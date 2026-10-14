"""The passkey-capacity report (report_capacity): chains are one recipe of one model from one seed, every number comes from the entries, stages in training are listed.

    python -m unittest tests.test_report_capacity -v
"""
import tempfile
import unittest
from pathlib import Path

from tests import _paths  # noqa: F401
import config as C
import report_capacity as RC
from tests.test_capacity import INIT, fake_result
import importlib.util

spec = importlib.util.spec_from_file_location("capacity", _paths.ROOT / "scripts" / "passkey_capacity.py")
CAP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CAP)


def stage(P, kind, steps, lr, seed, exact, pos, init="checkpoints/base.pt", cur=0):
    """(capacity entry, run dict) of one stage with the given validation exact match at N = 256 and per-position accuracy."""
    name = f"{kind}_pk{P}_steps{steps}_lr{lr:g}_s{seed}"
    res = fake_result(exact, token256=sum(pos) / len(pos), pos256=pos)
    res["config"]["train"].update({"steps": steps, "lr": lr, "curriculum_stage_steps": cur})
    entry = CAP.summarize(P, name, init, res)
    rows = [{**r, "ar_expired_stale_acc": 1.0, "ar_expired_acc": 1.0, "n_examples": 1000, "n_valid": 100, "n_expired_stale": 100} for r in res["rows"]]
    run = {"name": name, "seed": seed, "cfg": {"model": {"kind": kind}, "seed": seed}, "rows": rows}
    return entry, run


class CapacityReport(unittest.TestCase):
    def setUp(self):
        base = {"name": "base", "seed": 5, "cfg": {"model": {"kind": "swa_2pass"}, "seed": 5}, "rows": [{"split": s, "N": 256, "passkey_acc": 0.999} for s in ("validation", "test")]}
        b8, r8 = stage(8, "swa_2pass", 50000, 1e-3, 0, 0.95, [1.0] * 8, "checkpoints/base.pt", 2000)
        b16, r16 = stage(16, "swa_2pass", 50000, 1e-3, 0, 0.92, [1.0] * 16, "runs/x/model.pt", 2000)
        b32, r32 = stage(32, "swa_2pass", 50000, 1e-3, 0, 0.5, [1.0] * 20 + [0.1] * 12, "runs/x/model.pt", 2000)
        s8, q8 = stage(8, "swa_2pass", 10000, 3e-4, 0, 0.0, [1.0] * 4 + [0.06] * 4, "checkpoints/base.pt")
        f8, g8 = stage(8, "swa_fb", 50000, 1e-3, 0, 0.004, [1.0, 1, 1, 1, 0.07, 1, 1, 0.05], "checkpoints/fb.pt", 2000)
        self.capacity = {"threshold": 0.9, "stages": [], "budget_runs": [f8], "stages_swa_2pass": [s8], "stages_swa_2pass_budget": [b8, b16, b32]}
        self.runs = {r["name"]: r for r in (base, r8, r16, r32, q8, g8)}
        self.runs["fb"] = {"name": "fb", "seed": 2, "cfg": {"model": {"kind": "swa_fb"}, "seed": 2}, "rows": [{"split": "validation", "N": 256, "passkey_acc": 1.0}]}
        self.entries = (s8, b8, b16, b32, f8)

    def build(self, logs=None):
        with tempfile.TemporaryDirectory() as d:
            return RC.capacity_data(self.capacity, self.runs, Path(d) if logs is None else logs)

    def test_chains_are_one_recipe_of_one_model_from_one_seed(self):
        ch = RC.chains(self.capacity, self.runs)
        self.assertEqual(len(ch), 3)  # budget two-pass, short two-pass, budget vector loop
        budget = ch[0]
        self.assertEqual((budget["model"], budget["steps"], [e["passkey_len"] for e in budget["stages"]]), ("swa_2pass", 50000, [8, 16, 32]))
        self.assertEqual(budget["base"], ("base", 0.999))                       # the start checkpoint's own exact match is the P = 4 point
        self.assertIn("two-pass layer (segment 32), 50k steps, lr 0.001, curriculum, seed 0", budget["label"])
        self.assertEqual([c["model"] for c in ch], ["swa_2pass", "swa_2pass", "swa_fb"])

    def test_the_text_says_where_each_chain_holds_and_where_it_stops(self):
        data, title, h1 = self.build()
        text = " ".join(data["tldr"])
        self.assertEqual(title, "Passkey Capacity Report")
        self.assertIn("Holds P = 16 (64 bits): validation exact 0.92 at N = 256", text)       # the budget chain: 8 and 16 pass
        self.assertIn("Stops at P = 32: exact 0.50", text)                                      # and fails at 32 ...
        self.assertIn("20 of 32 positions at 0.9 or more", text)                              # ... with 20 positions stored
        self.assertIn("No stage reaches 0.9", text)                                              # the short fine-tune and the vector loop's P = 8 never pass
        self.assertIn("4 of 8 positions at 0.9 or more (below 0.9: positions 5, 6, 7, 8)", text)  # which tokens were not learned
        self.assertIn("6 of 8 positions at 0.9 or more (below 0.9: positions 5, 8)", text)

    def test_the_grid_has_a_row_per_stage_and_the_chart_a_line_per_chain_and_the_threshold(self):
        data, _, _ = self.build()
        self.assertEqual(len(data["rows"]), 5)
        self.assertEqual(data["ns"], [32, 64, 128, 256])
        key = next(r["key"] for r in data["rows"] if r["key"].endswith("_50000_0_32") and "swa_2pass" in r["key"])
        self.assertAlmostEqual(data["results"][key]["256"]["passkey"], 0.5)
        chart = next(f for f in data["extra_figs"] if f["key"] == "chart")["spec"]
        self.assertEqual(len(chart["series"]), 4)                                               # 3 chains + the threshold line
        budget = chart["series"][0]
        self.assertEqual([(p["x"], round(p["y"], 3)) for p in budget["points"]], [(4, 0.999), (8, 0.95), (16, 0.92), (32, 0.5)])
        self.assertEqual(chart["xTicks"], [4, 8, 16, 32])
        self.assertTrue(chart["series"][1]["dashed"] and not budget["dashed"])                  # short fine-tunes are dashed

    def test_seeds_of_one_recipe_are_counted_per_passkey_length(self):
        """Seeds 0 and 1 of the budget recipe: seed 0 passes P = 8 and fails 16, seed 1 passes 8 and 16 and fails 32."""
        b8, r8 = stage(8, "swa_2pass", 50000, 1e-3, 1, 0.97, [1.0] * 8, "checkpoints/base.pt", 2000)
        b16, r16 = stage(16, "swa_2pass", 50000, 1e-3, 1, 0.96, [1.0] * 16, "runs/x/model.pt", 2000)
        b32, r32 = stage(32, "swa_2pass", 50000, 1e-3, 1, 0.0, [1.0] * 14 + [0.1] * 18, "runs/x/model.pt", 2000)
        c0, d0 = stage(16, "swa_2pass", 50000, 1e-3, 0, 0.0, [1.0] * 9 + [0.1] * 7, "runs/x/model.pt", 2000)
        self.capacity["stages_swa_2pass_budget"] = [self.entries[1], c0]                    # seed 0: P = 8 passes, P = 16 fails
        self.capacity["stages_swa_2pass_budget_s1"] = [b8, b16, b32]
        self.runs.update({r["name"]: r for r in (r8, r16, r32, d0)})
        rates = RC.seed_rates(RC.chains(self.capacity, self.runs))
        self.assertEqual(len(rates), 1)                                                         # only the recipe with two seeds
        self.assertEqual(rates[0]["seeds"], [0, 1])
        self.assertEqual([(P, reached, passed) for P, reached, passed, _, _ in rates[0]["rows"]], [(8, 2, 2), (16, 2, 1), (32, 1, 0)])
        self.assertEqual(rates[0]["rows"][1][3], 12.5)                                          # median stored positions at P = 16: (9 + 16) / 2
        data, _, _ = self.build()
        text = " ".join(data["tldr"])
        self.assertIn("P = 8: 2 of 2 seeds that reached it pass; P = 16: 1 of 2 seeds that reached it pass; P = 32: 0 of 1 seeds that reached it pass", text)
        self.assertIn("The longest passkey held by any seed is P = 16 (64 bits)", text)
        self.assertTrue(any(f["key"] == "rates" for f in data["extra_figs"]))

    def rerun_setup(self):
        """Seed 0 of the budget recipe fails P = 16 (9 positions); the same stage with 150k steps from the same start stores 3."""
        c0, d0 = stage(16, "swa_2pass", 50000, 1e-3, 0, 0.0, [1.0] * 9 + [0.1] * 7, "runs/x/model.pt", 2000)
        long16, r_long = stage(16, "swa_2pass", 150000, 1e-3, 0, 0.0, [1.0] * 3 + [0.1] * 13, "runs/x/model.pt", 2000)
        self.capacity["stages_swa_2pass_budget"] = [self.entries[1], c0]
        self.capacity["budget_runs_swa_2pass_long"] = [long16]
        self.runs.update({r["name"]: r for r in (d0, r_long)})

    def test_a_longer_budget_rerun_of_a_stage_is_not_an_independent_chain(self):
        self.rerun_setup()
        ch = RC.chains(self.capacity, self.runs)
        long = next(c for c in ch if c["steps"] == 150000)
        self.assertEqual(long["rerun_of"]["steps"], 50000)
        self.assertIsNone(next(c for c in ch if c["steps"] == 50000 and c["model"] == "swa_2pass")["rerun_of"])
        self.assertIsNone(next(c for c in ch if c["steps"] == 10000)["rerun_of"])                  # another learning rate and no curriculum: a different recipe, not a re-run
        self.assertEqual(RC.seed_rates(ch), [])                                                     # one seed per recipe: nothing to count, the re-run is not a second seed
        data, _, _ = self.build()
        chart = next(f for f in data["extra_figs"] if f["key"] == "chart")["spec"]
        self.assertEqual([sr["short"] for sr in chart["series"]], ["two-pass 50k s0", "two-pass 10k s0", "m-loop 50k s0", "0.9"])   # the re-run is not a line
        self.assertEqual(len(data["rows"]), 5)                                                       # but it is a row of the grid
        text = " ".join(data["tldr"])
        self.assertIn("two-pass seed 0, P = 16: 9 of 16 positions stored at 50k steps, 3 at 150k", text)
        self.assertIn("0 of 1 pass", text)
        self.assertNotIn("150k steps, lr 0.001, curriculum, seed 0.</b>", text)                      # no chain bullet for the re-run
        reruns = next(f for f in data["extra_figs"] if f["key"] == "reruns")["html"]
        self.assertIn("9 / 16", reruns)
        self.assertIn("3 / 16", reruns)

    def test_the_position_grid_has_a_cell_per_position_of_each_long_or_failing_stage(self):
        data, _, _ = self.build()
        grid = next(f for f in data["extra_figs"] if f["key"] == "positions")["html"]
        # stages shown: P = 16 and 32 of the budget chain, and every stage with a position below 0.9 (two-pass 10k P = 8, vector loop P = 8); the passing P = 8 stage is left out
        self.assertEqual(grid.count("<tr><td"), 4)
        self.assertEqual(grid.count("class='pc c'"), 16 + 32 + 8 + 8)
        self.assertIn("position 21 of 32: 0.10", grid)
        self.assertIn("two-pass 50k s0, P = 32", grid)

    def test_positions_that_fail_in_most_stopping_stages_are_named(self):
        """Three seeds stop at P = 16; positions 10 and 16 are below 0.9 in all of them, position 12 in one."""
        entries, runs = [], {}
        for sd, bad in ((0, {10, 12, 16}), (1, {10, 16}), (2, {10, 11, 16})):
            pos = [0.1 if j + 1 in bad else 1.0 for j in range(16)]
            e8, r8 = stage(8, "swa_2pass", 50000, 1e-3, sd, 0.95, [1.0] * 8, "checkpoints/base.pt", 2000)
            e16, r16 = stage(16, "swa_2pass", 50000, 1e-3, sd, 0.0, pos, "runs/x/model.pt", 2000)
            self.capacity[f"stages_swa_2pass_s{sd}"] = [e8, e16]
            runs.update({r["name"]: r for r in (r8, r16)})
        self.capacity["stages_swa_2pass_budget"] = []
        self.runs.update(runs)
        data, _, _ = self.build()
        text = " ".join(data["tldr"])
        self.assertIn("Of the 3 chains that stop at P = 16, positions 10 (3 of 3), 16 (3 of 3) are below 0.9 in most of them", text)
        self.assertNotIn("12 (", text.split("Which positions fail")[1])
        self.assertIn("seed 0: P = 16, 13 of 16 positions stored; seed 1: P = 16, 14 of 16 positions stored; seed 2: P = 16, 13 of 16 positions stored", text)

    def test_the_labels_fit_into_the_chart_and_the_grid(self):
        """The end-of-line labels of the chart are short names and the margin on the right is sized from them (the long chain descriptions stay in the legend); grid row labels fit the label column."""
        data, _, _ = self.build()
        chart = next(f for f in data["extra_figs"] if f["key"] == "chart")["spec"]
        shorts = [sr["short"] for sr in chart["series"]]
        self.assertEqual(shorts, ["two-pass 50k s0", "two-pass 10k s0", "m-loop 50k s0", "0.9"])   # model, steps, seed (and the threshold line)
        self.assertTrue(all(len(x) <= 16 for x in shorts), shorts)
        for sr in chart["series"][:3]:
            self.assertIn("seed", sr["label"])                                  # the legend keeps the full description (model, segment, recipe, seed)
            self.assertIn("steps", sr["label"])
        W, m_l = 760, 52                                                         # the template's chart width and left margin
        longest = max(len(x) for x in shorts)
        self.assertGreaterEqual(chart["rightPad"], 14 + 6.5 * longest)           # a label starts 14 px right of the last point: it ends inside the margin
        self.assertLess(m_l + chart["rightPad"], W / 2)                          # and the margin leaves most of the width to the plot
        for r in data["rows"]:
            self.assertLessEqual(len(r["label"]) * 6.6, data["label_w"], r["label"])

    def test_the_table_lists_every_stage_with_its_positions_below_the_threshold(self):
        data, _, _ = self.build()
        table = next(f for f in data["extra_figs"] if f["key"] == "table")["html"]
        self.assertIn("5, 8", table)               # vector loop P = 8: positions 5 and 8 below 0.9
        self.assertIn("12 pos.", table)       # P = 32: too many to list
        self.assertIn("20 / 32", table)

    def test_stages_in_training_are_read_from_the_logs(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "capacity").mkdir()
            (Path(d) / "capacity" / "swa_2pass_pk64_steps50000_lr0.001_s0.log").write_text(
                "[swa_2pass_pk64_steps50000_lr0.001_s0] step 5000 loss 0.1 | n32: pk 0.10 v 1.00 e 1.00 n256: pk 0.02 v 1.00 e 1.00 | passkey positions >= 0.9 at N = 256: 3/64\\n"
                "[swa_2pass_pk64_steps50000_lr0.001_s0] step 6000 loss 0.1 | n32: pk 0.20 v 1.00 e 1.00 n256: pk 0.05 v 1.00 e 1.00 | passkey positions >= 0.9 at N = 256: 9/64\\n".replace("\\n", "\n"))
            data, _, _ = self.build(Path(d))
        self.assertIn("P = 64, 50k-step stage, seed 0: step 6,000, 9 of 64 positions at 0.9 or more, exact 0.05", " ".join(data["tldr"]))   # a short name in the summary ...
        running = next(f for f in data["extra_figs"] if f["key"] == "running")["html"]
        self.assertIn("swa_2pass_pk64_steps50000_lr0.001_s0", running)                                                                      # ... the full run name in the table
        self.assertTrue(any(f["key"] == "running" for f in data["extra_figs"]))
        self.assertFalse(any(f["key"] == "running" for f in self.build()[0]["extra_figs"]))

    def test_nothing_to_report_without_stages(self):
        with self.assertRaises(ValueError):
            RC.capacity_data({"threshold": 0.9}, {}, Path(tempfile.gettempdir()) / "nope")


if __name__ == "__main__":
    unittest.main()
