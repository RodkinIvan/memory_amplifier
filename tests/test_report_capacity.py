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
        self.assertIn("12 positions", table)       # P = 32: too many to list
        self.assertIn("20 / 32", table)

    def test_stages_in_training_are_read_from_the_logs(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "capacity").mkdir()
            (Path(d) / "capacity" / "swa_2pass_pk64_steps50000_lr0.001_s0.log").write_text(
                "[swa_2pass_pk64_steps50000_lr0.001_s0] step 5000 loss 0.1 | n32: pk 0.10 v 1.00 e 1.00 n256: pk 0.02 v 1.00 e 1.00 | passkey positions >= 0.9 at N = 256: 3/64\\n"
                "[swa_2pass_pk64_steps50000_lr0.001_s0] step 6000 loss 0.1 | n32: pk 0.20 v 1.00 e 1.00 n256: pk 0.05 v 1.00 e 1.00 | passkey positions >= 0.9 at N = 256: 9/64\\n".replace("\\n", "\n"))
            data, _, _ = self.build(Path(d))
        self.assertIn("P = 64, step 6,000, 9 of 64 positions at 0.9 or more, exact 0.05", " ".join(data["tldr"]))
        self.assertTrue(any(f["key"] == "running" for f in data["extra_figs"]))
        self.assertFalse(any(f["key"] == "running" for f in self.build()[0]["extra_figs"]))

    def test_nothing_to_report_without_stages(self):
        with self.assertRaises(ValueError):
            RC.capacity_data({"threshold": 0.9}, {}, Path(tempfile.gettempdir()) / "nope")


if __name__ == "__main__":
    unittest.main()
