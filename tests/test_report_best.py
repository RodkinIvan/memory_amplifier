"""The presentation sub-report (report_best): what it picks, what it says, and that every number comes from the runs.

    python -m unittest tests.test_report_best -v
"""
import json
import re
import unittest

from tests import _paths  # noqa: F401
import report_best as B
import report_groups as RG
from tests.test_report import fake_run, history

SEG8 = ["model.gain_init=0", "model.nonlin=none", "model.segment=8"]


def run(setup_steps, seed, test_pk, val_pk, hist=None):
    """A segment-8 no-tanh gain-0 run whose test and validation passkey differ, so the selection rule is visible."""
    r = fake_run("swa_fb.yaml", [*SEG8, f"train.steps={setup_steps}"], seed, test_pk, hist=hist)
    for row in r["rows"]:
        if row["split"] == "validation":
            row["passkey_acc"] = val_pk.get(row["N"], 0.0)
    return r


FULL = {n: 1.0 for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)}


class BestReport(unittest.TestCase):
    def setUp(self):
        h = history({2000: 0.0, 4000: 0.9}, {2000: 0.0, 4000: 0.6})
        # seed 2 is best on validation but not on test; seed 0 is best on test but not on validation: the checkpoint must follow validation
        self.a50 = [run(50000, 0, {**FULL, 16384: 0.95}, {**FULL, 16384: 0.50}, h), run(50000, 1, {256: 0.0}, {256: 0.0}, h), run(50000, 2, {**FULL, 16384: 0.90}, {**FULL, 16384: 0.99}, h)]
        self.a20 = [run(20000, s, {**FULL, 16384: v}, {**FULL, 16384: v}) for s, v in ((0, 0.9), (1, 0.0), (2, 0.7))]
        self.b50 = [fake_run("swa_fb.yaml", ["model.gain_init=0", "model.nonlin=none", "model.segment=32", "train.steps=50000"], s, {256: 1.0, 1024: 0.6}, hist=h) for s in range(3)]
        self.swa = [fake_run("swa.yaml", ["model.n_layers=2", "train.steps=50000", "train.curriculum_stage_steps=2000", f"model.window={w}"], s, {32: 1.0, 256: 0.98 if w == 1024 else 0.0})
                    for w in (64, 1024) for s in range(3)]
        def gdn(seed, pk):
            r = fake_run("exp1.yaml", ["model.alpha=1.0"], seed, pk)
            r["model"], r["alpha"] = "gdn_fixed", 1.0
            for row in r["rows"]:
                if row["split"] == "validation":
                    row["passkey_acc"] = pk.get(row["N"], 0.0)
            return r
        self.gdn = [gdn(0, {**{n: 1.0 for n in (32, 64, 128, 256, 512, 1024)}, 2048: 0.2}), gdn(1, {**{n: 1.0 for n in (32, 64, 128, 256, 512, 1024)}, 2048: 0.9}), gdn(2, {256: 1.0})]
        self.runs = [*self.a50, *self.a20, *self.b50, *self.swa, *self.gdn]

    def build(self):
        return B.best_data(self.runs, {}, [])

    def test_the_checkpoint_is_chosen_on_validation_and_reported_on_test(self):
        self.assertEqual(B.pick_best(self.a50 + self.a20)["name"], self.a50[2]["name"])
        data, _, _ = self.build()
        self.assertIn(self.a50[2]["name"], data["meta"])
        self.assertIn("seed 2", data["rows"][0]["label"])
        self.assertAlmostEqual(data["results"]["best"]["16384"]["passkey"], 0.90)  # the test value of seed 2, not its validation value 0.99
        self.assertIn("chosen on the validation split", " ".join(data["tldr"]))

    def test_setup_means_and_baselines_are_rows_and_their_values_are_means_of_the_seeds(self):
        data, _, _ = self.build()
        keys = [r["key"] for r in data["rows"]]
        self.assertEqual(keys, ["best", "A50", "A20", "B50", "w1024", "w64", "gdn_best", "gdn"])
        self.assertAlmostEqual(data["results"]["A50"]["16384"]["passkey"], (0.95 + 0.0 + 0.90) / 3, places=6)
        self.assertEqual(data["results"]["A20"]["16384"]["per"]["passkey"], [0.9, 0.0, 0.7])
        self.assertTrue(data["hide_scatter"])

    def test_text_numbers_come_from_the_data_and_nothing_is_left_open(self):
        data, title, h1 = self.build()
        text = " ".join(data["tldr"]) + data["intro"] + json.dumps(data["extra_figs"])
        self.assertNotIn("{fig:", json.dumps(data))
        self.assertNotIn("NaN", text)
        self.assertIn("0.90", " ".join(data["tldr"]))  # the checkpoint's test passkey at N = 16,384
        self.assertIn("0.95 / 0.00 / 0.90", " ".join(data["tldr"]))  # per-seed values of the best setup at N = 16,384 (instability)
        self.assertIn("0.90 / 0.00 / 0.70", " ".join(data["tldr"]))  # and of the 20k-step setup
        self.assertEqual(title, "SWA Segment Recurrence Report")

    def test_the_recipe_bullets_are_read_from_the_stored_config(self):
        data, _, _ = self.build()
        bullets = re.findall(r"<li>(.*?)</li>", data["intro"], re.S)
        text = " ".join(bullets)
        for part in ("segment of 8 tokens", "Loop gain initialised to 0", "50,000 steps", "Learning rate × 10", "32 → 64 → 128 → 256", "every 2,000 steps", "563,584 parameters", "weight decay 0.01"):
            self.assertIn(part, text, part)
        self.assertIn("50k-step seeds 0, 1, 2; 20k-step seeds 0, 1, 2", text)

    def test_figures_cover_every_n_and_the_method_diagram_is_there(self):
        data, _, _ = self.build()
        n_fig = next(f for f in data["extra_figs"] if f["key"] == "n")
        for s in n_fig["spec"]["series"]:
            self.assertEqual([p["x"] for p in s["points"]], B.NS, s["label"])
        self.assertEqual(len({s["short"] for s in n_fig["spec"]["series"]}), len(n_fig["spec"]["series"]), "end labels must be distinct")
        self.assertIn("<svg", data["intro"])
        self.assertIn("one vector per layer and segment", data["intro"])
        self.assertTrue(any(f["key"] == "c1024" for f in data["extra_figs"]))  # validation curves from the stored history

    def test_the_gdn_baseline_is_shown_as_its_best_seed_chosen_on_validation_and_as_the_mean(self):
        data, _, _ = self.build()
        self.assertEqual(B.pick_best_baseline(self.gdn)["seed"], 1)  # best average validation passkey, not seed 0
        self.assertIn("best seed (1)", next(r["label"] for r in data["rows"] if r["key"] == "gdn_best"))
        self.assertAlmostEqual(data["results"]["gdn_best"]["2048"]["passkey"], 0.9)
        self.assertAlmostEqual(data["results"]["gdn"]["2048"]["passkey"], (0.2 + 0.9 + 0.0) / 3, places=6)
        text = " ".join(data["tldr"])
        self.assertIn("best of 3 seeds, picked on validation like our checkpoint", text)
        self.assertIn("441k parameters, 30k steps without the curriculum", text)

    def test_gdn2_trained_like_our_model_is_a_separate_row_and_never_merged_with_the_fixed_alpha_gdn(self):
        def matched(seed, pk):
            r = fake_run("gdn2.yaml", ["train.steps=50000", "train.curriculum_stage_steps=2000"], seed, pk)
            r["model"], r["alpha"] = "gdn2_steps50000_cur2000", float("nan")
            for row in r["rows"]:
                if row["split"] == "validation":
                    row["passkey_acc"] = pk.get(row["N"], 0.0)
            return r
        m = [matched(0, {256: 1.0, 1024: 1.0, 4096: 0.6, 16384: 0.1}), matched(1, {256: 1.0, 1024: 0.5})]
        data, _, _ = B.best_data([*self.runs, *m], {}, [])
        keys = [r["key"] for r in data["rows"]]
        self.assertEqual(keys[-4:], ["gdn_best", "gdn", "gdn_m_best", "gdn_m"])
        self.assertEqual(len(data["results"]["gdn"]["256"]["per"]["passkey"]), 3)  # the 30k-step seeds only
        self.assertEqual(len(data["results"]["gdn_m"]["256"]["per"]["passkey"]), 2)
        self.assertAlmostEqual(data["results"]["gdn_m_best"]["4096"]["passkey"], 0.6)  # seed 0: the better validation curve
        text = " ".join(data["tldr"])
        self.assertIn("GDN-2 with the stock learned decay, trained like our model (50k steps, same curriculum; best of 2 seeds, picked on validation) scores", text)
        n_fig = next(f for f in data["extra_figs"] if f["key"] == "n")
        self.assertIn("GDN-2", [s["short"] for s in n_fig["spec"]["series"]])
        one, _, _ = B.best_data([*self.runs, m[0]], {}, [])
        self.assertNotIn("gdn_m", [r["key"] for r in one["rows"]])  # a single seed has no mean row

    def test_only_setups_that_exist_are_shown(self):
        no_a50 = [r for r in self.runs if not (r["cfg"]["model"]["kind"] == "swa_fb" and r["cfg"]["model"]["segment"] == 8 and r["cfg"]["train"]["steps"] == 50000)]
        data, _, _ = B.best_data(no_a50, {}, [])
        self.assertNotIn("A50", [r["key"] for r in data["rows"]])
        self.assertEqual(data["rows"][0]["key"], "best")
        with self.assertRaises(ValueError):
            B.best_data(list(self.swa), {}, [])


@unittest.skipUnless((_paths.ROOT / "results" / "raw").exists(), "needs results")
class OnTheCommittedResults(unittest.TestCase):
    def test_builds_and_renders(self):
        import tempfile
        from pathlib import Path
        import report as R
        data, title, h1 = B.best_data()
        self.assertEqual(data["rows"][0]["key"], "best")
        with tempfile.TemporaryDirectory() as d:
            R.render(data, title, h1, Path(d) / "r.html")
            self.assertIn("Segment-level recurrence", (Path(d) / "r.html").read_text())


if __name__ == "__main__":
    unittest.main()
