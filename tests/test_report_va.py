"""The value-adaptation report (report_va): rows are configurations, every number in the text comes from the runs, probe tables appear when probe results exist.

    python -m unittest tests.test_report_va -v
"""
import json
import unittest

from tests import _paths  # noqa: F401
import report_groups as RG
import report_va as V
from tests.test_report import fake_run, history

FB = ["model.gain_init=0", "model.nonlin=none", "train.steps=50000"]
H = lambda a, b: history({2000: a, 4000: b}, {2000: a / 2, 4000: b / 2})  # noqa: E731


class VaReport(unittest.TestCase):
    def setUp(self):
        self.va32 = [fake_run("swa_va.yaml", ["model.segment=32"], s, {256: 1.0, 1024: 0.9, 4096: 0.5 + 0.2 * s, 16384: 0.1 * s}, hist=H(0.1, 0.9)) for s in range(3)]
        self.va8 = [fake_run("swa_va.yaml", ["model.segment=8"], s, {256: 1.0, 1024: 1.0, 4096: 0.98, 16384: 0.0 if s == 1 else 0.9}, hist=H(0.2, 0.95)) for s in range(3)]
        self.fb32 = [fake_run("swa_fb.yaml", [*FB, "model.segment=32"], s, {256: 1.0, 1024: 0.6}, hist=H(0.1, 0.5)) for s in range(3)]
        self.fb8 = [fake_run("swa_fb.yaml", [*FB, "model.segment=8"], s, {256: 1.0, 1024: 1.0, 4096: 0.9, 16384: 0.99 if s == 2 else 0.05}, hist=H(0.2, 0.9)) for s in range(3)]
        self.swa = [fake_run("swa.yaml", ["model.n_layers=2", "train.steps=50000", "train.curriculum_stage_steps=2000"], s, {32: 1.0}) for s in range(3)]
        self.runs = [*self.va32, *self.va8, *self.fb32, *self.fb8, *self.swa]

    def build(self, probes=None):
        return V.va_data(self.runs, {}, probes if probes is not None else {})

    def test_the_value_adaptation_runs_are_rows_next_to_the_loop_and_the_baseline(self):
        data, title, h1 = self.build()
        self.assertEqual([r["key"] for r in data["rows"]], ["va32", "va32_best", "va8", "va8_best", "fb32", "fb8", "w64"])
        self.assertEqual(title, "SWA Value Adaptation Report")
        self.assertEqual(data["results"]["va32"]["16384"]["per"]["passkey"], [0.0, 0.1, 0.2])
        self.assertAlmostEqual(data["results"]["fb8"]["16384"]["passkey"], (0.05 + 0.05 + 0.99) / 3, places=6)  # means of the seeds of one configuration
        self.assertTrue(data["hide_scatter"])

    def test_the_text_numbers_come_from_the_runs(self):
        data, _, _ = self.build()
        text = " ".join(data["tldr"])
        self.assertIn("0.00 / 0.10 / 0.20", text)       # per seed at N = 16,384, segment 32
        self.assertIn("0.90 / 0.00 / 0.90", text)       # and segment 8
        self.assertIn("0.05 / 0.05 / 0.99", text)       # the vector loop at segment 8
        self.assertNotIn("NaN", text)
        self.assertNotIn("{fig:", json.dumps(data))

    def test_runs_that_differ_from_the_reference_in_something_else_are_not_merged_into_the_rows(self):
        odd = fake_run("swa_va.yaml", ["model.segment=32", "model.proj_std=0.5"], 3, {256: 1.0})
        data, _, _ = V.va_data([*self.runs, odd], {}, {})
        self.assertEqual(len(data["results"]["va32"]["256"]["per"]["passkey"]), 3)  # the odd run has its own configuration (appendix), not a row
        appendix = next(f for f in data["extra_figs"] if "appendix" in f["key"] or "configuration" in f["title"].lower())
        self.assertIn("proj_std 0.1 → 0.5", appendix["html"])

    def test_without_value_adaptation_runs_there_is_nothing_to_report(self):
        with self.assertRaises(ValueError):
            V.va_data(self.fb32, {}, {})

    def test_probe_results_become_tables(self):
        probes = {
            "cache": {"swa_va_seg32_s0": {"passkey_len": 4, "chance": 0.0625, "n_train": 3000, "n_test": 500, "distances": [96, 1024],
                      "models": {"model": {"run": "swa_va_seg32_s0", "reach": 126, "segment": 32, "by_distance": {
                          "96": {"cut_segment": 3, "tokens": 90, "v": {"mean": 0.99}, "k": {"mean": 0.9}, "dw": {"mean": 0.97}},
                          "1024": {"cut_segment": 32, "tokens": 1018, "v": {"mean": 0.8}, "k": {"mean": 0.1}, "dw": {"mean": 0.7}}}},
                                 "control": {"run": "swa_w64", "reach": 126, "segment": 32, "by_distance": {
                                     "96": {"cut_segment": 3, "tokens": 90, "v": {"mean": 0.9}, "k": {"mean": 0.9}},
                                     "1024": {"cut_segment": 32, "tokens": 1018, "v": {"mean": 0.07}, "k": {"mean": 0.06}}}}}}},
            "perturb": {"swa_va_seg32_s0": {"n_writes": 1024, "s_star": 30, "s_star_tokens": 960, "exact": {"clean": 0.9, "values_0.1": 0.9, "values_1": 0.5, "state_0.1": 0.9, "state_1": 0.8,
                                                                                                        "both_0.1": 0.9, "both_1": 0.4, "values_zero": 0.0, "state_zero": 0.7, "both_zero": 0.0},
                                            "contraction": {"1": 1.0, "16": 0.4}, "drift": {}}},
            "delta": {"swa_va_seg32_s0": {"n_writes": 4096, "segments": 400, "segment": 32, "layers": [
                {"gain": 0.8, "w_v_fro": 3.0, "at": {"0.5": {"segment": 200, "fro": 1.2, "fro_over_w_v": 0.4, "spectral": 0.5, "rank": 7.5, "max_entry": 0.3, "cos_next": 0.9, "cos_lag16": 0.6}}}]}},
        }
        data, _, _ = self.build(probes)
        titles = [f["title"] for f in data["extra_figs"]]
        self.assertTrue(any("read out of the cached window" in t for t in titles))
        self.assertTrue(any("Perturbing the memory" in t for t in titles))
        self.assertTrue(any("Size, rank and drift" in t for t in titles))
        cache = next(f for f in data["extra_figs"] if "read out" in f["title"])
        self.assertIn("0.99", cache["html"])
        self.assertIn("control: values", cache["html"])
        self.assertIn("0.07", cache["html"])


if __name__ == "__main__":
    unittest.main()
