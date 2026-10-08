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
        self.gdn2 = [fake_run("gdn2.yaml", ["train.steps=50000", "train.curriculum_stage_steps=2000"], s, {256: 1.0, 1024: 0.9, 4096: 0.2}) for s in range(2)]
        self.runs = [*self.va32, *self.va8, *self.fb32, *self.fb8, *self.swa, *self.gdn2]

    def build(self, probes=None):
        return V.va_data(self.runs, {}, probes if probes is not None else {})

    def test_the_value_adaptation_runs_are_rows_next_to_the_loop_and_the_baseline(self):
        data, title, h1 = self.build()
        self.assertEqual([r["key"] for r in data["rows"]], ["va32", "va8", "fb32", "fb8", "w64", "gdn2"])
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

    def test_open_issues_are_written_from_the_runs(self):
        long_hist = lambda a, b: history({38000: a, 50000: b}, {38000: a / 2, 50000: b / 2})  # noqa: E731
        va32 = [fake_run("swa_va.yaml", ["model.segment=32"], s, {256: 0.5, 1024: 0.1, 16384: 0.0}, hist=long_hist(0.2 + 0.1 * s, 0.4 + 0.1 * s)) for s in range(3)]
        va8 = [fake_run("swa_va.yaml", ["model.segment=8"], s, {32: 0.5 if s == 0 else 0.0}, valid=0.01 if s == 0 else 0.99, hist=long_hist(0.0, 0.0)) for s in range(3)]
        delta = {"r": {"n_writes": 4096, "segments": 400, "segment": 32, "layers": [{"gain": 1.0, "w_v_fro": 3.0, "at": {"0.5": {"segment": 2, "fro": 1.0, "fro_over_w_v": 0.4, "spectral": 0.5, "rank": 1.0, "max_entry": 0.3,
                                                                                                                       "cos_next": 0.9, "cos_lag16": 0.6, "saturated": 0.99, "mean_share": 0.5, "cov_fro_over_w_v": 0.1}}}]},
                 "big": {"n_writes": 4096, "segments": 400, "segment": 32, "layers": [{"gain": 1.0, "w_v_fro": 3.0, "at": {"0.5": {"segment": 2, "fro": 20.0, "fro_over_w_v": 6.5, "spectral": 19.0, "rank": 1.3, "max_entry": 1.0,
                                                                                                                       "cos_next": 0.9, "cos_lag16": 0.6, "saturated": 0.99, "mean_share": 0.5, "cov_fro_over_w_v": 2.1}}}]}}
        data, _, _ = V.va_data([*va32, *va8, *self.fb8, *self.swa], {}, {"delta": delta})
        text = data["next"]
        self.assertIn("0.20 / 0.30 / 0.40 per seed at step 38,000 and 0.40 / 0.50 / 0.60 at step 50,000", text)  # the runs were still improving
        self.assertIn("0.00 / 0.00 / 0.00", text)                      # segment-8 passkey at N = 256 per seed
        self.assertIn("0.01 / 0.99 / 0.99", text)                      # AR valid per seed: seed 0 never learned recall
        self.assertIn("99%", text)                                     # saturation of the codes, from the probe
        self.assertIn("in the 1 runs where the layer-0 update is at least as large as W_v (6.5–6.5 × ‖W_v‖)", text)  # and only the run with a big update is described as such
        self.assertNotIn("NaN", text)
        self.assertEqual(data["next"], V.va_data([*va32, *va8, *self.fb8, *self.swa], {}, {"delta": delta})[0]["next"])  # deterministic

    def test_probe_files_of_one_kind_are_merged_by_run(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "va_delta_w.json").write_text(json.dumps({"a": {"x": 1}}))
            (Path(d) / "va_delta_w_seg8.json").write_text(json.dumps({"b": {"x": 2}}))
            (Path(d) / "va_perturb.json").write_text(json.dumps({"a": {"y": 3}}))
            (Path(d) / "other.json").write_text(json.dumps({"z": 1}))
            got = V.load_probes(Path(d))
        self.assertEqual(got, {"delta": {"a": {"x": 1}, "b": {"x": 2}}, "perturb": {"a": {"y": 3}}})

    def test_probe_results_become_tables(self):
        probes = {
            "cache": {"swa_va_seg32_s0": {"passkey_len": 4, "chance": 0.0625, "n_train": 3000, "n_test": 500, "distances": [96, 1024],
                      "models": {"model": {"run": "swa_va_seg32_s0", "reach": 126, "segment": 32, "by_distance": {
                          "96": {"cut_segment": 3, "tokens": 90, "v": {"mean": 0.99, "per_token": [1.0, 1.0, 1.0, 0.96]}, "k": {"mean": 0.9}, "dw": {"mean": 0.97}},
                          "1024": {"cut_segment": 32, "tokens": 1018, "v": {"mean": 0.8, "per_token": [0.9, 0.5, 0.9, 0.9]}, "k": {"mean": 0.1}, "dw": {"mean": 0.7}}}},
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
        text = " ".join(data["tldr"])
        self.assertIn("0.80 at 1,018 tokens", text)             # the read-out at the farthest cut ...
        self.assertIn("0.90 / 0.50 / 0.90 / 0.90", text)         # ... token by token ...
        self.assertIn("plain SWA: 0.07", text)                   # ... and the control
        self.assertIn("keeps 0.90 clean", text)
        cache = next(f for f in data["extra_figs"] if "read out" in f["title"])
        self.assertIn("0.99", cache["html"])                    # the value read-out of the probed run ...
        self.assertIn("control: swa_w64", cache["html"])         # ... next to the plain-SWA control row
        self.assertIn("0.07", cache["html"])


if __name__ == "__main__":
    unittest.main()
