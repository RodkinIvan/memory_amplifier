"""The two-pass report (report_2pass): rows are configurations, every number in the text comes from the runs, probe tables appear when probe results exist.

    python -m unittest tests.test_report_2pass -v
"""
import json
import unittest

from tests import _paths  # noqa: F401
import report_2pass as R2
from tests.test_report import fake_run, history

FB = ["model.gain_init=0", "train.steps=50000"]
H = lambda a, b: history({38000: a, 50000: b}, {38000: a / 2, 50000: b / 2})  # noqa: E731


class TwoPassReport(unittest.TestCase):
    def setUp(self):
        mk = lambda ov, seeds, pk, hist=None: [fake_run("swa_2pass.yaml", ov, s, {n: v(s) if callable(v) else v for n, v in pk.items()}, hist=hist) for s in seeds]
        self.p2_32 = mk(["model.segment=32", "model.pass_output=2"], range(3), {256: 1.0, 1024: lambda s: 0.2 * s, 16384: 0.0}, H(0.2, 0.9))
        self.p1_32 = mk(["model.segment=32", "model.pass_output=1"], range(3), {256: 0.8, 1024: 0.1, 16384: 0.0}, H(0.1, 0.8))
        self.p2_64 = mk(["model.segment=64", "model.pass_output=2"], range(3), {256: 0.5, 1024: 0.05, 16384: 0.0}, H(0.1, 0.5))
        self.fb32 = [fake_run("swa_fb.yaml", [*FB, "model.segment=32", "model.nonlin=none"], s, {256: 1.0, 1024: 0.6}, hist=H(0.1, 0.5)) for s in range(3)]
        self.fb64 = [fake_run("swa_fb.yaml", [*FB, "model.segment=64", "model.nonlin=tanh"], s, {256: 0.9, 1024: 0.3}, hist=H(0.1, 0.5)) for s in range(3)]
        self.va32 = [fake_run("swa_va.yaml", ["model.segment=32"], s, {256: 0.4, 1024: 0.1}, hist=H(0.1, 0.4)) for s in range(3)]
        self.swa = [fake_run("swa.yaml", ["model.n_layers=2", "train.steps=50000", "train.curriculum_stage_steps=2000"], s, {32: 1.0}) for s in range(3)]
        self.runs = [*self.p2_32, *self.p1_32, *self.p2_64, *self.fb32, *self.fb64, *self.va32, *self.swa]

    def build(self, probes=None, runs=None):
        return R2.p2_data(runs or self.runs, {}, probes if probes is not None else {})

    def test_the_two_pass_configurations_are_rows_next_to_the_loop_adaptation_and_baseline(self):
        data, title, h1 = self.build()
        self.assertEqual([r["key"] for r in data["rows"]], ["p2_seg32_po2", "p2_seg32_po1", "p2_seg64_po2", "fb32", "fb64", "va32", "w64"])
        self.assertEqual(title, "SWA Two-Pass Report")
        self.assertEqual(data["results"]["p2_seg32_po2"]["1024"]["per"]["passkey"], [0.0, 0.2, 0.4])
        self.assertTrue(data["hide_scatter"])
        labels = " ".join(r["label"] for r in data["rows"])
        self.assertIn("output of pass 1", labels)

    def test_the_text_numbers_come_from_the_runs(self):
        data, _, _ = self.build()
        text = " ".join(data["tldr"])
        self.assertIn("0.00 / 0.20 / 0.40", text)      # per seed at N = 1,024, segment 32, output of pass 2
        self.assertIn("0.80 / 0.10 / 0.00 / 0.00", text)  # the output-of-pass-1 row, mean at N = 256 / 1,024 / 4,096 / 16,384
        self.assertNotIn("NaN", text)
        self.assertNotIn("{fig:", json.dumps(data))
        nxt = data["next"]
        self.assertIn("segment 32, output of pass 2: 3 of 3", nxt)   # validation passkey at N = 256 at the end of training: 0.9
        self.assertIn("segment 64, output of pass 2: 0 of 3", nxt)   # ... 0.5 is not above 0.5
        self.assertIn("Which pass feeds the next layer (segment 32)", nxt)
        self.assertIn("0.00 / 0.20 / 0.40", nxt)

    def test_a_gain_pilot_is_its_own_row_and_other_settings_are_not_merged(self):
        pilot = [fake_run("swa_2pass.yaml", ["model.segment=32", "model.pass_output=2", "model.gain_init=0.1"], 0, {256: 1.0}, hist=H(0.5, 1.0))]
        odd = [fake_run("swa_2pass.yaml", ["model.segment=32", "train.lr=0.002"], 0, {256: 1.0})]
        data, _, _ = self.build(runs=[*self.runs, *pilot, *odd])
        keys = [r["key"] for r in data["rows"]]
        self.assertIn("p2_seg32_po2_gi0.1", keys)
        self.assertEqual(len(data["results"]["p2_seg32_po2"]["256"]["per"]["passkey"]), 3)
        appendix = next(f for f in data["extra_figs"] if "configuration" in f["title"].lower())
        self.assertIn("lr 0.001 → 0.002", appendix["html"])  # the odd run has its own configuration in the appendix, not a row

    def test_without_two_pass_runs_there_is_nothing_to_report(self):
        with self.assertRaises(ValueError):
            R2.p2_data(self.fb32, {}, {})

    def test_the_method_figure_is_well_formed(self):
        import re
        svg = R2.method_svg()
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>"))
        self.assertEqual(svg.count("<svg"), 1)
        for line in re.findall(r"<text x='(\d+)' y='(?:16)' class='t'>(.*?)</text>", svg):
            self.assertLess(int(line[0]) + 6.9 * len(line[1]), 780, "the header line must fit the viewBox")

    def test_probe_results_become_figures_and_sentences(self):
        probes = {
            "cache": {"swa_2pass_seg32_po2_s0": {"passkey_len": 4, "chance": 0.0625, "n_train": 3000, "n_test": 500, "distances": [96, 1024],
                      "models": {"model": {"run": "swa_2pass_seg32_po2_s0", "reach": 126, "segment": 32, "by_distance": {
                          "96": {"cut_segment": 3, "tokens": 90, "v": {"mean": 0.99, "per_token": [1.0, 1.0, 1.0, 0.96]}, "k": {"mean": 0.9}},
                          "1024": {"cut_segment": 32, "tokens": 1018, "v": {"mean": 0.8, "per_token": [0.9, 0.5, 0.9, 0.9]}, "k": {"mean": 0.6}}}},
                                 "control": {"run": "swa_w64", "reach": 126, "segment": 32, "by_distance": {
                                     "96": {"cut_segment": 3, "tokens": 90, "v": {"mean": 0.9}, "k": {"mean": 0.9}},
                                     "1024": {"cut_segment": 32, "tokens": 1018, "v": {"mean": 0.07}, "k": {"mean": 0.06}}}}}}},
            "perturb": {"swa_2pass_seg32_po2_s0": {"kind": "swa_2pass", "n_writes": 1024, "s_star": 30, "s_star_tokens": 960,
                                                   "exact": {"clean": 0.9, "values_0.1": 0.9, "values_1": 0.7, "keys_0.1": 0.9, "keys_1": 0.5, "both_0.1": 0.9, "both_1": 0.4, "values_zero": 0.1, "keys_zero": 0.0, "both_zero": 0.0},
                                                   "contraction": {"1": 1.0, "16": 0.3}, "drift": {}}},
        }
        data, _, _ = self.build(probes)
        titles = [f["title"] for f in data["extra_figs"]]
        self.assertTrue(any("read out of the cached window" in t for t in titles))
        self.assertTrue(any("Perturbing the cached window" in t for t in titles))
        text = " ".join(data["tldr"])
        self.assertIn("values 0.80", text)
        self.assertIn("keys 0.60", text)
        self.assertIn("0.90 / 0.50 / 0.90 / 0.90", text)
        self.assertIn("plain SWA 0.07", text)
        self.assertIn("keeps 0.90 clean, 0.40 with its cached keys and values kicked by 1 × RMS, 0.10 with the values zeroed, 0.00 with the keys zeroed", text)
        pert = next(f for f in data["extra_figs"] if "Perturbing" in f["title"])
        self.assertIn("1.00 / 0.30", pert["html"])


if __name__ == "__main__":
    unittest.main()
