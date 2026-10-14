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
        self.assertEqual([r["key"] for r in data["rows"]], ["best_p2", "p2_seg32_po2", "p2_seg32_po1", "p2_seg64_po2", "fb32", "fb64", "va32", "w64"])  # the output-1 runs of segment 32 are the E4 pool of the best checkpoint
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

    def test_gains_and_pilots_are_reported_from_the_data(self):
        pilot = [fake_run("swa_2pass.yaml", ["model.segment=32", "model.pass_output=2", "train.loop_lr_mult=1"], 0, {32: 0.57, 256: 0.01, 1024: 0.0})]
        probes = {"gains": {"swa_2pass_seg32_po1_s0": {"name": "swa_2pass_seg32_po1_s0", "layers": [{"mean": 0.01, "abs_mean": 0.17, "abs_max": 0.9, "negative": 0.4}, {"mean": 0.21, "abs_mean": 0.31, "abs_max": 1.1, "negative": 0.1}]}}}
        data, _, _ = self.build(probes, runs=[*self.runs, *pilot])
        gains = next(f for f in data["extra_figs"] if f["key"] == "gains")
        self.assertIn("+0.21", gains["html"])
        self.assertIn("40%", gains["html"])
        self.assertIn("swa_2pass_seg32_po2_llr1_s0", data["next"])
        self.assertIn("0.57 / 0.01 / 0.00 at N = 32 / 256 / 1,024", data["next"])
        self.assertIn("loop_lr_mult 10 → 1", data["next"])
        self.assertNotIn("Pilots", self.build()[0]["next"])  # no pilot runs, no paragraph

    def test_every_chart_has_a_margin_that_fits_its_end_labels(self):
        """The end-of-line label of a series is drawn 14 px right of its last point (about 6.5 px per character): the margin must be at least as wide as the longest label."""
        data, _, _ = self.build()
        n = 0
        for f in data["extra_figs"]:
            if f.get("spec"):
                shorts = [s["short"] for s in f["spec"]["series"] if s.get("short")]
                self.assertGreaterEqual(f["spec"]["rightPad"], 14 + 6.5 * max(len(x) for x in shorts), (f["title"], shorts))
                n += 1
        self.assertGreaterEqual(n, 2)

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
        self.assertIn("seg32_po2_s0 0.80", text)                  # the value read-out at the farthest cut, per probed run (name without the model prefix) ...
        self.assertIn("plain SWA 0.07", text)                      # ... next to the control
        self.assertIn("Token by token for the best (seg32_po2_s0): 0.90 / 0.50 / 0.90 / 0.90", text)
        self.assertIn("seg32_po2_s0 0.90 → 0.40 → 0.00", text)    # perturbation: clean, kicked by 1 x RMS, zeroed
        self.assertIn("shrinks to 0.30–0.30 of its size within 16 segments", text)
        pert = next(f for f in data["extra_figs"] if "Perturbing" in f["title"])
        self.assertIn("1.00 / 0.30", pert["html"])


class LongerTraining(unittest.TestCase):
    """E4: the arms, the two rates (thresholds fixed before the runs), the switch-on step, the diff on every row, runs still in training."""

    BASE = ["model.segment=32", "model.pass_output=1"]

    def mk(self, extra, seed, pk256, pk4096, switch=None, **kw):
        hist = [{"step": st, "val/n256/passkey_acc": (1.0 if switch is not None and st >= switch else 0.1), "val/n256/ar_valid_acc": 0.99, "val/n4096/passkey_acc": 0.05, "val/n4096/ar_valid_acc": 0.99} for st in (2000, 4000, 6000, 8000)]
        return fake_run("swa_2pass.yaml", [*self.BASE, *extra], seed, {32: 1.0, 256: pk256, 1024: pk4096, 4096: pk4096, 16384: 0.0}, hist=hist, **kw)

    def setUp(self):
        # arm 50k: 6 seeds: learn (>= 0.9 at N = 256): seeds 0, 2, 3; extrapolate (>= 0.5 at N = 4096): seed 0 only
        v50 = [(1.0, 0.7, 4000), (0.6, 0.0, None), (0.95, 0.1, 6000), (0.9, 0.0, 8000), (0.3, 0.0, None), (0.89, 0.49, None)]
        self.a50 = [self.mk([], s, a, b, sw) for s, (a, b, sw) in enumerate(v50)]
        # arm 150k: 6 seeds: learn: all but seed 4; extrapolate: seeds 0, 1, 2, 5
        v150 = [(1.0, 0.8, 2000), (1.0, 0.6, 4000), (1.0, 0.5, 4000), (0.99, 0.2, 6000), (0.5, 0.0, None), (1.0, 0.9, 2000)]
        self.a150 = [self.mk(["train.steps=150000"], s, a, b, sw) for s, (a, b, sw) in enumerate(v150)]
        self.cont = [self.mk(["train.steps=50000", "train.init_from=runs/swa_2pass_seg32_po1_s1/model.pt"], 1, 0.95, 0.55, 2000)]
        self.other = [fake_run("swa_2pass.yaml", ["model.segment=64", "model.pass_output=1"], 0, {256: 1.0, 4096: 1.0}), fake_run("swa_2pass.yaml", ["model.segment=32", "model.pass_output=2"], 0, {256: 1.0, 4096: 1.0}),
                      fake_run("swa_2pass.yaml", [*self.BASE, "train.loop_lr_mult=1"], 0, {256: 1.0, 4096: 1.0})]
        self.runs = [*self.a50, *self.a150, *self.cont, *self.other]

    def test_runs_are_sorted_into_arms_and_other_runs_are_left_out(self):
        arms = R2.e4_runs(self.runs, {})
        self.assertEqual([r["seed"] for r in arms["50k"]["done"]], [0, 1, 2, 3, 4, 5])
        self.assertEqual([r["seed"] for r in arms["150k"]["done"]], [0, 1, 2, 3, 4, 5])
        self.assertEqual([r["name"] for r in arms["cont"]["done"]], ["swa_2pass_seg32_po1_steps50000_s1"])
        self.assertEqual({R2.e4_arm(r) for r in self.other}, {None})  # segment 64, output 2 and a changed gain learning rate are not E4 runs

    def test_the_rates_use_the_thresholds_fixed_in_advance(self):
        figs, sentences = R2.longer_training(self.runs, {})
        self.assertEqual(sentences[0], "50k steps: 3 of 6 finished seeds learn the training range, 1 of 6 extrapolate")   # 0.89 at 256 and 0.49 at 4096 do not count
        self.assertEqual(sentences[1], "150k steps: 5 of 6 finished seeds learn the training range, 4 of 6 extrapolate")
        self.assertIn("1 of 1 finished seeds learn the training range, 1 of 1 extrapolate", sentences[2])
        table = figs[0]["html"]
        self.assertIn("<b>50k steps: 3 of 6 seeds learn the training range, 1 of 6 extrapolate</b>", table)
        self.assertIn("<b>150k steps: 5 of 6 seeds learn the training range, 4 of 6 extrapolate</b>", table)

    def test_every_run_row_has_its_diff_its_values_at_all_n_and_its_switch_on_step(self):
        figs, _ = R2.longer_training(self.runs, {})
        table = figs[0]["html"]
        self.assertIn("output 1, 150k steps", table)              # the 150k rows say what differs from the reference ...
        self.assertIn("<td style='text-align:left'>output 1</td>", table)   # ... and so does every row (the 50k arm differs only in the output pass)
        self.assertIn("output 1, from seg32_po1_s1", table)       # the continuation names the weights it started from
        for n in ("<th>32</th>", "<th>2,048</th>", "<th>8,192</th>", "<th>16,384</th>"):
            self.assertIn(n, table)                               # all ten stream lengths, one column each
        self.assertEqual(R2.compact_diff({"model.pass_output": [2, 1], "train.steps": [50000, 150000], "train.lr": [0.001, 0.002]}), "output 1, 150k steps, lr 0.001 → 0.002")
        self.assertIn("4,000", table)                             # seed 0 of the 50k arm switches on at the validation at step 4,000
        self.assertIn("never", table)                             # seeds that never reach 0.9 at N = 256 in validation
        self.assertEqual(R2.switch_on_step(self.a50[2], {}), 6000)
        self.assertIsNone(R2.switch_on_step(self.a50[4], {}))

    def test_runs_still_in_training_are_listed_and_not_counted(self):
        logs = {"swa_2pass_seg32_po1_steps150000_s6": {2000: {256: {"pk": 0.2, "v": 0.99, "e": 1.0}, 4096: {"pk": 0.0, "v": 0.99, "e": 1.0}}, 4000: {256: {"pk": 0.9, "v": 0.99, "e": 1.0}, 4096: {"pk": 0.1, "v": 0.99, "e": 1.0}}}}
        figs, sentences = R2.longer_training(self.runs, logs)
        self.assertIn("(1 more still training)", sentences[1])
        only_running = R2.longer_training([r for r in self.runs if R2.e4_arm(r) != "150k"], logs)[1]
        self.assertIn("150k steps: no seed finished yet (1 still training)", only_running)  # an arm without a finished seed is not given a rate
        self.assertIn("still training", figs[0]["html"])
        self.assertIn("seed 6 (step 4,000)", figs[0]["html"])
        self.assertEqual([f["key"] for f in figs], ["e4_table", "e4_c256", "e4_c4096"])

    def test_the_fisher_test_matches_known_values(self):
        self.assertAlmostEqual(R2.fisher_two_sided(5, 6, 6, 6), 1.0)          # 5/6 against 6/6
        self.assertAlmostEqual(R2.fisher_two_sided(2, 6, 1, 6), 1.0)          # 2/6 against 1/6
        self.assertAlmostEqual(R2.fisher_two_sided(6, 6, 0, 6), 2 / 924, places=9)   # 6/6 against 0/6: 2 of the C(12,6) = 924 tables are this extreme
        self.assertAlmostEqual(R2.fisher_two_sided(3, 6, 3, 6), 1.0)

    def test_the_paragraph_says_what_the_data_say(self):
        text = R2.e4_paragraph(self.runs, {})
        self.assertIn("3 of 6 seeds at 50k steps, 5 of 6 at 150k", text)          # learning, same numbers as the table
        self.assertIn("1 of 6 against 4 of 6", text)                              # extrapolation
        self.assertIn(f"p = {R2.fisher_two_sided(3, 6, 5, 6):.2f}", text)
        self.assertIn("cannot be told apart", text)
        self.assertIn("seed 1: passkey 0.95 at N = 256 (was 0.60)", text)         # the continuation next to the run it continues
        self.assertEqual(R2.e4_paragraph([r for r in self.runs if R2.e4_arm(r) != "150k"], {}), "")  # nothing without both arms
        none_extrapolate = [r for r in self.runs if R2.e4_arm(r) != "150k"] + [self.mk(["train.steps=150000"], s, 1.0, 0.0, 2000) for s in range(3)]
        self.assertIn("p = ", R2.e4_paragraph(none_extrapolate, {}))              # an arm in which no seed extrapolates does not break the text

    def test_best_checkpoints_are_rows_picked_on_validation_and_reported_on_test(self):
        runs = list(self.runs)
        # make seed 3 of the 50k arm best on validation at 16,384 but not on test; seed 0 best on test but not on validation
        for r in runs:
            if R2.e4_arm(r) == "50k":
                for row in r["rows"]:
                    if row["N"] == 16384:
                        row["passkey_acc"] = {0: 0.9, 3: 0.2}.get(r["seed"], 0.0) if row["split"] == "test" else {0: 0.1, 3: 0.8}.get(r["seed"], 0.0)
                    if row["N"] == 4096:
                        row["passkey_acc"] = 0.3
        data, _, _ = R2.p2_data(runs, {}, {})
        self.assertEqual(data["rows"][0]["key"], "best_p2")
        self.assertIn("seed 3", data["rows"][0]["label"])                          # the validation winner
        self.assertAlmostEqual(data["results"]["best_p2"]["16384"]["passkey"], 0.2)    # reported on the test split
        self.assertIn("picked on the validation split", data["rows"][0]["long"])
        self.assertIn("Best checkpoints", " ".join(data["tldr"]))

    def test_the_report_carries_the_section_and_it_is_absent_without_e4_runs(self):
        data, _, _ = R2.p2_data(self.runs, {}, {})
        self.assertTrue(any(f["key"] == "e4_table" for f in data["extra_figs"]))
        self.assertIn("Longer training (E4", " ".join(data["tldr"]))
        data, _, _ = R2.p2_data([r for r in self.runs if R2.e4_arm(r) is None], {}, {})
        self.assertFalse(any(f["key"] == "e4_table" for f in data["extra_figs"]))
        self.assertEqual(R2.longer_training([], {}), ([], []))


if __name__ == "__main__":
    unittest.main()
