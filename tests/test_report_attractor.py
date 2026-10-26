"""The attractor report (report_attractor): every number in the text comes from the result files, the helpers do what they say (synthetic results, no GPU).

    python -m unittest tests.test_report_attractor -v
"""
import json
import re
import unittest

from tests import _paths  # noqa: F401
import report_attractor as RA

KS = RA.KS


def curve(n, start, floor, rate):
    return [floor + (start - floor) * (rate ** t) for t in range(1, n + 1)]


def fake(name, P=4, s0=(1.0, 1.0, 1.0, 1.0, 0.9), star=0.0, same_end=0.003, diff_end=0.45, noise_exact=1.0, forgets=False):
    dec = lambda vals: {k: {"exact": v, "per_position": [v] * P} for k, v in zip(KS, vals)}
    stab = lambda start: {sg: {"distance_ratio": curve(64, 0.8, 0.0029 / float(sg), 0.5) if start == "s0" else curve(64, 1.3, 20.0, 0.9),
                               "wander": [1.4] * 64, "exact": {"4": noise_exact if float(sg) < 1 else 0.5, "32": noise_exact if float(sg) < 1 else 0.4, "512": s0[4]}}
                          for sg in ("0.01", "0.03", "0.1", "0.3", "1.0")}
    basin = lambda kind: [{"alpha": a, **{K: {"match_a": 1.0 if a < 0.45 else 0.0, "match_b": 0.0 if a < 0.45 else 1.0, "other": 0.0, "exact_a": 1.0 if a < 0.45 else 0.0, "exact_b": 0.0 if a < 0.45 else 1.0} for K in ("32", "512")}}
                          for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    return {"checkpoint": name, "passkey_len": P, "segment": 32, "window": 64, "kind": "swa_2pass",
            "record": {"git": {"commit": "a" * 40, "dirty": False}, "lr_scale": 0.1, "n_pass": 16, "n_ctx": 4},
            "optimization": {"objective_start": 3.8, "objective_star": 0.02, "moved_from_s0": {"per_layer": [1.0, 1.0]}, "rms": {"s0": [1.6, 6.3], "star": [0.15, 1.6]},
                             "error_kept_vs_new": {"start": {"kept": [2.0, 2.0], "new": [1.7, 1.8]}, "star": {"kept": [0.002, 0.0], "new": [0.03, 0.01]}},
                             "path": [{"step": 0, "objective": 3.8, "exact_k4": 1.0, "moved": 0.0}, {"step": 5, "objective": 1.8, "exact_k4": 0.58, "moved": 0.38},
                                      {"step": 10, "objective": 1.1, "exact_k4": 0.0, "moved": 0.63}, {"step": 3000, "objective": 0.02, "exact_k4": 0.0, "moved": 1.37}]},
            "sphere": {"objective_star": 0.45, "moved_from_s0": [0.9, 0.8]},
            "decoding": {"s0": dec(s0), "star": dec([star] * 5), "sphere": dec([0.0] * 5)},
            "stability": {"s0": stab("s0"), "star": stab("star")},
            "basins": {"s0": {"all_positions": basin("a"), "one_position": basin("b")}, "star": {"all_positions": basin("a"), "one_position": basin("b")}},
            "passkey_difference": {"s0": {"one_position": {"distance_ratio": [1.0] * 64}, "all_positions": {"distance_ratio": [1.1] * 64}}, "star": {"one_position": {"distance_ratio": [1.0] * 64}, "all_positions": {"distance_ratio": [1.0] * 64}}},
            "trajectory_convergence": {"t": [0, 1, 2, 4, 8, 16, 32, 64], "same_passkey": [1.2, 0.9, 0.3, 0.07, 0.01, 0.004, 0.003, same_end] if not forgets else [1.0, 0.5, 0.1] + [0.0] * 5,
                                       "different_passkey": [1.0, 0.8, 0.45, 0.43, 0.42, 0.44, 0.45, diff_end] if not forgets else [0.9, 0.4, 0.05] + [0.0] * 5}}


def results():
    return {RA.MAIN: fake(RA.MAIN), RA.E4_CONTROL: fake(RA.E4_CONTROL, s0=(1.0, 1.0, 1.0, 1.0, 0.0)),
            RA.PLAIN: fake(RA.PLAIN, s0=(0.8, 0.0, 0.0, 0.0, 0.0), noise_exact=0.0, forgets=True),
            "swa_2pass_seg32_po1_pk8_steps50000_lr0.001_warmup_steps500_cur2000_s0": fake("swa_2pass_seg32_po1_pk8_steps50000_lr0.001_warmup_steps500_cur2000_s0", 8, s0=(1.0, 1.0, 1.0, 1.0, 0.1)),
            "swa_2pass_seg32_po1_pk16_steps50000_lr0.001_warmup_steps500_cur2000_s1": fake("swa_2pass_seg32_po1_pk16_steps50000_lr0.001_warmup_steps500_cur2000_s1", 16, s0=(1.0, 1.0, 1.0, 0.9, 0.0), noise_exact=0.4)}


class Helpers(unittest.TestCase):
    def test_settle_is_the_first_segment_after_which_a_curve_stays_near_its_floor(self):
        c = [0.8, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1]
        self.assertEqual(RA.settle(c), 4)            # from segment 4 on it is within 25% of 0.1
        self.assertEqual(RA.settle([0.1] * 5), 1)

    def test_crossing_interpolates_between_grid_points(self):
        curve_ = [{"alpha": 0.0, "32": {"match_a": 1.0}}, {"alpha": 0.5, "32": {"match_a": 0.5}}, {"alpha": 1.0, "32": {"match_a": 0.0}}]
        self.assertAlmostEqual(RA.crossing(curve_, "match_a", 0.9), 0.1)
        self.assertAlmostEqual(RA.crossing(curve_, "match_a", 0.1), 0.9)
        self.assertIsNone(RA.crossing(curve_, "match_a", 2.0))

    def test_the_y_axis_top_is_a_multiple_of_0_4_so_that_the_ticks_are_round(self):
        mk = lambda v: [{"points": [{"y": v}]}]
        self.assertEqual(RA.y_max(mk(1.19)), 1.2)
        self.assertEqual(RA.y_max(mk(0.3)), 0.4)
        self.assertEqual(RA.y_max(mk(1.37)), 1.6)

    def test_checkpoints_are_ordered_and_named_for_tables(self):
        res = results()
        names = RA.order(res)
        self.assertEqual(names[:3], [RA.MAIN, RA.E4_CONTROL, RA.PLAIN])
        self.assertEqual([RA.row_name(n, res[n]) for n in names[3:]], ["capacity P = 8, seed 0", "capacity P = 16, seed 1"])


class Report(unittest.TestCase):
    def build(self, res=None):
        data, title, h1 = RA.report_data(res or results())
        return data, title, h1

    def text(self, data):
        return " ".join(re.sub(r"<[^>]+>", "", t) for t in data["tldr"])

    def test_the_page_has_no_grids_and_every_figure(self):
        data, title, _ = self.build()
        self.assertEqual(title, "Attractor Report")
        self.assertTrue(data["hide_heatmaps"] and data["hide_scatter"])
        keys = [f["key"] for f in data["extra_figs"]]
        for k in ("decode", "decode_table", "opt_table", "path", "noise", "noise_table", "basin", "convergence", "conv_table", "diff_table"):
            self.assertIn(k, keys)
        self.assertIn("a" * 7, data["meta"])

    def test_the_numbers_in_the_text_come_from_the_results(self):
        res = results()
        text = self.text(self.build(res)[0])
        self.assertIn("1.00 / 1.00 / 1.00 / 0.90 after K = 0 / 4 / 32 / 512", text)                  # the best checkpoint's read-out (K = 0 / 4 / 32 / 512)
        self.assertIn("from 3.80 to 0.020", text)                                                      # the objective
        self.assertIn("9% / 25% of the size of S0", text)                                              # 0.15 / 1.6 against 1.6 / 6.3
        self.assertIn("the passkey is already gone after 10 steps, when the objective is still 29% of its start", text)   # 1.1 / 3.8
        self.assertIn("the sphere) the objective cannot go below 0.45", text)
        self.assertIn("switches from A to B at α = 0.5", text)
        self.assertIn("0.003 after 64 segments (from 1.20)", text)
        res[RA.MAIN]["decoding"]["s0"]["32"]["exact"] = 0.77                                         # change a result: the text follows
        self.assertIn("1.00 / 1.00 / 0.77 / 0.90 after K = 0", self.text(self.build(res)[0]))

    def test_the_noise_floor_sentence_uses_the_curves(self):
        text = self.text(self.build()[0])
        self.assertRegex(text, r"the same absolute distance for all of them \(0\.0029, 0\.0029, 0\.0029, 0\.0029 of the state's norm")
        self.assertIn("noise of 1 x RMS drops it to 0.40", text)

    def test_the_control_and_the_capacity_checkpoints_are_summarised(self):
        text = self.text(self.build()[0])
        self.assertIn("reads the passkey with exact match 0.80 at K = 0", text)
        self.assertIn("0.000 apart after 4 segments", text)                                            # plain SWA forgets
        self.assertIn("1 stage of P = 8", text)
        self.assertIn("P = 16 (seed 1): 1.00 / 0.90 / 0.00", text)
        self.assertIn("never reads the passkey out with exact match above 0.00", text)

    def test_every_axis_label_says_what_is_plotted_and_fits_the_chart(self):
        """Axis labels are rotated along a plot 268 px high (about 38 characters at 6.8 px each): at most 36 characters per line and two lines; x labels at most two lines of 80 characters;
        no label is the bare word 'share'; every chart has a formula or definition in its subtitle or caption."""
        data, _, _ = self.build()
        charts = [f for f in data["extra_figs"] if f.get("spec")]
        self.assertEqual(len(charts), 5)
        for f in charts:
            yl, xl = [f["spec"]["yLabel"]], [f["spec"]["xLabel"]]
            yl, xl = [x for v in yl for x in ([v] if isinstance(v, str) else v)], [x for v in xl for x in ([v] if isinstance(v, str) else v)]
            self.assertLessEqual(len(yl), 2, f["key"])
            self.assertLessEqual(len(xl), 2, f["key"])
            self.assertTrue(all(len(x) <= 36 for x in yl), (f["key"], yl))
            self.assertTrue(all(len(x) <= 80 for x in xl), (f["key"], xl))
            self.assertNotIn(" ".join(yl).strip().lower(), ("share", "share of initial distance"), f["key"])
            self.assertRegex(f["sub"] + f["caption"], r"=|‖|argmax", f["key"])

    def test_the_method_says_that_the_inputs_keep_the_datasets_grid(self):
        data, _, _ = self.build()
        self.assertIn("keeps the dataset's own grid of three-token units", data["intro"])
        self.assertIn("one token for P = 8", data["intro"])

    def test_the_basin_figure_says_which_positions_are_counted(self):
        data, _, _ = self.build()
        f = next(f for f in data["extra_figs"] if f["key"] == "basin")
        self.assertIn("The positions where A and B agree are not counted", f["sub"])
        self.assertIn("A = 1, B = 0 means that single token is read out as A's", f["caption"])
        self.assertIn("the plotted fractions equal the exact matches", f["caption"])        # the fixture reads the shared tokens right at every alpha
        self.assertTrue(any("the 1 token that differs" in s["label"] for s in f["spec"]["series"]))

    def test_the_weaker_stage_is_named_in_the_noise_caption(self):
        data, _, _ = self.build()
        cap = next(f for f in data["extra_figs"] if f["key"] == "noise")["caption"]
        self.assertIn("capacity P = 16, seed 1", cap)
        self.assertNotIn("capacity P = 8", cap)

    def test_a_report_without_the_best_checkpoint_still_builds(self):
        res = results()
        del res[RA.MAIN]
        data, _, _ = self.build(res)
        keys = [f["key"] for f in data["extra_figs"]]
        self.assertNotIn("path", keys)
        self.assertIn("decode_table", keys)

    def test_nothing_to_report_without_results(self):
        with self.assertRaises(ValueError):
            RA.report_data({})

    def test_the_committed_results_build_a_report(self):
        res = RA.load()
        if not res:
            self.skipTest("no results/probes/p2_attractor_*.json")
        data, _, _ = RA.report_data(res)
        self.assertGreaterEqual(len(data["extra_figs"]), 8)
        self.assertEqual(json.loads(json.dumps(data))["hide_heatmaps"], True)


if __name__ == "__main__":
    unittest.main()
