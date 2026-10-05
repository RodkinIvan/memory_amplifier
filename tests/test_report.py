"""The report builder, run on the committed Exp 1 results (CPU only).

    python -m unittest discover -s tests -v
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from tests import _paths  # noqa: F401
import config as C
import report as R
import report_groups as RG
import report_rounds as RR


class FbReport(unittest.TestCase):
    """The feedback-loop report builder on the committed results (CPU only)."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.title, cls.h1 = R.fb_data()

    def test_rows_have_every_stream_length_and_the_reference_comes_first(self):
        self.assertEqual(self.data["rows"][0]["label"], "SWA, no loop")
        for row in self.data["rows"]:
            self.assertEqual(sorted(map(int, self.data["results"][row["key"]])), R.NS, row["label"])

    def test_figures_have_a_chart_or_html_and_every_series_covers_all_n(self):
        for f in self.data["extra_figs"]:
            self.assertTrue(f["spec"] or f.get("html"), f["title"])
            for s in (f["spec"] or {"series": []})["series"]:
                xs = [p["x"] for p in s["points"]]
                self.assertTrue(xs, (f["title"], s["label"]))
                if f["spec"].get("xLog") is not False:  # log-scale charts: every point sits on a tick of its own chart (training-step charts have a linear axis)
                    self.assertTrue(set(xs) <= set(f["spec"]["xTicks"]), (f["title"], s["label"]))
                if f["spec"]["xTicks"] == R.NS:  # the charts over stream length cover every N
                    self.assertEqual(xs, R.NS, s["label"])

    def test_text_is_derived_without_placeholders(self):
        text = " ".join(self.data["tldr"]) + self.data["cap1"] + self.data["cap2"]
        for bad in ("NaN", "None", "nan"):
            self.assertNotIn(bad, text.replace("finance", ""), bad)
        self.assertGreaterEqual(len(self.data["tldr"]), 6)

    def test_only_runs_of_the_recipe_gain_are_in_the_rows(self):
        """Assert on the runs that went into each row: all have the recipe's loop gain (read from configs/swa_fb.yaml), and the per-seed
        lists of the row have one entry per run. The control row is plain SWA (no loop)."""
        raw = {r["name"]: r for r in R.A.load_runs(R.C.ROOT / "results" / "raw")}
        gain = self.data["recipe_gain"]
        for row in self.data["rows"]:
            names = self.data["row_runs"][row["key"]]
            self.assertTrue(names, row["label"])
            for nm in names:
                cfg = raw[nm]["cfg"]["model"]
                if row["key"] == "ref":
                    self.assertEqual(cfg["kind"], "swa", nm)
                else:
                    self.assertEqual((cfg["kind"], cfg["gain_init"]), ("swa_fb", gain), nm)
            self.assertEqual(len(self.data["results"][row["key"]]["32"]["per"]["passkey"]), len(names), row["label"])

    def test_every_recipe_run_is_in_some_row(self):
        raw = R.A.load_runs(R.C.ROOT / "results" / "raw")
        gain = self.data["recipe_gain"]
        recipe = {r["name"] for r in raw if r["cfg"]["model"]["kind"] == "swa_fb" and r["cfg"]["model"]["gain_init"] == gain}
        in_rows = {nm for k, names in self.data["row_runs"].items() if k != "ref" for nm in names}
        self.assertEqual(in_rows, recipe)  # nothing dropped, nothing from another gain mixed in

    def test_renders_to_a_page(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.html"
            R.render(self.data, self.title, self.h1, out)
            self.assertIn("SWA Feedback Loop Report", out.read_text())

    def test_every_run_is_in_exactly_one_configuration_row_and_rows_state_their_difference(self):
        """Review H7: the appendix lists every swa / swa_fb run once, in the row of its configuration (same diff to the frozen reference)."""
        raw = [r for r in R.A.load_runs(R.C.ROOT / "results" / "raw") if r["cfg"]["model"]["kind"] in ("swa", "swa_fb")]
        appendix = self.data["extra_figs"][-1]
        self.assertIn("Every configuration", appendix["title"])
        rows = re.findall(r"<tr><td title='([^']*)'>(.*?)<br>", appendix["html"])
        listed = [nm for names, _ in rows for nm in names.split(", ")]
        self.assertEqual(sorted(listed), sorted(r["name"] for r in raw))  # nothing dropped, nothing twice
        labels = [lab for _, lab in rows]
        self.assertEqual(len(labels), len(set(labels)), "two rows with the same label would hide a difference")
        by_name = {r["name"]: r for r in raw}
        for names, lab in rows:
            cfgs = {json.dumps({k: v for k, v in C.flatten(by_name[nm]["cfg"]).items() if not C.is_ignored(k)} | {k: C.CODE_DEFAULTS[k] for k in C.CODE_DEFAULTS if k not in C.flatten(by_name[nm]["cfg"])}, sort_keys=True) for nm in names.split(", ")}
            self.assertEqual(len(cfgs), 1, lab)  # identical configs apart from the seed

    def test_figures_are_numbered_in_order_and_every_reference_resolves(self):
        titles = [f["title"] for f in self.data["extra_figs"]]
        self.assertEqual([int(t.split(".")[0]) for t in titles], list(range(3, 3 + len(titles))))
        self.assertNotIn("{fig:", json.dumps(self.data))
        keys = [f["key"] for f in self.data["extra_figs"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_rows_name_window_and_segment_and_say_no_tanh_for_the_loop_without_tanh(self):
        labels = [r["label"] for r in self.data["rows"][1:]]
        self.assertTrue(all(lab.startswith("window 64, segment ") for lab in labels), labels)
        self.assertTrue(any("no tanh" in lab for lab in labels))
        self.assertNotIn("linear", " ".join(self.data["tldr"]).replace("linear read-out", ""))

    def test_the_far_propagation_figure_states_numbers_that_are_in_the_data(self):
        far = self.data["extra_figs"][0]
        self.assertIn("Far propagation", far["title"])
        raw = {r["name"]: r for r in R.A.load_runs(R.C.ROOT / "results" / "raw")}
        best = raw["swa_fb_gi0_seg8_nlnone_s0"]
        pk = next(x for x in best["rows"] if x["split"] == "test" and x["N"] == 16384)["passkey_acc"]
        self.assertIn(f"passkey {pk:.2f} on test", far["html"])


class GdnReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.title, cls.h1 = R.gdn_data()

    def test_rows_and_seeds(self):
        self.assertEqual([r["key"] for r in self.data["rows"]], ["1", "0.999", "0.995", "0.99", "0.98", "0.95", "0.9", "0.8"])
        for key in ("1", "0.8"):
            for n in R.NS:
                per = self.data["results"][key][str(n)]["per"]
                self.assertEqual({k: len(v) for k, v in per.items()}, {"passkey": 3, "valid": 3, "stale": 3, "expired": 3})

    def test_means_are_means_of_the_seeds(self):
        c = self.data["results"]["1"]["2048"]
        self.assertAlmostEqual(c["passkey"], sum(c["per"]["passkey"]) / 3)
        self.assertIsNone(self.data["results"]["1"]["32"]["stale"])  # no stale queries at N = 32

    def test_horizon_uses_the_seed_mean(self):
        cells = self.data["results"]
        self.assertEqual(R.horizon(cells["0.999"]), 512)
        self.assertEqual(R.horizon(cells["0.99"]), 256)
        self.assertIsNone(R.horizon(cells["0.9"]))

    def test_text_claims_follow_the_data(self):
        text = " ".join(self.data["tldr"])
        self.assertIn("all 24 runs", text)
        self.assertIn("α = 1: 1,024", text)  # the seed-mean horizon of alpha = 1

    def test_render_fills_every_placeholder_and_embeds_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.html"
            R.render(self.data, self.title, self.h1, out)
            html = out.read_text()
        self.assertNotIn("__TITLE__", html)
        self.assertNotIn("__H1__", html)
        self.assertNotIn("/*__DATA__*/", html)
        embedded = json.loads(re.search(r"const DATA = (.*);\nconst NS", html, re.S).group(1))
        self.assertEqual(embedded["ns"], R.NS)
        self.assertEqual(self.title, "passkey_swar Intermediate Report")


DATASET = _paths.ROOT / "data_cache" / "passkey_swar" / "n1024" / "validation.parquet"


@unittest.skipUnless(DATASET.exists(), "needs the generated dataset (data_cache/passkey_swar)")
class SwaReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.title, cls.h1 = R.swa_data()

    def test_the_matched_control_and_other_recipes_are_not_merged_into_the_baseline_rows(self):
        """Plain-SWA runs with the curriculum or another step count are separate configurations (review H7); they must not join the window-64 row."""
        for cells in self.data["results"].values():
            self.assertEqual(len(cells["256"]["per"]["passkey"]), 3)
        ctrl = [r for r in R.A.load_runs(R.C.ROOT / "results" / "raw") if r["cfg"]["model"]["kind"] == "swa" and r["cfg"]["train"]["steps"] != 30000]
        self.assertTrue(ctrl)  # the control exists, and still the row has 3 seeds

    def test_rows_are_ordered_by_reach_and_have_three_seeds(self):
        self.assertEqual([r["key"] for r in self.data["rows"]], ["w128_L4", "w128_L2", "w64_L4", "w64_L2", "w32_L4", "w32_L2"])
        self.assertEqual(self.data["rows"][0]["label"], "W128 \u00b7 L4 \u00b7 R508")  # reach = layers * (window - 1)
        for key in ("w128_L4", "w32_L2"):
            self.assertEqual(len(self.data["results"][key]["256"]["per"]["passkey"]), 3)

    def test_ar_half_is_solved_and_swa_is_not_given_a_decay(self):
        for key, cells in self.data["results"].items():
            for n in R.NS[1:]:
                self.assertGreaterEqual(min(cells[str(n)]["per"]["stale"]), 0.99, (key, n))
        self.assertIn("SWA has no decay", self.data["results_intro"])
        self.assertEqual(self.title, "SWA Baseline Report")

    def test_probe_figures_have_one_series_per_configuration(self):
        len_fig, dens_fig = self.data["extra_figs"]
        self.assertEqual(len(len_fig["spec"]["series"]), 6)
        self.assertEqual(len(dens_fig["spec"]["series"]), 7)  # 6 configurations + GDN alpha = 1 for contrast
        self.assertTrue(dens_fig["spec"]["series"][-1]["dashed"])


if __name__ == "__main__":
    unittest.main()


def fake_run(config: str, overrides: list[str], seed: int, passkey: dict, valid: float = 0.99, hist=None) -> dict:
    """A loaded run (as aggregate.load_runs returns it) with the given passkey accuracy per N; everything else plausible."""
    cfg, keys = C.load_config(_paths.ROOT / "configs" / config, [*overrides, f"seed={seed}"])
    rows = [{"split": sp, "N": n, "passkey_acc": passkey.get(n, 0.0), "ar_valid_acc": valid, "ar_expired_acc": 1.0, "ar_expired_stale_acc": 1.0, "ar_expired_never_acc": 1.0,
             "n_examples": 1000, "n_valid": 100, "n_expired_stale": 100, "n_expired_never": 10} for sp in ("validation", "test") for n in R.NS]
    return {"name": C.run_name(cfg, keys), "seed": seed, "cfg": cfg, "rows": rows, "final_train_loss": 0.02, "n_params": 563584, "train_time_s": 7200.0, "val_history": hist, "model": "x", "alpha": float("nan")}


def history(pk256_by_step: dict, pk1024_by_step: dict) -> list[dict]:
    return [{"step": st, "val/n256/passkey_acc": pk256_by_step[st], "val/n256/ar_valid_acc": 0.99, "val/n1024/passkey_acc": pk1024_by_step[st], "val/n1024/ar_valid_acc": 0.99}
            for st in sorted(pk256_by_step)]


FB50 = ["model.gain_init=0", "train.steps=50000"]


class ReportRounds(unittest.TestCase):
    """The figure builders of the third review round, on synthetic runs."""

    def setUp(self):
        mk = lambda ov, seed, pk, hist=None: fake_run("swa_fb.yaml", ov, seed, pk, hist=hist)
        self.b50 = [mk([*FB50, "model.segment=32", "model.nonlin=none"], s, {256: 1.0, 1024: 0.6 + 0.1 * s, 4096: 0.3, 16384: 0.1 * s},
                       history({2000: 0.1, 4000: 0.9, 6000: 1.0}, {2000: 0.0, 4000: 0.3, 6000: 0.6})) for s in range(3)]
        self.b20 = [mk(["model.gain_init=0", "model.segment=32", "model.nonlin=none"], s, {256: 1.0, 1024: 0.5, 4096: 0.1, 16384: 0.0}) for s in range(3)]
        self.swa = {w: [fake_run("swa.yaml", ["model.n_layers=2", "train.steps=50000", "train.curriculum_stage_steps=2000", f"model.window={w}"], s, {32: 1.0, 64: 1.0 if w >= 64 else 0.0, 256: 1.0 if w >= 256 else 0.0})
                        for s in range(3)] for w in (64, 128, 256, 512)}
        self.runs = [*self.b50, *self.b20, *(r for rs in self.swa.values() for r in rs)]
        self.dgroups = RG.diff_groups(self.runs)
        self.std = {RG.standard_key(g): g for g in self.dgroups if RG.standard_key(g)}

    def test_every_run_is_in_one_group_and_groups_are_told_apart_by_their_difference(self):
        self.assertEqual(sorted(r["name"] for g in self.dgroups for r in g["runs"]), sorted(r["name"] for r in self.runs))
        self.assertEqual(len(self.dgroups), 6)  # 50k loop, 20k loop, four windows
        labels = {RG.label(g) for g in self.dgroups}
        self.assertEqual(len(labels), 6)
        self.assertIn("window 64, segment 32, no-tanh loop · gain_init 0.1 → 0, 20000 → 50000 steps", labels)

    def test_runs_that_differ_in_an_unlisted_setting_are_not_merged(self):
        odd = fake_run("swa_fb.yaml", [*FB50, "model.segment=32", "model.nonlin=none", "model.proj_std=0.5"], 3, {})
        groups = RG.diff_groups([*self.b50, odd])
        self.assertEqual(sorted(len(g["runs"]) for g in groups), [1, 3])
        self.assertIsNone(RG.standard_key(next(g for g in groups if len(g["runs"]) == 1)))  # not a row of the main grid
        self.assertIn("proj_std 0.1 → 0.5", " ".join(RG.label(g) for g in groups))

    def test_long_runs_table_and_charts(self):
        out = RR.long_runs(self.dgroups, self.std, {})
        keys = [f["key"] for f in out["figs"]]
        self.assertEqual(keys[:2], ["long", "long_table"])
        chart = out["figs"][0]["spec"]
        self.assertEqual(len(chart["series"]), 1 + 1 + 4)  # B 20k and 50k, four windows
        for sr in chart["series"]:
            self.assertEqual([p["x"] for p in sr["points"]], R.NS)
        table = out["figs"][1]["html"]
        self.assertIn("126 tokens", table)  # window 64: reach 2 layers x 63
        self.assertIn("1,022 tokens", table)  # window 512
        self.assertIn("0.60 / 0.70 / 0.80", table)  # per-seed passkey at N = 1024 of setup B
        self.assertEqual(keys, ["long", "long_table", "curves1024"])  # the stored history has no N = 4096, so no curves for it
        self.assertIn("50k", out["tldr"])

    def test_gdn2_trained_like_the_loop_models_joins_the_long_run_comparison(self):
        g2 = [fake_run("gdn2.yaml", ["train.steps=50000", "train.curriculum_stage_steps=2000"], s, {256: 1.0, 1024: 1.0, 4096: 1.0 if s == 2 else 0.0, 8192: 0.9 if s == 2 else 0.0}) for s in range(3)]
        groups = [*self.dgroups, *RG.diff_groups(g2)]
        out = RR.long_runs(groups, self.std, {})
        chart = out["figs"][0]["spec"]
        self.assertIn("GDN-2", [sr["short"] for sr in chart["series"]])
        self.assertEqual(len(chart["series"]), 1 + 1 + 4 + 1)  # B 20k and 50k, four windows, GDN-2
        self.assertIn("recurrent state", out["figs"][1]["html"])
        self.assertIn("GDN-2 (stock learned decay, same training, 563k parameters, 3 seeds)", out["tldr"].replace("441k", "563k"))
        self.assertIn("0.00 / 0.00 / 1.00", out["tldr"])  # per-seed passkey at N = 4,096

    def test_the_text_says_a_plain_window_does_not_extrapolate_only_when_the_data_show_it(self):
        mk = lambda w, pk: [fake_run("swa.yaml", ["model.n_layers=2", "train.steps=50000", "train.curriculum_stage_steps=2000", f"model.window={w}"], s, pk) for s in range(3)]
        base = [*self.b50, *self.b20]
        # window 1024 (reach 2,046 tokens = streams up to N = 512) holds the passkey up to N = 256 only: the sentence appears
        runs = [*base, *mk(1024, {32: 1.0, 64: 1.0, 128: 1.0, 256: 0.98})]
        groups = RG.diff_groups(runs)
        std = {RG.standard_key(g): g for g in groups if RG.standard_key(g)}
        text = RR.long_runs(groups, std, {})["tldr"]
        self.assertIn("does not extrapolate in length even inside its reach", text)
        self.assertIn("up to N = 512, but holds the passkey only up to N = 256", text)
        # the same window holding the passkey at N = 512 (beyond the training streams): no such claim
        runs = [*base, *mk(1024, {32: 1.0, 64: 1.0, 128: 1.0, 256: 1.0, 512: 0.95})]
        groups = RG.diff_groups(runs)
        std = {RG.standard_key(g): g for g in groups if RG.standard_key(g)}
        self.assertNotIn("does not extrapolate", RR.long_runs(groups, std, {})["tldr"])

    def test_validation_curves_come_from_the_stored_history_or_from_the_log(self):
        run = self.b50[0]
        self.assertEqual(RR.curve(run, {}, 1024), [(2000, 0.0), (4000, 0.3), (6000, 0.6)])
        self.assertEqual(RR.switch_on_step(run, {}), 4000)  # first evaluation with passkey above 0.5 at N = 256
        logged = {"swa_fb_x": {2000: {1024: {"pk": 0.2, "v": 0.9, "e": 1.0}}, 4000: {1024: {"pk": 0.4, "v": 0.9, "e": 1.0}}}}
        self.assertEqual(RR.curve({"name": "swa_fb_x", "val_history": None}, logged, 1024), [(2000, 0.2), (4000, 0.4)])

    def test_log_lines_are_parsed_for_every_stream_length(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sess").mkdir()
            (Path(d) / "sess" / "swa_fb_run.log").write_text(
                "[swa_fb_run] params=1 wandb=off\n[swa_fb_run] step 2000 loss 0.5000 | n32: pk 0.10 v 0.90 e 1.00 n256: pk 0.20 v 0.80 e 1.00 n1024: pk 0.30 v 0.70 e 1.00 n4096: pk 0.40 v 0.60 e 1.00\n"
                "[swa_fb_run] step 2000 loss 0.4000 | n32: pk 0.15 v 0.90 e 1.00\nother text\n")
            out = RR.log_curves(Path(d))
        self.assertEqual(out["swa_fb_run"][2000], {"loss": 0.4, 32: {"pk": 0.15, "v": 0.9, "e": 1.0}})  # a restarted run's later line replaces the earlier one

    def test_segment_study_needs_two_segments_and_reports_the_switch_on_step(self):
        mk = lambda seg, seed: fake_run("swa_fb.yaml", [*FB50, f"model.segment={seg}", "model.nonlin=none"], seed, {256: 1.0, 1024: 0.5, 16384: 0.2},
                                        hist=history({2000: 0.0, 4000: 0.8}, {2000: 0.0, 4000: 0.5}))
        runs = [mk(seg, s) for seg in (64, 8) for s in range(2)]
        std = {RG.standard_key(g): g for g in RG.diff_groups(runs)}
        out = RR.segment_study(std, {})
        (fig,) = out["figs"]
        self.assertIn("4,000", fig["caption"])  # switch-on step in the table
        self.assertEqual([p["x"] for p in fig["spec"]["series"][0]["points"]], [8, 64])
        self.assertIsNone(RR.segment_study({k: v for k, v in std.items() if k[0] == 64}, {}))

    def test_nothing_is_built_without_the_runs(self):
        self.assertIsNone(RR.long_runs([], {}, {}))
        self.assertIsNone(RR.far_propagation({}, [], 0.1, {}))
        fig = RR.configuration_appendix(self.dgroups)
        self.assertEqual(fig["html"].count("<tr><td title="), 6)
