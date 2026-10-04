"""The report builder, run on the committed Exp 1 results (CPU only).

    python -m unittest discover -s tests -v
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from tests import _paths  # noqa: F401
import report as R


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
                self.assertTrue(xs and set(xs) <= set(f["spec"]["xTicks"]), (f["title"], s["label"]))  # every point sits on a tick of its own chart
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
