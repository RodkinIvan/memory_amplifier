"""Long heatmap row labels are wrapped onto two lines that fit the label column (report_groups.fit_label, applied by report.render for wide label columns).

    python -m unittest tests.test_report_labels -v
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from tests import _paths  # noqa: F401
import report as R
import report_groups as RG
import tests.test_report_2pass as T2  # (the modules, not their TestCase classes: those would be run a second time here)
import tests.test_report_capacity as TC

PX = 6.7  # px per character of the heatmap's row labels


def embedded(path: Path) -> dict:
    return json.loads(re.search(r"const DATA = (\{.*?\});\n", path.read_text(), re.S).group(1))


class FitLabel(unittest.TestCase):
    def test_a_label_that_fits_is_left_alone(self):
        self.assertEqual(RG.fit_label("two-pass, segment 32, mean of 3 seeds", 352), "two-pass, segment 32, mean of 3 seeds")

    def test_a_long_label_is_wrapped_at_spaces_into_lines_that_fit(self):
        text = "two-pass, segment 32, output of pass 2, gain 0.1 at the start, mean of 1 seed"
        out = RG.fit_label(text, 352)
        lines = out.split("\n")
        self.assertEqual(len(lines), 2)
        maxc = int((352 - 14) / PX)
        self.assertTrue(all(len(x) <= maxc for x in lines), lines)
        self.assertEqual(" ".join(lines), text)                       # nothing lost: only the break

    def test_two_lines_are_balanced_and_break_after_a_comma_or_colon_when_there_is_one(self):
        maxc = int((352 - 14) / PX)
        for text in ("two-pass, segment 32, output of pass 2, mean of 3 seeds", "two-pass, 50k steps, lr 0.001, curriculum, seed 0: P = 8",
                     "best two-pass checkpoint: segment 32, output of pass 1, seed 5, 50k steps", "two-pass, segment 32, output of pass 2, gain 0.1 at the start, mean of 1 seed"):
            a, b = RG.fit_label(text, 352).split("\n")
            self.assertTrue(a.endswith((",", ":")), (a, b))                       # a break after punctuation, never 'mean of 3 / seeds' or 'P = / 8'
            self.assertLessEqual(max(len(a), len(b)), maxc)
            self.assertLessEqual(max(len(a), len(b)), (len(text) + 1) // 2 + 16, (a, b))   # and not lopsided
            self.assertGreaterEqual(min(len(a), len(b)), 12, (a, b))                  # no orphan words
            self.assertEqual(a + " " + b, text)
        self.assertEqual(RG.fit_label("a" * 30 + " " + "b" * 30, 352), "a" * 30 + "\n" + "b" * 30)   # no punctuation: just two lines of the same length

    def test_more_than_two_lines_are_cut_with_an_ellipsis_and_one_long_word_too(self):
        out = RG.fit_label("word " * 60, 352)
        self.assertEqual(len(out.split("\n")), 2)
        self.assertTrue(out.endswith("…"))
        self.assertTrue(RG.fit_label("x" * 300, 352).endswith("…"))
        self.assertLessEqual(max(len(x) for x in RG.fit_label("x" * 300, 352).split("\n")), int((352 - 14) / PX))


class RenderedReports(unittest.TestCase):
    """The label column of the reports with long names: after render() every label is at most two lines that fit, nothing is cut, and the full name is the tooltip's."""

    def render(self, data):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.html"
            R.render(data, "t", "h", out)
            return embedded(out), out.read_text()

    def check(self, data):
        original = {r["key"]: r["label"] for r in data["rows"]}
        original_long = {r["key"]: r.get("long", r["label"]) for r in data["rows"]}
        got, html = self.render(data)
        maxc = int((got["label_w"] - 14) / PX)
        for r in got["rows"]:
            lines = r["label"].split("\n")
            self.assertLessEqual(len(lines), 2, r["label"])
            self.assertTrue(all(len(x) <= maxc for x in lines), (r["label"], maxc))
            self.assertNotIn("…", r["label"], "a label of these reports must fit without being cut")
            self.assertEqual(" ".join(lines), original[r["key"]])        # the wrapped label is the original label
            self.assertEqual(r["long"], original_long[r["key"]])                 # the tooltip keeps the full name
        return got, html

    def test_the_two_pass_report(self):
        t = T2.TwoPassReport()
        t.setUp()
        data, _, _ = t.build()
        got, _ = self.check(data)
        self.assertTrue(any("\n" in r["label"] for r in got["rows"]), "at least one label of this report is long enough to wrap")

    def test_the_capacity_report(self):
        t = TC.CapacityReport()
        t.setUp()
        self.check(t.build()[0])

    def test_a_narrow_label_column_is_not_touched(self):
        data = {"ns": [32], "rows": [{"key": "a", "label": "alpha = 0.95 with a long name that would wrap"}], "results": {}, "label_w": 124, "extra_figs": [], "tldr": []}
        got, _ = self.render(data)
        self.assertEqual(got["rows"][0]["label"], "alpha = 0.95 with a long name that would wrap")

    def test_the_template_draws_the_lines_and_keeps_tooltips_and_tables_on_one_line(self):
        html = R.TEMPLATE.read_text()
        self.assertIn("String(row.label).split('\\n')", html)
        self.assertIn("flat(row.long || row.label)", html)
        self.assertIn("flat(row.label), n, fmt2", html)


if __name__ == "__main__":
    unittest.main()
