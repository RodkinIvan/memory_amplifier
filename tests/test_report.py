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


if __name__ == "__main__":
    unittest.main()
