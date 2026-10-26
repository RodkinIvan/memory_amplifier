"""The capacity-grid page (src/report_capacity_grid.py): the numbers on it are computed from the chains, partial results are marked as such, and nothing is invented for a cell
that has no result.

    python -m unittest tests.test_report_capacity_grid -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from tests import _paths
import capacity_grid as G
import report_capacity_grid as R


def stage(P, exact, ok=None):
    return {"passkey_len": P, "validation": {"exact": {"256": exact}, "positions_ok": {"256": P if ok is None else ok}}}


def write_base(root: Path, s, w, g, exact):
    f = root / "results" / "raw" / f"{G.base_name(s, w, g)}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"rows": [{"split": "validation", "N": 256, "passkey_acc": exact}]}))


def fake(root: Path, caps: dict) -> dict:
    """caps[(s, w)] = [capacity of seed 0, 1, 2] in tokens (None: no result at all, 'run': base trained, chain unfinished)."""
    grid = {}
    for (s, w), vals in caps.items():
        for g, v in enumerate(vals):
            if v is None:
                continue
            write_base(root, s, w, g, 0.98)
            if v == "run":
                stages = [stage(8, 0.99)]
            elif v < 8:
                stages = [stage(8, 0.1, ok=5)]
            else:
                stages = []
                for P in (8, 16, 32, 64):
                    stages.append(stage(P, 0.97 if P <= v else 0.1, ok=P if P <= v else P - 3))
                    if P > v:
                        break
            grid[G.chain_key(s, w, g)] = stages
    return grid


class Page(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, caps):
        grid = fake(self.root, caps)
        return R.report_data(grid, self.root), grid

    def test_cell_summary_uses_only_finished_chains(self):
        grid = fake(self.root, {(8, 32): [8, 16, "run"]})
        sm = R.cell_summary(R.states(grid, self.root)[(8, 32)])
        self.assertEqual((sm["caps"], sm["done"], sm["running"], sm["pending"]), ([8, 16], 2, 1, 0))
        self.assertEqual(sm["median"], 12)

    def test_an_empty_grid_builds_and_says_nothing_has_finished(self):
        (data, title, h1), _ = self.build({})
        self.assertEqual(title, "Capacity Grid Report")
        self.assertIn("0 of 27 chains are finished", data["tldr"][0])
        html = data["extra_figs"][0]["html"]
        self.assertEqual(html.count("…"), 9)             # nine cells, none with a median
        self.assertNotIn("Capacity per cell", " ".join(data["tldr"]))

    def test_the_heatmap_shows_the_median_and_the_seeds_values(self):
        caps = {(s, w): [8, 16, 8] for s in G.SEGMENTS for w in G.WINDOWS}
        caps[(8, 128)] = [32, 32, 16]
        (data, _, _), _ = self.build(caps)
        html = data["extra_figs"][0]["html"]
        self.assertIn("<div class='big'>32</div>", html)       # the median of 32, 32, 16
        self.assertIn("32 · 32 · 16", html)
        self.assertIn("27 of 27 chains are finished", data["tldr"][0])
        self.assertIn("the largest is 32 tokens at segment 8, window 128", " ".join(data["tldr"]))

    def test_a_chain_stopped_at_the_top_is_marked_as_a_lower_bound(self):
        grid = {}
        write_base(self.root, 16, 128, 0, 0.99)
        grid[G.chain_key(16, 128, 0)] = [stage(P, 0.99) for P in (8, 16, 32, 64)]
        sm = R.cell_summary(R.states(grid, self.root)[(16, 128)])
        self.assertTrue(sm["censored"])
        self.assertIn(">=64", sm["labels"])
        self.assertIn("64+", R.heatmap_html({(16, 128): sm}))

    def test_first_stage_failed_is_4_when_the_base_passes_and_below_4_when_it_does_not(self):
        grid = fake(self.root, {(32, 32): [4, 4, 4]})
        write_base(self.root, 32, 32, 2, 0.2)
        sm = R.cell_summary(R.states(grid, self.root)[(32, 32)])
        self.assertEqual(sm["caps"], [4, 4, 0])
        self.assertEqual(sm["labels"], ["4", "4", "<4"])

    def test_trend_sentences_only_use_cells_with_all_three_seeds(self):
        caps = {(8, 32): [8, 8, 8], (8, 64): [16, 16, 16], (8, 128): [16, 16, "run"]}
        (data, _, _), grid = self.build(caps)
        text = " ".join(data["tldr"])
        self.assertIn("segment 8: window 32: 8, window 64: 16", text.replace("<b>Window.</b> ", ""))
        self.assertNotIn("window 128: 16", text)

    def test_pass_rates_count_the_chains_that_reached_a_stage(self):
        caps = {(8, 64): [8, 16, 16]}
        grid = fake(self.root, caps)
        pr = R.pass_rates(R.states(grid, self.root)[(8, 64)])
        self.assertEqual(pr[8], (3, 3))
        self.assertEqual(pr[16], (2, 3))        # seed 0 failed at 16, the other two passed it
        self.assertEqual(pr[32], (0, 2))        # the two that passed 16 failed 32

    def test_figure_numbers_are_substituted(self):
        (data, _, _), _ = self.build({(8, 32): [8, 8, 8]})
        self.assertNotIn("{fig:", json.dumps(data))
        self.assertTrue(data["extra_figs"][0]["title"].startswith("1. "))

    def test_the_real_results_build(self):
        data, title, _ = R.report_data()
        self.assertTrue(data["extra_figs"])


if __name__ == "__main__":
    unittest.main()
