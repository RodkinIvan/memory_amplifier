"""The capacity grid (src/capacity_grid.py, scripts/capacity_grid.py): which cells and jobs there are, that no two runs share a name (and none collides with an earlier chain), what a chain says
about the capacity of its cell, how jobs are handed to local and remote slots, and that the model of every cell is the plain sliding-window model at gain 0.

    python -m unittest tests.test_capacity_grid -v
"""
import importlib.util
import json
import unittest

import torch

from tests import _paths
import capacity_grid as G
import config as C
from models import build_swa_2pass

spec = importlib.util.spec_from_file_location("capacity_grid_script", _paths.ROOT / "scripts" / "capacity_grid.py")
GRID = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GRID)
PC = GRID.PC


def stage(P: int, exact: float) -> dict:
    return {"passkey_len": P, "validation": {"exact": {"256": exact}}}


class Cells(unittest.TestCase):
    def test_nine_cells_with_segment_at_most_window(self):
        cells = G.cells()
        self.assertEqual(len(cells), 9)
        self.assertEqual(set(cells), {(s, w) for s in (8, 16, 32) for w in (32, 64, 128)})
        self.assertTrue(all(s <= w for s, w in cells))
        self.assertIn((32, 32), cells)   # segment = window is a cell

    def test_jobs_are_seed_major_and_the_slow_cells_come_first(self):
        jobs = G.jobs()
        self.assertEqual(len(jobs), 27)
        self.assertEqual([g for _, _, g in jobs], [0] * 9 + [1] * 9 + [2] * 9)   # the heatmap is complete after one seed
        first = jobs[:9]
        self.assertEqual(first[0], (8, 128, 0))   # smallest segment, largest window first
        self.assertEqual([s for s, _, _ in first], sorted(s for s, _, _ in first))

    def test_the_base_of_the_32_64_cell_is_the_e4_run_and_the_others_have_their_own_names(self):
        self.assertEqual([G.base_name(32, 64, g) for g in G.SEEDS], [f"swa_2pass_seg32_po1_s{g}" for g in G.SEEDS])
        self.assertEqual(G.base_name(8, 128, 1), "swa_2pass_seg8_po1_w128_s1")
        self.assertEqual(G.base_name(16, 32, 2), "swa_2pass_seg16_po1_w32_s2")

    def test_the_base_config_of_the_32_64_cell_is_the_e4_config(self):
        """Same config (evaluation settings aside) as the stored E4 result, so the E4 run counts as this cell's base and is not trained again."""
        r = json.loads((_paths.ROOT / "results" / "raw" / "swa_2pass_seg32_po1_s0.json").read_text())
        cfg, _ = C.load_config(G.CONFIG, G.base_overrides(32, 64, 0))
        diff = {k: v for k, v in C.config_difference(r["config"], cfg).items() if not C.is_ignored(k)}
        self.assertEqual(diff, {})

    def test_every_run_name_is_unique_and_new(self):
        """27 bases and their stages P = 8 ... 64: no two share a name, and no stage has the name of a stage of an earlier chain (those started from another base, review K1)."""
        names = []
        for s, w, g in G.jobs():
            names.append(G.base_name(s, w, g))
            for P in (8, 16, 32, 64):
                names.append(PC.stage_name(P, G.base_weights(s, w, g), G.RECIPE["steps"], G.RECIPE["lr"], G.FT_SEED_OFFSET + g, G.RECIPE["warmup"], G.RECIPE["curriculum"], "swa_2pass", s,
                                           None if w == G.yaml_window() else w))
        self.assertEqual(len(names), len(set(names)))
        earlier = {e["run"] for k, v in json.loads((_paths.ROOT / "results" / "capacity.json").read_text()).items() if k.startswith(("stages", "budget")) for e in v}
        stages = {n for n in names if "_pk" in n}
        self.assertFalse(stages & earlier, stages & earlier)

    def test_window_is_only_overridden_when_it_is_not_the_files(self):
        self.assertFalse(any(o.startswith("model.window") for o in G.base_overrides(32, 64, 0)))
        self.assertIn("model.window=128", G.base_overrides(32, 128, 0))
        self.assertFalse(any(o.startswith("model.window") for o in PC.stage_overrides(8, "x.pt", 10, 1e-3, model="swa_2pass")))
        self.assertIn("model.window=32", PC.stage_overrides(8, "x.pt", 10, 1e-3, model="swa_2pass", window=32))

    def test_the_chain_recipe_is_the_one_of_the_earlier_two_pass_chains(self):
        r = json.loads((_paths.ROOT / "results" / "raw" / "swa_2pass_seg32_po1_pk8_steps50000_lr0.001_warmup_steps500_cur2000_s0.json").read_text())["config"]["train"]
        self.assertEqual((r["steps"], r["lr"], r["warmup_steps"], r["curriculum_stage_steps"]), (G.RECIPE["steps"], G.RECIPE["lr"], G.RECIPE["warmup"], G.RECIPE["curriculum"]))


class CapacityOf(unittest.TestCase):
    def test_largest_passed_stage_before_the_first_failure(self):
        self.assertEqual(G.capacity_of([stage(8, 0.99), stage(16, 0.95), stage(32, 0.3)], 0.99)["tokens"], 16)
        self.assertEqual(G.capacity_of([stage(8, 0.99), stage(16, 0.3), stage(32, 0.95)], 0.99)["tokens"], 8)   # a pass after a failure does not count (the chain stops at the failure anyway)

    def test_first_stage_failed_gives_4_when_the_base_passes_and_0_when_it_does_not(self):
        self.assertEqual(G.capacity_of([stage(8, 0.1)], 0.97)["tokens"], 4)
        self.assertEqual(G.capacity_of([stage(8, 0.1)], 0.5)["tokens"], 0)
        self.assertEqual(G.capacity_of([stage(8, 0.1)], None)["tokens"], 0)

    def test_a_base_below_the_threshold_does_not_stop_a_chain_that_passes(self):
        """The earlier E4 seed 1 had a base at 0.51 and then passed P = 8 and 16: the capacity is what the stages show."""
        c = G.capacity_of([stage(8, 0.97), stage(16, 0.96), stage(32, 0.0)], 0.507)
        self.assertEqual((c["tokens"], c["done"]), (16, True))

    def test_done_means_a_failed_stage_or_the_last_length_passed(self):
        self.assertFalse(G.capacity_of([stage(8, 0.99)], 0.99)["done"])
        self.assertTrue(G.capacity_of([stage(8, 0.99), stage(16, 0.1)], 0.99)["done"])
        top = G.capacity_of([stage(8, 0.99), stage(16, 0.99), stage(32, 0.99), stage(64, 0.95)], 0.99)
        self.assertEqual((top["tokens"], top["censored"], top["done"]), (64, True, True))   # at least 64: the chain does not go on
        self.assertFalse(G.capacity_of([stage(8, 0.99), stage(16, 0.99), stage(32, 0.99)], 0.99)["censored"])

    def test_threshold_is_inclusive(self):
        self.assertEqual(G.capacity_of([stage(8, 0.9), stage(16, 0.8999)], 0.99)["tokens"], 8)


class Slots(unittest.TestCase):
    def test_parse_slots(self):
        self.assertEqual(GRID.parse_slots(["local:1:2", "10.0.0.1:0:3"]), [("local", 1), ("10.0.0.1", 0), ("local", 1), ("10.0.0.1", 0), ("10.0.0.1", 0)])   # round-robin over the GPUs

    def test_a_remote_slot_never_gets_the_local_only_cell_and_a_local_slot_takes_it_first(self):
        pool = GRID.Pool([(8, 128, 0), (32, 64, 0), (32, 64, 1), (16, 32, 0)])
        self.assertEqual(pool.take(remote=True), (8, 128, 0))
        self.assertEqual(pool.take(remote=False), (32, 64, 0))   # local slots: the local-only cell before the others
        self.assertEqual(pool.take(remote=False), (32, 64, 1))
        self.assertEqual(pool.take(remote=False), (16, 32, 0))   # then whatever is left, in order
        self.assertIsNone(pool.take(remote=False))
        self.assertIsNone(pool.take(remote=True))

    def test_every_job_is_given_out_once(self):
        pool = GRID.Pool(G.jobs())
        got = [pool.take(remote=i % 2 == 0) for i in range(27)]
        self.assertEqual(sorted(got), sorted(G.jobs()))
        self.assertIsNone(pool.take(remote=False))

    def test_the_local_only_cell_is_the_one_with_the_e4_bases(self):
        self.assertTrue(GRID.local_only(32, 64))
        self.assertFalse(any(GRID.local_only(s, w) for s, w in G.cells() if (s, w) != (32, 64)))


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class EveryCellIsPlainSwaAtGainZero(unittest.TestCase):
    def test_zero_gain_equals_plain_swa_in_every_cell(self):
        """The two-pass model of each cell, at gain 0, gives the logits of the plain sliding-window model of that window (segment < window, = window)."""
        torch.manual_seed(0)
        for s, w in G.cells():
            m = build_swa_2pass(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=w, segment=s).cuda().eval()
            x = torch.randint(8, 40, (2, 300), device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                diff = (m(input_ids=x).logits.float() - m.base(input_ids=x).logits.float()).abs().max().item()
            self.assertLess(diff, 5e-2, (s, w, diff))


if __name__ == "__main__":
    unittest.main()
