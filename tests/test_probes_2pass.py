"""Diagnostics of the two-pass model (src/probes_2pass.py): the linear probe, the cache hooks, the perturbation probe (the model parts need a CUDA GPU).

    python -m unittest tests.test_probes_2pass -v
"""
import unittest
from unittest import mock

import numpy as np
import torch

from tests import _paths  # noqa: F401
import probes_2pass as PV
from models import build_swa_2pass

CUDA = torch.cuda.is_available()


class RidgeProbe(unittest.TestCase):
    def data(self, n, d, informative, seed=0, amplitude=3.0, spread=1):
        """Gaussian features; if `informative` the class is written into `spread` features of its own (16 * spread of the d features carry it)."""
        g = torch.Generator().manual_seed(seed)
        y = torch.randint(0, 16, (n,), generator=g)
        x = torch.randn(n, d, generator=g)
        if informative:
            x[:, :16 * spread] += amplitude * torch.nn.functional.one_hot(y, 16).repeat_interleave(spread, dim=1)
        return x, y

    def test_a_planted_class_is_found_and_noise_stays_at_chance(self):
        for informative, lo, hi in ((True, 0.65, 1.0), (False, 0.0, 0.14)):
            xt, yt = self.data(2000, 200, informative, 0)
            xe, ye = self.data(500, 200, informative, 1)
            r = PV.ridge_probe(xt, yt, xe, ye, 16)
            self.assertTrue(lo <= r["accuracy"] <= hi, (informative, r))

    def test_more_features_than_examples_and_the_ridge_is_chosen_without_the_test_set(self):
        xt, yt = self.data(400, 3000, True, 0, amplitude=1.5, spread=60)  # 3000 features, 400 examples; the class is spread over 960 features
        xe, ye = self.data(300, 3000, True, 1, amplitude=1.5, spread=60)
        r = PV.ridge_probe(xt, yt, xe, ye, 16)
        self.assertGreater(r["accuracy"], 0.8)
        self.assertGreater(r["heldout_accuracy"], 0.8)
        # shuffling the test labels destroys the score but cannot change the chosen lambda (it never sees them)
        r2 = PV.ridge_probe(xt, yt, xe, ye[torch.randperm(len(ye), generator=torch.Generator().manual_seed(5))], 16)
        self.assertEqual(r2["lambda"], r["lambda"])
        self.assertLess(r2["accuracy"], 0.2)


class CutPoints(unittest.TestCase):
    def test_a_cut_is_the_first_segment_end_at_or_after_the_distance(self):
        c = PV.cut_points([32, 96, 160, 256, 1024], 32, 4)  # the passkey ends at token 6 (stream starts at 6)
        self.assertEqual(c, {32: 1, 96: 3, 160: 5, 256: 8, 1024: 32})
        for d, s in c.items():
            end = (s + 1) * 32  # tokens in the cut stream
            self.assertGreaterEqual(end - 6, d)
            self.assertLess(end - 32 - 6, d)  # and no earlier segment would do
        self.assertEqual(PV.cut_points([100], 8, 8), {100: 13})  # passkey of 8: stream starts at 10 -> 110 tokens -> 14 segments of 8 = 112


class GainStats(unittest.TestCase):
    def test_per_layer_statistics_of_the_stored_gains(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d, mock.patch.object(PV.C, "ROOT", Path(d)):
            (Path(d) / "runs" / "r").mkdir(parents=True)
            torch.save({"loops.0.gain": torch.tensor([0.5, -0.5, 1.0, 0.0]), "loops.1.gain": torch.full((4,), 2.0), "base.x": torch.zeros(1)}, Path(d) / "runs" / "r" / "model.pt")
            r = PV.gain_stats("r")
        self.assertEqual(len(r["layers"]), 2)
        self.assertAlmostEqual(r["layers"][0]["mean"], 0.25)
        self.assertAlmostEqual(r["layers"][0]["abs_mean"], 0.5)
        self.assertAlmostEqual(r["layers"][0]["abs_max"], 1.0)
        self.assertAlmostEqual(r["layers"][0]["negative"], 0.25)
        self.assertEqual((r["layers"][1]["mean"], r["layers"][1]["negative"]), (2.0, 0.0))


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class TwoPassProbes(unittest.TestCase):
    """The same diagnostics on the two-pass model: cache hooks with keys / values / both, the window features, the perturbation probe end to end."""

    def model(self, W=16, C=8):
        torch.manual_seed(0)
        m = build_swa_2pass(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=W, segment=C).cuda().eval()
        for loop in m.loops:
            torch.nn.init.constant_(loop.gain, 0.5)
        return m

    def run_model(self, m, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(input_ids=x).logits.float()

    def test_keys_values_or_both_can_be_kicked_and_only_what_comes_after_moves(self):
        m = self.model()
        x = torch.randint(8, 40, (2, 64), device="cuda")
        clean = self.run_model(m, x)
        for which in ("v", "k", "kv"):
            for kw in ({"eps": 1.0}, {"mode": "zero"}):
                m.cache_hook = PV.CacheHook(s_star=3, seed=1, which=which, **kw)
                d = (self.run_model(m, x) - clean).abs().amax(-1)[0]
                self.assertEqual(float(d[:32].max()), 0.0, (which, kw))  # the kick lands after segment 3
                self.assertGreater(float(d[32:].max()), 0.0, (which, kw))
        m.cache_hook = None
        with self.assertRaises(ValueError):
            PV.CacheHook(which="q")

    def test_the_cache_hook_sees_the_window_and_changes_only_what_comes_after(self):
        m = self.model(16, 8)
        x = torch.randint(8, 40, (2, 64), device="cuda")
        clean = self.run_model(m, x)
        rec = PV.CacheHook(record_all=True)
        m.cache_hook = rec
        self.run_model(m, x)
        self.assertEqual(sorted(rec.snap), list(range(8)))
        k, v = rec.snap[7][0]
        self.assertEqual((tuple(k.shape), tuple(v.shape)), ((2, 15, 64), (2, 15, 64)))  # the last W - 1 tokens
        self.assertEqual(float((self.run_model(m, x) - clean).abs().max()), 0.0)  # recording changes nothing
        m.cache_hook = None

    def test_a_noise_kick_is_reproducible(self):
        m = self.model(16, 8)
        x = torch.randint(8, 40, (2, 64), device="cuda")
        outs = []
        for _ in range(2):
            m.cache_hook = PV.CacheHook(s_star=3, eps=0.5, seed=7)
            outs.append(self.run_model(m, x))
        self.assertEqual(float((outs[0] - outs[1]).abs().max()), 0.0)
        m.cache_hook = None

    def test_the_probe_features_of_a_cut_stream_come_from_the_right_segment(self):
        m = self.model(16, 8)
        cfg = {"train": {"precision": "bf16"}}
        toks = [list(np.random.default_rng(i).integers(8, 40, 64)) for i in range(3)]
        feats = PV.collect_window(m, cfg, toks, {16: 2, 40: 5}, batch=2)
        self.assertEqual(sorted(feats), [2, 5])
        self.assertEqual(sorted(feats[2]), ["k", "v"])           # keys and values only: the model has nothing else
        self.assertEqual(feats[2]["v"].shape, (3, 2 * 15 * 64))
        # the same stream cut at segment 2 gives the same features as a run on the first 3 segments alone
        ids = torch.tensor([t[:24] for t in toks], device="cuda")
        rec = PV.CacheHook(record_all=True)
        m.cache_hook = rec
        self.run_model(m, ids)
        m.cache_hook = None
        ref = torch.cat([v.flatten(1) for _, v in rec.snap[2]], 1).cpu().numpy()
        self.assertLess(float(np.abs(feats[2]["v"] - ref).max()), 5e-2)  # batch composition changes bf16 rounding a little

    def test_the_perturbation_probe_refuses_other_models(self):
        from models import build_swa
        plain = build_swa(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=16).cuda()
        with mock.patch.object(PV.P, "load_run", lambda name: (plain, {"model": {"kind": "swa"}, "train": {"precision": "bf16"}})):
            with self.assertRaises(ValueError):
                PV.perturbation_probe("x")

    def test_the_perturbation_probe_runs_end_to_end_on_the_two_pass_model(self):
        cfg = {"model": {"kind": "swa_2pass"}, "train": {"precision": "bf16"}}
        with mock.patch.object(PV.P, "load_run", lambda name: (self.model(64, 16), cfg)):
            r = PV.perturbation_probe("x", n_writes=64, n_exact=8, n_traj=4, frac=0.5, eps=(0.3, 3.0))
        self.assertEqual(r["kind"], "swa_2pass")
        want = {"clean", *(f"{k}_{e}" for k in ("values", "keys", "both") for e in ("0.3", "3")), "values_zero", "keys_zero", "both_zero"}
        self.assertEqual(set(r["exact"]), want)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in r["exact"].values()))
        self.assertTrue(r["contraction"])  # decay of a kick over later segments, from keys and values of the window


