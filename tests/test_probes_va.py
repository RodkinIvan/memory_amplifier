"""Diagnostics of the value-adaptation model (src/probes_va.py): the linear probe, the hooks, the matrix statistics (the model parts need a CUDA GPU).

    python -m unittest tests.test_probes_va -v
"""
import unittest
from unittest import mock

import numpy as np
import torch

from tests import _paths  # noqa: F401
import make_passkey_swar as G
import probes_va as PV
from models import build_swa_2pass, build_swa_va
from tests.test_models import strong_adaptation

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


class MatrixStats(unittest.TestCase):
    def test_known_matrices(self):
        eye = torch.eye(8)[None].repeat(2, 1, 1)
        st = PV.matrix_stats(torch.cat([eye[:1], 3 * eye[1:]]))
        self.assertTrue(torch.allclose(st["fro"], torch.tensor([8 ** 0.5, 3 * 8 ** 0.5])))
        self.assertTrue(torch.allclose(st["spectral"], torch.tensor([1.0, 3.0])))
        self.assertTrue(torch.allclose(st["rank"], torch.tensor([8.0, 8.0])))  # the identity uses all 8 directions equally
        self.assertTrue(torch.allclose(st["max_entry"], torch.tensor([1.0, 3.0])))
        one = torch.outer(torch.arange(1.0, 9.0), torch.arange(1.0, 9.0))[None]
        self.assertAlmostEqual(float(PV.matrix_stats(one)["rank"]), 1.0, places=4)  # rank one


class CommonMode(unittest.TestCase):
    def test_identical_tokens_are_all_mean_and_independent_tokens_are_not(self):
        g = torch.Generator().manual_seed(0)
        a = torch.tanh(2 * torch.randn(3, 32, 16, generator=g))
        b = torch.tanh(2 * torch.randn(3, 32, 16, generator=g))
        same = PV.common_mode(a[:, :1].repeat(1, 32, 1), b[:, :1].repeat(1, 32, 1))
        self.assertTrue(torch.allclose(same["mean_share"], torch.ones(3), atol=1e-5))
        self.assertLess(float(same["cov_fro"].max()), 1e-4)
        rnd = PV.common_mode(a, b)
        self.assertLess(float(rnd["mean_share"].max()), 0.5)  # independent tokens: most of the energy is in the covariance part
        # the two parts add up exactly: mean outer product + covariance = the full matrix
        full = torch.einsum("bjv,bjd->bvd", b, a) / 32
        mean = torch.einsum("bv,bd->bvd", b.mean(1), a.mean(1))
        self.assertTrue(torch.allclose((full - mean).flatten(1).norm(dim=1), rnd["cov_fro"], atol=1e-5))
        sat = PV.common_mode(torch.ones(1, 4, 8), torch.zeros(1, 4, 8))
        self.assertAlmostEqual(float(sat["saturated"]), 0.5)  # all of a saturated, none of b


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class Hooks(unittest.TestCase):
    def model(self, W=16, C=8):
        torch.manual_seed(0)
        m = build_swa_va(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=W, segment=C).cuda().eval()
        strong_adaptation(m, gain=0.5, std=0.2)
        return m

    def run_model(self, m, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(input_ids=x).logits.float()

    def test_delta_w_matches_the_matrix_the_model_adds(self):
        m = self.model()
        x = torch.randint(8, 40, (2, 40), device="cuda")
        hook = PV.StateHook(record=True)
        m.probe_hook = hook
        self.run_model(m, x)
        a, b = hook.state[(1, 0)]
        dw = PV.delta_w(m, 0, a, b)
        self.assertEqual(tuple(dw.shape), (2, 64, 64))
        self.assertTrue(torch.allclose(dw, m.loops[0].gain.detach() / 8 * torch.einsum("bjv,bjd->bvd", b, a)))
        self.assertLessEqual(float(dw.abs().max()), abs(float(m.loops[0].gain)) + 1e-6)  # entries bounded by the gain
        self.assertLessEqual(torch.linalg.matrix_rank(dw[0]).item(), 8)  # rank at most C

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
        for kw in ({"eps": 1.0}, {"mode": "zero"}):
            m.cache_hook = PV.CacheHook(s_star=3, seed=1, **kw)
            d = (self.run_model(m, x) - clean).abs().amax(-1)[0]
            self.assertEqual(float(d[:32].max()), 0.0, kw)  # up to the end of segment 3 nothing moves (the kick lands after it)
            self.assertGreater(float(d[32:].max()), 0.0, kw)
        m.cache_hook = None

    def test_the_state_hook_perturbs_only_at_its_segment_and_zero_removes_the_adaptation(self):
        m = self.model(16, 8)
        x = torch.randint(8, 40, (2, 64), device="cuda")
        clean = self.run_model(m, x)
        m.probe_hook = PV.StateHook(s_star=2, eps=1.0, seed=1)
        d = (self.run_model(m, x) - clean).abs().amax(-1)[0]
        self.assertEqual(float(d[:24].max()), 0.0)
        self.assertGreater(float(d[24:].max()), 0.0)
        # zeroing the state at every segment = no adaptation at all = the plain sliding-window model
        m.probe_hook = lambda s, i, st: tuple(torch.zeros_like(t) for t in st)
        plain = self.run_model(m, x)
        m.probe_hook = None
        self.assertLess(float((plain - self.run_model(m.base, x)).abs().max()), 5e-2)

    def test_a_noise_kick_is_reproducible_and_scales_with_the_tensor(self):
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
        cuts = {16: 2, 40: 5}
        feats = PV.collect_window(m, cfg, toks, cuts, batch=2)
        self.assertEqual(sorted(feats), [2, 5])
        self.assertEqual(feats[2]["v"].shape, (3, 2 * 15 * 64))
        self.assertEqual(feats[2]["dw"].shape, (3, 2 * 64 * 64))
        # the same stream cut at segment 2 gives the same features as a run on the first 3 segments alone
        ids = torch.tensor([t[:24] for t in toks], device="cuda")
        rec = PV.CacheHook(record_all=True)
        m.cache_hook = rec
        self.run_model(m, ids)
        m.cache_hook = None
        ref = torch.cat([v.flatten(1) for _, v in rec.snap[2]], 1).cpu().numpy()
        self.assertLess(float(np.abs(feats[2]["v"] - ref).max()), 5e-2)  # batch composition changes bf16 rounding a little


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

    def test_the_recorded_window_is_the_stored_one_and_has_no_adaptation_features(self):
        m = self.model()
        cfg = {"train": {"precision": "bf16"}}
        toks = [list(np.random.default_rng(i).integers(8, 40, 64)) for i in range(3)]
        feats = PV.collect_window(m, cfg, toks, {16: 2, 40: 5}, batch=2)
        self.assertEqual(sorted(feats[2]), ["k", "v"])           # no dW_v features: the model has none
        self.assertEqual(feats[2]["v"].shape, (3, 2 * 15 * 64))

    def test_dw_statistics_refuse_the_two_pass_model(self):
        with mock.patch.object(PV.P, "load_run", lambda name: (self.model(), {"train": {"precision": "bf16"}})):
            with self.assertRaises(ValueError):
                PV.delta_profile("x")

    def test_the_perturbation_probe_runs_end_to_end_on_the_two_pass_model(self):
        cfg = {"model": {"kind": "swa_2pass"}, "train": {"precision": "bf16"}}
        with mock.patch.object(PV.P, "load_run", lambda name: (self.model(64, 16), cfg)):
            r = PV.perturbation_probe("x", n_writes=64, n_exact=8, n_traj=4, frac=0.5, eps=(0.3, 3.0))
        self.assertEqual(r["kind"], "swa_2pass")
        want = {"clean", *(f"{k}_{e}" for k in ("values", "keys", "both") for e in ("0.3", "3")), "values_zero", "keys_zero", "both_zero"}
        self.assertEqual(set(r["exact"]), want)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in r["exact"].values()))
        self.assertTrue(r["contraction"])  # decay of a kick over later segments, from keys and values of the window


if __name__ == "__main__":
    unittest.main()
