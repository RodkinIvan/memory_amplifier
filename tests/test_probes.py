"""Probe helpers: on-the-fly streams, token distances and reach ceilings (CPU only).

    python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from tests import _paths  # noqa: F401
import make_passkey_swar as G
import probes as P
import sanity_checks as S


class Probes(unittest.TestCase):
    def test_make_packed_is_consistent_with_the_oracle(self):
        p = P.make_packed(G.GenParams(), 40, 8)
        self.assertEqual(len(p), 8)
        self.assertEqual(len(p.kinds), len(p.types))
        for i in range(len(p)):
            ids = p.ids[p.offs[i]:p.offs[i + 1]]
            lab, ty, kind, _ = S.oracle(ids, 16, 16, 16)
            np.testing.assert_array_equal(lab, p.labels[p.offs[i]:p.offs[i + 1]])
            np.testing.assert_array_equal(kind, p.kinds[p.toffs[i]:p.toffs[i + 1]])

    def test_probe_streams_use_their_own_seed(self):
        a = P.make_packed(G.GenParams(), 32, 4)
        for split_seed in (1, 2, 3):  # train / validation / test seeds of the dataset
            self.assertFalse(np.array_equal(a.ids[:a.offs[1]], P.make_packed(G.GenParams(), 32, 1, split_seed).ids))

    def test_distances_match_the_answer_rule(self):
        p = P.make_packed(G.GenParams(), 200, 30)
        valid, stale = P.query_distances(p)
        self.assertGreater(len(valid), 0)
        self.assertGreater(len(stale), 0)
        self.assertGreaterEqual(valid.min(), 2)  # a write is at least `v` then `Q k`: distance >= 2
        self.assertLess(valid.max(), stale.min())  # valid writes are always more recent than stale ones
        self.assertGreaterEqual(stale.min(), 3 * 32)  # more than 2w = 32 writes of 3 tokens each

    def test_reach_ceilings_are_monotone_and_bounded(self):
        c = P.reach_ceilings(P.make_packed(G.GenParams(), 200, 30), [30, 60, 100, 200, 500])
        vr = [c[r]["valid_reachable"] for r in sorted(c)]
        sv = [c[r]["stale_visible"] for r in sorted(c)]
        self.assertEqual(vr, sorted(vr))
        self.assertEqual(sv, sorted(sv))
        self.assertEqual(c[500]["valid_reachable"], 1.0)
        self.assertEqual(c[30]["stale_visible"], 0.0)
        self.assertTrue(all(0 <= v <= 1 for v in vr + sv))


class TokenProfile(unittest.TestCase):
    def test_histogram_and_per_token_accuracy(self):
        import numpy as np
        hits = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
        r = P.token_profile(hits)
        self.assertEqual(r["exact"], 0.5)
        self.assertEqual(r["per_token"], [0.75, 0.5, 0.5, 0.5])
        self.assertEqual(r["hist"], [0.25, 0.25, 0.0, 0.0, 0.5])  # shares of examples with 0..4 right tokens
        self.assertAlmostEqual(sum(r["hist"]), 1.0)


class AttractorProbeFunctions(unittest.TestCase):
    def test_contraction_profile_reads_the_decay_of_a_perturbation(self):
        S, B, D = 40, 3, 8
        clean, pert = np.zeros((S, B, D)), np.zeros((S, B, D))
        for s in range(10, S):
            pert[s] = 0.5 ** (s - 10)  # the perturbation introduced at segment 10 halves every hop
        prof = P.contraction_profile(clean, pert, 10, offsets=(1, 4, 16, 64))
        self.assertAlmostEqual(prof[1], 0.5)
        self.assertAlmostEqual(prof[4], 0.5 ** 4)
        self.assertAlmostEqual(prof[16], 0.5 ** 16)
        self.assertNotIn(64, prof)  # beyond the end of the stream

    def test_a_carried_perturbation_has_ratio_one_and_a_growing_one_more(self):
        S, B, D = 30, 2, 4
        clean = np.zeros((S, B, D))
        same, grow = np.ones((S, B, D)), np.cumprod(np.full((S, 1, 1), 1.1), axis=0) * np.ones((S, B, D))
        self.assertAlmostEqual(P.contraction_profile(clean, same, 5, offsets=(10,))[10], 1.0)
        self.assertAlmostEqual(P.contraction_profile(clean, grow, 5, offsets=(10,))[10], 1.1 ** 10)

    def test_drift_profile_of_a_constant_state_has_no_steps(self):
        state = np.ones((50, 3, 16)) * 0.5
        d = P.drift_profile(state)
        self.assertAlmostEqual(d["rms"][0.5], 0.5)
        self.assertEqual(d["step"][0.9], 0.0)


@unittest.skipUnless(__import__("torch").cuda.is_available(), "needs a GPU")
class LoopHookOnModel(unittest.TestCase):
    def model(self, gain):
        from models import build_swa_fb
        return build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=16, segment=8, gain_init=gain, proj_std=0.1).cuda().eval()

    def logits(self, m, x):
        import torch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(input_ids=x).logits.float()

    def test_hook_sees_every_segment_and_layer(self):
        import torch
        m = self.model(0.3)
        m.probe_hook = hook = P.LoopHook(record=True)
        self.logits(m, torch.randint(1, 40, (2, 64), device="cuda"))
        self.assertEqual(sorted(hook.m), [(s, i) for s in range(8) for i in range(2)])
        self.assertEqual(hook.states(2).shape, (8, 2, 128))

    def test_a_perturbation_changes_only_later_outputs(self):
        import torch
        m = self.model(0.5)
        x = torch.randint(1, 40, (2, 64), device="cuda")
        clean = self.logits(m, x)
        m.probe_hook = P.LoopHook(s_star=2, eps=1.0)
        pert = self.logits(m, x)
        first_affected = (2 + 1) * 8  # m of segment 2 is injected from segment 3 on
        self.assertEqual(float((clean[:, :first_affected] - pert[:, :first_affected]).abs().max()), 0.0)
        self.assertGreater(float((clean[:, first_affected:] - pert[:, first_affected:]).abs().max()), 1e-3)

    def test_jacobian_is_zero_without_injection_and_not_with_it(self):
        import torch
        x = torch.randint(1, 40, (80,), device="cuda")
        J0 = P.loop_jacobian(self.model(0.0), x, 3, torch.bfloat16)
        J1 = P.loop_jacobian(self.model(0.5), x, 3, torch.bfloat16)
        self.assertEqual(J0.shape, (128, 128))
        self.assertLess(float(np.abs(J0).max()), 1e-6)
        self.assertGreater(float(np.abs(J1).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
