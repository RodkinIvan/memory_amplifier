"""E5, the attractor analysis (src/attractor.py): the window state is the whole interface between segments, a run from a rebuilt state is the real run, the read-outs and the objective do what
they say (needs a CUDA GPU).

    python -m unittest tests.test_attractor -v
"""
import unittest

import numpy as np
import torch

from tests import _paths  # noqa: F401
import attractor as A
import make_passkey_swar as G
from models import build_swa_2pass

CUDA = torch.cuda.is_available()
V0 = G.GenParams().val0


def model(window=16, segment=8, gain=0.5, seed=0):
    torch.manual_seed(seed)
    m = build_swa_2pass(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=window, segment=segment).cuda().eval()
    for loop in m.loops:
        torch.nn.init.constant_(loop.gain, gain)
    return m


def tokens(n, length, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).integers(8, 40, (n, length))).cuda().long()


class Seeds(unittest.TestCase):
    def test_the_analysis_streams_use_seeds_the_dataset_and_the_earlier_probes_never_did(self):
        used = set(A.SEEDS.values())
        self.assertEqual(len(used), len(A.SEEDS))
        self.assertFalse(used & {1, 2, 3, 77, 78})  # train / validation / test of the dataset, the probes' test and training streams


class Streams(unittest.TestCase):
    def test_states_are_passkeys_times_contexts_plus_the_one_position_and_all_position_variants(self):
        p = G.GenParams()
        st = A.Streams(p, 32, n_pass=6, n_ctx=3)
        self.assertEqual(len(st), 6 * 3 + 6 + 6)
        self.assertEqual(st.prefix.shape, (len(st), 96))                       # the first three segments of 32 tokens
        self.assertTrue((st.prefix[:, 0] == G.BOS).all() and (st.prefix[:, 1] == G.PK).all())
        for k in range(6):
            a, b1, ball = (st.passkeys[st.index(kind, k, 0)] for kind in ("A", "B1", "Ball"))
            self.assertEqual(int((a != b1).sum()), 1)                          # one position differs ...
            self.assertEqual(int((a != b1)[st.pos_b1[k]]), 1)                  # ... the announced one
            self.assertTrue((a != ball).all())                                 # all positions differ
            self.assertTrue(np.array_equal(st.prefix[st.index("A", k, 0), 6:], st.prefix[st.index("B1", k, 0), 6:]))   # B has the context of A: only the passkey differs
            self.assertTrue(np.array_equal(st.prefix[st.index("A", k, 0), 2:6], a + 0))
        self.assertTrue(((st.passkeys >= V0) & (st.passkeys < V0 + p.n_values)).all())
        self.assertFalse(np.array_equal(st.prefix[st.index("A", 0, 0), 6:], st.prefix[st.index("A", 0, 1), 6:]))    # contexts differ
        p8 = G.GenParams(passkey_len=8)
        st8 = A.Streams(p8, 32, n_pass=4, n_ctx=2)
        self.assertEqual(st8.passkeys.shape[1], 8)
        self.assertEqual(sorted(set(st8.pos_b1)), [0, 1, 2, 3])


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class StateIsTheInterface(unittest.TestCase):
    def test_a_run_from_the_window_state_is_the_real_run_bit_for_bit(self):
        """The window state (the pass-2 inputs of the last W - 1 positions) is all that crosses a segment boundary: rebuilding the cache from it and running on gives the real logits."""
        m = model()
        x = tokens(3, 8 * 12)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            full = m(input_ids=x).logits.float()
            S = A.window_states(m, x, 5)                                         # after 5 segments (40 tokens)
            cache = A.cache_from_state(m, S, 40)
            hs = torch.cat([h for _, h, _ in A.advance(m, cache, x[:, 40:])], 1)
            got = A.logits_of(m, hs).float()
        self.assertEqual(tuple(S.shape), (3, 2, 15, 64))
        self.assertEqual(float((got - full[:, 40:]).abs().max()), 0.0)

    def test_the_next_window_is_the_window_after_one_more_segment(self):
        m = model()
        x = tokens(2, 8 * 8)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            S4, S5 = A.window_states(m, x, 4), A.window_states(m, x, 5)
            cache = A.cache_from_state(m, S4, 32)
            _, _, h2s = next(iter(A.advance(m, cache, x[:, 32:40])))
            S5b = A.next_window(S4, h2s, 8)
        self.assertEqual(float((S5b - S5).abs().max()), 0.0)

    def test_the_step_map_of_the_real_state_is_the_real_next_window(self):
        m = model(window=24, segment=8)
        x = tokens(2, 8 * 8)
        orig = A.N_SEGMENTS_BEFORE
        try:
            A.N_SEGMENTS_BEFORE = 4
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                S4, S5 = A.window_states(m, x, 4), A.window_states(m, x, 5)
                new, _ = A.step_map(m, S4, x[:, 32:40][:, None])
        finally:
            A.N_SEGMENTS_BEFORE = orig
        self.assertEqual(float((new[:, 0] - S5).abs().max()), 0.0)

    def test_a_cloned_cache_can_be_advanced_without_moving_the_original(self):
        m = model()
        x = tokens(2, 8 * 4)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            S = A.window_states(m, x, 3)
            cache = A.cache_from_state(m, S, 24)
            before = ([k.clone() for k in cache.k], list(cache.seen))
            branch = A.clone_cache(cache)
            for _ in A.advance(m, branch, x[:, 24:32]):
                pass
        self.assertEqual(cache.seen, before[1])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(cache.k, before[0])))
        self.assertNotEqual(branch.seen, cache.seen)


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class ReadOut(unittest.TestCase):
    def setUp(self):
        self.p = G.GenParams()
        self.m = model(window=16, segment=8)
        st = A.Streams(self.p, 8, n_pass=4, n_ctx=1)
        self.st = st
        self.ids = torch.from_numpy(st.prefix).cuda().long()
        dec = A.ar_contents(self.p, 92, 4, 120)
        self.ar = torch.from_numpy(np.stack([d[:8 * 6 + 8 * 3] for d in dec[:len(st)] * 4][:len(st)])).cuda().long()
        self.truth = torch.from_numpy(st.passkeys).cuda().long()

    def test_the_teacher_forced_read_out_equals_the_logits_of_the_whole_stream(self):
        """The branch read-out (state, K random segments, r filler tokens, PKQ, passkey) is what the model predicts on that whole token sequence."""
        orig = A.N_SEGMENTS_BEFORE
        try:
            A.N_SEGMENTS_BEFORE = 3
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                S = A.window_states(self.m, self.ids)
            for K, r in ((0, 5), (2, 9), (5, 4)):
                pred = A.read_out(self.m, S, self.ar, self.truth, (K,), r)[K]
                tail, r2 = A.tail_tokens(self.ar, self.truth, K, r, 8, True)
                full = torch.cat([self.ids, self.ar[:, :K * 8], tail], 1)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.m(input_ids=full).logits
                pos = self.ids.shape[1] + K * 8 + r2 + torch.arange(4, device="cuda")
                self.assertTrue(torch.equal(pred, logits[:, pos].argmax(-1)), (K, r))
                self.assertEqual((K * 8 + r2) % 3, 0)                         # PKQ follows whole triples of AR content, as in the data
        finally:
            A.N_SEGMENTS_BEFORE = orig

    def test_the_greedy_read_out_is_consistent_with_teacher_forcing_on_its_own_tokens(self):
        orig = A.N_SEGMENTS_BEFORE
        try:
            A.N_SEGMENTS_BEFORE = 3
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                S = A.window_states(self.m, self.ids)
            greedy = A.read_out(self.m, S, self.ar, self.truth, (3,), 6, greedy=True)[3]
            tf = A.read_out(self.m, S, self.ar, greedy, (3,), 6)[3]            # feed the model its own tokens: it predicts them again
            self.assertTrue(torch.equal(greedy, tf))
            first = A.read_out(self.m, S, self.ar, self.truth, (3,), 6)[3][:, 0]
            self.assertTrue(torch.equal(greedy[:, 0], first))                  # the first token does not depend on the fed passkey
        finally:
            A.N_SEGMENTS_BEFORE = orig

    def test_exact_match_and_positions(self):
        pred = torch.tensor([[1, 2, 3], [1, 0, 3]])
        truth = torch.tensor([[1, 2, 3], [1, 2, 3]])
        ex, pos = A.exact_and_positions(pred, truth)
        self.assertEqual(ex.tolist(), [True, False])
        self.assertEqual(pos.tolist(), [[True, True, True], [True, False, True]])


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class Objective(unittest.TestCase):
    def test_the_objective_is_the_error_of_the_step_map_normalised_by_the_energy_of_s0(self):
        m = model(window=24, segment=8)
        x = tokens(3, 8 * 4)
        orig = A.N_SEGMENTS_BEFORE
        try:
            A.N_SEGMENTS_BEFORE = 4
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                S0 = A.window_states(m, x, 4)
                S = S0 + 0.1 * torch.randn_like(S0)
                ids = tokens(3 * 2, 8, seed=5).view(3, 2, 8)
                per_state, slot = A.objective(m, S, S0, ids, per_slot=True)
                new, Sx = A.step_map(m, S, ids)
            err = (new - Sx).pow(2).sum((-1, -2))                              # [n, n_x, layers]
            want = (err.mean(1) / S0.pow(2).sum((2, 3))).sum(-1)
            self.assertTrue(torch.allclose(per_state, want, rtol=1e-4))
            self.assertEqual(tuple(slot.shape), (2, 23))
            # the kept, shifted entries are F(S)[i] = S[i + C]: their error is exactly the difference of S with itself shifted by C
            kept = (new[:, 0, :, :15] - Sx[:, 0, :, :15]).pow(2).sum(-1)
            shift = (S[:, :, 8:] - S[:, :, :15]).pow(2).sum(-1)
            self.assertTrue(torch.allclose(kept, shift, rtol=1e-4, atol=1e-5))
        finally:
            A.N_SEGMENTS_BEFORE = orig

    def test_adam_lowers_the_objective_and_leaves_s0_alone(self):
        m = model(window=24, segment=8)
        for p in m.parameters():
            p.requires_grad_(False)
        x = tokens(4, 8 * 4)
        orig = A.N_SEGMENTS_BEFORE
        try:
            A.N_SEGMENTS_BEFORE = 4
            with torch.autocast("cuda", dtype=torch.bfloat16):
                S0 = A.window_states(m, x, 4)
            keep = S0.clone()

            class Pool(A.SegmentPool):
                def __init__(self):
                    self.tokens, self.length = tokens(50, 200, seed=3), 200

            pool = Pool()
            ev = pool.sample((4, 4), 8, torch.Generator(device="cuda").manual_seed(1))
            S, trace, res = A.find_attractor(m, S0, pool, 60, 3e-2, 4, 0, ev, log_every=20, at_steps=(30,), callback=lambda s, S_: None)
        finally:
            A.N_SEGMENTS_BEFORE = orig
        self.assertTrue(torch.equal(S0, keep))
        self.assertLess(float(res["star"].mean()), 0.5 * float(res["start"].mean()))
        self.assertEqual([t[0] for t in trace], [20, 40, 60])
        self.assertLess(trace[-1][1], trace[0][1])


class Convergence(unittest.TestCase):
    @unittest.skipUnless(CUDA, "needs a CUDA GPU")
    def test_states_of_one_passkey_that_became_identical_have_distance_zero_and_ratio_zero(self):
        st = A.Streams(G.GenParams(), 8, n_pass=3, n_ctx=2)
        a = np.where(st.kind == "A")[0]
        S = torch.randn(len(st), 2, 15, 8).cuda()
        for k in range(3):                                                     # make the two contexts of a passkey the same state
            S[a[st.c[a] == 1][k]] = S[a[st.c[a] == 0][k]]
        r = A.convergence(st, S, S)
        self.assertAlmostEqual(r["star"]["same_passkey"], 0.0, places=5)
        self.assertGreater(r["star"]["different_passkey"], 0.5)
        self.assertAlmostEqual(r["star"]["ratio"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
