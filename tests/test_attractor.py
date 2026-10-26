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


def check_triples(toks, p: G.GenParams, until: int) -> None:
    """The tokens from position 2 + P to `until` are whole `W k v` / `Q k a` triples (head, key, value or NULL)."""
    keys, vals = range(G.KEY0, G.KEY0 + p.n_keys), range(p.val0, p.val0 + p.n_values)
    for pos in range(2 + p.passkey_len, until, 3):
        assert toks[pos] in (G.W, G.Q), (pos, toks[pos])
        assert toks[pos + 1] in keys, (pos + 1, toks[pos + 1])
        assert toks[pos + 2] in vals or toks[pos + 2] == G.NULL, (pos + 2, toks[pos + 2])


class StreamGrid(unittest.TestCase):
    """Review N3: the E5 streams keep the dataset's own triple grid across the state boundary, for every passkey length (the state is at position 96)."""

    def streams(self, P, segment=32):
        p = G.GenParams(passkey_len=P)
        st = A.Streams(p, segment, n_pass=3, n_ctx=2)
        dec = A.ar_contents(p, 92, 4, 400)
        return p, st, A.continuation(st, dec, 40 * segment)

    def test_the_cut_triple_is_completed_and_no_triple_is_broken_at_the_state_position(self):
        for P, phase, rest in ((4, 0, 0), (8, 2, 1), (16, 0, 0)):
            p, st, ar = self.streams(P)
            self.assertEqual((st.phase, st.rest.shape[1]), (phase, rest), P)                       # P = 8: the prefix ends two tokens into a triple
            for i in range(len(st)):
                toks = np.concatenate([st.prefix[i], ar[i]])
                check_triples(toks, p, len(toks) - 3)                                               # the triple that spans position 96 is whole too
                if rest:
                    self.assertEqual(int(st.prefix[i, -2]), int(toks[2 + P + 3 * ((96 - 2 - P) // 3)]))   # the cut triple's head and key are in the prefix, its value token follows

    def test_pkq_lands_where_the_dataset_puts_it_for_every_k_and_filler(self):
        for P in (4, 8, 16):
            p, st, ar = self.streams(P)
            ex = G.make_example(G.example_seed(5, 32, 0), 32, p)[0]
            dataset_phase = ex.index(G.PKQ) % 3
            arT = torch.from_numpy(ar).long()
            for K in (0, 1, 4, 32):
                for r in (3, 7, 10, 14):
                    tail, r2 = A.tail_tokens(arT, torch.from_numpy(st.passkeys).long(), K, r, 32, True, st.phase)
                    self.assertTrue(r - 2 <= r2 <= r)
                    for i in range(len(st)):
                        toks = np.concatenate([st.prefix[i], ar[i, :K * 32 + r2], tail[i, r2:].numpy()])
                        pkq = len(st.prefix[i]) + K * 32 + r2
                        self.assertEqual(int(toks[pkq]), G.PKQ)
                        self.assertEqual(pkq % 3, dataset_phase, (P, K, r))                           # the phase of PKQ in the dataset's streams
                        check_triples(toks, p, pkq)                                                  # whole triples all the way to PKQ
                        self.assertEqual(toks[pkq + 1:pkq + 1 + P].tolist(), st.passkeys[i].tolist())

    def test_the_old_layout_would_fail_for_p8(self):
        """Sensitivity: continuing with a new whole triple at position 96 (no rest, phase 0) breaks the grid for P = 8 and only for P = 8."""
        broken = {}
        for P in (4, 8, 16):
            p, st, ar = self.streams(P)
            dec = A.ar_contents(p, 92, 4, 400)
            old = np.stack([dec[i % 4][3 * (i // 4) * 7:][:ar.shape[1]] for i in range(len(st))])
            toks = np.concatenate([st.prefix[0], old[0]])
            try:
                check_triples(toks, p, len(toks) - 3)
                broken[P] = False
            except AssertionError:
                broken[P] = True
        self.assertEqual(broken, {4: False, 8: True, 16: False})


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class SegmentPhase(unittest.TestCase):
    """Review N2: the objective's segments start where the stream's grid is at the state's position."""

    def pool(self, P):
        p = G.GenParams(passkey_len=P)
        return p, A.SegmentPool(p, 91, 12, 120)

    def test_sampled_segments_start_at_the_requested_place_in_a_triple(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for P, phase in ((4, 0), (8, 2), (16, 0)):
            p, pool = self.pool(P)
            x = pool.sample((400,), 32, gen, phase)
            first = x[:, 0].cpu().numpy()
            if phase == 0:
                self.assertTrue(np.isin(first, [G.W, G.Q]).all(), P)                               # a segment starts at the head of a triple
            else:
                self.assertTrue(((first >= p.val0) | (first == G.NULL)).all(), P)                   # P = 8: at the third token of a triple (a value or the NULL answer)
            self.assertEqual(x.shape, (400, 32))
        x = pool.sample((400,), 32, gen)                                                              # the default is the triple head
        self.assertTrue(np.isin(x[:, 0].cpu().numpy(), [G.W, G.Q]).all())

    def test_the_old_sampling_would_start_inside_triples(self):
        p, pool = self.pool(4)
        gen = torch.Generator(device="cuda").manual_seed(0)
        o = torch.randint(0, pool.length - 31, (400,), generator=gen, device="cuda")
        first = pool.tokens[0, o].cpu().numpy()
        self.assertGreater(float((~np.isin(first, [G.W, G.Q])).mean()), 0.5)                         # arbitrary offsets: about 2 of 3 start mid-triple

    def test_the_optimisation_uses_the_phase_it_is_given(self):
        m = model(window=24, segment=8)
        for q in m.parameters():
            q.requires_grad_(False)
        x = tokens(4, 8 * 4)
        orig = A.N_SEGMENTS_BEFORE
        seen = []
        try:
            A.N_SEGMENTS_BEFORE = 4
            with torch.autocast("cuda", dtype=torch.bfloat16):
                S0 = A.window_states(m, x, 4)

            class Spy(A.SegmentPool):
                def __init__(self):
                    self.tokens, self.length = tokens(50, 200, seed=3), 200

                def sample(self, shape, C_, gen, phase=0):
                    seen.append(phase)
                    return super().sample(shape, C_, gen, phase)

            ev = Spy().sample((4, 4), 8, torch.Generator(device="cuda").manual_seed(1), 2)
            A.find_attractor(m, S0, Spy(), 3, 3e-2, 4, 0, ev, log_every=1, phase=2)
        finally:
            A.N_SEGMENTS_BEFORE = orig
        self.assertEqual(set(seen), {2})


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

    def test_the_read_out_of_a_p8_stream_equals_the_logits_of_its_whole_dataset_layout(self):
        """P = 8 puts the cut at the third token of a triple (phase 2): state, rest of the triple, K random segments, filler, PKQ, passkey is the logits of that one whole stream."""
        p8 = G.GenParams(passkey_len=8)
        st = A.Streams(p8, 8, n_pass=4, n_ctx=1)
        self.assertEqual(st.phase, 2)                                                                  # position 24: 24 - 10 = 14 tokens of AR, 14 % 3 = 2
        ids = torch.from_numpy(st.prefix).cuda().long()
        dec = A.ar_contents(p8, 92, 4, 120)
        ar = torch.from_numpy(A.continuation(st, dec, 8 * 6 + 8 * 3)).cuda().long()
        truth = torch.from_numpy(st.passkeys).cuda().long()
        orig = A.N_SEGMENTS_BEFORE
        try:
            A.N_SEGMENTS_BEFORE = 3
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                S = A.window_states(self.m, ids)
            for K, r in ((0, 5), (2, 9), (3, 4)):
                pred = A.read_out(self.m, S, ar, truth, (K,), r, phase=st.phase)[K]
                tail, r2 = A.tail_tokens(ar, truth, K, r, 8, True, st.phase)
                full = torch.cat([ids, ar[:, :K * 8], tail], 1)
                check_triples(full[0].cpu().numpy(), p8, ids.shape[1] + K * 8 + r2)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.m(input_ids=full).logits
                pos = ids.shape[1] + K * 8 + r2 + torch.arange(8, device="cuda")
                self.assertTrue(torch.equal(pred, logits[:, pos].argmax(-1)), (K, r))
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


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class Variants(unittest.TestCase):
    def setUp(self):
        self.m = model(window=24, segment=8)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.orig = A.N_SEGMENTS_BEFORE
        A.N_SEGMENTS_BEFORE = 4

    def tearDown(self):
        A.N_SEGMENTS_BEFORE = self.orig

    def test_the_sphere_variant_keeps_the_norm_of_every_layers_state(self):
        x = tokens(4, 8 * 4)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            S0 = A.window_states(self.m, x, 4)

        class Pool(A.SegmentPool):
            def __init__(self):
                self.tokens, self.length = tokens(50, 200, seed=3), 200

        ev = Pool().sample((4, 4), 8, torch.Generator(device="cuda").manual_seed(1))
        S, _, res = A.find_attractor(self.m, S0, Pool(), 40, 3e-2, 4, 0, ev, sphere=True)
        n0, n1 = S0.pow(2).sum((2, 3)).sqrt(), S.pow(2).sum((2, 3)).sqrt()
        self.assertLess(float((n1 / n0 - 1).abs().max()), 1e-4)                  # per state and layer
        self.assertGreater(float((S - S0).abs().max()), 1e-3)                    # and it did move
        S_free, _, res_free = A.find_attractor(self.m, S0, Pool(), 40, 3e-2, 4, 0, ev)
        self.assertLess(float((S_free.pow(2).sum((2, 3)).sqrt() / n0).mean()), 1.0)     # the free objective shrinks the state

    def test_the_trajectory_convergence_of_identical_starts_is_zero_and_of_different_ones_is_not(self):
        st = A.Streams(G.GenParams(), 8, n_pass=3, n_ctx=2)
        a = np.where(st.kind == "A")[0]
        S = torch.randn(len(st), 2, 23, 64).cuda()
        for k in range(3):                                                      # the two contexts of a passkey start from the same state
            S[a[st.c[a] == 1][k]] = S[a[st.c[a] == 0][k]]
        ar = tokens(len(st), 8 * 8, seed=2)
        r = A.trajectory_convergence(self.m, st, S, ar, 8, marks=(1, 4, 8))
        self.assertEqual(r["t"], [0, 1, 4, 8])
        self.assertAlmostEqual(r["same_passkey"][0], 0.0, places=5)
        self.assertAlmostEqual(r["same_passkey"][-1], 0.0, places=5)           # the same input from the same state: the same trajectory
        self.assertGreater(r["different_passkey"][0], 0.5)


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
