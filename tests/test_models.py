"""Tests for the fla-based models (need a CUDA GPU for the kernels).

    python -m unittest discover -s tests -v
"""
import math
import unittest

import torch

from tests import _paths  # noqa: F401
from models import FeedbackSWA, build_from_config, build_model, build_swa, build_swa_fb, swa_reach


def naive_gated_delta(q, k, v, alpha, beta_logit, neg):
    """fp32 reference: S_t = alpha S_{t-1} + beta_t (v_t - alpha S_{t-1} k_t) k_t^T, o_t = S_t q_t / sqrt(K)."""
    B, T, H, K = q.shape
    q = torch.nn.functional.normalize(q.float(), dim=-1) * K ** -0.5
    k = torch.nn.functional.normalize(k.float(), dim=-1)
    beta = torch.sigmoid(beta_logit.float()) * (2 if neg else 1)
    S = torch.zeros(B, H, v.shape[-1], K, device=q.device)
    out = []
    for t in range(T):
        S = S * alpha
        pred = torch.einsum("bhvk,bhk->bhv", S, k[:, t])
        S = S + torch.einsum("bhv,bhk->bhvk", (v[:, t].float() - pred) * beta[:, t, :, None], k[:, t])
        out.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]))
    return torch.stack(out, 1)


class BuilderArgs(unittest.TestCase):
    def test_alpha_only_for_fixed(self):
        with self.assertRaises(ValueError):
            build_model("gdn_fixed", vocab_size=40)
        with self.assertRaises(ValueError):
            build_model("gdn", vocab_size=40, alpha=0.9)
        with self.assertRaises(ValueError):
            build_model("gdn_fixed", vocab_size=40, alpha=1.5)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class FixedDecayKernel(unittest.TestCase):
    def test_kernel_matches_naive_recurrence(self):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        torch.manual_seed(0)
        for alpha in (1.0, 0.98, 0.8):
            for neg in (False, True):
                q, k, v = (torch.randn(2, 200, 1, 32, device="cuda") for _ in range(3))
                beta = torch.randn(2, 200, 1, device="cuda")
                g = torch.full((2, 200, 1), math.log(alpha), device="cuda")
                o, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True,
                                              use_beta_sigmoid_in_kernel=True, allow_neg_eigval=neg, state_v_first=True)
                ref = naive_gated_delta(q, k, v, alpha, beta, neg)
                self.assertLess(float((o.float() - ref).abs().max()), 1e-2 * float(ref.abs().max()) + 1e-3, (alpha, neg))

    def test_decay_is_actually_applied(self):
        torch.manual_seed(0)
        x = torch.randint(8, 40, (1, 256), device="cuda")
        outs = []
        for alpha in (1.0, 0.8):
            torch.manual_seed(1)  # same weights
            m = build_model("gdn_fixed", vocab_size=40, alpha=alpha).cuda().eval()
            with torch.no_grad():
                outs.append(m(input_ids=x).logits)
        self.assertGreater(float((outs[0] - outs[1]).abs().max()), 1e-3)

    def test_right_padding_is_harmless(self):
        torch.manual_seed(0)
        m = build_model("gdn_fixed", vocab_size=40, alpha=0.98).cuda().eval()
        x = torch.randint(8, 40, (1, 300), device="cuda")
        with torch.no_grad():
            a = m(input_ids=x).logits
            b = m(input_ids=torch.nn.functional.pad(x, (0, 212), value=0)).logits[:, :300]
        # not bit-exact (the padded length changes kernel blocking) but equal up to float rounding
        self.assertLess(float((a - b).abs().max()), 1e-3 * float(a.abs().max()))

    def test_forward_backward_and_param_count(self):
        m = build_model("gdn_fixed", vocab_size=40, alpha=0.95).cuda()
        names = {n for n, _ in m.named_parameters()}
        self.assertFalse(any(("A_log" in n or "dt_bias" in n or "a_proj" in n) for n in names))  # no learned decay
        x = torch.randint(8, 40, (2, 128), device="cuda")
        m(input_ids=x).logits.float().mean().backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))


class SWABuilder(unittest.TestCase):
    def test_argument_checks(self):
        with self.assertRaises(ValueError):
            build_swa(vocab_size=40, d_model=128, num_heads=3, window=8)  # 128 not divisible by 3
        with self.assertRaises(ValueError):
            build_swa(vocab_size=40, window=1)

    def test_dispatch_from_config_block(self):
        swa = build_from_config({"kind": "swa", "d_model": 128, "n_layers": 3, "num_heads": 4, "window": 16}, 40)
        self.assertEqual((swa.config.window_size, swa.config.num_hidden_layers, swa.config.hidden_size), (16, 3, 128))
        gdn = build_from_config({"kind": "gdn_fixed", "d_model": 128, "n_layers": 2, "num_heads": 1, "head_dim": 32,
                                 "expand_v": 1.0, "alpha": 0.9, "allow_neg_eigval": False, "norm_eps": 1e-5}, 40)
        self.assertEqual(sum(p.numel() for p in gdn.parameters()), 441_024)  # unchanged by the refactor

    def test_gdn2_is_the_stock_learned_decay_gdn(self):
        """kind 'gdn2' keeps the decay parameters that gdn_fixed removes, takes no alpha, and trains from the config like the other kinds."""
        mc = {"kind": "gdn2", "d_model": 128, "n_layers": 2, "num_heads": 1, "head_dim": 32, "expand_v": 1.0, "allow_neg_eigval": False, "norm_eps": 1e-5}
        gdn2 = build_from_config(mc, 40)
        for param in ("a_proj", "A_log", "dt_bias"):
            self.assertTrue(any(param in n for n, _ in gdn2.named_parameters()), param)
        self.assertGreater(sum(p.numel() for p in gdn2.parameters()), 441_024)  # the learned gate adds a_proj, A_log, dt_bias
        with self.assertRaises(ValueError):
            build_model("gdn2", vocab_size=40, alpha=0.9)

    def test_reach_formula(self):
        self.assertEqual(swa_reach(64, 2), 126)
        self.assertEqual(swa_reach(128, 4), 508)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class SWAReach(unittest.TestCase):
    def test_information_travels_exactly_layers_times_window_minus_one(self):
        """Changing token 0 must change the output at positions 0..R and leave every later position bit-identical."""
        torch.manual_seed(0)
        for window, layers in ((8, 3), (5, 2), (16, 1)):
            m = build_swa(vocab_size=40, d_model=64, n_layers=layers, num_heads=2, window=window).cuda().eval()
            R = swa_reach(window, layers)
            x = torch.randint(8, 40, (1, 3 * R + 10), device="cuda")
            x2 = x.clone()
            x2[0, 0] = 8 + (int(x[0, 0]) - 8 + 1) % 32
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                diff = (m(input_ids=x).logits.float() - m(input_ids=x2).logits.float()).abs().amax(-1)[0]
            changed = diff > 0
            self.assertTrue(bool(changed[:R + 1].all()), (window, layers, "no effect inside the reach"))
            self.assertFalse(bool(changed[R + 1:].any()), (window, layers, "effect leaked beyond the reach"))

    def test_runs_on_a_61k_token_sequence(self):
        m = build_swa(vocab_size=40, window=64, n_layers=2, num_heads=4).cuda().eval()
        x = torch.randint(8, 40, (1, 61440), device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.assertTrue(bool(torch.isfinite(m(input_ids=x).logits).all()))


def zero_feedback(model: FeedbackSWA):
    for loop in model.loops:
        torch.nn.init.zeros_(loop.proj.weight)
        torch.nn.init.zeros_(loop.proj.bias)


def strong_feedback(model: FeedbackSWA, std=0.5, gain=1.0):
    """Switch the loops on: random projection of scale `std`, injection gain `gain` (the default init has gain 0)."""
    for loop in model.loops:
        torch.nn.init.normal_(loop.proj.weight, std=std)
        torch.nn.init.constant_(loop.gain, gain)


class FeedbackBuilder(unittest.TestCase):
    def test_dispatch_and_defaults(self):
        mc = {"kind": "swa_fb", "d_model": 128, "n_layers": 2, "num_heads": 4, "window": 64}
        m = build_from_config(mc, 40)
        self.assertEqual((m.window, m.segment, m.loops[0].nonlin), (64, 64, "tanh"))  # segment 0/absent -> one per window
        self.assertEqual(build_from_config({**mc, "segment": 32, "nonlin": "none"}, 40).segment, 32)
        plain = build_from_config({**mc, "kind": "swa"}, 40)
        extra = sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in plain.parameters())
        self.assertEqual(extra, 2 * (128 * 128 + 128 + 128 + 128))  # per layer: proj weight+bias, norm weight, gain
        self.assertTrue(all(float(l.gain.abs().max()) == 0.0 for l in m.loops))  # starts as plain SWA
        self.assertEqual(build_from_config({**mc, "gain_init": 0.5}, 40).loops[0].gain[0].item(), 0.5)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            build_swa_fb(vocab_size=40, window=16, nonlin="relu")


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class FeedbackBehaviour(unittest.TestCase):
    def run_model(self, m, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(input_ids=x).logits.float()

    def test_zero_feedback_equals_plain_swa(self):
        """The segment loop with m = 0 must reproduce the plain sliding-window forward (checks tail handling)."""
        torch.manual_seed(0)
        for window, segment, length in ((16, 16, 100), (16, 8, 100), (16, 5, 77), (8, 24, 90)):
            fb = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=window, segment=segment).cuda().eval()
            zero_feedback(fb)
            x = torch.randint(8, 40, (2, length), device="cuda")
            diff = (self.run_model(fb, x) - self.run_model(fb.base, x)).abs().max().item()
            self.assertLess(diff, 5e-2, (window, segment, length, diff))

    def test_default_init_is_plain_swa_even_with_nonzero_projection(self):
        torch.manual_seed(0)
        fb = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=16, segment=16).cuda().eval()
        strong_feedback(fb, gain=0.0)  # a big projection, but gain 0 keeps the injection off
        x = torch.randint(8, 40, (2, 100), device="cuda")
        self.assertLess((self.run_model(fb, x) - self.run_model(fb.base, x)).abs().max().item(), 5e-2)

    def test_causality(self):
        torch.manual_seed(0)
        fb = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=8).cuda().eval()
        strong_feedback(fb)
        x = torch.randint(8, 40, (1, 96), device="cuda")
        x2 = x.clone()
        x2[0, 50] = 8 + (int(x[0, 50]) - 8 + 1) % 32
        d = (self.run_model(fb, x) - self.run_model(fb, x2)).abs().amax(-1)[0]
        self.assertEqual(float(d[:50].max()), 0.0)  # nothing before the changed token moves
        self.assertGreater(float(d[50:].max()), 0.0)

    def test_feedback_carries_information_past_the_window_reach(self):
        torch.manual_seed(0)
        window, layers = 8, 2
        R = swa_reach(window, layers)
        x = torch.randint(8, 40, (1, 8 * 15), device="cuda")
        x2 = x.clone()
        x2[0, 1] = 8 + (int(x[0, 1]) - 8 + 1) % 32
        far = slice(R + 2 * window, None)  # positions the window alone can never see
        for strength, expect_leak in ((None, False), (0.5, True)):
            fb = build_swa_fb(vocab_size=40, d_model=64, n_layers=layers, num_heads=2, window=window).cuda().eval()
            zero_feedback(fb) if strength is None else strong_feedback(fb, strength)
            d = (self.run_model(fb, x) - self.run_model(fb, x2)).abs().amax(-1)[0, far]
            self.assertEqual(bool((d > 0).any()), expect_leak, (strength, float(d.max())))

    def test_gradients_reach_every_loop_through_the_segments(self):
        fb = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=8).cuda()
        strong_feedback(fb, 0.3)
        x = torch.randint(8, 40, (2, 64), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fb(input_ids=x).logits.float()[:, -1].sum().backward()  # the last token depends on the first segment only via m
        for loop in fb.loops:
            self.assertGreater(float(loop.proj.weight.grad.abs().sum()), 0.0)
            self.assertGreater(float(loop.gain.grad.abs().sum()), 0.0)

    def test_state_dict_roundtrip(self):
        a = build_swa_fb(vocab_size=40, window=16, n_layers=2, num_heads=4).cuda().eval()
        b = build_swa_fb(vocab_size=40, window=16, n_layers=2, num_heads=4).cuda().eval()
        b.load_state_dict(a.state_dict())
        x = torch.randint(8, 40, (1, 50), device="cuda")
        self.assertEqual(float((self.run_model(a, x) - self.run_model(b, x)).abs().max()), 0.0)


class KvCacheEquivalence(unittest.TestCase):
    """`kv_cache=True` must give the same logits and gradients as recomputing the tail, with the loops switched on (m != 0).

    Strong feedback is chaotic: the tail path with weights perturbed by 1e-4 differs from itself by ~0.8 in logits after a few
    dozen hops, so rounding noise would swamp the comparison. The feedback here is the mild recipe scale (projection std 0.1,
    gain 0.1), where the two paths agree to bf16 rounding (~0.004 in logits of scale 0.13, 37 hops, window 64)."""

    CASES = ((16, 8, 100), (16, 16, 100), (16, 24, 90), (16, 5, 77), (8, 24, 90), (64, 16, 400), (64, 8, 300))  # window, segment, length: C < W, = W, > W, T % C != 0

    def pair(self, W, C, cache_cls=None):
        a = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=W, segment=C).cuda()
        strong_feedback(a, std=0.1, gain=0.1)
        b = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=W, segment=C, kv_cache=True).cuda()
        b.load_state_dict(a.state_dict())
        return a, b

    def logits_and_grads(self, m, x):
        m.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(input_ids=x).logits.float()
        logits.logsumexp(-1).mean().backward()
        return logits.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    def compare(self, W, C, T):
        torch.manual_seed(0)
        a, b = self.pair(W, C)
        x = torch.randint(1, 40, (3, T), device="cuda")
        (la, ga), (lb, gb) = self.logits_and_grads(a, x), self.logits_and_grads(b, x)
        names = [n for n in ga if ga[n].norm() > 1e-6]
        cos = min(float(torch.nn.functional.cosine_similarity(ga[n].flatten(), gb[n].flatten(), dim=0)) for n in names)
        rel = max(float((ga[n] - gb[n]).norm() / ga[n].norm()) for n in names)
        return float((la - lb).abs().max()), cos, rel, len(names)

    def test_logits_and_gradients_match_the_tail_recompute(self):
        for W, C, T in self.CASES:
            diff, cos, rel, n = self.compare(W, C, T)
            self.assertLess(diff, 0.01, (W, C, T, "logits"))  # rounding is at most ~0.004 here; a window one token too small gives ~0.017
            self.assertGreater(cos, 0.99, (W, C, T, "gradient direction"))
            self.assertLess(rel, 0.1, (W, C, T, "gradient size"))
            self.assertGreater(n, 10)  # many parameters carry a gradient, including the loops

    def test_the_comparison_would_notice_a_stock_sliding_cache(self):
        """Sensitivity check: a cache that trims to W *after* appending the chunk (what fla's own sliding cache does) shrinks the window
        of the first queries of every chunk; the comparison must fail for it, and for a window that is merely one token too small."""
        import models
        real = models._TailCache

        class StockLike(real):  # returns only the last W keys, chunk included
            def update(self, attn_state, layer_idx, offset=1, cache_kwargs=None, **_):
                out = super().update(attn_state, layer_idx, offset, cache_kwargs)["attn_state"]
                return {"attn_state": tuple(x[:, -(self.keep + 1):] for x in out)}

        class OneShort(real):
            def __init__(self, keep, n_layers):
                super().__init__(keep - 1, n_layers)

        for bad in (StockLike, OneShort):
            models._TailCache = bad
            try:
                diff, cos, rel, _ = self.compare(16, 8, 100)
            finally:
                models._TailCache = real
            self.assertGreater(diff, 0.01, bad.__name__)

    def test_cache_keeps_the_last_window_minus_one_tokens_and_returns_them_with_the_new_ones(self):
        from models import _TailCache
        c = _TailCache(keep=5, n_layers=1)
        k1, v1 = torch.arange(8.0).view(1, 8, 1), torch.arange(8.0).view(1, 8, 1) * 10
        out_k, _ = c.update((k1, v1), layer_idx=0)["attn_state"]
        self.assertEqual(out_k.flatten().tolist(), list(range(8)))              # nothing cached yet: just the new chunk
        self.assertEqual((c.get_seq_length(0), c.k[0].flatten().tolist()), (8, [3.0, 4.0, 5.0, 6.0, 7.0]))
        k2 = torch.arange(8.0, 10.0).view(1, 2, 1)
        out_k, _ = c.update((k2, k2 * 10), layer_idx=0)["attn_state"]
        self.assertEqual(out_k.flatten().tolist(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])  # cached W-1 keys, then the chunk
        self.assertEqual(c.k[0].flatten().tolist(), [5.0, 6.0, 7.0, 8.0, 9.0])

    def test_long_streams_beyond_the_position_table_are_refused(self):
        a = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=8, segment=8, kv_cache=True).cuda()
        a.base.config.max_position_embeddings = 64
        with self.assertRaises(ValueError):
            a(input_ids=torch.randint(1, 40, (1, 96), device="cuda"))


if __name__ == "__main__":
    unittest.main()
