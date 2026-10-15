"""Tests for the fla-based models (need a CUDA GPU for the kernels).

    python -m unittest discover -s tests -v
"""
import math
import unittest

import torch

from tests import _paths  # noqa: F401
from models import FeedbackSWA, TwoPassSWA, build_from_config, build_model, build_swa, build_swa_2pass, build_swa_fb, swa_reach


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
        mc = {"kind": "swa_fb", "d_model": 128, "n_layers": 2, "num_heads": 4, "window": 64, "kv_cache": True}  # E7: the config states the one path that is left
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


def reference_forward(fb: FeedbackSWA, input_ids: torch.Tensor) -> torch.Tensor:
    """The loop model written the plain way, as its definition says: for segment s and layer i,  inp = x_s + gain * m_i(s-1),  y = block_i over [the last W-1 inputs of the earlier
    segments ; inp] (the whole context is recomputed), m_i(s) = loop_i(y). This is the tail-recompute path that E7 removed from the model (it gave the same results as the key/value
    cache but was slower); it stays here as the reference the cache path is compared with."""
    B, T = input_ids.shape
    C, W = fb.segment, fb.window
    if T % C:
        input_ids = torch.nn.functional.pad(input_ids, (0, C - T % C), value=0)
    x = fb.base.model.embeddings(input_ids)
    layers = fb.base.model.layers
    tails, mems, outs = [None] * len(layers), [None] * len(layers), []
    for s in range(x.shape[1] // C):
        h = x[:, s * C:(s + 1) * C]
        for i, (layer, loop) in enumerate(zip(layers, fb.loops)):
            inp = h if mems[i] is None else loop.inject(h, mems[i])
            ctx = inp if tails[i] is None else torch.cat([tails[i], inp], dim=1)
            y = layer(ctx)[0][:, -C:]
            tails[i] = ctx[:, -(W - 1):]
            mems[i] = loop(y)
            h = y
        outs.append(h)
    return fb.base.lm_head(fb.base.model.norm(torch.cat(outs, dim=1)[:, :T]))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class CacheMatchesTheDefinition(unittest.TestCase):
    """The key/value cache path of `FeedbackSWA` (the only one since E7) must give the same logits and gradients as the definition written out in `reference_forward`, with the loops
    switched on (m != 0).

    Strong feedback is chaotic: the recomputing path with weights perturbed by 1e-4 differs from itself by ~0.8 in logits after a few dozen hops, so rounding noise would swamp the
    comparison. The feedback here is the mild recipe scale (projection std 0.1, gain 0.1), where the two agree to bf16 rounding (~0.004 in logits of scale 0.13, 37 hops, window 64)."""

    CASES = ((16, 8, 100), (16, 16, 100), (16, 24, 90), (16, 5, 77), (8, 24, 90), (64, 16, 400), (64, 8, 300))  # window, segment, length: C < W, = W, > W, T % C != 0

    def model(self, W, C):
        m = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=W, segment=C).cuda()
        strong_feedback(m, std=0.1, gain=0.1)
        return m

    def logits_and_grads(self, m, x, forward):
        m.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward(m, x).float()
        logits.logsumexp(-1).mean().backward()
        return logits.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    def compare(self, W, C, T):
        torch.manual_seed(0)
        m = self.model(W, C)
        x = torch.randint(1, 40, (3, T), device="cuda")
        (la, ga) = self.logits_and_grads(m, x, reference_forward)
        (lb, gb) = self.logits_and_grads(m, x, lambda mm, xx: mm(input_ids=xx).logits)
        names = [n for n in ga if ga[n].norm() > 1e-6]
        cos = min(float(torch.nn.functional.cosine_similarity(ga[n].flatten(), gb[n].flatten(), dim=0)) for n in names)
        rel = max(float((ga[n] - gb[n]).norm() / ga[n].norm()) for n in names)
        return float((la - lb).abs().max()), cos, rel, len(names)

    def test_logits_and_gradients_match_the_definition(self):
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
        a = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=8, segment=8).cuda()
        a.base.config.max_position_embeddings = 64
        with self.assertRaises(ValueError):
            a(input_ids=torch.randint(1, 40, (1, 96), device="cuda"))


def strong_two_pass(model: TwoPassSWA, gain=0.5):
    """Switch the second pass on: every gain `gain` (the default init has gain 0, i.e. plain SWA)."""
    for loop in model.loops:
        torch.nn.init.constant_(loop.gain, gain)


class TwoPassBuilder(unittest.TestCase):
    MC = {"kind": "swa_2pass", "d_model": 128, "n_layers": 2, "num_heads": 4, "window": 64, "pass_output": 1}  # E7: the config states the one output that is left

    def test_dispatch_defaults_and_parameter_names(self):
        m = build_from_config({**self.MC, "segment": 32}, 40)
        self.assertIsInstance(m, TwoPassSWA)
        self.assertEqual((m.window, m.segment), (64, 32))
        self.assertFalse(hasattr(m, "pass_output"))  # the output of pass 2 is gone
        self.assertEqual(build_from_config(self.MC, 40).segment, 64)  # segment 0/absent -> one per window
        plain = build_from_config({**self.MC, "kind": "swa"}, 40)
        extra = sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in plain.parameters())
        self.assertEqual(extra, 2 * 128)  # one gain per channel and layer, nothing else
        self.assertTrue(all(float(l.gain.abs().max()) == 0.0 for l in m.loops))  # starts as plain SWA
        self.assertAlmostEqual(float(build_from_config({**self.MC, "gain_init": 0.1}, 40).loops[1].gain[5]), 0.1, places=6)
        base_names = {f"base.{n}" for n in plain.state_dict()}
        self.assertEqual(set(m.state_dict()) - base_names, {f"loops.{i}.gain" for i in range(2)})
        self.assertTrue(base_names <= set(m.state_dict()))  # a plain SWA checkpoint loads into `base`

    def test_bad_arguments_and_the_removed_settings_are_refused(self):
        for bad in ({"pass_output": 2}, {"pass_output": 3}, {"kv_cache": False}):
            with self.assertRaises(ValueError, msg=str(bad)):
                build_from_config({**self.MC, **bad}, 40)
        build_from_config({**self.MC, "kv_cache": True}, 40)
        with self.assertRaises(ValueError):
            build_swa_2pass(vocab_size=40, window=16, segment=-1)

    def test_the_gains_are_in_the_loop_parameter_group(self):
        import train as T
        m = build_from_config(self.MC, 40)
        opt = T.make_optimizer(m, {"weight_decay": 0.01, "lr": 1e-3, "betas": [0.9, 0.95], "loop_lr_mult": 10.0})
        loop_group = [g for g in opt.param_groups if g.get("lr_mult") == 10.0]
        self.assertEqual(len(loop_group), 1)
        self.assertEqual(sorted(id(p) for p in loop_group[0]["params"]), sorted(id(l.gain) for l in m.loops))
        self.assertEqual(loop_group[0]["weight_decay"], 0.0)


class PeekCache(unittest.TestCase):
    def test_a_view_returns_what_the_cache_would_but_stores_nothing_and_does_not_advance(self):
        from models import _PeekCache, _TailCache
        c = _TailCache(keep=5, n_layers=1)
        k1 = torch.arange(8.0).view(1, 8, 1)
        c.update((k1, k1 * 10), layer_idx=0)
        before = (c.k[0].clone(), c.v[0].clone(), c.seen[0])
        view = _PeekCache(c)
        k2 = torch.arange(8.0, 10.0).view(1, 2, 1)
        out_k, out_v = view.update((k2, k2 * 10), layer_idx=0)["attn_state"]
        self.assertEqual(out_k.flatten().tolist(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])  # the cached W-1 entries, then the new ones: what the real update returns
        self.assertEqual(out_v.flatten().tolist(), [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
        self.assertEqual(view.get_seq_length(0), 8)
        self.assertTrue(torch.equal(c.k[0], before[0]) and torch.equal(c.v[0], before[1]) and c.seen[0] == before[2])  # nothing stored, the position did not move
        empty = _PeekCache(_TailCache(5, 1))
        self.assertEqual(empty.update((k2, k2), layer_idx=0)["attn_state"][0].flatten().tolist(), [8.0, 9.0])  # an empty cache: just the new entries


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TwoPassBehaviour(unittest.TestCase):
    def run_model(self, m, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(input_ids=x).logits.float()

    def build(self, window=16, segment=8, layers=2, **kw):
        return build_swa_2pass(vocab_size=40, d_model=64, n_layers=layers, num_heads=2, window=window, segment=segment, **kw).cuda().eval()

    def test_zero_gain_equals_plain_swa(self):
        """Gain 0: pass 2 writes what pass 1 read and the model is the plain sliding-window forward; C < W, C = W, C > W and a length that is not a multiple of C."""
        torch.manual_seed(0)
        for window, segment, length in ((16, 8, 100), (16, 16, 100), (16, 24, 90), (16, 5, 77), (8, 24, 90), (64, 8, 300)):
            m = self.build(window, segment)
            x = torch.randint(8, 40, (2, length), device="cuda")
            diff = (self.run_model(m, x) - self.run_model(m.base, x)).abs().max().item()
            self.assertLess(diff, 5e-2, (window, segment, length, diff))

    def test_the_second_pass_changes_what_later_segments_read_and_the_layer_outputs_pass_1(self):
        torch.manual_seed(0)
        x = torch.randint(8, 40, (2, 64), device="cuda")
        m = self.build(16, 8)
        strong_two_pass(m)
        plain = self.run_model(m.base, x)
        got = self.run_model(m, x)
        self.assertGreater((got - plain).abs().max().item(), 0.05)  # with the gain on, the second pass is not a no-op: later segments read what it wrote
        # the first segment reads an empty cache, so its pass 1 is the plain block whatever the gain is, and the layer passes pass 1 on: the logits of segment 0 are the plain model's
        self.assertLess((got[:, :8] - plain[:, :8]).abs().max().item(), 5e-2)

    def spy(self, m, x, pass_1_writes=False):
        """Run `m` with spies on both caches: every update of the real cache (pass 2) and of the read-only view (pass 1), per layer, with the keys / values passed in."""
        import models
        rec = {"write": [], "peek": [], "cache": []}
        real_tail, real_peek = models._TailCache, models._PeekCache

        class TailSpy(real_tail):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                rec["cache"].append(self)

            def update(self, attn_state, layer_idx, offset=1, cache_kwargs=None, **kw):
                rec["write"].append((layer_idx, attn_state[0].detach().clone(), attn_state[1].detach().clone()))
                return super().update(attn_state, layer_idx, offset, cache_kwargs, **kw)

        class PeekSpy(real_peek):
            def update(self, attn_state, layer_idx, offset=1, cache_kwargs=None, **kw):
                rec["peek"].append((layer_idx, attn_state[0].detach().clone(), attn_state[1].detach().clone()))
                return super().update(attn_state, layer_idx, offset, cache_kwargs, **kw)

        models._TailCache, models._PeekCache = TailSpy, (lambda cache: cache) if pass_1_writes else PeekSpy  # (the broken variant: pass 1 goes through the real cache)
        try:
            logits = self.run_model(m, x)
        finally:
            models._TailCache, models._PeekCache = real_tail, real_peek
        return logits, rec

    def test_each_token_is_stored_once_with_the_keys_and_values_of_pass_2(self):
        """Review E2: after a forward pass every token's keys / values are in the cache exactly once, `seen` is the number of tokens, and what is stored is pass 2's, not pass 1's
        (the layer's output is pass 1's, the cache holds pass 2's)."""
        torch.manual_seed(0)
        W, C, L, T = 16, 8, 2, 8 * 9
        m = self.build(W, C, L)
        strong_two_pass(m, 0.5)
        x = torch.randint(8, 40, (2, T), device="cuda")
        _, rec = self.spy(m, x)
        cache = rec["cache"][-1]
        self.assertEqual(len(rec["write"]), L * (T // C))   # one write per layer and segment: nothing stored twice
        self.assertEqual(len(rec["peek"]), L * (T // C))    # and one read-only call per layer and segment
        self.assertEqual(cache.seen, [T] * L)               # the position counter counts every token once
        for i in range(L):
            writes = torch.cat([k for l, k, v in rec["write"] if l == i], dim=1)
            values = torch.cat([v for l, k, v in rec["write"] if l == i], dim=1)
            peeks = torch.cat([k for l, k, v in rec["peek"] if l == i], dim=1)
            self.assertEqual(writes.shape[1], T)
            self.assertEqual(float((cache.k[i] - writes[:, -(W - 1):]).abs().max()), 0.0, i)  # the stored window is pass 2's keys ...
            self.assertEqual(float((cache.v[i] - values[:, -(W - 1):]).abs().max()), 0.0, i)  # ... and values
            self.assertGreater(float((cache.k[i] - peeks[:, -(W - 1):]).abs().max()), 1e-3, i)  # not pass 1's (the gain is on, so they differ)

    def test_the_bookkeeping_test_would_notice_a_first_pass_that_writes(self):
        """Sensitivity check: if pass 1 went through the real cache, every token would be stored twice and `seen` would double; the test above must fail for that."""
        m = self.build(16, 8, 2)
        strong_two_pass(m, 0.5)
        x = torch.randint(8, 40, (2, 72), device="cuda")
        _, rec = self.spy(m, x, pass_1_writes=True)
        self.assertEqual(rec["cache"][-1].seen, [144, 144])  # twice the tokens: the position counter and the cache are corrupted
        self.assertEqual(len(rec["write"]), 2 * 2 * 9)       # two writes per layer and segment instead of one: the check of the real model (one) fails for this

    def test_causality(self):
        torch.manual_seed(0)
        m = self.build(8, 8)
        strong_two_pass(m)
        x = torch.randint(8, 40, (1, 96), device="cuda")
        x2 = x.clone()
        x2[0, 50] = 8 + (int(x[0, 50]) - 8 + 1) % 32
        d = (self.run_model(m, x) - self.run_model(m, x2)).abs().amax(-1)[0]
        self.assertEqual(float(d[:50].max()), 0.0)
        self.assertGreater(float(d[50:].max()), 0.0)

    def test_the_second_pass_carries_information_past_the_window_reach(self):
        torch.manual_seed(0)
        window, layers = 8, 2
        R = swa_reach(window, layers)
        x = torch.randint(8, 40, (1, 8 * 15), device="cuda")
        x2 = x.clone()
        x2[0, 1] = 8 + (int(x[0, 1]) - 8 + 1) % 32
        far = slice(R + 2 * window, None)  # positions the window alone can never see
        for gain, expect_leak in ((0.0, False), (0.5, True)):
            m = self.build(window, window, layers)
            strong_two_pass(m, gain)
            d = (self.run_model(m, x) - self.run_model(m, x2)).abs().amax(-1)[0, far]
            self.assertEqual(bool((d > 0).any()), expect_leak, (gain, float(d.max())))

    def test_gradients_reach_the_gains_through_the_segments(self):
        """The loss is on every token of the last segment, which reads the cache of the segments before it (the very last token alone does not when C = W: its window is its own
        segment, and the last layer's gain acts only through what other tokens read)."""
        m = build_swa_2pass(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=8, segment=8).cuda()
        x = torch.randint(8, 40, (2, 64), device="cuda")
        for gain in (0.0, 0.5):
            m.zero_grad()
            strong_two_pass(m, gain)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(input_ids=x).logits.float()[:, -8:].sum().backward()
            for i, loop in enumerate(m.loops):
                self.assertGreater(float(loop.gain.grad.abs().sum()), 0.0, (gain, i))  # the gain can leave 0, in every layer

    def test_state_dict_roundtrip_and_a_plain_checkpoint_loads_into_base(self):
        a, b = self.build(16, 8), self.build(16, 8)
        strong_two_pass(a)
        b.load_state_dict(a.state_dict())
        x = torch.randint(8, 40, (1, 50), device="cuda")
        self.assertEqual(float((self.run_model(a, x) - self.run_model(b, x)).abs().max()), 0.0)
        plain = build_swa(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=16).cuda().eval()
        c = self.build(16, 8)
        c.base.load_state_dict(plain.state_dict())
        self.assertLess((self.run_model(c, x) - self.run_model(plain, x)).abs().max().item(), 5e-2)

    def test_long_streams_beyond_the_position_table_are_refused(self):
        m = self.build(8, 8)
        m.base.config.max_position_embeddings = 64
        with self.assertRaises(ValueError):
            m(input_ids=torch.randint(1, 40, (1, 96), device="cuda"))


def pass_two_as_a_whole_block(model: TwoPassSWA, input_ids: torch.Tensor, hook=None) -> torch.Tensor:
    """`TwoPassSWA.forward` as it was before E6, kept as the reference: pass 2 runs the whole block on h + g * (y1 - h) through the real cache (attention, output projection and MLP
    included) and drops what it returns. `hook(segment, cache)` is called after every segment like the model's `cache_hook`."""
    from models import _PeekCache, _TailCache
    B, T = input_ids.shape
    C, W = model.segment, model.window
    if T % C:
        input_ids = torch.nn.functional.pad(input_ids, (0, C - T % C), value=0)
    x = model.base.model.embeddings(input_ids)
    layers = model.base.model.layers
    cache = _TailCache(W - 1, len(layers))
    outs = []
    for s in range(x.shape[1] // C):
        h = x[:, s * C:(s + 1) * C]
        for layer, loop in zip(layers, model.loops):
            y1 = layer(h, past_key_values=_PeekCache(cache), use_cache=True)[0]
            h2 = h + (loop.gain * (y1 - h).float()).to(h.dtype)
            layer(h2, past_key_values=cache, use_cache=True)
            h = y1
        outs.append(h)
        if hook is not None:
            hook(s, cache)
    return model.base.lm_head(model.base.model.norm(torch.cat(outs, dim=1)[:, :T]))


class RecordCache:
    """A cache hook that keeps a copy of the window's keys and values of every layer after every segment."""

    def __init__(self):
        self.snap = {}

    def __call__(self, s, cache):
        self.snap[s] = [(cache.k[i].detach().clone(), cache.v[i].detach().clone()) for i in range(len(cache.k))]


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class PassTwoIsTheCacheWriteOnly(unittest.TestCase):
    """Review E6: with the output of pass 1, pass 2 matters only through the keys and values it writes, so the model computes only those (`write_kv`). The same function as running
    the block and dropping its output: equal cache contents after every segment and layer, equal logits, equal gradients."""

    def models(self, qk_norm=False, window=16, segment=8, seed=0):
        from fla.models import TransformerConfig, TransformerForCausalLM
        torch.manual_seed(seed)
        cfg = TransformerConfig(hidden_size=64, num_hidden_layers=2, num_heads=2, window_size=window, vocab_size=40, max_position_embeddings=1 << 17, tie_word_embeddings=True,
                                fuse_cross_entropy=False, use_cache=False, qk_norm=qk_norm)
        m = TwoPassSWA(TransformerForCausalLM(cfg), window=window, segment=segment).cuda().eval()
        strong_two_pass(m, 0.5)
        return m

    def go(self, m, x, forward, record=None):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return forward(m, x, record).float()

    def new(self, m, x, record=None):
        m.cache_hook = record
        try:
            return m(input_ids=x).logits
        finally:
            m.cache_hook = None

    def test_cache_contents_and_logits_are_equal_after_every_segment(self):
        for qk_norm in (False, True):
            m = self.models(qk_norm)
            x = torch.randint(8, 40, (3, 8 * 11), device="cuda")
            a, b = RecordCache(), RecordCache()
            la = self.go(m, x, lambda mm, xx, rec: pass_two_as_a_whole_block(mm, xx, rec), a)
            lb = self.go(m, x, self.new, b)
            self.assertEqual(sorted(a.snap), sorted(b.snap))
            for s in a.snap:
                for i, ((ka, va), (kb, vb)) in enumerate(zip(a.snap[s], b.snap[s])):
                    self.assertTrue(torch.equal(ka, kb) and torch.equal(va, vb), (qk_norm, s, i))
            self.assertEqual(float((la - lb).abs().max()), 0.0, qk_norm)

    def test_gradients_are_equal(self):
        m = self.models()
        x = torch.randint(8, 40, (3, 8 * 11), device="cuda")
        grads = []
        for forward in (lambda mm, xx: pass_two_as_a_whole_block(mm, xx), lambda mm, xx: mm(input_ids=xx).logits):
            m.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                forward(m, x).float().logsumexp(-1).mean().backward()
            grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None})
        self.assertEqual(set(grads[0]), set(grads[1]))
        self.assertTrue(any(float(g.abs().sum()) > 0 for n, g in grads[0].items() if n.startswith("loops.")))  # the gains get a gradient, through the cache only
        for n in grads[0]:
            self.assertLess(float((grads[0][n] - grads[1][n]).abs().max()), 1e-6 + 1e-3 * float(grads[0][n].abs().max()), n)

    def test_the_positions_of_the_stored_entries_are_the_caches(self):
        """The RoPE offset of pass 2 is the number of tokens stored before: a write with a wrong offset would change the keys of every later segment."""
        m = self.models()
        x = torch.randint(8, 40, (2, 8 * 6), device="cuda")
        rec = RecordCache()
        self.go(m, x, self.new, rec)
        ref = RecordCache()
        self.go(m, x, lambda mm, xx, r: pass_two_as_a_whole_block(mm, xx, r), ref)
        for s in (0, 1, 5):
            for (ka, _), (kb, _) in zip(rec.snap[s], ref.snap[s]):
                self.assertTrue(torch.equal(ka, kb), s)

    def test_nothing_but_the_key_and_value_projections_runs_in_pass_two(self):
        """The attention, its output projection and the MLP are called once per layer and segment (pass 1) instead of twice."""
        m = self.models()
        x = torch.randint(8, 40, (2, 8 * 6), device="cuda")
        calls = {"attn": 0, "mlp": 0, "o_proj": 0}
        hooks = []
        for layer in m.base.model.layers:
            hooks.append(layer.attn.register_forward_hook(lambda *a: calls.__setitem__("attn", calls["attn"] + 1)))
            hooks.append(layer.mlp.register_forward_hook(lambda *a: calls.__setitem__("mlp", calls["mlp"] + 1)))
            hooks.append(layer.attn.o_proj.register_forward_hook(lambda *a: calls.__setitem__("o_proj", calls["o_proj"] + 1)))
        self.go(m, x, self.new)
        for h in hooks:
            h.remove()
        self.assertEqual(calls, {"attn": 2 * 6, "mlp": 2 * 6, "o_proj": 2 * 6})  # layers x segments: one pass each (it was twice that)

    CKPT = _paths.ROOT / "checkpoints" / "swa_2pass_seg32_po1_s5.pt"
    DATA = _paths.ROOT / "data_cache" / "passkey_swar" / "n256" / "test.parquet"

    @unittest.skipUnless(CKPT.exists() and DATA.exists(), "needs the backed-up checkpoint and the dataset")
    def test_the_best_checkpoint_gives_the_same_cache_logits_and_gradients_eager_and_in_a_cuda_graph(self):
        import json
        import data as D
        import graphed
        import train as T
        cfg = json.loads((_paths.ROOT / "results" / "raw" / "swa_2pass_seg32_po1_s5.json").read_text())["config"]
        m = build_from_config(cfg["model"], 40).cuda()
        m.load_state_dict(torch.load(self.CKPT, map_location="cpu", weights_only=True))
        m.eval()
        ids_np, labs_np = D.pad_batch(D.load_packed(self.DATA), torch.arange(6).numpy(), 32)   # six real streams of N = 256: about 30 segments of 32 tokens
        ids, labs = torch.from_numpy(ids_np).cuda(), torch.from_numpy(labs_np).cuda()
        a, b = RecordCache(), RecordCache()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.zero_grad()
            la = pass_two_as_a_whole_block(m, ids, a)
            T.masked_ce(la, labs).backward()
            ga = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
            m.zero_grad()
            m.cache_hook = b
            lb = m(input_ids=ids).logits
            m.cache_hook = None
            T.masked_ce(lb, labs).backward()
            gb = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        self.assertGreaterEqual(len(a.snap), 29)
        for s in a.snap:
            for i, ((ka, va), (kb, vb)) in enumerate(zip(a.snap[s], b.snap[s])):
                self.assertTrue(torch.equal(ka, kb) and torch.equal(va, vb), (s, i))   # equal cache contents per segment and layer
        self.assertEqual(float((la - lb).abs().max()), 0.0)                            # equal logits
        self.assertEqual(set(ga), set(gb))
        for n in ga:                                                                     # equal gradients of the loss on this batch
            self.assertLess(float((ga[n] - gb[n]).abs().max()), 1e-6 + 1e-3 * float(ga[n].abs().max()), n)

        class Old(torch.nn.Module):   # the CUDA-graph step around the old forward
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, **_):
                from types import SimpleNamespace
                return SimpleNamespace(logits=pass_two_as_a_whole_block(self.model, input_ids))

        tc = {"lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.01, "loop_lr_mult": 10.0, "grad_clip": 1.0}
        out = []
        for wrap in (lambda mm: Old(mm), lambda mm: mm):
            mm = build_from_config(cfg["model"], 40).cuda()
            mm.load_state_dict(torch.load(self.CKPT, map_location="cpu", weights_only=True))
            mm.train()
            opt = T.make_optimizer(mm, tc, capturable=True)
            stepper = graphed.GraphedTrainStep(wrap(mm), opt, T.masked_ce, 1.0, torch.bfloat16)
            losses = [float(stepper.step(ids, labs, [1e-4 * g.get("lr_mult", 1.0) for g in opt.param_groups])[0]) for _ in range(3)]
            out.append((losses, [p.detach().clone() for p in mm.parameters()]))
        for x, y in zip(*[o[0] for o in out]):
            self.assertAlmostEqual(x, y, places=5)                                       # the graph steps follow each other
        self.assertLess(max(float((p - q).abs().max()) for p, q in zip(out[0][1], out[1][1])), 1e-5)


if __name__ == "__main__":
    unittest.main()
