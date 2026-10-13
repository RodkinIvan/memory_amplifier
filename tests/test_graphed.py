"""The CUDA-graph training step computes what the eager step computes (needs a GPU).

    python -m unittest tests.test_graphed -v
"""
import unittest

import torch

from tests import _paths  # noqa: F401
import graphed as G
import train as T
from models import build_swa_2pass, build_swa_fb, build_swa_va

TC = {"lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.01, "loop_lr_mult": 10.0, "grad_clip": 1.0}
LOSS = lambda logits, labs: T.weighted_ce(logits, labs, 1.0)  # noqa: E731


def model_and_opt(capturable: bool, segment=8, nonlin="tanh", kind="swa_fb"):
    torch.manual_seed(0)
    if kind in ("swa_2pass", "swa_2pass1"):  # two-pass layer, switched on (gain 0.3), second or first pass to the next layer
        m = build_swa_2pass(vocab_size=40, d_model=64, num_heads=2, window=16, segment=segment, gain_init=0.3, pass_output=1 if kind == "swa_2pass1" else 2).cuda()
        return m, T.make_optimizer(m, TC, capturable=capturable)
    if kind == "swa_va":  # value adaptation, switched on (gain 0.3): every parameter gets a gradient
        m = build_swa_va(vocab_size=40, d_model=64, num_heads=2, window=16, segment=segment, gain_init=0.3, proj_std=0.2).cuda()
        return m, T.make_optimizer(m, TC, capturable=capturable)
    m = build_swa_fb(vocab_size=40, d_model=64, num_heads=2, window=16, segment=segment, nonlin=nonlin, gain_init=0.1, proj_std=0.1).cuda()
    return m, T.make_optimizer(m, TC, capturable=capturable)


def batches(shapes, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for L in shapes:
        ids = torch.randint(0, 40, (8, L), generator=g)
        out.append((ids.cuda(), torch.where(torch.rand(8, L, generator=g) < 0.2, ids, torch.full_like(ids, -100)).cuda()))
    return out


def eager_steps(m, opt, data, lrs):
    losses = []
    for (ids, labs), lr in zip(data, lrs):
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(input_ids=ids).logits
        loss = LOSS(logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), TC["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    return losses


def graph_steps(m, opt, data, lrs):
    stepper = G.GraphedTrainStep(m, opt, LOSS, TC["grad_clip"], torch.bfloat16)
    losses = []
    for (ids, labs), lr in zip(data, lrs):
        loss, _ = stepper.step(ids, labs, [lr * g.get("lr_mult", 1.0) for g in opt.param_groups])
        losses.append(float(loss))
    return stepper, losses


def max_param_diff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class GraphedStep(unittest.TestCase):
    def test_one_step_equals_the_eager_step_up_to_float_rounding(self):
        for segment, nonlin in ((8, "tanh"), (4, "none")):
            data = batches([64])
            m1, o1 = model_and_opt(False, segment, nonlin)
            eager_steps(m1, o1, data, [2e-4])
            m2, o2 = model_and_opt(True, segment, nonlin)
            graph_steps(m2, o2, data, [2e-4])
            self.assertLess(max_param_diff(m1, m2), 1e-6, (segment, nonlin))

    def test_value_adaptation_model_is_captured_like_the_loop_model(self):
        """`swa_va` (forward hooks, key/value cache, scalar gains) in a CUDA graph: one step equals the eager step, and the loss trace follows."""
        data = batches([64])
        m1, o1 = model_and_opt(False, 8, kind="swa_va")
        eager_steps(m1, o1, data, [2e-4])
        m2, o2 = model_and_opt(True, 8, kind="swa_va")
        graph_steps(m2, o2, data, [2e-4])
        self.assertLess(max_param_diff(m1, m2), 1e-6)
        data, lrs = batches([32, 64, 32, 96, 64]), [2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
        m1, o1 = model_and_opt(False, 8, kind="swa_va")
        l1 = eager_steps(m1, o1, data, lrs)
        m2, o2 = model_and_opt(True, 8, kind="swa_va")
        _, l2 = graph_steps(m2, o2, data, lrs)
        self.assertLess(max(abs(a - b) for a, b in zip(l1, l2)), 2e-3, (l1, l2))

    def test_the_two_pass_model_is_captured_too(self):
        """`swa_2pass` (a read-only cache view, two block calls per segment) in a CUDA graph, for both outputs: one step equals the eager step, and the loss trace follows."""
        for kind in ("swa_2pass", "swa_2pass1"):
            data = batches([64])
            m1, o1 = model_and_opt(False, 8, kind=kind)
            eager_steps(m1, o1, data, [2e-4])
            m2, o2 = model_and_opt(True, 8, kind=kind)
            graph_steps(m2, o2, data, [2e-4])
            self.assertLess(max_param_diff(m1, m2), 1e-6, kind)
            data, lrs = batches([32, 64, 32, 96, 64]), [2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
            m1, o1 = model_and_opt(False, 8, kind=kind)
            l1 = eager_steps(m1, o1, data, lrs)
            m2, o2 = model_and_opt(True, 8, kind=kind)
            _, l2 = graph_steps(m2, o2, data, lrs)
            self.assertLess(max(abs(a - b) for a, b in zip(l1, l2)), 2e-3, (kind, l1, l2))

    def test_the_loss_trace_follows_the_eager_run_over_several_shapes(self):
        # Adam amplifies float rounding in parameters whose gradient is ~0, so weights drift apart after a few steps (an eager run and a
        # run with capturable AdamW drift just the same); the losses, which are what training follows, stay together.
        data = batches([32, 64, 32, 96, 64, 32])
        lrs = [2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 1e-3]
        m1, o1 = model_and_opt(False)
        l1 = eager_steps(m1, o1, data, lrs)
        m2, o2 = model_and_opt(True)
        stepper, l2 = graph_steps(m2, o2, data, lrs)
        self.assertLess(max(abs(a - b) for a, b in zip(l1, l2)), 2e-3, (l1, l2))
        self.assertEqual(sorted(stepper.graphs), [(8, 32), (8, 64), (8, 96)])  # one graph per batch shape, reused afterwards

    def test_capturing_does_not_train_the_model(self):
        m, o = model_and_opt(True)
        before = [p.detach().clone() for p in m.parameters()]
        stepper = G.GraphedTrainStep(m, o, LOSS, TC["grad_clip"], torch.bfloat16)
        ids, labs = batches([32])[0]
        stepper.step(ids, labs, [0.0, 0.0, 0.0])  # capture (3 warm-up steps, then the graph) and one replay at lr 0
        self.assertEqual(max((p - q).abs().max().item() for p, q in zip(m.parameters(), before)), 0.0)
        self.assertEqual({int(st["step"]) for st in o.state.values()}, {1})  # the warm-up steps are not counted; the one replay is

    def test_learning_rate_zero_changes_nothing_and_a_positive_one_does(self):
        m, o = model_and_opt(True)
        stepper = G.GraphedTrainStep(m, o, LOSS, TC["grad_clip"], torch.bfloat16)
        ids, labs = batches([32])[0]
        before = [p.detach().clone() for p in m.parameters()]
        stepper.step(ids, labs, [0.0] * 3)
        same = max((p - q).abs().max().item() for p, q in zip(m.parameters(), before))
        stepper.step(ids, labs, [1e-3] * 3)
        moved = max((p - q).abs().max().item() for p, q in zip(m.parameters(), before))
        self.assertEqual(same, 0.0)
        self.assertGreater(moved, 1e-4)

    def test_a_graphed_step_needs_tensor_learning_rates(self):
        m, o = model_and_opt(False)
        with self.assertRaises(TypeError):
            G.GraphedTrainStep(m, o, LOSS, 1.0, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
