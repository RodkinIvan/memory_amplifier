"""A training step (forward, loss, backward, gradient clipping, optimizer update) captured as one CUDA graph per input shape.

`swa_fb` runs ~130 small kernels per (segment, layer) call, so a step at segment 8 launches ~33k kernels and the GPU is idle most
of the time waiting for the CPU (review M8). Replaying a captured graph issues them all at once. It computes exactly what the eager
step computes: same kernels, same optimizer (AdamW with `capturable=True`, the learning rate is a device tensor that is filled before
every replay), so results agree with the eager path up to float rounding (tests/test_graphed.py).

Constraints: the loss weights must be constant (no passkey-weight annealing), every batch of one shape reuses the same static
input buffers, and the random streams of the graph (loop noise) are the graph-safe CUDA generator's, so a noisy run is not bit-identical
to its eager twin.
"""
from __future__ import annotations

import torch


class GraphedTrainStep:
    """`step(ids, labs, lrs)` runs one optimizer step on a batch and returns (loss, grad norm) as device tensors.

    The first call with a new input shape captures the graph: three warm-up iterations run first (kernel compilation and autotuning
    must not happen during capture) with every learning rate at 0, and the optimizer state is put back afterwards, so the warm-up
    does not change the training. All graphs share one memory pool and are replayed one at a time."""

    def __init__(self, model, opt: torch.optim.Optimizer, loss_fn, grad_clip: float, amp_dtype=None, warmup: int = 3):
        self.model, self.opt, self.loss_fn, self.grad_clip, self.amp_dtype, self.warmup = model, opt, loss_fn, grad_clip, amp_dtype, warmup
        self.params = [p for g in opt.param_groups for p in g["params"]]
        for g in opt.param_groups:
            if not isinstance(g["lr"], torch.Tensor):
                raise TypeError("a graphed step needs tensor learning rates (AdamW(..., capturable=True) with g['lr'] set to a device tensor)")
            g["lr"] = g["lr"].to(g["params"][0].device)  # a restored optimizer state brings its learning rates back on the CPU
        self.graphs: dict = {}
        self.pool = None

    # the work of one step; identical for the eager reference and for the graph
    def _work(self, ids, labs):
        with torch.autocast("cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None):
            logits = self.model(input_ids=ids).logits
        loss = self.loss_fn(logits, labs)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip, foreach=True)
        self.opt.step()
        return loss.detach(), gnorm

    def _state_tensors(self) -> list[torch.Tensor]:
        return [t for st in self.opt.state.values() for t in st.values() if isinstance(t, torch.Tensor)]

    def set_lrs(self, lrs: list[float]) -> None:
        for g, v in zip(self.opt.param_groups, lrs):
            g["lr"].fill_(float(v))

    def _capture(self, ids: torch.Tensor, labs: torch.Tensor) -> dict:
        had_state = bool(self.opt.state)
        saved = [t.clone() for t in self._state_tensors()]
        saved_lrs = [g["lr"].clone() for g in self.opt.param_groups]
        self.set_lrs([0.0] * len(saved_lrs))  # a step with lr 0 changes no parameter (decoupled weight decay is scaled by lr too)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self.opt.zero_grad(set_to_none=True)
                self._work(ids, labs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        # put the optimizer state back: the warm-up steps updated the moments and the step count
        if had_state:
            for t, s in zip(self._state_tensors(), saved):
                t.copy_(s)
        else:
            for t in self._state_tensors():
                t.zero_()
        for g, v in zip(self.opt.param_groups, saved_lrs):
            g["lr"].copy_(v)
        static_ids, static_labs = ids.clone(), labs.clone()
        graph = torch.cuda.CUDAGraph()
        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph, pool=self.pool):
            loss, gnorm = self._work(static_ids, static_labs)
        self.pool = self.pool or graph.pool()
        return {"graph": graph, "ids": static_ids, "labs": static_labs, "loss": loss, "gnorm": gnorm}

    def step(self, ids: torch.Tensor, labs: torch.Tensor, lrs: list[float]):
        key = tuple(ids.shape)
        if key not in self.graphs:
            self.graphs[key] = self._capture(ids, labs)
        g = self.graphs[key]
        g["ids"].copy_(ids)
        g["labs"].copy_(labs)
        self.set_lrs(lrs)
        g["graph"].replay()
        return g["loss"], g["gnorm"]
