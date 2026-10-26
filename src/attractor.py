#!/usr/bin/env python
"""E5: attractor states of the two-pass layer, and the passkey decoded from them (analysis of fixed checkpoints, no training).

    python src/attractor.py run --run swa_2pass_seg32_po1_s5 --out results/probes/p2_attractor_main.json

Hypothesis: the short-term store (the key/value window) settles into a state that is stationary under the incoming stream and from which the passkey can be decoded whatever the input.

The state S of a `TwoPassSWA` is, per layer, the pass-2 inputs h2 at the W - 1 = 63 most recent positions ([2 layers, 63, 128] at window 64): the keys and values of the window are computed from
them (`models.write_kv`), with RoPE at the window's positions; shifting the positions keeps the relative ones, so S carries no absolute position. Nothing else crosses a segment boundary.
The map F(S, x): build the cache from S, run one real segment x of C = 32 tokens, read the new window of h2 (the 31 entries kept, shifted, plus the 32 new ones).
S0 is the real window at the end of segment 2 (position 96) of a stream that saw the passkey; the objective is L(S) = E_x sum over layers of ||F(S, x) - S||^2 / ||S0||^2 over fresh random
segments x of associative-recall content (no PK / PKQ), minimised by Adam from S0. Tests of S* (and of S0 as the baseline): the passkey read out after K random segments, noise added to the
state, interpolation between the states of two passkeys, and whether starts with the same passkey reach the same S*. See `run` and the README of the results file for the recorded settings.

Streams are generated on the fly with their own seeds (SEEDS below), never the dataset's (1, 2, 3) or the earlier probes' (77, 78). All of it needs a CUDA GPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
import config as C  # noqa: E402
import make_passkey_swar as G  # noqa: E402

SEEDS = {"ar_pool": 91, "ar_decode": 92, "context": 93, "passkey": 94, "ar_eval": 95, "noise": 96, "optimizer": 97, "repeat": 98}
N_SEGMENTS_BEFORE = 3  # the state is the window at the end of segment 2: three segments of C tokens have gone by (position 96 at C = 32)
KS = (0, 1, 4, 32, 512)  # random segments between the state and the read-out
NOISE = (0.01, 0.03, 0.1, 0.3, 1.0)  # noise std relative to the RMS of the state (per layer)
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))


# ---- model and state machinery (no randomness) -------------------------------------------------------------------------------------------------------------------
def load_model(name: str, device="cuda"):
    """(TwoPassSWA, config, weights path, sha256) of a finished run: a swa_2pass run as it is; a plain `swa` run as a TwoPassSWA of segment 32 with every gain 0 (the same function), so the
    state of a plain window model is its layer inputs. Weights from checkpoints/<name>.pt (the local backup), else runs/<name>/model.pt."""
    import torch
    from models import TwoPassSWA, build_from_config
    cfg = json.loads((C.ROOT / "results" / "raw" / f"{name}.json").read_text())["config"]
    vocab = json.loads((C.data_dir_for(cfg) / "vocab.json").read_text())["size"]
    path = next(p for p in (C.ROOT / "checkpoints" / f"{name}.pt", C.ROOT / "runs" / name / "model.pt") if p.exists())
    model = build_from_config(cfg["model"], vocab)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    if cfg["model"]["kind"] == "swa":
        model = TwoPassSWA(model, window=cfg["model"]["window"], segment=32)
    elif cfg["model"]["kind"] != "swa_2pass":
        raise ValueError(f"{name}: E5 is for the two-pass layer (and a plain window model as its control), not {cfg['model']['kind']}")
    return model.to(device).eval(), cfg, path, hashlib.sha256(path.read_bytes()).hexdigest()


def cache_from_state(model, S, position: int):
    """A `_TailCache` holding the keys and values of the window state S [n, layers, W - 1, d], at the positions that end right before `position` (the position of the next token): the
    entries are at position - (W - 1) ... position - 1, so a run from this cache sees exactly what a run of the real stream sees at that point."""
    from models import _TailCache, write_kv
    layers = model.base.model.layers
    cache = _TailCache(model.window - 1, len(layers))
    for i, layer in enumerate(layers):
        cache.seen[i] = position - S.shape[2]
        write_kv(layer, S[:, i], cache)
        assert cache.seen[i] == position
    return cache


def clone_cache(cache):
    """A copy that can be advanced without moving the original (the tensors are replaced, never written in place, so copying the lists is enough)."""
    c = copy.copy(cache)
    c.k, c.v, c.seen = list(cache.k), list(cache.v), list(cache.seen)
    return c


def next_window(S, h2s, segment: int):
    """The window after a segment: per layer the last W - 1 entries of [S ; the segment's h2]."""
    import torch
    keep = S.shape[2]
    return torch.stack([torch.cat([S[:, i], h2s[i]], 1)[:, -keep:] for i in range(S.shape[1])], 1)


def window_states(model, ids, n_segments: int = N_SEGMENTS_BEFORE):
    """The real window after `n_segments` segments of the streams `ids` [n, >= n_segments * C]: [n, layers, W - 1, d] pass-2 inputs (no gradient)."""
    import torch
    from models import _TailCache
    C_ = model.segment
    cache = _TailCache(model.window - 1, len(model.base.model.layers))
    x = model.base.model.embeddings(ids[:, :n_segments * C_])
    keep = model.window - 1
    h2_all = [[] for _ in model.base.model.layers]
    with torch.no_grad():
        for s in range(n_segments):
            _, h2s = model.run_segment(x[:, s * C_:(s + 1) * C_], cache)
            for i, h2 in enumerate(h2s):
                h2_all[i].append(h2)
    return torch.stack([torch.cat(h, 1)[:, -keep:] for h in h2_all], 1).float()


def advance(model, cache, ids):
    """Run `ids` [n, T] (T a multiple of C) through the model from `cache` (advanced). Yields (segment index, output of the last layer [n, C, d], h2 of every layer)."""
    C_ = model.segment
    x = model.base.model.embeddings(ids)
    for s in range(ids.shape[1] // C_):
        h, h2s = model.run_segment(x[:, s * C_:(s + 1) * C_], cache)
        yield s, h, h2s


def pad_to_segment(ids, C_: int):
    import torch
    return torch.nn.functional.pad(ids, (0, -ids.shape[1] % C_), value=G.PAD)


def logits_of(model, h):
    return model.base.lm_head(model.base.model.norm(h))


# ---- held-out streams ---------------------------------------------------------------------------------------------------------------------------------------------
def ar_contents(params: G.GenParams, seed: int, n: int, n_writes: int) -> list[np.ndarray]:
    """The associative-recall part of `n` held-out streams (between `BOS PK p1 .. pP` and `PKQ p1 .. pP EOS`), whole `W k v` / `Q k a` triples."""
    out = []
    for i in range(n):
        t = G.make_example(G.example_seed(seed, n_writes, i), n_writes, params)[0]
        out.append(np.asarray(t[2 + params.passkey_len:-(params.passkey_len + 2)], np.int64))
    return out


def random_passkeys(params: G.GenParams, n: int, seed: int) -> np.ndarray:
    """[n, P] passkeys as token ids."""
    rng = np.random.default_rng(seed)
    return params.val0 + rng.integers(0, params.n_values, size=(n, params.passkey_len))


def other_values(params: G.GenParams, tokens: np.ndarray, rng) -> np.ndarray:
    """For each token another value token (uniformly among the other values)."""
    return params.val0 + (tokens - params.val0 + 1 + rng.integers(0, params.n_values - 1, size=tokens.shape)) % params.n_values


class Streams:
    """The states one run optimizes and tests: `n_pass` passkeys x `n_ctx` contexts (the first `START` tokens of held-out AR content), plus, for the context 0 of every passkey, a variant
    that differs in one position and one that differs in all. index(kind, k, c): `A` states are k * n_ctx + c; `B1` and `Ball` states follow.

    The dataset's stream is `BOS PK p1 .. pP` followed by whole `W k v` / `Q k a` triples, the first one at position 2 + P. The state is cut at position `position` (96), which is inside a
    triple unless (position - 2 - P) is a multiple of 3: that holds for P = 4 and P = 16 but not for P = 8 (review N3). `phase` = the number of tokens of the cut triple that the prefix
    already holds, `rest` = the other tokens of that triple, which the continuation of every state starts with (see `continuation`), so the stream keeps its own grid across the state boundary
    and PKQ lands where the dataset puts it (a multiple of 3 tokens after the first triple)."""

    def __init__(self, params: G.GenParams, segment: int, n_pass: int = 16, n_ctx: int = 4):
        self.params, self.n_pass, self.n_ctx = params, n_pass, n_ctx
        P = params.passkey_len
        start = self.position = N_SEGMENTS_BEFORE * segment
        self.first_triple = 2 + P
        self.phase = (start - self.first_triple) % 3  # tokens of a triple already in the prefix at the cut
        rest_len = (3 - self.phase) % 3
        self.pk_a = random_passkeys(params, n_pass, SEEDS["passkey"])
        rng = np.random.default_rng(SEEDS["passkey"] + 1000)
        self.pos_b1 = np.arange(n_pass) % P  # the position that differs in the one-position variant
        pk_b1 = self.pk_a.copy()
        pk_b1[np.arange(n_pass), self.pos_b1] = other_values(params, self.pk_a[np.arange(n_pass), self.pos_b1], rng)
        pk_ball = other_values(params, self.pk_a, rng)
        ctx = ar_contents(params, SEEDS["context"], n_ctx, 256)
        rows, rests, self.kind, self.k, self.c, pks = [], [], [], [], [], []
        for kind, pk_of, cs in (("A", lambda k: self.pk_a[k], range(n_ctx)), ("B1", lambda k: pk_b1[k], [0]), ("Ball", lambda k: pk_ball[k], [0])):
            for k in range(n_pass):
                for c in cs:
                    pk = pk_of(k)
                    rows.append(np.concatenate([[G.BOS, G.PK], pk, ctx[c]])[:start])
                    rests.append(ctx[c][start - self.first_triple:start - self.first_triple + rest_len])  # the tokens that complete the triple the cut broke
                    self.kind.append(kind), self.k.append(k), self.c.append(c), pks.append(pk)
        self.prefix, self.passkeys, self.rest = np.stack(rows), np.stack(pks), np.stack(rests).reshape(len(rows), rest_len)
        self.kind, self.k, self.c = np.array(self.kind), np.array(self.k), np.array(self.c)

    def index(self, kind: str, k, c: int = 0):
        base = {"A": 0, "B1": self.n_pass * self.n_ctx, "Ball": self.n_pass * self.n_ctx + self.n_pass}[kind]
        return base + (np.asarray(k) * self.n_ctx + c if kind == "A" else np.asarray(k))

    def __len__(self):
        return len(self.prefix)


def continuation(st: Streams, dec: list[np.ndarray], need: int) -> np.ndarray:
    """The held-out input that follows each state, [n_states, need]: the rest of the triple that the cut at the state's position broke (from the state's own context), then whole triples of
    held-out AR content, a different stretch per state (stream i % len(dec), shifted by a multiple of 21 tokens so that it still starts at a triple). The token at index j of a row is at
    position `st.position + j` of the stream, on the dataset's grid."""
    rows = [np.concatenate([st.rest[i], dec[i % len(dec)][3 * (i // len(dec)) * 7:]])[:need] for i in range(len(st))]
    assert all(len(r) == need for r in rows), "the held-out streams are too short"
    return np.stack(rows)


# ---- the objective -------------------------------------------------------------------------------------------------------------------------------------------------
class SegmentPool:
    """Random segments of C tokens from the AR content of held-out streams (no PK / PKQ), cut at the stream's own grid: the offset is a multiple of 3 plus `phase`, the number of tokens
    of a triple that precede the segment's first token in the stream (review N2). The step map F(S, x) puts x right after the state, so `phase` is `Streams.phase`."""

    def __init__(self, params: G.GenParams, seed: int, n_streams: int, n_writes: int = 256):
        import torch
        content = ar_contents(params, seed, n_streams, n_writes)
        self.length = min(len(c) for c in content)
        self.tokens = torch.from_numpy(np.stack([c[:self.length] for c in content])).cuda()

    def sample(self, shape: tuple, C_: int, gen, phase: int = 0):
        import torch
        n_streams = self.tokens.shape[0]
        flat = int(np.prod(shape))
        s = torch.randint(0, n_streams, (flat,), generator=gen, device=self.tokens.device)
        o = 3 * torch.randint(0, (self.length - C_ - phase) // 3 + 1, (flat,), generator=gen, device=self.tokens.device) + phase
        return self.tokens[s[:, None], o[:, None] + torch.arange(C_, device=self.tokens.device)].reshape(*shape, C_)


def step_map(model, S, ids):
    """F(S, x) for states S [n, layers, W - 1, d] and segments ids [n, n_x, C]: the window after each segment, [n, n_x, layers, W - 1, d]. S may need gradients."""
    import torch
    n, n_x, C_ = ids.shape
    keep = S.shape[2]
    Sx = S[:, None].expand(n, n_x, *S.shape[1:]).reshape(n * n_x, *S.shape[1:])
    cache = cache_from_state(model, Sx, N_SEGMENTS_BEFORE * C_)
    h = model.base.model.embeddings(ids.reshape(n * n_x, C_))
    _, h2s = model.run_segment(h, cache)
    new = torch.stack([torch.cat([Sx[:, i, C_:], h2s[i]], 1) for i in range(S.shape[1])], 1)
    assert new.shape[2] == keep
    return new.reshape(n, n_x, *S.shape[1:]), Sx.reshape(n, n_x, *S.shape[1:])


def objective(model, S, S0, ids, per_slot: bool = False):
    """E_x sum_layers ||F(S, x) - S||^2 / ||S0||^2 per state [n] (and per layer and slot [layers, W - 1] normalised by the mean slot energy of S0, averaged over states and segments)."""
    import torch
    new, Sx = step_map(model, S, ids)
    err = (new - Sx).float().pow(2).sum(-1)  # [n, n_x, layers, W - 1]
    norm0 = S0.pow(2).sum((2, 3))  # [n, layers]
    per_state = (err.sum(-1).mean(1) / norm0).sum(-1)
    if not per_slot:
        return per_state
    slot = err.mean((0, 1)) / (norm0.mean(0)[:, None] / S0.shape[2])
    return per_state, slot


def find_attractor(model, S0, pool: SegmentPool, steps: int, lr_scale: float, n_x: int, seed: int, eval_ids, log_every: int = 100, at_steps=(), callback=None, sphere: bool = False, phase: int = 0):
    """Adam from S0 on the objective with fresh segments every step (cosine decay of the learning rate to 10%). The learning rate is `lr_scale` times the RMS of S0. Returns S*, the trace
    (step, mean training objective of the last interval) and the held-out objective of S* and S0. `callback(step, S)` is called at the steps in `at_steps` (and at 0). With `sphere` every
    layer's state keeps the norm of S0 (rescaled after every step): the objective as specified lets the state shrink, because the error is divided by the energy of S0 (see `run`). `phase`: where in a triple the segments x start (`Streams.phase`)."""
    import torch
    gen = torch.Generator(device="cuda").manual_seed(seed)
    S = S0.clone().requires_grad_(True)
    norm0 = S0.pow(2).sum((2, 3), keepdim=True).sqrt()
    lr = lr_scale * float(S0.pow(2).mean().sqrt())
    opt = torch.optim.Adam([S], lr=lr)
    trace, acc = [], []
    if callback is not None:
        callback(0, S.detach())
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (step - 1) / max(1, steps - 1))))
        ids = pool.sample((S.shape[0], n_x), model.segment, gen, phase)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = objective(model, S, S0, ids)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        if sphere:
            with torch.no_grad():
                S.mul_(norm0 / S.pow(2).sum((2, 3), keepdim=True).sqrt())
        acc.append(float(loss.detach().mean()))
        if step % log_every == 0 or step == steps:
            trace.append((step, float(np.mean(acc))))
            acc = []
        if callback is not None and step in at_steps:
            callback(step, S.detach())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        star, slot_star = objective(model, S.detach(), S0, eval_ids, per_slot=True)
        base, slot_base = objective(model, S0, S0, eval_ids, per_slot=True)
    return S.detach(), trace, {"star": star, "start": base, "slot_star": slot_star, "slot_start": slot_base}


# ---- read-out ----------------------------------------------------------------------------------------------------------------------------------------------------------
def tail_tokens(ar, passkeys, K: int, r: int, C_: int, with_passkey: bool, phase: int = 0):
    """The tokens that follow K random segments: r' tokens of AR content (r' <= r, chosen so that the AR content before PKQ ends on a triple of the dataset's grid: `phase` tokens of AR
    content precede the state, so phase + K * C + r' is a multiple of 3), PKQ and, teacher forced, the passkey. [n, T]."""
    import torch
    n = passkeys.shape[0]
    r2 = r - (phase + K * C_ + r) % 3
    parts = [ar[:, K * C_:K * C_ + r2], torch.full((n, 1), G.PKQ, device=ar.device)]
    if with_passkey:
        parts.append(passkeys)
    return torch.cat(parts, 1).long(), r2


def read_out(model, S, ar, passkeys, Ks, r: int, greedy: bool = False, return_tokens: bool = False, phase: int = 0):
    """Start from the window state S [n, layers, W - 1, d], feed random AR content `ar` [n, >= max(K) * C + C] segment by segment and, after each K in Ks, branch: feed r more AR tokens, PKQ
    and read the passkey out. Teacher forced (the inputs hold the true passkey, `passkeys` [n, P]; exact match under teacher forcing equals greedy free-running generation) or `greedy`
    (the model's own tokens, re-running the tail segments for every token; needed when there is no true passkey). `phase` = `Streams.phase` (where the stream's triple grid is at the state).
    Returns {K: predicted tokens [n, P]}."""
    import torch
    C_, P = model.segment, passkeys.shape[1]
    out, done = {}, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cache = cache_from_state(model, S, N_SEGMENTS_BEFORE * C_)  # (under the autocast, like every forward: flash attention needs bf16 keys and values)
        for K in sorted(Ks):
            while done < K:
                for _ in advance(model, cache, ar[:, done * C_:(done + 1) * C_]):
                    pass
                done += 1
            if not greedy:
                tail, r2 = tail_tokens(ar, passkeys, K, r, C_, True, phase)
                ids = pad_to_segment(tail, C_)
                branch = clone_cache(cache)
                hs = torch.cat([h for _, h, _ in advance(model, branch, ids)], 1)
                pos = r2 + torch.arange(P, device=ids.device)  # the logits at PKQ and at the first P - 1 passkey tokens predict the passkey
                out[K] = logits_of(model, hs[:, pos]).argmax(-1)
            else:
                gen = torch.zeros(S.shape[0], 0, dtype=torch.long, device=S.device)
                for j in range(P):
                    tail, r2 = tail_tokens(ar, gen, K, r, C_, True, phase)
                    branch = clone_cache(cache)
                    hs = torch.cat([h for _, h, _ in advance(model, branch, pad_to_segment(tail, C_))], 1)
                    nxt = logits_of(model, hs[:, r2 + j]).argmax(-1)
                    gen = torch.cat([gen, nxt[:, None]], 1)
                out[K] = gen
    return out


def exact_and_positions(pred, truth):
    """(exact match [n] bool, per-position correctness [n, P] bool)."""
    ok = pred == truth
    return ok.all(1), ok


# ---- the whole experiment for one checkpoint --------------------------------------------------------------------------------------------------------------------
def git_state() -> dict:
    def git(*a):
        r = subprocess.run(["git", *a], cwd=C.ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def rel_dist(a, b):
    """||a - b|| / sqrt((||a||^2 + ||b||^2) / 2) over the last three dims (per state)."""
    num = (a - b).pow(2).sum((-3, -2, -1)).sqrt()
    den = ((a.pow(2).sum((-3, -2, -1)) + b.pow(2).sum((-3, -2, -1))) / 2).sqrt()
    return num / den


def run(name: str, steps: int = 3000, lr_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0), n_x: int = 8, n_pass: int = 16, n_ctx: int = 4, repeats: int = 3, track: int = 64, lr: float | None = None,
        pilot_steps: int = 300, log=print) -> dict:
    import torch
    t0 = time.time()
    model, cfg, weights, sha = load_model(name)
    for p in model.parameters():
        p.requires_grad_(False)
    P = int(cfg.get("passkey_len", 4))
    params = G.GenParams(passkey_len=P)
    C_ = model.segment
    st = Streams(params, C_, n_pass, n_ctx)
    ids = torch.from_numpy(st.prefix).cuda().long()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        S0 = window_states(model, ids)
    pool, eval_pool = SegmentPool(params, SEEDS["ar_pool"], 400), SegmentPool(params, SEEDS["ar_eval"], 100)
    eval_ids = eval_pool.sample((len(st), 16), C_, torch.Generator(device="cuda").manual_seed(SEEDS["ar_eval"]), st.phase)
    out = {"checkpoint": name, "weights": str(weights.relative_to(C.ROOT)), "sha256": sha, "passkey_len": P, "segment": C_, "window": model.window, "kind": cfg["model"]["kind"],
           "record": {"script": "src/attractor.py", "git": git_state(), "seeds": SEEDS, "n_pass": n_pass, "n_ctx": n_ctx, "n_states": len(st), "state_position": N_SEGMENTS_BEFORE * C_,
                      "grid": {"first_triple_at": st.first_triple, "phase_at_state": st.phase, "rest_tokens": int(st.rest.shape[1]), "segment_offsets_multiple_of_3_plus": st.phase}, "Ks": list(KS),
                      "noise": list(NOISE), "alphas": list(ALPHAS), "steps": steps, "n_x": n_x, "lr_grid": list(lr_grid), "pilot_steps": pilot_steps, "betas": [0.9, 0.999], "schedule": "cosine to 10%",
                      "repeats": repeats, "track_segments": track, "autocast": "bfloat16"}}
    # (1) the learning rate: the grid is run on a subset of the states for a short time and the lowest held-out objective wins (the decoding is not looked at)
    sub = torch.arange(0, len(st), max(1, len(st) // 24))
    if lr is None:
        grid = {}
        for lr_s in lr_grid:
            _, tr, ev = find_attractor(model, S0[sub], pool, pilot_steps, lr_s, n_x, SEEDS["optimizer"], eval_ids[sub], phase=st.phase)
            grid[lr_s] = float(ev["star"].mean())
            log(f"[{name}] lr {lr_s:g} x RMS: held-out objective {grid[lr_s]:.4f}")
        lr = min(grid, key=grid.get)
        out["record"]["lr_pilot"] = {str(k): v for k, v in grid.items()}
    out["record"]["lr_scale"] = lr
    A = torch.from_numpy(np.where(st.kind == "A")[0]).cuda()
    n_writes = int((max(KS) * C_ + 4 * C_) / 3.2) + 64
    dec = ar_contents(params, SEEDS["ar_decode"], 12, n_writes)
    need = max(KS) * C_ + 2 * C_
    ar = torch.from_numpy(continuation(st, dec, need)).cuda().long()  # what follows each state: the rest of the cut triple, then held-out AR content, a different stretch per state
    truth = torch.from_numpy(st.passkeys).cuda().long()
    r_values = [3 + int(x) for x in np.random.default_rng(SEEDS["repeat"]).choice(np.arange(0, C_ - P - 4), size=repeats, replace=False)]  # filler lengths so that PKQ + passkey fit one segment: random phases
    out["record"]["filler_r"] = r_values
    path_steps = sorted({s_ for s_ in (5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560) if s_ < steps} | {steps})
    path = []

    def on_path(step, S):  # the objective and the passkey read-out (after 4 random segments, teacher forced) along the optimisation: where is the passkey lost?
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            obj = float(objective(model, S, S0, eval_ids).mean())
        pred = read_out(model, S[A], ar[A], truth[A], (4,), r_values[0], phase=st.phase)
        path.append({"step": step, "objective": obj, "exact_k4": float(exact_and_positions(pred[4], truth[A])[0].float().mean()),
                     "moved": float(rel_dist(S[A], S0[A]).mean())})

    S_star, trace, ev = find_attractor(model, S0, pool, steps, lr, n_x, SEEDS["optimizer"], eval_ids, at_steps=path_steps, callback=on_path, phase=st.phase)
    log(f"[{name}] optimized in {time.time() - t0:.0f}s; objective {float(ev['start'].mean()):.4f} at S0 -> {float(ev['star'].mean()):.4f} at S*")
    L = S0.shape[1]
    kept = slice(0, S0.shape[2] - C_)  # slots 0 .. W - 2 - C are the kept, shifted entries of the previous window; the last C slots are new
    new = slice(S0.shape[2] - C_, S0.shape[2])
    out["optimization"] = {
        "trace": trace, "path": path, "objective_start": float(ev["start"].mean()), "objective_star": float(ev["star"].mean()), "objective_star_per_state_max": float(ev["star"].max()),
        "error_by_layer": {"start": [float(ev["slot_start"][i].mean()) for i in range(L)], "star": [float(ev["slot_star"][i].mean()) for i in range(L)]},
        "error_by_slot": {"start": ev["slot_start"].cpu().tolist(), "star": ev["slot_star"].cpu().tolist()},
        "error_kept_vs_new": {"start": {"kept": [float(ev["slot_start"][i, kept].mean()) for i in range(L)], "new": [float(ev["slot_start"][i, new].mean()) for i in range(L)]},
                              "star": {"kept": [float(ev["slot_star"][i, kept].mean()) for i in range(L)], "new": [float(ev["slot_star"][i, new].mean()) for i in range(L)]}},
        "moved_from_s0": {"per_layer": [float(((S_star[:, i] - S0[:, i]).pow(2).sum((1, 2)).sqrt() / S0[:, i].pow(2).sum((1, 2)).sqrt()).mean()) for i in range(L)]},
        "rms": {"s0": [float(S0[:, i].pow(2).mean().sqrt()) for i in range(L)], "star": [float(S_star[:, i].pow(2).mean().sqrt()) for i in range(L)]},
    }
    def decode(S, which=None):
        """Mean over the repeats of exact match and per-position accuracy for every K, over the states `which` (default: the A states)."""
        which = A if which is None else which
        res = {K: [[], []] for K in KS}
        for r in r_values:
            pred = read_out(model, S[which], ar[which], truth[which], KS, r, phase=st.phase)
            for K in KS:
                ex, pos = exact_and_positions(pred[K], truth[which])
                res[K][0].append(ex.float().mean().item())
                res[K][1].append(pos.float().mean(0).cpu().tolist())
        return {str(K): {"exact": float(np.mean(v[0])), "per_position": np.mean(v[1], 0).tolist()} for K, v in res.items()}

    S_sph, trace_sph, ev_sph = find_attractor(model, S0, pool, steps, lr, n_x, SEEDS["optimizer"], eval_ids, sphere=True, phase=st.phase)  # extension: the same objective on the sphere of ||S0|| (no shrinking)
    out["sphere"] = {"trace": trace_sph, "objective_star": float(ev_sph["star"].mean()),
                     "moved_from_s0": [float(((S_sph[:, i] - S0[:, i]).pow(2).sum((1, 2)).sqrt() / S0[:, i].pow(2).sum((1, 2)).sqrt()).mean()) for i in range(L)],
                     "error_kept_vs_new": {"kept": [float(ev_sph["slot_star"][i, kept].mean()) for i in range(L)], "new": [float(ev_sph["slot_star"][i, new].mean()) for i in range(L)]},
                     "convergence": convergence(st, S0, S_sph)}
    out["decoding"] = {"s0": decode(S0), "star": decode(S_star), "sphere": decode(S_sph)}
    log(f"[{name}] exact after K = {KS} random segments: S0 " + " ".join(f"{out['decoding']['s0'][str(K)]['exact']:.2f}" for K in KS) + " | S* " + " ".join(f"{out['decoding']['star'][str(K)]['exact']:.2f}" for K in KS) + " | sphere " + " ".join(f"{out['decoding']['sphere'][str(K)]['exact']:.2f}" for K in KS))
    out["stability"] = stability(model, S0, S_star, A, pool, ar, truth, r_values[0], track, st.phase)
    out["basins"] = basins(model, st, S0, S_star, ar, r_values)
    out["passkey_difference"] = difference_dynamics(model, st, S0, S_star, ar, track)
    out["convergence"] = convergence(st, S0, S_star)
    out["trajectory_convergence"] = trajectory_convergence(model, st, S0, ar, track)
    out["seconds"] = round(time.time() - t0)
    states = C.ROOT / "runs" / "attractor"  # the states themselves stay local (git-ignored, like every weight file)
    states.mkdir(parents=True, exist_ok=True)
    torch.save({"S0": S0.cpu(), "S_star": S_star.cpu(), "S_sphere": S_sph.cpu(), "kind": st.kind, "k": st.k, "c": st.c, "passkeys": st.passkeys}, states / f"{name}.pt")
    return out


def stability(model, S0, S_star, A, pool, ar, truth, r: int, track: int, phase: int = 0) -> dict:
    """Noise of std sigma x the RMS of the state (per layer) added to S* and to S0: how the difference to the clean trajectory under the same random input develops over `track` segments
    (relative to its start), how far the clean trajectory wanders from its start, and the passkey read out after K segments from the noisy state. All noise levels run in one batch."""
    import torch
    C_ = model.segment
    res = {}
    gen = torch.Generator(device="cuda").manual_seed(SEEDS["noise"])
    n, ns = len(A), len(NOISE)
    seg = ar[A][:, :track * C_].repeat(ns, 1)
    for tag, S in (("star", S_star[A]), ("s0", S0[A])):
        rms = S.pow(2).mean((2, 3), keepdim=True).sqrt()
        sig = torch.tensor(NOISE, device=S.device).repeat_interleave(n)[:, None, None, None]
        Sc = S.repeat(ns, 1, 1, 1)
        Sn = Sc + sig * rms.repeat(ns, 1, 1, 1) * torch.randn(Sc.shape, generator=gen, device=S.device)
        d0 = (Sn - Sc).pow(2).sum((1, 2, 3)).sqrt()
        dist, wander = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ca, cb = cache_from_state(model, Sc, N_SEGMENTS_BEFORE * C_), cache_from_state(model, Sn, N_SEGMENTS_BEFORE * C_)
            cur_a, cur_b = Sc, Sn
            for (s, _, h2a), (_, _, h2b) in zip(advance(model, ca, seg), advance(model, cb, seg)):
                cur_a, cur_b = next_window(cur_a, h2a, C_), next_window(cur_b, h2b, C_)
                dist.append((cur_b - cur_a).pow(2).sum((1, 2, 3)).sqrt() / d0)
                wander.append(rel_dist(cur_a, Sc))
        dist, wander = torch.stack(dist).view(track, ns, n).mean(2), torch.stack(wander).view(track, ns, n).mean(2)
        pred = read_out(model, Sn, ar[A].repeat(ns, 1), truth[A].repeat(ns, 1), (4, 32, 512), r, phase=phase)
        res[tag] = {}
        for j, sigma in enumerate(NOISE):
            res[tag][str(sigma)] = {"distance_ratio": dist[:, j].cpu().tolist(), "wander": wander[:, j].cpu().tolist(),
                                    "exact": {str(K): float(exact_and_positions(pred[K][j * n:(j + 1) * n], truth[A])[0].float().mean()) for K in pred}}
    return res


def basins(model, st: Streams, S0, S_star, ar, r_values) -> dict:
    """S_alpha = (1 - alpha) S(A) + alpha S(B) for the states of two passkeys that share their context (B differs from A in one position, or in all), then random input and the model's own
    greedy read-out: the share of the positions where A and B differ that are equal to A's token, to B's token, to neither, and the exact matches of each, as functions of alpha."""
    import torch
    n, na = st.n_pass, len(ALPHAS)
    ia, ib1, iball = [torch.from_numpy(st.index(k, np.arange(n), 0)).cuda() for k in ("A", "B1", "Ball")]
    truth = torch.from_numpy(st.passkeys).cuda().long()
    alpha = torch.tensor(ALPHAS, device="cuda")[:, None, None, None, None]
    res = {}
    for tag, S in (("star", S_star), ("s0", S0)):
        res[tag] = {}
        for kind, ib in (("one_position", ib1), ("all_positions", iball)):
            Sa = ((1 - alpha) * S[ia][None] + alpha * S[ib][None]).reshape(na * n, *S.shape[1:])  # all alphas in one batch
            arr, ta, tb = ar[ia].repeat(na, 1), truth[ia].repeat(na, 1), truth[ib].repeat(na, 1)
            rows = {K: {"match_a": [], "match_b": [], "exact_a": [], "exact_b": [], "other": []} for K in (32, 512)}
            for r in r_values[:2]:
                pred = read_out(model, Sa, arr, ta, (32, 512), r, greedy=True, phase=st.phase)
                for K, p in pred.items():
                    diff = (ta != tb).view(na, n, -1)
                    p, a_, b_ = p.view(na, n, -1), ta.view(na, n, -1), tb.view(na, n, -1)
                    rows[K]["match_a"].append(((p == a_) & diff).sum((1, 2)).float() / diff.sum((1, 2)))
                    rows[K]["match_b"].append(((p == b_) & diff).sum((1, 2)).float() / diff.sum((1, 2)))
                    rows[K]["other"].append(((p != a_) & (p != b_) & diff).sum((1, 2)).float() / diff.sum((1, 2)))
                    rows[K]["exact_a"].append((p == a_).all(2).float().mean(1))
                    rows[K]["exact_b"].append((p == b_).all(2).float().mean(1))
            res[tag][kind] = [{"alpha": ALPHAS[i], **{str(K): {k: float(torch.stack(v)[:, i].mean()) for k, v in rows[K].items()} for K in rows}} for i in range(na)]
    return res


def difference_dynamics(model, st: Streams, S0, S_star, ar, track: int) -> dict:
    """The distance between the states of two passkeys that share their context (B differs from A in one position, or in all), both run on the same random input, relative to the distance at the
    start, over `track` segments: does the part of the state that holds the passkey shrink like noise does (see `stability`) or persist? For S0 and S*."""
    import torch
    C_ = model.segment
    n = st.n_pass
    ia = torch.from_numpy(st.index("A", np.arange(n), 0)).cuda()
    res = {}
    for tag, S in (("s0", S0), ("star", S_star)):
        res[tag] = {}
        for kind in ("B1", "Ball"):
            ib = torch.from_numpy(st.index(kind, np.arange(n))).cuda()
            seg = ar[ia][:, :track * C_]
            d0 = (S[ib] - S[ia]).pow(2).sum((1, 2, 3)).sqrt()
            dist = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ca, cb = cache_from_state(model, S[ia], N_SEGMENTS_BEFORE * C_), cache_from_state(model, S[ib], N_SEGMENTS_BEFORE * C_)
                cur_a, cur_b = S[ia], S[ib]
                for (_, _, h2a), (_, _, h2b) in zip(advance(model, ca, seg), advance(model, cb, seg)):
                    cur_a, cur_b = next_window(cur_a, h2a, C_), next_window(cur_b, h2b, C_)
                    dist.append(float(((cur_b - cur_a).pow(2).sum((1, 2, 3)).sqrt() / d0).mean()))
            res[tag][{"B1": "one_position", "Ball": "all_positions"}[kind]] = {"distance_ratio": dist, "start_distance": float(rel_dist(S[ib], S[ia]).mean())}
    return res


def trajectory_convergence(model, st: Streams, S0, ar, track: int, marks=(1, 2, 4, 8, 16, 32, 64)) -> dict:
    """Do starts with the same passkey and different contexts converge when they are fed the same random input? The mean relative distance between the states of the same passkey
    (different contexts) and between different passkeys, after t = 0 (the start) and the given number of segments. An attractor per passkey would leave the first one at the rounding floor
    and the second one apart."""
    import torch
    C_ = model.segment
    a = np.where(st.kind == "A")[0]
    k, c = st.k[a], st.c[a]
    same, diff = (k[:, None] == k[None]) & (c[:, None] != c[None]), k[:, None] != k[None]
    ia = torch.from_numpy(a).cuda()
    seg = ar[ia[:1]][:, :track * C_].expand(len(a), -1)  # one input stretch for every state

    def dists(S):
        d = rel_dist(S[:, None], S[None]).cpu().numpy()
        return float(d[same].mean()), float(d[diff].mean())

    cur = S0[ia]
    res = {"t": [0], "same_passkey": [dists(cur)[0]], "different_passkey": [dists(cur)[1]]}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cache = cache_from_state(model, cur, N_SEGMENTS_BEFORE * C_)
        for s, _, h2s in advance(model, cache, seg):
            cur = next_window(cur, h2s, C_)
            if s + 1 in marks:
                d_same, d_diff = dists(cur)
                res["t"].append(s + 1), res["same_passkey"].append(d_same), res["different_passkey"].append(d_diff)
    return res


def convergence(st: Streams, S0, S_star) -> dict:
    """Do starts with the same passkey (different contexts) reach the same S*? Mean relative distance between the states of the same passkey and between different passkeys, for S0 and S*
    (and the ratio): the optimisation should make the first one small while keeping the second."""
    import torch
    a = np.where(st.kind == "A")[0]
    k, c = st.k[a], st.c[a]
    res = {}
    for tag, S in (("s0", S0), ("star", S_star)):
        X = S[torch.from_numpy(a).cuda()]
        d = rel_dist(X[:, None], X[None]).cpu().numpy()
        same = (k[:, None] == k[None]) & (c[:, None] != c[None])
        diff = (k[:, None] != k[None])
        res[tag] = {"same_passkey": float(d[same].mean()), "different_passkey": float(d[diff].mean()), "ratio": float(d[same].mean() / d[diff].mean())}
    return res


def fill_missing(name: str, res: dict, track: int = 64) -> dict:
    """Add the analyses a result was written without (from the saved states in runs/attractor/<name>.pt, no optimisation): `passkey_difference`, `trajectory_convergence`."""
    import torch
    model, cfg, _, _ = load_model(name)
    for p in model.parameters():
        p.requires_grad_(False)
    saved = torch.load(C.ROOT / "runs" / "attractor" / f"{name}.pt", weights_only=False)
    P = int(cfg.get("passkey_len", 4))
    params = G.GenParams(passkey_len=P)
    st = Streams(params, model.segment, res["record"]["n_pass"], res["record"]["n_ctx"])
    assert np.array_equal(st.passkeys, saved["passkeys"]), "the states were made from other passkeys"
    C_ = model.segment
    n_writes = int((max(KS) * C_ + 4 * C_) / 3.2) + 64
    dec = ar_contents(params, SEEDS["ar_decode"], 12, n_writes)
    need = max(KS) * C_ + 2 * C_
    ar = torch.from_numpy(continuation(st, dec, need)).cuda().long()
    if "passkey_difference" not in res:
        res["passkey_difference"] = difference_dynamics(model, st, saved["S0"].cuda(), saved["S_star"].cuda(), ar, track)
    if "trajectory_convergence" not in res:
        res["trajectory_convergence"] = trajectory_convergence(model, st, saved["S0"].cuda(), ar, track)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "fill"], help="run: the whole analysis of the given runs; fill: add the analyses a result file was written without, from the saved states")
    ap.add_argument("--run", nargs="+", required=True, help="finished runs (results/raw/<name>.json) with weights in checkpoints/ or runs/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=None, help="learning rate in units of the RMS of the state (default: picked from a grid by the held-out objective)")
    ap.add_argument("--n-pass", type=int, default=16)
    ap.add_argument("--n-ctx", type=int, default=4)
    a = ap.parse_args()
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for name in a.run:
        res[name] = run(name, steps=a.steps, lr=a.lr, n_pass=a.n_pass, n_ctx=a.n_ctx) if a.cmd == "run" else fill_missing(name, res[name])
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
