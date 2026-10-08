#!/usr/bin/env python
"""Diagnostics of the value-adaptation model (`swa_va`, design experiment E1): where does the passkey live once it has left the window's reach?

    # (1) linear probe: is the passkey decodable from the cached values (and keys, and dW_v) of the last window, at growing distance after it was written?
    python src/probes_va.py cache --runs swa_va_seg32_s2 --control swa_w64_L2_steps50000_cur2000_s0 --out results/probes/va_cache_probe.json
    # (2) size, effective rank and drift of dW_v along a long stream
    python src/probes_va.py delta --runs swa_va_seg32_s2 --out results/probes/va_delta_w.json
    # (3) perturb the cached values / the previous segment's encoding in the middle of the stream: does the passkey survive, does the perturbation die out?
    python src/probes_va.py perturb --runs swa_va_seg32_s2 --out results/probes/va_perturb.json

Streams are generated on the fly with their own seeds (77 for the probe's test set, 78 for its training set), never the dataset's. All probes need a CUDA GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
import config as C  # noqa: E402
import data as D  # noqa: E402
import make_passkey_swar as G  # noqa: E402
import probes as P  # noqa: E402

TRAIN_SEED = 78


# ---- the probe itself: ridge regression to one-hot targets, solved in the dual (a few thousand examples, tens of thousands of features) -------------------
def ridge_probe(x_train, y_train, x_test, y_test, n_classes: int, lambdas=(1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4), holdout: float = 0.2, seed: int = 0) -> dict:
    """Linear read-out of a class label from features. Features are standardised with the training statistics; the ridge strength is picked on a held-out part of
    the training set (never on the test set), then the model is refit on all of it. Returns accuracy on the test set, the chosen lambda and the held-out accuracy.
    x_*: [examples, features] float tensors on the GPU; y_*: [examples] int64 tensors."""
    import torch
    mean, std = x_train.mean(0, keepdim=True), x_train.std(0, keepdim=True) + 1e-6
    xt, xe = (x_train - mean) / std, (x_test - mean) / std
    onehot = torch.nn.functional.one_hot(y_train, n_classes).to(xt.dtype)
    onehot = onehot - onehot.mean(0, keepdim=True)  # centre the targets (no intercept in the dual form)
    n = len(xt)
    perm = torch.from_numpy(np.random.default_rng(seed).permutation(n)).to(xt.device)
    cut = int(n * (1 - holdout))
    tr, ho = perm[:cut], perm[cut:]

    def fit_predict(idx_fit, x_eval, lam):
        xf = xt[idx_fit]
        gram = xf @ xf.T
        alpha = torch.linalg.solve(gram + lam * xf.shape[1] * torch.eye(len(xf), device=xf.device, dtype=xf.dtype), onehot[idx_fit])
        return (x_eval @ xf.T) @ alpha  # scores [examples, classes]

    held = {lam: float((fit_predict(tr, xt[ho], lam).argmax(1) == y_train[ho]).float().mean()) for lam in lambdas}
    best = max(held, key=held.get)
    acc = float((fit_predict(torch.arange(n, device=xt.device), xe, best).argmax(1) == y_test).float().mean())
    return {"accuracy": acc, "lambda": best, "heldout_accuracy": held[best]}


# ---- hooks ---------------------------------------------------------------------------------------------------------------------------------------------------
class CacheHook:
    """`ValueAdaptSWA.cache_hook`: at the segments in `record` (all if None) it keeps a copy of the window's cached keys / values of every layer; at segment `s_star` it
    perturbs the cached values (or keys) of every layer: `noise` adds Gaussian noise of `eps` times the tensor's RMS (per example), `zero` sets them to 0."""

    def __init__(self, record=None, record_all: bool = False, s_star: int | None = None, eps: float = 0.0, mode: str = "noise", which: str = "v", seed: int = 0):
        self.record, self.record_all, self.s_star, self.eps, self.mode, self.which, self.seed = record, record_all, s_star, eps, mode, which, seed
        self.snap, self.gen = {}, None

    def __call__(self, s: int, cache) -> None:
        import torch
        if s == self.s_star and (self.eps > 0 or self.mode == "zero"):
            tensors = cache.v if self.which == "v" else cache.k
            for i, t in enumerate(tensors):
                if self.mode == "zero":
                    tensors[i] = torch.zeros_like(t)
                else:
                    if self.gen is None:
                        self.gen = torch.Generator(device=t.device)
                        self.gen.manual_seed(self.seed)
                    rms = t.detach().float().pow(2).mean((-1, -2), keepdim=True).sqrt()
                    tensors[i] = t + (self.eps * rms * torch.randn(t.shape, generator=self.gen, device=t.device)).to(t.dtype)
        if self.record_all or (self.record is not None and s in self.record):
            self.snap[s] = [(cache.k[i].detach().float(), cache.v[i].detach().float()) for i in range(len(cache.v))]


class StateHook:
    """`ValueAdaptSWA.probe_hook`: records the (keys, values) a segment hands to the next one (detached) when `record`, and at segment `s_star` perturbs them (`noise`:
    Gaussian noise of `eps` times their RMS, `zero`) -- the encoding of the previous segment's outputs, from which dW_v of the next segment is built."""

    def __init__(self, s_star: int | None = None, eps: float = 0.0, mode: str = "noise", seed: int = 0, record: bool = False):
        self.s_star, self.eps, self.mode, self.seed, self.record, self.state, self.gen = s_star, eps, mode, seed, record, {}, None

    def __call__(self, s: int, i: int, state):
        import torch
        if s == self.s_star and (self.eps > 0 or self.mode == "zero"):
            if self.gen is None:
                self.gen = torch.Generator(device=state[0].device)
                self.gen.manual_seed(self.seed)
            new = []
            for t in state:
                if self.mode == "zero":
                    new.append(torch.zeros_like(t))
                else:
                    rms = t.detach().float().pow(2).mean((-1, -2), keepdim=True).sqrt()
                    new.append(t + self.eps * rms * torch.randn(t.shape, generator=self.gen, device=t.device))
            state = tuple(new)
        if self.record:
            self.state[(s, i)] = tuple(t.detach().float() for t in state)
        return state


def delta_w(model, i: int, a, b):
    """dW_v = (gain / C) b^T a for layer `i`: [examples, d_value, d] from the keys a [examples, C, d] and values b [examples, C, d_value]."""
    import torch
    return model.loops[i].gain.detach().float() / a.shape[1] * torch.einsum("bjv,bjd->bvd", b, a)


def matrix_stats(dw) -> dict:
    """Per example statistics of a batch of matrices [examples, m, n]: Frobenius norm, spectral norm, participation-ratio rank (sum s^2)^2 / sum s^4, the largest entry."""
    import torch
    sv = torch.linalg.svdvals(dw.float())
    fro2 = (sv ** 2).sum(-1)
    return {"fro": fro2.sqrt(), "spectral": sv[:, 0], "rank": fro2 ** 2 / ((sv ** 4).sum(-1) + 1e-30), "max_entry": dw.abs().amax((-1, -2))}


def common_mode(a, b) -> dict:
    """Split (1/C) sum_j b_j a_j^T into the outer product of the means (b-bar a-bar^T: the same for every token of the segment, content-free if the means do not depend on the stream)
    and the covariance part (1/C) sum_j (b_j - b-bar)(a_j - a-bar)^T, which is where token-specific information can live. a [examples, C, d], b [examples, C, d_value].
    Returns per example: the share of the squared Frobenius norm in the mean part, the Frobenius norm of the covariance part, and the share of entries of a and b with |value| > 0.95."""
    import torch
    full = torch.einsum("bjv,bjd->bvd", b, a) / a.shape[1]
    mean_part = torch.einsum("bv,bd->bvd", b.mean(1), a.mean(1))
    cov = full - mean_part
    return {"mean_share": mean_part.pow(2).sum((-1, -2)) / full.pow(2).sum((-1, -2)).clamp_min(1e-30), "cov_fro": cov.flatten(1).norm(dim=1),
            "saturated": 0.5 * ((a.abs() > 0.95).float().mean((-1, -2)) + (b.abs() > 0.95).float().mean((-1, -2)))}


def autocast(model_cfg: dict, device):
    import torch
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=model_cfg["train"].get("precision", "fp32") == "bf16")


# ---- (2) dW_v along a long stream -----------------------------------------------------------------------------------------------------------------------------
def delta_profile(name: str, n_writes: int = 4096, n_examples: int = 8, fractions=(0.02, 0.1, 0.5, 0.9, 1.0)) -> dict:
    """Size, effective rank and drift of dW_v per layer along a stream: norms relative to the layer's own W_v, the cosine between the dW_v of consecutive segments and the
    one `lag` segments later, at the given fractions of the stream (mean over examples)."""
    import torch
    model, cfg = P.load_run(name)
    dev = torch.device("cuda")
    p = P.make_packed(G.GenParams(), n_writes, n_examples)
    ids, _ = D.pad_batch(p, np.arange(n_examples), model.segment)
    hook = StateHook(record=True)
    model.probe_hook = hook
    with torch.no_grad(), autocast(cfg, dev):
        model(input_ids=torch.from_numpy(ids).to(dev))
    model.probe_hook = None
    n_seg = max(s for s, _ in hook.state) + 1
    out = {"name": name, "segment": model.segment, "n_writes": n_writes, "segments": n_seg, "layers": []}
    for i in range(len(model.loops)):
        w_v = model.base.model.layers[i].attn.v_proj.weight.detach().float()
        dws = [delta_w(model, i, *hook.state[(s, i)]) for s in range(n_seg)]
        cos = lambda x, y: float((x.flatten(1) * y.flatten(1)).sum(1).div(x.flatten(1).norm(dim=1) * y.flatten(1).norm(dim=1) + 1e-30).mean())  # noqa: E731
        layer = {"gain": float(model.loops[i].gain), "w_v_fro": float(w_v.norm()), "at": {}}
        for f in fractions:
            s = min(n_seg - 1, int(f * (n_seg - 1)))
            st = matrix_stats(dws[s])
            cm = common_mode(*hook.state[(s, i)])
            gain = abs(float(model.loops[i].gain))
            layer["at"][f"{f:g}"] = {"segment": s, **{k: float(v.mean()) for k, v in st.items()}, "fro_over_w_v": float(st["fro"].mean() / w_v.norm()),
                                     "mean_share": float(cm["mean_share"].mean()), "cov_fro_over_w_v": float(gain * cm["cov_fro"].mean() / w_v.norm()), "saturated": float(cm["saturated"].mean()),
                                     "cos_next": cos(dws[s], dws[min(s + 1, n_seg - 1)]), "cos_lag16": cos(dws[s], dws[min(s + 16, n_seg - 1)])}
        out["layers"].append(layer)
    return out


# ---- (1) linear probe of the passkey from the cached window ---------------------------------------------------------------------------------------------------
def passkey_streams(n_writes: int, n: int, seed: int, params: G.GenParams | None = None) -> tuple[list[list[int]], np.ndarray]:
    """`n` token streams of `n_writes` AR writes and their passkeys [n, P] as value indices 0 .. n_values - 1 (tokens 2 .. 1 + P: the stream starts BOS PK p_1 .. p_P)."""
    p = params or G.GenParams()
    toks = [G.make_example(G.example_seed(seed, n_writes, i), n_writes, p)[0] for i in range(n)]
    return toks, np.asarray([t[2:2 + p.passkey_len] for t in toks]) - p.val0


def cut_points(distances: list[int], segment: int, passkey_len: int = G.PASSKEY_LEN) -> dict[int, int]:
    """For every distance d (tokens after the end of the passkey) the number of the segment whose end is the first one at or after it; the probe reads the cache at that
    segment boundary, so the window is full and holds no padding. Returns {d: segment index}."""
    start = 2 + passkey_len
    return {d: -(-(start + d) // segment) - 1 for d in distances}


def collect_window(model, cfg, toks: list[list[int]], cuts: dict[int, int], batch: int = 50) -> dict[int, dict[str, np.ndarray]]:
    """Run the model on `toks` and return, for every cut segment, the features of the cache after it: cached values `v`, cached keys `k` (both layers, flattened) and, for
    a swa_va model, dW_v of the next segment (`dw`: from that segment's state). Streams are cut at the last needed segment."""
    import torch
    dev = torch.device("cuda")
    segs, C_ = sorted(set(cuts.values())), model.segment
    feats: dict[int, dict[str, list]] = {s: {"v": [], "k": [], "dw": []} for s in segs}
    for b0 in range(0, len(toks), batch):
        chunk = toks[b0:b0 + batch]
        ids = torch.tensor([t[:(segs[-1] + 1) * C_] for t in chunk], device=dev)
        ch, sh = CacheHook(record=set(segs)), StateHook(record=hasattr(model, "loops") and hasattr(model, "cache_hook"))
        model.cache_hook = ch
        if sh.record:
            model.probe_hook = sh
        with torch.no_grad(), autocast(cfg, dev):
            model(input_ids=ids)
        model.cache_hook = model.probe_hook = None
        for s in segs:
            feats[s]["k"].append(torch.cat([k.flatten(1) for k, _ in ch.snap[s]], 1).cpu())
            feats[s]["v"].append(torch.cat([v.flatten(1) for _, v in ch.snap[s]], 1).cpu())
            if sh.record and (s, 0) in sh.state:
                feats[s]["dw"].append(torch.cat([delta_w(model, i, *sh.state[(s, i)]).flatten(1) for i in range(len(model.loops))], 1).cpu())
    return {s: {k: torch.cat(v).numpy() for k, v in f.items() if v} for s, f in feats.items()}


def cache_probe(name: str, distances=(32, 96, 160, 256, 512, 1024, 2048), n_train: int = 3000, n_test: int = 500, control: str | None = None) -> dict:
    """Linear read-out of every passkey token from the cached window of a finished run (values, keys, dW_v) at growing distance after the passkey; with `control` (a plain
    sliding-window run) the same read-out of its cache, whose values cannot carry the passkey beyond the reach (L * (W - 1) tokens)."""
    import torch
    dev = torch.device("cuda")
    models = {"model": name, **({"control": control} if control else {})}
    p0 = G.GenParams()
    n_writes = int(max(distances) / 2.5) + 64  # enough pairs for the longest distance (>= 3 tokens per AR unit at query rate 1; checked below)
    toks_tr, y_tr = passkey_streams(n_writes, n_train, TRAIN_SEED)
    toks_te, y_te = passkey_streams(n_writes, n_test, P.PROBE_SEED)
    need = 2 + p0.passkey_len + max(distances) + 64
    short = min(len(t) for t in toks_tr + toks_te)
    if short < need:
        raise ValueError(f"streams of {n_writes} writes are only {short} tokens, the longest cut needs {need}: raise n_writes")
    out = {"distances": list(distances), "n_train": n_train, "n_test": n_test, "chance": 1.0 / p0.n_values, "passkey_len": p0.passkey_len, "models": {}}
    for tag, run in models.items():
        model, cfg = P.load_run(run) if tag == "model" else load_plain_as_va(run)
        cuts = cut_points(list(distances), model.segment, p0.passkey_len)
        tr, te = collect_window(model, cfg, toks_tr, cuts), collect_window(model, cfg, toks_te, cuts)
        res = {"run": run, "segment": model.segment, "reach": 2 * (model.window - 1), "by_distance": {}}
        for d, s in cuts.items():
            res["by_distance"][str(d)] = {"cut_segment": s, "tokens": (s + 1) * model.segment - (2 + p0.passkey_len)}
            for feat in tr[s]:
                xs = torch.from_numpy(tr[s][feat]).to(dev)
                xe = torch.from_numpy(te[s][feat]).to(dev)
                accs = [ridge_probe(xs, torch.from_numpy(y_tr[:, k]).to(dev), xe, torch.from_numpy(y_te[:, k]).to(dev), p0.n_values)["accuracy"] for k in range(p0.passkey_len)]
                res["by_distance"][str(d)][feat] = {"per_token": accs, "mean": float(np.mean(accs))}
            print(f"[{tag}] d={d}: " + " ".join(f"{f} {res['by_distance'][str(d)][f]['mean']:.2f}" for f in tr[s]), flush=True)
        out["models"][tag] = res
    return out


def load_plain_as_va(name: str):
    """A finished plain sliding-window run (`swa`) as a `ValueAdaptSWA` with every gain 0, so its cache can be read with the same code (it computes exactly the plain model)."""
    import torch
    from models import ValueAdaptSWA
    base, cfg = P.load_run(name)
    seg = 32
    m = ValueAdaptSWA(base, window=cfg["model"]["window"], segment=seg).to(torch.device("cuda")).eval()
    return m, cfg


# ---- (3) perturbation ------------------------------------------------------------------------------------------------------------------------------------------
def perturbation_probe(name: str, n_writes: int = 1024, n_exact: int = 200, n_traj: int = 16, frac: float = 0.25, eps=(0.1, 0.3, 1.0, 3.0), eps_traj: float = 0.3) -> dict:
    """Passkey exact match when, at `frac` of the stream (far beyond the reach), the cached values of the window ("values"), the previous segment's encoding ("state", from
    which dW_v of the next segment is built) or both are perturbed by Gaussian noise of eps times their RMS, and when the cached values are zeroed. Plus how a perturbation of
    the cached values decays or grows over later segments (distance between the cached values of a clean and a perturbed run, relative to the distance right after the kick)."""
    import torch
    model, cfg = P.load_run(name)
    dev = torch.device("cuda")
    p = P.make_packed(G.GenParams(), n_writes, n_exact)
    C_ = model.segment
    s_star = int(frac * float(p.lengths().mean()) / C_)
    amp = torch.bfloat16 if cfg["train"].get("precision", "fp32") == "bf16" else None
    out = {"name": name, "segment": C_, "n_writes": n_writes, "s_star": s_star, "s_star_tokens": s_star * C_, "exact": {}}

    def exact(cache_kw=None, state_kw=None) -> float:
        model.cache_hook = CacheHook(s_star=s_star, seed=1, **cache_kw) if cache_kw else None
        model.probe_hook = StateHook(s_star=s_star, seed=2, **state_kw) if state_kw else None
        try:
            return P.token_profile(P.passkey_token_hits(model, p, dev, amp_dtype=amp))["exact"]
        finally:
            model.cache_hook = model.probe_hook = None

    out["exact"]["clean"] = exact()
    for e in eps:
        out["exact"][f"values_{e:g}"] = exact({"eps": e})
        out["exact"][f"state_{e:g}"] = exact(None, {"eps": e})
        out["exact"][f"both_{e:g}"] = exact({"eps": e}, {"eps": e})
    out["exact"]["values_zero"] = exact({"mode": "zero"})
    out["exact"]["state_zero"] = exact(None, {"mode": "zero"})
    out["exact"]["both_zero"] = exact({"mode": "zero"}, {"mode": "zero"})
    # trajectories of the cached values after a kick
    ids, _ = D.pad_batch(p, np.arange(n_traj), C_)
    x = torch.from_numpy(ids).to(dev)
    rec = {}
    for tag, e in (("clean", 0.0), ("perturbed", eps_traj)):
        model.cache_hook = hook = CacheHook(record_all=True, s_star=s_star, eps=e, seed=3)
        with torch.no_grad(), torch.autocast(dev.type, dtype=amp or torch.float32, enabled=amp is not None):
            model(input_ids=x)
        model.cache_hook = None
        full = [s for s in sorted(hook.snap) if hook.snap[s][0][1].shape[1] == model.window - 1]  # the first segments have a part-filled cache: leave them out
        first = full[0]
        rec[tag] = np.stack([torch.cat([v.flatten(1) for _, v in hook.snap[s]], 1).cpu().numpy() for s in full])  # [segments, examples, D]
    if s_star < first:
        raise ValueError(f"the kick at segment {s_star} lands before the cache is full (segment {first})")
    out["contraction"] = P.contraction_profile(rec["clean"], rec["perturbed"], s_star - first)
    out["drift"] = P.drift_profile(rec["clean"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("cache", "delta", "perturb"):
        s = sub.add_parser(cmd)
        s.add_argument("--runs", nargs="+", required=True)
        s.add_argument("--out", type=Path, required=True)
        if cmd == "cache":
            s.add_argument("--control", default=None, help="a plain sliding-window run whose cache is read in the same way")
            s.add_argument("--distances", type=int, nargs="+", default=[32, 96, 160, 256, 512, 1024, 2048])
            s.add_argument("--train", type=int, default=3000)
            s.add_argument("--test", type=int, default=500)
        if cmd in ("delta", "perturb"):
            s.add_argument("--n-writes", type=int, default=4096 if cmd == "delta" else 1024)
    a = ap.parse_args()
    res = {}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for name in a.runs:
        if a.cmd == "cache":
            res[name] = cache_probe(name, tuple(a.distances), a.train, a.test, a.control)
        elif a.cmd == "delta":
            res[name] = delta_profile(name, a.n_writes)
            print(name, json.dumps(res[name]["layers"][0]["at"]["0.5"]), flush=True)
        else:
            res[name] = perturbation_probe(name, a.n_writes)
            print(name, json.dumps(res[name]["exact"]), "| contraction", res[name]["contraction"], flush=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
