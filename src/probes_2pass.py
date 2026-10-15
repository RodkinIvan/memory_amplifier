#!/usr/bin/env python
"""Diagnostics of the two-pass model (`swa_2pass`, design experiment E2): where does the passkey live once it has left the window's reach? (They were written for the value-adaptation
model of E1 as well; that model was removed in E7, see `config.REMOVED_IN`, and the probe results of both stay in results/probes/ as an archive.)

    # (1) linear probe: is the passkey decodable from the cached keys and values of the last window, at growing distance after it was written?
    python src/probes_2pass.py cache --runs swa_2pass_seg32_po1_s5 --control swa_w64_L2_steps50000_cur2000_s0 --out results/probes/p2_cache_probe.json
    # (2) perturb the cached keys and / or values in the middle of the stream: does the passkey survive, does the perturbation die out?
    python src/probes_2pass.py perturb --runs swa_2pass_seg32_po1_s5 --out results/probes/p2_perturb.json
    # (3) statistics of the learned per-channel gains (CPU only)
    python src/probes_2pass.py gains --runs swa_2pass_seg32_po1_s5 --out results/probes/p2_gains.json

Streams are generated on the fly with their own seeds (77 for the probe's test set, 78 for its training set), never the dataset's. The cache and perturb probes need a CUDA GPU.
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
    """`cache_hook` of `TwoPassSWA`: at the segments in `record` (all if `record_all`) it keeps a copy of the window's cached keys / values of every layer; at
    segment `s_star` it perturbs the cached values (`which` = "v"), keys ("k") or both ("kv") of every layer: `noise` adds Gaussian noise of `eps` times the tensor's RMS (per
    example), `zero` sets them to 0."""

    def __init__(self, record=None, record_all: bool = False, s_star: int | None = None, eps: float = 0.0, mode: str = "noise", which: str = "v", seed: int = 0):
        if which not in ("v", "k", "kv"):
            raise ValueError(f"which must be 'v', 'k' or 'kv', got {which!r}")
        self.record, self.record_all, self.s_star, self.eps, self.mode, self.which, self.seed = record, record_all, s_star, eps, mode, which, seed
        self.snap, self.gen = {}, None

    def __call__(self, s: int, cache) -> None:
        import torch
        if s == self.s_star and (self.eps > 0 or self.mode == "zero"):
            for tensors in {"v": [cache.v], "k": [cache.k], "kv": [cache.k, cache.v]}[self.which]:  # the cache's own lists, replaced in place
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


def autocast(model_cfg: dict, device):
    import torch
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=model_cfg["train"].get("precision", "fp32") == "bf16")


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
    """Run the model on `toks` and return, for every cut segment, the features of the cache after it: cached values `v` and cached keys `k` (both layers, flattened).
    Streams are cut at the last needed segment."""
    import torch
    dev = torch.device("cuda")
    segs, C_ = sorted(set(cuts.values())), model.segment
    feats: dict[int, dict[str, list]] = {s: {"v": [], "k": []} for s in segs}
    for b0 in range(0, len(toks), batch):
        chunk = toks[b0:b0 + batch]
        ids = torch.tensor([t[:(segs[-1] + 1) * C_] for t in chunk], device=dev)
        ch = CacheHook(record=set(segs))
        model.cache_hook = ch
        with torch.no_grad(), autocast(cfg, dev):
            model(input_ids=ids)
        model.cache_hook = None
        for s in segs:
            feats[s]["k"].append(torch.cat([k.flatten(1) for k, _ in ch.snap[s]], 1).cpu())
            feats[s]["v"].append(torch.cat([v.flatten(1) for _, v in ch.snap[s]], 1).cpu())
    return {s: {k: torch.cat(v).numpy() for k, v in f.items() if v} for s, f in feats.items()}


def cache_probe(name: str, distances=(32, 96, 160, 256, 512, 1024, 2048), n_train: int = 3000, n_test: int = 500, control: str | None = None) -> dict:
    """Linear read-out of every passkey token from the cached window of a finished run (values, keys) at growing distance after the passkey; with `control` (a plain
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
        model, cfg = P.load_run(run) if tag == "model" else load_plain_as_2pass(run)
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


def load_plain_as_2pass(name: str):
    """A finished plain sliding-window run (`swa`) as a `TwoPassSWA` with every gain 0, so its cache can be read with the same code (it computes exactly the plain model)."""
    import torch
    from models import TwoPassSWA
    base, cfg = P.load_run(name)
    m = TwoPassSWA(base, window=cfg["model"]["window"], segment=32).to(torch.device("cuda")).eval()
    return m, cfg


# ---- (2) perturbation ------------------------------------------------------------------------------------------------------------------------------------------
def perturbation_probe(name: str, n_writes: int = 1024, n_exact: int = 200, n_traj: int = 16, frac: float = 0.25, eps=(0.1, 0.3, 1.0, 3.0), eps_traj: float = 0.3) -> dict:
    """Passkey exact match when, at `frac` of the stream (far beyond the reach), the cache is perturbed by Gaussian noise of eps times its RMS, or zeroed: the cached values ("values"),
    the cached keys ("keys") or both ("both"). Plus how a perturbation of the cache decays or grows over later segments (distance between the cached keys and values of a clean and a
    perturbed run, relative to the distance right after the kick)."""
    import torch
    from models import TwoPassSWA
    model, cfg = P.load_run(name)
    if not isinstance(model, TwoPassSWA):
        raise ValueError(f"{name}: the perturbation probe is for the two-pass model (swa_2pass), not {cfg['model']['kind']}")
    dev = torch.device("cuda")
    p = P.make_packed(G.GenParams(), n_writes, n_exact)
    C_ = model.segment
    s_star = int(frac * float(p.lengths().mean()) / C_)
    amp = torch.bfloat16 if cfg["train"].get("precision", "fp32") == "bf16" else None
    out = {"name": name, "kind": cfg["model"]["kind"], "segment": C_, "n_writes": n_writes, "s_star": s_star, "s_star_tokens": s_star * C_, "exact": {}}

    def exact(cache_kw=None) -> float:
        model.cache_hook = CacheHook(s_star=s_star, seed=1, **cache_kw) if cache_kw else None
        try:
            return P.token_profile(P.passkey_token_hits(model, p, dev, amp_dtype=amp))["exact"]
        finally:
            model.cache_hook = None

    out["exact"]["clean"] = exact()
    for e in eps:
        out["exact"][f"values_{e:g}"] = exact({"eps": e, "which": "v"})
        out["exact"][f"keys_{e:g}"] = exact({"eps": e, "which": "k"})
        out["exact"][f"both_{e:g}"] = exact({"eps": e, "which": "kv"})
    for key, which in (("values", "v"), ("keys", "k"), ("both", "kv")):
        out["exact"][f"{key}_zero"] = exact({"mode": "zero", "which": which})
    # trajectories of the cached entries after a kick
    ids, _ = D.pad_batch(p, np.arange(n_traj), C_)
    x = torch.from_numpy(ids).to(dev)
    rec = {}
    for tag, e in (("clean", 0.0), ("perturbed", eps_traj)):
        model.cache_hook = hook = CacheHook(record_all=True, s_star=s_star, eps=e, which="kv", seed=3)
        with torch.no_grad(), torch.autocast(dev.type, dtype=amp or torch.float32, enabled=amp is not None):
            model(input_ids=x)
        model.cache_hook = None
        full = [s for s in sorted(hook.snap) if hook.snap[s][0][1].shape[1] == model.window - 1]  # the first segments have a part-filled cache: leave them out
        first = full[0]
        rec[tag] = np.stack([torch.cat([t.flatten(1) for k, v in hook.snap[s] for t in (k, v)], 1).cpu().numpy() for s in full])  # [segments, examples, D]
    if s_star < first:
        raise ValueError(f"the kick at segment {s_star} lands before the cache is full (segment {first})")
    out["contraction"] = P.contraction_profile(rec["clean"], rec["perturbed"], s_star - first)
    out["drift"] = P.drift_profile(rec["clean"])
    return out


def gain_stats(name: str) -> dict:
    """Per layer statistics of the learned per-channel gains g of a finished two-pass run (CPU only): mean, mean |g|, largest |g| and the share of negative channels."""
    import torch
    state = torch.load(C.ROOT / "runs" / name / "model.pt", map_location="cpu", weights_only=True)
    layers = sorted(int(k.split(".")[1]) for k in state if k.startswith("loops.") and k.endswith(".gain"))
    out = {"name": name, "layers": []}
    for i in layers:
        g = state[f"loops.{i}.gain"].float()
        out["layers"].append({"mean": float(g.mean()), "abs_mean": float(g.abs().mean()), "abs_max": float(g.abs().max()), "negative": float((g < 0).float().mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("cache", "perturb", "gains"):
        s = sub.add_parser(cmd)
        s.add_argument("--runs", nargs="+", required=True)
        s.add_argument("--out", type=Path, required=True)
        if cmd == "cache":
            s.add_argument("--control", default=None, help="a plain sliding-window run whose cache is read in the same way")
            s.add_argument("--distances", type=int, nargs="+", default=[32, 96, 160, 256, 512, 1024, 2048])
            s.add_argument("--train", type=int, default=3000)
            s.add_argument("--test", type=int, default=500)
        if cmd == "perturb":
            s.add_argument("--n-writes", type=int, default=1024)
    a = ap.parse_args()
    res = {}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for name in a.runs:
        if a.cmd == "gains":
            res[name] = gain_stats(name)
            print(name, " | ".join(f"layer {i}: mean {l['mean']:+.2f} |g| {l['abs_mean']:.2f}" for i, l in enumerate(res[name]["layers"])), flush=True)
        elif a.cmd == "cache":
            res[name] = cache_probe(name, tuple(a.distances), a.train, a.test, a.control)
        else:
            res[name] = perturbation_probe(name, a.n_writes)
            print(name, json.dumps(res[name]["exact"]), "| contraction", res[name]["contraction"], flush=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
