#!/usr/bin/env python
"""Probes on trained checkpoints, using streams generated on the fly (so any N and any generator parameters).

    # passkey accuracy vs stream length for trained sliding-window models (shows the cliff at the reach)
    python src/probes.py passkey-length --runs swa_w64_L2_s0 swa_w128_L2_s0 --out results/probes/swa_passkey_length.json
    # passkey accuracy vs query density at fixed N (is the leak of a model driven by writes or by tokens?)
    python src/probes.py density --runs gdn_fixed_a1_s0 --out results/probes/gdn_leak_density.json
    # how many of the 4 passkey tokens survive per stream (graded decay or all-or-nothing retention?), on the dataset's test split
    python src/probes.py profile --runs swa_fb_nltanh_seg64_s2 --ns 256 1024 4096 --out results/probes/passkey_profile.json
    # is the loop an attractor? perturb m mid-stream (does the passkey survive, does the perturbation die out over later hops?),
    # Jacobian of m(s) -> m(s+1), and the size and drift of m over a long stream
    python src/probes.py attractor --runs swa_fb_gi0_nlnone_seg8_steps50000_s2 --out results/probes/attractor_probe.json

Probe streams use their own seed (77), distinct from the train/validation/test seeds of the dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import data as D  # noqa: E402
import make_passkey_swar as G  # noqa: E402
import sanity_checks as S  # noqa: E402

PROBE_SEED = 77


def make_packed(p: G.GenParams, n_writes: int, n_examples: int, split_seed: int = PROBE_SEED) -> D.Packed:
    """Generate `n_examples` streams with `n_writes` AR writes and return them as a Packed set with oracle kinds."""
    ids, labs, types, kinds, offs, toffs = [], [], [], [], [0], [0]
    for i in range(n_examples):
        t, l, ty = G.make_example(G.example_seed(split_seed, n_writes, i), n_writes, p)
        _, _, k, _ = S.oracle(np.asarray(t, np.int16), p.w, p.n_keys, p.n_values, p.passkey_len)
        ids += t
        labs += l
        types += ty
        kinds += k.tolist()
        offs.append(len(ids))
        toffs.append(len(types))
    a = np.asarray
    return D.Packed(a(ids, np.int16), a(labs, np.int16), a(offs, np.int64), a(types, np.int8), a(toffs, np.int64),
                    a(kinds, np.int8))


def query_distances(packed: D.Packed, passkey_len: int = G.PASSKEY_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Token distance from the queried key's latest write (its value token) to the query's key token.

    Returns (distances of valid queries, distances of stale-expired queries). Needs packed.kinds. `passkey_len` is the dataset's passkey length (tokens)."""
    valid, stale = [], []
    for i in range(len(packed)):
        t = packed.ids[packed.offs[i]:packed.offs[i + 1]].astype(int).tolist()
        ks = packed.kinds[packed.toffs[i]:packed.toffs[i + 1]][:-passkey_len]  # drop the passkey positions
        last_value_pos, j, q = {}, 2 + passkey_len, 0  # the stream starts after BOS PK and the passkey
        while t[j] != G.PKQ:
            if t[j] == G.W:
                last_value_pos[t[j + 1]] = j + 2
                j += 3
            else:
                kind = ks[q]
                q += 1
                if kind == 1:
                    valid.append(j + 1 - last_value_pos[t[j + 1]])
                elif kind == 2:
                    stale.append(j + 1 - last_value_pos[t[j + 1]])
                j += 3
    return np.asarray(valid), np.asarray(stale)


def reach_ceilings(packed: D.Packed, reaches: list[int], passkey_len: int = G.PASSKEY_LEN) -> dict[int, dict[str, float]]:
    """For a model that can pass information over `R` tokens (and no further): the share of valid answers whose write is
    reachable, and the share of stale writes that are still visible (those the model must recognise as expired by
    counting, since a pure window cannot tell they are older than 2w writes)."""
    valid, stale = query_distances(packed, passkey_len)
    return {int(r): {"valid_reachable": float(np.mean(valid <= r)), "stale_visible": float(np.mean(stale <= r))}
            for r in reaches}


def load_run(name: str, device="cuda"):
    """(model, config) of a finished run: results/raw/<name>.json + runs/<name>/model.pt."""
    import torch
    from models import build_from_config
    cfg = json.loads((C.ROOT / "results" / "raw" / f"{name}.json").read_text())["config"]
    data_dir = C.data_dir_for(cfg)
    vocab = json.loads((data_dir / "vocab.json").read_text())["size"]
    mc = {**cfg["model"], "kv_cache": True} if cfg["model"]["kind"] == "swa_fb" else cfg["model"]  # the loop model has only the key/value cache path since E7: the same function as the tail path some of these runs were trained with, up to rounding
    model = build_from_config(mc, vocab).to(device)
    model.load_state_dict(torch.load(C.ROOT / "runs" / name / "model.pt", map_location=device))
    return model.eval(), cfg


def token_profile(hits: np.ndarray) -> dict:
    """hits: [examples, P] bool, is each passkey token right (P = the passkey length). Exact match = all P; `hist[k]` = share of examples with exactly k right.
    A memory that decays gradually leaves many examples with 1 to P-1 right tokens; one that retains or loses the whole passkey
    puts almost all examples at 0 or P."""
    return {"exact": float(hits.all(1).mean()), "per_token": [float(x) for x in hits.mean(0)],
            "hist": [float(x) for x in np.bincount(hits.sum(1), minlength=hits.shape[1] + 1) / len(hits)]}


def passkey_token_hits(model, p: D.Packed, device, tokens_per_batch: int = 1 << 20, amp_dtype=None) -> np.ndarray:
    """[examples, P] bool: is each of the P passkey tokens predicted correctly under teacher forcing (same batching as the evaluator)."""
    import torch
    import evaluate as E
    L, B = E.eval_plan(p.lengths(), tokens_per_batch)
    rows = []
    with torch.no_grad():
        for start in range(0, len(p), B):
            idx = np.arange(start, min(start + B, len(p)))
            n_real = len(idx)
            if n_real < B <= len(p):
                idx = np.concatenate([idx, np.full(B - n_real, idx[-1])])
            ids, labs = D.pad_batch(p, idx, L)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(input_ids=torch.from_numpy(ids).to(device)).logits
            tgt = torch.from_numpy(labs[:, 1:]).to(device)
            mask = tgt != D.IGNORE
            correct, mask = ((logits[:, :-1].argmax(-1) == tgt) & mask).cpu().numpy(), mask.cpu().numpy()
            for j in range(n_real):
                kinds = p.kinds[p.toffs[idx[j]]:p.toffs[idx[j] + 1]]
                rows.append(correct[j][mask[j]][kinds == E.PASSKEY])
    return np.array(rows)


def passkey_profile(name: str, ns: list[int], split: str = "test") -> list[dict]:
    """token_profile of a finished run on the dataset's `split` for each N in `ns` (writes)."""
    import torch
    model, cfg = load_run(name)
    amp = torch.bfloat16 if cfg["train"].get("precision", "fp32") == "bf16" else None
    data_dir = C.data_dir_for(cfg)
    out = []
    for n in ns:
        p = D.load_packed(data_dir / f"n{n}" / f"{split}.parquet", kinds=True)
        out.append({"N": n, "split": split, **token_profile(passkey_token_hits(model, p, torch.device("cuda"), amp_dtype=amp))})
    return out


class LoopHook:
    """Probe hook for `FeedbackSWA.probe_hook`: records every loop vector m (detached, per segment and layer) and, if `eps` > 0, adds
    Gaussian noise of `eps` times the vector's own RMS to the vectors of segment `s_star` (all layers, each example)."""

    def __init__(self, s_star: int | None = None, eps: float = 0.0, seed: int = 0, record: bool = False):
        self.s_star, self.eps, self.seed, self.record, self.m, self.gen = s_star, eps, seed, record, {}, None

    def __call__(self, s: int, i: int, m):
        import torch
        if self.eps > 0 and s == self.s_star:
            if self.gen is None:
                self.gen = torch.Generator(device=m.device)
                self.gen.manual_seed(self.seed)
            rms = m.detach().float().pow(2).mean(-1, keepdim=True).sqrt()
            m = m + (self.eps * rms * torch.randn(m.shape, generator=self.gen, device=m.device)).to(m.dtype)
        if self.record:
            self.m[(s, i)] = m.detach().float()
        return m

    def states(self, n_layers: int) -> np.ndarray:
        """[segments, examples, n_layers * d] array of the recorded vectors."""
        import torch
        segs = sorted({s for s, _ in self.m})
        return torch.stack([torch.cat([self.m[(s, i)] for i in range(n_layers)], dim=-1) for s in segs]).cpu().numpy()


def contraction_profile(clean: np.ndarray, perturbed: np.ndarray, s_star: int, offsets=(1, 4, 16, 64, 256)) -> dict:
    """clean, perturbed: [segments, examples, D] loop states of the same streams with and without a perturbation at segment `s_star`.
    Returns {offset: median over examples of ||difference at s_star + offset|| / ||difference at s_star||}. A ratio far below 1 means
    the loop pulls the state back (an attractor), around 1 that the perturbation is carried along, above 1 that it grows."""
    d = np.linalg.norm(perturbed - clean, axis=-1)  # [segments, examples]
    base = np.maximum(d[s_star], 1e-12)
    return {int(k): float(np.median(d[s_star + k] / base)) for k in offsets if s_star + k < len(d)}


def drift_profile(clean: np.ndarray, fractions=(0.1, 0.5, 0.9)) -> dict:
    """RMS of the loop state (per coordinate) and the RMS step ||m_s - m_(s-1)|| / sqrt(D) at fractions of the stream (mean over examples)."""
    rms = np.sqrt((clean ** 2).mean(-1)).mean(-1)  # [segments]
    step = np.sqrt(((clean[1:] - clean[:-1]) ** 2).mean(-1)).mean(-1)
    pick = lambda a, f: float(a[min(len(a) - 1, int(f * len(a)))])
    return {"rms": {f: pick(rms, f) for f in fractions}, "step": {f: pick(step, f) for f in fractions}}


def loop_jacobian(model, ids, s_star: int, amp_dtype=None) -> np.ndarray:
    """Jacobian of the map m(s_star) -> m(s_star + 1) (all layers stacked: [L*d, L*d]) for one stream `ids` (1-D tensor): the loop vectors of
    segment s_star are replaced by leaf tensors, and the vectors of the next segment are differentiated with respect to them."""
    import torch
    L, C = len(model.loops), model.segment
    zs, outs = [None] * L, {}

    def hook(s, i, m):
        if s == s_star:
            zs[i] = m.detach().clone().requires_grad_(True)
            return zs[i]
        if s == s_star + 1:
            outs[i] = m
        return m

    model.probe_hook = hook
    try:
        with torch.enable_grad(), torch.autocast(ids.device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            model(input_ids=ids[: (s_star + 2) * C][None])
        y = torch.cat([outs[i][0] for i in range(L)])
        rows = [torch.cat([g.flatten() for g in torch.autograd.grad(y[k], zs, retain_graph=True)]) for k in range(len(y))]
    finally:
        model.probe_hook = None
    return torch.stack(rows).float().cpu().numpy()


def attractor_probe(name: str, n_writes: int = 4096, n_exact: int = 200, n_traj: int = 16, eps=(0.1, 0.3, 1.0, 3.0), frac: float = 0.25,
                    n_jac: int = 4, eps_traj: float = 0.3) -> dict:
    """The M2 probe of one finished run: (a) passkey exact match when m is perturbed at `frac` of the stream by `eps` times its RMS;
    (b) how the difference between a perturbed and a clean run evolves over later hops, and the size and drift of m; (c) singular
    values and spectral radius of the Jacobian m(s) -> m(s+1) at three positions of a few streams. Fresh streams of `n_writes` writes."""
    import torch
    import evaluate as E
    model, cfg = load_run(name)
    dev, C_ = torch.device("cuda"), model.segment
    amp = torch.bfloat16 if cfg["train"].get("precision", "fp32") == "bf16" else None
    L = len(model.loops)
    p = make_packed(G.GenParams(), n_writes, n_exact)
    s_star = int(frac * float(p.lengths().mean()) / C_)
    out = {"name": name, "segment": C_, "nonlin": cfg["model"]["nonlin"], "n_writes": n_writes, "s_star": s_star, "exact": {}}
    for e in (0.0, *eps):  # (a)
        model.probe_hook = LoopHook(s_star, e, seed=1)
        out["exact"][f"{e:g}"] = token_profile(passkey_token_hits(model, p, dev, amp_dtype=amp))["exact"]
    model.probe_hook = None
    # (b) trajectories on the first streams, clean and perturbed
    ids, _ = D.pad_batch(p, np.arange(n_traj), 256)
    x = torch.from_numpy(ids).to(dev)
    rec = {}
    for tag, e in (("clean", 0.0), ("perturbed", eps_traj)):
        model.probe_hook = hook = LoopHook(s_star, e, seed=2, record=True)
        with torch.no_grad(), torch.autocast(dev.type, dtype=amp or torch.float32, enabled=amp is not None):
            model(input_ids=x)
        rec[tag] = hook.states(L)
    model.probe_hook = None
    out["contraction"] = contraction_profile(rec["clean"], rec["perturbed"], s_star)
    out["drift"] = drift_profile(rec["clean"])
    # (c) Jacobian at a few positions of a few streams
    n_seg = len(rec["clean"])
    sing, rho = [], []
    for j in range(n_jac):
        n_tok = int(p.offs[j + 1] - p.offs[j])
        ids_j = torch.from_numpy(np.asarray(p.ids[p.offs[j]:p.offs[j + 1]], dtype=np.int64)).to(dev)
        for pos in (2, min(n_seg, n_tok // C_) // 4, min(n_seg, n_tok // C_) // 2):
            J = loop_jacobian(model, ids_j, pos, amp)
            sv = np.linalg.svd(J, compute_uv=False)
            sing.append(sv[:3].tolist())
            rho.append(float(np.abs(np.linalg.eigvals(J)).max()))
    out["jacobian"] = {"sigma_max_mean": float(np.mean([s[0] for s in sing])), "sigma_max": [round(s[0], 3) for s in sing],
                       "spectral_radius_mean": float(np.mean(rho)), "spectral_radius": [round(r, 3) for r in rho], "positions_per_stream": 3, "streams": n_jac}
    return out


def passkey_vs_length(name: str, ns: list[int], n_examples: int = 200) -> list[dict]:
    """Passkey exact match (and AR accuracies) of a finished run on fresh streams of each length in `ns` (writes)."""
    import torch
    import evaluate as E
    model, cfg = load_run(name)
    amp = torch.bfloat16 if cfg["train"].get("precision", "fp32") == "bf16" else None
    out = []
    for n in ns:
        p = make_packed(G.GenParams(), n, n_examples)
        m = E.evaluate_packed(model, p, torch.device("cuda"), amp_dtype=amp)
        out.append({"N": n, "tokens": float(p.lengths().mean()), **{k: m[k] for k in
                    ("passkey_acc", "ar_valid_acc", "ar_expired_acc", "ar_expired_stale_acc")}})
    return out


def passkey_vs_density(name: str, ns: list[int], rates: list[float], n_examples: int = 200) -> list[dict]:
    """Passkey exact match and AR accuracies of a finished run on streams with `rate` queries per write, for each N."""
    import torch
    from dataclasses import replace
    import evaluate as E
    model, cfg = load_run(name)
    amp = torch.bfloat16 if cfg["train"].get("precision", "fp32") == "bf16" else None
    out = []
    for rate in rates:
        for n in ns:
            p = make_packed(replace(G.GenParams(), query_rate=rate), n, n_examples)
            m = E.evaluate_packed(model, p, torch.device("cuda"), amp_dtype=amp)
            out.append({"N": n, "query_rate": rate, "tokens": float(p.lengths().mean()), **{k: m[k] for k in
                        ("passkey_acc", "ar_valid_acc", "ar_expired_acc", "ar_expired_stale_acc")}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("passkey-length")
    pl.add_argument("--runs", nargs="+", required=True)
    pl.add_argument("--ns", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 192, 256])
    pl.add_argument("--examples", type=int, default=200)
    pl.add_argument("--out", type=Path, required=True)
    de = sub.add_parser("density")
    de.add_argument("--runs", nargs="+", required=True)
    de.add_argument("--ns", type=int, nargs="+", default=[1024, 1536, 2048, 3072, 4096, 6144])
    de.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.25, 1.0, 2.0])
    de.add_argument("--examples", type=int, default=200)
    de.add_argument("--out", type=Path, required=True)
    at = sub.add_parser("attractor")
    at.add_argument("--runs", nargs="+", required=True)
    at.add_argument("--n-writes", type=int, default=4096)
    at.add_argument("--examples", type=int, default=200)
    at.add_argument("--out", type=Path, required=True)
    pr = sub.add_parser("profile")
    pr.add_argument("--runs", nargs="+", required=True)
    pr.add_argument("--ns", type=int, nargs="+", default=[256, 1024, 4096])
    pr.add_argument("--split", default="test")
    pr.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for name in a.runs:
        if a.cmd == "attractor":
            res[name] = attractor_probe(name, a.n_writes, a.examples)
            r = res[name]
            print(f"{name}: exact {r['exact']} | contraction {r['contraction']} | sigma_max {r['jacobian']['sigma_max_mean']:.2f} rho {r['jacobian']['spectral_radius_mean']:.2f}", flush=True)
        elif a.cmd == "profile":
            res[name] = passkey_profile(name, a.ns, a.split)
            print(f"{name}: " + " ".join(f"N{r['N']}: exact {r['exact']:.2f} hist {[round(h, 2) for h in r['hist']]}" for r in res[name]), flush=True)
        elif a.cmd == "density":
            res[name] = passkey_vs_density(name, a.ns, a.rates, a.examples)
            print(f"{name}: " + " ".join(f"q{r['query_rate']:g}/N{r['N']}:{r['passkey_acc']:.2f}" for r in res[name]), flush=True)
        else:
            res[name] = passkey_vs_length(name, a.ns, a.examples)
            print(f"{name}: " + " ".join(f"{r['N']}:{r['passkey_acc']:.2f}" for r in res[name]), flush=True)
        a.out.write_text(json.dumps(res, indent=1))  # after every run, so a long probe never loses finished runs


if __name__ == "__main__":
    main()
