#!/usr/bin/env python
"""Probes on trained checkpoints, using streams generated on the fly (so any N and any generator parameters).

    # passkey accuracy vs stream length for trained sliding-window models (shows the cliff at the reach)
    python src/probes.py passkey-length --runs swa_w64_L2_s0 swa_w128_L2_s0 --out results/probes/swa_passkey_length.json
    # passkey accuracy vs query density at fixed N (is the leak of a model driven by writes or by tokens?)
    python src/probes.py density --runs gdn_fixed_a1_s0 --out results/probes/gdn_leak_density.json

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
        _, _, k, _ = S.oracle(np.asarray(t, np.int16), p.w, p.n_keys, p.n_values)
        ids += t
        labs += l
        types += ty
        kinds += k.tolist()
        offs.append(len(ids))
        toffs.append(len(types))
    a = np.asarray
    return D.Packed(a(ids, np.int16), a(labs, np.int16), a(offs, np.int64), a(types, np.int8), a(toffs, np.int64),
                    a(kinds, np.int8))


def query_distances(packed: D.Packed) -> tuple[np.ndarray, np.ndarray]:
    """Token distance from the queried key's latest write (its value token) to the query's key token.

    Returns (distances of valid queries, distances of stale-expired queries). Needs packed.kinds."""
    valid, stale = [], []
    for i in range(len(packed)):
        t = packed.ids[packed.offs[i]:packed.offs[i + 1]].astype(int).tolist()
        ks = packed.kinds[packed.toffs[i]:packed.toffs[i + 1]][:-G.PASSKEY_LEN]  # drop the passkey positions
        last_value_pos, j, q = {}, 6, 0
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


def reach_ceilings(packed: D.Packed, reaches: list[int]) -> dict[int, dict[str, float]]:
    """For a model that can pass information over `R` tokens (and no further): the share of valid answers whose write is
    reachable, and the share of stale writes that are still visible (those the model must recognise as expired by
    counting, since a pure window cannot tell they are older than 2w writes)."""
    valid, stale = query_distances(packed)
    return {int(r): {"valid_reachable": float(np.mean(valid <= r)), "stale_visible": float(np.mean(stale <= r))}
            for r in reaches}


def load_run(name: str, device="cuda"):
    """(model, config) of a finished run: results/raw/<name>.json + runs/<name>/model.pt."""
    import torch
    from models import build_from_config
    cfg = json.loads((C.ROOT / "results" / "raw" / f"{name}.json").read_text())["config"]
    data_dir = Path(cfg["data_dir"])
    data_dir = data_dir if data_dir.is_absolute() else C.ROOT / data_dir
    vocab = json.loads((data_dir / "vocab.json").read_text())["size"]
    model = build_from_config(cfg["model"], vocab).to(device)
    model.load_state_dict(torch.load(C.ROOT / "runs" / name / "model.pt", map_location=device))
    return model.eval(), cfg


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
    a = ap.parse_args()
    res = {}
    for name in a.runs:
        if a.cmd == "density":
            res[name] = passkey_vs_density(name, a.ns, a.rates, a.examples)
            print(f"{name}: " + " ".join(f"q{r['query_rate']:g}/N{r['N']}:{r['passkey_acc']:.2f}" for r in res[name]), flush=True)
        else:
            res[name] = passkey_vs_length(name, a.ns, a.examples)
            print(f"{name}: " + " ".join(f"{r['N']}:{r['passkey_acc']:.2f}" for r in res[name]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
