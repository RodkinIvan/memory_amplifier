#!/usr/bin/env python
"""Scan the key-drift rate: is the 50/50 valid/expired ratio reachable, and how many expired queries are
*stale* (key written earlier, then not written for > 2w writes) rather than never-written?

Only stale queries test forgetting: an empty state answers NULL for a never-written key for free.

    python data/tune_drift.py --drifts 0.02 0.05 0.1 0.2 --ns 32 64 128 256 1024 --examples 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_passkey_swar as G  # noqa: E402
import sanity_checks as S  # noqa: E402


def scan(drift: float, n: int, examples: int, base: G.GenParams) -> dict:
    p = G.GenParams(**{**base.__dict__, "drift_rate": drift})
    st = S._zero_stats(p.w)
    for i in range(examples):
        t, _, _ = G.make_example(G.example_seed(99, n, i), n, p)
        st = S.merge_stats(st, S.oracle(np.asarray(t, np.int16), p.w, p.n_keys, p.n_values)[3])
    return S.summarize(st, p.w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drifts", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--ns", type=int, nargs="+", default=[32, 64, 128, 256, 1024])
    ap.add_argument("--examples", type=int, default=300)
    a = ap.parse_args()
    base = G.GenParams()
    print("drift |    N | valid  | expired stale-share | stale exp. queries/ex | live keys | gray keys | distinct keys written")
    for d in a.drifts:
        for n in a.ns:
            s = scan(d, n, a.examples, base)
            stale_per_ex = s["counts"]["expired_stale"] / s["examples"]
            print(f"{d:5.2f} | {n:4d} | {s['valid_frac']:.3f} | {s['expired_stale_share']:19.3f} | {stale_per_ex:21.2f} | "
                  f"{s['live_keys_at_query(age<=w)']:9.2f} | {s['gray_keys_at_query(w<age<=2w)']:9.2f} | "
                  f"{s['distinct_keys_written_per_ex']:.2f}", flush=True)
        print()
