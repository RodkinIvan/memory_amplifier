#!/usr/bin/env python
"""Deterministic generator for `irodkin/passkey_swar`
(Passkey + Sliding-Window Associative Recall).

Sequence layout
    BOS PK p1 p2 p3 p4  <stream>  PKQ p1 p2 p3 p4 EOS
    stream = N associative-recall writes `W k v`, interleaved with queries `Q k a`.

Answer rule for a query on key k, with age = (#writes so far) - (index of k's most recent write)
(age 1 == the immediately preceding write; overwrites therefore count):
    age <= w              -> valid,   a = value of that most recent write
    age >  2w or unwritten -> expired, a = NULL
    w < age <= 2w         -> never queried (gray zone)

Labels use the HF convention: `labels[t]` is the target *for the token at position t* (the shift
happens inside the loss); everything is -100 except the answer token of each `Q k a` and the four
passkey tokens after PKQ. `answer_type[j]` describes the j-th supervised position, in order.

Every example depends only on its own integer `seed` (column `seed`), so the output is independent of
the number of workers.  Per-example seed = split_seed * 10**12 + N * 10**6 + example_index.

Usage
    python data/make_passkey_swar.py --out-dir data_cache/passkey_swar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IGNORE = -100
SPECIALS = ["PAD", "BOS", "EOS", "PK", "PKQ", "W", "Q", "NULL"]
PAD, BOS, EOS, PK, PKQ, W, Q, NULL = range(len(SPECIALS))
KEY0 = len(SPECIALS)  # first key id; values follow the keys
TYPE_NAMES = ["passkey", "ar_valid", "ar_expired"]
PASSKEY, AR_VALID, AR_EXPIRED = range(3)
PASSKEY_LEN = 4  # the default passkey length; GenParams.passkey_len overrides it (capacity experiments use 8, 16, 32, ...)

ALL_NS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TRAIN_NS = [32, 64, 128, 256]


@dataclass(frozen=True)
class GenParams:
    w: int = 16
    n_keys: int = 16
    n_values: int = 16
    query_rate: float = 0.25  # expected queries per write
    valid_frac: float = 0.5  # target fraction of valid queries
    drift_rate: float = 0.10  # per-write probability of swapping one active key for an inactive one
    active_min: int = 6
    active_max: int = 8
    passkey_len: int = PASSKEY_LEN  # tokens of the passkey, each one of n_values values (4 bits at 16 values)

    @property
    def val0(self) -> int:
        return KEY0 + self.n_keys

    @property
    def vocab_size(self) -> int:
        return KEY0 + self.n_keys + self.n_values


def params_dict(p: GenParams) -> dict:
    """The generation parameters stored in gen_meta.json. `passkey_len` is written only when it is not the default, so the metadata (and the checksums
    that cover it) of a default dataset stay exactly what they were before the option existed."""
    d = asdict(p)
    if d["passkey_len"] == PASSKEY_LEN:
        del d["passkey_len"]
    return d


def vocab_dict(p: GenParams) -> dict:
    tokens = SPECIALS + [f"k{i}" for i in range(p.n_keys)] + [f"v{i}" for i in range(p.n_values)]
    return {
        "tokens": tokens,
        "token_to_id": {t: i for i, t in enumerate(tokens)},
        "size": len(tokens),
        "special": {t: i for i, t in enumerate(SPECIALS)},
        "key_ids": [KEY0, KEY0 + p.n_keys - 1],  # inclusive range
        "value_ids": [p.val0, p.val0 + p.n_values - 1],  # inclusive range
        "label_ignore": IGNORE,
        "answer_types": TYPE_NAMES,
    }


def example_seed(split_seed: int, n_writes: int, idx: int) -> int:
    assert 0 <= idx < 10**6 and n_writes < 10**6
    return split_seed * 10**12 + n_writes * 10**6 + idx


def make_example(seed: int, n_writes: int, p: GenParams):
    """Return (input_ids, labels, answer_type_codes) as python lists. Pure function of (seed, n_writes, p)."""
    rnd = random.Random(seed).random  # random() is stable across Python versions
    K, V = p.n_keys, p.n_values
    val0, w, w2 = p.val0, p.w, 2 * p.w
    p_more = p.query_rate / (1.0 + p.query_rate)

    m = p.active_min + int(rnd() * (p.active_max - p.active_min + 1))
    perm = list(range(K))
    for i in range(K - 1, 0, -1):  # Fisher-Yates with our own draws
        j = int(rnd() * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    active, inactive = perm[:m], perm[m:]

    passkey = [val0 + int(rnd() * V) for _ in range(p.passkey_len)]
    toks = [BOS, PK, *passkey]
    labs = [IGNORE] * len(toks)
    types: list[int] = []
    last = [-1] * K  # index (among writes) of each key's most recent write
    val = [0] * K
    c = 0  # writes emitted so far
    for _ in range(n_writes):
        if inactive and rnd() < p.drift_rate:  # drift: swap one active key for an inactive one
            i, j = int(rnd() * m), int(rnd() * len(inactive))
            active[i], inactive[j] = inactive[j], active[i]
        k = active[int(rnd() * m)]
        v = val0 + int(rnd() * V)
        toks += (W, KEY0 + k, v)
        labs += (IGNORE, IGNORE, IGNORE)
        last[k], val[k] = c, v
        c += 1
        while rnd() < p_more:  # queries at random positions: geometric number after each write
            valid = [x for x in range(K) if last[x] >= 0 and c - last[x] <= w]
            expired = [x for x in range(K) if last[x] < 0 or c - last[x] > w2]
            want_valid = rnd() < p.valid_frac
            if (want_valid and valid) or not expired:
                pool, t = valid, AR_VALID
            else:
                pool, t = expired, AR_EXPIRED
            k = pool[int(rnd() * len(pool))]
            a = val[k] if t == AR_VALID else NULL
            toks += (Q, KEY0 + k, a)
            labs += (IGNORE, IGNORE, a)
            types.append(t)
    toks += [PKQ, *passkey, EOS]
    labs += [IGNORE, *passkey, IGNORE]
    types += [PASSKEY] * p.passkey_len
    return toks, labs, types


def _gen_chunk(job):
    split_seed, n_writes, start, stop, p = job
    ids, labs, types, seeds, lens, tlens = [], [], [], [], [], []
    for i in range(start, stop):
        s = example_seed(split_seed, n_writes, i)
        t, l, ty = make_example(s, n_writes, p)
        ids.append(np.asarray(t, np.int16))
        labs.append(np.asarray(l, np.int16))
        types.append(np.asarray(ty, np.int8))
        seeds.append(s)
        lens.append(len(t))
        tlens.append(len(ty))
    return (np.concatenate(ids), np.concatenate(labs), np.concatenate(types),
            np.asarray(seeds, np.int64), np.asarray(lens, np.int64), np.asarray(tlens, np.int64))


def _list_array(flat: np.ndarray, lens: np.ndarray, value_type) -> pa.Array:
    offsets = np.zeros(len(lens) + 1, np.int32)
    np.cumsum(lens, out=offsets[1:])
    return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), pa.array(flat, value_type))


def build_table(parts, n_writes: int) -> pa.Table:
    ids, labs, types, seeds, lens, tlens = (np.concatenate([x[i] for x in parts]) for i in range(6))
    type_str = pa.array(np.asarray(TYPE_NAMES, dtype=object)[types], pa.string())
    offs = np.zeros(len(tlens) + 1, np.int32)
    np.cumsum(tlens, out=offs[1:])
    return pa.table({
        "input_ids": _list_array(ids, lens, pa.int16()),
        "labels": _list_array(labs, lens, pa.int16()),
        "answer_type": pa.ListArray.from_arrays(pa.array(offs, pa.int32()), type_str),
        "n_writes": pa.array(np.full(len(lens), n_writes), pa.int32()),
        "seed": pa.array(seeds, pa.int64()),
    })


def generate_config(pool, split_seed: int, n_writes: int, size: int, p: GenParams, path: Path) -> dict:
    approx_len = int(3.75 * n_writes + 12)
    chunk = max(1, min(500, 2_000_000 // approx_len))
    jobs = [(split_seed, n_writes, s, min(s + chunk, size), p) for s in range(0, size, chunk)]
    parts = list(pool.map(_gen_chunk, jobs))  # map preserves order -> deterministic output
    table = build_table(parts, n_writes)
    rg = max(1, min(math.ceil(size / 64), 8_000_000 // approx_len))  # >=64 row groups for parallel verification
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", row_group_size=rg)
    return {"rows": size, "bytes": path.stat().st_size}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_info(repo_root: Path, files: list[str]) -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=repo_root, capture_output=True, text=True)
        return r.stdout.rstrip("\n") if r.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),  # None if the repo has no commit yet
        "dirty_files": [ln[3:] for ln in (run("status", "--porcelain", "--", *files) or "").splitlines()],
        "blob_sha": {f: run("hash-object", f) for f in files},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = GenParams()
    ap.add_argument("--out-dir", type=Path, default=Path("data_cache/passkey_swar"))
    ap.add_argument("--w", type=int, default=d.w, help="AR window in writes; expired iff age > 2w")
    ap.add_argument("--n-keys", type=int, default=d.n_keys)
    ap.add_argument("--n-values", type=int, default=d.n_values, help="values, also used for passkey digits")
    ap.add_argument("--query-rate", type=float, default=d.query_rate, help="expected queries per write")
    ap.add_argument("--valid-frac", type=float, default=d.valid_frac, help="target share of valid queries")
    ap.add_argument("--drift-rate", type=float, default=d.drift_rate, help="per-write prob. of swapping an active key")
    ap.add_argument("--active-min", type=int, default=d.active_min)
    ap.add_argument("--active-max", type=int, default=d.active_max)
    ap.add_argument("--passkey-len", type=int, default=d.passkey_len, help="passkey length in tokens (default 4); the sequence is BOS PK p1..pP stream PKQ p1..pP EOS")
    ap.add_argument("--ns", type=int, nargs="+", default=ALL_NS, help="AR write counts (one HF config n{N} each)")
    ap.add_argument("--train-ns", type=int, nargs="+", default=TRAIN_NS)
    ap.add_argument("--train-size", type=int, default=100_000)
    ap.add_argument("--eval-size", type=int, default=1_000)
    ap.add_argument("--seed-train", type=int, default=1)
    ap.add_argument("--seed-validation", type=int, default=2)
    ap.add_argument("--seed-test", type=int, default=3)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--skip-sanity", action="store_true", help="do not run sanity_checks.py afterwards")
    args = ap.parse_args(argv)

    seeds = {"train": args.seed_train, "validation": args.seed_validation, "test": args.seed_test}
    assert len(set(seeds.values())) == 3, "each split needs its own seed"
    assert set(args.train_ns) <= set(args.ns)
    p = GenParams(args.w, args.n_keys, args.n_values, args.query_rate, args.valid_frac,
                  args.drift_rate, args.active_min, args.active_max, args.passkey_len)
    assert p.passkey_len >= 1
    assert p.active_max <= p.n_keys and 1 <= p.active_min <= p.active_max
    assert p.vocab_size < 2**15  # ids are stored as int16

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    meta = {
        "params": params_dict(p), "seeds": seeds, "ns": args.ns, "train_ns": args.train_ns,
        "sizes": {"train": args.train_size, "validation": args.eval_size, "test": args.eval_size},
        "generator": git_info(repo_root, ["data/make_passkey_swar.py", "data/sanity_checks.py"]),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pyarrow": pa.__version__},
        "files": {},
    }
    (out / "vocab.json").write_text(json.dumps(vocab_dict(p), indent=1))

    plan = [("train", n, args.train_size) for n in args.train_ns]
    plan += [(s, n, args.eval_size) for s in ("validation", "test") for n in args.ns]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for split, n, size in plan:
            path = out / f"n{n}" / f"{split}.parquet"
            info = generate_config(pool, seeds[split], n, size, p, path)
            info["sha256"] = sha256_file(path)
            meta["files"][f"n{n}/{split}.parquet"] = info
            print(f"[{time.time() - t0:7.1f}s] {split:10s} n{n:<6d} rows={size:<7d} {info['bytes'] / 1e6:8.1f} MB", flush=True)
    (out / "gen_meta.json").write_text(json.dumps(meta, indent=1))

    if not args.skip_sanity:
        import sanity_checks  # sibling module (script dir is on sys.path)
        ok = sanity_checks.run(out, repo_root / "reports", workers=args.workers)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
