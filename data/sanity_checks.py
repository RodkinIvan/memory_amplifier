#!/usr/bin/env python
"""Sanity checks for the passkey_swar parquet files (must pass before upload).

Everything here reads the *stored* parquet files, and the oracle re-derives all labels from the token
stream alone (brute force over the whole write history), independently of the generator's bookkeeping.

Checks
  1. oracle: reproduces 100% of labels and answer_type; gray-zone (w < age <= 2w) queries never occur;
     sequence structure is well formed; `n_writes` column is right.
  2. mutation test: the oracle rejects corrupted copies of examples (so "100%" is meaningful).
  3. determinism: examples regenerated from their stored `seed` are identical to the stored ones.
  4. leakage: no example (full input_ids) is shared between splits; seeds/splits are disjoint.
  5. statistics: valid/expired ratio, query-gap histogram, queried-key age histogram, key-reuse stats.

Usage
    python data/sanity_checks.py --data-dir data_cache/passkey_swar --report-dir reports
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_passkey_swar as G  # noqa: E402  (shared constants + make_example for the determinism check only)

KIND_NAMES = ["passkey", "ar_valid", "ar_expired_stale", "ar_expired_never"]  # finer than answer_type
GAP_BINS = 17  # inter-query gap (writes between consecutive queries): 0..15, >=16
N_EXP_BINS = 10  # expired age bins: (2w, 4w], (4w, 8w], ... ; 10th bin is everything above
N_OW = 4  # previous-write class of a write: never / live (<=w) / gray (w, 2w] / stale (>2w)
STAT_KEYS = [
    "examples", "tokens", "writes", "queries", "valid", "expired", "expired_never", "expired_stale",
    "valid_conflict", "distinct_keys_queried", "distinct_keys_written", "live_keys", "gray_keys",
    "ex_vf_sum", "ex_vf_sq", "ex_with_queries", "tokens_max",
]


class OracleError(AssertionError):
    pass


def _zero_stats(w: int) -> dict:
    s = {k: 0 for k in STAT_KEYS}
    s["age_valid"] = np.zeros(w, np.int64)  # index a-1 for age a in 1..w
    s["age_expired"] = np.zeros(N_EXP_BINS, np.int64)
    s["gap"] = np.zeros(GAP_BINS, np.int64)
    s["overwrite"] = np.zeros(N_OW, np.int64)
    return s


def merge_stats(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        out[k] = max(a[k], b[k]) if k == "tokens_max" else a[k] + b[k]
    return out


def oracle(ids: np.ndarray, w: int, n_keys: int, n_values: int, passkey_len: int = G.PASSKEY_LEN):
    """Brute-force re-derivation from input_ids alone.

    Returns (labels, answer_type codes, kind codes, per-example stats). `kind` refines `answer_type` by splitting
    expired into stale (key written earlier, idle > 2w writes) and never-written; codes index KIND_NAMES.
    """
    key0, val0 = G.KEY0, G.KEY0 + n_keys
    t = ids.tolist()
    n = len(t)
    P = passkey_len
    if not (n >= 2 * P + 4 and t[0] == G.BOS and t[1] == G.PK and t[-1] == G.EOS):
        raise OracleError("bad BOS/PK/EOS framing")
    pk = t[2:2 + P]
    if not all(val0 <= x < val0 + n_values for x in pk):
        raise OracleError("passkey tokens are not value tokens")
    n_w = int(np.count_nonzero(ids == G.W))
    wk = np.empty(n_w, np.int16)  # key of the c-th write
    wv = np.empty(n_w, np.int16)  # value token of the c-th write
    last_idx = [-1] * n_keys
    exp_lab = np.full(n, G.IGNORE, np.int16)
    exp_types: list[int] = []
    exp_kinds: list[int] = []
    st = _zero_stats(w)
    w2 = 2 * w
    c, i, last_q_c = 0, 2 + P, 0
    keys_written, keys_queried = set(), set()
    while i < n and t[i] != G.PKQ:
        tok = t[i]
        if tok == G.W:
            if i + 2 >= n:
                raise OracleError("truncated write")
            k, v = t[i + 1] - key0, t[i + 2]
            if not (0 <= k < n_keys and val0 <= v < val0 + n_values):
                raise OracleError(f"bad write tokens at {i}")
            prev = last_idx[k]  # class of the write we are overwriting (age measured before adding this write)
            if prev < 0:
                st["overwrite"][0] += 1
            else:
                age = c - prev
                st["overwrite"][1 if age <= w else 2 if age <= w2 else 3] += 1
            wk[c], wv[c], last_idx[k] = k, v, c
            keys_written.add(k)
            c += 1
            i += 3
        elif tok == G.Q:
            if i + 2 >= n:
                raise OracleError("truncated query")
            k, a = t[i + 1] - key0, t[i + 2]
            if not 0 <= k < n_keys:
                raise OracleError(f"bad query key at {i}")
            idx = np.flatnonzero(wk[:c] == k)  # brute force over the entire history
            if idx.size == 0:
                age, code, exp, kind = None, G.AR_EXPIRED, G.NULL, 3
            else:
                age = c - int(idx[-1])
                if age <= w:
                    code, exp, kind = G.AR_VALID, int(wv[idx[-1]]), 1
                elif age > w2:
                    code, exp, kind = G.AR_EXPIRED, G.NULL, 2
                else:
                    raise OracleError(f"gray-zone query (age={age}) at {i}")
            if a != exp:
                raise OracleError(f"answer token {a} != oracle {exp} at {i} (age={age})")
            exp_lab[i + 2] = exp
            exp_types.append(code)
            exp_kinds.append(kind)
            # ---- statistics
            st["queries"] += 1
            gap = min(c - last_q_c, GAP_BINS - 1)
            st["gap"][gap] += 1
            last_q_c = c
            keys_queried.add(k)
            n_live, n_live2 = (len(np.unique(wk[max(0, c - a_):c])) for a_ in (w, w2))
            st["live_keys"] += n_live
            st["gray_keys"] += n_live2 - n_live
            if code == G.AR_VALID:
                st["valid"] += 1
                st["age_valid"][age - 1] += 1
                if idx.size >= 2 and c - int(idx[-2]) <= w2 and wv[idx[-2]] != wv[idx[-1]]:
                    st["valid_conflict"] += 1  # an older, conflicting binding for this key is still recent
            else:
                st["expired"] += 1
                if age is None:
                    st["expired_never"] += 1
                else:
                    st["expired_stale"] += 1
                    st["age_expired"][min(((age - 1) // w2).bit_length() - 1, N_EXP_BINS - 1)] += 1
            i += 3
        else:
            raise OracleError(f"unexpected token {tok} at {i}")
    if i >= n or t[i] != G.PKQ or t[i + 1:i + 1 + P] != pk or i + 2 + P != n:
        raise OracleError("bad PKQ tail")
    exp_lab[i + 1:i + 1 + P] = pk
    exp_types += [G.PASSKEY] * P
    exp_kinds += [0] * P

    st["examples"], st["tokens"], st["tokens_max"], st["writes"] = 1, n, n, n_w
    st["distinct_keys_queried"], st["distinct_keys_written"] = len(keys_queried), len(keys_written)
    if st["queries"]:
        st["ex_with_queries"] = 1
        vf = st["valid"] / st["queries"]
        st["ex_vf_sum"], st["ex_vf_sq"] = vf, vf * vf
    return exp_lab, np.asarray(exp_types, np.int8), np.asarray(exp_kinds, np.int8), st


def _flat(col):
    """ChunkedArray<list<T>> -> (offsets starting at 0, flat values as an Arrow array)."""
    arr = col.combine_chunks()
    offs = arr.offsets.to_numpy()
    vals = arr.values.slice(int(offs[0]))
    return offs - offs[0], vals


def _read_group(path: str, rg: int):
    tbl = pq.ParquetFile(path).read_row_group(rg)
    offs, ids = _flat(tbl["input_ids"])
    _, labs = _flat(tbl["labels"])
    toffs, tvals = _flat(tbl["answer_type"])
    codes = pc.index_in(tvals, value_set=pa.array(G.TYPE_NAMES))
    if codes.null_count:
        raise OracleError("unknown answer_type string")
    return tbl, offs, ids.to_numpy(), labs.to_numpy(), toffs, codes.to_numpy(zero_copy_only=False)


def _verify_group(job):
    path, rg, params, n_expected, split_seed, mutate = job
    p = G.GenParams(**params)
    tbl, offs, ids, labs, toffs, tcodes = _read_group(path, rg)
    n_writes = tbl["n_writes"].to_numpy()
    seeds = tbl["seed"].to_numpy()
    # `seed` column must decode as split_seed * 10**12 + N * 10**6 + index (this is what keeps split seeds disjoint)
    seed_bad = int(np.count_nonzero((seeds // 10**12 != split_seed) | ((seeds // 10**6) % 10**6 != n_expected)))
    st, digests, errors = _zero_stats(p.w), [], []
    n_sup = 0
    mutation_caught = mutation_total = 0
    for r in range(len(offs) - 1):
        x = ids[offs[r]:offs[r + 1]]
        y = labs[offs[r]:offs[r + 1]]
        ty = tcodes[toffs[r]:toffs[r + 1]]
        try:
            if int(n_writes[r]) != n_expected:
                raise OracleError("n_writes column mismatch")
            exp_lab, exp_ty, _, s = oracle(x, p.w, p.n_keys, p.n_values, p.passkey_len)
            if int(s["writes"]) != n_expected:
                raise OracleError(f"stream has {s['writes']} writes, expected {n_expected}")
            if not np.array_equal(exp_lab, y):
                raise OracleError("labels differ from oracle")
            if not np.array_equal(exp_ty, ty):
                raise OracleError("answer_type differs from oracle")
            n_sup += len(exp_ty)
            st = merge_stats(st, s)
        except OracleError as e:
            errors.append((int(seeds[r]), str(e)))
        digests.append(hashlib.blake2b(x.tobytes(), digest_size=16).digest())
        if mutate and r % mutate == 0:  # mutation test: corrupt one supervised answer, oracle must object
            mutation_total += 1
            xm, ym = x.copy(), y.copy()
            pos = np.flatnonzero(ym != G.IGNORE)[0 if r % 2 else -1]
            old = int(ym[pos])  # replace the answer by a different, valid-looking token
            xm[pos] = ym[pos] = p.val0 + ((old - p.val0 + 1) % p.n_values) if old >= p.val0 else p.val0
            try:
                el, ety, _, _ = oracle(xm, p.w, p.n_keys, p.n_values, p.passkey_len)
                caught = not (np.array_equal(el, ym) and np.array_equal(ety, ty))
            except OracleError:
                caught = True
            mutation_caught += caught
    # determinism: regenerate a few examples from their stored seeds
    det_bad = 0
    for r in np.linspace(0, len(offs) - 2, num=min(3, len(offs) - 1), dtype=int):
        t_, l_, ty_ = G.make_example(int(seeds[r]), n_expected, p)
        if not (np.array_equal(t_, ids[offs[r]:offs[r + 1]]) and np.array_equal(l_, labs[offs[r]:offs[r + 1]])
                and np.array_equal(ty_, tcodes[toffs[r]:toffs[r + 1]])):
            det_bad += 1
    return {"stats": st, "digests": digests, "errors": errors, "n_sup": n_sup,
            "mut": (mutation_caught, mutation_total), "det_bad": det_bad, "det_checked": min(3, len(offs) - 1),
            "seed_bad": seed_bad, "seeds": seeds}


def check_file(pool, path: Path, params: dict, n_writes: int, split_seed: int, mutate: int):
    nrg = pq.ParquetFile(path).num_row_groups
    res = list(pool.map(_verify_group, [(str(path), g, params, n_writes, split_seed, mutate) for g in range(nrg)]))
    st = _zero_stats(params["w"])
    for r in res:
        st = merge_stats(st, r["stats"])
    return {
        "stats": st,
        "digests": [d for r in res for d in r["digests"]],
        "errors": [e for r in res for e in r["errors"]],
        "n_sup": sum(r["n_sup"] for r in res),
        "mut": (sum(r["mut"][0] for r in res), sum(r["mut"][1] for r in res)),
        "det_bad": sum(r["det_bad"] for r in res), "det_checked": sum(r["det_checked"] for r in res),
        "seed_bad": sum(r["seed_bad"] for r in res),
        "seeds": np.concatenate([r["seeds"] for r in res]),
    }


def summarize(st: dict, w: int) -> dict:
    q, ex = max(st["queries"], 1), max(st["examples"], 1)
    exq = max(st["ex_with_queries"], 1)
    mean_vf = st["ex_vf_sum"] / exq
    return {
        "examples": st["examples"],
        "mean_len": st["tokens"] / ex, "max_len": st["tokens_max"],
        "queries_per_example": st["queries"] / ex,
        "queries_per_write": st["queries"] / max(st["writes"], 1),
        "valid_frac": st["valid"] / q, "expired_frac": st["expired"] / q,
        "expired_never_share": st["expired_never"] / max(st["expired"], 1),
        "expired_stale_share": st["expired_stale"] / max(st["expired"], 1),
        "valid_frac_std_across_examples": float(np.sqrt(max(st["ex_vf_sq"] / exq - mean_vf ** 2, 0.0))),
        "valid_conflict_share": st["valid_conflict"] / max(st["valid"], 1),
        "distinct_keys_written_per_ex": st["distinct_keys_written"] / ex,
        "distinct_keys_queried_per_ex": st["distinct_keys_queried"] / ex,
        "live_keys_at_query(age<=w)": st["live_keys"] / q,
        "gray_keys_at_query(w<age<=2w)": st["gray_keys"] / q,
        "overwrite_class_share(never,live,gray,stale)": (st["overwrite"] / max(st["overwrite"].sum(), 1)).round(4).tolist(),
        "age_valid_share": (st["age_valid"] / max(st["valid"], 1)).round(4).tolist(),
        "age_expired_stale_share(bins (2w,4w],(4w,8w],..)": (st["age_expired"] / max(st["expired_stale"], 1)).round(4).tolist(),
        "gap_share(0..15,>=16)": (st["gap"] / q).round(4).tolist(),
        "gap_mean": float((np.arange(GAP_BINS) * st["gap"]).sum() / q),
        "counts": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()},
    }


def run(data_dir: Path, report_dir: Path, workers: int | None = None, mutate: int = 50) -> bool:
    if mutate < 1:
        raise ValueError("the mutation test cannot be disabled (mutate must be >= 1)")
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    meta = json.loads((data_dir / "gen_meta.json").read_text())
    params, w = meta["params"], meta["params"]["w"]
    workers = workers or min(32, os.cpu_count() or 1)
    results, hashes = {}, {}
    ok = True
    with ProcessPoolExecutor(workers) as pool:
        for rel in sorted(meta["files"], key=lambda r: (r.split("/")[1], int(r.split("/")[0][1:]))):
            cfg, split = rel.split("/")[0], rel.split("/")[1].removesuffix(".parquet")
            n = int(cfg[1:])
            if meta["files"][rel]["sha256"] != G.sha256_file(data_dir / rel):
                print(f"FAIL sha256 changed since generation: {rel}")
                ok = False
            r = check_file(pool, data_dir / rel, params, n, meta["seeds"][split], mutate)
            results[(cfg, split)] = r
            hashes[(cfg, split)] = r["digests"]
            good = not r["errors"] and r["det_bad"] == 0 and r["seed_bad"] == 0 and r["mut"][0] == r["mut"][1] > 0
            ok &= good
            print(f"{'OK  ' if good else 'FAIL'} {split:10s} {cfg:7s} rows={r['stats']['examples']:<7d} "
                  f"supervised={r['n_sup']:<9d} oracle_errors={len(r['errors'])} seed_col_bad={r['seed_bad']} "
                  f"mutation_caught={r['mut'][0]}/{r['mut'][1]} determinism_bad={r['det_bad']}/{r['det_checked']}",
                  flush=True)
            for e in r["errors"][:3]:
                print("     error seed=%d: %s" % e)

    # ---- leakage: no shared example across splits; seeds disjoint; no duplicates inside a split
    by_split = {s: set() for s in ("train", "validation", "test")}
    dup = {}
    for (cfg, split), digs in hashes.items():
        dup[(cfg, split)] = len(digs) - len(set(digs))
        by_split[split] |= set(digs)
    leak = {f"{a}&{b}": len(by_split[a] & by_split[b]) for a, b in
            (("train", "validation"), ("train", "test"), ("validation", "test"))}
    seed_sets = {sp: set() for sp in by_split}
    for (cfg, split), r in results.items():
        seed_sets[split] |= set(r["seeds"].tolist())
    seed_overlap = sum(len(seed_sets[a] & seed_sets[b]) for a, b in
                       (("train", "validation"), ("train", "test"), ("validation", "test")))
    seed_ok = len(set(meta["seeds"].values())) == 3 and seed_overlap == 0
    leak_ok = all(v == 0 for v in leak.values()) and seed_ok
    print(f"{'OK  ' if leak_ok else 'FAIL'} leakage shared-examples {leak}; split seeds distinct and example seeds disjoint={seed_ok}; "
          f"in-split duplicates={sum(dup.values())}")
    ok &= leak_ok

    summaries = {f"{cfg}/{split}": summarize(r["stats"], w) for (cfg, split), r in results.items()}
    report = {
        "passed": bool(ok), "params": params, "seeds": meta["seeds"], "generator": meta["generator"],
        "leakage": leak, "duplicates_within_split": {f"{c}/{s}": v for (c, s), v in dup.items()},
        "oracle_labels_checked": {f"{c}/{s}": r["n_sup"] for (c, s), r in results.items()},
        "oracle_errors": {f"{c}/{s}": len(r["errors"]) for (c, s), r in results.items()},
        "files_sha256": {k: v["sha256"] for k, v in meta["files"].items()},
        "summary": summaries,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "passkey_swar_stats.json").write_text(json.dumps(report, indent=1))
    (data_dir / "sanity_report.json").write_text(json.dumps({k: report[k] for k in
                                                          ("passed", "generator", "files_sha256")}, indent=1))
    (report_dir / "passkey_swar_stats.md").write_text(render_markdown(report))
    print(f"{'ALL SANITY CHECKS PASSED' if ok else 'SANITY CHECKS FAILED'}; report -> {report_dir}/passkey_swar_stats.md")
    return ok


def render_markdown(rep: dict) -> str:
    w = rep["params"]["w"]
    L = ["# passkey_swar dataset statistics\n", f"sanity checks passed: **{rep['passed']}**\n",
         "```\n" + json.dumps({"params": rep["params"], "seeds": rep["seeds"]}, indent=1) + "\n```\n",
         f"Leakage (shared examples): {rep['leakage']}\n"]
    order = sorted(rep["summary"], key=lambda k: (k.split("/")[1], int(k.split("/")[0][1:])))
    L += ["## Valid/expired ratio and lengths\n",
          "| split | cfg | examples | mean len | max len | queries/ex | queries/write | valid | expired | expired: never-written | expired: stale | std(valid frac) across examples | valid w/ conflicting old binding |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        s, (cfg, split) = rep["summary"][k], k.split("/")
        L.append(f"| {split} | {cfg} | {s['examples']} | {s['mean_len']:.0f} | {s['max_len']} | {s['queries_per_example']:.1f} | "
                 f"{s['queries_per_write']:.3f} | {s['valid_frac']:.3f} | {s['expired_frac']:.3f} | "
                 f"{s['expired_never_share']:.3f} | {s['expired_stale_share']:.3f} | "
                 f"{s['valid_frac_std_across_examples']:.3f} | {s['valid_conflict_share']:.3f} |")
    L += ["\n## Key reuse\n",
          f"Write overwrite class = state of the key's previous write when it is overwritten: never written / live (age<=w={w}) / gray ({w}<age<={2 * w}) / stale (>{2 * w}).\n",
          "| split | cfg | distinct keys written/ex | distinct keys queried/ex | live keys at query | gray keys at query | overwrite class share (never, live, gray, stale) |",
          "|---|---|---|---|---|---|---|"]
    for k in order:
        s, (cfg, split) = rep["summary"][k], k.split("/")
        L.append(f"| {split} | {cfg} | {s['distinct_keys_written_per_ex']:.2f} | {s['distinct_keys_queried_per_ex']:.2f} | "
                 f"{s['live_keys_at_query(age<=w)']:.2f} | {s['gray_keys_at_query(w<age<=2w)']:.2f} | "
                 f"{s['overwrite_class_share(never,live,gray,stale)']} |")
    L += ["\n## Age of queried key (writes since its last write)\n",
          f"Valid: share per age 1..{w}. Expired-stale: share per bin ({2 * w},{4 * w}], ({4 * w},{8 * w}], ... (doubling).\n",
          "| split | cfg | valid age 1..w | expired-stale age bins |", "|---|---|---|---|"]
    for k in order:
        s, (cfg, split) = rep["summary"][k], k.split("/")
        L.append(f"| {split} | {cfg} | {s['age_valid_share']} | {s['age_expired_stale_share(bins (2w,4w],(4w,8w],..)']} |")
    L += ["\n## Query-gap histogram (writes between consecutive queries; 0..15, >=16)\n",
          "| split | cfg | mean gap | share per gap |", "|---|---|---|---|"]
    for k in order:
        s, (cfg, split) = rep["summary"][k], k.split("/")
        L.append(f"| {split} | {cfg} | {s['gap_mean']:.2f} | {s['gap_share(0..15,>=16)']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data_cache/passkey_swar"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--mutate-every", type=int, default=50, help="mutation test: corrupt every k-th row of each row group and require the oracle to object")
    a = ap.parse_args()
    sys.exit(0 if run(a.data_dir, a.report_dir, a.workers, a.mutate_every) else 1)
