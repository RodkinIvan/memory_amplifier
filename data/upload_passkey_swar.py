#!/usr/bin/env python
"""Upload the verified passkey_swar parquet files to the Hugging Face Hub with `datasets.push_to_hub`.

Safety gates (all must hold, otherwise nothing is uploaded):
  * HF_TOKEN is read from the environment only (never stored, never printed).
  * sanity_report.json says `passed`, and every parquet file still has the sha256 that was verified.
  * the generator files are committed and unchanged since generation, so the dataset card can record the commit.
  * visibility must be chosen explicitly (--private / --public).

    HF_TOKEN=... python data/upload_passkey_swar.py --private                # or --public
    python data/upload_passkey_swar.py --dry-run                              # checks + writes the card locally, no network
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_passkey_swar as G  # noqa: E402

REPO_ID = "irodkin/passkey_swar"
REPO_ROOT = Path(__file__).resolve().parent.parent


def preflight(data_dir: Path, dry_run: bool) -> tuple[dict, dict, str | None]:
    meta = json.loads((data_dir / "gen_meta.json").read_text())
    san = json.loads((data_dir / "sanity_report.json").read_text())
    rep = json.loads((REPO_ROOT / "reports" / "passkey_swar_stats.json").read_text())
    problems = []
    if not san.get("passed") or not rep.get("passed"):
        problems.append("sanity checks did not pass")
    if rep["files_sha256"] != san["files_sha256"]:
        problems.append("reports/passkey_swar_stats.json was not produced from these files")
    for rel, info in meta["files"].items():
        if G.sha256_file(data_dir / rel) != info["sha256"] or san["files_sha256"].get(rel) != info["sha256"]:
            problems.append(f"{rel}: file changed since it was verified")
    files = ["data/make_passkey_swar.py", "data/sanity_checks.py"]
    now = G.git_info(REPO_ROOT, files)
    if now["blob_sha"] != meta["generator"]["blob_sha"]:
        problems.append("generator files changed since the dataset was generated; regenerate + re-verify")
    if now["commit"] is None:
        problems.append("git repo has no commit yet: commit the generator so the card can record its hash")
    elif now["dirty_files"]:
        problems.append(f"uncommitted generator changes: {now['dirty_files']}")
    # the commit the data was generated at; if it was generated from a dirty tree, the current commit
    # is equivalent because the blob hashes above are identical
    commit = meta["generator"]["commit"] if not meta["generator"]["dirty_files"] else now["commit"]
    if problems:
        print("PREFLIGHT " + ("(dry run, would fail)" if dry_run else "FAILED") + ":\n  - " + "\n  - ".join(problems))
        if not dry_run:
            sys.exit(1)
    return meta, rep, commit


def build_card(meta: dict, rep: dict, commit: str | None, yaml_header: str = "") -> str:
    p, S = meta["params"], rep["summary"]
    w = p["w"]
    rows = []
    for split in ("train", "validation", "test"):
        for n in meta["ns"]:
            k = f"n{n}/{split}"
            if k in S:
                s = S[k]
                rows.append(f"| n{n} | {split} | {s['examples']:,} | {s['mean_len']:.0f} | {s['queries_per_example']:.1f} | "
                            f"{s['valid_frac']:.3f} | {s['expired_frac']:.3f} | {s['expired_stale_share']:.3f} |")
    table = "\n".join(rows)
    return f"""{yaml_header}# passkey_swar — Passkey + Sliding-Window Associative Recall

A benchmark for fixed-size-state sequence models (linear attention / SSMs). One sequence contains
(i) a **passkey** that must be retained *indefinitely* and (ii) a stream of associative-recall (AR) writes and
queries that must be *forgotten* (queries on keys not written for more than 2w writes have answer `NULL`).
A single-head model with a scalar decay α cannot satisfy both: forgetting needs α < 1, indefinite retention needs α = 1.

## Format

```
BOS  PK p1 p2 p3 p4   <stream>   PKQ p1 p2 p3 p4 EOS
```

* passkey: 4 tokens from `v0..v{p['n_values'] - 1}` ({p['n_values'] ** 4:,} possibilities).
* stream: `N` AR writes `W k v`, interleaved with queries `Q k a` (`a` = answer). Queries appear at random
  positions: after every write, further queries follow with probability q/(1+q), q = {p['query_rate']} (≈ 1 query per {round(1 / p['query_rate'])} writes).
* keys are drawn from a **drifting active set** of {p['active_min']}–{p['active_max']} of {p['n_keys']} keys; after each write, with probability {p['drift_rate']}
  one active key is swapped for an inactive one.

## Answer rule

Let *age* = number of writes since the key's most recent write (the immediately preceding write has age 1;
overwrites count, only the latest write matters).

| key state | query? | answer | `answer_type` |
|---|---|---|---|
| age ≤ w = {w} | yes | value of that latest write | `ar_valid` |
| age > 2w = {2 * w}, or never written | yes | `NULL` | `ar_expired` |
| {w} < age ≤ {2 * w} | **never queried** (gray zone) | – | – |

Expired queries are what forces decay < 1: without them a delta-rule model could hold all keys and the passkey with α = 1.
Expired queries are either *never-written* keys or *stale* keys (written earlier, then idle > 2w writes); only the latter test
forgetting. With N ≤ 2w = {2 * w} no key can be stale, so `n32` has only never-written expired queries (see table).

## Vocabulary (`vocab.json`, {p['n_keys'] + p['n_values'] + 8} tokens)

`PAD=0 BOS=1 EOS=2 PK=3 PKQ=4 W=5 Q=6 NULL=7`, keys `k0..k{p['n_keys'] - 1}` = ids 8..{7 + p['n_keys']}, values `v0..v{p['n_values'] - 1}` = ids {8 + p['n_keys']}..{7 + p['n_keys'] + p['n_values']}.

## Configs and splits

Config `n{{N}}`, N = number of AR writes. `train`: n32–n256 (100,000 examples each); `validation` and `test`: all
N ∈ {{{', '.join(map(str, meta['ns']))}}} (1,000 examples each). Splits use different seeds and are checked to share no example.

## Columns

* `input_ids`: list[int16] — the full sequence.
* `labels`: list[int16] — `-100` except the answer token `a` of each `Q k a` and the 4 passkey tokens after `PKQ`.
  HF convention: `labels[t]` is the target for the token at position `t` (shift inside the loss).
* `answer_type`: list[str] — one entry per supervised position, in order: `ar_valid` / `ar_expired` per query, then `passkey` ×4.
* `n_writes`: int, `seed`: int64 — per-example seed = `split_seed * 10**12 + N * 10**6 + index`; `make_example(seed, N, params)` reproduces the example exactly.

## Statistics (measured on the released files)

| config | split | examples | mean length | queries/ex | valid | expired | stale share of expired |
|---|---|---|---|---|---|---|---|
{table}

Full tables (query-gap histogram, age histograms, key-reuse statistics) are in `stats.md` in this repo.

## Sanity checks (all passed before upload)

A brute-force oracle that re-derives every label from the token stream alone reproduces 100% of labels and `answer_type`
on every example ({sum(rep['oracle_labels_checked'].values()):,} supervised positions); it also rejects deliberately corrupted examples;
gray-zone queries never occur; regenerating from the stored seed reproduces stored examples; no example is shared between splits.

## Generation

Generator: `make_passkey_swar.py` (in this repo), git commit **`{commit or 'UNCOMMITTED'}`**.
Parameters: `{json.dumps(p)}`; split seeds `{json.dumps(meta['seeds'])}`; versions `{json.dumps(meta['versions'])}`.

```
python data/make_passkey_swar.py --w {p['w']} --n-keys {p['n_keys']} --n-values {p['n_values']} --query-rate {p['query_rate']} \\
    --drift-rate {p['drift_rate']} --seed-train {meta['seeds']['train']} --seed-validation {meta['seeds']['validation']} --seed-test {meta['seeds']['test']}
```

```python
from datasets import load_dataset
ds = load_dataset("{REPO_ID}", "n256", split="train")
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data_cache/passkey_swar"))
    ap.add_argument("--repo-id", default=REPO_ID)
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--private", action="store_true")
    vis.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="run all checks and write the card locally; no network")
    a = ap.parse_args()
    if not a.dry_run and not (a.private or a.public):
        ap.error("choose --private or --public explicitly")

    meta, rep, commit = preflight(a.data_dir, a.dry_run)
    card_path = a.data_dir / "README.md"
    if a.dry_run:
        card_path.write_text(build_card(meta, rep, commit))
        print(f"dry run: card written to {card_path}; nothing uploaded")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set in the environment")

    from datasets import Dataset, Features, Sequence, Value
    from huggingface_hub import DatasetCard, HfApi

    features = Features({"input_ids": Sequence(Value("int16")), "labels": Sequence(Value("int16")),
                         "answer_type": Sequence(Value("string")), "n_writes": Value("int32"), "seed": Value("int64")})
    api = HfApi(token=token)
    api.create_repo(a.repo_id, repo_type="dataset", private=a.private, exist_ok=True)
    order = sorted(meta["files"], key=lambda r: (r.split("/")[1], int(r.split("/")[0][1:])))
    for rel in order:
        cfg, split = rel.split("/")[0], rel.split("/")[1].removesuffix(".parquet")
        ds = Dataset.from_parquet(str(a.data_dir / rel), features=features, cache_dir=str(a.data_dir / ".hf_cache"))
        ds.push_to_hub(a.repo_id, config_name=cfg, split=split, private=a.private, token=token)
        print(f"pushed {cfg}/{split} ({len(ds)} rows)", flush=True)

    # aux files + card (keep the `configs:` YAML that push_to_hub wrote)
    for src, dst in ((a.data_dir / "vocab.json", "vocab.json"), (REPO_ROOT / "data/make_passkey_swar.py", "make_passkey_swar.py"),
                     (REPO_ROOT / "data/sanity_checks.py", "sanity_checks.py"), (REPO_ROOT / "reports/passkey_swar_stats.md", "stats.md")):
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=a.repo_id, repo_type="dataset")
    card = DatasetCard.load(a.repo_id, repo_type="dataset", token=token)
    header = "---\n" + card.data.to_yaml() + "\n---\n\n" if card.data.to_dict() else ""
    DatasetCard(build_card(meta, rep, commit, header)).push_to_hub(a.repo_id, repo_type="dataset", token=token)
    print(f"done: https://huggingface.co/datasets/{a.repo_id}")


if __name__ == "__main__":
    main()
