#!/usr/bin/env python
"""How much fits in the memory the segment loop creates? Fine-tune a trained checkpoint on longer and longer passkeys.

    python scripts/passkey_capacity.py --init checkpoints/swa_fb_gi0_nlnone_seg8_steps50000_s2.pt          # the chain, on GPU 0
    python scripts/passkey_capacity.py --budget-run --start 8 --steps 50000 --lr 1e-3 --warmup 500 --curriculum 2000   # one stage with the original recipe's budget
    python scripts/passkey_capacity.py --reevaluate      # fill in the per-position accuracy of stages that were run before it existed
    python scripts/passkey_capacity.py --model swa_2pass --steps 50000 --lr 1e-3 --warmup 500 --curriculum 2000   # the same chain for the two-pass layer, from its best checkpoint

The checkpoint was trained on streams of up to 256 AR pairs and a passkey of 4 tokens (4 bits each, so 16 bits). Stage P = 8, 16, 32, ... : generate the dataset with a passkey
of P tokens and the same streams statistics (N = 32 ... 256 AR pairs), fine-tune the weights of the previous stage on it, and evaluate the exact match of the passkey (all P tokens
right, teacher forced) on the validation and test split at N = 32 ... 256. The chain stops at the first stage whose validation exact match at N = 256 is below the threshold
(0.9). No longer streams are evaluated here. Each stage starts from the weights of the stage before (the first from `--init`).

Results: results/capacity.json (`stages`: one entry per stage, rewritten after every stage; `budget_runs`: single stages run with another budget), runs in results/raw/ as usual
(wandb group swa_fb_capacity). Every entry has the exact match, the per-token accuracy and the accuracy of every passkey position (and how many positions are at 0.9 or more), per N,
for both splits, plus the step at which each position first reached 0.9 at N = 256 (from the validation history). Datasets have the base dataset's sizes (100,000 training streams per N). Resumable: a finished stage is read
from its result file, an unfinished one resumes from its checkpoint; an out-of-memory crash waits for free memory and starts again.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))
import config as C  # noqa: E402

THRESHOLD = 0.9
NS = [32, 64, 128, 256]  # AR pairs: the training range of the checkpoint, nothing longer is evaluated
OUT = ROOT / "results" / "capacity.json"


MODELS = {  # what a stage fine-tunes: the config file, the wandb group, the model overrides (the checkpoint's architecture), the segment and the default weights of the checkpoint
    "swa_fb": {"config": "configs/swa_fb.yaml", "experiment": "swa_fb_capacity", "segment": 8, "overrides": ["model.nonlin=none", "model.gain_init=0"],
               "init": "checkpoints/swa_fb_gi0_nlnone_seg8_steps50000_s2.pt"},
    "swa_va": {"config": "configs/swa_va.yaml", "experiment": "swa_va_capacity", "segment": 32, "overrides": [], "init": None},
    "swa_2pass": {"config": "configs/swa_2pass.yaml", "experiment": "swa_2pass_capacity", "segment": 32, "overrides": ["model.pass_output=1"],
                  "init": "checkpoints/swa_2pass_seg32_po1_s5.pt"},  # the best two-pass checkpoint: highest validation passkey at N = 16,384 of the 14 output-1 runs of E4 (test 1.00 / 0.99 / 0.96 / 0.87 at N = 256 / 1,024 / 4,096 / 16,384)
}


def result_keys(model: str = "swa_fb", tag: str = "") -> tuple[str, str]:
    """Keys of capacity.json for the chain and for the single budget runs of a model. The first model keeps the original names (`stages`, `budget_runs`); `tag` (--tag) keeps two chains of
    one model with different recipes apart (`stages_swa_2pass_budget`)."""
    suffix = ("" if model == "swa_fb" else f"_{model}") + (f"_{tag}" if tag else "")
    return f"stages{suffix}", f"budget_runs{suffix}"


def stage_overrides(P: int, init: str, steps: int, lr: float, seed: int = 0, warmup: int = 200, curriculum: int = 0, model: str = "swa_fb", segment: int | None = None) -> list[str]:
    """Overrides of the model's config for one stage: the model of the checkpoint (for `swa_fb` segment 8, no tanh, gain 0), a fine-tuning schedule, the passkey length."""
    ns = "[" + ",".join(map(str, NS)) + "]"
    m = MODELS[model]
    return [f"experiment={m['experiment']}", f"seed={seed}", f"model.segment={segment or m['segment']}", *m["overrides"], f"passkey_len={P}", f"train.init_from={init}",
            f"train.steps={steps}", f"train.lr={lr:g}", f"train.warmup_steps={warmup}", f"train.curriculum_stage_steps={curriculum}", "train.eval_every=1000",
            f"train.eval_ns={ns}", f"eval.ns={ns}"]


def stage_name(P: int, init: str, steps: int, lr: float, seed: int = 0, warmup: int = 200, curriculum: int = 0, model: str = "swa_fb", segment: int | None = None) -> str:
    cfg, explicit = C.load_config(ROOT / MODELS[model]["config"], stage_overrides(P, init, steps, lr, seed, warmup, curriculum, model, segment))
    return C.run_name(cfg, explicit)


POS_OK = 0.9  # a passkey position counts as stored when it is right in at least this share of the streams


def position_switch_on(history: list[dict], n: int = 256, thr: float = POS_OK) -> list[int | None]:
    """For every passkey position the first validation step at which its accuracy at stream length n reached `thr` (None: never)."""
    key = f"val/n{n}/passkey_pos_acc"
    pts = [(h["step"], h[key]) for h in history if key in h]
    if not pts:
        return []
    return [next((st for st, acc in pts if acc[j] >= thr), None) for j in range(len(pts[0][1]))]


def summarize(P: int, name: str, init: str, result: dict, threshold: float = THRESHOLD) -> dict:
    """One entry of capacity.json from a result file: exact match, per-token accuracy and per-position accuracy of the passkey at every N, both splits."""
    rows = {(r["split"], r["N"]): r for r in result["rows"]}
    metric = lambda split, key: {str(n): round(rows[(split, n)][key], 4) for n in NS}

    def positions(split):
        if "passkey_pos_acc" not in rows[(split, 256)]:
            return {}  # a result written before the per-position metric existed (see --reevaluate)
        return {str(n): [round(x, 4) for x in rows[(split, n)]["passkey_pos_acc"]] for n in NS}

    def block(split):
        pos = positions(split)
        return {"exact": metric(split, "passkey_acc"), "token": metric(split, "passkey_token_acc"), "ar_valid": metric(split, "ar_valid_acc"),
                "positions": pos, "positions_ok": {n: sum(x >= POS_OK for x in v) for n, v in pos.items()}}

    return {"passkey_len": P, "bits": P * 4, "run": name, "init": init, "train_time_s": round(result["train_time_s"]), "steps": result["config"]["train"]["steps"],
            "train_size": (result.get("dataset") or {}).get("sizes", {}).get("train"),  # streams per N in the data the stage was trained on (None: not recorded)
            "recipe": {k: result["config"]["train"].get(k) for k in ("lr", "warmup_steps", "curriculum_stage_steps")},
            "validation": block("validation"), "test": block("test"),
            "switch_on_step": position_switch_on(result.get("val_history") or []),
            "passed": rows[("validation", 256)]["passkey_acc"] >= threshold}


@contextlib.contextmanager
def file_lock(path: Path):
    """Exclusive advisory lock on `path` (created if needed) for the length of the `with` block, across processes and threads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def write_capacity(update: dict, budget_run: dict | None = None, budget_key: str = "budget_runs") -> None:
    """Merge `update` into results/capacity.json (the chain's `stages` and the separate `budget_runs` live in one file). `budget_run` is added to (or replaces the
    same run in) the `budget_runs` list. The read-modify-write is under a file lock: two budget runs (seeds) may finish at the same time."""
    with file_lock(OUT.with_suffix(".lock")):
        data = json.loads(OUT.read_text()) if OUT.exists() else {}
        data.update({"threshold": THRESHOLD, "criterion": "validation exact match of the passkey at N = 256", **update})
        if budget_run is not None:
            data[budget_key] = [*(e for e in data.get(budget_key, []) if e["run"] != budget_run["run"]), budget_run]
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(OUT)


def merge_positions(updates: dict[str, dict]) -> None:
    """Write per-position results into the entries of capacity.json. `updates` = {run name: {split: {"positions": ..., "positions_ok": ...}}}. The file is read again under
    the lock, so an entry another process added while the (slow) evaluation ran is kept (review L12)."""
    with file_lock(OUT.with_suffix(".lock")):
        data = json.loads(OUT.read_text())
        for group in (k for k in data if k.startswith(("stages", "budget_runs"))):
            for entry in data[group]:
                for split, fields in updates.get(entry["run"], {}).items():
                    entry[split].update(fields)
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(OUT)


def chain(run_stage, first: int, max_len: int, init: str, threshold: float = THRESHOLD, key: str = "stages") -> list[dict]:
    """P = first, 2 first, 4 first, ... while the stage passes and P <= max_len. `run_stage(P, init)` returns (summary entry, path of the stage's weights)."""
    stages, P = [], first
    while P <= max_len:
        entry, weights = run_stage(P, init)
        stages.append(entry)
        write_capacity({key: stages})
        ok = entry["validation"]["positions_ok"].get("256")
        print(f"passkey {P} tokens ({P * 4} bits): validation exact {entry['validation']['exact']['256']:.3f} at N = 256"
              + (f", {ok} of {P} positions at {POS_OK} or more" if ok is not None else "") + f" -> {'passed' if entry['passed'] else 'FAILED, stop'}", flush=True)
        if not entry["passed"]:
            break
        init, P = weights, P * 2
    return stages


def free_gb(gpu: int) -> float:
    r = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return float(r.stdout.split()[0]) / 1024


TRAIN_SIZE = 100_000  # streams per N: the size of the base dataset, so the passkey length is the only difference between the data of a stage and the checkpoint's


def ensure_dataset(P: int, workers: int = 16, train_size: int = TRAIN_SIZE) -> Path:
    """The dataset folder for passkeys of P tokens, generated (and sanity-checked) when it does not exist. Under a lock per folder, so two drivers that need the
    same new dataset do not write the same files: the second waits and then finds it (review L13)."""
    out = C.data_dir_for({"data_dir": "data_cache/passkey_swar", "passkey_len": P})
    with file_lock(out.with_name(out.name + ".lock")):
        if (out / "gen_meta.json").exists():
            size = json.loads((out / "gen_meta.json").read_text())["sizes"]["train"]
            if size != train_size:
                raise SystemExit(f"{out} has {size} training streams per N, expected {train_size}: remove the folder to generate it again")
            return out
        import make_passkey_swar as G
        import sanity_checks as S
        ns = [str(n) for n in NS]
        G.main(["--out-dir", str(out), "--ns", *ns, "--train-ns", *ns, "--train-size", str(train_size), "--workers", str(workers), "--passkey-len", str(P), "--skip-sanity"])
        if not S.run(out, out / "reports", workers=workers):  # the oracle re-derives every label from the tokens; a dataset that fails is not used
            raise SystemExit(f"sanity checks failed for {out}")
        return out


def wait_for_memory(gpu: int, need_gb: float, poll: int = 60) -> None:
    """Block until the GPU has `need_gb` free: the GPU is shared with other people's jobs, so a stage does not start on a full one."""
    shown = False
    while (free := free_gb(gpu)) < need_gb:
        if not shown:
            print(f"waiting for {need_gb:.0f} GB free on GPU {gpu} ({free:.1f} GB now)", flush=True)
            shown = True
        time.sleep(poll)


def run_training(overrides: list[str], name: str, gpu: int, need_gb: float, log: Path, attempts: int = 30, config: str = "configs/swa_fb.yaml") -> None:
    """train.py in a subprocess on one GPU; after an out-of-memory crash wait for free memory and start again (it resumes from its checkpoint)."""
    cmd = [sys.executable, "src/train.py", "--config", config, "--set", *overrides]
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        start = log.stat().st_size if log.exists() else 0
        with log.open("ab") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).returncode
        if rc == 0:
            return
        tail = log.read_bytes()[start:].decode(errors="ignore")
        if not re.search(r"out of memory|OutOfMemoryError", tail, re.I):
            raise SystemExit(f"{name} failed (see {log})")
        print(f"{name}: out of memory (attempt {attempt}); waiting for {need_gb:.0f} GB free on GPU {gpu}", flush=True)
        time.sleep(300)
        while free_gb(gpu) < need_gb:
            time.sleep(60)
    raise SystemExit(f"{name}: gave up after {attempts} starts")


def reevaluate(gpu: int) -> None:
    """Add the per-position passkey accuracy to the stages of capacity.json that were run before it existed: evaluates their final weights on the validation and test split."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    import data as D
    import evaluate as E
    from models import build_from_config
    data = json.loads(OUT.read_text())
    updates: dict[str, dict] = {}
    for group in (k for k in data if k.startswith(("stages", "budget_runs"))):
        for entry in data[group]:
            if entry["validation"].get("positions"):
                continue
            res = json.loads((ROOT / "results" / "raw" / f"{entry['run']}.json").read_text())
            cfg = res["config"]
            ddir = C.data_dir_for(cfg)
            D.prepare_kinds(ddir, json.loads((ddir / "gen_meta.json").read_text())["params"], workers=8)  # the per-position kinds of the eval files (no-op when cached)
            model = build_from_config(cfg["model"], json.loads((ddir / "vocab.json").read_text())["size"])
            model.load_state_dict(torch.load(ROOT / "runs" / entry["run"] / "model.pt", map_location="cpu", weights_only=True))
            model = model.cuda()
            found = updates[entry["run"]] = {}
            for split in ("validation", "test"):
                pos = {}
                for n in NS:
                    m = E.evaluate_packed(model, D.load_packed(ddir / f"n{n}" / f"{split}.parquet", kinds=True), torch.device("cuda"), amp_dtype=torch.bfloat16)
                    pos[str(n)] = [round(x, 4) for x in m["passkey_pos_acc"]]
                found[split] = {"positions": pos, "positions_ok": {n: sum(x >= POS_OK for x in v) for n, v in pos.items()}}
            print(f"{entry['run']}: positions at {POS_OK} or more at N = 256: validation {found['validation']['positions_ok']['256']}, test {found['test']['positions_ok']['256']}", flush=True)
    if updates:
        merge_positions(updates)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="swa_fb", help="the model kind of the checkpoint (swa_va: value adaptation, swa_2pass: two-pass layer; the results of a model other than swa_fb go under `stages_<model>` / `budget_runs_<model>`)")
    ap.add_argument("--tag", default="", help="suffix of the result keys, to keep two chains of one model with different recipes apart (e.g. 'budget': `stages_swa_2pass_budget`)")
    ap.add_argument("--segment", type=int, default=None, help="segment of the checkpoint (default: 8 for swa_fb, 32 for swa_va)")
    ap.add_argument("--init", default=None, help="weights of the checkpoint trained on 4-token passkeys (default: the best checkpoint of the model, see MODELS)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200, help="warm-up steps (the original recipe: 500)")
    ap.add_argument("--curriculum", type=int, default=0, help="steps per stream-length stage, 0 = all lengths from the start (the original recipe: 2000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-run", action="store_true", help="run only the stage --start with the given budget and record it under `budget_runs` (no chain, no stop rule)")
    ap.add_argument("--reevaluate", action="store_true", help="fill in the per-position accuracy of recorded stages and exit")
    a = ap.parse_args()
    if a.reevaluate:
        return reevaluate(a.gpu)
    a.init = a.init or MODELS[a.model]["init"]
    if not a.init:
        ap.error(f"--init is required for --model {a.model}")

    def run_stage(P: int, init: str):
        ensure_dataset(P)
        overrides = stage_overrides(P, init, a.steps, a.lr, a.seed, a.warmup, a.curriculum, a.model, a.segment)
        cfg, explicit = C.load_config(ROOT / MODELS[a.model]["config"], overrides)
        name = C.run_name(cfg, explicit)
        result_file = ROOT / "results" / "raw" / f"{name}.json"
        if not result_file.exists():
            wait_for_memory(a.gpu, C.footprint_gb(cfg))
            print(f"stage {P}: training {name} from {init}", flush=True)
            run_training(overrides, name, a.gpu, C.footprint_gb(cfg), ROOT / "logs" / "capacity" / f"{name}.log", config=MODELS[a.model]["config"])
        result = json.loads(result_file.read_text())
        return summarize(P, name, init, result), str((ROOT / "runs" / name / "model.pt").relative_to(ROOT))

    if a.budget_run:
        entry, _ = run_stage(a.start, a.init)
        write_capacity({}, budget_run=entry, budget_key=result_keys(a.model, a.tag)[1])
        print(f"budget run, passkey {a.start}: validation exact {entry['validation']['exact']['256']:.3f} at N = 256, positions at {POS_OK}+: {entry['validation']['positions_ok'].get('256')}/{a.start}", flush=True)
        return
    chain(run_stage, a.start, a.max_len, a.init, key=result_keys(a.model, a.tag)[0])


if __name__ == "__main__":
    main()
