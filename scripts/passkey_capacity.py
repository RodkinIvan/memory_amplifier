#!/usr/bin/env python
"""How much fits in the memory the segment loop creates? Fine-tune a trained checkpoint on longer and longer passkeys.

    python scripts/passkey_capacity.py --gpu 1 --init checkpoints/swa_fb_gi0_nlnone_seg8_steps50000_s2.pt

The checkpoint was trained on streams of up to 256 AR pairs and a passkey of 4 tokens (4 bits each, so 16 bits). Stage P = 8, 16, 32, ... : generate the dataset with a passkey
of P tokens and the same streams statistics (N = 32 ... 256 AR pairs), fine-tune the weights of the previous stage on it, and evaluate the exact match of the passkey (all P tokens
right, teacher forced) on the validation and test split at N = 32 ... 256. The chain stops at the first stage whose validation exact match at N = 256 is below the threshold
(0.9). No longer streams are evaluated here. Each stage starts from the weights of the stage before (the first from `--init`).

Results: results/capacity.json (one entry per stage, rewritten after every stage), runs in results/raw/ as usual (wandb group swa_fb_capacity). Resumable: a finished stage is read
from its result file, an unfinished one resumes from its checkpoint; an out-of-memory crash waits for free memory and starts again.
"""
from __future__ import annotations

import argparse
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


def stage_overrides(P: int, init: str, steps: int, lr: float, seed: int = 0) -> list[str]:
    """Overrides of configs/swa_fb.yaml for one stage: the model of the checkpoint (segment 8, no tanh, gain 0), a fine-tuning schedule, the passkey length."""
    ns = "[" + ",".join(map(str, NS)) + "]"
    return ["experiment=swa_fb_capacity", f"seed={seed}", "model.segment=8", "model.nonlin=none", "model.gain_init=0", f"passkey_len={P}", f"train.init_from={init}",
            f"train.steps={steps}", f"train.lr={lr:g}", "train.warmup_steps=200", "train.curriculum_stage_steps=0", "train.eval_every=1000",
            f"train.eval_ns={ns}", f"eval.ns={ns}"]


def stage_name(P: int, init: str, steps: int, lr: float, seed: int = 0) -> str:
    cfg, explicit = C.load_config(ROOT / "configs" / "swa_fb.yaml", stage_overrides(P, init, steps, lr, seed))
    return C.run_name(cfg, explicit)


def summarize(P: int, name: str, init: str, result: dict, threshold: float = THRESHOLD) -> dict:
    """One entry of capacity.json from a result file: exact match and per-token accuracy of the passkey at every N, both splits."""
    rows = {(r["split"], r["N"]): r for r in result["rows"]}
    metric = lambda split, key: {str(n): round(rows[(split, n)][key], 4) for n in NS}
    return {"passkey_len": P, "bits": P * 4, "run": name, "init": init, "train_time_s": round(result["train_time_s"]), "steps": result["config"]["train"]["steps"],
            "validation": {"exact": metric("validation", "passkey_acc"), "token": metric("validation", "passkey_token_acc"), "ar_valid": metric("validation", "ar_valid_acc")},
            "test": {"exact": metric("test", "passkey_acc"), "token": metric("test", "passkey_token_acc"), "ar_valid": metric("test", "ar_valid_acc")},
            "passed": rows[("validation", 256)]["passkey_acc"] >= threshold}


def chain(run_stage, first: int, max_len: int, init: str, threshold: float = THRESHOLD) -> list[dict]:
    """P = first, 2 first, 4 first, ... while the stage passes and P <= max_len. `run_stage(P, init)` returns (summary entry, path of the stage's weights)."""
    stages, P = [], first
    while P <= max_len:
        entry, weights = run_stage(P, init)
        stages.append(entry)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps({"threshold": threshold, "criterion": "validation exact match of the passkey at N = 256", "stages": stages}, indent=1))
        print(f"passkey {P} tokens ({P * 4} bits): validation exact {entry['validation']['exact']['256']:.3f} at N = 256 -> {'passed' if entry['passed'] else 'FAILED, stop'}", flush=True)
        if not entry["passed"]:
            break
        init, P = weights, P * 2
    return stages


def free_gb(gpu: int) -> float:
    r = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return float(r.stdout.split()[0]) / 1024


def ensure_dataset(P: int, workers: int = 16) -> Path:
    out = C.data_dir_for({"data_dir": "data_cache/passkey_swar", "passkey_len": P})
    if (out / "gen_meta.json").exists():
        return out
    import make_passkey_swar as G
    import sanity_checks as S
    ns = [str(n) for n in NS]
    G.main(["--out-dir", str(out), "--ns", *ns, "--train-ns", *ns, "--train-size", "30000", "--workers", str(workers), "--passkey-len", str(P), "--skip-sanity"])
    if not S.run(out, out / "reports", workers=workers):  # the oracle re-derives every label from the tokens; a dataset that fails is not used
        raise SystemExit(f"sanity checks failed for {out}")
    return out


def run_training(overrides: list[str], name: str, gpu: int, need_gb: float, log: Path, attempts: int = 30) -> None:
    """train.py in a subprocess on one GPU; after an out-of-memory crash wait for free memory and start again (it resumes from its checkpoint)."""
    cmd = [sys.executable, "src/train.py", "--config", "configs/swa_fb.yaml", "--set", *overrides]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="checkpoints/swa_fb_gi0_nlnone_seg8_steps50000_s2.pt", help="weights of the checkpoint trained on 4-token passkeys")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--start", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    def run_stage(P: int, init: str):
        ensure_dataset(P)
        overrides = stage_overrides(P, init, a.steps, a.lr, a.seed)
        cfg, explicit = C.load_config(ROOT / "configs" / "swa_fb.yaml", overrides)
        name, result_file = C.run_name(cfg, explicit), None
        result_file = ROOT / "results" / "raw" / f"{name}.json"
        if not result_file.exists():
            print(f"stage {P}: training {name} from {init}", flush=True)
            run_training(overrides, name, a.gpu, C.footprint_gb(cfg), ROOT / "logs" / "capacity" / f"{name}.log")
        result = json.loads(result_file.read_text())
        return summarize(P, name, init, result), str((ROOT / "runs" / name / "model.pt").relative_to(ROOT))

    chain(run_stage, a.start, a.max_len, a.init)


if __name__ == "__main__":
    main()
