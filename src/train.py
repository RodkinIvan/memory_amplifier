#!/usr/bin/env python
"""Train one passkey_swar baseline and evaluate it on every N (validation + test).

    python src/train.py --config configs/exp1.yaml --set model.alpha=0.95 seed=1

The wandb run name is `<model kind>_<every overridden hyperparameter>_s<seed>`, e.g. `gdn_fixed_a0.95_s1`.
Results go to results/raw/<name>.json, the final weights to runs/<name>/model.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
from models import build_from_config  # noqa: E402

METRICS = ("passkey_acc", "ar_valid_acc", "ar_expired_acc", "ar_expired_stale_acc", "ar_expired_never_acc")


def load_wandb_key() -> None:
    """Use $WANDB_API_KEY, else the (git-ignored) `wandb_token` file in the repo root. The key is never printed."""
    f = C.ROOT / "wandb_token"
    if not os.environ.get("WANDB_API_KEY") and f.exists():
        os.environ["WANDB_API_KEY"] = f.read_text().strip()


def lr_at(step: int, tc: dict) -> float:
    if step <= tc["warmup_steps"]:
        return tc["lr"] * step / tc["warmup_steps"]
    frac = (step - tc["warmup_steps"]) / max(1, tc["steps"] - tc["warmup_steps"])
    return tc["lr"] * (tc["min_lr_ratio"] + (1 - tc["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))


def make_optimizer(model, tc: dict):
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]
    return torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=tc["lr"], betas=tuple(tc["betas"]))


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/exp1.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides, e.g. model.alpha=0.9")
    ap.add_argument("--force", action="store_true", help="rerun even if results/raw/<name>.json exists")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    cfg, explicit = C.load_config(args.config, args.set)
    name = C.run_name(cfg, explicit)
    out_json = C.ROOT / "results" / "raw" / f"{name}.json"
    if out_json.exists() and not args.force:
        if json.loads(out_json.read_text())["config"] != cfg:
            sys.exit(f"{out_json} exists with a different config (same name!); use --force or change an override")
        print(f"[{name}] already done, skipping", flush=True)
        return

    mc, tc, ec = cfg["model"], cfg["train"], cfg["eval"]
    data_dir = Path(cfg["data_dir"])
    data_dir = data_dir if data_dir.is_absolute() else C.ROOT / data_dir
    meta = json.loads((data_dir / "gen_meta.json").read_text())
    vocab_size = json.loads((data_dir / "vocab.json").read_text())["size"]
    D.prepare_kinds(data_dir, meta["params"], workers=8)  # no-op when cached

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = torch.device("cuda")
    model = build_from_config(mc, vocab_size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    amp = torch.bfloat16 if tc.get("precision", "fp32") == "bf16" else None  # flash-attention models need bf16

    load_wandb_key()
    import wandb
    run = wandb.init(project=cfg["project"], name=name, group=cfg["experiment"], config={**C.flatten(cfg), "n_params": n_params},
                     mode="disabled" if args.no_wandb else "online", dir=str(C.ROOT))
    print(f"[{name}] params={n_params:,} wandb={run.url if not args.no_wandb else 'off'}", flush=True)

    train = D.load_train(data_dir, tc["train_ns"])
    val = {n: D.load_packed(data_dir / f"n{n}" / "validation.parquet", kinds=True).subset(tc["eval_examples"])
           for n in tc["eval_ns"]}
    opt = make_optimizer(model, tc)
    model.train()
    t0, loss_acc, n_acc, last = time.time(), torch.zeros((), device=dev), 0, 0.0
    for step in range(1, tc["steps"] + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        ids, labs = D.pad_batch(train, rng.integers(0, len(train), size=tc["batch_size"]), tc["pad_multiple"])
        ids, labs = torch.from_numpy(ids).to(dev), torch.from_numpy(labs).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp is not None):
            logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labs[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_acc += loss.detach()
        n_acc += 1
        if step % tc["log_every"] == 0:
            last = float(loss_acc) / n_acc
            if not math.isfinite(last):
                raise RuntimeError(f"[{name}] non-finite loss at step {step}")
            wandb.log({"train/loss": last, "train/lr": opt.param_groups[0]["lr"], "train/grad_norm": float(gnorm),
                       "train/steps_per_s": step / (time.time() - t0)}, step=step)
            loss_acc.zero_()
            n_acc = 0
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            logs = {}
            for n, p in val.items():
                m = E.evaluate_packed(model, p, dev, amp_dtype=amp)
                logs.update({f"val/n{n}/{k}": m[k] for k in METRICS})
            wandb.log(logs, step=step)
            print(f"[{name}] step {step} loss {last:.4f} | " + " ".join(
                f"n{n}: pk {logs[f'val/n{n}/passkey_acc']:.2f} v {logs[f'val/n{n}/ar_valid_acc']:.2f} "
                f"e {logs[f'val/n{n}/ar_expired_acc']:.2f}" for n in (32, 256, 1024) if n in val), flush=True)
    train_time = time.time() - t0

    rows = []
    for split in ec["splits"]:
        for n in ec["ns"]:
            m = E.evaluate_packed(model, D.load_packed(data_dir / f"n{n}" / f"{split}.parquet", kinds=True), dev, ec["tokens_per_batch"], amp_dtype=amp)
            rows.append({"split": split, "N": n, **m})
            run.summary.update({f"final/{split}/n{n}/{k}": m[k] for k in METRICS})
        print(f"[{name}] {split}: " + " ".join(f"n{r['N']}:{r['passkey_acc']:.2f}/{r['ar_expired_acc']:.2f}" for r in rows if r["split"] == split), flush=True)

    ckpt = C.ROOT / "runs" / name / "model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    write_json_atomic(out_json, {"name": name, "config": cfg, "explicit_keys": explicit, "n_params": n_params,
                                 "final_train_loss": last, "train_time_s": train_time, "rows": rows})
    run.finish()
    print(f"[{name}] done in {time.time() - t0:.0f}s -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
