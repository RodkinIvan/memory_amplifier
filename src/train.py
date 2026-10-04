#!/usr/bin/env python
"""Train one passkey_swar baseline and evaluate it on every N (validation + test).

    python src/train.py --config configs/exp1.yaml --set model.alpha=0.95 seed=1

The wandb run name is `<model kind>_<every overridden hyperparameter>_s<seed>`, e.g. `gdn_fixed_a0.95_s1`.
Results go to results/raw/<name>.json, the final weights to runs/<name>/model.pt.

Resumable: at every periodic validation the full training state is written to runs/<name>/resume.pt (atomically). A run that is
started again with the same config continues from it (model, optimizer, both random streams, step, wandb run); `launch.py` workers
do this by themselves after an out-of-memory crash, and running the same launch command again continues stopped runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
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


def passkey_weight_at(step: int, tc: dict) -> float:
    """Loss weight of the passkey tokens at `step`: `train.passkey_weight` decays log-linearly to 1 over
    `train.passkey_weight_anneal_steps` steps (0 = constant). A large start value bootstraps a memory loop that would
    otherwise look like noise; annealing back to 1 makes the final model train on the plain objective."""
    w0, T = float(tc.get("passkey_weight", 1.0)), int(tc.get("passkey_weight_anneal_steps", 0))
    if T <= 0 or w0 == 1.0:
        return w0
    return math.exp(math.log(w0) * (1.0 - min(1.0, step / T)))


def active_train_ns(step: int, tc: dict) -> list[int]:
    """Training configs that may be sampled at `step`. With `train.curriculum_stage_steps` = S > 0 training starts on the
    first config of `train.train_ns` only (shortest streams, fewest memory-loop hops) and adds the next config every S
    steps; 0 = all configs from the start."""
    ns, stage = list(tc["train_ns"]), int(tc.get("curriculum_stage_steps", 0))
    if stage <= 0:
        return ns
    return ns[:min(len(ns), 1 + (step - 1) // stage)]


def sample_train_indices(rng, tc: dict, bounds: dict, step: int, n_total: int, batch: int):
    """Example indices into the pooled train set. `bounds[n] = (start, end)` is the slice of config n in the pool.
    Without a curriculum this is the plain uniform draw over the whole pool (same random stream as before)."""
    if int(tc.get("curriculum_stage_steps", 0)) <= 0:
        return rng.integers(0, n_total, size=batch)
    active = active_train_ns(step, tc)
    cfg = rng.integers(0, len(active), size=batch)
    lo = np.array([bounds[n][0] for n in active])[cfg]
    hi = np.array([bounds[n][1] for n in active])[cfg]
    return lo + (rng.random(batch) * (hi - lo)).astype(np.int64)


PERIODIC_TOKENS_PER_BATCH = 262144  # batch of the periodic validation (the final evaluation uses eval.tokens_per_batch)


def eval_examples_for(tc: dict, n: int) -> int:
    """Validation examples per N in the periodic evaluation: `train.eval_examples`, except for the N listed in
    `train.eval_examples_by_n` = [[N, count], ...] (long streams are expensive, a small subset is enough)."""
    return int(dict((int(a), int(b)) for a, b in tc.get("eval_examples_by_n") or []).get(n, tc["eval_examples"]))


def evaluate_with_backoff(model, packed, dev, tokens_per_batch: int, amp, min_tokens: int = 65536) -> dict:
    """`evaluate_packed`, but when the GPU is shared by many runs and a long-stream batch does not fit, retry with half
    the batch instead of losing a trained model. The metrics do not depend on the batch size (up to float rounding)."""
    while True:
        try:
            return E.evaluate_packed(model, packed, dev, tokens_per_batch, amp_dtype=amp)
        except torch.cuda.OutOfMemoryError:
            if tokens_per_batch // 2 < min_tokens:
                raise
        # outside the except block, so the traceback no longer pins the failed batch and empty_cache can release it
        torch.cuda.empty_cache()
        tokens_per_batch //= 2
        print(f"evaluation ran out of memory, retrying with tokens_per_batch={tokens_per_batch}", flush=True)


def make_optimizer(model, tc: dict):
    """AdamW with weight decay on matrices (not embeddings/norms); feedback-loop parameters ("loops.") get their own
    learning-rate multiplier `train.loop_lr_mult` and no weight decay."""
    named = list(model.named_parameters())
    loop = [p for n, p in named if ".loops." in "." + n or n.startswith("loops.")]
    rest = [(n, p) for n, p in named if not any(p is q for q in loop)]
    decay = [p for n, p in rest if p.ndim >= 2 and "embed" not in n]
    no_decay = [p for n, p in rest if p.ndim < 2 or "embed" in n]
    groups = [{"params": decay, "weight_decay": tc["weight_decay"], "lr_mult": 1.0},
              {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0}]
    if loop:
        groups.append({"params": loop, "weight_decay": 0.0, "lr_mult": float(tc.get("loop_lr_mult", 1.0))})
    return torch.optim.AdamW(groups, lr=tc["lr"], betas=tuple(tc["betas"]))


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, passkey_weight: float = 1.0) -> torch.Tensor:
    """Cross-entropy on supervised positions (label != -100). The last 4 supervised tokens of every row are the passkey;
    `passkey_weight` multiplies their weight (1.0 = the plain mean over supervised tokens)."""
    tgt = labels[:, 1:]
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), ignore_index=-100, reduction="none")
    ce = ce.view_as(tgt)
    sup = tgt != -100
    w = sup.float()
    if passkey_weight != 1.0:
        from_end = sup.flip(1).cumsum(1).flip(1)  # 1 for the last supervised token of a row, 2 for the one before, ...
        w = w * (1.0 + (passkey_weight - 1.0) * (sup & (from_end <= 4)).float())
    return (ce * w).sum() / w.sum()


def save_resume(path: Path, state: dict) -> None:
    """Write the training state next to the final weights; a crash while writing leaves the previous file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path: Path, cfg: dict) -> dict | None:
    """The saved state of an interrupted run with this exact config, None if there is none. A file written with another config
    (same run name) stops the run instead of being mixed in."""
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != cfg:
        sys.exit(f"{path} was written with a different config (same name!); remove it or use --force")
    return state


def git_state() -> dict:
    """Commit the code was started from and whether tracked files had uncommitted changes (empty outside a git checkout)."""
    def git(*a):
        r = subprocess.run(["git", *a], cwd=C.ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    return {"commit": commit, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))} if commit else {}


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/exp1.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides, e.g. model.alpha=0.9")
    ap.add_argument("--force", action="store_true", help="rerun even if results/raw/<name>.json exists; also discards runs/<name>/resume.pt")
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

    resume_file = C.ROOT / "runs" / name / "resume.pt"
    if args.force:
        resume_file.unlink(missing_ok=True)
    state = load_resume(resume_file, cfg)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = torch.device("cuda")
    model = build_from_config(mc, vocab_size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    amp = torch.bfloat16 if tc.get("precision", "fp32") == "bf16" else None  # flash-attention models need bf16

    load_wandb_key()
    import wandb
    run = wandb.init(project=cfg["project"], name=name, group=cfg["experiment"], notes=C.run_description(cfg),
                     config={**C.flatten(cfg), "n_params": n_params, "reference": C.reference_name(cfg), "diff": C.describe_diff(C.reference_diff(cfg))},
                     id=state["wandb_id"] if state else None, resume="allow" if state else None,
                     mode="disabled" if args.no_wandb else "online", dir=str(C.ROOT))
    print(f"[{name}] params={n_params:,} wandb={run.url if not args.no_wandb else 'off'}", flush=True)

    train = D.load_train(data_dir, tc["train_ns"])
    bounds, start = {}, 0
    for n in tc["train_ns"]:  # slice of every config inside the pooled train set (needed by the curriculum)
        rows = pq.ParquetFile(data_dir / f"n{n}" / "train.parquet").metadata.num_rows
        bounds[n] = (start, start + rows)
        start += rows
    val = {n: D.load_packed(data_dir / f"n{n}" / "validation.parquet", kinds=True).subset(eval_examples_for(tc, n))
           for n in tc["eval_ns"]}
    opt = make_optimizer(model, tc)
    model.train()
    loss_acc, n_acc, last, start, elapsed, resumed_at, history = torch.zeros((), device=dev), 0, 0.0, 1, 0.0, [], []
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        rng.bit_generator.state = state["np_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        loss_acc.fill_(state["loss_acc"])
        n_acc, last, start, elapsed, resumed_at = state["n_acc"], state["last_loss"], state["step"] + 1, state["train_time_s"], [*state["resumed_at_steps"], state["step"]]
        history = state["history"]
        print(f"[{name}] resuming from step {state['step']} ({resume_file})", flush=True)
    git = git_state()  # of this start; a resumed run may have been started from another commit
    t0 = time.time() - elapsed  # the time of the earlier attempts counts as training time
    for step in range(start, tc["steps"] + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc) * g.get("lr_mult", 1.0)
        ids, labs = D.pad_batch(train, sample_train_indices(rng, tc, bounds, step, len(train), tc["batch_size"]), tc["pad_multiple"])
        ids, labs = torch.from_numpy(ids).to(dev), torch.from_numpy(labs).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp is not None):
            logits = model(input_ids=ids).logits
        pk_w = passkey_weight_at(step, tc)
        loss = weighted_ce(logits, labs, pk_w)
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
            wandb.log({"train/loss": last, "train/lr": opt.param_groups[0]["lr"], "train/grad_norm": float(gnorm), "train/passkey_weight": pk_w, "train/active_max_n": max(active_train_ns(step, tc)),
                       "train/steps_per_s": step / (time.time() - t0)}, step=step)
            loss_acc.zero_()
            n_acc = 0
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            logs = {}
            for n, p in val.items():
                m = evaluate_with_backoff(model, p, dev, PERIODIC_TOKENS_PER_BATCH, amp)
                logs.update({f"val/n{n}/{k}": m[k] for k in METRICS})
            wandb.log(logs, step=step)
            history.append({"step": step, "train_loss": last, **logs})  # kept in the result file: the curves do not depend on wandb or on the logs
            print(f"[{name}] step {step} loss {last:.4f} | " + " ".join(
                f"n{n}: pk {logs[f'val/n{n}/passkey_acc']:.2f} v {logs[f'val/n{n}/ar_valid_acc']:.2f} "
                f"e {logs[f'val/n{n}/ar_expired_acc']:.2f}" for n in (32, 256, 1024, 4096, 16384) if n in val), flush=True)
            save_resume(resume_file, {"config": cfg, "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                                      "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                                      "loss_acc": float(loss_acc), "n_acc": n_acc, "last_loss": last, "train_time_s": time.time() - t0,
                                      "wandb_id": run.id, "resumed_at_steps": resumed_at, "history": history})
    train_time = time.time() - t0

    ckpt = C.ROOT / "runs" / name / "model.pt"  # saved before the (long) final evaluation so a failure there cannot lose the model
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)

    rows = []
    for split in ec["splits"]:
        for n in ec["ns"]:
            m = evaluate_with_backoff(model, D.load_packed(data_dir / f"n{n}" / f"{split}.parquet", kinds=True), dev, ec["tokens_per_batch"], amp)
            rows.append({"split": split, "N": n, **m})
            run.summary.update({f"final/{split}/n{n}/{k}": m[k] for k in METRICS})
        print(f"[{name}] {split}: " + " ".join(f"n{r['N']}:{r['passkey_acc']:.2f}/{r['ar_expired_acc']:.2f}" for r in rows if r["split"] == split), flush=True)

    write_json_atomic(out_json, C.stamp_diff({"name": name, "config": cfg, "explicit_keys": explicit, "n_params": n_params,
                                              "final_train_loss": last, "train_time_s": train_time, "resumed_at_steps": resumed_at, "git": git, "val_history": history, "rows": rows}))
    resume_file.unlink(missing_ok=True)  # the result file is the record now
    run.finish()
    print(f"[{name}] done in {time.time() - t0:.0f}s -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
