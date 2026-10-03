#!/usr/bin/env python
"""Launch a sweep of train.py runs in tmux: one window per worker, several workers per GPU, resumable.

    python src/launch.py --config configs/exp1.yaml --session exp1 --gpus 0 1 --procs-per-gpu 3 \\
        --sweep model.alpha=1.0,0.999,0.995,0.99,0.98,0.95,0.9,0.8 seed=0,1,2

The first --sweep key varies fastest (so seed 0 of every alpha finishes first). Runs whose results/raw/<name>.json
already exists are skipped. Logs: logs/<session>/<run name>.log ; failures are listed in logs/<session>/failed.txt.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import data as D  # noqa: E402


def build_jobs(config: str, base: list[str], sweep: list[str]) -> list[dict]:
    axes = []
    for s in sweep:
        key, _, vals = s.partition("=")
        axes.append([f"{key}={v}" for v in vals.split(",")])
    jobs = []
    # first sweep key varies fastest: reverse the axes for itertools.product, then restore the order
    for combo in itertools.product(*reversed(axes)):
        overrides = [*base, *reversed(combo)]
        cfg, explicit = C.load_config(config, overrides)
        jobs.append({"name": C.run_name(cfg, explicit), "overrides": overrides, "config": cfg})
    return jobs


def worker_header(gpu: int) -> list[str]:
    """First lines of a worker script. expandable_segments lets the CUDA allocator return and reuse memory in pieces, which keeps
    many processes on one GPU from each holding on to a large cache."""
    return ["#!/usr/bin/env bash", f"cd {shlex.quote(str(C.ROOT))}", f"export CUDA_VISIBLE_DEVICES={gpu}",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            # the token is read by the shell here and never written to a log or a command line
            '[ -f wandb_token ] && export WANDB_API_KEY="$(cat wandb_token)"']


def is_done(job: dict) -> bool:
    f = C.ROOT / "results" / "raw" / f"{job['name']}.json"
    return f.exists() and json.loads(f.read_text())["config"] == job["config"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/exp1.yaml")
    ap.add_argument("--session", required=True, help="tmux session name (also the logs/ subfolder)")
    ap.add_argument("--sweep", nargs="*", default=[], metavar="KEY=V1,V2,..")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every run")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--procs-per-gpu", type=int, default=2)
    ap.add_argument("--no-wandb", action="store_true", help="pass --no-wandb to every train.py call (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.config, args.set, args.sweep)
    todo = [j for j in jobs if not is_done(j)]
    print(f"{len(jobs)} runs in the sweep, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    for j in todo:
        print("  ", j["name"])
    if args.dry_run or not todo:
        return

    cfg0 = jobs[0]["config"]
    data_dir = Path(cfg0["data_dir"])
    data_dir = data_dir if data_dir.is_absolute() else C.ROOT / data_dir
    D.prepare_kinds(data_dir, json.loads((data_dir / "gen_meta.json").read_text())["params"], workers=32)

    workers = [(g, p) for g in args.gpus for p in range(args.procs_per_gpu)]
    logdir = C.ROOT / "logs" / args.session
    logdir.mkdir(parents=True, exist_ok=True)
    scripts = []
    for w, (gpu, _) in enumerate(workers):
        mine = todo[w::len(workers)]
        if not mine:
            continue
        lines = worker_header(gpu)
        for j in mine:
            cmd = [sys.executable, "src/train.py", "--config", args.config, *(["--no-wandb"] if args.no_wandb else []), "--set", *j["overrides"]]
            log = logdir / f"{j['name']}.log"
            lines.append(f"{shlex.join(cmd)} > {shlex.quote(str(log))} 2>&1 || echo FAILED {j['name']} >> {shlex.quote(str(logdir / 'failed.txt'))}")
        lines.append(f'echo "worker {w} (gpu {gpu}) finished"')
        script = logdir / f"worker{w}.sh"
        script.write_text("\n".join(lines) + "\n")
        scripts.append(script)

    if subprocess.run(["tmux", "has-session", "-t", args.session], capture_output=True).returncode == 0:
        sys.exit(f"tmux session {args.session!r} already exists (tmux attach -t {args.session}); pick another --session")
    for i, script in enumerate(scripts):
        cmd = f"bash {shlex.quote(str(script))}"
        if i == 0:
            subprocess.run(["tmux", "new-session", "-d", "-s", args.session, "-n", f"w{i}", cmd], check=True)
            subprocess.run(["tmux", "set-option", "-t", args.session, "remain-on-exit", "on"], check=True)
        else:
            subprocess.run(["tmux", "new-window", "-t", args.session, "-n", f"w{i}", cmd], check=True)
    print(f"started {len(scripts)} workers in tmux session {args.session!r}: tmux attach -t {args.session}")
    print(f"logs: {logdir}")


if __name__ == "__main__":
    main()
