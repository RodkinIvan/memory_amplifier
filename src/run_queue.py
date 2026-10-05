#!/usr/bin/env python
"""Run a queue of sweeps unattended: start the next one whenever the GPUs have free worker slots.

    tmux new -d -s fbqueue 'python src/run_queue.py --queue configs/queue.txt --watch fbsweep'

Each queue line is `<tmux session> | <arguments of src/launch.py>`; `#` starts a comment. The file is re-read on every
poll, so lines can be appended while the driver runs. A line is started once (the session name is the key, recorded in
logs/queue/launched.txt) when at most `--start-below` workers are still alive in the watched + already launched
sessions and every GPU the stage uses has free memory for all of the stage's workers (each worker's measured footprint, `config.footprint_gb`, e.g. 8 GB at segment 8 and
5 GB at segment 32 / 64; at least `--min-free-gb` in all). A stage with an 8 GB footprint gets at most 2 workers per GPU (`MAX_BIG_WORKERS`): the GPU is shared with other
people's jobs. launch.py skips runs whose result already exists, so restarting the driver is safe.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_queue(text: str) -> list[tuple[str, list[str]]]:
    """[(session, launch.py arguments)] in file order; blank lines and `#` comments are ignored."""
    stages = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        session, sep, args = line.partition("|")
        if not sep or not session.strip():
            raise ValueError(f"queue line needs '<session> | <launch.py args>': {raw!r}")
        stages.append((session.strip(), shlex.split(args)))
    sessions = [s for s, _ in stages]
    if len(set(sessions)) != len(sessions):
        raise ValueError("duplicate session names in the queue")
    return stages


def stage_gpus(args: list[str]) -> list[int]:
    """GPU ids a stage will use: the values after `--gpus` in its launch.py arguments (launch.py defaults to GPUs 0 and 1)."""
    if "--gpus" not in args:
        return [0, 1]
    out = []
    for a in args[args.index("--gpus") + 1:]:
        if a.startswith("--"):
            break
        out.append(int(a))
    return out


def stage_procs_per_gpu(args: list[str]) -> int:
    """Workers per GPU of a stage: the value after `--procs-per-gpu` in its launch.py arguments (launch.py defaults to 2)."""
    return int(args[args.index("--procs-per-gpu") + 1]) if "--procs-per-gpu" in args else 2


MAX_BIG_WORKERS = 2  # workers per GPU for stages whose footprint is BIG_GB or more
BIG_GB = 8.0


def stage_footprint_gb(args: list[str]) -> float | None:
    """The largest per-worker footprint (GB) among the runs of a stage, from the configs the sweep expands to; None if the arguments cannot be expanded."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import config as C
        import launch as L
        a = L.make_parser().parse_args(["--session", "x", *args])
        return max(C.footprint_gb(j["config"]) for j in L.build_jobs(str(ROOT / a.config), a.set, a.sweep))
    except Exception:  # noqa: BLE001 - an unreadable stage falls back to the --gb-per-worker value
        return None


def limit_workers(args: list[str], footprint: float | None) -> list[str]:
    """Stages with a big footprint run at most MAX_BIG_WORKERS workers per GPU; returns the (possibly rewritten) launch arguments."""
    if footprint is None or footprint < BIG_GB or stage_procs_per_gpu(args) <= MAX_BIG_WORKERS:
        return args
    out = list(args)
    out[out.index("--procs-per-gpu") + 1] = str(MAX_BIG_WORKERS)
    return out


def memory_needed_gb(args: list[str], gb_per_worker: float | None, min_free_gb: float) -> float:
    """Free memory a stage needs on each of its GPUs: the measured footprint of one worker (or `gb_per_worker`, the fallback / override) times the workers per GPU."""
    foot = stage_footprint_gb(args)
    per_worker = gb_per_worker if gb_per_worker is not None else (foot if foot is not None else 8.0)
    return max(min_free_gb, per_worker * stage_procs_per_gpu(limit_workers(args, foot)))


def free_gb(gpu: int) -> float | None:
    """Free memory of a GPU in GiB, None if nvidia-smi does not answer."""
    r = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return float(r.stdout.strip()) / 1024 if r.returncode == 0 else None
    except ValueError:
        return None


def can_start(live: int, start_below: int, free: dict[int, float | None], min_free_gb: float) -> bool:
    """Few enough workers alive, and enough free memory on every GPU of the stage (an unknown reading does not block)."""
    return live <= start_below and all(f is None or f >= min_free_gb for f in free.values())


def live_workers(session: str) -> int:
    """Worker windows of a tmux session whose script is still running (finished ones stay as dead panes)."""
    r = subprocess.run(["tmux", "list-panes", "-s", "-t", session, "-F", "#{pane_dead}"], capture_output=True, text=True)
    return sum(1 for x in r.stdout.split() if x == "0") if r.returncode == 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="configs/queue.txt")
    ap.add_argument("--watch", nargs="*", default=[], help="already running tmux sessions whose workers count as busy")
    ap.add_argument("--start-below", type=int, default=6, help="start the next stage when at most this many workers are alive")
    ap.add_argument("--min-free-gb", type=float, default=12.0, help="start a stage only when each of its GPUs has at least this much free memory")
    ap.add_argument("--gb-per-worker", type=float, default=None, help="override the measured per-worker footprint (config.footprint_gb) with one value for every stage")
    ap.add_argument("--poll", type=int, default=60, help="seconds between checks")
    ap.add_argument("--dry-run", action="store_true", help="print the stages and exit")
    args = ap.parse_args()

    qfile = ROOT / args.queue
    state = ROOT / "logs" / "queue" / "launched.txt"
    state.parent.mkdir(parents=True, exist_ok=True)
    stamp = lambda: time.strftime("%F %T")  # noqa: E731
    if args.dry_run:
        for session, a in parse_queue(qfile.read_text()):
            foot = stage_footprint_gb(a)
            print(f"{session}: footprint {foot} GB per worker, needs {memory_needed_gb(a, args.gb_per_worker, args.min_free_gb):.1f} GB per GPU: launch.py {shlex.join(limit_workers(a, foot))}")
        return

    while True:
        launched = set(state.read_text().split()) if state.exists() else set()
        pending = [(s, a) for s, a in parse_queue(qfile.read_text()) if s not in launched]
        live = sum(live_workers(s) for s in [*args.watch, *launched])
        if not pending and live == 0:
            print(f"{stamp()} queue empty and nothing running: done", flush=True)
            return
        if pending and can_start(live, args.start_below, {g: free_gb(g) for g in stage_gpus(pending[0][1])},
                                 memory_needed_gb(pending[0][1], args.gb_per_worker, args.min_free_gb)):
            session, a = pending[0]
            a = limit_workers(a, stage_footprint_gb(a))
            print(f"{stamp()} {live} workers alive -> starting {session}", flush=True)
            r = subprocess.run([sys.executable, str(ROOT / "src" / "launch.py"), "--session", session, *a], cwd=ROOT)
            if r.returncode:  # do not retry forever; the log shows why
                print(f"{stamp()} launch of {session} failed with code {r.returncode}", flush=True)
            with state.open("a") as f:
                f.write(session + "\n")
            time.sleep(120)  # let its workers start before counting again
            continue
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
