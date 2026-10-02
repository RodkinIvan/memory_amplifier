#!/usr/bin/env python
"""Let only a few runs at a time do their final evaluation on a shared GPU (a stop-gap for runs started before train.py got its
out-of-memory retry).

The final evaluation up to N = 16,384 needs about 6 GiB more than training; when several runs reach it together on a full GPU they
run out of memory and the trained model is lost. This script finds the train.py processes started before `--before`, and when more
than `--limit` of them per GPU are in the final evaluation (their log shows the last training step) it pauses the extra ones with
SIGSTOP and resumes them, one by one, as the running ones finish. Paused processes keep their GPU memory but do no work. Every paused
process is resumed when the script exits, also on SIGTERM / SIGHUP / Ctrl-C.

    python src/eval_gate.py --gpus 0 --limit 1 --before "2026-10-02 17:36"
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def decide(procs: list[dict], limit: int) -> tuple[list[int], list[int]]:
    """procs: [{pid, in_eval, stopped}] of one GPU in priority order. Returns (pids to pause, pids to resume) so that at most
    `limit` processes are in their final evaluation and not paused. Running evaluations are never interrupted in favour of others."""
    active = [p for p in procs if p["in_eval"] and not p["stopped"]]
    waiting = [p for p in procs if p["in_eval"] and p["stopped"]]
    pause, resume = [], []
    room = limit - len(active)
    if room < 0:  # too many running: pause the lowest-priority ones, keep the earlier ones
        pause = [p["pid"] for p in active[limit:]]
        room = 0
    resume = [p["pid"] for p in waiting[:room]]
    return pause, resume


def run_name(cmdline: list[str]) -> str | None:
    """Run name of a main-grid sweep process from its `--set model.nonlin=.. model.segment=.. seed=..` overrides."""
    if "--set" not in cmdline:
        return None
    kv = dict(x.split("=", 1) for x in cmdline[cmdline.index("--set") + 1:] if "=" in x)
    try:
        return f"swa_fb_nl{kv['model.nonlin']}_seg{kv['model.segment']}_s{kv['seed']}"
    except KeyError:
        return None


def proc_info(pid: int) -> dict | None:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        env = dict(x.split(b"=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in x)
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        started = os.stat(f"/proc/{pid}").st_ctime
    except (OSError, IndexError):
        return None
    cmd = [c.decode(errors="replace") for c in cmd]
    if not any(c.endswith("src/train.py") for c in cmd):
        return None
    return {"pid": pid, "cmd": cmd, "gpu": env.get(b"CUDA_VISIBLE_DEVICES", b"").decode(), "started": started, "stopped": state in ("T", "t")}


def in_final_eval(name: str | None, last_step: int) -> bool:
    """The run printed its evaluation of the last training step, so what follows is the final evaluation."""
    if not name:
        return False
    logs = sorted(glob.glob(str(ROOT / "logs" / "*" / f"{name}.log")), key=os.path.getmtime)
    if not logs:
        return False
    return f"] step {last_step} " in Path(logs[-1]).read_text(errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", nargs="+", default=["0"], help="GPUs to guard (CUDA_VISIBLE_DEVICES values of the runs)")
    ap.add_argument("--limit", type=int, default=1, help="runs per GPU allowed in the final evaluation at the same time")
    ap.add_argument("--before", required=True, help='only runs started before this time, "YYYY-MM-DD HH:MM"')
    ap.add_argument("--last-step", type=int, default=20000)
    ap.add_argument("--poll", type=float, default=2.0)
    a = ap.parse_args()
    cutoff = time.mktime(time.strptime(a.before, "%Y-%m-%d %H:%M"))
    paused: set[int] = set()

    def release(*_):
        for pid in list(paused):
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        print(f"{time.strftime('%T')} resumed {len(paused)} paused runs, exiting", flush=True)
        paused.clear()
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, release)
    try:
        while True:
            procs = [i for i in (proc_info(int(p)) for p in os.listdir("/proc") if p.isdigit()) if i and i["started"] < cutoff and i["gpu"] in a.gpus]
            if not procs:
                print(f"{time.strftime('%T')} no guarded runs left", flush=True)
                break
            for gpu in a.gpus:
                mine = sorted((p for p in procs if p["gpu"] == gpu), key=lambda p: p["pid"])
                for p in mine:
                    p["in_eval"] = in_final_eval(run_name(p["cmd"]), a.last_step)
                pause, resume = decide(mine, a.limit)
                for pid in pause:
                    os.kill(pid, signal.SIGSTOP)
                    paused.add(pid)
                    print(f"{time.strftime('%T')} gpu {gpu}: paused {pid} (another final evaluation is running)", flush=True)
                for pid in resume:
                    os.kill(pid, signal.SIGCONT)
                    paused.discard(pid)
                    print(f"{time.strftime('%T')} gpu {gpu}: resumed {pid}", flush=True)
            time.sleep(a.poll)
    finally:
        for pid in list(paused):
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass


if __name__ == "__main__":
    main()
