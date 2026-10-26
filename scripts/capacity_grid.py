#!/usr/bin/env python
"""Passkey capacity of the two-pass layer as a function of the segment size and the window: a heatmap over segment {8, 16, 32} x window {32, 64, 128} (segment <= window: 9 cells).

    python scripts/capacity_grid.py plan                                   # every job, what is done, the capacity found so far
    python scripts/capacity_grid.py run --slots local:1:5                  # run every unfinished job (resumable: finished runs are read from their result files)
    python scripts/capacity_grid.py job --segment 8 --window 32 --seed 0 --gpu 1

One job = one cell and one base seed g:
  1. the base: a two-pass model (pass_output 1) of that segment and window, trained from scratch on 4-token passkeys with the config file's recipe (50k steps, stream-length
     curriculum, seed g), evaluated up to N = 256. The base of (segment 32, window 64) is the E4 run swa_2pass_seg32_po1_s<g> (same config), it is reused;
  2. the capacity chain of scripts/passkey_capacity.py from the base's weights: P = 8, 16, 32, 64 tokens, each stage fine-tuned 50k steps (lr 1e-3, warm-up 500, curriculum 2000, the
     recipe of the earlier two-pass chains, fine-tuning seed g + 10) on N = 32 ... 256, stopping at the first stage whose validation exact match at N = 256 is below 0.9.
The earlier six chains of (32, 64) all started from ONE base (the best of the six E4 seeds); here every seed has its own base, so this cell is a different measurement.

Remote hosts: `--slots local:1:5 10.127.105.38:0:4 10.127.105.38:1:4` = host:gpu:jobs at once. A job on a remote host runs in a tmux session there (it survives a dropped connection;
`run` finds the session again after a restart), the driver polls it, then copies the job's raw results and weights (runs/<name>/model.pt, 2 MB) back and merges its chains into the local
results/capacity_grid.json. The remote hosts need the repo folder (`sync-remote` checks out HEAD from GitHub on it (push first) and copies the datasets), the conda env under REMOTE_PY (a copy of this machine's, the fla and
transformers versions of the hosts' own envs differ) and the git-ignored wandb_token. The cell (32, 64) is local only: its bases are the E4 runs, which exist on this machine only.

Results: results/capacity_grid.json (key stages_swa_2pass_grid_s<segment>_w<window>_b<g>, one list of stage entries per chain), raw runs in results/raw/, weights in runs/ (local),
logs in logs/capacity/. The page: `python src/report.py capgrid --out reports/capacity_grid_report.html`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))
import config as C  # noqa: E402
import capacity_grid as G  # noqa: E402

_spec = importlib.util.spec_from_file_location("passkey_capacity", ROOT / "scripts" / "passkey_capacity.py")
PC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PC)

MODEL = "swa_2pass"

REMOTE_ROOT = "/mnt/data/users/ivan.rodkin/lab/memory_amplifier"
REMOTE_PY = "/mnt/data/users/ivan.rodkin/lab/envs/py311_pt2.6_cu12.4_fla052/bin/python"   # a copy of this machine's py311_pt2.6_cu12.4 (not the hosts' own env of that name)
GITHUB = "https://github.com/RodkinIvan/memory_amplifier.git"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519_claude"
SSH_OPTS = ["-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30"]
DATA_DIRS = ["passkey_swar", "passkey_swar_pk8", "passkey_swar_pk16", "passkey_swar_pk32"]   # the datasets a chain needs up to P = 32 (P = 64 is generated on the host that first needs it)
POLL_S = 120


def ensure_base(segment: int, window: int, seed: int, gpu: int) -> str:
    """Train the base of a cell when its result file does not exist; stop when an existing result is not the result of this config. Returns the path of its weights."""
    overrides = G.base_overrides(segment, window, seed)
    cfg, explicit = C.load_config(G.CONFIG, overrides)
    name = C.run_name(cfg, explicit)
    result_file = ROOT / "results" / "raw" / f"{name}.json"
    if result_file.exists():
        diff = {k: v for k, v in C.config_difference(json.loads(result_file.read_text())["config"], cfg).items() if not C.is_ignored(k)}  # (evaluation settings do not matter: the E4 bases were evaluated further)
        if diff:
            raise SystemExit(f"{name}: results/raw/{name}.json was trained with a different config; differs in {diff}")
    else:
        need = C.footprint_gb(cfg)
        PC.wait_for_memory(gpu, need)
        print(f"base {name}", flush=True)
        PC.run_training(overrides, name, gpu, need, ROOT / "logs" / "capacity" / f"{name}.log", config=str(G.CONFIG.relative_to(ROOT)))
    weights = ROOT / "runs" / name / "model.pt"
    if not weights.exists():
        raise SystemExit(f"{name}: {weights} is missing (the result exists but the weights do not)")
    return str(weights.relative_to(ROOT))


def local_only(segment: int, window: int) -> bool:
    """The (32, 64) cell uses the E4 runs as bases (swa_2pass_seg32_po1_s<g>), which only this machine has."""
    return (segment, window) == (32, G.yaml_window())


def pipeline(segment: int, window: int, seed: int, gpu: int) -> list[dict]:
    """The base, then the chain; returns the chain's stage entries. Idempotent: a finished base or stage is read from its result file."""
    PC.OUT = G.RESULTS
    if local_only(segment, window) and not (ROOT / "results" / "raw" / f"{G.base_name(segment, window, seed)}.json").exists():
        raise SystemExit(f"{G.base_name(segment, window, seed)}: the base of this cell is an E4 run and its result file is missing here; it is not trained again under the same name")
    init = ensure_base(segment, window, seed, gpu)
    win = None if window == G.yaml_window() else window   # (same rule as the base: no override for the file's window, so stage names of the (32, 64) cell stay short)
    run_stage = PC.stage_runner(MODEL, G.RECIPE["steps"], G.RECIPE["lr"], G.FT_SEED_OFFSET + seed, G.RECIPE["warmup"], G.RECIPE["curriculum"], segment, win, gpu)
    stages = PC.chain(run_stage, G.FIRST_P, G.MAX_LEN, init, key=G.chain_key(segment, window, seed))
    return stages


def ssh(addr: str, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, f"ivan.rodkin@{addr}", command], capture_output=True, text=True, timeout=timeout)


def rsync(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(["rsync", "-a", "-e", "ssh " + " ".join(shlex.quote(x) for x in SSH_OPTS), *args], capture_output=True, text=True, timeout=timeout)


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def sync_code(addr: str, commit: str | None = None) -> str:
    """Put a commit (default: this checkout's HEAD) on the remote host as a git checkout fetched from GitHub (the public repo, over https), so the host's results carry the commit like
    the local ones. Stops when the commit is not on GitHub yet (push first) or, for HEAD, when code files have uncommitted changes (the host would run something else than this
    machine). `commit` names an older pushed commit, e.g. while HEAD only adds to this driver's own dispatch code. Returns the commit."""
    if commit is None and git("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "configs", "data"):
        raise SystemExit("src / scripts / configs / data have uncommitted changes: commit and push them first (the remote hosts run the commit, and their results record it)")
    head, branch = git("rev-parse", commit or "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    r = ssh(addr, f"cd {REMOTE_ROOT} && {{ [ -d .git ] || git init -q; }} && git fetch -q {GITHUB} {branch} && git checkout -q -f --detach {head} && git rev-parse HEAD", timeout=300)
    if r.returncode or r.stdout.strip() != head:
        raise SystemExit(f"{addr}: could not check out {head[:10]} from {GITHUB} (is it pushed?): {r.stderr.strip() or r.stdout.strip()}")
    return head


def sync_remote(addr: str, data: bool = True, commit: str | None = None) -> None:
    """The code (git checkout of HEAD), the datasets and the wandb token on a remote host; idempotent (rsync sends what changed)."""
    r = ssh(addr, f"mkdir -p {REMOTE_ROOT}/results/raw {REMOTE_ROOT}/runs {REMOTE_ROOT}/logs/capacity {REMOTE_ROOT}/data_cache {REMOTE_ROOT}/checkpoints")
    if r.returncode:
        raise SystemExit(f"{addr}: cannot prepare {REMOTE_ROOT}: {r.stderr.strip()}")
    sync_code(addr, commit)
    dest = f"ivan.rodkin@{addr}:{REMOTE_ROOT}/"
    if data:
        for d in DATA_DIRS:
            r = rsync([str(ROOT / "data_cache" / d), f"ivan.rodkin@{addr}:{REMOTE_ROOT}/data_cache/"])
            if r.returncode:
                raise SystemExit(f"{addr}: rsync of data_cache/{d} failed: {r.stderr.strip()}")
    if (ROOT / "wandb_token").exists():   # the user approved copying it to these hosts (2026-10-26); mode 600, never printed
        r = rsync(["--chmod=F600", str(ROOT / "wandb_token"), dest])
        if r.returncode:
            raise SystemExit(f"{addr}: rsync of the wandb token failed (not printed)")


def fetch_remote_results(addr: str) -> None:
    """Copy the host's raw results and the weights of its runs back and merge its chains into the local capacity_grid.json."""
    src = f"ivan.rodkin@{addr}:{REMOTE_ROOT}"
    for args in ([f"{src}/results/raw/", str(ROOT / "results" / "raw") + "/"],
                 ["--include=*/", "--include=model.pt", "--exclude=*", f"{src}/runs/", str(ROOT / "runs") + "/"],
                 ["--ignore-existing", f"{src}/logs/capacity/", str(ROOT / "logs" / "capacity") + "/"]):
        r = rsync(args)
        if r.returncode:
            raise RuntimeError(f"{addr}: fetching {args[-2]} failed: {r.stderr.strip()}")
    r = ssh(addr, f"cat {REMOTE_ROOT}/results/capacity_grid.json")
    if r.returncode:
        raise RuntimeError(f"{addr}: cannot read its capacity_grid.json: {r.stderr.strip()}")
    remote = json.loads(r.stdout)
    PC.OUT = G.RESULTS
    for key, stages in remote.items():
        if key.startswith("stages_"):
            local = G.load_results().get(key, [])
            if len(stages) >= len(local):
                PC.write_capacity({key: stages})


def remote_job(addr: str, gpu: int, segment: int, window: int, seed: int) -> None:
    """One job on a remote host: a tmux session there runs `capacity_grid.py job`; poll for its exit file, then fetch the results. Re-attaches to a session that is already running."""
    name = G.tag(segment, window, seed)
    exit_file, log = f"logs/capacity/{name}.exit", f"logs/capacity/{name}.driver.log"
    if ssh(addr, f"tmux has-session -t {name} 2>/dev/null").returncode != 0:
        cmd = (f"cd {REMOTE_ROOT} && rm -f {exit_file} && {REMOTE_PY} scripts/capacity_grid.py job --segment {segment} --window {window} --seed {seed} --gpu {gpu} >> {log} 2>&1; echo $? > {exit_file}")
        r = ssh(addr, f"tmux new-session -d -s {name} {shlex.quote('bash -c ' + shlex.quote(cmd))}")
        if r.returncode:
            raise SystemExit(f"{addr}: cannot start {name}: {r.stderr.strip()}")
    misses = 0
    while True:
        time.sleep(POLL_S)
        try:
            r = ssh(addr, f"cat {REMOTE_ROOT}/{exit_file} 2>/dev/null; true")
        except subprocess.TimeoutExpired:
            r = None
        if r is None or r.returncode:
            misses += 1
            if misses > 60:   # two hours without an answer
                raise SystemExit(f"{addr}: no answer for {misses} polls ({name})")
            continue
        misses = 0
        if r.stdout.strip():
            break
    for attempt in range(5):   # the copy back can fail on a network blip; the remote files stay
        try:
            fetch_remote_results(addr)
            break
        except RuntimeError as e:
            print(f"fetch from {addr} failed ({e}), retry {attempt + 1}", flush=True)
            time.sleep(60)
    else:
        raise SystemExit(f"{addr}: could not fetch the results of {name}")
    if r.stdout.strip() != "0":
        raise SystemExit(f"{addr}: job {name} exited with {r.stdout.strip()} (see {REMOTE_ROOT}/{log})")


def fmt_state(s: dict) -> str:
    cap = s["capacity"]
    if s["base_exact"] is None:
        return "base not trained"
    stages = ", ".join(f"P={e['passkey_len']}: {e['validation']['exact']['256']:.2f}" for e in s["stages"]) or "no stage yet"
    if cap["done"]:
        return f"base {s['base_exact']:.2f} | {stages} | DONE capacity {'>=' if cap['censored'] else ''}{cap['tokens']}"
    return f"base {s['base_exact']:.2f} | {stages} | running (capacity so far {cap['tokens']})"


def plan(seeds) -> None:
    grid = G.load_results()
    for (s, w, g) in G.jobs(seeds):
        print(f"segment {s:>2} window {w:>3} seed {g}  {G.base_name(s, w, g)}\n    {fmt_state(G.job_state(s, w, g, grid))}")


class Pool:
    """The jobs still to start. A remote slot does not take the local-only cell, a local slot takes it first (nobody else can, and it is the cheap one); a job is given out once."""

    def __init__(self, jobs):
        self.jobs, self.lock = list(jobs), threading.Lock()

    def take(self, remote: bool):
        with self.lock:
            order = [i for i, (s, w, g) in enumerate(self.jobs) if not local_only(s, w)] if remote else \
                    sorted(range(len(self.jobs)), key=lambda i: not local_only(*self.jobs[i][:2]))
            return self.jobs.pop(order[0]) if order else None


def run_jobs(job_list, slots: list[tuple[str, int]], stagger: int) -> list[tuple]:
    """Run the jobs on `slots` (host, gpu): "local" or the address of a remote host; a GPU listed n times runs n jobs at once. The first starts of the slots are `stagger` s apart so the
    memory of one job is allocated before the next one looks at the free memory."""
    pool = Pool(job_list)
    failures: list[tuple] = []

    def worker(i: int, host: str, gpu: int) -> None:
        time.sleep(i * stagger)
        while (job := pool.take(host != "local")) is not None:
            s, w, g = job
            t0 = time.time()
            print(f"[{time.strftime('%d %b %H:%M')}] {host} GPU {gpu}: start segment {s} window {w} seed {g}", flush=True)
            try:
                pipeline(s, w, g, gpu) if host == "local" else remote_job(host, gpu, s, w, g)
            except SystemExit as e:   # the helpers stop with a message (config mismatch, a failed run); the other jobs go on
                failures.append(((s, w, g), str(e)))
                print(f"FAILED segment {s} window {w} seed {g} on {host}: {e}", flush=True)
            except Exception:
                failures.append(((s, w, g), traceback.format_exc()))
                print(f"FAILED segment {s} window {w} seed {g} on {host}:\n{traceback.format_exc()}", flush=True)
            else:
                print(f"[{time.strftime('%d %b %H:%M')}] {host} GPU {gpu}: done segment {s} window {w} seed {g} in {(time.time() - t0) / 3600:.1f} h", flush=True)

    threads = [threading.Thread(target=worker, args=(i, host, gpu), daemon=True) for i, (host, gpu) in enumerate(slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failures


def parse_slots(specs: list[str]) -> list[tuple[str, int]]:
    """`host:gpu:n` -> n slots on that GPU ("local" = this machine), interleaved round-robin over the GPUs: the jobs are handed out in the order of the slots, so the heavy ones of the
    first jobs land on different GPUs instead of piling up on the first."""
    lists = []
    for spec in specs:
        host, gpu, n = spec.split(":")
        lists.append([(host, int(gpu))] * int(n))
    return [lst[i] for i in range(max(map(len, lists), default=0)) for lst in lists if i < len(lst)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run", "job", "sync-remote"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(G.SEEDS), help="base seeds to run (default: all)")
    ap.add_argument("--slots", nargs="*", default=["local:1:5"], metavar="HOST:GPU:N", help="where jobs run (run): N at once on GPU `GPU` of HOST (local, or the address of a remote host); a job needs ~4 GB, config.footprint_gb says 7")
    ap.add_argument("--hosts", nargs="*", default=[], help="addresses of the remote hosts (sync-remote)")
    ap.add_argument("--commit", default=None, help="commit to check out on the remote hosts (default: HEAD, which must be pushed and have no uncommitted code changes)")
    ap.add_argument("--stagger", type=int, default=60, help="seconds between the first starts of the slots (run)")
    ap.add_argument("--segment", type=int)
    ap.add_argument("--window", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.command == "plan":
        return plan(a.seeds)
    if a.command == "job":
        if (a.segment, a.window) not in G.cells():
            ap.error(f"(segment, window) = ({a.segment}, {a.window}) is not a cell of the grid {G.cells()}")
        pipeline(a.segment, a.window, a.seed, a.gpu)
        return
    if a.command == "sync-remote":
        for addr in a.hosts:
            sync_remote(addr, commit=a.commit)
            print(f"{addr}: synced", flush=True)
        return
    slots = parse_slots(a.slots)
    for addr in sorted({h for h, _ in slots if h != "local"}):
        sync_remote(addr, data=False, commit=a.commit)   # the code is current; the datasets were copied by `sync-remote` (a missing one is an error on the host, not copied by every run)
    failures = run_jobs(G.jobs(a.seeds), slots, a.stagger)
    if failures:
        print(f"{len(failures)} job(s) failed:", *[f"  {j}: {m.splitlines()[0] if m else ''}" for j, m in failures], sep="\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
