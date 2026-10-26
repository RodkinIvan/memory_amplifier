#!/usr/bin/env python3
"""Route experiments to the hosts of a hosts file: prepare each host (conda env, code), start jobs where a GPU is free, fetch the results when they are done.

    python cluster/route.py hosts                                    # which hosts answer, their GPUs (free memory), conda and the env
    python cluster/route.py setup jobs.json                          # only prepare the hosts (env + code), start nothing
    python cluster/route.py run   jobs.json                          # prepare, run every job that is not done, fetch results (resumable)
    python cluster/route.py run   jobs.json --dry-run                # where the jobs would go, nothing started
    python cluster/route.py status jobs.json                         # what the state file says
    python cluster/route.py kill  jobs.json [job ...]                # stop the sessions of running jobs

Works from any machine that has ssh access to the hosts (needs ssh, rsync and tmux here and on the hosts; Python >= 3.8, standard library only).

hosts file (default cluster/hosts, `--hosts-file`): ssh-config syntax, used as `ssh -F <file>`, so aliases and keys work without a ~/.ssh/config:

    Host *
        IdentityFile ~/.ssh/id_ed25519
        IdentitiesOnly yes
    Host nlp3
        HostName 192.0.2.14
        User youruser
        #router: root=/path/to/lab per-gpu=4

`#router:` lines (key=value ...) belong to the Host block above them, or to every host when they come before the first Host. Keys: root (base folder on the host, the project goes in
<root>/<project>; default $HOME/router), per-gpu (jobs per GPU, default 4), gpus (comma list of the GPU indices the router may use, default all), skip=1 (leave the host out),
conda (path to the conda executable, default: found on the PATH and in the usual places), local=1 (this machine: no ssh; a host whose address is one of this machine's is local too).

jobs file (JSON):

    {"project": "memory_amplifier",                   # folder name under the host's root
     "env": {"file": "cluster/env.yml"},              # conda env spec (relative to the project dir here); its `name:` is the env; <stem>_post.sh next to it runs after creation
     "sync": ["src", "scripts", "configs", "data"],   # copied to the hosts with rsync (relative to the project dir here)
     "exclude": ["__pycache__", "*.pyc"],
     "secrets": ["wandb_token"],                      # copied with mode 600, only what is listed here
     "fetch": ["results/raw/{name}*.json"],           # copied back when a job is done (rsync paths relative to the project dir, may have *); {name} = the job
     "jobs": [{"name": "a", "cmd": "{python} scripts/x.py --gpu {gpu}", "gpus": 1, "gb": 7,
               "expect": ["results/raw/a.json"], "hosts": ["nlp3"], "retries": 1}]}

In `cmd`, `fetch` and `expect`: {python} = the env's python on the host, {gpu} = the GPU index(es) given to the job (comma-separated), {name}, {host}, {project_dir} (on the host).
A job starts with CUDA_VISIBLE_DEVICES = its GPUs, in a tmux session on the host (it survives a lost connection; `run` finds it again after a restart); its output goes to
logs/router/<name>.log in the project dir on the host and is fetched with the results. A GPU takes a job when it has gb + headroom GB free and fewer than per-gpu router jobs;
memory other people use is respected because the free memory is read from nvidia-smi on every poll. `expect` lists local files that must exist after the fetch, else the job counts as failed.
State (which job went where, done / failed) is kept in <jobs file>.state.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_HOSTS = HERE / "hosts"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", "-o", "StrictHostKeyChecking=accept-new"]
ENV_MARKER = ".router_env_sha256"
CONDA_PLACES = ["$HOME/miniconda3/bin/conda", "$HOME/anaconda3/bin/conda", "$HOME/miniforge3/bin/conda", "$HOME/mambaforge/bin/conda", "/opt/conda/bin/conda", "/usr/local/miniconda3/bin/conda"]


class RouterError(Exception):
    pass


# ------------------------------------------------------------------------------------------------------------------------ hosts file
@dataclass
class Host:
    name: str
    hostname: str = ""
    user: str = ""
    root: str = "$HOME/router"
    per_gpu: int = 4
    gpus: list[int] | None = None
    skip: bool = False
    conda: str = ""
    local: bool = False


def parse_router_options(line: str) -> dict[str, str]:
    body = line.split(":", 1)[1] if ":" in line else ""
    return dict(tok.split("=", 1) for tok in body.split() if "=" in tok)


def apply_options(h: Host, opts: dict[str, str]) -> None:
    for k, v in opts.items():
        k = k.replace("_", "-")
        if k == "root":
            h.root = v
        elif k == "per-gpu":
            h.per_gpu = int(v)
        elif k == "gpus":
            h.gpus = [int(x) for x in v.split(",") if x != ""]
        elif k == "skip":
            h.skip = v not in ("0", "false", "no")
        elif k == "conda":
            h.conda = v
        elif k == "local":
            h.local = v not in ("0", "false", "no")
        else:
            raise RouterError(f"unknown #router option {k!r} (known: root per-gpu gpus skip conda local)")


def local_addresses() -> set[str]:
    out = {"127.0.0.1", "localhost", socket.gethostname()}
    try:
        out.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        out.update(subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def parse_hosts(text: str, self_addresses: set[str] | None = None) -> list[Host]:
    """The Host blocks of an ssh-config file with their `#router:` options. `Host *` blocks (ssh defaults) are not hosts; `#router:` lines before the first Host apply to all hosts."""
    defaults: dict[str, str] = {}
    hosts: list[Host] = []
    cur: Host | None = None
    star = False
    own = local_addresses() if self_addresses is None else self_addresses
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line[1:].strip().lower().startswith("router:"):
                opts = parse_router_options(line[1:].strip())
                if cur is not None:
                    apply_options(cur, opts)
                elif not star:
                    defaults.update(opts)
            continue
        key, _, value = line.partition(" ")
        key, value = key.lower(), value.strip().strip('"')
        if key == "host":
            names = value.split()
            if any(c in n for n in names for c in "*?!"):
                cur, star = None, True
                continue
            star = False
            cur = Host(name=names[0])
            apply_options(cur, defaults)
            hosts.append(cur)
        elif cur is not None and key == "hostname":
            cur.hostname = value
        elif cur is not None and key == "user":
            cur.user = value
    for h in hosts:
        if h.hostname in own or h.name in own:
            h.local = True
    names = [h.name for h in hosts]
    if len(set(names)) != len(names):
        raise RouterError(f"a host alias appears twice in the hosts file: {sorted({n for n in names if names.count(n) > 1})}")
    return hosts


# ------------------------------------------------------------------------------------------------------------------------ jobs file
@dataclass
class Job:
    name: str
    cmd: str
    gpus: int = 1
    gb: float = 0.0
    hosts: list[str] | None = None
    fetch: list[str] = field(default_factory=list)
    expect: list[str] = field(default_factory=list)
    retries: int = 0


@dataclass
class Spec:
    project: str
    env_file: str | None = None
    sync: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", ".git"])
    secrets: list[str] = field(default_factory=list)
    fetch: list[str] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def load_spec(text: str) -> Spec:
    d = json.loads(text)
    unknown = set(d) - {"project", "env", "sync", "exclude", "secrets", "fetch", "jobs"}
    if unknown:
        raise RouterError(f"unknown keys in the jobs file: {sorted(unknown)}")
    if not d.get("project") or not re.fullmatch(r"[A-Za-z0-9_.-]+", d["project"]):
        raise RouterError("'project' (a folder name) is required")
    jobs = []
    for j in d.get("jobs", []):
        bad = set(j) - {"name", "cmd", "gpus", "gb", "hosts", "fetch", "expect", "retries"}
        if bad:
            raise RouterError(f"job {j.get('name')!r}: unknown keys {sorted(bad)}")
        if not j.get("name") or not j.get("cmd"):
            raise RouterError(f"a job needs 'name' and 'cmd': {j}")
        jobs.append(Job(**j))
    names = [j.name for j in jobs]
    if len(set(names)) != len(names):
        raise RouterError(f"job names must be unique: {sorted({n for n in names if names.count(n) > 1})}")
    if len({sanitize(n) for n in names}) != len(names):
        raise RouterError("two job names are the same after replacing characters outside A-Za-z0-9_")
    spec = Spec(project=d["project"], env_file=(d.get("env") or {}).get("file"), sync=d.get("sync", []), secrets=d.get("secrets", []), fetch=d.get("fetch", []), jobs=jobs)
    if "exclude" in d:
        spec.exclude = d["exclude"]
    return spec


def env_name(env_file: Path) -> str:
    m = re.search(r"^name:\s*(\S+)\s*$", env_file.read_text(), re.M)
    if not m:
        raise RouterError(f"{env_file} has no `name:` line")
    return m.group(1)


def env_hash(env_file: Path) -> str:
    """Hash of the env spec and its post-create script: stored in the env on the host, to tell later whether the env was made from these files."""
    h = hashlib.sha256(env_file.read_bytes())
    post = env_file.with_name(env_file.stem + "_post.sh")
    if post.exists():
        h.update(post.read_bytes())
    return h.hexdigest()


PLACEHOLDERS = ("python", "gpu", "name", "host", "project_dir")


def fill(text: str, **values: str) -> str:
    """Replace {python} {gpu} {name} {host} {project_dir}; other braces are left alone."""
    return re.sub(r"\{(%s)\}" % "|".join(PLACEHOLDERS), lambda m: str(values.get(m.group(1), m.group(0))), text)


def quote_glob(path: str) -> str:
    """shell-quote a path but leave * ? [ ] to the shell (rsync fetch patterns)."""
    return "".join(c if c in "*?[]" else shlex.quote(c) if not re.fullmatch(r"[A-Za-z0-9_./@%+=:,-]", c) else c for c in path)


# ------------------------------------------------------------------------------------------------------------------------ connections
class Conn:
    """Commands and rsync on one host: over ssh -F <hosts file>, or directly for the local machine."""

    def __init__(self, host: Host, hosts_file: Path):
        self.host, self.hosts_file = host, hosts_file

    def ssh_cmd(self) -> list[str]:
        return ["ssh", "-F", str(self.hosts_file), *SSH_OPTS]

    def sh(self, script: str, timeout: float = 120, stdin: str | None = None) -> subprocess.CompletedProcess:
        cmd = ["bash", "-lc", script] if self.host.local else [*self.ssh_cmd(), self.host.name, "bash -lc " + shlex.quote(script)]
        return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)

    def must(self, script: str, what: str, timeout: float = 120, stdin: str | None = None) -> str:
        r = self.sh(script, timeout, stdin)
        if r.returncode:
            raise RouterError(f"{self.host.name}: {what} failed: {(r.stderr or r.stdout).strip()[-400:]}")
        return r.stdout

    def remote(self, path: str) -> str:
        return path if self.host.local else f"{self.host.name}:{path}"

    def rsync(self, args: str, timeout: float = 7200, cwd: Path | None = None) -> subprocess.CompletedProcess:
        e = "" if self.host.local else "-e " + shlex.quote(" ".join(shlex.quote(x) for x in self.ssh_cmd())) + " "
        return subprocess.run(f"rsync -a {e}{args}", shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)


# ------------------------------------------------------------------------------------------------------------------------ host preparation
@dataclass
class Ctx:
    """A prepared host: absolute project dir, the env's python, the GPUs the router may use."""
    host: Host
    conn: Conn
    project_dir: str
    python: str
    prefix: str
    warnings: list[str] = field(default_factory=list)


def find_conda(conn: Conn) -> str:
    hint = conn.host.conda
    places = " ".join(shlex.quote(hint) if hint else p.replace("$HOME", "\"$HOME\"") for p in ([hint] if hint else CONDA_PLACES))
    out = conn.sh(f"for c in {places}; do [ -x \"$c\" ] && echo \"$c\" && exit 0; done; command -v conda || true").stdout.strip().splitlines()
    return out[0] if out else ""


def env_prefix(conn: Conn, conda: str, name: str) -> str:
    r = conn.sh(f"{shlex.quote(conda)} env list --json", timeout=120)
    if r.returncode:
        raise RouterError(f"{conn.host.name}: `conda env list` failed: {r.stderr.strip()[-300:]}")
    for p in json.loads(r.stdout).get("envs", []):
        if Path(p).name == name:
            return p
    return ""


def wait_for_exit(conn: Conn, exit_file: str, what: str, timeout: float, poll: float = 20) -> int:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = conn.sh(f"cat {shlex.quote(exit_file)} 2>/dev/null; true", timeout=60)
        except subprocess.TimeoutExpired:
            r = None
        if r is not None and r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip())
        time.sleep(poll)
    raise RouterError(f"{conn.host.name}: {what} did not finish within {timeout / 3600:.1f} h (the tmux session keeps running)")


def ensure_env(conn: Conn, env_file: Path, project_dir: str, recreate: bool, strict: bool, timeout: float, log=print) -> tuple[str, list[str]]:
    """The prefix of the env named in `env_file` on the host: found by name; created from the file (and its _post.sh) when there is none. Returns (prefix, warnings)."""
    name, sha, warnings = env_name(env_file), env_hash(env_file), []
    conda = find_conda(conn)
    if not conda:
        raise RouterError(f"{conn.host.name}: conda not found (install miniconda there, or give its path with `#router: conda=/path/to/conda`)")
    prefix = env_prefix(conn, conda, name)
    if prefix and recreate:
        log(f"{conn.host.name}: removing env {name} (--recreate-env)")
        conn.must(f"{shlex.quote(conda)} env remove -y -n {shlex.quote(name)}", "env remove", timeout=1800)
        prefix = ""
    if prefix:
        marker = conn.sh(f"cat {shlex.quote(prefix + '/' + ENV_MARKER)} 2>/dev/null; true").stdout.strip()
        if marker != sha:
            msg = (f"{conn.host.name}: env {name} exists but was {'created from other files' if marker else 'not created by this router'}; its package versions may differ from {env_file.name} "
                   f"(--recreate-env rebuilds it, --strict-env refuses to run on it)")
            if strict:
                raise RouterError(msg)
            warnings.append(msg)
        return prefix, warnings
    rd = f"{project_dir}/.router"
    conn.must(f"mkdir -p {shlex.quote(rd)}", "mkdir")
    files = [env_file] + ([env_file.with_name(env_file.stem + "_post.sh")] if env_file.with_name(env_file.stem + "_post.sh").exists() else [])
    r = conn.rsync(" ".join(shlex.quote(str(f)) for f in files) + " " + shlex.quote(conn.remote(rd + "/")))
    if r.returncode:
        raise RouterError(f"{conn.host.name}: cannot copy the env files: {r.stderr.strip()[-300:]}")
    session, done = sanitize(f"rt_env_{name}"), f"{rd}/env_{sanitize(name)}.exit"
    q = shlex.quote
    post = f"{rd}/{env_file.stem}_post.sh"
    steps = [f"{q(conda)} env create -y -n {q(name)} -f {q(rd + '/' + env_file.name)}",
             f"PREFIX=$({q(conda)} run -n {q(name)} python -c 'import sys; print(sys.prefix)')"]
    if len(files) > 1:
        steps.append(f"PY=$PREFIX/bin/python CONDA_PREFIX=$PREFIX PATH=$PREFIX/bin:$PATH bash {q(post)}")
    steps.append(f"echo {sha} > $PREFIX/{ENV_MARKER}")
    log_file = q(rd + '/env_' + sanitize(name) + '.log')
    # a failed creation removes what it made: a half-built env of that name would be taken for an existing one by the next run
    script = (f"cd {q(rd)} && rm -f {q(done)} && ( {' && '.join(steps)} ) > {log_file} 2>&1; rc=$?; "
              f"if [ $rc -ne 0 ]; then {q(conda)} env remove -y -n {q(name)} >> {log_file} 2>&1; fi; echo $rc > {q(done)}")
    if conn.sh(f"tmux has-session -t {session} 2>/dev/null").returncode != 0:
        log(f"{conn.host.name}: creating env {name} (conda env create; log {rd}/env_{sanitize(name)}.log)")
        conn.must(f"tmux new-session -d -s {session} {shlex.quote('bash -c ' + shlex.quote(script))}", "start env creation")
    rc = wait_for_exit(conn, done, f"creating env {name}", timeout)
    if rc != 0:
        raise RouterError(f"{conn.host.name}: creating env {name} failed (exit {rc}); see {rd}/env_{sanitize(name)}.log on the host")
    prefix = env_prefix(conn, conda, name)
    if not prefix:
        raise RouterError(f"{conn.host.name}: env {name} is not listed by conda after its creation")
    return prefix, warnings


def sync_project(conn: Conn, spec: Spec, project_dir: Path, remote_dir: str) -> None:
    """rsync the `sync` paths (relative to the project dir here) to the project dir on the host, then the secrets with mode 600."""
    ex = " ".join(f"--exclude {shlex.quote(p)}" for p in spec.exclude)
    paths = [p for p in spec.sync if (project_dir / p).exists()]
    missing = [p for p in spec.sync if p not in paths]
    if missing:
        raise RouterError(f"sync paths not found in {project_dir}: {missing}")
    if paths:
        r = conn.rsync(f"-R {ex} " + " ".join(shlex.quote("./" + p.rstrip("/")) for p in paths) + " " + shlex.quote(conn.remote(remote_dir + "/")), cwd=project_dir)
        if r.returncode:
            raise RouterError(f"{conn.host.name}: rsync of the code failed: {r.stderr.strip()[-300:]}")
    for s in spec.secrets:
        if not (project_dir / s).exists():
            raise RouterError(f"secret {s!r} not found in {project_dir}")
        r = conn.rsync("-R --chmod=F600 " + shlex.quote("./" + s) + " " + shlex.quote(conn.remote(remote_dir + "/")), cwd=project_dir)
        if r.returncode:
            raise RouterError(f"{conn.host.name}: copying the secret {s!r} failed (the value is not shown)")


def prepare_host(host: Host, hosts_file: Path, spec: Spec, project_dir: Path, a, log=print) -> Ctx:
    conn = Conn(host, hosts_file)
    r = conn.sh("echo ok", timeout=30)
    if r.returncode or r.stdout.strip() != "ok":
        raise RouterError(f"{host.name}: not reachable: {(r.stderr or r.stdout).strip()[-200:]}")
    base = conn.must(f"mkdir -p {host.root} && cd {host.root} && pwd", "resolve the root folder").strip()
    remote_dir = f"{base}/{spec.project}"
    conn.must(f"mkdir -p {shlex.quote(remote_dir)}/.router {shlex.quote(remote_dir)}/logs/router", "mkdir")
    python, prefix, warnings = "python3", "", []
    if spec.env_file:
        prefix, warnings = ensure_env(conn, project_dir / spec.env_file, remote_dir, a.recreate_env, a.strict_env, a.env_timeout, log)
        python = prefix + "/bin/python"
    sync_project(conn, spec, project_dir, remote_dir)
    return Ctx(host, conn, remote_dir, python, prefix, warnings)


# ------------------------------------------------------------------------------------------------------------------------ polling and placement
@dataclass
class HostState:
    gpus: dict[int, tuple[float, float]]    # index -> (free GB, total GB)
    exits: dict[str, int]                   # job (sanitized name) -> exit code, from <project>/.router/*.exit
    sessions: set[str]


def parse_probe(out: str) -> HostState | None:
    if "END" not in out:
        return None
    gpus, exits, sessions = {}, {}, set()
    for line in out.splitlines():
        if line.startswith("GPU "):
            idx, free, total = [x.strip() for x in line[4:].split(",")]
            gpus[int(idx)] = (float(free) / 1024, float(total) / 1024)
        elif line.startswith("EXIT "):
            parts = line.split()
            if len(parts) == 3:
                exits[parts[1]] = int(parts[2])
        elif line.startswith("SESSION "):
            sessions.add(line[8:].strip())
    return HostState(gpus, exits, sessions)


def probe(ctx: Ctx) -> HostState | None:
    rd = shlex.quote(ctx.project_dir + "/.router")
    script = (f"nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | sed 's/^/GPU /'; "
              f"for f in {rd}/*.exit; do [ -f \"$f\" ] && echo \"EXIT $(basename \"$f\" .exit) $(cat \"$f\")\"; done; "
              f"tmux ls -F '#S' 2>/dev/null | sed 's/^/SESSION /'; echo END")
    try:
        r = ctx.conn.sh(script, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    return parse_probe(r.stdout) if r.returncode == 0 else None


@dataclass
class Slot:
    host: str
    gpus: list[int]


def place(job: Job, hosts: dict[str, Host], states: dict[str, HostState], load: dict[tuple[str, int], int], reserved: dict[tuple[str, int], float], headroom: float,
          launched_at: dict[str, float], stagger: float, now: float, host_jobs: dict[str, int], cpu_cap: int = 4) -> Slot | None:
    """Where `job` can start now: a host with job.gpus GPUs that are each below per-gpu router jobs and have gb + headroom GB free (minus what jobs just started there will still claim).
    Prefers the GPUs with the fewest router jobs, then the most free memory. CPU jobs (gpus = 0) go to the host with the fewest router jobs."""
    best, best_key = None, None
    for name, h in hosts.items():
        st = states.get(name)
        if st is None or (job.hosts and name not in job.hosts) or now - launched_at.get(name, -1e9) < stagger:
            continue
        if job.gpus == 0:
            if host_jobs.get(name, 0) < cpu_cap and (best is None or host_jobs.get(name, 0) < best_key[0]):
                best, best_key = Slot(name, []), (host_jobs.get(name, 0), 0.0)
            continue
        ok = []
        for idx, (free, total) in st.gpus.items():
            if h.gpus is not None and idx not in h.gpus:
                continue
            eff = free - reserved.get((name, idx), 0.0)
            if load.get((name, idx), 0) < h.per_gpu and eff >= job.gb + headroom:
                ok.append((load.get((name, idx), 0), -eff, idx))
        if len(ok) >= job.gpus:
            ok.sort()
            pick = ok[:job.gpus]
            key = (sum(p[0] for p in pick), sum(p[1] for p in pick))
            if best is None or key < best_key:
                best, best_key = Slot(name, sorted(p[2] for p in pick)), key
    return best


# ------------------------------------------------------------------------------------------------------------------------ state file
def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"jobs": {}}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------------------------------ the router
class Router:
    def __init__(self, spec: Spec, hosts: list[Host], hosts_file: Path, project_dir: Path, state_path: Path, a, log=print):
        self.spec, self.hosts, self.hosts_file, self.project_dir, self.state_path, self.a, self.log = spec, {h.name: h for h in hosts}, hosts_file, project_dir, state_path, a, log
        self.state = load_state(state_path)
        self.ctx: dict[str, Ctx] = {}
        self.launched_at: dict[str, float] = {}
        self.reserved_until: list[tuple[float, tuple[str, int], float]] = []
        self.pool = cf.ThreadPoolExecutor(max_workers=16)

    # -- setup
    def prepare(self) -> None:
        futs = {n: self.pool.submit(prepare_host, h, self.hosts_file, self.spec, self.project_dir, self.a, self.log) for n, h in self.hosts.items()}
        for n, f in futs.items():
            try:
                self.ctx[n] = f.result()
                for w in self.ctx[n].warnings:
                    self.log("WARNING " + w)
                self.log(f"{n}: ready ({self.ctx[n].project_dir}, python {self.ctx[n].python})")
            except (RouterError, subprocess.SubprocessError, OSError) as e:
                self.log(f"{n}: skipped: {e}")
        if not self.ctx:
            raise RouterError("no host is ready")

    # -- jobs
    def session(self, job: Job) -> str:
        return sanitize(f"rt_{self.spec.project}_{job.name}")[:100]

    def values(self, job: Job, ctx: Ctx, gpus: list[int]) -> dict[str, str]:
        return {"python": ctx.python, "gpu": ",".join(map(str, gpus)), "name": job.name, "host": ctx.host.name, "project_dir": ctx.project_dir}

    def script(self, job: Job, ctx: Ctx, gpus: list[int]) -> str:
        v, key = self.values(job, ctx, gpus), sanitize(job.name)
        env = f"export ROUTER_JOB={shlex.quote(job.name)} ROUTER_HOST={shlex.quote(ctx.host.name)} ROUTER_GPU={shlex.quote(v['gpu'])}\n"
        if gpus:
            env += f"export CUDA_VISIBLE_DEVICES={shlex.quote(v['gpu'])}\n"
        path = f"export PATH={shlex.quote(ctx.prefix + '/bin')}:$PATH\n" if ctx.prefix else ""
        return (f"#!/bin/bash\ncd {shlex.quote(ctx.project_dir)}\n{path}{env}rm -f .router/{key}.exit\n( {fill(job.cmd, **v)} ) >> logs/router/{key}.log 2>&1\necho $? > .router/{key}.exit\n")

    def start(self, job: Job, slot: Slot) -> None:
        ctx, key = self.ctx[slot.host], sanitize(job.name)
        ctx.conn.must(f"cat > {shlex.quote(ctx.project_dir + '/.router/' + key + '.sh')}", "write the job script", stdin=self.script(job, ctx, slot.gpus))
        ctx.conn.must(f"cd {shlex.quote(ctx.project_dir)} && rm -f .router/{key}.exit && tmux new-session -d -s {self.session(job)} {shlex.quote('bash .router/' + key + '.sh')}", "start the session")
        self.launched_at[slot.host] = time.time()
        for g in slot.gpus:
            self.reserved_until.append((time.time() + self.a.ramp, (slot.host, g), job.gb))
        self.state["jobs"][job.name] = {"status": "running", "host": slot.host, "gpus": slot.gpus, "started": time.strftime("%F %T"), "attempt": self.state["jobs"].get(job.name, {}).get("attempt", 0) + 1}
        save_state(self.state_path, self.state)
        self.log(f"[{time.strftime('%d %b %H:%M')}] {job.name} -> {slot.host} GPU {','.join(map(str, slot.gpus)) or '-'}")

    def fetch(self, job: Job, rc: int) -> list[str]:
        """Copy the job's results (spec fetch + job fetch + its log) back; returns problems."""
        st = self.state["jobs"][job.name]
        ctx, problems = self.ctx[st["host"]], []
        v = self.values(job, ctx, st["gpus"])
        for p in (fill(x, **v) for x in [*self.spec.fetch, *job.fetch, f"logs/router/{sanitize(job.name)}.log"]):
            r = ctx.conn.rsync("-R " + quote_glob(ctx.conn.remote(ctx.project_dir + "/./" + p)) + " " + shlex.quote(str(self.project_dir) + "/"))
            if r.returncode not in (0, 23, 24):   # 23 / 24: some files vanished or were not there (a pattern may match nothing); `expect` decides whether that matters
                problems.append(f"fetching {p}: {r.stderr.strip()[-200:]}")
        if rc == 0:
            problems += [f"expected file missing after the fetch: {e}" for e in (fill(x, **v) for x in job.expect) if not (self.project_dir / e).exists()]
        return problems

    def finish(self, job: Job, rc: int) -> None:
        st = self.state["jobs"][job.name]
        problems = self.fetch(job, rc)
        failed = rc != 0 or bool(problems)
        if failed and st["attempt"] <= job.retries:
            st["status"] = "retry"
            self.log(f"[{time.strftime('%d %b %H:%M')}] {job.name} failed on {st['host']} (exit {rc}{'; ' + '; '.join(problems) if problems else ''}); retry {st['attempt']} of {job.retries}")
        else:
            st["status"], st["exit"], st["finished"] = ("failed" if failed else "done"), rc, time.strftime("%F %T")
            st["problems"] = problems
            self.log(f"[{time.strftime('%d %b %H:%M')}] {job.name} {'FAILED' if failed else 'done'} on {st['host']} (exit {rc}){': ' + '; '.join(problems) if problems else ''}")
        save_state(self.state_path, self.state)

    # -- the loop
    def status_of(self, name: str) -> str | None:
        return self.state["jobs"].get(name, {}).get("status")

    def observe(self, jobs: dict[str, Job], states: dict[str, HostState | None], lost_polls: dict[str, int]) -> None:
        """Look at the running jobs: finished ones are fetched, jobs whose session has vanished without an exit file twice in a row are marked lost."""
        for name, st in [(n, s) for n, s in self.state["jobs"].items() if s["status"] == "running" and n in jobs]:
            hs = states.get(st["host"])
            if hs is None:
                continue    # no answer this time: the job keeps running on the host
            key = sanitize(name)
            if key in hs.exits:
                self.finish(jobs[name], hs.exits[key])
                lost_polls.pop(name, None)
            elif self.session(jobs[name]) not in hs.sessions:
                lost_polls[name] = lost_polls.get(name, 0) + 1
                if lost_polls[name] >= 2:
                    self.log(f"{name}: its session on {st['host']} is gone without an exit file; it will be started again")
                    st["status"] = "lost"
                    save_state(self.state_path, self.state)
            else:
                lost_polls.pop(name, None)

    def counts(self, jobs: dict[str, Job]):
        load: dict[tuple[str, int], int] = {}
        host_jobs: dict[str, int] = {}
        for n, st in self.state["jobs"].items():
            if st["status"] == "running" and n in jobs:
                host_jobs[st["host"]] = host_jobs.get(st["host"], 0) + 1
                for g in st["gpus"]:
                    load[(st["host"], g)] = load.get((st["host"], g), 0) + 1
        reserved: dict[tuple[str, int], float] = {}
        for _, key, gb in self.reserved_until:
            reserved[key] = reserved.get(key, 0.0) + gb
        return load, host_jobs, reserved

    def plan(self, jobs: dict[str, Job], pending: list[str], states: dict[str, HostState | None]) -> list[tuple[str, Slot | None]]:
        """Greedy placement of the pending jobs on what is free now (nothing is started): the --dry-run output."""
        load, host_jobs, reserved = self.counts(jobs)
        hosts, live = {n: self.hosts[n] for n in self.ctx}, {n: s for n, s in states.items() if s is not None}
        out = []
        for name in pending:
            slot = place(jobs[name], hosts, live, load, reserved, self.a.headroom, {}, 0.0, time.time(), host_jobs)
            if slot:
                host_jobs[slot.host] = host_jobs.get(slot.host, 0) + 1
                for g in slot.gpus:
                    load[(slot.host, g)] = load.get((slot.host, g), 0) + 1
                    reserved[(slot.host, g)] = reserved.get((slot.host, g), 0.0) + jobs[name].gb
            out.append((name, slot))
        return out

    def run(self, dry: bool = False) -> int:
        jobs = {j.name: j for j in self.spec.jobs}
        for n, st in self.state["jobs"].items():     # a job recorded as running on a host that is not ready now cannot be looked after: leave it alone
            if n in jobs and st["status"] == "running" and st["host"] not in self.ctx:
                self.log(f"WARNING {n} is recorded as running on {st['host']}, which is not ready now; leaving it alone (status 'unknown')")
                st["status"] = "unknown"
        done = [n for n in jobs if self.status_of(n) in ("done", "failed", "unknown")]
        self.log(f"{len(done)} of {len(jobs)} jobs already finished, {len(jobs) - len(done)} to go")
        lost_polls: dict[str, int] = {}
        last_wait = 0.0
        while True:
            pending = [n for n in jobs if self.status_of(n) in (None, "retry", "lost", "killed")]
            running = [n for n in jobs if self.status_of(n) == "running"]
            if not pending and not running:
                break
            states = {n: probe(c) for n, c in self.ctx.items()}
            self.reserved_until = [r for r in self.reserved_until if r[0] > time.time()]
            self.observe(jobs, states, lost_polls)
            pending = [n for n in jobs if self.status_of(n) in (None, "retry", "lost", "killed")]
            if dry:
                for name, slot in self.plan(jobs, pending, states):
                    self.log(f"would start {name} on " + (f"{slot.host} GPU {','.join(map(str, slot.gpus)) or '-'}" if slot else "nothing now (no host with a free GPU)"))
                return 0
            load, host_jobs, reserved = self.counts(jobs)
            for name in pending:
                slot = place(jobs[name], {n: self.hosts[n] for n in self.ctx}, {n: s for n, s in states.items() if s is not None}, load, reserved, self.a.headroom,
                             self.launched_at, self.a.stagger, time.time(), host_jobs)
                if slot is None:
                    continue
                try:
                    self.start(jobs[name], slot)
                except RouterError as e:
                    self.log(f"could not start {name} on {slot.host}: {e}")
                    continue
                host_jobs[slot.host] = host_jobs.get(slot.host, 0) + 1
                for g in slot.gpus:
                    load[(slot.host, g)] = load.get((slot.host, g), 0) + 1
                    reserved[(slot.host, g)] = reserved.get((slot.host, g), 0.0) + jobs[name].gb
            pending = [n for n in jobs if self.status_of(n) in (None, "retry", "lost", "killed")]
            running = [n for n in jobs if self.status_of(n) == "running"]
            if pending and not running and not any(self.can_ever_place(jobs[n], states) for n in pending):
                raise RouterError(f"no host can run {pending[:5]} (not enough GPUs or memory, or the job's hosts are not ready)")
            if pending and time.time() - last_wait > 600:
                last_wait = time.time()
                self.log(f"[{time.strftime('%d %b %H:%M')}] {len(running)} running, {len(pending)} waiting for a free GPU")
            if not pending and not running:
                break
            time.sleep(self.a.poll)
        return 1 if any(self.status_of(n) == "failed" for n in jobs) else 0

    def can_ever_place(self, job: Job, states: dict[str, HostState | None]) -> bool:
        """Whether some ready host has (as far as its last answer says) enough GPUs, each big enough, for the job; a host that did not answer counts as possible."""
        for n, h in self.hosts.items():
            if n not in self.ctx or (job.hosts and n not in job.hosts):
                continue
            st = states.get(n)
            if job.gpus == 0 or st is None:
                return True
            usable = [t for i, (f, t) in st.gpus.items() if (h.gpus is None or i in h.gpus) and t >= job.gb + self.a.headroom]
            if len(usable) >= job.gpus:
                return True
        return False


# ------------------------------------------------------------------------------------------------------------------------ commands
def read_hosts(a) -> tuple[list[Host], Path]:
    f = Path(a.hosts_file)
    if not f.exists():
        raise RouterError(f"hosts file {f} not found (copy cluster/hosts.example to {f} and edit it)")
    hosts = [h for h in parse_hosts(f.read_text()) if not h.skip]
    if a.hosts:
        unknown = set(a.hosts) - {h.name for h in hosts}
        if unknown:
            raise RouterError(f"not in the hosts file (or skipped): {sorted(unknown)}")
        hosts = [h for h in hosts if h.name in a.hosts]
    return hosts, f


def cmd_hosts(a) -> int:
    hosts, f = read_hosts(a)
    env_file = Path(a.env) if a.env else None

    def one(h: Host) -> str:
        conn = Conn(h, f)
        try:
            r = conn.sh("echo \"load $(cut -d' ' -f1 /proc/loadavg) disk $(df -h $HOME | tail -1 | awk '{print $4}') free\"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null", timeout=40)
        except subprocess.TimeoutExpired:
            return f"{h.name:<14} no answer"
        if r.returncode:
            return f"{h.name:<14} not reachable: {(r.stderr or r.stdout).strip()[-90:]}"
        lines = r.stdout.strip().splitlines()
        conda = find_conda(conn)
        envs = ""
        if env_file and conda:
            name = env_name(env_file)
            try:
                pre = env_prefix(conn, conda, name)
                mk = conn.sh(f"cat {shlex.quote(pre + '/' + ENV_MARKER)} 2>/dev/null; true").stdout.strip() if pre else ""
                envs = f"env {name}: " + ("missing" if not pre else "from this spec" if mk == env_hash(env_file) else "exists (not made by the router, versions unknown)")
            except RouterError as e:
                envs = str(e)[-80:]
        gpus = "; ".join(l.split(", ")[0] + ": " + l.split(", ")[2] + " / " + l.split(", ")[3] for l in lines[1:]) or "no GPU"
        return f"{h.name:<14} {'local' if h.local else 'up':<5} {lines[0]}  [{gpus}]  conda {'yes' if conda else 'NO'}  {envs}"

    with cf.ThreadPoolExecutor(8) as ex:
        for line in ex.map(one, hosts):
            print(line)
    return 0


def make_router(a) -> Router:
    hosts, f = read_hosts(a)
    jobs_file = Path(a.jobs).resolve()
    spec = load_spec(jobs_file.read_text())
    project_dir = Path(a.project_dir).resolve() if a.project_dir else HERE.parent
    state_path = jobs_file.with_suffix(".state.json")
    return Router(spec, hosts, f, project_dir, state_path, a)


def cmd_run(a, setup_only: bool = False) -> int:
    r = make_router(a)
    r.prepare()
    if setup_only:
        return 0
    return r.run(dry=a.dry_run)


def cmd_status(a) -> int:
    spec, state = load_spec(Path(a.jobs).read_text()), load_state(Path(a.jobs).resolve().with_suffix(".state.json"))
    for j in spec.jobs:
        s = state["jobs"].get(j.name)
        print(f"{j.name:<40} " + (f"{s['status']:<8} {s.get('host', '')} GPU {','.join(map(str, s.get('gpus', [])))} since {s.get('started', '')}" + (f" exit {s['exit']}" if 'exit' in s else "") if s else "not started"))
    return 0


def cmd_kill(a) -> int:
    r = make_router(a)
    r.prepare()
    for j in r.spec.jobs:
        s = r.state["jobs"].get(j.name)
        if s and s["status"] == "running" and (not a.names or j.name in a.names) and s["host"] in r.ctx:
            r.ctx[s["host"]].conn.sh(f"tmux kill-session -t {r.session(j)}")
            s["status"] = "killed"
            print(f"killed {j.name} on {s['host']}")
    save_state(r.state_path, r.state)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["hosts", "setup", "run", "status", "kill"])
    ap.add_argument("jobs", nargs="?", help="the jobs file (not needed for `hosts`)")
    ap.add_argument("names", nargs="*", help="kill: the jobs to stop (default: all running)")
    ap.add_argument("--hosts-file", default=str(DEFAULT_HOSTS))
    ap.add_argument("--hosts", nargs="*", default=[], help="use only these hosts of the hosts file")
    ap.add_argument("--project-dir", default=None, help="the folder the jobs file's paths are relative to (default: the parent of cluster/)")
    ap.add_argument("--env", default=None, help="hosts: an env spec file to check the hosts' envs against")
    ap.add_argument("--headroom", type=float, default=4.0, help="GB of GPU memory kept free beyond a job's own gb")
    ap.add_argument("--stagger", type=float, default=30.0, help="seconds between two job starts on one host (their memory shows up in nvidia-smi after a while)")
    ap.add_argument("--ramp", type=float, default=240.0, help="seconds a started job's gb stay reserved before nvidia-smi is trusted for it")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between polls of the hosts")
    ap.add_argument("--recreate-env", action="store_true", help="remove and create again the env on the hosts")
    ap.add_argument("--strict-env", action="store_true", help="refuse hosts whose env of that name was not made from this spec")
    ap.add_argument("--env-timeout", type=float, default=3 * 3600, help="seconds to wait for an env to be created")
    ap.add_argument("--dry-run", action="store_true", help="run: prepare the hosts and show nothing started (the first poll only)")
    a = ap.parse_args(argv)
    try:
        if a.command == "hosts":
            return cmd_hosts(a)
        if not a.jobs:
            ap.error("the jobs file is required")
        if a.command == "status":
            return cmd_status(a)
        if a.command == "kill":
            a.names = a.names or []
            return cmd_kill(a)
        return cmd_run(a, setup_only=a.command == "setup")
    except RouterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
