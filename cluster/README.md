# cluster/route.py: run experiments on the machines you can ssh to

One script, standard library only (Python >= 3.8; `ssh`, `rsync` and `tmux` on this machine and on the hosts). It works from any machine that has ssh access to the hosts in the
hosts file; the hosts need no shared filesystem.

```
python cluster/route.py hosts --env cluster/env.yml     # who answers, GPUs (used / total), conda, is the env there
python cluster/route.py setup jobs.json                 # prepare the hosts (env + code + secrets), start nothing
python cluster/route.py run   jobs.json                 # prepare, start every job that is not done on a host with a free GPU, fetch results (resumable)
python cluster/route.py run   jobs.json --dry-run       # where the jobs would go right now
python cluster/route.py status jobs.json                # the state file
python cluster/route.py kill  jobs.json [job ...]       # stop running jobs
```

## What `run` does

1. **Hosts file** (`cluster/hosts`, git-ignored, see `hosts.example`): ssh-config syntax, used as `ssh -F <file>`. `#router: key=value` lines set `root`, `per-gpu`, `gpus`, `conda`, `skip`, `local`.
   A host whose address is one of this machine's counts as local (no ssh).
2. **Env**: the `name:` in `cluster/env.yml` is looked up with `conda env list` on each host. Missing: it is created (`conda env create -f`, then `cluster/env_post.sh` if it exists,
   e.g. flash-attn which needs torch first) in a tmux session on the host. The hash of both files is stored in the env. An env of that name that was made from other files is used but
   reported (`WARNING ... may differ`); `--recreate-env` rebuilds it, `--strict-env` refuses it. Conda must be installed on the host (not installed by the router).
3. **Code**: the `sync` paths of the jobs file (relative to the project dir here, the parent of `cluster/`) are copied with rsync to `<root>/<project>/` on each host, nothing through git,
   uncommitted changes included. `secrets` are copied with mode 600, only the files listed there.
4. **Jobs**: a job starts in a tmux session on a host that has a GPU with `gb` + headroom GB free (read from nvidia-smi at every poll, so other people's jobs count) and fewer than
   `per-gpu` router jobs. `CUDA_VISIBLE_DEVICES` is set to the job's GPU; `{gpu}` in `cmd` is its index (also `ROUTER_GPU`). The output goes to `logs/router/<job>.log` on the host.
   The session survives a lost connection; after a restart `run` finds running jobs again from the state file.
5. **Results**: when a job exits, the paths in `fetch` (spec and job) and its log are copied back with rsync; `expect` files must exist afterwards or the job counts as failed. Failed jobs
   are retried `retries` times. Exit code of `run`: 0 when every job is done, 1 when some failed.

## Jobs file

See `jobs.example.json`. `{python}`, `{gpu}`, `{name}`, `{host}`, `{project_dir}` are replaced in `cmd`, `fetch` and `expect`. Jobs run with the project dir on the host as working directory.

## Notes

* ssh uses `StrictHostKeyChecking=accept-new`: a host's key is remembered at the first contact (a changed key is refused).
* The state file is `<jobs file>.state.json` (git-ignored); delete a job's entry to run it again.
* Tests: `python -m unittest tests.test_cluster_router` (the whole-run tests use this machine as a host with a temporary root and need tmux and rsync).
