#!/usr/bin/env bash
# Rebuild everything that is derived from the finished runs: stamp the diff into every result json, the summary tables and the HTML reports.
# Does not commit and does not publish. With --after <tmux session> it first waits until that session (the queue driver) has ended.
#   tmux new -d -s fbrefresh 'bash scripts/refresh_reports.sh --after fbqueue3'
set -u
cd "$(dirname "$0")/.."
PY=/home/ivan.rodkin/miniconda3/envs/py311_pt2.6_cu12.4/bin/python
if [ "${1:-}" = "--after" ]; then
  while tmux has-session -t "$2" 2>/dev/null; do sleep 300; done
fi
{
  date
  "$PY" src/config.py results/raw/*.json results/explore/*.json
  "$PY" src/aggregate.py --no-plots
  for r in gdn swa fb; do "$PY" src/report.py "$r" --out "reports/${r}_report.html"; done
  date
} >> logs/refresh.log 2>&1
