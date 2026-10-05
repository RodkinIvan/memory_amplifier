# Attractor feedback memory for linear RNNs — phase 1: benchmark and baseline trade-off

Linear-attention RNNs / SSMs (Gated DeltaNet, Mamba-2, RWKV) keep a fixed-size state
`S_t = α_t · S_{t-1} + (write)`. With a **single head and a scalar decay α** they face a trade-off:
tasks that need **forgetting** (bounded interference on an infinite stream) need α < 1, tasks that need
**indefinite retention** (a passkey seen once at the start) need α = 1. Phase 1 builds the benchmark that makes this
trade-off measurable and establishes the baselines; the feedback model comes later.

## Benchmark: `irodkin/passkey_swar` (Passkey + Sliding-Window Associative Recall)

```
BOS  PK p1 p2 p3 p4   <stream>   PKQ p1 p2 p3 p4 EOS          stream = N writes `W k v` + queries `Q k a`
```

* **Passkey**: 4 tokens from 16 values, to be repeated at the end after `PKQ`.
* **AR queries** (about 1 per 4 writes, at random positions) on keys drawn from a drifting active set:
  * the key's latest write is ≤ `w = 16` writes old → answer is its value (`ar_valid`; overwrites count);
  * older than `2w = 32` writes, or never written → answer `NULL` (`ar_expired`);
  * keys last written 17–32 writes ago are never queried (gray zone).
* Configs `n{N}`, N ∈ {32 … 16384} AR writes (up to ~61k tokens). Train: n32–n256 (100k each); validation/test: all N (1k each).
* Every label is re-derived by a brute-force oracle from the token stream; see `reports/passkey_swar_stats.md`
  (ratios, query-gap and age histograms, key-reuse statistics) and `data/sanity_checks.py`.

```bash
python data/make_passkey_swar.py --out-dir data_cache/passkey_swar   # generate + run all sanity checks (~1 min)
HF_TOKEN=... python data/upload_passkey_swar.py --private            # push_to_hub + dataset card (checks gate the upload)
```

Expired queries are either *never-written* keys (free for an empty state) or *stale* keys (test forgetting). With N ≤ 2w no
key can be stale, so the smallest configs only contain never-written expired queries; the evaluation reports both.

## Experiments (phase 1)

Small models on stock `flash-linear-attention` blocks, d_model 128, loss only on labels. GDN: 2 layers, **1 head of dim 32** (fp32). SWA: 2 or 4 layers, 4 heads of dim 32 (bf16, flash-attention needs it).
Plan and rationale: `reports/exp1_plan.md`.

| Exp | Models | Status |
|---|---|---|
| 1 | GDN, decay **fixed** to α ∈ {1, .999, .995, .99, .98, .95, .9, .8}, 3 seeds; plot passkey vs expired-AR accuracy | done, `reports/gdn_report.html` |
| 1b | Sliding-window attention (SWA), window ∈ {32, 64, 128} × layers ∈ {2, 4}, 3 seeds. No decay: forgetting is a hard cutoff, reach R = layers·(window−1) tokens | done, `reports/swa_report.html` |
| 1c | SWA + **segment feedback loop** (memory amplification): once per segment each layer turns its output into one vector that is added to its input in the next segment. Window 64, segment ∈ {64, 32, 16, 8}, tanh or no-tanh loop, stream-length curriculum; extras: loop noise, 40k steps, loop gain 0 instead of 0.1, a plain-SWA control with the same curriculum, an attractor probe | 20k-step sweep done; round 3 (50k steps at gain 0, plain SWA at windows 64–1024, segment study) queued/running, see `configs/queue.txt`; `reports/fb_report.html`, exploration in `results/explore/` |
| 2 | **GDN-2** = GDN with the stock learned decay (`kind: gdn2`, `configs/gdn2.yaml`), trained like the loop models (50k steps, curriculum); then GDN with eigenvalues in [−1, 1], Mamba-2 | GDN-2 done (3 seeds, `results/raw/gdn2_*`): best seed passkey 1.00 up to N = 4,096, 0.97 at 8,192, 0.15 at 16,384; the other two seeds fail from N = 512 |
| 3 | GDN, 2 heads, per-head α statistics | not started |
| 4 | attractor cleanup in isolation (numpy/torch) | not started |

```bash
python src/launch.py --config configs/exp1.yaml --session exp1 --gpus 0 1 --procs-per-gpu 3 \
    --sweep model.alpha=1.0,0.999,0.995,0.99,0.98,0.95,0.9,0.8 seed=0,1,2     # tmux, resumable, logs to wandb "passkey_swar"
python src/launch.py --config configs/swa.yaml --session swa --gpus 0 1 --procs-per-gpu 3 \
    --sweep model.window=32,64,128 model.n_layers=2,4 seed=0,1,2              # sliding-window baseline
python src/launch.py --config configs/swa_fb.yaml --session fbsweep --gpus 0 1 --procs-per-gpu 9 \
    --sweep model.nonlin=tanh,none model.segment=64,32,16 seed=0,1,2          # feedback loop; defaults of the yaml = the recipe
tmux new -d -s fbqueue 'python src/run_queue.py --queue configs/queue.txt --watch fbsweep'   # unattended queue of further sweeps
#   (a stage starts only with free GPU memory for its workers; add --set model.kv_cache=true to a sweep to cut the loop model's memory 2-4x, same results, no speed-up)
#   queue stages are gated on the measured per-worker memory (config.footprint_gb: 8 GB at segment 8, 5 GB at segments 32/64; segment-8 stages run at most 2 workers per GPU), and a
#   worker waits for that much free memory before it restarts a run after an out-of-memory crash
#   a run that dies with CUDA out-of-memory (also in mid-training) is restarted up to 30 times (5 min apart) and resumes from runs/<name>/resume.pt (written at every periodic
#   validation); to continue stopped runs, kill the old tmux session and launch the same command again. --set train.cuda_graph=true replays one CUDA graph per batch shape
#   instead of ~130 kernel launches per (segment, layer) call (same results up to rounding, tests/test_graphed.py)
python src/config.py results/raw/*.json                                          # stamp each result with its diff against the frozen reference configs/ref/*_v1.yaml
python src/aggregate.py                                                          # results/summary*.csv + plots
python src/probes.py passkey-length --runs swa_w128_L4_s0 --out results/probes/x.json   # probes on checkpoints (see --help)
python src/probes.py profile --runs swa_fb_nltanh_seg64_s2 --ns 256 1024 4096 --out results/probes/passkey_profile.json   # tokens right per stream
python src/probes.py attractor --runs swa_fb_seg32_noise0.3_nltanh_s0 --out results/probes/attractor_probe.json   # perturb the loop vector, contraction, Jacobian
python src/report.py swa --out reports/swa_report.html                          # self-contained HTML report (gdn | swa | fb | best)
python src/report.py best --out reports/best_report.html                        # presentation sub-report: the method, the best checkpoint, seed-average lines
```

wandb run names are `<model>_<overridden hyperparameters>_s<seed>`, e.g. `gdn_fixed_a0.95_s1`. The wandb key is read from a
git-ignored file `wandb_token` (or `$WANDB_API_KEY`).

**What differs from the reference recipe.** Run names list only the overrides of one launch, so every result file also stores its diff against a
frozen reference config of its model kind (`configs/ref/gdn_fixed_v1.yaml`, `swa_v1.yaml`, `swa_fb_v1.yaml`; never edited, hash-checked in
`tests/test_reference.py`; a new recipe gets a new file). The diff ignores only the seed, the wandb group, logging and the evaluation set / batch
(`config.IGNORED_KEYS`); the wandb run notes carry it too. Report rows are *configurations*: runs with the same diff. Runs that differ in anything
else are never averaged, and `reports/fb_report.html` ends with a table of every configuration with its diff, run names and per-seed numbers.

## Layout

```
data/    make_passkey_swar.py  generator (deterministic, per-example seeds)
         sanity_checks.py      oracle, mutation test, determinism, leakage, statistics
         upload_passkey_swar.py, tune_drift.py, vocab.json
src/     models.py (fixed-decay GDN, sliding-window attention, segment-feedback SWA), data.py, evaluate.py, train.py (resumable), graphed.py (CUDA-graph step),
         launch.py, run_queue.py (unattended queue of sweeps), aggregate.py, probes.py (checkpoint probes on fresh streams), report.py + report_groups.py +
         report_rounds.py (HTML reports; rows = configurations), config.py (configs, run names, diff against the frozen references)
checkpoints/ the best weights (2.3 MB each, manifest.json with hashes and metrics; `python scripts/backup_checkpoints.py <run>`)
configs/ experiment YAMLs, configs/ref/ frozen references      reports/ statistics + plans      results/ summary tables, plots, raw per-run json
tests/   python -m unittest discover -s tests -t .
```

Environment: PyTorch 2.6 + `flash-linear-attention` 0.5.2 (Triton 3.2), two 48 GB GPUs.
