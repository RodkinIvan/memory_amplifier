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
| 2 | GDN (learned decay), GDN with eigenvalues in [−1, 1], Mamba-2 | not started |
| 3 | GDN, 2 heads, per-head α statistics | not started |
| 4 | attractor cleanup in isolation (numpy/torch) | not started |

```bash
python src/launch.py --config configs/exp1.yaml --session exp1 --gpus 0 1 --procs-per-gpu 3 \
    --sweep model.alpha=1.0,0.999,0.995,0.99,0.98,0.95,0.9,0.8 seed=0,1,2     # tmux, resumable, logs to wandb "passkey_swar"
python src/launch.py --config configs/swa.yaml --session swa --gpus 0 1 --procs-per-gpu 3 \
    --sweep model.window=32,64,128 model.n_layers=2,4 seed=0,1,2              # sliding-window baseline
python src/aggregate.py                                                          # results/summary*.csv + plots
python src/probes.py passkey-length --runs swa_w128_L4_s0 --out results/probes/x.json   # probes on checkpoints (see --help)
python src/report.py swa --out reports/swa_report.html                          # self-contained HTML report
```

wandb run names are `<model>_<overridden hyperparameters>_s<seed>`, e.g. `gdn_fixed_a0.95_s1`. The wandb key is read from a
git-ignored file `wandb_token` (or `$WANDB_API_KEY`).

## Layout

```
data/    make_passkey_swar.py  generator (deterministic, per-example seeds)
         sanity_checks.py      oracle, mutation test, determinism, leakage, statistics
         upload_passkey_swar.py, tune_drift.py, vocab.json
src/     models.py (fixed-decay GDN, sliding-window attention), data.py, evaluate.py, train.py, launch.py, aggregate.py,
         probes.py (checkpoint probes on fresh streams), report.py (HTML reports), config.py
configs/ experiment YAMLs      reports/ statistics + plans      results/ summary tables, plots, raw per-run json
tests/   python -m unittest discover -s tests -t .
```

Environment: PyTorch 2.6 + `flash-linear-attention` 0.5.2 (Triton 3.2), two 48 GB GPUs.
