# Exp 1 plan — baseline trade-off curve (awaiting confirmation; nothing has been trained)

## Setup
- **Env:** conda `py311_pt2.6_cu12.4` (torch 2.6, triton 3.2, `flash-linear-attention` 0.5.2 installed into it). 2x RTX 6000 Ada (48 GB).
  Triton 3.2 is below fla's recommended 3.3; smoke tests (below) ran fine.
- **Model:** stock fla `GatedDeltaNetForCausalLM` blocks (RMSNorm, SwiGLU MLP ratio 4, tied embeddings), 2 layers, d_model 128,
  **1 head, head_dim 32, expand_v 1** (q/k/v all 32-d, 441k params), vocab 40. The GDN mixer is replaced by `FixedDecayGDN`
  (`src/models.py`): `g = log(alpha)` constant, everything else (short conv, beta gate, output gate, output RMSNorm) unchanged.
  alpha = 1.0 is the plain delta rule. Both layers use the same alpha.
- **Sweep:** alpha in {1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8} x 3 seeds = 24 runs.
- **Train data:** pooled `n32..n256` (400k examples), random batches of 32 (right-padded, PAD labels = -100; causal model so
  padding is inert). Loss = cross-entropy on supervised positions only (mean over supervised tokens).
- **Optimisation (untuned defaults):** AdamW lr 1e-3, betas (0.9, 0.95), wd 0.01, 500 warmup steps, cosine to 10%, grad-clip 1.0,
  30k steps (~2.4 epochs of the pool), fp32 params/activations (bf16 is not faster at this size). No checkpoint selection: the
  final model is evaluated once.
- **Eval:** every config n32..n16384, validation and test (1,000 examples each). Reported in `results/summary.csv` from **test**;
  validation goes to `results/summary_val.csv`. Passkey = exact match of all 4 tokens (teacher-forced == greedy for exact match);
  `ar_valid_acc`, `ar_expired_acc` = token accuracy on those positions. Sequences right-padded to a multiple of 4096 (causal model, so
  padding cannot influence earlier positions; logits agree up to float rounding, max diff ~3e-5 on logits of scale ~2; avoids
  per-length Triton recompiles: 3 ms/example at 15k tokens, 7.5 ms at 61k tokens once warm).
- **Extra (not in the requested columns, `results/summary_extra.csv`):** `ar_expired` split into `stale` vs `never-written`
  (derived from the tokens with the oracle), because only stale queries test forgetting, and n32/n64 expired queries are 100%/92%
  never-written.
- **Plot:** passkey exact-match (y) vs `ar_expired` accuracy (x), one point per alpha (mean of 3 seeds, seeds as faint dots),
  panels N = 256, 1024, 4096, 16384 -> `results/exp1_tradeoff.png`. Second version with stale-only expired accuracy.
- **Logging:** wandb project `passkey_swar` (train loss every 50 steps, validation metrics n32..n1024 every 2k steps, final table).
- **Config:** one YAML per experiment in `configs/`; `python src/train.py --config configs/exp1.yaml --alpha 0.95 --seed 0`.
- **Compute:** ~8 ms fwd+bwd per step -> ~6-8 min per run + ~1-2 min eval. 24 runs ~ 4 h sequential; ~1 h with several
  processes per GPU.

## What alpha means here (decay is per *token*, ~3.75 tokens per AR write)
The window w = 16 writes is ~60 tokens, 2w ~ 120 tokens; N = 256 / 4096 / 16384 writes ~ 1k / 15k / 61k tokens.

| alpha | alpha^60 (age w) | alpha^120 (age 2w) | passkey alpha^tokens, N=256 | N=1024 | N=4096 | N=16384 |
|---|---|---|---|---|---|---|
| 1.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| 0.999 | 0.94 | 0.89 | 0.38 | 2e-2 | 2e-7 | 2e-27 |
| 0.995 | 0.74 | 0.55 | 8e-3 | 4e-9 | ~0 | ~0 |
| 0.99 | 0.55 | 0.30 | 6e-5 | ~0 | ~0 | ~0 |
| 0.98 | 0.30 | 0.09 | 4e-9 | ~0 | ~0 | ~0 |
| 0.95 | 0.05 | 2e-3 | ~0 | ~0 | ~0 | ~0 |
| 0.9 | 2e-3 | 3e-6 | ~0 | ~0 | ~0 | ~0 |
| 0.8 | 2e-6 | 2e-12 | ~0 | ~0 | ~0 | ~0 |

Expected picture: alpha >= 0.995 barely separates age w from age 2w (cannot answer NULL for stale keys); alpha <= 0.9 has
almost nothing left even at age w; only alpha ~ 0.95-0.98 forgets on the right scale, and none of those keeps the passkey past
N ~ 256. Only alpha = 1.0 keeps the passkey at N = 16384 (if the delta rule does not erase it: 16 AR keys + 4 passkey slots in 32 dims).

## Calibration gate (before the 24 runs, ~15 min, needs your go-ahead as it is training)
GDN applies an RMSNorm to the mixer output (`o_norm`, eps 1e-5). It removes the magnitude of a decayed read-out, so "old =
small = NULL" is only expressible through the eps floor. That could make even a well-chosen alpha fail `ar_expired`, for a reason
unrelated to the trade-off, and would contaminate the sweep.
1. seed 0, alpha = 0.95 and 0.9: require `ar_valid` and `ar_expired(stale)` >= 95% at n256. If not, rerun with a larger
   `norm_eps` (1e-2 / 1e-1) or without `o_norm`, and use that for **all** alphas (the choice is already a config knob).
2. seed 0, alpha = 1.0: require passkey exact match >= 95% at n256 (capacity check for head_dim 32). If not, try head_dim 64.

## How it runs (confirmed by the user)
- Everything runs in **tmux** via `src/launch.py` (one window per worker, several workers per GPU, resumable: finished runs are skipped):
  `python src/launch.py --config configs/exp1.yaml --session exp1 --gpus 0 1 --procs-per-gpu 3 --sweep model.alpha=1.0,0.999,0.995,0.99,0.98,0.95,0.9,0.8 seed=0,1,2`
- Logging to wandb project `passkey_swar`, group = experiment. The run name is the model kind plus **every hyperparameter that
  is overridden for that run** plus the seed, e.g. `gdn_fixed_a0.95_s1`, `gdn_fixed_a0.95_eps0.01_s0` (`src/config.py`).
  The wandb key is read from the git-ignored `wandb_token` file (the repo is public), never printed or committed.
- Per-run results: `results/raw/<name>.json`; weights: `runs/<name>/model.pt` (git-ignored); logs: `logs/<session>/<name>.log`.
  `python src/aggregate.py` builds `results/summary*.csv` and the trade-off plots.

## Open points
- Exp 2 (later): fla's Mamba-2 needs `mamba_ssm` for its SSD kernels; it is not installed (needs a prebuilt wheel for torch 2.6 / cu12).
- Exp 2/3 GDN with learned decay uses the stock `gdn` builder already in `src/models.py` (`allow_neg_eigval=True` for Grazzi et al.).
