#!/usr/bin/env python
"""Collect results/raw/*.json into the summary tables and the Exp 1 trade-off plots.

    python src/aggregate.py                      # tables + plots for the fixed-decay baseline
    python src/aggregate.py --no-plots

Writes
  results/summary.csv        test split; columns model, alpha, seed, N, passkey_acc, ar_valid_acc, ar_expired_acc
  results/summary_val.csv    same columns, validation split
  results/summary_extra.csv  both splits, plus expired split into stale / never-written, counts, hyperparameters
  results/exp1_tradeoff.{png,pdf}        passkey exact match vs expired-AR accuracy, one point per alpha
  results/exp1_tradeoff_stale.{png,pdf}  same with stale-only expired queries (the ones that test forgetting)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SPEC_COLS = ["model", "alpha", "seed", "N", "passkey_acc", "ar_valid_acc", "ar_expired_acc"]
METRICS = ["passkey_acc", "ar_valid_acc", "ar_expired_acc", "ar_expired_stale_acc", "ar_expired_never_acc"]
COUNTS = ["n_examples", "n_valid", "n_expired_stale", "n_expired_never"]

# reference palette (dataviz skill): chart surface, ink, muted, hairlines, and the blue sequential ramp steps 250..700.
# The ramp is monotone in lightness, single-hue and its light end clears the surface (validate_palette.js --ordinal), but
# 8 levels cannot all pass the ordinal adjacent-step check (dL >= 0.06; this ramp fits ~5). That is acceptable because colour
# is only a redundant cue here: every mean point is directly labeled with its alpha, the neutral line orders the points along
# alpha, and the legend names each value.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]  # light -> dark


def load_runs(raw_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(Path(raw_dir).glob("*.json")):
        r = json.loads(f.read_text())
        cfg, kind = r["config"], r["config"]["model"]["kind"]
        variant = [k for k in r["explicit_keys"] if k not in ("seed", "model.kind", "model.alpha")]
        label = kind + "".join(f"_{C.SHORT.get(k, k.split('.')[-1])}{C.fmt(C.get_nested(cfg, k))}" for k in variant)
        runs.append({"name": r["name"], "model": label, "alpha": cfg["model"]["alpha"] if kind == "gdn_fixed" else np.nan,
                     "seed": cfg["seed"], "cfg": cfg, "rows": r["rows"], "final_train_loss": r["final_train_loss"]})
    return runs


def to_frame(runs: list[dict]) -> pd.DataFrame:
    recs = []
    for r in runs:
        mc, tc = r["cfg"]["model"], r["cfg"]["train"]
        for row in r["rows"]:
            recs.append({"split": row["split"], "model": r["model"], "alpha": r["alpha"], "seed": r["seed"], "N": row["N"],
                         **{k: row[k] for k in METRICS + COUNTS},
                         # model-specific hyperparameters; blank for the model family that does not have them
                         "norm_eps": mc.get("norm_eps"), "head_dim": mc.get("head_dim"), "window": mc.get("window"),
                         "n_layers": mc["n_layers"], "num_heads": mc["num_heads"],
                         "reach": mc["n_layers"] * (mc["window"] - 1) if mc.get("window") else None,
                         "d_model": mc["d_model"], "steps": tc["steps"], "final_train_loss": r["final_train_loss"], "run": r["name"]})
    df = pd.DataFrame(recs)
    return df.sort_values(["split", "model", "alpha", "seed", "N"], ascending=[True, True, False, True, True]).reset_index(drop=True)


def write_tables(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df[df.split == "test"][SPEC_COLS].to_csv(out / "summary.csv", index=False)
    df[df.split == "validation"][SPEC_COLS].to_csv(out / "summary_val.csv", index=False)
    df.to_csv(out / "summary_extra.csv", index=False)


def alpha_colors(alphas: list[float]) -> dict[float, str]:
    """One-hue ordinal ramp: alpha closest to 1 is darkest. Subsamples the 8-step ramp evenly when there are fewer."""
    order = sorted(alphas, reverse=True)
    pick = np.linspace(len(BLUE_RAMP) - 1, 0, len(order)).round().astype(int)
    return {a: BLUE_RAMP[i] for a, i in zip(order, pick)}


def cluster_points(pts: dict[float, tuple[float, float]], tol: float = 0.02) -> list[tuple[float, float, str, list[float]]]:
    """Group alphas whose mean points coincide (within tol on both axes) -> [(x, y, label, alphas)].

    Several alphas often land on exactly the same point (e.g. everything at 0% passkey); one label per group lists them all
    (long groups are compacted to "first ... last (n)") so no alpha is left unlabeled and no marker hides another."""
    groups: list[dict] = []
    for a in sorted(pts, reverse=True):
        x, y = pts[a]
        for g in groups:
            if abs(g["x"] - x) <= tol and abs(g["y"] - y) <= tol:
                g["alphas"].append(a)
                break
        else:
            groups.append({"x": x, "y": y, "alphas": [a]})
    out = []
    for g in groups:
        names = [f"{a:g}" for a in g["alphas"]]
        label = ", ".join(names) if len(names) <= 4 else f"{names[0]} \u2026 {names[-1]} ({len(names)})"
        out.append((g["x"], g["y"], "\u03b1 " + label, g["alphas"]))
    return out


def plot_tradeoff(df: pd.DataFrame, out_base: Path, model: str = "gdn_fixed", split: str = "test", x: str = "ar_expired_acc",
                  ns=(256, 1024, 4096, 16384), title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = df[(df.split == split) & (df.model == model)]
    ns = [n for n in ns if n in set(d.N)]
    if d.empty or not ns:
        print(f"nothing to plot for model={model} split={split}")
        return
    colors = alpha_colors(sorted(d.alpha.unique()))
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, len(ns), figsize=(3.7 * len(ns) + 0.6, 4.7), sharex=True, sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, ns):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        g = d[d.N == n]
        mean = g.groupby("alpha")[["passkey_acc", x, "ar_valid_acc"]].mean()
        for a in mean.index:  # faint dots: the individual seeds
            seeds = g[g.alpha == a]
            ax.scatter(seeds[x], seeds.passkey_acc, s=14, color=colors[a], alpha=0.35, linewidths=0, zorder=3)
        for gx, gy, label, members in cluster_points({a: (r[x], r.passkey_acc) for a, r in mean.iterrows()}):
            for a in sorted(members):  # lowest alpha first, so the highest alpha of a cluster ends up on top
                hollow = mean.loc[a, "ar_valid_acc"] < 0.9
                ax.scatter([mean.loc[a, x]], [mean.loc[a, "passkey_acc"]], s=85, zorder=5, facecolor=SURFACE if hollow else colors[a],
                           edgecolor=colors[a] if hollow else SURFACE, linewidths=2.0 if hollow else 1.5)
            right = gx > 0.55
            ax.annotate(label, (gx, gy), xytext=(-9 if right else 9, 9 if gy < 0.5 else -14), textcoords="offset points",
                        ha="right" if right else "left", fontsize=8.5, color=INK2, zorder=6)
        tok = int(round(g.N.iloc[0] * 3.75))
        ax.set_title(f"N = {n:,} writes  (~{tok / 1000:.0f}k tokens)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(-0.04, 1.06)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"])
        ax.set_xlabel("expired-query accuracy" if x == "ar_expired_acc" else "stale expired-query accuracy")
    axes[0].set_ylabel("passkey exact match")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=colors[a], markeredgecolor=SURFACE,
                      label=f"{a:g}") for a in sorted(colors, reverse=True)]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=SURFACE, markeredgecolor=MUTED,
                          markeredgewidth=1.8, label="hollow: ar_valid < 90%"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9, handletextpad=0.3,
                     columnspacing=1.2, title="decay alpha per token (darker = closer to 1)", title_fontsize=9)
    for t in leg.get_texts() + [leg.get_title()]:
        t.set_color(INK2)
    fig.suptitle(title or f"Fixed-decay GDN, {split} split: passkey vs expired-AR accuracy (mean of {d.seed.nunique()} seeds, faint dots = seeds; up and right is better)",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=C.ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=C.ROOT / "results")
    ap.add_argument("--model", default="gdn_fixed", help="model label to plot")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    runs = load_runs(a.raw)
    if not runs:
        sys.exit(f"no results in {a.raw}")
    df = to_frame(runs)
    write_tables(df, a.out)
    print(f"{len(runs)} runs, {df.model.nunique()} model variants -> {a.out}/summary*.csv")
    if not a.no_plots:
        plot_tradeoff(df, a.out / "exp1_tradeoff", model=a.model)
        plot_tradeoff(df, a.out / "exp1_tradeoff_stale", model=a.model, x="ar_expired_stale_acc", ns=(256, 1024, 4096, 16384),
                      title="Fixed-decay GDN, test split: passkey vs stale-expired accuracy (only stale queries test forgetting)")


if __name__ == "__main__":
    main()
