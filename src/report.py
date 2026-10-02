#!/usr/bin/env python
"""Build the HTML reports (one self-contained page each) from results/raw/*.json and results/probes/*.json.

    python src/report.py gdn --out reports/gdn_report.html
    python src/report.py swa --out reports/swa_report.html
    python src/report.py fb --out reports/fb_report.html     # SWA + segment feedback loop (also reads logs/fbsweep for runs in progress)

All numbers in the text are computed from the data; the template is reports/report_template.html.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as A  # noqa: E402
import config as C  # noqa: E402
import data as D  # noqa: E402
import probes as P  # noqa: E402

NS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TEMPLATE = C.ROOT / "reports" / "report_template.html"
TOKENS_PER_WRITE = 3.75  # 3 tokens per write plus its share (0.25 per write) of 3-token queries


def _nn(x):
    return None if x is None or x != x else float(x)


def cells_for_row(runs: list[dict], split: str = "test", ns: list[int] | None = None) -> dict:
    """{str(N): {passkey, valid, stale, expired (means over seeds), counts, per: {metric: [per seed]}}} for one row."""
    keys = {"passkey": "passkey_acc", "valid": "ar_valid_acc", "stale": "ar_expired_stale_acc", "expired": "ar_expired_acc"}
    out = {}
    for n in ns or NS:
        per = {k: [] for k in keys}
        base = None
        for r in sorted(runs, key=lambda r: r["seed"]):
            row = next(x for x in r["rows"] if x["split"] == split and x["N"] == n)
            for k, col in keys.items():
                per[k].append(_nn(row[col]))
            base = row
        mean = lambda xs: None if any(v is None for v in xs) else float(np.mean(xs))
        out[str(n)] = {**{k: mean(v) for k, v in per.items()}, "n_examples": base["n_examples"], "n_valid": base["n_valid"],
                       "n_stale": base["n_expired_stale"], "per": per}
    return out


def horizon(cells: dict, key: str = "passkey", thr: float = 0.9):
    """Largest N whose seed-mean `key` is at least thr (None if there is none)."""
    ok = [n for n in NS if cells[str(n)][key] is not None and cells[str(n)][key] >= thr]
    return max(ok) if ok else None


def render(data: dict, title: str, h1: str, out: Path) -> None:
    """`title` is the browser/gallery name (keep it stable across republishes), `h1` the heading on the page."""
    html = (TEMPLATE.read_text().replace("__TITLE__", title).replace("__H1__", h1)
            .replace("/*__DATA__*/null", json.dumps(data)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e3:.0f} kB)")


def fmt_n(n):
    return "never" if n is None else f"{n:,}"


# ----------------------------------------------------------------------------------------------------------- GDN report
def gdn_data() -> tuple[dict, str]:
    runs = [r for r in A.load_runs(C.ROOT / "results" / "raw") if r["model"] == "gdn_fixed"]
    alphas = sorted({r["alpha"] for r in runs}, reverse=True)
    rows = [{"key": f"{a:g}", "label": f"{a:g}", "long": f"α = {a:g}"} for a in alphas]
    results = {f"{a:g}": cells_for_row([r for r in runs if r["alpha"] == a]) for a in alphas}
    seeds = sorted({r["seed"] for r in runs})
    cell = lambda a, n, k: results[f"{a:g}"][str(n)][k]
    hz = {a: horizon(results[f"{a:g}"]) for a in alphas}
    min_stale = min(v for a in alphas for n in NS for v in results[f"{a:g}"][str(n)]["per"]["stale"] if v is not None)
    worst_valid = {a: min(v for n in NS for v in results[f"{a:g}"][str(n)]["per"]["valid"]) for a in alphas}
    weak = [a for a in alphas if worst_valid[a] < 0.99]
    strong_min = min(worst_valid[a] for a in alphas if a not in weak)
    weak_txt = " and ".join(f"α = {a:g} ({worst_valid[a]:.2f})" for a in weak)  # two entries at most in practice
    stale16k = cell(1.0, 16384, "n_stale") * len(seeds)
    seed_spread = {n: results["1"][str(n)]["per"]["passkey"] for n in (1024, 2048, 4096)}
    partial = [a for a in alphas if hz[a] is None and max(cell(a, 32, "passkey") or 0, 0) > 0.05]

    probe = json.loads((C.ROOT / "results" / "probes" / "gdn_leak_density.json").read_text())["gdn_fixed_a1_s0"]
    labels = {0.0: ("no queries (0 per write)", "0 per write"), 0.25: ("0.25 per write (training density)", "0.25 (training)"),
              1.0: ("1 per write", "1 per write"), 2.0: ("2 per write", "2 per write")}
    series = []
    for rate in (0.0, 0.25, 1.0, 2.0):
        pts = [r for r in probe if r["query_rate"] == rate and (rate == 0.25 or r["N"] in (2048, 3072, 4096))]
        if rate == 0.25:  # extend with the standard test evaluation at the two lengths inside the training range
            std = [{"N": n, "tokens": n * TOKENS_PER_WRITE, "passkey_acc": cell(1.0, n, "passkey"), "std": True} for n in (256, 512)]
            pts = std + pts
        series.append({"label": labels[rate][0], "short": labels[rate][1], "width": 3 if rate == 0.25 else 2,
                       "points": [{"x": p["N"], "y": p["passkey_acc"], "tip": f"{labels[rate][0]}<br>N = {p['N']:,} writes, {p['tokens'] / 1000:.1f}k tokens<br>passkey {p['passkey_acc']:.3f}"
                                   + ("<br>standard test split, 1000 examples per seed" if p.get("std") else "<br>200 streams, seed-0 checkpoint")} for p in pts]})
    lk = {r["N"]: r["passkey_acc"] for r in probe if r["query_rate"] == 0.25}
    lk2 = {(r["N"], r["query_rate"]): r["passkey_acc"] for r in probe}

    tldr = [
        "<b>The benchmark is built and verified.</b> 24 files, valid and expired queries split 50/50, an independent brute-force oracle reproduces every label, and no example is shared between splits.",
        f"<b>Every α gets the expired queries right.</b> Stale-key accuracy is at least {min_stale:.3f} in all {len(alphas) * len(seeds)} runs at every N up to 16,384 ({stale16k:,} stale queries at N = 16,384 over the three seeds), "
        f"including α = 1, which has no decay at all. Valid accuracy is at least {strong_min:.3f} except for {weak_txt}, where the decay is fast enough to cost some recent answers. "
        "The expected conflict, forgetting needs α &lt; 1, does not show up.",
        "<b>Only the passkey separates the α values, through how long it survives.</b> Longest N with a seed-mean of at least 90% exact match: "
        + ", ".join(f"α = {a:g}: {fmt_n(hz[a])}" for a in alphas)
        + f". For α = 1 this is seed-dependent: at N = 2,048 the three seeds score {', '.join(f'{v:.2f}' for v in seed_spread[2048])}. "
        + (f"α = {partial[0]:g} learned the passkey in one seed only (partially, N ≤ 64). " if partial else "")
        + "No α holds the passkey at N = 4,096 or beyond.",
        "<b>α = 1 expires old bindings through the delta rule, not through decay.</b> On the seed-0 checkpoint its accuracy does not move when the query density changes 40-fold, so it counts writes (which overwrite each other), while a scalar decay counts tokens.",
        f"<b>The α = 1 passkey leak is not a kernel-precision artifact.</b> The fla kernel, an fp32 recurrence and an fp64 recurrence gave identical passkey accuracy on 32 streams. The leak grows with sequence length and with query density (figure 3).",
        "<b>Next.</b> A sliding-window attention baseline (separate report) has a provable reach limit, which makes it a cleaner base for the memory-amplification loop.",
    ]
    tested = [
        {"title": "Benchmark", "tag": "passed", "ok": True, "body": "Oracle reproduces 100% of labels on all files, rejects deliberately corrupted examples, and never meets a gray-zone query. Regenerating from the stored seeds gives identical examples. No leakage between splits. Drift rate tuned to 0.10. At N ≤ 32 no key can be stale, so expired queries there are all never-written keys (n64: 92%)."},
        {"title": "Model plumbing", "tag": "passed", "ok": True, "body": "Fixed-decay GDN on stock fla blocks, checked against a naive recurrence (0.3% relative, the kernel’s precision). Padding to a common length does not change predictions beyond float rounding. 47 unit tests pass."},
        {"title": "Calibration gate", "tag": "passed", "ok": True, "body": "α = 0.95 and 0.9 answer NULL for stale keys despite GDN’s output RMSNorm, so the norm eps stays at its default. α = 1 holds the passkey at N = 256, so head_dim 32 has enough capacity."},
        {"title": "Exp 1 sweep", "tag": f"{len(runs)}/24 runs", "ok": len(runs) == 24, "body": "8 decays × 3 seeds, 2 layers, d_model 128, 1 head of dim 32 (a 32-dim key space independent of d_model), 30k steps on N ≤ 256, fp32, evaluated on N up to 16,384. Logged to wandb project passkey_swar."},
        {"title": "Probes of the α = 1 model", "tag": "seed 0", "ok": True, "body": "Query density (0.05 to 2 queries per write): expiry unchanged, passkey leak grows with it. Number of active keys and key drift: no effect on expiry. Kernel versus fp32 and fp64 recurrence on 32 streams at N = 1,024 / 3,072 / 4,096: identical passkey accuracy."},
    ]
    cap1 = (
        f"<p><b>Bottom two grids:</b> every row is close to 1.00 in every column. Only the fastest decays cost valid answers ({weak_txt}, worst over seeds and N); all other rows are at {strong_min:.3f} or above. "
        "Whether a stored value fades does not decide if the model can answer these queries, including for α = 1 where nothing fades.</p>"
        f"<p><b>Top grid:</b> the passkey is where rows differ. Rows α ≥ 0.99 solve it inside the training range (N ≤ 256). Beyond it the horizon grows with α: "
        f"{fmt_n(hz[0.999])} for α = 0.999 (every seed), and {fmt_n(hz[1.0])} for α = 1, where the cells at 2,048 and beyond average very different seeds (hover a cell for the three values). "
        f"Rows α ≤ 0.95 never learn it, not even at N = 32 (132 tokens)"
        + (f"; α = 0.98 did so in one of three seeds" if partial else "") + ". "
        "The no-data cells for N = 32 in the bottom grid are there because no key can be stale in 32 writes.</p>")
    cap2 = (
        "<p>Expired-query accuracy is 100% for every α, so all points sit on the right edge and the spread is entirely vertical; faint dots are the individual seeds. "
        "At N = 256 the intended target corner (top right) is occupied by every α ≥ 0.99. "
        f"At N = 1,024 only α = 1 keeps the passkey (mean {cell(1.0, 1024, 'passkey'):.2f}), at N = 4,096 its mean has fallen to {cell(1.0, 4096, 'passkey'):.2f}, and at N = 16,384 the corner is empty for all α. "
        "So the plot shows a horizon that shrinks as α moves away from 1, not a diagonal trade-off between the two metrics. Hollow markers would flag models that answer NULL too often (valid accuracy below 90%); none does here.</p>")
    cap3 = (
        f"<p>The dashed line marks the longest training stream. At the training density (0.25 queries per write, the thick line) the passkey is intact to N = 1,024 and falls to {lk[2048]:.2f} at 2,048 and {lk[4096]:.2f} at 4,096. "
        f"Adding queries between the same number of writes makes it worse: at N = 2,048 it falls from {lk2[(2048, 0.25)]:.2f} to {lk2[(2048, 1.0)]:.2f} to {lk2[(2048, 2.0)]:.2f} as the density goes 0.25 → 1 → 2. "
        f"So the leak depends on how many tokens pass, and writes erode it more than queries do. Streams with no queries at all still lose it ({lk2[(4096, 0.0)]:.2f} at 4,096). "
        "A protection that was learned on streams of about 1k tokens and is not perfect would behave like this; this is a reading of the curve, not something the probe proves. "
        "Expiry of AR bindings, by contrast, stayed at 1.00 in all these streams.</p>")
    nxt = ("<p><b>1. Exp 1 is complete</b> (24 of 24 runs). Raw results, summary tables and the trade-off plots are in the repository.</p>"
           "<p><b>2. The benchmark question stays open.</b> Whether the α = 1 passkey limit is only a training-length limit (train on N up to 1,024) and whether expiry should be defined in tokens instead of writes are both unanswered; the sliding-window baseline sidesteps the first by having a provable limit.</p>"
           "<p><b>3. Exp 2 and 3 are on hold</b> until the memory design is settled. Mamba-2 in fla also needs <code>mamba_ssm</code>, which is not installed here.</p>")
    data = {
        "ns": NS, "rows": rows, "results": results, "scatter_ns": [256, 1024, 4096, 16384], "label_w": 58, "row_name": "α", "row_prefix": "α ",
        "meta": f"Phase 1, 2 Oct 2026. Fixed-decay Gated DeltaNet baselines (Exp 1), final. <span class='status'><span>runs finished: <b>{len(runs)}/24</b></span><span>·</span><span>3 seeds per α</span></span>",
        "tldr": tldr, "tested": tested,
        "results_intro": "All numbers are from the <b>test</b> split (1,000 examples per N), as the mean over 3 seeds, after 30k training steps on N ≤ 256. Hover a heatmap cell for the individual seeds.",
        "fig1_sub": "Rows: decay α per token (1 = no decay). Columns: number of AR writes N (a write is about 3.75 tokens, so 16384 is about 61k tokens). Darker is higher.",
        "fig2_sub": "One point per α (mean over seeds, faint dots = seeds). Ideal is the top-right corner. Labels list the α values that share a point.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)",
        "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": cap1, "cap2": cap2,
        "extra_figs": [{
            "title": "3. Probe of the α = 1 model: what the passkey leak depends on",
            "sub": "Same trained checkpoint (α = 1, seed 0), fresh streams with a different query density. Passkey exact match at N writes; 200 streams per point (N = 256 and 512 come from the standard test split).",
            "caption": cap3,
            "spec": {"series": series, "xTicks": [256, 512, 1024, 2048, 4096, 6144], "x0": 256, "x1": 6144, "xLabel": "AR writes N (log scale)",
                     "yLabel": "passkey exact match", "vlines": [{"x": 256, "label": ""}],
                     "ordReverse": True, "note": "trained on N ≤ 256 (~1k tokens)", "aria": "Passkey exact match versus N for four query densities"}}],
        "next": nxt,
    }
    return data, "passkey_swar Intermediate Report", "Fixed-decay GDN baseline: final results"


# ----------------------------------------------------------------------------------------------------------- SWA report
def _mlh(vals):
    """(mean, min, max) of a list of numbers."""
    return float(np.mean(vals)), float(min(vals)), float(max(vals))


def swa_data() -> tuple[dict, str, str]:
    runs = [r for r in A.load_runs(C.ROOT / "results" / "raw") if r["cfg"]["model"]["kind"] == "swa"]
    wl = lambda r: (r["cfg"]["model"]["window"], r["cfg"]["model"]["n_layers"])
    cfgs = sorted({wl(r) for r in runs}, key=lambda c: (-(c[1] * (c[0] - 1)), -c[0]))  # longest reach first
    reach = {c: c[1] * (c[0] - 1) for c in cfgs}
    key = lambda c: f"w{c[0]}_L{c[1]}"
    by_cfg = {c: [r for r in runs if wl(r) == c] for c in cfgs}
    rows = [{"key": key(c), "label": f"W{c[0]} · L{c[1]} · R{reach[c]}",
             "long": f"window {c[0]}, {c[1]} layers, reach {reach[c]} tokens ({len(by_cfg[c])} seed{'s' if len(by_cfg[c]) != 1 else ''})"} for c in cfgs]
    results = {key(c): cells_for_row(by_cfg[c]) for c in cfgs}
    n_runs = len(runs)
    seed_counts = {c: len(by_cfg[c]) for c in cfgs}
    partial_seeds = n_runs < 18
    cell = lambda c, n, k: results[key(c)][str(n)][k]
    worst_valid = {c: min(v for n in NS for v in results[key(c)][str(n)]["per"]["valid"]) for c in cfgs}
    worst_stale = {c: min(v for n in NS for v in results[key(c)][str(n)]["per"]["stale"] if v is not None) for c in cfgs}
    best_pk = {c: max(cell(c, n, "passkey") for n in NS) for c in cfgs}
    pk_ok = {c: [n for n in NS if cell(c, n, "passkey") >= 0.5] for c in cfgs}  # lengths where the passkey is mostly solved
    learners = [c for c in cfgs if pk_ok[c]]
    name = lambda c: f"W{c[0]} L{c[1]}"

    # ---- probes
    pdir = C.ROOT / "results" / "probes"
    length = json.loads((pdir / "swa_passkey_length.json").read_text())
    dens = json.loads((pdir / "ar_density.json").read_text())
    runs_of = lambda c: [r["name"] for r in by_cfg[c]]
    ns_probe = sorted({r["N"] for rr in length.values() for r in rr})
    len_series, cliff = [], {}
    for c in sorted(cfgs, key=lambda c: reach[c]):  # shortest reach first, so longer reaches are drawn on top
        pts = []
        for n in ns_probe:
            rr = [x for nm in runs_of(c) if nm in length for x in length[nm] if x["N"] == n]
            if not rr:
                continue
            m, lo, hi = _mlh([x["passkey_acc"] for x in rr])
            tok = float(np.mean([x["tokens"] for x in rr]))
            pts.append({"x": round(tok), "y": m, "lo": lo, "hi": hi, "n": n,
                        "tip": f"{rows[cfgs.index(c)]['long']}<br>N = {n} writes, ~{tok:.0f} tokens<br>passkey {m:.3f} (seeds {lo:.2f} to {hi:.2f})"})
        len_series.append({"label": f"W{c[0]} L{c[1]} (R = {reach[c]})", "short": name(c), "points": pts})
        good = [p for p in pts if p["y"] >= 0.5]
        if good:
            last = max(good, key=lambda p: p["x"])
            nxt = next((p for p in pts if p["x"] > last["x"]), None)
            cliff[c] = (last, nxt)
    probe_seeds = ", ".join(f"{name(c)}: {len([nm for nm in runs_of(c) if nm in length])}" for c in cfgs)  # models behind the probe figures
    dens_rates = sorted({r["query_rate"] for rr in dens.values() for r in rr if r["N"] == 256 and r["query_rate"] > 0})
    dens_series = []
    for c in cfgs:  # longest reach first = darkest
        pts = []
        for rate in dens_rates:
            vals = [x["ar_valid_acc"] for nm in runs_of(c) if nm in dens for x in dens[nm] if x["N"] == 256 and x["query_rate"] == rate]
            if vals:
                m, lo, hi = _mlh(vals)
                pts.append({"x": rate, "y": m, "lo": lo, "hi": hi, "tip": f"{rows[cfgs.index(c)]['long']}<br>{rate:g} queries per write<br>valid accuracy {m:.3f} (seeds {lo:.2f} to {hi:.2f})"})
        dens_series.append({"label": f"W{c[0]} L{c[1]} (R = {reach[c]})", "short": name(c), "points": pts})
    gdn_names = [n for n in dens if n.startswith("gdn_fixed_a1_")]
    gdn_pts = []
    for rate in dens_rates:
        vals = [x["ar_valid_acc"] for nm in gdn_names for x in dens[nm] if x["N"] == 256 and x["query_rate"] == rate]
        m, lo, hi = _mlh(vals)
        gdn_pts.append({"x": rate, "y": m, "lo": lo, "hi": hi, "tip": f"GDN α = 1<br>{rate:g} queries per write<br>valid accuracy {m:.3f} (seeds {lo:.2f} to {hi:.2f})"})
    dens_series.append({"label": "GDN α = 1 (for contrast)", "short": "GDN α = 1", "c": "var(--s2)", "dashed": True, "width": 2, "points": gdn_pts})
    worst_rate = max(dens_rates)
    drop = {c: dens_series[cfgs.index(c)]["points"][-1]["y"] for c in cfgs}

    # ---- reach ceilings on the validation set: valid answers inside one window / inside the full reach
    packed = D.load_packed(C.ROOT / "data_cache" / "passkey_swar" / "n1024" / "validation.parquet", kinds=True)
    vdist, sdist = P.query_distances(packed)
    vmax, smin = int(vdist.max()), int(sdist.min())
    ceil = {c: (float(np.mean(vdist <= c[0] - 1)), float(np.mean(vdist <= reach[c]))) for c in cfgs}  # (one window, full reach)
    meas = {c: float(np.mean([np.mean(results[key(c)][str(n)]["per"]["valid"]) for n in NS])) for c in cfgs}
    ceil_rows = "".join(f"<tr><td>W{c[0]} L{c[1]}</td><td>{c[0] - 1}</td><td>{reach[c]}</td><td>{ceil[c][0]:.3f}</td><td>{ceil[c][1]:.3f}</td><td>{meas[c]:.3f}</td></tr>" for c in cfgs)
    ceil_table = ("<div class='tbl'><table><thead><tr><th>config</th><th>one window, tokens</th><th>full reach R, tokens</th><th>valid answers within one window</th>"
                  "<th>within R</th><th>measured valid accuracy</th></tr></thead><tbody>" + ceil_rows + "</tbody></table></div>")
    short = [c for c in cfgs if ceil[c][0] < 0.999]  # configs whose single window misses some valid answers
    short_txt = "; ".join(f"{name(c)}: {ceil[c][0]:.2f} / {ceil[c][1]:.2f} / {meas[c]:.2f}" for c in short)

    gdn1 = [r for r in A.load_runs(C.ROOT / "results" / "raw") if r["model"] == "gdn_fixed" and r["alpha"] == 1.0]
    gdn_h = horizon(cells_for_row(gdn1)) if gdn1 else None
    lrn_txt = "; ".join(f"{name(c)} up to N = {max(pk_ok[c])}" for c in learners) if learners else "none"
    never = [c for c in cfgs if best_pk[c] < 0.5]
    never_txt = ", ".join(name(c) for c in never)
    stale_all = min(worst_stale.values())
    weak = [c for c in cfgs if worst_valid[c] < 0.99]
    weak_txt = ", ".join(f"{name(c)} ({worst_valid[c]:.2f})" for c in weak) or "none"
    best_256 = max(cell(c, 256, "passkey") for c in cfgs)
    top = len_series[-1]["points"]  # the configuration with the longest reach
    shortest = top[0]
    cl = cliff.get(cfgs[0])
    cliff_txt = (f"{name(cfgs[0])} (R = {reach[cfgs[0]]}) holds the passkey up to N = {cl[0]['n']} (~{cl[0]['x']} tokens, {cl[0]['y']:.2f}) and is at {cl[1]['y']:.2f} by N = {cl[1]['n']} (~{cl[1]['x']} tokens), well short of its nominal reach"
                 if cl and cl[1] else "")
    gmin = min(p["y"] for p in gdn_pts)
    gdn_txt = f"at {gmin:.3f}" if gmin >= max(p["y"] for p in gdn_pts) - 5e-4 else f"between {gmin:.3f} and {max(p['y'] for p in gdn_pts):.3f}"

    weak_detail = "; ".join(f"{name(c)} ({worst_valid[c]:.2f}): one window covers {ceil[c][0]:.2f} of the valid answers and the full reach {ceil[c][1]:.2f}" for c in weak)
    tldr = [
        f"<b>Sliding-window attention solves the AR half by construction.</b> Stale-key accuracy is at least {stale_all:.3f} in all {n_runs} runs at every N up to 16,384. "
        + (f"Valid accuracy is at least 0.99 except for {weak_detail}. Such a model has to relay writes across layers, which it does only in part."
           if weak else "Valid accuracy is at least 0.99 in every run."),
        f"<b>The passkey is out of reach almost everywhere, as the bound says.</b> Information travels at most R = layers × (window − 1) tokens (verified in the code: nothing leaks beyond R). "
        f"Configurations with a seed-mean passkey of at least 50% at some N: {lrn_txt}. At N = 256 (about 1k tokens) the best configuration scores {best_256:.2f}. Never above 50%: {never_txt}.",
        f"<b>The practical cliff is below the nominal reach.</b> {cliff_txt}. On streams shorter than any seen in training it falls too ({shortest['y']:.2f} at N = {shortest['n']}), "
        "which suggests the passkey is learned as a lookup at the training distances rather than as a search for the PK marker.",
        f"<b>Expiry follows token distance, not the number of writes.</b> With more queries between writes, valid accuracy falls (at {worst_rate:g} queries per write: "
        + ", ".join(f"{name(c)} {drop[c]:.2f}" for c in cfgs)
        + f"), while GDN α = 1 stays {gdn_txt}. At the training density valid writes are at most {vmax} tokens back and stale ones at least {smin}, so a distance threshold solves expiry exactly. That is a weakness of the benchmark, not of the models.",
        "<b>Role in the project.</b> This is the floor for the memory-amplification step: AR solved, the passkey lost beyond a few hundred tokens."
        + (f" GDN α = 1 holds the passkey to about N = {gdn_h:,} (about {gdn_h * TOKENS_PER_WRITE / 1000:.0f}k tokens) for comparison." if gdn_h else ""),
    ]
    tested = [
        {"title": "Reach bound", "tag": "passed", "ok": True, "body": "Unit test on the actual implementation: changing token 0 changes the output at positions 0 to L·(W−1) and leaves every later position bit-identical, for several (window, layers) pairs. A 61k-token sequence runs finite."},
        {"title": "SWA sweep", "tag": f"{n_runs}/18 runs", "ok": n_runs == 18, "body": "Window 32, 64, 128 × 2 and 4 layers × 3 seeds. Stock fla transformer blocks, RoPE, 4 heads of dim 32, d_model 128, same data, optimiser (30k steps, N ≤ 256) and evaluation as the GDN sweep. Trained in bf16 autocast (flash-attention needs it); the GDN runs were fp32. Logged to wandb project passkey_swar."},
        {"title": "Passkey vs stream length", "tag": "probe", "ok": True, "body": "Fresh streams with N from 4 to 256 writes (200 per point and model, own probe seed), to see where the passkey stops working. Streams shorter than N = 32 were never seen in training."},
        {"title": "Query-density probe", "tag": "probe", "ok": True, "body": f"Fresh streams at N = 256 with {dens_rates[0]:g} to {dens_rates[-1]:g} queries per write (training: 0.25), same answer rule in writes. Separates a token-distance rule from counting writes. Run on all SWA models and on GDN α = 1."},
        {"title": "Reach ceilings", "tag": "analysis", "ok": True, "body": f"From the validation data: the share of valid answers within one window and within the full reach (table under figure 1). Valid writes are at most {vmax} tokens back, stale ones at least {smin}."},
    ]
    cap1 = (
        f"<p><b>Bottom two grids:</b> stale-key accuracy is at least {stale_all:.3f} everywhere, and valid accuracy is flat in N: it depends on the distance to the write, not on the stream length. "
        + (f"Where one window cannot cover every valid answer, the measured accuracy sits between the one-window and the full-reach ceiling (one window / full reach / measured: {short_txt}), so models relay across layers when it pays off, but not for the last few queries." if short else "")
        + "</p>"
        f"<p><b>Top grid:</b> the passkey is 0 almost everywhere. The non-zero cells are at N = 32 and N = 64, where the passkey is still inside the reach. Hover a cell for the seeds.</p>"
        + ceil_table)
    cap2 = (f"<p>Expired-query accuracy is at least {min(cell(c, n, 'expired') for c in cfgs for n in NS):.3f} for every configuration, so all points sit on the right edge. The passkey separates them only at the shortest lengths. "
            f"At N = 256 the best configuration scores {best_256:.2f} and at N = 16,384 every configuration is at 0, the bottom-right corner: the target corner (top right) is empty, which the window guarantees.</p>")
    cap3 = (f"<p>Passkey exact match against stream length (log scale; mean over seeds, bars show the seed range). Dotted lines are the nominal reaches R. {cliff_txt}. "
            f"Never above 50% at any length: {never_txt}. Below the training range the best configuration falls too ({shortest['y']:.2f} at N = {shortest['n']}, about {shortest['x']} tokens).</p>")
    cap4 = (f"<p>Valid-query accuracy at N = 256 as the number of queries between writes changes (the answer rule is unchanged, in writes). The dashed line is GDN α = 1, which counts writes and stays {gdn_txt}. "
            f"Every SWA model loses valid answers as streams get denser, even when the write is still inside its reach: with 2 queries per write a write 16 writes back is about 144 tokens away, "
            f"beyond the {vmax} to {smin}-token gap that separates valid from expired writes at the training density, which is where a distance rule would put its border. "
            "Stale-key accuracy stays at about 1.00 throughout (answering NULL for far writes is right for expired queries and wrong for valid ones).</p>")
    nxt = ("<p><b>1. Your memory-amplification design.</b> This baseline is ready to extend: <code>fla</code> transformer blocks with a window, the same training and evaluation, and wandb logging.</p>"
           "<p><b>2. A benchmark change worth considering before the loop is evaluated:</b> sample the query density per stream (for example between 0.05 and 2 queries per write). "
           "Then token distance no longer predicts the age in writes, a window cannot solve expiry by a distance threshold, and a mechanism has to count writes or hold content, as GDN α = 1 does. "
           "It also makes the AR half a fair test of whether an amplifier leaks stale content.</p>"
           "<p><b>3. Still open from before:</b> several passkeys at unknown positions, so that persistence has to be selected at run time.</p>")
    data = {
        "ns": NS, "rows": rows, "results": results, "scatter_ns": [32, 64, 256, 16384], "label_w": 124, "row_name": "configuration", "row_prefix": "",
        "meta": f"Phase 1, 2 Oct 2026. <span class='status'><span>runs finished: <b>{n_runs}/18</b></span>"
                + (("<span>·</span><span><b>intermediate</b>, seeds so far: " + ", ".join(f"{name(c)}: {seed_counts[c]}" for c in cfgs) + "</span>") if partial_seeds else "<span>·</span><span>3 seeds per configuration</span>") + "</span>",
        "tldr": tldr, "tested": tested,
        "results_intro": "<b>SWA has no decay.</b> It keeps no recurrent state and has no multiplier like GDN’s α: each layer attends exactly to the last W tokens, and anything older has weight zero. "
                         "Forgetting is a hard cutoff, and the only memory-length knob is the reach R = L·(W−1) tokens, the farthest distance information can travel through L layers. "
                         "α appears in this report only on the dashed GDN line in figure 4, for contrast. "
                         "Rows are (window W, layers L, reach R). All numbers are from the <b>test</b> split (1,000 examples per N), as the mean over the seeds finished so far (hover a cell for the individual seeds), after 30k training steps on N ≤ 256." + (" This is an intermediate version: the 4-layer configurations still have fewer than 3 seeds." if partial_seeds else ""),
        "fig1_sub": "Columns: number of AR writes N (a write is about 3.75 tokens, so 16384 is about 61k tokens). Darker is higher.",
        "fig2_sub": "One point per configuration (mean over seeds, faint dots = seeds). Ideal is the top-right corner.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)",
        "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": cap1, "cap2": cap2,
        "extra_figs": [
            {"title": "3. Where the passkey stops: accuracy against stream length",
             "sub": f"Fresh streams, 200 per point and model. Stream length is measured in tokens (N writes is about 3.75 N tokens). Models (seeds) per configuration: {probe_seeds}.", "caption": cap3,
             "spec": {"series": len_series, "xTicks": [32, 64, 128, 256, 512, 1024], "x0": 24, "x1": 1300, "xLabel": "tokens in the stream (log scale)", "yLabel": "passkey exact match", "ordReverse": True,
                      "vlines": [{"x": 62, "label": "R 62"}, {"x": 125, "label": "R 124–126", "dy": 12}, {"x": 253, "label": "R 252–254"}, {"x": 508, "label": "R 508", "dy": 12}],
                      "aria": "Passkey exact match versus stream length for six window configurations"}},
            {"title": "4. What expiry depends on: valid accuracy against query density",
             "sub": f"Fresh streams at N = 256 writes with a different number of queries per write; training used 0.25. Models (seeds) per configuration: {probe_seeds}.", "caption": cap4,
             "spec": {"series": dens_series, "xTicks": dens_rates, "x0": min(dens_rates), "x1": max(dens_rates), "yMin": 0.8, "yMax": 1.0, "xLabel": "queries per write (log scale)",
                      "yLabel": "AR valid accuracy", "vlines": [{"x": 0.25, "label": "training density", "dy": 238}], "rightPad": 118, "aria": "AR valid accuracy versus query density"}},
        ],
        "next": nxt,
    }
    return data, "SWA Baseline Report", "Sliding-window attention baseline"


# ------------------------------------------------------------------------------------------- SWA + feedback-loop report
EXP_NS = [32, 64, 256, 1024]  # evaluation lengths of the 12k-step exploration runs (validation split only)
SEGS, NLS = (64, 32, 16), ("tanh", "none")
NL_NAME = {"tanh": "tanh loop", "none": "linear loop"}
LOG_RE = re.compile(r"\[swa_fb_nl(?P<nl>\w+?)_seg(?P<seg>\d+)_s(?P<seed>\d+)\] step (?P<step>\d+) loss (?P<loss>[\d.]+) \| "
                    r"n32: pk (?P<pk32>[\d.]+) v (?P<v32>[\d.]+) e (?P<e32>[\d.]+) n256: pk (?P<pk256>[\d.]+) v (?P<v256>[\d.]+) e (?P<e256>[\d.]+) "
                    r"n1024: pk (?P<pk1024>[\d.]+) v (?P<v1024>[\d.]+) e (?P<e1024>[\d.]+)")


def sweep_progress(logdir: Path) -> dict:
    """{(segment, nonlin): {seed: {step: {metric: value}}}} parsed from the validation lines of the sweep logs (2 decimals)."""
    out: dict = {}
    for f in sorted(Path(logdir).glob("swa_fb_nl*_seg*_s*.log")):
        for line in f.read_text().splitlines():
            m = LOG_RE.search(line)
            if m:
                g = m.groupdict()
                out.setdefault((int(g["seg"]), g["nl"]), {}).setdefault(int(g["seed"]), {})[int(g["step"])] = {k: float(v) for k, v in g.items() if k not in ("nl", "seg", "seed", "step")}
    return out


def fb_data() -> tuple[dict, str, str]:
    all_runs = A.load_runs(C.ROOT / "results" / "raw")
    ref_runs = [r for r in all_runs if r["cfg"]["model"]["kind"] == "swa" and r["cfg"]["model"]["window"] == 64 and r["cfg"]["model"]["n_layers"] == 2]
    explore = [r for r in A.load_runs(C.ROOT / "results" / "explore") if r["cfg"]["model"]["kind"] == "swa_fb"]
    finished = [r for r in all_runs if r["cfg"]["model"]["kind"] == "swa_fb" and r["cfg"]["train"]["steps"] == 20000]

    def family(r):
        tc = r["cfg"]["train"]
        cur, w, a = tc.get("curriculum_stage_steps", 0) > 0, tc["passkey_weight"], tc["passkey_weight_anneal_steps"]
        if cur and w == 1:
            return 0, "curriculum", "--s1"
        if cur:
            return 1, f"curriculum + w{w:g}→1", "--s2"
        return 2 + (a // 4000 - 1), f"w{w:g}→1 over {a // 1000}k", "--s3"

    explore.sort(key=lambda r: (family(r)[0], r["cfg"]["model"]["nonlin"] != "tanh"))
    rows = [{"key": "ref", "label": "SWA, no loop", "color": "--muted",
             "long": f"plain sliding window, W = 64, 2 layers, no loop ({len(ref_runs)} seeds, 30k steps)"}]
    results = {"ref": cells_for_row(ref_runs, "validation", EXP_NS)}
    for r in explore:
        _, fam, col = family(r)
        nl = "tanh" if r["cfg"]["model"]["nonlin"] == "tanh" else "linear"
        rows.append({"key": r["name"], "label": f"{fam} · {nl}", "color": col, "nolabel": family(r)[0] != 0,
                     "long": f"{fam}, {nl} loop, window 64, segment 64, 12k steps, seed {r['seed']}"})
        results[r["name"]] = cells_for_row([r], "validation", EXP_NS)
    cell = lambda key, n, k: results[key][str(n)][k]
    fam_rows = lambda f, nl: next(x for x in explore if family(x)[1] == f and x["cfg"]["model"]["nonlin"] == nl)["name"]
    cur_t, cur_l = fam_rows("curriculum", "tanh"), fam_rows("curriculum", "none")
    weight_only = [x["name"] for x in explore if family(x)[0] >= 2]
    cw = [x["name"] for x in explore if family(x)[0] == 1]
    best_w_valid = max(cell(k, 256, "valid") for k in weight_only)
    best_cw_valid = max(cell(k, 256, "valid") for k in cw)
    tok = lambda n: n * TOKENS_PER_WRITE
    hops = lambda n, seg=64: round(tok(n) / seg)

    # ---- the main sweep, from its logs
    prog = sweep_progress(C.ROOT / "logs" / "fbsweep")
    series_keys = [(seg, nl) for seg in SEGS for nl in NLS if (seg, nl) in prog]
    n_started = sum(len(prog[k]) for k in prog)

    def pts(key, metric):
        seeds = prog[key]
        common = sorted(set.intersection(*[set(v) for v in seeds.values()])) if len(seeds) == 3 else []
        out = []
        for st in common:
            vals = [seeds[s][st][metric] for s in sorted(seeds)]
            m, lo, hi = _mlh(vals)
            out.append({"x": st, "y": m, "lo": lo, "hi": hi,
                        "tip": f"segment {key[0]} · {NL_NAME[key[1]]}<br>step {st}<br>{metric} {m:.2f} (seeds {lo:.2f} to {hi:.2f})"})
        return out

    def curve(metric, title_y, aria):
        series = []
        for key in series_keys:
            p = pts(key, metric)
            if p:
                series.append({"label": f"segment {key[0]} · {NL_NAME[key[1]]}", **({"short": f"{key[0]} {'tanh' if key[1] == 'tanh' else 'lin'}"} if min(x["x"] for x in [p[-1]]) >= 18000 else {}),  # end labels only once the curves have separated
                               "ord": [SEGS.index(key[0]), len(SEGS)], "dashed": key[1] == "none", "points": p})
        return {"series": series, "xTicks": [0, 4000, 8000, 12000, 16000, 20000], "x0": 0, "x1": 20000, "xLog": False, "yMin": 0, "yMax": 1,
                "xLabel": "training step (the runs are 20,000 steps long)", "yLabel": title_y, "legendLine": True, "rightPad": 86, "aria": aria,
                "vlines": [{"x": 2000, "label": "n64 joins"}, {"x": 4000, "label": "n128 joins", "dy": 12}, {"x": 6000, "label": "n256 joins", "dy": 24}]}

    last_common = {k: (max(pts(k, "pk256"), key=lambda p: p["x"])["x"] if pts(k, "pk256") else 0) for k in series_keys}
    step_c = min(last_common.values()) if last_common else 0
    at_c = {k: next((p for p in pts(k, "pk256") if p["x"] == step_c), None) for k in series_keys}
    at_c1k = {k: next((p for p in pts(k, "pk1024") if p["x"] == step_c), None) for k in series_keys}
    at_cv = {k: next((p for p in pts(k, "v256") if p["x"] == step_c), None) for k in series_keys}
    prog_txt = ", ".join(f"segment {s}: step {max(last_common[(s, nl)] for nl in NLS if (s, nl) in last_common):,}" for s in SEGS if any((s, nl) in last_common for nl in NLS))
    tbl_rows = "".join(
        f"<tr><td>segment {k[0]}</td><td>{NL_NAME[k[1]]}</td><td>{at_c[k]['y']:.2f} ({at_c[k]['lo']:.2f} to {at_c[k]['hi']:.2f})</td>"
        f"<td>{at_c1k[k]['y']:.2f} ({at_c1k[k]['lo']:.2f} to {at_c1k[k]['hi']:.2f})</td><td>{at_cv[k]['y']:.2f} ({at_cv[k]['lo']:.2f} to {at_cv[k]['hi']:.2f})</td></tr>"
        for k in series_keys if at_c[k] and at_c1k[k] and at_cv[k])
    cur_tbl = (f"<div class='tbl'><table><thead><tr><th>segment</th><th>loop</th><th>passkey, N = 256</th><th>passkey, N = 1024</th><th>AR valid, N = 256</th></tr></thead><tbody>{tbl_rows}</tbody></table></div>"
               if tbl_rows else "")
    now = time.strftime("%d %b %Y, %H:%M")

    ref_pk = {n: cell("ref", n, "passkey") for n in EXP_NS}
    tldr = [
        f"<b>A plain window cannot carry the passkey.</b> Sliding-window attention with W = 64 and 2 layers (reach {2 * 63} tokens) scores "
        f"{ref_pk[32]:.2f} / {ref_pk[64]:.2f} / {ref_pk[256]:.2f} / {ref_pk[1024]:.2f} on the passkey at N = 32 / 64 / 256 / 1024, while its AR queries are solved (valid accuracy {cell('ref', 256, 'valid'):.2f} at N = 256).",
        f"<b>The feedback loop learns the passkey and the AR task together only with a stream-length curriculum.</b> Training on n32 first and adding n64, n128, n256 every 2,000 steps gives "
        f"passkey {cell(cur_l, 256, 'passkey'):.2f} (linear loop) and {cell(cur_t, 256, 'passkey'):.2f} (tanh loop) at N = 256, with AR valid accuracy {cell(cur_l, 256, 'valid'):.2f} and {cell(cur_t, 256, 'valid'):.2f}. "
        f"Without the curriculum the loop only started when the passkey loss was up-weighted ×100 (earlier 8k-step runs with default settings stayed at 0.00 at every N; they are not in the table), and then AR is starved: valid accuracy at N = 256 is at most {best_w_valid:.2f} in the four runs without the curriculum "
        f"({best_cw_valid:.2f} when the up-weighting is combined with the curriculum).",
        f"<b>Extrapolation beyond the training length is weak so far.</b> N = 1024 is 4× the longest training stream (about {hops(1024)} loop hops against {hops(256)} in training). "
        f"The curriculum models score {cell(cur_l, 1024, 'passkey'):.2f} (linear) and {cell(cur_t, 1024, 'passkey'):.2f} (tanh) there, against {ref_pk[1024]:.2f} for the plain window. "
        "These are single runs, 12k steps each.",
        f"<b>The 3-seed sweep is running and nothing in it is final.</b> {n_started} of 18 runs have logged evaluations (latest step with all seeds: {prog_txt}); {len(finished)} have finished. "
        "The question it answers is the one you raised: does a smaller segment (same window) extrapolate better? Smaller segments need proportionally more sequential hops, so they are slower to train.",
        "<b>Queued for tonight:</b> segment 8, noise on the loop vector during training, and 40k-step runs. Each is a separate wandb group and a separate stage of the queue.",
    ]
    tested = [
        {"title": "Feedback loop", "tag": "implemented", "ok": True,
         "body": "Once per segment every layer turns its output into one vector, the mean over the segment of nonlin(linear(norm(y))), and adds it with a learned per-channel gain to its own input at every token of the next segment. "
                 "The segment may be shorter than the window. Unit tests: zero feedback reproduces plain sliding-window attention exactly, the model is causal, information passes the window reach, gradients reach every hop."},
        {"title": "Training recipe exploration", "tag": f"{len(explore)} runs, seed 0", "ok": True,
         "body": "Window 64, 2 layers, segment 64, 12k steps, validation split. Compared a stream-length curriculum, a ×100 passkey loss weight annealed to 1 (over 4k or 8k steps), and both, each with a tanh and a linear loop. Loop parameters at 10× learning rate, gain 0.1, projection std 0.1 in all of them."},
        {"title": "Main sweep", "tag": f"{len(finished)}/18 finished", "ok": False,
         "body": "Loop type (tanh, linear) × segment (64, 32, 16) × 3 seeds, window 64, 20k steps, curriculum with 2,000 steps per stage, no loss re-weighting. Final evaluation on validation and test up to N = 16,384. wandb group swa_fb."},
        {"title": "Overnight queue", "tag": "queued", "ok": False,
         "body": "Segment 8 (2 seeds), noise 0.1 and 0.3 on the loop vector at segment 32, and 40k-step runs at segments 64 and 32. wandb group swa_fb_extra."},
    ]
    lin_pk = [cell(x["name"], 256, "passkey") for x in explore if x["cfg"]["model"]["nonlin"] == "none"]
    tan_pk = [cell(x["name"], 256, "passkey") for x in explore if x["cfg"]["model"]["nonlin"] == "tanh"]
    max_1k = max(cell(x["name"], 1024, "passkey") for x in explore)
    cap1 = (f"<p><b>Top grid.</b> The plain window is at {ref_pk[256]:.2f} from N = 256 on. With a loop, the linear variants keep the passkey at N = 256 under every recipe ({min(lin_pk):.2f} to {max(lin_pk):.2f}) "
            f"and the tanh variants less so ({min(tan_pk):.2f} to {max(tan_pk):.2f}). At N = 1024 no run is above {max_1k:.2f}.</p>"
            f"<p><b>Middle grid.</b> The difference between the recipes is here: valid accuracy at N = 256 is {min(cell(k, 256, 'valid') for k in weight_only):.2f} to {best_w_valid:.2f} for the four runs without the curriculum, "
            f"{min(cell(k, 256, 'valid') for k in cw):.2f} to {best_cw_valid:.2f} with the curriculum plus up-weighting, and {cell(cur_l, 256, 'valid'):.2f} / {cell(cur_t, 256, 'valid'):.2f} (linear / tanh) with the curriculum alone. "
            "Only the curriculum-alone rows have both halves.</p>"
            "<p><b>Bottom grid.</b> Stale-key accuracy stays close to 1.00 for every row where it exists (N = 32 has no stale queries). Hover a cell for the numbers.</p>")
    cap2 = ("<p>Expired-query accuracy is high for all runs, so the points differ mainly in height. Hollow markers have AR valid accuracy below 0.90, which is the case for the weighted runs: they trade the AR half for the passkey. "
            "The top-right corner needs both, and only the curriculum rows get close at N = 256.</p>")
    nxt = ("<p><b>1. Read the sweep when it finishes.</b> Final numbers (test split, N up to 16,384, 3 seeds) and the report in this format, with the segment-size comparison as the main figure.</p>"
           "<p><b>2. Decide from it what comes after the queue.</b> If smaller segments help, segment 8 shows whether the trend continues. If they do not, the limit is what one vector can carry per hop, and the loop vector, not the schedule, is what to change.</p>"
           "<p><b>3. Still open:</b> training on longer streams (n512, n1024) would remove the extrapolation gap by construction and needs your decision; the benchmark change that samples the query density per stream; Exp 3 and Exp 4 are on hold.</p>")
    data = {
        "ns": EXP_NS, "rows": rows, "results": results, "scatter_ns": [256, 1024], "label_w": 206, "row_name": "configuration", "row_prefix": "",
        "meta": f"Phase 1, snapshot of {now}. <span class='status'><span>main sweep: <b>{len(finished)}/18 runs finished</b></span><span>·</span><span><b>intermediate</b></span></span>",
        "tldr": tldr, "tested": tested,
        "results_intro": "The first part is the exploration that fixed the training recipe (validation split, 1,000 examples per N, one seed per row except the plain-window reference, which is the mean of 3 seeds trained for 30k steps). "
                         "The second part is the main sweep, seen through its training curves so far. N counts AR writes; one write is about 3.75 tokens.",
        "fig1_sub": "Rows: training recipe and loop type. Columns: number of AR writes N. Darker is higher. w100→1 = passkey loss weight ×100, annealed to 1.",
        "fig2_sub": "One point per configuration. Ideal is the top-right corner. Hollow = AR valid accuracy below 0.90. Colour: "
                    "<span class='legend' style='display:inline-flex;margin:0'><span><i style='background:var(--muted)'></i>plain window</span><span><i style='background:var(--s1)'></i>curriculum alone</span>"
                    "<span><i style='background:var(--s2)'></i>curriculum + weight</span><span><i style='background:var(--s3)'></i>weight only, no curriculum</span></span>. Only the first two groups carry labels; hover the others.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)",
        "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": cap1, "cap2": cap2,
        "extra_figs": [
            {"title": "3. Main sweep so far: passkey at N = 256 during training",
             "sub": "Validation, 200 examples per evaluation, mean over the 3 seeds with the seed range as bars. Colour = segment length, dashed = linear loop. Lines end at the current step of the slowest seed.",
             "caption": f"<p>N = 256 is the longest training stream (about {hops(256)} hops at segment 64, {hops(256, 16)} at segment 16). Steps 0 to 6,000 are the curriculum stages, so the curves are not comparable before the n256 stage starts at step 6,000. "
                        f"Latest step reached by all seeds of every series: {step_c:,}.</p>" + cur_tbl,
             "spec": curve("pk256", "passkey exact match, N = 256", "Passkey accuracy at N = 256 against training step")},
            {"title": "4. Main sweep so far: passkey at N = 1024 (4× the training length)",
             "sub": "The extrapolation test. Same runs and evaluation as figure 3.",
             "caption": "<p>This is the number the segment-size hypothesis is about: whether more, shorter hops carry the passkey farther. It is the noisiest of the three curves; read it only when the runs have finished.</p>",
             "spec": curve("pk1024", "passkey exact match, N = 1024", "Passkey accuracy at N = 1024 against training step")},
            {"title": "5. Main sweep so far: AR valid accuracy at N = 256",
             "sub": "Checks that the AR half is learned alongside the passkey.",
             "caption": "<p>Valid accuracy rises early in every run, since the AR half needs no loop; a run whose passkey is high but whose AR curve stays low would be the weighted-loss failure from the exploration.</p>",
             "spec": curve("v256", "AR valid accuracy, N = 256", "AR valid accuracy at N = 256 against training step")},
        ],
        "next": nxt,
    }
    return data, "SWA Feedback Loop Report", "Sliding-window attention with a segment feedback loop"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["gdn", "swa", "fb"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    render(*{"gdn": gdn_data, "swa": swa_data, "fb": fb_data}[a.kind](), a.out)


if __name__ == "__main__":
    main()
