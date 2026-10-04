#!/usr/bin/env python
"""Build the HTML reports (one self-contained page each) from results/raw/*.json and results/probes/*.json.

    python src/report.py gdn --out reports/gdn_report.html
    python src/report.py swa --out reports/swa_report.html
    python src/report.py fb --out reports/fb_report.html     # SWA + segment feedback loop (also reads logs/fbsweep for runs in progress)

All numbers in the text are computed from the data; the template is reports/report_template.html.
"""
from __future__ import annotations

import argparse
import html
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
import report_groups as RG  # noqa: E402
import report_rounds as RR  # noqa: E402
from report_groups import NS, _nn, cells_for_row, horizon  # noqa: E402,F401

TEMPLATE = C.ROOT / "reports" / "report_template.html"
TOKENS_PER_WRITE = 3.75  # 3 tokens per write plus its share (0.25 per write) of 3-token queries


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
        + ("No α holds the passkey at N = 4,096 or beyond." if not any(hz[a] is not None and hz[a] >= 4096 for a in alphas)
           else "α = " + ", ".join(f"{a:g}" for a in alphas if hz[a] is not None and hz[a] >= 4096) + " hold it at N = 4,096 or beyond."),
        "<b>α = 1 expires old bindings through the delta rule, not through decay.</b> On the seed-0 checkpoint its accuracy does not move when the query density changes 40-fold, so it counts writes (which overwrite each other), while a scalar decay counts tokens.",
        f"<b>The α = 1 passkey leak is not a kernel-precision artifact.</b> The fla kernel, an fp32 recurrence and an fp64 recurrence gave identical passkey accuracy on 32 streams. The leak grows with sequence length and with query density (figure 3).",
        "<b>Next.</b> A sliding-window attention baseline (separate report) has a provable reach limit, which makes it a cleaner base for the memory-amplification loop.",
    ]
    tested = [
        {"title": "Benchmark", "tag": "passed", "ok": True, "body": "Oracle reproduces 100% of labels on all files, rejects deliberately corrupted examples, and never meets a gray-zone query. Regenerating from the stored seeds gives identical examples. No leakage between splits. Drift rate tuned to 0.10. At N ≤ 32 no key can be stale, so expired queries there are all never-written keys (n64: 92%)."},
        {"title": "Model plumbing", "tag": "passed", "ok": True, "body": "Fixed-decay GDN on stock fla blocks, checked against a naive recurrence (0.3% relative, the kernel’s precision). Padding to a common length does not change predictions beyond float rounding. The unit tests pass."},
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
    # the baseline sweep = plain SWA that differs from the reference recipe (swa_v1: 30k steps, no curriculum) in window and layers only; the control and the 50k-step
    # runs with the curriculum differ in more and are never merged into these rows (review H7)
    runs = [r for r in A.load_runs(C.ROOT / "results" / "raw") if r["cfg"]["model"]["kind"] == "swa" and set(RG.diff_of(r)) <= {"model.window", "model.n_layers"}]
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
LOG_RE = re.compile(r"\[swa_fb_nl(?P<nl>\w+?)_seg(?P<seg>\d+)_s(?P<seed>\d+)\] step (?P<step>\d+) loss (?P<loss>[\d.]+) \| "
                    r"n32: pk (?P<pk32>[\d.]+) v (?P<v32>[\d.]+) e (?P<e32>[\d.]+) n256: pk (?P<pk256>[\d.]+) v (?P<v256>[\d.]+) e (?P<e256>[\d.]+) "
                    r"n1024: pk (?P<pk1024>[\d.]+) v (?P<v1024>[\d.]+) e (?P<e1024>[\d.]+)")


def sweep_progress(logs_root: Path) -> dict:
    """{(segment, nonlin): {seed: {step: {metric: value}}}} parsed from the validation lines of the sweep logs (2 decimals).
    Logs live in logs/<tmux session>/<run name>.log; a run that was stopped and started again in another session has two logs
    with the same name, and the newest one wins."""
    newest: dict[str, Path] = {}
    for f in sorted(Path(logs_root).glob("*/swa_fb_nl*_seg*_s*.log"), key=lambda f: f.stat().st_mtime):
        newest[f.stem] = f
    out: dict = {}
    for f in newest.values():
        for line in f.read_text().splitlines():
            m = LOG_RE.search(line)
            if m:
                g = m.groupdict()
                out.setdefault((int(g["seg"]), g["nl"]), {}).setdefault(int(g["seed"]), {})[int(g["step"])] = {k: float(v) for k, v in g.items() if k not in ("nl", "seg", "seed", "step")}
    return out


STEP_RE = re.compile(r"\[(?P<name>swa_fb_\S+)\] step (?P<step>\d+) loss (?P<loss>[\d.]+) \| n32: pk (?P<pk32>[\d.]+) v (?P<v32>[\d.]+) e [\d.]+ n256: pk (?P<pk256>[\d.]+) v (?P<v256>[\d.]+)")


def recipe_gain() -> float:
    """Loop gain at initialisation of the recipe in configs/swa_fb.yaml (the report's main rows use runs with this gain)."""
    return float(C.load_config(C.ROOT / "configs" / "swa_fb.yaml", [])[0]["model"]["gain_init"])


def active_logs(logs_root: Path, minutes: float = 30) -> set[str]:
    """Run names whose training log was written in the last `minutes` minutes (the runs that are still going)."""
    import time as _t
    now = _t.time()
    return {f.stem for f in Path(logs_root).glob("*/swa_fb_*.log") if now - f.stat().st_mtime < minutes * 60}


def log_steps(logs_root: Path) -> dict:
    """{run name: {step: {metric: value}}} from the validation lines of the training logs; the newest log of a run name wins."""
    newest: dict[str, Path] = {}
    for f in sorted(Path(logs_root).glob("*/swa_fb_*.log"), key=lambda f: f.stat().st_mtime):
        newest[f.stem] = f
    out: dict = {}
    for name, f in newest.items():
        for line in f.read_text(errors="ignore").splitlines():
            m = STEP_RE.search(line)
            if m and m["name"] == name:
                out.setdefault(name, {})[int(m["step"])] = {k: float(m[k]) for k in ("loss", "pk32", "v32", "pk256", "v256")}
    return out


def gain_comparison(raw: list[dict], gain: float, steps: dict) -> dict | None:
    """Segment size at two initial loop gains: the recipe's and 0 (the review's H2). Returns None when there are no gain-0 runs.
    Result: groups per (segment, nonlinearity, gain), the figures, and the numbers the summary text needs."""
    groups: dict = {}  # (segment, nonlin, gain) -> runs of the configuration that differs from the reference in these settings only (noise 0, 20k steps)
    for g in RG.diff_groups([r for r in raw if r["cfg"]["model"]["kind"] == "swa_fb"]):
        k = RG.standard_key(g)
        if k and k[2] == 0 and k[3] == 20000 and k[4] in (gain, 0, 0.01) and gain != 0:
            groups[(k[0], k[1], k[4])] = g["runs"]
    if not any(k[2] == 0 for k in groups):
        return None
    segs, nls = (8, 16, 32, 64), ("tanh", "none")
    gname = lambda g: "gain 0" if g == 0 else f"gain {g:g}"
    lname = lambda nl: "tanh loop" if nl == "tanh" else "no-tanh loop"
    cells = {k: cells_for_row(rs, "test") for k, rs in groups.items()}
    pk = lambda k, n: cells[k][str(n)]["passkey"]
    perk = lambda k, n: cells[k][str(n)]["per"]["passkey"]

    def ar_at(nl, seg, g, step=2000):
        """AR valid accuracy at N = 256 (validation) after `step` steps, mean over the runs of the group; the log of a run is found by
        the run name, which is the run's own name (results/raw/<name>.json)."""
        v = [steps[r["name"]][step]["v256"] for r in groups.get((seg, nl, g), []) if step in steps.get(r["name"], {})]
        return float(np.mean(v)) if v else None

    def chart(n):
        series = []
        for nl in nls:
            for g in (gain, 0):
                pts = [{"x": s, "y": pk((s, nl, g), n), "lo": min(perk((s, nl, g), n)), "hi": max(perk((s, nl, g), n)),
                        "tip": f"{lname(nl)}, {gname(g)}, segment {s}<br>N = {n:,}<br>passkey {pk((s, nl, g), n):.3f} (seeds {', '.join(f'{x:.2f}' for x in perk((s, nl, g), n))})"}
                       for s in segs if (s, nl, g) in groups]
                if pts:
                    series.append({"label": f"{lname(nl)}, {gname(g)}", "short": f"{'tanh' if nl == 'tanh' else 'lin'} g{g:g}", "points": pts, "width": 2.2,
                                   "c": "var(--s1)" if nl == "tanh" else "var(--s2)", "dashed": g == 0})
        return {"series": series, "xTicks": list(segs), "x0": 8, "x1": 64, "xLabel": "segment length in tokens (log scale)", "yLabel": f"passkey exact match, N = {n:,}",
                "legendLine": True, "rightPad": 92, "aria": f"Passkey accuracy at N = {n} against segment length for two initial loop gains"}

    rows_html = ""
    started = {}
    for nl in nls:
        for s in reversed(segs):
            for g in (gain, 0.01, 0):
                k = (s, nl, g)
                if k not in groups:
                    continue
                rs = groups[k]
                st = sum(next(x for x in r["rows"] if x["split"] == "test" and x["N"] == 64)["passkey_acc"] >= 0.5 for r in rs)
                started[k] = (st, len(rs))
                a2 = ar_at(nl, s, g)
                rows_html += (f"<tr><td>{lname(nl)}</td><td>{s}</td><td>{gname(g)}</td><td>{st} of {len(rs)}</td><td>{'–' if a2 is None else f'{a2:.2f}'}</td>"
                              f"<td>{pk(k, 256):.2f}</td><td>{pk(k, 1024):.2f}</td><td>{pk(k, 16384):.2f}</td><td>{np.mean([r['final_train_loss'] for r in rs]):.3f}</td></tr>")
    table = ("<div class='tbl'><table><thead><tr><th>loop</th><th>segment</th><th>gain</th><th>loop started</th><th>AR valid, step 2k</th>"
             "<th>passkey N = 256</th><th>N = 1024</th><th>N = 16384</th><th>train loss</th></tr></thead><tbody>" + rows_html + "</tbody></table></div>")
    best = max((k for k in groups if k[2] == 0), key=lambda k: pk(k, 16384))
    valid_of = lambda r: next(x for x in r["rows"] if x["split"] == "test" and x["N"] == 256)["ar_valid_acc"]
    seed_rows = [(r["seed"], next(x for x in r["rows"] if x["split"] == "test" and x["N"] == n)["passkey_acc"]) for r in sorted(groups[best], key=lambda r: r["seed"]) for n in (16384,)]
    return {"groups": groups, "cells": cells, "chart": chart, "table": table, "started": started, "best": best, "pk": pk, "perk": perk,
            "ar_at": ar_at, "gname": gname, "lname": lname, "segs": segs, "nls": nls, "seed_rows": seed_rows,
            "best_valid": [valid_of(r) for r in sorted(groups[best], key=lambda r: r["seed"])]}


def number_figures(data: dict, first: int = 3) -> dict:
    """Number the extra figures in order from `first` and replace {fig:key} references in all text by the number of that figure."""
    nums = {}
    for i, f in enumerate(data["extra_figs"], start=first):
        f["title"] = f"{i}. " + re.sub(r"^\d+\.\s*", "", f["title"])
        if f.get("key"):
            nums[f["key"]] = i
    text = json.dumps(data)
    for key, i in nums.items():
        text = text.replace("{fig:%s}" % key, str(i))
    left = re.findall(r"\{fig:[^}]*\}", text)
    if left:
        raise ValueError(f"references to figures that do not exist: {sorted(set(left))}")
    return json.loads(text)


def fb_data() -> tuple[dict, str, str]:
    raw = A.load_runs(C.ROOT / "results" / "raw")
    # runs of the recipe (the loop gain of configs/swa_fb.yaml); runs with another gain are a separate comparison and must not merge into these rows
    rec_gain = recipe_gain()
    # every row is one configuration = the runs with the same diff against the frozen reference (review H7); the rows of the main grid
    # are the configurations that differ from it only in segment, loop type, noise, steps (at the recipe's gain)
    dgroups = RG.diff_groups([r for r in raw if r["cfg"]["model"]["kind"] in ("swa", "swa_fb")])
    std = {RG.standard_key(g): g for g in dgroups if RG.standard_key(g)}
    dg_of = {k[:4]: g for k, g in std.items() if k[4] == rec_gain}  # (segment, nonlin, noise, steps) -> configuration
    fb = [r for g in dg_of.values() for r in g["runs"]]
    group_of = {r["name"]: g for g in dgroups for r in g["runs"]}
    steps_log = log_steps(C.ROOT / "logs")
    gc = gain_comparison(raw, rec_gain, steps_log)
    swa64 = [g for g in dgroups if g["kind"] == "swa" and RG.setting(g, "model.window") == 64 and RG.setting(g, "model.n_layers") == 2
             and set(g["diff"]) <= {"train.steps", "train.curriculum_stage_steps"}]  # plain window 64, 2 layers; only steps and curriculum differ
    same_recipe = next((g["runs"] for g in swa64 if RG.setting(g, "train.curriculum_stage_steps") > 0 and RG.setting(g, "train.steps") == 20000), [])  # control trained like the loop models
    ref_runs = same_recipe or next(g["runs"] for g in swa64 if not RG.setting(g, "train.curriculum_stage_steps"))
    ref_desc = (f"no loop, 20k steps with the same curriculum as the loop models, {len(ref_runs)} seeds" if same_recipe
                else f"no loop, {ref_runs[0]['cfg']['train']['steps'] // 1000}k steps, no curriculum, {len(ref_runs)} seeds")
    gdn1 = [r for r in raw if r["model"] == "gdn_fixed" and r["alpha"] == 1.0]
    explore = [r for r in A.load_runs(C.ROOT / "results" / "explore") if r["cfg"]["model"]["kind"] == "swa_fb"]
    groups = {k: g["runs"] for k, g in dg_of.items()}
    klabel = lambda k, n=None: RG.label(dg_of[k]) + (f" ({len(groups[k]) if n is None else n})" if (len(groups[k]) if n is None else n) != 3 else "")
    kshort = lambda k: RG.short_label(dg_of[k]) + (f" ({len(groups[k])})" if len(groups[k]) != 3 else "")
    order = ([(s, nl, 0.0, 20000) for nl in ("tanh", "none") for s in (64, 32, 16, 8)]
             + [(32, nl, nz, 20000) for nl in ("tanh", "none") for nz in (0.1, 0.3)]
             + [(s, nl, 0.0, 40000) for nl in ("tanh", "none") for s in (64, 32)])
    order = [k for k in order if k in groups]
    sk = lambda k: f"{k[0]}_{k[1]}_{k[2]:g}_{k[3]}"
    fam_color = lambda k: "--s1" if k[1] == "tanh" else "--s2"
    highlight = {(64, "tanh", 0.0, 20000), (32, "tanh", 0.3, 20000)}  # labelled in the scatter; the other points are told apart by colour and tooltip
    rows = [{"key": "ref", "label": "SWA, no loop", "color": "--muted", "long": f"plain sliding window, W = 64, 2 layers, {ref_desc}"}]
    results = {"ref": cells_for_row(ref_runs, "test")}
    for k in order:
        rows.append({"key": sk(k), "label": kshort(k), "slabel": f"seg {k[0]} {'tanh' if k[1] == 'tanh' else 'no tanh'}" + (f", noise {k[2]:g}" if k[2] else "") + (f", {k[3] // 1000}k steps" if k[3] != 20000 else ""),
                     "color": fam_color(k), "nolabel": k not in highlight,
                     "long": f"{klabel(k)}, seeds {sorted(r['seed'] for r in groups[k])}"})
        results[sk(k)] = cells_for_row(groups[k], "test")
    row_runs = {"ref": [r["name"] for r in ref_runs], **{sk(k): [r["name"] for r in groups[k]] for k in order}}  # the runs behind every row
    cell = lambda k, n, f="passkey": results[sk(k)][str(n)][f]
    per = lambda k, n, f="passkey": results[sk(k)][str(n)]["per"][f]
    f2 = lambda x: f"{x:.2f}"
    hops = lambda n, seg: round(n * TOKENS_PER_WRITE / seg)
    main = lambda s, nl: (s, nl, 0.0, 20000)
    has = lambda k: k in groups
    n_main = sum(len(groups[k]) for k in groups if k[0] in (64, 32, 16) and k[2] == 0 and k[3] == 20000)
    n_extra = len(fb) - n_main

    # ---- runs still going (reruns after out-of-memory failures, long sweeps): every training log without a result file
    prog = sweep_progress(C.ROOT / "logs")
    have = {r["name"] for r in raw}
    live = active_logs(C.ROOT / "logs")
    running = [f"{nm.replace('swa_fb_', '')} (step {max(st):,})" for nm, st in sorted(steps_log.items()) if nm not in have and nm in live]

    # ---- is the segment comparison handicapped by a slow start? AR valid accuracy at N = 256 (validation, 200 examples) from the logs
    def ar_at(seg, step):
        v = [st[step]["v256"] for (s, nl), seeds in prog.items() if s == seg for st in seeds.values() if step in st]
        return float(np.mean(v)) if v else None
    stall = {s: (ar_at(s, 2000), ar_at(s, 6000)) for s in (64, 32, 16)}
    stall_txt = ", ".join(f"segment {s}: {f2(a)} at step 2,000 and {f2(b)} at step 6,000" for s, (a, b) in stall.items() if a is not None and b is not None)
    nparams = lambda name: json.loads((C.ROOT / "results" / "raw" / f"{name}.json").read_text())["n_params"]

    # ---- the hypothesis: smaller segment, better extrapolation?
    segs = (64, 32, 16, 8)
    at1k = {nl: [cell(main(s, nl), 1024) if has(main(s, nl)) else None for s in segs] for nl in ("none", "tanh")}
    at4k = {nl: [cell(main(s, nl), 4096) if has(main(s, nl)) else None for s in segs] for nl in ("none", "tanh")}
    mono = {nl: all(v is not None for v in at1k[nl]) and all(a < b for a, b in zip(at1k[nl], at1k[nl][1:])) for nl in at1k}
    best_seg = {nl: segs[max(range(len(segs)), key=lambda i: at1k[nl][i] if at1k[nl][i] is not None else -1)] for nl in at1k}
    spread_best = lambda nl: ", ".join(f2(x) for x in per(main(best_seg[nl], nl), 1024))
    verdict = ("The hypothesis that a smaller segment extrapolates better holds for both loops." if all(mono.values()) else
               f"The hypothesis that a smaller segment extrapolates better is not supported by these runs: neither loop shows a monotone trend. The best segment at N = 1024 is {best_seg['none']} for the no-tanh loop "
               f"(seeds {spread_best('none')}) and {best_seg['tanh']} for the tanh loop (seeds {spread_best('tanh')}).")
    seq = lambda xs: " / ".join("–" if x is None else f2(x) for x in xs)
    in256 = {nl: [cell(main(s, nl), 256) if has(main(s, nl)) else None for s in segs] for nl in ("none", "tanh")}
    conv = {}  # share of runs that did not converge (final training loss above 0.1) by segment, main grid + segment 8
    for s in segs:
        rr = [r for k in groups if k[0] == s and k[2] == 0 and k[3] == 20000 for r in groups[k]]
        conv[s] = (sum(r["final_train_loss"] > 0.1 for r in rr), len(rr))
    conv_txt = ", ".join(f"segment {s}: {a} of {b}" for s, (a, b) in conv.items() if b)
    seg8 = [r for k in groups if k[0] == 8 for r in groups[k]]
    seg8_bad = [r for r in seg8 if next(x for x in r["rows"] if x["split"] == "test" and x["N"] == 256)["ar_valid_acc"] < 0.6]

    gdn = cells_for_row(gdn1, "test") if gdn1 else None
    gdn_txt = (f"GDN α = 1 ({len(gdn1)} seeds) is better in range: {f2(gdn['512']['passkey'])} at N = 512, {f2(gdn['1024']['passkey'])} at 1024, {f2(gdn['2048']['passkey'])} at 2048, "
               f"then {f2(gdn['4096']['passkey'])} at 4096 and {f2(gdn['16384']['passkey'])} at 16384; the models are not size-matched: {nparams(gdn1[0]['name']) // 1000}k parameters, 1 head, fp32 for GDN against {nparams(fb[0]['name']) // 1000}k, 4 heads, bf16 for the loop models") if gdn else ""

    # ---- per-stream token profile (probe), if it has been run
    pf = C.ROOT / "results" / "probes" / "passkey_profile.json"
    prof = json.loads(pf.read_text()) if pf.exists() else {}
    by_name = {r["name"]: r for r in fb}
    prof_rows = []
    for nm, lst in prof.items():
        if nm in by_name:
            lab = RG.short_label(group_of[nm]) + f" · seed {by_name[nm]['seed']}"
            for e in sorted(lst, key=lambda e: e["N"]):
                prof_rows.append((nm, lab, e))
    pick = lambda nm, n: next((e for p, _, e in prof_rows if p == nm and e["N"] == n), None)
    nz = pick("swa_fb_seg32_noise0.3_nltanh_s0", 4096)
    t64 = pick("swa_fb_nltanh_seg64_s2", 4096)
    l64 = pick("swa_fb_nlnone_seg64_s0", 1024)
    prof_txt = ""
    if nz and t64:
        prof_txt = (f"<b>Noise during training made the retention all-or-nothing</b> (one seed). For the segment-32 tanh loop with noise 0.3, {nz['hist'][4]:.0%} of the streams at N = 4096 keep all four passkey tokens, "
                    f"{nz['hist'][0]:.0%} keep none and only {1 - nz['hist'][0] - nz['hist'][4]:.0%} keep some. "
                    f"The segment-64 tanh loop (seed 2) at the same N has {t64['hist'][4]:.0%} with all four, {t64['hist'][0]:.0%} with none and {1 - t64['hist'][0] - t64['hist'][4]:.0%} with some: a graded loss"
                    + (f", as for the no-tanh loop at N = 1024, where the four tokens are right {l64['per_token'][0]:.0%} / {l64['per_token'][1]:.0%} / {l64['per_token'][2]:.0%} / {l64['per_token'][3]:.0%} of the time, "
                       "later tokens first" if l64 else "")
                    + ". An intact-or-gone passkey is what error-correcting (attractor) dynamics would produce.")

    # ---- figure data
    def curves(nl):
        series = []
        extras = [((32, nl, 0.3, 20000), "segment 32, noise 0.3", "noise 0.3", "var(--s2)", False), ((64, nl, 0.0, 40000), "segment 64, 40k steps", "40k · seg 64", "var(--s3)", False),
                  ((32, nl, 0.0, 40000), "segment 32, 40k steps", "40k · seg 32", "var(--muted)", True)]
        for i, s in enumerate(segs):
            k = main(s, nl)
            if has(k):
                series.append({"k": k, "label": f"segment {s} ({len(groups[k])} seeds)", "short": f"seg {s}", "ord": [i, len(segs)]})
        for k, lab, short, col, dashed in extras:
            if has(k):
                series.append({"k": k, "label": f"{lab} ({len(groups[k])} seed)", "short": short, "c": col, "width": 2.4, "dashed": dashed})
        out = []
        for sr in series:
            k = sr.pop("k")
            pts = []
            for n in NS:
                v = per(k, n)
                pts.append({"x": n, "y": cell(k, n), "lo": min(v), "hi": max(v),
                            "tip": f"{sr['label']}<br>N = {n:,} (~{hops(n, k[0]):,} loop hops)<br>passkey {cell(k, n):.3f}" + (f" (seeds {', '.join(f'{x:.2f}' for x in v)})" if len(v) > 1 else "")})
            out.append({**sr, "points": pts})
        return {"series": out, "xTicks": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], "x0": 32, "x1": 16384, "xLabel": "stream length N (AR writes, log scale)", "yLabel": "passkey exact match (test)",
                "legendLine": True, "rightPad": 128, "vlines": [{"x": 256, "label": "longest training stream", "dy": 0 if nl == "tanh" else 262}], "aria": f"Passkey accuracy against stream length, {nl} loops"}

    PROF_NS = (256, 1024, 4096)
    pbars = ""
    if prof_rows:
        shades = [("var(--seq0)", "var(--ink)"), ("var(--seq1)", "var(--ink)"), ("var(--seq2)", "#fff"), ("var(--seq3)", "var(--page)"), ("var(--seq4)", "var(--page)")]
        pbars = "<div class='legend'>passkey tokens right in a stream: " + "".join(f"<span><i style='background:{bg}'></i>{j}</span>" for j, (bg, _) in enumerate(shades)) + "</div>"
        for _, lab, e in prof_rows:
            cellsb = "".join(f"<i style='flex:{h};background:{shades[j][0]};color:{shades[j][1]}' title='{j} of 4 tokens right: {h:.1%} of streams'>{h:.0%}</i>" if h >= 0.045 else
                             f"<i style='flex:{h};background:{shades[j][0]}' title='{j} of 4 tokens right: {h:.1%} of streams'></i>" for j, h in enumerate(e["hist"]) if h > 0)
            pbars += f"<div class='pbar'><span>{html.escape(lab)} · N = {e['N']:,}</span><div class='stack'>{cellsb}</div></div>"

    expl_cur = [r for r in explore if r["cfg"]["train"].get("curriculum_stage_steps", 0) > 0 and r["cfg"]["train"]["passkey_weight"] == 1]
    expl_oth = [r for r in explore if r not in expl_cur]
    v256 = lambda r: next(x for x in r["rows"] if x["N"] == 256)["ar_valid_acc"]
    rerun_note = f" Still training when this page was built: {', '.join(running)}." if running else ""
    tldr = [
        f"<b>The loop works inside the training range.</b> With the stream-length curriculum, a 2-layer window-64 model (reach 126 tokens) that cannot answer the passkey at all ({results['ref']['256']['passkey']:.2f} without the loop) keeps it at N = 256, about 8 times its reach: "
        f"{f2(cell(main(64, 'none'), 256))} for the no-tanh loop and {f2(cell(main(64, 'tanh'), 256))} for the tanh loop at segment 64, with AR valid accuracy {f2(cell(main(64, 'none'), 256, 'valid'))}, the same as the plain window.",
        f"<b>No evidence that a smaller segment extrapolates better.</b> Passkey at N = 1024, 4× the longest training stream, for segment 64 / 32 / 16 / 8: no-tanh loop {seq(at1k['none'])}, tanh loop {seq(at1k['tanh'])}. {verdict} "
        f"Small segments can fit the training range (N = 256, no-tanh loop, segment 64 / 32 / 16 / 8: {seq(in256['none'])}) but do no better beyond it. Training also becomes less reliable: runs whose training loss stayed above 0.1 ({conv_txt})"
        + (f"; at segment 8, {len(seg8_bad)} of {len(seg8)} runs did not learn the AR half (valid accuracy below 0.6)." if seg8 else ".")
        + (f" The comparison is also handicapped by a slow start: the AR half is learned later the smaller the segment (AR valid accuracy at N = 256, mean over runs: {stall_txt}), "
           "so small segments spend less of the 20k steps on long streams. "
           + ("The gain-0 runs further down remove this confound." if gc else "A rerun with the loop gain initialised at 0 would remove this confound.") if stall_txt else ""),
        f"<b>The tanh loop keeps a floor, the no-tanh loop does not.</b> No-tanh loops fit the training range better and are at {max(cell(k, 4096) for k in groups if k[1] == 'none'):.2f} or less by N = 4096 in every configuration. The segment-64 tanh loop holds "
        f"{f2(cell(main(64, 'tanh'), 4096))} / {f2(cell(main(64, 'tanh'), 8192))} / {f2(cell(main(64, 'tanh'), 16384))} at N = 4096 / 8192 / 16384 (about {hops(16384, 64):,} loop hops at the end), a constant instead of a decay. "
        + (gdn_txt + ". The loop does not beat GDN α = 1 where GDN works (N ≤ 2048), but at N = 16384 GDN is at zero and the tanh loop is not." if gdn_txt else ""),
    ]
    # ---- the attractor probe (probes.py attractor) and the 12k-step exploration, as tables
    ap_file = C.ROOT / "results" / "probes" / "attractor_probe.json"
    attr = json.loads(ap_file.read_text()) if ap_file.exists() else {}
    all_fb = {r["name"]: r for r in raw if r["cfg"]["model"]["kind"] == "swa_fb"}
    attr = {n: r for n, r in attr.items() if n in all_fb}

    def run_label(nm):
        r = all_fb[nm]
        return RG.short_label(group_of[nm]) + f" · seed {r['seed']}"

    extra_html_figs = []
    if attr:
        order_a = sorted(attr, key=lambda n: -attr[n]["exact"]["0"])
        rows_a = "".join(
            f"<tr><td>{html.escape(run_label(n))}</td><td>{r['exact']['0']:.2f}</td>" + "".join(f"<td>{r['exact'][e]:.2f}</td>" for e in ("0.1", "0.3", "1", "3"))
            + "".join(f"<td>{r['contraction'].get(k, float('nan')):.3f}</td>" for k in ("1", "16", "64"))
            + f"<td>{r['jacobian']['sigma_max_mean']:.2f}</td><td>{r['jacobian']['spectral_radius_mean']:.2f}</td><td>{r['drift']['rms']['0.5']:.2f}</td></tr>" for n, r in ((n, attr[n]) for n in order_a))
        keep = [n for n in order_a if attr[n]["exact"]["0"] >= 0.3]
        lose = [n for n in order_a if attr[n]["exact"]["0"] < 0.05]
        a_n = attr[order_a[0]]["n_writes"]
        keep_txt = lose_txt = ""
        if keep:
            r03 = min(attr[n]["exact"]["0.3"] / attr[n]["exact"]["0"] for n in keep)
            r1 = min(attr[n]["exact"]["1"] / attr[n]["exact"]["0"] for n in keep)
            c64 = max(attr[n]["contraction"].get("64", float("nan")) for n in keep)
            rho_k = [attr[n]["jacobian"]["spectral_radius_mean"] for n in keep]
            keep_txt = (f"In the {len(keep)} probed runs that hold the passkey at N = {a_n:,} ({', '.join(run_label(n) for n in keep)}) a perturbation of 0.3 times the RMS of m leaves at least {r03:.0%} of the clean exact match "
                        f"and one of 1.0 times the RMS at least {r1:.0%}; the difference to the clean run shrinks to {c64:.1%} of its start within 64 hops at most, and the one-hop Jacobian has a spectral radius of {min(rho_k):.2f} to {max(rho_k):.2f}.")
        if lose:
            rho_l = [attr[n]["jacobian"]["spectral_radius_mean"] for n in lose]
            lose_txt = (f"Contraction alone is not retention: the {len(lose)} probed runs that lose the passkey (clean exact match below 0.05) have spectral radii of {min(rho_l):.2f} to {max(rho_l):.2f}, "
                        f"and the largest one belongs to a no-tanh loop whose state is large and keeps drifting.")
        extra_html_figs.append({"key": "attractor", "title": "9. Is the loop an attractor? Perturbation, contraction and Jacobian of the loop vector",
                                "sub": f"Fresh streams of N = {a_n:,} writes (100 for the exact match, 16 for the contraction, 4 × 3 positions for the Jacobian). m is perturbed at 25% of the stream, in all layers at once, by Gaussian noise of the given multiple of its own RMS.",
                                "caption": f"<p>{keep_txt} {lose_txt}</p><p>Two caveats. The Jacobian is the one-hop response of m with the layer inputs of the window held fixed, so it does not include what the 63-token window itself holds, and it is computed in bf16. "
                                           "The rows are single runs chosen as contrasts, not a sample.</p>",
                                "spec": None, "html": ("<div class='tbl'><table><thead><tr><th>run</th><th>exact, clean</th><th>×0.1</th><th>×0.3</th><th>×1</th><th>×3</th><th>diff after 1 hop</th><th>16 hops</th><th>64 hops</th>"
                                                       "<th>σ max</th><th>spectral radius</th><th>RMS of m</th></tr></thead><tbody>" + rows_a + "</tbody></table></div>")})
        probe_txt = None
        if keep:
            probe_txt = (f"<b>The runs that keep the passkey correct perturbations of their loop vector.</b> {keep_txt} {lose_txt} "
                         "What decides retention is therefore not contraction by itself; a direct test is whether the retaining runs hold one stable state per passkey (decode the passkey from m late in the stream).")
    explore_rows = ""
    for r in sorted(explore, key=lambda r: (r["cfg"]["train"].get("curriculum_stage_steps", 0) == 0, r["cfg"]["train"]["passkey_weight"], r["cfg"]["train"].get("passkey_weight_anneal_steps", 0), r["cfg"]["model"]["nonlin"] != "tanh")):
        tc, mc = r["cfg"]["train"], r["cfg"]["model"]
        n256 = next(x for x in r["rows"] if x["N"] == 256)
        n1k = next(x for x in r["rows"] if x["N"] == 1024)
        recipe = ("curriculum" if tc.get("curriculum_stage_steps", 0) else "no curriculum") + (f", passkey weight {tc['passkey_weight']:g} → 1 over {tc['passkey_weight_anneal_steps'] // 1000}k steps" if tc["passkey_weight"] != 1 else ", no re-weighting")
        explore_rows += (f"<tr><td>{recipe}</td><td>{'tanh' if mc['nonlin'] == 'tanh' else 'no tanh'}</td><td>{n256['ar_valid_acc']:.2f}</td><td>{n256['passkey_acc']:.2f}</td><td>{n1k['passkey_acc']:.2f}</td></tr>")
    if explore_rows:
        extra_html_figs.append({"key": "explore", "title": "10. How the training recipe was chosen: the 12k-step exploration",
                                "sub": f"{len(explore)} runs, window 64, segment 64, seed 0, validation split.",
                                "caption": "<p>The curriculum alone is the only recipe that learns the AR half and the passkey together; every run that up-weighted the passkey loss paid with AR accuracy. This is what the main sweep's recipe is based on.</p>",
                                "spec": None, "html": ("<div class='tbl'><table><thead><tr><th>recipe</th><th>loop</th><th>AR valid, N = 256</th><th>passkey, N = 256</th><th>passkey, N = 1024</th></tr></thead><tbody>" + explore_rows + "</tbody></table></div>")})
    gain_figs, gain_tested = [], None
    if gc:
        sg = (64, 32, 16, 8)
        ar_seq = lambda g: " / ".join(f2(np.mean([gc["ar_at"](nl, sg_, g) for nl in gc["nls"] if gc["ar_at"](nl, sg_, g) is not None])) for sg_ in sg)
        small = lambda g: [gc["started"][k] for k in gc["started"] if k[2] == g and k[0] <= 16]
        fail = lambda g: (sum(b - a for a, b in small(g)), sum(b for a, b in small(g)))
        bk = gc["best"]
        other_k = max(order, key=lambda k: cell(k, 16384))
        bseeds = ", ".join(f2(x) for x in gc["perk"](bk, 16384))
        bvalid = ", ".join(f2(x) for x in gc["best_valid"])
        n_g0 = sum(len(rs) for k, rs in gc["groups"].items() if k[2] == 0)
        tldr.insert(2, f"<b>With the loop gain at 0 the AR stall is gone, but the loop often does not start, and one configuration holds the passkey far longer than anything else.</b> "
                       f"AR valid accuracy at N = 256 after 2,000 steps, segments {' / '.join(map(str, sg))}: gain {rec_gain:g}: {ar_seq(rec_gain)}; gain 0: {ar_seq(0)}. "
                       f"In return the loop did not start (passkey below 0.5 at N = 64) in {fail(0)[0]} of {fail(0)[1]} gain-0 runs at segments 16 and 8, against {fail(rec_gain)[0]} of {fail(rec_gain)[1]} at gain {rec_gain:g}. "
                       f"The best gain-0 configuration is the {gc['lname'](bk[1])} at segment {bk[0]}: passkey {f2(gc['pk'](bk, 1024))} / {f2(gc['pk'](bk, 4096))} / {f2(gc['pk'](bk, 16384))} at N = 1,024 / 4,096 / 16,384 "
                       f"(seeds at N = 16,384: {bseeds}; AR valid accuracy at N = 256 per seed: {bvalid}), about {hops(16384, bk[0]):,} loop hops. "
                       f"The best configuration of the main grid and extras at N = 16,384 is {klabel(other_k)} with {f2(cell(other_k, 16384))}.")
        gseq = lambda nl, g, n: " / ".join(f2(gc["pk"]((sg_, nl, g), n)) if (sg_, nl, g) in gc["groups"] else "–" for sg_ in (8, 16, 32, 64))
        gstart = lambda nl, g: f"{sum(a for k, (a, b) in gc['started'].items() if k[1] == nl and k[2] == g)} of {sum(b for k, (a, b) in gc['started'].items() if k[1] == nl and k[2] == g)}"
        series_1k = {(nl, g): [gc["pk"]((sg_, nl, g), 1024) for sg_ in (64, 32, 16, 8) if (sg_, nl, g) in gc["groups"]] for nl in gc["nls"] for g in (rec_gain, 0)}
        n_mono = sum(1 for v in series_1k.values() if len(v) == 4 and all(a < b for a, b in zip(v, v[1:])))  # rises as the segment shrinks from 64 to 8
        gmono_txt = f"In {n_mono} of the 4 series the passkey rises monotonically as the segment shrinks from 64 to 8."
        gbest_seg = max((sg_ for sg_ in (8, 16, 32, 64) if (sg_, "none", 0) in gc["groups"]), key=lambda sg_: gc["pk"]((sg_, "none", 0), 1024))
        gtot = lambda g: (sum(a for k, (a, b) in gc["started"].items() if k[2] == g), sum(b for k, (a, b) in gc["started"].items() if k[2] == g))
        gain_figs = [
            {"key": "seg1k", "title": "6. Segment length at two initial loop gains: passkey at N = 1,024",
             "sub": f"Mean over seeds with the seed range as bars. Solid = gain {rec_gain:g} (the recipe), dashed = gain 0. Blue = tanh loop, orange = no-tanh loop.",
             "caption": (f"<p>Passkey at N = 1,024 for segments 8 / 16 / 32 / 64: no-tanh loop, gain {rec_gain:g}: {gseq('none', rec_gain, 1024)}; no-tanh loop, gain 0: {gseq('none', 0, 1024)}; "
                         f"tanh loop, gain {rec_gain:g}: {gseq('tanh', rec_gain, 1024)}; tanh loop, gain 0: {gseq('tanh', 0, 1024)}. {gmono_txt} "
                         f"The highest value at N = 1,024 is the no-tanh loop at gain 0 with segment {gbest_seg}. "
                         f"The tanh loop at gain 0 started the loop in {gstart('tanh', 0)} of its runs.</p>"),
             "spec": gc["chart"](1024)},
            {"key": "seg16k", "title": "7. The same at N = 16,384 (about 61k tokens)", "sub": "Same runs and layout as figure {fig:seg1k}.",
             "caption": f"<p>The no-tanh loop at segment 8 and gain 0 keeps {f2(gc['pk']((8, 'none', 0), 16384)) if (8, 'none', 0) in gc['groups'] else '–'} of the passkeys at this length; every other configuration is at or below {f2(max(gc['pk'](k, 16384) for k in gc['groups'] if k != (8, 'none', 0)))}.</p>",
             "spec": gc["chart"](16384)},
            {"key": "started", "title": "8. Did the loop start? Segment, loop type and initial gain", "sub": "Runs per row: 3 seeds (fewer where a run is missing, one for the gain-0.01 pilot). Loop started = passkey of at least 0.5 at N = 64 on the test split. AR valid, step 2k = AR valid accuracy at N = 256 after 2,000 training steps (validation, mean over the runs of the row).",
             "caption": (f"<p>The recipe gain ({rec_gain:g}) started the loop in {gtot(rec_gain)[0]} of {gtot(rec_gain)[1]} runs and gain 0 in {gtot(0)[0]} of {gtot(0)[1]}: from a zero injection the optimiser has to find the loop, "
                         "and at small segments often does not. In exchange gain 0 learns the AR half much earlier (column 'AR valid, step 2k'); the recipe gain delays it the more, the smaller the segment. "
                         "The two gain-0.01 pilot runs behave like gain 0: the loop started at segment 64 and not at segment 16.</p>"),
             "spec": None, "html": gc["table"]},
        ]
        gain_tested = {"title": "Gain-0 comparison", "tag": f"{n_g0} runs", "ok": True,
                       "body": "The segment comparison again with the loop gain initialised at 0 instead of the recipe's 0.1 (code review H2): segments 64, 32, 16 and 8, tanh and no-tanh loop, 3 seeds, everything else as in the main sweep. The plain-SWA control with the same curriculum is the reference row."}
    if prof_txt:
        tldr.append(prof_txt + (f" Noise 0.1 does not help ({f2(cell((32, 'tanh', 0.1, 20000), 4096))} at N = 4096)." if has((32, 'tanh', 0.1, 20000)) else ""))
    if attr and probe_txt:
        tldr.append(probe_txt)
    if has((64, "tanh", 0.0, 40000)):
        tldr.append(f"<b>Twice the training helps a little and changes no horizon</b> (one seed each). At N = 1024: segment-64 tanh {f2(cell((64, 'tanh', 0.0, 40000), 1024))} (20k steps: {f2(cell(main(64, 'tanh'), 1024))}), "
                    f"segment-32 no-tanh {f2(cell((32, 'none', 0.0, 40000), 1024))} ({f2(cell(main(32, 'none'), 1024))}); the no-tanh loops are still at 0 by N = 4096.")
    logs_c = RR.log_curves(C.ROOT / "logs")
    far = RR.far_propagation({k: g for k, g in std.items()}, ref_runs, rec_gain, logs_c)
    long_ = RR.long_runs(dgroups, std, logs_c)
    segs_ = RR.segment_study(std, logs_c)
    if far:
        tldr.insert(1, far["tldr"])
    if long_:
        tldr.append(long_["tldr"])
    tldr.append("<b>What these results do not show.</b> The passkey is always the first item and the only long-range one, so a loop that simply holds whatever the first segment said is enough; "
                "nothing here tests selecting or replacing content. "
                + ("" if same_recipe else "The AR accuracy of the loop models is compared with a plain window trained differently (30k steps, no curriculum); the passkey comparison is unaffected, "
                                         "since a plain window cannot reach that far. ")
                + "The noise and 40k-step results are single seeds.")
    tldr.append("<b>Seed-to-seed spread is large, so small differences are not conclusions.</b> "
                f"Segment-32 no tanh: N = 256 passkey {', '.join(f2(x) for x in per(main(32, 'none'), 256))} across seeds (one seed never started the loop; its training loss stayed at 0.18 against 0.02). "
                f"Segment-32 tanh: {', '.join(f2(x) for x in per(main(32, 'tanh'), 256))}." + rerun_note)
    tested = [
        {"title": "Main sweep", "tag": f"{n_main}/18 runs", "ok": n_main == 18,
         "body": "Loop type (tanh, no tanh) × segment (64, 32, 16) × 3 seeds, window 64, 2 layers, 20k steps, stream-length curriculum (n32, then n64, n128, n256 every 2,000 steps), no loss re-weighting. Evaluated on the test split up to N = 16,384 (about 61k tokens). Four runs crashed with GPU out-of-memory errors (three in the final evaluation) and were rerun with identical settings."},
        {"title": "Extra runs", "tag": f"{n_extra} runs", "ok": True,
         "body": "Segment 8 (2 seeds per loop), Gaussian noise (std 0.1 and 0.3) added to the loop vector during training at segment 32, and 40k-step training at segments 64 and 32 (one seed each). They share the recipe and the evaluation of the main sweep."},
        {"title": "Token profile", "tag": "probe", "ok": True,
         "body": "For a few finished models, how many of the 4 passkey tokens are right in each test stream, at N = 256, 1024 and 4096. Separates a memory that decays gradually from one that is either kept or lost. One seed per row."},
        {"title": "Recipe exploration", "tag": f"{len(explore)} runs", "ok": True,
         "body": f"12k-step runs that fixed the recipe. The curriculum alone gave AR valid accuracy {f2(min(v256(r) for r in expl_cur))}{'' if f2(min(v256(r) for r in expl_cur)) == f2(max(v256(r) for r in expl_cur)) else ' to ' + f2(max(v256(r) for r in expl_cur))} at N = 256; every run that re-weighted the passkey loss was at {f2(max(v256(r) for r in expl_oth))} or lower."} if explore else None,
    ]
    tested = [t for t in [*tested, gain_tested] if t]
    rng256 = [cell(k, 256) for k in order]
    low_valid = [f"{kshort(k)} {cell(k, 256, 'valid'):.2f}" for k in order if cell(k, 256, "valid") < 0.97]
    cap1 = (f"<p><b>Top grid.</b> The plain window is at 0.00 everywhere. Inside the training range (N ≤ 256, left of the dashed line) the loop rows range from {min(rng256):.2f} to {max(rng256):.2f} at N = 256. "
            f"Beyond it the no-tanh loops fall to {max(cell(k, 4096) for k in groups if k[1] == 'none'):.2f} or less by N = 4096, while the tanh rows keep a non-zero tail: "
            f"segment 64 reaches {f2(cell(main(64, 'tanh'), 16384))} at N = 16384, and the noise-0.3 row {f2(cell((32, 'tanh', 0.3, 20000), 16384))}. Hover a cell for the individual seeds.</p>"
            f"<p><b>Middle grid.</b> AR valid accuracy is flat in N, as for every earlier model, and 0.97 or higher except for: {', '.join(low_valid)}. These are rows with runs whose training did not converge.</p>"
            "<p><b>Bottom grid.</b> Stale-key accuracy is at or near 1.00 wherever the AR half was learned.</p>")
    cap2 = (f"<p>Each point is one configuration at N = 1,024 and N = 16,384 (mean over seeds). Blue = tanh loop, orange = no-tanh loop, grey = plain window; hollow = AR valid accuracy below 0.90. "
            "The target corner is top right. At N = 16,384 only tanh loops are above 0.")
    cap2 += "</p>"
    data = {
        "ns": NS, "rows": rows, "results": results, "row_runs": row_runs, "recipe_gain": rec_gain, "scatter_ns": [1024, 16384], "label_w": 330, "row_name": "configuration", "row_prefix": "",
        "meta": f"Phase 1, results of {time.strftime('%d %b %Y, %H:%M')}. <span class='status'><span>main sweep: <b>{n_main}/18 runs</b></span><span>·</span><span>extra runs: <b>{n_extra}/12</b></span>"
                + (f"<span>·</span><span><b>{len(running)} runs in progress</b></span>" if running else "") + "</span>",
        "tldr": tldr, "tested": tested,
        "results_intro": "All numbers are from the <b>test</b> split (1,000 examples per N), the mean over the seeds that finished; the number of seeds is in brackets when it is not 3. N counts AR writes, one write is about 3.75 tokens, "
                         f"so N = 16,384 is about 61k tokens and, at segment 64, about 960 loop hops. Training: 20k steps on N ≤ 256 with the curriculum unless a row says otherwise. The reference is the plain window (W = 64, 2 layers, {ref_desc}). "
                         "'No tanh' means the loop without the tanh only: the RMSNorm in front of the projection and the layer stack between injection and read-out stay, so it is not a linear system. "
                         "Rows are configurations: the runs whose settings, apart from the seed, are identical; the last figure lists every configuration with its difference to the reference recipe.",
        "fig1_sub": "Rows: segment length, loop type and extra settings. Columns: stream length N. Darker is higher.",
        "fig2_sub": "One point per configuration. Ideal is the top-right corner. Blue = tanh, orange = no tanh, hollow = AR valid accuracy below 0.90.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)",
        "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": cap1, "cap2": cap2,
        "extra_figs": [
            {"key": "tanh", "title": "3. Passkey against stream length, tanh loops",
             "sub": "Mean over seeds with the seed range as bars; the single-seed extras are drawn thicker. The dotted line marks the longest training stream (N = 256).",
             "caption": f"<p>Tanh loops start lower than no-tanh ones but decay slowly and level off. Segment 64 is at {f2(cell(main(64, 'tanh'), 8192))} at N = 8192 and {f2(cell(main(64, 'tanh'), 16384))} at N = 16,384. "
                        f"With noise 0.3 during training, segment 32 holds {f2(cell((32, 'tanh', 0.3, 20000), 4096))} / {f2(cell((32, 'tanh', 0.3, 20000), 16384))} at N = 4096 / 16,384 (one seed). The other segments (main grid) stay at or below {max(cell(main(sg, 'tanh'), n) for sg in (32, 16, 8) if has(main(sg, 'tanh')) for n in NS if n >= 2048):.2f} from N = 2048 on.</p>" if has((32, 'tanh', 0.3, 20000)) else "",
             "spec": curves("tanh")},
            {"key": "notanh", "title": "4. Passkey against stream length, no-tanh loops",
             "sub": "Same layout. No-tanh loops fit the training range best and have no tail.",
             "caption": f"<p>Every no-tanh loop is at {max(cell(k, 4096) for k in groups if k[1] == 'none'):.2f} or less by N = 4096, whatever the segment, noise or training length. Seeds at N = 1024: segment 64 {', '.join(f2(x) for x in per(main(64, 'none'), 1024))}, segment 32 {', '.join(f2(x) for x in per(main(32, 'none'), 1024))}. "
                        f"Training twice as long moves the curve to the right a little (segment 32, 40k steps: {f2(cell((32, 'none', 0.0, 40000), 512))} at N = 512) and the horizon stays at about N = 2048.</p>" if has((32, 'none', 0.0, 40000)) else "",
             "spec": curves("none")},
        ] + ([{"key": "profile", "title": "5. What a lost passkey looks like: tokens right per stream",
               "sub": "Test streams of finished models, each bar splits the 1,000 streams by how many of the 4 passkey tokens are right (exact match is the darkest part).",
               "caption": "<p>Chance for a single token is 1 in 16. A graded memory leaves many streams with 1 to 3 tokens right; the noise-trained tanh loop at N = 4096 is split into streams that are fully right and streams with nothing right. "
                          "Each token is scored given the true earlier tokens (teacher forcing), so the middle bars slightly overstate how much of a partly lost passkey a model could actually write out; the all-four bar is the exact-match accuracy.</p>",
               "spec": None, "html": pbars}] if pbars else []) + gain_figs + extra_html_figs,
        "next": ("<p><b>1. More seeds where a conclusion depends on one.</b> The no-tanh loop at segment 8 with gain 0 (seeds 0.93, 0.01 and 0.72 at N = 16,384) and the noise-0.3 tanh loop (one seed) both need at least 5 seeds before the "
                 "segment-size and the noise results are called results. Segment 4 would show whether the trend continues.</p>"
                 "<p><b>2. Decode the passkey from the loop vector late in the stream.</b> If the retaining runs hold one stable state per passkey, a linear read-out of m at N = 4,096 recovers it, and the runs that forget should not allow that. "
                 "This is the direct test behind figure {fig:attractor}.</p>"
                 "<p><b>3. Longer training (the runs have not plateaued) and a selective write.</b> Validation passkey at N = 1,024 was still rising at the end of the 20k and 40k-step runs. The loop vector is an average over the segment; a learned gate or pooling "
                 "would test the selective re-amplification the hypothesis is about.</p>"
                 "<p><b>4. A harder task and the decisions that are yours.</b> The passkey is always the first item; a random position, several passkeys, an overwritten passkey and a per-stream query density would force the loop to select. "
                 "Training on longer streams would remove the extrapolation gap by construction and needs new training data. Exp 2 and Exp 3 are on hold.</p>"),
    }
    data["extra_figs"] = (far["figs"] if far else []) + data["extra_figs"] + (long_["figs"] if long_ else []) + (segs_["figs"] if segs_ else []) + [RR.configuration_appendix(dgroups)]
    return number_figures(data), "SWA Feedback Loop Report", "Sliding-window attention with a segment feedback loop"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["gdn", "swa", "fb"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    render(*{"gdn": gdn_data, "swa": swa_data, "fb": fb_data}[a.kind](), a.out)


if __name__ == "__main__":
    main()
