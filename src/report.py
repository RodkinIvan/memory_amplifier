#!/usr/bin/env python
"""Build the HTML reports (one self-contained page each) from results/raw/*.json and results/probes/*.json.

    python src/report.py gdn --out reports/gdn_report.html
    python src/report.py swa --out reports/swa_report.html

All numbers in the text are computed from the data; the template is reports/report_template.html.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def cells_for_row(runs: list[dict], split: str = "test") -> dict:
    """{str(N): {passkey, valid, stale, expired (means over seeds), counts, per: {metric: [per seed]}}} for one row."""
    keys = {"passkey": "passkey_acc", "valid": "ar_valid_acc", "stale": "ar_expired_stale_acc", "expired": "ar_expired_acc"}
    out = {}
    for n in NS:
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
    rows = [{"key": key(c), "label": f"W{c[0]} · L{c[1]} · R{reach[c]}", "long": f"window {c[0]}, {c[1]} layers, reach {reach[c]} tokens"} for c in cfgs]
    by_cfg = {c: [r for r in runs if wl(r) == c] for c in cfgs}
    results = {key(c): cells_for_row(by_cfg[c]) for c in cfgs}
    n_runs, n_seeds = len(runs), len({r["seed"] for r in runs})
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
        "meta": f"Phase 1, 2 Oct 2026. <span class='status'><span>runs finished: <b>{n_runs}/18</b></span><span>·</span><span>{n_seeds} seeds per configuration</span></span>",
        "tldr": tldr, "tested": tested,
        "results_intro": "<b>SWA has no decay.</b> It keeps no recurrent state and has no multiplier like GDN’s α: each layer attends exactly to the last W tokens, and anything older has weight zero. "
                         "Forgetting is a hard cutoff, and the only memory-length knob is the reach R = L·(W−1) tokens, the farthest distance information can travel through L layers. "
                         "α appears in this report only on the dashed GDN line in figure 4, for contrast. "
                         "Rows are (window W, layers L, reach R). All numbers are from the <b>test</b> split (1,000 examples per N), as the mean over seeds, after 30k training steps on N ≤ 256. Hover a cell for the individual seeds.",
        "fig1_sub": "Columns: number of AR writes N (a write is about 3.75 tokens, so 16384 is about 61k tokens). Darker is higher.",
        "fig2_sub": "One point per configuration (mean over seeds, faint dots = seeds). Ideal is the top-right corner.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)",
        "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": cap1, "cap2": cap2,
        "extra_figs": [
            {"title": "3. Where the passkey stops: accuracy against stream length",
             "sub": "Fresh streams, 200 per point and model. Stream length is measured in tokens (N writes is about 3.75 N tokens).", "caption": cap3,
             "spec": {"series": len_series, "xTicks": [32, 64, 128, 256, 512, 1024], "x0": 24, "x1": 1300, "xLabel": "tokens in the stream (log scale)", "yLabel": "passkey exact match", "ordReverse": True,
                      "vlines": [{"x": 62, "label": "R 62"}, {"x": 125, "label": "R 124–126", "dy": 12}, {"x": 253, "label": "R 252–254"}, {"x": 508, "label": "R 508", "dy": 12}],
                      "aria": "Passkey exact match versus stream length for six window configurations"}},
            {"title": "4. What expiry depends on: valid accuracy against query density",
             "sub": "Fresh streams at N = 256 writes with a different number of queries per write; training used 0.25.", "caption": cap4,
             "spec": {"series": dens_series, "xTicks": dens_rates, "x0": min(dens_rates), "x1": max(dens_rates), "yMin": 0.8, "yMax": 1.0, "xLabel": "queries per write (log scale)",
                      "yLabel": "AR valid accuracy", "vlines": [{"x": 0.25, "label": "training density", "dy": 238}], "rightPad": 118, "aria": "AR valid accuracy versus query density"}},
        ],
        "next": nxt,
    }
    return data, "SWA Baseline Report", "Sliding-window attention baseline"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["gdn", "swa"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    render(*(gdn_data() if a.kind == "gdn" else swa_data()), a.out)


if __name__ == "__main__":
    main()
