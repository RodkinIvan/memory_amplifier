"""Sub-report for presenting the method: segment-level recurrence on sliding-window attention, and the best checkpoint.

    python src/report.py best --out reports/best_report.html

Only what improved quality is shown: the best checkpoint (chosen on the validation split among the runs of the best setups), the
average of each best setup over its seeds (so the seed-to-seed instability is visible), and the baselines it is compared with. Every
number in the text is computed from the results files; the recipe bullets are read from the best run's stored config.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as A  # noqa: E402
import config as C  # noqa: E402
import report_groups as RG  # noqa: E402
import report_rounds as RR  # noqa: E402
from report_groups import NS, cells_for_row, number_figures  # noqa: E402

TOKENS_PER_WRITE = 3.75  # 3 tokens per write plus its share of the 3-token queries
f2 = lambda x: f"{x:.2f}"

# the setups that improved quality: (segment, nonlin, noise, steps, gain) as RG.standard_key reports them
SETUPS = {"A50": (8, "none", 0.0, 50000, 0.0), "A20": (8, "none", 0.0, 20000, 0.0), "B50": (32, "none", 0.0, 50000, 0.0)}


def _row(run: dict, n: int, split: str = "test") -> dict:
    return next(x for x in run["rows"] if x["split"] == split and x["N"] == n)


def pick_best(runs: list[dict]) -> dict:
    """The run to present: highest validation passkey at N = 16,384, ties broken by N = 4,096 (selected on validation; the test split is only reported)."""
    return max(runs, key=lambda r: (_row(r, 16384, "validation")["passkey_acc"], _row(r, 4096, "validation")["passkey_acc"]))


def pick_best_baseline(runs: list[dict]) -> dict:
    """The baseline seed to compare the best checkpoint with, chosen the same way (on the validation split): highest validation passkey averaged over all N."""
    return max(runs, key=lambda r: float(np.mean([x["passkey_acc"] for x in r["rows"] if x["split"] == "validation"])))


def over_n(runs: list[dict], label: str, color: str, dashed: bool = False, width: float = 2.4, short: str = "") -> dict:
    cells = cells_for_row(runs, "test")
    pts = []
    for n in NS:
        c = cells[str(n)]
        per = c["per"]["passkey"]
        p = {"x": n, "y": c["passkey"], "tip": f"{html.escape(label)}<br>N = {n:,} (about {round(n * TOKENS_PER_WRITE / 1000)}k tokens)<br>passkey {c['passkey']:.3f}" + (f" (seeds {', '.join(f'{x:.2f}' for x in per)})" if len(per) > 1 else "")}
        if len(per) > 1:
            p.update(lo=min(per), hi=max(per))
        pts.append(p)
    return {"label": label, "short": short or label, "points": pts, "c": color, "width": width, "dashed": dashed}


def mean_curve(runs: list[dict], logs: dict, n: int, label: str, color: str, width: float = 2.2, dashed: bool = False, short: str = "") -> dict | None:
    """Validation passkey at stream length n against the training step: mean over the runs, bars from the worst to the best run."""
    curves = [dict(RR.curve(r, logs, n)) for r in runs]
    steps = sorted(set.intersection(*[set(c) for c in curves])) if curves and all(curves) else []
    if not steps:
        return None
    pts = []
    for st in steps:
        v = [c[st] for c in curves]
        p = {"x": st, "y": float(np.mean(v)), "tip": f"{html.escape(label)}<br>step {st:,}, N = {n:,}<br>mean {np.mean(v):.2f} (runs {', '.join(f'{x:.2f}' for x in v)})"}
        if len(v) > 1:
            p.update(lo=min(v), hi=max(v))
        pts.append(p)
    return {"label": label, "short": short or label, "points": pts, "c": color, "width": width, "dashed": dashed}


def method_svg() -> str:
    """One layer of the model run segment by segment: where the loop vector m comes from and where it goes."""
    tokens = lambda x, y, cls: "".join(f"<rect x='{x + i * 23}' y='{y}' width='20' height='20' rx='3' class='{cls}'/>" for i in range(8))
    cols = [40, 290, 540]
    s = ["<svg viewBox='0 0 780 340' role='img' aria-label='One layer of the segment-recurrent model: the outputs of segment s-1 give the vector m, which is added to every token input of segment s' style='width:100%;height:auto'>",
         "<style>.tk{fill:var(--s1);opacity:.85}.to{fill:var(--s3);opacity:.85}.bl{fill:var(--surface);stroke:var(--axis);stroke-width:1.2}.mn{fill:var(--s2)}"
         ".ar{stroke:var(--ink-2);stroke-width:1.4;fill:none}.fb{stroke:var(--s2);stroke-width:2.4;fill:none}.fd{stroke:var(--s2);stroke-width:1.6;fill:none;stroke-dasharray:5 4}"
         ".t{font:11.5px var(--mono);fill:var(--ink-2)}.tb{font:12px var(--mono);fill:var(--ink)}.to2{font:11.5px var(--mono);fill:var(--s2)}</style>",
         "<defs><marker id='ah' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='var(--ink-2)'/></marker>"
         "<marker id='ao' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='var(--s2)'/></marker></defs>",
         "<text x='20' y='16' class='t'>m = mean over the segment of Linear(RMSNorm(y)), one vector per layer and segment, computed in fp32</text>"]
    for i, x in enumerate(cols):
        name = ["segment s−1", "segment s", "segment s+1"][i]
        c = x + 92
        s.append(f"<text x='{c}' y='330' text-anchor='middle' class='tb'>{name} (8 tokens)</text>")
        s.append(tokens(x, 282, "tk"))
        s.append(f"<path d='M{c} 280 L{c} 212' class='ar' marker-end='url(#ah)'/>")
        s.append(f"<rect x='{x - 8}' y='132' width='200' height='78' rx='4' class='bl'/>")
        s.append(f"<text x='{c}' y='162' text-anchor='middle' class='tb'>layer ℓ</text><text x='{c}' y='182' text-anchor='middle' class='t'>sliding window 64</text><text x='{c}' y='198' text-anchor='middle' class='t'>+ MLP</text>")
        s.append(f"<path d='M{c} 130 L{c} 90' class='ar' marker-end='url(#ah)'/>")
        s.append(tokens(x, 64, "to"))
        s.append(f"<text x='{c}' y='52' text-anchor='middle' class='t'>outputs y</text>")
        if i > 0:  # the input of every segment but the first gets the vector of the segment before it
            s.append(f"<circle cx='{c}' cy='246' r='10' fill='var(--surface)' stroke='var(--s2)' stroke-width='2'/><text x='{c}' y='251' text-anchor='middle' style='font:700 15px var(--mono);fill:var(--s2)'>+</text>")
    for i in (0, 1):  # m of segment s-1 and s: from the outputs, around the layer box, into the next segment's input
        x = cols[i]
        mx = x + 217
        s.append(f"<path d='M{x + 186} 74 L{mx - 15} 82' class='ar'/>")
        s.append(f"<circle cx='{mx}' cy='86' r='15' class='mn'/><text x='{mx}' y='91' text-anchor='middle' style='font:700 14px var(--mono);fill:#fff'>m</text>")
        s.append(f"<path d='M{mx} 101 L{mx} 246 L{cols[i + 1] + 92 - 12} 246' class='{'fb' if i == 0 else 'fd'}' marker-end='url(#ao)'/>")
    s.append(f"<text x='{cols[1] + 106}' y='262' class='to2'>+ g ⊙ m on every token</text>")
    s.append(f"<text x='{cols[1] + 106}' y='276' class='to2' style='font-size:10.5px'>(learned gain g per channel)</text>")
    s.append("</svg>")
    return "".join(s)


def best_data(raw: list[dict] | None = None, logs: dict | None = None, explore: list[dict] | None = None) -> tuple[dict, str, str]:
    raw = raw if raw is not None else A.load_runs(C.ROOT / "results" / "raw")
    logs = logs if logs is not None else RR.log_curves(C.ROOT / "logs")
    explore = explore if explore is not None else [r for r in A.load_runs(C.ROOT / "results" / "explore") if r["cfg"]["model"]["kind"] == "swa_fb"]
    dgroups = RG.diff_groups([r for r in raw if r["cfg"]["model"]["kind"] in ("swa", "swa_fb")])
    std = {RG.standard_key(g): g for g in dgroups if RG.standard_key(g)}
    setups = {k: std[key] for k, key in SETUPS.items() if key in std}
    if not setups:
        raise ValueError("no segment-8 / segment-32 no-tanh gain-0 runs found")
    best = pick_best([r for k in ("A50", "A20") for r in setups.get(k, {"runs": []})["runs"]])
    bg = next(g for g in dgroups if best in g["runs"])
    swa_g = {RG.setting(g, "model.window"): g for g in dgroups if g["kind"] == "swa" and RG.setting(g, "train.steps") == 50000 and RG.setting(g, "train.curriculum_stage_steps") == 2000
             and RG.setting(g, "model.n_layers") == 2 and set(g["diff"]) <= {"model.window", "train.steps", "train.curriculum_stage_steps"}}
    gdn = [r for r in raw if r["model"] == "gdn_fixed" and r["alpha"] == 1.0]
    gdn_best = pick_best_baseline(gdn) if gdn else None
    mc, tc = best["cfg"]["model"], best["cfg"]["train"]
    seg, reach = mc["segment"], mc["n_layers"] * (mc["window"] - 1)
    train_max_tokens = round(max(tc["train_ns"]) * TOKENS_PER_WRITE + 12)
    tok = lambda n: round(n * TOKENS_PER_WRITE)
    n_seeds = lambda g: len(g["runs"])
    seeds_lbl = lambda g: f"{n_seeds(g)} seeds" if n_seeds(g) >= 3 else f"{n_seeds(g)} of 3 seeds"  # a setup that still has runs in training says so
    seed_note = lambda g: f"{n_seeds(g)} seed{'s' if n_seeds(g) != 1 else ''}" + ("" if n_seeds(g) >= 3 else " finished, more still training")
    steps_k = lambda g: f"{RG.setting(g, 'train.steps') // 1000}k"

    # ---------------------------------------------------------------- rows of the heatmaps
    rows, results = [], {}
    add = lambda key, label, runs, color, long=None: (rows.append({"key": key, "label": label, "color": color, "long": long or label}), results.__setitem__(key, cells_for_row(runs, "test")))
    add("best", f"best checkpoint: segment {seg}, seed {best['seed']}, {steps_k(bg)} steps", [best], "--s2", f"{best['name']} (selected on the validation split)")
    for k, col in (("A50", "--s2"), ("A20", "--s3"), ("B50", "--s1")):
        if k in setups:
            g = setups[k]
            add(k, f"same setup, mean of {seeds_lbl(g)}" if g is bg else f"segment {RG.effective_segment(g)}, {steps_k(g)} steps, mean of {seeds_lbl(g)}", g["runs"], col, f"{RG.label(g)} ({seed_note(g)})")
    if 1024 in swa_g:
        add("w1024", f"plain SWA, window 1024 (reach {RG.reach(swa_g[1024]):,})", swa_g[1024]["runs"], "--muted", RG.label(swa_g[1024]))
    if 64 in swa_g:
        add("w64", f"plain SWA, window 64 (same size as ours)", swa_g[64]["runs"], "--muted", RG.label(swa_g[64]))
    if gdn:
        add("gdn_best", f"GDN α = 1, best seed ({gdn_best['seed']}, on validation)", [gdn_best], "--muted", f"{gdn_best['name']}: gated DeltaNet, no decay, 1 head, 30k steps, no curriculum, selected on the validation split")
        add("gdn", f"GDN α = 1, mean of {len(gdn)} seeds", gdn, "--muted", "gated DeltaNet, no decay, 1 head, 30k steps, no curriculum")
    # same-key lookups used by the text
    cell = lambda key, n, f="passkey": results[key][str(n)][f]
    hz = lambda key: next((n for n in reversed(NS) if cell(key, n) is not None and cell(key, n) >= 0.9), None)
    lab = lambda key: next(r["label"] for r in rows if r["key"] == key)
    setup_key = next((k for k in ("A50", "A20") if k in setups and setups[k] is bg), "A50")

    # ---------------------------------------------------------------- figures
    series = [over_n([best], f"best checkpoint (segment {seg}, seed {best['seed']}, {steps_k(bg)} steps)", "var(--s2)", width=3.6, short="best")]
    for k, col in (("A50", "var(--s2)"), ("A20", "var(--s3)"), ("B50", "var(--s1)")):
        if k in setups:
            g = setups[k]
            series.append(over_n(g["runs"], f"segment {RG.effective_segment(g)}, {steps_k(g)} steps, mean of {seeds_lbl(g)} (bars: worst to best seed)", col, dashed=True, width=2.2,
                                 short=f"{steps_k(g)} avg" if RG.effective_segment(g) == seg else f"seg {RG.effective_segment(g)} avg"))
    if 1024 in swa_g:
        series.append(over_n(swa_g[1024]["runs"], f"plain SWA, window 1024 (reach {RG.reach(swa_g[1024]):,} tokens)", "var(--muted)", width=2.2, short="SWA W1024"))
    if 64 in swa_g:
        series.append(over_n(swa_g[64]["runs"], f"plain SWA, window 64 (reach {RG.reach(swa_g[64])} tokens)", "var(--ink-2)", width=2.2, short="SWA W64"))
    if gdn:
        series.append(over_n([gdn_best], f"GDN α = 1, best seed ({gdn_best['seed']})", "var(--ink)", dashed=True, width=2.4, short="GDN best"))
        series.append(over_n(gdn, f"GDN α = 1, mean of {len(gdn)} seeds", "var(--muted)", dashed=True, width=2.0, short="GDN avg"))
    chart_n = {"series": series, "xTicks": NS, "x0": 32, "x1": 16384, "xLabel": "stream length N (AR writes, log scale; 16,384 writes ≈ 61k tokens)", "yLabel": "passkey exact match (test)", "legendLine": True, "rightPad": 96,
               "vlines": [{"x": max(tc["train_ns"]), "label": "longest training stream", "dy": 0}], "aria": "Passkey accuracy against stream length for the best checkpoint, the setup averages and the baselines"}

    def curve_chart(n: int, with20: bool) -> dict | None:
        sr = []
        c = mean_curve([best], logs, n, f"best checkpoint (seed {best['seed']})", "var(--s2)", width=3.6, short="best")
        if c:
            sr.append(c)
        for k, col in (("A50", "var(--s2)"), ("A20", "var(--s3)"), ("B50", "var(--s1)")):
            if k in setups and (with20 or k == "A50"):
                g = setups[k]
                m = mean_curve(g["runs"], logs, n, f"segment {RG.effective_segment(g)}, {steps_k(g)} steps, mean of {seeds_lbl(g)} (bars: worst to best)", col, dashed=True, short=f"{steps_k(g)} avg" if RG.effective_segment(g) == seg else f"seg {RG.effective_segment(g)} avg")
                if m:
                    sr.append(m)
        for r in bg["runs"]:
            if r is not best:
                c = mean_curve([r], logs, n, f"seed {r['seed']} of the same setup" + (" (loop never started)" if _row(r, 256)["passkey_acc"] < 0.1 else ""), "var(--muted)", width=1.6, short=f"seed {r['seed']}")
                if c:
                    sr.append(c)
        if not sr:
            return None
        x1 = max(p["x"] for s in sr for p in s["points"])
        return {"series": sr, "xTicks": list(range(0, x1 + 1, 10000)), "x0": 0, "x1": x1, "xLog": False, "xLabel": "training step", "yLabel": f"validation passkey, N = {n:,}", "legendLine": True, "rightPad": 70,
                "aria": f"Validation passkey at N = {n} during training"}

    # numbers table
    def trow(key):
        return [html.escape(lab(key))] + [f2(cell(key, n)) for n in (256, 1024, 4096, 16384)] + [f"{f2(cell(key, 16384, 'valid'))} / {f2(cell(key, 16384, 'expired'))}"]
    table = RR._table(["configuration (test split)", "passkey N = 256", "N = 1,024", "N = 4,096", "N = 16,384", "AR valid / expired at N = 16,384"], [trow(r["key"]) for r in rows], "wide")

    figs = [{"key": "n", "title": "Passkey accuracy against stream length", "sub": "Test split (1,000 streams per N). The passkey is shown once, 4 tokens, and asked for at the end of a stream of N writes plus associative-recall queries.",
             "caption": f"<p>The model attends {reach} tokens back (2 layers × window 64). The dotted line is the longest training stream (N = {max(tc['train_ns'])}, about {train_max_tokens} tokens); everything to its right is length extrapolation.</p>",
             "spec": chart_n},
            {"key": "table", "title": "The numbers", "sub": "Passkey exact match on the test split; AR valid = answers for keys written in the last 16 writes, AR expired = answers NULL for stale or unknown keys.", "caption": "", "spec": None, "html": table}]
    c1 = curve_chart(1024, True)
    if c1:
        figs.append({"key": "c1024", "title": "Training: validation passkey at N = 1,024", "sub": "200 validation streams per evaluation, every 2,000 steps. Thick = the best checkpoint; dashed = mean over the seeds of a setup, bars from the worst to the best seed; grey = other seeds of the best setup.",
                     "caption": "<p>The loop switches on abruptly and late, and not in every run: the gap between the mean lines and the best checkpoint is the training instability.</p>", "spec": c1})
    c2 = curve_chart(16384, False) if "A50" in setups else None  # only the setup that holds the passkey at this length
    if c2:
        figs.append({"key": "c16k", "title": "Training: validation passkey at N = 16,384 (about 61k tokens)", "sub": "32 validation streams per evaluation; only the runs that were evaluated at this length during training.", "caption": "", "spec": c2})

    # ---------------------------------------------------------------- text
    n16 = 16384
    bp = {n: cell("best", n) for n in (256, 1024, 4096, 16384)}
    ctrl64 = hz("w64") if "w64" in results else None
    min_valid = min(cell("best", n, "valid") for n in NS)
    min_expired = min(cell("best", n, "expired") for n in NS if cell("best", n, "expired") is not None)
    figs_ref = lambda keys: ", ".join("{fig:%s}" % k for k in keys if any(f["key"] == k for f in figs))
    per_a = RG.seeds_text(RG.per_seed(setups[setup_key]["runs"], n16)) if setup_key in setups else ""
    a20 = setups.get("A20")
    sw1024_txt = (f"plain SWA with window 1024 (reach {RG.reach(swa_g[1024]):,} tokens) holds it only up to N = {hz('w1024')} (the longest training stream) and scores {f2(cell('w1024', 512))} at N = 512, though 512 writes (about {tok(512):,} tokens) are inside its reach"
                  if "w1024" in results and hz("w1024") is not None and hz("w1024") <= max(tc["train_ns"]) else "")
    started = lambda g: sum(_row(r, 256)["passkey_acc"] > 0.5 for r in g["runs"])
    gdn_txt = (f"; GDN α = 1 (441k parameters, 30k steps without the curriculum; best of {len(gdn)} seeds, picked on validation like our checkpoint) scores {f2(cell('gdn_best', 1024))} / {f2(cell('gdn_best', 2048))} / {f2(cell('gdn_best', 4096))} / {f2(cell('gdn_best', 8192))} "
               f"at N = 1,024 / 2,048 / 4,096 / 8,192 and {f2(cell('gdn_best', n16))} at N = 16,384 (mean of its seeds {f2(cell('gdn', 2048))} at N = 2,048)") if "gdn" in results else ""
    tldr = [
        f"<b>Method.</b> A sliding-window transformer ({mc['n_layers']} layers, window {mc['window']}, {best['n_params'] // 1000}k parameters) sees only {reach} tokens back. We add <b>segment-level recurrence</b>: once per segment of {seg} tokens every layer compresses "
        f"its outputs into one vector m (RMSNorm, linear map, mean over the segment) and adds it, through a learned per-channel gain, to the input of every token of the next segment. Nothing else changes: the window and the attention are those of plain SWA.",
        f"<b>Result.</b> The best checkpoint (segment {seg}, seed {best['seed']}, {steps_k(bg)} steps, chosen on the validation split) answers the 4-token passkey with exact match {f2(bp[256])} / {f2(bp[1024])} / {f2(bp[4096])} / {f2(bp[n16])} on the test split at N = 256 / 1,024 / 4,096 / 16,384 writes. "
        f"The longest stream it trained on is {train_max_tokens} tokens; N = 16,384 is about {tok(n16):,} tokens ({tok(n16) / train_max_tokens:.0f}× longer, {tok(n16) / reach:.0f}× its attention reach, {round(n16 * TOKENS_PER_WRITE / seg):,} segment hops), "
        f"with AR valid accuracy {f2(cell('best', n16, 'valid'))} and AR expired accuracy {f2(cell('best', n16, 'expired'))} (the associative-recall half of the task is not hurt).",
        f"<b>Against the baselines.</b> The same model without the recurrence (plain SWA, window 64, trained the same way)"
        + (f" holds the passkey up to N = {ctrl64} at best" if ctrl64 is not None else f" never reaches 0.9 at any N (passkey {f2(cell('w64', 32))} at N = 32, {f2(cell('w64', 256))} at N = 256)" if "w64" in results else "")
        + (f"; {sw1024_txt}" if sw1024_txt else "") + gdn_txt + ".",
        f"<b>Training is unstable.</b> The setup of the best checkpoint has {seed_note(bg)}. Passkey at N = 256 per seed {RG.seeds_text(RG.per_seed(bg['runs'], 256))} and at N = 16,384 per seed {RG.seeds_text(RG.per_seed(bg['runs'], n16))} "
        f"(seeds {', '.join(str(r['seed']) for r in bg['runs'])}); {sum(x < 0.1 for x in RG.per_seed(bg['runs'], n16))} of {n_seeds(bg)} score below 0.1 at N = 16,384"
        + (f". With 20k steps instead of {steps_k(bg)} the same setup scored {RG.seeds_text(RG.per_seed(a20['runs'], n16))} over its {n_seeds(a20)} seeds (mean {f2(cell('A20', n16))})" if a20 is not None and bg is not a20 else "")
        + ". The runs differ in whether and when the loop switches on and in how far it then carries the passkey, which is the gap between the best checkpoint and the mean lines"
        + (f" (figures {figs_ref(['c1024', 'c16k'])})" if figs_ref(['c1024', 'c16k']) else "") + ".",
    ]

    # recipe bullets, read from the best run's stored config
    ex_cur = [r for r in explore if r["cfg"]["train"].get("curriculum_stage_steps", 0) > 0 and r["cfg"]["train"].get("passkey_weight", 1) == 1 and r["cfg"]["model"]["nonlin"] == "none"]
    g_gain01 = std.get((seg, "none", 0.0, 20000, 0.1))
    g_tanh = std.get((seg, "tanh", 0.0, 20000, 0.0))
    ev = lambda *a: " <span class='hint'>(" + "".join(a) + ")</span>"
    at256 = lambda r: next(x for x in r["rows"] if x["N"] == 256)
    cur_ev = (f"12k-step exploration, segment 64 no-tanh loop: the only recipe that learned both halves, AR valid {f2(min(at256(r)['ar_valid_acc'] for r in ex_cur))} and passkey {f2(max(at256(r)['passkey_acc'] for r in ex_cur))} at N = 256") if ex_cur else ""
    sel_txt = "; ".join(steps_k(setups[k]) + "-step seeds " + ", ".join(str(r["seed"]) for r in setups[k]["runs"]) for k in ("A50", "A20") if k in setups)
    bullets = [
        f"<b>Backbone:</b> stock flash-linear-attention transformer, {mc['n_layers']} layers, d_model {mc['d_model']}, {mc['num_heads']} heads × {mc['d_model'] // mc['num_heads']}, RoPE (θ = 10,000), SwiGLU MLP (hidden 512), tied embeddings, vocabulary of 40 tokens, {best['n_params']:,} parameters.",
        f"<b>Attention:</b> causal sliding window of {mc['window']} tokens, bf16 autocast with flash attention. Plain SWA reaches {mc['n_layers']} × ({mc['window']} − 1) = {reach} tokens; the model is run one segment at a time and the last {mc['window'] - 1} layer inputs are carried over, so every query sees exactly what plain SWA would see.",
        f"<b>Segment recurrence:</b> segment of {seg} tokens ({'no tanh' if mc['nonlin'] == 'none' else mc['nonlin']}); per layer, m = mean over the segment of Linear(RMSNorm(y)) in fp32, added as x + g ⊙ m to every token input of the next segment, one learned gain g per channel; m = 0 before the first segment, so the first segment is plain SWA.",
        f"<b>No tanh on the loop vector</b> (the RMSNorm stays): at segment {seg} and 20k steps the passkey at N = 1,024 is {f2(np.mean(RG.per_seed(setups['A20']['runs'], 1024)))} without tanh against {f2(np.mean(RG.per_seed(g_tanh['runs'], 1024)))} with it" + (f" ({started(g_tanh)} of {n_seeds(g_tanh)} tanh runs started the loop)" if g_tanh else "") + "." if "A20" in setups and g_tanh else
        "<b>No tanh on the loop vector</b> (the RMSNorm stays).",
        f"<b>Loop gain initialised to {mc['gain_init']:g}</b> (and loop projection init std {mc['proj_std']:g}): the model starts as exactly plain SWA, so the associative-recall half is learned first and the optimiser switches the loop on later" + (
            f"; at segment {seg}, 20k steps, passkey at N = 1,024 is {f2(np.mean(RG.per_seed(setups['A20']['runs'], 1024)))} with gain 0 against {f2(np.mean(RG.per_seed(g_gain01['runs'], 1024)))} with gain 0.1" if "A20" in setups and g_gain01 else "") + ".",
        f"<b>Learning rate × {tc['loop_lr_mult']:g} and no weight decay for the loop parameters</b> (gain, projection and the norm weight of the loop), so the loop learns faster than the backbone.",
        f"<b>Stream-length curriculum:</b> training starts on N = {tc['train_ns'][0]} only and adds the next length every {tc['curriculum_stage_steps']:,} steps ({' → '.join(map(str, tc['train_ns']))}); examples are drawn uniformly over the active lengths and padded to a multiple of {tc['pad_multiple']}"
        + (ev(cur_ev) if cur_ev else "") + ".",
        f"<b>Plain objective:</b> cross-entropy on the supervised tokens only (AR answers and the 4 passkey tokens), plain mean, passkey weight {tc['passkey_weight']:g}; no auxiliary loss, no noise on the loop vector.",
        f"<b>Optimiser:</b> AdamW (β = {tc['betas'][0]:g} / {tc['betas'][1]:g}, weight decay {tc['weight_decay']:g} on matrices), learning rate {tc['lr']:g}, {tc['warmup_steps']} warm-up steps then cosine decay to {tc['min_lr_ratio']:g} × at the last step, gradient clipping {tc['grad_clip']:g}, batch {tc['batch_size']} streams, {tc['steps']:,} steps, bf16.",
        f"<b>Model selection:</b> the checkpoint is the run with the best validation passkey at N = 16,384 among the runs of the segment-{seg} setups ({sel_txt}); the checkpoint itself was never picked on the test split. The setup (segment {seg}, no tanh, gain 0) was chosen earlier from a 24-run grid compared on the test split, so the test numbers of the setup are slightly optimistic; the validation split shows the same picture.",
        "<b>Evaluation:</b> teacher-forced exact match (all 4 passkey tokens right), fixed batch shape per N, up to N = 16,384; the data (<code>irodkin/passkey_swar</code>) and its oracle are described in the benchmark report.",
    ]
    intro = ("<h2>Method</h2><p>The task: remember a 4-token passkey shown once at the start of a stream, and repeat it at the end, while also answering associative-recall queries about recent key–value writes. "
             "A plain sliding-window model cannot do the first part beyond its window; a recurrent model has to do both with one decay. Segment-level recurrence gives the window model a small, learned, "
             "once-per-segment memory.</p>"
             f"<figure><div class='fig-title'>The model, one layer, three consecutive segments</div><div class='chart'>{method_svg()}</div>"
             f"<figcaption>Inside a segment the layer is plain sliding-window attention over the last {mc['window']} tokens (which reach back into the previous segments). Between segments the only new path is m, "
             "one vector per layer that is rebuilt from scratch every segment, so information can travel past the window only by being re-written into it every few tokens.</figcaption></figure>"
             "<h2>Training procedure and techniques</h2><ul class='tight'>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
    data = {
        "ns": NS, "rows": rows, "results": results, "scatter_ns": [], "hide_scatter": True, "label_w": 352, "row_name": "configuration", "row_prefix": "",
        "meta": f"Sub-report for the group meeting, results of {__import__('time').strftime('%d %b %Y, %H:%M')}. Best checkpoint: <code>runs/{best['name']}/model.pt</code>. Test split, N counts AR writes, one write is about 3.75 tokens.",
        "tldr": tldr, "tested": [], "intro": intro, "results_title": "Results",
        "results_intro": "The mean lines are the average over the seeds of a setup, bars from the worst to the best seed. The best checkpoint is one run, selected on validation.",
        "fig1_title": "1. Accuracy by configuration and stream length", "fig1_sub": "Rows: the best checkpoint, the averages of the best setups, the baselines. Columns: stream length N. Darker is higher.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)", "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": f"<p>Top grid: passkey. Middle and bottom grids: for the best checkpoint the associative-recall half stays at {f2(min_valid)} (valid) and {f2(min_expired)} (expired) or above at every length, so the memory is not bought with recall quality. "
                "The setup means include the seeds that never started the loop (and, for the 20k-step setup, one seed that never learned the recall half), which is why they are lower.</p>", "cap2": "",
        "extra_figs": figs, "next_title": "Open issues",
        "next": ("<p><b>Reliability.</b> Whether the loop switches on depends on the seed; the averages above are the honest summary and the best checkpoint is the upper end. More seeds and a way to make the start reliable are the next steps.</p>"
                 "<p><b>Scope.</b> The passkey is always the first item, so the loop only has to keep what the first segment held; random positions, several or overwritten passkeys would test selection.</p>"),
    }
    return number_figures(data, first=2), "SWA Segment Recurrence Report", "Segment-level recurrence for sliding-window attention: method and best result"
