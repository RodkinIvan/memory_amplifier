"""Report of design experiment E1: value adaptation of the short-term store (`swa_va`).

    python src/report.py va --out reports/va_report.html

The model keeps its long-range memory in the *values* of the sliding-window key/value cache (a low-rank, test-time adaptation of W_v built from the
previous segment's outputs); no loop vector crosses the segment boundary. The report compares it with the segment-recurrent model (`swa_fb`, the vector m) at the same
segments, with plain sliding-window attention and with GDN-2, all trained with the same recipe (50k steps, curriculum 2000, 2 layers, window 64). Every number in the text is
computed from the results files; rows are configurations, grouped by the diff of their stored config against the frozen reference (H7).
"""
from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as A  # noqa: E402
import config as C  # noqa: E402
import report_groups as RG  # noqa: E402
import report_rounds as RR  # noqa: E402
from report_best import _row, mean_curve, over_n, pick_best  # noqa: E402
from report_groups import NS, cells_for_row, number_figures  # noqa: E402

f2 = lambda x: f"{x:.2f}"
PROBE_DIR = C.ROOT / "results" / "probes"
COLORS = {"va32": "var(--s2)", "va8": "var(--s3)", "fb32": "var(--s2)", "fb8": "var(--s3)"}


def method_svg() -> str:
    """One layer of the value-adaptation model over two consecutive segments: from the outputs of segment s-1 to the values of segment s."""
    tok = lambda x, y, cls, n=6: "".join(f"<rect x='{x + i * 21}' y='{y}' width='18' height='18' rx='3' class='{cls}'/>" for i in range(n))
    s = ["<svg viewBox='0 0 780 330' role='img' aria-label='Value adaptation: the outputs of segment s-1 give a low-rank matrix that is added to W_v for the tokens of segment s' style='width:100%;height:auto'>",
         "<style>.tk{fill:var(--s1);opacity:.85}.to{fill:var(--s3);opacity:.85}.bl{fill:var(--surface);stroke:var(--axis);stroke-width:1.2}.kv{fill:var(--s2);opacity:.9}"
         ".ar{stroke:var(--ink-2);stroke-width:1.4;fill:none}.fb{stroke:var(--s2);stroke-width:2.4;fill:none}.t{font:11.5px var(--mono);fill:var(--ink-2)}.tb{font:12px var(--mono);fill:var(--ink)}"
         ".to2{font:11.5px var(--mono);fill:var(--s2)}</style>",
         "<defs><marker id='ah' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='var(--ink-2)'/></marker>"
         "<marker id='ao' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='var(--s2)'/></marker></defs>",
         "<text x='16' y='16' class='t'>per layer and segment: keys a_j = tanh(W_y·norm(o_j)), values b_j = tanh(W_x·norm(o_j)) from the block outputs o_j; ΔW_v = (g / C) Σ_j b_j a_jᵀ</text>"]
    # left: segment s-1
    s.append("<text x='120' y='318' text-anchor='middle' class='tb'>segment s−1 (C tokens)</text>")
    s.append(tok(40, 270, "tk"))
    s.append("<path d='M120 266 L120 206' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='30' y='132' width='180' height='72' rx='4' class='bl'/><text x='120' y='160' text-anchor='middle' class='tb'>layer ℓ</text><text x='120' y='178' text-anchor='middle' class='t'>window attention + MLP</text>")
    s.append("<path d='M120 130 L120 98' class='ar' marker-end='url(#ah)'/>")
    s.append(tok(40, 76, "to"))
    s.append("<text x='120' y='64' text-anchor='middle' class='t'>outputs o_j</text>")
    # middle: the matrix
    s.append("<path d='M212 86 L292 112' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='296' y='84' width='150' height='62' rx='4' class='bl'/><text x='371' y='108' text-anchor='middle' class='tb'>ΔW_v</text>")
    s.append("<text x='371' y='126' text-anchor='middle' class='t'>(g/C) Σ b_j a_jᵀ</text><text x='371' y='140' text-anchor='middle' class='t'>rank ≤ C, |entries| ≤ |g|</text>")
    # right: segment s
    s.append("<text x='640' y='318' text-anchor='middle' class='tb'>segment s</text>")
    s.append(tok(560, 270, "tk"))
    s.append("<path d='M640 266 L640 206' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='530' y='132' width='220' height='72' rx='4' class='bl'/><text x='640' y='156' text-anchor='middle' class='tb'>layer ℓ</text><text x='640' y='174' text-anchor='middle' class='t'>v_t = W_v h_t + ΔW_v h_t</text><text x='640' y='192' text-anchor='middle' class='to2'>stored once in the window's cache</text>")
    s.append("<path d='M640 130 L640 98' class='ar' marker-end='url(#ah)'/>")
    s.append(tok(560, 76, "to"))
    s.append("<path d='M448 112 L526 150' class='fb' marker-end='url(#ao)'/>")
    s.append("<text x='474' y='214' class='to2'>queries and keys unchanged;</text><text x='474' y='228' class='to2'>no other path between segments</text>")
    s.append("</svg>")
    return "".join(s)


def load_probes(directory: Path = PROBE_DIR) -> dict:
    """{'cache': ..., 'delta': ..., 'perturb': ...} from results/probes/va_*.json (whatever exists)."""
    out = {}
    for key, name in (("cache", "va_cache_probe.json"), ("delta", "va_delta_w.json"), ("perturb", "va_perturb.json")):
        f = directory / name
        if f.exists():
            out[key] = json.loads(f.read_text())
    return out


def probe_figures(probes: dict) -> list[dict]:
    """Tables of the three diagnostics, for the runs that were probed (empty when none was)."""
    figs = []
    for run, res in probes.get("cache", {}).items():
        rows = []
        for d, e in res["models"]["model"]["by_distance"].items():
            ctl = res["models"].get("control", {}).get("by_distance", {}).get(d)
            rows.append([f"{int(d):,}", f"{e['tokens']:,}", f2(e["v"]["mean"]), f2(e["k"]["mean"]), f2(e["dw"]["mean"]) if "dw" in e else "–"]
                        + ([f2(ctl["v"]["mean"]), f2(ctl["k"]["mean"])] if ctl else []))
        ctl_run = res["models"].get("control", {}).get("run")
        head = ["distance after the passkey (tokens)", "cut at token", "values", "keys", "ΔW_v"] + (["control: values", "control: keys"] if ctl_run else [])
        figs.append({"key": f"probe_cache_{run}", "title": f"Passkey read out of the cached window: {run}",
                     "sub": f"Linear (ridge) read-out of each passkey token from the {res['models']['model']['reach'] // 2} cached positions of both layers; mean accuracy over the {res['passkey_len']} tokens, chance {res['chance']:.3f}; "
                            f"{res['n_train']} streams to fit, {res['n_test']} to test.",
                     "caption": "<p>The read-out never sees the passkey tokens themselves, only the cached values (or keys, or the adaptation matrix) after the given number of tokens. "
                                f"The reach of a plain window model with these dimensions is {res['models']['model']['reach']} tokens" + (f"; the control ({html.escape(ctl_run)}) is a plain sliding-window model, whose cache cannot hold the passkey beyond that." if ctl_run else ".") + "</p>",
                     "spec": None, "html": RR._table(head, rows)})
    for run, res in probes.get("delta", {}).items():
        rows = []
        for i, layer in enumerate(res["layers"]):
            for frac, e in layer["at"].items():
                rows.append([str(i), f"{float(frac):.0%}", f"{e['segment']:,}", f"{layer['gain']:.3f}", f"{e['fro']:.3f}", f2(e["fro_over_w_v"]), f"{e['spectral']:.3f}", f"{e['rank']:.1f}", f"{e['max_entry']:.3f}", f2(e["cos_next"]), f2(e["cos_lag16"])])
        figs.append({"key": f"probe_delta_{run}", "title": f"Size, rank and drift of ΔW_v along a stream: {run}",
                     "sub": f"{res['n_writes']:,} writes per stream (segments: {res['segments']:,}), mean over the probe streams; rank = participation ratio of the singular values (at most C = {res['segment']}).",
                     "caption": "<p>Size is relative to the layer's own W_v. A cosine near 1 between ΔW_v of consecutive segments (or 16 segments apart) means the adaptation has settled into a fixed point; near 0 that it is rewritten every segment.</p>", "spec": None,
                     "html": RR._table(["layer", "position in stream", "segment", "gain g", "‖ΔW_v‖_F", "÷ ‖W_v‖_F", "spectral norm", "rank", "largest entry", "cos next", "cos +16"], rows)})
    for run, res in probes.get("perturb", {}).items():
        ex = res["exact"]
        eps = sorted({float(k.split("_")[1]) for k in ex if k.split("_")[0] in ("values", "state", "both") and k.split("_")[1] != "zero"})
        rows = [[who] + [f2(ex[f"{key}_{e:g}"]) for e in eps] + [f2(ex[f"{key}_zero"])] for who, key in (("cached values", "values"), ("previous segment's encoding (state)", "state"), ("both", "both"))]
        con = res["contraction"]
        figs.append({"key": f"probe_perturb_{run}", "title": f"Perturbing the memory in mid-stream: {run}",
                     "sub": f"Passkey exact match at N = {res['n_writes']:,} (clean: {f2(ex['clean'])}); the kick lands after segment {res['s_star']:,} (token {res['s_star_tokens']:,}), far beyond the reach. Columns: noise of ε × the tensor's RMS, and zeroing.",
                     "caption": "<p>Distance between the cached values of a clean and a perturbed run, relative to right after the kick (median over streams), after "
                                + ", ".join(f"{k} segments: {v:.2f}" for k, v in con.items()) + ". Well below 1 means the memory pulls itself back to the clean trajectory; near 1 that the kick is carried along unchanged.</p>", "spec": None,
                     "html": RR._table(["perturbed"] + [f"ε = {e:g}" for e in eps] + ["zeroed"], rows)})
    return figs


def va_data(raw: list[dict] | None = None, logs: dict | None = None, probes: dict | None = None) -> tuple[dict, str, str]:
    raw = raw if raw is not None else A.load_runs(C.ROOT / "results" / "raw")
    logs = logs if logs is not None else RR.log_curves(C.ROOT / "logs")
    probes = probes if probes is not None else load_probes()
    dgroups = RG.diff_groups([r for r in raw if r["cfg"]["model"]["kind"] in ("swa", "swa_fb", "swa_va", "gdn2")])
    va = {RG.effective_segment(g): g for g in dgroups if g["kind"] == "swa_va" and set(g["diff"]) <= {"model.segment"}}
    if not va:
        raise ValueError("no swa_va runs found")
    fb = {seg: g for g in dgroups if (key := RG.standard_key(g)) and key[1:] == ("none", 0.0, 50000, 0.0) and (seg := key[0]) in (8, 32)}
    swa64 = next((g for g in dgroups if g["kind"] == "swa" and RG.setting(g, "model.window") == 64 and RG.setting(g, "train.steps") == 50000 and RG.setting(g, "train.curriculum_stage_steps") == 2000
                  and RG.setting(g, "model.n_layers") == 2 and set(g["diff"]) <= {"model.window", "train.steps", "train.curriculum_stage_steps"}), None)
    gdn2 = next((g for g in dgroups if g["kind"] == "gdn2" and RG.setting(g, "train.steps") == 50000 and RG.setting(g, "train.curriculum_stage_steps") == 2000
                 and set(g["diff"]) <= {"train.steps", "train.curriculum_stage_steps"}), None)
    segs = sorted(va, reverse=True)  # 32 first
    ref_mc, tc = C.load_reference({"model": {"kind": "swa_va"}})["model"], next(iter(va.values()))["runs"][0]["cfg"]["train"]
    reach = ref_mc["n_layers"] * (ref_mc["window"] - 1)
    seeds_lbl = lambda g: f"{len(g['runs'])} seed{'s' if len(g['runs']) != 1 else ''}"

    # ---------------------------------------------------------------- heatmap rows
    rows, results = [], {}
    add = lambda key, label, runs, color, long=None: (rows.append({"key": key, "label": label, "color": color, "long": long or label}), results.__setitem__(key, cells_for_row(runs, "test")))
    for seg in segs:
        g = va[seg]
        best = pick_best(g["runs"]) if all(any(x["N"] == 16384 and x["split"] == "validation" for x in r["rows"]) for r in g["runs"]) else g["runs"][0]
        add(f"va{seg}", f"value adaptation, segment {seg}, mean of {seeds_lbl(g)}", g["runs"], COLORS[f"va{seg}"], f"{RG.label(g)} ({seeds_lbl(g)}: {', '.join(r['name'] for r in g['runs'])})")
        if len(g["runs"]) > 1:
            add(f"va{seg}_best", f"value adaptation, segment {seg}, best seed ({best['seed']})", [best], COLORS[f"va{seg}"], f"{best['name']} (best seed by validation passkey at N = 16,384)")
    for seg in segs:
        if seg in fb:
            g = fb[seg]
            add(f"fb{seg}", f"vector loop m, segment {seg}, mean of {seeds_lbl(g)}", g["runs"], "--s1", RG.label(g))
    if swa64:
        add("w64", "plain SWA, window 64 (reach 126)", swa64["runs"], "--muted", RG.label(swa64))
    if gdn2:
        add("gdn2", f"GDN-2 (learned decay), mean of {seeds_lbl(gdn2)}", gdn2["runs"], "--muted", "gated DeltaNet, stock learned decay, our recipe")
    cell = lambda key, n, f="passkey": results[key][str(n)][f]
    hz = lambda key: next((n for n in reversed(NS) if cell(key, n) is not None and cell(key, n) >= 0.9), None)
    per = lambda key, n: results[key][str(n)]["per"]["passkey"]

    # ---------------------------------------------------------------- figures
    series = []
    for seg in segs:
        series.append(over_n(va[seg]["runs"], f"value adaptation, segment {seg}, mean of {seeds_lbl(va[seg])} (bars: worst to best seed)", COLORS[f"va{seg}"], width=3.0, short=f"VA seg {seg}"))
    for seg in segs:
        if seg in fb:
            series.append(over_n(fb[seg]["runs"], f"vector loop m, segment {seg}, mean of {seeds_lbl(fb[seg])}", "var(--s1)", dashed=True, width=2.0, short=f"m seg {seg}"))
    if swa64:
        series.append(over_n(swa64["runs"], "plain SWA, window 64", "var(--ink-2)", width=2.0, short="SWA W64"))
    if gdn2:
        series.append(over_n(gdn2["runs"], f"GDN-2, mean of {seeds_lbl(gdn2)}", "var(--ink)", dashed=True, width=2.0, short="GDN-2"))
    chart_n = {"series": series, "xTicks": NS, "x0": 32, "x1": 16384, "xLabel": "stream length N (AR writes, log scale; 16,384 writes ≈ 61k tokens)", "yLabel": "passkey exact match (test)", "legendLine": True, "rightPad": 96,
               "vlines": [{"x": max(tc["train_ns"]), "label": "longest training stream", "dy": 0}], "aria": "Passkey accuracy against stream length for value adaptation, the vector loop and the baselines"}

    def curves(n: int) -> dict | None:
        sr = []
        for seg in segs:
            for g, tag, col, dash in ((va[seg], "value adaptation", COLORS[f"va{seg}"], False), (fb.get(seg), "vector loop m", "var(--s1)", True)):
                if g:
                    c = mean_curve(g["runs"], logs, n, f"{tag}, segment {seg}, mean of {seeds_lbl(g)} (bars: worst to best)", col, width=2.6 if not dash else 1.8, dashed=dash, short=f"{'VA' if not dash else 'm'} seg {seg}")
                    if c:
                        sr.append(c)
        if not sr:
            return None
        x1 = max(p["x"] for s_ in sr for p in s_["points"])
        return {"series": sr, "xTicks": list(range(0, x1 + 1, 10000)), "x0": 0, "x1": x1, "xLog": False, "xLabel": "training step", "yLabel": f"validation passkey, N = {n:,}", "legendLine": True, "rightPad": 70,
                "aria": f"Validation passkey at N = {n} during training"}

    def trow(key):
        return [html.escape(next(r["label"] for r in rows if r["key"] == key))] + [f2(cell(key, n)) for n in (256, 1024, 4096, 16384)] + [f"{f2(cell(key, 16384, 'valid'))} / {f2(cell(key, 16384, 'expired'))}"]
    table = RR._table(["configuration (test split)", "passkey N = 256", "N = 1,024", "N = 4,096", "N = 16,384", "AR valid / expired at N = 16,384"], [trow(r["key"]) for r in rows], "wide")
    figs = [{"key": "n", "title": "Passkey accuracy against stream length", "sub": "Test split (1,000 streams per N). The passkey is shown once, 4 tokens, and asked for at the end of a stream of N writes plus associative-recall queries.",
             "caption": f"<p>Plain window attention reaches {reach} tokens; the dotted line is the longest training stream (N = {max(tc['train_ns'])}). Everything to its right is length extrapolation.</p>", "spec": chart_n},
            {"key": "table", "title": "The numbers", "sub": "Passkey exact match on the test split; AR valid = answers for keys written in the last 16 writes, AR expired = answers NULL for stale or unknown keys.", "caption": "", "spec": None, "html": table}]
    for n, key in ((256, "c256"), (1024, "c1024"), (16384, "c16k")):
        c = curves(n)
        if c:
            figs.append({"key": key, "title": f"Training: validation passkey at N = {n:,}", "sub": "Validation streams, every 2,000 steps (N = 4,096: 100 streams, N = 16,384: 32 streams). Solid = value adaptation, dashed = the vector loop m at the same segment; bars from the worst to the best seed.",
                         "caption": "", "spec": c})
    figs += probe_figures(probes)
    figs.append(RR.configuration_appendix([g for g in dgroups if g["kind"] == "swa_va" or g in list(fb.values()) + [x for x in (swa64, gdn2) if x]]))

    # ---------------------------------------------------------------- text
    tldr = [
        "<b>Idea.</b> Instead of adding a loop vector m to the inputs of the next segment, the new model changes the <b>values</b> of the sliding-window store: per layer and segment, the layer's outputs on the previous segment give "
        f"keys a_j and values b_j (tanh of linear maps of the normalised outputs), and every token of the next segment gets v_t = W_v h_t + ΔW_v h_t with ΔW_v = (g / C) Σ_j b_j a_jᵀ, a matrix of rank at most C with entries bounded by |g| "
        f"(g: one learned scalar per layer, 0 at the start). Queries and keys are untouched, each value is computed once and stays in the cache for the {ref_mc['window'] - 1} tokens it lives; "
        f"nothing else crosses the segment boundary. {RG.setting(next(iter(va.values())), 'model.n_layers')} layers, window {ref_mc['window']}, {next(iter(va.values()))['runs'][0]['n_params']:,} parameters (the vector loop: 564k).",
    ]
    for seg in segs:
        key = f"va{seg}"
        h = hz(key)
        tldr.append(f"<b>Segment {seg}.</b> Value adaptation ({seeds_lbl(va[seg])}) scores passkey exact match {f2(cell(key, 256))} / {f2(cell(key, 1024))} / {f2(cell(key, 4096))} / {f2(cell(key, 16384))} (mean, test) at N = 256 / 1,024 / 4,096 / 16,384; "
                    f"per seed at N = 16,384 {RG.seeds_text(per(key, 16384))}; the mean stays at 0.9 or above up to " + (f"N = {h:,}" if h else "no N") + ". "
                    + (f"The vector loop m at the same segment ({seeds_lbl(fb[seg])}) scores {f2(cell(f'fb{seg}', 256))} / {f2(cell(f'fb{seg}', 1024))} / {f2(cell(f'fb{seg}', 4096))} / {f2(cell(f'fb{seg}', 16384))} (per seed at 16,384: {RG.seeds_text(per(f'fb{seg}', 16384))})." if seg in fb else ""))
    base = []
    if swa64:
        base.append(f"plain SWA (window 64) {f2(cell('w64', 256))} at N = 256 and {f2(cell('w64', 1024))} at N = 1,024")
    if gdn2:
        base.append(f"GDN-2 (mean of {seeds_lbl(gdn2)}) {f2(cell('gdn2', 1024))} / {f2(cell('gdn2', 4096))} / {f2(cell('gdn2', 16384))} at N = 1,024 / 4,096 / 16,384")
    if base:
        tldr.append("<b>Baselines, same recipe.</b> " + "; ".join(base) + ".")
    data = {
        "ns": NS, "rows": rows, "results": results, "scatter_ns": [], "hide_scatter": True, "label_w": 352, "row_name": "configuration", "row_prefix": "",
        "meta": f"Design experiment E1, results of {time.strftime('%d %b %Y, %H:%M')}. Test split, N counts AR writes, one write is about 3.75 tokens. Every row is one configuration; seeds are averaged only within a configuration.",
        "tldr": tldr, "tested": [], "results_title": "Results",
        "intro": ("<h2>Method</h2><p>The task: remember a 4-token passkey shown once at the start of a stream and repeat it at the end, while answering associative-recall queries about recent key–value writes. "
                  "A plain sliding-window model cannot do the first part beyond its window. Here the long-term memory is meant to live in the short-term store itself: the values of the cached window are adapted, "
                  "segment by segment, by an associative read of the layer's own previous outputs.</p>"
                  f"<figure><div class='fig-title'>One layer, two consecutive segments</div><div class='chart'>{method_svg()}</div>"
                  "<figcaption>The adaptation is a low-rank test-time update of W_v, ΔW_v = (g / C) Σ b_j a_jᵀ, built from the previous segment and never formed as a matrix (Δv_t = (g / C) Σ_j b_j ⟨a_j, h_t⟩). "
                  "The first segment is plain attention. The model runs on a key/value cache: a value computed for a token keeps the adaptation of its own segment for as long as it stays in the window.</figcaption></figure>"),
        "results_intro": "Value adaptation against the vector loop m at the same segment, plain SWA and GDN-2, all with the same recipe (50k steps, stream-length curriculum, 2 layers, window 64).",
        "fig1_title": "1. Accuracy by configuration and stream length", "fig1_sub": "Rows: value adaptation (mean of the seeds and best seed), the vector loop at the same segments, baselines. Columns: stream length N. Darker is higher.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)", "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": "<p>Top grid: passkey. Middle and bottom grids: the associative-recall half.</p>", "cap2": "", "extra_figs": figs, "next_title": "Open issues",
        "next": "<p><b>Seeds.</b> Both model families switch their memory on in some seeds only; the means above are the summary, not the best seed.</p>",
    }
    return number_figures(data, first=2), "SWA Value Adaptation Report", "Value adaptation of the sliding-window store: a memory in the values of the cache"
