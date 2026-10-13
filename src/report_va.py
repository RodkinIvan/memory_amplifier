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
from report_best import mean_curve, over_n  # noqa: E402
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
         "<text x='16' y='16' class='t'>from the block outputs o_j of segment s−1:  a_j = tanh(W_y·norm(o_j)),  b_j = tanh(W_x·norm(o_j))</text>",
         "<text x='16' y='32' class='t'>ΔW_v = (g / C) Σ_j b_j a_jᵀ is added to W_v for segment s; g: one learned scalar per layer, 0 at the start</text>"]
    # left: segment s-1
    s.append("<text x='120' y='318' text-anchor='middle' class='tb'>segment s−1 (C tokens)</text>")
    s.append(tok(40, 270, "tk"))
    s.append("<path d='M120 266 L120 206' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='30' y='132' width='180' height='72' rx='4' class='bl'/><text x='120' y='160' text-anchor='middle' class='tb'>layer ℓ</text><text x='120' y='178' text-anchor='middle' class='t'>window attention + MLP</text>")
    s.append("<path d='M120 130 L120 98' class='ar' marker-end='url(#ah)'/>")
    s.append(tok(40, 76, "to"))
    s.append("<text x='120' y='64' text-anchor='middle' class='t'>outputs o_j</text>")
    # middle: the matrix
    s.append("<path d='M212 86 L282 112' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='286' y='84' width='172' height='66' rx='4' class='bl'/><text x='372' y='106' text-anchor='middle' class='tb'>ΔW_v</text>")
    s.append("<text x='372' y='124' text-anchor='middle' class='t'>(g/C) Σ b_j a_jᵀ</text><text x='372' y='140' text-anchor='middle' class='t'>rank ≤ C, |entry| ≤ |g|</text>")
    # right: segment s
    s.append("<text x='640' y='318' text-anchor='middle' class='tb'>segment s</text>")
    s.append(tok(560, 270, "tk"))
    s.append("<path d='M640 266 L640 206' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='530' y='132' width='220' height='72' rx='4' class='bl'/><text x='640' y='156' text-anchor='middle' class='tb'>layer ℓ</text><text x='640' y='174' text-anchor='middle' class='t'>v_t = W_v h_t + ΔW_v h_t</text><text x='640' y='192' text-anchor='middle' class='to2'>stored once in the cache</text>")
    s.append("<path d='M640 130 L640 98' class='ar' marker-end='url(#ah)'/>")
    s.append(tok(560, 76, "to"))
    s.append("<path d='M460 118 L526 152' class='fb' marker-end='url(#ao)'/>")
    s.append("<text x='474' y='214' class='to2'>queries and keys unchanged;</text><text x='474' y='228' class='to2'>no other path between segments</text>")
    s.append("</svg>")
    return "".join(s)


def load_probes(directory: Path = PROBE_DIR, prefix: str = "va") -> dict:
    """{'cache': ..., 'delta': ..., 'perturb': ...} from results/probes/<prefix>_cache_probe*.json, <prefix>_delta_w*.json, <prefix>_perturb*.json (whatever exists; several files of
    one kind, e.g. one per segment length, are merged by run name). `va` = value adaptation (E1), `p2` = two-pass (E2)."""
    out: dict = {}
    for key, stem in (("cache", f"{prefix}_cache_probe"), ("delta", f"{prefix}_delta_w"), ("perturb", f"{prefix}_perturb")):
        for f in sorted(directory.glob(f"{stem}*.json")):
            out.setdefault(key, {}).update(json.loads(f.read_text()))
    return out


def probe_figures(probes: dict, kinds: tuple = ("cache", "delta", "perturb")) -> list[dict]:
    """One figure per kind of diagnostic, all probed runs in it (empty when nothing was probed). `kinds` picks which; the perturbation figure is the value-adaptation one
    (cached values, previous encoding), the two-pass model has its own in report_2pass."""
    figs = []
    cache = probes.get("cache", {}) if "cache" in kinds else {}
    if cache:
        runs = list(cache)
        first = cache[runs[0]]
        dists = list(first["models"]["model"]["by_distance"])
        tokens = [first["models"]["model"]["by_distance"][d]["tokens"] for d in dists]
        ctl = next((res["models"]["control"] for res in cache.values() if "control" in res["models"]), None)
        head = ["run"] + [f"{t:,} tokens" for t in tokens]

        def table(feature: str) -> str:
            rows = [[html.escape(r)] + [f2(cache[r]["models"]["model"]["by_distance"][d][feature]["mean"]) if feature in cache[r]["models"]["model"]["by_distance"][d] else "–" for d in dists] for r in runs]
            if ctl:
                rows.append([f"control: {html.escape(ctl['run'])}"] + [f2(ctl["by_distance"][d][feature]["mean"]) if feature in ctl["by_distance"][d] else "–" for d in dists])
            return RR._table(head, rows)
        reach = first["models"]["model"]["reach"]
        figs.append({"key": "probe_cache", "title": "Passkey read out of the cached window",
                     "sub": f"Linear (ridge) read-out of each passkey token from the {reach // 2} cached positions of both layers, at growing numbers of tokens after the passkey; mean accuracy over the {first['passkey_len']} tokens, chance {first['chance']:.3f}; "
                            f"{first['n_train']} streams to fit, {first['n_test']} to test. Tables: cached values, cached keys, the adaptation matrix ΔW_v.",
                     "caption": "<p>The read-out never sees the passkey tokens themselves, only the cached values (or keys, or ΔW_v) after the given number of tokens. A plain window model with these dimensions reaches "
                                f"{reach} tokens" + (f"; the control ({html.escape(ctl['run'])}) is a plain sliding-window model, whose cache cannot hold the passkey beyond that." if ctl else ".") + "</p>"
                                , "spec": None, "html": "<p><b>Values</b></p>" + table("v") + "<p><b>Keys</b></p>" + table("k") + "<p><b>ΔW_v</b></p>" + table("dw")})
    delta = probes.get("delta", {}) if "delta" in kinds else {}
    if delta:
        rows = []
        for run, res in delta.items():
            for i, layer in enumerate(res["layers"]):
                e = layer["at"].get("0.5") or next(iter(layer["at"].values()))
                rows.append([html.escape(run), str(i), f"{layer['gain']:.3f}", f2(e["fro_over_w_v"]), f"{e['spectral']:.2f}", f"{e['rank']:.1f}", f"{e['max_entry']:.3f}",
                             f2(e["mean_share"]) if "mean_share" in e else "–", f2(e["cov_fro_over_w_v"]) if "cov_fro_over_w_v" in e else "–", f2(e["saturated"]) if "saturated" in e else "–", f2(e["cos_next"]), f2(e["cos_lag16"])])
        res0 = next(iter(delta.values()))
        figs.append({"key": "probe_delta", "title": "Size, rank and drift of ΔW_v along a stream",
                     "sub": f"{res0['n_writes']:,} writes per stream, taken in the middle of the stream, mean over the probe streams; rank = participation ratio of the singular values (at most C).",
                     "caption": "<p>Size is relative to the layer's own W_v. The mean part is the outer product of the average value vector and the average key vector of the segment (the same for every token); the token-specific part is the rest, where stream content can live. "
                                "Saturated: share of key and value entries with |tanh| above 0.95. A cosine near 1 between ΔW_v of consecutive segments (or 16 segments apart) means the adaptation has settled into a fixed point; near 0 that it is rewritten every segment.</p>",
                     "spec": None, "html": RR._table(["run", "layer", "gain g", "‖ΔW_v‖ ÷ ‖W_v‖", "spectral norm", "rank", "largest entry", "share of the mean part", "token-specific part ÷ ‖W_v‖", "saturated", "cos next", "cos +16"], rows)})
    pert = {r: res for r, res in probes.get("perturb", {}).items() if res["exact"]["clean"] >= 0.05 and "state_zero" in res["exact"]} if "perturb" in kinds else {}
    if pert:
        rows = []
        for run, res in pert.items():
            ex = res["exact"]
            eps = sorted({float(k.split("_")[1]) for k in ex if k.split("_")[0] in ("values", "state", "both") and k.split("_")[1] != "zero"})
            for who, key in (("cached values", "values"), ("previous encoding", "state"), ("both", "both")):
                rows.append([html.escape(run), who, f2(ex["clean"])] + [f2(ex[f"{key}_{e:g}"]) for e in eps] + [f2(ex[f"{key}_zero"])]
                            + ([" / ".join(f"{v:.2f}" for v in res["contraction"].values())] if key == "values" else [""]))
        res0 = next(iter(pert.values()))
        eps0 = sorted({float(k.split("_")[1]) for k in res0["exact"] if k.split("_")[0] == "values" and k.split("_")[1] != "zero"})
        figs.append({"key": "probe_perturb", "title": "Perturbing the memory in mid-stream",
                     "sub": f"Passkey exact match at N = {res0['n_writes']:,}; the kick lands after segment {res0['s_star']:,} (token {res0['s_star_tokens']:,}), far beyond the reach. Columns: noise of ε × the tensor's RMS, and zeroing. Only runs that hold the passkey (clean 0.05 or more).",
                     "caption": "<p>The last column: distance between the cached values of a clean and a perturbed run, relative to right after the kick (median over streams), after "
                                + " / ".join(res0["contraction"]) + " segments. Well below 1 means the memory pulls itself back to the clean trajectory; near 1 that the kick is carried along unchanged. "
                                "Runs whose passkey exact match is below 0.05 are not shown (nothing to perturb).</p>", "spec": None,
                     "html": RR._table(["run", "perturbed", "clean"] + [f"ε = {e:g}" for e in eps0] + ["zeroed", "kick size after later segments"], rows)})
    return figs


def probe_bullets(probes: dict) -> list[str]:
    """Summary sentences of the diagnostics (numbers from the probe files; one sentence per kind of probe, over the probed runs)."""
    out = []
    cache = probes.get("cache", {})
    if cache:
        parts = []
        for run, res in cache.items():
            m, ctl = res["models"]["model"], res["models"].get("control")
            far = [d for d, e in m["by_distance"].items() if e["tokens"] >= 2 * m["reach"]]
            if not far:
                continue
            d0, d1 = far[0], far[-1]
            at = lambda d: m["by_distance"][d]["v"]["mean"]
            tok = lambda d: f"{m['by_distance'][d]['tokens']:,}"
            both = d0 != d1
            txt = (f"{html.escape(run)}: {f2(at(d0))}" + (f" / {f2(at(d1))}" if both else "") + f" at {tok(d0)}" + (f" / {tok(d1)}" if both else "")
                   + f" tokens (token by token at {tok(d1)}: {' / '.join(f2(x) for x in m['by_distance'][d1]['v']['per_token'])})")
            if ctl:
                txt += f" (plain SWA: {f2(ctl['by_distance'][d0]['v']['mean'])}" + (f" / {f2(ctl['by_distance'][d1]['v']['mean'])}" if both else "") + ")"
            parts.append(txt)
        if parts:
            ex = next(iter(cache.values()))
            out.append(f"<b>Where the passkey is.</b> A linear read-out of each passkey token from the cached <i>values</i> of the last window (no passkey token in sight; mean accuracy over the {ex['passkey_len']} tokens, chance {ex['chance']:.2f}) reaches "
                       + "; ".join(parts) + f". The window itself is at most {ex['models']['model']['reach'] // 2} tokens, so what is read out was carried by the adaptation. Per-token accuracies are in the probe table.")
    delta = probes.get("delta", {})
    if delta:
        parts = []
        for run, res in delta.items():
            l0 = res["layers"][0]
            e = l0["at"].get("0.5") or next(iter(l0["at"].values()))
            parts.append(f"{html.escape(run)}: layer 0 has ‖ΔW_v‖ = {e['fro_over_w_v']:.1f} × ‖W_v‖, participation rank {e['rank']:.1f} (of at most {res['segment']})"
                         + (f", {e['saturated']:.0%} of the key and value entries saturated, {e['mean_share']:.0%} of its energy in the segment-mean part" if "saturated" in e else ""))
        out.append("<b>What the adaptation looks like.</b> " + "; ".join(parts) + ".")
    pert = probes.get("perturb", {})
    if pert:
        parts = []
        for run, res in pert.items():
            ex = res["exact"]
            if ex["clean"] < 0.05:
                continue  # no memory to perturb
            parts.append(f"{html.escape(run)} keeps {f2(ex['clean'])} clean, {f2(ex.get('values_1', float('nan')))} with its cached values kicked by 1 × RMS, {f2(ex.get('state_zero', float('nan')))} with the previous segment's encoding zeroed, {f2(ex.get('both_zero', float('nan')))} with both zeroed; "
                         f"a kick to the cached values shrinks to {res['contraction'].get('16', float('nan')):.2f} of its size after 16 segments")
        if parts:
            out.append("<b>Perturbation in mid-stream</b> (far beyond the reach): " + "; ".join(parts) + ".")
    return out


def probe_sentence8(g8: dict, probes: dict) -> str:
    """What the probes say about the segment-8 runs: the cached-value read-out far beyond the reach, per run, with the run's associative-recall accuracy."""
    cache = probes.get("cache", {})
    parts = []
    for r in g8["runs"]:
        res = cache.get(r["name"])
        if not res:
            continue
        m = res["models"]["model"]
        far = max(m["by_distance"], key=int)
        parts.append(f"seed {r['seed']} (AR valid {RG.per_seed([r], 256, 'ar_valid_acc')[0]:.2f}): {f2(m['by_distance'][far]['v']['mean'])}")
    return (f" Read-out of the passkey from the cached values {m['by_distance'][far]['tokens']:,} tokens after it was shown: " + "; ".join(parts) + f" (chance {res['chance']:.2f}).") if parts else ""


def next_text(va: dict, fb: dict, logs: dict, probes: dict) -> str:
    """The open issues, from the numbers: what the runs did at the end of training, what failed, what the diagnostics point to. Suggestions are marked as not run."""
    paras = []
    seen = lambda g, n, step: [dict(RR.curve(r, logs, n)).get(step) for r in g["runs"]]
    g32 = va.get(32)
    if g32 and all(len(r.get("val_history") or []) > 0 for r in g32["runs"]):
        last = max(h["step"] for h in g32["runs"][0]["val_history"])
        early = last - 12000
        a, b = seen(g32, 256, early), seen(g32, 256, last)
        if None not in a + b:
            paras.append(f"<p><b>Training length.</b> At segment 32 the validation passkey at N = 256 was {RG.seeds_text(a)} per seed at step {early:,} and {RG.seeds_text(b)} at step {last:,}, "
                         f"with the learning rate already cosine-decayed to {g32['runs'][0]['cfg']['train']['min_lr_ratio']:.0%} at the end: the runs were still improving when the schedule ended, so a longer schedule is the first thing to try (not run).</p>")
    g8 = va.get(8)
    if g8:
        valid = RG.per_seed(g8["runs"], 256, "ar_valid_acc")
        pk = RG.per_seed(g8["runs"], 256)
        pk32 = RG.per_seed(g8["runs"], 32)
        paras.append(f"<p><b>Segment 8.</b> Value adaptation does not work here: passkey at N = 256 per seed {RG.seeds_text(pk)} (at N = 32: {RG.seeds_text(pk32)}), AR valid accuracy {RG.seeds_text(valid)}. "
                     + (f"The vector loop m at segment 8, the best configuration of the earlier rounds, scores {RG.seeds_text(RG.per_seed(fb[8]['runs'], 16384))} per seed at N = 16,384. " if 8 in fb else "")
                     + f"A seed that never learned the associative-recall half (valid {min(valid):.2f}) has the only passkey signal, which decays with N."
                     + probe_sentence8(g8, probes) + f" Whether the narrower channel (rank at most {RG.effective_segment(g8)} per segment, against 32) is the cause is not tested.</p>")
    delta = probes.get("delta", {})
    sat = [e["saturated"] for res in delta.values() for l in res["layers"] for e in l["at"].values() if "saturated" in e]
    big = [(res["layers"][0]["at"].get("0.5") or next(iter(res["layers"][0]["at"].values()))) for res in delta.values()]
    big = [e for e in big if e["fro_over_w_v"] >= 1.0]
    if sat:
        paras.append(f"<p><b>Saturated codes.</b> In the probed runs {np.mean(sat):.0%} of the key and value entries are saturated (|tanh| above 0.95)"
                     + (f"; in the {len(big)} runs where the layer-0 update is at least as large as W_v ({min(e['fro_over_w_v'] for e in big):.1f}–{max(e['fro_over_w_v'] for e in big):.1f} × ‖W_v‖) it is mostly one or two directions "
                        f"(participation rank {min(e['rank'] for e in big):.1f}–{max(e['rank'] for e in big):.1f})" if big else "")
                     + ". The adaptation behaves like a binary hash with a large constant part rather than a graded associative memory. "
                       "Keeping the codes unsaturated (a scale or a norm before the tanh) or removing the segment mean are the natural variants (not run).</p>")
    cache = probes.get("cache", {})
    per_tok = [np.array(m["by_distance"][max(m["by_distance"], key=int)]["v"]["per_token"]) for res in cache.values() for m in [res["models"]["model"]] if "seg32" in res["models"]["model"]["run"]]
    if per_tok:
        mean_tok = np.mean(per_tok, axis=0)
        paras.append(f"<p><b>Which passkey tokens are kept.</b> At the longest cut the read-out of the four passkey tokens from the cached values is {' / '.join(f2(x) for x in mean_tok)} (mean over the probed segment-32 runs): "
                     "an alternating pattern, as the loop model showed a period-3 pattern at 8 passkey tokens. Why the stored positions alternate is not explained here.</p>")
    paras.append("<p><b>Seeds.</b> Both families switch their memory on in some seeds only; the means above are the summary, not the best seed.</p>")
    return "".join(paras)


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
        add(f"va{seg}", f"value adaptation, segment {seg}, mean of {seeds_lbl(g)}", g["runs"], COLORS[f"va{seg}"], f"{RG.label(g)} ({seeds_lbl(g)}: {', '.join(r['name'] for r in g['runs'])})")
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
                    f"per seed at N = 16,384 {RG.seeds_text(per(key, 16384))}; " + (f"the mean stays at 0.9 or above up to N = {h:,}" if h else "the mean never reaches 0.9") + ". "
                    + (f"The vector loop m at the same segment ({seeds_lbl(fb[seg])}) scores {f2(cell(f'fb{seg}', 256))} / {f2(cell(f'fb{seg}', 1024))} / {f2(cell(f'fb{seg}', 4096))} / {f2(cell(f'fb{seg}', 16384))} (per seed at 16,384: {RG.seeds_text(per(f'fb{seg}', 16384))})." if seg in fb else ""))
    base = []
    if swa64:
        base.append(f"plain SWA (window 64) {f2(cell('w64', 256))} at N = 256 and {f2(cell('w64', 1024))} at N = 1,024")
    if gdn2:
        base.append(f"GDN-2 (mean of {seeds_lbl(gdn2)}) {f2(cell('gdn2', 1024))} / {f2(cell('gdn2', 4096))} / {f2(cell('gdn2', 16384))} at N = 1,024 / 4,096 / 16,384")
    if base:
        tldr.append("<b>Baselines, same recipe.</b> " + "; ".join(base) + ".")
    tldr += probe_bullets(probes)
    done = {r["name"] for r in raw}
    running = [f"{nm} (step {max(st):,})" for nm, st in sorted(logs.items()) if nm.startswith("swa_va_") and nm not in done]
    data = {
        "ns": NS, "rows": rows, "results": results, "scatter_ns": [], "hide_scatter": True, "label_w": 352, "row_name": "configuration", "row_prefix": "",
        "meta": f"Design experiment E1, results of {time.strftime('%d %b %Y, %H:%M')}. Test split, N counts AR writes, one write is about 3.75 tokens. Every row is one configuration; seeds are averaged only within a configuration."
                + (f" Still training (not in the tables): {', '.join(running)}." if running else ""),
        "tldr": tldr, "tested": [], "results_title": "Results",
        "intro": ("<h2>Method</h2><p>The task: remember a 4-token passkey shown once at the start of a stream and repeat it at the end, while answering associative-recall queries about recent key–value writes. "
                  "A plain sliding-window model cannot do the first part beyond its window. Here the long-term memory is meant to live in the short-term store itself: the values of the cached window are adapted, "
                  "segment by segment, by an associative read of the layer's own previous outputs.</p>"
                  f"<figure><div class='fig-title'>One layer, two consecutive segments</div><div class='chart'>{method_svg()}</div>"
                  "<figcaption>The adaptation is a low-rank test-time update of W_v, ΔW_v = (g / C) Σ b_j a_jᵀ, built from the previous segment and never formed as a matrix (Δv_t = (g / C) Σ_j b_j ⟨a_j, h_t⟩). "
                  "The first segment is plain attention. The model runs on a key/value cache: a value computed for a token keeps the adaptation of its own segment for as long as it stays in the window.</figcaption></figure>"),
        "results_intro": "Value adaptation against the vector loop m at the same segment, plain SWA and GDN-2, all with the same recipe (50k steps, stream-length curriculum, 2 layers, window 64).",
        "fig1_title": "1. Accuracy by configuration and stream length", "fig1_sub": "Rows: value adaptation (mean of the seeds), the vector loop at the same segments, baselines. Columns: stream length N. Darker is higher.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)", "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": "<p>Top grid: passkey. Middle and bottom grids: the associative-recall half.</p>", "cap2": "", "extra_figs": figs, "next_title": "Open issues",
        "next": next_text(va, fb, logs, probes),
    }
    return number_figures(data, first=2), "SWA Value Adaptation Report", "Value adaptation of the sliding-window store: a memory in the values of the cache"
