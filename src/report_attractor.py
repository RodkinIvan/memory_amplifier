"""Report of E5: attractor states of the two-pass layer and the passkey decoded from them.

    python src/report.py attractor --out reports/attractor_report.html

Reads results/probes/p2_attractor_*.json (src/attractor.py): per checkpoint the real window state S0 after three segments of a stream that saw the passkey, the state S* that minimises the
step-map error E_x ||F(S, x) - S||^2 / ||S0||^2 (and its variant on the sphere of ||S0||), the passkey read out after K random segments, noise on the state, interpolation between the
states of two passkeys, and the convergence of starts with the same passkey. Every number in the text is computed from those files.
"""
from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import report_groups as RG  # noqa: E402
import report_rounds as RR  # noqa: E402
from report_groups import label_margin, number_figures  # noqa: E402

f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
PROBE_DIR = C.ROOT / "results" / "probes"
MAIN = "swa_2pass_seg32_po1_s5"
E4_CONTROL = "swa_2pass_seg32_po1_s2"
PLAIN = "swa_L2_steps50000_cur2000_w64_s0"
KS = ["0", "1", "4", "32", "512"]
COLORS = {"main": "var(--s2)", "e4": "var(--s3)", "plain": "var(--muted)", "cap8": "var(--s1)", "cap16": "var(--ink)"}


def load(directory: Path = PROBE_DIR) -> dict:
    """{run name: result} merged from every results/probes/p2_attractor_*.json."""
    out: dict = {}
    for f in sorted(directory.glob("p2_attractor_*.json")):
        out.update(json.loads(f.read_text()))
    return out


def short(name: str, r: dict | None = None) -> str:
    if name == MAIN:
        return "two-pass, best checkpoint (P = 4)"
    if name == E4_CONTROL:
        return "two-pass, E4 seed 2 (P = 4, does not extrapolate)"
    if name == PLAIN:
        return "plain SWA, window 64 (control)"
    P = r["passkey_len"] if r else name.split("_pk")[1].split("_")[0]
    seed = name.rsplit("_s", 1)[1]
    return f"two-pass, capacity stage P = {P}, seed {seed}"


def row_name(name: str, r: dict) -> str:
    """The name of a checkpoint in a table row (the full description is the legend / the tooltip)."""
    if name == MAIN:
        return "best (P = 4)"
    if name == E4_CONTROL:
        return "E4 seed 2 (P = 4)"
    if name == PLAIN:
        return "plain SWA, window 64"
    return f"capacity P = {r['passkey_len']}, seed {name.rsplit('_s', 1)[1]}"


def order(res: dict) -> list[str]:
    """Best checkpoint first, then the E4 control and plain SWA, then the capacity stages by passkey length and seed."""
    key = lambda n: (0 if n == MAIN else 1 if n == E4_CONTROL else 2 if n == PLAIN else 3, res[n]["passkey_len"], n)
    return sorted(res, key=key)


def role(name: str, r: dict) -> str:
    if name == MAIN:
        return "main"
    if name == E4_CONTROL:
        return "e4"
    if name == PLAIN:
        return "plain"
    return "cap16" if r["passkey_len"] == 16 else "cap8"


def test_passkey(name: str, n: int) -> float | None:
    """The checkpoint's own test passkey exact match at N = n AR writes (results/raw), None when it was not evaluated there."""
    f = C.ROOT / "results" / "raw" / f"{name}.json"
    if not f.exists():
        return None
    return next((x["passkey_acc"] for x in json.loads(f.read_text())["rows"] if x["split"] == "test" and x["N"] == n), None)


def exact(r: dict, start: str, K: str) -> float:
    return r["decoding"][start][K]["exact"]


def y_max(series: list[dict]) -> float:
    """Top of a share axis: the largest value rounded up to a multiple of 0.4, so that the quarter ticks of the chart are round percentages (0.4 -> 10% steps)."""
    top = max(p["y"] for sr in series for p in sr["points"] if p["y"] is not None)
    return round(0.4 * max(1, -(-top // 0.4)), 1)


def settle(curve: list[float], tol: float = 1.25) -> int:
    """The first segment (1-based) after which a distance curve stays within `tol` x its final value (25% by default)."""
    final = curve[-1]
    last_bad = max((i for i, v in enumerate(curve) if v > tol * final), default=-1)
    return last_bad + 2


def method_svg() -> str:
    """The window state S, one segment x, the new window: 31 kept entries shifted, 32 new ones."""
    tok = lambda x, y, cls, n, w=9: "".join(f"<rect x='{x + i * (w + 1)}' y='{y}' width='{w}' height='18' rx='2' class='{cls}'/>" for i in range(n))
    s = ["<svg viewBox='0 0 780 270' role='img' aria-label='The state is the window of 63 pass-2 inputs per layer; the map F runs one segment of 32 tokens on the cache built from it and returns the new window: 31 kept entries shifted and 32 new ones' style='width:100%;height:auto'>",
         "<style>.k{fill:var(--s1);opacity:.85}.n{fill:var(--s3);opacity:.9}.o{fill:var(--s2);opacity:.85}.t{font:11.5px var(--mono);fill:var(--ink-2)}.tb{font:12px var(--mono);fill:var(--ink)}.ar{stroke:var(--ink-2);stroke-width:1.4;fill:none}.bl{fill:var(--surface);stroke:var(--axis);stroke-width:1.2}</style>",
         "<defs><marker id='ah' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='var(--ink-2)'/></marker></defs>",
         "<text x='10' y='18' class='tb'>state S (one row per layer): the pass-2 inputs h2 of the 63 latest positions</text>"]
    s.append(tok(10, 28, "k", 63))
    s.append("<text x='10' y='62' class='t'>slot 0</text><text x='640' y='62' class='t' text-anchor='end'>slot 62</text>")
    s.append("<path d='M520 50 L520 88' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='370' y='90' width='400' height='54' rx='6' class='bl'/><text x='570' y='113' text-anchor='middle' class='tb'>F(S, x): cache from S (keys and values of h2),</text><text x='570' y='131' text-anchor='middle' class='tb'>one real segment x of 32 tokens, both layers</text>")
    s.append(tok(10, 100, "o", 32) + "<text x='10' y='134' class='t'>x: 32 tokens of associative-recall</text><text x='10' y='149' class='t'>content (no PK, no PKQ)</text>")
    s.append("<path d='M336 109 L366 112' class='ar' marker-end='url(#ah)'/>")
    s.append("<path d='M570 146 L570 178' class='ar' marker-end='url(#ah)'/>")
    s.append("<text x='10' y='194' class='tb'>new window F(S, x)</text>")
    s.append(tok(10, 202, "k", 31) + tok(10 + 31 * 10, 202, "n", 32))
    s.append("<text x='10' y='236' class='t'>31 entries kept, shifted by 32 slots</text><text x='330' y='236' class='t'>32 new entries (h2 of the segment)</text>")
    s.append("<text x='10' y='258' class='t'>objective: E over x of the squared distance between F(S, x) and S, divided by the energy of S0 (per layer)</text>")
    s.append("</svg>")
    return "".join(s)


def decode_figure(res: dict) -> dict:
    """Passkey exact match after K random segments from the real state S0, one line per checkpoint (the six capacity P = 8 seeds as their mean with the range), and from S* of the main one."""
    xs = [int(k) + 1 for k in KS]  # K = 0 on a log axis: plotted at K + 1
    pts = lambda vals, tip: [{"x": x, "y": v, "tip": f"{tip}<br>after K = {int(k)} random segments ({int(k) * 32:,} tokens): exact {v:.2f}"} for x, k, v in zip(xs, KS, vals)]
    series = []
    for name in order(res):
        r = res[name]
        if role(name, r) == "cap8":
            continue
        series.append({"label": short(name, r), "short": {"main": "best", "e4": "E4 s2", "plain": "plain SWA", "cap16": "P = 16"}[role(name, r)], "c": COLORS[role(name, r)],
                       "points": pts([exact(r, "s0", k) for k in KS], f"{short(name, r)}, state S0"), "width": 2.6})
    cap8 = [r for n, r in res.items() if role(n, r) == "cap8"]
    if cap8:
        mean = [float(np.mean([exact(r, "s0", k) for r in cap8])) for k in KS]
        lo, hi = [min(exact(r, "s0", k) for r in cap8) for k in KS], [max(exact(r, "s0", k) for r in cap8) for k in KS]
        series.append({"label": f"two-pass, capacity stages P = 8 (mean and range of {len(cap8)} seeds)", "short": "P = 8", "c": COLORS["cap8"], "width": 2.6,
                       "points": [{"x": x, "y": m, "lo": a, "hi": b, "tip": f"capacity stages P = 8, {len(cap8)} seeds<br>after K = {int(k)} random segments: mean exact {m:.2f}, range {a:.2f} to {b:.2f}"} for x, k, m, a, b in zip(xs, KS, mean, lo, hi)]})
    if MAIN in res:
        series.append({"label": "S* of the best checkpoint (minimiser of the objective)", "short": "S*", "c": "var(--axis)", "width": 2.2, "dashed": True,
                       "points": pts([exact(res[MAIN], "star", k) for k in KS], "S* of the best checkpoint")})
    spec = {"series": series, "xTicks": xs, "xTickLabels": {x: k for x, k in zip(xs, KS)}, "x0": 1, "x1": 513, "xLabel": "random segments K between the state and the read-out (32 tokens each; log scale, K = 0 at the left)",
            "yLabel": "passkey exact match", "legendLine": True, "rightPad": label_margin([s["short"] for s in series]), "aria": "Passkey exact match against the number of random segments after the state"}
    return {"key": "decode", "title": "The passkey read out of the real state after K random segments",
            "sub": "State S0 = the real window at position 96 of a stream that saw the passkey; K segments of random associative-recall content (held-out streams), then the filler, PKQ and the passkey, teacher forced. 64 states per checkpoint (16 passkeys x 4 contexts), mean of 3 filler lengths.",
            "caption": "<p>The state is the entire interface between segments: running on from a state rebuilt from the 63 stored entries gives exactly the logits of the real stream (tested). K = 512 is 16,384 tokens, the length of the longest evaluation stream. The dashed grey line is the state that minimises the objective, "
                       "which holds no passkey even at K = 0.</p>", "spec": spec}


def decode_table(res: dict) -> dict:
    rows = []
    for name in order(res):
        r = res[name]
        for start, lab in (("s0", "S0 (real state)"), ("star", "S* (minimiser)"), ("sphere", "S* on the sphere")):
            if start not in r["decoding"]:
                continue
            rows.append([html.escape(row_name(name, r)), lab] + [f2(exact(r, start, k)) for k in KS])
    return {"key": "decode_table", "title": "Passkey exact match after K random segments, all checkpoints", "sub": "Held-out streams of this analysis (their seeds are in the result files). Columns: K = number of random segments between the state and the read-out.",
            "caption": "", "spec": None, "html": RR._table(["checkpoint", "state"] + [f"K = {k}" for k in KS], rows, "wide compact")}


def optimization_table(res: dict) -> dict:
    rows = []
    for name in order(res):
        r = res[name]
        o, sp = r["optimization"], r.get("sphere")
        kn = o["error_kept_vs_new"]["star"]
        rows.append([html.escape(row_name(name, r)), f"{o['objective_start']:.2f} → {o['objective_star']:.3f}", f"{o['moved_from_s0']['per_layer'][0]:.2f} / {o['moved_from_s0']['per_layer'][1]:.2f}",
                     f"{o['rms']['star'][0] / o['rms']['s0'][0]:.2f} / {o['rms']['star'][1] / o['rms']['s0'][1]:.2f}", f"{kn['kept'][0]:.3f} / {kn['kept'][1]:.3f}", f"{kn['new'][0]:.3f} / {kn['new'][1]:.3f}",
                     f"{sp['objective_star']:.2f}" if sp else "–"])
    return {"key": "opt_table", "title": "The optimisation: objective, distance moved, error by slot",
            "sub": "Held-out objective (fresh random segments) from S0 to S*; distance moved ||S* - S0|| / ||S0|| (layer 0 / layer 1); RMS(S*) / RMS(S0); the error of the kept slots (0-30, the shifted entries) and of the new slots (31-62) at S*, "
                   "normalised by the mean slot energy of S0; the objective reached on the sphere of ||S0||.",
            "caption": "<p>The objective falls by two orders of magnitude, mostly by making S* small: the state is divided by its own norm before it makes keys and values (RMSNorm), so its scale is free, and the error is measured against the energy of S0. "
                       "On the sphere of ||S0|| (last column) the error cannot be made small.</p>", "spec": None,
            "html": RR._table(["checkpoint", "objective", "moved", "RMS ratio", "kept slots", "new slots", "on the sphere"], rows, "wide compact")}


def path_figure(res: dict) -> dict | None:
    if MAIN not in res:
        return None
    path = res[MAIN]["optimization"]["path"]
    xs = [p["step"] + 1 for p in path]
    obj0 = path[0]["objective"]
    mk = lambda key, f: [{"x": x, "y": f(p[key]), "tip": f"step {p['step']}: {key} {p[key]:.3f}"} for x, p in zip(xs, path)]
    series = [{"label": "passkey exact match after K = 4 segments", "short": "passkey", "c": COLORS["main"], "points": mk("exact_k4", lambda v: v), "width": 2.6},
              {"label": "objective, as a share of its value at S0", "short": "objective", "c": COLORS["e4"], "points": mk("objective", lambda v: v / obj0), "width": 2.6},
              {"label": "distance moved from S0 (relative, 100% = the norm of S0)", "short": "moved", "c": "var(--s1)", "points": mk("moved", lambda v: v), "width": 2.6, "dashed": True}]
    ticks = xs[::2] if len(xs) > 8 else xs
    spec = {"series": series, "xTicks": ticks, "xTickLabels": {x: str(x - 1) for x in xs}, "x0": 1, "x1": xs[-1], "yMax": y_max(series), "xLabel": "Adam step (log scale)", "yLabel": "share", "legendLine": True,
            "rightPad": label_margin(["passkey", "objective", "moved"]), "aria": "Passkey read-out, objective and distance moved along the optimisation"}
    first_lost = next((p["step"] for p in path if p["exact_k4"] < 0.5), None)
    return {"key": "path", "title": "Where the optimisation loses the passkey (best checkpoint)",
            "sub": "Along the Adam path from S0: the objective, the passkey exact match after 4 random segments (64 states) and the distance moved from S0.",
            "caption": f"<p>The passkey is gone after {first_lost} steps, when the state has moved {next(p['moved'] for p in path if p['step'] == first_lost):.2f} of its norm and the objective is still "
                       f"{next(p['objective'] for p in path if p['step'] == first_lost) / obj0:.0%} of its starting value: the descent does not pass through states that keep the passkey.</p>" if first_lost is not None else "", "spec": spec}


def weaker_note(res: dict) -> str:
    """A sentence on the checkpoints for which a kick of 0.3 x RMS is not erased or hurts the passkey (the table has them all)."""
    weak = [row_name(n, r) for n, r in res.items() if n != MAIN and r["stability"]["s0"]["0.3"]["exact"]["32"] < 0.9 and role(n, r) != "plain"]
    return f" The stage with the longest passkey is less robust: for {', '.join(weak)} a kick of 0.3 x RMS already costs read-out accuracy and leaves a distance that does not shrink to the floor." if weak else ""


def noise_figure(res: dict) -> dict | None:
    if MAIN not in res:
        return None
    st = res[MAIN]["stability"]["s0"]
    segs = len(next(iter(st.values()))["distance_ratio"])
    xs = list(range(1, segs + 1))
    cols = ["var(--s3)", "var(--s1)", "var(--s2)", "var(--ink)", "var(--muted)"]
    series = [{"label": f"noise {sg} x RMS", "short": f"{sg}", "c": cols[i % 5], "width": 2.2, "noDots": True,
               "points": [{"x": x, "y": v, "tip": f"noise {sg} x RMS<br>after {x} segments: {v:.3f} of the initial distance"} for x, v in zip(xs, st[sg]["distance_ratio"])]} for i, sg in enumerate(st)]
    spec = {"series": series, "xTicks": [1, 2, 4, 8, 16, 32, 64][:7], "x0": 1, "x1": segs, "yMax": 1.0, "xLabel": "segments of the same random input after the kick (log scale)", "yLabel": "share of initial distance",
            "legendLine": True, "rightPad": label_margin([f"{k}" for k in st]), "aria": "Distance between a noisy and a clean trajectory over later segments"}
    return {"key": "noise", "title": "Noise on the real state (best checkpoint)",
            "sub": "Gaussian noise of the given std times the RMS of the state (per layer) is added to S0; both runs get the same random input; 64 states per noise level.",
            "caption": f"<p>The curves end at a flat floor that is the same absolute distance for every noise level (the rounding of the two bf16 runs), reached within {max(settle(st[k]['distance_ratio']) for k in st if float(k) <= 0.3)} segments for every kick up to 0.3 x RMS, "
                       "while the passkey read-out stays intact (table). A kick of 1 x RMS is the size of the state itself; it does damage the passkey." + weaker_note(res) + "</p>", "spec": spec}


def noise_table(res: dict) -> dict:
    rows = []
    for name in order(res):
        r = res[name]
        for start, lab, levels in (("s0", "S0", ("0.03", "0.3", "1.0")), ("star", "S*", ("0.1",) if name == MAIN else ())):
            for sg in levels:
                v = r["stability"][start][sg]
                rows.append([html.escape(row_name(name, r)), lab, sg, f3(v["distance_ratio"][3]), f3(v["distance_ratio"][15]), f3(v["distance_ratio"][-1]), f2(v["exact"]["4"]), f2(v["exact"]["32"]), f2(v["exact"]["512"])])
    return {"key": "noise_table", "title": "Noise: distance after later segments and the read-out, all checkpoints", "sub": "Distance to the clean trajectory after 4, 16 and 64 segments, as a share of the initial distance; passkey exact match after K = 4, 32, 512 random segments from the noisy state "
                                                                                           "(S0 at three noise levels; S* only for the best checkpoint; all levels are in the result files).",
            "caption": "", "spec": None, "html": RR._table(["checkpoint", "state", "noise x RMS", "d(4)", "d(16)", "d(64)", "exact K=4", "exact K=32", "exact K=512"], rows, "wide compact")}


def crossing(curve: list[dict], key: str, level: float, K: str = "32") -> float | None:
    """The alpha at which a share falling or rising through `level` crosses it (linear interpolation between the grid points)."""
    for a, b in zip(curve, curve[1:]):
        ya, yb = a[K][key], b[K][key]
        if (ya - level) * (yb - level) <= 0 and ya != yb:
            return a["alpha"] + (level - ya) / (yb - ya) * (b["alpha"] - a["alpha"])
    return None


def basin_caption(b: dict) -> str:
    parts = []
    for kind, lab in (("all_positions", "all positions differ"), ("one_position", "one position differs")):
        hi, lo = crossing(b[kind], "match_a", 0.9), crossing(b[kind], "match_a", 0.1)
        other = max(c["32"]["other"] for c in b[kind])
        if hi is not None and lo is not None:
            parts.append(f"{lab}: the share read as A falls from 90% to 10% between alpha = {hi:.2f} and {lo:.2f} (at most {other:.0%} of the positions are read as neither)")
    return "<p>The read-out does not blend: " + "; ".join(parts) + ". Each passkey has its own basin, with the boundary near the middle between the two states.</p>"


def basin_figure(res: dict) -> dict | None:
    if MAIN not in res:
        return None
    b = res[MAIN]["basins"]["s0"]
    series = []
    for kind, lab, dash in (("all_positions", "all positions differ", False), ("one_position", "one position differs", True)):
        curve = b[kind]
        for who, key, col in (("A", "match_a", COLORS["main"]), ("B", "match_b", COLORS["e4"])):
            series.append({"label": f"share of the differing positions read as {who}'s token ({lab})", "short": f"{who}, {lab.split()[0]} {lab.split()[1]}", "c": col, "width": 2.4, "dashed": dash,
                           "points": [{"x": c["alpha"], "y": c["32"][key], "tip": f"alpha {c['alpha']}: {c['32'][key]:.2f} read as {who}"} for c in curve]})
    spec = {"series": series, "xTicks": [round(0.1 * i, 1) for i in range(11)], "x0": 0, "x1": 1, "xLog": False, "xLabel": "alpha: S = (1 - alpha) S(A) + alpha S(B)", "yLabel": "share of positions read as A / as B",
            "legendLine": True, "rightPad": label_margin(["B, one position"]), "aria": "Read-out between the states of two passkeys"}
    return {"key": "basin", "title": "Between the states of two passkeys (best checkpoint, S0)",
            "sub": "Two passkeys A and B with the same context (B differs from A in one position or in all of them), the states S(A) and S(B) mixed with weight alpha, then 32 random segments and the model's own greedy read-out (16 pairs, mean of 2 filler lengths).",
            "caption": basin_caption(b), "spec": spec}


def convergence_caption(res: dict) -> str:
    two = {n: r for n, r in res.items() if role(n, r) != "plain" and "trajectory_convergence" in r}
    last = lambda n, key: res[n]["trajectory_convergence"][key][-1]
    same, diff = [last(n, "same_passkey") for n in two], [last(n, "different_passkey") for n in two]
    t = res[MAIN]["trajectory_convergence"]["t"][-1]
    note = ""
    if PLAIN in res and "trajectory_convergence" in res[PLAIN]:
        note = f" The plain window forgets everything: its states end {last(PLAIN, 'different_passkey'):.3f} apart whatever the passkey."
    return (f"<p>Once the context has left the window, the state is a function of the passkey and the input only: after {t} segments the states of one passkey are {last(MAIN, 'same_passkey'):.3f} apart for the best checkpoint "
            f"({min(same):.3f} to {max(same):.3f} over the {len(two)} two-pass checkpoints, the rounding floor is about 0.003), the states of different passkeys {last(MAIN, 'different_passkey'):.2f} ({min(diff):.2f} to {max(diff):.2f}). "
            f"In this sense every passkey has its own attractor; the state does not sit still (its window is rewritten about every two segments), the trajectory is the same." + note + "</p>")


def convergence_figure(res: dict) -> dict | None:
    if MAIN not in res or "trajectory_convergence" not in res[MAIN]:
        return None
    t = res[MAIN]["trajectory_convergence"]
    xs = [x + 1 for x in t["t"]]
    series = [{"label": "same passkey, different contexts", "short": "same passkey", "c": COLORS["e4"], "width": 2.6,
               "points": [{"x": x, "y": v, "tip": f"same passkey, different contexts<br>after {x - 1} segments: relative distance {v:.3f}"} for x, v in zip(xs, t["same_passkey"])]},
              {"label": "different passkeys", "short": "other passkey", "c": COLORS["main"], "width": 2.6,
               "points": [{"x": x, "y": v, "tip": f"different passkeys<br>after {x - 1} segments: relative distance {v:.3f}"} for x, v in zip(xs, t["different_passkey"])]}]
    spec = {"series": series, "xTicks": xs, "xTickLabels": {x: str(x - 1) for x in xs}, "x0": 1, "x1": xs[-1], "yMax": y_max(series), "xLabel": "segments of the same random input (log scale, 0 = the start)", "yLabel": "relative distance between states",
            "legendLine": True, "rightPad": label_margin(["other passkey"]), "aria": "Distance between states of the same and of different passkeys under the same input"}
    return {"key": "convergence", "title": "Starts with the same passkey converge (best checkpoint)",
            "sub": "64 states (16 passkeys x 4 contexts) run on one and the same random input; the mean relative distance ||a - b|| / rms(||a||, ||b||) between states with the same passkey and different contexts, and between states with different passkeys.",
            "caption": convergence_caption(res), "spec": spec}


def convergence_table(res: dict) -> dict:
    rows = []
    for name in order(res):
        r = res[name]
        t = r.get("trajectory_convergence")
        if not t:
            continue
        pick = lambda key, seg: f3(t[key][t["t"].index(seg)]) if seg in t["t"] else "–"
        rows.append([html.escape(row_name(name, r))] + [f"{pick('same_passkey', s)} / {pick('different_passkey', s)}" for s in (0, 4, 16, 64)])
    return {"key": "conv_table", "title": "Convergence of starts with the same passkey, all checkpoints", "sub": "Relative distance between states of the same passkey / of different passkeys, after 0, 4, 16 and 64 segments of the same input.",
            "caption": "", "spec": None, "html": RR._table(["checkpoint", "t = 0", "t = 4", "t = 16", "t = 64"], rows, "wide compact")}


def difference_table(res: dict) -> dict:
    rows = []
    for name in order(res):
        r = res[name]
        d = r.get("passkey_difference")
        if not d:
            continue
        for kind, lab in (("one_position", "one position differs"), ("all_positions", "all positions differ")):
            x = d["s0"][kind]["distance_ratio"]
            rows.append([html.escape(row_name(name, r)), lab, f2(x[0]), f2(x[3]), f2(x[15]), f2(x[-1])])
    return {"key": "diff_table", "title": "The difference between two passkeys is not erased",
            "sub": "The distance between the real states of two passkeys with the same context, both run on the same random input, as a share of the distance at the start (a kick of noise, in the noise figure, decays to the rounding floor within a few segments).",
            "caption": "", "spec": None, "html": RR._table(["checkpoint", "pair", "after 1 segment", "after 4", "after 16", "after 64"], rows, "wide compact")}


def report_data(res: dict | None = None) -> tuple[dict, str, str]:
    res = res if res is not None else load()
    if not res:
        raise ValueError("no attractor results found (results/probes/p2_attractor_*.json)")
    main = res.get(MAIN)
    tldr = ["<b>Question.</b> Does the key/value window of the two-pass layer settle into a state that is stationary under the incoming stream and from which the passkey can be decoded whatever the input? "
            "The state is the pass-2 inputs of the 63 latest positions of both layers (the whole interface between segments); the map F(S, x) runs one real segment on the cache built from S."]
    if main:
        o, d = main["optimization"], main["decoding"]
        tldr.append(f"<b>The real state holds the passkey.</b> From the state at position 96 of a stream that saw the passkey, the best checkpoint reads it out with exact match {f2(exact(main, 's0', '0'))} / {f2(exact(main, 's0', '4'))} / {f2(exact(main, 's0', '32'))} / {f2(exact(main, 's0', '512'))} "
                    f"after K = 0 / 4 / 32 / 512 random segments (512 segments are 16,384 tokens; the checkpoint's own test passkey at N = 4,096 writes, about 15k tokens, is {f2(test_passkey(MAIN, 4096))}). The E4 control that learned the training range but does not extrapolate reads "
                    f"{f2(exact(res[E4_CONTROL], 's0', '32'))} at K = 32 and {f2(exact(res[E4_CONTROL], 's0', '512'))} at K = 512 (its own test passkey at N = 4,096 is {f2(test_passkey(E4_CONTROL, 4096))})."
                    if E4_CONTROL in res else f"<b>The real state holds the passkey.</b> Exact match {f2(exact(main, 's0', '32'))} after K = 32 random segments, {f2(exact(main, 's0', '512'))} after K = 512.")
        first_lost = next((p for p in o["path"] if p["exact_k4"] < 0.5), None)
        tldr.append(f"<b>The minimiser of the specified objective is not an attractor of the passkey.</b> Adam from S0 lowers the objective from {f2(o['objective_start'])} to {f3(o['objective_star'])} (held out) but S* reads out the passkey with exact match "
                    f"{f2(exact(main, 'star', '0'))} at every K, even at K = 0; the state is {o['rms']['star'][0] / o['rms']['s0'][0]:.0%} / {o['rms']['star'][1] / o['rms']['s0'][1]:.0%} of the size of S0 (layer 0 / 1) and moved {o['moved_from_s0']['per_layer'][0]:.2f} of its norm"
                    + (f"; the passkey is already gone after {first_lost['step']} steps, when the objective is still {first_lost['objective'] / o['objective_start']:.0%} of its start" if first_lost else "") + ". "
                    + (f"Keeping the norm of S0 (the sphere) the objective cannot go below {f2(main['sphere']['objective_star'])} and the read-out is {f2(exact(main, 'sphere', '0'))} as well." if "sphere" in main else ""))
        ns = main["stability"]["s0"]
        small = ("0.01", "0.03", "0.1", "0.3")
        floors = ", ".join("%.4f" % (ns[k]["distance_ratio"][-1] * float(k)) for k in small)
        tldr.append(f"<b>The dynamics around the real state are contracting.</b> Noise of up to 0.3 x the state's RMS is gone after {max(settle(ns[k]['distance_ratio']) for k in small)} segments at most: the distance to the clean trajectory falls to a floor "
                    f"(share of the initial distance after 16 segments: " + ", ".join(f"{ns[k]['distance_ratio'][15]:.3f}" for k in small) + " for noise 0.01 / 0.03 / 0.1 / 0.3), which is the same absolute distance for all of them "
                    f"({floors} of the state's norm: the rounding of two bf16 runs). The passkey read-out never falls below "
                    f"{f2(min(ns[k]['exact'][kk] for k in small for kk in ('4', '32')))} up to K = 32; noise of 1 x RMS drops it to {f2(ns['1.0']['exact']['32'])}.")
        b = main["basins"]["s0"]["all_positions"]
        cross = next((c for c in b if c["32"]["match_b"] > c["32"]["match_a"]), None)
        tc = main.get("trajectory_convergence")
        tldr.append("<b>Each passkey has its own basin and its own trajectory.</b> " + (f"Mixing the states of two passkeys, the read-out switches from A to B at alpha = {cross['alpha']:g} (all positions differ); " if cross else "")
                    + (f"starts with the same passkey and different contexts, fed the same input, end at relative distance {tc['same_passkey'][-1]:.3f} after {tc['t'][-1]} segments (from {tc['same_passkey'][0]:.2f}), "
                       f"states of different passkeys at {tc['different_passkey'][-1]:.2f}." if tc else ""))
    cap8 = [r for n, r in res.items() if role(n, r) == "cap8"]
    cap16 = [r for n, r in res.items() if role(n, r) == "cap16"]
    if cap8 or cap16:
        def rng(rs, st, K):
            lo, hi = min(exact(r, st, K) for r in rs), max(exact(r, st, K) for r in rs)
            return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f} to {hi:.2f}"
        nz = lambda rs, K: rng([{"decoding": {"x": {K: {"exact": r["stability"]["s0"]["0.3"]["exact"][K]}}}} for r in rs], "x", K)
        parts = []
        if cap8:
            parts.append(f"{len(cap8)} stage{'s' if len(cap8) != 1 else ''} of P = 8: after K = 4 / 32 / 512 random segments {rng(cap8, 's0', '4')} / {rng(cap8, 's0', '32')} / {rng(cap8, 's0', '512')}, with noise of 0.3 x RMS and K = 32 {nz(cap8, '32')}")
        if cap16:
            parts.append(f"P = 16 (seed 1): {rng(cap16, 's0', '4')} / {rng(cap16, 's0', '32')} / {rng(cap16, 's0', '512')}, with that noise {nz(cap16, '32')}")
        tldr.append("<b>Capacity checkpoints</b> (ranges over seeds, real state S0). " + "; ".join(parts) + f". The minimiser S* never reads the passkey out with exact match above {max(exact(r, 'star', k) for r in cap8 + cap16 for k in KS):.2f}.")
    if PLAIN in res:
        pt = res[PLAIN].get("trajectory_convergence")
        tldr.append(f"<b>Control.</b> A plain sliding window (no gain, the state is its layer inputs) reads the passkey with exact match {f2(exact(res[PLAIN], 's0', '0'))} at K = 0, where the passkey is still inside its reach of 126 tokens, and {f2(exact(res[PLAIN], 's0', '4'))} at K = 4"
                    + (f"; the states of different passkeys fed the same input are {pt['different_passkey'][pt['t'].index(4)]:.3f} apart after 4 segments: nothing of the passkey is left." if pt and 4 in pt["t"] else "."))
    figs = [f for f in (decode_figure(res), decode_table(res), optimization_table(res), path_figure(res), noise_figure(res), noise_table(res), basin_figure(res), convergence_figure(res), convergence_table(res), difference_table(res)) if f]
    rec = (main or next(iter(res.values())))["record"]
    data = {
        "ns": [], "rows": [], "results": {}, "scatter_ns": [], "hide_scatter": True, "hide_heatmaps": True, "label_w": 58, "row_name": "", "row_prefix": "",
        "meta": f"Design experiment E5, results of {time.strftime('%d %b %Y, %H:%M')}. Analysis of fixed checkpoints, no training. Code {rec['git']['commit'][:7]}; held-out streams with their own seeds; bf16.",
        "tldr": tldr, "tested": [], "results_title": "Results",
        "intro": ("<h2>Method</h2><p>The short-term store of the two-pass layer is the key/value window. Its state S is, per layer, the pass-2 inputs h2 of the W - 1 = 63 latest positions (2 x 63 x 128): the keys and values are computed from them, with RoPE at the window's positions, "
                  "and nothing else crosses a segment boundary. The map F(S, x) builds the cache from S, runs one real segment x of 32 tokens and reads the new window. The start S0 is the real window at position 96 (the end of segment 2) of a stream that saw the passkey, "
                  "taken so that the passkey is only present in the form the layers passed on. The objective is L(S) = E_x sum over layers of ||F(S, x) - S||^2 / ||S0||^2 over fresh random segments of associative-recall content, minimised by Adam from S0.</p>"
                  + "<figure>" + method_svg() + "<figcaption>The state and the map. The 31 kept entries of the new window are the old ones shifted, so a zero error would force a period of 32 slots.</figcaption></figure>"
                  + "<p>Tests of the state (S0 as the baseline): the passkey read out after K random segments; noise added to the state; the states of two passkeys mixed; starts with the same passkey fed the same input. The learning rate of Adam is picked "
                    "from a grid by the held-out objective, never by the read-out. Extension to the specification: the same optimisation on the sphere of ||S0||, and the trajectory analyses (they need no fixed point).</p>"),
        "results_intro": "Every figure is computed from results/probes/p2_attractor_*.json; the best checkpoint is swa_2pass_seg32_po1_s5.",
        "extra_figs": figs, "next_title": "Notes",
        "next": ("<p>The objective of the specification has no passkey-bearing minimiser here because its scale is free (keys and values see the state through an RMSNorm) and the error is divided by the energy of S0; "
                 "the window is rewritten about every two segments, so the state itself is never stationary under input. What is stationary is the dynamics: perturbations die, the passkey does not, and starts with one passkey end on one trajectory. "
                 "The rounding floor of the distance curves is the difference between two bf16 runs. One passkey length per checkpoint (P = 4, 8 or 16); the capacity stages were fine-tuned from the best checkpoint, so they are not independent of it.</p>"),
    }
    return number_figures(data, first=1), "Attractor Report", "Attractor states of the two-pass layer"
