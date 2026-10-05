"""Figures of the third review round for the feedback-loop report: the far-propagation case (M7), the 50k-step runs against plain SWA at
several windows (H5/H6), the segment study (M9), the validation curves, and the table of every configuration (H7).

Every builder takes the loaded runs / configuration groups and returns figure dicts (`{key, title, sub, caption, spec | html}`) and
data-derived text; nothing here is written by hand around a number. A builder returns nothing when its runs do not exist yet.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_groups as RG  # noqa: E402
from report_groups import NS, cells_for_row, horizon  # noqa: E402

TOKENS_PER_WRITE = 3.75
LOG_LINE = re.compile(r"\[(?P<name>[^\]]+)\] step (?P<step>\d+) loss (?P<loss>[\d.]+) \| (?P<rest>.*)")
LOG_BLOCK = re.compile(r"n(?P<n>\d+): pk (?P<pk>[\d.]+) v (?P<v>[\d.]+) e (?P<e>[\d.]+)")
f2 = lambda x: f"{x:.2f}"


def log_curves(logs_root: Path) -> dict:
    """{run name: {step: {"loss": x, N: {"pk", "v", "e"}}}} from the periodic-validation lines of every training log (newest log wins,
    and a restarted run's repeated steps overwrite the earlier ones)."""
    newest: dict[str, Path] = {}
    for f in sorted(Path(logs_root).glob("*/*.log"), key=lambda f: f.stat().st_mtime):
        newest[f.stem] = f
    out: dict = {}
    for name, f in newest.items():
        for line in f.read_text(errors="ignore").splitlines():
            m = LOG_LINE.search(line)
            if m and m["name"] == name:
                e = {"loss": float(m["loss"])}
                for b in LOG_BLOCK.finditer(m["rest"]):
                    e[int(b["n"])] = {"pk": float(b["pk"]), "v": float(b["v"]), "e": float(b["e"])}
                out.setdefault(name, {})[int(m["step"])] = e
    return out


def curve(run: dict, logs: dict, n: int, what: str = "pk") -> list[tuple[int, float]]:
    """(step, value) of a validation metric at stream length n: from the run's stored history if it has one, else from its log."""
    key = {"pk": "passkey_acc", "v": "ar_valid_acc", "e": "ar_expired_acc"}[what]
    if run.get("val_history"):
        return [(h["step"], h[f"val/n{n}/{key}"]) for h in run["val_history"] if f"val/n{n}/{key}" in h]
    return sorted((st, e[n][what]) for st, e in logs.get(run["name"], {}).items() if n in e)


def switch_on_step(run: dict, logs: dict, n: int = 256, thr: float = 0.5):
    """First periodic validation with passkey accuracy above `thr` at stream length n (None if there is none)."""
    return next((st for st, v in curve(run, logs, n) if v > thr), None)


def _table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    return (f"<div class='tbl {cls}'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>")


def _kv_table(rows: list[list[str]]) -> str:
    return "<div class='tbl kv'><table><tbody>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</tbody></table></div>"


def _row_of(run: dict, n: int, split: str = "test") -> dict:
    return next(x for x in run["rows"] if x["split"] == split and x["N"] == n)


# --------------------------------------------------------------------------------------------------------------- M7
def far_propagation(std: dict, ref_runs: list[dict], rec_gain: float, logs: dict, best_key=(8, "none", 0.0, 20000, 0.0)) -> dict | None:
    """The case that carries the passkey furthest (window 64, segment 8, no tanh, gain 0): its numbers next to the plain window and to the
    other configurations, and its validation curve. Returns {figs, tldr, best}."""
    if best_key not in std:
        return None
    G = std[best_key]
    runs = G["runs"]
    seg = best_key[0]
    top = max(runs, key=lambda r: _row_of(r, 16384)["passkey_acc"])
    hops = lambda n: round(n * TOKENS_PER_WRITE / seg)
    reach = RG.setting(G, "model.n_layers") * (RG.setting(G, "model.window") - 1)  # tokens the attention alone can see back
    t, v = _row_of(top, 16384), _row_of(top, 16384, "validation")
    ctrl = cells_for_row(ref_runs, "test") if ref_runs else None
    g0 = [g for k, g in std.items() if k[4] == 0 and k[2] == 0 and k[3] == 20000]  # the gain-0 grid
    both = lambda r: _row_of(r, 256)["ar_valid_acc"] > 0.9 and _row_of(r, 1024)["passkey_acc"] > 0.5
    n_g0 = sum(len(g["runs"]) for g in g0)
    n_both = sum(both(r) for g in g0 for r in g["runs"])
    seeds = sorted(runs, key=lambda r: r["seed"])
    facts = [
        ["configuration", html.escape(RG.label(G)) + f"; trained on N ≤ 256 (about {round(256 * TOKENS_PER_WRITE):,} tokens), {RG.setting(G, 'train.steps'):,} steps"],
        [f"best seed ({top['seed']}), N = 16,384", f"passkey {f2(t['passkey_acc'])} on test, {f2(v['passkey_acc'])} on validation; AR valid {f2(t['ar_valid_acc'])}, AR expired {f2(t['ar_expired_acc'])}; "
                                                   f"about {round(16384 * TOKENS_PER_WRITE):,} tokens and {hops(16384):,} loop hops, {16384 * TOKENS_PER_WRITE / reach:.0f} times the attention reach of {reach} tokens"],
        ["all seeds, N = 16,384", "passkey " + RG.seeds_text([_row_of(r, 16384)["passkey_acc"] for r in seeds]) + " (seeds " + ", ".join(str(r["seed"]) for r in seeds)
         + "); AR valid " + RG.seeds_text([_row_of(r, 16384)["ar_valid_acc"] for r in seeds])],
    ]
    if ctrl:
        first0 = next((n for n in NS if ctrl[str(n)]["passkey"] is not None and ctrl[str(n)]["passkey"] < 0.05), None)
        facts.append(["plain window, same curriculum and steps (no loop)", f"passkey {f2(ctrl['256']['passkey'])} at N = 256 and {f2(ctrl['16384']['passkey'])} at N = 16,384"
                      + (f", below 0.05 from N = {first0}" if first0 else "") + f"; AR valid {f2(ctrl['256']['valid'])}"])
    facts.append(["gain-0 runs that learned both halves", f"{n_both} of {n_g0} (AR valid above 0.9 at N = 256 and passkey above 0.5 at N = 1,024)"])
    seg_rows = []
    for nl in ("none", "tanh"):
        for g in (0.0, rec_gain):
            vals = [std[(s, nl, 0.0, 20000, g)] for s in (64, 32, 16, 8) if (s, nl, 0.0, 20000, g) in std]
            if len(vals) == 4:
                seg_rows.append([("no tanh" if nl == "none" else "tanh") + f", gain {g:g}"] + [f2(np.mean(RG.per_seed(x["runs"], 1024))) for x in vals])
    series_vals = [[float(x) for x in r[1:]] for r in seg_rows]
    n_up = sum(all(a < b for a, b in zip(v, v[1:])) for v in series_vals)
    n_down = sum(all(a > b for a, b in zip(v, v[1:])) for v in series_vals)
    fig1 = {"key": "far", "title": "Far propagation: the best configuration",
            "sub": f"Window 64, segment {seg}, no tanh, loop gain 0 at the start; test split unless stated.",
            "caption": (f"<p>Mean passkey at N = 1,024 for segments 64 / 32 / 16 / 8, by loop type and initial gain. {n_up} of the {len(seg_rows)} rows rise steadily as the segment shrinks from 64 to 8, "
                        f"{n_down} fall steadily and {len(seg_rows) - n_up - n_down} are not monotone.</p>" + _table(["loop, gain", "segment 64", "32", "16", "8"], seg_rows)) if seg_rows else "",
            "spec": None, "html": _kv_table(facts)}

    def pts(r, n, what):
        return [{"x": st, "y": val, "tip": f"seed {r['seed']}, step {st:,}<br>{what} at N = {n:,}: {val:.2f}"} for st, val in curve(r, logs, n, "v" if what == "AR valid" else "pk")]

    series = []
    for n, what, col, dashed in ((256, "passkey", "var(--s1)", False), (1024, "passkey", "var(--s2)", False), (256, "AR valid", "var(--muted)", True)):
        p = pts(top, n, what)
        if p:
            series.append({"label": f"{what}, N = {n:,}", "short": f"{what} N={n}", "points": p, "c": col, "width": 2.4, "dashed": dashed})
    fig2 = None
    if series:
        sw = switch_on_step(top, logs)
        fig2 = {"key": "far_curve", "title": f"Far propagation: how the best seed (seed {top['seed']}) learned it",
                "sub": "Validation accuracy (200 streams per N) at every periodic evaluation of the training run.",
                "caption": f"<p>The loop {'switched on at step ' + format(sw, ',') + ' (first evaluation with passkey above 0.5 at N = 256)' if sw else 'did not switch on at N = 256'}; "
                           f"the final training loss was {top['final_train_loss']:.3f}.</p>",
                "spec": {"series": series, "xTicks": sorted({s for sr in series for s in (p["x"] for p in sr["points"]) if s % 4000 == 0}), "x0": 0, "x1": max(p["x"] for sr in series for p in sr["points"]),
                         "xLog": False, "xLabel": "training step", "yLabel": "validation accuracy", "legendLine": True, "rightPad": 112,
                         "aria": "Validation accuracy against training step for the best run"}}
    tldr = (f"<b>One configuration carries the passkey to about {16384 * TOKENS_PER_WRITE / reach:.0f} times the attention reach, but not reliably.</b> Window 64, segment {seg}, no tanh, gain 0 (seed {top['seed']}): passkey {f2(t['passkey_acc'])} at N = 16,384 "
            f"(about {round(16384 * TOKENS_PER_WRITE / 1000)}k tokens, {hops(16384):,} loop hops) with AR valid {f2(t['ar_valid_acc'])} and AR expired {f2(t['ar_expired_acc'])}, on validation {f2(v['passkey_acc'])}; "
            f"the plain window with the same training is at {f2(ctrl['16384']['passkey']) if ctrl else '–'}. "
            f"It is not reliable: the three seeds score {RG.seeds_text([_row_of(r, 16384)['passkey_acc'] for r in seeds])} at N = 16,384, and only {n_both} of {n_g0} gain-0 runs learned both halves (figure {{fig:far}}).")
    return {"figs": [f for f in (fig1, fig2) if f], "tldr": tldr, "best": top["name"]}


# ------------------------------------------------------------------------------------------------------- H5 / H6 / M9
def _series_over_n(runs: list[dict], label: str, short: str, color: str | None, dashed: bool = False, split: str = "test") -> dict:
    cells = cells_for_row(runs, split)
    pts = []
    for n in NS:
        c = cells[str(n)]
        per = c["per"]["passkey"]
        pts.append({"x": n, "y": c["passkey"], "lo": min(per), "hi": max(per),
                    "tip": f"{html.escape(label)}<br>N = {n:,}<br>passkey {c['passkey']:.3f} (seeds {', '.join(f'{x:.2f}' for x in per)})"})
    sr = {"label": f"{label} ({len(runs)} seeds)" if len(runs) != 3 else label, "short": short, "points": pts, "width": 2.4, "dashed": dashed}
    if color:
        sr["c"] = color
    return sr


def long_runs(dgroups: list[dict], std: dict, logs: dict) -> dict | None:
    """H5 (50k-step loop runs) and H6 (plain SWA windows trained the same way) against the 20k-step runs of the same setup."""
    h5 = {"B": (32, "none"), "C": (64, "tanh")}
    loops = {name: std.get((seg, nl, 0.0, 50000, 0.0)) for name, (seg, nl) in h5.items()}
    swa = sorted((g for g in dgroups if g["kind"] == "swa" and RG.setting(g, "train.steps") == 50000 and RG.setting(g, "train.curriculum_stage_steps") == 2000
                  and RG.setting(g, "model.n_layers") == 2 and set(g["diff"]) <= {"model.window", "train.steps", "train.curriculum_stage_steps"}),
                 key=lambda g: RG.setting(g, "model.window"))
    loops = {k: g for k, g in loops.items() if g}
    gdn2 = next((g for g in dgroups if g["kind"] == "gdn2" and RG.setting(g, "train.steps") == 50000 and RG.setting(g, "train.curriculum_stage_steps") == 2000
                 and set(g["diff"]) <= {"train.steps", "train.curriculum_stage_steps"}), None)  # the recurrent baseline trained like the loop models
    if not loops and not swa:
        return None
    ctrl20 = next((g for g in dgroups if g["kind"] == "swa" and RG.setting(g, "model.window") == 64 and RG.setting(g, "train.steps") == 20000
                   and RG.setting(g, "train.curriculum_stage_steps") == 2000 and set(g["diff"]) <= {"train.steps", "train.curriculum_stage_steps"}), None)
    short20 = {name: std.get((seg, nl, 0.0, 20000, 0.0)) for name, (seg, nl) in h5.items()}
    rows, series = [], []
    colour = {"B": "var(--s2)", "C": "var(--s1)"}
    named = lambda g: RG.short_label(g) + ("" if "train.steps" in g["diff"] else f", {RG.setting(g, 'train.steps') // 1000}k steps")  # a changed step count is already in the label
    for name, (seg, nl) in h5.items():
        g = loops.get(name)
        if short20[name]:
            series.append(_series_over_n(short20[name]["runs"], named(short20[name]), f"{name} 20k", colour[name], True))
        if g:
            series.append(_series_over_n(g["runs"], named(g), f"{name} 50k", colour[name]))
    for i, g in enumerate(swa):
        series.append(_series_over_n(g["runs"], f"plain SWA, window {RG.setting(g, 'model.window')}", f"W{RG.setting(g, 'model.window')}", f"var(--seq{min(4, i)})"))
    if gdn2:
        series.append(_series_over_n(gdn2["runs"], "GDN-2 (learned decay), same training", "GDN-2", "var(--ink)", False))
    if ctrl20:
        series.append(_series_over_n(ctrl20["runs"], "plain SWA, window 64, 20k steps", "W64 20k", "var(--muted)", True))
    train_max = max(max(r["cfg"]["train"]["train_ns"]) for g in dgroups for r in g["runs"][:1] if g["kind"] in ("swa", "swa_fb"))  # longest training stream (N)
    for g in [*(short20[n] for n in h5 if short20[n]), *loops.values(), *([ctrl20] if ctrl20 else []), *swa, *([gdn2] if gdn2 else [])]:
        rs = g["runs"]
        cells = cells_for_row(rs, "test")
        hz = horizon(cells, "passkey", 0.9)
        t = [r["train_time_s"] for r in rs if r.get("train_time_s")]
        reach = RG.reach(g)
        rows.append([html.escape(RG.label(g)), f"{reach:,} tokens" if reach else ("recurrent state" if g["kind"] == "gdn2" else "loop"), f"{rs[0]['n_params'] // 1000}k", "–" if not t else f"{np.mean(t) / 3600:.1f} h",
                     RG.seeds_text(RG.per_seed(rs, 256)), RG.seeds_text(RG.per_seed(rs, 1024)), RG.seeds_text(RG.per_seed(rs, 4096)), RG.seeds_text(RG.per_seed(rs, 16384)),
                     "none" if hz is None else f"N = {hz:,}", f2(np.mean(RG.per_seed(rs, 256, "ar_valid_acc")))])
    table = _table(["configuration", "attention reach", "params", "train time", "passkey N = 256, per seed", "N = 1,024", "N = 4,096", "N = 16,384", "longest N with mean ≥ 0.9", "AR valid, N = 256"], rows, "wide")
    figs = [{"key": "long", "title": "50k-step runs against plain windows, passkey against stream length",
             "sub": "Test split, mean over seeds with the seed range as bars. Dashed = the 20k-step runs of the same setup. Grey ramp = plain sliding-window attention with the same curriculum, 50k steps and 2 layers, at windows 64 / 128 / 256 / 512 / 1,024.",
             "caption": "<p>A plain window of W tokens reaches 2(W−1) tokens back, so it can hold the passkey while the stream is shorter than that. The 50k-step runs change only the number of steps (and the cosine schedule that stretches with it) against the 20k-step gain-0 runs.</p>",
             "spec": {"series": series, "xTicks": NS, "x0": 32, "x1": 16384, "xLabel": "stream length N (AR writes, log scale)", "yLabel": "passkey exact match (test)", "legendLine": True, "rightPad": 112,
                      "vlines": [{"x": 256, "label": "longest training stream", "dy": 0}], "aria": "Passkey accuracy against stream length for 50k-step loop runs and plain windows"}},
            {"key": "long_table", "title": "50k-step runs against plain windows, table", "sub": "Attention reach L·(W−1) of a plain window; the loop models have window 64, reach 126 tokens, plus the loop. Training time is the mean over seeds, on a GPU shared with other jobs.",
             "caption": "", "spec": None, "html": table}]
    for n in (1024, 4096):
        cs = []
        for name, g in loops.items():
            for r in g["runs"]:
                p = [{"x": st, "y": v, "tip": f"{name} seed {r['seed']}, step {st:,}<br>passkey at N = {n:,}: {v:.2f}"} for st, v in curve(r, logs, n)]
                if p:
                    cs.append({"label": f"{name} seed {r['seed']}", "short": f"{name} s{r['seed']}", "points": p, "c": colour[name], "width": 1.8, "dashed": r["seed"] % 2 == 1})
        if cs:
            figs.append({"key": f"curves{n}", "title": f"Validation passkey at N = {n:,} during the 50k-step runs",
                         "sub": f"Per seed ({'solid / dashed alternate' if len(cs) > 2 else 'one line per run'}); 100 streams at N = 4,096 and 32 at N = 16,384, 200 below. B = segment 32 no tanh, C = segment 64 tanh.",
                         "caption": "", "spec": {"series": cs, "xTicks": list(range(0, 50001, 10000)), "x0": 0, "x1": 50000, "xLog": False, "xLabel": "training step", "yLabel": f"validation passkey, N = {n:,}",
                                                 "legendLine": True, "rightPad": 70, "aria": f"Validation passkey at N = {n} during training"}})
    mean_at = lambda g, n: float(np.mean(RG.per_seed(g["runs"], n)))
    parts = []
    for name, g in loops.items():
        prev = short20[name]
        parts.append(f"{RG.short_label(g).replace(', ', ', ', 1)} ({len(g['runs'])} seed{'s' if len(g['runs']) != 1 else ''}): passkey at N = 1,024 / 4,096 / 16,384 {RG.seeds_text(RG.per_seed(g['runs'], 1024))} | {RG.seeds_text(RG.per_seed(g['runs'], 4096))} | "
                     f"{RG.seeds_text(RG.per_seed(g['runs'], 16384))} per seed" + (f", means {mean_at(g, 1024):.2f} / {mean_at(g, 4096):.2f} / {mean_at(g, 16384):.2f} against {mean_at(prev, 1024):.2f} / {mean_at(prev, 4096):.2f} / {mean_at(prev, 16384):.2f} after 20k steps" if prev else ""))
    if gdn2:
        parts.append(f"GDN-2 (stock learned decay, same training, {gdn2['runs'][0]['n_params'] // 1000}k parameters, {len(gdn2['runs'])} seeds): passkey at N = 1,024 / 4,096 / 8,192 / 16,384 per seed "
                     f"{RG.seeds_text(RG.per_seed(gdn2['runs'], 1024))} | {RG.seeds_text(RG.per_seed(gdn2['runs'], 4096))} | {RG.seeds_text(RG.per_seed(gdn2['runs'], 8192))} | {RG.seeds_text(RG.per_seed(gdn2['runs'], 16384))}")
    swa_txt = ""
    if swa:
        wins = []
        for g in swa:
            reach, hz = RG.reach(g), horizon(cells_for_row(g["runs"], "test"), "passkey", 0.9)
            n_in = max([n for n in NS if n * TOKENS_PER_WRITE + 12 <= reach], default=None)
            wins.append((RG.setting(g, "model.window"), reach, n_in, hz))
        swa_txt = ("Plain SWA with the same training, window (reach in tokens): longest N with mean passkey ≥ 0.9 " + ", ".join(
            f"{w} ({r:,}): " + ("none" if hz is None else f"{hz:,}") for w, r, n_in, hz in wins) + ". ")
        out = [(w, r, n_in, hz) for w, r, n_in, hz in wins if n_in is not None and n_in > train_max]
        if out and all(hz is not None and hz <= train_max for _, _, _, hz in out):
            swa_txt += (" ".join(f"Window {w} reaches {r:,} tokens, enough for streams up to N = {n_in:,}, but holds the passkey only up to N = {hz:,}, the longest training stream." for w, r, n_in, hz in out)
                        + " So a plain window does not extrapolate in length even inside its reach, whereas "
                        + "; ".join(f"the loop with segment {RG.effective_segment(g)} and a {RG.setting(g, 'model.n_layers') * (RG.setting(g, 'model.window') - 1)}-token attention reach scores {mean_at(g, 1024):.2f} at N = 1,024 (about {round(1024 * TOKENS_PER_WRITE):,} tokens)" for g in loops.values())
                        + ". ")
    tldr = ("<b>50k steps instead of 20k (gain 0, same curriculum), and plain windows trained the same way.</b> " + "; ".join(parts) + ". " + swa_txt
            + " (figures {fig:long}, {fig:long_table}" + (", validation curves {fig:curves1024}" if any(f["key"] == "curves1024" for f in figs) else "") + ".)")
    # is the loop still improving at the end? validation passkey at N = 1,024, mean of the last 5 evaluations against the 5 before (10k steps each at eval_every 2000)
    deltas = []
    for name, g in loops.items():
        for r in g["runs"]:
            c = [v for _, v in curve(r, logs, 1024)]
            if len(c) >= 10:
                deltas.append((f"{name} seed {r['seed']}", float(np.mean(c[-5:]) - np.mean(c[-10:-5]))))
    rising = (f"At the end of the 50k-step runs the validation passkey at N = 1,024 (mean of the last 5 evaluations minus the 5 before) changed by "
              + ", ".join(f"{d:+.2f} ({n})" for n, d in deltas) + f"; it rose by more than 0.05 in {sum(d > 0.05 for _, d in deltas)} of {len(deltas)} runs, so training had not plateaued.") if deltas else ""
    return {"figs": figs, "tldr": tldr, "rising": rising}


def segment_study(std: dict, logs: dict, steps: int = 50000, gain: float = 0.0) -> dict | None:
    """M9: the segment as the only variable at the same recipe (window 64, `steps`, gain 0), per loop type that has runs at 2 or more segments."""
    figs = []
    for nl in ("none", "tanh"):
        gs = {s: std[(s, nl, 0.0, steps, gain)] for s in (64, 32, 16, 8) if (s, nl, 0.0, steps, gain) in std}
        if len(gs) < 2:
            continue
        rows = []
        for s, g in sorted(gs.items(), reverse=True):
            for r in g["runs"]:
                sw = switch_on_step(r, logs)
                rows.append([str(s), f"{round(1024 * TOKENS_PER_WRITE / s):,} / {round(16384 * TOKENS_PER_WRITE / s):,}", str(r["seed"]), "never" if sw is None else f"{sw:,}",
                             f2(_row_of(r, 1024)["passkey_acc"]), f2(_row_of(r, 16384)["passkey_acc"]), f2(_row_of(r, 256)["ar_valid_acc"])])
        series = []
        for n, col, dashed in ((1024, "var(--s1)", False), (16384, "var(--s2)", True)):
            pts = []
            for s, g in sorted(gs.items()):
                per = RG.per_seed(g["runs"], n)
                pts.append({"x": s, "y": float(np.mean(per)), "lo": min(per), "hi": max(per), "tip": f"segment {s}, N = {n:,}<br>passkey {np.mean(per):.3f} (seeds {', '.join(f'{x:.2f}' for x in per)})<br>about {round(n * TOKENS_PER_WRITE / s):,} loop hops"})
            series.append({"label": f"passkey, N = {n:,}", "short": f"N={n}", "points": pts, "c": col, "width": 2.4, "dashed": dashed})
        name = "no tanh" if nl == "none" else "tanh"
        figs.append({"key": f"segstudy_{nl}", "title": f"Segment study at {steps // 1000}k steps, {name} loop, gain 0",
                     "sub": "Window 64; the segment is the only variable. Mean over seeds with the seed range as bars.",
                     "caption": "<p>Table: loop hops at N = 1,024 / 16,384, the step at which the loop switched on (first periodic validation with passkey above 0.5 at N = 256), and the test results per seed.</p>"
                                + _table(["segment", "loop hops, N = 1,024 / 16,384", "seed", "switch-on step", "passkey N = 1,024", "N = 16,384", "AR valid N = 256"], rows),
                     "spec": {"series": series, "xTicks": sorted(gs), "x0": min(gs), "x1": max(gs), "xLabel": "segment length in tokens (log scale)", "yLabel": "passkey exact match (test)",
                              "legendLine": True, "rightPad": 92, "aria": f"Passkey against segment length, {name} loop, {steps} steps"}})
    return {"figs": figs} if figs else None


# --------------------------------------------------------------------------------------------------------------- H7
def configuration_appendix(dgroups: list[dict]) -> dict:
    """Every configuration of the report's model kinds, one row each, with the diff against the reference and the per-seed numbers."""
    order = sorted(dgroups, key=lambda g: (g["kind"] != "swa", RG.setting(g, "model.window"), RG.effective_segment(g) or 0, g["kind"], str(sorted(g["diff"].items(), key=str))))
    return {"key": "appendix", "title": "Every configuration and what differs from the reference recipe",
            "sub": f"{sum(len(g['runs']) for g in dgroups)} runs in {len(dgroups)} configurations: runs are grouped by the diff of their stored config against the frozen reference (swa_v1 / swa_fb_v1, "
                   "seed, logging and evaluation settings ignored), so runs that differ in any other setting are never averaged. Hover the first column for the run names.",
            "caption": "<p>Reference swa_fb_v1: window 64, 2 layers, tanh loop, segment = window, gain 0.1, 20k steps with the stream-length curriculum; reference swa_v1: window 64, 2 layers, 30k steps, no curriculum.</p>",
            "spec": None, "html": RG.configuration_table(order)}
