"""Report of the passkey-capacity experiment: how many passkey tokens fit in the memory a model builds inside the short-term store.

    python src/report.py capacity --out reports/capacity_report.html

A checkpoint trained on streams of up to 256 AR writes and a passkey of 4 tokens (16 bits) is fine-tuned on the same streams with a passkey of P = 8, 16, 32, ... tokens (4 bits each); each
stage starts from the weights of the one before and the chain stops at the first stage whose validation exact match is below 0.9 at N = 256. Nothing longer than N = 256 is evaluated.
The page reads results/capacity.json (one chain per recipe, model and seed) and the stages' result files; every number in the text is computed from them.
"""
from __future__ import annotations

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
import report_rounds as RR  # noqa: E402
from report_groups import cells_for_row, label_margin, number_figures  # noqa: E402

f2 = lambda x: f"{x:.2f}"
NS = [32, 64, 128, 256]
THRESHOLD, POS_OK = 0.9, 0.9
MODEL_LABEL = {"swa_fb": "vector loop (segment 8)", "swa_2pass": "two-pass layer (segment 32)"}
MODEL_SHORT = {"swa_fb": "vector loop", "swa_2pass": "two-pass"}
CHART_SHORT = {"swa_fb": "m-loop", "swa_2pass": "two-pass"}  # end-of-line labels of the chart
COLORS = ["var(--s2)", "var(--s3)", "var(--s1)", "var(--ink)", "var(--muted)"]


def recipe_label(steps: int, lr: float, curriculum: int) -> str:
    return f"{steps // 1000}k steps, lr {lr:g}" + (", curriculum" if curriculum else "")


def chain_short(c: dict) -> str:
    """Short name of a chain for the end of its line in the chart: 'two-pass 50k s1' (model, steps, seed). The full description is in the legend and the tooltips."""
    return f"{CHART_SHORT.get(c['model'], c['model'])} {c['steps'] // 1000}k s{c['seed']}"


def chains(capacity: dict, runs: dict[str, dict]) -> list[dict]:
    """[{label, model, recipe, seed, stages: [entries sorted by P], base: (run name, validation exact at N = 256) of the checkpoint the chain started from, rerun_of: the chain with fewer steps whose stages
    this one repeats with a longer budget, or None}]. Entries of every `stages*` and `budget_runs*` list are grouped by the model, the recipe and the seed of their run, so a chain is one recipe of one model from one seed."""
    groups: dict = {}
    for key in capacity:
        if not key.startswith(("stages", "budget_runs")) or not isinstance(capacity[key], list):
            continue
        for e in capacity[key]:
            r = runs.get(e["run"])
            if r is None:
                continue
            cfg = r["cfg"]
            rec = e["recipe"]
            gk = (cfg["model"]["kind"], int(e["steps"]), float(rec["lr"]), int(rec.get("curriculum_stage_steps") or 0), int(cfg["seed"]))
            groups.setdefault(gk, []).append(e)
    out = []
    for (kind, steps, lr, cur, seed), entries in groups.items():
        entries.sort(key=lambda e: e["passkey_len"])
        first = entries[0]
        init = Path(first["init"]).stem if str(first["init"]).startswith("checkpoints/") else None
        base = runs.get(init) if init else None
        base_exact = next((x["passkey_acc"] for x in base["rows"] if x["split"] == "validation" and x["N"] == 256), None) if base else None
        out.append({"model": kind, "steps": steps, "lr": lr, "curriculum": cur, "seed": seed, "stages": entries, "base": (init, base_exact),
                    "label": f"{MODEL_LABEL.get(kind, kind)}, {recipe_label(steps, lr, cur)}, seed {seed}"})
    out.sort(key=lambda c: (c["model"] != "swa_2pass", -c["steps"], c["seed"]))
    for c in out:   # a chain of the same model, seed and recipe but fewer steps that has a passkey length in common: this one is a longer-budget re-run of its stages, not an independent chain
        shorter = [d for d in out if (d["model"], d["seed"], d["lr"], d["curriculum"]) == (c["model"], c["seed"], c["lr"], c["curriculum"]) and d["steps"] < c["steps"]
                   and {e["passkey_len"] for e in d["stages"]} & {e["passkey_len"] for e in c["stages"]}]
        c["rerun_of"] = max(shorter, key=lambda d: d["steps"]) if shorter else None
    return out


def in_progress(logs_root: Path, done: set[str]) -> list[dict]:
    """Stages that are training now: the newest log of every capacity run without a result file, with the last periodic validation (step, passkey positions at 0.9 or more at N = 256, exact match)."""
    out = []
    for f in sorted(Path(logs_root).glob("capacity/*_pk*_steps*.log")):
        name = f.stem
        if name in done:
            continue
        last = None
        for line in f.read_text(errors="ignore").splitlines():
            m = re.search(r"\] step (\d+) loss [\d.]+ \|.*n256: pk ([\d.]+).*passkey positions >= 0.9 at N = 256: (\d+)/(\d+)", line)
            if m:
                last = (int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)))
        if last:
            m = re.search(r"_steps(\d+)_.*_s(\d+)$", name)
            out.append({"run": name, "step": last[0], "exact256": last[1], "positions_ok": last[2], "P": last[3], "steps": int(m.group(1)) if m else None, "seed": int(m.group(2)) if m else None})
    return out


def seed_rates(ch: list[dict]) -> list[dict]:
    """For every recipe of every model that has more than one seed: per passkey length P, how many seeds reached the stage and how many passed it (validation exact >= THRESHOLD at N = 256), with the
    median number of stored positions (>= POS_OK at N = 256) among the seeds that reached it. [{label, rows: [(P, reached, passed, median stored positions, median exact)]}]."""
    groups: dict = {}
    for c in ch:
        if not c["rerun_of"]:
            groups.setdefault((c["model"], c["steps"], c["lr"], c["curriculum"]), []).append(c)
    out = []
    for (model, steps, lr, cur), cs in groups.items():
        if len(cs) < 2:
            continue
        rows = []
        for P in sorted({e["passkey_len"] for c in cs for e in c["stages"]}):
            es = [e for c in cs for e in c["stages"] if e["passkey_len"] == P]
            rows.append((P, len(es), sum(e["validation"]["exact"]["256"] >= THRESHOLD for e in es), float(np.median([e["validation"]["positions_ok"].get("256", 0) for e in es])),
                         float(np.median([e["validation"]["exact"]["256"] for e in es]))))
        out.append({"label": f"{MODEL_LABEL.get(model, model)}, {recipe_label(steps, lr, cur)}", "short": f"{CHART_SHORT.get(model, model)}, {steps // 1000}k steps", "seeds": sorted(c["seed"] for c in cs), "rows": rows, "chains": sorted(cs, key=lambda c: c["seed"])})
    return out


def stop_stage(c: dict) -> dict | None:
    """The first stage of a chain below the threshold (the one where it stops), or None."""
    return next((e for e in c["stages"] if e["validation"]["exact"]["256"] < THRESHOLD), None)


def rerun_pairs(ch: list[dict]) -> list[dict]:
    """For every longer-budget re-run: per passkey length it shares with the shorter chain, the two stages side by side. [{seed, model, P, short, long, same_start}] with the stage entries."""
    out = []
    for c in ch:
        d = c["rerun_of"]
        if not d:
            continue
        for e in c["stages"]:
            o = next((x for x in d["stages"] if x["passkey_len"] == e["passkey_len"]), None)
            if o:
                out.append({"model": c["model"], "seed": c["seed"], "P": e["passkey_len"], "short": o, "long": e, "short_steps": d["steps"], "long_steps": c["steps"], "same_start": o["init"] == e["init"]})
    return out


def failing_positions(ch: list[dict]) -> dict[int, dict[int, int]]:
    """For every passkey length: {position: in how many of the independent chains' failing stages (the stage where the chain stops) that position is below POS_OK}; only stages of models with a count."""
    out: dict = {}
    for c in ch:
        e = stop_stage(c)
        pos = e["validation"]["positions"].get("256") if e else None
        if c["rerun_of"] or not pos:
            continue
        for j, x in enumerate(pos):
            if x < POS_OK:
                out.setdefault(e["passkey_len"], {}).setdefault(j + 1, 0)
                out[e["passkey_len"]][j + 1] += 1
    return out


def position_grid(ch: list[dict]) -> str:
    """Which passkey positions are stored at the stages with P >= 16 or with a position below POS_OK: one row per stage, one coloured cell per position (validation, N = 256), darker is higher."""
    rows = []
    for c in ch:
        for e in c["stages"]:
            pos = e["validation"]["positions"].get("256")
            if pos and (e["passkey_len"] >= 16 or any(x < POS_OK for x in pos)):
                rows.append((e["passkey_len"], c, e, pos))
    if not rows:
        return ""
    rows.sort(key=lambda t: t[0])
    width = max(len(p) for _, _, _, p in rows)
    head = "<tr><th></th>" + "".join(f"<th class='pc'>{j + 1 if (j + 1) % 4 == 0 or j == 0 else ''}</th>" for j in range(width)) + "</tr>"
    body = ""
    for P, c, e, pos in rows:
        cells = "".join(f"<td class='pc c' style='background:color-mix(in srgb, var(--s1) {round(100 * x)}%, var(--surface));' title='position {j + 1} of {P}: {x:.2f}'></td>" for j, x in enumerate(pos))
        body += f"<tr><td title='{html.escape(c['label'], quote=True)}'>{html.escape(chain_short(c))}, P = {P}</td>{cells}</tr>"
    return f"<div class='tbl'><table class='posgrid'><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def capacity_data(capacity: dict | None = None, runs: dict[str, dict] | None = None, logs_root: Path | None = None) -> tuple[dict, str, str]:
    capacity = capacity if capacity is not None else json.loads((C.ROOT / "results" / "capacity.json").read_text())
    runs = runs if runs is not None else {r["name"]: r for r in A.load_runs(C.ROOT / "results" / "raw", all_passkeys=True)}
    logs_root = logs_root if logs_root is not None else C.ROOT / "logs"
    ch = chains(capacity, runs)
    if not ch:
        raise ValueError("no capacity stages found")
    done = {e["run"] for c in ch for e in c["stages"]}
    prog = in_progress(logs_root, done | set(runs))

    # ---------------------------------------------------------------- heatmap rows: one per stage, columns = N
    rows, results = [], {}
    for i, c in enumerate(ch):
        for e in c["stages"]:
            key = f"{c['model']}_{c['steps']}_{c['seed']}_{e['passkey_len']}"
            rows.append({"key": key, "label": f"{MODEL_SHORT.get(c['model'], c['model'])}, {recipe_label(c['steps'], c['lr'], c['curriculum'])}, seed {c['seed']}: P = {e['passkey_len']}", "color": COLORS[i % len(COLORS)],
                         "long": f"{e['run']} ({e['bits']} bits)"})
            results[key] = cells_for_row([runs[e["run"]]], "validation", ns=NS)
    # ---------------------------------------------------------------- the chart: validation exact match at N = 256 against P
    series = []
    for i, c in enumerate(ch):
        if c["rerun_of"]:   # a longer-budget re-run of a stage is one point on top of another line: it is in the grid and the tables, not in the chart
            continue
        pts = []
        if c["base"][1] is not None:
            pts.append({"x": 4, "y": c["base"][1], "tip": f"{html.escape(c['label'])}<br>start checkpoint (P = 4, 16 bits)<br>validation exact {c['base'][1]:.3f}"})
        for e in c["stages"]:
            v = e["validation"]
            pts.append({"x": e["passkey_len"], "y": v["exact"]["256"], "tip": f"{html.escape(c['label'])}<br>P = {e['passkey_len']} ({e['bits']} bits)<br>validation exact {v['exact']['256']:.3f}, "
                                                                                 f"token accuracy {v['token']['256']:.3f}, positions at 0.9+: {v['positions_ok'].get('256', '?')} of {e['passkey_len']}"})
        series.append({"label": c["label"], "short": chain_short(c), "points": pts, "c": COLORS[i % len(COLORS)], "width": 2.6, "dashed": c["steps"] < 20000})
    ps = sorted({4} | {e["passkey_len"] for c in ch if not c["rerun_of"] for e in c["stages"]})
    series.append({"label": "threshold 0.9 (the chain stops below it)", "short": "0.9", "points": [{"x": p, "y": THRESHOLD, "tip": "threshold 0.9"} for p in ps], "c": "var(--axis)", "width": 1.2, "dashed": True})
    # the end-of-line labels are the short names; the margin on the right is sized from the longest one (about 6.5 px per character at the chart's font size, plus the offset of the label)
    chart = {"series": series, "xTicks": ps, "x0": 4, "x1": max(ps), "xLabel": "passkey tokens P (4 bits each; log scale)", "yLabel": "validation exact match at N = 256", "legendLine": True,
             "rightPad": label_margin([sr["short"] for sr in series]),
             "aria": "Passkey exact match at N = 256 against the passkey length for every chain"}

    # ---------------------------------------------------------------- tables
    def stage_row(c, e):
        v, t = e["validation"], e["test"]
        pos = v["positions"].get("256")
        failing = [j + 1 for j, x in enumerate(pos) if x < POS_OK] if pos else []
        sw = [s for s in e["switch_on_step"] if s]
        r = runs[e["run"]]
        return [f"<span title='{html.escape(c['label'], quote=True)}'>{html.escape(chain_short(c))}</span>", str(e["passkey_len"])] + [f2(v["exact"][str(n)]) for n in NS] + [f2(t["exact"]["256"]), f2(v["token"]["256"]),
                f"{v['positions_ok'].get('256', '?')} / {e['passkey_len']}", (", ".join(map(str, failing)) if len(failing) <= 4 else f"{len(failing)} pos.") if pos else "?",
                (f"{np.median(sw) / 1000:g}k ({max(sw) / 1000:g}k)" if sw else "never"), f2(t["ar_valid"]["256"]), f"{e['train_time_s'] / 3600:.1f} h" if e.get("train_time_s") else "–"]
    head = ["chain", "P"] + [f"N={n}" for n in NS] + ["test 256", "token acc.", "stored", "below 0.9", "switch-on", "AR valid", "time"]
    table = RR._table(head, [stage_row(c, e) for c in ch for e in c["stages"]], "wide compact")
    rates = seed_rates(ch)
    grid = position_grid(ch)
    pairs = rerun_pairs(ch)
    rate_rows = [[html.escape(r["short"]), str(P), str(4 * P), f"{reached}", f"{passed} of {reached}", f"{pos:g} of {P}", f2(ex)] for r in rates for P, reached, passed, pos, ex in r["rows"]]
    figs = [{"key": "chart", "title": "Exact match of the passkey against its length", "sub": "Validation split at N = 256 (1,000 streams); the point at P = 4 is the checkpoint the chain started from. Dashed lines are the short fine-tunes.",
             "caption": "<p>One line per chain: a model, a fine-tuning recipe and a seed. Each stage starts from the weights of the stage before it; a chain stops at the first stage below 0.9, so the last point of a line is its failing stage. "
                        "Exact match is all or nothing at these lengths, so lines of different seeds lie on top of each other; the next figure shows how much of the passkey each seed does store.</p>", "spec": chart},
            *([{"key": "positions", "title": "Which passkey positions are stored", "sub": "Validation split at N = 256: one row per stage with P of 16 or more, or with a position below 0.9; one cell per passkey position, darker is higher (hover for the value).",
                "caption": "<p>A cell is the share of validation streams in which that passkey token is right when the passkey is read out. White cells are at chance (1 in 16). Stages that pass have every cell dark.</p>", "spec": None, "html": grid}] if grid else []),
            *([{"key": "rates", "title": "How many seeds hold each passkey length",
                "sub": "Recipes with more than one seed: per P, the seeds that reached the stage (the chain stops at the first failure) and how many of them passed it.",
                "caption": "<p>Reached: seeds whose chain got to stage P (a seed reaches stage P only if it passed the stage before, so this falls as P grows). Passed: validation exact match at N = 256 of at least 0.9, out of those that reached it. "
                           "Stored: passkey positions right in at least 90% of the validation streams, and exact: the validation exact match, both medians over the seeds that reached the stage.</p>",
                "spec": None, "html": RR._table(["recipe", "P", "bits", "reached", "passed", "stored (median)", "exact (median)"], rate_rows, "wide")}] if rate_rows else []),
            *([{"key": "reruns", "title": "Longer budget on a stage that stopped a chain",
                "sub": "The stage that failed, trained again from the same starting weights with more steps (same learning rate, warm-up and curriculum). Validation split, N = 256.",
                "caption": "", "spec": None,
                "html": RR._table(["stage", "steps", "exact", "stored", "longer: steps", "exact", "stored"],
                                  [[f"{html.escape(MODEL_SHORT.get(q['model'], q['model']))}, seed {q['seed']}, P = {q['P']}", f"{q['short_steps'] // 1000}k", f2(q["short"]["validation"]["exact"]["256"]),
                                    f"{q['short']['validation']['positions_ok'].get('256', '?')} / {q['P']}", f"{q['long_steps'] // 1000}k", f2(q["long"]["validation"]["exact"]["256"]),
                                    f"{q['long']['validation']['positions_ok'].get('256', '?')} / {q['P']}"] for q in pairs], "wide")}] if pairs else []),
            {"key": "table", "title": "Every stage", "sub": "Columns N = 32 ... 256: validation exact match of the whole passkey (all P tokens right, teacher forced) at that stream length; test N = 256: the same on the test split; token acc.: per-token accuracy at N = 256 (validation); "
                                                      "stored: passkey positions right in at least 90% of the validation streams at N = 256; below 0.9: which positions are not; switch-on: the median (and the last) training step, in thousands, at which a stored position first reached 0.9; AR valid: associative recall (test, N = 256). "
                                                      "Hover a chain name for its full description (the position grid above shows every position).", "caption": "", "spec": None, "html": table}]
    if prog:
        figs.append({"key": "running", "title": "Stages in training", "sub": "Last periodic validation (every 1,000 steps).", "caption": "", "spec": None,
                     "html": RR._table(["run", "P", "step", "exact N = 256", "positions ≥ 0.9 at N = 256"], [[html.escape(p["run"]), str(p["P"]), f"{p['step']:,}", f2(p["exact256"]), f"{p['positions_ok']} / {p['P']}"] for p in prog], "wide")})

    # ---------------------------------------------------------------- text
    tldr = ["<b>Question.</b> A model trained on a 4-token passkey (16 bits) is fine-tuned on passkeys of P = 8, 16, 32, ... tokens, with the same streams of up to 256 AR writes: how long a passkey can the memory built inside the short-term store hold "
            "(validation exact match at N = 256 of at least 0.9)? Each stage starts from the weights of the previous one; nothing longer than N = 256 is evaluated."]
    in_group = {id(c) for r in rates for c in r["chains"]}   # chains of a recipe with several seeds get one sentence together (below), not one each
    for c in ch:
        if c["rerun_of"] or id(c) in in_group:
            continue
        passed = [e for e in c["stages"] if e["validation"]["exact"]["256"] >= THRESHOLD]
        failed = [e for e in c["stages"] if e["validation"]["exact"]["256"] < THRESHOLD]
        best = max(passed, key=lambda e: e["passkey_len"]) if passed else None
        txt = f"<b>{html.escape(c['label'])}.</b> " + (f"Holds P = {best['passkey_len']} ({best['bits']} bits): validation exact {f2(best['validation']['exact']['256'])} at N = 256, test {f2(best['test']['exact']['256'])}. " if best
                                                          else "No stage reaches 0.9. ")
        if failed:
            f = failed[0]
            pos = f["validation"]["positions"].get("256")
            txt += (f"Stops at P = {f['passkey_len']}: exact {f2(f['validation']['exact']['256'])}, token accuracy {f2(f['validation']['token']['256'])}"
                    + (f", {f['validation']['positions_ok']['256']} of {f['passkey_len']} positions at 0.9 or more" + (f" (below 0.9: positions {', '.join(str(j + 1) for j, x in enumerate(pos) if x < POS_OK)})" if sum(x < POS_OK for x in pos) <= 8 else "") if pos else "") + ".")
        else:
            txt += "No stage has failed yet" + (" (a stage may still be training)." if prog else ".")
        tldr.append(txt)
    for r in rates:
        top = [(P, reached, passed) for P, reached, passed, _, _ in r["rows"] if passed]
        last = r["rows"][-1]
        tldr.append(f"<b>Across seeds ({html.escape(r['label'])}, {len(r['seeds'])} seeds).</b> " + "; ".join(f"P = {P}: {passed} of {reached} seeds that reached it pass" for P, reached, passed, _, _ in r["rows"])
                    + (f". The longest passkey held by any seed is P = {max(t[0] for t in top)} ({4 * max(t[0] for t in top)} bits)" if top else ". No seed holds any of the lengths") + ".")
        stops = [(c["seed"], e) for c in r["chains"] if (e := stop_stage(c))]
        if stops:
            tldr.append(f"<b>Where the seeds stop ({html.escape(r['short'])}).</b> " + "; ".join(f"seed {sd}: P = {e['passkey_len']}, {e['validation']['positions_ok'].get('256', '?')} of {e['passkey_len']} positions stored" for sd, e in stops) + ".")
    fails = failing_positions([c for r in rates for c in r["chains"]])
    for P, cnt in sorted(fails.items()):
        n = sum(1 for r in rates for c in r["chains"] if (e := stop_stage(c)) and e["passkey_len"] == P)
        common = [j for j, k in sorted(cnt.items()) if k >= max(2, 0.6 * n)]
        if n >= 3 and common:
            tldr.append(f"<b>Which positions fail (P = {P}).</b> Of the {n} chains that stop at P = {P}, position{'s' if len(common) > 1 else ''} " + ", ".join(f"{j} ({cnt[j]} of {n})" for j in common)
                        + f" {'are' if len(common) > 1 else 'is'} below 0.9 in most of them; the others differ from seed to seed.")
    if pairs:
        better = sum(q["long"]["validation"]["exact"]["256"] >= THRESHOLD for q in pairs)
        tldr.append("<b>Longer budget.</b> The stage that stopped a chain, trained again from the same weights with more steps: " + "; ".join(
            f"{html.escape(MODEL_SHORT.get(q['model'], q['model']))} seed {q['seed']}, P = {q['P']}: {q['short']['validation']['positions_ok'].get('256', '?')} of {q['P']} positions stored at {q['short_steps'] // 1000}k steps, "
            f"{q['long']['validation']['positions_ok'].get('256', '?')} at {q['long_steps'] // 1000}k" for q in pairs) + f". {better} of {len(pairs)} pass."
            + ("" if all(q["same_start"] for q in pairs) else " (Not all re-runs start from the weights of the original stage.)"))
    if prog:
        name = lambda p: f"P = {p['P']}, {p['steps'] // 1000}k-step stage, seed {p['seed']}" if p.get("steps") else html.escape(p["run"])   # (the full run names are in the table below)
        tldr.append("<b>In training now.</b> " + "; ".join(f"{name(p)}: step {p['step']:,}, {p['positions_ok']} of {p['P']} positions at 0.9 or more, exact {f2(p['exact256'])}" for p in prog) + ".")
    data = {
        "ns": NS, "rows": rows, "results": results, "scatter_ns": [], "hide_scatter": True, "label_w": 400, "row_name": "chain and stage", "row_prefix": "",
        "meta": f"Passkey capacity, results of {time.strftime('%d %b %Y, %H:%M')}. Validation split, N counts AR writes. Chains: {len(ch)}.",
        "tldr": tldr, "tested": [], "results_title": "Results",
        "intro": ("<h2>Setup</h2><p>The checkpoint was trained on streams of up to 256 AR writes with a 4-token passkey. Stage P = 8, 16, 32, ...: a dataset with the same streams and a passkey of P tokens "
                  "(each token one of 16 values, 4 bits) is generated, the weights of the previous stage are fine-tuned on it, and the exact match of the passkey (all P tokens right) is measured on the validation and test split at N = 32 ... 256. "
                  "A chain stops at the first stage below 0.9 at N = 256. The recipes: a 10k-step fine-tune (lr 3e-4, no curriculum), and the original budget (50k steps, lr 1e-3, 500 warm-up steps, stream-length curriculum), "
                  "which turned out necessary for the vector loop.</p>"),
        "results_intro": "Validation exact match by stage and stream length (grid), and against the passkey length (line chart).",
        "fig1_title": "1. Exact match by stage and stream length", "fig1_sub": "Rows: stages of every chain. Columns: stream length N (AR writes). Validation split. Darker is higher.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)", "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": "<p>Top grid: passkey exact match. Middle and bottom grids: the associative-recall half, which should not suffer from the longer passkey.</p>", "cap2": "", "extra_figs": figs, "next_title": "Notes",
        "next": "<p>A chain stops at the first stage below the threshold, so the stop point of a short fine-tune can come earlier than that of a longer one on the same model: compare chains of one model before comparing models. "
                "A position counts as stored when it is right in at least 90% of the validation streams; at a failing stage the positions below 0.9 show which tokens were not learned.</p>",
    }
    return number_figures(data, first=2), "Passkey Capacity Report", "Passkey capacity: how many tokens fit in the memory inside the short-term store"
