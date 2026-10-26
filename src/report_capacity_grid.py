"""The capacity-grid page: passkey capacity of the two-pass layer as a function of the segment size and the window (scripts/capacity_grid.py).

    python src/report.py capgrid --out reports/capacity_grid_report.html

Everything on the page is computed from results/capacity_grid.json and the result files of the bases; cells and seeds that are not finished yet are marked as such, so the page can be built
while the jobs run.
"""
from __future__ import annotations

import html
import statistics
import time

import config as C  # noqa: E402
import capacity_grid as G  # noqa: E402
import report_rounds as RR  # noqa: E402
from report_groups import number_figures  # noqa: E402

PS = (4, 8, 16, 32, 64)
EARLIER = "stages_swa_2pass_budget"      # the six earlier chains of (32, 64) from the best base: keys stages_swa_2pass_budget[_s1 ... _s5]
f2 = lambda x: f"{x:.2f}"


def tokens_label(c: dict) -> str:
    """'16', '>=64' or '<4' for one chain's capacity."""
    t = c["tokens"]
    return "<4" if t == 0 else (f">={t}" if c["censored"] else str(t))


def states(grid: dict | None = None, root=G.ROOT) -> dict:
    """{(segment, window): [job_state of seed 0, 1, 2]}."""
    grid = G.load_results() if grid is None else grid
    return {(s, w): [G.job_state(s, w, g, grid, root) for g in G.SEEDS] for s, w in G.cells()}


def cell_summary(sts: list[dict]) -> dict:
    """{caps: capacities of the finished chains, running: the number of chains that have a base but are not finished, pending: chains without a base result, median: of the finished ones}."""
    done = [s["capacity"] for s in sts if s["capacity"]["done"]]
    caps = [c["tokens"] for c in done]
    return {"caps": caps, "labels": [tokens_label(c) for c in done], "done": len(done), "running": sum(1 for s in sts if s["base_exact"] is not None and not s["capacity"]["done"]),
            "pending": sum(1 for s in sts if s["base_exact"] is None), "median": statistics.median(caps) if caps else None, "censored": any(c["censored"] for c in done)}


def shade(tokens: float | None) -> str:
    """Background of a heatmap cell: darker is a larger capacity, on a log scale from 4 to 64 tokens (nothing: the surface)."""
    if tokens is None:
        return "var(--surface)"
    import math
    t = 0.12 if tokens < 4 else min(1.0, 0.18 + 0.82 * math.log2(max(tokens, 4) / 4) / 4)
    return f"color-mix(in srgb, var(--s1) {round(100 * t)}%, var(--surface))"


def num(x: float) -> str:
    return f"{x:g}"


def heatmap_html(by_cell: dict) -> str:
    """Rows: segment; columns: window. Big number: the median capacity of the finished seeds; below: the seeds' values."""
    head = "<tr><th></th>" + "".join(f"<th class='pc'>window {w}</th>" for w in G.WINDOWS) + "</tr>"
    body = ""
    for s in G.SEGMENTS:
        cells = ""
        for w in G.WINDOWS:
            sm = by_cell.get((s, w))
            if sm is None:
                cells += "<td class='c'></td>"
                continue
            seeds = " · ".join(sm["labels"]) if sm["labels"] else ""
            status = []
            if sm["running"]:
                status.append(f"{sm['running']} training")
            if sm["pending"]:
                status.append(f"{sm['pending']} waiting")
            big = ("…" if sm["median"] is None else ("&lt;4" if sm["median"] < 4 else num(sm["median"]) + ("+" if sm["censored"] else "")))
            tip = f"segment {s}, window {w}: capacity per seed (finished chains) {seeds or 'none yet'}" + (f"; {', '.join(status)}" if status else "")
            cells += (f"<td class='c' style='background:{shade(sm['median'])};' title='{html.escape(tip, quote=True)}'><div class='big'>{big}</div>"
                      f"<div class='small'>{seeds or '&nbsp;'}</div><div class='small'>{html.escape(', '.join(status)) if status else '&nbsp;'}</div></td>")
        body += f"<tr><td class='rl'>segment {s}</td>{cells}</tr>"
    return f"<div class='tbl'><table class='capgrid'><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def pass_rates(sts: list[dict]) -> dict:
    """{P: (seeds whose stage P passed, seeds whose chain reached stage P with a result)}; the denominator counts the chains that have a stage P result (a chain stops at its first failure)."""
    out = {}
    for P in PS[1:]:
        reached = [e for s in sts for e in s["stages"] if e["passkey_len"] == P]
        out[P] = (sum(e["validation"]["exact"]["256"] >= G.THRESHOLD for e in reached), len(reached))
    return out


def rates_table(by_state: dict) -> str:
    rows = []
    for (s, w), sts in sorted(by_state.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        pr = pass_rates(sts)
        cells = []
        for P in PS[1:]:
            ok, n = pr[P]
            cells.append(f"<span style='display:inline-block;min-width:3.2em;padding:0 4px;border-radius:3px;background:{shade(P if n and ok else None) if ok else 'var(--surface)'}'>{ok} / {n}</span>" if n else "–")
        base = [x["base_exact"] for x in sts if x["base_exact"] is not None]
        rows.append([f"segment {s}, window {w}", f"{sum(b >= G.THRESHOLD for b in base)} / {len(base)}", *cells])
    return RR._table(["cell", "base passes P = 4"] + [f"P = {P}" for P in PS[1:]], rows, "compact")


def chain_table(by_state: dict) -> str:
    rows = []
    for (s, w), sts in sorted(by_state.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for st in sts:
            by_p = {e["passkey_len"]: e for e in st["stages"]}
            cells = []
            for P in PS[1:]:
                e = by_p.get(P)
                if e is None:
                    cells.append("")
                    continue
                ok = e["validation"]["positions_ok"].get("256")
                cells.append(f2(e["validation"]["exact"]["256"]) + (f" ({ok}/{P})" if ok is not None and e["validation"]["exact"]["256"] < G.THRESHOLD else ""))
            cap = st["capacity"]
            state = ("not started" if st["base_exact"] is None else "done" if cap["done"] else "running")
            rows.append([f"segment {s}, window {w}", str(st["seed"]), "" if st["base_exact"] is None else f2(st["base_exact"]), *cells,
                         "" if st["base_exact"] is None else (tokens_label(cap) if cap["done"] else f"({cap['tokens']}) {state}")])
    return RR._table(["cell", "seed", "base (P = 4)"] + [f"P = {P}" for P in PS[1:]] + ["capacity"], rows, "compact")


def earlier_cell(grid: dict) -> list[int]:
    """Capacity of the six earlier chains of (32, 64) (one base, six fine-tuning seeds), from results/capacity.json."""
    import json
    cap = json.loads((C.ROOT / "results" / "capacity.json").read_text())
    out = []
    for k, v in sorted(cap.items()):
        if k.startswith(EARLIER) and isinstance(v, list) and v and not k.startswith("stages_swa_2pass_budget_runs"):
            c = G.capacity_of(v, None)
            out.append(c["tokens"])
    return out


def trend(by_cell: dict, axis: str) -> str | None:
    """A sentence on the effect of one axis on the median capacity of the cells that are finished: segment at fixed window, or window at fixed segment."""
    parts = []
    for fixed in (G.WINDOWS if axis == "segment" else G.SEGMENTS):
        line = [(v, by_cell[(v, fixed) if axis == "segment" else (fixed, v)]) for v in (G.SEGMENTS if axis == "segment" else G.WINDOWS) if ((v, fixed) if axis == "segment" else (fixed, v)) in by_cell]
        line = [(v, sm["median"]) for v, sm in line if sm["median"] is not None and sm["done"] == len(G.SEEDS)]
        if len(line) >= 2:
            parts.append(("window " if axis == "segment" else "segment ") + f"{fixed}: " + ", ".join(f"{axis} {v}: {num(m)}" for v, m in line))
    return "; ".join(parts) if parts else None


def report_data(grid: dict | None = None, root=G.ROOT) -> tuple[dict, str, str]:
    grid = G.load_results() if grid is None else grid
    by_state = states(grid, root)
    by_cell = {k: cell_summary(v) for k, v in by_state.items()}
    n_done = sum(sm["done"] for sm in by_cell.values())
    n_total = len(by_cell) * len(G.SEEDS)
    n_started = sum(sm["done"] + sm["running"] for sm in by_cell.values())
    tldr = [f"<b>Question.</b> How many passkey tokens (4 bits each) does the two-pass layer hold, as a function of the segment size and of the window? Nine cells, segment 8 / 16 / 32 x window 32 / 64 / 128, "
            f"three seeds each; every seed trains its own base on 4-token passkeys and is then fine-tuned on passkeys of 8, 16, 32, 64 tokens until a stage fails. <b>{n_done} of {n_total} chains are finished</b>"
            f"{'' if n_started == n_done else f', {n_started - n_done} are running'}."]
    done_cells = {k: sm for k, sm in by_cell.items() if sm["median"] is not None}
    if done_cells:
        best = max(done_cells.items(), key=lambda kv: (kv[1]["median"], kv[0][1] * -1))
        worst = min(done_cells.items(), key=lambda kv: (kv[1]["median"], kv[0][1]))
        line = (f"<b>Capacity per cell</b> (median over the finished seeds, figure {{fig:heat}}): the largest is {num(best[1]['median'])} tokens at segment {best[0][0]}, window {best[0][1]}"
                f"{' (a lower bound: the chain stops after 64)' if best[1]['censored'] else ''}, the smallest {num(worst[1]['median'])} at segment {worst[0][0]}, window {worst[0][1]}.")
        tldr.append(line)
    for axis, label in (("window", "<b>Window.</b> Median capacity as the window grows, segment fixed"), ("segment", "<b>Segment.</b> Median capacity as the segment grows, window fixed")):
        t = trend(by_cell, axis)
        if t:
            tldr.append(f"{label} (cells with all three seeds finished): {t}.")
    ref = earlier_cell(grid)
    if ref:
        tldr.append(f"<b>Reference.</b> The six earlier chains of (segment 32, window 64) all started from one base, the best of six, and reached {', '.join(str(x) for x in sorted(ref))} tokens; they are not seeds of this grid "
                    f"(here every seed has its own base), and the grid's own (32, 64) cell is shown in the figure.")
    figs = [
        {"key": "heat", "title": "Capacity by segment size and window",
         "sub": "Rows: tokens per segment. Columns: attention window in tokens. Big number: median over the finished seeds of the capacity = the largest passkey length (in tokens, 4 bits each) whose stage passed, "
                f"i.e. validation exact match of the passkey at least {G.THRESHOLD} at N = 256 AR writes, all smaller stages passed too; &lt;4 means the stage of 8 tokens failed and the base did not reach the threshold at 4 either; + means the chain was stopped at 64. "
                "Small numbers: the seeds' own values. Darker is larger (log scale).",
         "caption": "A chain starts from a base trained on 4-token passkeys and fine-tunes it on 8, 16, 32, 64 tokens (50k steps each, the original recipe) and stops at the first failing stage; the next stage would be twice as long, so a capacity of 16 means "
                    "that 16 passed and 32 failed: the true limit is between 16 and 31 tokens.", "spec": None, "html": heatmap_html(by_cell)},
        {"key": "rates", "title": "How many seeds pass each passkey length",
         "sub": "Per cell: seeds whose stage of that length passed / seeds whose chain has a result for that stage (a chain stops at its first failure, so the denominators fall with P). 'base passes' counts bases that reach the threshold at 4 tokens.",
         "caption": "", "spec": None, "html": rates_table(by_state)},
        {"key": "chains", "title": "Every chain",
         "sub": "Validation exact match of the passkey at N = 256 after each stage; for a failed stage the number of passkey positions stored (accuracy at least 0.9) in brackets. The base column is the base's own validation exact match at 4 tokens.",
         "caption": "", "spec": None, "html": chain_table(by_state)},
    ]
    data = {
        "ns": [], "rows": [], "results": {}, "scatter_ns": [], "hide_scatter": True, "hide_heatmaps": True, "label_w": 58, "row_name": "", "row_prefix": "",
        "meta": f"Capacity grid of the two-pass layer, results of {time.strftime('%d %b %Y, %H:%M')}. Validation split, N counts AR writes. Chains finished: {n_done} of {n_total}.",
        "tldr": tldr, "tested": [], "results_title": "Results",
        "intro": ("<h2>Method</h2><p>A two-pass layer (pass 1 reads the cached window and is the layer's output, pass 2 only writes the cache) is trained for each of nine (segment, window) pairs with segment ≤ window: "
                  "segment 8 / 16 / 32 tokens, window 32 / 64 / 128 tokens, two layers, width 128. <b>Base:</b> from scratch on 4-token passkeys with the usual recipe (50k steps, stream-length curriculum, seed g = 0, 1, 2; evaluated up to N = 256). "
                  "<b>Chain:</b> the base is fine-tuned on passkeys of 8 tokens (50k steps, learning rate 1e-3, warm-up 500, stream-length curriculum of 2000 steps per stage, fine-tuning seed g + 10), the result on 16 tokens, "
                  "and so on up to 64, stopping at the first stage whose validation exact match of the whole passkey at N = 256 is below 0.9. The streams have 32 to 256 AR writes, nothing longer is evaluated.</p>"
                  "<p>The earlier capacity chains of the two-pass layer (segment 32, window 64) all started from a single base, the best of six; here the bases are independent, so the (32, 64) cell is a new measurement with the E4 runs "
                  "swa_2pass_seg32_po1_s0-s2 as bases.</p>"),
        "results_intro": "Every number is computed from results/capacity_grid.json and the result files in results/raw/.",
        "extra_figs": figs, "next_title": "Notes",
        "next": ("<p>Capacity is measured on the doubling ladder 4, 8, 16, 32, 64, so each value stands for an interval (a capacity of 16 means that 16 passed and 32 failed). With three seeds the median is the second largest value; "
                 "the spread between seeds of one cell is of the size of the differences between neighbouring cells in the earlier six-seed chains, so differences of one step between cells should not be read as an effect. "
                 "A smaller segment means more segments per stream and a slower model: segment 8 trains about four times slower per step than segment 32.</p>"),
    }
    return number_figures(data, first=1), "Capacity Grid Report", "Passkey capacity by segment size and window"
