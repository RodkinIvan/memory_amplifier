"""Report of design experiment E2: the two-pass layer (`swa_2pass`), the short-term store re-written with what it read.

    python src/report.py p2 --out reports/p2_report.html

Every layer runs twice per segment on the key/value cache: pass 1 reads the window and stores nothing, pass 2 re-runs the block on the input plus the gained update of pass 1 and is
the only one that writes keys and values; nothing but the cache crosses a segment boundary. The report compares it (segment 32 with either pass as the output to the next layer,
segment 64) with the segment-recurrent vector loop `swa_fb` (H5 B / C), the value adaptation `swa_va` (E1), plain sliding-window attention and GDN-2, all with the same recipe
(50k steps, curriculum 2000, 2 layers, window 64). Every number in the text is computed from the results files; rows are configurations, grouped by the diff of their stored config
against the frozen reference (H7).
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
from report_best import mean_curve, over_n, pick_best  # noqa: E402
from report_groups import NS, cells_for_row, label_margin, number_figures  # noqa: E402

f2 = lambda x: f"{x:.2f}"
COLORS = ["var(--s2)", "var(--s3)", "var(--s1)", "var(--ink)"]
PROBE_DIR = C.ROOT / "results" / "probes"


def load_probes(directory: Path = PROBE_DIR, prefix: str = "p2") -> dict:
    """{'cache': ..., 'perturb': ..., 'gains': ...} from results/probes/<prefix>_cache_probe*.json, <prefix>_perturb*.json, <prefix>_gains*.json (whatever exists; several files of one kind
    are merged by run name). `p2` = the two-pass probes (src/probes_2pass.py)."""
    out: dict = {}
    for key, stem in (("cache", f"{prefix}_cache_probe"), ("perturb", f"{prefix}_perturb"), ("gains", f"{prefix}_gains")):
        for f in sorted(directory.glob(f"{stem}*.json")):
            out.setdefault(key, {}).update(json.loads(f.read_text()))
    return out


def cache_figure(probes: dict) -> list[dict]:
    """The linear read-out of the passkey from the cached keys and values: one table per feature, all probed runs (empty when nothing was probed)."""
    cache = probes.get("cache", {})
    if not cache:
        return []
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
    return [{"key": "probe_cache", "title": "Passkey read out of the cached window",
             "sub": f"Linear (ridge) read-out of each passkey token from the {reach // 2} cached positions of both layers, at growing numbers of tokens after the passkey; mean accuracy over the {first['passkey_len']} tokens, chance {first['chance']:.3f}; "
                    f"{first['n_train']} streams to fit, {first['n_test']} to test. Tables: cached values, cached keys.",
             "caption": "<p>The read-out never sees the passkey tokens themselves, only the cached values (or keys) after the given number of tokens. A plain window model with these dimensions reaches "
                        f"{reach} tokens" + (f"; the control ({html.escape(ctl['run'])}) is a plain sliding-window model, whose cache cannot hold the passkey beyond that." if ctl else ".") + "</p>",
             "spec": None, "html": "<p><b>Values</b></p>" + table("v") + "<p><b>Keys</b></p>" + table("k")}]


def method_svg() -> str:
    """One layer over one segment: the window is read twice, only the second pass writes."""
    tok = lambda x, y, cls, n=6: "".join(f"<rect x='{x + i * 21}' y='{y}' width='18' height='18' rx='3' class='{cls}'/>" for i in range(n))
    s = ["<svg viewBox='0 0 780 330' role='img' aria-label='Two-pass layer: pass 1 reads the cached window and stores nothing; pass 2 re-runs the block on the input plus the gained update and writes keys and values' style='width:100%;height:auto'>",
         "<style>.tk{fill:var(--s1);opacity:.85}.to{fill:var(--s3);opacity:.85}.bl{fill:var(--surface);stroke:var(--axis);stroke-width:1.2}.ca{fill:var(--surface);stroke:var(--s2);stroke-width:1.6}"
         ".ar{stroke:var(--ink-2);stroke-width:1.4;fill:none}.rd{stroke:var(--ink-2);stroke-width:1.2;fill:none;stroke-dasharray:5 4}.wr{stroke:var(--s2);stroke-width:2.4;fill:none}"
         ".t{font:11.5px var(--mono);fill:var(--ink-2)}.tb{font:12px var(--mono);fill:var(--ink)}.to2{font:11.5px var(--mono);fill:var(--s2)}</style>",
         "<defs><marker id='ah' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='var(--ink-2)'/></marker>"
         "<marker id='ao' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='var(--s2)'/></marker></defs>",
         "<text x='16' y='16' class='t'>one layer, one segment: h = input, y1 = block(h) read-only, y2 = block(h + g ⊙ (y1 − h))</text>"]
    # cache (left), the two passes (middle), notes (right); texts are at most ~6.9 px per character wide and every box is wider than its longest line
    s.append("<rect x='10' y='70' width='170' height='166' rx='4' class='ca'/><text x='95' y='130' text-anchor='middle' class='tb'>cache</text>")
    s.append("<text x='95' y='154' text-anchor='middle' class='t'>keys and values of</text><text x='95' y='170' text-anchor='middle' class='t'>the last W − 1 tokens,</text><text x='95' y='186' text-anchor='middle' class='to2'>as written by pass 2</text>")
    # input tokens and the two passes
    s.append(tok(287, 286, "tk"))
    s.append("<text x='350' y='322' text-anchor='middle' class='tb'>h (segment s)</text>")
    s.append("<path d='M350 282 L350 250' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='250' y='204' width='200' height='44' rx='4' class='bl'/><text x='350' y='223' text-anchor='middle' class='tb'>pass 1: block(h)</text><text x='350' y='240' text-anchor='middle' class='t'>reads, stores nothing</text>")
    s.append("<path d='M350 202 L350 172' class='ar' marker-end='url(#ah)'/>")
    s.append("<text x='358' y='192' class='t'>y1 − h, × g</text>")
    s.append("<circle cx='350' cy='156' r='15' fill='var(--surface)' stroke='var(--s2)' stroke-width='2'/><text x='350' y='162' text-anchor='middle' style='font:700 16px var(--mono);fill:var(--s2)'>+</text>")
    s.append("<path d='M416 295 L482 295 L482 156 L367 156' class='ar' marker-end='url(#ah)'/><text x='490' y='152' class='t'>h</text>")
    s.append("<path d='M350 140 L350 114' class='ar' marker-end='url(#ah)'/>")
    s.append("<rect x='250' y='68' width='200' height='44' rx='4' class='bl'/><text x='350' y='87' text-anchor='middle' class='tb'>pass 2: block(h2)</text><text x='350' y='104' text-anchor='middle' class='t'>same weights, writes K, V</text>")
    s.append("<path d='M350 66 L350 38' class='ar' marker-end='url(#ah)'/><text x='358' y='54' class='t'>y2 or y1 → next layer</text>")
    # reads and the write (the labels sit in the gap between the cache and the boxes, off the arrows)
    s.append("<path d='M182 206 L246 224' class='rd' marker-end='url(#ah)'/><text x='188' y='204' class='t'>read</text>")
    s.append("<path d='M182 128 L246 102' class='rd' marker-end='url(#ah)'/><text x='190' y='146' class='t'>read</text>")
    s.append("<path d='M246 80 L184 92' class='wr' marker-end='url(#ao)'/><text x='192' y='76' class='to2'>writes</text>")
    # notes
    for i, line in enumerate(["pass 1 reads, writes nothing", "pass 2: same block and weights,", "writes its keys and values", "g = 0: both passes equal (SWA)", "only the cache crosses segments"]):
        s.append(f"<text x='535' y='{104 + i * 20}' class='t'>{html.escape(line)}</text>")
    s.append("</svg>")
    return "".join(s)


def perturb_figure(probes: dict) -> list[dict]:
    """Perturbing the cached values, keys or both of the window in the middle of the stream, for the probed two-pass runs that hold the passkey."""
    pert = {r: res for r, res in probes.get("perturb", {}).items() if res["exact"]["clean"] >= 0.05 and "keys_zero" in res["exact"]}
    if not pert:
        return []
    rows = []
    for run, res in pert.items():
        ex = res["exact"]
        eps = sorted({float(k.split("_")[1]) for k in ex if k.split("_")[0] in ("values", "keys", "both") and k.split("_")[1] != "zero"})
        for who, key in (("cached values", "values"), ("cached keys", "keys"), ("both", "both")):
            rows.append([html.escape(run), who, f2(ex["clean"])] + [f2(ex[f"{key}_{e:g}"]) for e in eps] + [f2(ex[f"{key}_zero"])]
                        + [" / ".join(f"{v:.2f}" for v in res["contraction"].values()) if key == "both" else ""])
    res0 = next(iter(pert.values()))
    eps0 = sorted({float(k.split("_")[1]) for k in res0["exact"] if k.split("_")[0] == "values" and k.split("_")[1] != "zero"})
    return [{"key": "probe_perturb", "title": "Perturbing the cached window in mid-stream",
             "sub": f"Passkey exact match at N = {res0['n_writes']:,}; the kick lands after segment {res0['s_star']:,} (token {res0['s_star_tokens']:,}), far beyond the reach. Columns: noise of ε × the tensor's RMS, and zeroing. Only runs that hold the passkey (clean 0.05 or more).",
             "caption": "<p>The last column (rows 'both'): distance between the cached keys and values of a clean and a kicked run, relative to right after the kick (median over streams), after "
                        + " / ".join(res0["contraction"]) + " segments. Well below 1 means the store pulls itself back to the clean trajectory; near 1 that the kick is carried along.</p>", "spec": None,
             "html": RR._table(["run", "perturbed", "clean"] + [f"ε = {e:g}" for e in eps0] + ["zeroed", "kick size after later segments"], rows)}]


def gains_figure(probes: dict) -> list[dict]:
    """The learned gains g per layer of the probed runs (E2 hypothesis: with output 2 the gain is a signed tweak around 0, with output 1 one layer has a clearly positive mean)."""
    gains = probes.get("gains", {})
    if not gains:
        return []
    rows = []
    for run, res in gains.items():
        for i, l in enumerate(res["layers"]):
            rows.append([html.escape(run), str(i), f"{l['mean']:+.2f}", f"{l['abs_mean']:.2f}", f"{l['abs_max']:.2f}", f"{l['negative']:.0%}"])
    return [{"key": "gains", "title": "Learned gains per layer",
             "sub": "The per-channel gain g of the fed-back update y1 − h, read from the final weights; initialised to 0.",
             "caption": "<p>With output 2 the gain also changes the forward pass of the current tokens; with output 1 it acts only through the keys and values later segments read. The question is whether the two end up using it differently, "
                        "e.g. as signed tweaks around 0 against a clearly positive gain in some layer. Which layer is positive varies between seeds, and its size does not by itself say which seeds extrapolate.</p>",
             "spec": None, "html": RR._table(["run", "layer", "mean g", "mean |g|", "largest |g|", "negative channels"], rows)}]


def pilot_paragraph(raw: list[dict]) -> str:
    """The seed-0 pilots of the output-of-pass-2 variant (gain at 1x learning rate, a nonzero starting gain): what they reached."""
    pilots = [r for r in raw if r["cfg"]["model"]["kind"] == "swa_2pass" and (RG.diff_of(r).keys() & {"train.loop_lr_mult"} or (r["cfg"]["model"].get("gain_init") or 0) > 0)
              and not (RG.diff_of(r).keys() - {"model.segment", "model.pass_output", "model.gain_init", "train.loop_lr_mult"})]
    if not pilots:
        return ""
    parts = []
    for r in sorted(pilots, key=lambda r: r["name"]):
        row = {x["N"]: x["passkey_acc"] for x in r["rows"] if x["split"] == "test"}
        parts.append(f"{html.escape(r['name'])} ({html.escape(C.describe_diff(RG.diff_of(r)))}): {f2(row[32])} / {f2(row[256])} / {f2(row[1024])} at N = 32 / 256 / 1,024")
    return "<p><b>Pilots (output of pass 2, seed 0, not part of the comparison rows).</b> " + "; ".join(parts) + ".</p>"


def cache_bullet(probes: dict) -> list[str]:
    """One sentence of the cache read-out (numbers from the probe files): the mean accuracy over the passkey tokens at the farthest cut for every probed run, best first, with the token-by-token values of the best run;
    the full tables are in the figure."""
    cache = probes.get("cache", {})
    items, ex, ctl_far = [], None, None
    for run, res in cache.items():
        m, ctl = res["models"]["model"], res["models"].get("control")
        far = [d for d, e in m["by_distance"].items() if e["tokens"] >= 2 * m["reach"]]
        if not far:
            continue
        ex, d1 = res, far[-1]
        items.append((m["by_distance"][d1]["v"]["mean"], run, m["by_distance"][d1]["tokens"], m["by_distance"][d1]["v"]["per_token"]))
        if ctl:
            ctl_far = ctl["by_distance"][d1]["v"]["mean"]
    if not items:
        return []
    items.sort(reverse=True)
    short = lambda r: html.escape(r.replace("swa_2pass_", ""))
    best = items[0]
    return [f"<b>Where the passkey is.</b> A linear read-out of each passkey token from the cached values of the last window (no passkey token in sight; mean accuracy over the {ex['passkey_len']} tokens, chance {ex['chance']:.2f}) "
            f"about {best[2]:,} tokens after the passkey, {len(items)} probed runs: " + ", ".join(f"{short(r)} {f2(v)}" for v, r, _, _ in items)
            + (f"; plain SWA {f2(ctl_far)}" if ctl_far is not None else "") + f". Token by token for the best ({short(best[1])}): " + " / ".join(f2(x) for x in best[3])
            + f". The window holds at most {ex['models']['model']['reach'] // 2} tokens, so anything read out beyond that was carried by the store being re-written; "
              "a run can hold the passkey in the store and still fail to answer (the read-out here is linear and sees the cache only), and the other way round."]


def perturb_bullet(probes: dict) -> list[str]:
    """One sentence on the perturbation probe: per run that holds the passkey, the exact match clean, with keys and values kicked by 1 x RMS, and with both zeroed, and how fast a kick dies out."""
    parts, shrink = [], []
    for run, res in probes.get("perturb", {}).items():
        ex = res["exact"]
        if ex["clean"] < 0.05 or "keys_zero" not in ex:
            continue
        parts.append(f"{html.escape(run.replace('swa_2pass_', ''))} {f2(ex['clean'])} → {f2(ex.get('both_1', float('nan')))} → {f2(ex['both_zero'])}")
        shrink.append(res["contraction"].get("16", float("nan")))
    if not parts:
        return []
    return [f"<b>Perturbation in mid-stream</b> (far beyond the reach; exact match clean → keys and values kicked by 1 × RMS → both zeroed): " + "; ".join(parts)
            + f". A kick shrinks to {min(shrink):.2f}–{max(shrink):.2f} of its size within 16 segments."]


LEARN_N, LEARN_MIN = 256, 0.9      # E4 metric (fixed before the runs): a seed "learns the training range" when its test passkey at N = 256 is at least 0.9 ...
EXTRA_N, EXTRA_MIN = 4096, 0.5     # ... and "extrapolates" when its test passkey at N = 4,096 is at least 0.5.
SWITCH_N, SWITCH_MIN = 256, 0.9    # the loop "switches on" at the first periodic validation with passkey >= 0.9 at N = 256.
E4_ARMS = [("50k steps", "50k"), ("150k steps", "150k"), ("50k + 50k more", "cont")]  # (label, key); seeds 0-5 in each of the first two, the continuation of seeds 1 and 2


def e4_arm(run: dict) -> str | None:
    """Which E4 arm a finished run belongs to ('50k', '150k', 'cont'), None if it is not one of them (segment 32, output of pass 1, gain 0, the standard recipe otherwise)."""
    d = RG.diff_of(run)
    if run["cfg"]["model"]["kind"] != "swa_2pass" or RG.setting({"kind": "swa_2pass", "diff": d}, "model.segment") != 32:
        return None
    if set(d) - {"model.pass_output", "train.steps", "train.init_from"} or "model.pass_output" not in d or d["model.pass_output"][1] != 1:
        return None
    steps, init = run["cfg"]["train"]["steps"], run["cfg"]["train"].get("init_from")
    return "cont" if init and steps == 50000 else "150k" if steps == 150000 and not init else "50k" if steps == 50000 and not init else None


def switch_on_step(run: dict, logs: dict) -> int | None:
    """First periodic validation with passkey >= SWITCH_MIN at N = SWITCH_N (None if there is none)."""
    return next((st for st, v in RR.curve(run, logs, SWITCH_N) if v >= SWITCH_MIN), None)


def e4_runs(raw: list[dict], logs: dict) -> dict[str, dict]:
    """{arm key: {"done": [finished runs], "running": [stub runs with the name and seed only, for the curves of runs still in training]}}."""
    import re
    arms = {k: {"done": [], "running": []} for _, k in E4_ARMS}
    for r in raw:
        k = e4_arm(r)
        if k:
            arms[k]["done"].append(r)
    done = {r["name"] for r in raw}
    pattern = {"50k": r"swa_2pass_seg32_po1_s(\d+)", "150k": r"swa_2pass_seg32_po1_steps150000_s(\d+)", "cont": r"swa_2pass_seg32_po1_steps50000_s(\d+)"}
    for name in sorted(logs):
        for k, pat in pattern.items():
            m = re.fullmatch(pat, name)
            if m and name not in done:
                arms[k]["running"].append({"name": name, "seed": int(m.group(1))})
    for v in arms.values():
        v["done"].sort(key=lambda r: r["seed"])
        v["running"].sort(key=lambda r: r["seed"])
    return arms


def compact_diff(diff: dict) -> str:
    """The diff of an E4 run against swa_2pass_v1 in a few words: 'output 1', '150k steps', 'from seg32_po1_s1' (the weights it started from)."""
    parts = []
    for k, (a, b) in diff.items():
        if k == "model.pass_output":
            parts.append(f"output {b}")
        elif k == "train.steps":
            parts.append(f"{b // 1000}k steps")
        elif k == "train.init_from":
            parts.append("from " + Path(str(b)).parent.name.replace("swa_2pass_", ""))
        else:
            parts.append(f"{k.split('.')[-1]} {C.fmt(a)} → {C.fmt(b)}")
    return ", ".join(parts)


def e4_table(head: list[str], rows: list) -> str:
    """Table of the E4 runs: a row is a list of cells, or {'span': html} for a summary line over the whole width; the first two columns are left-aligned text."""
    def cell(i, c, tag="td"):
        style = " style='text-align:left'" if i < 2 else ""
        return f"<{tag}{style}>{c}</{tag}>"
    body = "".join(f"<tr><td colspan='{len(head)}' style='text-align:left'>{r['span']}</td></tr>" if isinstance(r, dict) else "<tr>" + "".join(cell(i, c) for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='tbl compact'><table><thead><tr>{''.join(cell(i, h, 'th') for i, h in enumerate(head))}</tr></thead><tbody>{body}</tbody></table></div>"


def longer_training(raw: list[dict], logs: dict) -> tuple[list[dict], list[str]]:
    """E4: one table with the arms, per seed (test passkey at every N, the step at which the loop switches on) and the two rates per arm, the configuration diff on every row; two curve charts
    (validation passkey at N = 256 and N = 4,096 against the step, mean of an arm with bars from the worst to the best seed). Returns (figures, summary sentences); nothing when there are no E4 runs."""
    arms = e4_runs(raw, logs)
    if not any(v["done"] or v["running"] for v in arms.values()):
        return [], []
    n_cols = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    head = ["run", "differs from swa_2pass_v1"] + [f"{n:,}" for n in n_cols] + ["on at", "learns", "extrap."]
    rows, rates, sentences = [], {}, []
    for label, key in E4_ARMS:
        done = arms[key]["done"]
        if not done and not arms[key]["running"]:
            continue
        learn = extra = 0
        for r in done:
            row = {x["N"]: x["passkey_acc"] for x in r["rows"] if x["split"] == "test"}
            sw = switch_on_step(r, logs)
            ok_l, ok_e = row[LEARN_N] >= LEARN_MIN, row[EXTRA_N] >= EXTRA_MIN
            learn, extra = learn + ok_l, extra + ok_e
            rows.append([f"{label.replace(' steps', '').replace('50k + 50k more', '50k+50k')}, seed {r['seed']}", html.escape(compact_diff(RG.diff_of(r)))] + [f2(row[n]) for n in n_cols]
                        + [f"{sw:,}" if sw else "never", "yes" if ok_l else "no", "yes" if ok_e else "no"])
        run_txt = ", ".join(f"seed {r['seed']} (step {max(logs[r['name']]):,})" for r in arms[key]["running"] if logs.get(r["name"]))
        n_total = len(done) + len(arms[key]["running"])
        sw_all = [x for x in (switch_on_step(r, logs) for r in done) if x]
        rows.append({"span": f"<b>{label}: {learn} of {len(done)} seeds learn the training range, {extra} of {len(done)} extrapolate</b>" + (f"; median switch-on step {int(np.median(sw_all)):,}" if sw_all else "")
                             + (f" <i>(still training: {run_txt})</i>" if run_txt else "")})
        rates[key] = (learn, extra, len(done), n_total)
    figs = [{"key": "e4_table", "title": "E4: does longer training fix the seed instability?",
             "sub": f"Two-pass layer, segment 32, output of pass 1. Test split; the numbered columns are the passkey exact match at that stream length N. A seed learns the training range when its passkey at N = {LEARN_N} is at least {LEARN_MIN}, extrapolates when its passkey at N = {EXTRA_N:,} is at least {EXTRA_MIN}; "
                    f"the loop switches on at the first periodic validation with passkey at least {SWITCH_MIN} at N = {SWITCH_N}.",
             "caption": "<p>Rows are runs, labelled with the diff of their stored config against the frozen reference; the bold rows are the per-arm rates. 'on at': the step at which the loop switches on (first periodic validation with passkey ≥ 0.9 at N = 256). The 150k arm uses the same warm-up, a cosine decay to 10% at its last step and a curriculum stage "
                        "every 2,000 steps, so it is not the 50k run continued. The continuation row starts from the final weights of a 50k run and warms the learning rate up again: it is its own row, not the 100k point of a curve.</p>",
             "spec": None, "html": e4_table(head, rows)}]
    for n, key in ((256, "e4_c256"), (4096, "e4_c4096")):
        sr = []
        for (label, k), col in zip(E4_ARMS, ("var(--s2)", "var(--s1)", "var(--s3)")):
            runs = arms[k]["done"] + arms[k]["running"]
            if runs:
                c = mean_curve(runs, logs, n, f"{label}, mean of {len(runs)} seed{'s' if len(runs) != 1 else ''} (bars: worst to best)", col, width=2.6, short=label)
                if c:
                    sr.append(c)
        if sr:
            x1 = max(p["x"] for s_ in sr for p in s_["points"])
            figs.append({"key": key, "title": f"E4: validation passkey at N = {n:,} against the training step",
                         "sub": "Validation streams, every 2,000 steps (N = 4,096: 100 streams). Runs still in training are included up to the step they have reached; the bars come from the steps all runs of an arm share.",
                         "caption": "", "spec": {"series": sr, "xTicks": list(range(0, x1 + 1, 25000)), "x0": 0, "x1": x1, "xLog": False, "xLabel": "training step", "yLabel": f"validation passkey, N = {n:,}", "legendLine": True, "rightPad": label_margin([x["short"] for x in sr], 70),
                                                  "aria": f"Validation passkey at N = {n} during training for the E4 arms"}})
    for label, key in E4_ARMS:
        if key in rates:
            learn, extra, nd, nt = rates[key]
            if nd:
                sentences.append(f"{label}: {learn} of {nd} finished seeds learn the training range, {extra} of {nd} extrapolate" + (f" ({nt - nd} more still training)" if nt > nd else ""))
            else:
                sentences.append(f"{label}: no seed finished yet ({nt} still training)")
    return figs, sentences


def fisher_two_sided(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided exact test (Fisher) that the rates a / n1 and b / n2 come from the same share: the probability, among tables with the same totals, of one at most as likely as the observed."""
    import math
    k, n = a + b, n1 + n2
    pmf = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    p0 = pmf(a)
    return float(sum(pmf(x) for x in range(max(0, k - n2), min(n1, k) + 1) if pmf(x) <= p0 + 1e-12))


def e4_paragraph(raw: list[dict], logs: dict) -> str:
    """E4 in words, computed: the two rates of the 50k and 150k arms and whether they can be told apart, what a threefold schedule did to the passkey at N = 1,024, when the loop switched on,
    whether the seeds that do not extrapolate ever had a validation signal at N = 4,096, and the continuation rows. Nothing is written when an arm has no finished seed."""
    arms = e4_runs(raw, logs)
    a50, a150, cont = arms["50k"]["done"], arms["150k"]["done"], arms["cont"]["done"]
    if not (a50 and a150):
        return ""
    test = lambda r, n: next(x["passkey_acc"] for x in r["rows"] if x["split"] == "test" and x["N"] == n)
    rate = lambda runs, n, thr: sum(test(r, n) >= thr for r in runs)
    l50, l150, e50, e150 = rate(a50, LEARN_N, LEARN_MIN), rate(a150, LEARN_N, LEARN_MIN), rate(a50, EXTRA_N, EXTRA_MIN), rate(a150, EXTRA_N, EXTRA_MIN)
    pl, pe = fisher_two_sided(l50, len(a50), l150, len(a150)), fisher_two_sided(e50, len(a50), e150, len(a150))
    m1024 = lambda runs: float(np.mean([test(r, 1024) for r in runs]))
    sw = lambda runs: [x for x in (switch_on_step(r, logs) for r in runs) if x]
    best4096 = lambda runs: [max(v for _, v in RR.curve(r, logs, EXTRA_N)) for r in runs if test(r, EXTRA_N) < EXTRA_MIN and RR.curve(r, logs, EXTRA_N)]
    b150 = best4096(a150)
    txt = (f"<p><b>E4: training budget or seed? (segment 32, output of pass 1.)</b> Learning the training range (test passkey ≥ {LEARN_MIN} at N = {LEARN_N}): {l50} of {len(a50)} seeds at 50k steps, {l150} of {len(a150)} at 150k "
           f"(two-sided exact test p = {pl:.2f}). Extrapolating (≥ {EXTRA_MIN} at N = {EXTRA_N:,}): {e50} of {len(a50)} against {e150} of {len(a150)} (p = {pe:.2f}); with {len(a50)} seeds per arm, rates this far apart cannot be told apart. "
           f"Tripling the schedule did not raise the mean passkey at N = 1,024 either ({m1024(a50):.2f} at 50k, {m1024(a150):.2f} at 150k). "
           f"The loop switched on (validation passkey ≥ {SWITCH_MIN} at N = {SWITCH_N}) at step {int(np.median(sw(a50))):,} (median of {len(sw(a50))} seeds) at 50k and {int(np.median(sw(a150))):,} ({len(sw(a150))} seeds, latest {max(sw(a150)):,}) at 150k, "
           f"so the extra steps came long after it. ")
    early = [v for r in a150 if test(r, EXTRA_N) >= EXTRA_MIN for st, v in RR.curve(r, logs, EXTRA_N) if st <= 30000]
    if b150:
        txt += (f"At 150k steps the {len(b150)} seeds that do not extrapolate had a validation passkey at N = {EXTRA_N:,} of at most {max(b150):.2f} at any evaluation of the whole run"
                + (f", and the extrapolating seed{'s' if e150 > 1 else ''} already had {max(early):.2f} by step 30,000: extrapolation was there early or not at all. " if early else ". "))
    if cont:
        txt += ("Continuing seeds " + " and ".join(str(r["seed"]) for r in cont) + " for another 50k steps (the learning rate warmed up again) gives "
                + "; ".join(f"seed {r['seed']}: passkey {test(r, 256):.2f} at N = 256 (was {test(next(x for x in a50 if x['seed'] == r['seed']), 256):.2f}), {test(r, 1024):.2f} at N = 1,024 (was {test(next(x for x in a50 if x['seed'] == r['seed']), 1024):.2f}), {test(r, EXTRA_N):.2f} at N = {EXTRA_N:,}"
                            for r in cont if any(x["seed"] == r["seed"] for x in a50)) + ".")
    return txt + "</p>"


def p2_data(raw: list[dict] | None = None, logs: dict | None = None, probes: dict | None = None) -> tuple[dict, str, str]:
    raw = raw if raw is not None else A.load_runs(C.ROOT / "results" / "raw")
    logs = logs if logs is not None else RR.log_curves(C.ROOT / "logs")
    probes = probes if probes is not None else load_probes()
    dgroups = RG.diff_groups([r for r in raw if r["cfg"]["model"]["kind"] in ("swa", "swa_fb", "swa_va", "swa_2pass", "gdn2")])
    p2 = {(RG.effective_segment(g), int(RG.setting(g, "model.pass_output")), float(RG.setting(g, "model.gain_init"))): g for g in dgroups
          if g["kind"] == "swa_2pass" and set(g["diff"]) <= {"model.segment", "model.pass_output", "model.gain_init"}}
    if not p2:
        raise ValueError("no swa_2pass runs found")
    fb = {seg: g for g in dgroups if (key := RG.standard_key(g)) and (key[0], key[1:]) in ((32, ("none", 0.0, 50000, 0.0)), (64, ("tanh", 0.0, 50000, 0.0))) and (seg := key[0]) in (32, 64)}
    va32 = next((g for g in dgroups if g["kind"] == "swa_va" and set(g["diff"]) <= {"model.segment"} and RG.effective_segment(g) == 32), None)
    swa64 = next((g for g in dgroups if g["kind"] == "swa" and RG.setting(g, "model.window") == 64 and RG.setting(g, "train.steps") == 50000 and RG.setting(g, "train.curriculum_stage_steps") == 2000
                  and RG.setting(g, "model.n_layers") == 2 and set(g["diff"]) <= {"model.window", "train.steps", "train.curriculum_stage_steps"}), None)
    gdn2 = next((g for g in dgroups if g["kind"] == "gdn2" and RG.setting(g, "train.steps") == 50000 and RG.setting(g, "train.curriculum_stage_steps") == 2000
                 and set(g["diff"]) <= {"train.steps", "train.curriculum_stage_steps"}), None)
    order = sorted(p2, key=lambda k: (k[0], -k[1], k[2]))  # segment 32 first, pass 2 before pass 1, gain 0 before a nonzero gain
    ref = C.load_reference({"model": {"kind": "swa_2pass"}})
    default_po = int(C.load_config(C.ROOT / "configs" / "swa_2pass.yaml", [])[0]["model"]["pass_output"])  # the working config's default (E3); the reference keeps its own
    reach = ref["model"]["n_layers"] * (ref["model"]["window"] - 1)
    tc = next(iter(p2.values()))["runs"][0]["cfg"]["train"]
    seeds_lbl = lambda g: f"{len(g['runs'])} seed{'s' if len(g['runs']) != 1 else ''}"
    tag = lambda k: f"seg{k[0]}_po{k[1]}" + (f"_gi{k[2]:g}" if k[2] else "")
    name = lambda k: f"two-pass, segment {k[0]}, output of pass {k[1]}" + (f", gain {k[2]:g} at the start" if k[2] else "")

    rows, results = [], {}
    add = lambda key, label, runs, color, long=None: (rows.append({"key": key, "label": label, "color": color, "long": long or label}), results.__setitem__(key, cells_for_row(runs, "test")))
    has_val = lambda r: all(any(x["N"] == n and x["split"] == "validation" for x in r["rows"]) for n in (4096, 16384))
    # the best checkpoints, picked on the validation split (same rule as the presentation sub-report): all output-1 two-pass runs of the E4 study, and the best vector-loop checkpoint (segment 8)
    p2_pool = [r for r in raw if r["cfg"]["model"]["kind"] == "swa_2pass" and e4_arm(r) in ("50k", "150k") and has_val(r)]
    best2 = pick_best(p2_pool) if p2_pool else None
    fb8 = next((g for g in dgroups if (key := RG.standard_key(g)) and key == (8, "none", 0.0, 50000, 0.0)), None)
    bestfb = pick_best([r for r in fb8["runs"] if has_val(r)]) if fb8 and any(has_val(r) for r in fb8["runs"]) else None
    if best2:
        add("best_p2", f"best two-pass checkpoint: segment 32, output of pass 1, seed {best2['seed']}, {best2['cfg']['train']['steps'] // 1000}k steps", [best2], "--s2", f"{best2['name']} (picked on the validation split among the {len(p2_pool)} output-1 runs of the E4 study)")
    if bestfb:
        add("best_fb", f"best vector-loop checkpoint: segment 8, no tanh, seed {bestfb['seed']}, 50k steps", [bestfb], "--s1", f"{bestfb['name']} (picked on the validation split among the runs of that setup)")
    for i, k in enumerate(order):
        g = p2[k]
        add(f"p2_{tag(k)}", f"{name(k)}, mean of {seeds_lbl(g)}", g["runs"], COLORS[i % 3], f"{RG.label(g)} ({seeds_lbl(g)}: {', '.join(r['name'] for r in g['runs'])})")
    for seg in (32, 64):
        if seg in fb:
            add(f"fb{seg}", f"vector loop m, segment {seg}, {'no tanh' if seg == 32 else 'tanh'}, mean of {seeds_lbl(fb[seg])}", fb[seg]["runs"], "--s1", RG.label(fb[seg]))
    if va32:
        add("va32", f"value adaptation (E1), segment 32, mean of {seeds_lbl(va32)}", va32["runs"], "--s1", RG.label(va32))
    if swa64:
        add("w64", "plain SWA, window 64 (reach 126)", swa64["runs"], "--muted", RG.label(swa64))
    if gdn2:
        add("gdn2", f"GDN-2 (learned decay), mean of {seeds_lbl(gdn2)}", gdn2["runs"], "--muted", "gated DeltaNet, stock learned decay, our recipe")
    cell = lambda key, n, f="passkey": results[key][str(n)][f]
    per = lambda key, n: results[key][str(n)]["per"]["passkey"]
    hz = lambda key: next((n for n in reversed(NS) if cell(key, n) is not None and cell(key, n) >= 0.9), None)

    series = []
    if best2:
        series.append(over_n([best2], f"best two-pass checkpoint (seed {best2['seed']}, {best2['cfg']['train']['steps'] // 1000}k steps, picked on validation)", "var(--s2)", width=3.8, short="best 2-pass"))
    if bestfb:
        series.append(over_n([bestfb], f"best vector-loop checkpoint (segment 8, seed {bestfb['seed']}, picked on validation)", "var(--s1)", width=3.8, short="best m-loop"))
    for i, k in enumerate(order):
        series.append(over_n(p2[k]["runs"], f"{name(k)}, mean of {seeds_lbl(p2[k])} (bars: worst to best seed)", COLORS[i % 3], width=3.0, short=f"2-pass {tag(k)}"))
    for seg in (32, 64):
        if seg in fb:
            series.append(over_n(fb[seg]["runs"], f"vector loop m, segment {seg}, mean of {seeds_lbl(fb[seg])}", "var(--s1)", dashed=True, width=2.0, short=f"m seg {seg}"))
    if va32:
        series.append(over_n(va32["runs"], f"value adaptation (E1), segment 32, mean of {seeds_lbl(va32)}", "var(--muted)", dashed=True, width=2.0, short="VA seg 32"))
    if swa64:
        series.append(over_n(swa64["runs"], "plain SWA, window 64", "var(--ink-2)", width=2.0, short="SWA W64"))
    if gdn2:
        series.append(over_n(gdn2["runs"], f"GDN-2, mean of {seeds_lbl(gdn2)}", "var(--ink)", dashed=True, width=2.0, short="GDN-2"))
    chart_n = {"series": series, "xTicks": NS, "x0": 32, "x1": 16384, "xLabel": "stream length N (AR writes, log scale; 16,384 writes ≈ 61k tokens)", "yLabel": "passkey exact match (test)", "legendLine": True, "rightPad": label_margin([x["short"] for x in series], 96),
               "vlines": [{"x": max(tc["train_ns"]), "label": "longest training stream", "dy": 0}], "aria": "Passkey accuracy against stream length for the two-pass layer, the vector loop, value adaptation and the baselines"}

    def curves(n: int) -> dict | None:
        sr = []
        for i, k in enumerate(order):
            c = mean_curve(p2[k]["runs"], logs, n, f"{name(k)}, mean of {seeds_lbl(p2[k])} (bars: worst to best)", COLORS[i % 3], width=2.6, short=f"2-pass {tag(k)}")
            if c:
                sr.append(c)
        for seg in (32, 64):
            if seg in fb:
                c = mean_curve(fb[seg]["runs"], logs, n, f"vector loop m, segment {seg}, mean of {seeds_lbl(fb[seg])}", "var(--s1)", width=1.8, dashed=True, short=f"m seg {seg}")
                if c:
                    sr.append(c)
        if not sr:
            return None
        x1 = max(p["x"] for s_ in sr for p in s_["points"])
        return {"series": sr, "xTicks": list(range(0, x1 + 1, 10000)), "x0": 0, "x1": x1, "xLog": False, "xLabel": "training step", "yLabel": f"validation passkey, N = {n:,}", "legendLine": True, "rightPad": label_margin([x["short"] for x in sr], 70),
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
            figs.append({"key": key, "title": f"Training: validation passkey at N = {n:,}", "sub": "Validation streams, every 2,000 steps (N = 4,096: 100 streams, N = 16,384: 32 streams). Solid = two-pass, dashed = the vector loop m at the same segment; bars from the worst to the best seed.",
                         "caption": "", "spec": c})
    e4_figs, e4_sentences = longer_training(raw, logs)
    figs += e4_figs + cache_figure(probes) + perturb_figure(probes) + gains_figure(probes)
    figs.append(RR.configuration_appendix([g for g in dgroups if g["kind"] == "swa_2pass" or g in list(fb.values()) + [x for x in (va32, swa64, gdn2) if x]]))

    # ---------------------------------------------------------------- text
    first = next(iter(p2.values()))
    tldr = [
        "<b>Idea.</b> No loop vector, no adaptation matrix: the only thing that crosses a segment boundary is the sliding-window key/value cache, and the cache is re-written with what the layer read. "
        "Per segment every layer runs twice: pass 1 reads the window (cached entries plus the segment, causal) and stores nothing; pass 2 re-runs the same block on h + g ⊙ (y1 − h), the input plus the gained update of pass 1, "
        "with queries, keys and values all from that new input, and only its keys and values are stored. Memory persists only by being re-read and re-written. "
        f"g is one learned gain per channel and layer (0 at the start, so the model starts as plain SWA); the next layer gets the output of pass {default_po} (the default of the working config since E3) or of pass {3 - default_po} "
        f"(the frozen reference recipe uses pass {ref['model']['pass_output']}; runs of the working config carry that difference in their diff). {first['runs'][0]['n_params']:,} parameters, window {ref['model']['window']}, {ref['model']['n_layers']} layers.",
    ]
    for k in order:
        key = f"p2_{tag(k)}"
        g = p2[k]
        h = hz(key)
        sent = (f"<b>{name(k).capitalize()}.</b> Passkey exact match {f2(cell(key, 256))} / {f2(cell(key, 1024))} / {f2(cell(key, 4096))} / {f2(cell(key, 16384))} (mean of {seeds_lbl(g)}, test) at N = 256 / 1,024 / 4,096 / 16,384; "
                f"per seed at N = 1,024 {RG.seeds_text(per(key, 1024))}, at N = 16,384 {RG.seeds_text(per(key, 16384))}; "
                + (f"the mean stays at 0.9 or above up to N = {h:,}." if h else "the mean never reaches 0.9."))
        if k[0] in fb:
            sent += f" The vector loop m at segment {k[0]} scores {f2(cell(f'fb{k[0]}', 256))} / {f2(cell(f'fb{k[0]}', 1024))} / {f2(cell(f'fb{k[0]}', 4096))} / {f2(cell(f'fb{k[0]}', 16384))} (per seed at 1,024: {RG.seeds_text(per(f'fb{k[0]}', 1024))})."
        tldr.append(sent)
    if best2:
        b = lambda r, n: next(x["passkey_acc"] for x in r["rows"] if x["split"] == "test" and x["N"] == n)
        txt = f"<b>Best checkpoints</b> (picked on the validation split, reported on test, passkey at N = 256 / 1,024 / 4,096 / 16,384). Two-pass, seed {best2['seed']} ({best2['cfg']['train']['steps'] // 1000}k steps): " + " / ".join(f2(b(best2, n)) for n in (256, 1024, 4096, 16384))
        if bestfb:
            txt += f". Vector loop, segment 8, seed {bestfb['seed']}: " + " / ".join(f2(b(bestfb, n)) for n in (256, 1024, 4096, 16384))
        tldr.append(txt + ". Seeds that converge well exist in both families; the means above are what a new run can be expected to do.")
    base = []
    if va32:
        base.append(f"value adaptation (E1, segment 32) {f2(cell('va32', 256))} / {f2(cell('va32', 1024))} / {f2(cell('va32', 16384))} at N = 256 / 1,024 / 16,384")
    if swa64:
        base.append(f"plain SWA (window 64) {f2(cell('w64', 256))} at N = 256 and {f2(cell('w64', 1024))} at N = 1,024")
    if gdn2:
        base.append(f"GDN-2 (mean of {seeds_lbl(gdn2)}) {f2(cell('gdn2', 1024))} / {f2(cell('gdn2', 4096))} / {f2(cell('gdn2', 16384))} at N = 1,024 / 4,096 / 16,384")
    if base:
        tldr.append("<b>For comparison, same recipe.</b> " + "; ".join(base) + ".")
    if e4_sentences:
        tldr.append("<b>Longer training (E4, segment 32, output of pass 1).</b> " + "; ".join(e4_sentences) + ". Per-seed values and the switch-on steps are in the E4 table.")
    tldr += cache_bullet(probes) + perturb_bullet(probes)
    running = [f"{nm} (step {max(st):,})" for nm, st in sorted(logs.items()) if nm.startswith("swa_2pass_") and nm not in {r["name"] for r in raw}]
    data = {
        "ns": NS, "rows": rows, "results": results, "scatter_ns": [], "hide_scatter": True, "label_w": 352, "row_name": "configuration", "row_prefix": "",
        "meta": f"Design experiment E2, results of {time.strftime('%d %b %Y, %H:%M')}. Test split, N counts AR writes, one write is about 3.75 tokens. Every row is one configuration; seeds are averaged only within a configuration."
                + ((f" Still training (not in the tables): {', '.join(running)}." if len(running) <= 4 else
                    f" {len(running)} runs are still training and are not in the tables (steps {min(int(x.split('step ')[1].rstrip(')').replace(',', '')) for x in running):,} to {max(int(x.split('step ')[1].rstrip(')').replace(',', '')) for x in running):,}).") if running else ""),
        "tldr": tldr, "tested": [], "results_title": "Results",
        "intro": ("<h2>Method</h2><p>The task: remember a 4-token passkey shown once at the start of a stream and repeat it at the end, while answering associative-recall queries about recent key–value writes. "
                  "A plain sliding-window model cannot do the first part beyond its window. Here the long-term memory is meant to be the short-term store itself, kept alive by rehearsal: each segment, the layer reads the "
                  "window, and the entries it stores for the segment are re-computed from what it read.</p>"
                  f"<figure><div class='fig-title'>One layer, one segment</div><div class='chart'>{method_svg()}</div>"
                  "<figcaption>Pass 1 attends over the cached window plus the segment (causal) and writes nothing. Pass 2 is the same block with the same weights on h + g ⊙ (y1 − h); its queries, keys and values all come from that input, "
                  "and its keys and values are what later segments see. Both passes use the same positions. With g = 0 pass 2 equals pass 1 and the model is plain sliding-window attention.</figcaption></figure>"),
        "results_intro": "The two-pass layer (segment 32 with the output of either pass, segment 64) against the vector loop m, value adaptation, plain SWA and GDN-2, all with the same recipe (50k steps, stream-length curriculum, 2 layers, window 64).",
        "fig1_title": "1. Accuracy by configuration and stream length", "fig1_sub": "Rows: the two-pass configurations (mean of the seeds), the vector loop, value adaptation, baselines. Columns: stream length N. Darker is higher.",
        "valid_title": "AR valid accuracy (key written within the last 16 writes)", "stale_title": "AR expired accuracy, stale keys only (last write more than 32 writes ago)",
        "cap1": "<p>Top grid: passkey. Middle and bottom grids: the associative-recall half.</p>", "cap2": "", "extra_figs": figs, "next_title": "Open issues",
        "next": e4_paragraph(raw, logs) + next_text(p2, order, fb, logs, probes) + pilot_paragraph(raw),
    }
    return number_figures(data, first=2), "SWA Two-Pass Report", "Two-pass layers: the sliding-window store re-written with what it read"


def next_text(p2: dict, order: list, fb: dict, logs: dict, probes: dict) -> str:
    """The open issues, from the numbers: did the gain switch on, which output pass works better, what the diagnostics point to. Suggestions are marked as not run."""
    paras = []
    gains = {}
    for k in order:
        vh = [r.get("val_history") or [] for r in p2[k]["runs"]]
        if all(vh):
            last = [h[-1][f"val/n256/passkey_acc"] for h in vh]
            gains[k] = last
    on = {k: sum(v > 0.5 for v in vals) for k, vals in gains.items()}
    if on:
        paras.append("<p><b>Did the second pass switch on.</b> Seeds with a validation passkey above 0.5 at N = 256 at the end of training: "
                     + "; ".join(f"{name_of(k)}: {n} of {len(p2[k]['runs'])}" for k, n in on.items()) + ".</p>")
    byp = {(k[0], k[1]): k for k in order if not k[2]}
    for seg in sorted({k[0] for k in byp}):
        if (seg, 2) in byp and (seg, 1) in byp:
            a, b = p2[byp[(seg, 2)]], p2[byp[(seg, 1)]]
            paras.append(f"<p><b>Which pass feeds the next layer (segment {seg}).</b> Passkey at N = 1,024 per seed: output of pass 2 {RG.seeds_text(RG.per_seed(a['runs'], 1024))}, output of pass 1 {RG.seeds_text(RG.per_seed(b['runs'], 1024))}; "
                         f"at N = 16,384: {RG.seeds_text(RG.per_seed(a['runs'], 16384))} against {RG.seeds_text(RG.per_seed(b['runs'], 16384))}. With {len(a['runs'])} seeds for output 2, {len(b['runs'])} for output 1 and a large seed-to-seed spread this is a comparison of tendencies, not a test.</p>")
    paras.append("<p><b>Seeds.</b> The model families so far switch their memory on in some seeds only; the means are the summary, not the best seed.</p>")
    return "".join(paras)


def name_of(k: tuple) -> str:
    return f"segment {k[0]}, output of pass {k[1]}" + (f", gain {k[2]:g}" if k[2] else "")
