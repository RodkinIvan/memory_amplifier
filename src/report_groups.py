"""Group runs by what differs from the frozen reference recipe (review H7) and describe the groups.

A report row is one *configuration*: all runs of one model kind whose stored configs have the same diff against the reference
(`config.reference_diff`, which ignores only seed, wandb grouping, logging and evaluation settings). Runs that differ in anything else
are never averaged together, and a row's label states what differs.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

NS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

STANDARD = ("model.segment", "model.nonlin", "model.noise", "model.gain_init", "train.steps")  # the settings the main grid varies


def diff_of(run: dict) -> dict:
    """The diff of a run's stored config against its reference (`run["cfg"]`, as loaded by aggregate.load_runs)."""
    return C.reference_diff(run["cfg"])


def diff_groups(runs: list[dict]) -> list[dict]:
    """[{kind, reference, diff, runs}] in a stable order; every run is in exactly one group, and the runs of a group have identical
    configs apart from the ignored settings (seed, ...)."""
    groups: dict = {}
    for r in runs:
        kind = r["cfg"]["model"]["kind"]
        groups.setdefault((kind, json.dumps(diff_of(r), sort_keys=True)), []).append(r)
    out = [{"kind": k, "reference": C.reference_name({"model": {"kind": k}}), "diff": json.loads(d), "runs": sorted(rs, key=lambda r: r["seed"])}
           for (k, d), rs in groups.items()]
    return sorted(out, key=lambda g: (g["kind"], sorted(g["diff"].items(), key=str)))


def ref_config(kind: str) -> dict:
    return C.load_reference({"model": {"kind": kind}})


def setting(group: dict, key: str):
    """Value of a setting for a group: the run's value if it differs from the reference, else the reference's."""
    return group["diff"][key][1] if key in group["diff"] else C.flatten(ref_config(group["kind"])).get(key, C.CODE_DEFAULTS.get(key))


def effective_segment(group: dict) -> int | None:
    seg = setting(group, "model.segment")
    return None if group["kind"] not in ("swa_fb", "swa_va", "swa_2pass") else (seg or setting(group, "model.window"))


def standard_key(group: dict) -> tuple | None:
    """(segment, nonlin, noise, steps, gain) for a swa_fb group that differs from the reference only in the settings of the main grid, else None."""
    if group["kind"] != "swa_fb" or not set(group["diff"]) <= set(STANDARD):
        return None
    return (effective_segment(group), setting(group, "model.nonlin"), float(setting(group, "model.noise")),
            int(setting(group, "train.steps")), float(setting(group, "model.gain_init")))


def friendly_diff(diff: dict, drop: tuple = (), compact: bool = False) -> str:
    """'no tanh, gain_init 0.1 → 0, 20000 → 50000 steps' (segment, nonlin and window are told by the label's structure part).
    `compact` shortens it for row labels ('noise 0.3', '50k steps'); the full form with both values is what the appendix shows."""
    parts = []
    for k in sorted(diff, key=lambda k: (k.split(".")[0] != "model", k.split(".")[0] != "train", k)):
        if k in drop:
            continue
        a, b = diff[k]
        name = k.split(".")[-1]
        if k == "model.nonlin":
            parts.append("no tanh" if b == "none" else f"{b} loop")
        elif k == "train.steps":
            parts.append(f"{b // 1000}k steps" if compact and b % 1000 == 0 else f"{C.fmt(b) if compact else f'{C.fmt(a)} → {C.fmt(b)}'} steps")
        elif compact and k in ("model.noise", "model.gain_init"):
            parts.append(f"{name.replace('_init', '')} {C.fmt(b)}")
        elif k == "train.curriculum_stage_steps":
            parts.append(f"curriculum {C.fmt(a)} → {C.fmt(b)} steps per stage")
        else:
            parts.append(f"{name} {C.fmt(a)} → {C.fmt(b)}")
    return ", ".join(parts)


def structure(group: dict) -> str:
    """What the model is, so that window and segment cannot be mixed up: 'window 64, segment 32, tanh loop' or 'plain SWA, window 128, 2 layers'."""
    kind, w, nl = group["kind"], setting(group, "model.window"), setting(group, "model.n_layers")
    if kind == "swa":
        return f"plain SWA, window {w}, {nl} layers"
    if kind == "swa_fb":
        return f"window {w}, segment {effective_segment(group)}, {'tanh loop' if setting(group, 'model.nonlin') == 'tanh' else 'no-tanh loop'}"
    if kind == "swa_va":
        return f"value adaptation, window {w}, segment {effective_segment(group)}"
    if kind == "swa_2pass":
        return f"two-pass layer, window {w}, segment {effective_segment(group)}, output of pass {setting(group, 'model.pass_output')}"
    if kind in ("gdn", "gdn2"):
        return "GDN-2 (learned decay)"
    return f"{kind}, alpha {setting(group, 'model.alpha'):g}"


def label(group: dict) -> str:
    """Structure plus every other difference to the reference: 'window 64, segment 8, no-tanh loop · gain_init 0.1 → 0'."""
    rest = friendly_diff(group["diff"], drop=("model.segment", "model.nonlin", "model.window", "model.n_layers", "model.alpha", "model.pass_output"))
    return structure(group) + (f" · {rest}" if rest else "")


def short_label(group: dict) -> str:
    """Compact row label for the heatmaps: 'window 64, segment 8 · no tanh · gain_init 0.1 → 0' (window and segment always spelled out)."""
    rest = friendly_diff(group["diff"], drop=("model.segment", "model.nonlin", "model.window", "model.n_layers", "model.alpha", "model.pass_output"), compact=True)
    if group["kind"] == "swa_fb":
        head = f"window {setting(group, 'model.window')}, segment {effective_segment(group)} · {'tanh' if setting(group, 'model.nonlin') == 'tanh' else 'no tanh'}"
    elif group["kind"] == "swa_va":
        head = f"value adaptation, window {setting(group, 'model.window')}, segment {effective_segment(group)}"
    elif group["kind"] == "swa_2pass":
        head = f"two-pass layer, window {setting(group, 'model.window')}, segment {effective_segment(group)}, output pass {setting(group, 'model.pass_output')}"
    else:
        head = structure(group)
    return head + (f" · {rest}" if rest else "")


def reach(group: dict) -> int | None:
    """Attention reach of a plain SWA group: layers x (window - 1) tokens (a loop model has no fixed reach)."""
    return setting(group, "model.n_layers") * (setting(group, "model.window") - 1) if group["kind"] == "swa" else None


def label_margin(labels: list[str], minimum: int = 70, px_per_char: float = 6.5, offset: int = 18) -> int:
    """Pixels a line chart needs on the right of the plot for end-of-line labels of these texts: the template draws a label 14 px right of its line's last point, about 6.5 px per
    character at its font size. A fixed margin overflows as soon as a label is longer than it was sized for."""
    return max(minimum, int(offset + px_per_char * max((len(x) for x in labels), default=0)))


def per_seed(runs: list[dict], n: int, metric: str = "passkey_acc", split: str = "test") -> list[float]:
    return [float(next(x for x in r["rows"] if x["split"] == split and x["N"] == n)[metric]) for r in runs]


def seeds_text(vals: list[float]) -> str:
    return " / ".join(f"{v:.2f}" for v in vals)


def configuration_table(groups: list[dict], ns: tuple = (256, 1024, 4096, 16384)) -> str:
    """HTML table with one row per configuration: label, reference, runs, size, training time, per-seed passkey at the given N."""
    head = ("<th>configuration (what differs from the reference recipe)</th><th>runs</th><th>params</th><th>train time</th>"
            + "".join(f"<th>passkey, N = {n:,}<br><span class='hint'>per seed</span></th>" for n in ns) + "<th>AR valid, N = 256<br><span class='hint'>mean</span></th>")
    rows = []
    for g in groups:
        rs = g["runs"]
        names = ", ".join(r["name"] for r in rs)
        t = [r["train_time_s"] for r in rs if r.get("train_time_s")]
        rows.append(
            f"<tr><td title='{html.escape(names)}'>{html.escape(label(g))}<br><span class='hint'>reference {g['reference']}; seeds {', '.join(str(r['seed']) for r in rs)}</span></td>"
            f"<td>{len(rs)}</td><td>{rs[0].get('n_params', 0) // 1000}k</td><td>{'–' if not t else f'{np.mean(t) / 3600:.1f} h'}</td>"
            + "".join(f"<td>{seeds_text(per_seed(rs, n))}</td>" for n in ns)
            + f"<td>{np.mean(per_seed(rs, 256, 'ar_valid_acc')):.2f}</td></tr>")
    return f"<div class='tbl wide'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _nn(x):
    return None if x is None or x != x else float(x)


def cells_for_row(runs: list[dict], split: str = "test", ns: list[int] | None = None) -> dict:
    """{str(N): {passkey, valid, stale, expired (means over seeds), counts, per: {metric: [per seed]}}} for one row."""
    keys = {"passkey": "passkey_acc", "valid": "ar_valid_acc", "stale": "ar_expired_stale_acc", "expired": "ar_expired_acc"}
    out = {}
    for n in ns or NS:
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
