"""Definitions of the capacity grid (two-pass layer, passkey capacity as a function of the segment size and the window): which cells and jobs it has, the names of their
runs, the keys of their chains in results/capacity_grid.json, and what a finished chain says about the capacity of its cell. Used by scripts/capacity_grid.py (runs the jobs)
and src/report_capacity_grid.py (the page); no GPU, no training in here.
"""
from __future__ import annotations

import json
from pathlib import Path

import config as C

ROOT = C.ROOT
CONFIG = ROOT / "configs" / "swa_2pass.yaml"
RESULTS = ROOT / "results" / "capacity_grid.json"

SEGMENTS = (8, 16, 32)
WINDOWS = (32, 64, 128)
SEEDS = (0, 1, 2)             # base seeds: one base model per cell and seed, trained from scratch, and one chain from each
FT_SEED_OFFSET = 10           # fine-tuning seed of a chain = base seed + 10, so the stage names differ from the earlier chains of the best (segment 32, window 64) checkpoint, which used seeds 0 ... 5
FIRST_P, MAX_LEN = 8, 64      # the chain starts at 8-token passkeys (the bases are trained on 4) and stops after P = 64 passes ("at least 64")
THRESHOLD = 0.9               # validation exact match of the passkey at N = 256 (the chain's stop rule, scripts/passkey_capacity.py)
RECIPE = {"steps": 50000, "lr": 1e-3, "warmup": 500, "curriculum": 2000}   # of every stage: the budget of the original recipe, as for the earlier two-pass chains
EVAL_NS = "[32,64,128,256]"   # the bases are only evaluated where the chain evaluates (the cell is about the passkey length, not the stream length: no N = 16,384 passes)


def cells() -> list[tuple[int, int]]:
    """(segment, window) with segment <= window, windows major."""
    return [(s, w) for w in WINDOWS for s in SEGMENTS if s <= w]


def jobs(seeds=SEEDS) -> list[tuple[int, int, int]]:
    """(segment, window, base seed): first every cell with seed 0, then seed 1, ... so the heatmap fills in completely before it is refined; within a seed the slow cells (small
    segment, large window) first, which keeps the longest jobs from starting last."""
    return [(s, w, g) for g in seeds for (s, w) in sorted(cells(), key=lambda c: (c[0], -c[1]))]


def yaml_window() -> int:
    return C.load_config(CONFIG, [])[0]["model"]["window"]


def base_overrides(segment: int, window: int, seed: int) -> list[str]:
    """Overrides of configs/swa_2pass.yaml for the base of a cell. The window is only written when it differs from the file's, so the base of the (32, 64) cell has the run name
    of the earlier E4 runs (swa_2pass_seg32_po1_s<seed>) and is reused, not trained again."""
    return ["experiment=swa_2pass_grid", f"seed={seed}", f"model.segment={segment}", "model.pass_output=1", *([f"model.window={window}"] if window != yaml_window() else []),
            f"train.eval_ns={EVAL_NS}", f"eval.ns={EVAL_NS}"]


def base_name(segment: int, window: int, seed: int) -> str:
    cfg, explicit = C.load_config(CONFIG, base_overrides(segment, window, seed))
    return C.run_name(cfg, explicit)


def base_weights(segment: int, window: int, seed: int) -> str:
    return f"runs/{base_name(segment, window, seed)}/model.pt"


def tag(segment: int, window: int, seed: int) -> str:
    return f"grid_s{segment}_w{window}_b{seed}"


def chain_key(segment: int, window: int, seed: int) -> str:
    """Key of the chain in results/capacity_grid.json (scripts/passkey_capacity.py: `stages_<model>_<tag>`)."""
    return f"stages_swa_2pass_{tag(segment, window, seed)}"


def capacity_of(stages: list[dict], base_exact: float | None, max_len: int = MAX_LEN, threshold: float = THRESHOLD) -> dict:
    """What a chain says about its cell. `stages`: the chain's entries of capacity.json (passkey_len, validation exact at N = 256). The capacity is the largest P whose stage passed
    with every smaller stage passed too. {tokens: that P; when the first stage (P = 8) failed: 4 if the base itself reaches the threshold on 4-token passkeys, else 0 (below 4);
    censored: the last stage passed and P = max_len, so the capacity is at least that; done: the chain stopped (a stage failed or max_len passed), nothing more will be added}."""
    by_p = sorted(stages, key=lambda e: e["passkey_len"])
    tokens = 4 if (base_exact is not None and base_exact >= threshold) else 0
    stopped = False
    for e in by_p:
        if e["validation"]["exact"]["256"] >= threshold:
            tokens = e["passkey_len"]
        else:
            stopped = True
            break
    censored = bool(by_p) and not stopped and tokens >= max_len
    return {"tokens": tokens, "censored": censored, "done": stopped or censored, "stages": [e["passkey_len"] for e in by_p]}


def load_results(path: Path = RESULTS) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def raw_result(name: str, root: Path = ROOT) -> dict | None:
    f = root / "results" / "raw" / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def base_exact(result: dict | None) -> float | None:
    """Validation exact match of the passkey at N = 256 of a base run's result file (None: no result yet)."""
    if not result:
        return None
    return next((r["passkey_acc"] for r in result["rows"] if r["split"] == "validation" and r["N"] == 256), None)


def job_state(segment: int, window: int, seed: int, grid: dict | None = None, root: Path = ROOT) -> dict:
    """{base: run name, base_exact: validation exact at N = 256 or None while the base is not trained, capacity: capacity_of(...) (tokens None before the base is done)}."""
    grid = load_results() if grid is None else grid
    name = base_name(segment, window, seed)
    be = base_exact(raw_result(name, root))
    stages = grid.get(chain_key(segment, window, seed), [])
    cap = capacity_of(stages, be) if be is not None else {"tokens": None, "censored": False, "done": False, "stages": []}
    return {"segment": segment, "window": window, "seed": seed, "base": name, "base_exact": be, "capacity": cap, "stages": stages}
