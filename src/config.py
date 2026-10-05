"""YAML config + dotted-key overrides + run naming (no heavy imports; used by train.py and launch.py)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# short names for the run name; any other key falls back to its last path component
SHORT = {"model.alpha": "a", "model.window": "w", "model.segment": "seg", "model.nonlin": "nl", "model.noise": "noise", "model.kv_cache": "kv", "train.cuda_graph": "cg", "model.gain_init": "gi", "model.proj_std": "ps", "train.loop_lr_mult": "llr", "train.passkey_weight": "pkw", "train.passkey_weight_anneal_steps": "pka", "train.curriculum_stage_steps": "cur", "model.norm_eps": "eps", "model.head_dim": "hd", "model.num_heads": "nh",
         "model.d_model": "d", "model.n_layers": "L", "model.expand_v": "ev", "model.allow_neg_eigval": "neg",
         "train.lr": "lr",
         "train.steps": "steps", "train.batch_size": "bs", "seed": "s"}


def parse_value(text: str):
    """int -> float -> YAML (so `1e-5` becomes a float, which plain YAML 1.1 would read as a string)."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def set_nested(cfg: dict, dotted: str, value) -> None:
    *path, last = dotted.split(".")
    d = cfg
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def get_nested(cfg: dict, dotted: str):
    for k in dotted.split("."):
        cfg = cfg[k]
    return cfg


def load_config(path: str | Path, overrides: list[str]) -> tuple[dict, list[str]]:
    """Returns (config, explicitly overridden dotted keys in order)."""
    cfg = yaml.safe_load(Path(path).read_text())
    keys = []
    for o in overrides:
        k, _, v = o.partition("=")
        if not v:
            raise ValueError(f"override must look like key=value, got {o!r}")
        set_nested(cfg, k, parse_value(v))
        if k not in keys:
            keys.append(k)
    return cfg, keys


def fmt(v) -> str:
    if isinstance(v, (list, tuple)):  # [32, 64] -> "32-64": no brackets or spaces in run names and file names
        return "-".join(fmt(x) for x in v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def name_keys(explicit_keys: list[str]) -> list[str]:
    """The overridden keys that describe a run. Evaluation / logging settings, the wandb group and the seed do not, so they stay
    out of run names and of the labels in the summary tables."""
    return [k for k in explicit_keys if k != "model.kind" and not is_ignored(k)]


def run_name(cfg: dict, explicit_keys: list[str]) -> str:
    """model kind + every hyperparameter that was overridden for this run (sweep axes, --set) + seed."""
    keys = name_keys(explicit_keys)
    parts = [f"{SHORT.get(k, k.split('.')[-1])}{fmt(get_nested(cfg, k))}" for k in keys]
    parts.append(f"s{cfg['seed']}")
    return "_".join([cfg["model"]["kind"], *parts])


def flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        out.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


# ---- what differs from the reference recipe (review H7) --------------------------------------------------------------------------
# Every run is labelled by the diff of its stored config against a frozen reference file of its model kind. Run names list only the
# overrides of one launch (relative to whatever the YAML said then); the diff does not depend on that.
REFERENCES = {"gdn_fixed": "gdn_fixed_v1", "gdn2": "gdn2_v1", "swa": "swa_v1", "swa_fb": "swa_fb_v1"}  # model.kind -> configs/ref/<name>.yaml (never edited)

# Settings that do not change what is trained or how it is scored: seed, wandb grouping, logging, and the evaluation set / batch.
# Everything else counts, also implementation switches such as model.kv_cache (equivalent results, but a different config).
IGNORED_KEYS = ("seed", "experiment", "project", "train.log_every", "train.eval_every", "train.eval_examples",
                "train.eval_examples_by_n", "train.eval_ns")
IGNORED_PREFIXES = ("eval.",)

# What the code uses for a key a stored config does not have (configs written before the key existed): `.get(key, default)` in
# models.py / train.py. Keep in sync with those calls.
CODE_DEFAULTS = {"model.segment": 0, "model.nonlin": "tanh", "model.noise": 0.0, "model.gain_init": 0.0, "model.proj_std": 0.02,
                 "model.kv_cache": False, "train.cuda_graph": False, "train.precision": "fp32", "train.loop_lr_mult": 1.0, "train.passkey_weight": 1.0,
                 "train.passkey_weight_anneal_steps": 0, "train.curriculum_stage_steps": 0}


def is_ignored(key: str) -> bool:
    return key in IGNORED_KEYS or key.startswith(IGNORED_PREFIXES)


def reference_name(cfg: dict) -> str:
    kind = cfg["model"]["kind"]
    if kind not in REFERENCES:
        raise KeyError(f"no frozen reference for model kind {kind!r}: add configs/ref/{kind}_v1.yaml and list it in config.REFERENCES")
    return REFERENCES[kind]


def load_reference(cfg: dict) -> dict:
    return yaml.safe_load((ROOT / "configs" / "ref" / f"{reference_name(cfg)}.yaml").read_text())


def reference_diff(cfg: dict, ref: dict | None = None) -> dict:
    """{dotted key: [reference value, this run's value]} for every setting of `cfg` that differs from its frozen reference,
    apart from IGNORED_KEYS. A key a config lacks counts as the code default, so old configs compare correctly."""
    a = {**CODE_DEFAULTS, **flatten(ref if ref is not None else load_reference(cfg))}
    b = {**CODE_DEFAULTS, **flatten(cfg)}
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if not is_ignored(k) and a.get(k) != b.get(k)}


def describe_diff(diff: dict, none: str = "reference recipe") -> str:
    """'segment 0 → 32, gain_init 0.1 → 0': model keys first, then train keys, then the rest."""
    if not diff:
        return none
    order = sorted(diff, key=lambda k: (k.split(".")[0] != "model", k.split(".")[0] != "train", k))
    return ", ".join(f"{k.split('.')[-1]} {fmt(diff[k][0])} → {fmt(diff[k][1])}" for k in order)


def run_description(cfg: dict) -> str:
    """Text for the wandb run notes: the reference, what differs from it, and the facts that are easy to misread."""
    tc = cfg["train"]
    stage = int(tc.get("curriculum_stage_steps", 0))
    curriculum = (f"curriculum: one stage per {stage} steps (train N added in order {tc['train_ns']}), unchanged by the number of steps"
                  if stage > 0 else f"no curriculum (all of {tc['train_ns']} from the start)")
    return (f"{cfg['model']['kind']}, reference {reference_name(cfg)}; differs in: {describe_diff(reference_diff(cfg))}. "
            f"Schedule: {tc['warmup_steps']} warmup steps, then cosine decay to {tc['min_lr_ratio']:g} x lr at the LAST step ({tc['steps']}), "
            f"so a run with more steps is not a shorter run continued. {curriculum}. Seed {cfg['seed']}.")


def stamp_diff(result: dict) -> dict:
    """Add `reference` and `diff` (from the stored config) to a results dict; returns it."""
    result["reference"] = reference_name(result["config"])
    result["diff"] = reference_diff(result["config"])
    return result


def clone(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def stamp_files(paths: list[str]) -> int:
    """Write `reference` and `diff` into results json files (finished runs from before the field existed). Returns how many changed."""
    import json
    changed = 0
    for p in map(Path, paths):
        r = json.loads(p.read_text())
        before = (r.get("reference"), r.get("diff"))
        stamp_diff(r)
        if (r["reference"], r["diff"]) != before:
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(r, indent=1))
            tmp.rename(p)
            changed += 1
    return changed


if __name__ == "__main__":  # python src/config.py results/raw/*.json  -> stamp the diff against the frozen reference into each file
    import sys
    print(f"{stamp_files(sys.argv[1:])} of {len(sys.argv) - 1} files changed")
