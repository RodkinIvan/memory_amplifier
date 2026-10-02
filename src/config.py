"""YAML config + dotted-key overrides + run naming (no heavy imports; used by train.py and launch.py)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# short names for the run name; any other key falls back to its last path component
SHORT = {"model.alpha": "a", "model.window": "w", "model.segment": "seg", "model.nonlin": "nl", "model.noise": "noise", "model.gain_init": "gi", "model.proj_std": "ps", "train.loop_lr_mult": "llr", "train.passkey_weight": "pkw", "train.passkey_weight_anneal_steps": "pka", "train.curriculum_stage_steps": "cur", "model.norm_eps": "eps", "model.head_dim": "hd", "model.num_heads": "nh",
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
    return f"{v:g}" if isinstance(v, float) else str(v)


def run_name(cfg: dict, explicit_keys: list[str]) -> str:
    """model kind + every hyperparameter that was overridden for this run (sweep axes, --set) + seed."""
    # evaluation / logging settings and the wandb group do not describe the model, so they stay out of the name
    skip = ("seed", "model.kind", "experiment", "train.eval_every", "train.log_every", "train.eval_examples", "train.eval_ns")
    keys = [k for k in explicit_keys if k not in skip and not k.startswith("eval.")]
    parts = [f"{SHORT.get(k, k.split('.')[-1])}{fmt(get_nested(cfg, k))}" for k in keys]
    parts.append(f"s{cfg['seed']}")
    return "_".join([cfg["model"]["kind"], *parts])


def flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        out.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def clone(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
