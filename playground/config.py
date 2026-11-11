"""Load an experiment config: defaults, then the experiment file, then command-line overrides."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))   # keep downloaded models in this repo
OPEN_SECTIONS = (["generation"], ["backends"])           # passed on as keyword arguments, so new keys are legal


def _merge(base, extra):
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _read(path):
    return yaml.safe_load(Path(path).read_text())


def apply_overrides(cfg, overrides):
    for item in overrides:                     # e.g. internet.kernel=true
        dotted_key, value = item.split("=", 1)
        *parents, last = dotted_key.split(".")
        node = cfg
        for key in parents:
            node = node[key]
        if last not in node and parents[:1] not in OPEN_SECTIONS:
            raise KeyError(f"Unknown setting: {dotted_key}")
        node[last] = yaml.safe_load(value)
    return cfg


def load_config(path, overrides=()):
    cfg = apply_overrides(_merge(_read(ROOT / "configs/default.yaml"), _read(path)), overrides)
    cfg["model_spec"] = _read(ROOT / "playground/models.yaml")[cfg["model"]]
    return cfg


def load_run_config(run_dir):
    return _read(Path(run_dir) / "config.yaml")
