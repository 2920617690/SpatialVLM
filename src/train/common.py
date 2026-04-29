from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    parent = config.get("inherits")
    if not parent:
        return config

    inherited_path = Path(parent)
    if not inherited_path.is_absolute():
        inherited_path = path.parent.parent / inherited_path
    base = load_config(inherited_path)
    return _merge_dicts(base, {key: value for key, value in config.items() if key != "inherits"})


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_output_dir(config: Dict[str, Any], stage_name: str) -> Path:
    root = Path(config["project"]["output_root"])
    output_dir = root / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir
