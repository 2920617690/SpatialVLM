from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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
    return _merge_dicts(base, {k: v for k, v in config.items() if k != "inherits"})


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
