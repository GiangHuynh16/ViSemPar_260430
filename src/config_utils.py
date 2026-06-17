"""Shared YAML config loading helpers."""

import copy
from pathlib import Path
from typing import Union

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_config(config_path: Union[str, Path]) -> dict:
    path = Path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if 'base' in cfg:
        base_path = path.parent / cfg['base']
        with open(base_path) as f:
            base = yaml.safe_load(f)
        cfg = deep_merge(base, cfg)
    return cfg
