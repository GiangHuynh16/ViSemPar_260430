"""Shared YAML config loading helpers."""

import copy
import ast
from pathlib import Path
from typing import Union

try:
    import yaml
except ImportError:
    yaml = None


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _parse_scalar(value: str):
    value = value.strip()
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [item.strip() for item in value[1:-1].split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    return value


def _yaml_lines(text: str) -> list[tuple[int, str]]:
    rows = []
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        rows.append((len(line) - len(line.lstrip(" ")), line.strip()))
    return rows


def _parse_block(rows: list[tuple[int, str]], index: int, indent: int):
    if index >= len(rows):
        return {}, index
    is_list = rows[index][1].startswith("- ")
    if is_list:
        out = []
        while index < len(rows):
            row_indent, text = rows[index]
            if row_indent != indent or not text.startswith("- "):
                break
            value = text[2:].strip()
            index += 1
            if value:
                out.append(_parse_scalar(value))
            else:
                child, index = _parse_block(rows, index, rows[index][0])
                out.append(child)
        return out, index

    out = {}
    while index < len(rows):
        row_indent, text = rows[index]
        if row_indent != indent or text.startswith("- "):
            break
        if ":" not in text:
            raise ValueError(f"Unsupported YAML line: {text!r}")
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        index += 1
        if value:
            out[key] = _parse_scalar(value)
        elif index < len(rows) and rows[index][0] > row_indent:
            child, index = _parse_block(rows, index, rows[index][0])
            out[key] = child
        else:
            out[key] = None
    return out, index


def _mini_safe_load(stream) -> dict:
    text = stream.read() if hasattr(stream, "read") else str(stream)
    rows = _yaml_lines(text)
    if not rows:
        return {}
    parsed, index = _parse_block(rows, 0, rows[0][0])
    if index != len(rows):
        raise ValueError("Unsupported YAML structure")
    return parsed


def _format_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _mini_safe_dump(data, sort_keys: bool = False, indent: int = 0) -> str:
    lines = []
    items = data.items()
    if sort_keys:
        items = sorted(items)
    for key, value in items:
        prefix = " " * indent + f"{key}:"
        if isinstance(value, dict):
            lines.append(prefix)
            lines.append(_mini_safe_dump(value, sort_keys=sort_keys, indent=indent + 2))
        elif isinstance(value, list):
            lines.append(prefix)
            for item in value:
                if isinstance(item, dict):
                    lines.append(" " * (indent + 2) + "-")
                    lines.append(_mini_safe_dump(item, sort_keys=sort_keys, indent=indent + 4))
                else:
                    lines.append(" " * (indent + 2) + f"- {_format_scalar(item)}")
        else:
            lines.append(f"{prefix} {_format_scalar(value)}")
    return "\n".join(lines)


class _MiniYaml:
    @staticmethod
    def safe_load(stream):
        return _mini_safe_load(stream)

    @staticmethod
    def safe_dump(data, sort_keys: bool = False):
        return _mini_safe_dump(data, sort_keys=sort_keys) + "\n"


if yaml is None:
    yaml = _MiniYaml()


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
