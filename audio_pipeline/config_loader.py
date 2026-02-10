from __future__ import annotations

import copy
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import PipelineConfig, validate_config
from .profiles import PROFILES, aggressive


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import path
        raise RuntimeError("YAML config requested but PyYAML is not installed. Install with: pip install pyyaml") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Config file root must be a mapping/object")
    return payload


def _read_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except Exception as exc:  # pragma: no cover - Python <3.11
        raise RuntimeError("TOML config is unsupported on this Python version") from exc

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file root must be a mapping/object")
    return payload


def load_config_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _read_yaml(path)
    if suffix == ".toml":
        return _read_toml(path)
    raise ValueError(f"Unsupported config extension '{suffix}'. Use .yaml/.yml or .toml")


def deep_update_dataclass(obj: Any, values: Dict[str, Any], prefix: str = "") -> None:
    if not is_dataclass(obj):
        raise TypeError("deep_update_dataclass expects a dataclass instance")

    field_map = {f.name: f for f in fields(obj)}
    for key, val in values.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if key not in field_map:
            raise ValueError(f"Unknown config key: {full_key}")

        cur = getattr(obj, key)
        if is_dataclass(cur):
            if not isinstance(val, dict):
                raise ValueError(f"Config key '{full_key}' must be an object/mapping")
            deep_update_dataclass(cur, val, prefix=full_key)
            continue

        setattr(obj, key, val)


def apply_cli_overrides(config: PipelineConfig, overrides: Iterable[tuple[str, Any]]) -> None:
    for dotted_key, value in overrides:
        if value is None:
            continue
        parts = dotted_key.split(".")
        target: Any = config
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)


def resolve_config(
    *,
    profile_name: Optional[str] = None,
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Iterable[tuple[str, Any]]] = None,
) -> PipelineConfig:
    base = aggressive()
    config = copy.deepcopy(base)

    if profile_name:
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown profile '{profile_name}'. Choose from: {', '.join(sorted(PROFILES))}")
        config = copy.deepcopy(PROFILES[profile_name]())

    if config_path:
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        file_values = load_config_file(config_path)
        deep_update_dataclass(config, file_values)

    if cli_overrides:
        apply_cli_overrides(config, cli_overrides)

    validate_config(config)
    return config


def config_pretty_json(config: PipelineConfig) -> str:
    return json.dumps(config.as_dict(), indent=2, ensure_ascii=False)
