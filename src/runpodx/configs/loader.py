from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypeVar

from runpodx.http.errors import ConfigError

T = TypeVar("T")


def load_toml(path: str | Path) -> dict:
    """Load a TOML file and return the raw dict."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if path.suffix != ".toml":
        raise ConfigError(f"Expected a .toml file, got: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_config(config_cls: type[T], source: str | Path | dict | T) -> T:
    """Resolve a config from a TOML path, raw dict, or existing instance.

    This is the core dispatch function used by all resource managers:
        client.pods.create("pod.toml")         -> loads TOML, validates
        client.pods.create({"name": "test"})   -> validates dict
        client.pods.create(PodCreateConfig())  -> passes through
    """
    if isinstance(source, config_cls):
        return source
    if isinstance(source, (str, Path)):
        raw = load_toml(source)
        return config_cls.model_validate(raw)
    if isinstance(source, dict):
        return config_cls.model_validate(source)
    raise ConfigError(
        f"Expected a TOML file path, dict, or {config_cls.__name__} instance, "
        f"got {type(source).__name__}"
    )
