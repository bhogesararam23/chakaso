"""Typed, explicit, versioned configuration.

Configuration is TOML parsed with the standard library (ADR-0006) and loaded
explicitly: no file is discovered by scanning the filesystem, and no environment
variable is silently substituted for a file value.

    from chakaso.config import load_config

    loaded = load_config("configs/default.toml", overrides=["configs/local.toml"])
    loaded.config.model.adapter
    loaded.provenance["model.adapter"]

The schema is deliberately small. A field exists here when something that already
exists requires it, not when the roadmap anticipates it.
"""

from __future__ import annotations

from chakaso.config.errors import (
    ConfigError,
    ConfigFileError,
    ConfigParseError,
    ConfigValidationError,
)
from chakaso.config.loader import BUILT_IN_SOURCE, LoadedConfig, ResolvedValue, load_config
from chakaso.config.schema import SPECS, Config, FieldSpec, ModelConfig

__all__ = [
    "BUILT_IN_SOURCE",
    "SPECS",
    "Config",
    "ConfigError",
    "ConfigFileError",
    "ConfigParseError",
    "ConfigValidationError",
    "FieldSpec",
    "LoadedConfig",
    "ModelConfig",
    "ResolvedValue",
    "load_config",
]
