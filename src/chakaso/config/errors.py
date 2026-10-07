"""Configuration errors.

Every configuration failure names the file and the field involved. A configuration
system whose errors say "invalid config" costs more time than it saves, because the
failure surfaces somewhere else entirely.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "ConfigFileError",
    "ConfigParseError",
    "ConfigValidationError",
]


class ConfigError(Exception):
    """Base class for every configuration problem.

    Callers that want to handle any configuration failure catch this. Callers that
    want to distinguish a missing file from an invalid value catch the subclasses.
    """


class ConfigFileError(ConfigError):
    """A configuration file could not be read."""


class ConfigParseError(ConfigError):
    """A configuration file is not valid TOML."""


class ConfigValidationError(ConfigError):
    """A configuration value is missing, mistyped, or outside its permitted range."""
