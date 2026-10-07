"""Adapter selection: the one place configuration becomes a model.

ADR-0002 requires that application code never imports a concrete implementation
and that configuration selects one in a single place. This module is that place.

A registry maps an adapter name to a factory. The name is the same string that
appears in ``configs/default.toml`` under ``model.adapter``, and it is validated
against the same pattern the configuration schema uses, so a name that cannot be
configured cannot be registered either.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from chakaso.config import Config, ModelConfig
from chakaso.config.schema import ADAPTER_NAME_PATTERN
from chakaso.models.base import LanguageModel
from chakaso.models.deterministic import DETERMINISTIC_MODEL_ID, DeterministicModel
from chakaso.models.errors import UnknownModelError

__all__ = ["ModelFactory", "ModelRegistry", "create_model", "default_registry"]

#: A factory takes the model section of the resolved configuration and returns a
#: ready model. It takes configuration rather than nothing so that adding an
#: adapter-specific setting later does not change the signature of every factory.
ModelFactory = Callable[[ModelConfig], LanguageModel]


class ModelRegistry:
    """Maps adapter names to factories, and creates models from configuration."""

    def __init__(self, factories: Mapping[str, ModelFactory] | None = None) -> None:
        self._factories: dict[str, ModelFactory] = {}
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    def register(self, name: str, factory: ModelFactory) -> None:
        """Register ``factory`` under ``name``.

        Raises:
            ValueError: ``name`` is not a valid adapter name, or is already taken.
                Re-registering is rejected rather than allowed to shadow: a silent
                overwrite is how two tests end up disagreeing about what a name
                means.
        """
        if re.fullmatch(ADAPTER_NAME_PATTERN, name) is None:
            message = (
                f"{name!r} is not a valid adapter name. Expected a lowercase name "
                f"matching {ADAPTER_NAME_PATTERN!r}, because the same string is written "
                "in a configuration file."
            )
            raise ValueError(message)
        if name in self._factories:
            message = f"adapter {name!r} is already registered"
            raise ValueError(message)
        self._factories[name] = factory

    def names(self) -> tuple[str, ...]:
        """Registered adapter names, sorted."""
        return tuple(sorted(self._factories))

    def create(self, config: ModelConfig) -> LanguageModel:
        """Build the model named by ``config.adapter``.

        Raises:
            UnknownModelError: nothing is registered under that name. The message
                lists what is, since the usual cause is a typo in a config file.
        """
        try:
            factory = self._factories[config.adapter]
        except KeyError as exc:
            known = ", ".join(self.names()) or "none"
            message = (
                f"no model adapter is registered as {config.adapter!r}. "
                f"Registered adapters: {known}."
            )
            raise UnknownModelError(message) from exc
        return factory(config)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)


def _build_deterministic(_config: ModelConfig) -> LanguageModel:
    # The double has no configurable behaviour, so it ignores the model section.
    # The parameter is kept for signature compatibility with adapter-specific
    # settings that will exist.
    return DeterministicModel()


def default_registry() -> ModelRegistry:
    """A registry containing the adapters built into this package.

    A fresh registry each time, so that a test which registers something cannot
    affect another test.
    """
    return ModelRegistry({DETERMINISTIC_MODEL_ID: _build_deterministic})


def create_model(config: Config, *, registry: ModelRegistry | None = None) -> LanguageModel:
    """Create the language model selected by ``config``.

    This is the single composition point ADR-0002 requires. Pass ``registry`` to
    substitute a different set of adapters; omit it to use the built-in ones.
    """
    target = registry if registry is not None else default_registry()
    return target.create(config.model)
