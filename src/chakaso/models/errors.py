"""Model-layer errors."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = ["ModelError", "UnknownModelError", "UnsupportedCapabilityError"]


class ModelError(ChakasoError):
    """Base class for failures at the language-model boundary."""


class UnknownModelError(ModelError, LookupError):
    """No adapter is registered under the requested name.

    Deriving from :class:`LookupError` as well means a caller resolving a name from
    configuration can catch the built-in type it would already expect.
    """


class UnsupportedCapabilityError(ModelError):
    """The selected model does not support what was asked of it.

    Raised at the boundary, naming the model and the capability, rather than deep
    inside a call stack where the cause would be unrecognisable.
    """
