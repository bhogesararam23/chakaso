"""Model-layer errors."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = [
    "CorruptedReleaseError",
    "IncompatibleReleaseError",
    "ModelError",
    "UnknownModelError",
    "UnsupportedCapabilityError",
]


class ModelError(ChakasoError):
    """Base class for failures at the language-model boundary."""


class CorruptedReleaseError(ModelError):
    """A release document is unreadable, incomplete, or does not match its own checksum."""


class IncompatibleReleaseError(ModelError):
    """The release speaks a schema this code does not implement, or names an artifact it cannot honour.

    A missing statement (no licence, no limitations, no checkpoint) and a checkpoint that hashes to
    different bytes than the contract declares are the same class of failure: the directory is not the
    release it claims to be, so loading it must stop rather than proceed on assumption.
    """


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
