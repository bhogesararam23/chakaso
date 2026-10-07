"""Retrieval-layer errors."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = ["ChunkingError", "RetrievalError"]


class RetrievalError(ChakasoError):
    """Base class for failures in the retrieval layer."""


class ChunkingError(RetrievalError, ValueError):
    """A document cannot be split with the given settings.

    Deriving from :class:`ValueError` as well means a caller passing a bad chunking
    setting gets the built-in type it would already expect.
    """
