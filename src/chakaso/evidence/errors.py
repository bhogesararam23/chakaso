"""Evidence-layer errors."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = ["EvidenceError", "SourceRecordError", "UrlError"]


class EvidenceError(ChakasoError):
    """Base class for failures in the evidence layer."""


class UrlError(EvidenceError, ValueError):
    """A URL cannot be used as a source.

    Deriving from :class:`ValueError` as well means untrusted input — a URL found
    in a page or a model's output — can be validated with the built-in type.
    """


class SourceRecordError(EvidenceError, ValueError):
    """A source or chunk record would be internally inconsistent."""
