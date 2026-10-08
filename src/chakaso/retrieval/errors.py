"""Retrieval-layer errors."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = [
    "ChunkingError",
    "ContentDecodingError",
    "DocumentTooLargeError",
    "DocumentUnavailableError",
    "IngestionError",
    "RetrievalError",
    "UnsupportedDocumentError",
]


class RetrievalError(ChakasoError):
    """Base class for failures in the retrieval layer."""


class ChunkingError(RetrievalError, ValueError):
    """A document cannot be split with the given settings.

    Deriving from :class:`ValueError` as well means a caller passing a bad chunking
    setting gets the built-in type it would already expect.
    """


class IngestionError(RetrievalError, ValueError):
    """A source cannot be read into evidence.

    Every failure subclass names the offending path and the expectation, so a caller
    knows whether to fix the path, the format or the limit rather than guessing. The
    family derives from :class:`ValueError` because the inputs are untrusted.
    """


class DocumentUnavailableError(IngestionError):
    """The path is empty, does not exist, or is not a regular file.

    A directory and a nonexistent path are reported alike: ingestion reads the one
    file it was explicitly given and never crawls, so there is nothing to do but say
    the path is not an ingestable document.
    """


class UnsupportedDocumentError(IngestionError):
    """The file's format is not one ingestion reads.

    Only the declared extensions are accepted, so an unreadable binary — including the
    private documentation pack — is refused on format rather than mangled into text.
    """


class DocumentTooLargeError(IngestionError):
    """The file is larger than the ingestable limit.

    It is checked before the bytes are read, so a pathological file cannot exhaust
    memory; a document over the limit is a hard failure, never a silent truncation.
    """


class ContentDecodingError(IngestionError):
    """The bytes are not valid text in the expected encoding.

    Rejected rather than decoded with replacement characters, because a corrupted byte
    would otherwise become evidence that quietly says something the source did not.
    """
