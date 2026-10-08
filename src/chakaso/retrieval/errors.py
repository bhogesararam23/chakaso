"""Retrieval-layer errors."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = [
    "AcquisitionError",
    "BlockedDestinationError",
    "ChunkingError",
    "ContentDecodingError",
    "CorpusError",
    "DocumentTooLargeError",
    "DocumentUnavailableError",
    "FetchTimeoutError",
    "IngestionError",
    "InvalidSchemeError",
    "NetworkFailureError",
    "ResponseTooLargeError",
    "RetrievalError",
    "UnsupportedContentTypeError",
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


class CorpusError(RetrievalError, ValueError):
    """A chunk is added to a corpus under a source that does not own it.

    Membership is what keeps provenance intact: a chunk that claims a source the corpus
    has never seen would let retrieval hand back evidence with nothing behind it.
    """


class AcquisitionError(RetrievalError, ValueError):
    """Base class for a web fetch that was refused or failed.

    The subclasses name the reason a fetch did not produce content — scheme, destination,
    limit, timeout, content type, decoding or transport — so a caller can tell a blocked
    destination from a network blip without parsing a message, and none of them collapse
    into a generic "something went wrong".
    """


class InvalidSchemeError(AcquisitionError):
    """The URL's scheme is not one the fetch policy allows."""


class BlockedDestinationError(AcquisitionError):
    """The URL points at a destination the network policy refuses.

    Loopback, private, link-local and metadata addresses are blocked. This is a
    best-effort check on the address literal; it is not a complete SSRF defence and does
    not claim to be (ADR-0012).
    """


class ResponseTooLargeError(AcquisitionError):
    """The response body exceeded the configured maximum and was not buffered."""


class FetchTimeoutError(AcquisitionError):
    """The request exceeded the configured timeout."""


class UnsupportedContentTypeError(AcquisitionError):
    """The response's content type is not one the pipeline can turn into text."""


class NetworkFailureError(AcquisitionError):
    """The transport failed for a reason that is not a policy decision.

    A DNS failure, a refused connection or a mid-body break is reported here, distinct
    from a deliberate refusal, so an operator can tell "we chose to block this" from "the
    network did not deliver."
    """
