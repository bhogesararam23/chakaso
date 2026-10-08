"""Turning acquired bytes into evidence — the web path into the same records as local files.

The fetcher produces bytes under a policy; this stage decodes them, extracts a document's
text and title (HTML via the narrow reader in :mod:`chakaso.retrieval.html`, plain text as
is), normalizes deterministically, and splits into evidence chunks — producing exactly the
`IngestedDocument` that local ingestion produces, so the corpus, retriever and evidence
pack do not learn a second representation for web content.

Identity comes from the validated canonical URL of the final response, never from anything
the page says about itself (ADR-0003). A page's text is data: a line that reads like an
instruction is carried into a chunk and is never executed or treated as a directive.
Decoding is strict — a byte sequence that is not valid in the declared charset is an error,
not a silent replacement that would turn a corrupted byte into evidence.
"""

from __future__ import annotations

from chakaso.evidence import SourceRecord
from chakaso.retrieval.acquire import AcquiredSource
from chakaso.retrieval.chunking import ChunkingConfig, chunk_document
from chakaso.retrieval.errors import ContentDecodingError, UnsupportedContentTypeError
from chakaso.retrieval.html import parse_html
from chakaso.retrieval.ingest import IngestedDocument
from chakaso.retrieval.normalize import normalize_document

__all__ = ["ingest_acquired"]

_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_TEXT_TYPES = frozenset({"text/plain"})


def ingest_acquired(
    acquired: AcquiredSource,
    *,
    config: ChunkingConfig | None = None,
) -> IngestedDocument:
    """Decode, parse, normalize and chunk an acquired web source into evidence.

    Raises:
        UnsupportedContentTypeError: the media type is neither HTML nor plain text. A
            binary is refused rather than decoded into mangled text.
        ContentDecodingError: the bytes are not valid in the declared charset.
    """
    text = _decode(acquired)

    if acquired.media_type in _HTML_TYPES:
        parsed = parse_html(text)
        title = parsed.title
        body = parsed.text
        fmt = "html"
    elif acquired.media_type in _TEXT_TYPES:
        title = ""
        body = text
        fmt = "text"
    else:
        message = (
            f"cannot ingest {acquired.final_url!r}: media type {acquired.media_type!r} is "
            "neither HTML nor plain text"
        )
        raise UnsupportedContentTypeError(message)

    normalized = normalize_document(body)
    record = SourceRecord.create(
        url=acquired.final_url,
        title=title,
        content=normalized,
        retrieved_at=acquired.retrieved_at,
        metadata={"format": fmt, "media_type": acquired.media_type},
    )
    chunks = chunk_document(record.source_id, normalized, config=config)
    return IngestedDocument(source=record, chunks=chunks)


def _decode(acquired: AcquiredSource) -> str:
    """Decode the acquired bytes, honouring a declared charset and stripping a UTF-8 BOM."""
    charset = acquired.charset or "utf-8-sig"
    try:
        return acquired.content.decode(charset)
    except LookupError as exc:
        message = f"unknown charset {acquired.charset!r} for {acquired.final_url!r}"
        raise ContentDecodingError(message) from exc
    except UnicodeDecodeError as exc:
        message = f"fetched bytes from {acquired.final_url!r} are not valid {charset}"
        raise ContentDecodingError(message) from exc
