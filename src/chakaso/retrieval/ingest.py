"""Reading explicitly supplied documents into evidence records.

This is the first stage that turns content into the structures the evidence layer
already defines. For one source it runs the whole path — identify, read, normalize,
split into sections and chunks — and hands back a :class:`SourceRecord` with its
:class:`~chakaso.evidence.EvidenceChunk` records, the exact shape
:meth:`ConversationManager.send` consumes. No parallel representation is invented: an
ingested document is the same ``SourceRecord`` a caller could have built by hand.

Nothing here crawls. A caller names one file, or passes one block of text, and that is
the only thing read. There is no directory scan, no walk of a home directory, and no
reading a file because it happens to exist — a source is never read merely because it
is present. The filesystem-safety rules below exist to keep it that way, and each one is
covered by a test that deliberately tries to break it.

The safety rules are deliberately about *what can be read*, not a permission model for
a hostile caller. The API takes an explicit path; there is no base-directory to escape,
so path traversal is not a boundary here — the caller already chose the path. What is
bounded is the ways a document can be unusable or dangerous to load: it might not be a
regular file, might be a format that would be mangled into text, might be large enough
to exhaust memory, or might not be decodable at all. Each of those is rejected with a
specific error rather than read anyway.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from chakaso.evidence import EvidenceChunk, SourceRecord
from chakaso.retrieval.chunking import ChunkingConfig, chunk_document
from chakaso.retrieval.errors import (
    ContentDecodingError,
    DocumentTooLargeError,
    DocumentUnavailableError,
    IngestionError,
    UnsupportedDocumentError,
)
from chakaso.retrieval.normalize import normalize_document

__all__ = [
    "DEFAULT_MAX_BYTES",
    "SUPPORTED_FORMATS",
    "IngestedDocument",
    "ingest_file",
    "ingest_text",
]

#: Extensions ingestion will read, mapped to a format label recorded as metadata.
#: Plain text and Markdown only: they are genuinely useful, need no parsing
#: dependency, and their structure is what the chunker understands. Anything else —
#: including the author's private ``.docx`` pack — is refused on format rather than
#: decoded into text and mangled.
SUPPORTED_FORMATS: Final[dict[str, str]] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
}

#: Refuse to read a document larger than this, checked against the file's size before
#: its bytes are loaded. It is a parameter of every call, not a hidden constant at the
#: call site, so a caller that expects larger inputs passes a larger bound.
DEFAULT_MAX_BYTES: Final[int] = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    """One source and the evidence chunks read from it.

    The chunks are empty when the document has no content — a source with nothing in it
    is a real, valid record that simply cannot support any claim, not an error.
    """

    source: SourceRecord
    chunks: tuple[EvidenceChunk, ...]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def ingest_file(
    path: str | Path,
    *,
    title: str | None = None,
    config: ChunkingConfig | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    now: Callable[[], datetime] | None = None,
) -> IngestedDocument:
    """Read one explicitly named local document into a source record and its chunks.

    The file is identified by its resolved path (a ``file`` reference, ADR-0011), read
    as UTF-8, normalized deterministically, and split into evidence chunks. ``title``
    defaults to the file name — a description of the source, not a claim about its
    content. ``now`` is the retrieval clock, injectable so tests do not depend on
    wall-clock time.

    Raises:
        DocumentUnavailableError: the path is empty, absent, or not a regular file.
        UnsupportedDocumentError: the extension is not one ingestion reads.
        DocumentTooLargeError: the file is bigger than ``max_bytes``.
        ContentDecodingError: the bytes are not valid UTF-8 text.
    """
    candidate = _validated_path(path)

    suffix = candidate.suffix.lower()
    fmt = SUPPORTED_FORMATS.get(suffix)
    if fmt is None:
        allowed = ", ".join(sorted(SUPPORTED_FORMATS))
        message = (
            f"unsupported document format {suffix!r} for {candidate}; ingestion reads {allowed}"
        )
        raise UnsupportedDocumentError(message)

    raw_text = _read_text(candidate, max_bytes=max_bytes)
    normalized = normalize_document(raw_text)

    record = SourceRecord.create_file(
        path=candidate,
        title=title if title is not None else candidate.name,
        content=normalized,
        retrieved_at=_resolved_time(now),
        raw_text_path=str(candidate.resolve()),
        metadata={"format": fmt},
    )
    chunks = chunk_document(record.source_id, normalized, config=config)
    return IngestedDocument(source=record, chunks=chunks)


def ingest_text(
    text: str,
    *,
    label: str | None = None,
    title: str = "",
    config: ChunkingConfig | None = None,
    now: Callable[[], datetime] | None = None,
) -> IngestedDocument:
    """Ingest a block of supplied text as a source with no persistent location.

    Identity is the content (a ``text`` reference, ADR-0011); an optional ``label``
    distinguishes passages that would otherwise deduplicate. This is the "retrieve this
    from the text I am handing you" path that makes the pipeline testable without any
    filesystem or network at all.
    """
    normalized = normalize_document(text)
    record = SourceRecord.create_text(
        content=normalized,
        retrieved_at=_resolved_time(now),
        title=title,
        label=label,
    )
    chunks = chunk_document(record.source_id, normalized, config=config)
    return IngestedDocument(source=record, chunks=chunks)


def _validated_path(path: str | Path) -> Path:
    """Return ``path`` as a candidate file, or raise before anything is read.

    Existence and "is a regular file" are checked first so that a directory, an absent
    path or a device is rejected as unavailable without consulting format or size.
    """
    if isinstance(path, str) and not path.strip():
        message = "ingestion needs an explicit non-empty path; it was given none"
        raise DocumentUnavailableError(message)

    candidate = Path(path).expanduser()
    try:
        is_file = candidate.is_file()
        exists = candidate.exists()
    except OSError as exc:  # a path that cannot even be stat'd is unavailable
        message = f"document path could not be inspected: {candidate}: {exc}"
        raise DocumentUnavailableError(message) from exc

    if not exists:
        message = f"document does not exist: {candidate}"
        raise DocumentUnavailableError(message)
    if not is_file:
        message = f"path is not a regular file: {candidate}"
        raise DocumentUnavailableError(message)
    return candidate


def _read_text(path: Path, *, max_bytes: int) -> str:
    """Read ``path`` as UTF-8 text, bounding its size before the bytes are loaded."""
    size = path.stat().st_size
    if size > max_bytes:
        message = f"document is {size} bytes, over the {max_bytes}-byte ingestable limit: {path}"
        raise DocumentTooLargeError(message)

    try:
        # utf-8-sig strips a leading byte-order mark if one is present; a document that
        # is not valid UTF-8 raises rather than being decoded with replacements, which
        # would turn a corrupted byte into evidence that says something the source did not.
        return path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        message = f"document is not valid UTF-8 text: {path}"
        raise ContentDecodingError(message) from exc
    except OSError as exc:
        message = f"document could not be read: {path}: {exc}"
        raise IngestionError(message) from exc


def _resolved_time(now: Callable[[], datetime] | None) -> datetime:
    moment = (now if now is not None else _utcnow)()
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        message = "the ingestion clock must return a timezone-aware datetime"
        raise IngestionError(message)
    return moment
