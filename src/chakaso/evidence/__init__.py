"""Evidence: source records, chunks, packs and citation resolution.

The retrieval layer owns source identity (ADR-0003), derives identifiers from
content rather than assigning them at random (ADR-0007), and supplies a bounded set
of evidence to each generation call. This package holds the records and the pack,
and enforces that a reference outside the pack is never resolved to a URL.

Nothing here fetches anything. There is no retriever, no fetcher and no store; a
record is built from content that a caller already has.
"""

from __future__ import annotations

from chakaso.evidence.errors import (
    EvidenceError,
    SourceRecordError,
    SourceReferenceError,
    UrlError,
)
from chakaso.evidence.identity import (
    SourceKind,
    SourceReference,
    file_reference,
    text_reference,
    web_reference,
)
from chakaso.evidence.pack import Citation, CitationResolution, EvidencePack, resolve_citations
from chakaso.evidence.records import EvidenceChunk, SourceRecord
from chakaso.evidence.urls import canonicalize_url, host_of

__all__ = [
    "Citation",
    "CitationResolution",
    "EvidenceChunk",
    "EvidenceError",
    "EvidencePack",
    "SourceKind",
    "SourceRecord",
    "SourceRecordError",
    "SourceReference",
    "SourceReferenceError",
    "UrlError",
    "canonicalize_url",
    "file_reference",
    "host_of",
    "resolve_citations",
    "text_reference",
    "web_reference",
]
