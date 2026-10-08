"""Source and evidence records.

Identity and provenance are assigned here, on the retrieval side, and never by a
model (ADR-0003). A record is immutable and its identifier is derived from its
content (ADR-0007), so a page that changed produces a new record rather than
overwriting the one an earlier answer depends on.

Both types are created through class methods — :meth:`SourceRecord.create`,
:meth:`SourceRecord.create_file`, :meth:`SourceRecord.create_text` and
:meth:`EvidenceChunk.create` — rather than by filling in fields directly. Those
methods derive the identifier, the reference and the content hash from the content
itself, so there is no way to build a record whose identifier does not match what it
names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Self

from chakaso.core.hashing import is_sha256_hex, sha256_hex
from chakaso.core.identifiers import ChunkId, SourceId, derive_chunk_id, derive_source_id
from chakaso.evidence.errors import SourceRecordError
from chakaso.evidence.identity import (
    SourceKind,
    SourceReference,
    file_reference,
    require_consistent,
    text_reference,
    web_reference,
)
from chakaso.evidence.urls import host_of

__all__ = ["EvidenceChunk", "SourceRecord"]

_EMPTY_METADATA: Mapping[str, str] = MappingProxyType({})


def _require_aware(moment: datetime, field_name: str) -> datetime:
    """Return ``moment`` if it carries a timezone, else raise.

    A naive timestamp in provenance is a bug waiting to happen: it silently means
    "whatever the machine's timezone was", which is exactly the ambiguity a
    reproducibility record exists to remove.
    """
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        message = (
            f"{field_name} must be timezone-aware, got a naive datetime. "
            "Use datetime.now(datetime.UTC) or attach an explicit offset."
        )
        raise SourceRecordError(message)
    return moment


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One canonical source reference with one content body.

    Immutable, including its metadata mapping. The content itself is not stored
    here: a record is provenance, and the bytes live wherever the store puts them.

    A source may be a web page, a local document or supplied text (ADR-0011).
    ``kind`` records which, and ``canonical_reference`` is its canonical form — a URL
    for a web source, a ``file`` URI for a document, a ``text`` reference otherwise.
    Build records through :meth:`create`, :meth:`create_file` or :meth:`create_text`,
    which derive the identifier, the reference and the content hash from the content
    itself, so a record whose identifier does not match what it names cannot be built.

    ``raw_text_path`` is where the stored bytes live, if anywhere. The storage layout
    is deliberately undecided (see ``docs/architecture.md``), so this is usually
    ``None``; it is present because ``docs/retrieval.md`` specifies it, and a record
    without it would have nowhere to point once a store exists.
    """

    source_id: SourceId
    canonical_reference: str
    kind: SourceKind
    title: str
    domain: str
    retrieved_at: datetime
    published_at: datetime | None
    content_hash: str
    raw_text_path: str | None = None
    # default_factory rather than a bare default: Python 3.11's dataclasses reject
    # any default whose type is unhashable, and mappingproxy is unhashable, so
    # `field(default=_EMPTY_METADATA)` raises at import time on 3.11 while working
    # on 3.12 and later. The shared object is immutable, so one instance for every
    # record is safe.
    metadata: Mapping[str, str] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        # The reference and its kind are checked by the shared rule, so a hand-built
        # record cannot carry a ``FILE`` kind beside an ``http`` value: an inconsistent
        # provenance row is worse than one that fails at construction.
        require_consistent(self.kind, self.canonical_reference)
        if not is_sha256_hex(self.content_hash):
            message = (
                f"content_hash must be a full lowercase SHA-256 digest, got {self.content_hash!r}"
            )
            raise SourceRecordError(message)
        _require_aware(self.retrieved_at, "retrieved_at")
        if self.published_at is not None:
            _require_aware(self.published_at, "published_at")

    @classmethod
    def create(
        cls,
        *,
        url: str,
        title: str,
        content: str | bytes,
        retrieved_at: datetime,
        published_at: datetime | None = None,
        raw_text_path: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        """Build a record for a web source from a URL and its retrieved content.

        Identity is derived from the canonical URL, so this produces exactly the
        identifier ADR-0007 defines for a web source; generalizing identity to other
        kinds does not move it.
        """
        return cls._from_reference(
            web_reference(url),
            title=title,
            content=content,
            retrieved_at=retrieved_at,
            published_at=published_at,
            raw_text_path=raw_text_path,
            metadata=metadata,
        )

    @classmethod
    def create_file(
        cls,
        *,
        path: str | Path,
        title: str,
        content: str | bytes,
        retrieved_at: datetime,
        published_at: datetime | None = None,
        raw_text_path: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        """Build a record for a local document from its path and content.

        The path identifies the source and ``content`` is the text read from it. A
        local file is named by a ``file`` URI and never routed through URL
        canonicalization (ADR-0011), so identifying it cannot widen what a future
        fetcher is permitted to retrieve.
        """
        return cls._from_reference(
            file_reference(path),
            title=title,
            content=content,
            retrieved_at=retrieved_at,
            published_at=published_at,
            raw_text_path=raw_text_path,
            metadata=metadata,
        )

    @classmethod
    def create_text(
        cls,
        *,
        content: str | bytes,
        retrieved_at: datetime,
        title: str = "",
        label: str | None = None,
        published_at: datetime | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        """Build a record for content supplied directly, with no location to cite.

        Identity is the content. An optional ``label`` distinguishes two passages that
        would otherwise deduplicate to one opaque source.
        """
        return cls._from_reference(
            text_reference(label),
            title=title,
            content=content,
            retrieved_at=retrieved_at,
            published_at=published_at,
            raw_text_path=None,
            metadata=metadata,
        )

    @classmethod
    def _from_reference(
        cls,
        reference: SourceReference,
        *,
        title: str,
        content: str | bytes,
        retrieved_at: datetime,
        published_at: datetime | None,
        raw_text_path: str | None,
        metadata: Mapping[str, str] | None,
    ) -> Self:
        """Assemble a record from an already-built reference and its content.

        ``domain`` is the reference's host where one exists; a file or text reference
        has no host, so it is empty rather than guessed. ``published_at`` is left
        ``None`` when unknown: a guessed publication date is worse than a missing one,
        because it lets a stale source look current with no way to tell later.
        """
        canonical = reference.canonical
        return cls(
            source_id=derive_source_id(canonical, content),
            canonical_reference=canonical,
            kind=reference.kind,
            title=title,
            domain=host_of(canonical),
            retrieved_at=_require_aware(retrieved_at, "retrieved_at"),
            published_at=published_at,
            content_hash=sha256_hex(content),
            raw_text_path=raw_text_path,
            metadata=MappingProxyType(dict(metadata or {})),
        )

    def matches_content(self, content: str | bytes) -> bool:
        """Whether ``content`` is the content this record was built from.

        This is the verification that makes immutability meaningful. A record
        stores only a hash, so it cannot check itself; when the bytes are available
        — from a store, or from a re-fetch — this says whether the record still
        describes them. It is how a stored record is confirmed before an old answer
        is re-examined against it.
        """
        return (
            sha256_hex(content) == self.content_hash
            and derive_source_id(self.canonical_reference, content) == self.source_id
        )

    @property
    def is_freshness_known(self) -> bool:
        """Whether this source's publication date is known.

        Callers should say "the publication date is unknown" rather than implying
        that an undated source is current.
        """
        return self.published_at is not None


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """One evidence unit within one source.

    ``section`` is the heading path the chunk sits under, or ``None`` when the
    document has no usable structure. ``None`` rather than an empty string, so that
    "no heading" and "a heading whose text is empty" stay distinguishable.
    """

    chunk_id: ChunkId
    source_id: SourceId
    text: str
    position: int
    section: str | None = None
    token_count: int | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None

    def __post_init__(self) -> None:
        if self.position < 0:
            message = f"position must not be negative, got {self.position}"
            raise SourceRecordError(message)
        if not self.text:
            message = "chunk text must not be empty; an empty chunk is not evidence"
            raise SourceRecordError(message)
        if self.token_count is not None and self.token_count < 1:
            message = f"token_count must be at least 1 when present, got {self.token_count}"
            raise SourceRecordError(message)

    @classmethod
    def create(
        cls,
        *,
        source_id: SourceId,
        text: str,
        position: int,
        section: str | None = None,
        token_count: int | None = None,
        retrieval_score: float | None = None,
        rerank_score: float | None = None,
    ) -> Self:
        """Build a chunk whose identifier is derived from its source, position and text.

        Scores are recorded, never presented. A retrieval score is a ranking
        artefact, not a probability that a claim is true, and nothing in the system
        may display one to a user as confidence.
        """
        # Checked here as well as in __post_init__ so that both entry points raise
        # the same error type. Deriving the identifier first would surface an
        # IdentifierError from a different module, which is a confusing thing to
        # get from a bad chunk position.
        if position < 0:
            message = f"position must not be negative, got {position}"
            raise SourceRecordError(message)
        return cls(
            chunk_id=derive_chunk_id(source_id, position, text),
            source_id=source_id,
            text=text,
            position=position,
            section=section,
            token_count=token_count,
            retrieval_score=retrieval_score,
            rerank_score=rerank_score,
        )
