"""Source and evidence records.

Identity and provenance are assigned here, on the retrieval side, and never by a
model (ADR-0003). A record is immutable and its identifier is derived from its
content (ADR-0007), so a page that changed produces a new record rather than
overwriting the one an earlier answer depends on.

Both types are created through :meth:`SourceRecord.create` and
:meth:`EvidenceChunk.create` rather than by filling in fields directly. Those class
methods derive the identifier, the host and the content hash from the content
itself, so there is no way to build a record whose identifier does not match what it
names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Self

from chakaso.core.hashing import is_sha256_hex, sha256_hex
from chakaso.core.identifiers import ChunkId, SourceId, derive_chunk_id, derive_source_id
from chakaso.evidence.errors import SourceRecordError
from chakaso.evidence.urls import canonicalize_url, host_of

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
    """One canonical URL with one retrieved content body.

    Immutable, including its metadata mapping. The content itself is not stored
    here: a record is provenance, and the bytes live wherever the store puts them.

    ``raw_text_path`` is where those bytes live, if anywhere. The storage layout is
    deliberately undecided (see ``docs/architecture.md``), so this is usually
    ``None``; it is present because ``docs/retrieval.md`` specifies it, and a
    record without it would have nowhere to point once a store exists.
    """

    source_id: SourceId
    canonical_url: str
    title: str
    domain: str
    retrieved_at: datetime
    published_at: datetime | None
    content_hash: str
    raw_text_path: str | None = None
    metadata: Mapping[str, str] = field(default=_EMPTY_METADATA)

    def __post_init__(self) -> None:
        if not self.canonical_url:
            message = "canonical_url must not be empty"
            raise SourceRecordError(message)
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
        """Build a record, deriving identity, host and content hash from ``content``.

        ``published_at`` is left as ``None`` when unknown. A guessed publication
        date is worse than a missing one: it lets a stale source look current, and
        afterwards there is no way to tell which dates were guesses.
        """
        canonical_url = canonicalize_url(url)
        return cls(
            source_id=derive_source_id(canonical_url, content),
            canonical_url=canonical_url,
            title=title,
            domain=host_of(canonical_url),
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
            and derive_source_id(self.canonical_url, content) == self.source_id
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
