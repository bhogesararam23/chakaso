"""An in-memory corpus: the set of evidence available to retrieval.

A corpus answers the questions retrieval needs — what sources and chunks exist, how
they are added, how they are enumerated, and where a chunk came from — and nothing else.
It stores nothing on disk and talks to no service. Persistent storage is a separate
decision for when there is a reason for it: a database introduced because one might
eventually help is a database maintained against a guess, and an index is a cache, never
the source of truth.

Membership preserves provenance rather than re-deriving it. Chunks are added with the
source that owns them, and a chunk whose identifier names a source the corpus has not
seen is refused: retrieval that returned such a chunk would be handing back evidence with
nothing behind it. Re-adding a source or a chunk that is already present is a no-op,
because identifiers are content-derived, so "the same thing twice" is literally the same
keys — duplicate documents do not grow the corpus.

Enumeration is deterministic in a way that does not depend on insertion order: sources
come back ordered by identifier and chunks by (source identifier, position), so two runs
that add the same documents in a different order see the same corpus.
"""

from __future__ import annotations

from collections.abc import Iterable

from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.evidence import EvidenceChunk, SourceRecord
from chakaso.retrieval.errors import CorpusError
from chakaso.retrieval.ingest import IngestedDocument

__all__ = ["Corpus"]


class Corpus:
    """A deterministic, in-memory collection of sources and their evidence chunks."""

    def __init__(self) -> None:
        self._sources: dict[SourceId, SourceRecord] = {}
        self._chunks: dict[ChunkId, EvidenceChunk] = {}

    @classmethod
    def of(cls, documents: Iterable[IngestedDocument]) -> Corpus:
        """Build a corpus from ingested documents."""
        corpus = cls()
        for document in documents:
            corpus.add_document(document)
        return corpus

    def add(
        self,
        source: SourceRecord,
        chunks: Iterable[EvidenceChunk] = (),
    ) -> None:
        """Add a source and the chunks that belong to it, idempotently.

        Raises:
            CorpusError: a chunk names a source other than ``source``.
        """
        self._sources[source.source_id] = source
        for chunk in chunks:
            if chunk.source_id != source.source_id:
                message = (
                    f"chunk {chunk.chunk_id} belongs to {chunk.source_id}, not to the "
                    f"source {source.source_id} it was added under"
                )
                raise CorpusError(message)
            self._chunks[chunk.chunk_id] = chunk

    def add_document(self, document: IngestedDocument) -> None:
        """Add an ingested document — its source and its chunks together."""
        self.add(document.source, document.chunks)

    def add_documents(self, documents: Iterable[IngestedDocument]) -> None:
        """Add every document in ``documents``."""
        for document in documents:
            self.add_document(document)

    @property
    def sources(self) -> tuple[SourceRecord, ...]:
        """Every source, ordered by identifier."""
        return tuple(self._sources[identifier] for identifier in sorted(self._sources))

    @property
    def chunks(self) -> tuple[EvidenceChunk, ...]:
        """Every chunk, ordered by source identifier then position within the source."""
        return tuple(
            sorted(self._chunks.values(), key=lambda chunk: (chunk.source_id, chunk.position))
        )

    def source_for(self, source_id: SourceId) -> SourceRecord | None:
        """The source with ``source_id``, or ``None`` if the corpus has no such source."""
        return self._sources.get(source_id)

    def chunk_for(self, chunk_id: ChunkId) -> EvidenceChunk | None:
        """The chunk with ``chunk_id``, or ``None`` if the corpus has no such chunk."""
        return self._chunks.get(chunk_id)

    def chunks_for(self, source_id: SourceId) -> tuple[EvidenceChunk, ...]:
        """The chunks belonging to ``source_id``, in document order."""
        return tuple(chunk for chunk in self.chunks if chunk.source_id == source_id)

    @property
    def source_count(self) -> int:
        """How many distinct sources are in the corpus."""
        return len(self._sources)

    @property
    def chunk_count(self) -> int:
        """How many distinct chunks are in the corpus."""
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        """Whether the corpus holds no chunks."""
        return not self._chunks

    def __len__(self) -> int:
        """The number of chunks; a corpus is counted in evidence units."""
        return len(self._chunks)
