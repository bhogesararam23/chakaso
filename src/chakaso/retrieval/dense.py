"""An exact, in-memory dense index and retriever over the embedding boundary.

This is the second implementation of the `Retriever` protocol — the lexical BM25 retriever
is the first (ADR-0022 named the ``dense`` mode; this is what finally backs it). It scores a
query by similarity to stored vectors rather than by shared terms, using the
`EmbeddingModel` boundary in `chakaso.retrieval.embeddings`. The index is *exact*: it scans
every vector and computes every similarity, so it is correct and reproducible at the small
scale this project runs at. An approximate structure (FAISS or similar) is deferred until a
measured corpus is large enough to need one — introducing it now would be a guess, and a
heavy dependency the CPU-first, offline posture (ADR-0001) does not want.

Honesty about quality is the load-bearing constraint. With the fixture embedding, a dense
result is a hashed bag-of-tokens similarity — token overlap wearing a vector costume, not
semantic retrieval. Nothing here claims meaning: a result records the ``dense`` strategy that
produced it and the similarity score it computed, and the score is a ranking artefact, not a
probability and not an understanding. A real embedding model is what would make ``dense``
mean something a lexical baseline cannot; the boundary is what lets it drop in unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.evidence import EvidenceChunk
from chakaso.retrieval.embeddings import (
    EmbeddingMetadata,
    EmbeddingModel,
    EmbeddingVector,
    SimilarityMetric,
)
from chakaso.retrieval.errors import RetrievalError
from chakaso.retrieval.lexical import RetrievalResult
from chakaso.retrieval.strategy import RetrievalStrategy

__all__ = ["DenseIndex", "DenseRetriever", "build_dense_index"]


class DenseIndex:
    """A fixed set of chunks with a vector each, built once and searched many times."""

    def __init__(self, chunks: Sequence[EvidenceChunk], vectors: Sequence[EmbeddingVector]) -> None:
        if len(chunks) != len(vectors):
            message = f"got {len(chunks)} chunks but {len(vectors)} vectors; they must match"
            raise RetrievalError(message)
        if chunks:
            dimension = vectors[0].dimension
            for index, vector in enumerate(vectors):
                if vector.dimension != dimension:
                    message = (
                        f"vector at position {index} has dimension {vector.dimension}, not the "
                        f"index dimension {dimension}; a dense index must be uniform"
                    )
                    raise RetrievalError(message)
        chunk_ids: set[ChunkId] = set()
        for chunk in chunks:
            if chunk.chunk_id in chunk_ids:
                message = f"duplicate chunk {chunk.chunk_id} cannot be indexed twice"
                raise RetrievalError(message)
            chunk_ids.add(chunk.chunk_id)

        self._chunks = tuple(chunks)
        self._vectors = tuple(vectors)

    @classmethod
    def from_chunks(cls, chunks: Iterable[EvidenceChunk], embedder: EmbeddingModel) -> DenseIndex:
        """Embed ``chunks`` with ``embedder`` and build the index."""
        indexed = tuple(chunks)
        vectors = embedder.embed([chunk.text for chunk in indexed])
        return cls(indexed, vectors)

    @property
    def document_count(self) -> int:
        """How many chunks are indexed."""
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        """The shared vector length, or ``0`` for an empty index."""
        return self._vectors[0].dimension if self._vectors else 0

    @property
    def chunks(self) -> tuple[EvidenceChunk, ...]:
        """The indexed chunks, in the order they were added."""
        return self._chunks

    @property
    def vectors(self) -> tuple[EmbeddingVector, ...]:
        """The vector for each indexed chunk, positionally aligned with `chunks`."""
        return self._vectors


def build_dense_index(chunks: Iterable[EvidenceChunk], embedder: EmbeddingModel) -> DenseIndex:
    """Build a :class:`DenseIndex` over ``chunks`` using ``embedder``."""
    return DenseIndex.from_chunks(chunks, embedder)


class DenseRetriever:
    """A `Retriever` that ranks indexed chunks by vector similarity to the query."""

    def __init__(
        self,
        index: DenseIndex,
        embedder: EmbeddingModel,
        *,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        if index.document_count and index.dimension != embedder.metadata.dimension:
            message = (
                f"the index dimension {index.dimension} does not match the embedding model "
                f"dimension {embedder.metadata.dimension}"
            )
            raise RetrievalError(message)
        self._index = index
        self._embedder = embedder
        self._metric = metric

    @property
    def embedding_metadata(self) -> EmbeddingMetadata:
        """The identity of the embedding model this retriever ranks with."""
        return self._embedder.metadata

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[SourceId] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Return the best ``top_k`` chunks by similarity to ``query``, best first.

        Mirrors the lexical retriever's contract exactly: ``top_k`` below 1 is an error, a
        ``source_ids`` filter restricts candidates, and ties break by chunk identifier for a
        total, reproducible order. A query that embeds to a zero vector (nothing to compare)
        returns nothing rather than an arbitrary ranking; a result is tagged ``dense`` and
        carries no matched terms, because a similarity search matched none.
        """
        if top_k < 1:
            message = f"top_k must be at least 1, got {top_k}"
            raise RetrievalError(message)
        if self._index.document_count == 0:
            return ()

        (query_vector,) = self._embedder.embed([query])
        if _norm(query_vector) == 0.0:
            return ()

        wanted = None if source_ids is None else frozenset(source_ids)
        scored: list[tuple[float, EvidenceChunk]] = []
        for chunk, vector in zip(self._index.chunks, self._index.vectors, strict=True):
            if wanted is not None and chunk.source_id not in wanted:
                continue
            scored.append((self._similarity(query_vector, vector), chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))

        return tuple(
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
                strategy=RetrievalStrategy.DENSE,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        )

    def _similarity(self, left: EmbeddingVector, right: EmbeddingVector) -> float:
        """Compare two vectors under the configured metric, treating a zero vector as 0."""
        dot = sum(a * b for a, b in zip(left.values, right.values, strict=True))
        if self._metric is SimilarityMetric.DOT:
            return dot
        denominator = _norm(left) * _norm(right)
        if denominator == 0.0:
            return 0.0
        return dot / denominator


def _norm(vector: EmbeddingVector) -> float:
    return math.sqrt(sum(value * value for value in vector.values))
