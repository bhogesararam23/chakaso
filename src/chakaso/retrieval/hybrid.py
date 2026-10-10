"""Hybrid retrieval: fuse the lexical and dense retrievers into one ranking.

Two retrievers now exist behind the `Retriever` protocol — lexical BM25 and dense similarity —
and they fail differently: lexical is exact on shared terms, dense reaches near-matches a real
embedding would (though with the fixture double it largely reproduces term overlap). A hybrid
retriever composes them so a caller gets one ranked list. It is a *composition*, not a
super-class: it holds two `Retriever`s and a fusion rule, adding no orchestration beyond the
merge (ADR-0022's "no giant orchestrator" applies here too).

The fusion is Reciprocal Rank Fusion, chosen because it needs only the ranks each component
produced — not scores on a comparable scale, which a BM25 score and a cosine similarity are
not. Its formula is written out, not hidden:

    fused(c) = lexical_weight / (rrf_k + rank_lexical(c))   for the components that returned c
             + dense_weight    / (rrf_k + rank_dense(c))

with a missing component contributing nothing. RRF's `1/(k+rank)` is the published method
(Cormack et al.); `rrf_k` smooths rank-1 dominance, the weights let one component be favored.
Deterministic by construction: components are summed in a fixed order and ties break by chunk
identifier.

Every fused result is tagged ``hybrid`` and carries `HybridProvenance` — the lexical and dense
ranks and scores that produced it and the fused score that ordered it — so a hybrid hit is
auditable back to its parts rather than an opaque number. The fused score is an RRF value, a
ranking artefact, not a similarity or a probability.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.evidence import EvidenceChunk
from chakaso.retrieval.errors import RetrievalError
from chakaso.retrieval.lexical import RetrievalResult, Retriever
from chakaso.retrieval.strategy import HybridProvenance, RetrievalStrategy

__all__ = ["FusionMethod", "HybridConfig", "HybridRetriever"]


class FusionMethod(StrEnum):
    """The rank-fusion rule. Only RRF is implemented today."""

    RRF = "rrf"


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Typed tuning for the hybrid retriever (not a second configuration system).

    Attributes:
        method: The fusion rule; only ``RRF`` exists, and anything else is refused rather
            than silently ignored.
        rrf_k: RRF smoothing constant. Higher values flatten the rank influence; the
            published default is 60.
        candidate_k: How many results each component retrieves before fusion. ``None`` means
            use the caller's ``top_k``; setting it larger over-retrieves for a better pool.
        lexical_weight / dense_weight: How much each component's reciprocal rank contributes.
            Non-negative, and not both zero (a fusion of nothing would rank nothing).
    """

    method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60
    candidate_k: int | None = None
    lexical_weight: float = 1.0
    dense_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.method is not FusionMethod.RRF:
            message = f"only RRF fusion is implemented; got method {self.method.value!r}"
            raise RetrievalError(message)
        if self.rrf_k < 1:
            message = f"rrf_k must be at least 1, got {self.rrf_k}"
            raise RetrievalError(message)
        if self.candidate_k is not None and self.candidate_k < 1:
            message = f"candidate_k must be at least 1 when given, got {self.candidate_k}"
            raise RetrievalError(message)
        if self.lexical_weight < 0 or self.dense_weight < 0:
            message = "fusion weights must be non-negative"
            raise RetrievalError(message)
        if self.lexical_weight == 0 and self.dense_weight == 0:
            message = "both fusion weights are zero; at least one component must contribute"
            raise RetrievalError(message)


@dataclass(slots=True)
class _Fused:
    """Mutable accumulator of one chunk's per-component hits during fusion."""

    chunk: EvidenceChunk
    lexical: RetrievalResult | None = None
    dense: RetrievalResult | None = None


class HybridRetriever:
    """A `Retriever` that fuses a lexical and a dense retriever by reciprocal rank fusion."""

    def __init__(
        self,
        lexical: Retriever,
        dense: Retriever,
        *,
        config: HybridConfig | None = None,
    ) -> None:
        self._lexical = lexical
        self._dense = dense
        self._config = config if config is not None else HybridConfig()

    @property
    def config(self) -> HybridConfig:
        """The fusion configuration this retriever applies."""
        return self._config

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[SourceId] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Fuse the two components' rankings for ``query`` and return the best ``top_k``.

        A ``source_ids`` filter is passed through to both components, so a follow-up
        constrained to prior sources stays constrained through the fusion. A chunk nothing
        returned is never produced: results come only from what the components actually
        retrieved.
        """
        if top_k < 1:
            message = f"top_k must be at least 1, got {top_k}"
            raise RetrievalError(message)

        config = self._config
        pool = config.candidate_k if config.candidate_k is not None else top_k
        lexical_results = self._lexical.retrieve(query, top_k=pool, source_ids=source_ids)
        dense_results = self._dense.retrieve(query, top_k=pool, source_ids=source_ids)

        fused: dict[ChunkId, _Fused] = {}
        for result in lexical_results:
            fused[result.chunk.chunk_id] = _Fused(chunk=result.chunk, lexical=result)
        for result in dense_results:
            entry = fused.get(result.chunk.chunk_id)
            if entry is None:
                fused[result.chunk.chunk_id] = _Fused(chunk=result.chunk, dense=result)
            else:
                entry.dense = result

        ranked: list[tuple[float, ChunkId, _Fused]] = []
        for chunk_id, entry in fused.items():
            score = 0.0
            if entry.lexical is not None:
                score += config.lexical_weight / (config.rrf_k + entry.lexical.rank)
            if entry.dense is not None:
                score += config.dense_weight / (config.rrf_k + entry.dense.rank)
            ranked.append((score, chunk_id, entry))

        ranked.sort(key=lambda item: (-item[0], item[1]))

        return tuple(
            RetrievalResult(
                chunk=entry.chunk,
                score=score,
                rank=rank,
                strategy=RetrievalStrategy.HYBRID,
                matched_terms=entry.lexical.matched_terms if entry.lexical is not None else (),
                provenance=_provenance(entry, score),
            )
            for rank, (score, _chunk_id, entry) in enumerate(ranked[:top_k], start=1)
        )


def _provenance(entry: _Fused, fused_score: float) -> HybridProvenance:
    return HybridProvenance(
        lexical_rank=None if entry.lexical is None else entry.lexical.rank,
        lexical_score=None if entry.lexical is None else entry.lexical.score,
        dense_rank=None if entry.dense is None else entry.dense.rank,
        dense_score=None if entry.dense is None else entry.dense.score,
        fused_score=fused_score,
    )
