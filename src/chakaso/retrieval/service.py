"""Retrieval orchestration: query to evidence pack, with provenance intact.

This is the component that coordinates the pipeline for one query:

```text
query -> corpus index -> retriever -> ranked chunks -> EvidencePack
```

It is deliberately *not* the conversation manager. The manager will call a retrieval
service and pass the resulting pack to `send`; keeping them separate is what lets
retrieval be developed, tested and replaced on its own, and lets a turn decide whether to
retrieve at all (query planning) without dragging the manager into fetching or ranking.

The service owns three things the evidence contract cares about:

* **Provenance, not manufacture.** Sources come from the corpus the chunks were drawn
  from; nothing is invented. A retrieval that finds nothing produces an empty pack, not
  fabricated evidence — an empty pack is an honest answer, an invented one is the failure
  ADR-0003 exists to prevent.
* **Scores recorded, never invented.** Each returned chunk carries the retriever's own
  `retrieval_score`, recorded onto a copy so the stored chunk is untouched. A score is a
  ranking artefact, and the pack records it as such.
* **Rebuildable cache semantics.** The index is built once from the corpus the service is
  given. A corpus that grows afterwards needs a fresh service, because an index is a
  cache over the corpus, never its source of truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Final

from chakaso.core.identifiers import SourceId
from chakaso.evidence import EvidenceChunk, EvidencePack, SourceRecord
from chakaso.retrieval.corpus import Corpus
from chakaso.retrieval.errors import RetrievalError
from chakaso.retrieval.lexical import (
    LexicalRetriever,
    RetrievalResult,
    Retriever,
    build_lexical_index,
)

__all__ = ["DEFAULT_TOP_K", "RetrievalOutcome", "RetrievalService"]

#: How many chunks to return when the caller does not say. A documented default, not a
#: measured optimum and not yet a configuration field.
DEFAULT_TOP_K: Final[int] = 5


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """The result of one retrieval: the pack to hand generation, and the ranking behind it.

    ``results`` keeps the full ranked explanation — score, rank, matched terms — for a
    developer or the evaluation layer; ``pack`` is the validated evidence the model is
    allowed to cite. They describe the same chunks, so nothing computed here is lost
    between retrieval and generation.
    """

    query: str
    pack: EvidencePack
    results: tuple[RetrievalResult, ...]
    retrieval_config: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


class RetrievalService:
    """Retrieves ranked evidence for a query from one corpus."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        retriever: Retriever | None = None,
        retriever_name: str = "lexical",
    ) -> None:
        """Create a service over ``corpus``.

        Args:
            corpus: The evidence to search. The index is built from its current chunks.
            retriever: An implementation of the :class:`Retriever` boundary. Defaults to
                a BM25 lexical retriever over the corpus. Injecting a different one — a
                dense or hybrid retriever later — is the whole point of the boundary.
            retriever_name: A label recorded in the pack's provenance. It must describe
                the retriever actually used, so a pack from a dense retriever does not
                claim to be lexical.
        """
        self._corpus = corpus
        self._retriever = (
            retriever
            if retriever is not None
            else LexicalRetriever(build_lexical_index(corpus.chunks))
        )
        self._retriever_name = retriever_name

    @property
    def corpus(self) -> Corpus:
        """The corpus this service searches."""
        return self._corpus

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        source_ids: Iterable[SourceId] | None = None,
    ) -> RetrievalOutcome:
        """Retrieve the best chunks for ``query`` and assemble an evidence pack.

        The returned pack is valid by construction: every chunk's source is present and
        the chunks carry the retriever's recorded scores. When nothing matches, the pack
        is empty — an honest "no evidence," never a fabricated one.

        Raises:
            RetrievalError: ``top_k`` is below 1.
        """
        if top_k < 1:
            message = f"top_k must be at least 1, got {top_k}"
            raise RetrievalError(message)

        results = self._retriever.retrieve(query, top_k=top_k, source_ids=source_ids)
        scored = tuple(replace(r.chunk, retrieval_score=r.score) for r in results)
        config = {"retriever": self._retriever_name, "top_k": str(top_k)}

        sources = self._sources_for(scored)
        pack = EvidencePack.of(scored, sources, retrieval_config=config)
        return RetrievalOutcome(
            query=query,
            pack=pack,
            results=results,
            retrieval_config=MappingProxyType(config),
        )

    def _sources_for(self, chunks: tuple[EvidenceChunk, ...]) -> list[SourceRecord]:
        """Collect the distinct sources the chunks belong to, in first-reference order.

        Every chunk came from this service's corpus, so its source is present; if one is
        not, the corpus and the chunks disagree, which is a defect the pack validation
        would also catch. It is raised here with a clearer message.
        """
        seen: dict[SourceId, None] = {}
        for chunk in chunks:
            seen.setdefault(chunk.source_id, None)

        sources: list[SourceRecord] = []
        for source_id in seen:
            source = self._corpus.source_for(source_id)
            if source is None:
                message = f"the corpus has no source {source_id} for retrieved chunks"
                raise RetrievalError(message)
            sources.append(source)
        return sources
