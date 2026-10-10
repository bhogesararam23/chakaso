"""A deterministic lexical index and retriever — the replaceable retrieval boundary.

This is the smallest thing that turns a corpus into ranked evidence: an inverted term
index scored with Okapi BM25. It is lexical on purpose. It matches shared words and
nothing else — no embeddings, no vector index, no model, no network — so it is exactly
reproducible, and it is the baseline a future dense or hybrid retriever will have to beat
in evaluation. Its limitations (synonyms, morphology, paraphrase, anything needing
meaning) are the reason the project wants a measured baseline rather than an assumed
one.

Three properties the rest of the system depends on:

* **Determinism.** The same corpus and query give the same ranking every time. Scores
  are summed in sorted term order so floating-point accumulation cannot drift between
  runs, and ties are broken by chunk identifier, never by dictionary order.
* **An inspectable boundary.** :class:`Retriever` is a protocol; a later retriever
  implements it and nothing above has to change.
* **Honest explanation.** A result carries only what the retriever actually computed — the
  strategy that produced it, matched terms, a score, a rank. It computes no confidence and
  claims no understanding, and the score is a ranking artefact, not a probability that
  anything is true.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from chakaso.core.identifiers import SourceId
from chakaso.evidence import EvidenceChunk
from chakaso.retrieval.errors import RetrievalError
from chakaso.retrieval.strategy import RetrievalStrategy

__all__ = [
    "DEFAULT_B",
    "DEFAULT_K1",
    "LexicalIndex",
    "LexicalRetriever",
    "RetrievalResult",
    "Retriever",
    "build_lexical_index",
    "tokenize",
]

#: BM25 term-frequency saturation. A starting value for a small local corpus, not a
#: measured optimum, and not configuration: nothing reads it per run yet.
DEFAULT_K1: Final[float] = 1.2

#: BM25 length normalization. As above: a documented default a caller may override.
DEFAULT_B: Final[float] = 0.75

# A token is a run of Unicode word characters. Punctuation separates tokens and is
# dropped, which is exactly what a lexical matcher wants; case is folded so that "File"
# and "file" match. Deterministic and dependency-free.
_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> tuple[str, ...]:
    """Split ``text`` into lowercased word tokens for lexical matching."""
    return tuple(_TOKEN.findall(text.casefold()))


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked chunk, with only the explanation the retriever actually computed.

    ``strategy`` records which mechanism produced the result — a lexical match, a dense
    similarity, or a hybrid fusion — so evaluation can attribute a hit to its source rather
    than infer it from the retriever object that ran (ADR-0022/ADR-0023). ``matched_terms``
    is what a lexical match actually overlapped; a dense result matched no terms and leaves
    it empty, because inventing term overlaps it did not compute would be a false explanation.
    """

    chunk: EvidenceChunk
    score: float
    rank: int
    strategy: RetrievalStrategy
    matched_terms: tuple[str, ...] = ()


@runtime_checkable
class Retriever(Protocol):
    """The boundary retrieval is called through, so a lexical one can be replaced."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[SourceId] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Return the best ``top_k`` chunks for ``query``, best first.

        An empty result means nothing matched — it never means fabricated evidence, and
        the caller must not invent a pack when retrieval returns nothing.
        """
        ...


class LexicalIndex:
    """An inverted index over a fixed set of chunks, built once and queried many times.

    Construction is the expensive part and happens once; :class:`LexicalRetriever` reads
    the prepared postings, term frequencies and length statistics without reparsing the
    text for every query.
    """

    def __init__(
        self,
        chunks: tuple[EvidenceChunk, ...],
        term_frequencies: tuple[Counter[str], ...],
        document_frequency: dict[str, int],
        lengths: tuple[int, ...],
    ) -> None:
        self._chunks = chunks
        self._term_frequencies = term_frequencies
        self._document_frequency = document_frequency
        self._lengths = lengths
        self._total_length = sum(lengths)

    @classmethod
    def from_chunks(cls, chunks: Iterable[EvidenceChunk]) -> LexicalIndex:
        """Build an index over ``chunks``."""
        indexed = tuple(chunks)
        frequencies: list[Counter[str]] = []
        lengths: list[int] = []
        document_frequency: dict[str, int] = {}
        for chunk in indexed:
            counts = Counter(tokenize(chunk.text))
            frequencies.append(counts)
            lengths.append(len(counts))
            for term in counts:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        return cls(indexed, tuple(frequencies), document_frequency, tuple(lengths))

    @property
    def chunks(self) -> tuple[EvidenceChunk, ...]:
        """The indexed chunks, in the order they were added."""
        return self._chunks

    @property
    def document_count(self) -> int:
        """How many chunks are indexed."""
        return len(self._chunks)

    @property
    def average_length(self) -> float:
        """Mean token count per chunk, the length-normalization denominator."""
        if not self._chunks:
            return 0.0
        return self._total_length / len(self._chunks)


def build_lexical_index(chunks: Iterable[EvidenceChunk]) -> LexicalIndex:
    """Build a :class:`LexicalIndex` over ``chunks``."""
    return LexicalIndex.from_chunks(chunks)


class LexicalRetriever:
    """A BM25 :class:`Retriever` over a :class:`LexicalIndex`."""

    def __init__(
        self,
        index: LexicalIndex,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if k1 <= 0:
            message = f"k1 must be positive, got {k1}"
            raise RetrievalError(message)
        if not 0 <= b <= 1:
            message = f"b must be between 0 and 1, got {b}"
            raise RetrievalError(message)
        self._index = index
        self._k1 = k1
        self._b = b

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[SourceId] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Score every indexed chunk against ``query`` and return the best ``top_k``.

        ``source_ids``, when given, restricts candidates to those sources — a filter,
        not a re-ranking. Ties are broken by chunk identifier so the order is total and
        reproducible.
        """
        if top_k < 1:
            message = f"top_k must be at least 1, got {top_k}"
            raise RetrievalError(message)

        if self._index.document_count == 0 or self._index.average_length == 0.0:
            return ()

        terms = sorted(set(tokenize(query)))
        if not terms:
            return ()

        wanted = None if source_ids is None else frozenset(source_ids)
        scored: list[tuple[float, EvidenceChunk, tuple[str, ...]]] = []
        for position, chunk in enumerate(self._index.chunks):
            if wanted is not None and chunk.source_id not in wanted:
                continue
            result = self._score_chunk(position, terms)
            if result is not None:
                scored.append(result)

        # Sort by descending score, then ascending chunk identifier, so equal scores
        # still produce one deterministic order rather than an arbitrary one.
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))

        return tuple(
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
                strategy=RetrievalStrategy.LEXICAL,
                matched_terms=matched,
            )
            for rank, (score, chunk, matched) in enumerate(scored[:top_k], start=1)
        )

    def _score_chunk(
        self,
        position: int,
        terms: list[str],
    ) -> tuple[float, EvidenceChunk, tuple[str, ...]] | None:
        """Return one chunk's BM25 score for ``terms``, or ``None`` if it matched none."""
        frequencies = self._index._term_frequencies[position]
        chunk = self._index._chunks[position]
        length = self._index._lengths[position]
        total = self._index.document_count
        average = self._index.average_length

        score = 0.0
        matched: list[str] = []
        for term in terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            matched.append(term)
            document_frequency = self._index._document_frequency[term]
            idf = math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
            normalization = self._k1 * (1.0 - self._b + self._b * length / average)
            score += idf * (frequency * (self._k1 + 1.0)) / (frequency + normalization)

        if not matched:
            return None
        return score, chunk, tuple(matched)
