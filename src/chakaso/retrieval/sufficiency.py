"""Evidence sufficiency: a deterministic, corpus-statistical decision at the retrieval boundary.

Retrieval's job is to hand back ranked candidates — `Retriever.retrieve` returns every chunk that
shares a term with the query, which is correct for *finding* but useless for deciding *whether there is
anything here to answer from*. E2603 measured the consequence: every unanswerable question produced a
non-empty pack because incidental function-word overlaps (`a`, `the`, `is`, `library`) scored like real
matches. This module draws the missing line, and does it at a separate seam so retrieval itself and its
contract are unchanged (ADR-0032).

The rule is deliberately *not* a raw BM25 score threshold. BM25 magnitudes are query-length- and
corpus-dependent and are not probabilities, and they are not comparable across the lexical/dense/hybrid
strategies — so a single number would be an uncalibrated magic constant. Instead the decision uses the
one signal that actually separated the failures: a retrieved chunk is *adequate* evidence only if it
shares a **discriminating** query term — a term that appears in a small fraction of the corpus, so its
presence tells you something — rather than only corpus-common words. A query whose every candidate match
is on non-discriminating words has nothing here that supports an answer, and the outcome is
`insufficient`.

Everything here is deterministic, offline and numpy-free, and the decision names the terms behind it so a
report is inspectable rather than a bare boolean. The `max_document_fraction` knob is a development
heuristic tuned to this corpus, **not** a calibrated, transferable threshold; that is stated in ADR-0032
and re-examined in P29 rather than asserted here. A differently-phrased but genuine passage can still be
judged insufficient (the lexical morphology/synonym limit that made `renewal` fail to match `renewed`):
that is a measured false-abstention, not something this boundary pretends to solve.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from chakaso.evidence import EvidenceChunk
from chakaso.retrieval.corpus import Corpus
from chakaso.retrieval.errors import RetrievalError
from chakaso.retrieval.lexical import tokenize

__all__ = [
    "DEFAULT_MAX_DOCUMENT_FRACTION",
    "STOP_WORDS",
    "DiscriminatingTermSufficiencyPolicy",
    "EvidenceSufficiency",
    "EvidenceSufficiencyPolicy",
    "SufficiencyOutcome",
]

#: Default corpus-document fraction at or below which a term counts as discriminating. A documented
#: development value for this workload, not a calibrated constant — see ADR-0032. On the shipped
#: 13-document corpus the policy separates abstention from answer correctly only over a narrow band
#: (about 0.23-0.31); 0.25 is chosen inside it. That narrowness is a real limitation and is re-
#: examined, not hidden, in P29.
DEFAULT_MAX_DOCUMENT_FRACTION: Final[float] = 0.25

#: Closed-class function words. A term is *discriminating* only if it is not one of these: on a small
#: corpus even `to`, `on` and `and` can appear in a single document, so document frequency alone does not
#: separate a meaningful content overlap from an incidental function-word one (the E2603 abstention
#: queries matched exactly such words). This list is deliberately short and inspectable — a starting
#: closed-class set, not an exhaustive stopword list, and its effect is re-examined in P29.
STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "must",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


class SufficiencyOutcome(StrEnum):
    """Whether the retrieved evidence is adequate to ground an answer.

    Two states, not three: an empty pack or an empty corpus is `INSUFFICIENT` (there is nothing to
    ground on), which is a decision, not an indeterminate. A policy that cannot read a corpus is a
    construction error raised at build time, never a value returned here.
    """

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class EvidenceSufficiency:
    """The outcome of one sufficiency decision, with the evidence that justified it.

    ``discriminating_terms`` lists the query terms (sorted) that a retrieved chunk actually shared and
    that were corpus-discriminating — the reason for a `SUFFICIENT` verdict, and empty on an
    `INSUFFICIENT` one. The counts make the decision inspectable and reproducible.
    """

    outcome: SufficiencyOutcome
    discriminating_terms: tuple[str, ...]
    query_term_count: int
    candidate_chunk_count: int

    @property
    def is_sufficient(self) -> bool:
        """Whether the evidence was judged adequate to answer from."""
        return self.outcome is SufficiencyOutcome.SUFFICIENT


@runtime_checkable
class EvidenceSufficiencyPolicy(Protocol):
    """The boundary a retrieval decision is made through, so a policy can be replaced."""

    def assess(self, query: str, chunks: Iterable[EvidenceChunk]) -> EvidenceSufficiency:
        """Decide whether ``chunks`` retrieved for ``query`` are adequate evidence."""
        ...


class DiscriminatingTermSufficiencyPolicy:
    """Sufficiency by discriminating-term overlap, using document frequency from the corpus it searched.

    The corpus is read once at construction to compute document frequencies deterministically from the
    same chunks retrieval searched (ADR-0007 content-derived identities make this reproducible); a term
    is *discriminating* for a query when its document frequency is between 1 and
    ``max_document_fraction * document_count`` inclusive. A term present in no document (frequency 0) is
    never discriminating — it supports nothing — which is exactly what separates the E2603 abstention
    cases from the answerable ones.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        max_document_fraction: float = DEFAULT_MAX_DOCUMENT_FRACTION,
    ) -> None:
        if not 0.0 < max_document_fraction <= 1.0:
            message = f"max_document_fraction must be in (0, 1], got {max_document_fraction}"
            raise RetrievalError(message)
        self._max_document_fraction = max_document_fraction
        self._chunks = tuple(corpus.chunks)
        self._document_count = len(self._chunks)
        document_frequency: Counter[str] = Counter()
        for chunk in self._chunks:
            for term in set(tokenize(chunk.text)):
                document_frequency[term] += 1
        self._document_frequency = document_frequency

    def assess(self, query: str, chunks: Iterable[EvidenceChunk]) -> EvidenceSufficiency:
        """Report whether any retrieved chunk shares a discriminating query term."""
        query_terms = set(tokenize(query))
        candidates = tuple(chunks)
        if self._document_count == 0 or not query_terms or not candidates:
            return EvidenceSufficiency(
                outcome=SufficiencyOutcome.INSUFFICIENT,
                discriminating_terms=(),
                query_term_count=len(query_terms),
                candidate_chunk_count=len(candidates),
            )

        ceiling = self._max_document_fraction * self._document_count
        discriminating = {
            term
            for term in query_terms
            if term not in STOP_WORDS and 1 <= self._document_frequency.get(term, 0) <= ceiling
        }

        matched: set[str] = set()
        for chunk in candidates:
            chunk_terms = set(tokenize(chunk.text))
            matched |= discriminating & chunk_terms

        outcome = SufficiencyOutcome.SUFFICIENT if matched else SufficiencyOutcome.INSUFFICIENT
        return EvidenceSufficiency(
            outcome=outcome,
            discriminating_terms=tuple(sorted(matched)),
            query_term_count=len(query_terms),
            candidate_chunk_count=len(candidates),
        )
