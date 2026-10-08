"""Benchmark case: a query, the evidence it should retrieve, and the judgements to score.

A case is a fixed, reviewable unit of the thing the project wants to measure — did
retrieval find the right evidence, and does the answer stay supported by it. Its shape is
built from the identifiers the evidence layer already owns (`SourceId`, `ChunkId`), so a
case refers to *the same chunks* retrieval returns, not to a parallel description of them.

Two sets carry the retrieval judgement: gold identifiers (acceptable evidence) and
forbidden identifiers (evidence that must not be used). The forbidden set is what catches
citation laundering — a system that cites something loosely related scores the same as one
that cites the right thing if only "relevant" is recorded. A chunk cannot be in both sets;
that is not a hard case, it is a broken case, and construction refuses it.

Cases are immutable and their collections are canonicalized (deduplicated, sorted) on
construction, so two authorings that mean the same thing compare equal and fingerprint to
the same version. The version is a fingerprint of the *judgements*, not of the identifier:
renaming a case keeps its version, editing what it expects changes it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from chakaso.benchmark.errors import InvalidCaseError
from chakaso.benchmark.identity import CaseId, content_version
from chakaso.core.identifiers import ChunkId, SourceId

__all__ = ["BenchmarkCase", "BenchmarkCategory", "ExpectedBehavior"]


class BenchmarkCategory(StrEnum):
    """The kind of retrieval/grounding problem a case probes.

    These are the failure modes the project actually wants to see, not arbitrary labels:
    a distractor is relevant-looking-but-wrong, a conflict is two sources disagreeing, a
    missing-evidence case has no source that supports an answer and should produce an
    abstain, and so on. Each category is a shape the fixtures are built to exhibit.
    """

    DIRECT_RETRIEVAL = "direct-retrieval"
    MULTI_SOURCE = "multi-source"
    DISTRACTOR = "distractor"
    CONFLICTING = "conflicting"
    MISSING_EVIDENCE = "missing-evidence"
    TEMPORAL = "temporal"
    AMBIGUOUS = "ambiguous"
    FOLLOW_UP = "follow-up"
    UNSUPPORTED_PREMISE = "unsupported-premise"
    CITATION_CHALLENGE = "citation-challenge"


class ExpectedBehavior(StrEnum):
    """What the system should do with the evidence it has — not what it will do.

    A judgement the benchmark asserts, so a failure to abstain on a missing-evidence case
    is measurable rather than a matter of taste.
    """

    ANSWER = "answer"
    QUALIFY = "qualify"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One versioned benchmark case: a query and the evidence/claims it should and must not reach."""

    case_id: CaseId
    category: BenchmarkCategory
    query: str
    gold_chunk_ids: tuple[ChunkId, ...] = ()
    forbidden_chunk_ids: tuple[ChunkId, ...] = ()
    gold_source_ids: tuple[SourceId, ...] = ()
    required_claims: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    expected_behavior: ExpectedBehavior = ExpectedBehavior.ANSWER
    requires_citation: bool = True

    def __post_init__(self) -> None:
        query = self.query.strip()
        if not query:
            message = f"case {self.case_id} has a blank query"
            raise InvalidCaseError(message)
        object.__setattr__(self, "query", query)

        object.__setattr__(self, "gold_chunk_ids", _canonical_ids(self.gold_chunk_ids))
        object.__setattr__(self, "forbidden_chunk_ids", _canonical_ids(self.forbidden_chunk_ids))
        object.__setattr__(self, "gold_source_ids", _canonical_ids(self.gold_source_ids))
        object.__setattr__(
            self, "required_claims", _canonical_text(self.required_claims, self.case_id)
        )
        object.__setattr__(
            self, "forbidden_claims", _canonical_text(self.forbidden_claims, self.case_id)
        )

        overlap = set(self.gold_chunk_ids) & set(self.forbidden_chunk_ids)
        if overlap:
            both = ", ".join(sorted(str(identifier) for identifier in overlap))
            message = f"case {self.case_id} lists {both} as both gold and forbidden evidence"
            raise InvalidCaseError(message)

    @property
    def relevant_chunks(self) -> frozenset[ChunkId]:
        """The gold chunk identifiers, as a set for membership tests."""
        return frozenset(self.gold_chunk_ids)

    @property
    def unacceptable_chunks(self) -> frozenset[ChunkId]:
        """The forbidden chunk identifiers, as a set for membership tests."""
        return frozenset(self.forbidden_chunk_ids)

    def canonical_definition(self) -> dict[str, object]:
        """The judgement-bearing fields as a canonical mapping, for versioning.

        The identifier is excluded: it is the name, not part of what is judged. A case
        renamed is the same case; a case whose expected evidence changes is a different
        version.
        """
        return {
            "category": self.category.value,
            "expected_behavior": self.expected_behavior.value,
            "forbidden_chunk_ids": [str(i) for i in self.forbidden_chunk_ids],
            "forbidden_claims": list(self.forbidden_claims),
            "gold_chunk_ids": [str(i) for i in self.gold_chunk_ids],
            "gold_source_ids": [str(i) for i in self.gold_source_ids],
            "query": self.query,
            "required_claims": list(self.required_claims),
            "requires_citation": self.requires_citation,
        }

    @property
    def version(self) -> str:
        """A deterministic fingerprint of this case's judgements."""
        return content_version(self.canonical_definition())


_Identifier = TypeVar("_Identifier", ChunkId, SourceId)


def _canonical_ids(items: Iterable[_Identifier]) -> tuple[_Identifier, ...]:
    """Deduplicate and order identifier collections so equal cases compare and hash equal."""
    return tuple(sorted(dict.fromkeys(items), key=str))


def _canonical_text(items: Iterable[str], case_id: CaseId) -> tuple[str, ...]:
    """Deduplicate and order claim strings, refusing a blank one."""
    seen: dict[str, None] = {}
    for item in items:
        text = item.strip()
        if not text:
            message = f"case {case_id} has a blank required or forbidden claim"
            raise InvalidCaseError(message)
        seen.setdefault(text, None)
    return tuple(sorted(seen))
