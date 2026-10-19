"""Grounded-answer case, source and provenance schema.

A grounded case is a *distinct* judgement object from the retrieval `BenchmarkCase`. A
retrieval case asks only "did the right chunks come back"; a grounded case asks the question
the answer pipeline must survive — what should the system *do* with the evidence it has, which
facts must an acceptable answer carry, which claims must it never make, and what does the case
exist to check. It records all of that against a source pool it can name evidence by, so a gold
reference is the same content-derived `ChunkId` retrieval produces (ADR-0007), not a parallel
description of it.

The provenance record is not metadata for a report footer. It is the schema field that makes
the workload *legally usable*: every source states who authored it, under what licence, and
whether it may be redistributed (ADR-0031). A source without provenance is refused here, at
construction, rather than discovered later as a licence problem in a shipped dataset.

Cases are immutable and their collections are canonicalized on construction, and the version is
a fingerprint of the judgements only — never of the identifier — so a renamed case keeps its
version while any change to what it expects bumps it, exactly the rule ADR-0013 sets for the
retrieval benchmark this reuses the identity layer of.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from chakaso.benchmark.cases import BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.errors import InvalidCaseError, MissingProvenanceError
from chakaso.benchmark.identity import CaseId, content_version

__all__ = [
    "CaseSplit",
    "GroundedCase",
    "GroundedProvenance",
    "GroundedSource",
]

#: A source-passage or label slug: readable, safe inside a ``text:<label>`` reference and a
#: report column, and bounded so it cannot be an unbounded blob.
_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


class CaseSplit(StrEnum):
    """Which side of the workload a case sits on.

    Deliberately two honest names rather than train/dev/test: the project selects a
    configuration on ``development`` and reports a number on ``heldout`` only. The
    ``heldout`` split is what makes a reported figure a held-out measurement rather than a
    restatement of whatever was tuned on; a case id reused across both is refused by the
    dataset, and so is a source that supports cases on either side.
    """

    DEVELOPMENT = "development"
    HELDOUT = "heldout"


@dataclass(frozen=True, slots=True)
class GroundedProvenance:
    """Where a source came from and under what terms it may be used.

    Every field is required and non-empty. This is the guard ADR-0031 is built on: a
    passage the repository cannot trace to an author and a licence is not a case with a
    blank optional field, it is evidence the project is not permitted to ship.
    """

    origin: str
    licence: str
    collection_method: str
    redistribution: str
    created_on: str = ""

    def __post_init__(self) -> None:
        for field_name in ("origin", "licence", "collection_method", "redistribution"):
            if not getattr(self, field_name).strip():
                message = f"provenance is missing required field {field_name!r}"
                raise MissingProvenanceError(message)

    def canonical_definition(self) -> dict[str, object]:
        """The provenance as a canonical mapping, for fingerprinting."""
        return {
            "collection_method": self.collection_method,
            "created_on": self.created_on,
            "licence": self.licence,
            "origin": self.origin,
            "redistribution": self.redistribution,
        }


@dataclass(frozen=True, slots=True)
class GroundedSource:
    """One source passage in the workload, with its provenance attached.

    The ``label`` is the stable name a case refers to this passage by; it is resolved to the
    content-derived identifiers of the chunks the passage ingests into. ``text`` is the exact
    evidence, authored so it fits a single chunk, so naming a passage is naming its evidence.
    """

    label: str
    title: str
    text: str
    provenance: GroundedProvenance

    def __post_init__(self) -> None:
        if _LABEL_PATTERN.fullmatch(self.label) is None:
            message = (
                f"source label {self.label!r} is not a valid slug. Expected lowercase, "
                "starting with a letter or digit, then letters, digits, '.', '_' or '-'."
            )
            raise InvalidCaseError(message)
        if not self.text.strip():
            message = f"source {self.label} has empty evidence text"
            raise InvalidCaseError(message)
        if not self.title.strip():
            message = f"source {self.label} has a blank title"
            raise InvalidCaseError(message)


@dataclass(frozen=True, slots=True)
class GroundedCase:
    """One versioned grounded-answer case: a question and everything needed to judge an answer."""

    case_id: CaseId
    category: BenchmarkCategory
    split: CaseSplit
    question: str
    rationale: str
    expected_behavior: ExpectedBehavior = ExpectedBehavior.ANSWER
    gold_labels: tuple[str, ...] = ()
    forbidden_labels: tuple[str, ...] = ()
    required_evidence_labels: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    prohibited_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        question = self.question.strip()
        if not question:
            message = f"case {self.case_id} has a blank question"
            raise InvalidCaseError(message)
        object.__setattr__(self, "question", question)

        rationale = self.rationale.strip()
        if not rationale:
            message = f"case {self.case_id} has no evaluation rationale"
            raise InvalidCaseError(message)
        object.__setattr__(self, "rationale", rationale)

        object.__setattr__(self, "gold_labels", _canonical_labels(self.gold_labels, self.case_id))
        object.__setattr__(
            self, "forbidden_labels", _canonical_labels(self.forbidden_labels, self.case_id)
        )
        object.__setattr__(
            self,
            "required_evidence_labels",
            _canonical_labels(self.required_evidence_labels, self.case_id),
        )
        object.__setattr__(
            self, "required_facts", _canonical_text(self.required_facts, self.case_id)
        )
        object.__setattr__(
            self, "prohibited_claims", _canonical_text(self.prohibited_claims, self.case_id)
        )

        overlap = set(self.gold_labels) & set(self.forbidden_labels)
        if overlap:
            both = ", ".join(sorted(overlap))
            message = f"case {self.case_id} lists {both} as both gold and forbidden evidence"
            raise InvalidCaseError(message)

        dangling = set(self.required_evidence_labels) - set(self.gold_labels)
        if dangling:
            both = ", ".join(sorted(dangling))
            message = f"case {self.case_id} requires evidence {both} that is not in its gold set"
            raise InvalidCaseError(message)

        if self.expected_behavior is ExpectedBehavior.ABSTAIN and self.gold_labels:
            message = f"case {self.case_id} expects abstention but names gold evidence"
            raise InvalidCaseError(message)
        if self.expected_behavior is ExpectedBehavior.ANSWER and not self.gold_labels:
            message = f"case {self.case_id} expects an answer but names no gold evidence"
            raise InvalidCaseError(message)
        if self.expected_behavior is ExpectedBehavior.ABSTAIN and self.required_facts:
            message = f"case {self.case_id} expects abstention but lists required facts"
            raise InvalidCaseError(message)

    @property
    def answerable(self) -> bool:
        """Whether the correct behaviour is to produce an answer rather than abstain."""
        return self.expected_behavior is not ExpectedBehavior.ABSTAIN

    def canonical_definition(self) -> dict[str, object]:
        """The judgement-bearing fields as a canonical mapping, for versioning.

        The identifier is excluded (ADR-0013): it is the address, not part of what is judged.
        """
        return {
            "category": self.category.value,
            "expected_behavior": self.expected_behavior.value,
            "forbidden_labels": list(self.forbidden_labels),
            "gold_labels": list(self.gold_labels),
            "prohibited_claims": list(self.prohibited_claims),
            "question": self.question,
            "rationale": self.rationale,
            "required_evidence_labels": list(self.required_evidence_labels),
            "required_facts": list(self.required_facts),
            "split": self.split.value,
        }

    @property
    def version(self) -> str:
        """A deterministic fingerprint of this case's judgements."""
        return content_version(self.canonical_definition())


def _canonical_labels(items: Iterable[str], case_id: CaseId) -> tuple[str, ...]:
    """Deduplicate and order source labels, refusing a blank one."""
    seen: dict[str, None] = {}
    for item in items:
        label = item.strip()
        if not label:
            message = f"case {case_id} has a blank evidence label"
            raise InvalidCaseError(message)
        seen.setdefault(label, None)
    return tuple(sorted(seen))


def _canonical_text(items: Iterable[str], case_id: CaseId) -> tuple[str, ...]:
    """Deduplicate and order fact/claim strings, refusing a blank one."""
    seen: dict[str, None] = {}
    for item in items:
        text = item.strip()
        if not text:
            message = f"case {case_id} has a blank required fact or prohibited claim"
            raise InvalidCaseError(message)
        seen.setdefault(text, None)
    return tuple(sorted(seen))
