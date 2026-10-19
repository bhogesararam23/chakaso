"""A small, self-authored grounded-answer workload.

This is the P27 corpus: a handful of questions over a handful of passages, each passage written
for Chakaso so the repository can prove it is allowed to ship it (ADR-0031; the same self-authored,
Apache-2.0 basis as the training corpus `chakaso-toy-char-v1`). No external text was copied and no
licence to someone else's corpus is being assumed. Every passage carries a provenance record, and
every case carries the judgements an answer evaluation needs — the evidence it must cite, the
evidence it must not, the facts a correct answer has to state, the claims it must never make, and
what the case exists to check.

It is deliberately tiny and deliberately shaped. It is *not* a real-world corpus and cannot support
a claim about answer quality on natural language (K-001 stands); it is a trustworthy seam the
pipeline can be pointed at and measured against, small enough that a person can read all of it. The
questions span the situations that separate a grounded system from a fluent one: a single-source
fact, a multi-source fact, a source that must not be used, two sources that conflict, a question
nothing in the pool answers, a question carrying a false premise, and an ambiguous question whose
right output is a qualification rather than a confident single number.

The cases are split into a development side and a held-out side, and no passage grounds a case on
both — configuration and prompt choices are made on development, and only the held-out cases feed a
reported number, so a selection-set passage cannot inflate it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from chakaso.benchmark.cases import BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.grounded_cases import (
    CaseSplit,
    GroundedCase,
    GroundedProvenance,
    GroundedSource,
)
from chakaso.benchmark.grounded_dataset import (
    GroundedDataset,
    build_corpus,
)
from chakaso.benchmark.identity import CaseId, DatasetId
from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.retrieval import Corpus

__all__ = [
    "GROUNDED_DATASET_ID",
    "GROUNDED_SOURCES",
    "GROUNDED_WORKLOAD_VERSION",
    "SELF_AUTHORED_PROVENANCE",
    "development_grounded_workload",
    "load_development_grounded_workload",
]

#: The one provenance every passage in this workload shares, because every passage was written
#: here. It is stated once and attached to each source rather than left to a README.
SELF_AUTHORED_PROVENANCE = GroundedProvenance(
    origin="Authored for Chakaso (self-created; no external text was copied).",
    licence="Apache-2.0 (the project's own licence; no third-party terms apply).",
    collection_method="hand-authored policy prose about a fictional library",
    redistribution="no restriction; the project owns this text",
    created_on="2026-10-19",
)

GROUNDED_DATASET_ID = DatasetId("ga-kestrel-library")
GROUNDED_WORKLOAD_VERSION = "1.0.0"


def _source(label: str, title: str, text: str) -> GroundedSource:
    return GroundedSource(label=label, title=title, text=text, provenance=SELF_AUTHORED_PROVENANCE)


#: Each passage is one factual sentence, authored to fit a single chunk, so naming a label names
#: its evidence exactly. Texts are distinct so no two passages deduplicate in the corpus.
GROUNDED_SOURCES: tuple[GroundedSource, ...] = (
    _source("book-loan", "Standard Book Loan", "A standard book loan lasts twenty eight days."),
    _source("renewal", "Renewal", "A borrowed item may be renewed once for fourteen days."),
    _source("late-fee", "Late Fee", "An overdue book accrues a late fee of fifty cents per day."),
    _source(
        "max-items", "Borrowing Limit", "A borrower may have at most fifteen items checked out."
    ),
    _source(
        "digital-loan",
        "E Book Licence",
        "E books are licensed for seven days and cannot be renewed.",
    ),
    _source(
        "interlibrary",
        "Interlibrary Loan",
        "Interlibrary items are borrowed for the period the supplying library sets.",
    ),
    _source(
        "conflict-loan-2023",
        "Loan 2023",
        "In 2023 the standard book loan was stated as thirty days.",
    ),
    _source("closed-date", "Closures", "The library is closed on public holidays."),
    _source(
        "rare-books",
        "Rare Books",
        "Rare books are consulted only in the supervised reading room.",
    ),
    _source(
        "governance-chair",
        "Board Chair",
        "The Library Board is chaired by a member of the public.",
    ),
    _source(
        "governance-term",
        "Board Term",
        "Board members serve a term of two years.",
    ),
    _source(
        "reading-hours", "Reading Room Hours", "The reading room opens at nine in the morning."
    ),
    _source(
        "cafe",
        "Cafe",
        "The cafe attached to the library serves coffee until four in the afternoon.",
    ),
)


def _case(case_id: str, **fields: object) -> GroundedCase:
    return GroundedCase(case_id=CaseId(case_id), **fields)  # type: ignore[arg-type]


def _development_cases() -> tuple[GroundedCase, ...]:
    return (
        _case(
            "ga-direct-loan",
            category=BenchmarkCategory.DIRECT_RETRIEVAL,
            split=CaseSplit.DEVELOPMENT,
            question="how long does a standard book loan last",
            rationale="a single-source fact: retrieval must find the loan term and the answer cite it",
            gold_labels=("book-loan",),
            required_evidence_labels=("book-loan",),
            required_facts=("twenty eight days",),
        ),
        _case(
            "ga-renewal",
            category=BenchmarkCategory.DIRECT_RETRIEVAL,
            split=CaseSplit.DEVELOPMENT,
            question="how long is a renewal period",
            rationale="a single-source fact distinct from the loan term",
            gold_labels=("renewal",),
            forbidden_labels=("book-loan",),
            required_evidence_labels=("renewal",),
            required_facts=("fourteen days",),
        ),
        _case(
            "ga-multi-loan-renew",
            category=BenchmarkCategory.MULTI_SOURCE,
            split=CaseSplit.DEVELOPMENT,
            question="how long is a book loan and how long is a renewal",
            rationale="two facts from two sources must both be retrieved, cited and stated",
            gold_labels=("book-loan", "renewal"),
            required_evidence_labels=("book-loan", "renewal"),
            required_facts=("twenty eight days", "fourteen days"),
        ),
        _case(
            "ga-digital-no-renew",
            category=BenchmarkCategory.CITATION_CHALLENGE,
            split=CaseSplit.DEVELOPMENT,
            question="can an e book be renewed and for how long is it licensed",
            rationale="the answer must not launder the renewal rule onto the e book licence",
            gold_labels=("digital-loan",),
            forbidden_labels=("renewal",),
            required_evidence_labels=("digital-loan",),
            required_facts=("seven days",),
            prohibited_claims=("E books may be renewed for fourteen days.",),
        ),
        _case(
            "ga-late-fee",
            category=BenchmarkCategory.DIRECT_RETRIEVAL,
            split=CaseSplit.DEVELOPMENT,
            question="what is the late fee for an overdue book",
            rationale="a numeric fact that must be retrieved and quoted accurately",
            gold_labels=("late-fee",),
            required_facts=("fifty cents",),
        ),
        _case(
            "ga-conflicting-loan",
            category=BenchmarkCategory.CONFLICTING,
            split=CaseSplit.DEVELOPMENT,
            question="what is the standard book loan period",
            rationale="two sources disagree; both must surface and the answer must not silently pick",
            gold_labels=("book-loan", "conflict-loan-2023"),
            required_facts=("twenty eight days",),
        ),
        _case(
            "ga-missing-dinosaur",
            category=BenchmarkCategory.MISSING_EVIDENCE,
            split=CaseSplit.DEVELOPMENT,
            question="how much does it cost to renew a dinosaur skeleton",
            rationale="the pool has nothing on dinosaurs; the only correct behaviour is to abstain",
            expected_behavior=ExpectedBehavior.ABSTAIN,
        ),
        _case(
            "ga-false-premise-quiet-floor",
            category=BenchmarkCategory.UNSUPPORTED_PREMISE,
            split=CaseSplit.DEVELOPMENT,
            question="why does the library enforce silence on its fourth floor",
            rationale="a false premise (there is no fourth floor); abstain rather than play along",
            expected_behavior=ExpectedBehavior.ABSTAIN,
        ),
        _case(
            "ga-ambiguous-how-long",
            category=BenchmarkCategory.AMBIGUOUS,
            split=CaseSplit.DEVELOPMENT,
            question="how long can I keep a book",
            rationale="a physical loan and an e book licence give different durations; qualify",
            expected_behavior=ExpectedBehavior.QUALIFY,
            gold_labels=("book-loan", "digital-loan"),
        ),
    )


def _heldout_cases() -> tuple[GroundedCase, ...]:
    return (
        _case(
            "ga-held-max-items",
            category=BenchmarkCategory.DIRECT_RETRIEVAL,
            split=CaseSplit.HELDOUT,
            question="how many items may a borrower have checked out",
            rationale="a held-out single-source fact",
            gold_labels=("max-items",),
            required_evidence_labels=("max-items",),
            required_facts=("fifteen items",),
        ),
        _case(
            "ga-held-interlibrary",
            category=BenchmarkCategory.DIRECT_RETRIEVAL,
            split=CaseSplit.HELDOUT,
            question="how long may an interlibrary item be kept",
            rationale="a held-out fact whose answer is a conditional, not a number",
            gold_labels=("interlibrary",),
            required_facts=("the period the supplying library sets",),
        ),
        _case(
            "ga-held-rare-books",
            category=BenchmarkCategory.DIRECT_RETRIEVAL,
            split=CaseSplit.HELDOUT,
            question="where may rare books be consulted",
            rationale="a held-out location fact",
            gold_labels=("rare-books",),
            required_evidence_labels=("rare-books",),
            required_facts=("supervised reading room",),
        ),
        _case(
            "ga-held-closed",
            category=BenchmarkCategory.DIRECT_RETRIEVAL,
            split=CaseSplit.HELDOUT,
            question="when is the library closed",
            rationale="a held-out fact with an obvious distractor in the reading-room-hours passage",
            gold_labels=("closed-date",),
            forbidden_labels=("reading-hours",),
            required_facts=("public holidays",),
        ),
        _case(
            "ga-held-missing-planet",
            category=BenchmarkCategory.MISSING_EVIDENCE,
            split=CaseSplit.HELDOUT,
            question="what is the distance between the library and the moon",
            rationale="a held-out question the pool cannot answer; abstain",
            expected_behavior=ExpectedBehavior.ABSTAIN,
        ),
    )


def development_grounded_workload() -> GroundedDataset:
    """Build the versioned grounded-answer workload from its self-authored passages."""
    return GroundedDataset(
        dataset_id=GROUNDED_DATASET_ID,
        version=GROUNDED_WORKLOAD_VERSION,
        sources=GROUNDED_SOURCES,
        cases=(*_development_cases(), *_heldout_cases()),
    )


def load_development_grounded_workload(
    *, now: Callable[[], datetime] | None = None
) -> tuple[Corpus, GroundedDataset]:
    """Build the corpus and the workload together, so their identifiers agree."""
    dataset = development_grounded_workload()
    corpus = build_corpus(dataset.sources, now=now)
    return corpus, dataset


def case_evidence(
    dataset: GroundedDataset, resolved: dict[str, tuple[SourceId, tuple[ChunkId, ...]]]
) -> dict[str, tuple[tuple[ChunkId, ...], tuple[ChunkId, ...], tuple[ChunkId, ...]]]:
    """Resolve each case's label sets into chunk identifiers.

    Returns a mapping from case id to ``(gold, forbidden, required)`` chunk tuples. A case with no
    gold evidence (an abstention) resolves to an empty gold set, which is exactly what the runner
    scores as "retrieval should find nothing".
    """
    out: dict[str, tuple[tuple[ChunkId, ...], tuple[ChunkId, ...], tuple[ChunkId, ...]]] = {}
    for case in dataset.cases:
        gold = tuple(resolved[label][1][0] for label in case.gold_labels)
        forbidden = tuple(resolved[label][1][0] for label in case.forbidden_labels)
        required = tuple(resolved[label][1][0] for label in case.required_evidence_labels)
        out[str(case.case_id)] = (gold, forbidden, required)
    return out
