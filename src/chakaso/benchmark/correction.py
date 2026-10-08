"""The correction development benchmark: versioned cases and curated fixtures.

This applies the retrieval benchmark's reproducibility discipline (ADR-0013) to the correction
decision. A ``CorrectionCase`` records a prior answer's claims (each with the status it held when
asserted and the evidence label it cited), the evidence to reassess against, which claims are
required to stay evidenced, the *expected* decision, and — for the cases structural matching cannot
reach — the per-claim semantic relation a fixture judge will assert. Every number a run produces is
tied to a case version and a dataset content fingerprint, so an edit to a judgement is a visible
version change, not a silent one.

These cases are **synthetic and hand-built**, written to exhibit the six outcomes
``docs/correction.md`` describes (retain, qualify, correct, needs-review, abstain) plus an
irrelevant-evidence control. They are a development instrument: they score a *decision rule*, not
real-world correction quality, and no claim about general performance is supportable from them.

Evidence is named by fixture *label*, resolved to real ``ChunkId``s by ingesting the shared corpus
at run time (see :mod:`chakaso.benchmark.correction_runner`), exactly as the retrieval benchmark
resolves gold evidence — so identifiers always match and the build stays deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chakaso.benchmark.errors import (
    BenchmarkError,
    DuplicateCaseError,
    InvalidCaseError,
    UnknownCaseError,
)
from chakaso.benchmark.identity import CaseId, DatasetId, content_version
from chakaso.claims.model import ClaimStatus
from chakaso.correction.decision import CorrectionDecision
from chakaso.grounding.semantic import SemanticJudgement

__all__ = [
    "CorrectionCase",
    "CorrectionClaim",
    "CorrectionDataset",
    "development_correction_benchmark",
]


@dataclass(frozen=True, slots=True)
class CorrectionClaim:
    """One claim of the prior answer, as the reassessment needs to see it."""

    text: str
    prior_status: ClaimStatus
    cite_label: str | None = None
    #: A relation a *fixture* semantic judge asserts, for outcomes structural matching cannot
    #: produce (contradicted, uncertain). ``None`` means the case runs through the structural
    #: evaluator — the honesty boundary of ADR-0015/0019 preserved in the data, not the code.
    semantic_relation: SemanticJudgement | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            message = "a correction claim must have text"
            raise InvalidCaseError(message)

    def to_material(self) -> dict[str, object]:
        """The deterministic view used to fingerprint a case."""
        return {
            "text": self.text,
            "prior_status": self.prior_status.value,
            "cite_label": self.cite_label,
            "semantic_relation": (
                self.semantic_relation.value if self.semantic_relation is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class CorrectionCase:
    """One reassessment scenario and the decision it is expected to reach."""

    case_id: CaseId
    description: str
    claims: tuple[CorrectionClaim, ...]
    reassessment_labels: tuple[str, ...]
    expected_decision: CorrectionDecision
    #: Positions of claims that must stay evidenced. ``None`` lets the runner default to the
    #: claims the prior answer asserted as supported; an explicit set is how a case exercises
    #: abstention on claims that were asserted but never evidenced.
    required_positions: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.description.strip():
            message = f"correction case {self.case_id} must have a description"
            raise InvalidCaseError(message)
        if not self.claims:
            message = f"correction case {self.case_id} must carry at least one claim"
            raise InvalidCaseError(message)
        object.__setattr__(
            self, "reassessment_labels", tuple(sorted(set(self.reassessment_labels)))
        )
        if self.required_positions is not None:
            positions = tuple(sorted(set(self.required_positions)))
            unknown = [p for p in positions if not 0 <= p < len(self.claims)]
            if unknown:
                message = (
                    f"correction case {self.case_id} requires evidence at unknown claim "
                    f"positions {unknown}"
                )
                raise InvalidCaseError(message)
            object.__setattr__(self, "required_positions", positions)

    @property
    def version(self) -> str:
        """A deterministic fingerprint of the case's definition."""
        return content_version(self.to_material())

    def to_material(self) -> dict[str, object]:
        """The canonical, order-insensitive view a version is computed from."""
        return {
            "case_id": str(self.case_id),
            "description": self.description,
            "claims": [claim.to_material() for claim in self.claims],
            "reassessment_labels": list(self.reassessment_labels),
            "required_positions": (
                list(self.required_positions) if self.required_positions is not None else None
            ),
            "expected_decision": self.expected_decision.value,
        }


@dataclass(frozen=True, slots=True)
class CorrectionDataset:
    """A versioned, identified set of correction cases."""

    dataset_id: DatasetId
    version: str
    cases: tuple[CorrectionCase, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            message = f"dataset {self.dataset_id} must declare a non-empty version"
            raise BenchmarkError(message)
        seen: set[CaseId] = set()
        for case in self.cases:
            if case.case_id in seen:
                message = f"dataset {self.dataset_id} has a duplicate case id {case.case_id}"
                raise DuplicateCaseError(message)
            seen.add(case.case_id)

    @property
    def content_fingerprint(self) -> str:
        """A fingerprint over every case's identity and version, sorted by identifier."""
        summary = [
            {"case_id": str(case.case_id), "version": case.version}
            for case in sorted(self.cases, key=lambda case: str(case.case_id))
        ]
        return content_version(summary)

    def case(self, case_id: CaseId | str) -> CorrectionCase:
        wanted = case_id if isinstance(case_id, CaseId) else CaseId(case_id)
        for case in self.cases:
            if case.case_id == wanted:
                return case
        message = f"dataset {self.dataset_id} has no case {wanted}"
        raise UnknownCaseError(message)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterable[CorrectionCase]:
        return iter(self.cases)


def _development_cases() -> tuple[CorrectionCase, ...]:
    return (
        CorrectionCase(
            case_id=CaseId("retain-supported"),
            description="New evidence agrees with the cited claim.",
            claims=(
                CorrectionClaim(
                    "The submission window closes on 14 April.",
                    ClaimStatus.SUPPORTED,
                    cite_label="deadlines",
                ),
            ),
            reassessment_labels=("deadlines",),
            expected_decision=CorrectionDecision.RETAIN,
        ),
        CorrectionCase(
            case_id=CaseId("correct-contradiction"),
            description="A supplied judgement finds the cited evidence contradicts the claim.",
            claims=(
                CorrectionClaim(
                    "The appeal review budget is 12000 credits.",
                    ClaimStatus.SUPPORTED,
                    cite_label="budget-2024",
                    semantic_relation=SemanticJudgement.CONTRADICTED,
                ),
            ),
            reassessment_labels=("budget-2024",),
            expected_decision=CorrectionDecision.CORRECT,
        ),
        CorrectionCase(
            case_id=CaseId("qualify-lost-support"),
            description="The evidence the claim relied on is gone, with nothing contradicting it.",
            claims=(
                CorrectionClaim(
                    "The appeal review budget is 12000 credits.",
                    ClaimStatus.SUPPORTED,
                    cite_label="budget-2024",
                ),
            ),
            # Reassessed against only the prior-year figure: the cited chunk is no longer supplied.
            reassessment_labels=("budget-2023",),
            expected_decision=CorrectionDecision.QUALIFY,
        ),
        CorrectionCase(
            case_id=CaseId("review-unsettled-conflict"),
            description="Evidence bears on the claim but a supplied judgement leaves it unsettled.",
            claims=(
                CorrectionClaim(
                    "The submission window closes on 14 April.",
                    ClaimStatus.SUPPORTED,
                    cite_label="deadlines",
                    semantic_relation=SemanticJudgement.UNCERTAIN,
                ),
            ),
            reassessment_labels=("deadlines",),
            expected_decision=CorrectionDecision.NEEDS_REVIEW,
        ),
        CorrectionCase(
            case_id=CaseId("abstain-total-grounding-loss"),
            description="Every required claim is asserted without supporting evidence.",
            claims=(
                CorrectionClaim(
                    "The appeals panel meets on Tuesdays.",
                    ClaimStatus.NOT_EVALUATED,
                    cite_label=None,
                ),
            ),
            # New evidence is real but supports nothing the answer asserted; the claim is required
            # explicitly, since it was never evidenced as supported.
            reassessment_labels=("parental-leave",),
            expected_decision=CorrectionDecision.ABSTAIN,
            required_positions=(0,),
        ),
        CorrectionCase(
            case_id=CaseId("retain-irrelevant-evidence"),
            description="New, unrelated evidence changes nothing about the supported claim.",
            claims=(
                CorrectionClaim(
                    "The submission window closes on 14 April.",
                    ClaimStatus.SUPPORTED,
                    cite_label="deadlines",
                ),
            ),
            reassessment_labels=("deadlines", "parental-leave"),
            expected_decision=CorrectionDecision.RETAIN,
        ),
    )


def development_correction_benchmark() -> CorrectionDataset:
    """The curated synthetic correction benchmark, version 1.0.0."""
    return CorrectionDataset(
        dataset_id=DatasetId("dev-correction"),
        version="1.0.0",
        cases=_development_cases(),
    )
