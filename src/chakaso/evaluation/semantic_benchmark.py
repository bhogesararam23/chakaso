"""A versioned synthetic semantic-evaluation dataset and a deterministic scorer.

This is a development instrument, not a scientific benchmark, and it makes no claim about semantic
capability. It exists so that a *future* real semantic evaluator — one that does not exist in this
repository (see ADR-0019/ADR-0027) — has a fixed, versioned set of labelled claim/evidence cases to
be measured against, and so that the measurement itself (per-class precision/recall, a confusion
matrix, how unresolved and failed cases are handled) is written down and testable now.

The cases are hand-authored and small, and their "expected relation" is a human label, never a
machine judgement. The only judge available today is the caller-parameterised
:class:`~chakaso.grounding.semantic.FixtureSemanticJudge`; scoring it only exercises the arithmetic
of this harness — a fixture that is handed the expected relations scores perfectly, which measures
the scorer, not understanding. Two explicitly-named probe judges (support-all, always-uncertain)
show the scorer distinguishes a judge that over-claims from one that abstains. No number here
describes general language-understanding performance, and none is presented as if it did.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from chakaso.claims.model import Claim
from chakaso.core.hashing import short_sha256_hex
from chakaso.evidence import EvidencePack
from chakaso.grounding.semantic import (
    SemanticEvaluation,
    SemanticJudge,
    SemanticJudgeError,
    SemanticJudgement,
)
from chakaso.retrieval import ingest_text

__all__ = [
    "SEMANTIC_DATASET_VERSION",
    "ProbeSupportAllJudge",
    "ProbeUncertainJudge",
    "SemanticBenchmarkRun",
    "SemanticCase",
    "SemanticCaseOutcome",
    "dataset_fingerprint",
    "development_semantic_dataset",
    "format_semantic_report",
    "materialize_case",
    "score_semantic_judge",
    "semantic_report_json",
]

#: The version of this hand-built dataset. Adding, removing or re-labelling a case bumps it — a
#: measured difference against a different version is not a regression, it is a different question.
SEMANTIC_DATASET_VERSION: Final[str] = "dev-semantic-eval-0.1"

#: Every judgement a scored case can be assigned. Fixed order so metrics and confusion matrices are
#: deterministic regardless of how a dict happens to iterate.
JUDGEMENT_CLASSES: Final[tuple[SemanticJudgement, ...]] = (
    SemanticJudgement.SUPPORTED,
    SemanticJudgement.CONTRADICTED,
    SemanticJudgement.UNCERTAIN,
    SemanticJudgement.NOT_EVALUATED,
)

#: Judgements that mean "the evaluator did not settle the claim" — counted separately from a
#: verdict, because abstention is neither support nor contradiction.
_UNRESOLVED: Final[frozenset[SemanticJudgement]] = frozenset(
    {SemanticJudgement.UNCERTAIN, SemanticJudgement.NOT_EVALUATED}
)

#: A fixed clock. Chunk and claim identifiers are content-derived, so the dataset does not depend on
#: the wall clock, but a fixed time keeps materialisation byte-stable across runs.
_FIXED_TIME: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
_CLOCK: Final[Callable[[], datetime]] = lambda: _FIXED_TIME  # noqa: E731


@dataclass(frozen=True, slots=True)
class SemanticCase:
    """One labelled claim/evidence pair: what a real evaluator *should* be asked to decide.

    ``expected`` is the human-authored relation label, and ``category`` records the linguistic
    situation the case probes (negation, numbers/units, entity mismatch, conflicting evidence …).
    ``conflicting_evidence`` names, by index into ``evidence_texts``, the chunks that disagree when
    the label is a contradiction, so a judge that reports them can be checked for structured
    conflict rather than a bare verdict.
    """

    case_id: str
    category: str
    claim_text: str
    evidence_texts: tuple[str, ...]
    expected: SemanticJudgement
    conflicting_evidence: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id or not self.category or not self.claim_text:
            message = f"semantic case {self.case_id!r} is missing an id, category or claim"
            raise ValueError(message)
        if not self.evidence_texts:
            message = f"semantic case {self.case_id!r} must supply at least one evidence text"
            raise ValueError(message)
        for index in self.conflicting_evidence:
            if not 0 <= index < len(self.evidence_texts):
                message = f"semantic case {self.case_id!r} names an out-of-range conflict"
                raise ValueError(message)


def development_semantic_dataset() -> tuple[SemanticCase, ...]:
    """The versioned, hand-built synthetic set covering the situations a judge must distinguish."""
    return (
        SemanticCase(
            "support-direct",
            "direct_support",
            "Paris is the capital of France.",
            ("Paris is the capital of France.",),
            SemanticJudgement.SUPPORTED,
        ),
        SemanticCase(
            "contradict-direct",
            "direct_contradiction",
            "The window closes in April.",
            ("The window stays open until June and does not close in April.",),
            SemanticJudgement.CONTRADICTED,
        ),
        SemanticCase(
            "support-partial",
            "partial_support",
            "The report has three sections and was published in May.",
            ("The report was published in May.",),
            SemanticJudgement.UNCERTAIN,
        ),
        SemanticCase(
            "unsupported-none",
            "unsupported_claim",
            "The library is open on Sundays.",
            ("The library's opening hours vary by branch.",),
            SemanticJudgement.NOT_EVALUATED,
        ),
        SemanticCase(
            "irrelevant",
            "irrelevant_evidence",
            "Chakaso grounds answers in evidence.",
            ("The recipe calls for two eggs and a cup of flour.",),
            SemanticJudgement.NOT_EVALUATED,
        ),
        SemanticCase(
            "negation",
            "negation",
            "The treatment cures the disease.",
            ("The treatment does not cure the disease.",),
            SemanticJudgement.CONTRADICTED,
        ),
        SemanticCase(
            "numbers-units",
            "numbers_and_units",
            "The tower is 300 metres tall.",
            ("The tower is 300 feet tall.",),
            SemanticJudgement.UNCERTAIN,
        ),
        SemanticCase(
            "entity-mismatch",
            "entity_mismatch",
            "Marie Curie won the 1903 Nobel Prize in Physics.",
            ("Linus Pauling won the 1954 Nobel Prize in Chemistry.",),
            SemanticJudgement.NOT_EVALUATED,
        ),
        SemanticCase(
            "conflicting",
            "conflicting_evidence",
            "The meeting is at noon.",
            ("The meeting is at noon.", "The meeting is at 3pm."),
            SemanticJudgement.UNCERTAIN,
            conflicting_evidence=(0, 1),
        ),
        SemanticCase(
            "ambiguous",
            "ambiguous_or_insufficient",
            "It increased significantly.",
            ("The figure changed.",),
            SemanticJudgement.UNCERTAIN,
        ),
        SemanticCase(
            "multi-sentence",
            "multi_sentence_evidence",
            "The company was founded in 1998 and is headquartered in Seattle.",
            (
                "The company was founded in 1998 in a garage. "
                "It later moved its headquarters to Seattle.",
            ),
            SemanticJudgement.SUPPORTED,
        ),
        SemanticCase(
            "citation-mismatch",
            "citation_evidence_mismatch",
            "The capital is Paris.",
            ("Berlin is the capital of Germany.",),
            SemanticJudgement.NOT_EVALUATED,
        ),
    )


def dataset_fingerprint(cases: Iterable[SemanticCase] | None = None) -> str:
    """A short content hash of the dataset, so a change to any case is detectable."""
    items = tuple(cases) if cases is not None else development_semantic_dataset()
    parts = [
        f"{item.case_id}|{item.category}|{item.claim_text}|{item.expected.value}|"
        f"{list(item.evidence_texts)}|{list(item.conflicting_evidence)}"
        for item in items
    ]
    return short_sha256_hex("\n".join(parts), length=16)


def materialize_case(case: SemanticCase) -> tuple[Claim, EvidencePack]:
    """Build the ``Claim`` and ``EvidencePack`` for ``case`` deterministically.

    Identifiers are content-derived, so the same case always yields the same claim id and chunk ids
    — which is what lets a fixture be pointed at a specific claim by id, and what makes the run
    reproducible.
    """
    documents = [
        ingest_text(text, label=f"{case.case_id}:e{index}", now=_CLOCK)
        for index, text in enumerate(case.evidence_texts)
    ]
    chunks = tuple(chunk for document in documents for chunk in document.chunks)
    sources = [document.source for document in documents]
    pack = EvidencePack.of(chunks, sources)
    claim = Claim(answer_id=case.case_id, position=0, text=case.claim_text)
    return claim, pack


@dataclass(frozen=True, slots=True)
class SemanticCaseOutcome:
    """One scored case: the label, what the judge said, and whether the judge raised."""

    case_id: str
    category: str
    expected: SemanticJudgement
    predicted: SemanticJudgement
    correct: bool
    evaluator_error: bool


@dataclass(frozen=True, slots=True)
class SemanticBenchmarkRun:
    """The full result of scoring one judge over the dataset: per-case outcomes plus aggregates."""

    dataset_version: str
    fingerprint: str
    evaluator_name: str
    evaluator_version: str
    outcomes: tuple[SemanticCaseOutcome, ...]

    @property
    def size(self) -> int:
        return len(self.outcomes)

    @property
    def accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for outcome in self.outcomes if outcome.correct) / self.size

    @property
    def unresolved_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.predicted in _UNRESOLVED)

    @property
    def error_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.evaluator_error)

    @property
    def confusion(self) -> dict[str, int]:
        """Counts keyed ``"expected->predicted"``, present only where a pair actually occurred."""
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            key = f"{outcome.expected.value}->{outcome.predicted.value}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def per_class(self) -> dict[str, dict[str, float | int | None]]:
        """Per-class support / precision / recall, with ``None`` where a rate is undefined.

        Precision and recall are ``None`` when their denominator is zero — no verdict is reported as
        a spurious 0.0 when the class simply did not appear.
        """
        metrics: dict[str, dict[str, float | int | None]] = {}
        for klass in JUDGEMENT_CLASSES:
            tp = sum(1 for o in self.outcomes if o.expected is klass and o.predicted is klass)
            fp = sum(1 for o in self.outcomes if o.predicted is klass and o.expected is not klass)
            fn = sum(1 for o in self.outcomes if o.expected is klass and o.predicted is not klass)
            support = tp + fn
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / support if support else None
            metrics[klass.value] = {
                "support": support,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
            }
        return metrics

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": {
                "kind": "development",
                "subject": "semantic-evaluation",
                "dataset_version": self.dataset_version,
                "fingerprint": self.fingerprint,
                "size": self.size,
                "note": (
                    "Synthetic, hand-labelled cases over a fixture or probe judge. This measures the "
                    "scoring harness, not semantic understanding; no number here is general "
                    "language-understanding performance."
                ),
            },
            "evaluator": {"name": self.evaluator_name, "version": self.evaluator_version},
            "metrics": {
                "accuracy": self.accuracy,
                "unresolved_count": self.unresolved_count,
                "error_count": self.error_count,
                "per_class": self.per_class(),
            },
            "confusion": self.confusion,
            "outcomes": [
                {
                    "case_id": o.case_id,
                    "category": o.category,
                    "expected": o.expected.value,
                    "predicted": o.predicted.value,
                    "correct": o.correct,
                    "evaluator_error": o.evaluator_error,
                }
                for o in self.outcomes
            ],
        }


def score_semantic_judge(
    judge: SemanticJudge,
    *,
    cases: Sequence[SemanticCase] | None = None,
    evaluator_name: str | None = None,
    evaluator_version: str | None = None,
) -> SemanticBenchmarkRun:
    """Score ``judge`` over the dataset and return a deterministic run.

    The judge is asked per case; a :class:`SemanticJudgeError` (a failure or abstention) is recorded
    as ``not_evaluated`` with ``evaluator_error=True`` — never silently as support.
    """
    resolved_cases = tuple(cases) if cases is not None else development_semantic_dataset()
    outcomes: list[SemanticCaseOutcome] = []
    for case in resolved_cases:
        claim, pack = materialize_case(case)
        try:
            evaluation = judge.judge(claim, pack)
            predicted = evaluation.judgement
            raised = False
        except SemanticJudgeError:
            predicted = SemanticJudgement.NOT_EVALUATED
            raised = True
        outcomes.append(
            SemanticCaseOutcome(
                case_id=case.case_id,
                category=case.category,
                expected=case.expected,
                predicted=predicted,
                correct=predicted is case.expected,
                evaluator_error=raised,
            )
        )
    return SemanticBenchmarkRun(
        dataset_version=SEMANTIC_DATASET_VERSION,
        fingerprint=dataset_fingerprint(resolved_cases),
        evaluator_name=evaluator_name
        if evaluator_name is not None
        else str(getattr(judge, "name", "judge")),
        evaluator_version=(
            evaluator_version
            if evaluator_version is not None
            else str(getattr(judge, "version", ""))
        ),
        outcomes=tuple(outcomes),
    )


class ProbeSupportAllJudge:
    """A scoring probe, not an evaluator: it claims support for every claim.

    It exists only to show the harness surfaces a judge that over-claims — it scores poorly against
    the labels. It must never be mistaken for a semantic capability; it has none.
    """

    name = "probe-support-all"
    version = "0.1"

    def judge(self, claim: Claim, evidence: EvidencePack) -> SemanticEvaluation:  # noqa: ARG002
        return SemanticEvaluation(
            claim_id=claim.claim_id,
            judgement=SemanticJudgement.SUPPORTED,
            evaluator_name=self.name,
            evaluator_version=self.version,
            rationale="probe: always claims support",
        )


class ProbeUncertainJudge:
    """A scoring probe, not an evaluator: it abstains on every claim (all unresolved)."""

    name = "probe-uncertain-all"
    version = "0.1"

    def judge(self, claim: Claim, evidence: EvidencePack) -> SemanticEvaluation:  # noqa: ARG002
        return SemanticEvaluation(
            claim_id=claim.claim_id,
            judgement=SemanticJudgement.UNCERTAIN,
            evaluator_name=self.name,
            evaluator_version=self.version,
            rationale="probe: always abstains",
        )


def format_semantic_report(run: SemanticBenchmarkRun) -> str:
    """A deterministic text report that labels itself a development instrument."""
    lines = [
        "Semantic-evaluation development report — synthetic, hand-labelled cases; a fixture/probe "
        "score measures the harness, NOT semantic understanding.",
        f"evaluator: {run.evaluator_name} (version {run.evaluator_version or '-'})",
        f"dataset: {run.dataset_version}  fingerprint={run.fingerprint}  cases={run.size}",
        f"accuracy: {run.accuracy:.3f}  unresolved: {run.unresolved_count}  evaluator_errors: {run.error_count}",
        "per-class (support, precision, recall):",
    ]
    for klass, metrics in run.per_class().items():
        precision = _fmt_rate(metrics["precision"])
        recall = _fmt_rate(metrics["recall"])
        lines.append(
            f"  {klass:<12} support={metrics['support']!r}  precision={precision}  recall={recall}"
        )
    lines.append("confusion (expected->predicted):")
    if not run.confusion:
        lines.append("  (no cases)")
    for key, count in run.confusion.items():
        lines.append(f"  {key}: {count}")
    return "\n".join(lines)


def semantic_report_json(run: SemanticBenchmarkRun) -> str:
    """The machine-readable form of the same deterministic run."""
    return json.dumps(run.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)


def _fmt_rate(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
