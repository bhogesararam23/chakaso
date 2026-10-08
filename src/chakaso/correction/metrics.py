"""Correction metrics — structural rates over a caller-labelled set of reassessments.

``docs/correction.md`` names three measures, and all three can be gamed by a system that simply
agrees with every challenge or never agrees with one. They are therefore defined so both failure
directions show up separately, over cases whose *expected decision a caller or benchmark supplies*:

* **correction success rate** — of the cases where the evidence required the answer to change, how
  many the system decided to change the way the case expected;
* **unjustified persistence rate** — of those same cases, how many it left unchanged anyway
  (defending the earlier answer for its own sake);
* **unnecessary revision rate** — of the cases that did *not* require a change, how many it changed
  regardless.

These are decision-agreement rates, not a claim the revised content is correct: judging whether a
revised answer is right needs a semantic evaluator and a language model, neither of which exists.
A case's ``expected_decision`` is the benchmark's label, and the honest limit of the metric is
that it measures agreement with that label, nothing more. Rates whose denominator has no cases are
``None`` rather than a fabricated ``0.0`` or ``1.0``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from chakaso.correction.decision import CorrectionDecision
from chakaso.correction.record import CorrectionRecord

__all__ = ["CorrectionMetrics", "LabeledCorrection", "correction_metrics"]


@dataclass(frozen=True, slots=True)
class LabeledCorrection:
    """One reassessment paired with the decision a case expected it to reach."""

    decision: CorrectionDecision
    expected_decision: CorrectionDecision

    @property
    def produced_revision(self) -> bool:
        """Whether the system decided to change the answer at all."""
        return self.decision is not CorrectionDecision.RETAIN

    @property
    def requires_revision(self) -> bool:
        """Whether the case's own label says the answer should have changed."""
        return self.expected_decision is not CorrectionDecision.RETAIN

    @classmethod
    def from_record(
        cls, record: CorrectionRecord, *, expected_decision: CorrectionDecision
    ) -> Self:
        """Label a recorded reassessment against the decision a case expected."""
        return cls(decision=record.decision, expected_decision=expected_decision)


class CorrectionMetrics:
    """Correction rates and their raw counts, so a good rate cannot hide a thin denominator."""

    def __init__(
        self,
        *,
        cases: int,
        required_cases: int,
        successful_cases: int,
        persisted_cases: int,
        not_required_cases: int,
        unnecessarily_revised_cases: int,
    ) -> None:
        self.cases = cases
        self.required_cases = required_cases
        self.successful_cases = successful_cases
        self.persisted_cases = persisted_cases
        self.not_required_cases = not_required_cases
        self.unnecessarily_revised_cases = unnecessarily_revised_cases

    @property
    def correction_success_rate(self) -> float | None:
        """Required cases that reached the expected decision; ``None`` if there were none."""
        if not self.required_cases:
            return None
        return self.successful_cases / self.required_cases

    @property
    def unjustified_persistence_rate(self) -> float | None:
        """Required cases the system left unchanged; ``None`` if there were none."""
        if not self.required_cases:
            return None
        return self.persisted_cases / self.required_cases

    @property
    def unnecessary_revision_rate(self) -> float | None:
        """Not-required cases the system changed anyway; ``None`` if there were none."""
        if not self.not_required_cases:
            return None
        return self.unnecessarily_revised_cases / self.not_required_cases


def correction_metrics(cases: Iterable[LabeledCorrection]) -> CorrectionMetrics:
    """Aggregate labelled reassessments into the three correction rates and their counts."""
    collected = tuple(cases)
    return CorrectionMetrics(
        cases=len(collected),
        required_cases=sum(1 for c in collected if c.requires_revision),
        successful_cases=sum(
            1 for c in collected if c.requires_revision and c.decision is c.expected_decision
        ),
        persisted_cases=sum(
            1 for c in collected if c.requires_revision and not c.produced_revision
        ),
        not_required_cases=sum(1 for c in collected if not c.requires_revision),
        unnecessarily_revised_cases=sum(
            1 for c in collected if not c.requires_revision and c.produced_revision
        ),
    )
