"""The correction record: an append-only account of what a reassessment decided.

Correction is only real if it is visible and historical. ``docs/correction.md`` is explicit:
the earlier answer stays in the record with a note that it was superseded, and history is
appended to, never rewritten — silently mutating an earlier answer destroys exactly the
evidence needed to evaluate whether correction works. This module is the smallest record that
honours that: it names the answer, the decision, the per-claim dispositions, the conflict it was
given, the evaluator that produced them, and, for a revision, the identifier of the answer it
supersedes.

It records a *decision*, not a rewritten answer. There is no language model, so the revised
prose does not exist; ``summary`` is a structural tally of what changed, never fluent text made
to look like a correction.
"""

from __future__ import annotations

from dataclasses import dataclass

from chakaso.claims.model import ClaimId
from chakaso.core.hashing import join_parts, short_sha256_hex
from chakaso.correction.decision import (
    ClaimDisposition,
    ClaimReassessment,
    CorrectionDecision,
)
from chakaso.correction.reassess import Reassessment
from chakaso.grounding import Contradiction

__all__ = ["CorrectionRecord", "record_reassessment"]

#: Prefix for a correction record's content-derived identifier, mirroring the other
#: content-derived identifiers in the system (ADR-0007).
RECORD_ID_PREFIX = "cor_"


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    """An immutable account of one reassessment and what it decided to do."""

    answer_id: str
    decision: CorrectionDecision
    claims: tuple[ClaimReassessment, ...]
    contradictions: tuple[Contradiction, ...]
    evaluator_name: str
    supersedes: str | None = None

    @property
    def record_id(self) -> str:
        """A deterministic identifier derived from what this record decided.

        The same reassessment of the same answer always yields the same identifier, so a chain
        of corrections can reference one another without a mutable registry — the same reason
        claim and chunk identifiers are content-derived (ADR-0007, ADR-0014).
        """
        claim_parts = ";".join(
            f"{r.claim_id}:{r.disposition.value}"
            for r in sorted(self.claims, key=lambda r: str(r.claim_id))
        )
        digest = short_sha256_hex(
            join_parts(
                self.answer_id,
                self.decision.value,
                self.evaluator_name,
                self.supersedes or "-",
                claim_parts,
            ),
            length=16,
        )
        return f"{RECORD_ID_PREFIX}{digest}"

    @property
    def is_revision(self) -> bool:
        """Whether this record supersedes an earlier answer (a non-RETAIN outcome that names one)."""
        return self.supersedes is not None and self.decision is not CorrectionDecision.RETAIN

    def claims_with(self, disposition: ClaimDisposition) -> tuple[ClaimId, ...]:
        """Claim identifiers decided to have that disposition, ordered by their string form."""
        return tuple(
            sorted(
                (r.claim_id for r in self.claims if r.disposition is disposition),
                key=str,
            )
        )

    @property
    def summary(self) -> str:
        """A structural tally of what changed — counts, not rewritten prose.

        It is deliberately a machine-shaped summary: producing the sentence a user would read
        is a model step that does not exist, and fabricating one here would be the appearance
        of a correction without a correction.
        """
        counts = ", ".join(
            f"{disposition.value}={sum(1 for r in self.claims if r.disposition is disposition)}"
            for disposition in ClaimDisposition
        )
        conflicts = len(self.contradictions)
        return f"{self.decision.value} ({counts}; {conflicts} recorded conflict(s))"


def record_reassessment(
    reassessment: Reassessment, *, supersedes: str | None = None
) -> CorrectionRecord:
    """Turn a reassessment into an append-only correction record.

    ``supersedes`` is the identifier of the answer this correction replaces (typically the prior
    answer's record id). It stays ``None`` for a reassessment that changed nothing, which is how
    the record keeps "the earlier answer was superseded" honest rather than implied.
    """
    return CorrectionRecord(
        answer_id=reassessment.answer_id,
        decision=reassessment.decision,
        claims=reassessment.claims,
        contradictions=reassessment.contradictions,
        evaluator_name=reassessment.evaluator_name,
        supersedes=supersedes,
    )
