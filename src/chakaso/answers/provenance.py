"""The historical evaluation snapshot of a completed turn.

An ``AnswerRecord`` carries two evaluation-shaped things, and they are deliberately
different (ADR-0028):

* :class:`~chakaso.evaluation.AnswerEvaluation` — the *live* view. It is computed on demand
  from the record's claims and evidence by whatever evaluator runs now, and it is **not**
  persisted (ADR-0021), because freezing a judgement made by a present-day evaluator into
  durable history would let the record claim what a later evaluator thinks rather than what
  the original turn actually reached.
* :class:`EvaluationProvenance` — the *historical* snapshot defined here. It records, as an
  immutable fact, the evaluation outcomes the turn itself produced: the citation-structure
  counts, the per-claim grounding statuses, the identity and version of the evaluator that
  produced them, and the evidence references found in the answer text that did not resolve.

Persisting the snapshot does not contradict "evaluation is recomputed." They answer different
questions: the snapshot says *what was decided then, by whom, against the evidence that was
supplied at the time*; the recomputed view says *what a current evaluator decides now*. A reader
can hold both and see where they differ — which is the whole point of making a grounded answer
auditable after the fact.

The dimensions stay separate, exactly as ``docs/evaluation.md`` insists: citation structure,
claim grounding and text-reference resolution are recorded as distinct facts, never merged into
one quality number. ``evidence_coverage`` and ``is_grounded`` below are *derived* from the stored
counts and statuses, mirroring :class:`~chakaso.evaluation.AnswerEvaluation`'s own definitions,
and they are labelled what they are — a structural support rate decided by the named evaluator —
never a truth verdict (ADR-0014, ADR-0015).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from chakaso.answers.errors import InvalidAnswerError
from chakaso.claims.model import ClaimId, ClaimStatus
from chakaso.evaluation import AnswerEvaluation
from chakaso.evidence import CitationResolution

__all__ = ["EvaluationProvenance", "GroundedClaim"]


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    """One claim's grounding status, as the turn's evaluator decided it.

    It is kept separate from the ``Claim`` itself on purpose: the claim is *what was asserted*,
    an immutable part of the answer; the status is *how the evidence at that moment bore on it*,
    a historical evaluation fact. Reassessing later never rewrites the claim, and it must not
    rewrite this recorded outcome either — a reassessment is a new event, not an edit.
    """

    claim_id: ClaimId
    status: ClaimStatus

    def __post_init__(self) -> None:
        if not isinstance(self.status, ClaimStatus):
            message = f"grounded claim {self.claim_id} has a non-ClaimStatus value: {self.status!r}"
            raise InvalidAnswerError(message)


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    """What a completed turn's evaluation actually reached, frozen as a historical record."""

    claims: int
    citations_made: int
    citations_valid: int
    citations_unknown: int
    citations_irrelevant: int
    required_claims: int
    uncited_claims: int
    grounded_claims: tuple[GroundedClaim, ...] = ()
    grounding_evaluator: str = "structural-0.1"
    grounding_evaluator_version: str = ""
    unresolved_reference_ids: tuple[str, ...] = ()
    malformed_reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Deterministic, de-duplicated order so a snapshot serializes and compares stably
        # regardless of the order the evaluator happened to iterate claims in.
        object.__setattr__(self, "grounded_claims", _ordered_claims(self.grounded_claims))
        for name in (
            "claims",
            "citations_made",
            "citations_valid",
            "citations_unknown",
            "citations_irrelevant",
            "required_claims",
            "uncited_claims",
        ):
            if getattr(self, name) < 0:
                message = f"provenance field {name!r} must not be negative"
                raise InvalidAnswerError(message)

        # A citation is exactly one of valid / unknown / irrelevant, so the three must sum to the
        # total. A stored snapshot that fails this is corrupt, not merely unusual, and is refused
        # rather than trusted.
        if self.citations_valid + self.citations_unknown + self.citations_irrelevant != (
            self.citations_made
        ):
            message = (
                "provenance citation counts are inconsistent: valid+unknown+irrelevant "
                f"({self.citations_valid}+{self.citations_unknown}+{self.citations_irrelevant}) "
                f"!= citations made ({self.citations_made})"
            )
            raise InvalidAnswerError(message)
        if self.uncited_claims > self.required_claims:
            message = (
                f"provenance reports {self.uncited_claims} uncited claims but only "
                f"{self.required_claims} were required to cite"
            )
            raise InvalidAnswerError(message)
        if len(self.grounded_claims) > self.claims:
            message = "provenance grounds more claims than the answer had"
            raise InvalidAnswerError(message)
        if not self.grounding_evaluator.strip():
            message = "a provenance snapshot must name its evaluator (even if only 'unknown')"
            raise InvalidAnswerError(message)

    # -- derived views (never stored, so they cannot drift from the facts) ----------

    @property
    def statuses(self) -> Mapping[ClaimId, ClaimStatus]:
        """The recorded grounding status per claim."""
        return MappingProxyType({item.claim_id: item.status for item in self.grounded_claims})

    def _with_status(self, status: ClaimStatus) -> tuple[ClaimId, ...]:
        return tuple(item.claim_id for item in self.grounded_claims if item.status is status)

    @property
    def supported_claims(self) -> tuple[ClaimId, ...]:
        return self._with_status(ClaimStatus.SUPPORTED)

    @property
    def unsupported_claims(self) -> tuple[ClaimId, ...]:
        return self._with_status(ClaimStatus.UNSUPPORTED)

    @property
    def contradicted_claims(self) -> tuple[ClaimId, ...]:
        return self._with_status(ClaimStatus.CONTRADICTED)

    @property
    def uncertain_claims(self) -> tuple[ClaimId, ...]:
        return self._with_status(ClaimStatus.UNCERTAIN)

    @property
    def not_evaluated_claims(self) -> tuple[ClaimId, ...]:
        return self._with_status(ClaimStatus.NOT_EVALUATED)

    @property
    def citation_precision(self) -> float:
        """Valid citations over citations made; 1.0 when nothing was cited (mirrors metrics)."""
        return self.citations_valid / self.citations_made if self.citations_made else 1.0

    @property
    def citation_recall(self) -> float:
        """Required claims carrying a valid citation; 1.0 when none were required."""
        if not self.required_claims:
            return 1.0
        return (self.required_claims - self.uncited_claims) / self.required_claims

    @property
    def evidence_coverage(self) -> float:
        """Supported claims over all claims; 0.0 when there are no claims (mirrors evaluation)."""
        if self.claims == 0:
            return 0.0
        return len(self.supported_claims) / self.claims

    @property
    def is_grounded(self) -> bool:
        """The structural grounding the turn reached — never a claim that the answer is true.

        It mirrors :attr:`chakaso.evaluation.AnswerEvaluation.is_grounded`: every claim supported,
        none contradicted, and no citation to unsupplied evidence. It is decided by the named,
        versioned evaluator recorded here, which today is a structural evaluator that cannot know
        entailment (ADR-0015), so ``True`` means "every claim cited supplied evidence", nothing
        more.
        """
        return (
            self.claims > 0
            and bool(self.grounded_claims)
            and all(item.status is ClaimStatus.SUPPORTED for item in self.grounded_claims)
            and not self.contradicted_claims
            and self.citations_unknown == 0
        )

    @property
    def has_unresolved_references(self) -> bool:
        """Whether the answer's text referenced evidence it was not given, or fumbled a format.

        This is the condition K-006 said nothing persisted. It is now a durable fact on any
        answer recorded through the store, so the rate of invented references can be computed
        from history rather than only from a live reply that is already gone.
        """
        return bool(self.unresolved_reference_ids or self.malformed_reference_ids)

    @classmethod
    def from_turn(
        cls,
        evaluation: AnswerEvaluation,
        resolution: CitationResolution,
    ) -> EvaluationProvenance:
        """Freeze the outcomes of a completed turn from its live evaluation and citation resolution.

        Both arguments are the objects the manager already holds at the end of a turn. Nothing is
        inferred or re-decided here — every field is read straight off what the turn produced, so
        the snapshot is a record of that evaluation, not a fresh judgement. ``chakaso.answers``
        depends on ``chakaso.evaluation`` in one direction only (the evaluation and evidence layers
        never import the answer store), so importing those types here introduces no cycle.
        """
        metrics = evaluation.citations
        grounding = evaluation.grounding
        return cls(
            claims=metrics.claims,
            citations_made=metrics.citations,
            citations_valid=metrics.valid,
            citations_unknown=metrics.unknown,
            citations_irrelevant=metrics.irrelevant,
            required_claims=metrics.required_claims,
            uncited_claims=metrics.uncited_claims,
            grounded_claims=tuple(
                GroundedClaim(claim_id, status) for claim_id, status in grounding.statuses.items()
            ),
            grounding_evaluator=grounding.evaluator_name,
            grounding_evaluator_version=grounding.evaluator_version,
            unresolved_reference_ids=resolution.unknown_identifiers,
            malformed_reference_ids=resolution.malformed_identifiers,
        )


def _ordered_claims(claims: tuple[GroundedClaim, ...]) -> tuple[GroundedClaim, ...]:
    """De-duplicate grounded claims by identifier and order them deterministically.

    Ordering by the claim's string identifier — not the evaluator's iteration order — is what lets
    a snapshot serialize and compare stably, so an identical turn yields an identical record.
    """
    unique: dict[ClaimId, GroundedClaim] = {}
    for item in claims:
        unique[item.claim_id] = item
    return tuple(unique[key] for key in sorted(unique, key=str))
