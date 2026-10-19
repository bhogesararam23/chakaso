"""Pure, hand-checkable metric primitives for grounded-answer evaluation.

Each function computes one clearly-defined quantity with an explicit denominator, so a number in a
report can be recomputed by hand from the counts that fed it. They measure *distinct* properties on
purpose: the project forbids a single merged quality score, because one number hiding another is how
a confidently-wrong system looks fine.

Two rules govern every rate here:

* a rate over an empty denominator is ``0.0``, never a division by zero and never an invented 1.0.
  An answer with no claims did not achieve full coverage; a case with no citations made no valid
  ones. Reporting those as ``0.0`` keeps a missing measurement from reading as a success;
* a denominator is stated in the function's own docstring, and the tests check it against an
  independently-computed value rather than the implementation's arithmetic.

What these do *not* measure is as important as what they do: none of them is a semantic judgement.
Citation validity is set membership against supplied evidence, not proof a source supports a claim
(ADR-0015/ADR-0027); claim coverage is presence of an author-required string, not understanding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

__all__ = [
    "NO_DENOMINATOR",
    "citation_validity_rate",
    "mean",
    "prohibited_claim_rate",
    "rate",
    "required_fact_coverage",
    "unsupported_claim_rate",
]

#: An empty denominator yields this value: a missing measurement, never a fabricated success.
NO_DENOMINATOR: Final[float] = 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean of ``values``; ``0.0`` for an empty collection."""
    if not values:
        return NO_DENOMINATOR
    return sum(values) / len(values)


def rate(numerator: int, denominator: int) -> float:
    """``numerator / denominator``; ``0.0`` when the denominator is zero.

    The denominator is always the caller's, and always named at the call site so a rate can be read
    as "X of Y" rather than a bare fraction.
    """
    if denominator <= 0:
        return NO_DENOMINATOR
    return numerator / denominator


def citation_validity_rate(*, valid: int, unknown: int, irrelevant: int) -> float:
    """Valid citations over all citations *issued* (valid + unknown + irrelevant).

    Uncited claims are deliberately excluded from the denominator: a citation that was never made is
    an absence, not an invalid citation, and conflating the two would punish a claim that simply
    needs more evidence exactly like a reference to a chunk that does not exist. ``0.0`` when no
    citations were issued at all.
    """
    return rate(valid, valid + unknown + irrelevant)


def unsupported_claim_rate(*, unsupported: int, total_claims: int) -> float:
    """Claims that needed evidence and had none valid, over all claims in the answer.

    This is the fraction of an answer's assertions that the supplied evidence does not carry — the
    complement of evidence coverage over the claims actually produced. ``0.0`` when the answer made
    no claims (nothing to be unsupported about, which the abstention metric scores separately).
    """
    return rate(unsupported, total_claims)


def required_fact_coverage(*, facts_present: int, facts_required: int) -> float:
    """Required facts present in the answer, over the facts the case required.

    A micro denominator: summing across cases treats each required fact as one item. ``0.0`` for a
    case that required no facts (an abstention), which the abstention metric is the one to read.
    """
    return rate(facts_present, facts_required)


def prohibited_claim_rate(*, prohibited_present: int, prohibited_total: int) -> float:
    """Prohibited claims that appeared, over the claims the case forbade.

    Any non-zero value is a grounding failure the citation and coverage numbers could otherwise hide:
    an answer can cite the right source and still assert a claim the evidence forbids. ``0.0`` for a
    case that forbade nothing.
    """
    return rate(prohibited_present, prohibited_total)
