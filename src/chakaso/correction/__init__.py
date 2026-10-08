"""Correction: revising an earlier answer when new evidence changes what it was entitled to say.

This is the reassessment foundation described in ``docs/correction.md``. It is deliberately
a *foundation*, not a finished loop: it turns a prior answer's claims and a re-evaluation
against supplied evidence into an explicit per-claim disposition and a recorded outcome, so
that "recognise the contradiction, do not defend the earlier answer merely because it came
first" is a rule in code rather than a hope inside one generation call.

Two limits are stated up front rather than hidden:

* there is no language model, so nothing here generates the *prose* of a revised answer. The
  correction path decides and records what should change and why; producing the new wording is
  a model step that does not exist yet;
* a status is always "supported / unsupported / contradicted / uncertain *by the supplied
  evidence*" (ADR-0014, ADR-0015) — never a claim that something is true.

The reassessment engine builds on the claim, citation and grounding boundaries already in the
repo rather than inventing parallel ones.
"""

from __future__ import annotations

from chakaso.correction.decision import (
    ClaimDisposition,
    ClaimReassessment,
    CorrectionDecision,
    decide_claim,
)
from chakaso.correction.reassess import Reassessment, decide_answer, reassess

__all__ = [
    "ClaimDisposition",
    "ClaimReassessment",
    "CorrectionDecision",
    "Reassessment",
    "decide_answer",
    "decide_claim",
    "reassess",
]
