"""Correction-path errors that are not a store or model failure.

Reassessment reads a stored answer (whose own failures are ``chakaso.answers`` errors) and decides
against supplied evidence. The one error class the *composition* front edge needs is the refusal to
act on an ambiguous instruction: when a follow-up names more than one stored answer, the runtime will
not pick one, because choosing between them by a heuristic is precisely the model-like inference this
project refuses to fabricate (ADR-0003's discipline applied to intent rather than identifiers).
"""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = ["AmbiguousReassessmentTargetError", "CorrectionError"]


class CorrectionError(ChakasoError):
    """Base class for the correction path's own errors."""


class AmbiguousReassessmentTargetError(CorrectionError):
    """A follow-up named more than one stored answer, so no target can be chosen safely.

    It is raised before any reassessment runs and touches nothing: an ambiguous instruction is
    refused, not guessed at. The message lists the identifiers the trigger saw, so the caller can
    re-issue the request naming exactly one.
    """
