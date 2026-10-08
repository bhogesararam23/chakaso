"""Errors raised by the claim layer."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = ["ClaimError", "InvalidClaimError"]


class ClaimError(ChakasoError, ValueError):
    """Base class for failures in the claim layer."""


class InvalidClaimError(ClaimError):
    """A claim cannot be constructed — it has no text, a negative position, or no answer.

    A claim is the unit correction operates on, so a malformed one is not a nuisance; it
    is a unit that could not later be classified, cited or superseded without ambiguity.
    """
