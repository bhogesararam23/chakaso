"""Errors shared across Chakaso.

A single base class lets a caller catch everything this project raises without
also catching a ``KeyError`` from an unrelated bug. Specific failures derive from
it and, where the standard library has a fitting built-in, from that too, so that
code written against ``ValueError`` still works.
"""

from __future__ import annotations

__all__ = ["ChakasoError", "IdentifierError"]


class ChakasoError(Exception):
    """Base class for every error Chakaso raises deliberately."""


class IdentifierError(ChakasoError, ValueError):
    """A string is not a valid evidence identifier.

    Deriving from :class:`ValueError` as well means callers that validate
    untrusted input can catch the built-in type they would already expect.
    """
