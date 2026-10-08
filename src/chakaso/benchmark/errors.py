"""Errors raised by the benchmark package."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = ["BenchmarkError", "InvalidCaseError", "InvalidIdentifierError"]


class BenchmarkError(ChakasoError, ValueError):
    """Base class for every failure the benchmark layer raises deliberately.

    Deriving from :class:`ValueError` as well lets callers validate untrusted input —
    a benchmark file from disk, a case written by hand — with the built-in type they
    would already expect.
    """


class InvalidIdentifierError(BenchmarkError):
    """A case or dataset identifier is not a valid stable slug.

    Identifiers are the addresses a benchmark is referred to by across runs, machines
    and versions; a malformed one is not a formatting nicety but a case that cannot be
    cited back later.
    """


class InvalidCaseError(BenchmarkError):
    """A benchmark case definition is internally inconsistent or unusable.

    Raised for a blank query, an empty judgement, or an identifier listed as both gold
    and forbidden — a case whose "relevant" and "unacceptable" sets overlap is asking to
    score a chunk as simultaneously right and wrong, which no metric can mean anything by.
    """
