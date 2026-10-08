"""Errors raised by the benchmark package."""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = [
    "BenchmarkError",
    "DuplicateCaseError",
    "IncompatibleBenchmarkError",
    "InvalidCaseError",
    "InvalidIdentifierError",
    "MalformedCaseEntryError",
    "UnknownCaseError",
]


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


class DuplicateCaseError(BenchmarkError):
    """A dataset contains the same case identifier twice.

    An identifier is the address a result is reported against; two cases sharing one make
    every per-case number ambiguous.
    """


class UnknownCaseError(BenchmarkError):
    """A case identifier was requested from a dataset that does not contain it."""


class MalformedCaseEntryError(BenchmarkError):
    """A serialized case entry is not a valid case.

    Raised for a JSONL line that is not an object, a missing required field, an unknown
    field, an unrecognized category or behaviour, or a value of the wrong shape — a
    benchmark that fails silently into a partially-read case would produce a number no one
    could trust.
    """


class IncompatibleBenchmarkError(BenchmarkError):
    """A run was asked to use a dataset whose identity or version it cannot honour.

    A result must be traceable to the exact judgements behind it; running against, or
    recording a result for, a dataset the caller did not intend is how a benchmark number
    silently stops meaning what its label says.
    """
