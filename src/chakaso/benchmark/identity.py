"""Benchmark identity: stable addresses and content-derived versions.

A benchmark case has to be *addressable* — referred to by the same name across runs,
machines and commits — and a benchmark result has to be traceable to *which version* of
a case produced it. Those are two different things and this module keeps them separate:

* an **identifier** (`CaseId`, `DatasetId`) is a stable, human-written slug. It is the
  name, and it does not change when the case's content is edited.
* a **version** (`content_version`) is a deterministic fingerprint of a case's or a
  dataset's definition. It changes the moment the definition changes, so a result can
  never silently be compared against a different set of judgements.

The split matters for reproducibility: editing a gold-answer list must not turn a case
into a different case (its identifier stays), but it *must* change its version, so that
an evaluation run records which definition it scored against.

Identifiers are deliberately readable slugs rather than the content-derived hexadecimal
identifiers used for evidence (`chakaso.core.identifiers`). A curated case is few and
small and is reviewed by a person; a stable name a human can type is worth more there
than an opaque digest, and a case is not deduplicated by content the way evidence is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final

from chakaso.benchmark.errors import InvalidIdentifierError
from chakaso.core.hashing import sha256_hex

__all__ = [
    "CaseId",
    "DatasetId",
    "canonical_json",
    "content_version",
]

#: A stable slug: lowercase, starting alphanumeric, then letters/digits/dot/dash/underscore.
#: 128 characters is generous for a readable name and bounded so an identifier cannot be
#: an unbounded blob smuggled into a path or a report column.
_SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


@dataclass(frozen=True, slots=True, order=True)
class CaseId:
    """The stable, addressable name of one benchmark case within a dataset."""

    value: str

    def __post_init__(self) -> None:
        _require_slug(self.value, "case")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class DatasetId:
    """The stable, addressable name of one benchmark dataset."""

    value: str

    def __post_init__(self) -> None:
        _require_slug(self.value, "dataset")

    def __str__(self) -> str:
        return self.value


def _require_slug(value: str, kind: str) -> None:
    if _SLUG_PATTERN.fullmatch(value) is None:
        message = (
            f"{kind} identifier {value!r} is not a valid slug. Expected lowercase, "
            "starting with a letter or digit, then letters, digits, '.', '_' or '-', "
            "at most 128 characters."
        )
        raise InvalidIdentifierError(message)


def canonical_json(value: object) -> str:
    """Serialize ``value`` to a deterministic JSON string for fingerprinting.

    Keys are sorted and whitespace is removed so two structurally equal definitions
    always produce the same bytes. Lists keep their order on purpose: callers pass
    already-sorted collections when order must not affect the version, and a list whose
    order is meaningful to the case is allowed to change the version when reordered.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_version(value: object) -> str:
    """Return a short deterministic fingerprint of ``value``.

    Used as a case or dataset version. It is a truncation of the SHA-256 of the
    canonical JSON, matching how evidence identifiers are truncated (ADR-0007): long
    enough to be collision-free at benchmark scale, short enough to sit in a report.
    """
    return sha256_hex(canonical_json(value))[:16]
