"""The answer record: the durable account of one produced answer.

An ``AnswerRecord`` is where the layers meet. A conversation turn says *what was said*; an
evidence pack says *what was supplied*; a claim set says *what was asserted*; grounding says
*how the evidence bears on those assertions*. Each of those already exists and each is the
authority on itself — this record does not duplicate them. It names them by identifier and adds
the facts that belong to no other layer: which model produced the answer, whether that model is
a development double, what retrieval configuration supplied the evidence, when the answer
happened, and — the piece that makes a correction a *history* rather than a replacement — which
earlier answer, if any, it supersedes.

Immutable, append-only, and validated on construction. A record that cites evidence it was not
given, or that claims to correct itself, cannot be built: those are exactly the impossible states
that would later make the correction history untrustworthy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from chakaso.answers.errors import InvalidAnswerError
from chakaso.answers.identity import AnswerId
from chakaso.claims.model import Claim
from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.evaluation import AnswerEvaluation

__all__ = ["AnswerRecord"]

#: A single shared empty mapping for defaulted metadata fields. It is immutable, so one
#: instance is safe across records, and a ``default_factory`` (not a bare default) is required
#: because Python 3.11 rejects an unhashable dataclass default.
_EMPTY_METADATA: Mapping[str, str] = MappingProxyType({})


def _require_aware(moment: datetime, field_name: str) -> datetime:
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        message = (
            f"{field_name} must be timezone-aware; a naive timestamp makes an answer's "
            "place in a history ambiguous"
        )
        raise InvalidAnswerError(message)
    return moment


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    """One answer, recorded with everything needed to cite, evaluate and supersede it."""

    answer_id: AnswerId
    conversation_id: str
    text: str
    created_at: datetime
    model_id: str
    model_is_development_double: bool
    claims: tuple[Claim, ...] = ()
    evidence_source_ids: tuple[SourceId, ...] = ()
    cited_source_ids: tuple[SourceId, ...] = ()
    cited_chunk_ids: tuple[ChunkId, ...] = ()
    retrieval_metadata: Mapping[str, str] = field(default_factory=lambda: _EMPTY_METADATA)
    evaluation: AnswerEvaluation | None = None
    turn_index: int | None = None
    correction_of: AnswerId | None = None
    provenance: Mapping[str, str] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        if not self.conversation_id.strip():
            message = "an answer record must name the conversation it belongs to"
            raise InvalidAnswerError(message)
        if not self.text.strip():
            message = f"an answer record must have text (answer {self.answer_id})"
            raise InvalidAnswerError(message)
        if not self.model_id.strip():
            message = (
                f"an answer record must name the model that produced it (answer {self.answer_id})"
            )
            raise InvalidAnswerError(message)
        _require_aware(self.created_at, "created_at")

        object.__setattr__(self, "evidence_source_ids", _dedup(self.evidence_source_ids))
        object.__setattr__(self, "cited_source_ids", _dedup(self.cited_source_ids))
        object.__setattr__(self, "cited_chunk_ids", _dedup(self.cited_chunk_ids))

        # ADR-0003's rule, applied to the record rather than the text: an answer cannot cite
        # evidence it was not given. Kept as a set check so a citation to an unsupplied source
        # is refused here, not silently recorded as a real reference.
        cited = set(self.cited_source_ids)
        evidence = set(self.evidence_source_ids)
        if not cited <= evidence:
            unexpected = sorted(str(source_id) for source_id in cited - evidence)
            message = (
                f"answer {self.answer_id} cites sources it was not given as evidence: {unexpected}"
            )
            raise InvalidAnswerError(message)

        # A claim's identity is (answer_id, position, text); a claim carried under a record whose
        # answer_id it does not name belongs to a different answer and would corrupt extraction.
        for claim in self.claims:
            if claim.answer_id != str(self.answer_id):
                message = (
                    f"claim at position {claim.position} names answer {claim.answer_id!r}, "
                    f"but is recorded under {str(self.answer_id)!r}"
                )
                raise InvalidAnswerError(message)

        if self.correction_of is not None and self.correction_of == self.answer_id:
            message = f"answer {self.answer_id} cannot correct itself"
            raise InvalidAnswerError(message)

    @property
    def is_correction(self) -> bool:
        """Whether this answer supersedes an earlier one."""
        return self.correction_of is not None

    @property
    def claim_count(self) -> int:
        """How many claims the answer was decomposed into."""
        return len(self.claims)


def _dedup(identifiers: Iterable[ChunkId | SourceId]) -> tuple[ChunkId | SourceId, ...]:
    """Preserve order while removing duplicate identifiers."""
    return tuple(dict.fromkeys(identifiers))
