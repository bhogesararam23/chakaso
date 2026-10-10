"""Serializing an ``AnswerRecord`` to a durable form and back.

Durable storage must round-trip an answer faithfully — same identifier, same text, same
references, same correction link — or the persisted history would not be the history that was
written. This codec defines *what is persisted*: the answer's substance (identity, text, model,
claims, the evidence it was given, the sources and chunks it cited, retrieval/provenance
metadata, turn position and its ``correction_of`` link).

What is deliberately **not** persisted is the in-memory ``evaluation`` object. An answer's
evaluation is a view computed from its claims and evidence by a particular evaluator; freezing
it into durable history would capture a possibly-stale judgement of an ephemeral computation
rather than the fact that endures. The answer, its claims and its references are the record; the
evaluation is recomputed against the current evaluator when needed.

Decoding is strict (the storage ADR): a malformed or inconsistent stored document raises rather
than producing a corrupted ``AnswerRecord``, because silently repairing research history is
exactly what this project refuses to do.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime

from chakaso.answers.errors import AnswerSerializationError
from chakaso.answers.identity import AnswerId
from chakaso.answers.record import AnswerRecord
from chakaso.claims.model import Claim, ClaimStatus
from chakaso.core.identifiers import ChunkId, SourceId

__all__ = ["decode_document", "decode_json", "encode_document", "encode_json"]


def encode_document(record: AnswerRecord) -> dict[str, object]:
    """Convert ``record`` into a JSON-serializable mapping of its durable fields."""
    return {
        "answer_id": str(record.answer_id),
        "conversation_id": record.conversation_id,
        "text": record.text,
        "created_at": record.created_at.isoformat(),
        "model_id": record.model_id,
        "model_is_development_double": record.model_is_development_double,
        "claims": [_encode_claim(claim) for claim in record.claims],
        "evidence_source_ids": [str(source_id) for source_id in record.evidence_source_ids],
        "cited_source_ids": [str(source_id) for source_id in record.cited_source_ids],
        "cited_chunk_ids": [str(chunk_id) for chunk_id in record.cited_chunk_ids],
        "retrieval_metadata": dict(record.retrieval_metadata),
        "turn_index": record.turn_index,
        "correction_of": None if record.correction_of is None else str(record.correction_of),
        "provenance": dict(record.provenance),
    }


def decode_document(document: Mapping[str, object]) -> AnswerRecord:
    """Rebuild an ``AnswerRecord`` from a stored mapping, validating as it goes.

    Raises:
        AnswerSerializationError: a required field is missing, mistyped or unusable.
        Malformed identifiers and inconsistent records raise their own invariant types
        (``IdentifierError``, ``InvalidAnswerError``) from the layers that own them.
    """
    try:
        answer_id = AnswerId(_require(document, "answer_id"))
        created_at = datetime.fromisoformat(_require(document, "created_at"))
        correction_raw = document.get("correction_of")
        return AnswerRecord(
            answer_id=answer_id,
            conversation_id=_require(document, "conversation_id"),
            text=_require(document, "text"),
            created_at=created_at,
            model_id=_require(document, "model_id"),
            model_is_development_double=_require_bool(document, "model_is_development_double"),
            claims=tuple(_decode_claim(item) for item in _as_list(document.get("claims", []))),
            evidence_source_ids=tuple(
                SourceId(value) for value in _as_str_list(document.get("evidence_source_ids", []))
            ),
            cited_source_ids=tuple(
                SourceId(value) for value in _as_str_list(document.get("cited_source_ids", []))
            ),
            cited_chunk_ids=tuple(
                ChunkId(value) for value in _as_str_list(document.get("cited_chunk_ids", []))
            ),
            retrieval_metadata=_as_str_mapping(document.get("retrieval_metadata", {})),
            turn_index=_optional_int(document.get("turn_index")),
            correction_of=None if correction_raw is None else AnswerId(str(correction_raw)),
            provenance=_as_str_mapping(document.get("provenance", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        # A stored document that cannot be interpreted is a corruption to report, never a
        # record to guess at. Our own serialization errors carry the precise message and are
        # re-raised unchanged; identifier/answer invariant errors surface with their own type.
        if isinstance(exc, AnswerSerializationError):
            raise
        message = f"malformed stored answer document: {exc}"
        raise AnswerSerializationError(message) from exc


def encode_json(record: AnswerRecord) -> str:
    """Serialize a record to a deterministic JSON string (sorted keys)."""
    return json.dumps(encode_document(record), sort_keys=True, ensure_ascii=False)


def decode_json(payload: str) -> AnswerRecord:
    """Parse a JSON string produced by :func:`encode_json` back into a record.

    Raises:
        AnswerSerializationError: the payload is not valid JSON or not an object.
    """
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        message = f"stored answer is not valid JSON: {exc}"
        raise AnswerSerializationError(message) from exc
    if not isinstance(loaded, dict):
        message = "stored answer document must be a JSON object"
        raise AnswerSerializationError(message)
    return decode_document(loaded)


def _encode_claim(claim: Claim) -> dict[str, object]:
    return {
        "answer_id": claim.answer_id,
        "position": claim.position,
        "text": claim.text,
        "cited_evidence_ids": [str(chunk_id) for chunk_id in claim.cited_evidence_ids],
        "status": claim.status.value,
    }


def _decode_claim(item: object) -> Claim:
    if not isinstance(item, dict):
        message = "each claim must be an object"
        raise AnswerSerializationError(message)
    mapping: Mapping[str, object] = item
    return Claim(
        answer_id=_require(mapping, "answer_id"),
        position=_require_int(mapping, "position"),
        text=_require(mapping, "text"),
        cited_evidence_ids=tuple(
            ChunkId(value) for value in _as_str_list(mapping.get("cited_evidence_ids", []))
        ),
        status=ClaimStatus(_require(mapping, "status")),
    )


def _require(mapping: Mapping[str, object], key: str) -> str:
    if key not in mapping or mapping[key] is None:
        message = f"stored answer document is missing {key!r}"
        raise AnswerSerializationError(message)
    value = mapping[key]
    if not isinstance(value, str):
        message = f"stored answer field {key!r} must be a string"
        raise AnswerSerializationError(message)
    return value


def _require_bool(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        message = f"stored answer field {key!r} must be a boolean"
        raise AnswerSerializationError(message)
    return value


def _require_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    # bool is a subclass of int; a position is never a boolean, so reject it explicitly.
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"stored answer field {key!r} must be an integer"
        raise AnswerSerializationError(message)
    return value


def _optional_int(value: object) -> int | None:
    # bool is a subclass of int; a turn index is never a boolean, so reject it explicitly.
    if value is None or isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        message = "expected a list in the stored answer document"
        raise AnswerSerializationError(message)
    return value


def _as_str_list(value: object) -> list[str]:
    items: list[str] = []
    for item in _as_list(value):
        if not isinstance(item, str):
            message = "expected string identifiers in the stored answer document"
            raise AnswerSerializationError(message)
        items.append(item)
    return items


def _as_str_mapping(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        message = "expected an object for a metadata field"
        raise AnswerSerializationError(message)
    return {str(key): str(item) for key, item in value.items()}
