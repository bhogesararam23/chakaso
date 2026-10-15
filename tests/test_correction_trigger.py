"""Tests for the deterministic reassessment trigger (P21's front edge).

The trigger detects *intent* only — it never runs reassessment — so these tests pin down that a
follow-up is read as a reassessment request exactly when, and only when, it explicitly names a stored
answer. They are the independent half of "intent detection is separately testable from execution": no
store, no model, no evidence, no side effects — just a deterministic classification of a string.
"""

from __future__ import annotations

import pytest

from chakaso.answers import AnswerId, new_answer_id
from chakaso.correction import (
    DeterministicReassessmentTrigger,
    ReassessmentTriggerKind,
    detect_reassessment_request,
)

_ID_A = "ans_0000000000000000"
_ID_B = "ans_1111111111111111"


def test_a_follow_up_naming_one_answer_is_an_explicit_target() -> None:
    decision = detect_reassessment_request(f"please reconsider {_ID_A} against the new source")

    assert decision.kind is ReassessmentTriggerKind.EXPLICIT
    assert decision.triggers_reassessment is True
    assert decision.is_ambiguous is False
    assert decision.answer_id == AnswerId(_ID_A)
    assert decision.references == (AnswerId(_ID_A),)


def test_an_ordinary_follow_up_names_nothing_and_does_not_trigger() -> None:
    # A plain continuation must stay an ordinary conversational turn, never a guessed correction.
    decision = detect_reassessment_request("and what about the deadline?")

    assert decision.kind is ReassessmentTriggerKind.NO_TARGET
    assert decision.triggers_reassessment is False
    assert decision.answer_id is None
    assert decision.references == ()


def test_an_empty_message_is_no_target() -> None:
    assert detect_reassessment_request("").kind is ReassessmentTriggerKind.NO_TARGET


def test_two_distinct_answers_are_ambiguous_and_carry_no_target() -> None:
    decision = detect_reassessment_request(f"compare {_ID_A} with {_ID_B}")

    assert decision.kind is ReassessmentTriggerKind.AMBIGUOUS
    assert decision.is_ambiguous is True
    assert decision.triggers_reassessment is False
    assert decision.answer_id is None  # an ambiguous outcome never pretends to have chosen one
    assert decision.references == (AnswerId(_ID_A), AnswerId(_ID_B))


def test_the_same_answer_named_twice_is_one_unambiguous_target() -> None:
    decision = detect_reassessment_request(f"reassess {_ID_A}, i mean {_ID_A}")

    assert decision.kind is ReassessmentTriggerKind.EXPLICIT
    assert decision.answer_id == AnswerId(_ID_A)
    assert decision.references == (AnswerId(_ID_A),)


def test_evidence_and_source_identifiers_are_not_answer_references() -> None:
    # Only the ``ans_`` namespace addresses an answer; a chunk or source id is not a target, so a
    # normal grounded turn that cites evidence is never read as a reassessment request.
    chunk = "chk_abcdef0123456789"
    source = "src_0123456789abcdef"

    for message in (f"look again at {chunk}", f"the source {source} matters"):
        assert detect_reassessment_request(message).kind is ReassessmentTriggerKind.NO_TARGET


@pytest.mark.parametrize(
    "malformed",
    [
        "ans_123",  # too short
        "ans_GGGGGGGGGGGGGGGG",  # not hexadecimal
        "ans_000000000000000000",  # too long: a run of more than sixteen hex digits
        "ANS_0000000000000000",  # wrong case in the prefix
        "preans_0000000000000000",  # not a standalone token
    ],
)
def test_a_malformed_or_embedded_identifier_is_not_a_target(malformed: str) -> None:
    assert detect_reassessment_request(f"reconsider {malformed} now").kind is (
        ReassessmentTriggerKind.NO_TARGET
    )


def test_detection_is_deterministic_and_repeatable() -> None:
    trigger = DeterministicReassessmentTrigger()
    message = f"reassess {_ID_A}"

    first = trigger.detect(message)
    second = trigger.detect(message)

    assert first == second


def test_a_real_generated_identifier_triggers_and_is_returned_typed() -> None:
    answer_id = new_answer_id()

    decision = detect_reassessment_request(f"reconsider {answer_id}")

    assert decision.kind is ReassessmentTriggerKind.EXPLICIT
    assert decision.answer_id == answer_id
