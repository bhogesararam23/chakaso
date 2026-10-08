"""Tests for answer identity.

The whole reason ``AnswerId`` exists as its own type is the ADR-0017 decision: an answer is a
per-event record, not a content address. So these tests pin the shape and validation it shares
with every other identifier, and the one property that distinguishes it — two answers get
different identifiers even when their would-be content is identical, which is what lets a
corrected answer stay distinguishable from the answer it supersedes.
"""

from __future__ import annotations

import pytest

from chakaso.answers import (
    ANSWER_ID_PREFIX,
    AnswerError,
    AnswerId,
    AnswerStoreError,
    InvalidAnswerError,
    new_answer_id,
)
from chakaso.core.errors import IdentifierError


def test_new_answer_id_has_the_prefixed_digest_shape() -> None:
    answer_id = new_answer_id()
    body = str(answer_id).removeprefix(ANSWER_ID_PREFIX)

    assert str(answer_id).startswith("ans_")
    assert len(body) == 16
    assert all(character in "0123456789abcdef" for character in body)


def test_two_answer_ids_differ_even_for_identical_answers() -> None:
    # ADR-0017: identity is per-event, so nothing about an answer's content collides two
    # records. A re-affirming correction must never overwrite what it corrects.
    assert new_answer_id() != new_answer_id()


def test_answer_id_is_constructed_from_a_valid_string_and_rejects_a_bad_one() -> None:
    assert str(AnswerId("ans_" + "0" * 16)) == "ans_" + "0" * 16
    with pytest.raises(IdentifierError):
        AnswerId.parse("ans_not-hex-nope")


def test_answer_id_sorts_and_hash_like_other_identifiers() -> None:
    first = AnswerId("ans_" + "0" * 16)
    second = AnswerId("ans_" + "1" * 16)

    assert sorted([second, first]) == [first, second]
    assert len({first, second, first}) == 2


def test_answer_errors_share_a_base_and_are_value_errors() -> None:
    assert issubclass(InvalidAnswerError, AnswerError)
    assert issubclass(AnswerStoreError, AnswerError)
    assert issubclass(AnswerError, ValueError)
