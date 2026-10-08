"""Tests for benchmark identity and content versioning.

Two properties carry the whole design: an identifier is a stable, human-written address
that survives edits, and a version is a deterministic fingerprint that changes when the
definition changes. Both must be provable, because a benchmark number is only as
trustworthy as the case-and-version it is recorded against.
"""

from __future__ import annotations

import pytest

from chakaso.benchmark import (
    CaseId,
    DatasetId,
    InvalidIdentifierError,
    canonical_json,
    content_version,
)

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def test_a_valid_slug_is_accepted_and_stringifies_to_itself() -> None:
    identifier = CaseId("direct-retrieval.deadline_1")

    assert str(identifier) == "direct-retrieval.deadline_1"
    assert identifier.value == "direct-retrieval.deadline_1"


def test_identifiers_compare_by_value() -> None:
    assert CaseId("a") == CaseId("a")
    assert CaseId("a") < CaseId("b")
    assert DatasetId("corpus") == DatasetId("corpus")


@pytest.mark.parametrize(
    "bad",
    ["", "-leading", ".leading", "UPPER", "with space", "with/slash", "tail-" * 40],
)
def test_an_invalid_slug_is_rejected(bad: str) -> None:
    with pytest.raises(InvalidIdentifierError, match="not a valid slug"):
        CaseId(bad)


def test_dataset_identifier_uses_the_same_rule() -> None:
    with pytest.raises(InvalidIdentifierError):
        DatasetId("Bad Id")


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_drops_whitespace() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_canonical_json_preserves_non_ascii_without_escaping() -> None:
    assert canonical_json({"t": "café"}) == '{"t":"café"}'


# ---------------------------------------------------------------------------
# Content versioning
# ---------------------------------------------------------------------------


def test_content_version_is_deterministic() -> None:
    definition = {"query": "when?", "gold": ["chk_0000000000000000"]}

    assert content_version(definition) == content_version(dict(reversed(list(definition.items()))))


def test_content_version_ignores_key_order_but_not_list_order() -> None:
    # Dict key order is normalized; list order is not, because a list's order is the
    # caller's to have already made canonical if it should not matter.
    assert content_version({"a": 1, "b": 2}) == content_version({"b": 2, "a": 1})
    assert content_version({"g": ["a", "b"]}) != content_version({"g": ["b", "a"]})


def test_content_version_changes_when_the_definition_changes() -> None:
    before = content_version({"query": "q", "gold": ["a"]})
    after = content_version({"query": "q", "gold": ["a", "b"]})

    assert before != after


def test_content_version_is_a_16_hex_fingerprint() -> None:
    version = content_version({"x": 1})

    assert len(version) == 16
    assert all(character in "0123456789abcdef" for character in version)
