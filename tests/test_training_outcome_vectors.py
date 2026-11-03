"""The per-item outcome encoding that lets a committed probe extract be re-paired (K-032).

The fixtures are two items and one group, small enough that every expected bit string and key listing below can be
read off the fixture by hand. That is the point: K-032 was filed because a committed extract left a reader able to
check what was *reported* and unable to recompute the test, so an encoding whose output nobody can re-derive is not
a fix.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import pytest

from chakaso.training.outcome_vectors import (
    OutcomeEncodingError,
    compact_probe_result,
    decode_outcomes,
    encode_outcomes,
    outcome_layout,
    restore_probe_result,
)

_ROWS = [
    ["aux-0001", "auxiliary_agreement", 1.0],
    ["aux-0002", "auxiliary_agreement", 0.0],
    ["det-0001", "determiner_noun_agreement", 1.0],
]
_GROUPS = {
    "pmi_prior_span": [["ctx-a", 1.0], ["ctx-b", 0.0]],
    "pmi_reversed": [["ctx-a", 0.0], ["ctx-b", 0.0]],
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_the_layout_of_each_shape_the_probe_runners_write_is_named() -> None:
    assert outcome_layout(_ROWS) == "rows"
    assert outcome_layout(_GROUPS) == "groups"


def test_a_block_that_is_not_per_item_outcomes_is_left_alone() -> None:
    # The extractor must not mistake a table for an outcome vector: `by_phenomenon` is a mapping of mappings, a
    # list of bare numbers carries no item identity to pair on, and a value outside {0.0, 1.0} is not a verdict.
    assert outcome_layout({"auxiliary_agreement": {"correct": 40}}) is None
    assert outcome_layout([1.0, 0.0, 1.0]) is None
    assert outcome_layout([["a", 0.4], ["b", 1.0]]) is None
    assert outcome_layout([]) is None
    assert outcome_layout([["a", "b", 1.0], ["c", "d", 2.0]]) is None
    assert outcome_layout({"g": [["a", 1.0]], "h": []}) is None


def test_an_encoded_vector_round_trips_to_the_value_it_came_from() -> None:
    for value in (_ROWS, _GROUPS):
        assert decode_outcomes(encode_outcomes(value)) == value


def test_the_recorded_digest_is_of_the_value_the_extractor_replaced() -> None:
    for value in (_ROWS, _GROUPS):
        assert (
            encode_outcomes(value)["compact_v1"]["canonical_json_sha256"]
            == sha256(_canonical(value).encode("utf-8")).hexdigest()
        )


def test_the_bits_are_one_character_per_item_in_the_row_order() -> None:
    rows = encode_outcomes(_ROWS)["compact_v1"]
    assert rows["layout"] == "rows"
    assert rows["vectors"][""] == "101"
    assert rows["keys"] == [
        ["aux-0001", "auxiliary_agreement"],
        ["aux-0002", "auxiliary_agreement"],
        ["det-0001", "determiner_noun_agreement"],
    ]

    grouped = encode_outcomes(_GROUPS)["compact_v1"]
    assert grouped["layout"] == "groups"
    assert grouped["vectors"] == {"pmi_prior_span": "10", "pmi_reversed": "00"}
    assert grouped["keys"] == {
        "pmi_prior_span": ["ctx-a", "ctx-b"],
        "pmi_reversed": ["ctx-a", "ctx-b"],
    }


def test_decoding_refuses_a_block_whose_bits_do_not_match_its_keys() -> None:
    block = encode_outcomes(_ROWS)
    block["compact_v1"]["vectors"][""] = "1010"
    with pytest.raises(OutcomeEncodingError, match="3 items"):
        decode_outcomes(block)


def test_decoding_refuses_a_digest_that_does_not_match_what_it_rebuilds() -> None:
    block = encode_outcomes(_ROWS)
    block["compact_v1"]["keys"][0][0] = "aux-9999"
    with pytest.raises(OutcomeEncodingError, match="digest"):
        decode_outcomes(block)


def _probe_document(models: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_set": {"digest": "d", "name": "n", "version": "2.0.0"},
        "models": {
            name: {"step": 0, "result": {"accuracy": 0.5, "outcomes": outcomes}}
            for name, outcomes in models.items()
        },
    }


def test_a_probe_result_keeps_one_key_listing_for_models_asked_the_same_items() -> None:
    document = _probe_document({"m1": _ROWS, "m2": [list(row) for row in _ROWS]})
    compacted = compact_probe_result(document)
    assert compacted["outcome_items"]["result.outcomes"] == [[row[0], row[1]] for row in _ROWS]
    for name in ("m1", "m2"):
        block = compacted["models"][name]["result"]["outcomes"]["compact_v1"]
        assert "keys" not in block
        assert block["keys_shared"] == "outcome_items.result.outcomes"
        assert block["vectors"][""] == "101"
    # Nothing else moved: the accuracy and the pair set are still whole, and no stub block appeared.
    assert compacted["models"]["m1"]["result"]["accuracy"] == 0.5
    assert compacted["pair_set"] == document["pair_set"]
    restored = restore_probe_result(compacted)
    assert restored["models"]["m1"]["result"]["outcomes"] == _ROWS
    assert restored["models"]["m2"]["result"]["outcomes"] == _ROWS


def test_models_asked_different_items_keep_their_own_keys() -> None:
    other = [["aux-0001", "auxiliary_agreement", 1.0], ["aux-0002", "auxiliary_agreement", 1.0]]
    compacted = compact_probe_result(_probe_document({"m1": _ROWS, "m2": other}))
    assert "outcome_items" not in compacted
    for name, value in (("m1", _ROWS), ("m2", other)):
        block = compacted["models"][name]["result"]["outcomes"]["compact_v1"]
        assert block["keys"] == [[row[0], row[1]] for row in value]
        assert "keys_shared" not in block
    restored = restore_probe_result(compacted)
    assert restored["models"]["m1"]["result"]["outcomes"] == _ROWS
    assert restored["models"]["m2"]["result"]["outcomes"] == other


def test_compacting_an_already_compacted_document_is_idempotent() -> None:
    # The extractor refuses to overwrite a committed extract, so a re-run always works from the whole report.
    # Idempotence is still what a reader wants to see: a compact block is recognised as one and left alone, so a
    # document cannot be compacted twice into a smaller and smaller shape.
    document = _probe_document({"m1": _ROWS, "m2": [list(row) for row in _ROWS]})
    once = compact_probe_result(document)
    assert compact_probe_result(once) == once
    assert restore_probe_result(once)["models"]["m1"]["result"]["outcomes"] == _ROWS


def test_a_model_with_several_outcome_blocks_compacts_each_one_under_its_own_path() -> None:
    document = {
        "models": {
            "m1": {"result": {"outcomes": _ROWS, "secondary": {"nested": _GROUPS}}},
            "m2": {
                "result": {"outcomes": [list(r) for r in _ROWS], "secondary": {"nested": _GROUPS}}
            },
        }
    }
    compacted = compact_probe_result(document)
    assert set(compacted["outcome_items"]) == {
        "result.outcomes",
        "result.secondary.nested",
    }
    restored = restore_probe_result(compacted)
    assert restored["models"]["m1"]["result"]["outcomes"] == _ROWS
    assert restored["models"]["m2"]["result"]["secondary"]["nested"] == _GROUPS


def test_a_document_without_the_models_block_is_returned_unchanged() -> None:
    document = {"transformer": {"test": {"bits_per_character": 2.05}}}
    assert compact_probe_result(document) == document
    assert "outcome_items" not in document


def test_a_ten_model_probe_result_compacts_to_a_fraction_of_its_rows() -> None:
    # K-032's measured shape: E2626 scored ten models on the same 800 pairs, 56 KB of rows per model. One shared
    # item listing plus one bit per item per model is what makes the whole array committable under the 1 MiB bound
    # instead of stubbed away from it.
    rows = [
        [f"{phenomenon}-{index:04d}", phenomenon, 1.0 if index % 2 else 0.0]
        for index, phenomenon in enumerate(
            ["auxiliary_agreement", "determiner_noun_agreement"] * 400
        )
    ]
    models = {f"step{step}": [list(row) for row in rows] for step in range(200, 2200, 200)}
    document = _probe_document(models)
    whole = len(_canonical(document))
    compacted = compact_probe_result(document)
    assert len(_canonical(compacted)) < whole / 8
    restored = restore_probe_result(compacted)
    for name in models:
        assert restored["models"][name]["result"]["outcomes"] == rows
