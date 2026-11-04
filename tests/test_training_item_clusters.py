"""Tests for the probe cluster listing: what it names, what it refuses, and the digest that binds it.

The listing exists so a clustering claim costs nothing to check (K-035), so the properties under test are the ones a
reader of a committed extract would rely on: that the pairing is digested, that a shuffled or edited pairing fails
rather than mis-attributing a verdict, and that an item present in one side and missing from the other is a refusal
instead of a dropped row.
"""

from __future__ import annotations

import pytest

from chakaso.training.context_continuations import ContextItem
from chakaso.training.item_clusters import (
    CLUSTER_BLOCK_SCHEMA,
    PHENOMENON_CLUSTER_UNIT,
    WORK_CLUSTER_UNIT,
    ClusterListing,
    ClusterListingError,
    cluster_block,
    keys_for_items,
    load_cluster_block,
)

_ROWS = (("a-1", "prose-w0"), ("b-2", "prose-w1"), ("c-3", "prose-w0"))

#: Hand-derived: sha256 of ``[["a-1","prose-w0"],["b-2","prose-w1"],["c-3","prose-w0"]]``.
_ROWS_DIGEST = "5aabdac1dc7b890c4376e8e9ba272102d7f1697840b889cff11ea8203947767c"


def _listing() -> ClusterListing:
    return ClusterListing(unit=WORK_CLUSTER_UNIT, rows=_ROWS)


# ------------------------------------------------------------------ the listing itself


def test_the_digest_binds_the_rows_in_order_not_only_their_content() -> None:
    assert _listing().digest == _ROWS_DIGEST
    moved = ClusterListing(
        unit=WORK_CLUSTER_UNIT, rows=(("b-2", "prose-w1"), ("a-1", "prose-w0"), ("c-3", "prose-w0"))
    )
    assert moved.digest != _ROWS_DIGEST


def test_a_cluster_reassignment_moves_the_digest_because_the_pairing_is_the_information() -> None:
    reassigned = ClusterListing(
        unit=WORK_CLUSTER_UNIT, rows=(("a-1", "prose-w1"), ("b-2", "prose-w1"), ("c-3", "prose-w0"))
    )
    assert reassigned.digest != _ROWS_DIGEST


def test_the_clusters_are_named_in_order_of_first_appearance() -> None:
    assert _listing().clusters == ("prose-w0", "prose-w1")


def test_an_item_listed_twice_is_refused_because_a_paired_row_must_be_one_item() -> None:
    with pytest.raises(ClusterListingError, match="names an item twice"):
        ClusterListing(unit=WORK_CLUSTER_UNIT, rows=(("a-1", "w0"), ("a-1", "w1")))


@pytest.mark.parametrize("unit", ["", "book", None])
def test_an_unnamed_cluster_unit_is_refused(unit: object) -> None:
    with pytest.raises(ClusterListingError, match="cluster unit"):
        ClusterListing(unit=str(unit), rows=_ROWS)  # type: ignore[arg-type]


def test_an_empty_listing_is_refused_because_no_items_means_no_measurement() -> None:
    with pytest.raises(ClusterListingError, match="zero items"):
        ClusterListing(unit=WORK_CLUSTER_UNIT, rows=())


def test_a_blank_cluster_name_is_refused_rather_than_becoming_an_unknown_work() -> None:
    with pytest.raises(ClusterListingError, match="both an item identifier and a cluster name"):
        ClusterListing(unit=WORK_CLUSTER_UNIT, rows=(("a-1", "  "),))


# ------------------------------------------------------------------ serialization


def test_the_block_states_its_schema_counts_and_digest_and_round_trips_exactly() -> None:
    block = _listing().as_dict()
    assert block["schema_version"] == CLUSTER_BLOCK_SCHEMA
    assert block["unit"] == WORK_CLUSTER_UNIT
    assert block["items"] == 3
    assert block["clusters"] == 2
    assert load_cluster_block(block) == _listing()


def test_a_block_whose_stored_digest_does_not_recompute_is_refused() -> None:
    block = _listing().as_dict()
    block["rows"][1][1] = "prose-w0"
    with pytest.raises(
        ClusterListingError, match="the pairing of items to clusters is not the one"
    ):
        load_cluster_block(block)


def test_a_block_that_counts_items_wrong_is_refused_even_when_its_digest_matches() -> None:
    block = _listing().as_dict()
    del block["items"]
    with pytest.raises(ClusterListingError, match="item count as None"):
        load_cluster_block(block)


def test_a_block_with_no_rows_is_refused_rather_than_read_as_an_empty_pairing() -> None:
    block = _listing().as_dict()
    block["rows"] = []
    with pytest.raises(ClusterListingError, match="no rows"):
        load_cluster_block(block)


def test_a_row_that_is_not_a_pair_is_refused() -> None:
    block = _listing().as_dict()
    block["rows"][0] = ["a-1"]
    with pytest.raises(ClusterListingError, match="item identifier and its cluster"):
        load_cluster_block(block)


def test_a_block_from_another_schema_is_refused_by_name() -> None:
    block = _listing().as_dict()
    block["schema_version"] = "chakaso-probe-clusters/2"
    with pytest.raises(ClusterListingError, match="declares schema"):
        load_cluster_block(block)


def test_a_bare_list_is_not_a_cluster_block() -> None:
    with pytest.raises(ClusterListingError, match="not a bare list"):
        load_cluster_block([["a-1", "prose-w0"]])


# ------------------------------------------------------------------ built from items


def _item(identifier: str, work: str) -> ContextItem:
    return ContextItem(
        identifier=identifier,
        register="prose",
        document=f"{work}/part-0001",
        work=work,
        offset=0,
        prefix=(1, 2),
        true_continuation=(3, 4),
        distractors=(("prior_span", 0, (5, 6)),),
    )


def test_cluster_block_reads_the_work_off_each_item_in_the_order_it_was_given() -> None:
    block = cluster_block(
        [_item("a-1", "prose-w0"), _item("b-2", "prose-w1")], unit=WORK_CLUSTER_UNIT
    )
    assert block["rows"] == [["a-1", "prose-w0"], ["b-2", "prose-w1"]]
    assert load_cluster_block(block).unit == WORK_CLUSTER_UNIT


def test_an_item_with_no_work_is_refused_rather_than_listed_under_an_empty_cluster() -> None:
    with pytest.raises(ClusterListingError, match="carries no work"):
        cluster_block([_item("a-1", "")], unit=WORK_CLUSTER_UNIT)


def test_authored_pairs_are_listed_under_the_phenomenon_because_no_work_lies_behind_them() -> None:
    block = cluster_block([_item("a-1", "auxiliary_agreement")], unit=PHENOMENON_CLUSTER_UNIT)
    assert block["unit"] == PHENOMENON_CLUSTER_UNIT


# ------------------------------------------------------------------ alignment


def test_keys_follow_the_callers_item_order_not_the_listings() -> None:
    keys = keys_for_items(_listing(), ["c-3", "a-1", "b-2"])
    assert keys == ["prose-w0", "prose-w0", "prose-w1"]


def test_an_item_scored_but_not_listed_is_refused_rather_than_dropped() -> None:
    with pytest.raises(ClusterListingError, match="does not name it"):
        keys_for_items(_listing(), ["a-1", "b-2", "z-9"])


def test_an_item_listed_but_not_scored_is_refused_because_the_sides_would_differ() -> None:
    with pytest.raises(ClusterListingError, match="the outcomes carry 2"):
        keys_for_items(_listing(), ["a-1", "b-2"])


def test_the_same_item_twice_in_the_outcomes_is_refused_rather_than_counted_twice() -> None:
    with pytest.raises(ClusterListingError, match="twice; a paired row is one item"):
        keys_for_items(_listing(), ["a-1", "a-1", "b-2"])
