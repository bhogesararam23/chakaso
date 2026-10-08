"""Tests for the evaluation metric functions.

These assert the arithmetic of each metric on hand-built ranked lists and relevance sets.
They are *not* a benchmark and report no result about the system; they check that the
functions compute their definitions correctly so that, once a real benchmark supplies the
relevance judgements, the numbers will mean what they claim to mean.
"""

from __future__ import annotations

import pytest

from chakaso.evaluation import (
    duplicate_count,
    mean_reciprocal_rank,
    measure_latency,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unresolved_reference_count,
)
from chakaso.evidence import CitationResolution

# ---------------------------------------------------------------------------
# Recall / precision
# ---------------------------------------------------------------------------


def test_recall_at_k_counts_relevant_hits_over_total_relevant() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = frozenset({"b", "x"})

    assert recall_at_k(retrieved, relevant, k=2) == 0.5  # only b is in the top 2
    assert recall_at_k(retrieved, relevant, k=4) == 0.5  # x never retrieved
    assert recall_at_k(["b", "x"], relevant, k=2) == 1.0


def test_recall_with_no_relevant_items_is_zero_not_fabricated() -> None:
    assert recall_at_k(["a"], frozenset(), k=3) == 0.0


def test_precision_at_k_divides_by_the_window_size() -> None:
    retrieved = ["a", "b", "c"]
    relevant = frozenset({"a", "c"})

    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # [a, b] -> one hit of two
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_precision_of_an_empty_window_is_zero() -> None:
    assert precision_at_k([], frozenset({"a"}), k=3) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k])
def test_k_below_one_is_rejected(metric: object) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        metric(["a"], frozenset({"a"}), k=0)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_of_the_first_relevant_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], frozenset({"b"})) == 0.5
    assert reciprocal_rank(["a", "b"], frozenset({"z"})) == 0.0


def test_mean_reciprocal_rank_averages_runs() -> None:
    runs = [(["a", "b"], frozenset({"b"})), (["x", "y"], frozenset({"z"}))]

    assert mean_reciprocal_rank(runs) == 0.25  # (0.5 + 0.0) / 2


def test_mean_reciprocal_rank_of_no_runs_is_zero() -> None:
    assert mean_reciprocal_rank([]) == 0.0


# ---------------------------------------------------------------------------
# Duplicates and citation counts
# ---------------------------------------------------------------------------


def test_duplicate_count_reports_repeated_identifiers() -> None:
    assert duplicate_count(["a", "b", "b", "a", "c"]) == 2
    assert duplicate_count(["a", "b", "c"]) == 0
    assert duplicate_count([]) == 0


def test_unresolved_reference_count_adds_unknown_and_malformed() -> None:
    resolution = CitationResolution(
        unknown_identifiers=("src_aaaaaaaaaaaaaaaa",),
        malformed_identifiers=("src_0123", "chk_deadbeef"),
    )

    assert unresolved_reference_count(resolution) == 3


def test_a_clean_resolution_has_no_unresolved_references() -> None:
    assert unresolved_reference_count(CitationResolution()) == 0


# ---------------------------------------------------------------------------
# Latency observation (an observation, not a claim about the system)
# ---------------------------------------------------------------------------


def test_measure_latency_invokes_the_callable_and_returns_a_duration() -> None:
    ran = False

    def work() -> None:
        nonlocal ran
        ran = True

    elapsed = measure_latency(work)

    assert ran is True
    assert elapsed >= 0.0
