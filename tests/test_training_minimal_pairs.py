"""The minimal-pair instrument: scoring is a comparison of two log-probabilities, nothing more, and its guards are real."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chakaso.training.errors import CorpusError
from chakaso.training.minimal_pairs import (
    MinimalPair,
    PairSet,
    contaminated_pairs,
    load_pair_set,
    score_pairs,
    wilson_interval,
)


def _pairs() -> tuple[MinimalPair, ...]:
    return (
        MinimalPair("p1", "agreement", "The dog runs.", "The dog run."),
        MinimalPair("p2", "agreement", "The dogs run.", "The dogs runs."),
        MinimalPair("p3", "order", "The old man sat.", "Old the man sat."),
        MinimalPair("p4", "order", "A tall tree fell.", "Tall a tree fell."),
    )


def _oracle(text: str) -> float:
    """A scorer that prefers any sentence that does not appear in the known-bad list."""
    return (
        -10.0
        if text in {"The dog run.", "The dogs runs.", "Old the man sat.", "Tall a tree fell."}
        else -5.0
    )


def test_a_scorer_that_prefers_every_good_sentence_is_perfect() -> None:
    result = score_pairs(_pairs(), _oracle)

    assert result.accuracy == 1.0
    assert result.by_phenomenon == {"agreement": 1.0, "order": 1.0}
    assert result.pairs == 4


def test_a_scorer_that_prefers_every_bad_sentence_scores_zero() -> None:
    result = score_pairs(_pairs(), lambda text: -_oracle(text))

    assert result.accuracy == 0.0


def test_a_tie_counts_half_so_a_constant_scorer_sits_at_chance() -> None:
    result = score_pairs(_pairs(), lambda _text: -7.0)

    assert result.accuracy == 0.5
    assert result.ties == 4


def test_the_interval_is_wilson_and_brackets_the_estimate() -> None:
    low, high = wilson_interval(successes=80, trials=100)

    assert low < 0.8 < high
    assert low == pytest.approx(0.7112, abs=1e-3)
    assert high == pytest.approx(0.8666, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_the_result_reports_whether_chance_lies_inside_the_interval() -> None:
    chance = score_pairs(_pairs(), lambda _t: 0.0)
    perfect = score_pairs(_pairs() * 25, _oracle)

    assert chance.chance_inside_interval is True
    assert perfect.chance_inside_interval is False


def test_a_length_only_reading_is_reported_so_a_length_bias_cannot_pass_for_grammar() -> None:
    pairs = (
        MinimalPair("a", "x", "The dog runs.", "The dog run."),  # good is longer
        MinimalPair("b", "x", "The dogs run.", "The dogs runs."),  # good is shorter
        MinimalPair("c", "x", "It is.", "Is it."),  # equal length
    )

    result = score_pairs(pairs, lambda text: -float(len(text)))

    # A scorer that simply prefers the shorter sentence gets only the pair where good is shorter, and half of the tie.
    assert result.length_only_accuracy == pytest.approx((0 + 1 + 0.5) / 3)
    assert result.good_longer == 1
    assert result.good_shorter == 1


def test_a_pair_with_identical_sentences_is_refused() -> None:
    with pytest.raises(CorpusError, match="identical"):
        MinimalPair("p", "x", "The same.", "The same.")


def test_a_pair_set_refuses_duplicate_identifiers_and_empty_sets() -> None:
    first = MinimalPair("p", "x", "A b.", "B a.")
    with pytest.raises(CorpusError, match="twice"):
        PairSet(name="s", version="1", licence="l", provenance="p", pairs=(first, first))
    with pytest.raises(CorpusError, match="no pairs"):
        PairSet(name="s", version="1", licence="l", provenance="p", pairs=())


def test_a_pair_set_round_trips_and_carries_a_digest_of_its_pairs(tmp_path: Path) -> None:
    pair_set = PairSet(
        name="s",
        version="1.0.0",
        licence="Apache-2.0",
        provenance="written for tests",
        pairs=_pairs(),
    )
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(pair_set.as_dict()), encoding="utf-8")

    loaded = load_pair_set(path)

    assert loaded.pairs == pair_set.pairs
    assert loaded.digest == pair_set.digest
    altered = json.loads(path.read_text(encoding="utf-8"))
    altered["pairs"][0]["good"] = "The dog runs fast."
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(CorpusError, match="digest"):
        load_pair_set(path)


def test_contamination_counts_pairs_whose_good_sentence_shares_a_word_ngram_with_the_training_text() -> (
    None
):
    texts = ["Long ago the old man sat by the river and the dog runs home."]
    pairs = (
        MinimalPair("seen", "x", "The old man sat by the river.", "The old man sit by the river."),
        MinimalPair("unseen", "x", "Birds sing at dawn.", "Birds sings at dawn."),
    )

    flagged = contaminated_pairs(pairs, texts, n=5)

    assert flagged == ("seen",)


def test_contamination_ignores_case_and_punctuation() -> None:
    flagged = contaminated_pairs(
        (MinimalPair("p", "x", "The OLD man, sat by the river!", "The old man sit by the river."),),
        ["the old man sat by the river"],
        n=5,
    )

    assert flagged == ("p",)


def test_an_ngram_longer_than_the_sentence_cannot_flag_it() -> None:
    pairs = (MinimalPair("p", "x", "Birds sing.", "Birds sings."),)

    assert contaminated_pairs(pairs, ["birds sing"], n=5) == ()


def test_wilson_interval_is_symmetric_at_one_half() -> None:
    low, high = wilson_interval(50, 100)

    assert low + high == pytest.approx(1.0)
    assert math.isfinite(low) and math.isfinite(high)


# ------------------------------------------------ a token-count reading beside the length-only one (E2621)


def test_a_token_count_reading_is_reported_when_a_counter_is_given() -> None:
    pairs = (
        MinimalPair("a", "x", "Good one.", "Bad one longer."),  # good has fewer words
        MinimalPair("b", "x", "Good number two here.", "Bad two."),  # good has more
        MinimalPair("c", "x", "Same size ok.", "Same size no."),  # equal
    )

    result = score_pairs(pairs, lambda _text: 0.0, token_count=lambda text: len(text.split()))

    # "Prefer fewer tokens" gets pair a, misses pair b, ties pair c.
    assert result.token_count_only_accuracy == pytest.approx((1 + 0 + 0.5) / 3)
    assert result.good_fewer_tokens == 1
    assert result.good_more_tokens == 1


def test_without_a_counter_the_token_count_reading_is_absent_not_zero() -> None:
    result = score_pairs(_pairs(), _oracle)

    assert result.token_count_only_accuracy is None
    assert result.as_dict()["token_count_only_accuracy"] is None


def test_pairs_balanced_in_tokens_make_the_token_count_reading_exactly_chance() -> None:
    pairs = (
        MinimalPair("a", "x", "one two", "two one"),
        MinimalPair("b", "x", "red fox", "fox red"),
    )

    result = score_pairs(pairs, lambda _text: 0.0, token_count=lambda text: len(text.split()))

    assert result.token_count_only_accuracy == 0.5
    assert result.good_fewer_tokens == result.good_more_tokens == 0


# ------------------------------------------------------------------ the per-item record


def test_the_result_carries_which_way_every_pair_went() -> None:
    """A paired comparison of two models on one pair set needs the per-item outcome, not the mean.

    E2622 could only set two Wilson intervals beside each other and had to withdraw the ordering it produced;
    the item-level record is what McNemar's test counts.
    """
    result = score_pairs(_pairs(), _oracle)

    assert [entry[0] for entry in result.outcomes] == ["p1", "p2", "p3", "p4"]
    assert [entry[1] for entry in result.outcomes] == ["agreement", "agreement", "order", "order"]
    assert all(entry[2] == 1.0 for entry in result.outcomes)
    assert result.as_dict()["outcomes"] == [
        ["p1", "agreement", 1.0],
        ["p2", "agreement", 1.0],
        ["p3", "order", 1.0],
        ["p4", "order", 1.0],
    ]


def test_a_tie_and_a_failure_are_kept_apart_in_the_per_item_record() -> None:
    # One pair the scorer cannot tell (equal log-probability, half credit), one it gets wrong.
    def scorer(text: str) -> float:
        return {
            "The dog runs.": -5.0,
            "The dog run.": -5.0,
            "The dogs run.": -1.0,
            "The dogs runs.": -5.0,
        }.get(text, 0.0)

    result = score_pairs(_pairs()[:2], scorer)

    assert result.outcomes[0] == ("p1", "agreement", 0.5)
    assert result.outcomes[1] == ("p2", "agreement", 1.0)
    assert result.ties == 1
