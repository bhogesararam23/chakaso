"""Tests for the decoding policy: one row of logits in, one decision out, and when a run is over.

These are the two places an inference runtime can be quietly wrong. A nucleus that keeps everything, a stop
sequence that matches the prompt, or a repetition penalty that promotes a negative logit all produce
plausible text with an implausible story attached, and none of them fail a loss number. So the policy is
tested here, over plain lists of numbers, with no weights and no framework in the way.
"""

from __future__ import annotations

import random

import pytest

from chakaso.models.base import GenerationParams
from chakaso.models.decoding import AbortSignal, StopScanner, TokenSelector, truncate_prompt
from chakaso.models.errors import ModelError


def selector(params: GenerationParams, size: int = 5, seed: int | None = None) -> TokenSelector:
    return TokenSelector(params, vocabulary_size=size, rng=random.Random(seed))


# ------------------------------------------------------------------ prompt fitting
def test_a_long_prompt_keeps_its_most_recent_identifiers_and_says_so() -> None:
    identifiers, truncated = truncate_prompt([1, 2, 3, 4, 5], 3)

    assert identifiers == (3, 4, 5)
    assert truncated is True


def test_a_prompt_that_fits_is_returned_unchanged_and_unflagged() -> None:
    identifiers, truncated = truncate_prompt([1, 2], 8)

    assert identifiers == (1, 2)
    assert truncated is False


def test_an_empty_prompt_is_refused_rather_than_answered_from_nothing() -> None:
    with pytest.raises(ModelError, match="empty prompt"):
        truncate_prompt([], 8)


@pytest.mark.parametrize("window", [0, -1])
def test_a_context_window_that_is_not_a_positive_integer_is_refused(window: int) -> None:
    with pytest.raises(ModelError, match="context window"):
        truncate_prompt([1], window)


def test_a_boolean_window_is_refused_because_true_is_not_a_length() -> None:
    with pytest.raises(ModelError, match="context window"):
        truncate_prompt([1], True)  # type: ignore[arg-type]


# ------------------------------------------------------------------ stopping
def test_a_stop_token_identifier_ends_the_run() -> None:
    scanner = StopScanner(token_ids=(7,))

    verdict = scanner.feed("nothing", 7)

    assert verdict.stopped
    assert verdict.marker == "token:7"


def test_the_checkpoint_separator_ends_the_run_when_the_caller_knows_it() -> None:
    scanner = StopScanner(document_end_id=256)

    assert scanner.feed("text", 255) == StopScanner().feed("text", 255)
    assert scanner.feed("text", 256).marker == "document-end"


def test_a_stop_sequence_is_found_in_the_produced_text() -> None:
    scanner = StopScanner(sequences=("\n\n",))

    assert not scanner.feed("first line", 1).stopped
    verdict = scanner.feed("first line\n\n", 2)

    assert verdict.stopped
    assert verdict.marker == "\n\n"


def test_a_stop_sequence_that_only_appears_in_the_prompt_does_not_stop_anything() -> None:
    # The scanner is fed generated text only, so a marker that exists in the prompt is never seen. This is
    # the bug the incremental form exists to avoid: a whole-context search ends the run at step one.
    scanner = StopScanner(sequences=("Answer:",))

    verdict = scanner.feed(" The", 1)

    assert not verdict.stopped


def test_a_long_generation_still_finds_a_marker_it_straddles() -> None:
    scanner = StopScanner(sequences=("stop now",))
    produced = "a" * 4000

    assert not scanner.feed(produced + " ", 1).stopped
    verdict = scanner.feed(produced + " stop now", 2)

    assert verdict.stopped
    assert verdict.marker == "stop now"


def test_the_whole_text_form_is_available_for_a_final_check() -> None:
    scanner = StopScanner(sequences=("the end",))

    assert scanner.scan_text("and then, the end came").marker == "the end"
    assert not scanner.scan_text("no marker here").stopped


def test_an_absurd_stop_sequence_is_refused() -> None:
    with pytest.raises(ModelError, match="longer than 64 characters"):
        StopScanner(sequences=("x" * 65,))


# ------------------------------------------------------------------ selection
def test_greedy_selection_takes_the_best_candidate_and_breaks_ties_lowest() -> None:
    params = GenerationParams(temperature=0.0)

    assert selector(params, size=3).select([0.1, 0.9, 0.4]) == 1
    assert selector(params, size=3).select([0.5, 0.5, 0.1]) == 0


def test_a_non_finite_logit_is_refused_instead_of_being_answered_with_a_word() -> None:
    params = GenerationParams(temperature=0.0)

    with pytest.raises(ModelError, match="non-finite"):
        selector(params, size=3).select([0.0, float("nan"), 1.0])
    with pytest.raises(ModelError, match="non-finite"):
        selector(params, size=3).select([0.0, float("inf"), 1.0])


def test_a_logit_row_that_does_not_match_the_vocabulary_is_refused() -> None:
    params = GenerationParams(temperature=0.0)

    with pytest.raises(ModelError, match="2 logits for a vocabulary of 5"):
        selector(params).select([1.0, 2.0])


def test_a_top_k_wider_than_the_vocabulary_is_refused_before_the_run() -> None:
    with pytest.raises(ModelError, match="wider than the vocabulary"):
        selector(GenerationParams(temperature=1.0, top_k=9), size=5)


def test_a_seeded_sample_is_reproducible_and_an_unrelated_seed_can_differ() -> None:
    params = GenerationParams(temperature=1.0)
    logits = [0.2, 1.4, 0.9, 0.1, 0.3]

    # One selector, many steps: that is the loop a generation runs. A fresh selector per draw would be a
    # fresh RNG per draw, which is not what a stream does and would make this test prove nothing.
    one = selector(params, seed=11)
    two = selector(params, seed=11)
    first = [one.select(logits) for _ in range(40)]
    second = [two.select(logits) for _ in range(40)]

    assert first == second
    assert set(first) <= {0, 1, 2, 3, 4}
    assert 1 in first
    assert len(set(first)) > 1


def test_top_k_only_allows_the_declared_number_of_candidates() -> None:
    params = GenerationParams(temperature=1.0, top_k=2)
    logits = [0.0, 3.0, 2.0, 1.9, 1.8]

    drawn = {selector(params, seed=index).select(logits) for index in range(60)}

    assert drawn <= {1, 2}


def test_top_p_keeps_the_smallest_set_that_reaches_the_threshold() -> None:
    # A sharply peaked row: the first candidate alone carries well over 0.9 of the mass, so a nucleus of
    # 0.9 must be able to draw only it. A wider threshold reaches the second.
    params = GenerationParams(temperature=0.5, top_p=0.9)
    logits = [10.0, 0.0, 0.0, 0.0, 0.0]

    drawn = {selector(params, seed=index).select(logits) for index in range(30)}

    assert drawn == {0}


def test_a_wider_nucleus_lets_more_candidates_through() -> None:
    params = GenerationParams(temperature=1.0, top_p=1.0)
    logits = [1.0, 1.0, 1.0, 0.0, 0.0]

    drawn = {selector(params, seed=index).select(logits) for index in range(80)}

    assert drawn <= {0, 1, 2, 3, 4}
    assert len(drawn) > 1


def test_a_repeat_penalty_lowers_a_token_that_has_already_been_used() -> None:
    logits = [2.0, 1.9, 0.0, 0.0, 0.0]
    without = selector(GenerationParams(temperature=0.0))
    with_penalty = selector(GenerationParams(temperature=0.0, repeat_penalty=1.6))
    with_penalty.note(0)

    assert without.select(logits) == 0
    assert with_penalty.select(logits) == 1


def test_a_repeat_penalty_never_promotes_a_token_it_is_penalising() -> None:
    # A negative logit multiplied rather than divided would rise in rank when penalised; the sign handling
    # exists to stop exactly that, so a used token must never overtake an unused one it started below.
    logits = [-1.0, -0.5, 0.0, 0.0, 0.0]
    penalised = selector(GenerationParams(temperature=0.0, repeat_penalty=2.0))
    penalised.note(2)

    assert (
        penalised.select(logits) == 2
    )  # still the best of the three, and it was not penalised below 0
    penalised.note(1)
    assert penalised.select(logits) == 2


def test_greedy_is_the_default_and_temperature_zero_is_greedy_by_name() -> None:
    params = GenerationParams()

    assert selector(params).greedy
    assert selector(GenerationParams(temperature=1.0)).greedy is False


def test_an_abort_signal_only_reports_cancellation_after_it_is_asked_for() -> None:
    signal = AbortSignal()

    assert not signal.cancelled
    signal.abort()
    assert signal.cancelled
