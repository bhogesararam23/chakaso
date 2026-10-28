"""Tests for the calibrated generation-quality suite: metrics, prompt selection, gates and comparison.

A scripted fake stands in for the model, so every number below can be checked by hand and the suite runs in
the dependency-free matrix. What is under test is the *procedure* — that a looping model fails, that real
text passes against itself, that an over-diverse text is refused as well, and that two reports about
different prompts cannot be compared — not any claim about a trained model.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from chakaso.models import (
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    ModelError,
)
from chakaso.models.base import GenerationChunk
from chakaso.training.data import TextUnit
from chakaso.training.quality_eval import (
    DEFAULT_CONFIGS,
    GATES,
    HELD_OUT_TEST_FRACTION,
    HELD_OUT_VALIDATION_FRACTION,
    DecodingConfig,
    GateSpec,
    HeldOutPrompt,
    assess_gates,
    compare_quality_reports,
    distinct_ratio,
    evaluate_quality,
    heldout_test_units,
    run_release_quality,
    select_heldout_prompts,
    summarise_quality,
    text_statistics,
    words,
)
from chakaso.training.subword import train_subword_vocabulary

_PROSE = (
    "The old captain walked slowly along the harbour wall while the grey tide came in beneath him and "
    "the gulls wheeled over the empty boats. Nobody spoke to him, for the town had long ago learned "
    "that he preferred the sound of water to the sound of voices, and that a question put to him at "
    "that hour would only be answered with a look toward the horizon. "
)
_VOCABULARY = train_subword_vocabulary([_PROSE * 4], target_size=300, min_frequency=2)


def _units(count: int = 6) -> list[TextUnit]:
    return [
        TextUnit(identifier=f"doc-{index:02d}", text=(f"Chapter {index}. " + _PROSE * 12))
        for index in range(count)
    ]


class _Scripted:
    """A model whose output is a function of the request, so gates can be driven to either verdict."""

    def __init__(self, produce: Callable[[str, GenerationParams], str]) -> None:
        self._produce = produce
        self.calls = 0

    def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        self.calls += 1
        text = self._produce(messages[-1].content, params)
        return GenerationResult(
            text=text,
            model_id="scripted",
            finish_reason=FinishReason.LENGTH,
            token_ids=(1,) * 3,
            new_tokens=3,
        )

    def stream(
        self, messages: list[Message], params: GenerationParams
    ) -> Iterator[GenerationChunk]:
        text = self._produce(messages[-1].content, params)
        for character in text:
            yield GenerationChunk(text=character, token_id=1)
        yield GenerationChunk(text="", token_id=-1, final=True, finish_reason=FinishReason.LENGTH)


def _prompts(count: int = 4) -> tuple[HeldOutPrompt, ...]:
    return select_heldout_prompts(
        _units(), _VOCABULARY, count=count, prompt_chars=120, reference_tokens=200, seed=5
    )


def test_words_drops_punctuation_and_keeps_a_curly_apostrophe_inside_a_word() -> None:
    assert words("Don\u2019t stop, he said.") == ["don\u2019t", "stop", "he", "said"]


def test_distinct_ratio_is_one_for_unique_text_and_falls_as_phrases_repeat() -> None:
    unique = words("one two three four five six seven eight nine ten")
    looped = words("the cat the cat the cat the cat the cat")

    assert distinct_ratio(unique, 1) == 1.0
    assert distinct_ratio(unique, 2) == 1.0
    assert distinct_ratio(looped, 1) == pytest.approx(0.2)
    assert distinct_ratio(looped, 2) == pytest.approx(2 / 9)


def test_distinct_ratio_is_undefined_for_a_text_too_short_to_say() -> None:
    assert distinct_ratio(["a", "b", "c"], 2) is None


def test_distinct_ratio_refuses_a_nonsense_order() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        distinct_ratio(["a"] * 10, 0)


def test_text_statistics_flags_a_tail_loop_and_a_replacement_character() -> None:
    stats = text_statistics("so it went and went and went and went and went \ufffd")

    assert stats["replacement_characters"] == 1
    assert stats["periodic_run_repeats"] >= 3
    assert stats["distinct1"] is not None and stats["distinct1"] < 0.7


def test_trailing_whitespace_is_not_a_loop_but_a_repeated_phrase_still_is() -> None:
    # Regression: a sampled continuation ending in "\n     " was flagged as a tail loop of one space.
    trailing = (
        "He attended to the employed feeling, and showed a sensible tone to put off the\n     "
    )
    looping = "He said so. " + "and so it went " * 6

    assert text_statistics(trailing)["tail_loop"] is False
    assert (
        text_statistics("Chapter one.\n\n\n\n\nChapter two begins here")["periodic_run_repeats"] < 3
    )
    assert text_statistics(looping)["tail_loop"] is True


def test_prompt_selection_is_deterministic_and_takes_one_prompt_per_document() -> None:
    first = _prompts(5)
    second = _prompts(5)

    assert first == second
    assert len({prompt.unit_id for prompt in first}) == 5


def test_prompt_selection_changes_with_the_seed() -> None:
    other = select_heldout_prompts(
        _units(), _VOCABULARY, count=4, prompt_chars=120, reference_tokens=40, seed=6
    )

    assert other != _prompts(4)


def test_prompts_end_at_a_word_boundary_and_the_reference_really_follows_in_the_document() -> None:
    units = {unit.identifier: unit.text for unit in _units()}
    for prompt in _prompts(4):
        document = units[prompt.unit_id]
        assert document[prompt.start : prompt.start + len(prompt.prompt)] == prompt.prompt
        after = document[prompt.start + len(prompt.prompt) :].lstrip()
        assert after.startswith(prompt.reference.lstrip()[:30])
        assert not prompt.prompt.endswith(" ")


def test_prompt_selection_refuses_to_underfill_the_suite() -> None:
    with pytest.raises(ValueError, match="were requested"):
        select_heldout_prompts(
            _units(2), _VOCABULARY, count=5, prompt_chars=120, reference_tokens=40
        )


def test_prompt_selection_ignores_documents_too_short_to_hold_a_reference() -> None:
    units = [*_units(3), TextUnit(identifier="short", text="tiny")]

    chosen = select_heldout_prompts(
        units, _VOCABULARY, count=3, prompt_chars=120, reference_tokens=40
    )

    assert "short" not in {prompt.unit_id for prompt in chosen}


def _human_like(prompt: str, _params: GenerationParams) -> str:
    # Returns the *reference* of whichever prompt it is asked about, i.e. exactly the human text.
    return _REFERENCE_BY_PROMPT[prompt]


_PROMPT_SET = _prompts(4)
_REFERENCE_BY_PROMPT = {prompt.prompt: prompt.reference for prompt in _PROMPT_SET}


def test_a_model_that_writes_exactly_the_human_text_passes_every_gate() -> None:
    report = evaluate_quality(
        _Scripted(_human_like), _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8
    )

    gates = report["configs"]["greedy"]["gates"]
    assert gates["all_passed"] is True
    assert gates["distinct1"]["gap"] == 0.0


def test_a_looping_model_fails_the_diversity_looping_and_periodic_gates() -> None:
    looper = _Scripted(lambda _prompt, _params: "and the world of the " * 8)

    report = evaluate_quality(looper, _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8)

    gates = report["configs"]["greedy"]["gates"]
    assert gates["all_passed"] is False
    assert gates["distinct1"]["direction"] == "repetitive"
    assert gates["looping"]["passed"] is False
    assert gates["periodic"]["passed"] is False


def test_text_that_never_repeats_a_word_is_refused_as_over_diverse() -> None:
    # A repetition penalty can push a model past the human level of variety. That is also off-target.
    unique = " ".join(f"word{index}" for index in range(60))
    model = _Scripted(lambda _prompt, _params: unique)

    report = evaluate_quality(model, _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8)

    verdict = report["configs"]["greedy"]["gates"]["distinct1"]
    assert verdict["passed"] is False
    assert verdict["direction"] == "over_diverse"


def test_a_single_replacement_character_fails_the_decoding_gate() -> None:
    model = _Scripted(lambda prompt, _params: _REFERENCE_BY_PROMPT[prompt] + "\ufffd")

    report = evaluate_quality(model, _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8)

    gate = report["configs"]["greedy"]["gates"]["replacement_characters"]
    assert gate["passed"] is False
    assert gate["observed"] == len(_PROMPT_SET)


def test_sampled_configurations_run_once_per_seed_and_greedy_runs_once() -> None:
    model = _Scripted(_human_like)
    configs = (DecodingConfig("greedy"), DecodingConfig("hot", temperature=1.0))

    report = evaluate_quality(
        model, _PROMPT_SET, configs=configs, seeds=(1, 2, 3), max_new_tokens=8
    )

    assert report["configs"]["greedy"]["aggregate"]["samples"] == len(_PROMPT_SET)
    assert report["configs"]["hot"]["aggregate"]["samples"] == 3 * len(_PROMPT_SET)
    assert {row["seed"] for row in report["configs"]["hot"]["rows"]} == {1, 2, 3}


def test_a_refusal_is_counted_and_not_scored() -> None:
    def refuse(_prompt: str, _params: GenerationParams) -> str:
        message = "declined"
        raise ModelError(message)

    model = _Scripted(refuse)
    report = evaluate_quality(model, _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8)

    # The refusal propagates from determinism checks only if generation itself did not; here it is
    # swallowed per sample and the aggregate says nothing was scored.
    assert report["configs"]["greedy"]["aggregate"]["refused"] == len(_PROMPT_SET)
    assert report["configs"]["greedy"]["aggregate"]["samples"] == 0


def test_an_evaluation_with_nothing_to_run_is_refused_not_passed() -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        evaluate_quality(_Scripted(_human_like), (), max_new_tokens=8)


def test_determinism_checks_catch_a_model_that_does_not_repeat_itself() -> None:
    counter = {"n": 0}

    def unstable(prompt: str, _params: GenerationParams) -> str:
        counter["n"] += 1
        return _REFERENCE_BY_PROMPT[prompt] + f" {counter['n']}"

    report = evaluate_quality(
        _Scripted(unstable), _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8
    )

    assert report["determinism"][0]["repeat_identical"] is False


def test_the_report_carries_its_schema_prompt_digest_gates_and_limit() -> None:
    report = evaluate_quality(
        _Scripted(_human_like), _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8
    )

    assert report["schema"] == "chakaso-generation-quality/1"
    assert len(report["prompt_digest"]) == 64
    assert report["gates"] == GATES.as_dict()
    assert report["model_id"] == "scripted"
    assert "surface statistics" in report["interpretation_limit"]


def test_comparison_reports_per_config_deltas_for_the_same_prompts() -> None:
    good = evaluate_quality(
        _Scripted(_human_like), _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8
    )
    bad = evaluate_quality(
        _Scripted(lambda _p, _q: "and the world of the " * 8),
        _PROMPT_SET,
        configs=(DEFAULT_CONFIGS[0],),
        max_new_tokens=8,
    )

    delta = compare_quality_reports(good, bad)["greedy"]

    assert delta["distinct1"] < 0
    assert delta["all_passed"] == {"baseline": True, "candidate": False}


def test_comparison_refuses_reports_made_from_different_prompts() -> None:
    first = evaluate_quality(
        _Scripted(_human_like), _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8
    )
    second = copy.deepcopy(first)
    second["prompt_digest"] = "0" * 64

    with pytest.raises(ValueError, match="prompt_digest"):
        compare_quality_reports(first, second)


def test_gate_bands_are_two_sided_and_declared_as_data() -> None:
    aggregate = {
        "distinct1": 0.80,
        "distinct2": 0.9,
        "distinct3": 0.9,
        "looping_fraction": 0.0,
        "periodic_fraction": 0.0,
        "replacement_characters": 0,
    }
    reference = {"distinct1": 0.75, "distinct2": 0.9, "distinct3": 0.9}

    assert assess_gates(aggregate, reference, GateSpec(distinct1_band=0.05))["distinct1"]["passed"]
    assert not assess_gates(aggregate, reference, GateSpec(distinct1_band=0.04))["distinct1"][
        "passed"
    ]


def _snapshot(directory: Path, documents: int = 14) -> Path:
    from chakaso.training.manifest import build_manifest, write_snapshot

    docs = directory / "docs"
    docs.mkdir(parents=True)
    for index in range(documents):
        (docs / f"book-{index:02d}.md").write_text(
            f"Book {index}. " + _PROSE * 12, encoding="utf-8", newline="\n"
        )
    manifest = build_manifest(
        directory,
        name="quality-test",
        version="1",
        licence="Apache-2.0",
        source="built by tests",
        include=("docs/",),
        extensions=(".md",),
    )
    target = directory / "snapshot.json"
    write_snapshot(manifest, target)
    return target


def test_the_held_out_documents_are_the_experiment_drivers_test_side(tmp_path: Path) -> None:
    from chakaso.training.data import split_text_units
    from chakaso.training.manifest import read_snapshot

    snapshot = _snapshot(tmp_path)

    held_out = heldout_test_units(snapshot, split_seed=7)
    expected = split_text_units(
        read_snapshot(snapshot).manifest.text_units(),
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=7,
    )

    assert held_out == tuple(expected.test)
    train_ids = {unit.identifier for unit in expected.train}
    validation_ids = {unit.identifier for unit in expected.validation}
    assert held_out
    assert not {unit.identifier for unit in held_out} & (train_ids | validation_ids)


def test_a_different_split_seed_holds_out_different_documents(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    assert heldout_test_units(snapshot, split_seed=7) != heldout_test_units(snapshot, split_seed=8)


def test_the_quality_suite_refuses_a_model_with_no_trained_vocabulary(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    with pytest.raises(ModelError, match="no trained vocabulary"):
        run_release_quality(_Scripted(_human_like), snapshot, split_seed=7)


def test_the_summary_table_names_each_failed_gate() -> None:
    looper = _Scripted(lambda _prompt, _params: "and the world of the " * 8)
    report = evaluate_quality(looper, _PROMPT_SET, configs=(DEFAULT_CONFIGS[0],), max_new_tokens=8)

    table = summarise_quality(report)

    assert "human" in table
    assert "FAIL" in table and "distinct1" in table and "looping" in table


def test_the_split_fractions_match_the_experiment_driver() -> None:
    pytest.importorskip("torch")
    from chakaso.training import subword_experiment

    assert HELD_OUT_VALIDATION_FRACTION == subword_experiment._VALIDATION_FRACTION
    assert HELD_OUT_TEST_FRACTION == subword_experiment._TEST_FRACTION


def test_an_offset_gives_a_second_prompt_set_that_shares_no_document() -> None:
    first = select_heldout_prompts(
        _units(), _VOCABULARY, count=3, prompt_chars=120, reference_tokens=40, seed=5
    )
    second = select_heldout_prompts(
        _units(), _VOCABULARY, count=3, prompt_chars=120, reference_tokens=40, seed=5, offset=3
    )

    assert not {p.unit_id for p in first} & {p.unit_id for p in second}
    assert len(second) == 3


def test_an_offset_past_the_available_documents_is_refused() -> None:
    with pytest.raises(ValueError, match="after skipping 4"):
        select_heldout_prompts(
            _units(6), _VOCABULARY, count=3, prompt_chars=120, reference_tokens=40, offset=4
        )


def test_the_declared_sweep_names_are_unique_and_the_sets_are_selectable() -> None:
    from chakaso.training.quality_eval import CONFIG_SETS, SWEEP_CONFIGS

    names = [config.name for config in SWEEP_CONFIGS]

    assert len(names) == len(set(names)) == 19
    assert set(CONFIG_SETS) == {"default", "sweep"}
