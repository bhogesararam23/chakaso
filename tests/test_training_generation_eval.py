"""Tests for the generation evaluation suite.

Two things are checked separately here: the arithmetic (repetition ratios, repeated blocks, loop detection),
which is hand-computable and must be exactly right, and the suite's behaviour over the model boundary, which
must report what a model did — including a refusal or a crash — rather than only what a well-behaved model
does.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from chakaso.models import (
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    ModelError,
)
from chakaso.models.deterministic import DeterministicModel
from chakaso.training.generation_eval import (
    CONTINUATION_PROBES,
    EDGE_PROBES,
    INTERPRETATION_LIMIT,
    GenerationProbe,
    detect_loop,
    evaluate_generation,
    longest_periodic_run,
    longest_repeated_block,
    ngram_repetition_ratio,
    summarise_generation_evaluation,
)

# ------------------------------------------------------------------ arithmetic


def test_the_repetition_ratio_counts_repeats_of_an_ngram_within_one_text() -> None:
    assert ngram_repetition_ratio("aaaaa", 2) == 0.75
    assert ngram_repetition_ratio("abcde", 2) == 0.0
    assert ngram_repetition_ratio("ababab", 2) == 0.6
    assert ngram_repetition_ratio("ababab", 3) == 0.5
    assert ngram_repetition_ratio("abcabc", 2) == 0.4
    assert ngram_repetition_ratio("abcabc", 3) == 0.25


def test_an_undefined_repetition_ratio_is_none_and_not_a_flattering_zero() -> None:
    assert ngram_repetition_ratio("", 2) is None
    assert ngram_repetition_ratio("ab", 2) is None
    assert ngram_repetition_ratio("x", 1) is None
    assert ngram_repetition_ratio("aabb", 2) is None


def test_the_repetition_ratio_refuses_a_degenerate_ngram() -> None:
    with pytest.raises(ValueError, match="n >= 1"):
        ngram_repetition_ratio("abc", 0)


def test_the_longest_repeated_block_counts_overlapping_occurrences() -> None:
    assert longest_repeated_block("aaaaa") == ("aaaa", 2)
    assert longest_repeated_block("abcabcabc") == ("abcabc", 2)
    assert longest_repeated_block("the the the the the") == ("the the the the", 2)
    assert longest_repeated_block("abcde") == ("", 0)
    assert longest_repeated_block("") == ("", 0)


def test_a_single_repeated_character_is_the_finding_when_nothing_longer_repeats() -> None:
    block, times = longest_repeated_block("aabb")

    assert times == 2
    assert len(block) == 1


def test_loop_detection_names_the_smallest_block_that_sustains_the_tail() -> None:
    assert detect_loop("the the the the the").as_dict() == {
        "looping": True,
        "period": 4,
        "block": " the",
        "repeats": 4,
    }
    assert (detect_loop("ababab").period, detect_loop("ababab").repeats) == (2, 3)
    assert detect_loop("aaaaa").as_dict() == {
        "looping": True,
        "period": 1,
        "block": "a",
        "repeats": 5,
    }


def test_a_repeated_phrase_inside_varied_text_is_not_reported_as_a_loop() -> None:
    """The failure being looked for is a decoding loop, which presents as a periodic *tail*."""
    text = "- Chakaso. - Chakaso. and then a sentence that goes somewhere new entirely"

    assert detect_loop(text).looping is False
    assert (
        longest_repeated_block(text)[1] >= 2
    )  # the repetition is real, and it is still not a loop


def test_the_periodic_run_finds_a_cycle_in_the_middle_that_the_tail_rule_misses() -> None:
    """Measured on E2605's own output, which reports no tail loop and is nonetheless one."""
    text = "grounded grounded grounded and then a heading ## Consequences"

    assert detect_loop(text).looping is False
    assert longest_periodic_run(text) == ("grounded ", 3)


def test_the_periodic_run_reports_no_cycle_as_one_repeat_of_nothing() -> None:
    assert longest_periodic_run("solo text here") == ("", 1)
    assert longest_periodic_run("") == ("", 1)
    assert longest_periodic_run("aaaa") == ("a", 4)
    assert longest_periodic_run("abcabcabc") == ("abc", 3)


def test_the_periodic_run_refuses_a_non_positive_period() -> None:
    with pytest.raises(ValueError, match="min_period must be at least 1"):
        longest_periodic_run("aaaa", min_period=0)


def test_the_suite_records_both_the_cycle_and_the_tail_verdict() -> None:
    model = _ScriptedModel({"p": "grounded grounded grounded tail"})

    report = evaluate_generation(
        model, probes=(GenerationProbe(label="mid", prompt="p", max_new_tokens=64),), edge_probes=()
    )

    sample = report["samples"][0]
    assert sample["longest_periodic_run"] == "grounded "
    assert sample["longest_periodic_run_repeats"] == 3
    assert sample["loop"]["looping"] is False
    assert report["automated_summary"]["periodic_run_samples"] == ["mid"]


def test_loop_detection_needs_two_repeats_and_a_positive_period() -> None:
    assert detect_loop("abab", min_repeats=2).as_dict()["repeats"] == 2
    assert detect_loop("abab").looping is False  # two repeats is below the suite's default of three
    with pytest.raises(ValueError, match="at least two repeats"):
        detect_loop("abab", min_repeats=1)
    with pytest.raises(ValueError, match="min_period must be at least 1"):
        detect_loop("abab", min_period=0)


# ------------------------------------------------------------------ probes


def test_a_probe_that_cannot_be_run_is_refused_on_construction() -> None:
    with pytest.raises(ValueError, match="needs a label"):
        GenerationProbe(label="", prompt="x")
    with pytest.raises(ValueError, match="at least one token"):
        GenerationProbe(label="zero", prompt="x", max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        GenerationProbe(label="hot", prompt="x", temperature=3.0)


def test_the_declared_suite_is_named_and_its_edge_cases_are_fixed() -> None:
    assert {probe.label for probe in CONTINUATION_PROBES} == {
        "document-heading",
        "status-section",
        "because-clause",
        "list-continuation",
        "table-row",
    }
    assert {probe.label for probe in EDGE_PROBES} == {
        "empty-prompt",
        "single-character",
        "non-ascii",
        "oversized-prompt",
    }
    assert all(probe.temperature == 0.0 for probe in CONTINUATION_PROBES)
    assert all(probe.why for probe in CONTINUATION_PROBES + EDGE_PROBES)


# ------------------------------------------------------------------ the suite over a model


class _ScriptedModel:
    """A model that returns fixed text per prompt, so the suite's bookkeeping is what is under test."""

    def __init__(
        self,
        outputs: dict[str, str],
        *,
        fail: Sequence[str] = (),
        refuse: Sequence[str] = (),
    ) -> None:
        self._outputs = outputs
        self._fail = tuple(fail)
        self._refuse = tuple(refuse)
        self.prompts: list[str] = []

    def generate(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> GenerationResult:
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if prompt in self._fail:
            message = f"this model breaks on {prompt!r}"
            raise ValueError(message)
        if prompt in self._refuse:
            message = f"this model refuses {prompt!r}"
            raise ModelError(message)
        text = self._outputs.get(prompt, "")
        bound = params is not None and len(text) >= params.max_new_tokens
        return GenerationResult(
            text=text,
            model_id="scripted",
            finish_reason=FinishReason.LENGTH if bound else FinishReason.STOP,
        )


def test_a_refusal_and_a_crash_are_not_recorded_as_the_same_event() -> None:
    looping = " the the the the the the"
    varied = "a sentence about retrieval and one about tokenization, both short"
    model = _ScriptedModel(
        {
            "heading prompt": looping,
            "status prompt": varied,
        },
        fail=("because prompt",),
        refuse=("empty prompt",),
    )
    probes = (
        GenerationProbe(label="heading", prompt="heading prompt", max_new_tokens=64),
        GenerationProbe(label="status", prompt="status prompt", max_new_tokens=64),
        GenerationProbe(label="because", prompt="because prompt", max_new_tokens=8),
        GenerationProbe(label="empty", prompt="empty prompt", max_new_tokens=8),
    )

    report = evaluate_generation(model, probes=probes, edge_probes=())

    statuses = {sample["label"]: sample["status"] for sample in report["samples"]}
    assert statuses == {
        "heading": "generated",
        "status": "generated",
        "because": "crashed",
        "empty": "refused",
    }
    summary = report["automated_summary"]
    assert summary["samples_generated"] == 2
    assert summary["samples_crashed"] == 1
    assert summary["samples_refused"] == 1
    # A refusal produced no text, so it contributes nothing to the repetition statistics — counting it as a
    # zero-repetition sample would flatter a model that answered nothing.
    assert summary["looping_samples"] == ["heading"]
    assert summary["looping_fraction"] == 0.5
    assert report["interpretation_limit"] == INTERPRETATION_LIMIT
    assert "surface statistics only" in INTERPRETATION_LIMIT
    line = summarise_generation_evaluation(report)
    assert "1 refused" in line and "1 crashed" in line


def test_an_empty_output_is_recorded_as_empty_not_as_a_zero_length_success() -> None:
    model = _ScriptedModel({"p": ""})

    report = evaluate_generation(
        model,
        probes=(GenerationProbe(label="blank", prompt="p", max_new_tokens=8),),
        edge_probes=(),
    )

    sample = report["samples"][0]
    assert sample["characters"] == 0
    assert sample["empty_output"] is True
    assert sample["bigram_repetition"] is None
    assert report["automated_summary"]["empty_outputs"] == ["blank"]


def test_a_reaching_the_requested_bound_is_recorded_as_termination_behaviour() -> None:
    """Whether a sample stopped or ran out of budget is part of what a length number means."""
    model = _ScriptedModel({"p": "x" * 8})

    report = evaluate_generation(
        model,
        probes=(GenerationProbe(label="bound", prompt="p", max_new_tokens=8),),
        edge_probes=(),
    )

    assert report["samples"][0]["finish_reason"] == "length"
    assert report["automated_summary"]["bound_reached"] == ["bound"]


def test_the_suite_survives_every_declared_edge_probe_against_the_development_double() -> None:
    """The double is not a language model, but it is a model behind the boundary: input must not break it."""
    model = DeterministicModel()

    report = evaluate_generation(model, probes=(), edge_probes=EDGE_PROBES)

    assert [sample["status"] for sample in report["edge_samples"]] == ["generated"] * len(
        EDGE_PROBES
    )
    for probe, sample in zip(EDGE_PROBES, report["edge_samples"], strict=True):
        assert sample["label"] == probe.label
        assert sample["characters"] <= probe.max_new_tokens


def test_the_suite_runs_over_the_full_declared_prompts() -> None:
    model = DeterministicModel()

    report = evaluate_generation(model)

    assert [sample["label"] for sample in report["samples"]] == [
        probe.label for probe in CONTINUATION_PROBES
    ]
    assert report["model_id"] == "deterministic"
    assert report["probes"] == [probe.label for probe in CONTINUATION_PROBES]
    line = summarise_generation_evaluation(report)
    assert "samples" in line and "looping" in line


def test_every_probe_reaches_the_model_as_one_user_message() -> None:
    model = _ScriptedModel({})

    evaluate_generation(model, probes=CONTINUATION_PROBES, edge_probes=EDGE_PROBES)

    assert model.prompts == [probe.prompt for probe in CONTINUATION_PROBES + EDGE_PROBES]


def test_a_model_can_be_probed_through_another_entry_point() -> None:
    """`sampler` is how a caller that generates through `sample` reuses the same probes and statistics."""

    class _OnlySample:
        def sample(self, prompt: str, **_ignored: object) -> GenerationResult:
            return GenerationResult(
                text=prompt.upper(), model_id="upper", finish_reason=FinishReason.STOP
            )

    def _sampler(
        model: object, messages: list[Message], _params: GenerationParams
    ) -> GenerationResult:
        return model.sample(messages[-1].content)  # type: ignore[attr-defined]

    report = evaluate_generation(
        _OnlySample(),
        probes=(GenerationProbe(label="loud", prompt="quiet", max_new_tokens=16),),
        edge_probes=(),
        sampler=_sampler,
    )

    assert report["samples"][0]["text"] == "QUIET"
    assert report["samples"][0]["model_id"] == "upper"


def test_an_empty_suite_is_refused_rather_than_reported_as_a_run() -> None:
    with pytest.raises(ValueError, match="no probes"):
        evaluate_generation(_ScriptedModel({}), probes=(), edge_probes=())
