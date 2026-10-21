"""A deterministic generation evaluation suite that runs over the model boundary, not behind it.

A held-out loss number says something about next-token prediction and nothing about what the model actually
produces when asked to continue text. E2605's committed generations loop — that is how the record describes
them — and "it loops" is an observation, not a measurement. This module turns that observation into numbers
that can be compared across seeds and across corpora, and it is deliberately honest about their scope:

* **Everything here is a surface statistic.** Repetition ratios, loop periods, emitted lengths and refusal
  behaviour are computed from the characters the model produced. None of them measures meaning, and the
  output says so in the same payload that carries the numbers, so a reader cannot separate the two.
* **Human judgement is a separate act.** The suite records the raw text of every sample; whether any of it
  is *useful* is a question for a person, and no number in this module is offered as its substitute.
* **It takes a** :class:`chakaso.models.LanguageModel`. Nothing here imports a concrete adapter, so the same
  suite runs against the development double in the dependency-free matrix, against the character checkpoint
  behind ``[train]``, and against the Transformer checkpoint behind ``[nn]``. A model that cannot survive a
  probe is reported as what it is — a refusal, an exception, or a loop — rather than skipped.

The prompts are fixed here and are not derived from any model's output. They are continuations of the kind of
prose the project's own corpus contains, which is a fair test of what this model was trained on and an unfair
test of anything else; that limitation is stated rather than hidden by choosing prompts the model happens to
score well on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from chakaso.models import FinishReason, GenerationParams, LanguageModel, Message, Role

__all__ = [
    "CONTINUATION_PROBES",
    "EDGE_PROBES",
    "GenerationProbe",
    "detect_loop",
    "evaluate_generation",
    "longest_periodic_run",
    "longest_repeated_block",
    "ngram_repetition_ratio",
    "summarise_generation_evaluation",
]

#: What the automated numbers do and do not claim. Carried inside the report so it travels with the data.
INTERPRETATION_LIMIT = (
    "surface statistics only: repetition, length, termination and refusal behaviour computed from emitted "
    "characters. No semantic quality, usefulness or groundedness is measured here, and none may be inferred "
    "from these numbers. Judging whether the text means anything is a separate, human act."
)

#: The smallest output that supports a bigram repetition ratio. Below this the statistic is undefined and is
#: reported as ``None`` rather than as a flattering zero.
_MINIMUM_TEXTS_FOR_REPETITION = 4


@dataclass(frozen=True, slots=True)
class GenerationProbe:
    """One fixed request, with the reason it is in the suite.

    Attributes:
        label: A stable name, so a report can be compared across runs and experiments.
        prompt: The text handed to the model as a user turn.
        max_new_tokens: The generation bound declared up front, so an output length has a meaning.
        temperature: 0.0 is greedy decoding — deterministic, and therefore comparable across runs.
        why: What the probe is checking. A suite of unexplained prompts cannot be audited later.
    """

    label: str
    prompt: str
    max_new_tokens: int = 48
    temperature: float = 0.0
    why: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            message = "a generation probe needs a label to be comparable across runs"
            raise ValueError(message)
        if self.max_new_tokens < 1:
            message = (
                f"probe {self.label!r} must ask for at least one token, got {self.max_new_tokens}"
            )
            raise ValueError(message)
        if not 0.0 <= self.temperature <= 2.0:
            message = f"probe {self.label!r} has a temperature outside [0, 2]: {self.temperature}"
            raise ValueError(message)


#: Continuations of the project's own prose. Fixed, hand-written for this suite, and never chosen using a
#: model's output.
CONTINUATION_PROBES: tuple[GenerationProbe, ...] = (
    GenerationProbe(
        label="document-heading",
        prompt="# Chakaso\n\nChakaso is a",
        why="the corpus's dominant shape: a markdown heading followed by prose about the project",
    ),
    GenerationProbe(
        label="status-section",
        prompt="## Status\n\nThe retrieval layer",
        why="a status paragraph, which is where the corpus's own claims about itself live",
    ),
    GenerationProbe(
        label="because-clause",
        prompt="The tokenizer is trained by the project because",
        max_new_tokens=64,
        why="a causal clause — the shape a model must complete to say anything with structure",
    ),
    GenerationProbe(
        label="list-continuation",
        prompt="Commands:\n\n```bash\npython -m pytest",
        max_new_tokens=48,
        why="a fenced code block, which has a closing shape the corpus contains many times",
    ),
    GenerationProbe(
        label="table-row",
        prompt="| Gate | Result |\n| --- | --- |\n| `pytest` |",
        max_new_tokens=32,
        why="table markup: the most pattern-bound prose in the corpus, so the easiest case",
    ),
)

#: Inputs a model must survive without an unhandled failure. Each names the behaviour being checked.
EDGE_PROBES: tuple[GenerationProbe, ...] = (
    GenerationProbe(
        label="empty-prompt", prompt="", max_new_tokens=8, why="an empty turn is legal input"
    ),
    GenerationProbe(
        label="single-character",
        prompt="#",
        max_new_tokens=8,
        why="a context shorter than the window",
    ),
    GenerationProbe(
        label="non-ascii",
        prompt="-notes: \u4f60\u597d\u4e16\u754c \U0001f600\n",
        max_new_tokens=8,
        why="characters outside any corpus alphabet must encode, not crash the tokenizer",
    ),
    GenerationProbe(
        label="oversized-prompt",
        prompt="repeated context " * 200,
        max_new_tokens=8,
        why="a prompt longer than the model's window: the boundary decides, the model must not raise",
    ),
)


def ngram_repetition_ratio(text: str, n: int) -> float | None:
    """The share of ``text``'s n-grams that have already occurred earlier in the same text.

    0.0 means every n-gram is new; 1.0 means the text is one n-gram repeated. ``None`` is returned when the
    text is too short to have n-grams to compare, because an undefined statistic reported as 0.0 would look
    like perfect non-repetition.

    Known failure modes, stated because they are how this number gets misread: a legitimate refrain, a
    heading style repeated by a document, or markdown boilerplate all raise this ratio without the text
    being bad; and a very short output gets a low ratio simply by having almost nothing to repeat.
    """
    if n < 1:
        message = f"repetition is defined over n >= 1, got {n}"
        raise ValueError(message)
    grams = [text[index : index + n] for index in range(len(text) - n + 1)]
    if len(grams) < _MINIMUM_TEXTS_FOR_REPETITION:
        return None
    seen: set[str] = set()
    repeats = 0
    for gram in grams:
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return round(repeats / len(grams), 6)


def longest_repeated_block(text: str) -> tuple[str, int]:
    """The longest substring of ``text`` that occurs at least twice, and how many times it occurs.

    ``("", 0)`` when nothing repeats. Overlapping occurrences count, which is why the search runs to
    ``len(text) - 1`` rather than stopping at half the length: in ``"aaaaa"`` the block ``"aaaa"`` occurs
    twice, at offsets 0 and 1, and a half-length cap would report the shorter ``"aa"`` as the longest.
    """
    best: tuple[str, int] = ("", 0)
    for length in range(1, max(1, len(text))):
        counts: dict[str, int] = {}
        for start in range(len(text) - length + 1):
            block = text[start : start + length]
            counts[block] = counts.get(block, 0) + 1
        repeated = [(block, times) for block, times in counts.items() if times > 1]
        if not repeated:
            continue
        block, times = max(repeated, key=lambda item: (len(item[0]), item[1], item[0]))
        best = (block, times)
    return best


@dataclass(frozen=True, slots=True)
class LoopFinding:
    """A periodic tail: the end of the output is one block repeated, which is the failure it looks for."""

    looping: bool
    period: int
    block: str
    repeats: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "looping": self.looping,
            "period": self.period,
            "block": self.block,
            "repeats": self.repeats,
        }


def detect_loop(text: str, *, min_period: int = 1, min_repeats: int = 3) -> LoopFinding:
    """Whether ``text`` ends with one block repeated at least ``min_repeats`` times.

    Only the *tail* is examined, because that is how a decoding loop presents: the model settles into a
    cycle and stays in it. A repeated phrase in the middle of an otherwise varied output is reported as not
    looping. ``period`` is the smallest block that sustains the tail run, so "ababab" is a period-2 loop of
    three repeats rather than a period-6 of one.
    """
    if min_period < 1:
        message = f"min_period must be at least 1, got {min_period}"
        raise ValueError(message)
    if min_repeats < 2:
        message = f"a loop needs at least two repeats, got {min_repeats}"
        raise ValueError(message)
    for period in range(min_period, len(text) // min_repeats + 1):
        block = text[-period:]
        if len(block) != period:
            continue
        repeats = 0
        cursor = len(text)
        while cursor >= period and text[cursor - period : cursor] == block:
            repeats += 1
            cursor -= period
        if repeats >= min_repeats:
            return LoopFinding(looping=True, period=period, block=block, repeats=repeats)
    return LoopFinding(looping=False, period=0, block="", repeats=0)


def longest_periodic_run(text: str, *, min_period: int = 1) -> tuple[str, int]:
    """The longest block that repeats consecutively anywhere in ``text``, and how many times it repeats.

    This exists because the tail rule below was measured against E2605's own outputs and missed the case it
    is supposed to catch: a sample whose middle is ``grounded grounded grounded`` and whose end is a heading
    is repetitive, and reports no tail loop. The two numbers answer different questions — *did the decode
    settle into a cycle* (tail) versus *is there a cycle in this text at all* (here) — so both are recorded.

    One occurrence counts as a repeat of 1, so the pair ``("", 1)`` means no consecutive repetition of any
    block at all.
    """
    if min_period < 1:
        message = f"min_period must be at least 1, got {min_period}"
        raise ValueError(message)
    best_block, best_repeats = "", 1
    for start in range(len(text)):
        for period in range(min_period, (len(text) - start) // 2 + 1):
            block = text[start : start + period]
            repeats = 1
            cursor = start + period
            while text[cursor : cursor + period] == block:
                repeats += 1
                cursor += period
            if repeats < 2:
                continue  # one occurrence is not a cycle, and the longest such block means nothing
            if repeats > best_repeats or (repeats == best_repeats and len(block) > len(best_block)):
                best_block, best_repeats = block, repeats
    return best_block, best_repeats


def _run_probe(
    model: LanguageModel, probe: GenerationProbe, *, sampler: Callable[..., Any] | None = None
) -> dict[str, Any]:
    """Ask one probe and record what came back, including a refusal or an unexpected failure."""
    messages = [Message(role=Role.USER, content=probe.prompt)]
    params = GenerationParams(max_new_tokens=probe.max_new_tokens, temperature=probe.temperature)
    record: dict[str, Any] = {
        "label": probe.label,
        "prompt": probe.prompt,
        "why": probe.why,
        "requested_max_new_tokens": probe.max_new_tokens,
        "temperature": probe.temperature,
    }
    try:
        result = (
            model.generate(messages, params)
            if sampler is None
            else sampler(model, messages, params)
        )
    except Exception as exc:
        return {**record, "status": "raised", "error": f"{type(exc).__name__}: {exc}"}
    text = result.text
    block, times = longest_repeated_block(text)
    run_block, run_repeats = longest_periodic_run(text)
    return {
        **record,
        "status": "generated",
        "model_id": result.model_id,
        "finish_reason": result.finish_reason.value,
        "characters": len(text),
        "reached_token_bound": result.finish_reason is FinishReason.LENGTH,
        "text": text,
        "bigram_repetition": ngram_repetition_ratio(text, 2),
        "trigram_repetition": ngram_repetition_ratio(text, 3),
        "longest_repeated_block": block,
        "longest_repeated_block_count": times,
        "longest_periodic_run": run_block,
        "longest_periodic_run_repeats": run_repeats,
        "loop": detect_loop(text).as_dict(),
        "empty_output": text == "",
    }


def evaluate_generation(
    model: LanguageModel,
    *,
    probes: Sequence[GenerationProbe] = CONTINUATION_PROBES,
    edge_probes: Sequence[GenerationProbe] = EDGE_PROBES,
    sampler: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the suite against one model and return the raw samples with their automated statistics.

    Args:
        model: Anything behind the boundary. The suite does not know or care which adapter it is, which is
            the point: the same probes must be survivable by a development double and by a trained model.
        probes: The continuation prompts. Defaults to the fixed suite; passing something else makes a
            comparison to a recorded run invalid, so a report names what it was given.
        edge_probes: The inputs that must not crash the model.
        sampler: An alternative entry to the model, used by callers that generate through
            :meth:`chakaso.models.base.LanguageModel`-adjacent APIs such as ``sample``. Left ``None`` in a
            recorded run, which goes through ``generate`` exactly as the runtime would.

    Raises:
        ValueError: both probe lists are empty. An evaluation that ran no probes would return a report with
            zero samples, which reads exactly like a report that was produced.
    """
    continuation = [_run_probe(model, probe, sampler=sampler) for probe in probes]
    edges = [_run_probe(model, probe, sampler=sampler) for probe in edge_probes]
    if not probes and not edge_probes:
        message = (
            "the generation suite was given no probes; an empty evaluation would report zero samples and "
            "read like a run that happened"
        )
        raise ValueError(message)
    all_samples = continuation + edges
    generated = [sample for sample in continuation if sample["status"] == "generated"]
    loops = [sample for sample in generated if sample["loop"]["looping"]]
    ratios = [
        sample["bigram_repetition"]
        for sample in generated
        if sample["bigram_repetition"] is not None
    ]
    return {
        "probes": [probe.label for probe in probes],
        "edge_probes": [probe.label for probe in edge_probes],
        "model_id": continuation[0].get("model_id") if continuation else None,
        "samples": continuation,
        "edge_samples": edges,
        "automated_summary": {
            "samples_generated": len(generated),
            "samples_raised": sum(1 for sample in all_samples if sample["status"] == "raised"),
            "looping_samples": [sample["label"] for sample in loops],
            "periodic_run_samples": [
                sample["label"]
                for sample in generated
                if sample["longest_periodic_run_repeats"] >= 3
            ],
            "looping_fraction": round(len(loops) / len(generated), 4) if generated else None,
            "mean_bigram_repetition": round(sum(ratios) / len(ratios), 6) if ratios else None,
            "empty_outputs": [
                sample["label"] for sample in all_samples if sample.get("empty_output")
            ],
            "bound_reached": [
                sample["label"] for sample in generated if sample["reached_token_bound"]
            ],
        },
        "interpretation_limit": INTERPRETATION_LIMIT,
    }


def summarise_generation_evaluation(report: dict[str, Any]) -> str:
    """One line: how many samples loop, how many repeat a cycle anywhere, and what repetition averaged.

    Both loop numbers are printed because they disagree, and the disagreement is the finding: a sample can
    repeat a block eight times in its middle and still end on something else, which the tail rule reports as
    no loop and the whole-output rule reports as a cycle.
    """
    summary = report["automated_summary"]
    looping = summary["looping_samples"]
    periodic = summary.get("periodic_run_samples", [])
    fraction = summary["looping_fraction"]
    mean = summary["mean_bigram_repetition"]
    return (
        f"{summary['samples_generated']} samples, "
        f"{len(looping)} tail-looping"
        f"{f' ({fraction:.0%})' if fraction is not None else ''}, "
        f"{len(periodic)} with a periodic run of 3 or more, "
        f"mean bigram repetition {mean if mean is not None else 'undefined'}, "
        f"{len(summary['empty_outputs'])} empty"
    )
