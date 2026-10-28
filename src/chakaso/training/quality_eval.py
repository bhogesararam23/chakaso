"""A calibrated generation-quality suite: repetition measured against real text, over held-out prompts.

:mod:`chakaso.training.generation_eval` runs five hand-written prompts at greedy decoding and reports
character-level repetition. Two properties of that design made its headline number hard to read, both found
while diagnosing the released checkpoint's repetition:

* **Nothing was calibrated.** A character bigram ratio of 0.55 sounds damning until the same statistic is
  computed on *real* held-out prose of the same length and comes out at 0.34. A repetition number is only
  evidence relative to the text a person would have written.
* **The prompts were out of distribution** (markdown about this repository, for a model trained on 19th
  century prose), and decoding was only ever greedy, so "the model repeats" and "greedy decoding of this
  model repeats" were the same sentence.

This module replaces both with a procedure that can be run again and compared:

* prompts are drawn from the **test side** of a document split, by a deterministic, versioned procedure
  (:data:`PROMPT_PROCEDURE`), and each prompt keeps the true continuation of the same passage, so every
  statistic has a human reference computed the same way;
* the metrics are word-level (distinct-n, loops, periodic runs) because the characters of English repeat
  whatever the words do;
* every decoding configuration is run over several seeds, and **gates are declared as data before a run**
  (:data:`GATES`) and applied two-sided: a text that is *less* repetitive than people are is also off-target
  (a repetition penalty can do this), and is reported as such rather than celebrated.

What it does not do, stated where the numbers travel: these are surface statistics. Passing every gate means
the output is not repetitive in the ways measured; it says nothing about whether the output is true, coherent
or useful, and a model can pass by being a fluent nonsense generator.

Standard library only, so the metrics and gate arithmetic run in the dependency-free matrix. Whatever model
is passed in only needs the :class:`~chakaso.models.base.LanguageModel` boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chakaso.models import GenerationParams, LanguageModel, Message, ModelError, Role
from chakaso.models.base import StreamingGenerator
from chakaso.training.data import TextUnit, split_text_units
from chakaso.training.generation_eval import detect_loop, longest_periodic_run
from chakaso.training.manifest import read_snapshot
from chakaso.training.subword import SubwordVocabulary

__all__ = [
    "CONFIG_SETS",
    "DEFAULT_CONFIGS",
    "GATES",
    "HELD_OUT_TEST_FRACTION",
    "HELD_OUT_VALIDATION_FRACTION",
    "PROMPT_PROCEDURE",
    "QUALITY_SCHEMA",
    "SWEEP_CONFIGS",
    "DecodingConfig",
    "GateSpec",
    "HeldOutPrompt",
    "assess_gates",
    "compare_quality_reports",
    "distinct_ratio",
    "evaluate_quality",
    "heldout_test_units",
    "run_release_quality",
    "select_heldout_prompts",
    "summarise_quality",
    "text_statistics",
    "words",
]

#: Names the prompt-selection rules, so a report states which procedure produced its prompts.
PROMPT_PROCEDURE = "chakaso-heldout-prompts/1"
#: The shape of the machine-readable report; bumped when a field's meaning changes.
QUALITY_SCHEMA = "chakaso-generation-quality/1"

#: The seeds a sampled configuration is repeated over. Fixed in code, not chosen from scores.
DEFAULT_SEEDS: tuple[int, ...] = (11, 23, 37)

_SPACE_RUN = re.compile(r"\s+")
_WORD = re.compile(r"[\w\u2019']+")
#: Fewest words a text needs before a distinct-n ratio means anything; below it the ratio is undefined.
_MINIMUM_WORDS = 8


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    """One named way of turning a model's distribution into text, fixed before the run."""

    name: str
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    repeat_penalty: float = 1.0

    @property
    def sampled(self) -> bool:
        """Whether the configuration draws, and so must be repeated over seeds."""
        return self.temperature > 0.0

    def params(self, *, max_new_tokens: int, seed: int | None) -> GenerationParams:
        """The boundary request this configuration stands for."""
        return GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            seed=seed,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }


#: The fixed sweep. ``greedy`` is what every earlier report measured; the rest are the standard controls a
#: reader would ask for, with ``greedy_rp1.2`` there to show what a repetition penalty does to the gates.
DEFAULT_CONFIGS: tuple[DecodingConfig, ...] = (
    DecodingConfig("greedy"),
    DecodingConfig("greedy_rp1.2", repeat_penalty=1.2),
    DecodingConfig("t0.7_p0.9", temperature=0.7, top_p=0.9),
    DecodingConfig("t0.8_k40", temperature=0.8, top_k=40),
    DecodingConfig("t1.0", temperature=1.0),
)


#: The decoding sweep: temperature x nucleus for sampling, plus greedy with a repetition penalty. Declared
#: here so a sweep is a named set rather than whatever a script happened to loop over.
SWEEP_CONFIGS: tuple[DecodingConfig, ...] = (
    DecodingConfig("greedy"),
    *(DecodingConfig(f"greedy_rp{p}", repeat_penalty=p) for p in (1.05, 1.1, 1.2)),
    *(
        DecodingConfig(f"t{t}_p{p}" if p else f"t{t}", temperature=t, top_p=p)
        for t in (0.5, 0.7, 0.8, 0.9, 1.0)
        for p in (None, 0.9, 0.95)
    ),
)

#: Named configuration sets a runner may select.
CONFIG_SETS: dict[str, tuple[DecodingConfig, ...]] = {
    "default": DEFAULT_CONFIGS,
    "sweep": SWEEP_CONFIGS,
}


@dataclass(frozen=True, slots=True)
class GateSpec:
    """The acceptance gates, as numbers a run cannot adjust after seeing its results.

    Attributes:
        distinct1_band / distinct2_band / distinct3_band: How far a configuration's mean distinct-n may sit
            from the human reference on the same prompts, in either direction.
        max_looping_fraction: Largest share of samples whose tail is one block repeated three or more times.
        max_periodic_fraction: Largest share of samples with a consecutive repeat of three or more anywhere.
        max_replacement_characters: U+FFFD characters allowed across all samples. The reference is clean, so
            the allowance is zero: one means a decoding defect.
    """

    distinct1_band: float = 0.05
    distinct2_band: float = 0.03
    distinct3_band: float = 0.02
    max_looping_fraction: float = 0.0
    max_periodic_fraction: float = 0.05
    max_replacement_characters: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "distinct1_band": self.distinct1_band,
            "distinct2_band": self.distinct2_band,
            "distinct3_band": self.distinct3_band,
            "max_looping_fraction": self.max_looping_fraction,
            "max_periodic_fraction": self.max_periodic_fraction,
            "max_replacement_characters": self.max_replacement_characters,
        }


GATES = GateSpec()


@dataclass(frozen=True, slots=True)
class HeldOutPrompt:
    """A passage start from a held-out document, with the words that really followed it."""

    unit_id: str
    start: int
    prompt: str
    reference: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "start": self.start,
            "prompt": self.prompt,
            "reference": self.reference,
        }


def words(text: str) -> list[str]:
    """Lower-cased word tokens, with a curly apostrophe kept inside a word and punctuation dropped."""
    return _WORD.findall(text.lower())


def distinct_ratio(tokens: Sequence[str], n: int) -> float | None:
    """Share of the n-grams of ``tokens`` that are unique; ``None`` when the text is too short to say.

    1.0 means no n-gram occurs twice; a text that is one phrase repeated tends to 0. Defined over *all*
    n-grams of one text, so it measures a single output's self-repetition, never diversity across outputs.
    """
    if n < 1:
        message = f"n must be at least 1, got {n}"
        raise ValueError(message)
    if len(tokens) < max(_MINIMUM_WORDS, n):
        return None
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def text_statistics(text: str) -> dict[str, Any]:
    """Every per-text number the gates use, from the emitted characters alone.

    Loop and periodic-run detection run on the text with whitespace collapsed to single spaces and the ends
    trimmed. Prose legitimately ends on, or contains, runs of newlines and indentation; counting a trailing
    ``"\\n     "`` as a repeated block flagged a sampled continuation that was not looping at all (found on
    the first baseline run). The human reference goes through the same function, so the treatment is shared.
    """
    tokens = words(text)
    squeezed = _SPACE_RUN.sub(" ", text).strip()
    _, run_repeats = longest_periodic_run(squeezed)
    return {
        "words": len(tokens),
        "distinct1": distinct_ratio(tokens, 1),
        "distinct2": distinct_ratio(tokens, 2),
        "distinct3": distinct_ratio(tokens, 3),
        "tail_loop": detect_loop(squeezed).looping,
        "periodic_run_repeats": run_repeats,
        "replacement_characters": text.count("\ufffd"),
    }


def select_heldout_prompts(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    *,
    count: int = 24,
    prompt_chars: int = 240,
    reference_tokens: int = 64,
    seed: int = 1234,
    offset: int = 0,
) -> tuple[HeldOutPrompt, ...]:
    """Draw ``count`` prompts, one per document, from ``units`` — which the caller passes as the test side.

    The procedure is deterministic in ``(units, seed)``: eligible documents are ordered by identifier and
    shuffled with ``seed``; each yields one passage that starts at a word boundary, a prompt of about
    ``prompt_chars`` characters ending at a word boundary, and the following ``reference_tokens`` tokens as
    the human continuation. One prompt per document, so no document is over-represented and no two prompts
    can quote each other. ``offset`` skips that many documents of the shuffled order, so ``offset=count``
    gives a second set sharing no document with the first: the tuning/confirmation split a decoding sweep
    needs, so a setting chosen on one set is judged on the other. (Draws are sequential, so a set is a
    function of ``(units, seed, offset, count)`` together, not of a document alone.)

    Raises:
        ValueError: fewer eligible documents than ``count`` (an under-filled suite would report a smaller
            sample as if it were the declared one), or a non-positive size.
    """
    if count < 1 or prompt_chars < 20 or reference_tokens < 8:
        message = (
            "count, prompt_chars and reference_tokens must be positive and large enough to measure"
        )
        raise ValueError(message)
    reserve = reference_tokens * 8  # characters read after the prompt before cutting to tokens
    eligible = sorted(
        (unit for unit in units if len(unit.text) >= 400 + prompt_chars + reserve),
        key=lambda unit: unit.identifier,
    )
    if offset < 0 or len(eligible) < offset + count:
        message = (
            f"{len(eligible)} documents are long enough for a prompt and a reference, but "
            f"{offset + count} were requested (count {count} after skipping {offset})"
        )
        raise ValueError(message)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen: list[HeldOutPrompt] = []
    for unit in eligible[offset : offset + count]:
        text = unit.text
        start = rng.randrange(200, len(text) - prompt_chars - reserve)
        while start < len(text) and not text[start - 1].isspace():
            start += 1
        end = start + prompt_chars
        while end > start and not text[end - 1].isspace():
            end -= 1
        prompt = text[start:end].rstrip()
        if not prompt:
            continue
        reference_ids = vocabulary.encode(text[end : end + reserve])[:reference_tokens]
        chosen.append(
            HeldOutPrompt(
                unit_id=unit.identifier,
                start=start,
                prompt=prompt,
                reference=vocabulary.decode(reference_ids),
            )
        )
    if len(chosen) < count:
        message = f"only {len(chosen)} of {count} prompts could be cut at a word boundary"
        raise ValueError(message)
    return tuple(chosen)


def _mean(values: Sequence[float]) -> float | None:
    return round(math.fsum(values) / len(values), 6) if values else None


def _sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    centre = math.fsum(values) / len(values)
    return round(math.sqrt(math.fsum((v - centre) ** 2 for v in values) / (len(values) - 1)), 6)


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-sample statistics into the numbers a gate reads."""
    out: dict[str, Any] = {"samples": len(rows)}
    for key in ("distinct1", "distinct2", "distinct3"):
        defined = [row[key] for row in rows if row[key] is not None]
        out[key] = _mean(defined)
        out[f"{key}_sd"] = _sd(defined)
        out[f"{key}_defined"] = len(defined)
    out["looping_fraction"] = _mean([1.0 if row["tail_loop"] else 0.0 for row in rows])
    out["periodic_fraction"] = _mean(
        [1.0 if row["periodic_run_repeats"] >= 3 else 0.0 for row in rows]
    )
    out["replacement_characters"] = sum(row["replacement_characters"] for row in rows)
    out["mean_words"] = _mean([float(row["words"]) for row in rows])
    return out


def _check_determinism(
    model: LanguageModel, prompt: str, config: DecodingConfig, tokens: int
) -> dict[str, Any]:
    """Same request twice must give the same text, and streaming must join to the generated result."""
    messages = [Message(role=Role.USER, content=prompt)]
    seed = None if not config.sampled else DEFAULT_SEEDS[0]
    params = config.params(max_new_tokens=tokens, seed=seed)
    try:
        first = model.generate(messages, params)
        second = model.generate(messages, params)
        streamed = (
            "".join(chunk.text for chunk in model.stream(messages, params))
            if isinstance(model, StreamingGenerator)
            else None
        )
    except ModelError as exc:
        # A refusal is a named outcome, not a reason to lose the whole report.
        return {"config": config.name, "refused": f"{type(exc).__name__}: {exc}"}
    return {
        "config": config.name,
        "repeat_identical": first.text == second.text and first.token_ids == second.token_ids,
        "stream_equals_generate": None if streamed is None else streamed == first.text,
    }


def evaluate_quality(
    model: LanguageModel,
    prompts: Sequence[HeldOutPrompt],
    *,
    configs: Sequence[DecodingConfig] = DEFAULT_CONFIGS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_new_tokens: int = 64,
    gates: GateSpec = GATES,
    keep_text: bool = True,
) -> dict[str, Any]:
    """Run every configuration over every prompt (and seed, if sampled), then apply the gates.

    The returned mapping is the machine-readable report: ``schema``, ``procedure``, the prompts' digest, the
    gates that applied, the human reference, per-configuration aggregates and gate verdicts, the
    determinism checks, and the interpretation limit.

    Raises:
        ValueError: no prompts or no configurations — an empty run would read like a passing one.
    """
    if not prompts or not configs:
        message = "a quality evaluation needs at least one prompt and one decoding configuration"
        raise ValueError(message)

    reference_rows = [text_statistics(prompt.reference) for prompt in prompts]
    reference = _aggregate(reference_rows)

    by_config: dict[str, Any] = {}
    model_id: str | None = None
    for config in configs:
        rows: list[dict[str, Any]] = []
        refused = 0
        for index, prompt in enumerate(prompts):
            messages = [Message(role=Role.USER, content=prompt.prompt)]
            for seed in seeds if config.sampled else (None,):
                try:
                    result = model.generate(
                        messages, config.params(max_new_tokens=max_new_tokens, seed=seed)
                    )
                except ModelError:
                    refused += 1
                    continue
                model_id = model_id or result.model_id
                row = text_statistics(result.text)
                row.update(
                    {
                        "prompt_index": index,
                        "seed": seed,
                        "finish_reason": result.finish_reason.value,
                        "new_tokens": result.new_tokens,
                    }
                )
                if keep_text:
                    row["text"] = result.text
                rows.append(row)
        aggregate = _aggregate(rows)
        aggregate["refused"] = refused
        aggregate["bound_fraction"] = _mean(
            [1.0 if row["finish_reason"] == "length" else 0.0 for row in rows]
        )
        by_config[config.name] = {
            "config": config.as_dict(),
            "aggregate": aggregate,
            "gates": assess_gates(aggregate, reference, gates),
            "rows": rows,
        }

    determinism = [
        _check_determinism(model, prompts[0].prompt, config, min(max_new_tokens, 24))
        for config in (configs[0], next((c for c in configs if c.sampled), configs[0]))
    ]
    return {
        "schema": QUALITY_SCHEMA,
        "procedure": PROMPT_PROCEDURE,
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "seeds": list(seeds),
        "prompt_digest": hashlib.sha256(
            json.dumps([p.as_dict() for p in prompts], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "prompts": [p.as_dict() for p in prompts],
        "gates": gates.as_dict(),
        "reference": reference,
        "configs": by_config,
        "determinism": determinism,
        "interpretation_limit": (
            "surface statistics against a human reference on the same prompts. Passing a gate means the "
            "output is not repetitive in the ways measured; it does not mean the output is true, coherent "
            "or useful, and a model can pass by writing fluent nonsense."
        ),
    }


def assess_gates(
    aggregate: dict[str, Any], reference: dict[str, Any], gates: GateSpec = GATES
) -> dict[str, Any]:
    """Apply the declared gates to one configuration's aggregate, two-sided where a band is declared.

    Returns each gate's boolean and the numbers it was decided on. A distinct-n value *above* the human
    band fails as ``over_diverse``: text that never reuses a word is as unlike prose as text that loops.
    """
    verdict: dict[str, Any] = {}
    for key, band in (
        ("distinct1", gates.distinct1_band),
        ("distinct2", gates.distinct2_band),
        ("distinct3", gates.distinct3_band),
    ):
        observed, human = aggregate[key], reference[key]
        if observed is None or human is None:
            verdict[key] = {"passed": False, "reason": "undefined"}
            continue
        gap = round(observed - human, 6)
        verdict[key] = {
            "passed": abs(gap) <= band,
            "observed": observed,
            "human": human,
            "gap": gap,
            "band": band,
            "direction": "within"
            if abs(gap) <= band
            else ("over_diverse" if gap > 0 else "repetitive"),
        }
    verdict["looping"] = {
        "passed": (aggregate["looping_fraction"] or 0.0) <= gates.max_looping_fraction,
        "observed": aggregate["looping_fraction"],
        "limit": gates.max_looping_fraction,
    }
    verdict["periodic"] = {
        "passed": (aggregate["periodic_fraction"] or 0.0) <= gates.max_periodic_fraction,
        "observed": aggregate["periodic_fraction"],
        "limit": gates.max_periodic_fraction,
    }
    verdict["replacement_characters"] = {
        "passed": aggregate["replacement_characters"] <= gates.max_replacement_characters,
        "observed": aggregate["replacement_characters"],
        "limit": gates.max_replacement_characters,
    }
    verdict["all_passed"] = all(
        item["passed"] for key, item in verdict.items() if key != "all_passed"
    )
    return verdict


#: The split fractions every subword experiment uses. Repeated here because importing them from the
#: experiment driver would pull in torch; a gated test asserts the two stay equal.
HELD_OUT_VALIDATION_FRACTION = 0.15
HELD_OUT_TEST_FRACTION = 0.15


def heldout_test_units(snapshot_path: Path, *, split_seed: int) -> tuple[TextUnit, ...]:
    """The documents a run with this ``split_seed`` held out for testing, read from a verified snapshot.

    :func:`~chakaso.training.manifest.read_snapshot` re-checks every digest, so the text returned is the
    pinned corpus rather than whatever is on disk. The split is the experiment driver's own (same
    function, fractions and seed), so these are the documents the model never trained on or selected by.
    """
    snapshot = read_snapshot(snapshot_path)
    split = split_text_units(
        snapshot.manifest.text_units(),
        validation_fraction=HELD_OUT_VALIDATION_FRACTION,
        test_fraction=HELD_OUT_TEST_FRACTION,
        seed=split_seed,
    )
    return tuple(split.test)


def run_release_quality(
    model: LanguageModel,
    snapshot_path: Path,
    *,
    split_seed: int,
    count: int = 24,
    max_new_tokens: int = 64,
    prompt_seed: int = 1234,
    prompt_offset: int = 0,
    config_set: str = "default",
) -> dict[str, Any]:
    """Evaluate a loaded trained model on prompts drawn from its own held-out documents.

    Raises:
        ModelError: the model exposes no vocabulary (the development double), so a human reference cannot
            be cut into its tokens and nothing honest can be measured.
    """
    artifact = getattr(model, "artifact", None)
    vocabulary = getattr(artifact, "vocabulary", None)
    if vocabulary is None:
        message = (
            "this model carries no trained vocabulary; the quality suite measures trained models"
        )
        raise ModelError(message)
    units = heldout_test_units(snapshot_path, split_seed=split_seed)
    if config_set not in CONFIG_SETS:
        message = f"unknown configuration set {config_set!r}; choose one of {sorted(CONFIG_SETS)}"
        raise ValueError(message)
    prompts = select_heldout_prompts(
        units, vocabulary, count=count, seed=prompt_seed, offset=prompt_offset
    )
    report = evaluate_quality(
        model, prompts, configs=CONFIG_SETS[config_set], max_new_tokens=max_new_tokens
    )
    report["config_set"] = config_set
    report["prompt_offset"] = prompt_offset
    report["split_seed"] = split_seed
    report["heldout_documents"] = len(units)
    return report


def summarise_quality(report: dict[str, Any]) -> str:
    """A fixed-width table: one row per configuration against the human reference and the gates."""
    reference = report["reference"]
    lines = [
        f"{report['schema']} model={report['model_id']} prompts={len(report['prompts'])} "
        f"max_new_tokens={report['max_new_tokens']} digest={report['prompt_digest'][:12]}",
        f"{'config':<14}{'n':>5}{'dist1':>8}{'dist2':>8}{'dist3':>8}{'loop':>7}{'period':>8}{'fffd':>6}  gates",
        f"{'human':<14}{reference['samples']:>5}{reference['distinct1']:>8.3f}"
        f"{reference['distinct2']:>8.3f}{reference['distinct3']:>8.3f}",
    ]
    for name, entry in report["configs"].items():
        agg, gates = entry["aggregate"], entry["gates"]
        failed = [key for key, item in gates.items() if key != "all_passed" and not item["passed"]]
        verdict = "PASS" if gates["all_passed"] else "FAIL " + ",".join(failed)
        lines.append(
            f"{name:<14}{agg['samples']:>5}{_cell(agg['distinct1'])}{_cell(agg['distinct2'])}"
            f"{_cell(agg['distinct3'])}{_cell(agg['looping_fraction'], 7, 2)}"
            f"{_cell(agg['periodic_fraction'], 8, 2)}{agg['replacement_characters']:>6}  {verdict}"
        )
    return "\n".join(lines)


def _cell(value: float | None, width: int = 8, places: int = 3) -> str:
    return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.{places}f}"


def compare_quality_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Per-configuration differences between two reports, refused unless they used the same prompts.

    A comparison across different prompt sets is a comparison of prompts, so the digests must match.

    Raises:
        ValueError: the reports disagree about schema, prompt digest or token bound.
    """
    for field in ("schema", "prompt_digest", "max_new_tokens"):
        if baseline.get(field) != candidate.get(field):
            message = (
                f"reports are not comparable: {field} differs "
                f"({baseline.get(field)!r} vs {candidate.get(field)!r})"
            )
            raise ValueError(message)
    deltas: dict[str, Any] = {}
    for name, entry in candidate["configs"].items():
        if name not in baseline["configs"]:
            continue
        before, after = baseline["configs"][name]["aggregate"], entry["aggregate"]
        deltas[name] = {
            key: (
                None
                if before[key] is None or after[key] is None
                else round(after[key] - before[key], 6)
            )
            for key in (
                "distinct1",
                "distinct2",
                "distinct3",
                "looping_fraction",
                "periodic_fraction",
                "replacement_characters",
            )
        }
        deltas[name]["all_passed"] = {
            "baseline": baseline["configs"][name]["gates"]["all_passed"],
            "candidate": entry["gates"]["all_passed"],
        }
    return deltas
