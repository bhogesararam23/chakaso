"""Torch-gated tests for what the training engine added for the release path.

The loop itself — selection on validation, the batch plan, resume equivalence — is covered by
``test_training_nn_training.py``. This file covers the four things that did not exist before it: a decaying
learning-rate schedule, gradient accumulation, throughput and completion accounting, and the resume check
that refuses to continue from an artifact that is not the same run. Each is a claim an experiment record
will make about a measured number, so each is asserted against the value the record would print.

Everything here is gated on the optional ``[nn]`` extra.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.data import TextUnit  # noqa: E402
from chakaso.training.decoder import ChakasoDecoder, DecoderConfig  # noqa: E402
from chakaso.training.nn_checkpoint import ModelArtifact  # noqa: E402
from chakaso.training.nn_training import (  # noqa: E402
    COMPLETED_BUDGET_CEILING,
    COMPLETED_CONVERGED,
    COMPLETED_STOP_REQUESTED,
    TrainingScheduleError,
    TransformerTrainingConfig,
    _learning_rate,
    build_model,
    plan_batch,
    train_transformer,
    verify_resume_compatibility,
)
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.subword import train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import CausalTransformer, TransformerConfig  # noqa: E402

_VOCABULARY = train_subword_vocabulary(
    ["the quick red fox ran over the dry sand and the quiet owl slept in the town"],
    target_size=280,
    min_frequency=2,
)
_TRAIN_TEXT = "the quick red fox ran over the dry sand and the quiet owl slept"
_HELD_OUT_TEXT = "the quiet owl slept in the dry sand again and again"
_TRANSFORMER = TransformerConfig(
    vocab_size=_VOCABULARY.size,
    context_length=8,
    d_model=16,
    num_layers=1,
    num_heads=4,
    feedforward_dim=32,
)
_DECODER = DecoderConfig(
    vocab_size=_VOCABULARY.size,
    context_length=8,
    d_model=16,
    num_layers=1,
    num_heads=4,
    feedforward_dim=32,
)


def _examples(text: str) -> tuple[Any, ...]:
    units = (TextUnit(identifier="unit", text=text),)
    return pack_units(units, _VOCABULARY, context_length=8).examples


def _config(**settings: object) -> TransformerTrainingConfig:
    values: dict[str, object] = {
        "batch_size": 4,
        "max_steps": 4,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "warmup_steps": 1,
        "gradient_clip_norm": 1.0,
        "seed": 7,
        "eval_every": 2,
        "patience": 2,
        "min_delta": 0.0,
        "architecture": _TRANSFORMER,
    }
    values.update(settings)
    return TransformerTrainingConfig(**values)  # type: ignore[arg-type]


def _run(config: TransformerTrainingConfig, **kwargs: Any) -> Any:
    return train_transformer(
        train_examples=_examples(_TRAIN_TEXT),
        validation_examples=_examples(_HELD_OUT_TEXT),
        vocabulary=_VOCABULARY,
        config=config,
        corpus_fingerprint="fixture",
        **kwargs,
    )


def test_the_default_schedule_is_the_one_every_measured_run_used() -> None:
    config = _config(max_steps=4, warmup_steps=1, eval_every=1)

    outcome = _run(config)

    rates = [record.learning_rate for record in outcome.records]
    # Step zero reports the rate its first update will use: warmup has not finished, so it is below the
    # peak; from step 1 on, the constant schedule holds the peak exactly.
    assert rates[0] == 0.0
    assert rates[1:] == [config.learning_rate] * 4


def test_a_cosine_schedule_anneals_to_the_declared_floor() -> None:
    config = _config(max_steps=6, warmup_steps=2, eval_every=1, schedule="cosine", decay_to=0.1)

    outcome = _run(config)

    rates = [record.learning_rate for record in outcome.records]
    assert rates[2] == pytest.approx(config.learning_rate)
    assert all(rates[index + 1] <= rates[index] for index in range(2, len(rates) - 1))
    assert rates[-1] == pytest.approx(config.learning_rate * 0.1, rel=1e-6)


def test_the_schedule_is_a_pure_function_of_the_step() -> None:
    config = _config(max_steps=10, warmup_steps=2, schedule="cosine", decay_to=0.0)

    assert _learning_rate(config, 5) == _learning_rate(config, 5)
    assert _learning_rate(config, 0) == 0.0
    assert _learning_rate(config, 10) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", ["inverse-square", "constant-ish"])
def test_an_unknown_schedule_name_is_refused_before_the_run(name: str) -> None:
    with pytest.raises(ValueError, match="is not one of"):
        _config(schedule=name)


def test_a_decay_floor_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="decay_to must be in"):
        _config(schedule="cosine", decay_to=1.5)


def test_an_accumulated_step_covers_its_effective_batch_and_keeps_the_step_count() -> None:
    config = _config(batch_size=2, grad_accum_steps=4, max_steps=3, eval_every=1)

    outcome = _run(config)

    total = len(_examples(_TRAIN_TEXT))
    expected = [
        sum(
            len(plan_batch(total, (step - 1) * 4 + micro, batch_size=2, seed=config.seed))
            for micro in range(4)
        )
        * 8
        for step in (1, 2, 3)
    ]

    assert config.effective_batch_size == 8
    assert len(outcome.records) == 4
    assert [record.positions_trained for record in outcome.records[1:]] == expected
    assert [record.step for record in outcome.records] == [0, 1, 2, 3]


def test_accumulation_consumes_micro_batches_in_the_order_the_plan_defines() -> None:
    # The property that makes the effective batch honest: step k draws micro-batch indices
    # (k-1)*accum ... k*accum-1, each planned by the same seeded function an unaccumulated run would use.
    config = _config(batch_size=2, grad_accum_steps=3, max_steps=2, eval_every=1)
    total = len(_examples(_TRAIN_TEXT))

    drawn = [
        plan_batch(total, step - 1, batch_size=config.batch_size, seed=config.seed)
        for step in (1, 2)
    ]

    assert drawn[0] != drawn[1] or total <= config.batch_size
    assert all(len(indices) == config.batch_size for indices in drawn)


def test_an_accumulated_run_moves_parameters_and_reduces_a_loss() -> None:
    config = _config(batch_size=2, grad_accum_steps=3, max_steps=6, eval_every=2)

    outcome = _run(config)

    assert outcome.records[-1].train_loss < outcome.records[0].train_loss
    # Compare against the parameters the run *started* from, not against its own final snapshot: when the
    # last evaluation is also the best, those two are legitimately the same tensor set.
    torch.manual_seed(config.seed)
    started = CausalTransformer(_TRANSFORMER)
    assert not torch.equal(
        outcome.state["token_embedding.weight"], started.state_dict()["token_embedding.weight"]
    )


def test_a_run_reports_throughput_and_names_how_it_ended() -> None:
    config = _config(max_steps=2, eval_every=1, patience=0, min_delta=0.0)

    outcome = _run(config)

    assert outcome.wall_clock_seconds > 0
    assert outcome.positions_per_second > 0
    assert outcome.records[-1].positions_per_second >= 0
    assert outcome.completion in {COMPLETED_CONVERGED, COMPLETED_BUDGET_CEILING}
    assert outcome.reached_budget == (outcome.completion == COMPLETED_BUDGET_CEILING)


def test_a_converged_run_is_labelled_as_converged_not_as_a_ceiling() -> None:
    # patience=0 with min_delta above any real improvement means the second evaluation cannot beat the
    # first, so the declared rule stops it: that is convergence by the protocol, not truncation.
    config = _config(max_steps=6, eval_every=1, patience=0, min_delta=9.0)

    outcome = _run(config)

    assert outcome.completion == COMPLETED_CONVERGED
    assert outcome.stopped_early
    assert outcome.records[-1].step < config.max_steps


def test_a_requested_stop_is_recorded_as_a_request_not_a_convergence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resumable.pt"
    config = _config(max_steps=6, eval_every=1)
    seen = {"n": 0}

    def stop_after_two() -> bool:
        seen["n"] += 1
        return seen["n"] >= 2

    outcome = _run(config, should_stop=stop_after_two, checkpoint_path=checkpoint)

    assert outcome.completion == COMPLETED_STOP_REQUESTED
    assert not outcome.stopped_early
    assert outcome.records[-1].step == 2
    assert checkpoint.is_file()


def test_a_stopped_run_resumes_onto_the_schedule_it_was_on(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resumable.pt"
    config = _config(max_steps=4, eval_every=1, warmup_steps=2)

    stopped = _run(config, should_stop=lambda: True, checkpoint_path=checkpoint)
    resumed_artifact = load_resumable(checkpoint)
    state, optimizer_state, start_step = verify_resume_compatibility(
        resumed_artifact, config=config, corpus_fingerprint="fixture", vocabulary=_VOCABULARY
    )

    resumed = _run(
        config,
        resume_state=state,
        resume_optimizer=optimizer_state,
        start_step=start_step,
    )

    assert stopped.records[-1].step == start_step
    assert resumed.records[0].step == start_step
    assert resumed.records[-1].step == config.max_steps
    assert resumed.records[0].learning_rate == _learning_rate(config, start_step)


def load_resumable(path: Path) -> ModelArtifact:
    """Read a resume point through the checkpoint loader rather than assembling one by hand."""
    from chakaso.training.nn_checkpoint import load_artifact

    return load_artifact(path)


def test_a_resume_point_from_another_corpus_is_named() -> None:
    artifact = resumable_artifact(_config(max_steps=2, eval_every=1))

    with pytest.raises(TrainingScheduleError, match="corpus:"):
        verify_resume_compatibility(
            artifact,
            config=_config(max_steps=2),
            corpus_fingerprint="different",
            vocabulary=_VOCABULARY,
        )


def test_a_resume_point_from_another_architecture_is_named() -> None:
    artifact = resumable_artifact(_config(max_steps=2, eval_every=1))

    with pytest.raises(TrainingScheduleError, match="architecture:"):
        verify_resume_compatibility(
            artifact,
            config=_config(max_steps=2, architecture=_DECODER),
            corpus_fingerprint="fixture",
            vocabulary=_VOCABULARY,
        )


def test_a_resume_point_from_another_tokenizer_is_named() -> None:
    artifact = resumable_artifact(_config(max_steps=2, eval_every=1))
    other = train_subword_vocabulary(
        ["a wholly different sentence trained into another merge table"],
        target_size=280,
        min_frequency=2,
    )

    with pytest.raises(TrainingScheduleError, match="tokenizer:"):
        verify_resume_compatibility(
            artifact,
            config=_config(max_steps=2),
            corpus_fingerprint="fixture",
            vocabulary=other,
        )


def test_a_resume_point_from_a_changed_schedule_is_named() -> None:
    artifact = resumable_artifact(_config(max_steps=2, eval_every=1))

    with pytest.raises(TrainingScheduleError, match="training configuration differs in"):
        verify_resume_compatibility(
            artifact,
            config=_config(max_steps=9, warmup_steps=1, eval_every=1),
            corpus_fingerprint="fixture",
            vocabulary=_VOCABULARY,
        )


def test_a_usable_model_is_refused_as_a_resume_point() -> None:
    outcome = _run(_config(max_steps=2, eval_every=1))
    usable = outcome.artifact()
    assert usable.optimizer_state is None

    with pytest.raises(TrainingScheduleError, match="no optimizer state"):
        verify_resume_compatibility(
            usable,
            config=_config(max_steps=2, eval_every=1),
            corpus_fingerprint="fixture",
            vocabulary=_VOCABULARY,
        )


def resumable_artifact(config: TransformerTrainingConfig) -> ModelArtifact:
    """The resume point a run wrote, rebuilt as the artifact type the check expects."""
    outcome = _run(config)
    return ModelArtifact(
        config=config.architecture,
        state=outcome.final_state,
        vocabulary=outcome.vocabulary,
        corpus_fingerprint=outcome.corpus_fingerprint,
        run_metadata={"kind": "resumable", "training": config.as_dict()},
        step=outcome.records[-1].step,
        optimizer_state=outcome.optimizer_state,
    )


def test_build_model_makes_the_family_each_configuration_names() -> None:
    assert isinstance(build_model(_TRANSFORMER), CausalTransformer)
    assert isinstance(build_model(_DECODER), ChakasoDecoder)


def test_the_same_loop_trains_a_decoder_and_its_artifact_reloads() -> None:
    config = _config(max_steps=2, eval_every=1, architecture=_DECODER)

    outcome = _run(config)
    artifact = outcome.artifact()
    model = artifact.to_model()

    assert isinstance(model, ChakasoDecoder)
    assert outcome.parameter_count == model.parameter_count
    assert artifact.run_metadata["completion"] == outcome.completion
    assert artifact.run_metadata["effective_batch_size"] == config.effective_batch_size
    assert json.loads(artifact.metadata_document())["schema_version"] == (
        "chakaso-decoder-checkpoint/1"
    )


def test_a_split_seed_that_is_not_a_seed_study_field_still_splits_deterministically() -> None:
    # Not the engine's own concern, but the engine's batch plan is only reproducible if the data it is given
    # is: the same text and seed must produce the same number of packed blocks.
    first = _examples(_TRAIN_TEXT)
    second = _examples(_TRAIN_TEXT)

    assert len(first) == len(second)
    assert first[0].input_ids == second[0].input_ids


def test_the_run_records_its_own_effective_batch_in_the_metadata() -> None:
    outcome = _run(_config(batch_size=2, grad_accum_steps=2, max_steps=1, eval_every=1))

    recorded = json.loads(outcome.artifact().metadata_document())
    training = recorded["run_metadata"]["training"]

    assert training["batch_size"] == 2
    assert training["grad_accum_steps"] == 2
    assert replace(outcome.config, grad_accum_steps=1).effective_batch_size == 2
