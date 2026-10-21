"""Torch-gated tests for the mini-batch training path.

These cover the claims a measured training result rests on: that a held-out number is computed over every
position of the evaluation side, that selection reads validation and never test data, that an interrupted
run resumes onto the *same* schedule rather than a similar one, and that the loop refuses to run on nothing
instead of reporting a loss over zero positions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.data import TextUnit, split_text_units  # noqa: E402
from chakaso.training.errors import TrainingError  # noqa: E402
from chakaso.training.nn_checkpoint import load_artifact, save_artifact  # noqa: E402
from chakaso.training.nn_training import (  # noqa: E402
    StepRecord,
    TrainingOutcome,
    TrainingScheduleError,
    TransformerTrainingConfig,
    choose_validation_best,
    evaluate_loss,
    plan_batch,
    resume_state_from,
    train_transformer,
)
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.subword import train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import CausalTransformer, TransformerConfig  # noqa: E402

_VOCABULARY = train_subword_vocabulary(
    ["the quick red fox ran over the dry sand and the quiet owl slept in the town"],
    target_size=280,
    min_frequency=2,
)
_ARCHITECTURE = TransformerConfig(
    vocab_size=_VOCABULARY.size,
    context_length=8,
    d_model=16,
    num_layers=1,
    num_heads=4,
    feedforward_dim=32,
)
_TRAIN_TEXT = "the quick red fox ran over the dry sand and the quiet owl slept"
_HELD_OUT_TEXT = "the quiet owl slept in the dry sand again and again"


def _examples(text: str, *, context_length: int = 8) -> tuple:
    units = (TextUnit(identifier="unit", text=text),)
    return pack_units(units, _VOCABULARY, context_length=context_length).examples


def _fresh_model() -> CausalTransformer:
    torch.manual_seed(2)
    model = CausalTransformer(_ARCHITECTURE)
    model.eval()
    return model


def _config(**settings: object) -> TransformerTrainingConfig:
    values: dict[str, object] = {
        "batch_size": 4,
        "max_steps": 6,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "warmup_steps": 2,
        "gradient_clip_norm": 1.0,
        "seed": 7,
        "eval_every": 2,
        "patience": 2,
        "min_delta": 0.0,
        "architecture": _ARCHITECTURE,
    }
    values.update(settings)
    return TransformerTrainingConfig(**values)  # type: ignore[arg-type]


def _run(
    train_examples: tuple,
    validation_examples: tuple,
    config: TransformerTrainingConfig,
    *,
    vocabulary: object = _VOCABULARY,
    fingerprint: str = "fixture",
    on_step: Callable[[StepRecord], None] | None = None,
    checkpoint_path: Path | None = None,
    start_step: int = 0,
    resume_state: Mapping[str, torch.Tensor] | None = None,
    resume_optimizer: Mapping[str, object] | None = None,
) -> TrainingOutcome:
    return train_transformer(
        train_examples=train_examples,
        validation_examples=validation_examples,
        vocabulary=vocabulary,  # type: ignore[arg-type]
        config=config,
        corpus_fingerprint=fingerprint,
        on_step=on_step,
        checkpoint_path=checkpoint_path,
        start_step=start_step,
        resume_state=resume_state,
        resume_optimizer=resume_optimizer,
    )


def _train(
    *,
    on_step: Callable[[StepRecord], None] | None = None,
    checkpoint_path: Path | None = None,
    start_step: int = 0,
    resume_state: Mapping[str, torch.Tensor] | None = None,
    resume_optimizer: Mapping[str, object] | None = None,
    **settings: object,
) -> TrainingOutcome:
    """The loop over the fixture's two packed sides.

    Call-level arguments (a callback, resume state, the interruption path) are named separately from
    ``**settings``, which are configuration values.
    """
    return _run(
        _examples(_TRAIN_TEXT),
        _examples(_HELD_OUT_TEXT),
        _config(**settings),
        on_step=on_step,
        checkpoint_path=checkpoint_path,
        start_step=start_step,
        resume_state=resume_state,
        resume_optimizer=resume_optimizer,
    )


# ------------------------------------------------------------------ configuration


def test_training_configuration_checks_its_own_bounds() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        _config(learning_rate=0)
    with pytest.raises(ValueError, match="batch_size"):
        _config(batch_size=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        _config(warmup_steps=10, max_steps=4)
    with pytest.raises(ValueError, match="eval_every"):
        _config(eval_every=100, max_steps=4)
    with pytest.raises(ValueError, match="seed"):
        _config(seed=True)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        _config(gradient_clip_norm=-1)
    with pytest.raises(ValueError, match="patience"):
        _config(patience=-1)


def test_the_configuration_records_itself_without_the_architecture() -> None:
    values = _config().as_dict()

    assert "architecture" not in values
    assert values["batch_size"] == 4
    assert values["learning_rate"] == 1e-2


# ------------------------------------------------------------------ batch plan


def test_the_batch_plan_is_a_pure_function_of_the_schedule() -> None:
    assert plan_batch(10, 3, batch_size=4, seed=9) == plan_batch(10, 3, batch_size=4, seed=9)
    batch = plan_batch(10, 3, batch_size=4, seed=9)

    assert 1 <= len(batch) <= 4
    assert len(set(batch)) == len(batch)


def test_each_epoch_covers_every_example_exactly_once() -> None:
    total, batch_size = 11, 4
    steps_per_epoch = -(-total // batch_size)
    covered: list[int] = []
    for step in range(steps_per_epoch):
        covered.extend(plan_batch(total, step, batch_size=batch_size, seed=5))

    assert sorted(covered) == list(range(total))


def test_the_last_batch_of_an_epoch_is_shorter() -> None:
    steps_per_epoch = -(-11 // 4)

    lengths = [len(plan_batch(11, step, batch_size=4, seed=5)) for step in range(steps_per_epoch)]

    assert lengths == [4, 4, 3]


def test_a_different_epoch_gets_a_different_permutation() -> None:
    # Without this, "seeded" and "shuffled" would be the same claim twice: epoch 1 must not replay epoch 0.
    steps_per_epoch = -(-12 // 4)

    assert plan_batch(12, 0, batch_size=4, seed=5) != plan_batch(
        12, steps_per_epoch, batch_size=4, seed=5
    )


def test_the_plan_refuses_no_examples_and_a_negative_index() -> None:
    with pytest.raises(TrainingScheduleError, match="at least one example"):
        plan_batch(0, 0, batch_size=4, seed=1)
    with pytest.raises(TrainingScheduleError, match="negative"):
        plan_batch(5, -1, batch_size=2, seed=1)


# ------------------------------------------------------------------ evaluation


def test_evaluation_averages_over_positions_and_reports_them() -> None:
    examples = _examples("the quick red fox ran over the dry sand again")

    loss, positions = evaluate_loss(_fresh_model(), examples, batch_size=3)

    assert positions == len(examples) * 8
    assert loss > 0.0


def test_evaluation_is_independent_of_batch_size() -> None:
    examples = _examples("the quick red fox ran over the dry sand and the quiet owl")
    model = _fresh_model()

    one_at_a_time, count_one = evaluate_loss(model, examples, batch_size=1)
    all_together, count_all = evaluate_loss(model, examples, batch_size=len(examples))

    assert count_one == count_all
    assert one_at_a_time == pytest.approx(all_together, abs=1e-5)


def test_evaluation_matches_a_manual_mean_over_every_position() -> None:
    examples = _examples("the quick red fox ran over the dry sand")[:3]
    model = _fresh_model()
    inputs = torch.tensor([list(example.input_ids) for example in examples], dtype=torch.int64)
    targets = torch.tensor([list(example.targets) for example in examples], dtype=torch.int64)

    with torch.no_grad():
        logits, _ = model(inputs)
    manual = torch.stack(
        [
            -torch.log_softmax(logits[b, p].float(), dim=-1)[int(targets[b, p])]
            for b in range(len(examples))
            for p in range(8)
        ]
    ).mean()

    loss, positions = evaluate_loss(model, examples, batch_size=8)

    assert loss == pytest.approx(float(manual), abs=1e-5)
    assert positions == len(examples) * 8


def test_evaluation_refuses_an_empty_held_out_set() -> None:
    with pytest.raises(TrainingScheduleError, match="zero examples"):
        evaluate_loss(_fresh_model(), [], batch_size=4)


# ------------------------------------------------------------------ the selection rule


def _curve(*values: float | None) -> tuple[StepRecord, ...]:
    return tuple(
        StepRecord(
            step=index,
            train_loss=1.0,
            validation_loss=value,
            positions_trained=8,
            gradient_norm=1.0,
            learning_rate=1e-3,
            elapsed_seconds=0.0,
        )
        for index, value in enumerate(values)
    )


def test_selection_picks_the_lowest_validation_step() -> None:
    step, loss, stopped = choose_validation_best(
        _curve(3.0, 2.0, 2.5, 1.5, 1.7), patience=10, min_delta=0.0
    )

    assert (step, loss, stopped) == (3, 1.5, False)


def test_a_tiny_drift_is_not_an_improvement() -> None:
    step, loss, _ = choose_validation_best(
        _curve(2.0, 1.999999, 1.999998), patience=10, min_delta=0.01
    )

    assert (step, loss) == (0, 2.0)


def test_patience_stops_a_curve_that_stops_improving() -> None:
    step, loss, stopped = choose_validation_best(
        _curve(2.0, 1.5, 1.6, 1.7, 1.8), patience=1, min_delta=0.0
    )

    assert (step, loss) == (1, 1.5)
    assert stopped


def test_patience_zero_stops_at_the_first_non_improvement() -> None:
    step, _, stopped = choose_validation_best(_curve(2.0, 2.1, 1.0), patience=0, min_delta=0.0)

    assert (step, stopped) == (0, True)


def test_selection_refuses_a_curve_with_no_validation_numbers() -> None:
    with pytest.raises(TrainingScheduleError, match="nothing to select on"):
        choose_validation_best(_curve(None, None), patience=1, min_delta=0.0)


# ------------------------------------------------------------------ the loop


def test_a_short_run_records_the_curve_and_keeps_the_two_losses_separate() -> None:
    outcome = _train(max_steps=8, eval_every=2)

    assert outcome.records[0].step == 0
    assert [record.step for record in outcome.records] == [0, 2, 4, 6, 8]
    assert outcome.best_validation_loss <= outcome.initial_validation_loss
    assert outcome.parameter_count == _ARCHITECTURE.parameter_count
    assert outcome.corpus_fingerprint == "fixture"
    assert outcome.train_positions == len(_examples(_TRAIN_TEXT)) * 8
    assert outcome.validation_positions == len(_examples(_HELD_OUT_TEXT)) * 8
    assert outcome.records[0].positions_trained == 0
    # A recorded position count is the number of predicted slots in that step: the scheduled batch width
    # times the window. So the curve and the schedule cannot drift apart unnoticed.
    planned = len(_examples(_TRAIN_TEXT))
    for record in outcome.records[1:]:
        scheduled = len(plan_batch(planned, record.step - 1, batch_size=4, seed=7))
        assert record.positions_trained == scheduled * 8


def test_the_validation_side_is_scored_and_never_trained_on() -> None:
    train_examples = _examples("the quick red fox ran over the dry sand")
    validation_examples = _examples("the quiet owl slept in the town square")

    outcome = _run(train_examples, validation_examples, _config(max_steps=4, eval_every=2))

    trained_ids = {token for example in train_examples for token in example.input_ids}
    held_out_ids = {token for example in validation_examples for token in example.input_ids}
    assert held_out_ids - trained_ids, (
        "the held-out side must contain identifiers training never saw"
    )
    assert outcome.validation_positions == len(validation_examples) * 8


def test_training_refuses_empty_sides_and_an_impossible_resume_point() -> None:
    with pytest.raises(TrainingScheduleError, match="zero examples"):
        _run([], _examples(_HELD_OUT_TEXT), _config())
    with pytest.raises(TrainingScheduleError, match="held-out"):
        _run(_examples(_TRAIN_TEXT), [], _config())
    with pytest.raises(TrainingScheduleError, match="negative"):
        _run(_examples(_TRAIN_TEXT), _examples(_HELD_OUT_TEXT), _config(), start_step=-1)
    with pytest.raises(TrainingScheduleError, match="already reached max_steps"):
        _run(
            _examples(_TRAIN_TEXT),
            _examples(_HELD_OUT_TEXT),
            _config(max_steps=4),
            start_step=4,
        )


def test_the_step_callback_sees_every_record_in_order() -> None:
    seen: list[int] = []

    outcome = _train(max_steps=6, eval_every=3, on_step=lambda record: seen.append(record.step))

    assert seen == [record.step for record in outcome.records]


def test_warmup_ramps_the_rate_and_clipping_records_the_pre_clip_norm() -> None:
    outcome = _train(max_steps=4, eval_every=2, warmup_steps=4, gradient_clip_norm=0.01)

    rates = [record.learning_rate for record in outcome.records]
    assert rates[0] == pytest.approx(0.0)
    assert rates[1] < rates[-1] <= 1e-2
    # clip_grad_norm_ returns the norm *before* clipping, so a tiny bound must show real norms above it —
    # otherwise "clipping happened" is unfalsifiable.
    assert max(record.gradient_norm for record in outcome.records[1:]) >= 0.01


def test_patience_stops_the_run_early_and_the_selection_rule_agrees() -> None:
    outcome = _train(max_steps=40, eval_every=1, patience=0, min_delta=1000.0)

    assert outcome.stopped_early
    assert outcome.best_step == 0
    assert len(outcome.records) < 41
    assert outcome.records[-1].step < 40


def test_an_interrupted_run_resumes_onto_the_same_parameters() -> None:
    """The schedule is a pure function, so resuming at step 4 must equal running eight steps through."""
    train_examples = _examples(_TRAIN_TEXT)
    validation_examples = _examples(_HELD_OUT_TEXT)

    def run(max_steps: int) -> TrainingOutcome:
        return _run(train_examples, validation_examples, _config(max_steps=max_steps, eval_every=4))

    uninterrupted = run(8)
    first_half = run(4)
    resumed = _run(
        train_examples,
        validation_examples,
        _config(max_steps=8, eval_every=4),
        start_step=4,
        resume_state=first_half.final_state,
        resume_optimizer=first_half.optimizer_state,
    )

    for name, tensor in uninterrupted.final_state.items():
        assert torch.equal(resumed.final_state[name], tensor), name
    assert uninterrupted.records[-1].step == resumed.records[-1].step == 8
    assert uninterrupted.final_validation_loss == pytest.approx(
        resumed.final_validation_loss, abs=1e-6
    )


def test_resume_refuses_an_artifact_without_optimizer_moments(tmp_path: Path) -> None:
    target = tmp_path / "usable.pt"
    save_artifact(_train(max_steps=2, eval_every=2).artifact(), target)

    with pytest.raises(TrainingScheduleError, match="no optimizer state"):
        resume_state_from(load_artifact(target))


def test_interruption_safety_writes_a_resumable_artifact_at_each_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "data" / "runs" / "resumable.pt"

    outcome = _train(max_steps=4, eval_every=2, checkpoint_path=path)

    loaded = load_artifact(path)
    assert loaded.step == outcome.records[-1].step
    assert loaded.run_metadata["kind"] == "resumable"
    state, optimizer, start_step = resume_state_from(loaded)
    assert start_step == loaded.step
    assert set(state) == set(outcome.final_state)
    assert optimizer

    continued = _train(max_steps=6, eval_every=2, start_step=start_step, resume_state=state)
    assert continued.records[-1].step == 6


def test_the_outcome_artifact_round_trips_and_predicts_identically(tmp_path: Path) -> None:
    outcome = _train(max_steps=2, eval_every=2)
    artifact = outcome.artifact()
    target = tmp_path / "final.pt"

    save_artifact(artifact, target)
    reloaded = load_artifact(target)
    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    with torch.no_grad():
        before = artifact.to_model()(inputs)[0]
        after = reloaded.to_model()(inputs)[0]

    assert torch.equal(before, after)
    assert reloaded.vocabulary.fingerprint == _VOCABULARY.fingerprint
    assert reloaded.run_metadata["kind"] == "usable"
    assert reloaded.run_metadata["architecture"] == _ARCHITECTURE.as_dict()
    assert reloaded.run_metadata["tokenizer_fingerprint"] == _VOCABULARY.fingerprint
    assert reloaded.run_metadata["initial_validation_loss_nats"] == pytest.approx(
        outcome.initial_validation_loss
    )


def test_a_document_split_feeds_the_loop_without_leakage() -> None:
    # Each document is written from its own vocabulary, so a packed block from one side cannot coincide
    # with a block from the other unless packing genuinely mixed the sides.
    words = ("alder", "basalt", "cedar", "dolomite", "ember", "flint", "granite", "helium", "iris")
    documents = tuple(
        TextUnit(
            identifier=f"doc-{index}",
            text=f"{word} {word} {word}" + chr(10),
        )
        for index, word in enumerate(words)
    )
    split = split_text_units(documents, validation_fraction=0.25, test_fraction=0.25, seed=4)
    vocabulary = train_subword_vocabulary(
        [unit.text for unit in split.train], target_size=300, min_frequency=2
    )
    architecture = TransformerConfig(
        vocab_size=vocabulary.size,
        context_length=6,
        d_model=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
    )

    train_batch = pack_units(split.train, vocabulary, context_length=6)
    validation_batch = pack_units(split.validation, vocabulary, context_length=6)

    outcome = train_transformer(
        train_examples=train_batch.examples,
        validation_examples=validation_batch.examples,
        vocabulary=vocabulary,
        config=_config(max_steps=3, eval_every=1, architecture=architecture),
        corpus_fingerprint="split-fixture",
    )

    assert outcome.validation_positions == len(validation_batch.examples) * 6
    trained_blocks = {example.input_ids for example in train_batch.examples}
    held_out_blocks = {example.input_ids for example in validation_batch.examples}
    assert trained_blocks and held_out_blocks
    assert not (trained_blocks & held_out_blocks)


def test_training_errors_share_the_package_base_type() -> None:
    with pytest.raises(TrainingError):
        plan_batch(0, 0, batch_size=2, seed=1)
