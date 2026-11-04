"""Torch-gated tests for what a resumed run is required to have kept.

The claim the loop makes is stronger than "it carried on": a run that was interrupted after its checkpoint and then
continued is the *same computation* as one that was never interrupted — the same parameters, the same Adam moments,
the same position on the learning-rate schedule, the same step number, and a curve that repeats no update and skips
none. Every test here goes through the whole chain a recovery actually uses (write the resumable artifact, load it
through the checkpoint loader, verify it, continue from what the check returns) rather than passing state over by
hand, because the hand-off is where this kind of work loses a number.

There is no test that a generator state is restored, and there is no state to restore: the batch plan is a pure
function of seed, update index and data size, the architecture's dropout defaults to zero so nothing draws from a
generator mid-run, and initialization is seeded only when a model is built rather than resumed. What that buys is
testable, so it is tested as an equivalence rather than asserted as a property.

Gated on the optional ``[nn]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.data import TextUnit  # noqa: E402
from chakaso.training.nn_checkpoint import load_artifact  # noqa: E402
from chakaso.training.nn_training import (  # noqa: E402
    COMPLETED_STOP_REQUESTED,
    TrainingOutcome,
    TransformerTrainingConfig,
    _learning_rate,
    resume_state_from,
    train_transformer,
    verify_resume_compatibility,
)
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.subword import SubwordVocabulary, train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import TransformerConfig  # noqa: E402

_VOCABULARY: SubwordVocabulary = train_subword_vocabulary(
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


def _examples(text: str) -> tuple[Any, ...]:
    return pack_units(
        (TextUnit(identifier="unit", text=text),), _VOCABULARY, context_length=8
    ).examples


def _config(**settings: object) -> TransformerTrainingConfig:
    values: dict[str, object] = {
        "batch_size": 4,
        "max_steps": 6,
        "learning_rate": 1e-2,
        "weight_decay": 0.01,
        "warmup_steps": 1,
        "gradient_clip_norm": 1.0,
        "seed": 7,
        "eval_every": 2,
        "patience": 8,
        "min_delta": 0.0,
        "architecture": _ARCHITECTURE,
    }
    values.update(settings)
    return TransformerTrainingConfig(**values)  # type: ignore[arg-type]


def _train(config: TransformerTrainingConfig, **kwargs: Any) -> TrainingOutcome:
    return train_transformer(
        train_examples=_examples(_TRAIN_TEXT),
        validation_examples=_examples(_HELD_OUT_TEXT),
        vocabulary=_VOCABULARY,
        config=config,
        corpus_fingerprint="fixture",
        experiment="E2699",
        split_fingerprint="split-fixture",
        **kwargs,
    )


def _recover(path: Path, config: TransformerTrainingConfig, **kwargs: Any) -> TrainingOutcome:
    """Continue a run the way a recovery does: read the file, verify it, train from what it attests."""
    point = verify_resume_compatibility(
        load_artifact(path),
        config=config,
        corpus_fingerprint="fixture",
        vocabulary=_VOCABULARY,
        experiment="E2699",
        split_fingerprint="split-fixture",
        **kwargs,
    )
    return _train(
        config,
        start_step=point.start_step,
        resume_state=point.state,
        resume_optimizer=point.optimizer_state,
    )


def _tensors(state: dict[str, Any]) -> dict[str, Any]:
    """Every tensor inside an optimizer state dict, keyed by its path, so equality is per-value."""
    found: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            found[str(key)] = value
        elif isinstance(value, dict):
            found.update({f"{key}.{name}": tensor for name, tensor in _tensors(value).items()})
    return found


def test_a_continued_run_is_the_uninterrupted_one_down_to_the_last_parameter(
    tmp_path: Path,
) -> None:
    config = _config()
    checkpoint = tmp_path / "E2699-transformer-resumable.pt"

    uninterrupted = _train(config)
    interrupted = _train(config, should_stop=lambda: True, checkpoint_path=checkpoint)
    assert interrupted.completion == COMPLETED_STOP_REQUESTED
    stopped_at = interrupted.records[-1].step
    assert 0 < stopped_at < config.max_steps

    resumed = _recover(checkpoint, config)

    for name, tensor in uninterrupted.final_state.items():
        assert torch.equal(resumed.final_state[name], tensor), name
    for name, tensor in uninterrupted.state.items():
        assert torch.equal(resumed.state[name], tensor), f"selected weights differ at {name}"
    assert resumed.best_step == uninterrupted.best_step
    assert resumed.best_validation_loss == pytest.approx(uninterrupted.best_validation_loss)
    assert resumed.stopped_early == uninterrupted.stopped_early
    assert resumed.records[-1].step == config.max_steps


def test_the_moments_are_what_make_a_continuation_the_same_run(tmp_path: Path) -> None:
    """Weights alone continue the parameters and reset the schedule, and the difference is measurable.

    ``train_transformer`` documents that supplying one of the two halves is the caller's choice rather than a
    default, and this is the reason it is worth documenting: Adam's bias correction restarts, so the first
    update after a moments-less resume takes a different step than the run being reproduced took.
    """
    config = _config()
    checkpoint = tmp_path / "E2699-transformer-resumable.pt"

    uninterrupted = _train(config)
    _train(config, should_stop=lambda: True, checkpoint_path=checkpoint)
    point = resume_state_from(load_artifact(checkpoint))

    weights_only = _train(config, start_step=point[2], resume_state=point[0])

    differing = [
        name
        for name, tensor in uninterrupted.final_state.items()
        if not torch.equal(weights_only.final_state[name], tensor)
    ]
    assert differing, "resuming without the moments reproduced the run, so moments do nothing"
    assert weights_only.records[-1].step == config.max_steps


def test_the_optimizer_state_crosses_the_boundary_rather_than_being_dropped(
    tmp_path: Path,
) -> None:
    config = _config()
    checkpoint = tmp_path / "E2699-transformer-resumable.pt"

    uninterrupted = _train(config)
    _train(config, should_stop=lambda: True, checkpoint_path=checkpoint)

    resumed = _recover(checkpoint, config)

    stored = _tensors(dict(uninterrupted.optimizer_state))
    continued = _tensors(dict(resumed.optimizer_state))
    assert stored.keys() == continued.keys()
    assert stored, "an empty optimizer state would make this test vacuous"
    for name, tensor in stored.items():
        assert torch.equal(continued[name], tensor), name


def test_a_resumed_curve_repeats_no_update_and_skips_none(tmp_path: Path) -> None:
    config = _config()
    checkpoint = tmp_path / "E2699-transformer-resumable.pt"

    uninterrupted = _train(config)
    interrupted = _train(config, should_stop=lambda: True, checkpoint_path=checkpoint)
    resumed = _recover(checkpoint, config)

    whole = [record.step for record in uninterrupted.records]
    head = [record.step for record in interrupted.records]
    tail = [record.step for record in resumed.records]

    assert tail == whole[len(head) - 1 :]
    assert tail[0] == head[-1], "the continuation must open where the interruption stopped"
    assert tail == sorted(set(tail)), "a step must be recorded once"


def test_the_resumed_run_sits_where_the_schedule_says_it_should(tmp_path: Path) -> None:
    config = _config(warmup_steps=2, schedule="cosine", decay_to=0.1)
    checkpoint = tmp_path / "E2699-transformer-resumable.pt"

    _train(config, should_stop=lambda: True, checkpoint_path=checkpoint)

    resumed = _recover(checkpoint, config)

    for record in resumed.records:
        assert record.learning_rate == pytest.approx(_learning_rate(config, record.step))
    assert resumed.records[0].learning_rate > 0.0


def test_writing_a_checkpoint_does_not_move_the_run_it_is_written_from(tmp_path: Path) -> None:
    """Checkpointing may not be a numerical setting.

    ``save_artifact`` rebuilds the model to check the artifact is loadable, and rebuilding initialises
    parameters from the global generator; unless that is forked aside, the act of saving a resume point would
    consume randomness the run had not spent (K-030). The interruption-safety guarantee is worth nothing if
    taking it changes the numbers.
    """
    config = _config(checkpoint_every=1)

    without = _train(config)
    with_checkpoints = _train(config, checkpoint_path=tmp_path / "cadence.pt")

    for name, tensor in without.final_state.items():
        assert torch.equal(with_checkpoints.final_state[name], tensor), name


def test_the_checkpoint_a_run_writes_is_the_step_that_run_reached(tmp_path: Path) -> None:
    """The resume point carries the *current* weights and the moments, not the selected ones.

    Continuing from the validation-selected snapshot would put Adam's moments against weights they were never
    computed for — a schedule that looks right and is not, which is why the usable artifact and the resume
    point are two different files of one run.
    """
    config = _config()
    checkpoint = tmp_path / "E2699-transformer-resumable.pt"

    outcome = _train(config, checkpoint_path=checkpoint)
    stored = load_artifact(checkpoint)

    assert stored.step == outcome.records[-1].step == config.max_steps
    assert stored.run_metadata["kind"] == "resumable"
    assert stored.run_metadata["current_step"] == config.max_steps
    assert stored.run_metadata["best_step"] == outcome.best_step
    assert stored.optimizer_state is not None
    for name, tensor in _tensors(dict(outcome.optimizer_state)).items():
        assert torch.equal(_tensors(dict(stored.optimizer_state))[name], tensor), name
