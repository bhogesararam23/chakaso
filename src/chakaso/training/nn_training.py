"""Mini-batch training for the Transformer, as a Python API rather than a command.

[ADR-0033](../../../docs/decisions/ADR-0033-model-centred-product-contract.md) requires that training never
needs an interactive session and that a checkpoint be produced by a callable, testable path. This module is
that path: it takes packed examples from one side of a split, an optimization configuration and a seed, and
returns the measured curve, the validation-selected parameters, and everything an experiment record has to
state.

The properties it is built around:

* **The batch plan is a pure function of seed, update index and data size.** No Python iteration order and
  no framework RNG state left to chance, so a resumed run draws exactly the batches the uninterrupted run
  would have drawn. ``max_steps`` is the absolute ceiling for a run, not an additional count, for the same
  reason.
* **Training loss and held-out loss are separate numbers, always.** A recorded training number is the loss
  of the batch actually updated at that step; a held-out number is computed by :func:`evaluate_loss` over
  every position of the evaluation set, recombined by position count rather than averaged per batch.
  Neither is derived from the other, and the held-out side is supplied by the caller — this module cannot
  see the training documents and score them as unseen.
* **Selection happens on validation, and the rule is written down once.** The returned artifact is the
  parameter snapshot at the step where validation loss improved by more than ``min_delta``; patience counts
  evaluations without that improvement. Test data is never read here. The loop applies the rule
  incrementally and then checks itself against :func:`choose_validation_best`, the pure form of the same
  rule, and refuses to return if the two disagree.
* **Interruption safety.** When a path is given, a *resumable* artifact — current parameters plus Adam
  moments plus the step — is rewritten at every evaluation, so a killed run leaves something recoverable on
  disk. That file is deliberately not the usable artifact: the selected parameters arrive with the outcome.

Requires the optional ``[nn]`` extra.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from chakaso.training.errors import TrainingError
from chakaso.training.nn_checkpoint import ModelArtifact, describe_environment, save_artifact
from chakaso.training.packing import PackedExample
from chakaso.training.subword import SubwordVocabulary
from chakaso.training.transformer import CausalTransformer, TransformerConfig

__all__ = [
    "StepRecord",
    "TrainingOutcome",
    "TrainingScheduleError",
    "TransformerTrainingConfig",
    "choose_validation_best",
    "evaluate_loss",
    "plan_batch",
    "resume_state_from",
    "train_transformer",
]


class TrainingScheduleError(TrainingError):
    """A training request cannot be scheduled: empty data, or a configuration that cannot make progress."""


@dataclass(frozen=True, slots=True)
class TransformerTrainingConfig:
    """Optimization, evaluation and reproducibility settings for one Transformer run.

    Attributes:
        batch_size: Sequences per optimizer step.
        max_steps: Absolute ceiling on optimizer steps, counted from zero and not from ``start_step``.
        learning_rate: Peak AdamW learning rate, approached by linear warmup.
        weight_decay: AdamW decoupled weight decay.
        warmup_steps: Updates over which the learning rate ramps to its peak.
        gradient_clip_norm: Maximum gradient norm; zero disables clipping.
        seed: Master seed for initialization and for the batch plan.
        eval_every: Evaluate and record every this-many-th step (and at the final step).
        patience: Evaluations without a ``min_delta`` improvement before the run stops.
        min_delta: Improvement, in nats, that counts against patience.
        architecture: The Transformer this run builds when no model is supplied.
    """

    batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_clip_norm: float
    seed: int
    eval_every: int
    patience: int
    min_delta: float
    architecture: TransformerConfig

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_steps", "warmup_steps", "eval_every", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                message = f"{name} must be an integer >= 0, got {value!r}"
                raise ValueError(message)
        for name in ("batch_size", "max_steps", "eval_every"):
            if getattr(self, name) < 1:
                message = f"{name} must be >= 1, got {getattr(self, name)}"
                raise ValueError(message)
        if self.warmup_steps > self.max_steps:
            message = (
                f"warmup_steps ({self.warmup_steps}) cannot exceed max_steps ({self.max_steps})"
            )
            raise ValueError(message)
        if self.eval_every > self.max_steps:
            message = (
                f"eval_every ({self.eval_every}) cannot exceed max_steps ({self.max_steps}); "
                "a run that never evaluates cannot select on validation"
            )
            raise ValueError(message)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            message = f"seed must be an integer, got {self.seed!r}"
            raise ValueError(message)
        for name in ("learning_rate", "weight_decay", "gradient_clip_norm", "min_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                message = f"{name} must be a number >= 0, got {value!r}"
                raise ValueError(message)
        if self.learning_rate <= 0:
            message = f"learning_rate must be positive, got {self.learning_rate!r}"
            raise ValueError(message)

    def as_dict(self) -> dict[str, Any]:
        """The configuration as plain data, minus the architecture, which is recorded separately."""
        values = asdict(self)
        values.pop("architecture")
        return values


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One recorded moment: what was trained, what the held-out set said, and the gradient scale."""

    step: int
    train_loss: float
    validation_loss: float | None
    positions_trained: int
    gradient_norm: float
    learning_rate: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """A finished run: the curve, the selected parameters, and the identity of everything used."""

    config: TransformerTrainingConfig
    records: tuple[StepRecord, ...]
    state: Mapping[str, torch.Tensor]
    final_state: Mapping[str, torch.Tensor]
    best_step: int
    best_validation_loss: float
    final_validation_loss: float
    stopped_early: bool
    vocabulary: SubwordVocabulary
    corpus_fingerprint: str
    train_positions: int
    validation_positions: int
    parameter_count: int
    optimizer_state: Mapping[str, Any] | None = None

    @property
    def initial_validation_loss(self) -> float:
        """The held-out loss before any update — the number the curve is measured against."""
        value = self.records[0].validation_loss
        if value is None:
            message = "the recorded curve has no step-zero validation loss"
            raise TrainingError(message)
        return value

    def artifact(self) -> ModelArtifact:
        """The usable artifact of this run: selected parameters plus the identity that reproduces them.

        No optimizer state is attached on purpose. An artifact that carries the *selected* parameters but
        claims to be resumable would let a later run continue from moments computed against different
        weights — a schedule that looks right and is not.
        """
        return ModelArtifact(
            config=self.config.architecture,
            state=self.state,
            vocabulary=self.vocabulary,
            corpus_fingerprint=self.corpus_fingerprint,
            run_metadata={
                "kind": "usable",
                "training": self.config.as_dict(),
                "architecture": self.config.architecture.as_dict(),
                "tokenizer_fingerprint": self.vocabulary.fingerprint,
                "train_positions": self.train_positions,
                "validation_positions": self.validation_positions,
                "parameter_count": self.parameter_count,
                "stopped_early": self.stopped_early,
                "best_step": self.best_step,
                "best_validation_loss_nats": self.best_validation_loss,
                "final_validation_loss_nats": self.final_validation_loss,
                "initial_validation_loss_nats": self.initial_validation_loss,
            },
            step=self.best_step,
            optimizer_state=None,
            environment=describe_environment(),
        )


def plan_batch(
    total_examples: int,
    update_index: int,
    *,
    batch_size: int,
    seed: int,
) -> tuple[int, ...]:
    """The example indices for the ``update_index``-th optimizer update, as a pure function.

    ``update_index`` counts updates already performed, so a run's first update is index 0 and a run resumed
    after ``s`` updates asks for index ``s``. Updates run in epochs of ``ceil(total_examples / batch_size)``
    steps; each epoch is a seeded shuffle of the example indices, and the last batch of an epoch may be
    shorter. Nothing depends on iteration order or process state, which is what lets a resumed run land on
    exactly the batches the uninterrupted run would have drawn.

    Raises:
        TrainingScheduleError: there are no examples to schedule, or the index is negative.
    """
    if total_examples < 1:
        message = "a training schedule needs at least one example"
        raise TrainingScheduleError(message)
    if update_index < 0:
        message = f"update_index must not be negative, got {update_index}"
        raise TrainingScheduleError(message)
    steps_per_epoch = -(-total_examples // batch_size)
    epoch = update_index // steps_per_epoch
    position = update_index % steps_per_epoch
    order = list(range(total_examples))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    start = position * batch_size
    return tuple(order[start : start + batch_size])


def choose_validation_best(
    records: Sequence[StepRecord], *, patience: int, min_delta: float
) -> tuple[int, float, bool]:
    """Apply the selection rule to a recorded curve: return ``(best_step, best_loss, stopped_early)``.

    Only steps carrying a validation number are candidates, and an improvement must exceed ``min_delta`` —
    otherwise a curve drifting by 1e-9 forever would never stop. The first record is the pre-update
    baseline and is a candidate, so a run that learns nothing selects step 0 rather than a lucky early step.
    """
    scored = [record for record in records if record.validation_loss is not None]
    if not scored:
        message = "no recorded step carries a validation loss, so there is nothing to select on"
        raise TrainingScheduleError(message)

    best_step = scored[0].step
    best_loss = float(scored[0].validation_loss or 0.0)
    stagnant = 0
    for record in scored[1:]:
        value = float(record.validation_loss or 0.0)
        if value < best_loss - min_delta:
            best_loss = value
            best_step = record.step
            stagnant = 0
        else:
            stagnant += 1
            if stagnant > patience:
                return best_step, best_loss, True
    return best_step, best_loss, False


def evaluate_loss(
    model: CausalTransformer,
    examples: Sequence[PackedExample],
    *,
    batch_size: int,
) -> tuple[float, int]:
    """Mean next-token cross-entropy over *every* position of ``examples``, and the position count.

    Batch means are recombined by position count rather than averaged, so a final short batch cannot tilt
    the number: what is reported is loss per predicted token, exactly.

    Raises:
        TrainingScheduleError: there are no examples to score.
    """
    if not examples:
        message = "cannot evaluate held-out loss over zero examples"
        raise TrainingScheduleError(message)
    model.eval()
    total_nats = 0.0
    positions = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            inputs = torch.tensor([list(example.input_ids) for example in chunk], dtype=torch.int64)
            targets = torch.tensor([list(example.targets) for example in chunk], dtype=torch.int64)
            _, loss = model(inputs, targets)
            assert loss is not None
            count = int(inputs.numel())
            total_nats += float(loss.detach()) * count
            positions += count
    return total_nats / positions, positions


def _snapshot(model: CausalTransformer) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _learning_rate(config: TransformerTrainingConfig, step: int) -> float:
    """Linear warmup to the peak rate; the schedule is a function of step, not of accumulated state."""
    if config.warmup_steps <= 0 or step >= config.warmup_steps:
        return config.learning_rate
    return config.learning_rate * (step / config.warmup_steps)


def train_transformer(
    *,
    train_examples: Sequence[PackedExample],
    validation_examples: Sequence[PackedExample],
    vocabulary: SubwordVocabulary,
    config: TransformerTrainingConfig,
    corpus_fingerprint: str,
    model: CausalTransformer | None = None,
    resume_state: Mapping[str, torch.Tensor] | None = None,
    resume_optimizer: Mapping[str, Any] | None = None,
    start_step: int = 0,
    on_step: Callable[[StepRecord], None] | None = None,
    checkpoint_path: Path | None = None,
) -> TrainingOutcome:
    """Train a Transformer on packed examples, selecting the returned parameters on validation loss.

    ``max_steps`` is the absolute ceiling: a resumed run reaches the same step number an uninterrupted run
    would have reached. ``resume_state`` restores weights and ``resume_optimizer`` restores Adam moments, so
    the batch plan lines up with the original schedule (see :func:`plan_batch`). Supplying one without the
    other continues the run but not the schedule, and is the caller's documented choice, not a default.

    Raises:
        TrainingScheduleError: either side has no examples, or ``start_step`` is negative or already at the
            ceiling.
    """
    if not train_examples:
        message = "cannot train on zero examples"
        raise TrainingScheduleError(message)
    if not validation_examples:
        message = "cannot select on validation without at least one held-out example"
        raise TrainingScheduleError(message)
    if start_step < 0:
        message = f"start_step must not be negative, got {start_step}"
        raise TrainingScheduleError(message)
    if start_step >= config.max_steps:
        message = (
            f"start_step ({start_step}) has already reached max_steps ({config.max_steps}); max_steps is "
            "the absolute ceiling, so resuming past it would train a schedule the configuration does not "
            "describe"
        )
        raise TrainingScheduleError(message)

    if model is None:
        torch.manual_seed(config.seed)
        model = CausalTransformer(config.architecture)
    if resume_state is not None:
        model.load_state_dict(dict(resume_state), strict=True)

    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if resume_optimizer is not None:
        optimizer.load_state_dict(dict(resume_optimizer))

    baseline_validation, validation_positions = evaluate_loss(
        model, validation_examples, batch_size=config.batch_size
    )
    # The batch the next update will draw, so a fresh run's first training number is a pre-update number.
    baseline_indices = plan_batch(
        len(train_examples), start_step, batch_size=config.batch_size, seed=config.seed
    )
    baseline_train, _ = evaluate_loss(
        model,
        [train_examples[index] for index in baseline_indices],
        batch_size=config.batch_size,
    )
    records: list[StepRecord] = [
        StepRecord(
            step=start_step,
            train_loss=baseline_train,
            validation_loss=baseline_validation,
            positions_trained=0,
            gradient_norm=0.0,
            learning_rate=_learning_rate(config, start_step),
            elapsed_seconds=0.0,
        )
    ]
    if on_step is not None:
        on_step(records[-1])

    best_state = _snapshot(model)
    best_step = start_step
    best_validation = baseline_validation
    stagnant = 0
    stopped_early = False
    started = time.perf_counter()
    train_positions = len(train_examples) * len(train_examples[0].input_ids)

    for offset in range(1, config.max_steps - start_step + 1):
        step = start_step + offset
        # Planned by absolute update index, so a resumed run draws exactly the batches the uninterrupted run
        # would have drawn; that is what makes the resume property equivalence rather than resemblance.
        indices = plan_batch(
            len(train_examples), step - 1, batch_size=config.batch_size, seed=config.seed
        )
        batch = [train_examples[index] for index in indices]
        inputs = torch.tensor([list(example.input_ids) for example in batch], dtype=torch.int64)
        targets = torch.tensor([list(example.targets) for example in batch], dtype=torch.int64)

        learning_rate = _learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        model.train()
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ returns the norm *before* clipping, which is what makes "clipping happened"
        # observable in the record rather than a claim about a value that was thrown away.
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters,
                config.gradient_clip_norm if config.gradient_clip_norm > 0 else float("inf"),
            )
        )
        optimizer.step()

        if step % config.eval_every == 0 or step == config.max_steps:
            validation_loss, _ = evaluate_loss(
                model, validation_examples, batch_size=config.batch_size
            )
            record = StepRecord(
                step=step,
                train_loss=float(loss.detach()),
                validation_loss=validation_loss,
                positions_trained=int(inputs.numel()),
                gradient_norm=norm,
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - started,
            )
            records.append(record)
            if on_step is not None:
                on_step(record)
            if validation_loss < best_validation - config.min_delta:
                best_validation = validation_loss
                best_step = step
                best_state = _snapshot(model)
                stagnant = 0
            else:
                stagnant += 1
            if checkpoint_path is not None:
                save_artifact(
                    _interim_artifact(
                        model,
                        config=config,
                        vocabulary=vocabulary,
                        corpus_fingerprint=corpus_fingerprint,
                        step=step,
                        best_step=best_step,
                        best_validation=best_validation,
                        train_positions=train_positions,
                        validation_positions=validation_positions,
                        optimizer_state=optimizer.state_dict(),
                    ),
                    checkpoint_path,
                    allow_overwrite=True,
                )
            if stagnant > config.patience:
                stopped_early = True
                break

    final_state = _snapshot(model)
    last_validation = records[-1].validation_loss
    final_validation = baseline_validation if last_validation is None else float(last_validation)
    model.load_state_dict(best_state, strict=True)

    checked_step, checked_loss, checked_stopped = choose_validation_best(
        records, patience=config.patience, min_delta=config.min_delta
    )
    if (
        checked_step != best_step
        or abs(checked_loss - best_validation) > 1e-12
        or checked_stopped != stopped_early
    ):
        message = (
            "the loop's incremental selection and choose_validation_best disagree: loop step "
            f"{best_step} loss {best_validation!r} stopped {stopped_early} vs rule step {checked_step} "
            f"loss {checked_loss!r} stopped {checked_stopped}"
        )
        raise TrainingScheduleError(message)

    return TrainingOutcome(
        config=config,
        records=tuple(records),
        state=best_state,
        final_state=final_state,
        best_step=best_step,
        best_validation_loss=best_validation,
        final_validation_loss=final_validation,
        stopped_early=stopped_early,
        vocabulary=vocabulary,
        corpus_fingerprint=corpus_fingerprint,
        train_positions=train_positions,
        validation_positions=validation_positions,
        parameter_count=model.parameter_count,
        optimizer_state=optimizer.state_dict(),
    )


def resume_state_from(
    artifact: ModelArtifact,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any], int]:
    """The ``(state, optimizer_state, start_step)`` triple that lets a saved run continue.

    Only an artifact written during training carries optimizer moments. A finished run's artifact carries
    the selected parameters and nothing else, and resuming from one would silently restart Adam's moments —
    a different schedule that looks like the same one. That is refused rather than approximated.

    Raises:
        TrainingScheduleError: the artifact carries no optimizer state.
    """
    if artifact.optimizer_state is None:
        message = (
            "this artifact carries no optimizer state, so it cannot be continued as a run; it is a usable "
            "model, not a resume point"
        )
        raise TrainingScheduleError(message)
    return dict(artifact.state), dict(artifact.optimizer_state), artifact.step


def _interim_artifact(
    model: CausalTransformer,
    *,
    config: TransformerTrainingConfig,
    vocabulary: SubwordVocabulary,
    corpus_fingerprint: str,
    step: int,
    best_step: int,
    best_validation: float,
    train_positions: int,
    validation_positions: int,
    optimizer_state: Mapping[str, Any],
) -> ModelArtifact:
    """A resume point: the *current* parameters, the moments, and the step — not the selected weights."""
    return ModelArtifact(
        config=config.architecture,
        state=_snapshot(model),
        vocabulary=vocabulary,
        corpus_fingerprint=corpus_fingerprint,
        run_metadata={
            "kind": "resumable",
            "current_step": step,
            "best_step": best_step,
            "best_validation_loss_nats": best_validation,
            "train_positions": train_positions,
            "validation_positions": validation_positions,
            "parameter_count": model.parameter_count,
            "training": config.as_dict(),
            "architecture": config.architecture.as_dict(),
            "tokenizer_fingerprint": vocabulary.fingerprint,
        },
        step=step,
        optimizer_state=dict(optimizer_state),
        environment=describe_environment(),
    )
