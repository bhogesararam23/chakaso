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

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from chakaso.training.decoder import ChakasoDecoder, DecoderConfig
from chakaso.training.errors import TrainingError, TrainingScheduleError
from chakaso.training.nn_checkpoint import ModelArtifact, describe_environment, save_artifact
from chakaso.training.packing import PackedExample
from chakaso.training.subword import SubwordVocabulary
from chakaso.training.transformer import CausalTransformer, TransformerConfig

__all__ = [
    "COMPLETED_BUDGET_CEILING",
    "COMPLETED_CONVERGED",
    "COMPLETED_STOP_REQUESTED",
    "ResumePoint",
    "StepRecord",
    "TrainingOutcome",
    "TrainingScheduleError",
    "TransformerTrainingConfig",
    "build_model",
    "choose_validation_best",
    "evaluate_loss",
    "plan_batch",
    "resume_state_from",
    "train_transformer",
    "verify_resume_compatibility",
]


@dataclass(frozen=True, slots=True)
class TransformerTrainingConfig:
    """Optimization, evaluation and reproducibility settings for one Transformer run.

    Attributes:
        batch_size: Sequences per optimizer step, before accumulation.
        max_steps: Absolute ceiling on optimizer steps, counted from zero and not from ``start_step``.
        learning_rate: Peak AdamW learning rate, approached by linear warmup.
        weight_decay: AdamW decoupled weight decay.
        warmup_steps: Updates over which the learning rate ramps to its peak.
        gradient_clip_norm: Maximum gradient norm; zero disables clipping.
        seed: Master seed for initialization and for the batch plan.
        eval_every: Evaluate and record every this-many-th step (and at the final step).
        patience: Evaluations without a ``min_delta`` improvement before the run stops.
        min_delta: Improvement, in nats, that counts against patience.
        architecture: The model this run builds when none is supplied.
        schedule: How the learning rate moves after warmup. ``constant`` is what E2605, E2606 and E2607
            ran with, and it stays the default so a measured protocol is reproducible by name. ``cosine``
            anneals to ``decay_to`` by the ceiling, which is the change E2607's budget-truncated curve
            asks for.
        decay_to: Fraction of the peak rate a cosine schedule ends at. Ignored by ``constant``.
        grad_accum_steps: Micro-batches per optimizer step. One accumulates gradients over this many
            ``batch_size`` slices, which buys a larger effective batch on a machine that cannot hold it.
        checkpoint_every: How often to rewrite the resumable artifact. Zero means every evaluation, which
            is what the measured runs did; a long run on a small machine may want it less often.
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
    architecture: TransformerConfig | DecoderConfig
    schedule: str = "constant"
    decay_to: float = 0.1
    grad_accum_steps: int = 1
    checkpoint_every: int = 0

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
        if self.schedule not in _SCHEDULES:
            message = f"schedule {self.schedule!r} is not one of {sorted(_SCHEDULES)}"
            raise ValueError(message)
        if isinstance(self.decay_to, bool) or not isinstance(self.decay_to, (int, float)):
            message = f"decay_to must be a number, got {self.decay_to!r}"
            raise ValueError(message)
        if not 0.0 <= self.decay_to < 1.0:
            message = f"decay_to must be in [0, 1), got {self.decay_to}"
            raise ValueError(message)
        if isinstance(self.grad_accum_steps, bool) or not isinstance(self.grad_accum_steps, int):
            message = f"grad_accum_steps must be an integer, got {self.grad_accum_steps!r}"
            raise ValueError(message)
        if self.grad_accum_steps < 1:
            message = f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}"
            raise ValueError(message)
        if isinstance(self.checkpoint_every, bool) or not isinstance(self.checkpoint_every, int):
            message = f"checkpoint_every must be an integer, got {self.checkpoint_every!r}"
            raise ValueError(message)
        if self.checkpoint_every < 0:
            message = f"checkpoint_every must be >= 0, got {self.checkpoint_every}"
            raise ValueError(message)

    @property
    def effective_batch_size(self) -> int:
        """Sequences per optimizer step, accumulation included — the number a record should state."""
        return self.batch_size * self.grad_accum_steps

    def as_dict(self) -> dict[str, Any]:
        """The configuration as plain data, minus the architecture, which is recorded separately."""
        values = asdict(self)
        values.pop("architecture")
        return values


_SCHEDULES = frozenset({"constant", "cosine"})


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
    positions_per_second: float = 0.0


#: How a run ended, named rather than inferred from a combination of booleans by whoever reads the record.
COMPLETED_CONVERGED = "converged"
COMPLETED_BUDGET_CEILING = "budget_ceiling"
COMPLETED_STOP_REQUESTED = "stopped_by_request"


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
    completion: str = COMPLETED_BUDGET_CEILING
    wall_clock_seconds: float = 0.0
    positions_per_second: float = 0.0

    @property
    def reached_budget(self) -> bool:
        """Whether the run trained to its declared ceiling without converging or being stopped."""
        return self.completion == COMPLETED_BUDGET_CEILING

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
                "completion": self.completion,
                "wall_clock_seconds": round(self.wall_clock_seconds, 2),
                "positions_per_second": round(self.positions_per_second, 1),
                "effective_batch_size": self.config.effective_batch_size,
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
    """The rate at ``step``: linear warmup, then the declared schedule to the ceiling.

    ``constant`` holds the peak, which is what E2605 to E2607 ran. ``cosine`` anneals from the peak at
    ``warmup_steps`` to ``learning_rate * decay_to`` at ``max_steps``. The schedule is a pure function of the
    step rather than accumulated scheduler state, so a resumed run reads the same rate the uninterrupted one
    would have — which is the only reason resumption can be called the same run.
    """
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    if config.schedule == "constant":
        return config.learning_rate
    span = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = config.decay_to
    return config.learning_rate * (floor + (1.0 - floor) * cosine)


def build_model(architecture: TransformerConfig | DecoderConfig) -> Any:
    """Construct the model one configuration names. One dispatch, so no caller guesses a family."""
    if isinstance(architecture, DecoderConfig):
        return ChakasoDecoder(architecture)
    return CausalTransformer(architecture)


@dataclass(frozen=True, slots=True)
class ResumePoint:
    """What a verified resume gives back: the state, the step it stopped at, and any budget change.

    Attributes:
        state: The saved parameters.
        optimizer_state: The saved Adam moments — without them the continuation is not the same schedule.
        start_step: The step the next update will be.
        budget_extended_from: The ceiling the saved run declared, when this run declares a larger one.
            Extending a ceiling is a real change of protocol, so it comes back as a fact for the record to
            print rather than being absorbed into a run that looks uninterrupted.
    """

    state: Mapping[str, torch.Tensor]
    optimizer_state: Mapping[str, Any]
    start_step: int
    budget_extended_from: int | None = None


def verify_resume_compatibility(
    artifact: ModelArtifact,
    *,
    config: TransformerTrainingConfig,
    corpus_fingerprint: str,
    vocabulary: SubwordVocabulary,
) -> ResumePoint:
    """Check that a saved resume point is the one *this* run may continue.

    A resume that is merely plausible is the dangerous case: continuing from another corpus, another
    architecture or another tokenizer produces a curve that looks like a run and is a mixture of two. Those
    are refused by name. A larger ``max_steps`` is treated differently, because a run that stopped at its
    ceiling — which is exactly what E2607 measured — is continued by raising the ceiling, and refusing that
    would leave no honest way to do it. Any other difference in the optimization settings is refused: the
    learning rate, the batch, the schedule or the seed would make the continuation a different protocol.

    Raises:
        TrainingScheduleError: an identity field disagrees, the optimizer state is absent, or a setting
            other than the step ceiling differs.
    """
    if artifact.optimizer_state is None:
        message = (
            "this artifact carries no optimizer state, so it cannot be continued as a run; it is a usable "
            "model, not a resume point"
        )
        raise TrainingScheduleError(message)
    disagreements: list[str] = []
    if artifact.corpus_fingerprint != corpus_fingerprint:
        disagreements.append(
            f"corpus: resumed from {artifact.corpus_fingerprint}, this run is on {corpus_fingerprint}"
        )
    if artifact.config.as_dict() != config.architecture.as_dict():
        disagreements.append(
            f"architecture: resumed from {artifact.config.as_dict()}, this run declares "
            f"{config.architecture.as_dict()}"
        )
    if artifact.vocabulary.fingerprint != vocabulary.fingerprint:
        disagreements.append(
            f"tokenizer: resumed from {artifact.vocabulary.fingerprint}, this run trained "
            f"{vocabulary.fingerprint}"
        )

    extended: int | None = None
    recorded = artifact.run_metadata.get("training")
    wanted = config.as_dict()
    if isinstance(recorded, dict):
        differing = sorted(
            key for key in set(recorded) | set(wanted) if recorded.get(key) != wanted.get(key)
        )
        if differing == ["max_steps"] and int(wanted["max_steps"]) > int(recorded["max_steps"]):
            extended = int(recorded["max_steps"])
        elif differing:
            disagreements.append(f"training configuration differs in: {differing}")
    if disagreements:
        message = "this checkpoint cannot resume the requested run — " + "; ".join(disagreements)
        raise TrainingScheduleError(message)

    state, optimizer_state, start_step = resume_state_from(artifact)
    return ResumePoint(
        state=state,
        optimizer_state=optimizer_state,
        start_step=start_step,
        budget_extended_from=extended,
    )


def train_transformer(
    *,
    train_examples: Sequence[PackedExample],
    validation_examples: Sequence[PackedExample],
    vocabulary: SubwordVocabulary,
    config: TransformerTrainingConfig,
    corpus_fingerprint: str,
    model: Any | None = None,
    resume_state: Mapping[str, torch.Tensor] | None = None,
    resume_optimizer: Mapping[str, Any] | None = None,
    start_step: int = 0,
    on_step: Callable[[StepRecord], None] | None = None,
    checkpoint_path: Path | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainingOutcome:
    """Train on packed examples, selecting the returned parameters on validation loss.

    ``max_steps`` is the absolute ceiling: a resumed run reaches the same step number an uninterrupted run
    would have reached. ``resume_state`` restores weights and ``resume_optimizer`` restores Adam moments, so
    the batch plan and the learning-rate schedule line up with the original run (see :func:`plan_batch` and
    :func:`_learning_rate`). Supplying one without the other continues the run but not the schedule, and is
    the caller's documented choice, not a default.

    ``should_stop`` is consulted at every evaluation; when it answers yes the run ends early, writes its
    resume point and reports :data:`COMPLETED_STOP_REQUESTED` — an interruption a caller asked for is not a
    failure and not a convergence, and the record says which.

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
        model = build_model(config.architecture)
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
        len(train_examples),
        start_step * config.grad_accum_steps,
        batch_size=config.batch_size,
        seed=config.seed,
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
    completion = COMPLETED_BUDGET_CEILING
    started = time.perf_counter()
    trained_positions = 0
    # Throughput is measured over gradient work only. Timing an interval that spans an evaluation would
    # charge the model for reading the validation set, and the number would then depend on how often the
    # caller asked to evaluate rather than on how fast the model trains.
    train_seconds = 0.0
    last_train_seconds = 0.0
    train_positions = len(train_examples) * len(train_examples[0].input_ids)

    for offset in range(1, config.max_steps - start_step + 1):
        step = start_step + offset
        learning_rate = _learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        # One optimizer step may be the sum of several micro-batches. Each micro-batch is planned by its own
        # absolute index, so enabling accumulation changes the effective batch and nothing else: the same
        # sequence of examples, the same schedule, the same step count.
        optimizer.zero_grad(set_to_none=True)
        positions = 0
        step_loss = 0.0
        step_started = time.perf_counter()
        for micro in range(config.grad_accum_steps):
            # Planned by absolute update index, so a resumed run draws exactly the batches the uninterrupted
            # run would have drawn; that is what makes the resume property equivalence rather than resemblance.
            micro_index = (step - 1) * config.grad_accum_steps + micro
            indices = plan_batch(
                len(train_examples), micro_index, batch_size=config.batch_size, seed=config.seed
            )
            batch = [train_examples[index] for index in indices]
            inputs = torch.tensor([list(example.input_ids) for example in batch], dtype=torch.int64)
            targets = torch.tensor([list(example.targets) for example in batch], dtype=torch.int64)
            positions += int(inputs.numel())

            model.train()
            _, loss = model(inputs, targets)
            assert loss is not None
            # Gradients average across micro-batches, so an accumulated step has the same gradient magnitude
            # as one big batch would — otherwise accumulation would silently be a larger learning rate.
            (loss / config.grad_accum_steps).backward()
            step_loss += float(loss.detach())

        train_seconds += time.perf_counter() - step_started
        trained_positions += positions
        average_loss = step_loss / config.grad_accum_steps
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
            elapsed = time.perf_counter() - started
            spent = train_seconds - last_train_seconds
            last_train_seconds = train_seconds
            record = StepRecord(
                step=step,
                train_loss=average_loss,
                validation_loss=validation_loss,
                positions_trained=positions,
                gradient_norm=norm,
                learning_rate=learning_rate,
                elapsed_seconds=elapsed,
                positions_per_second=positions / spent if spent > 0 else 0.0,
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
            cadence = config.checkpoint_every or config.eval_every
            if checkpoint_path is not None and (
                step % cadence == 0 or step == config.max_steps or stagnant > config.patience
            ):
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
                completion = COMPLETED_CONVERGED
                break
            if should_stop is not None and should_stop():
                completion = COMPLETED_STOP_REQUESTED
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

    elapsed = time.perf_counter() - started
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
        completion=completion,
        wall_clock_seconds=elapsed,
        positions_per_second=trained_positions / train_seconds if train_seconds > 0 else 0.0,
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
