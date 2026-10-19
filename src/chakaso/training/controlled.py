"""A controlled, multi-seed follow-up to the first experiment (E2602).

E2601 produced a single-seed point result whose measured held-out curve overfits. The honest next step is
not a bigger model but a tighter protocol around the one that exists:

* **three-way split** — a validation set for *selection* and a separate test set for *reporting only*, so
  early stopping on validation never contaminates the number the result cites;
* **early stopping** — the reported test loss is taken at the step where validation loss was lowest, under
  a rule fixed here in code, not chosen by looking at test;
* **several seeds** — the result is a mean and a spread across seeds, not a single high-variance point.

Every value this module returns is measured from a real training run; it has no canned numbers. It depends
on ``numpy`` (the optional ``[train]`` extra, ADR-0030).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from chakaso.training.baselines import UnigramBaseline, uniform_next_token_loss
from chakaso.training.config import TrainingConfig
from chakaso.training.corpus import CORPUS
from chakaso.training.data import character_windows, fit_vocabulary, split_three_way
from chakaso.training.model import CharCausalMLP, Params, batches_from_examples
from chakaso.training.optimizer import Adam

__all__ = [
    "CONTROLLED_CONFIG",
    "CONTROLLED_SEEDS",
    "ControlledReport",
    "SeedResult",
    "run_controlled_experiment",
]

#: The fixed architecture/optimization protocol for every seed — deliberately identical to E2601 so the
#: only new things are early stopping and the multi-seed spread, not a different model.
CONTROLLED_CONFIG = TrainingConfig(
    context_size=4,
    embed_dim=8,
    hidden_dim=12,
    learning_rate=0.02,
    seed=7,
    steps=300,
    eval_every=10,
)

#: The seeds the controlled experiment sweeps. Pre-declared, not chosen after seeing results.
CONTROLLED_SEEDS: tuple[int, ...] = (7, 21, 1337, 2024, 42)


@dataclass(frozen=True, slots=True)
class SeedResult:
    """One seed's measured outcome under the fixed protocol."""

    seed: int
    vocab_size: int
    best_step: int
    best_validation_loss: float
    test_loss_at_best: float
    final_test_loss: float
    uniform_baseline: float
    unigram_baseline: float
    train_tokens: int
    validation_tokens: int
    test_tokens: int


@dataclass(frozen=True, slots=True)
class ControlledReport:
    """The per-seed results and their aggregate spread."""

    config: TrainingConfig
    results: tuple[SeedResult, ...]
    mean_test_loss_at_best: float
    stdev_test_loss_at_best: float | None
    mean_test_loss_final: float
    mean_unigram_baseline: float

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(result.seed for result in self.results)


def _copy_params(params: Params) -> Params:
    return Params(
        embedding=params.embedding.copy(),
        hidden_weight=params.hidden_weight.copy(),
        hidden_bias=params.hidden_bias.copy(),
        output_weight=params.output_weight.copy(),
        output_bias=params.output_bias.copy(),
    )


def _run_one_seed(
    config: TrainingConfig, seed: int, validation_fraction: float, test_fraction: float
) -> SeedResult:
    split = split_three_way(
        CORPUS.text, validation_fraction=validation_fraction, test_fraction=test_fraction, seed=seed
    )
    vocabulary = fit_vocabulary(split)
    train_examples = character_windows(split.train_text, vocabulary, config.context_size)
    validation_examples = character_windows(split.validation_text, vocabulary, config.context_size)
    test_examples = character_windows(split.test_text, vocabulary, config.context_size)
    if not (train_examples and validation_examples and test_examples):
        message = (
            f"seed {seed} produced an empty train/validation/test set; cannot run the protocol"
        )
        raise ValueError(message)

    train_contexts, train_targets = batches_from_examples(train_examples, config.context_size)
    val_contexts, val_targets = batches_from_examples(validation_examples, config.context_size)
    test_contexts, test_targets = batches_from_examples(test_examples, config.context_size)

    model = CharCausalMLP(
        vocab_size=vocabulary.size,
        context_size=config.context_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
    )
    params = model.initialize(np.random.default_rng(seed))
    optimizer = Adam(learning_rate=config.learning_rate)

    best_validation_loss = model.forward(params, val_contexts, val_targets)
    best_step = 0
    best_params = _copy_params(params)

    for step in range(1, config.steps + 1):
        params = optimizer.step(params, model.gradient(params, train_contexts, train_targets))
        if step % config.eval_every == 0 or step == config.steps:
            validation_loss = model.forward(params, val_contexts, val_targets)
            # The selection rule: the first step that achieves a new minimum validation loss wins. It
            # reads validation only, never test.
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_params = _copy_params(params)

    test_loss_at_best = model.forward(best_params, test_contexts, test_targets)
    final_test_loss = model.forward(params, test_contexts, test_targets)

    train_target_ids = [example.target for example in train_examples]
    test_target_ids = [example.target for example in test_examples]
    unigram = UnigramBaseline.fit(train_target_ids, vocabulary.size).cross_entropy(test_target_ids)

    return SeedResult(
        seed=seed,
        vocab_size=vocabulary.size,
        best_step=best_step,
        best_validation_loss=best_validation_loss,
        test_loss_at_best=test_loss_at_best,
        final_test_loss=final_test_loss,
        uniform_baseline=uniform_next_token_loss(vocabulary.size),
        unigram_baseline=unigram,
        train_tokens=len(train_examples),
        validation_tokens=len(validation_examples),
        test_tokens=len(test_examples),
    )


def run_controlled_experiment(
    config: TrainingConfig = CONTROLLED_CONFIG,
    seeds: tuple[int, ...] = CONTROLLED_SEEDS,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> ControlledReport:
    """Run the fixed protocol across ``seeds`` and aggregate the measured held-out (test) losses."""
    if not seeds:
        message = "a controlled experiment needs at least one seed"
        raise ValueError(message)

    results = tuple(
        _run_one_seed(config, seed, validation_fraction, test_fraction) for seed in seeds
    )
    best_losses = [result.test_loss_at_best for result in results]
    final_losses = [result.final_test_loss for result in results]
    return ControlledReport(
        config=config,
        results=results,
        mean_test_loss_at_best=statistics.fmean(best_losses),
        stdev_test_loss_at_best=statistics.stdev(best_losses) if len(best_losses) > 1 else None,
        mean_test_loss_final=statistics.fmean(final_losses),
        mean_unigram_baseline=statistics.fmean([r.unigram_baseline for r in results]),
    )
