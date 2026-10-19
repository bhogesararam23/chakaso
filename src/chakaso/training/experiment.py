"""The driver that runs the first from-scratch experiment and measures it honestly.

This is the piece the experiment record points at: it wires the split, the loop, the baselines and the
evaluator into one reproducible run and returns every number the record will state — nothing invented,
nothing hidden. Whether the model *beats* each baseline is computed and reported as a boolean, so a
negative or a narrow result surfaces as itself rather than being spun into a success.

It re-derives the validation arrays and the model through the public layers so the reported held-out
metrics come from the same evaluator a reader could call themselves, not from a quantity the training
loop happened to keep internally.
"""

from __future__ import annotations

from dataclasses import dataclass

from chakaso.training.baselines import UnigramBaseline, uniform_next_token_loss
from chakaso.training.checkpoint import Checkpoint
from chakaso.training.config import TrainingConfig
from chakaso.training.corpus import CORPUS
from chakaso.training.data import character_windows, fit_vocabulary, split_corpus
from chakaso.training.evaluation import HeldOutMetrics, evaluate
from chakaso.training.loop import EvaluationPoint, run_training
from chakaso.training.model import CharCausalMLP, batches_from_examples

__all__ = ["DEFAULT_CONFIG", "ExperimentOutcome", "run_char_experiment"]

#: The configuration E2601 runs and the record cites. A modest budget for a 344-token corpus; the
#: resulting curve overfits after roughly step 50, which is itself a measured finding, not hidden.
DEFAULT_CONFIG = TrainingConfig(
    context_size=4,
    embed_dim=8,
    hidden_dim=12,
    learning_rate=0.02,
    seed=7,
    steps=300,
    eval_every=25,
)


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """Every measured value from one experiment run, plus the identity that reproduces it."""

    config: TrainingConfig
    vocab_size: int
    vocab_fingerprint: str
    dataset_digest: str
    parameter_count: int
    train_tokens: int
    validation_tokens: int
    initial_validation_loss: float
    final_validation_loss: float
    uniform_baseline_loss: float
    unigram_baseline_loss: float
    model_validation: HeldOutMetrics
    beats_uniform: bool
    beats_unigram: bool
    checkpoint_fingerprint: str
    curve: tuple[EvaluationPoint, ...]


def run_char_experiment(
    config: TrainingConfig, *, validation_fraction: float = 0.2
) -> ExperimentOutcome:
    """Run the full learning loop on the experiment corpus and measure it against both baselines."""
    split = split_corpus(CORPUS.text, validation_fraction=validation_fraction, seed=config.seed)
    trained = run_training(split, config)

    vocabulary = fit_vocabulary(split)
    train_examples = character_windows(split.train_text, vocabulary, config.context_size)
    validation_examples = character_windows(split.validation_text, vocabulary, config.context_size)
    train_targets = [example.target for example in train_examples]
    validation_targets = [example.target for example in validation_examples]

    uniform_loss = uniform_next_token_loss(vocabulary.size)
    unigram_loss = UnigramBaseline.fit(train_targets, vocabulary.size).cross_entropy(
        validation_targets
    )

    contexts, targets = batches_from_examples(validation_examples, config.context_size)
    model = CharCausalMLP(
        vocab_size=vocabulary.size,
        context_size=config.context_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
    )
    metrics = evaluate(trained.final_params, model, contexts, targets)

    checkpoint = Checkpoint(
        params=trained.final_params,
        config=config,
        vocab_size=vocabulary.size,
        vocab_fingerprint=vocabulary.fingerprint,
        dataset_digest=trained.dataset_digest,
        vocab_chars=vocabulary.chars,
    )

    return ExperimentOutcome(
        config=config,
        vocab_size=vocabulary.size,
        vocab_fingerprint=vocabulary.fingerprint,
        dataset_digest=trained.dataset_digest,
        parameter_count=trained.parameter_count,
        train_tokens=trained.train_tokens,
        validation_tokens=trained.validation_tokens,
        initial_validation_loss=trained.curve[0].validation_loss,
        final_validation_loss=trained.final_validation_loss,
        uniform_baseline_loss=uniform_loss,
        unigram_baseline_loss=unigram_loss,
        model_validation=metrics,
        beats_uniform=metrics.loss < uniform_loss,
        beats_unigram=metrics.loss < unigram_loss,
        checkpoint_fingerprint=checkpoint.fingerprint(),
        curve=trained.curve,
    )


def measure(
    config: TrainingConfig = DEFAULT_CONFIG, *, validation_fraction: float = 0.2
) -> dict[str, object]:
    """Run the experiment and return a deterministic JSON-ready report of every measured value.

    The curve is included so the record can show the held-out trajectory — including the overfitting at
    the tail — rather than only the endpoints. Floats are rounded to a fixed precision so the report is
    stable and readable; the unrounded values are what the checks use.
    """
    outcome = run_char_experiment(config, validation_fraction=validation_fraction)
    return {
        "experiment": "E2601",
        "corpus": {"name": CORPUS.name, "version": CORPUS.version, "licence": CORPUS.licence},
        "config": config.as_dict(),
        "vocab_size": outcome.vocab_size,
        "vocab_fingerprint": outcome.vocab_fingerprint,
        "dataset_digest": outcome.dataset_digest,
        "parameter_count": outcome.parameter_count,
        "train_tokens": outcome.train_tokens,
        "validation_tokens": outcome.validation_tokens,
        "uniform_baseline_loss_nats": round(outcome.uniform_baseline_loss, 6),
        "unigram_baseline_loss_nats": round(outcome.unigram_baseline_loss, 6),
        "initial_validation_loss_nats": round(outcome.initial_validation_loss, 6),
        "final_validation_loss_nats": round(outcome.final_validation_loss, 6),
        "model_validation_bits_per_char": round(outcome.model_validation.bits_per_char, 6),
        "model_validation_perplexity": round(outcome.model_validation.perplexity, 6),
        "model_validation_accuracy": round(outcome.model_validation.accuracy, 6),
        "beats_uniform": outcome.beats_uniform,
        "beats_unigram": outcome.beats_unigram,
        "checkpoint_fingerprint": outcome.checkpoint_fingerprint,
        "curve": [
            {
                "step": point.step,
                "train_loss": round(point.train_loss, 6),
                "validation_loss": round(point.validation_loss, 6),
            }
            for point in outcome.curve
        ],
    }
