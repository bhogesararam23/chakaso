"""The resolved configuration for a training run.

Every number that shapes the experiment lives here, validated on construction, so a run is fully
described by one frozen object that can be hashed into a fingerprint and reproduced exactly. This is
the discipline ADR-0020 applies to results and ``configs/`` applies to the runtime: a value that could
differ between runs is configuration, and a bad value fails at once.

The module is deliberately pure standard library: a configuration object is *data*, not numerics, so it
can be imported, validated and type-checked on a dependency-free machine where ``numpy`` is absent. The
training loop that consumes it is the part that needs ``numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TrainingConfig"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """The architecture, optimization and reproducibility settings of one experiment.

    Attributes:
        context_size: How many previous characters the model conditions on.
        embed_dim: Width of each character embedding row.
        hidden_dim: Width of the single hidden layer.
        learning_rate: Adam step size.
        seed: The master seed for random initialization and any data shuffling.
        steps: How many full-batch optimization steps to run.
        eval_every: Record a held-out evaluation every this-many-th step (and at the final step).
    """

    context_size: int
    embed_dim: int
    hidden_dim: int
    learning_rate: float
    seed: int
    steps: int
    eval_every: int

    def __post_init__(self) -> None:
        for name in ("context_size", "embed_dim", "hidden_dim", "steps", "eval_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                message = f"{name} must be an integer >= 1, got {value!r}"
                raise ValueError(message)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            message = f"seed must be an integer, got {self.seed!r}"
            raise ValueError(message)
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            message = f"learning_rate must be a number, got {self.learning_rate!r}"
            raise ValueError(message)
        if self.learning_rate <= 0:
            message = f"learning_rate must be positive, got {self.learning_rate!r}"
            raise ValueError(message)

    def as_dict(self) -> dict[str, object]:
        """The configuration as a plain mapping, for recording and fingerprinting."""
        values: dict[str, object] = {
            "context_size": self.context_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "steps": self.steps,
            "eval_every": self.eval_every,
        }
        return values
