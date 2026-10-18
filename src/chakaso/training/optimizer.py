"""A real optimizer: Adam over the model's parameter bundle.

The gradient computed by :mod:`chakaso.training.model` is only meaningful if an actual update
procedure consumes it. This module implements Adam — a standard, small, well-understood first-order
optimizer — as an explicit, inspectable step over the ``Params`` bundle, rather than pulling an
optimizer from a framework. Keeping the update rule in-project is the same verifiability argument that
made the gradients hand-derived (ADR-0030): the experiment can state, and a test can check, exactly how
parameters move.

Adam carries per-parameter moment state, so it is a small stateful object; a fresh instance starts
clean and the state is not part of the trained artifact (only the final parameters are checkpointed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import typing as npt

from chakaso.training.model import Params

__all__ = ["Adam"]

F64 = npt.NDArray[np.float64]


@dataclass(slots=True)
class Adam:
    """The Adam optimizer (Kingma & Ba), with bias-corrected first and second moments.

    Attributes:
        learning_rate: Step size; must be positive.
        beta1: Exponential decay rate for the first moment.
        beta2: Exponential decay rate for the second moment.
        epsilon: Numerical guard in the denominator.
    """

    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    _step: int = field(default=0, init=False)
    _moment: Params | None = field(default=None, init=False)
    _velocity: Params | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            message = f"learning_rate must be positive, got {self.learning_rate!r}"
            raise ValueError(message)
        for name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if not 0.0 <= value < 1.0:
                message = f"{name} must be in [0, 1), got {value!r}"
                raise ValueError(message)
        if self.epsilon <= 0:
            message = f"epsilon must be positive, got {self.epsilon!r}"
            raise ValueError(message)

    def step(self, params: Params, grads: Params) -> Params:
        """Apply one Adam update and return the new parameters.

        ``params`` and ``grads`` must have matching shapes; the returned bundle is a fresh object and
        the inputs are not mutated, so the update is an explicit, reviewable function of its arguments.
        """
        if self._moment is None or self._velocity is None:
            self._moment = params.zeros_like()
            self._velocity = params.zeros_like()

        self._step += 1
        new_moment_arrays: dict[str, F64] = {}
        new_velocity_arrays: dict[str, F64] = {}
        updated: dict[str, F64] = {}

        for name, array in params.items():
            grad = _named(grads, name)
            moment = _named(self._moment, name)
            velocity = _named(self._velocity, name)

            new_moment = self.beta1 * moment + (1.0 - self.beta1) * grad
            new_velocity = self.beta2 * velocity + (1.0 - self.beta2) * (grad * grad)

            # Bias correction, because the moments are initialized at zero.
            moment_hat = new_moment / (1.0 - self.beta1**self._step)
            velocity_hat = new_velocity / (1.0 - self.beta2**self._step)
            updated[name] = array - self.learning_rate * moment_hat / (
                np.sqrt(velocity_hat) + self.epsilon
            )

            new_moment_arrays[name] = new_moment
            new_velocity_arrays[name] = new_velocity

        self._moment = Params(**new_moment_arrays)
        self._velocity = Params(**new_velocity_arrays)
        return Params(**updated)


def _named(bundle: Params, name: str) -> F64:
    for key, array in bundle.items():
        if key == name:
            return array
    message = f"parameter {name!r} is not present in the bundle"
    raise KeyError(message)
