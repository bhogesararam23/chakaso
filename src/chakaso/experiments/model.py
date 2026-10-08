"""The experiment and its result: reproducible research artifacts.

Chakaso is becoming a research system, so a run has to be a *reproducible artifact*, not a
screenshot of a console. An ``Experiment`` names what is being tested — a hypothesis, which
development benchmark it runs against, and the configuration under which it runs. An
``ExperimentResult`` records what that run produced: the dataset identity and version, the metrics,
the evaluator(s) that decided, the implementation version, and a separate block of environment
metadata.

The reproducibility rule (ADR-0020) is why the shape is what it is. ``result_id`` is a fingerprint
over everything that determines the *number* — experiment, benchmark identity and version,
configuration, evaluator, implementation version, and the metrics themselves — and deliberately
excludes the timestamp and the machine the run happened on. Two runs of the same experiment on the
same code therefore share a ``result_id`` even if seconds apart on different hosts; a change to any
input that should matter changes it. There is no ML platform here: no database, no experiment
tracker, just immutable, serializable records a person can commit.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from chakaso.benchmark.identity import content_version
from chakaso.core.errors import ChakasoError

__all__ = [
    "BenchmarkKind",
    "Experiment",
    "ExperimentError",
    "ExperimentId",
    "ExperimentResult",
    "InvalidExperimentError",
]

_EXPERIMENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")

_EMPTY_CONFIGURATION: Mapping[str, str] = MappingProxyType({})


class ExperimentError(ChakasoError, ValueError):
    """Base class for failures in the experiment layer."""


class InvalidExperimentError(ExperimentError):
    """An experiment or result is malformed — a blank name, an unaddressable id, a naive time."""


@dataclass(frozen=True, slots=True, order=True)
class ExperimentId:
    """A stable, readable identifier for one experiment definition."""

    value: str

    def __post_init__(self) -> None:
        if _EXPERIMENT_ID_PATTERN.fullmatch(self.value) is None:
            message = (
                f"experiment identifier {self.value!r} is not a valid slug: lowercase, starting "
                "with a letter or digit, then letters, digits, '.', '_' or '-', max 128."
            )
            raise InvalidExperimentError(message)

    def __str__(self) -> str:
        return self.value


class BenchmarkKind(StrEnum):
    """Which development benchmark an experiment runs against."""

    RETRIEVAL = "retrieval"
    CORRECTION = "correction"


@dataclass(frozen=True, slots=True)
class Experiment:
    """What is being tested: a hypothesis, a benchmark, and the configuration under test."""

    experiment_id: ExperimentId
    name: str
    hypothesis: str
    benchmark_kind: BenchmarkKind
    configuration: Mapping[str, str] = field(default_factory=lambda: _EMPTY_CONFIGURATION)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            message = f"experiment {self.experiment_id} must have a name"
            raise InvalidExperimentError(message)
        if not self.hypothesis.strip():
            message = (
                f"experiment {self.experiment_id} must state a hypothesis; an experiment that "
                "tests nothing is not an experiment"
            )
            raise InvalidExperimentError(message)
        object.__setattr__(self, "configuration", MappingProxyType(dict(self.configuration)))

    def to_material(self) -> dict[str, object]:
        """The deterministic view of the experiment definition."""
        return {
            "experiment_id": str(self.experiment_id),
            "name": self.name,
            "hypothesis": self.hypothesis,
            "benchmark_kind": self.benchmark_kind.value,
            "configuration": dict(sorted(self.configuration.items())),
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """The recorded outcome of running one experiment once."""

    experiment_id: ExperimentId
    name: str
    hypothesis: str
    benchmark_kind: BenchmarkKind
    configuration: Mapping[str, str]
    dataset_id: str
    dataset_version: str
    dataset_content_fingerprint: str
    evaluators: tuple[str, ...]
    metrics: Mapping[str, float | None]
    implementation_version: str
    environment: Mapping[str, str]
    recorded_at: datetime

    def __post_init__(self) -> None:
        if (
            self.recorded_at.tzinfo is None
            or self.recorded_at.tzinfo.utcoffset(self.recorded_at) is None
        ):
            message = f"result for experiment {self.experiment_id} needs a timezone-aware timestamp"
            raise InvalidExperimentError(message)
        object.__setattr__(self, "configuration", MappingProxyType(dict(self.configuration)))
        object.__setattr__(self, "evaluators", tuple(sorted(set(self.evaluators))))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))

    @property
    def result_id(self) -> str:
        """A fingerprint of everything that determines the number — excluding when/where it ran.

        Two runs of the same experiment on the same code share this id regardless of timestamp or
        host; any change to an input that ought to matter (configuration, benchmark version,
        evaluator, implementation, the metrics) moves it (ADR-0020).
        """
        return content_version(self._reproducibility_material())

    def _reproducibility_material(self) -> dict[str, object]:
        return {
            "experiment_id": str(self.experiment_id),
            "name": self.name,
            "hypothesis": self.hypothesis,
            "benchmark_kind": self.benchmark_kind.value,
            "configuration": dict(sorted(self.configuration.items())),
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_content_fingerprint": self.dataset_content_fingerprint,
            "evaluators": list(self.evaluators),
            "metrics": _normalise_metrics(self.metrics),
            "implementation_version": self.implementation_version,
        }

    def to_dict(self) -> dict[str, object]:
        """A full, deterministic, JSON-serializable view, environment kept separate."""
        return {
            "result_id": self.result_id,
            "experiment": {
                "experiment_id": str(self.experiment_id),
                "name": self.name,
                "hypothesis": self.hypothesis,
                "benchmark_kind": self.benchmark_kind.value,
                "configuration": dict(sorted(self.configuration.items())),
            },
            "benchmark": {
                "dataset_id": self.dataset_id,
                "dataset_version": self.dataset_version,
                "dataset_content_fingerprint": self.dataset_content_fingerprint,
            },
            "evaluators": list(self.evaluators),
            "metrics": _normalise_metrics(self.metrics),
            "implementation_version": self.implementation_version,
            "environment": dict(sorted(self.environment.items())),
            "recorded_at": self.recorded_at.isoformat(),
        }


def _normalise_metrics(metrics: Mapping[str, float | None]) -> dict[str, float | None]:
    """Sort metric keys and coerce integral floats to a stable representation for fingerprinting.

    ``1`` and ``1.0`` must fingerprint the same whether a count or a rate produced them, so an
    integral float is stored as a float consistently. Non-integral values keep full precision;
    ``None`` (a rate with no applicable cases) is preserved rather than turned into a number.
    """
    normalised: dict[str, float | None] = {}
    for key in sorted(metrics):
        value = metrics[key]
        normalised[key] = None if value is None else float(value)
    return normalised
