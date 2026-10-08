"""Running an experiment: execute its development benchmark and record the result.

``run_experiment`` is deliberately thin glue between the experiment definition and the benchmark
that already exists (ADR-0020). It does no measuring of its own — it runs the retrieval or
correction development benchmark, lifts that run's aggregate metrics and dataset identity into an
``ExperimentResult``, and stamps on the implementation version, an evaluator set, and a separate
environment block. The clock is injectable so a test can pin a result's timestamp while still
getting a stable ``result_id`` (which excludes the timestamp by design).

Because every underlying run is deterministic and offline, so is the experiment: the same
experiment on the same code yields the same ``result_id``. No model runs; the numbers are
development measurements over synthetic fixtures, not real-world performance.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from datetime import UTC, datetime

from chakaso import __version__
from chakaso.benchmark import (
    CorrectionBenchmarkRun,
    CorrectionBenchmarkRunner,
    RetrievalBenchmarkRunner,
    development_correction_benchmark,
    load_development_benchmark,
)
from chakaso.benchmark.runner import BenchmarkRun
from chakaso.experiments.model import BenchmarkKind, Experiment, ExperimentResult
from chakaso.retrieval import DEFAULT_TOP_K, RetrievalService


def _default_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def run_experiment(
    experiment: Experiment,
    *,
    now: Callable[[], datetime] | None = None,
    implementation_version: str | None = None,
    environment: dict[str, str] | None = None,
) -> ExperimentResult:
    """Run ``experiment`` against its development benchmark and return the recorded result.

    Args:
        experiment: what to test and under what configuration.
        now: clock for the timestamp; injectable so tests pin it. Excluded from ``result_id``.
        implementation_version: defaults to the installed package version, so a result is traceable
            to the code that produced it.
        environment: host metadata, recorded separately from the measured numbers; a different host
            with the same code and configuration produces the same ``result_id``.

    Raises:
        ExperimentError is not raised here; an unknown benchmark kind is a programming error and
        surfaces as a plain ``ValueError`` from the dispatch.
    """
    if experiment.benchmark_kind is BenchmarkKind.RETRIEVAL:
        retrieval_run = _run_retrieval(experiment)
        return _retrieval_result(
            experiment, retrieval_run, now, implementation_version, environment
        )
    if experiment.benchmark_kind is BenchmarkKind.CORRECTION:
        correction_run = _run_correction()
        return _correction_result(
            experiment, correction_run, now, implementation_version, environment
        )
    message = f"unknown benchmark kind {experiment.benchmark_kind!r}"
    raise ValueError(message)


def _run_retrieval(experiment: Experiment) -> BenchmarkRun:
    top_k = int(experiment.configuration.get("top_k", DEFAULT_TOP_K))
    corpus, dataset = load_development_benchmark()
    runner = RetrievalBenchmarkRunner(RetrievalService(corpus), top_k=top_k)
    return runner.run(dataset)


def _run_correction() -> CorrectionBenchmarkRun:
    return CorrectionBenchmarkRunner().run(development_correction_benchmark())


def _recorded_at(now: Callable[[], datetime] | None) -> datetime:
    return now() if now is not None else datetime.now(UTC)


def _retrieval_result(
    experiment: Experiment,
    run: BenchmarkRun,
    now: Callable[[], datetime] | None,
    implementation_version: str | None,
    environment: dict[str, str] | None,
) -> ExperimentResult:
    metrics = run.metrics
    return ExperimentResult(
        experiment_id=experiment.experiment_id,
        name=experiment.name,
        hypothesis=experiment.hypothesis,
        benchmark_kind=BenchmarkKind.RETRIEVAL,
        configuration=experiment.configuration,
        dataset_id=run.dataset_id,
        dataset_version=run.dataset_version,
        dataset_content_fingerprint=run.dataset_content_fingerprint,
        evaluators=(run.retriever_name,),
        metrics={
            "decision_cases": float(metrics.total_cases),
            "scored_cases": float(metrics.scored_cases),
            "abstention_cases": float(metrics.abstention_cases),
            "recall_at_k": metrics.mean_recall_at_k,
            "precision_at_k": metrics.mean_precision_at_k,
            "mrr": metrics.mrr,
            "hit_rate": metrics.hit_rate,
            "forbidden_hits": float(metrics.forbidden_hits),
            "false_retrievals": float(metrics.false_retrievals),
            "duplicates": float(metrics.duplicates),
        },
        implementation_version=implementation_version or __version__,
        environment=environment if environment is not None else _default_environment(),
        recorded_at=_recorded_at(now),
    )


def _correction_result(
    experiment: Experiment,
    run: CorrectionBenchmarkRun,
    now: Callable[[], datetime] | None,
    implementation_version: str | None,
    environment: dict[str, str] | None,
) -> ExperimentResult:
    metrics = run.metrics
    evaluators = tuple({outcome.evaluator_name for outcome in run.outcomes})
    return ExperimentResult(
        experiment_id=experiment.experiment_id,
        name=experiment.name,
        hypothesis=experiment.hypothesis,
        benchmark_kind=BenchmarkKind.CORRECTION,
        configuration=experiment.configuration,
        dataset_id=run.dataset_id,
        dataset_version=run.dataset_version,
        dataset_content_fingerprint=run.dataset_content_fingerprint,
        evaluators=evaluators,
        metrics={
            "decision_accuracy": metrics.decision_accuracy,
            "unjustified_persistence_rate": metrics.unjustified_persistence_rate,
            "unnecessary_revision_rate": metrics.unnecessary_revision_rate,
            "appropriate_abstention_rate": metrics.appropriate_abstention_rate,
            "appropriate_review_rate": metrics.appropriate_review_rate,
            "total_cases": float(metrics.total_cases),
            "passed_cases": float(metrics.passed_cases),
        },
        implementation_version=implementation_version or __version__,
        environment=environment if environment is not None else _default_environment(),
        recorded_at=_recorded_at(now),
    )
