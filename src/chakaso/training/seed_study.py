"""One protocol, several pre-declared seeds, and an aggregate rebuilt from the files each run wrote.

E2605 measured a Transformer against a character model on a single seed and reported a single number. One
observation carries no spread, so the gap it found — 1.8657 against 3.1343 bits per character — could not be
separated from the luck of one initialization and one batch order. This module runs a declared list of seeds
through the *same* protocol and aggregates what came back, which is the only way the difference becomes an
effect with a size attached to it.

Three rules make the resulting numbers worth reading, and the code enforces each:

* **Seeds are declared before the run.** They arrive as a list the study validates on construction; nothing
  here chooses, adds or drops a seed after seeing a score.
* **Only the seed moves.** The document split is drawn with its own seed, fixed across the study, so a run
  that differs from its neighbours differs in initialization and batch order and in nothing else — the
  held-out documents, the tokenizer and the packing are identical, and the study report proves it by
  comparing every run's recorded fingerprints.
* **The aggregate is rebuilt from disk.** Each run writes its own report; the summary reads those files
  again. A result that only exists in the memory of the process that produced it is not auditable, and a
  missing or failed run is named in the output rather than silently shrinking the sample.

Requires the same optional extras as :mod:`chakaso.training.subword_experiment`.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chakaso.core.atomic import write_text_atomically
from chakaso.core.hashing import sha256_hex
from chakaso.training.errors import StudyProtocolError
from chakaso.training.subword_experiment import ExperimentPlan, run_experiment, write_report

__all__ = [
    "DEFAULT_METRICS",
    "REPRODUCTION_TOLERANCES",
    "RunOutcome",
    "SeedStudy",
    "aggregate_metric",
    "aggregate_study",
    "build_study_report",
    "collect_run_reports",
    "compare_with_reference",
    "protocol_identity",
    "run_seed_study",
    "study_summary",
]

#: Every number the aggregate reports, named once so a record and a reader use the same label for it. The
#: paths address the per-run report produced by :func:`chakaso.training.subword_experiment.run_experiment`.
DEFAULT_METRICS: dict[str, tuple[str, ...]] = {
    "transformer_bits_per_character": ("transformer", "test", "bits_per_character"),
    "transformer_nats_per_token": ("transformer", "test", "loss_nats_per_token"),
    "transformer_perplexity": ("transformer", "test", "perplexity"),
    "transformer_token_accuracy": ("transformer", "test", "token_accuracy"),
    "transformer_best_step": ("transformer", "budget", "best_step"),
    "transformer_wall_clock_s": ("transformer", "budget", "wall_clock_s"),
    "transformer_parameter_count": ("transformer", "parameter_count"),
    "character_bits_per_character": ("character_baseline", "test", "bits_per_character"),
    "character_nats_per_token": ("character_baseline", "test", "loss_nats_per_token"),
    "character_wall_clock_s": ("character_baseline", "budget", "wall_clock_s"),
    "unigram_bits_per_character": ("baselines_subword", "unigram", "bits_per_character"),
    "uniform_bits_per_character": ("baselines_subword", "uniform", "bits_per_character"),
}

#: Fields that must be identical across every run of one study. A difference here is not variation between
#: seeds, it is a different experiment, and the aggregate would be averaging numbers that do not belong
#: together.
_PROTOCOL_FIELDS: dict[str, tuple[str, ...]] = {
    "experiment": ("experiment",),
    "corpus_fingerprint": ("corpus", "fingerprint"),
    "split_fingerprint": ("corpus", "split_fingerprint"),
    "split_seed": ("corpus", "split_seed"),
    "tokenizer_fingerprint": ("tokenizer", "fingerprint"),
    "tokenizer_size": ("tokenizer", "size"),
    "architecture": ("transformer", "architecture"),
    "parameter_count": ("transformer", "parameter_count"),
    "batch_size": ("transformer", "budget", "batch_size"),
    "max_steps": ("transformer", "budget", "max_steps"),
    "eval_every": ("transformer", "budget", "eval_every"),
    "patience": ("transformer", "budget", "patience"),
    "min_delta": ("transformer", "budget", "min_delta"),
    "learning_rate": ("transformer", "budget", "learning_rate"),
    "weight_decay": ("transformer", "budget", "weight_decay"),
    "warmup_steps": ("transformer", "budget", "warmup_steps"),
    "gradient_clip_norm": ("transformer", "budget", "gradient_clip_norm"),
    "character_steps": ("character_baseline", "budget", "max_steps"),
    "test_positions": ("transformer", "test", "positions_scored"),
    "test_characters": ("transformer", "test", "characters_scored"),
}

#: Declared in E2606's preregistration, before any seed of that study ran, and not adjusted afterwards.
#: Half of the last decimal place the record reports: agreement to the reported precision is the claim, and
#: anything larger is a deviation to explain rather than a rounding difference.
REPRODUCTION_TOLERANCES: dict[str, float] = {
    "transformer_bits_per_character": 0.0005,
    "transformer_nats_per_token": 0.0005,
    "transformer_perplexity": 0.005,
    "transformer_token_accuracy": 0.0005,
    "character_bits_per_character": 0.0005,
    "character_nats_per_token": 0.0005,
    "unigram_bits_per_character": 0.0,
    "uniform_bits_per_character": 0.0,
    "transformer_parameter_count": 0.0,
    "transformer_best_step": 0.0,
}


def _dig(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Follow ``path`` through a nested report, refusing a missing key rather than yielding ``None``."""
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            message = f"report has no {'/'.join(path)!r} (stopped at {key!r})"
            raise StudyProtocolError(message)
        current = current[key]
    return current


@dataclass(frozen=True, slots=True)
class SeedStudy:
    """A declared list of seeds over one protocol template.

    Attributes:
        experiment: The experiment identifier these runs belong to; names every artifact and report field.
        plan: The protocol. Its ``seed`` and ``work_dir`` are ignored — each run gets its own — and its
            ``split_seed`` and ``expected_corpus_fingerprint`` are the fixed parts of the contract.
        seeds: The pre-declared seeds, in the order they run. Chosen before any result exists; a study that
            adds a seed after seeing scores is a different study.
    """

    experiment: str
    plan: ExperimentPlan
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.experiment or "/" in self.experiment or "\\" in self.experiment:
            message = f"experiment {self.experiment!r} cannot be used as a directory name"
            raise StudyProtocolError(message)
        if not self.seeds:
            message = "a seed study with no declared seeds measures nothing"
            raise StudyProtocolError(message)
        if len(set(self.seeds)) != len(self.seeds):
            message = f"seeds must be distinct, got {list(self.seeds)}"
            raise StudyProtocolError(message)
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                message = f"seed must be an integer, got {seed!r}"
                raise StudyProtocolError(message)
        if self.plan.split_seed is None:
            message = (
                "a seed study must fix its document split with split_seed; varying the master seed alone "
                "moves the held-out documents as well, and the runs would then differ in their data"
            )
            raise StudyProtocolError(message)
        if self.plan.expected_corpus_fingerprint is None:
            message = (
                "a seed study must name the corpus fingerprint every run trains on, or a run that picked "
                "up different bytes would still be aggregated with the others"
            )
            raise StudyProtocolError(message)

    def expected_identity(self, seed: int) -> dict[str, Any]:
        """The protocol fields this study can state before any run: the plan's own numbers.

        Not every identity field is knowable in advance — the split fingerprint, the tokenizer fingerprint,
        the measured parameter count and the scored positions are outputs — so a reused report is checked
        against what the plan fixes here, and against every sibling run in
        :func:`build_study_report`, which compares the outputs too.
        """
        plan = self.plan
        return {
            "experiment": self.experiment,
            "corpus_fingerprint": plan.expected_corpus_fingerprint,
            "split_seed": plan.effective_split_seed,
            "batch_size": plan.batch_size,
            "max_steps": plan.max_steps,
            "eval_every": plan.eval_every,
            "patience": plan.patience,
            "min_delta": plan.min_delta,
            "learning_rate": plan.learning_rate,
            "weight_decay": plan.weight_decay,
            "warmup_steps": plan.warmup_steps,
            "gradient_clip_norm": plan.gradient_clip_norm,
            "character_steps": plan.character_steps,
            "model_seed": seed,
        }

    def run_dir(self, seed: int) -> Path:
        """This run's own directory, so no seed can overwrite another's artifacts."""
        return self.plan.work_dir / self.experiment / f"seed-{seed}"

    def plan_for(self, seed: int) -> ExperimentPlan:
        """The protocol with one seed applied and its own output directory.

        ``overwrite_artifacts`` is forced on inside a seed's own directory, because the report — not the
        checkpoint — is what makes a run recorded: a directory with no report holds nothing authoritative,
        and a rerun that refuses to write over its own half-finished checkpoint would make the preregistered
        retry policy impossible. A run that *does* have a report is never reached here; it is reused, or the
        operator has declared a rerun with ``overwrite_artifacts``.
        """
        return replace(
            self.plan,
            seed=seed,
            work_dir=self.run_dir(seed),
            experiment_id=self.experiment,
            overwrite_artifacts=True,
        )


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What happened to one declared seed."""

    seed: int
    status: str
    report_path: str
    report_digest: str | None = None
    wall_clock_s: float | None = None
    reused: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.status,
            "report_path": self.report_path,
            "report_digest": self.report_digest,
            "wall_clock_s": self.wall_clock_s,
            "reused": self.reused,
            "error": self.error,
        }


def protocol_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """The fields that define the protocol a run measured under, pulled out of its report."""
    return {name: _dig(report, path) for name, path in _PROTOCOL_FIELDS.items()}


def _require_same_protocol(
    report: Mapping[str, Any], expected: Mapping[str, Any], seed: int
) -> None:
    """Refuse to aggregate a run that did not use this protocol.

    Resume is where this bites: a report left in a run directory by an earlier, different study has a
    plausible shape and would otherwise be counted as one of this study's seeds. ``expected`` carries what
    the plan fixes in advance plus the seed this directory belongs to; the outputs (split, tokenizer and
    parameter fingerprints) are compared across runs by :func:`build_study_report`.
    """
    actual = {
        **protocol_identity(report),
        "model_seed": _dig(report, ("corpus", "model_seed")),
    }
    drifted = sorted(name for name in expected if actual.get(name) != expected[name])
    if drifted:
        message = (
            f"seed {seed}'s existing report does not match this study's protocol in {drifted}; delete the "
            "run directory or point the study somewhere else, because aggregating it would mix protocols"
        )
        raise StudyProtocolError(message)


def run_seed_study(
    study: SeedStudy,
    *,
    verbose: bool = False,
    reuse_completed: bool = True,
    reference: Mapping[str, Any] | None = None,
    reference_seed: int | None = None,
) -> dict[str, Any]:
    """Run every declared seed, then build the aggregate from the reports the runs wrote.

    A seed that raises is recorded as failed and the study continues: losing forty minutes of completed runs
    because the third seed crashed is worse than a report with a named failure in it, and a failure is data.
    Nothing here decides whether a failed run is rerun — that is the preregistered policy's job, and the
    aggregate always shows what is missing.

    Args:
        reference: A recorded report to reproduce against, with ``reference_seed`` naming which run is the
            reproduction. Both are optional; a study with no reference only reports its spread.

    Returns:
        The study report from :func:`build_study_report`, whose per-run numbers were read back off disk.
    """
    outcomes: list[RunOutcome] = []
    if not reuse_completed and not study.plan.overwrite_artifacts:
        recorded = sorted(
            seed for seed in study.seeds if (study.run_dir(seed) / "report.json").is_file()
        )
        if recorded:
            message = (
                f"seeds {recorded} already have a completed report in {str(study.plan.work_dir)!r}; rerunning "
                "them would replace a recorded result, which needs overwrite_artifacts=True (or leave reuse "
                "on, which is the point of writing the reports)"
            )
            raise StudyProtocolError(message)

    for seed in study.seeds:
        directory = study.run_dir(seed)
        report_path = directory / "report.json"
        if reuse_completed and report_path.is_file():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            _require_same_protocol(existing, study.expected_identity(seed), seed)
            outcomes.append(
                RunOutcome(
                    seed=seed,
                    status="completed",
                    report_path=str(report_path.relative_to(study.plan.work_dir)),
                    report_digest=sha256_hex(report_path.read_bytes().decode("utf-8")),
                    reused=True,
                )
            )
            if verbose:
                print(  # noqa: T201 - operator-visible progress
                    f"seed {seed}: reusing the report already on disk",
                    flush=True,
                )
            continue

        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            report = run_experiment(study.plan_for(seed), verbose=verbose)
        except (
            Exception
        ) as exc:  # recorded, never swallowed: the aggregate has to show this seed failed
            outcomes.append(
                RunOutcome(
                    seed=seed,
                    status="failed",
                    report_path=str(report_path.relative_to(study.plan.work_dir)),
                    wall_clock_s=round(time.perf_counter() - started, 2),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        wall = round(time.perf_counter() - started, 2)
        digest = write_report(report, report_path, allow_overwrite=study.plan.overwrite_artifacts)
        outcomes.append(
            RunOutcome(
                seed=seed,
                status="completed",
                report_path=str(report_path.relative_to(study.plan.work_dir)),
                report_digest=digest,
                wall_clock_s=wall,
            )
        )

    return build_study_report(
        experiment=study.experiment,
        work_dir=study.plan.work_dir,
        seeds=study.seeds,
        outcomes=outcomes,
        reference=reference,
        reference_seed=reference_seed,
        study=study,
    )


def collect_run_reports(
    *, experiment: str, work_dir: Path, seeds: Sequence[int]
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Read each declared seed's report from disk; return the ones found and the seeds that have none."""
    found: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for seed in seeds:
        report_path = work_dir / experiment / f"seed-{seed}" / "report.json"
        if report_path.is_file():
            found[seed] = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            missing.append(seed)
    return found, missing


def aggregate_metric(values: Sequence[float]) -> dict[str, Any]:
    """Describe a set of measurements across runs: centre, spread and extent, with nothing invented.

    ``sample_stdev`` is the sample standard deviation of the runs, not a confidence interval — five runs do
    not establish a sampling distribution, and E2606's preregistration says so in the same words.
    """
    count = len(values)
    if count == 0:
        return {"runs": 0, "mean": None, "sample_stdev": None}
    if count == 1:
        return {
            "runs": 1,
            "mean": values[0],
            "sample_stdev": None,
            "median": values[0],
            "min": values[0],
            "max": values[0],
            "range": 0.0,
        }
    return {
        "runs": count,
        "mean": round(statistics.fmean(values), 6),
        "sample_stdev": round(statistics.stdev(values), 6),
        "median": round(statistics.median(values), 6),
        "min": min(values),
        "max": max(values),
        "range": round(max(values) - min(values), 6),
    }


def aggregate_study(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    metrics: Mapping[str, tuple[str, ...]] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Aggregate every declared metric over the runs that produced one, keyed by metric name."""
    aggregate: dict[str, Any] = {}
    for name, path in metrics.items():
        values = [_dig(report, path) for report in reports.values()]
        aggregate[name] = aggregate_metric([float(value) for value in values])
    return aggregate


def protocol_agreement(reports: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Prove from the reports that the runs shared one protocol, and name the seeds that broke it."""
    identities = {seed: protocol_identity(report) for seed, report in reports.items()}
    if not identities:
        return {"runs_compared": 0, "shared": {}, "disagreements": {}}
    first = next(iter(identities.values()))
    disagreements: dict[str, list[int]] = {}
    for name in first:
        offenders = [seed for seed, identity in identities.items() if identity[name] != first[name]]
        if offenders:
            disagreements[name] = sorted(offenders)
    return {
        "runs_compared": len(identities),
        "shared": first,
        "disagreements": disagreements,
    }


def compare_with_reference(
    report: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    tolerances: Mapping[str, float] = REPRODUCTION_TOLERANCES,
) -> dict[str, Any]:
    """Score a fresh run against a recorded report: identical where identity is claimed, else within a
    pre-declared tolerance.

    The two kinds of agreement are kept apart on purpose. Fingerprints, parameter counts and configurations
    must match exactly or the run reproduced a different protocol; loss numbers may differ by the declared
    tolerance because floating-point reduction order across library versions is not a protocol change — and
    a deviation that eats a declared tolerance is reported as a deviation, not rounded away.
    """
    identity_fields = {
        "corpus_fingerprint": ("corpus", "fingerprint"),
        "split_fingerprint": ("corpus", "split_fingerprint"),
        "tokenizer_fingerprint": ("tokenizer", "fingerprint"),
        "architecture": ("transformer", "architecture"),
        "parameter_count": ("transformer", "parameter_count"),
        "best_step": ("transformer", "budget", "best_step"),
        "stopped_early": ("transformer", "budget", "stopped_early"),
    }
    identity: dict[str, Any] = {}
    for name, path in identity_fields.items():
        expected = _dig(reference, path)
        actual = _dig(report, path)
        identity[name] = {"recorded": expected, "reproduced": actual, "match": expected == actual}

    metrics: dict[str, Any] = {}
    for name, path in DEFAULT_METRICS.items():
        if path[-1] in {"best_step", "wall_clock_s"}:
            continue
        expected = _dig(reference, path)
        actual = _dig(report, path)
        tolerance = tolerances.get(name, 0.0)
        difference = abs(float(actual) - float(expected))
        metrics[name] = {
            "recorded": expected,
            "reproduced": actual,
            "difference": round(difference, 6),
            "tolerance": tolerance,
            "within_tolerance": difference <= tolerance,
        }

    deviations = sorted(
        [name for name, result in identity.items() if not result["match"]]
        + [name for name, result in metrics.items() if not result["within_tolerance"]]
    )
    return {
        "identity": identity,
        "metrics": metrics,
        "deviations": deviations,
        "verdict": "reproduced" if not deviations else "deviated",
    }


def build_study_report(
    *,
    experiment: str,
    work_dir: Path,
    seeds: Sequence[int],
    outcomes: Sequence[RunOutcome] | None = None,
    metrics: Mapping[str, tuple[str, ...]] = DEFAULT_METRICS,
    reference: Mapping[str, Any] | None = None,
    reference_seed: int | None = None,
    study: SeedStudy | None = None,
) -> dict[str, Any]:
    """Rebuild the study's aggregate from the reports on disk, without running anything.

    This is the entry point that makes a record auditable after the fact: point it at the same directory and
    the same declared seed list and it recomputes the summary from the files each run left behind.

    Args:
        reference: A recorded report to reproduce against, and ``reference_seed``: which of this study's runs
            is the reproduction of it. Both are optional; a study with no reference only reports its spread.
        study: When given, every report on disk is also checked against what this study declares its runs
            must look like — the budget, the split seed and the corpus included. Cross-run agreement alone
            cannot catch a directory of five runs that all share a *different* protocol from the one being
            collected, which is how a rebuilt aggregate would quietly describe another experiment.

    Raises:
        StudyProtocolError: a run on disk speaks a different protocol than its siblings, or than the declared
            study, or a reference is declared for a seed that has no report.
    """
    reports, missing = collect_run_reports(experiment=experiment, work_dir=work_dir, seeds=seeds)
    if study is not None:
        for seed, report in reports.items():
            _require_same_protocol(report, study.expected_identity(seed), seed)
    agreement = protocol_agreement(reports)
    if agreement["disagreements"]:
        message = (
            f"the runs in {str(work_dir)!r} do not share one protocol; fields that differ: "
            f"{agreement['disagreements']}"
        )
        raise StudyProtocolError(message)

    reproduction = None
    if reference is not None:
        if reference_seed is None or reference_seed not in reports:
            message = (
                "a reproduction comparison needs the reference seed to have a completed report in this "
                f"study; it has {sorted(reports)}"
            )
            raise StudyProtocolError(message)
        reproduction = compare_with_reference(reports[reference_seed], reference)

    per_run = {
        str(seed): {
            "metrics": {name: _dig(report, path) for name, path in metrics.items()},
            "artifact_checksum": _dig(report, ("transformer", "artifact", "checksum")),
            "corpus_origin": _dig(report, ("corpus", "prepared_from")),
            "environment": report["environment"],
        }
        for seed, report in sorted(reports.items())
    }

    return {
        "study": experiment,
        "declared_seeds": list(seeds),
        "completed_seeds": sorted(reports),
        "missing_seeds": sorted(missing),
        "failed_seeds": sorted(
            outcome.seed for outcome in outcomes or () if outcome.status == "failed"
        ),
        "runs": [outcome.as_dict() for outcome in outcomes or ()],
        "protocol": agreement["shared"],
        "aggregate": aggregate_study(reports, metrics=metrics),
        "per_run": per_run,
        "reproduction": reproduction,
    }


def write_study_report(report: Mapping[str, Any], destination: Path) -> str:
    """Write the aggregate deterministically and return the digest a record can cite."""
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    write_text_atomically(payload, destination)
    return sha256_hex(payload)


def study_summary(report: Mapping[str, Any]) -> str:
    """One line naming the sample, its centre and its spread — and the runs that are not there.

    A study with nothing completed says so rather than formatting a ``None``: this line is what an operator
    reads after an hour of training, and it has to survive the failure case.
    """
    transformer = report["aggregate"]["transformer_bits_per_character"]
    declared = len(report["declared_seeds"])
    head = f"{report['study']}: {transformer['runs']}/{declared} seeds"
    if transformer["runs"] == 0:
        line = f"{head}, nothing measured yet"
    else:
        character = report["aggregate"]["character_bits_per_character"]
        spread = (
            f"{transformer['sample_stdev']:.4f}"
            if transformer["sample_stdev"] is not None
            else "single run"
        )
        line = (
            f"{head}, transformer {transformer['mean']:.4f} bits/char (sample stdev {spread}), "
            f"character {character['mean']:.4f}"
        )
    if report["missing_seeds"]:
        line += f", missing {report['missing_seeds']}"
    if report["failed_seeds"]:
        line += f", failed {report['failed_seeds']}"
    reproduction = report.get("reproduction")
    if reproduction:
        line += f", reproduction {reproduction['verdict']}"
    return line
