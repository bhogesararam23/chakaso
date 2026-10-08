# ADR-0020: Experiments are content-fingerprinted, environment-separated research artifacts

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0013](ADR-0013-benchmark-identity-and-versioning.md), [ADR-0019](ADR-0019-semantic-grounding-boundary.md)

## Context

The system now has development benchmarks (retrieval, correction) and a correction decision rule.
For this to be a research project rather than a codebase with tests, a run has to be a
*reproducible artifact*: something that records what was tested, against which version of which
benchmark, under what configuration, and what number came out — in a form another person can check
and a later change can be compared against. `docs/evaluation.md` already forbids reporting a result
without its configuration, versions and seed.

The temptation is to reach for an experiment tracker (MLflow, Weights & Biases) or a results
database. Both are wrong here for the same reason a dense index or a model is wrong now: they add a
dependency and a schema before the work has shown it needs one, and an external tracker is a
networked, non-local service (ADR-0001).

## Problem

How should experiments and their results be modelled so they are reproducible, comparable, and
committable as research history, without introducing a database, a tracker, or any dependency that
breaks the local-first, offline, deterministic posture?

## Options considered

- **An external experiment tracker.** Rejected: networked, not local-first, heavyweight, and it
  would own the reproducibility story the project needs to define for itself.
- **Write result files into the repo from the code.** Rejected for now: the runtime does not perform
  filesystem writes (nothing here persists answers either, ADR-0018); a result is produced,
  serialized, and committed by a person, which keeps the "no persistence / no side effects by
  default" property intact.
- **Immutable, serializable, content-fingerprinted records produced by running a benchmark.** Chosen.

## Decision

`chakaso.experiments` defines:

- `Experiment` — an identifier, a name, a **hypothesis**, the development benchmark it runs against,
  and a configuration mapping. An experiment with no hypothesis is refused: a run that tests nothing
  is not an experiment.
- `ExperimentResult` — the outcome: the benchmark's dataset identity, version and content
  fingerprint, the evaluator(s) that decided, the metrics, the implementation version, a separate
  `environment` block, and a timezone-aware timestamp. It exposes `to_dict()` (deterministic JSON)
  and a `result_id`.

**Reproducibility rule.** `result_id` is a fingerprint over everything that determines the *number*
— experiment, benchmark identity/version/fingerprint, configuration, evaluator, implementation
version, and the metrics — and explicitly **excludes** the timestamp and the environment. Two runs
of the same experiment on the same code therefore share a `result_id` even seconds apart on
different hosts; a change to any input that ought to matter moves it. Environment metadata (Python,
platform) is recorded but kept in a separate block so it can never silently alter an evaluation
semantics — reproducibility is a property of the measured part, and the machine is provenance, not
part of the number.

A `compare_results` baseline operation compares two results only when they address the same
experiment, benchmark identity and version, configuration and evaluator set; it reports per-metric
deltas with a declared better/worse direction, and refuses incompatible pairs rather than comparing
differently-defined metrics silently. There is no statistical-significance machinery — a
deterministic delta over a small synthetic benchmark is not the place for a confidence interval.

## Reasoning

- **The number and the machine are different facts.** Folding the timestamp or hostname into the
  result identity would make every run "different" and destroy the ability to detect that two runs
  actually reproduced each other. Separating them is the difference between a reproducible artifact
  and a log line.
- **Compare only like with like.** A metric is only a measurement of the thing that defined it; a
  fingerprint or evaluator difference means the two numbers answer different questions, so the
  comparison must fail loudly, not interpolate.
- **No new dependency earns its keep yet.** The benchmarks are deterministic and small; a
  content-fingerprint over a serialized result gives reproducibility without a database, exactly as
  the retrieval benchmark's pinned fingerprint guards its corpus (ADR-0013).

## Trade-offs

- Results are not persisted by the runtime; committing them is a human/repo action. That keeps the
  no-side-effects default and is honest about there being no persistence layer yet (ADR-0018,
  K-007).
- `result_id` fingerprints the *computed* metrics, so a genuine regression changes it — it identifies
  a result, it does not hide a bad one. That is intended.

## Consequences

- Experiments can be recorded and compared reproducibly; a baseline comparison is a first, honest
  regression guard (`chakaso.experiments.compare`).
- Every result traces to a benchmark version and evaluator, so a future change to a fixture or the
  decision rule shows up as a moved `result_id` and a comparison, not a silent drift.
- A trained model would run against the *same* experiment/benchmark/result machinery: generating
  answers is a new input to an experiment, not a reason to redesign this layer.
- Nothing here claims a measured model-quality result; all numbers are development measurements over
  synthetic fixtures.
