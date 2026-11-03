# ADR-0050: a run can write usable models at declared points on its own curve

- Status: Accepted (2026-11-03). Implemented in `chakaso.training.nn_training` and
  `chakaso.training.subword_experiment`.
- Date: 2026-11-03
- Related: [K-001](../agent/KNOWN_ISSUES.md),
  [E2610](../../experiments/E2610/record.md),
  [E2622](../../experiments/E2622/record.md),
  [E2623](../../experiments/E2623/record.md),
  [capability benchmarks](../research/chakaso-capability-benchmarks.md),
  [ADR-0035](ADR-0035-transformer-checkpoint-format.md),
  [ADR-0049](ADR-0049-a-runs-optimizer-settings-are-declared-not-inherited.md).

## Context

The project's sharpest open question is not about the model. It is: **can any instrument here detect a real
improvement in what the model has learned?** Two capability probes exist and both were measured against their
own untrained floor — the grammatical minimal-pair probe (E2622) and the held-out continuation probe (E2623).
E2623 came back **near ceiling** (0.958–0.986 across six trained models whose intervals all overlap) and its
own record says it "cannot rank the trained arms"; E2622's addendum measured the probe's seed-to-seed standard
deviation at 0.015–0.048 and withdrew its single-seed ordering. Held-out bits per character, by contrast, is
measured at a seed spread of 0.0023–0.0123 (E2614). So the loss moves finer than the probes can see, and every
statement about what a probe detects was made against *one* endpoint: a fully trained arm versus a random
initialisation.

That comparison is unavailable in the other direction. A run writes exactly one usable artifact, at the end,
after the selected weights are reloaded (`subword_experiment.run_experiment`), and the interim file it writes
during training is a *resume point* carrying optimizer moments — a different artifact kind, overwritten at
each cadence, that `resume_state_from` deliberately refuses to treat as a model. There is therefore no way to
ask whether a probe moves **as the model learns**, which is the only sensitivity test that does not need a new
hypothesis about the data: the training curve is an ordering the harness already measures and already records
at every evaluation.

Without rungs, the question "is this instrument tracking learning, or saturating long before the loss stops
improving?" cannot be asked at all, and every future arm comparison inherits the ceiling problem E2623 found.

## Decision

1. **A plan can name optimizer steps at which it also writes a usable model artifact**:
   `ExperimentPlan.curve_artifact_steps`, forwarded to `TransformerTrainingConfig.curve_artifact_steps`, empty
   by default. Empty is what every committed record ran, and an empty list is elided from the checkpoint
   metadata document (ADR-0049's rule), so a re-run of an existing protocol reproduces its artifact checksum.
2. **A rung must be an evaluation point.** `evaluate_loss` runs at every `eval_every` and at the ceiling; a
   step outside that set has no recorded held-out loss, and a rung that cannot be placed on the curve it came
   from is a file, not a measurement. The configuration refuses a step that is not an evaluation point, one
   beyond `max_steps`, and a repeated step.
3. **The loop hands over a copy, and the caller owns the bytes.** `train_transformer` takes
   `curve_artifact_writer(step, weights, validation_loss)` and calls it with `_snapshot(model)` — the same
   private clone the resume path uses, because the optimizer keeps writing into the live tensors and a rung
   whose bytes quietly became the next rung's is worse than no rung. Naming steps with no writer is refused
   rather than silently skipped: a run that reported a ladder it never wrote is the failure mode this record
   exists to prevent.
4. **A rung is a model, not a resume point.** `rung_artifact` builds a `ModelArtifact` with `kind = "usable"`,
   the step, the architecture, the tokenizer fingerprint, the frequency floor and `rung_validation_loss_nats`,
   and carries **no** optimizer state, so `resume_state_from` refuses it by the rule E2607's budget work
   already established. It is written through `save_artifact`, which verifies by rebuilding the model before
   the bytes land, honours the ADR-0035 refuse-to-clobber guard, and digests the metadata it wrote.
5. **The report lists them.** `transformer.curve_artifacts` holds one entry per rung with its step, file name,
   the validation loss recorded for that step and the artifact checksum, so a later measurement can cite the
   bytes it scored. The key sits outside every block `protocol_digest` hashes, and a test asserts that a
   ladder run and the same run without rungs produce equal digests **and equal selected weights**.

## What this does not decide

Whether any probe tracks the curve. That is the experiment this seam exists for, preregistered separately;
this record decides only that a run can be measured at more than one point on it. It also does not decide how
often a rung is worth writing: the cost is one artifact of the model's own size per rung (17.9 MB at the
reference configuration, measured in E2608's artifact block) and an evaluation-aligned step list.

## Consequences

One training run of an already-measured protocol yields a set of models ordered by a criterion the harness
records for each of them, which is the cheapest available sensitivity test for an instrument and needs no new
data, no new architecture and no new hypothesis about the corpus. The same mechanism serves a future
decoding study of the E2610 kind, which currently can only look at the endpoint. Nothing already on disk
changes: the default writes what it always wrote, at the same path, with the same checksum.
