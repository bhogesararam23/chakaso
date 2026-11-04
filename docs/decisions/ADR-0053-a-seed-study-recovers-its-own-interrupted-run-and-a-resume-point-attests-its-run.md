# ADR-0053: a seed study recovers its own interrupted run, and a resume point attests the run it came from

- Status: Accepted (2026-11-05; implemented in the same change)
- Date: 2026-11-05
- Related: [ADR-0035](ADR-0035-transformer-checkpoint-format.md) (the checkpoint format),
  [ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md) (a measured record stays re-trainable),
  [ADR-0046](ADR-0046-checkpoint-records-the-tokenizer-frequency-floor-as-an-optional-field.md) (a format field that
  appears only when it carries information), [ADR-0048](ADR-0048-the-pre-tokenizer-is-a-named-part-of-a-vocabulary-s-identity.md),
  [ADR-0033](ADR-0033-model-centred-product-contract.md) (the model is the unit of work),
  K-036, K-037, K-038.

## Context

`train_transformer` already writes a *resumable* artifact — the current parameters, the Adam moments and the step —
into the run's own directory at every checkpoint interval, and `ExperimentPlan` already carries a `resume_from` field
that `run_experiment` verifies through `verify_resume_compatibility` before its first update. The single-run path
works, and [E2608's record](../../experiments/E2608/record.md) names the checks it makes.

A seed study did not use any of it. `run_seed_study` treats `report.json` as the only evidence a seed produced —
correctly, since the report is what makes a run recorded — and when the report is absent it calls
`run_experiment(study.plan_for(seed))` with `resume_from` unset. So a seed interrupted at step *n* re-trained from step
0 while the file that would have continued it sat in that seed's own directory. Measured on 2026-11-05 against the
synthetic study fixture: the recovery pass reported `resumed_from_step: None` over a curve of `[0, 4, 8, 12]`, twelve
steps it had just thrown away. This is K-036, and it is not a hypothetical cost — [E2636](../../experiments/E2636/record.md)
is preregistered at three times the step ceiling any previous run of this project reached.

Wiring discovery in the obvious way exposed a second defect. `report["plan"]` holds every scalar field of the plan, and
`protocol_digest` removes `_MACHINE_LOCAL_PLAN_FIELDS` from it before hashing. `resume_from` was not in that set, so an
absolute path into one operator's working directory was hashed as a *condition the numbers were measured under*. A
resumed run is defined as the uninterrupted one — same seed, same batch plan, same ceiling — so a recovered seed hashed
differently from its siblings, `protocol_agreement` reported `{"protocol_digest": [the recovered seed]}`, and
`build_study_report` refused to aggregate the runs the study had just paid for. This is K-037.

Both defects were latent rather than visible because K-037 makes K-036's fix inert: no study had ever resumed, so no
study had ever hashed a resume path.

## The constraint that makes this a decision

The checksum a record cites is taken over the checkpoint's metadata document, which includes `run_metadata`. Writing a
new key there at its default re-digests every artifact the code before it produced, and a record names those digests —
[E2606](../../experiments/E2606/record.md) records exactly that event happening when the environment block gained two
keys, and calls it "a format-of-the-record change rather than a change in the trained object", filed as a deviation
rather than smoothed over. ADR-0046's route is the one this project takes: **a field appears when it carries
information, and its absence reads as a stated absence.**

`protocol_digest` has the same constraint in a sharper form, because four committed `results.json` files state a
protocol digest as a literal. A change to what the digest covers is only compatible if no committed report changes
value.

## Decision

1. **A seed study discovers its own resume point, and discovery addresses a path rather than searching for one.**
   `SeedStudy.discover_resume_point(seed)` returns
   `run_dir(seed) / resumable_checkpoint_filename(experiment, model_family)` when that file exists and names a step
   strictly between 0 and the plan's ceiling, and `None` otherwise. The name is the same helper the writer uses, so the
   two cannot drift apart. No glob, no "most recent `.pt`", no reading a sibling seed's moments.
2. **Discovery is not a second validator.** It reads the artifact only for its step and hands the path to
   `run_experiment`, which loads and verifies it through the existing `verify_resume_compatibility`. A file that will
   not load raises `CorruptedCheckpointError` and is recorded as that seed's failure — the study continues, and nothing
   retrains over bytes nobody can explain.
3. **A checkpoint at or beyond the plan's ceiling is not a run left to continue.** There is no step to resume onto;
   `train_transformer` refuses a `start_step` at the ceiling, and it is right to. The seed starts clean, which
   reproduces the curve it replaces rather than a different one, because the seed, the split and the batch plan are all
   fixed.
4. **A resume point attests the run that wrote it.** `train_transformer` accepts `experiment` and
   `split_fingerprint` and records each in the resumable artifact only when its caller named it.
   `verify_resume_compatibility` accepts the same two, and refuses a checkpoint that attests a value that disagrees or
   attests neither. The experiment field closes the case where two runs agree on corpus, architecture, tokenizer and
   every budget number and differ only in which record cites them; the split field closes the case where two runs train
   on the same bytes and hold out different documents, so a continued curve would be selected on a validation set it
   never saw.
5. **`resume_from` is machine-local.** It joins the plan fields `protocol_digest` ignores, on the same footing as the
   `resumed_from_step` that was already excluded as an observed outcome of one execution.
6. **A declared rerun does not resume.** `run_seed_study(..., reuse_completed=False)` — which the guard at the top of
   the pass already makes cost `overwrite_artifacts=True` — is the operator saying "train these seeds again", so
   discovery is switched off for that pass. Silent continuation of a run the operator asked to repeat is a different
   run from the one they asked for, and without this rule the only way to retrain a recovered seed would be to delete
   its checkpoint by hand.

## Consequences

- An interrupted multi-seed study costs the remaining steps of the interrupted seeds and nothing more, and a seed that
  completed still reuses its report.
- A resume point written before this change records neither identity field, so **automatic recovery refuses it**. This
  is a deliberate cut: no committed record depends on it — [E2608](../../experiments/E2608/record.md) states that
  neither arm reported `resumed_from_step`, so the path was covered only by unit tests — and every artifact the new
  code writes attests itself. An operator who knows where an unattested file came from keeps the explicit
  `--resume-from` path, which passes no identity requirement and records that it was not checked.
- No committed protocol digest moved, and the reason is checkable rather than argued: the report's plan block keeps
  only `int`, `float`, `str` and `Path` values, so a run that did not resume never wrote the key at all. Every per-run
  report under `experiments/` was read to confirm none carries it.
- Artifact checksums for runs made from now on do move, because the resumable artifact now names its experiment and
  split. No record's cited checksum changes: those describe files already written, and this milestone retrains
  nothing.
- The report still states `resume_from` and `resumed_from_step`. Excluding a field from the digest is not removing it
  from the record.
- K-038 stays open: `budget_extended_from` is hashed as a condition, so a study in which only *some* seeds continued
  from checkpoints written under an older, smaller ceiling cannot aggregate. Closing it needs the digest re-baselined,
  which is a decision record of its own.

## Alternatives rejected

- **Glob for the newest resumable artifact in the seed directory.** It would pick up a file another experiment was
  copied in, and "newest" makes recovery depend on mtime — a fact a record cannot cite.
- **Re-validate in `SeedStudy`.** Two validators of one checkpoint is two that can disagree, and the weaker one wins in
  production.
- **Store run identity in a new top-level block of the checkpoint file.** `load_artifact` refuses unexpected blocks by
  design, so this is a schema change rather than an additive one, and it buys nothing over `run_metadata`.
- **Fall back to a clean start when a discovered checkpoint fails validation.** That is the silently-resumed
  incompatible run this milestone exists to prevent, expressed as a silent *discarded* one.
- **Let a checkpoint that attests nothing resume anyway, treating the absence as compatible.** An unattributable file
  is precisely the case recovery cannot judge; ADR-0046's "absent means the default" works there because
  `min_frequency`'s default is a fact about the reader, whereas the absence of an experiment name is not a fact about
  which run wrote the file.

## Verification

`tests/test_training_seed_study.py` runs the real pipeline on the synthetic fixture: a seed interrupted mid-curve is
continued and its curve opens at the step it stopped at; a recovered seed and a sibling that never stopped aggregate; a
seed with nothing on disk starts clean; a checkpoint at the ceiling is not resumed; a completed seed's report beats its
own checkpoint; an unreadable checkpoint fails one seed and names itself; a foreign experiment's file is invisible to
discovery. `tests/test_training_nn_engine.py` covers the four identity refusals and the field-elision rule.
`tests/test_training_subword_experiment.py` requires a resumed run's `protocol_digest` to equal the uninterrupted run
it reproduces. All of it measured locally on Windows 11 with the `[train,nn]` extras on 2026-11-05; the Linux runner had
not been read for these tests at the time this record was written.
