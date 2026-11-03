# ADR-0049: a run's optimizer settings are declared by this project, and the non-matrix parameters can be exempted from decay

- Status: Accepted (2026-11-03). Implemented in `chakaso.training.nn_training`; the decay exemption is an
  option, not a new default, and the arm that would decide it is preregistered separately.
- Date: 2026-11-03
- Related: [K-030 item 3](../agent/KNOWN_ISSUES.md),
  [ADR-0034](ADR-0034-optional-pytorch-model-extra.md),
  [ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md),
  [ADR-0046](ADR-0046-checkpoint-records-the-tokenizer-frequency-floor-as-an-optional-field.md).

## Context

Every measured run in this repository trained with `torch.optim.AdamW(parameters, lr=…, weight_decay=…)`. Its
betas, its epsilon and its `amsgrad` were never stated: the report recorded them, but read them *off the
library's defaults* at the moment the report was written (the `optimizer` block added on 2026-11-01). So a
re-run under a torch that had changed a default would train a different recipe and record it honestly — the
record would say what happened, but the protocol a previous record named would no longer mean the same thing,
and two runs on different library versions would not be comparable as one protocol. K-030 item 3 filed exactly
this: "declared nowhere and recorded in no report" for the first half, and "whether norms *should* be excluded
from decay remains an unrun arm" for the second.

The second half is a real fork in the road, not a style question. In this architecture the embedding table and
the tied output head are 1,048,576 of 3,413,760 parameters (30.7 %, measured in E2608's arithmetic check), and
the RMSNorm scales and the biases are the only one-dimensional parameters. The published recipes that use this
optimiser at small scale exempt the one-dimensional ones from decay; Chakaso has never tested whether that
matters at 3.4M parameters and a truncated budget.

The constraint is the one ADR-0036 and ADR-0046 already imposed: no committed number, fingerprint, checksum or
protocol digest may move.

## Decision

1. **The optimizer's betas, epsilon and `amsgrad` are named constants of this project** — `ADAM_BETAS`,
   `ADAM_EPSILON`, `ADAM_AMSGRAD` — and `build_optimizer` passes them explicitly. Their values are the ones
   `torch.optim.AdamW` used for every committed record (0.9, 0.999, 1e-8, off), so no run changes. Two tests
   hold the two halves of the claim: one compares the constants against the installed library's own defaults,
   so a future change fails a named test instead of moving a loss; the other perturbs the constants and asserts
   the built optimizer follows them, which is what proves they are the source rather than a comment.
2. **Weight decay can be exempted from the one-dimensional parameters, off by default.**
   `weight_decay_excludes_vectors` splits the parameters by **shape** — `ndim >= 2` decays, `ndim < 2` does
   not — because the two families name their modules differently and a name list would stop matching silently
   the day a module is renamed. The default is `False`, which is what every committed record ran; one group
   with the plan's decay is built, which is what `AdamW(parameters, weight_decay=…)` builds today.
3. **The report states which of the two it ran**, in the same `decay_applied_to` string it already carried.
   No key was added to `optimizer_settings`: `protocol_digest` hashes the budget block, so a new key would
   make a faithful re-run of E2612 or E2614 stop aggregating with the study it belongs to. A test pins the key
   set for that reason.
4. **A new configuration field appears in the checkpoint metadata only when it carries information.**
   `TransformerTrainingConfig.as_dict()` elides `weight_decay_excludes_vectors` and (see
   [ADR-0050](ADR-0050-a-run-can-write-usable-models-at-declared-points-on-its-own-curve.md))
   `curve_artifact_steps` at their default, because that dictionary is inside the metadata document a
   checksum covers. This is ADR-0046's route, and it keeps a re-run of a committed protocol reproducing its
   artifact checksum. An artifact whose metadata lacks the key reads as the default, not as unknown, and
   `verify_resume_compatibility` therefore still refuses a resume that would change the recipe.

## What this does not decide

Whether exempting the one-dimensional parameters helps. That is an arm against an existing control at the
reference configuration, preregistered as its own record, judged by E2614's measured margins. Until it is run,
the option is a route, not a recommendation — the same standing ADR-0040 gave document-boundary masking.

## Consequences

Nothing on disk changes: no committed fingerprint, checksum, digest or number moves, and the dependency-free
and `[train, nn]` matrices pass with the change. A measured recipe can now be re-run on a library whose
defaults have moved and train the same optimisation. One more arm becomes statable in a protocol file.
