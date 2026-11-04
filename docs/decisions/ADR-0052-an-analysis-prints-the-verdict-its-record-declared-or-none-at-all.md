# ADR-0052: an analysis prints the verdict its record declared, or none at all

- Status: Accepted (2026-11-04). Implemented in `chakaso.training.outcome_rules`,
  `chakaso.training.probe_sensitivity`, `scripts/analyse_arm_pairs.py` and
  `scripts/analyse_probe_sensitivity.py`; the declarations are committed under `configs/analyses/`.
- Date: 2026-11-04
- Related: [ADR-0049](ADR-0049-a-runs-optimizer-settings-are-declared-not-inherited.md) (a recipe a run inherits is a
  recipe no re-run controls — the same argument one layer down from this one),
  [ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md) (a measured result stays re-trainable),
  [K-034](../agent/KNOWN_ISSUES.md), [E2626](../../experiments/E2626/record.md),
  [E2629](../../experiments/E2629/record.md), [E2634](../../experiments/E2634/record.md),
  [the record format](../internal/experiments/README.md).

## Context

Two shared tools embedded other records' verdicts:

* `chakaso.training.probe_sensitivity` decided every instrument's verdict as `S-a`…`S-d` against the constants
  `MATERIAL_COVERAGE = 0.9` and `TRACKING_THRESHOLD = 0.9`, and `scripts/analyse_probe_sensitivity.py` stamped
  `materiality_declared_in = "experiments/E2626/record.md"` into every result it wrote — the 0.9 bounds and that
  attribution were E2626's declaration, made in that record before its run, and no other record signed them.
* `scripts/analyse_arm_pairs.py` decided the family outcome as `Z-a`…`Z-d`. **E2634 preregistered `G-a…G-d` and was
  reported a `Z-a`** — its own deviation 1: *"the analysis script prints its own outcome letters, and they are not this
  record's … The tool's rules were written for another record and were not re-declared here; where they differ, the
  record governs."*

The numbers were never wrong. That is what made it survivable: a verdict is not a number, it is the statement that a
declared condition held, and the condition was declared in a file the tool had never read. A second record with
different rules gets reported under the first record's outcome space, and the difference has to be explained in prose
by whoever notices. Nothing in the artifact says which contract produced a letter.

## Decision

**An analysis computes facts. A record declares what they mean. A tool prints no letter it was not handed.**

1. `chakaso.training.outcome_rules` is the mechanism: an outcome vocabulary is a named list of letters, each with a
   condition over the analysis's facts and the sentence that states it, in a document of schema
   `chakaso-outcome-vocabulary/1`. The condition language is flags, numeric bounds and `all_of` / `any_of` nesting,
   plus an explicit `{"always": true}` for a catch-all — a rule with no condition must say it is one.
2. `compare_instruments` and the arm-pair analysis now return their **facts** — for instruments: `rho`,
   `material_pairs`, `material_pairs_separated`, `material_coverage`, `has_material_pairs`, `contradicted_pairs`
   (:data:`~chakaso.training.probe_sensitivity.VERDICT_FACT_NAMES`); for the family: the eight names in
   `FAMILY_FACT_NAMES` — and take an optional vocabulary. Without one every verdict slot is `null` and the statistics
   are complete.
3. Both scripts **refuse** rather than print an inherited letter: `--outcome-vocabulary` (and
   `--instrument-vocabulary` for the arm-pair run) or `--statistics-only`, chosen explicitly. The refusal happens
   before any file is read.
4. Three declarations are committed so the records that ran without the mechanism still reproduce:
   [`arm-pairs-e2629-v1.json`](../../configs/analyses/arm-pairs-e2629-v1.json),
   [`arm-pairs-e2634-v1.json`](../../configs/analyses/arm-pairs-e2634-v1.json) and
   [`probe-sensitivity-e2626-v1.json`](../../configs/analyses/probe-sensitivity-e2626-v1.json), the last carrying the
   0.9 bounds that used to be constants.

**Letters are read in declaration order, and every letter that also matched is reported.** This is not a loophole:
E2629's `Z-a` ("every material pair ranked") *implies* its `Z-b` ("at least one ranked"), so the two were always
separated by precedence rather than by disjoint conditions, and a mechanism that refused every overlap could not
reproduce the record that started the series. A record that wants disjointness says `"exclusive": true` and the tool
then refuses two matches instead of choosing. Either way the artifact carries `precedence` and `also_matched`, so an
overlap is visible to a reader instead of decided in a code path nobody reads.

## Consequences

**What gets better.** A new record invents its own letters by writing a file:
`test_a_record_that_declared_its_own_letters_gets_them_from_the_same_facts` runs the same analysis over the same facts
under two declarations and reports `Z-a` and `G-a`. A mis-typed condition is a load-time refusal naming the fact and
listing the ones that exist (`test_a_fact_the_analysis_does_not_compute_is_refused_when_the_vocabulary_is_loaded`), so
a rule can no longer quietly never fire. Facts that no rule covers are a refusal that prints the facts, which turns
"the declaration did not anticipate this" into a finding instead of a silent else-branch.

**What costs something.** Every reproduction command written before this change needs a vocabulary flag. E2626's,
E2629's and E2634's records carry commands that now stop at the refusal, so each has a dated addendum naming the flag
and the file that reproduces its published letters. Nothing committed is rewritten: E2634's
[`results.json`](../../experiments/E2634/results.json) still holds the `Z-a` the tool gave it, and
[`results-under-declared-vocabulary.json`](../../experiments/E2634/results-under-declared-vocabulary.json) is the same
analysis re-run under the record's own declaration — its `pairs` array is byte-identical and the outcome letter is
`G-a`, which is the evidence that the mechanism reproduces the arithmetic and moves only the contract.

**What is explicitly not claimed.** A vocabulary makes a verdict *traceable*, not *right*. E2634's `G-a` fires on a
reading whose untrained floor is 0.119 below chance, and no declaration changes that; the letters describe what was
tested, never whether the model is useful. K-001 is untouched.

## Alternatives rejected

- **Keep the constants as defaults, overridable.** Rejected for ADR-0049's reason: an inherited default is the one a
  re-run does not control, and here it is worse — the default was another record's outcome space, so an override that
  was never typed silently mislabels a result. `load_plan`'s pre-tokenizer gate and `LEGACY_DEFAULT_PRE_TOKENIZER` are
  the same argument about a different field.
- **First-match-wins without reporting the overlap.** Rejected because it is how the bug survived: the tool picked and
  nobody saw that two rules held.
- **Require every vocabulary to be disjoint.** Rejected on the evidence above — it would make E2629's own declaration
  unrepresentable, and a mechanism that cannot express the record that started the series gets worked around.
- **Embed the vocabulary in the record markdown and parse it.** Rejected: the record is prose a human edits, and a
  verdict rule has to be machine-checked against the fact list. It lives in `configs/analyses/`, and the record cites
  it the way it cites a protocol file.

## Verification

Executed 2026-11-04. `tests/test_training_outcome_rules.py` (25 tests, hand-derived letters and the three refusals),
plus the two script suites: 16 and 19 tests, including the refusal-before-reading and the two-declarations-one-facts
case. The committed declarations were read against the real artifacts — E2634's analysis re-run over its three probe
files and the `E2629` arm reports reproduces `results.json`'s `pairs` byte for byte, every instrument verdict
identically, with the family letter now `G-a`. Dependency-free 2,196 pass / 34 skip; `ruff check` and `format` clean;
`mypy src` clean on 125 files and the model-path configuration clean on 174.
