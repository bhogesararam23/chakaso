# Research campaign report — 2026-11-03

**Status:** Research. A factual close of the work done on 2026-11-03 in the session that began at `c498eb8`.
Every number below is either read from a committed experiment record in this repository or is a command output
quoted with the command that produced it. Where a count is given it was taken from `git log` at the time of writing.
Nothing here is a capability claim, and [K-001](../../agent/KNOWN_ISSUES.md) — no trained model produces usable text —
is unchanged by every experiment in this report.

---

## What this phase set out to do, and what it delivered

The campaign's open problem was stated by [E2626](../../../experiments/E2626/record.md)'s own limitation: the project has
two capability probes, neither is known to detect a real improvement in what a model has learned, and every sentence of
the form "this arm is better" rests on held-out loss. Three preregistered experiments closed that question, and the
answer is a negative the project has to live with.

| Unit | What it asked | What came back |
| --- | --- | --- |
| [E2626](../../../experiments/E2626/record.md) | Can either probe detect an improvement this project would call real, measured against one training curve? | **No S-a.** The minimal-pair probe tracks held-out loss (ρ 0.951) and separates 63 of 74 material rung pairs down to a 0.0218-bit gap, but contradicts the loss on one pair (rungs 1600 vs 1800, 24:2 discordant, p = 1.0 × 10⁻⁵) → S-c. The continuation probe tracks (ρ 0.918), separates 23, resolves nothing below 0.294 and nothing at all past update 800 → S-b. |
| [E2627](../../../experiments/E2627/record.md) | Is the continuation probe's ceiling the distractor's topic cue? Rebuild the distractor from the anchor document and find out. | **No.** 215 same-region items; three of four kinds sit at chance untrained (0.4638 / 0.4684 / 0.4678) and `permuted` fails its floor at 0.4425; only `prior_span` has headroom (0.9395) and earns S-b; **no kind separates any of the ten pairs among five independently trained seeds** (smallest p 0.065); E2623's v1 number reproduced to the last digit, 0.986046511627907. T-a on one kind of four. |
| [E2628](../../../experiments/E2628/record.md) | Does the space-prefixed tokenizer's gain survive a second, independent initialisation, on the host that has not carried it? | **Yes, and it grew.** O1 +0.029403831 and whole pack +0.034949980 against the control's own 2,400 updates and 2,756 documents, all five partitions improving, 2.93 characters per identifier against 2.10. Y-a. No default changed. |
| [E2629](../../../experiments/E2629/record.md) | Can either probe rank two **independently trained** arms whose loss gap is known? | **No (Z-d).** E2612's mixture against its prose-only control — 0.108246 bits per character apart, proved matched before any statistic — ranked by nothing. The grammatical probe tied the two arms (0.89000 each) and put the worst model first (0.90000), ρ = −0.866; both continuation readings ordered all three arms correctly and separated nothing, closest p = 0.049 against Holm's 0.016667. |

Four records, all preregistered and committed before the run or the scoring they describe: `cafbb8a` (E2626), `1cadc1a`
with its pre-scoring correction `10ff9c0` (E2627), `ef639f0` (E2628), `9277702` (E2629).

---

## What was built

Model-path code, all of it dependency-free arithmetic except where it loads a checkpoint:

* **`chakaso.training.nn_training`** — a run now declares its optimizer recipe (`ADAM_BETAS`, `ADAM_EPSILON`,
  `ADAM_AMSGRAD`, and a `weight_decay_excludes_vectors` policy that exempts the one-dimensional parameters by shape),
  and can publish a loadable artifact at each declared evaluation point (`curve_artifact_steps`). The declared key set
  is frozen so no committed `protocol_digest` or artifact checksum moves; `as_dict()` elides the new fields at their
  defaults, which is why every pre-existing record's bytes still verify. [ADR-0049](../../decisions/ADR-0049-a-runs-optimizer-settings-are-declared-not-inherited.md),
  [ADR-0050](../../decisions/ADR-0050-a-run-can-write-usable-models-at-declared-points-on-its-curve.md).
* **`chakaso.training.subword_experiment`** — rung writing through the same verifying, refuse-to-clobber writer, a
  `curve_artifacts` report block naming step, path, checksum and the loss recorded for it, and `score_artifacts()` to
  measure several models of one protocol over one prepared held-out pack, refusing a foreign corpus or tokenizer. The
  per-partition packing and unigram baseline are now hoisted out of the model loop (`45aca65`), which is what made a
  27-model scoring pass affordable.
* **`chakaso.training.paired_stats`** — exact McNemar, Spearman ρ with average ranks over ties, Holm's step-down, and
  sign agreement. All four are hand-derived in their tests; the first Holm implementation was wrong and those tests
  caught it.
* **`chakaso.training.probe_sensitivity`** — `compare_instruments` gives, per instrument and never pooled: rank
  correlation against the criterion, share of material pairs separated, smallest gap resolved, largest material gap
  missed, contradicted pairs, and a verdict S-a…S-d.
* **`chakaso.training.context_continuations`** — the same-region distractor construction E2623 named as its own fix,
  with its own digest so v1's stays what E2623 cites.
* **Scripts**: `analyse_probe_sensitivity.py` (curve), `evaluate_context_continuations.py` (shares v1's held-out split
  by import so the two probes cannot drift apart), `analyse_arm_pairs.py` (arms; refuses through
  `chakaso.training.arm_comparison` before it computes anything), and `extract_report.py`, which turns the
  compacted-extract convention four records had been doing by hand into a command.

Checkpoint compatibility was held: every new field is optional and elided at its default, old artifacts verify and
re-save byte for byte, and `MODEL_CARD.md` and `chakaso-base@0.1.0` were not touched.

---

## What was verified about the mechanism, not just claimed

E2626 exists to measure instruments, so it first had to prove the new mechanism did not perturb what it measured. It
did, at the strongest level available: all **43 tensors** of E2614's seed-7 artifact are bit-identical to E2626's final
model and to its rung 2400, the test and validation blocks equal the committed report to the last digit, and the six
intermediate evaluations E2614 never had fall where its curve says. Writing twelve artifacts and evaluating twice as
often cost **15 % of wall clock and moved nothing**.

E2629 added two unplanned reproductions: E2612-T scores 0.82500 across all 800 pairs and 0.89000 on the usable subset —
equal to E2626's `step2400` rung to every digit — and re-scoring the same-region items returned the digest E2627 fixed
(`f49c128e…bcc1`) with the endpoint accuracies that record prints.

---

## Tests and gates, as measured

| Tier | Environment | Result at the final head |
| --- | --- | --- |
| Dependency-free matrix | `.venv` (`[dev]`) | **1,982 passed, 34 skipped**; `mypy src` clean on 125 files; `ruff check` clean; `ruff format --check` clean on 473 files |
| `[train]` | `.venv-train` (numpy) | **2,061 passed, 24 skipped** |
| `[train, nn]` | `.venv-nn` (numpy + torch) | **2,558 passed, 1 skipped** |
| Model-path types | `.venv-nn` | `mypy --config-file configs/mypy-model-path.toml` clean on 168 files |

**Seven test files were added and six extended** (`git diff --name-status` over the phase's range); the new ones are
`test_training_paired_stats.py`,
`test_training_probe_sensitivity.py`, `test_analyse_probe_sensitivity_script.py`,
`test_training_context_continuations.py`, `test_evaluate_context_continuations_script.py`,
`test_extract_report_script.py` and `test_analyse_arm_pairs_script.py`.

CI was read back from the runner, not inferred: **all eight jobs** (core / train / model path across Python 3.11, 3.12
and 3.13) green on `775974c` and again on `35c2504` (run `41605612713`). Three runs during the phase were red and each
had a named cause that was fixed forward — a model-path type error and format drift that the local dependency-free
gates could not see (`da0ef81`), an `importorskip` missing from a test, and a fixture the metrics-extract test never
asked for (`71abc83`). One run was cancelled by the workflow's concurrency rule when a later push superseded it
(`e8bddc2`). No test was weakened or deleted to get green.

---

## What this phase changed about how the project may speak

1. **No capability instrument in this repository has ever ranked two models that differ.** E2621–E2623 showed trained
   models beat random initialisations (floors 0.36–0.50 against the arms' 0.83–0.99); that stands and is a
   floor-versus-ceiling statement, not a ranking. From here, "this arm is better" is a claim about held-out bits per
   character and nothing else measured here corroborates it.
2. **E2626's "rungs are not arms" caveat is now a measurement.** The bound was loose enough to invert: the instrument
   that resolved 63 of 74 material rung pairs down to 0.0218 bits resolved neither of two material arm pairs at 0.108
   and ordered the arms the opposite way while doing it.
3. **A ranking claim needs to clear the family correction, and the record says so when it nearly did.** Two
   continuation tests on the right side of the gap landed at p = 0.031 and p = 0.049 against Holm's 0.05/3 = 0.016667.
   Had E2629 declared one pair rather than three, that would have been reported as a separation.
4. **H17 is a two-seed candidate on two hosts**, and the remaining step is a decision record plus a refit-sensitivity
   check, not a third seed.
5. **The continuation probe's ceiling is a property of the construction** at this scale, not of the distractor's topic
   cue, because the harder construction E2623 proposed was built and did not move it.

---

## Blockers and open issues, honestly

* **[K-001](../../agent/KNOWN_ISSUES.md)** — unchanged. Every arm in this phase truncated at the 2,400-step ceiling with
  validation still falling, so all absolute levels are floors, and every generated sample hit its token bound.
* **[K-032](../../agent/KNOWN_ISSUES.md)** — filed by E2629's deviation. A committed probe extract cannot be re-paired:
  the extractor keeps the tables a record prints and stubs `outcomes`, the per-item array a paired test computes from,
  because a full multi-model probe result is 1.45–2.28 MB and `tests/test_repository_hygiene.py` refuses tracked files
  above 1 MiB. Three shapes are on record and choosing one is a policy decision, not a bug fix.
* **[K-030](../../agent/KNOWN_ISSUES.md) item 3** — ADR-0049 closed the declaration half; the arm that would decide the
  vector-decay exemption is still unrun.
* **[K-031](../../agent/KNOWN_ISSUES.md)** — the raw bytes of the pinned corpus exist only on this host; the drifted
  `science-pg3489` original is not recoverable.
* **One seed per arm everywhere.** E2628 gives H17 two; E2629's three arms are all single draws at seed 7. Nothing here
  replicates a gap's size, only its direction.
* **The criterion is loss, and nothing in this repository validates loss against anything.**
* **Data**: one corpus, one split, archaic English, one vocabulary per arm family, one truncated budget.

---

## Research done in this phase

No new external source was consulted or cited in this phase, and no citation was added to any tracked document. The
decisions came from measurements already in the repository and from what the earlier phases' literature work already
recorded ([open-model landscape](../../research/open-model-landscape.md): nine families, 0 adopt / 1 adapt / 2 test /
8 reject for frontier mechanisms at this scale; the SOTA-audit and Pythia seed-suite notes referenced from
[hypotheses](../../research/hypotheses.md)). Where this phase could have used external evidence — published practice on
paired significance testing and item counts in small-LM evaluation, which is what E2629's next step turns on — it
recorded the question instead of improvising a citation, per the project's rule that an unread source is not written down.

---

## Commits

**36 commits**, `a77ea4e` (07:02 IST) onward, all on `main`; the 34 that did the work run through `35c2504` (14:15 IST)
and the last two carry this report. All were authored and committed by the configured
identity (`bhogesararam23`); **no commit carries a co-author trailer, a generated-by notice, or any added attribution**,
and no `docs/*.docx` private material is tracked (`git ls-files | grep -c docx` returns 0). Of the 34 that did the work:
11 `feat:`, 9 `research:`, 7 `docs:`, 4 `fix:`, 2 `test:`, 1 `perf:`. Across `c498eb8..35c2504` that is
sixty files changed, 100,349 insertions and 27
deletions — the insertions are dominated by the committed probe results and report extracts, which are the point: each
of E2626, E2627, E2628 and E2629 commits its machine-readable evidence beside its record.

**The brief's 50-commit minimum was not met, and the history was not padded to reach it.** The honest reason is that
this phase's work was four experiments and three of them were gated by machine time: E2626's ladder arm ran 5,757 s,
E2628's arm 7,633 s, and E2629's three scoring passes over 215-item and 800-pair instruments took about ten
minutes in total. The repository's own rule — one commit is one coherent change, and a single change is not split into several
commits — means the remaining units were either real experiments or noise. Measured against the stretch target this is a
miss; measured against what a reader can verify from `git log`, each of the 36 is a unit that stands alone, and every
one of the four records was committed before the run or scoring it describes. For context, `main` carried **175 commits since 2026-11-01**
as measured at `d690e1b`, of which these 36 are the last; the count is pinned to that commit rather than to a date so
this report's own commits cannot make it wrong.

Push verification: every batch was pushed as it landed and each push was confirmed — `775974c` and `35c2504` were read
back as green from the runner with all eight jobs, and this report's commits went up on top of them. The end state of
the phase, checked with:

```text
$ git status --short --branch
## main...origin/main
$ git rev-list --left-right --count origin/main...main
0	0
```

— a clean tree with nothing ahead of or behind the remote. No force-push, history rewrite or destructive operation
happened at any point in the phase.

---

## What should happen next, in order

1. **Enlarge the item set, not the model set** (E2629's open lever). The models on the arm pair are already scored and
   the two continuation tests that pointed the right way died on 215 items, not on the gap. A declared 2× item set with
   its digest fixed before scoring is a scoring pass, and it is the only cheap way to find out whether any instrument
   here can resolve an arm-level difference at all. If it cannot, the honest conclusion is that this project needs a
   different instrument, not a bigger run.
2. **H17's adoption record.** Two seeds on two hosts exceed the separable margin; ADR-0048's own "does not decide"
   clause and a refit-sensitivity check on a second fitting sample (measured at 6.6 s for 1,002,483 characters) are
   what stand between the measurement and a default change.
3. **Decide K-032's shape**, because it decides what future probe records can be re-analysed at all.
4. **A second ladder** at seed 11 to test whether E2626's 1600-versus-1800 contradiction reproduces — now demoted,
   since E2629 showed a rung disagreement says little about arms.
5. **The scale question is still gated by data weighting**, not by any instrument: E2612's rule 2 requires an objective,
   the 25 % prose share is measured but not an objective, and the model is still a toy. That, not a new architecture,
   is the next large unit.
