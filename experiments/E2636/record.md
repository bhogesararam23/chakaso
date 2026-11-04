# E2636 — does the reference configuration ever turn its validation curve over, at three times the ceiling?

**Status:** preregistered, **not run**. Written and committed on 2026-11-04 before a single arm exists; no number in this
file is a result, and the `## Results` section will be added by a later commit or not at all. **Type:** Research
experiment, **preregistered**, training-budget class. It is the first record whose protocol pair names the adopted
pre-tokenizer as a property of its own plan ([ADR-0051](../docs/decisions/ADR-0051-the-space-prefixed-pre-tokenizer-is-adopted-and-a-run-stops-inheriting-one.md)),
and the first whose verdict band is derived from the adopted rule's own measured seed spread rather than inherited from
the rule that rule replaced.
**Protocol files:** [`configs/experiments/E2636-adopted-budget-2400.toml`](../configs/experiments/E2636-adopted-budget-2400.toml)
and [`configs/experiments/E2636-adopted-budget-7200.toml`](../configs/experiments/E2636-adopted-budget-7200.toml),
read by `chakaso experiment study --protocol … --seeds N`. A test holds the pair to exactly one differing field.

## Motivation

Every arm this repository has trained at the reference configuration stopped at a declared step ceiling with its
validation loss still falling, and it says so in its own record: E2607 "no run early-stopped and all three selected the
step ceiling, which makes that level a budget floor rather than a converged loss"; E2608 and E2611 "both arms truncated
at the step ceiling"; E2630's P4 "all four arms completed at the 2,400-update ceiling with validation still falling".
The five arms E2630 and E2628 trained under the adopted pre-tokenizer are the strongest case: they are the same
configuration, on this host, and every one of E2630's four reports `best_step == max_steps` with
`completion: budget_ceiling`. `best_step == max_steps` with `completion: budget_ceiling`.

So every absolute held-out figure in the series is a floor of a curve that was never observed turning, and the project
has never seen a validation minimum for a trained model. That is not a tokenizer problem, a data problem or a capability
claim; it is one unanswered question about the budget, and it is the item the last campaign's own next-step list put
first among the things that need a training arm rather than an afternoon. The decision this informs is real and narrow:
**if the curve turns, the next unit is a converged budget and later records can select on validation; if it does not
turn at three times the ceiling, then "longer" is not the lever this project should keep pulling, and the next unit is
data or regularisation instead.**

It tests no hypothesis in [`hypotheses.md`](../research/hypotheses.md). H17 is closed by
[ADR-0051](../docs/decisions/ADR-0051-the-space-prefixed-pre-tokenizer-is-adopted-and-a-run-stops-inheriting-one.md)
and this record changes nothing about it: the vocabulary is a committed input, not a treatment. Nothing here is about
capability either — see *Known limits*.

## Question

**At the reference configuration, under the adopted pre-tokenizer, on one host and one session: does tripling the
update ceiling from 2,400 to 7,200 produce a validation minimum inside the horizon, and does the longer arm's held-out
loss differ from the same run at 2,400 by more than the adopted rule's own band?**

## What is held fixed, and what moves

Held fixed by both protocol files, field for field: corpus `gutenberg-multiregister-v1` 1.0.0 (`312571f9fa8351a8`); the
whole-work split at split seed 7 (`adf650435c3a9068`); the 3,413,760-parameter rotary decoder at `vocab_size` 4096,
`context_length` 256, `d_model` 256, 4 layers, 8 heads, 640 feedforward; AdamW 6e-4 with weight decay 0.01 and the
recipe [ADR-0049](../docs/decisions/ADR-0049-a-runs-optimizer-settings-are-declared-not-inherited.md) declares, decay
applied to every parameter (the exemption stays off, because E2631 measured it and it did not help);
`effective_batch_size` 32 (batch 32 × grad accumulation 1); cosine to 0.1; `warmup_steps` 200; `eval_every` 400;
`patience` 99, so **neither arm early-stops** and the curve is observed rather than truncated by a selector; the packed
training pool `max_training_tokens` 14,033,424 — the same 2,756 whole documents; the evaluation caps 250,000
identifiers per side; and the same five `test_partitions`.

Moves: `max_steps` only, 2,400 → 7,200. Each pair shares its training `seed`, taken from the declared list.

**Vocabulary: fitted by no run.** Both arms read `experiments/E2620/vocabulary-4a47f0281c81a9e5.json` as a committed
input and check its fingerprint ([ADR-0044](../docs/decisions/ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md)),
and both files declare its rule by name:

```toml
tokenizer_pre_tokenizer = "space-prefixed-word|space-prefixed-symbol|whitespace-run"   # SPACE_PREFIXED_PRE_TOKENIZER
```

which is `SPACE_PREFIXED_PRE_TOKENIZER`, the rule
[ADR-0051](../docs/decisions/ADR-0051-the-space-prefixed-pre-tokenizer-is-adopted-and-a-run-stops-inheriting-one.md)
adopted. It is declared rather than inherited — the loader refuses a protocol that fits its own vocabulary without
naming a rule, and E2624 and E2625 already named theirs while pinning a file, which is the precedent this follows.
Nothing is refitted: E2632 measured the rule's refit sensitivity (four fits, within-sample compression ratios
1.405175–1.409317, relative spread 0.30 %) and found compression saturated at the identifier count, so a fresh fit
would buy a fingerprint and not information. The held-out pack is selected with the **legacy**-rule vocabulary
`434debe82f949347` (E2612's, committed beside that record), which is what keeps the scored documents the same text
E2614, E2625 and E2630 scored; a checkpoint this run writes records `tokenizer_pre_tokenizer` because its rule differs
from `LEGACY_DEFAULT_PRE_TOKENIZER`, so an artifact from before the adoption still reads back as the word-run rule it
was trained under.

## Matched, and not matched

| Comparison | Matched? | Why |
| --- | --- | --- |
| The two arms of one pair (same seed) | **matched**, on every field but `max_steps` | Same host, same session, same protocol pair; `scripts/read_budget_extension.py` refuses the reading unless corpus, split, tokenizer, architecture, parameter count, seed, whole optimizer recipe, packed pool, both evaluation caps and the `evaluation_sample` block agree, and unless the update counts sit at exactly the declared 3× ratio |
| This record's short arms against E2630's committed treatment arms | **not matched** | Different session, and possibly a different machine. E2617 measured a cross-host difference of ≤ 0.0025 bits per character on the whole pack — **under the legacy rule** — so that figure is transferred here as an assumption with its origin stated, not as a bound this record has earned |
| Held-out loss against training exposure | **not confounded, and not separated either** | 7,200 × 8,192 positions reads 58,982,400 identifiers out of the same 14,033,424-token pool: about 4.2 passes where the short arm reads 1.40. A difference here is a **budget** effect, meaning compute and exposure together. E2619 (passes at fixed updates, Q-a) and E2624 (equal text, W-b) own the two separations; this record takes neither apart and must not be read as if it did |
| Learning-rate path at a given step | **not matched** | The cosine horizon is each arm's own `max_steps`, so at step 400 the long arm is near peak LR and the short arm is not. This is a longer *run*, not the same run observed later — which is the only version of "more budget" that is a single intervention on the budget |
| The evaluation pack between the two arms | **matched, and proved so** | Same cap, same selection vocabulary, same split seed, and the reader refuses if the two reports' `evaluation_sample` blocks differ. This is K-033's rule enforced by a tool: a whole-pack figure and a register table are different samples, and both must name their own |

## Metrics, bands and decision rules, declared before any arm runs

Primary **O1** — the equal-weight mean of the five register partitions' held-out bits per character, which is what the
series calls O1 and not a pooled figure (K-027). Secondary: the **whole pack**, pooled over the capped test sample,
printed beside O1 and never confused with it. Tertiary and descriptive: the per-register table, the validation curve in
nats per token, and the train/validation gap at the endpoint.

Δ = short arm − long arm, positive when the **longer** budget scores fewer bits per character.

Two bands, both computed from committed evidence by the reading tool, neither typed into it:

* **the adopted rule's own band, 0.005** — `1.96·√2·σ` over the adopted rule's measured seed spread σ(O1) = **0.001595**
  from [E2630's committed `results.json`](E2630/results.json) `treatment_spread.o1`, which is 0.004418, rounded up to the
  next 0.005 by the replacement rule E2614 declared for a band it retires. E2630 measured that figure as the *treatment's*
  own spread — the number every earlier record in the series said it could not state — and it is smaller than the legacy
  rule's 0.002240, which is why a band built on it is tighter rather than looser.
* **the inherited band, 0.02** — E2614's, still carried in that record's `margins.o1_separable`, printed beside the
  derived one.

**The relationship, stated rather than assumed.** E2614's 0.02 was measured under the pre-tokenizer ADR-0051 replaced;
0.005 is 0.250 of it. That does not retire 0.02: every verdict already recorded against it — E2612's H2 prose cost,
E2613's mathematics reading, E2614's own R1/R2/R3 table — was read on the old band, and silently moving a threshold
would rewrite what those records said. Both figures are therefore printed, the verdict names the one it was read on, and
the derived band is the *primary* because these arms run under the rule that band was measured from. Where a difference
clears 0.005 but not 0.02, the record says so in those words. The whole pack gets no derived band at all: E2630's
`treatment_spread` carries O1 and the five registers but no whole-pack σ, so the whole-pack reading stays on the
inherited 0.00933 tie margin and is labelled as inherited rather than quietly re-derived.

The arms run in the declared order **7, 13, 17** (the first three of E2630's own seed list, taken in that list's order so
the choice cannot be accused of cherry-picking). A pair is complete when both of its arms finish. An arm that dies is
relaunched once; a second failure drops that seed with its name in the record, and the reading is made over the
remaining pairs — the same failure policy E2630 declared. **The tool refuses below two completed pairs**, because a mean
over one draw is that draw, so a single surviving pair is reported as a shortfall with its raw difference and no letter.

Decision rules, checked in declaration order; **C-d is the declared residue** and is shadowed by every other letter,
which the artifact reports in `also_matched` rather than hiding. The full conditions and the sentence each letter means
are committed in [`configs/analyses/budget-extension-e2636-v1.json`](../configs/analyses/budget-extension-e2636-v1.json)
([ADR-0052](../docs/decisions/ADR-0052-an-analysis-prints-the-verdict-its-record-declared-or-none-at-all.md)); a letter
is printed only when that file is handed to the tool, and `--statistics-only` prints none.

* **C-a — the curve turned.** Every declared long arm's validation minimum falls strictly inside its own horizon, the
  mean Δ(O1) clears 0.005 and its 95 % Student-*t* interval excludes zero. Consequence: this configuration has a
  converged budget, the 2,400-update figures across the series are floors of an unfinished curve and can be replaced by a
  level, and a later record can select on validation instead of running to a ceiling.
* **C-b — the ceiling is still binding at three times.** No long arm bottoms out, and the mean Δ(O1) clears 0.005.
  Consequence: "longer" keeps helping and has not yet stopped; every absolute number in this series stays a floor, and
  the next budget question is a scaling question rather than a run at this configuration.
* **C-c — more compute cost held-out loss.** The mean Δ(O1) is at or below −0.005. Consequence: the step ceiling was
  never the binding constraint at this scale, and the next unit on the model path is data or regularisation, not budget.
* **C-d — anything else**: a turn in some arms and not others, a gain inside the band, or a declared set that did not
  complete. Every paired difference, both bands, both intervals and every curve are reported and no mechanism is claimed.

**No outcome here is a capability claim**, and none moves **K-001**: no model in this repository answers a grounded
question or produces useful text, and a lower held-out loss at 7,200 updates would be a lower held-out loss at 7,200
updates on archaic English.

## Descriptive predictions (none decision-bearing)

* **P1.** Every long arm still reports `completion: budget_ceiling` and `best_step` at or near its own ceiling, because
  every arm in the series has behaved that way. A long arm that bottoms out early would be the surprise, and C-a is the
  letter that names it.
* **P2.** Δ(O1) is positive on every pair, and smaller per additional update than the first 2,400 updates bought:
  E2630's arms sit at O1 ≈ 1.996 and the curve has been falling at roughly 0.03 bits per character per doubling of
  nothing in particular, so the honest expectation is a gain of the same order as one more interval of decay, not a
  step change.
* **P3.** The register ordering at the endpoint is unchanged, with mathematics still the loosest partition (its σ is
  0.006712 under the adopted rule, four times technology's) and medicine still the thinnest held-out sample.
* **P4.** The train/validation gap at 7,200 is wider than at 2,400, since the longer arm reads 4.2 passes of the same
  2,756 documents. If it is not, the model is not memorising the pool and the budget reading is cleaner than expected.
* **P5.** No pair is dropped for a machine reason.

## Known limits, declared in advance

* One corpus, one split, one vocabulary, one configuration, three seeds. This is a statement about **this** curve at
  **this** scale, not a scaling law.
* Both arms are still truncated by the same *kind* of rule; a longer run that turns over at step 7,100 would produce C-a
  on the strength of the curve, and the record would then report where the minimum sits rather than treating 7,200 as a
  converged budget.
* The band transfer is an assumption. σ(O1) = 0.001595 is measured at 2,400 updates; nothing in this repository has
  measured the seed spread of a 7,200-update arm, so the derived band is the one the arms at this horizon earned and no
  more. E2614's own note — "a transfer is assumed in all of them" — applies here with the direction reversed.
* Δ is read between same-seed arms, which removes initialisation variance from the difference but not its interaction
  with budget; the Student-*t* interval is over seeds, so it describes the spread of the *difference* across seeds.
* The passes/confound above is not separated. A C-b or C-a result licenses saying "more budget helps / stops helping at
  this shape"; it does not license saying which of compute or exposure did it.
* Nothing here is an instrument validation. E2629's Z-d and E2635's D-b stand: no capability probe in this repository
  separates two independently trained arms, so whatever the loss says at 7,200 updates, the probes will not corroborate
  it.

## Cost, from measured runs on this configuration

Measured, this host, the same configuration at 2,400 updates: **3,045–3,536 s** per arm
([campaign report §1](../docs/internal/engineering/research-campaign-report-2026-11-04.md): "a 3,413,760-parameter arm at
2,400 updates costs 3,045–3,536 s here when the machine is otherwise idle"), **5,834–6,759 positions per second**, peak
resident **2,502–2,544 MB** per process, and `resources.available_at_launch` in each report. E2631 measured 3,353 s at
the same shape; E2628's seed-11 arm cost 7,634 s on a busy host, which is why the arms run one at a time.

The long arm is arithmetic on those measured rates, **not a measurement**: 58,982,400 positions at 5,834–6,759
positions per second is **8,727–10,110 s**, so **2.4–2.8 hours**. Six arms (three pairs) therefore cost
**35,316–40,938 s ≈ 9.8–11.4 hours** of single-process wall clock on this machine, plus about 43 s of corpus preparation
per arm's snapshot read and no network. Peak resident stays ~2.5 GB because one process runs at a time; the six
checkpoints are ~18 MB each and live in `data/`, git-ignored, cited by digest.

Two things follow. First, **this record is not scheduled here**: it is preregistered for a separately approved compute
session, and nothing in it has run. Second, the rates are host-specific, so the session runs **seed 7's pair first** and
prints the long arm's measured `positions_per_second` before the remaining two pairs are launched; if that measured rate
puts the whole record beyond the session's budget, the shortfall is reported as a deviation with the completed pairs
named, and the rules above do not move.

## Reproduction

Nothing below has been executed. It is written now so that the later run cannot be adjusted to fit its own result.

```bash
cd /d/chakaso
SNAP=data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json

# Six arms, in the declared order: seed 7's pair first, and the long arm's measured rate read before the rest launch.
for SEED in 7 13 17; do
  chakaso experiment study --work-dir data/experiments --experiment E2636 --seeds "$SEED" --split-seed 7 \
    --expected-corpus-fingerprint 312571f9fa8351a8 --corpus-snapshot "$SNAP" \
    --protocol configs/experiments/E2636-adopted-budget-2400.toml \
    --write-report data/experiments/E2636/short-$SEED/study-$SEED.json
  chakaso experiment study --work-dir data/experiments --experiment E2636 --seeds "$SEED" --split-seed 7 \
    --expected-corpus-fingerprint 312571f9fa8351a8 --corpus-snapshot "$SNAP" \
    --protocol configs/experiments/E2636-adopted-budget-7200.toml \
    --write-report data/experiments/E2636/long-$SEED/study-$SEED.json
done

# Each arm's compact report is committed beside this record (scripts/extract_report.py) before the reading runs.
python scripts/read_budget_extension.py \
  --short 7=experiments/E2636/E2636-short-7-report.json \
  --short 13=experiments/E2636/E2636-short-13-report.json \
  --short 17=experiments/E2636/E2636-short-17-report.json \
  --long 7=experiments/E2636/E2636-long-7-report.json \
  --long 13=experiments/E2636/E2636-long-13-report.json \
  --long 17=experiments/E2636/E2636-long-17-report.json \
  --bands experiments/E2630/results.json \
  --expected-ratio 3 --expected-pairs 3 \
  --outcomes configs/analyses/budget-extension-e2636-v1.json \
  --out experiments/E2636/results.json
```

The reader refuses a pair that differs in anything but `max_steps`, refuses an arm that did not finish at its own
ceiling, and refuses a report whose `best_step` disagrees with the validation argmin it computes from the curve. `--bands`
is [E2630's committed aggregate](E2630/results.json), so the band is read from the record that measured it instead of
being typed into a protocol. Running `--statistics-only` instead prints the numbers with no letter, which is what a
record that has not declared its outcomes gets (ADR-0052).

## What the run will report

Nothing has been measured, so there is no results section here — a `## Results` heading appears in the commit that reads
the run, and the hygiene gate that enforces that is the reason this record is written before it. That section will carry:
the six arms' completion state with any dropped seed named, the per-pair Δ(O1) and Δ(whole pack) with both bands beside
each, each long arm's validation minimum step and final-quarter slope, the register table, the `also_matched` residue the
declaration leaves, and a Corrections/deviations block in the series' shape if any prediction above fails.

## Follow-up

1. **C-a** → the converged budget becomes the reference schedule, E2614's 2,400-update configuration is retired with a
   dated note beside its numbers rather than edited, and the probe records get a curve that has a floor to approach.
   **C-b** → budget stops being the next unit; the data side (K-022, ADR-0041's equal-token experiment) becomes it.
   **C-c** → the same, with a reason attached.
2. Whichever letter lands, the passes/updates confound this record declares stays open: separating it is a
   fixed-passes-at-larger-pool design, which E2619 built the shape of at 2,400 updates.
3. A cross-host repeat of one completed pair, bounded by E2617's figure rather than assuming it, which is what makes the
   comparison to E2630's committed arms something other than a hope.
