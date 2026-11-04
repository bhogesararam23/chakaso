# E2626 — can either capability probe detect an improvement this project would call real? A training curve is the ordering criterion

**Status:** measured — **neither instrument earns S-a. The minimal-pair probe tracks the curve (ρ 0.951) and separates 63 of the 74 materially-different rung pairs, down to a 0.0218-bit gap, but it contradicts the loss on one pair; the continuation probe tracks it (ρ 0.918) and separates 23, and after update 400 it resolves nothing.** Preregistered and committed (`cafbb8a`) before the run, with this record and [`configs/experiments/E2626-curve-ladder.toml`](../../configs/experiments/E2626-curve-ladder.toml) as one declaration; the Results section was added after the report and the two probe outputs were on disk.
**Type:** Research experiment, **preregistered**. One arm. It is an instrument measurement, not a model-quality measurement: nothing about the
mixture, the tokenizer or the budget changes, and the model it trains is E2614's at E2612 arm T's configuration.
**Protocol file:** [`configs/experiments/E2626-curve-ladder.toml`](../../configs/experiments/E2626-curve-ladder.toml), read by
`chakaso experiment study --protocol … --seeds 7`. A test
(`tests/test_experiment_protocol_files.py::test_the_ladder_differs_from_e2614_only_in_how_often_it_evaluates_and_what_it_writes`)
holds it to E2614's plan on exactly two fields, `eval_every` and `curve_artifact_steps`.

## Why this experiment exists

The open question is not about the model. The campaign's own summary puts it plainly: **no instrument in this repository is currently known to
detect a real improvement in what the model has learned.** Both capability probes have been measured only against one endpoint — a fully trained
arm versus a random initialisation:

* the grammatical minimal-pair probe, version 2 ([E2622](../E2622/record.md)): valid at the headline level on **three of eight phenomena**
  (0.86–0.95 trained against a 0.47–0.51 untrained floor), and its seed-to-seed standard deviation on those 300 pairs is **0.015–0.048** — larger
  than most of the differences between the arms it was asked to rank, which is why its single-seed ordering was withdrawn;
* the held-out continuation probe ([E2623](../E2623/record.md)): **near ceiling** — six trained models between 0.958 and 0.986 on the easy
  distractor, all intervals overlapping, "it cannot rank the trained arms".

Held-out bits per character, by contrast, is measured at a seed spread of 0.0023–0.0123 (E2614). So the project's fine-grained criterion moves in a
range its capability instruments cannot resolve, and the sharpest thing anyone here could say about a gain is "the loss fell".

To find out whether an instrument can see a real improvement, it has to be scored against an ordering that is **already measured, already monotone,
and costs nothing to produce**. A training curve is that ordering: the loop records a held-out number at every evaluation, the updates advance by a
function of the absolute step, and until now no run could publish a model at any point before its end. ADR-0050 gives it that: at each declared
evaluation the run writes a usable artifact beside the loss recorded for that same step.

**Question.** Along one training curve, for pairs of rungs whose held-out bits per character differ by more than the project's own materiality
threshold, does either probe separate them — and if not, what is the smallest loss gap each probe can resolve at all?

It measures two instruments against one criterion. It says nothing about whether the model is useful (K-001), and it does not test a hypothesis about
data, architecture or budget.

## What is held fixed, and what moves

Held fixed — everything of [E2614](../E2614/record.md) seed 7 (= E2612 arm T = the control of E2617, E2620 and E2625): corpus
`gutenberg-multiregister-v1` 1.0.0 (`312571f9fa8351a8`), whole-work split with split seed 7 (`adf650435c3a9068`), the committed vocabulary
`434debe82f949347` read as an input (ADR-0044), the 3,413,760-parameter rotary decoder, 2,400 updates × 8,192 positions, cosine to 0.1, AdamW 6e-4
with the settings ADR-0049 now declares, warmup 200, 20,000,000 training identifiers, seed 7, this host.

Moves — **how often the run looks, and what it leaves behind**: `eval_every` 400 → 200 and `curve_artifact_steps` = (200, 400, …, 2400). Neither
touches a weight: `plan_batch` and `_learning_rate` are functions of the absolute update index, an evaluation consumes no random numbers, and
`save_artifact` validates inside `torch.random.fork_rng` (K-030 item 2). That claim is not assumed here — it is **checked** by P4 below, and by a
test that runs the same plan with and without rungs and asserts the selected weights and the protocol digest are identical.

## The criterion, in the project's own unit

For every rung the analysis script computes **test-pack bits per character**, and the **equal-weight mean of the five register partitions (O1)**, over
exactly the pack the run's own final evaluation used, reconstructed from the same snapshot, split and vocabulary. Two rungs are then
**materially different** when their test-pack bits per character differ by more than **0.02** — the threshold this project has used since E2611 to
call two arms different, and the one E2614 judged against a measured seed spread (0.00625 at its centre on prose, 0.00933 on the whole pack).

Why test-pack bpc and not the validation curve the run already records: it is the number every other record is written in, so a sensitivity figure
produced here can be set beside an arm comparison without a unit conversion. The validation curve is still reported, as the ordering the loop
measured without any extra work.

Two rungs of one curve are **not** two arms: they share their initialisation, their data order and their entire weight trajectory, and the later one
is the earlier one after more updates. The comparison is therefore *more* favourable to the loss criterion than a real arm comparison is, and that is
stated rather than smoothed: a probe that cannot resolve these pairs certainly cannot resolve two independently trained arms whose loss gap is the
same size.

## The instruments, unchanged

Both are used exactly as committed, with no new scoring rule, no new pair and no new item:

* **minimal pairs v2** — `configs/probes/grammatical-minimal-pairs-v2.json` (digest `2487a4ab…07208`), scored by
  `scripts/evaluate_minimal_pairs.py`; the decision-bearing reading is the **three usable phenomena** (300 pairs) that E2622 identified, with all 800
  reported;
* **continuation** — the 215 items of `E2623` (digest `9557a327…acdcd`), scored by `scripts/evaluate_continuations.py`; the decision-bearing reading
  is **PMI against the easy distractor**, with the hard one reported.

The untrained initialisation is the curve's rung 0: `--control` builds it from the artifact's architecture with seed 7, which is the same
`torch.manual_seed(7); build_model(...)` the run started from, and its test-pack bpc is the step-0 number the run itself records.

**Paired separability.** Two probe scores on the same item set are a paired comparison, and E2621–E2623 compared marginal Wilson intervals — the
weaker test. Here each rung pair is judged by **McNemar's exact binomial test** on the items where the two rungs disagree (two-sided, α = 0.05),
Holm-corrected within each instrument across the pairs tested. A pair is **separated** when the test rejects. This is stated now because it favours
the instruments: a paired test resolves differences the marginal intervals could not, so an instrument that still fails here fails for a real reason.

## Decision rules, declared before the run

Let *M* be the set of rung pairs (including rung 0) that are materially different as defined above. For an instrument, let *R* ⊆ *M* be the pairs it
separates, and call a separation **wrong-directed** when it favours the rung with the *higher* loss. Spearman ρ is taken over all rungs between the
instrument's score and the instrument-free criterion (test-pack bpc).

* **S-a — the instrument detects a real improvement.** *R* covers at least 90 % of *M*, no separation is wrong-directed, and ρ ≥ 0.9. Its smallest
  resolved gap is reported. Consequence: the probe may be cited as corroboration of a loss change at this scale.
* **S-b — it detects only large improvements.** ρ ≥ 0.9, no wrong-directed separation, but under 90 % of *M* is resolved. The headline number is
  **the smallest loss gap it resolves**, and the plateau: the loss remaining after its score stops moving.
* **S-c — it does not track.** ρ < 0.9, or any wrong-directed separation, or it fails to separate rung 0 from the endpoint (which E2622 and E2623
  already measured, so this would contradict those records and be reported as a defect in this run rather than a finding).
* **S-d — anything else**, with the numbers.

Each instrument gets its own outcome; they are judged separately and neither is allowed to rescue the other. Reported for both, not decision-bearing:
the hard distractor, all 800 pairs, every phenomenon and every register, and the rung-to-rung adjacent steps.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** The continuation probe separates rungs only in the first third of the curve (up to about update 800) and never among the last six; its
  smallest resolved gap exceeds 0.10 bits per character.
* **P2.** The minimal-pair probe resolves somewhat further down the curve than P1's figure, because E2622 measured it at 0.65 → 0.95 across arms
  while E2623's probe sits at 0.96–0.99; its ceiling phenomenon set is the same three phenomena.
* **P3.** At least one adjacent-rung difference is non-monotone for the minimal-pair probe. E2622's own seed spread (0.015–0.048 on 300 pairs) is
  larger than the top-of-curve differences, so exact monotonicity would be the surprise.
* **P4 — the validity check on the mechanism.** The endpoint rung (2,400) reproduces the numbers of the artifact already on disk from E2614 seed 7:
  the same test-pack bits per character (2.05544739454246, exact) and the same probe accuracies, since the probe scripts are deterministic. If it
  does not, the ladder perturbed the run it was recording, ADR-0050's central claim is false, and **no rung in this record may be read as anything**
  — that is stated here so a failure has a declared consequence rather than an explanation written afterwards.

## Cost

Preparation 42 s (measured on E2614 seed 7 here); the arm ≈ 83 min at 4,164 positions/s on this host, plus twelve held-out evaluations instead of
six and twelve artifact writes. Scoring: E2622 measured 5 min 38 s for thirteen models batched, so thirteen rungs and the control is about the same;
the per-rung test-pack scoring is roughly a run-end evaluation repeated twelve times. **About 1.5–2 machine-hours, sequential.** The rungs cost
12 × 17.9 MB ≈ 215 MB in `data/`, which is git-ignored.

## Reproduction

```bash
chakaso experiment study --work-dir data/experiments --experiment E2626 --seeds 7 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2626-curve-ladder.toml --write-report data/experiments/E2626/study.json
python scripts/evaluate_minimal_pairs.py --pairs configs/probes/grammatical-minimal-pairs-v2.json \
  --control --checkpoint E2626-step200=data/experiments/E2626/seed-7/E2626-step200-decoder.pt … \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --out experiments/E2626/minimal-pairs.json
python scripts/evaluate_continuations.py \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --vocabulary experiments/E2612/vocabulary-434debe82f949347.json --expected-split-fingerprint adf650435c3a9068 \
  --registers prose-:100,science-:55,technology-:40,mathematics-:20 --max-per-work 5 --control \
  --checkpoint E2626-step200=… --out experiments/E2626/continuations.json
python scripts/analyse_probe_sensitivity.py --report data/experiments/E2626/seed-7/report.json \
  --protocol configs/experiments/E2626-curve-ladder.toml --work-dir data/experiments/E2626/seed-7 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --split-seed 7 --expected-corpus-fingerprint 312571f9fa8351a8 \
  --pairs experiments/E2626/minimal-pairs.json --continuations experiments/E2626/continuations.json \
  --usable-phenomena auxiliary_agreement,determiner_noun_agreement,reflexive_pronoun_agreement \
  --materiality 0.02 --out experiments/E2626/results.json
```

*The reproduction block was corrected on 2026-11-03 after the analysis was built: as preregistered it named only the two probe files, the
materiality and the output, because the criterion had to be computed per rung and the arguments that reconstruct the held-out pack (the
protocol, the work directory, the snapshot, the split seed and the corpus fingerprint) and the `--usable-phenomena` list — which E2622
declares and the script must be told — had not been spelled out. Nothing about the measurement changed: the same pair set, the same item
set, the same 0.02 band and the same rules. The exact command that ran is recorded in `data/run_e2626_analysis.sh` alongside the two probe
invocations above.*

## Known limits, declared in advance

* **One curve, one seed, one configuration.** The ordering is along one trajectory in weight space; two independently trained arms differ in more
  than that, and a probe's sensitivity here is an upper bound on what it will show across arms, not a lower one.
* **The criterion is loss, and the project has seen loss and text quality disagree** (E2612: the better-loss arm produced *more* repetitive text).
  This record measures agreement between an instrument and the loss; it does not validate the loss against anything, and it cannot say that a probe
  which tracks the loss tracks usefulness.
* Items within the minimal-pair set are templated from a 16-noun, 10-verb lexicon and continuation items are capped at five per work, so the paired
  tests' independence assumption is optimistic for both — the same limit E2622 and E2623 carried, unchanged here.
* Rung 0 is an initialisation of the same architecture with the same tokenizer; it is not a different model in any other respect.
* McNemar treats each item as scored identically by both rungs; it is, because both use the same committed pair and item files, which is why the
  comparison is paired at all.
* Twelve evaluations of one run are autocorrelated by construction; no interval across rungs is computed, only separations.
* Everything this record touches is archaic English from one corpus family (K-023's within-work caveat does not apply: the split is whole-work), and
  nothing here is a statement about capability (K-001).

---

## Results (added 2026-11-03)

Produced by [`scripts/analyse_probe_sensitivity.py`](../../scripts/analyse_probe_sensitivity.py) from files committed beside this record:
[`E2626-report.json`](E2626-report.json) (the run, compacted by `scripts/extract_report.py`, rung list and artifact digests intact),
[`minimal-pairs.json`](minimal-pairs.json) and [`continuations.json`](continuations.json) (the two probes' per-model tables; each model's per-item
outcome array is replaced by the SHA-256 of its canonical JSON, because the raw files are 1.4 MB and 1.5 MB against a 1 MiB tracked-file limit, and
the bytes those stubs digest are in `data/experiments/E2626/`), and [`results.json`](results.json) — the comparison itself: every pair, its loss gap,
its discordant counts, its p-value, whether Holm rejected, and which way the direction fell.

### What the per-item arrays look like now (added 2026-11-04)

[`minimal-pairs-with-outcomes.json`](minimal-pairs-with-outcomes.json) and
[`continuations-with-outcomes.json`](continuations-with-outcomes.json) are new extracts of the *same two runs*, written
by the same `scripts/extract_report.py` after K-032 was fixed, and they carry every per-item outcome array the stubbed
files above left out — 127 KB and 87 KB against the 1.45 MB and 1.55 MB the scoring wrote. Their `report_file_sha256`
equals the digest the extracts committed on 2026-11-03 recorded, so both files describe the same bytes on disk, and
nothing in this record's numbers is restated by the new pair: they exist so a reader can recompute the tests rather
than check the arithmetic that was reported.

### P4 first, because everything else depends on it

The endpoint guard passed. Rung 2400 scores **2.05544739454246** bits per character with the same `positions_scored` and `characters_scored` as the
run's own reported test block, and its `test_partitions` block equals the report's field for field. Beyond the report, the weights were compared
directly: **all 43 tensors of [E2614](../E2614/record.md)'s seed-7 artifact on this host are bit-identical to E2626's final artifact, and rung 2400's
tensors are bit-identical to both.** The validation curve agrees at every point E2614 recorded — 3.489956081890669, 3.190452286454498,
3.0628919796865492, 2.9782929967661373, 2.923530496534754, 2.9031519811661517 at steps 400 through 2400, to the last digit — and the ladder adds the
six intermediate evaluations E2614 never had. Writing twelve usable artifacts and evaluating twice as often did not perturb the curve being recorded:
ADR-0050's central claim holds at full scale on a real run, not only in the in-process test. Wall clock rose from 4,991 s to 5,757 s (+15 %) for the
six extra evaluations and twelve writes; throughput fell from 4,164 to 3,766 positions/s on a busier machine, which is a property of the host and not
of the mechanism.

### The curve

Whole-pack test bits per character, the equal-weight register mean, and the two probes' decision-bearing readings:

| Rung | whole pack | O1 | minimal pairs, 300 usable | all 800 | continuation PMI, easy | hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (untrained) | 5.7533 | 5.6797 | 0.4967 | 0.4725 | 0.4930 | 0.4930 |
| 200 | 2.7365 | 2.6818 | 0.4500 | 0.5350 | 0.8093 | 0.7814 |
| 400 | 2.4424 | 2.3966 | 0.5300 | 0.6325 | 0.9395 | 0.9209 |
| 600 | 2.3187 | 2.2736 | 0.6133 | 0.7188 | 0.9581 | 0.9395 |
| 800 | 2.2421 | 2.2017 | 0.7333 | 0.7562 | 0.9488 | 0.9256 |
| 1000 | 2.1939 | 2.1555 | 0.7233 | 0.7475 | 0.9721 | 0.9442 |
| 1200 | 2.1569 | 2.1171 | 0.8167 | 0.7987 | 0.9674 | 0.9209 |
| 1400 | 2.1343 | 2.0933 | 0.7633 | 0.7662 | 0.9860 | 0.9349 |
| 1600 | 2.1092 | 2.0701 | 0.8800 | 0.8263 | 0.9767 | 0.9256 |
| 1800 | 2.0859 | 2.0486 | 0.8067 | 0.7900 | 0.9814 | 0.9488 |
| 2000 | 2.0692 | 2.0338 | 0.8667 | 0.8100 | 0.9860 | 0.9442 |
| 2200 | 2.0640 | 2.0284 | 0.8867 | 0.8263 | 0.9814 | 0.9535 |
| 2400 | 2.0554 | 2.0214 | 0.8900 | 0.8250 | 0.9860 | 0.9535 |

78 pairs were tested per instrument; 74 of them are material by the declared rule (a loss gap above 0.02 bits per character).

### The preregistered reading

* **`minimal_pairs_usable` — S-c.** ρ = **+0.9505**, coverage **63 of 74 material pairs = 0.851**, and one separation **contradicts** the criterion:
  rungs 1800 and 1600, where the loss says 1800 is better by **0.0233** bits per character and the probe says 1600 is better — 24 discordant items to 2,
  p = 1.0 × 10⁻⁵. By the rule as written (any wrong-direction separation is S-c) that is the verdict, and the rule was written that way on purpose: an
  instrument that can be confidently wrong about which of two models learned more is not one to corroborate a loss change with.
* **`minimal_pairs_all_800` — S-c**, on the same pair; ρ = +0.9216, coverage 64/74.
* **`continuation_pmi_easy` — S-b.** ρ = **+0.9184**, no contradiction anywhere, but coverage **23/74 = 0.311** and a **smallest separated gap of 0.2941
  bits per character** — fifteen times the materiality band. Against the endpoint it separates only rungs 0 and 200; from rung 400 onward, **0.387 bits
  per character of held-out improvement later**, it resolves no pair against the endpoint at all, and among the 36 pairs whose rungs both sit at or past
  update 800 it separates **none** of the 32 that are material. E2623's ceiling was not an overlap of intervals; it is insensitivity over the last two
  thirds of the curve, and the size of it is now measured.
* **`continuation_pmi_hard` — S-c**: ρ = +0.8398, below the tracking threshold; coverage 23/74.
* **No instrument earned S-a.** The honest headline: the repository now has **one instrument that demonstrably moves with what the model has learned
  across most of a training curve**, resolving material gaps down to **0.0218 bits per character** — inside the band this project uses to call two arms
  different — and **one that has a working range only over the first third of the curve**. It also has its first documented counter-example: at the top of
  the curve, more training and a better held-out loss produced a significantly worse grammatical reading.

### Sensitivity is not a threshold

The minimal-pair probe separates pairs whose loss gaps run from **0.0166** (an *immaterial* gap — a rung pair the band calls a tie) up to 3.70, while
failing to separate **3.31** and **3.02** — the untrained floor against rungs 400 and 200, where both models sit at chance on the instrument (0.4967
against 0.5300 and 0.4500) — and failing on several top-of-curve pairs of 0.048 to 0.108 bits. Reported without the multiplicity correction the shape is
unchanged and so are the verdicts: 65 of 74 material pairs at p < 0.05 for the minimal-pair probe with a minimum material gap of 0.0218, and 33 of 74 for
the continuation probe with a minimum of 0.1077. The correction is therefore not what produced this answer — which is worth saying, because 78 comparisons
drawn from one autocorrelated trajectory are not 78 independent tests and Holm treats them as nearer to that than they are. What a reader should take from
the numbers is that **these are upper bounds on the fineness a probe resolves along a curve and lower bounds on how often it is right**: adjacent rungs
share almost all their weights and almost all their item outcomes, and the discordant counts that carry the paired test are often single digits.

### The predictions

* **P1 held, more strongly than predicted.** Nothing among the last six rungs is separated by the continuation probe, and its smallest resolved gap is
  0.294 — nearly three times the 0.10 the record expected.
* **P2 held.** The minimal-pair probe resolves to 0.0218 on material pairs where the continuation probe needs 0.294, and its usable set is the same three
  phenomena E2622 named.
* **P3 held.** The probe is non-monotone against the loss in at least three places: 800 → 1000 (0.7333 → 0.7233 while the loss falls), 1600 → 1800
  (0.8800 → 0.8067 while the loss falls, and significantly so), 2200 → 2400 (0.8867 → 0.8900 for 0.0086, a tie by any band).
* **P4 held**, and it is stated first above because the rest depends on it.

### What this changes, and what it does not

1. **The campaign's sharpest open question now has a number attached.** Where
   [`chakaso-research-campaign-2026-11.md`](../../docs/research/chakaso-research-campaign-2026-11.md) §7 said no instrument here is *known* to detect a
   real improvement, it is now known that the minimal-pair probe detects held-out-loss improvements of about 0.022 bits per character and larger along a
   training curve, and that the continuation probe does not detect them below 0.29. The first is usable as corroboration of a loss change; the second is
   usable as a floor test and nothing more.
2. **Neither is yet a ranker for the comparisons the project actually makes.** E2615/E2618's mixture differences are 0.009–0.014 bits per character on O1
   and E2625's tokenizer gain is 0.0229. Resolving a 0.0218 gap *along one trajectory* is not resolving a 0.022 gap *between two independently trained
   arms*: E2622 measured this probe's seed-to-seed spread at 0.015–0.048 on the same 300 pairs, which is the same size. That difference is exactly what
   this record cannot close, because rungs are not arms — and it is the reason the verdict is S-c rather than "the probe works".
3. **The 1600/1800 contradiction is a finding, not noise to average away.** It is E2612's pattern — loss and a text measure moving opposite ways —
   reproduced inside one model, at one seed, with a paired test and a p-value. Any later claim that a loss improvement is "also" a capability improvement
   has to survive it.
4. **The ladder stays.** Twelve usable models from one run, bit-identical to the run that would have existed without them, is a permanent capability of
   the experiment runner (ADR-0050), and it bought the six intermediate points on the curve for 15 % more wall clock.

### What was not measured here

No new item set, no new scoring rule, no per-register split of the verdict (the probe files carry it; it is not decision-bearing), no alternative family
definition for the multiplicity correction, and no second seed. The ordering criterion is one trajectory, so this record cannot say whether the 1600/1800
contradiction reproduces across initialisations — that would need the same ladder at another seed. [E2627](../E2627/record.md) is the next instrument
measurement and is already preregistered.

### Addendum (2026-11-04) — the S-letters this record printed are now a declaration it owns

Nothing above is edited. [ADR-0052](../../docs/decisions/ADR-0052-an-analysis-prints-the-verdict-its-record-declared-or-none-at-all.md)
removed the `S-a`…`S-d` constants and the 0.9 bounds that were `MATERIAL_COVERAGE` and `TRACKING_THRESHOLD` from
`chakaso.training.probe_sensitivity`, because this record declared them and later records did not. The declaration is
committed at
[`configs/analyses/probe-sensitivity-e2626-v1.json`](../../configs/analyses/probe-sensitivity-e2626-v1.json) — the same
conditions, the same order, this record's thresholds — and re-running this analysis now requires
`--outcome-vocabulary` pointing at it, or `--statistics-only` to read the numbers with no letter. The file was verified
by re-running [E2629](../E2629/record.md)'s arm-pair analysis through it: all eight instrument verdicts came back
identical to that record's committed `results.json`, so the declaration reproduces what the constants computed. This
record's own rung-ladder command was not re-run here, because it re-scores thirteen artifacts for a check that the
verdict table does not need.
