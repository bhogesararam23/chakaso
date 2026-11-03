# E2626 — can either capability probe detect an improvement this project would call real? A training curve is the ordering criterion

**Status:** preregistered — the record and [`configs/experiments/E2626-curve-ladder.toml`](../../configs/experiments/E2626-curve-ladder.toml)
are committed **before** the run, and the decision rules, the materiality convention and the predictions below are fixed now.
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
  --pairs experiments/E2626/minimal-pairs.json --continuations experiments/E2626/continuations.json \
  --materiality 0.02 --out experiments/E2626/results.json
```

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
