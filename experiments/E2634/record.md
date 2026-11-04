# E2634 — does four times the items let the continuation probe rank the arms that loss ranks?

**Status:** preregistered — committed before any enlarged item set was scored. **Type:** Research measurement,
**preregistered**. It scores three checkpoints that are already on disk; it trains nothing and produces no loss number.

## Question

[E2629](../E2629/record.md) asked both capability instruments to rank two independently trained arms and got
**Z-d: nothing ranked it.** The grammatical minimal-pair probe tied E2612's mixture against its prose-only control at
0.89000 and put the *worse* model on top (0.90000), ρ = −0.866 against the loss ordering. Both continuation readings
ordered all three arms correctly and separated nothing, the closest being **p = 0.049 against Holm's 0.016667 step**
over three pairs. K-001 stands, and every claim of the form "this arm is better" rested on held-out loss and nothing
else measured.

There are two reasons an instrument can rank without separating, and E2629 could not tell them apart:

* it has too few items — 215 passages, and a pair with a real but modest effect sits at p = 0.049, one step short; or
* the effect it measures does not track what loss measures — in which case more items buy a *sharper* wrong answer.

[E2633](../E2633/record.md) supplies the comparison that makes this answerable: on the same two arms the loss itself
separates by a paired, work-clustered interval of [−0.209, −0.037] around −0.108 bits per character. So the question is
narrow: **with the same arms, the same split and the same instrument construction, does 3.5× the items turn a correct
ordering into a separation?**

## The arms, the pairs, and what is already known about them

| Arm | Whole-pack bits per character | Loss against | E2633's paired reading |
| --- | --- | --- | --- |
| `E2612-T` (technical mixture) | 2.055447 | — | — |
| `E2612-P` (prose only) | 2.163693 | T −0.108246 | interval [−0.209325, −0.037284], **separates** |
| `E2613-H` (half prose, T's vocabulary) | 2.168218 | T −0.112770 | interval [−0.217418, −0.039099], **separates** |
| | | P −0.004525 | interval [−0.014888, +0.002641], **includes zero** |

One vocabulary (`434debe82f949347`), one split (`adf650435c3a9068`, seed 7), one corpus (`312571f9fa8351a8`), one
budget, three independent initialisations. The two material pairs and the pair the 0.02 band calls equal are the same
three pairs E2629 tested; E2633 re-tested them on the loss and this record tests them on the probes.

## The instruments, and the item sets declared before any scoring

Three instruments, two of which are enlarged and one deliberately not. All item sets are cut with the arms' own
vocabulary, because the runners refuse a checkpoint whose tokenizer is not the one the items were cut with — which is
how two item sets become two measurements.

| Instrument | Items | Digest, fixed now | Relation to E2629 |
| --- | --- | --- | --- |
| `continuation_pmi_easy`, `continuation_pmi_hard` (`scripts/evaluate_continuations.py`) | **745** | `f5ae1effc9a97f27db8dcc82fc3618b9146598f3c64d56beae611511bb479e66` | the same construction at 3.47× the passages |
| `context_pmi_prior_span`, `context_pmi_next_span`, `context_pmi_permuted`, `context_pmi_reversed` (`scripts/evaluate_context_continuations.py`) | **745** | `b48c01eb0e40a65ff064eb32fbb6027a01c5d478cb39a5dd73c65a0c2b5bbbc0` | E2627's same-region instrument, same 3.47× |
| `minimal_pairs_usable` (`scripts/evaluate_minimal_pairs.py`) | 300 of 800 pairs, file digest `2487a4ab4424ce2feb2e7b570a0a99ab099e2a40f8f16f9486686c2d99f07208` | unchanged | the control that does **not** scale: E2629's failure here was an *inversion* at 300 pairs with 24 discordant items to 2, not a shortage of items |

Quotas and the reason for each number: `--registers prose-:400,science-:170,technology-:120,mathematics-:55
--max-per-work 20` over the held-out side's 1,443 documents. The builder refuses to pad a register that cannot supply
its quota ("yielded only N eligible passage(s) of the M asked for"), and this record was written *after* two such
refusals: science- asked for 220 at `max-per-work` 20 and yielded 172, technology- asked for 130 and yielded 120. The
quotas above are the largest that pass, so the item count is what the corpus's held-out works can support rather than
what would have been nice. **Medicine is absent** for the same reason it is absent from the whole-pack loss sample
(K-033): one held-out work cannot supply independent passages, and a register represented by one work is not a register
in a ranking test.

Each command carries its declared digest, and the runner computes the digest of the set it has built and stops **before
loading a checkpoint** if it differs (the gate this campaign added after E2627 fixed a digest by hand).

## Metrics, margins and rules, declared before the run

Per instrument, per pair: **exact McNemar** over the paired per-item verdicts (the same test E2629 used, on the same
rows — the item verdicts are what survives in the committed extracts, K-032), **Holm step-down** across the three pairs
within one instrument at α = 0.05, and the direction (which arm's verdict is the better-loss arm). Across arms:
**Spearman ρ** of the instrument's per-arm mean score against the committed whole-pack bits per character. Reported
beside every rejection: the **number of distinct works** whose items contributed the discordant pairs, because 745
passages drawn from 25 works are not 745 independent observations and E2633 measured exactly how much that matters
(its leave-one-out range crossed zero for a difference the pooled estimate called large).

* **G-a — the shortage was power.** At least one material pair (`T/P` or `T/H`) is separated by Holm in at least one
  continuation instrument, with the better-loss arm favoured, and no inside-the-band pair (`P/H`) separates.
  Consequence: the continuation probe becomes a candidate *arm* instrument — candidate, because a second arm pair at a
  different loss gap is what turns "it worked on the pair we already had" into an instrument.
* **G-b — the shortage was not power.** Every continuation instrument still separates nothing after the enlargement
  (no Holm rejection on any pair). Consequence: 215 items were not the limit; the instrument's resolution between
  independently trained models is where E2626's ρ = 0.918 curve leaves it, and the record says the item count that
  failed rather than implying more would have helped.
* **G-c — the band is contradicted.** The inside-the-band pair `P/H` separates. Consequence: no ranking claim is made
  from this instrument at any item count, and this becomes the second measured contradiction between a probe and the
  loss criterion in the series (E2626's rung pair was the first).
* **G-d — the ordering inverts.** Any continuation instrument's ρ against loss is ≤ 0 at the enlarged set (E2629's were
  +1.0 on three arms). Consequence: the enlargement made the wrong answer sharper, which is the outcome this record
  exists to be able to report honestly.

The minimal-pair instrument is **reported and not decision-bearing** at all four letters: E2629's inversion there is
not an item-count story, and it is kept in the run as the control that shows the enlargement did not change what a
fixed instrument says.

**Descriptive predictions (none decision-bearing).**

* **P1.** `T/P` clears Holm. E2629 sat at p = 0.049 with 215 items; the discordant count scales with items at a fixed
  effect, so 745 items put the same effect at roughly a quarter of that p — under the 0.016667 step.
* **P2.** ρ = +1.000 for both continuation instruments, as in E2629: three arms, correct order, at any item count.
* **P3.** `P/H` stays alone: the band calls it equal and E2633's interval includes zero, so a probe that separated it
  would be measuring something the loss does not.
* **P4.** The digests reproduce: re-running each command on the same snapshot and vocabulary returns the declared
  digest, which is the property the new flag turns into a refusal rather than a hope.
* **P5.** A rejection, if one comes, will not rest on many works: the registers with the fewest held-out works
  (mathematics, 5) contribute the fewest items and the least clustering headroom, so the discordant pairs of a
  separating pair are expected to concentrate in prose and science.

## Known limits, declared in advance

* **Items are not works.** McNemar treats each item as an independent paired observation; the items of one held-out
  book are not independent of each other. E2633 built the clustered reading for the loss
  (`chakaso.training.paired_loss`) and this record does not re-derive one for binary verdicts: it reports the work
  count behind every rejection instead, and a reader should treat a p-value here as **optimistic by an unknown margin**
  rather than as the quantity E2633 printed.
* Three arms give three pairs, one of which is inside the band. A ranking instrument validated on one material pair is
  validated on one material pair.
* One corpus, one split, one tokenizer, one register set with medicine excluded. The passages are archaic English
  prose and technical text, and the phenomena are the probe's own construction.
* **No capability claim, and this is the sentence that matters.** A probe that ranks two models by held-out loss is
  measuring whether the probe tracks loss. It is not evidence that either model is useful, and K-001 is unchanged by
  whatever G-letter this record reports.

## Cost

No training. Three scoring passes (two continuation scripts over three checkpoints each, plus the unchanged minimal-pair
pass) and one analysis pass. E2629 scored 215 items in "minutes each" on this host and E2626 ran the same scripts over
thirteen rungs in tens of minutes, so 745 items × 3 checkpoints × 2 instruments is expected at **10–35 minutes of CPU**
in total, one model loaded at a time (2.5 GB class). Peak memory is one 3,413,760-parameter decoder.

## Reproduction

```bash
cd /d/chakaso
QPYTHON=.venv-nn/Scripts/python.exe   # torch is needed to load the checkpoints; nothing is trained
ARMS=(--checkpoint "E2612-T=data/experiments/E2612/mixture/E2612-T-decoder.pt"
      --checkpoint "E2612-P=data/experiments/E2612/prose/E2612-P-decoder.pt"
      --checkpoint "E2613-H=data/experiments/E2613/half-prose-e2612-vocab/E2613-H-decoder.pt")
SNAP=data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json
VOCAB=experiments/E2612/vocabulary-434debe82f949347.json
QUOTAS=(--registers "prose-:400,science-:170,technology-:120,mathematics-:55" --max-per-work 20)

"$QPYTHON" scripts/evaluate_minimal_pairs.py \
  --pairs configs/probes/grammatical-minimal-pairs-v2.json \
  --expected-pair-set-digest 2487a4ab4424ce2feb2e7b570a0a99ab099e2a40f8f16f9486686c2d99f07208 \
  --control "${ARMS[@]}" --out experiments/E2634/minimal-pairs.json

"$QPYTHON" scripts/evaluate_continuations.py \
  --snapshot "$SNAP" --vocabulary "$VOCAB" --expected-split-fingerprint adf650435c3a9068 \
  "${QUOTAS[@]}" \
  --expected-items-digest f5ae1effc9a97f27db8dcc82fc3618b9146598f3c64d56beae611511bb479e66 \
  --control "${ARMS[@]}" --out experiments/E2634/continuations.json

"$QPYTHON" scripts/evaluate_context_continuations.py \
  --snapshot "$SNAP" --vocabulary "$VOCAB" --expected-split-fingerprint adf650435c3a9068 \
  "${QUOTAS[@]}" \
  --expected-items-digest b48c01eb0e40a65ff064eb32fbb6027a01c5d478cb39a5dd73c65a0c2b5bbbc0 \
  --control "${ARMS[@]}" --out experiments/E2634/context.json

"$QPYTHON" scripts/analyse_arm_pairs.py \
  --arm "E2612-T=experiments/E2629/E2612-T-report.json" \
  --arm "E2612-P=experiments/E2629/E2612-P-report.json" \
  --arm "E2613-H=experiments/E2629/E2613-H-report.json" \
  --reference E2612-T \
  --pairs experiments/E2634/minimal-pairs.json \
  --continuations experiments/E2634/continuations.json \
  --context experiments/E2634/context.json \
  --usable-phenomena auxiliary_agreement,determiner_noun_agreement,reflexive_pronoun_agreement \
  --decision-bearing continuation_pmi_easy,continuation_pmi_hard,context_pmi_prior_span,context_pmi_next_span,context_pmi_permuted,context_pmi_reversed \
  --materiality 0.02 \
  --out experiments/E2634/results.json
```

The analysis reads the arm *reports* (committed extracts of the runs, with the denominators the comparison is checked
against) and the three probe files, and writes every pair's loss gap, discordant counts, exact p-value, Holm step,
whether it rejected and which way the direction fell. Per K-032's rule, the committed probe files carry their per-item
verdicts in the compact digest-checked form, so a reader recomputes every McNemar count in `results.json` rather than
checking the arithmetic that was reported.
