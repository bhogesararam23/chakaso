# E2634 — does four times the items let the continuation probe rank the arms that loss ranks?

**Status:** measured — **G-a on the letter of the rule, and the qualification under it is the finding.** Preregistered
at 2026-11-04T03:07:40Z (`2c0261a`) and the three scoring passes ran 76 seconds later, 03:08:56Z → 03:11:29Z. **Type:** Research measurement,
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

The analysis reads the arm *reports* (committed extracts of the runs, whose denominators the comparison checks against)
and the three probe files, and writes every pair's loss gap, discordant counts, exact p-value, Holm step, whether it
rejected and which way the direction fell.

Two of the three probe files are larger than the tracked-file bound, so what the repository carries is the compacted
extract of each — the raw 1.57 MB and 1.10 MB files live in `data/E2634/`, as every generated artifact does — and the
analysis is then re-run from the committed bytes to prove they hold the same verdicts:

```bash
"$QPYTHON" scripts/extract_report.py --report experiments/E2634/continuations.json \
  --out experiments/E2634/continuations-with-outcomes.json      # 1,567,222 → 161,704 bytes
"$QPYTHON" scripts/extract_report.py --report experiments/E2634/context.json \
  --out experiments/E2634/context-with-outcomes.json            # 1,098,085 → 142,769 bytes

# and the analyse_arm_pairs.py command above with --continuations and --context pointed at those two files, which
# reproduces experiments/E2634/results.json byte for byte (13,464 bytes, deviation 3).
```

## Results (added 2026-11-04)

**Outcome: G-a — the shortage was power, for one instrument, and that instrument is one the series has already
disqualified.** `context_pmi_permuted` separates **both** material pairs with Holm's correction applied and the
better-loss arm favoured each time, and no inside-the-band pair separates anywhere. It is the only one of the six
decision-bearing readings that separates anything at 3.47× the items, and its untrained floor is **0.3812** — 0.119
below chance, the reading E2627 refused to use for exactly that reason. Among the readings whose floor is at chance,
the best p-value moved from E2629's 0.049 to **0.031539** against a first Holm step of 0.016667: enlargement helped,
not enough, and the record says what the limit actually is rather than implying another 3.47× would have done it.

### Every instrument, at 745 items

| Instrument | ρ against the loss ordering | material pairs separated | smallest separated gap | largest missed | discordant items on the closest pair |
| --- | --- | --- | --- | --- | --- |
| `continuation_pmi_easy` | +1.000 | 0 of 2 | — | 0.112770 | 19 (15:4), p = 0.019211 |
| `continuation_pmi_hard` | +0.500 | 0 of 2 | — | 0.112770 | 47 (25:22), p = 0.242960 |
| `context_pmi_prior_span` | +0.500 | 0 of 2 | — | 0.112770 | 43 (29:14), p = 0.031539 |
| `context_pmi_next_span` | +1.000 | 0 of 2 | — | 0.112770 | 40 (25:15), p = 0.059584 |
| **`context_pmi_permuted`** | **+1.000** | **2 of 2** | **0.108246** | — | 43 (30:13) p = 0.013718 and 44 (32:12) **p = 0.003658** |
| `context_pmi_reversed` | +1.000 | 0 of 2 | — | 0.112770 | 41 (27:14), p = 0.024093 |

`P/H`, the pair the 0.02 band calls equal (0.004525 bits per character), was separated by **no** instrument — G-c did
not fire — and no instrument's ordering inverted against the criterion, so G-d did not fire either. G-b (nothing
separates at any count) is the reading five of the six instruments gave.

The unchanged control behaved as a control should: `minimal_pairs_usable` reproduced E2629's inversion exactly at the
same 300 pairs, ρ = **−0.866**, and the full 800-pair set ρ = **−1.000**. That instrument's failure is not an item
count, and this record did not pretend it was.

### The floors, which are the finding

Untrained-network accuracy on the same 745 items, per reading — the quantity that decides whether a high score means
anything:

| Reading | untrained floor | E2612-T | E2612-P | E2613-H |
| --- | --- | --- | --- | --- |
| `context_pmi_prior_span` | 0.4671 | 0.9409 | 0.9208 | 0.9248 |
| `context_pmi_next_span` | 0.4591 | 0.9597 | 0.9463 | 0.9423 |
| **`context_pmi_permuted`** | **0.3812** | 0.9664 | 0.9436 | 0.9396 |
| `context_pmi_reversed` | 0.4617 | 0.9597 | 0.9463 | 0.9369 |
| v1 `easy` | 0.4631 | 0.9879 | 0.9758 | 0.9732 |
| v1 `hard` | 0.4671 | 0.9463 | 0.9342 | 0.9423 |

Three things follow from that table, and they are the reason the headline is qualified.

1. **The separating reading is the disqualified one.** A permuted span is scored better by a *random* network only
   38.1 % of the time, not 50 %: permuting identifiers changes their frequency statistics, so the reading measures an
   artifact of the construction alongside whatever the model knows. E2627 established this floor at 215 items
   (**T-a**: only `prior_span` passed) and this record re-establishes it at 745. A ranking that comes from a reading
   with an off-chance floor is a ranking a reader should not spend.
2. **`easy` is saturated, not undersampled.** Its accuracies are 0.9879 / 0.9758 / 0.9732: two arms agree on 98 % of
   745 items, so only 19 items disagree and no Holm step at three pairs can be cleared by 19 coin flips. Adding items
   to an instrument whose arms already agree at 98 % adds almost no discordance — this is the ceiling E2626's rung
   analysis pointed at, measured between arms.
3. **The clean readings are effect-size limited.** `prior_span` disagrees on 43 of 745 items (5.8 %). Scaling p with
   √n predicts a 1.86× improvement from 3.47× the items; E2629's closest 0.049 became 0.0315, a 1.56× improvement —
   roughly what theory says, and still a factor of two short of the 0.016667 step. Clearing it at this discordance
   rate needs on the order of 1,700–2,000 items, and the held-out works will not supply them: the builder refused
   220 science items at 20 per work and yielded 172, and 130 technology and yielded 120. **The corpus, not the budget,
   is the limit.**

### The clustering the record promised

E2633's lesson is that items from one held-out book are not independent observations, so every rejection is reported
with the works behind it. The per-item rows identify a register and a hash, not a work, so the mapping was rebuilt
from the same flags — and the rebuilt set's digest came back **`b48c01eb0e40a65f…`**, equal to the declared one, so the
745 items are the same 745 items that were scored:

| Pair | discordant | works carrying them | largest single work |
| --- | --- | --- | --- |
| `E2612-T` / `E2612-P` (separated, p = 0.013718) | 43 | **24** | `mathematics-pg302`, 5 of 43 (11.6 %) |
| `E2612-T` / `E2613-H` (separated, p = 0.003658) | 44 | **26** | `mathematics-pg302`, 3 of 44 (6.8 %) |
| `E2612-P` / `E2613-H` (inside the band) | 43 | 25 | `science-pg36504`, 5 of 43 |

P5 predicted the discordant items would concentrate in the registers with the most items and leave mathematics
under-represented. It is **wrong in the useful direction**: the separations are spread over 24 and 26 of the works in
the pack, and no single work carries more than 12 % of the disagreement — so unlike the loss difference E2633 found
one mathematics book carrying a third of, these verdict differences are not one book's artifact. That strengthens the
statistics and does not touch the floor problem: a spread-out bias in an off-chance reading is still a bias.

### The predictions

* **P1 — held.** `T/P` cleared Holm: p = 0.013718 against the 0.016667 first step, exactly the direction predicted
  from E2629's 0.049 at a third of the items.
* **P2 — half held.** ρ = +1.000 for three of the six readings (`easy`, `next_span`, `reversed`, and `permuted`);
  `hard` and `prior_span` came out +0.500, i.e. one of the three arms sits out of the loss order — a *ranking* failure
  at 745 items that E2629 at 215 had not shown, and the clearest sign that more items do not monotonically improve an
  instrument.
* **P3 — held.** `P/H` stayed alone under every instrument.
* **P4 — held.** Both declared digests reproduced: `f5ae1effc9a97f27…` and `b48c01eb0e40a65f…` are what the runs wrote,
  and the rebuild for the work mapping returned the second one to the character. The declared-digest gate never fired,
  which is the property: it is a refusal that stayed silent because there was nothing to refuse.
* **P5 — failed,** as recorded above: no concentration in one or two works.

### What this changes about the instruments

E2629's **Z-d** ("nothing ranked it") now has a measured reason attached to it, and it is two reasons rather than one:
for `easy` the arms agree on 98 % of items, so the instrument has nothing to resolve; for the clean same-region and
hard readings the disagreement is ~6 % of items, which no item count this corpus supports can separate under a
multiple-comparison correction. The only reading that did separate is the one with an off-chance floor.

**Consequence, stated narrowly.** Neither continuation instrument becomes an arm-ranking instrument on this evidence.
What the campaign gets from E2634 is the number that bounds the question: **at this corpus's held-out works, a probe
that agrees with loss on 94–99 % of items cannot separate two arms that differ by 0.108 bits per character.** The next
unit is therefore not a bigger item set. It is one of: (a) a *work-clustered* statistic for binary verdicts — the
instrument E2633 built for loss, ported to paired verdicts, which changes what "separated" means rather than what was
sampled; or (b) accepting E2626's result as the boundary — a probe tracks loss along one training curve and does not
rank independent initialisations — and spending the next measurement on the model instead.

### Deviations

1. **The analysis script prints its own outcome letters, and they are not this record's.** `analyse_arm_pairs.py`
   reports `Z-a` ("context_pmi_permuted ranks every material pair in the criterion's direction") and per-instrument
   `S-a`/`S-b`/`S-c` verdicts inherited from E2626's and E2629's rule space. This record's outcome is **G-a** under
   the rules committed before the run, and the mapping is one line: the tool's `Z-a` is the event G-a names. The tool's
   rules were written for another record and were not re-declared here; where they differ, the record governs. The
   follow-up is for the runner to take its outcome vocabulary from the record instead of embedding one.
2. **The per-item rows do not carry the work**, so the clustering table above was computed by rebuilding the item set
   from the same flags — digest-verified to be the same set. A future record should not have to rebuild anything to
   state a clustering claim: the fix is work identity in the row, and it is not in this commit because the row layout
   is the one K-032's compact form and three measured records already cite.
3. **The two enlarged probe files exceed the 1 MiB tracked-file bound** (1.57 MB and 1.10 MB), so the committed
   evidence is the compacted extract of each — `continuations-with-outcomes.json` at 161,704 bytes and
   `context-with-outcomes.json` at 142,769, each carrying every per-item verdict plus the digest of the array it
   replaced (K-032). The analysis was re-run pointing at the extracts and produced a `results.json` **byte-identical**
   to the one the raw files gave (13,464 bytes), so what is committed recomputes what is claimed. `minimal-pairs.json`
   fits at 446,764 bytes and is committed as the runner wrote it.
4. **The cost block over-estimated by an order of magnitude again, in the same direction E2632 did.** Declared
   10–35 minutes; measured **2 m 33 s** end to end (minimal pairs 11 s, continuations 61 s, same-region 81 s,
   analysis under a second) with one model loaded at a time. 745 items over four models is a small amount of compute;
   the records' fear of scoring passes is not supported by the measurement.

