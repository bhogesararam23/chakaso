# E2633 — the held-out loss tested as a paired difference, resampled by work

**Status:** preregistered — written and committed before any arm was scored here.
**Type:** Research experiment, **preregistered**. It measures a statistic on artifacts already on disk; it trains
nothing.
**Tool:** [`scripts/paired_loss_test.py`](../../scripts/paired_loss_test.py), built on
[`chakaso.training.paired_loss`](../../src/chakaso/training/paired_loss.py) and
[`chakaso.training.nn_evaluation.document_losses`](../../src/chakaso/training/nn_evaluation.py).

## Question

Every arm comparison in this project has been read against a band. E2614 measured the seed-to-seed spread of the
reference configuration — 0.00226 bits per character on prose, 0.00337 on the whole pack, 0.01232 on mathematics —
and the campaign then called a one-seed difference *material* above 2.77·*s*. That is a threshold on **model**
variation: it asks whether a difference is bigger than what re-initialising the same protocol produces.

It is not a test of the measurement. Two arms score the *same* held-out documents, so the difference can be paired
document by document, and how well the pooled difference is known — given this text — is a separate quantity with its
own uncertainty. Nobody in this repository has computed it. E2629 made the gap concrete from the other side: its
capability probes could not separate E2612's mixture from its prose-only control at 0.108 bits per character, so the
only evidence that the mixture was better was a number standing beside a threshold.

**The question is therefore narrow and answerable here:** on this 62-work held-out pack, does a paired, work-clustered
test of the held-out loss separate the two arms that the band separates, and does it leave alone the pair the band
calls equal?

## The arms, and the pairs

All four share the corpus `gutenberg-multiregister-v1` (`312571f9fa8351a8`), the whole-work split at seed 7
(`adf650435c3a9068`), the vocabulary `434debe82f949347`, the 3,413,760-parameter rotary decoder and 2,400 updates.
Their whole-pack bits per character are read from the reports committed beside the records that measured them.

| Arm | Whole pack | What it is |
| --- | --- | --- |
| `E2612-T` | 2.05544739454246 | five registers, seed 7 |
| `E2612-P` | 2.163693122795988 | prose only, seed 7 |
| `E2613-H` | 2.1682177974655765 | prose at half the identifiers, seed 7 |
| `E2614-s7` | 2.05544739454246 | the reference configuration, seed 7 (identical artifact to `E2612-T`) |
| `E2614-s11` | 2.0587330335285547 | the same configuration at seed 11 |

* **The material pair.** `E2612-T` against `E2612-P`, 0.108246 apart, and `E2612-T` against `E2613-H`, 0.112770. These
  are the pairs [E2629](../E2629/record.md) handed to four capability probes, which separated neither (Z-d).
* **The pair inside the band.** `E2612-P` against `E2613-H`, 0.004525 apart — below the 0.02 separable margin, so the
  band calls them equal. It is the test's false-positive check: a measurement that "significantly" separates a pair
  the project's own noise floor calls indistinguishable is either anticonservative or telling a different story.
* **The same-configuration pair, descriptive.** `E2614-s7` against `E2614-s11` differ by −0.003286 and differ *only*
  in their initialisation and data order. They are not an experiment about which model is better; they are the
  clearest available demonstration of what this test measures and what it does not.

## The statistic, declared before the run

Both arms score the identical packed held-out side: the test documents of the split, capped at the runs' own
`max_test_tokens` of 250,000 identifiers, counted with the selection vocabulary when the plan named one. The loss is
split into documents from the packer's `segment_ids`, with a block's last target attributed to the next block's first
document — the shift the packed stream implies — and documents are aggregated to **works** by the split's own group.

For work *w*, with *c_w* scored characters, the paired observation is
`d_w = bpc_first(w) − bpc_second(w)`, and the reported quantity is the character-weighted mean
`Σ c_w·d_w / Σ c_w`, which is algebraically the pooled whole-pack difference. The interval is a **cluster bootstrap
over works** with replacement: 2,000 resamples, seed 20261104, two-sided 95 %. The exact sign test over documents is
reported beside it and is labelled the optimistic reading, because it assumes documents are independent and a book
writes its documents in one voice.

Before any number is read, the rebuild is checked against the arms' committed reports: positions scored, characters
scored and the pooled bits per character must reproduce what each record published, to 1e-9. A pack whose denominator
moved is a different measurement and the script refuses it.

## Outcome rules, fixed now

`M` is the material pair `E2612-T`/`E2612-P`, `M2` is `E2612-T`/`E2613-H`, `N` is the inside-the-band pair
`E2612-P`/`E2613-H`. "Clears zero" means the 95 % cluster interval excludes 0.

* **L-a — the test has power here and does not cry wolf.** `M` and `M2` both clear zero, and `N` does not.
  Consequence: a paired, work-clustered interval belongs beside the band in every future arm comparison, stated for
  what it is — evidence about *this held-out text* — and the record says which of the two uncertainties answers which
  question.
* **L-b — the test separates the pair the band calls equal.** `M` clears zero and `N` clears zero too. Consequence:
  within-work clustering is not enough at this scale to make the interval a statement about generalisation, the band
  stays the decision-bearing rule, and the interval is reported as descriptive only.
* **L-c — the test cannot see the gap the band calls material.** Either `M` or `M2` fails to clear zero. Consequence:
  on 62 works even a 0.108-bit difference is not separated from zero by resampling text, the campaign's arm verdicts
  rest on the band and on nothing else, and E2629's negative result is the *mild* version of the problem.
* **L-d — anything else**, with every interval named.

**Descriptive predictions (none decision-bearing).**

* **P1.** L-a. A 0.108-bit gap is 30 times the whole-pack seed deviation, and it is not one book's doing: E2612
  reported the mixture better on all four technical partitions and worse on prose.
* **P2.** The `N` interval straddles zero but sits near it, because −0.0045 is a real difference in mean and only a
  fifth of the band.
* **P3.** `E2614-s7` against `E2614-s11` — a difference the band calls a tie — comes out on whichever side of zero the
  text happens to favour, and its interval is the same width as `N`'s. If it clears zero, that is the honest reason
  the same-configuration pair is in the record.
* **P4.** The pooled bits per character of each arm reproduces its committed report to the stated tolerance, on the
  first attempt.

## Known limits, declared in advance

* **This is an interval over text, not over models.** It answers "is this difference distinguishable from zero on
  samples of held-out works drawn from these 62?" — not "would a re-trained arm differ by the same amount?". The
  second question is E2614's, and its answer is the 0.02 band. Reporting one where the other is asked is the error
  this record exists to avoid making.
* 62 held-out works is the cluster count. A 2,000-draw bootstrap over 62 clusters is coarse at the tails and the
  interval is conditional on *these* works, which one seeded split chose.
* The pack is 250,000 identifiers of the test side, not the whole side; the registers are archaic English prose, four
  of them at a handful of works each.
* The document-to-work attribution uses the split's group; a corpus whose documents carry no group falls back to one
  cluster per document, which would make this interval *narrower*, and is refused here because these arms' corpus
  does have groups.
* Nothing about capability. Loss is not usefulness (K-001), and E2629's probes are still the only attempt to measure
  anything else.

## Cost

No training. Preparation is a snapshot read and a document split (tens of seconds); five checkpoints scored over
713 blocks each, on CPU, at the rate the experiment's held-out evaluations run — measured to take **under five
minutes per arm**, so the whole record is well under half an hour of CPU. It shares the machine with E2630's arms and
is scheduled between them.

## Reproduction

```bash
QPYTHON=.venv-nn/Scripts/python.exe
"$QPYTHON" scripts/paired_loss_test.py \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --vocabulary experiments/E2612/vocabulary-434debe82f949347.json \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --expected-split-fingerprint adf650435c3a9068 \
  --split-seed 7 --context-length 256 --max-test-tokens 250000 \
  --arm "E2612-T=data/experiments/E2612/mixture/E2612-T-decoder.pt" \
  --arm "E2612-P=data/experiments/E2612/prose/E2612-P-decoder.pt" \
  --arm "E2613-H=data/experiments/E2613/half-prose-e2612-vocab/E2613-H-decoder.pt" \
  --arm "E2614-s11=data/experiments/E2614/seed-11/E2614-decoder.pt" \
  --report "E2612-T=experiments/E2629/E2612-T-report.json" \
  --report "E2612-P=experiments/E2629/E2612-P-report.json" \
  --report "E2613-H=experiments/E2629/E2613-H-report.json" \
  --report "E2614-s11=experiments/E2628/E2614-seed-11-control-report.json" \
  --pair E2612-T/E2612-P --pair E2612-T/E2613-H --pair E2612-P/E2613-H \
  --pair E2612-T/E2614-s11 \
  --seed 20261104 --resamples 2000 --out experiments/E2633/results.json
```

`E2614-s7` is the same artifact as `E2612-T` — E2626 proved all 43 tensors identical — so it is not scored twice, and
the same-configuration pair is read as `E2612-T` against `E2614-s11`, whose committed control extract is also the
report this run checks itself against. That fourth pair is descriptive by the rules above: it decides nothing, and it
is here because it is the clearest available demonstration that a paired interval over text and a band over models are
different quantities.

## Results (added 2026-11-04)

**Outcome: L-a.** The paired, work-clustered test separates both material pairs and leaves the inside-the-band pair
alone, and in doing so it measures two things the band never stated: how much of a headline difference one book
carries, and how weak the per-document sign is even when the pooled gap is large.

Machine-readable evidence: [`results.json`](results.json), produced by the command in the Reproduction block. The
pack it scored is the runs' own: 996 blocks, 254,976 positions, 535,282 characters, split `adf650435c3a9068`,
vocabulary `434debe82f949347`, corpus `312571f9fa8351a8`.

### The four readings

| Pair | Published difference | Paired mean | 95 % interval over works | Clears zero | Works favouring first | Sign test p | Leave-one-out range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `E2612-T` / `E2612-P` | −0.108246 | −0.108246 | [−0.209325, −0.037284] | **yes** | 15 of 25 | 0.21218 | [−0.122913, −0.072495] |
| `E2612-T` / `E2613-H` | −0.112770 | −0.112770 | [−0.217418, −0.039099] | **yes** | 15 of 25 | 0.21218 | [−0.128205, −0.076367] |
| `E2612-P` / `E2613-H` | −0.004525 | −0.004525 | [−0.014888, +0.002641] | no | 14 of 25 | 0.34502 | [−0.005292, −0.000645] |
| `E2612-T` / `E2614-s11` | −0.003286 | −0.003286 | [−0.020684, +0.007283] | no | 14 of 25 | 0.34502 | [−0.005166, +0.004016] |

The rules fired in the order the record fixed them: both material pairs clear zero and the pair the band calls equal
does not, which is **L-a** and nothing else. P1 held. P2 held, and nearer than expected: the inside-band interval is
[−0.0149, +0.0026], straddling zero but with 85 % of its mass on the side the point estimate sits. P3 held in form and
surprised in substance — the same-configuration pair's interval (±0.014) is *wider* than the inside-band arm pair's
(±0.009), and its leave-one-out range crosses zero while the arm pair's does not. P4 held exactly: four published
whole-pack numbers reproduced to better than 1e-9 from the checkpoints on disk, and the run was repeated after the
tool gained its leave-one-out block, with the pack, the arms and every interval identical to the last digit.

### What the numbers say that the band could not

**The mixture's 0.108-bit advantage is a weight effect, not a majority effect.** E2612-T is better on 15 of the 25
works and worse on 10 — including the largest prose work in the pack (`prose-pg78424`, 9.0 % of its characters, +0.0407
against it). The exact sign test over the 25 works cannot reject zero at any of these gaps (p = 0.21218 for the 0.108
difference), and over documents the same reading gives the same p, because the sign is 15:10. What makes the pooled
difference large is that the works it wins include the hardest ones: one book, `mathematics-pg302`, is 1.2472 bits per
character better under the mixture, 3.0 % of the pack's characters, and removing it moves the pooled difference from
−0.108246 to −0.072495 — a third of the headline effect is one held-out book.

**It survives that, and that is the useful part.** Dropping any *single* work leaves both material readings negative
and leaves the inside-band reading negative too (its leave-one-out range is [−0.005292, −0.000645]). A pure seed
difference of comparable size does not behave that way: `E2612-T` against `E2614-s11` flips sign when one work is
removed (+0.004016 at its most influential). Two arms that differ in their data move together across the whole pack;
two arms that differ only in their initialisation do not. That is the closest thing to a capability-free discriminator
between "a different model" and "a different draw" this repository has measured, and it is descriptive — this record
did not preregister a rule on it.

**The interval is not the band, and the two disagree in both directions.** The 0.02-bit band is a statement about
re-initialising the protocol; the interval is a statement about re-drawing the held-out works. On 25 works the
mixture's 0.108-bit gap has an interval half-width of 0.086 — eight times the band — while the inside-band 0.0045
difference has a half-width of 0.009, below the band. So the band is *wider* than the text-level uncertainty for small
differences and *narrower* for large ones, which means the two quantities answer different questions and a verdict that
quotes one where the other is asked is wrong in a way neither number shows.

### Deviations from the preregistered plan

1. **The held-out pack was rebuilt by the experiment driver, not by this script.** The preregistered reproduction
   block pointed the tool at a vocabulary and a split fingerprint and had it take the test side in split order. That
   reconstruction was refused by the arms' own committed reports: 256,256 positions and 483,895 characters against the
   254,976 and 535,282 every report prints, and a pooled 3.253976 bits per character for a model whose published
   number is 2.055447. The cause is that `max_test_tokens` caps a **seeded sample** of the test side
   (`_sampled_units`), not its prefix, so the two packs hold different documents. The tool now obtains its pack through
   `chakaso.training.subword_experiment.work_losses`, which pins, splits and packs through the run's own code path,
   and the driver gained `pin_and_split` and `evaluation_sides` so that path is shared rather than copied. The refusal
   is kept in the script as `check_against_reports` and the numbers above are why.
2. **The pack covers 25 works, not the 62 the limits section said.** The split holds out 62 works; a pack capped at
   250,000 identifiers samples 34 documents from 25 of them, and each of those 25 carries at least 2.7 % of its
   characters. Every interval above is over 25 clusters. This weakens the sentence in the preregistration that "62
   held-out works is the cluster count" and is the reason P3's surprise is worth reporting: with 25 clusters the
   honest interval is wide.
3. **The leave-one-out table was added to the tool after the preregistration and before the reported reading.** It is
   descriptive: no outcome letter, margin, seed, resample count or metric depends on it. The declared readings were
   recomputed with it present and are identical to the run without it, to the last digit, so no rule moved.

4. **The pack this test paired is not the pack the register tables are measured over, and it holds no medicine at
   all.** The command's `--pair` differences are computed on the run's whole held-out pack: 34 documents from 25
   works, 535,282 characters, registers science 46.6 %, prose 41.5 %, mathematics 6.0 %, technology 5.9 % — **no
   medicine work is in it**. The partition numbers each record prints beside that figure come from a different
   selection: `partition_packs` samples each register prefix separately at the same `max_test_tokens`, and the
   committed `E2612-T` report scores 510,290 / 101,008 / 539,534 / 538,564 / 547,586 characters across the five
   registers — 2,236,982 in total, 4.2 times the whole-pack denominator. So the interval above bounds the whole-pack
   difference only; O1, the campaign's primary objective, weights a register this test never scored. Filed as
   [K-033](../../docs/agent/KNOWN_ISSUES.md), because every record in the series prints the two kinds of number in one
   table and a reader can reasonably take one for the pooled version of the other.

### What this changes, and what it does not

1. **A paired interval belongs beside every future arm difference, labelled for what it measures.** L-a's consequence
   is exactly that: report the band for model variation and the interval for text variation, and never one in place of
   the other. `scripts/paired_loss_test.py` is the tool and `results.json` is its shape.
2. **E2629's negative result is sharpened, not repaired.** The probes still cannot rank two independently trained arms;
   this record adds that even the loss, tested properly, leans on a handful of hard works, and that the sign of the
   difference across books is 15:10 at a gap the campaign calls large.
3. **K-001 stands.** This is a better measurement of a difference in held-out prediction. It is not a capability, not
   a grounded answer, and not a model that produces useful text.
