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
