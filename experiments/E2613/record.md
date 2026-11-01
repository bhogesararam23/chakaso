# E2613 — exposure control: does the mixture cost prose, or does prose dose?

**Status:** completed 2026-11-01 — preregistered in `536649f`; the first launch deviated (see "Deviation 1") and is
reported as run R1 only; the valid arm ran after the cause was fixed. **Outcome: H8-b** — the prose cost belongs to the
mixture, with the two-pass confound this record declared still standing. One seed.
**Type:** Research experiment, **preregistered**
**Unit:** T3 of [ACTIVE_TASK](../../docs/agent/ACTIVE_TASK.md); tests hypothesis
[H8](../../docs/research/hypotheses.md).

**Question.** [E2612](../E2612/record.md) found that the prose-plus-technical mixture (arm T) scored its prose
partition **0.03026** bits per character worse than the prose control (arm P), 1.5× the 0.02 band, and declared in
advance that the packing cap made T's prose exposure smaller. Is that prose cost the cost of *having technical text
in the mix*, or only the cost of *seeing less prose*?

This is a data-dose question about a 3.4M-parameter decoder at 19.66M positions. It is not a capability
question, and nothing here bears on K-001 or Level 3.

## What is held fixed, and what is not

Everything that E2612's two arms held equal is held equal again, by reusing the same inputs rather than
re-deriving them:

| Condition | Value | How it is held |
| --- | --- | --- |
| Corpus | `gutenberg-multiregister-v1` 1.0.0, fingerprint `312571f9fa8351a8` | same pinned snapshot, `--expected-corpus-fingerprint` |
| Split | whole-work, seed 7, 312 / 56 / 62 works, zero on more than one side | same seed and `--whole-work-splits` |
| Tokenizer | `434debe82f949347`, 4,096 identifiers, `min_frequency` 3 | **the E2612 cache, read not re-fitted** (see the note below) |
| Model | rotary decoder, 3,413,760 parameters, context 256 | same flags |
| Optimisation | AdamW 6e-4, weight decay 0.01, 200 warmup, cosine to 0.1, batch 32 × 1, **2,400 steps**, seed 7 | same flags |
| Evaluation packs | 250,000 identifiers per partition, seeded | same caps and seed |

**Declared difference, the only one:** arm **H** trains on the prose works alone, capped at **10,000,000** training
identifiers (`--max-training-tokens 10000000`) instead of 20,000,000.

| Arm | Training works | Cap | Unique prose identifiers seen | Technical identifiers seen |
| --- | --- | --- | --- | --- |
| P (E2612) | prose only | 20M | ~20M (≈80 % of the ~25M prose pool) | 0 |
| T (E2612) | all five registers | 20M | ~10M (≈42 % of a 50.51 %-prose pool) | ~10M |
| **H (this record)** | prose only | **10M** | **~10M** | 0 |

Arm H has the prose *content* that T had — about 10M unique identifiers drawn by the same seeded shuffle — and
none of the technical text. If H scores T's prose, the extra loss in E2612 belongs to prose dose; if H scores
better, technical text displaced prose.

**The tokenizer note.** The vocabulary is read from the E2612 cache, not re-fitted. The merge loop was fixed
after E2612 (K-024) and an exact refit of the same sample now yields `3abe4df4b123d4c5`, not `434debe82f949347`;
refitting would put arm H on a different vocabulary from the two arms it is compared against and make every
difference uninterpretable. Reading the cache is the only way to keep the comparison valid, and it means this
record is reproducible only from E2612's git-ignored cache, as E2612 already is.

## A confound this design cannot remove, declared before the run

At 2,400 steps × 8,192 positions = 19,660,800 positions, arm H makes **about two passes** over its ~10M-identifier
pool; arm T made about one pass over each of its two ~10M-identifier halves. "Equal unique prose" is therefore
not "equal prose positions seen": H sees roughly twice as many prose *positions* as T, by repetition. Muennighoff et
al. (arXiv:2305.16264) report that a few epochs over repeated data are close in value to fresh data, but that is a
claim about their scale, and this record does not assume it. A follow-up arm at 1,200 steps (one pass, half the
compute and a different schedule length) would separate pass count from composition; it is **not** part of this
record and is not run unless this one's result makes the question live.

## Metrics and declared bands

Held-out bits per character on each of the five register partitions and the whole pack, from the report's
`test_partitions` blocks, weighted by `metrics.characters_scored` where an aggregate is formed (K-027). The
comparison is on the **prose partition**. The 0.02-bit materiality band is inherited from E2611/E2612 and is not a
measured noise floor (E2609 was never run; see T4). A difference is *outside* the band at 0.02 or more and inside
it below 0.02, as in E2612; a difference inside the band is reported as indistinguishable, not as a direction.

E2612's prose values, from its committed [`results.json`](../E2612/results.json): **P 1.80926**, **T 1.83952**.
Write **X** for arm H's prose partition. Four outcomes, mutually exclusive and exhaustive on the pair (X − T,
X − P):

* **H8-a — the cost is dose.** |X − T| < 0.02 **and** X − P ≥ 0.02. H sits with T and clear of P: having
  technical text in the mix is not what raised T's prose loss; seeing less prose was.
* **H8-b — the cost is the mixture.** X ≤ T − 0.02. At equal unique prose, an arm with no technical text is
  materially better on prose than T, so technical text displaces something prose needs. H8 as worded in
  `hypotheses.md` is *falsified* by this outcome.
* **H8-c — technical text helps prose.** X ≥ T + 0.02. Reported as positive transfer from the technical
  registers; there is then no prose cost of mixing to explain.
* **H8-d — unresolved.** |X − T| < 0.02 **and** X − P < 0.02. H is inside the band of both E2612 arms, which
  differ by 0.03026; one seed cannot separate them. The record says so and picks no story.

## Decision rule, declared before the run

1. **H8-a** — E2612's prose regression is a budget artefact. The next unit is the prose-share sweep (H9, T8) with
   the mixture recipe unchanged, and the mixture is not treated as having an intrinsic prose cost.
2. **H8-b** — the mixture itself costs prose at this scale. The sweep (H9) is run to find the share that trades
   least, and E2612's decision rule 2 stands as written.
3. **H8-c** — the mixture's prose cost in E2612 was exposure, and the technical text offset part of it; the sweep
   is still the next unit, with prose and technical shares treated as complementary rather than competing.
4. **H8-d** — nothing is concluded about the mixture's prose cost, and the next unit is T4, the decoder seed
   spread, before another data arm is spent.
5. **Whatever lands, the seed spread (T4) is the measurement that gives these bands meaning.** One seed per arm is
   a single-draw estimate, and a gap near the band edge is not decided by this record.

No band, metric, budget or flag is changed after a result is read. A re-run with a different budget is a new
record.

## Deviation 1 — the first launch refitted the vocabulary (disclosed 2026-11-01, before any result was read)

The arm was first launched at `536649f` (2026-11-01 10:27) into `data/experiments/E2613/half-prose`. Within four
minutes a second file, with vocabulary checksum `2f747d1fbf52b229…`, appeared beside E2612's
(`2dd2f5273e1fada3…`) in the shared tokenizer directory: the run had **missed E2612's cache and fitted a different
vocabulary**, so it does not hold the tokenizer fixed and is not the arm this record preregisters.

*Cause (mine).* The tokenizer cache is keyed on the split fingerprint, and that fingerprint hashed the list of
excluded duplicates. `7704fe2` fixed K-025 by making the whole-work split report its three duplicates, which moved
E2612's split fingerprint and with it the cache key. Nothing in the preregistration or the launch checked that the
cache would hit; the record said "read, not re-fitted" without verifying it before starting. The fingerprint now
covers the side assignment only (`b9d7b1a`), and the E2612 split reproduces its recorded `adf650435c3a9068` from the
pinned snapshot.

*Consequence.* The first launch's report, when it completes, is **run R1: arm H under a refitted vocabulary**. It
differs from the preregistered arm in the tokenizer, so it is not compared against E2612's arms for H8 and no
outcome letter is assigned from it. It is reported in the Results section as an additional, differently-conditioned
observation, with the vocabulary checksums beside it. It cannot be stopped from this session (the attempt to end the
process was refused), so it runs to completion.

*Correction.* The valid arm is launched after `b9d7b1a` into a new directory, `half-prose-e2612-vocab`, and must report
`reused_from_cache: true` with tokenizer fingerprint `434debe82f949347` and split fingerprint `adf650435c3a9068` before
its numbers are read; if it does not, it is also a deviation. **No band, outcome, decision rule or flag other than the
work directory changes**, and none is chosen with any result of R1 in view: R1 has no result on disk when this section
is committed.

## Results — run R1, the first launch (not the preregistered arm)

Disclosed here **before the valid arm has finished**, so that nothing in the preregistered rules above can be said to
have been chosen with R1 in view: none of them was changed after R1 completed, and R1 is not assigned an outcome
letter. R1 finished on 2026-11-01 (12:10) into `data/experiments/E2613/half-prose`; its report file digest is
`face74161f8636c33caf9501d11774b7aea6ae0d17d47dd2424fa4569c36b861` (a file hash, with the properties K-028 records).

What it was: arm H's data and budget, trained under a **refitted** vocabulary. Measured conditions:

| Condition | R1 | The preregistered arm requires |
| --- | --- | --- |
| Vocabulary fingerprint | `face1a050155274f` (refitted) | `434debe82f949347` |
| Split fingerprint | `59d54432fc87b0c1` (includes the 3 reported duplicates) | `adf650435c3a9068` |
| Parameters / steps / completion | 3,413,760 / 2,400 / `budget_ceiling` | the same |
| Training identifiers packed (declared tolerance 9.5M–10.5M) | **10,007,299** — within | ~10M |
| Training documents packed | 1,383 | — |
| Wall clock / peak RSS | 5,732 s / 2,829 MB (machine shared with other work throughout) | observed, not a condition |

Bits per character on the held-out partitions, with E2612's two arms beside them **for orientation only** — they were
scored under a different vocabulary, so the differences are *not* a comparison:

| Partition | E2612 P (prose, 20M) | E2612 T (mixture, 20M) | R1 (prose, 10M, refitted vocabulary) |
| --- | ---: | ---: | ---: |
| prose | 1.80926 | 1.83952 | 1.81115 |
| science | 2.05867 | 1.94862 | 2.06098 |
| technology | 2.18647 | 1.90273 | 2.18680 |
| medicine | 1.99522 | 1.83814 | 1.99578 |
| mathematics | 3.04361 | 2.57777 | 3.11763 |
| whole test pack | 2.16369 | 2.05545 | 2.17231 |

*What this can and cannot say.* R1's technical partitions sit where a prose-only model's would, which is a sanity check
that it trained on prose alone. Its prose partition is near E2612's P and not near T. That is suggestive and nothing
more: a different vocabulary is a different model input, the one-seed noise floor is still unmeasured (E2614), and the
preregistered question is about the E2612 vocabulary. **R1 does not decide H8 either way.**

## Results — the preregistered arm

**Outcome: H8-b — the cost is the mixture, not the dose.** Arm H ran to the declared 2,400-step ceiling on
2026-11-01 (12:10–13:15, 3,853 s of training on a shared host) into `data/experiments/E2613/half-prose-e2612-vocab`.
This section was written after the report existed; the declarations above are unchanged. Machine-readable aggregate:
[`results.json`](results.json), rebuilt by `scripts/compare_arms.py` from the three reports on disk.

### Matched conditions were verified before any number was read

`compare_arms` reports **all conditions matched** across P, T and H: corpus `312571f9fa8351a8`, split
`adf650435c3a9068` (the fingerprint the fix of `b9d7b1a` restored), vocabulary `434debe82f949347` — `reused_from_cache:
true`, so no vocabulary was fitted — 3,413,760 parameters, seed 7, 2,400 steps, cosine, effective batch 32, learning
rate 6e-4, and identical scored tokens and characters on every partition. The only differences are the declared ones:

| Field | P | T | H |
| --- | --- | --- | --- |
| `training_group_prefixes` | prose | none | prose |
| `max_training_tokens` | 20,000,000 | 20,000,000 | 10,000,000 |
| training identifiers packed | 20,000,402 | 20,000,103 | **10,001,764** (inside the declared 9.5M–10.5M) |

### Held-out bits per character (lower is better)

| Partition | P (prose, 20M) | T (mixture, 20M) | **H (prose, 10M)** | H − T | H − P |
| --- | ---: | ---: | ---: | ---: | ---: |
| **prose** | 1.80926 | 1.83952 | **1.80621** | **−0.03331** | −0.00305 |
| science | 2.05867 | 1.94862 | 2.05934 | +0.11071 | +0.00067 |
| technology | 2.18647 | 1.90273 | 2.18996 | +0.28723 | +0.00349 |
| medicine | 1.99522 | 1.83814 | 2.00057 | +0.16243 | +0.00535 |
| mathematics | 3.04361 | 2.57777 | 3.10860 | +0.53083 | +0.06499 |
| whole test pack | 2.16369 | 2.05545 | 2.16822 | +0.11277 | +0.00453 |

Technical aggregate, weighted by scored characters (K-027): P 2.39222, T 2.11638, **H 2.41341**.

### Verdict against the declared bands

X = 1.80621. **X − T = −0.03331**, which is at least 0.02 below T, so the outcome is **H8-b**: at equal unique prose, an
arm with no technical text scored the prose partition materially better than the mixture arm, and H8 as worded in
`hypotheses.md` is **falsified**. X − P = −0.00305 is inside the band: halving the unique prose (20M → 10M) cost the
prose-only arm nothing measurable. H's technical partitions sit with P's, as a prose-only arm's should (a sanity check
on the arm, not a finding).

By the decision rule declared before the run: **rule 2.** The mixture itself costs prose at this scale; the prose-share
sweep (H9) is the next data unit, and E2612's decision rule 2 stands as written.

### What this does and does not establish

* **The declared confound did not go away, and it matters here.** H made about two passes over its ~10M unique prose
  identifiers; T made one pass over each of its two ~10M halves. So H saw roughly twice as many prose *positions* as T.
  H ≈ P despite having half of P's *unique* prose says repetition cost nothing at this budget, which points at prose
  positions seen, not unique prose, as what the prose partition tracks. The result is therefore best read as: **a fixed
  step budget spent partly on technical text costs prose, and the cost is about what spending those positions on prose
  would have bought** — not as a finding that technical text actively harms prose representations. The 1,200-step arm
  (one pass, half the compute) that would separate pass count from composition was declared as a follow-up "if this
  result makes the question live"; it now does, and it is **not run**.
* **One seed, and the band is unmeasured.** −0.03331 is 1.67 × the inherited 0.02 band; E2614 will say whether 0.02 is
  above noise (rule R3 there would retire it and re-read this verdict against the replacement). Until then this verdict
  is a single-draw result against a convention.
* The technical-aggregate difference between H and P (+0.02120) sits just outside the band in the other direction, with
  mathematics (+0.06499) carrying it; H's mathematics partition is the one place the halved data shows. It is reported,
  not interpreted.
* **Annotation after [E2614](../E2614/record.md) (same day; the numbers above are unedited).** The band was measured on the mixture configuration: prose s = 0.00226, so H8-b's −0.03331 is 1.86× the upper end of the one-seed band (2.77·s<sub>hi</sub> = 0.01795) and stands; −0.00305 is 0.17× of it and stands. The prose-only arms' own spread is not measured. **The remark above that mathematics (+0.06499) carries the H-against-P gap is withdrawn:** mathematics s = 0.01232 gives a one-seed band of 0.0341 (0.0981 at the upper end), so +0.06499 cannot be told from noise.
* Generation (`held_out` prompts): 5 samples, 0 looping, 0 periodic, 1 refused; mean bigram repetition 0.5874. These are
  repetition statistics, not quality (K-020, E2612).

### Artifacts

| Artifact | SHA-256 of the file |
| --- | --- |
| Report `data/experiments/E2613/half-prose-e2612-vocab/report.json` | `051c943f5f30b2dccb0d75515adb5ae786c59e4b4b6a5a6637be974986ad4070` |
| Checkpoint `E2613-H-decoder.pt` | `c4152994878320932f6636cb034ace250f24101a28724f30a824932ca1dadf7d` |
| Aggregate, committed [`results.json`](results.json) | rebuilt from the three reports; its H protocol digest is `e18748b9c0b9557b14102f851232fb9442ddb4187c93f82c02fdd7f1646b1305` |

As K-028 records, the report-file digest embeds timings and will not reproduce; the protocol digest is the part a re-run
can reproduce exactly. Run R1 (above) is not part of this verdict.

## Generation block

Run with `--sample-prompts held_out`, which takes test-side document openings, not the repository's own
headings (K-026). The generation numbers describe repetition statistics, which E2612 showed can move opposite to
held-out loss; they are reported, not interpreted as quality.

## Known limits, declared in advance

* One seed, one arm. The comparison is against E2612's two single-seed arms, run on 2026-10-30 on the same
  machine under the same code *except* the merge-loop fix (`cfdcd1a`), which changes tokenizer fitting only. The
  loaded vocabulary and its encodings are read from the cache E2612's arms used, not re-derived.
* Between E2612 and this run the held-out scorer changed (`2a5a5c3`: it can now apply the document mask). The
  default is off and this run does not enable masking, so the numbers are computed exactly as E2612's were.
* The prose pool is roughly 25M identifiers; "about 10M" is the seeded cap, and the report states the exact count
  packed. If the packed count is outside 9.5M–10.5M, the record says so before the result is read.
* Public-domain English from 1800s–early-1900s books; no transfer claim to modern text.

## Reproduction

```bash
chakaso experiment model --work-dir data/experiments/E2613/half-prose-e2612-vocab --experiment E2613-H \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --model-family decoder --vocab-size 4096 --context-length 256 --d-model 256 --num-layers 4 \
  --num-heads 8 --feedforward-dim 640 --batch-size 32 --grad-accum-steps 1 \
  --learning-rate 0.0006 --weight-decay 0.01 --warmup-steps 200 --schedule cosine --decay-to 0.1 \
  --max-steps 2400 --eval-every 400 --patience 99 --seed 7 \
  --whole-work-splits --shuffle-training-units --training-group-prefix prose- \
  --max-training-tokens 10000000 --max-validation-tokens 250000 --max-test-tokens 250000 \
  --tokenizer-sample-characters 12000000 --tokenizer-cache-dir data/experiments/E2612/tokenizer \
  --test-partition prose- --test-partition science- --test-partition technology- \
  --test-partition medicine- --test-partition mathematics- \
  --sample-prompts held_out \
  --no-character-baseline --write-report data/experiments/E2613/half-prose-e2612-vocab/report.json
```
