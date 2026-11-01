# Does E2612's whole-work split still leak? Near-duplicate overlap, measured

**Status:** Research — a measurement of one split of one corpus. It says nothing about any model.
**Date:** 2026-11-01. **Tool:** [`scripts/measure_split_overlap.py`](../../../scripts/measure_split_overlap.py) over
[`chakaso.training.overlap`](../../../src/chakaso/training/overlap.py) (seven offline tests that build each failure on
purpose: a reissued text with a new header, a partial quotation, shared vocabulary without shared windows).

## The question

[ADR-0042](../../decisions/ADR-0042-held-out-means-held-out-works.md) fixed the first leak: chapters of one book on both
sides of a split (K-023: 100 % of E2608's test characters came from books also in training). A whole-work split puts no
*work* on two sides. It cannot see two *works* that are the same text — two editions of one book, an anthology holding
a novel that is also its own catalogue entry, a translation sharing a source's passages. The split's own duplicate
screen is an exact document digest, which an extra preface defeats. So the second leak is open until someone measures
it, and E2612's 62 held-out works were never checked.

## Method

Reproduce the exact split E2612 used (snapshot fingerprint `312571f9fa8351a8`; whole-work; seed 7; 15 % / 15 %; split
fingerprint `adf650435c3a9068`, which the script's split reproduces), join each held-out work's documents, cut every
text into overlapping **word n-grams** (lower-cased `\w+` runs), hash each to 64 bits (BLAKE2b, so identical in every
process), and report the share of a held-out work's distinct n-grams that occur anywhere in the **entire** training
side (312 works, 17,754,263 distinct 8-gram windows). No threshold is chosen here; the shares are reported.

```bash
python scripts/measure_split_overlap.py --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --split-seed 7 --whole-works --side test --n 8
```

About 2 minutes and a few hundred MB on one core.

## Result

| Held-out side | n | Works | Windows | Also in training | Share | Largest single work | Works above 1 % | Above 10 % |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| test | 8 | 62 | 3,948,396 | 8,568 | **0.217 %** | `prose-pg8210`, 2.27 % (2,238 of 98,398) | 5 | **0** |
| test | 16 | 62 | 3,959,137 | 6,189 | **0.156 %** | `prose-pg8210`, 1.80 % | 4 | **0** |
| validation | 8 | 56 | 3,975,028 | 5,644 | **0.142 %** | `science-pg52923`, 3.00 % (3,275 of 109,265) | 1 | **0** |

Further detail for the test side at n = 8: the median work shares 0.052 %; the single most-overlapping work accounts
for 26.1 % of all shared windows; excluding the three most-overlapping works the pooled share is 0.149 %.

**Read plainly:** no held-out work shares more than 3 % of its 8-word windows with training, none comes near 10 %, and
the pooled share is about one window in 640. Doubling the window to 16 words removes only about 28 % of the shared
windows, which says the overlap that exists is mostly **long repeated passages** — the signature of a quotation, a
shared preface or a reprinted passage — and not of coincidental phrases, which would vanish at n = 16.

## Interpretation, and what it does not establish

* **The K-023 failure does not recur here.** There it was 100 % of test characters; here it is a fraction of a percent,
  and the largest single work is under 3 %.
* **It is not zero**, and this measurement does not bound how much it flatters a held-out loss. A shared window is
  eight words; an isolated one covers eight words and a run of *k* consecutive windows covers *k* + 7, so the share of
  *words* touched is somewhere between the window share and about eight times it. Whether a model of this size
  memorises such passages well enough to lower bits per character measurably was not tested.
* **E2612's arm comparison is unaffected in kind**: both arms scored the same test text, so a shared bias cancels in
  the difference. What this bears on is the *absolute* held-out numbers, which stay floors for other reasons already
  recorded.
* **It sees exact word-level repetition only.** Translations, paraphrase, OCR variants and re-typeset text with
  changed spellings are invisible to it. A clean result here is not a proof of independence.
* **The training side is the whole 312-work split, not the 20M-identifier sample each arm actually trained on.** The
  overlap with what a given arm *saw* is at most this, and for the prose-only arm, smaller on the technical works.
* **The scored pack is a sample (250,000 identifiers per partition) of these works**; the overlap was computed on the
  whole held-out works, so a scored sample could be cleaner or dirtier than the population by chance. Per-work shares
  are in the output for anyone who wants to weight them.

## What changes

* A reusable instrument now exists for the contamination check the data review required before any new corpus (item 7
  of its preregistration list: exact-digest and n-gram overlap against the Gutenberg test side). `us-pd-press-v1` should
  be measured with it before use, in both directions.
* [K-023](../../agent/KNOWN_ISSUES.md) gains a dated line: the second leak class was measured on E2612 and is small.
* No code in the training path changes, and no recorded number is restated.
