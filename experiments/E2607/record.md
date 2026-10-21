# E2607 — Does the advantage survive on text that is not Chakaso's own?

- Status: running (preregistered before its runs; three seeds, not five — the reason is measured, below)
- Date: 2026-10-21
- Hypothesis: not one of H1–H7. This is the generalization step that E2605's and E2606's own limitation
  sections name: a model measured on the project's documentation has never been tested on text it has no
  relationship to.
- Owner: the project
- Git commit: the record is committed before the runs. The commit that executes them is named in Results
  when they exist.
- Config hash: the resolved configuration of each run is written by that run into
  `data/experiments/E2607/E2607/seed-<n>/report.json`; the aggregate carries each digest.

## Motivation

E2605 measured 1.8657 bits per character against 3.1343 for the character model, and E2606 will say how much
of that is the seed. Both numbers come from one corpus: this repository's own 780,223 characters of
markdown — one author, one domain, technical prose about the project itself, with the tokenizer learning
`docs/` and `##` shapes that recur hundreds of times. On that data a subword model has an easy job, and an
easy job on your own writing is not evidence about language.

The question this experiment answers is narrow and prior to any capacity question: **does the same
architecture, with the same budget and the same evaluation code, still beat the character model and both
trivial baselines when the text is somebody else's?** If it does not, the E2605 result is about this
repository's prose and must be described that way everywhere it appears. If it does, the comparison has
survived its first independent test and the next question — capacity — becomes worth asking.

## Dataset, licence and provenance

| Field | Value |
| --- | --- |
| name / version | `gutenberg-english-prose-v1` / `1.0.0` |
| sources | pg11 *Alice's Adventures in Wonderland* (Lewis Carroll); pg84 *Frankenstein; or, the modern prometheus* (Mary Wollstonecraft Shelley); pg6130 *The Iliad* (Homer; the file's own header names no translator, and none is claimed) |
| URLs | `https://www.gutenberg.org/cache/epub/11/pg11.txt`, `…/84/pg84.txt`, `…/6130/pg6130.txt` |
| retrieved | 2026-10-21, over TLS with certificate verification (this machine's Python trust store rejected the chain; `curl` and a `certifi` bundle both verified, and verification was not disabled) |
| raw byte sizes | 174,311 / 448,885 / 1,161,373 |
| raw byte digests (SHA-256) | pg11 `01b38ea4c710a84b…`, pg84 `7810cd483cffcf2c…`, pg6130 `ca4f23115b384b02…` — the same values `sha256sum` reports for the downloaded files |
| licence | Public domain in the United States, distributed by Project Gutenberg under the Project Gutenberg License, which permits copying, giving away and re-use. Each file's own preamble states that users outside the United States must check local law before use; that sentence is part of the recorded licence, not a footnote. |
| redistribution policy | **The corpus bytes are not committed.** They live under `data/`, git-ignored, per [ADR-0020](../../docs/decisions/ADR-0020-experiment-artifacts.md). What is committed is `corpus-manifest.json` (105 document digests, no text) and this acquisition record, so the corpus can be rebuilt from the URLs and verified against the digests. |
| documents / characters after ingestion | 105 documents, 1,660,274 characters |
| corpus fingerprint | `931b07a635ca5a0c` |
| snapshot checksum | `776afa22a2d54f49a2ff9d25e276c10b…` at `data/experiments/E2607/gutenberg-english-prose-v1-snapshot.json` |

### How the documents were made, and why that is not a leak

`chakaso.training.source_corpus` does two things and nothing else:

1. **Cuts the source's own wrappers.** Everything between the `*** START OF THE PROJECT GUTENBERG EBOOK … ***`
   and `*** END OF … ***` lines is the corpus; a file that does not carry both markers in order is refused.
   Gutenberg's licence preamble and its "END" disclaimer are therefore excluded by a stated rule rather than
   by hand-editing, which means nobody can re-create this corpus by pasting text into an editor.
2. **Groups whole paragraphs up to 16,384 characters into a document.** Paragraphs are blank-line separated;
   a paragraph longer than the bound becomes a document on its own rather than being cut, because splitting
   mid-paragraph would put the two halves of one sentence on opposite sides of the split. Chunking is a pure
   function of the text, so the same bytes always produce the same 105 documents in the same order.

No other normalization is applied except the manifest layer's line-ending normalization. Nothing was
lower-cased, de-accented, deduplicated across sources, or re-ordered.

**Contamination check performed:** the three texts are 19th-century and classical prose, and the repository's
corpus is 21st-century technical markdown; no document in this corpus shares a path, a digest or a sentence
with `chakaso-repo-text-v1`. This is asserted as a *design* property of the two corpora, not measured as a
n-gram overlap study — no overlap metric was computed, and none is claimed.

## Split

| Field | Value |
| --- | --- |
| function | `chakaso.training.data.split_text_units`, whole documents, deduplicated by content first |
| split seed | **7 for every run** — the same rule E2606 declares, chosen here before any run produced a number |
| fractions | validation 0.15, test 0.15 |
| sides, computed before any run | **73 train / 16 validation / 16 test** documents, 0 duplicates dropped; 1,160,177 / 244,332 / 255,765 characters |
| split fingerprint | `93be3cc6f9e80423` — every run must report it, and the study refuses to aggregate runs that do not agree |
| leakage protection | tokenizer fitted on the **train** documents only; the character baseline's vocabulary likewise; the test side is scored after selection and never selected from |

## Model and configuration — unchanged from E2605 and E2606 on purpose

Every number below is the one E2605 used. The corpus is the only thing that differs, because a comparison
that changes two things at once answers neither question.

```text
architecture   vocab_size 1024, context_length 128, d_model 128, num_layers 3, num_heads 4,
               feedforward_dim 384, dropout 0.0, tie_embeddings true, initialization_std 0.02
optimizer      learning_rate 6e-4, weight_decay 0.01, warmup_steps 60, gradient_clip_norm 1.0
budget         batch_size 32, max_steps 2400, eval_every 60, patience 8, min_delta 0.001 nats
tokenizer      byte-pair subword, target 1024, min_frequency 3, fitted on the train side only
character      context 16 characters, embed 32, hidden 64, batch 256 windows, 4000 steps, lr 2e-2
baselines      uniform (log vocab) and add-alpha unigram (alpha = 1) on the same test positions
seed           the declared study seed (initialization and batch plan only)
```

**The budget is a real limitation here, and it is stated rather than hidden.** 2,400 optimizer steps over a
1.66 MB corpus is fewer epochs over the data than 2,400 steps over 0.78 MB — the model sees proportionally
less of the larger text. Keeping the budget fixed was the deliberate choice: the question is whether the
*same experiment* transfers, and re-tuning the budget per corpus would make the two records
non-comparable. If this study finds a smaller gap than E2606's, "the budget, not the architecture" is one of
the explanations, and the follow-up names the test that would separate them.

## Seeds and number of runs — declared, with the reason measured

**Three seeds: 7, 11, 13.**

Five was the target, and it is not what this session can afford honestly. The cost was measured before this
record was committed, on the runs of E2606 that had already completed on this machine: one full-protocol run
of the internal corpus took 513.6 s of Transformer training as recorded by E2605 and 955.4 s as measured
under E2606's contended load, plus 62–70 s of character baseline and 10–13 s of preparation. This corpus is
2.13× the size, so preparation, tokenizer fitting and packing scale up accordingly while the step budget
does not shrink. Three runs is what fits alongside the remaining work of this session; a fourth and fifth
would have been started and left incomplete, and an incomplete study is worse than a small one.

Seeds 7, 11 and 13 are E2606's first three, chosen by that study's rule (E2605's own seed, then the next
primes), not by any score. Two corpora at the same three seeds also makes the two studies directly
comparable per seed.

**Consequence for reading the numbers:** with n=3 the sample standard deviation is a weak description of
seed-to-seed spread, and nothing in this record will be presented as a confidence interval or as evidence
about populations of seeds. The gap is reported per seed, as three observations, not as one mean.

## Primary metric and baselines

Held-out **bits per character** over the decoded whole target stream of the test documents, separator and
unknown-character targets excluded from the denominator and reported by count — the same function
(`chakaso.training.model_metrics.metrics_from_scores`), so a subword model cannot win by splitting sentences
into more tokens. Secondary: nats per token and perplexity (labelled tokenizer-bound), token accuracy,
selected step, wall clock, checkpoint checksum. Baselines: uniform, add-alpha unigram fitted on the training
documents, and E2601's character architecture, all on the same held-out documents.

The context asymmetry is unchanged and still applies: 128 subword tokens (about 2.08 characters each on the
internal corpus; this corpus's ratio is measured and recorded per run) against the character model's 16
characters.

## Success, inconclusive and failure, fixed in advance

- **Success:** the Transformer's held-out bits per character is below the character model's and below both
  trivial baselines on all three seeds.
- **Inconclusive:** fewer than three seeds complete, or the corpus cannot be re-pinned from the recorded
  URLs and digests.
- **Failure:** any seed on which the Transformer is not below the character model, or on which it is not
  below the unigram baseline. That would mean E2605's advantage was a property of this repository's prose,
  and every document that quotes 1.8657 has to say so.

A negative result is a valid outcome and will be written up as one. Failing to beat the *unigram* baseline
on independent text is the most informative failure available here, because it would say the model learned
the shape of Chakaso's documentation and nothing else.

## Failure and missing-run policy

As preregistered for E2606: a seed that raises is recorded as failed with its exception and elapsed time,
the study continues, the aggregate names what is missing, a run is retried only for an infrastructure cause
at the same seed and configuration, and a completed run with a bad score is never dropped.

## Planned outputs

- `data/experiments/E2607/study.json` (aggregate, digest cited here) and one `report.json` per seed.
- Committed: this record, `corpus-manifest.json`, and the aggregate `results.json` if it stays under the
  repository's tracked-size bound.
- A table of seed × bits per character for both architectures and both baselines, every seed shown.
- Generation samples: the fixed continuation prompts do not apply to this corpus (they quote Chakaso's own
  headings), so the qualitative samples are continuations *of held-out documents of this corpus*, recorded
  verbatim with the repetition and loop statistics from `chakaso.training.generation_eval`, separated from
  the loss numbers and labelled as surface statistics.

## Known limitations and threats to validity

- **Three seeds, one machine, one budget.** Spread is described, not inferred.
- **The three texts are not one domain.** A children's book, a gothic novel and an epic translation in
  prose are three registers; held-out performance mixes them. That is closer to real text than the internal
  corpus and further than any single work would be, and neither property is presented as a virtue.
- **Translation is not the original.** *The Iliad* here is an English translation; the file's header names no
  translator, and this record does not either.
- **The 16,384-character chunking is a choice.** Chunk size changes what a document-level split holds out;
  a different bound is a different experiment, not a robustness check of this one.
- **Copyright and licence are assessed for the United States** by the source's own statement. Anyone
  rebuilding this corpus elsewhere must check their own law, as the files themselves say.
- **Still no workload.** Nothing here asks a question or grounds an answer. A lower held-out loss on
  external text would mean the model learns statistical structure in prose it was not written about. It
  would not mean the model is useful, coherent, or able to answer anything.

## Follow-up

Whichever way this lands: the generator's looping behaviour on this corpus is the next measurement, then the
capacity question that both corpora bound. If the advantage does not survive, the next experiment is about
the corpus (size and domain), not the architecture.
