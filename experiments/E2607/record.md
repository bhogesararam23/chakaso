# E2607 — Does the advantage survive on text that is not Chakaso's own?

- Status: completed (measured; all three declared seeds, preregistered before any of them ran)
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

## Follow-up, as declared before the runs

Whichever way this lands: the generator's looping behaviour on this corpus is the next measurement, then the
capacity question that both corpora bound. If the advantage does not survive, the next experiment is about
the corpus (size and domain), not the architecture.

---

## Results

**All three declared seeds completed. None failed, none was dropped, none was rerun, and no seed was added
after a score was seen.** Every run loaded the pinned snapshot (checksum `776afa22a2d54f49…`) and reported the
same corpus fingerprint `931b07a635ca5a0c`, split fingerprint `93be3cc6f9e80423` (73 / 16 / 16 documents,
exactly the assignment declared above), tokenizer fingerprint `48b813f9269bf039` at 1,024 identifiers and 767
merges, 643,840 parameters, and 135,040 scored positions over 253,629 decoded characters with 15 separator
targets excluded. The aggregate refuses to build if any of those disagree across runs.

| Seed | Transformer bits/char | nats/token | perplexity | token acc | selected step | early-stopped | character bits/char | gap | checkpoint checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 2.004910 | 2.610098 | 13.6004 | 0.448860 | 2400 | no | 3.204348 | 1.199438 | `f46610501aa0…` |
| 11 | 2.019749 | 2.629416 | 13.8657 | 0.446083 | 2400 | no | 3.227964 | 1.208215 | `c1a0a2378bb1…` |
| 13 | 2.028269 | 2.640508 | 14.0203 | 0.443024 | 2400 | no | 3.233075 | 1.204806 | `aa7daf6874bd…` |

| Statistic (n=3) | Transformer bits/char | Character bits/char | Gap | Unigram | Uniform |
| --- | --- | --- | --- | --- | --- |
| mean | **2.017643** | **3.221796** | **1.204153** | 3.611479 | 5.324312 |
| sample standard deviation | 0.011821 | 0.015325 | 0.004425 | 0.0 | 0.0 |
| median | 2.019749 | 3.227964 | 1.204806 | 3.611479 | 5.324312 |
| min – max | 2.004910 – 2.028269 | 3.204348 – 3.233075 | 1.199438 – 1.208215 | 3.611479 | 5.324312 |
| range | 0.023359 | 0.028727 | 0.008777 | 0.0 | 0.0 |

Wall clock per run: preparation 26.43–46.46 s, character baseline 123.30–175.95 s, Transformer
971.09–1,357.63 s (mean 1,124.74, sample SD 205.09). The aggregate is
`data/experiments/E2607/study.json`, digest `efe541727c46c3699c02a48757d7abab…`, rebuilt from the three
per-run reports by `--collect-only`; the same statistics were computed a second time directly over the values
in the table above and agree.

**The preregistered success condition holds on every seed.** The Transformer is below the character model on
all three (by 1.1994 to 1.2082 bits per character), below the unigram baseline on all three (by 1.5832 to
1.6066), and below uniform on all three (by 3.2960 to 3.3194). The two trivial baselines are identical across
seeds, as they must be: they depend on the corpus and the split, both of which are fixed.

## The budget did not finish, and that is the headline caveat

On this corpus **no run early-stopped.** All three selected step **2400 — the ceiling of the budget** — with
41 evaluations recorded and the validation loss still falling at the last one:

| Seed | validation nats at step 0 | at step 2400 | minimum over the curve | step of that minimum |
| --- | --- | --- | --- | --- |
| 7 | 6.9759 | 2.6763 | 2.6763 | 2400 |
| 11 | 6.9856 | 2.6922 | 2.6922 | 2400 |
| 13 | 6.9758 | 2.7039 | 2.7039 | 2400 |

The minimum sits at the final evaluation on every seed, which is the definition of a truncated curve.
E2606's runs on the internal corpus stopped between step 1260 and 1380 of the same 2,400.

That is the limitation this record declared in advance, and it happened exactly as stated: 2,400 steps of 32
sequences is proportionally fewer epochs over 1.66 MB of prose than over 780 KB, so the reported model is
**budget-limited, not converged**. Two consequences, neither of which may be glossed:

- **2.0176 bits per character is a floor on what this architecture reaches on this data, not its optimum.** A
  longer budget would lower it, and would very likely also lower the character baseline's number, whose own
  selection landed at step 3000, 3400 and 3600 of its 4,000-step budget rather than early-stopping either.
- **The comparison stays internally fair**, because both architectures and both baselines were run to the same
  declared ceiling on the same split. What the budget limits is the *absolute* level, not the direction or the
  size of the gap.

So the honest cross-corpus comparison is a comparison of gaps, and the gaps are close: **1.2042 bits per
character here (sample SD 0.0046 over three seeds) against 1.2143 there (SD 0.0336 over five).** The
Transformer's advantage over the character architecture survived the move off the project's own prose to within
0.010 bits, which is 0.8% of the gap.

## Generation on prose the model was not written about

`chakaso.training.generation_eval` ran over each of the three checkpoints. The continuation prompts were taken
from **held-out documents of this corpus**, not from Chakaso's headings — the internal suite's prompts quote
this repository's own section titles, which would make the comparison meaningless. They are recorded by source
identifier: `pg11/part-0001.md` (whose first paragraph is the placeholder line `[Illustration]` — an honest
artifact of the corpus rather than a curated prompt), `pg6130/part-0006.md` and `pg6130/part-0014.md`. All raw
output is retained in `data/experiments/E2607/generation-E2607-seed-*.json`.

| Seed | samples (of 3) with a periodic run of ≥3 | tail loop | mean bigram repetition |
| --- | --- | --- | --- |
| 7 | 1 | 0 | 0.5772 |
| 11 | 0 | 0 | 0.5618 |
| 13 | 1 | 1 | 0.6956 |

Representative output, verbatim and unedited:

```text
[prompt: "The armies being ready to engage, a single combat is agreed"]
  → " the\nbirth of the poorial poems of the poetry. The poet of the poet and the\npoetry of the poetry"

[prompt: "[Illustration]"]
  → " THE OF OF OF OF THE OF OF OF OF OF THE OF OF OF OF OF OF OF OF OF OF "

[prompt: "The speeches are to be considered as they flow from the"]
  → " body of the poems. The\npalm of the brass of the body of the brass of the brass of the poets\nand"
```

The model produces pseudo-Homeric noun phrases and stops coining words that are not in the corpus (`poorial`,
`pooriginal`, `matigin`) — which is what a next-token predictor does when it has learned the surface music of a
text without learning anything that constrains it to be true. The repetition statistics are the same order as
on the internal corpus (mean bigram repetition 0.4870–0.6904 there, 0.5618–0.6956 here), and on the external
text the looping is if anything more naked, since one sample above is 38 characters of `OF` repeated.

**The measured improvement in held-out loss did not become usable text. It was never expected to, and this
record states that rather than letting the loss number imply otherwise.**

## Interpretation

1. **The subword Transformer's advantage is not an artifact of this project's own prose.** On three seeds of
   independently licensed text, in three different registers, with the same architecture, budget, tokenizer
   configuration and metric code, the held-out gap to the character model is 1.2042 bits per character against
   1.2143 on the internal corpus. The direction, the size and the ordering of the four quantities (Transformer
   < character < unigram < uniform) all replicate.
2. **Absolute numbers do not transfer, and must not be compared across corpora.** 2.0176 against 1.8794 bits
   per character is not the model getting worse at language: the denominators are different text, and the
   subword tokenizer's compression falls from 2.0829 to 1.8782 characters per identifier, which alone moves
   bits-per-character in the unfavourable direction at a fixed per-token loss. Bits per character is comparable
   *within* a corpus across tokenizations, which is why it is the primary metric; across corpora it describes
   that corpus.
3. **The seed spread is again far smaller than the gap**: SD 0.0118 against a gap of 1.2042, roughly 1/102 —
   the same order as E2606's 1/158, measured over fewer seeds, so read it as "small", not as a refined
   estimate.
4. **What the failure would have looked like, and did not.** Had the Transformer landed above the unigram
   baseline on this prose, every claim made from E2605's 1.8657 would have had to be reworded as a statement
   about Chakaso's own documentation. It did not — but it did not on three seeds of three 19th-century books,
   which is a long way from a general claim about language.

## Deviations from the preregistration

1. **The budget risk declared in advance materialized**: no run early-stopped and all three selected the
   2400-step ceiling. The protocol was not changed mid-study — no budget was raised after the curve was seen —
   and the consequence is reported as a floor on the level, not a revision of the gap.
2. **The generation prompts moved to held-out documents of this corpus**, which the Planned outputs already
   specified; the recorded identifiers and the `[Illustration]` artifact are disclosed rather than swapped for
   prettier prompts.
3. Nothing else. The corpus, split, tokenizer rules, model configuration, baselines, seeds, aggregation and
   reporting rules are what was declared before the runs, and no seed was added, dropped, rerun or
   substituted.

## Limitations beyond those declared in advance

- **Three seeds, and a standard deviation over three numbers.** The gap's SD (0.0046) is not stable at this n,
  and the 1/102 spread-to-gap ratio should be read as "small", not as a measurement.
- **Public-domain prose is not a workload.** Three books are *somebody else's text*, which is what makes the
  comparison independent; nobody asked the model to do anything with them.
- **Chunking leaves some documents beginning with a placeholder or a front-matter line**, and the split assigns
  whole documents, so a held-out prompt can start awkwardly — one of the three here does.
- **The Iliad is a translation and by far the largest contributor** (69 of 105 documents), so "external prose"
  leans toward one register rather than balancing three.
- **No peak-memory figure was obtained**, as elsewhere in this repository: the process-memory interfaces
  available from this shell return nothing, so nothing is estimated in its place.

## Conclusion

Believed, at the strength three seeds of externally licensed prose support: **the subword Transformer's
held-out advantage over the character architecture and over both trivial baselines is not a property of
Chakaso's own documentation.** On 1.66 MB of Project Gutenberg text, with the budget and protocol unchanged, the
gap is 1.2042 bits per character against 1.2143 measured on the internal corpus, and every seed beats every
baseline. Also believed, and unchanged by that result: **the generated text is not usable.** Its loss improved;
its sequences did not become statements, and on the external corpus one degenerates into repeated function
words.

Not established, and not claimed here: that the model generalizes to language in general, that the reported
level is where this architecture converges (no run finished its budget), or that anything about a real
workload has been measured. K-001 stays open.

## Follow-up

The next experiment is the one this record's own caveat makes unavoidable rather than optional: **the same
external corpus at a budget that lets training converge.** The validation curve was still falling at step 2400
on all three seeds, so the level reported here is truncated. Two candidates, varied one at a time: raise the
step budget on this corpus, or hold the budget and vary the corpus size. Raising the budget first is the
cheaper inference, because the truncation is measured and its direction is known, and a capacity question asked
against an unconverged baseline cannot be answered.

## Corrections applied after the run (2026-10-22)

Two numbers in this record disagreed with the per-run reports it aggregates. Corrected in place, with the
original wording kept, per the rule that a measured record is not silently edited.

| Where | Said | Correct | How it was caught |
| --- | --- | --- | --- |
| *Results*, success-condition paragraph | below the unigram baseline "by 1.5918 to 1.6066" | the three gaps are **1.5832, 1.5917, 1.6066** — the range is 1.5832 to 1.6066 (seed 13 has the smallest gap) | recomputed from the three `report.json` files |
| *Results*, aggregate table | gap sample standard deviation **0.004642** | **0.004425** (digits transposed); the mean 1.204153, median 1.204806, min–max and range columns are unaffected | recomputed from the same three values |

The aggregate `results.json` (digest `efe541727c46c369…`, the value this record already cited) is now committed
in this directory, so the statistics above can be re-derived without the gitignored `data/` tree.
