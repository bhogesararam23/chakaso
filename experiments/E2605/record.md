# E2605 — Does a subword Transformer trained from scratch on the project's own text beat a character model and the trivial baselines, in bits per character?

- Status: completed (written and committed before the run, per the record rules)
- Date: 2026-10-21
- Hypothesis: none of H1–H7 — this is a model-capability experiment on the project's own text, and the
  first the repository runs with a Transformer. A result here informs whether the training path is worth
  scaling, not whether grounding or citation behaviour works on real language.
- Owner: Chakaso autonomous engineering session (P31)
- Git commit: `9b8a478` (this record) with driver `0cbed79` and CLI action `d878828`; the run's tree
  also carried the driver's interval clamp, committed after it as `bcdfa7d`
- Config: the library defaults in `chakaso.training.subword_experiment.ExperimentPlan`, listed below; no
  value is set anywhere but in that dataclass

## Motivation

E2601 and E2602 measured a 984-parameter character MLP on a 344-token hand-written corpus. That is a
demonstrated learning loop, not a language model, and ADR-0033 made the model the product precisely
because further application work could not close the gap it was waiting on. Three things have to be
established before anything can be claimed about scaling: that a real corpus can be named and pinned
(the manifest), that a tokenizer trained by the project behaves correctly (byte-pair subword), and that
the training path produces a model that measurably beats both a trivial predictor and the previous
architecture.

An experiment that would not change a decision should not be run. This one decides: if the subword
Transformer cannot beat the character model or the unigram baseline on held-out documents, the next step
is a data and protocol problem, not a scaling one, and the roadmap must say so instead of growing the
model.

## Baseline

Measured on the same split, the same test documents, through the same metric function
(`chakaso.training.model_metrics.metrics_from_scores`):

- **Uniform** over the subword vocabulary: `ln(V)` nats per token — the loss of knowing only the alphabet.
- **Unigram**: Laplace-smoothed frequencies of the *training* tokens, scored on the test tokens — the loss
  of knowing how often identifiers occur, ignoring order.
- **Character model**: `CharCausalMLP` (E2601's architecture, hand-derived gradients verified by finite
  differences), mini-batched over the same training documents with a 16-character window — the previous
  best thing this project has trained.

Bits per character is the unit that makes the character model and the subword model comparable at all;
per-token loss and perplexity are reported too but are labelled as tokenizer-bound.

## Configuration

Transformer: vocabulary 1024 (byte-pair, trained on the training side only, `min_frequency=3`), context
128 tokens, `d_model` 128, 3 layers, 4 heads, feed-forward 384, embeddings tied to the head, dropout 0.0,
AdamW with peak rate 6e-4 and weight decay 0.01, 60-step linear warmup, gradient clipping at norm 1.0,
batch 32 sequences, ceiling 2400 optimizer steps, validation every 60 steps, early stopping after 8
evaluations without a 0.001-nat improvement, seed 7.

Character baseline: window 16 characters, embedding 32, hidden 64, Adam at 2e-2, batch 256 windows,
4000 steps, validation every 200 steps, same seed. Its vocabulary is fit on the training documents alone.

Both models are selected on the validation documents and reported on the test documents.

## Data

`chakaso-repo-text-v1`: Chakaso's own tracked text — every markdown file under `docs/`, every markdown
file at the repository root, plus `LICENSE` and `AGENTS.md`, discovered by the include spec on the
experiment plan. Author: this project. Licence: Apache-2.0 (ADR-0004); no third-party text, no download,
no scraping, so there is no licence question left open and no contamination from a public benchmark.

As pinned by the run: **71 documents, 780,223 characters, manifest fingerprint `e45993f9badf706c`**, split
49 / 11 / 11 documents (train / validation / test) with no duplicate content dropped, split fingerprint
`87067cca96275def`. The tokenized sides hold 240,418 / 82,376 / 46,577 identifiers at vocabulary 1,024
(767 learned merges, fingerprint `2605bf8f50183a6e`, `min_frequency=3`), and the test side decodes to
2.0829 characters per identifier. The pinned manifest is committed beside this record as
`corpus-manifest.json`; the text itself is not, per the artifact policy.

The corpus is pinned by content rather than by a path list: `CorpusManifest` records each document's
relative path and the SHA-256 of its normalized text (line endings to `\n` before hashing, so a CRLF
checkout and an LF checkout produce one fingerprint), and `verify_corpus` refuses to prepare the corpus
if a file has changed, disappeared, or newly matched the include spec since the manifest was written.
Preparation, licence, size and the pinned manifest live beside this record; the bytes themselves are not
committed (artifact policy).

Split: whole documents, seeded by the experiment seed, 15% validation / 15% test, deduplicated by content
before selection so a repeated document cannot be trained on twice and held out at once. The character and
subword pipelines consume the *same* split object — a second split anywhere would be a different
experiment, and an earlier draft of the driver made exactly that mistake.

**The limitation, stated plainly:** this is one author's technical prose about one project. It is real
English text of substantial size, and a model can learn its distribution; it is not a general corpus, and
no number here may be read as general language ability. Held-out documents are also documents the model's
author wrote about the system, so "learns the corpus" here includes memorizing a house style.

## Model

Initialized from random parameters: a named zero-mean normal (std 0.02) for every embedding and linear
weight, zero biases, unit normalization scales — set from the seed, with no framework default left
implicit. Parameter count is computed from the architecture (`TransformerConfig.parameter_count`) and
cross-checked against the built model by test, because tied weights make a byte-level sum over the stored
tensors overstate it.

The artifact is `chakaso-transformer-checkpoint/1` (ADR-0035): tensors plus metadata plus a checksum over
both, read with `weights_only=True`, carrying the tokenizer inline. The checkpoint's bytes are not
committed; its checksum is recorded below once the run finishes.

## Environment

Recorded by the artifact itself at save time (`describe_environment`): torch version, Python version,
platform string, device `cpu`. Observed for this run: 18 logical CPUs, 16.57 GB physical RAM with the
machine otherwise loaded, no CUDA device, ~168 GB free disk. No accelerator was used, and no throughput
claim is made beyond the wall-clock the run reports.

## Reproducibility

```bash
python -m pip install -e ".[dev,train,nn]"
python -m chakaso experiment model \
  --work-dir data/experiments/E2605 \
  --write-report experiments/E2605/results.json
```

Seed 7 throughout. The corpus is fixed by the manifest committed with this record, so the same commit
reprepares the same data even after later documentation edits — and a mismatch is a refusal, not a
substitution. Exact replay is claimed for this interpreter and CPU build; bit-identical replay across
torch versions or hardware is explicitly **not** claimed (ADR-0034 rule 6).

## Metrics

- `loss_nats`: mean next-token cross-entropy over every scored position of the test side.
- `perplexity`: `exp(loss_nats)`, meaningful only inside one identifier space.
- `bits_per_character`: total description length in bits divided by the characters the scored tokens
  decode to. The denominator is the decoded whole stream, not a per-token character count.
- `token_accuracy`, `positions_scored`, `characters_scored`, `separator_targets`: what was predicted, over
  how much, and how many scored positions were document boundaries.
- Training loss is recorded per step and is never the reported held-out number; the curve is included so
  the overfitting is visible rather than summarized away.
- Implementation: `chakaso.training.nn_evaluation`, `chakaso.training.model_metrics`, and for the character
  side `chakaso.training.evaluation.per_position_cross_entropies`.

## Results

Measured on 2026-10-21 by the command in Reproducibility above, run from the tree at `9b8a478` in the
`[nn]` environment (Python 3.13.5, torch 2.13.0+cpu, device `cpu`). That tree carried one change not yet
committed when the run started — the driver's evaluation-interval clamp, committed after the run as
`bcdfa7d` — which computes the same 200-step interval this plan already used, so it cannot have moved a
number here. The full machine-readable report is committed as [`results.json`](results.json), digest
`7b05bec0aedf7b1ca6ef29f5f17f744a64501cc9f949d783b4d78f484e942183` (the SHA-256 of the canonical JSON
payload as written; normalizing CRLF to LF before hashing reproduces it from the committed file).

**Held-out (test documents), every model-and-baseline number produced by the same metric function over the
same positions:**

| Predictor | Identifier space | Bits per character | Loss (nats/token) | Perplexity | Token accuracy | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| Subword Transformer | 1,024 BPE | **1.8657** | 2.6936 | 14.79 | 0.4501 | 643,840 |
| Unigram over subword tokens | 1,024 BPE | 3.3224 | 4.7967 | 121.12 | 0.2693 | 0 |
| Uniform over subword tokens | 1,024 BPE | 4.8010 | 6.9315 | 1,024.00 | 0.0000 | 0 |
| Character MLP (E2601 architecture) | 116 chars | **3.1343** | 2.1724 | 8.78 | 0.4059 | 44,084 |
| Unigram over characters | 116 chars | 4.7764 (derived) | 3.3108 | — | — | 0 |
| Uniform over characters | 116 chars | 6.8580 (derived) | 4.7536 | — | — | 0 |

The two derived rows divide the measured mean cross-entropy in nats by `ln 2`, which is exact for a
character identifier space up to the 5 unknown-character targets among 97,084 positions (0.0003
bits/character, in the same direction for both rows). Their perplexities are `exp` of the nats column.
**A baseline's bits per character is not tokenizer-free**: uniform over 1,024 subwords (4.80) is cheaper
than uniform over 116 characters (6.86), because each subword covers about two characters. The
cross-tokenization comparison is therefore between *models* (1.8657 against 3.1343), while each model is
judged inside its own identifier space, where it wins by 1.4567 bits/character (Transformer over its
unigram) and 1.6422 bits/character (character model over its own).

Scored denominators, so every figure above can be re-derived: the Transformer was scored over 46,208
positions covering 96,246 characters (10 of those positions had a document separator as target); the
character model over 97,084 positions covering 97,079 characters, of which **5** targets were characters
absent from the training vocabulary, encoded to the reserved unknown identifier and scored as such —
0.005% of positions, which does not move the comparison.

**Training and selection, read off validation.** The snapshot reported is step **1380** of a 2400-step
budget, at validation loss 2.678459 nats. One later evaluation (step 1440, 2.678170) recorded the curve's
lowest number but did not beat the incumbent by more than `min_delta` = 0.001, so under the stated rule it
did not become the selection — the selected snapshot is the first point of the plateau, not the argmin.
After it, validation rose while training loss kept falling (step 1920: train 1.6940, validation 2.8219).
Early stopping triggered at step 1920: the implemented rule stops when the count of non-improving
evaluations *exceeds* patience, so with patience 8 it stopped on the ninth, and the run's guard
(`choose_validation_best`, the pure form of the rule) agreed with the loop's incremental selection. The
model that would have been reported had the budget run out is 0.1434 nats worse on held-out text than the
one that was. Wall clock: 513.6 s for the Transformer, 69.9 s for the character baseline, 13.2 s to pin the
corpus, train the tokenizer and pack both sides. The character baseline selected at step 3800 of its 4000
at validation 2.108838 nats.

**Artifact:** `E2605-transformer.pt`, schema `chakaso-transformer-checkpoint/1`, checksum
`95c849b3cffa5cb0d18a3d2983bfc542da38ff52b4c869d796a7e666865d1416`, carrying the 1,024-entry tokenizer
inline. Its bytes are not committed.

**Generation samples** — greedy (temperature 0), 96-token ceiling, produced by loading the checkpoint
through `chakaso.models.load_model` and calling `generate`, exactly as an application would. They are quoted
as stored in `results.json`, one line each, with newlines escaped as `\n` and nothing added or wrapped,
because they are the honest evidence of what this model is:

```console
prompt: "# Chakaso\n\nChakaso is a"                        finish: length
text:   " model that is a model that is a model that that is not a model to a model that it.\n\n## Consequence is a model\n\n- **A boundary** rested as a real model that a model that reachable to in a\nreal"

prompt: "## Status\n\nThe retrieval layer"                 finish: length
text:   " is a real model and a real model model (`chakaso benchmarks`, ADR-0031).\n\n## Consequences\n\n- The model is a real model (the model boundary)\n\n- **A model* is a real model* that a model that not a model"

prompt: "The tokenizer is trained by the project because"   finish: stop
text:   " it is not a model to a model that it.\n"
```

**Correction, recorded rather than quietly edited.** The Data section was written before the run, and its
sentence naming the corpus carried "754,789 characters" — a figure measured in an earlier scratch
preparation, before the audit, the research note and the new ADRs were committed. The run pinned **71
documents and 780,223 characters**, which is what the manifest, the table above and `results.json` report.
The two places outside this record that quote 754,789 as the current corpus (README and the changelog) are
corrected in the commit that follows; the research-log entry that measured it is left as the dated
observation it was, and the growth is logged as a new entry.

## Interpretation

The measured answer to the question in the title is **yes, on this corpus**: a subword-tokenized causal
Transformer trained from random initialization beats the architecture this project had already measured by
1.2685 bits per character (3.1343 to 1.8657, a 40.5% reduction), and beats the unigram baseline of its own
identifier space by 1.4567 bits per character. The comparison separates two mechanisms rather than crediting
one: attention over a 128-token context reaches across a document in a way a 16-character window cannot, and
the subword vocabulary spends 2.08 characters per predicted identifier, so the same description length per
token buys fewer bits per character.

Bits per character is the number that means something here. Perplexity 14.79 does not compare against the
character model's 8.78 — they are branching factors over different alphabets, and the character model's
lower perplexity alongside its much worse bits-per-character is exactly the confusion this unit exists to
avoid.

The early-stopping discipline was load-bearing, not decorative: the model at the end of the budget was worse
on held-out text than the one at the selected step, which is the same effect E2601 saw on 344 training
positions and E2602 reproduced across five seeds, now over 240,418 training identifiers — 699 times as many
prediction targets.

What the samples establish is the limit rather than the capability. The model has learned this corpus's
*shape* — markdown headings, bullet syntax, the words "model", "real", "boundary", a plausible `## `
structure — and it loops ("a model that is a model that is a model"), restates a claim it has just made, and
finishes a sentence about the tokenizer with "it is not a model to a model that it." The third sample
stopping on its own is the document-separator semantics working, not a model deciding it had said enough. A
model that describes this text at 1.87 bits per character has learned real structure in this distribution and
still means nothing by it.

One more honest note, of the kind that only appears after the run: the corpus grew between the moment this
record was written and the moment the run pinned it — the realignment's own documentation added 25,434
characters of what the model then trained on. That is mild self-reference (the project is training on text
partly about this work) and it is why the manifest pins content rather than a path list. It does not affect
the comparison, since both models saw the same split; it is a further reason to want an external corpus.

## Limitations

The first four bullets were fixed in advance, before the run. Each later addition marks what only the
measured run could show, and is labelled as such rather than presented as pre-declared:

- A single corpus, one author, one domain, one style; the held-out set was written by the same person.
  Every one of the 22 validation and test documents is documentation or an experiment record *about this
  work*, and the corpus grew by 25,434 characters between this record's writing and the run's pinning.
- A single seed for the Transformer. E2602 measured 0.18 nats of sample SD across five seeds on a far
  smaller problem, so the gaps in the table above are point estimates, not effects with an error bar.
- CPU-only at a scale chosen to finish inside one session (513.6 s of Transformer training); a larger model
  or corpus is not measured here, and peak memory was never observed.
- No external benchmark: perplexity on the project's own documentation is not a language-ability test, and
  the generation samples above are offered as evidence of what the model is *not*.
- Bits per character rewards whatever segmentation compresses the corpus; on single-author technical prose
  the subword advantage is partly an artefact of that corpus's redundancy — the vocabulary is 1,024
  identifiers over 240,418 training tokens, so it is dense in exactly this style.
- The reported selection is the first point of a validation plateau, not the curve's argmin (step 1440 was
  0.0003 nats lower and did not clear `min_delta`). At this scale that is a rounding-level difference; at a
  larger one the rule's wording would need to say which of the two it intends.

## Conclusion

Believed, at the strength the evidence supports:

1. **The training path works end to end on a named, pinned corpus** — content-hashed manifest, document
   split, learned subword tokenizer, packing, Transformer, mini-batch training with validation-selected
   early stopping, self-describing checksummed checkpoint, and a `load_model` that generates and scores from
   Python. Every stage is executed and measured or covered by tests; none is a stub. *(Strong.)*
2. **Subword tokenization plus attention beats this project's previous architecture on this corpus** by
   1.2685 bits per character, and both beat their own unigram baselines. *(Strong for this corpus; a claim
   about this distribution, not about language.)*
3. **Validation-selected early stopping matters at this scale**: the un-stopped model was 0.1434 nats worse
   on held-out text. *(Strong — the same effect E2602 measured across five seeds, now on real text.)*
4. **The model is not useful.** 643,840 parameters over 780 KB of one author's technical prose; generations
   loop and repeat. *(Certain.)*
5. **The sizes carry no error bar.** One seed, one corpus, one machine: 1.8657 must not be quoted as an
   expected effect, because E2602 showed a point estimate of this kind moves by 0.18 nats (sample SD) across
   seeds on a far smaller problem. *(Certain.)*

**Reproduction, and what it needs.** Re-preparing this corpus in this checkout now refuses: the pinned
manifest is re-verified against the live tree (`verify_corpus`), and 13 of the 71 documents — including this
record, which was part of the corpus in its pre-run form — have changed since the pin. That is the drift
refusal doing its job: a saved result cannot be restated against text the model never saw. A fresh work
directory re-pins the *current* text, which is a different corpus and therefore a different experiment, and
it is exactly what a reader must not mistake for a reproduction. Closing the gap is a unit of work, not a
flag: the driver must write a snapshot of the pinned text beside the manifest (`write_snapshot` exists and is
tested, but the driver never called it), and `prepare_corpus` must accept a snapshot as its source.

## Follow-up

The branch was fixed in advance and the first branch applies. In order:

0. **Make a measured result reproducible after the repository moves on**: write the corpus snapshot beside the
   manifest and let the preparation consume it, so a re-run can train on the pinned bytes rather than either
   refusing or silently retargeting a newer corpus. The Reproducibility section above, written before the
   run, said the committed manifest lets the same commit reprepare the same data "even after later
   documentation edits" — the same *commit* does; a later one refuses or re-pins something else. That was
   imprecise, and it is recorded rather than quietly corrected.
1. **Reproduce E2605 across pre-declared seeds** (the E2602 discipline) so the bits-per-character gap
   carries a spread rather than being a point estimate.
2. **An external corpus with a recorded licence**, prepared from a manifest that pins content digests, so
   "one author's house style" is no longer the whole of what is measured. The pipeline is already
   corpus-agnostic; the corpus is the missing decision.
3. **Only then** a capacity question (context length, layers, vocabulary size) — a scale change is not
   interpretable until 1 and 2 say what the baseline variance is and what the data can support.
4. The deferred application-side items (P29 calibration, P32 runtime consumption) stay deferred: nothing in
   this result makes them more executable, and ADR-0033's ordering still holds.
