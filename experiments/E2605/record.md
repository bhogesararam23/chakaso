# E2605 — Does a subword Transformer trained from scratch on the project's own text beat a character model and the trivial baselines, in bits per character?

- Status: running (written and committed before the run, per the record rules)
- Date: 2026-10-21
- Hypothesis: none of H1–H7 — this is a model-capability experiment on the project's own text, and the
  first the repository runs with a Transformer. A result here informs whether the training path is worth
  scaling, not whether grounding or citation behaviour works on real language.
- Owner: Chakaso autonomous engineering session (P31)
- Git commit: `d878828` (the CLI action); driver `0cbed79`; all layers as of this commit
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

Pending. This record is committed before the run so that what follows is what the procedure produced, not
what it was hoped to produce.

## Interpretation

Pending.

## Limitations

Fixed in advance, so they are not written after seeing the numbers:

- A single corpus, one author, one domain, one style; the held-out set was written by the same person.
- A single seed for the Transformer unless the run is later reproduced across seeds (E2602 showed that
  matters for a result of this size).
- CPU-only at a scale chosen to finish inside one session; a larger model or corpus is not measured here.
- No external benchmark: perplexity on the project's own documentation is not a language-ability test, and
  no generation sample below is offered as evidence that the model means anything.
- Bits per character rewards whatever segmentation compresses the corpus; on single-author technical prose
  the subword advantage may be partly an artefact of that corpus's redundancy.

## Conclusion

Pending.

## Follow-up

Decided by the outcome, in advance: if the Transformer beats both baselines and the character model, the
next unit is a multi-seed protocol at this scale plus a genuinely external corpus with a licence record;
if it does not, the next unit is the data and protocol diagnosis, not a bigger model.
