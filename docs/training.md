# Training and model development

Chakaso intends to train its own language model from randomly initialised
parameters. This document describes that plan, and is explicit about what has and
has not been executed. As of 2026-10-24 the training pipeline exists
(`chakaso.training`, gated by [ADR-0030](decisions/ADR-0030-minimal-optional-training-dependency.md)
and, for the model itself, [ADR-0034](decisions/ADR-0034-optional-pytorch-model-extra.md)) and six
experiments have measured from-scratch training: [E2601](../experiments/E2601/record.md) trained a
984-parameter character model that beats trivial baselines on held-out loss and
[E2602](../experiments/E2602/record.md) reproduced that across five seeds,
[E2605](../experiments/E2605/record.md) trained a 643,840-parameter subword causal Transformer that beats it,
in bits per character, on held-out documents, [E2606](../experiments/E2606/record.md) put a five-seed spread on
that result and reproduced its artifact tensor-for-tensor, [E2607](../experiments/E2607/record.md) repeated the
protocol on externally licensed prose, and [E2608](../experiments/E2608/record.md) trained a
3,413,760-parameter rotary decoder to 1.6865 bits per character against 1.9043 for the transformer
architecture at a matched token budget — the artifact it selected is the first one released under a contract, at
Level 2 of [ADR-0039](decisions/ADR-0039-three-level-model-release-gate.md). None of them is a useful model, and
that distinction is what the rest of this document keeps honest.

## Strategy

The model is the product ([ADR-0033](decisions/ADR-0033-model-centred-product-contract.md)); the system
around it exists so the model can be measured, reproduced and eventually replaced by a better one.
Retrieval, conversation, evaluation and the interfaces between them were built first and are kept, because
they are what makes a model number checkable — but no further application capability is allowed to stand in
front of a model result, and that ordering is what changed in 2026-10-21.

Building the system first was not a detour. Evaluation infrastructure, evidence handling and correction
behaviour all had to exist before a trained model could be judged, and a model judged by nothing is a model
nobody can improve. What they could not do was close the gap the model itself has to close.

## What "scratch-trained" means here

Parameters are initialised independently rather than copied from a pretrained
foundation model. Using PyTorch, a tokenizer library, a data-loading library and
open-source tooling does not change that: those are infrastructure, not learned
weights.

This definition is written down because it is the kind of claim that quietly stops
being true — a fine-tune of an open base model is a useful thing to build, but it
is not the thing this project says it is building, and conflating them would make
the project's central claim false.

## Stages

| Stage | Goal | Output | Status |
| --- | --- | --- | --- |
| M0 | Train nothing; validate the system architecture | Web-grounded prototype | **Superseded, and honest about which half happened.** The architecture validation is done and shipped: retrieval, evidence identity, citation validation, grounding boundaries, a durable answer store and a local runtime all exist and are tested without a provider. The *web-grounded prototype* half was never the model's milestone and stays unmet — no model answers a grounded question here (K-001), and default fetching is off ([ADR-0001](decisions/ADR-0001-local-first-runtime.md)) |
| M1 | Train a tokenizer on a controlled corpus | Versioned tokenizer | **Done for both tokenizations** — a deterministic character vocabulary (used by E2601/E2602) and, since P30, a learned byte-pair subword vocabulary trained on pinned training-split text (`chakaso.training.subword`, ADR-0033's model-centred contract) |
| M2 | Train a tiny Transformer from random initialisation | Chakaso-0 checkpoint | **Done** — [E2605](../experiments/E2605/record.md) trained a 643,840-parameter subword causal Transformer from random initialization on the pinned corpus and measured 1.8657 bits per character on held-out documents, against 3.1343 for the character architecture. The artifact is `E2605-transformer.pt`: an experiment result, not a release, and not named Chakaso-0 |
| M3 | Pretrain a small useful base model | Chakaso Base | **Half met, and the half that matters is not.** An artifact named `chakaso-base@0.1.0` now exists and is released under a contract — E2608's 3,413,760-parameter rotary decoder, 1.6865 bits per character on held-out licensed prose, loadable, verifiable, serveable, Level 2 of [ADR-0039](decisions/ADR-0039-three-level-model-release-gate.md). "**Useful**" is not met: its generations repeat phrases, every measured sample hit its token bound, and Level 3 is withheld in [the model card](../MODEL_CARD.md) on that measurement rather than on caution |
| M4 | Instruction tune | Chakaso Instruct | Planned |
| M5 | Teach grounding and correction behaviour | Chakaso Grounded | Planned |
| M6 | Preference optimization, if justified | Aligned variant | Research |
| M7 | Domain adaptation after niche selection | Domain model | Research |

No stage past M2 has produced a *useful* model. E2601 wrote a checkpoint and trained a 984-parameter
character model from random initialization — the project's first measured training result, and a precursor to
M2. E2605 then trained the Transformer itself and measured it against that precursor and against trivial
predictors of both identifier spaces. E2605's and E2607's artifacts are experiment results cited by checksum;
the released artifact is E2608's decoder arm, whose **contract** is published in
[`MODEL_CARD.md`](../MODEL_CARD.md) while **no checkpoint bytes are committed to the repository** and no
version has been put on a package index. A model that answers a grounded question still does not exist: the
trained checkpoint is loadable through the `LanguageModel` boundary from Python, and no runtime path uses it
as a default.

## Sizing policy

| Model | Purpose | Constraint |
| --- | --- | --- |
| Chakaso-0 | Prove the pipeline runs end to end | Very small; must train on free or low-cost compute |
| Chakaso-0.05B–0.1B class | Language-learning research | Small enough for repeated experiments |
| Chakaso-0.3B class | Potentially useful small model | Only after the pipeline and evaluation are stable |
| Larger | Future | A compute-driven decision, not a parameter-count target |

Parameter count is not a goal. A model is scaled when an experiment needs more
capacity to answer a question, and when the evaluation exists to tell whether the
result is better.

## Initial architecture

A decoder-only Transformer with causal self-attention:

- token embeddings plus a positional representation,
- feed-forward blocks,
- residual connections and normalization,
- a linear language-model head,
- configuration-driven dimensions, so parameter count is a configuration value
  rather than a code change.

Attention is the mechanism that makes this tractable; optimized
scaled-dot-product attention primitives can be adopted later without changing the
conceptual architecture.

## Tokenizer

Trained as part of the project rather than borrowed, because tokenizer behaviour
is a research variable: vocabulary size and segmentation rule change how much text
fits in a context window and how well a small model generalises.

Planned approach:

- start with BPE and Unigram as experimental candidates,
- measure token efficiency on representative text rather than assuming,
- include domain text only after a domain is selected,
- freeze tokenizer versions per model family,
- never silently retrain a tokenizer while continuing to train a model against it.

The last rule matters more than it looks. A tokenizer change silently invalidates
every checkpoint trained against the old one, and the failure appears as a model
that suddenly cannot produce coherent text.

## Dataset pipeline

```text
raw sources
  -> licence and usage check
  -> language and quality filtering
  -> normalization
  -> deduplication
  -> contamination checks
  -> document metadata
  -> train / validation / test split
  -> tokenization
  -> packed sequences
```

Data hygiene comes before benchmark optimisation. Leakage, duplication and
train/test contamination can make a model look better than it is, and a benchmark
number obtained that way is worse than no number, because it will be trusted.

| Data type | Purpose |
| --- | --- |
| General text | Learn language structure |
| High-quality reference text | Reduce noisy learning |
| Instruction examples | Teach useful interaction |
| Grounded QA | Teach evidence-conditioned answering |
| Correction pairs | Teach revision after contradictory evidence |
| Abstention examples | Teach appropriate uncertainty |
| Ambiguity examples | Teach clarification rather than guessing |
| Preference pairs | Teach better response behaviour |
| Domain examples | Later specialization |

### What is actually pinned today

Three corpora exist, each named by content rather than described, and the licence column is filled in from
what each source states about itself:

| Corpus | Documents | Characters | Licence | Used by |
| --- | --- | --- | --- | --- |
| `chakaso-repo-text-v1` | 71 | 780,223 | Apache-2.0, the project's own licence; the text is authored for Chakaso | E2605, E2606 |
| `gutenberg-english-prose-v1` | 105 | 1,660,274 | Public domain in the United States, distributed under the Project Gutenberg License, whose own preamble tells users outside the United States to check local law | E2607 |
| `gutenberg-english-prose-v2` | 919 | 14,490,005 | the same Project Gutenberg terms, recorded per source in `configs/corpora/gutenberg-english-prose-v2.toml` | E2608 |

The two external corpora each have a data card — [`gutenberg-english-prose-v1`](data/gutenberg-english-prose-v1.md),
[`gutenberg-english-prose-v2`](data/gutenberg-english-prose-v2.md) — generated from the artifact itself by
`chakaso corpus report`, carrying the measured composition, the document-length percentiles and the
near-duplicate screen rather than a description of the collection. `chakaso-repo-text-v1` has no card, because
its composition *is* the repository: a card would restate the manifest and the tracked-file list, which a
reader already has, and the snapshot a record cites is recovered from a named revision rather than declared
independently.

A corpus is declared before it is used. `chakaso corpus declare` reads each downloaded file's own header and
bytes and writes the catalogue; `chakaso corpus prepare` refuses any file whose digest is not the declared
one, ingests the rest into a digest-keyed cache, and pins the snapshot a run trains from:

```bash
chakaso corpus declare --raw-dir data/corpora/gutenberg-v2 --name gutenberg-english-prose-v2 \
  --version 1.0.0 --licence "…" --retrieved 2026-10-22 \
  --output configs/corpora/gutenberg-english-prose-v2.toml
chakaso corpus prepare --catalog configs/corpora/gutenberg-english-prose-v2.toml \
  --raw-dir data/corpora/gutenberg-v2 --cache-dir data/corpora/gutenberg-english-prose-v2 \
  --snapshot data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json \
  --report data/experiments/E2608/dataset-report.json \
  --data-card docs/data/gutenberg-english-prose-v2.md
```

The instruction, grounded-QA, correction-pair, abstention and preference rows above are still **no dataset
collected**, and nothing in this table suggests otherwise: neither pinned corpus contains a question and an
answer grounded in evidence. The Gutenberg corpora are 19th-century prose, a classical epic and a scientific
essay in English, chosen because they are somebody else's text and legally re-usable, not because they are
training data for an assistant.

## Reproducing a recorded run

The corpus a result was measured on is not the corpus that happens to be on disk. Once a repository edits
its own documentation, the drift guard in `chakaso.training.manifest` refuses to re-prepare a pinned corpus —
correctly, because re-preparing it would silently move the data under a saved number. Reproduction therefore
recovers the bytes from the revision that recorded them ([ADR-0036](decisions/ADR-0036-corpus-snapshot-recovery-from-history.md)),
and every later run reads the recovered snapshot instead of the tree. Nothing is downloaded, and the
`git archive` stays local to this clone.

```bash
# 1. the tier the model path needs
python -m pip install -e ".[dev,train,nn]"

# 2. recover E2605's exact corpus from the revision its record names, and prove it
chakaso corpus restore \
  --manifest experiments/E2605/corpus-manifest.json \
  --revision 9b8a478 \
  --output data/experiments/E2605/chakaso-repo-text-v1-snapshot.json
chakaso corpus verify \
  --manifest experiments/E2605/corpus-manifest.json \
  --snapshot data/experiments/E2605/chakaso-repo-text-v1-snapshot.json

# 3. re-run E2605 against the recovered bytes and the corpus it names
chakaso experiment model \
  --work-dir data/experiments/E2605-repro \
  --corpus-snapshot data/experiments/E2605/chakaso-repo-text-v1-snapshot.json \
  --expected-corpus-fingerprint e45993f9badf706c \
  --split-seed 7 --seed 7 --write-report data/experiments/E2605-repro/results.json

# 4. the five-seed study, and its aggregate rebuilt from the reports on disk
chakaso experiment study --work-dir data/experiments/E2606 --experiment E2606 \
  --seeds 7,11,13,17,19 --split-seed 7 \
  --corpus-snapshot data/experiments/E2605/chakaso-repo-text-v1-snapshot.json \
  --expected-corpus-fingerprint e45993f9badf706c \
  --reference experiments/E2605/results.json --reference-seed 7 \
  --write-report data/experiments/E2606/study.json
chakaso experiment study --work-dir data/experiments/E2606 --experiment E2606 \
  --seeds 7,11,13,17,19 --split-seed 7 --expected-corpus-fingerprint e45993f9badf706c --collect-only

# 5. generate from a saved checkpoint through the library, with no terminal and no config file
python -c "
from pathlib import Path
from chakaso.models import load_model
model = load_model(Path('data/experiments/E2605/E2605-transformer.pt'))
print(model.sample('Chakaso is a', max_new_tokens=24).text)
print(model.score(Path('docs/training.md').read_text(encoding='utf-8')))
"
```

Three rules make those commands mean something, and the code enforces them rather than recommending them:

- **A run that names a corpus trains on that corpus.** `--expected-corpus-fingerprint` is checked before any
  step is taken; a snapshot of a different corpus is a refusal, not a substitution.
- **A recorded artifact is not overwritten by accident.** Manifests, snapshots, checkpoints and reports all
  refuse to replace an existing file unless `--overwrite` says so, and writes are staged so a run that dies
  mid-write leaves no half artifact to be mistaken for a result.
- **A re-run either refuses or reproduces.** `verify_corpus` checks changed, missing and newly-selected
  documents against the pinned digests; a snapshot restored from a revision one edit away from the pin is
  rejected by the same function a stale working tree is rejected by.

The external corpus is pinned the same way but from a download rather than from history: the three Project
Gutenberg texts are named by URL and byte digest in
[`experiments/E2607/record.md`](../experiments/E2607/record.md), `chakaso.training.source_corpus` cuts them
by the source's own wrapper markers and chunks them at whole-paragraph boundaries, and the resulting
snapshot — not the repository's text — is what those runs load. The corpus bytes are not committed;
[ADR-0020](decisions/ADR-0020-experiment-artifacts.md) keeps them in `data/`, and the record says how to
rebuild them.

## Training records

Every run records the following, and does so as a machine-readable file rather than a note:

```text
ExperimentRecord
    experiment_id, git_commit, config_hash,
    dataset_version, tokenizer_version, seed,
    hardware, optimizer, scheduler,
    checkpoint_path, training_steps,
    validation_metrics, notes
```

That shape is what `chakaso.training.subword_experiment.run_experiment` now returns, and
`write_report` writes it: the corpus block carries the name, version, licence, source, discovery rules,
fingerprint, the snapshot's checksum and acquisition, the split fingerprint and both seed roles; the
tokenizer block its fingerprint, size and merges; the model blocks their architecture, parameter count,
budget, validation curve, held-out metrics and artifact checksum; and the environment block the torch and
numpy versions, the platform, the device and the thread count the run actually used. The digest of the
report is cited in the experiment record, so a reader can confirm the file without regenerating a run.

A run whose configuration cannot be reconstructed is not reproducible, and a
result that is not reproducible is not a result.

## Running on interrupted, limited compute

The plan assumes runs get interrupted:

- design for checkpoint resume rather than assuming a run completes,
- keep datasets streamable and preprocessable in batches,
- use gradient accumulation when memory is the constraint,
- prefer many small repeatable experiments over one long run,
- separate data preparation from training so the expensive stage can resume,
- never assume a single session will finish a run.

## Document boundaries and attention

Packing concatenates documents and terminates each with a separator, but the default attention is causal over
the whole packed block: a token can read every earlier token, including tokens from unrelated documents.
[ADR-0040](decisions/ADR-0040-document-boundary-attention-masking.md) adds an **opt-in** document-boundary
mask. `pack_units(..., track_documents=True)` records the source document of every position;
`ExperimentPlan.document_masking` (CLI: `--document-masking`) makes the decoder attend only within each
position's own document. The default is off because every measured protocol in this repository trained with
full causal attention, and changing that silently would make E2605–E2611 unreproducible by name. The option
is decoder-only for now; the transformer family refuses it. A masked run states the flag in its report and
records it in the resumable checkpoint, and a checkpoint written before the field existed is read as
unmasked. **No benefit is claimed until a controlled masked/unmasked experiment measures one**; the planned
comparison is a pair of decoder arms at the same corpus, split, seed, architecture, token budget and schedule.

## Preference optimization

Direct preference optimization is a plausible later stage, because it optimizes
preference pairs directly without the machinery of a PPO-style pipeline. It is not
introduced before supervised behaviour and evaluation are stable. Optimizing
preferences against an unmeasured baseline produces a model that is better at
something nobody defined.

## Status

| Item | Status |
| --- | --- |
| Corpus and provenance | Implemented — `chakaso.training.corpus` (the self-authored toy text), `chakaso.training.manifest` (pins a real document set by path and content digest and refuses drift), `chakaso.training.snapshot` (recovers a pinned corpus from the revision that recorded it and writes a checksum-verified snapshot), `chakaso.training.source_corpus` (turns a downloaded external text into a pinned corpus offline) and `chakaso.training.dataset_catalog` / `dataset_report` (the catalogue a corpus is declared from, and the measured data card it generates). E2605/E2606 trained on the pinned repository text: 71 documents, 780,223 characters, fingerprint `e45993f9badf706c`; E2607 on 105 documents of Project Gutenberg prose, fingerprint `931b07a635ca5a0c`; E2608 on 919 documents from 24 texts, 14,490,005 characters, fingerprint `e9fc2e92f59d88ea` |
| Tokenizer | Implemented — a deterministic character vocabulary (`chakaso.training.tokenizer`, used by E2601/E2602) and a learned byte-pair subword vocabulary (`chakaso.training.subword`, dependency-free, total over Unicode, self-describing and checksum-verified on load) |
| Dataset pipeline | Implemented — document-level three-way split with deduplication before selection (`chakaso.training.data.split_text_units`), sequence packing with per-position targets (`chakaso.training.packing`), split/manifest fingerprints, and a split seed separate from the model seed so a study can vary initialization without moving the held-out documents. Packing can optionally record each position's source document for the document-boundary attention mask (ADR-0040, default off) |
| Model code | Implemented and measured — the character MLP (E2601, 984 parameters, every gradient finite-difference verified), the decoder-only causal Transformer (`chakaso.training.transformer`, 643,840 parameters in E2605, autograd gradients cross-checked against hand-computed losses and sampled finite differences per ADR-0034), and since E2608 a second family: `chakaso.training.decoder`, a rotary-position, RMSNorm, tied-embedding decoder with an ignore-index target mask and a key/value cache whose stepped forward pass is tested equal to the full one (3,413,760 parameters measured) |
| Training loop | Implemented and measured — full-batch Adam over the character model (E2601/E2602) and mini-batch AdamW with warmup, gradient clipping, validation-selected early stopping and bit-identical resume for the Transformer (`chakaso.training.nn_training`) |
| Checkpoints | Implemented — character JSON schema/2 (checksum-verified, vocabulary embedded), and one tensor-plus-metadata file carrying **two** architecture schemas: `chakaso-transformer-checkpoint/1` (ADR-0035) and `chakaso-decoder-checkpoint/1` (ADR-0037). A file names its own family, the loader builds the class from that and never guesses, the checksum covers tensors plus metadata, `weights_only=True` is explicit, incompatible artifacts are refused by name, and usable and resumable kinds are distinguished |
| Model evaluation | Implemented and measured — `chakaso.training.model_metrics` reports nats per identifier, perplexity and **bits per character** through one arithmetic for any identifier space, with uniform and unigram baselines scored the same way; `chakaso.training.generation_eval` measures what a model emits over the boundary (repetition ratios, longest repeated block, cycles anywhere in the sample and periodic tails, length and termination, refusal on empty input) and carries its own interpretation limit, because none of it measures meaning |
| Seed studies | Implemented — `chakaso.training.seed_study` runs a declared seed list over one fixed protocol, one directory per run, and rebuilds its aggregate from the reports on disk: mean, sample standard deviation, median, range, with no confidence interval published and the reason stated. A reused report is accepted only after its configuration, corpus fingerprint and seed match the study |
| Using a trained model | Implemented — `chakaso.models.load_model(path)` returns a `LanguageModel` from a checkpoint **or from a verified release directory**, with no terminal, no config and no application import; generation is bounded by one validated parameter object (temperature over a stated range, top-k, top-p, a seed, stop sequences, a repeat penalty), `stream()` yields as the model steps and stops when the caller cancels, `score()` returns the same held-out cross-entropy the experiments report, `describe()` states what the runtime is, and `chakaso model inspect|verify|export|generate|stream|score|serve` is the same surface from a terminal — thin handlers over that library, never a second implementation of it. The runtime's own default adapter is still the development double, and `chakaso.serve` refuses to serve it |
| Experiments | E2601 through E2608 are **measured results**, all eight committed: the character model across two records, the retrieval and sufficiency measurements across two, the Transformer across three (E2605, E2606's five seeds, E2607 on licensed prose), and E2608's two-arm comparison on a 919-document licensed corpus. Nothing has been measured on a real *workload* (K-001), and no measured model is useful |

The two halves of this table used to contradict each other — the header cited a measured E2601 while the
rows below said the tokenizer, model code, training loop and checkpoints did not exist. They were written
before any of it ran. The rows now state what is implemented and what has been measured, with the experiment
that measured it named; the substance has changed only where a number arrived.
