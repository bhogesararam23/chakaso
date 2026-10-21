# Training and model development

Chakaso intends to train its own language model from randomly initialised
parameters. This document describes that plan, and is explicit about what has and
has not been executed. As of 2026-10-21 the training pipeline exists
(`chakaso.training`, gated by [ADR-0030](decisions/ADR-0030-minimal-optional-training-dependency.md)
and, for the model itself, [ADR-0034](decisions/ADR-0034-optional-pytorch-model-extra.md)) and two
experiments have measured from-scratch training: [E2601](../experiments/E2601/record.md) trained a
984-parameter character model that beats trivial baselines on held-out loss, and
[E2605](../experiments/E2605/record.md) trained a 643,840-parameter subword causal Transformer that beats it,
in bits per character, on held-out documents. Neither is a useful model, and that distinction is what the
rest of this document keeps honest.

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
| M0 | Train nothing; validate the system architecture | Web-grounded prototype | In progress |
| M1 | Train a tokenizer on a controlled corpus | Versioned tokenizer | **Done for both tokenizations** — a deterministic character vocabulary (used by E2601/E2602) and, since P30, a learned byte-pair subword vocabulary trained on pinned training-split text (`chakaso.training.subword`, ADR-0033's model-centred contract) |
| M2 | Train a tiny Transformer from random initialisation | Chakaso-0 checkpoint | **Done** — [E2605](../experiments/E2605/record.md) trained a 643,840-parameter subword causal Transformer from random initialization on the pinned corpus and measured 1.8657 bits per character on held-out documents, against 3.1343 for the character architecture. The artifact is `E2605-transformer.pt`: an experiment result, not a release, and not named Chakaso-0 |
| M3 | Pretrain a small useful base model | Chakaso Base | Planned |
| M4 | Instruction tune | Chakaso Instruct | Planned |
| M5 | Teach grounding and correction behaviour | Chakaso Grounded | Planned |
| M6 | Preference optimization, if justified | Aligned variant | Research |
| M7 | Domain adaptation after niche selection | Domain model | Research |

No stage past M2 has produced a *useful* model. E2601 wrote a checkpoint and trained a 984-parameter
character model from random initialization — the project's first measured training result, and a precursor to
M2. E2605 then trained the Transformer itself and measured it against that precursor and against trivial
predictors of both identifier spaces. Both checkpoints are reproducible from the seed, config and pinned
corpus manifest, and **neither is committed to the repository**; there are no released weights. A model that
answers a grounded question still does not exist: the trained checkpoint is loadable through the
`LanguageModel` boundary from Python, and no runtime path uses it as a default.

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

No dataset has been collected. Whether any specific corpus is usable depends on
its licence, which will be recorded when it is chosen.

## Training records

Every run will record:

```text
ExperimentRecord
    experiment_id, git_commit, config_hash,
    dataset_version, tokenizer_version, seed,
    hardware, optimizer, scheduler,
    checkpoint_path, training_steps,
    validation_metrics, notes
```

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

## Preference optimization

Direct preference optimization is a plausible later stage, because it optimizes
preference pairs directly without the machinery of a PPO-style pipeline. It is not
introduced before supervised behaviour and evaluation are stable. Optimizing
preferences against an unmeasured baseline produces a model that is better at
something nobody defined.

## Status

| Item | Status |
| --- | --- |
| Corpus and provenance | Implemented — `chakaso.training.corpus` (the self-authored toy text) and `chakaso.training.manifest`, which pins a real document set by path and content digest and refuses drift. E2605 trained on the pinned repository text: 71 documents, 780,223 characters, fingerprint `e45993f9badf706c` |
| Tokenizer | Implemented — a deterministic character vocabulary (`chakaso.training.tokenizer`, used by E2601/E2602) and a learned byte-pair subword vocabulary (`chakaso.training.subword`, dependency-free, total over Unicode, self-describing and checksum-verified on load) |
| Dataset pipeline | Implemented — document-level three-way split with deduplication before selection (`chakaso.training.data.split_text_units`), sequence packing with per-position targets (`chakaso.training.packing`), and split/manifest fingerprints |
| Model code | Implemented and measured — the character MLP (E2601, 984 parameters, every gradient finite-difference verified) and the decoder-only causal Transformer (`chakaso.training.transformer`, 643,840 parameters in E2605, autograd gradients cross-checked against hand-computed losses and sampled finite differences per ADR-0034) |
| Training loop | Implemented and measured — full-batch Adam over the character model (E2601/E2602) and mini-batch AdamW with warmup, gradient clipping, validation-selected early stopping and bit-identical resume for the Transformer (`chakaso.training.nn_training`) |
| Checkpoints | Implemented — character JSON schema/2 (checksum-verified, vocabulary embedded) and `chakaso-transformer-checkpoint/1` (ADR-0035): one file, checksum over tensors plus metadata, tokenizer inline, `weights_only=True`, incompatible artifacts refused by name, usable and resumable kinds distinguished |
| Model evaluation | Implemented and measured — `chakaso.training.model_metrics` reports nats per identifier, perplexity and **bits per character** through one arithmetic for any identifier space, with uniform and unigram baselines scored the same way |
| Using a trained model | Implemented — `chakaso.models.load_model(path)` returns a `LanguageModel` from a checkpoint with no terminal, no config and no application import; the runtime's own default adapter is still the development double |
| Experiments | E2601, E2602 and E2605 are measured results; nothing has been measured on a real workload or an external corpus (K-001), and no measured model is useful |

The two halves of this table used to contradict each other — the header cited a measured E2601 while the
rows below said the tokenizer, model code, training loop and checkpoints did not exist. They were written
before any of it ran. The rows now state what is implemented and what has been measured, with the experiment
that measured it named; the substance has changed only where a number arrived.
