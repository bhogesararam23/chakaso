# Training and model development

Chakaso intends to train its own language model from randomly initialised
parameters. This document describes that plan, and is explicit about the fact that
none of it has been executed.

## Strategy

The system and the model are separate efforts inside one repository. The system is
built first, around a replaceable model, so that the eventual Chakaso model can
replace the initial one without redesigning retrieval, conversation, evaluation or
the interfaces around them ([ADR-0002](decisions/ADR-0002-language-model-boundary.md)).

Building the system first is not a detour. Evaluation infrastructure, evidence
handling and correction behaviour all have to exist before a trained model can be
judged, and a model judged by nothing is a model nobody can improve.

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
| M1 | Train a tokenizer on a controlled corpus | Versioned tokenizer | Planned |
| M2 | Train a tiny Transformer from random initialisation | Chakaso-0 checkpoint | Planned |
| M3 | Pretrain a small useful base model | Chakaso Base | Planned |
| M4 | Instruction tune | Chakaso Instruct | Planned |
| M5 | Teach grounding and correction behaviour | Chakaso Grounded | Planned |
| M6 | Preference optimization, if justified | Aligned variant | Research |
| M7 | Domain adaptation after niche selection | Domain model | Research |

No stage past M0 has produced anything. There are no weights in this repository
and no checkpoint has ever been written.

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
| Tokenizer | Not started |
| Dataset pipeline | Not started |
| Model code | Not started |
| Training loop | Not started |
| Checkpoints | None exist |
| Experiments | None run |

Nothing in this document has been executed. The sequence above is a plan, and it
will be revised when the first experiment contradicts it.
