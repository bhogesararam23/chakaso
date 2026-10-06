# Artifact policy

What is committed, what is stored elsewhere, and what has not been decided.

## Committed

Small, human-readable, reproducible-by-recreation artifacts:

- source code, tests, configuration, documentation,
- decision records and research notes,
- experiment *records* (the description of a run), which are small text,
- evaluation dataset definitions and metadata, once they exist, on the condition
  that they are small enough and licensed for redistribution.

## Never committed

Enforced by `.gitignore`:

| Artifact | Why |
| --- | --- |
| The private planning pack (`docs/*.docx`) | Private working material |
| Model weights and checkpoints (`*.safetensors`, `*.pt`, `*.bin`, `*.gguf`, …) | Large, and reproducible from recorded configuration |
| Dataset shards and extraction caches (`data/raw`, `data/processed`, `data/cache`) | Large, and reproducible from a recorded source and licence |
| Local experiment outputs (`experiments/**/outputs`, `experiments/**/runs`) | Machine-specific and large |
| Virtual environments, tool caches, coverage reports | Regenerable |
| Secrets and local environment files (`.env`, `configs/local.toml`) | Secrets never enter version control |

The rule of thumb: if it can be regenerated from something committed plus a
recorded environment, it is not committed.

## Not yet decided

These are genuinely open. Recording them is better than guessing, because a guess
becomes a convention by accident.

### Where do released weights live?

Git is unsuitable for weights at any realistic size, and Git LFS for multi-gigabyte
checkpoints is awkward. The plausible options are a model registry, object storage
with published checksums, or a release attachment for small checkpoints. Nothing
exists to store yet, so the decision is deferred until the first checkpoint exists
and its size is known.

### Which dataset artifacts are redistributable?

Depends on the licence of each source corpus, which depends on the corpus, which
has not been chosen. Each dataset will carry a licence note and a data card when it
exists. No blanket policy is possible.

### How are research artifacts (indexes, tokenizers) versioned?

A tokenizer must be versioned and frozen per model family
([`../../training.md`](../../training.md)). Whether tokenizer artifacts live in Git
(a small BPE vocabulary file plausibly can) or alongside weights depends on size,
and so is deferred with the weights decision.

## Consequence for now

There is nothing large to store. The policy exists so that the first checkpoint and
the first dataset do not get committed by default simply because nobody had
decided otherwise.
