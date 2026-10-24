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

**The premise of this question expired on 2026-10-22.** Checkpoints exist, and their sizes are measured rather
than imagined: E2608's usable decoder artifact is **17,906,053 bytes** and its resumable one **45,254,464**, the
usable transformer arm 6,301,473, and the corpus snapshot that made them re-trainable 15,760,296 — against a
repository hygiene bound of **1 MiB per tracked file** (`MAX_TRACKED_FILE_BYTES` in
[`tests/test_repository_hygiene.py`](../../../tests/test_repository_hygiene.py)). Git is therefore excluded by
the project's own test, not by taste: a checkpoint could not be committed even if the policy allowed it.

What is still undecided is the destination. The plausible options remain a model registry, object storage with
published checksums, or a release attachment for an artifact of this size. Today the answer for one artifact is
*neither*: `chakaso-base@0.1.0` is distributed as a contract that lives in Git (`model.json`, 3,529 bytes) plus
bytes each consumer rebuilds from [E2608's record](../../../experiments/E2608/record.md) with the commands that
record states. That works because the corpus is licensed and the training is reproducible; it stops being
enough the moment someone is expected to load a weight they cannot spend an hour rebuilding.

### Which dataset artifacts are redistributable?

Depends on the licence of each source corpus — and since E2607 that dependency is recorded rather
than assumed: three pinned corpora with per-document content digests and snapshots whose checksums the records
cite, data cards for the two external ones ([`docs/data/`](../../data/)) carrying measured composition, length
distribution and a duplication screen, a machine-readable catalogue for `gutenberg-english-prose-v2`
([`configs/corpora/gutenberg-english-prose-v2.toml`](../../../configs/corpora/gutenberg-english-prose-v2.toml)),
Project Gutenberg's own terms including its sentence that users outside the United States must check local law,
and this repository's prose under Apache-2.0. The corpus *bytes* stay in `data/`; what a reader gets is the
digests and, where a catalogue exists, the URLs to re-fetch from, so "redistributable" here means "rebuildable
from its own source, verifiable against a digest we publish".
[ADR-0004](../../decisions/ADR-0004-apache-2.0-license.md)'s reservation of weights and datasets for their own
release policy is untouched; a blanket policy is still not possible, and now it is not necessary either.

### How are research artifacts (indexes, tokenizers) versioned?

A tokenizer is versioned and frozen per model family ([`../../training.md`](../../training.md)), and the
measured size question has an answer: the E2608 vocabulary file is **39,139 bytes** — 4,096 identifiers and
3,839 merges, 26 times below the tracked-file bound above. It is not committed anyway,
for two reasons that are not size. It is *cached by identity* — corpus fingerprint, split, target size and
frequency floor are part of its key, so a reuse cannot silently mean something else — and its merge table
travels **inline inside every checkpoint** that used it
([ADR-0035](../../decisions/ADR-0035-transformer-checkpoint-format.md)), where the loader re-hashes it against
the recorded fingerprint before a weight is read. A committed copy beside that would be a second source of
identity, free to disagree.

## Consequence for now

There is something large to store, and nothing large is stored in the repository. The policy earned its keep:
the first checkpoints, snapshots and reports landed under `data/` cited by digest with no debate, because the
decision had already been written down. What remains genuinely open is the destination for a published weight,
which is now a hosting question with a known 17.9 MB input rather than an abstraction.
