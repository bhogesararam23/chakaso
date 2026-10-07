# Current state

Last updated: 2026-10-07, at commit `feat: add core identifier and hashing primitives`.

This file is the authority on what exists. If it disagrees with any other
document, this file is right and the other document is a defect.

A private planning note is never evidence that something is implemented.

## What exists

| Thing | Where | Notes |
| --- | --- | --- |
| Repository conventions | `.gitignore`, `.gitattributes`, `.editorconfig` | Private docx pack excluded by `.gitignore` and guarded by a test |
| License | `LICENSE` | Apache-2.0; rationale in ADR-0004 |
| Public documentation | `README.md`, `docs/*.md`, `docs/research/` | Specifications with mandatory status labels |
| Decision records | `docs/decisions/` | ADR-0001 to ADR-0007 |
| Contribution guide | `CONTRIBUTING.md` | |
| Agent contract | `AGENTS.md`, `docs/agent/` | |
| Python package | `src/chakaso/` | Installs; typed; `py.typed` ships |
| CLI | `src/chakaso/cli.py` | `--version`, `info`, `config show` |
| Configuration | `src/chakaso/config/`, `configs/default.toml` | Typed schema, explicit loading, per-value provenance |
| Core primitives | `src/chakaso/core/` | Content-derived identifiers, SHA-256 hashing, error base |
| Tests | `tests/` | 100 tests across package, CLI, configuration, primitives and repository hygiene |
| CI | `.github/workflows/ci.yml` | Green on Python 3.11, 3.12, 3.13 |
| Model interface | **does not exist** | Next unit |
| Retrieval, evidence store, correction | **do not exist** | Planned |

## What works

- `python -m pip install -e ".[dev]"` installs the package and its dev tools.
- `python -m pytest` runs 100 tests. All pass.
- `python -m ruff check .`, `python -m ruff format --check .` and `python -m mypy src`
  pass under strict settings.
- `python -m chakaso --version` and `python -m chakaso info` report the version,
  Python version and platform.
- `python -m chakaso config show` prints every configuration value with the source
  it came from, and exits non-zero with a specific message when a configuration
  file is missing, malformed, or contains an unknown key or an out-of-range value.
- Source and chunk identifiers can be derived from content and validated from a
  string, which is the path a citation from generated text will take.
- CI runs all of the above on three Python versions, with no secrets and no network
  access to a model provider.

No conversation can be held, nothing is retrieved, no answer is generated and no
citation is resolved. The `model.adapter` setting names an adapter that does not
exist yet, so any code attempting to resolve it would fail — and no such code
exists.

## What is tested

| Area | Coverage |
| --- | --- |
| Package installation | Version shape, distribution metadata agreement, console script entry point, `py.typed` |
| CLI | Version, help, unknown command, `info`, `config show` including failure paths |
| Configuration | Defaults, file layering and precedence, provenance, unknown keys, wrong types, boolean-vs-integer, ranges, name pattern, missing file, directory, invalid TOML, array values, immutability, schema/dataclass agreement, shipped file vs built-in defaults |
| Identifiers and hashing | Digest agreement with `hashlib`, UTF-8 handling, truncation bounds, part-separator ambiguity, derivation determinism, deduplication, content-change distinction, position and text sensitivity, malformed identifier rejection, ordering and hashing |
| Repository invariants | Private pack never tracked, `.gitignore` rule present, no secret-shaped files, no tracked file over 1 MiB, no commercial provider dependency, ADR numbering and indexing, documentation links resolve |

## What is experimental

Nothing. There is no code whose approach is unsettled; the code that exists does
one small thing each.

## What is not implemented

- Language-model interface, registry and every adapter
- Conversation state and the conversation manager
- Query planner
- Retrieval: fetch, parse, chunk, rank, index, embeddings
- Evidence records and citation resolution
- Grounding and citation validation
- Reassessment and the correction loop
- Evaluation harness, benchmarks and metrics
- Tokenizer, dataset pipeline, model code, training loop
- Any model weights, of any size

## Decisions made

| Record | Decision |
| --- | --- |
| ADR-0001 | No required commercial inference or search dependency at runtime |
| ADR-0002 | Application code depends on a language-model boundary, not a model |
| ADR-0003 | The retrieval layer owns evidence identifiers; the model only references them |
| ADR-0004 | Apache-2.0 for code; weights and datasets need their own policy before release |
| ADR-0005 | `src` layout, with training code as a module inside one distribution |
| ADR-0006 | TOML configuration parsed with the standard library, read-only |
| ADR-0007 | Evidence identifiers are derived from content and the canonical URL, not assigned at random |

Two process facts are recorded outside the ADR series because they are repository
history rather than architecture:

- The repository history was restarted from an empty tree; the previous head is
  preserved as tag `archive/bootstrap-attempt-1`
  ([`../internal/engineering/repository-history.md`](../internal/engineering/repository-history.md)).
- The private planning pack lives in `docs/*.docx`, is git-ignored, and is never
  committed.

## Known gaps and risks

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md). K-001 still stands and is the important
one: nothing in this repository has been executed against a real workload, so every
architectural decision made so far rests on reasoning rather than measurement.

## Next step

The language-model boundary (ADR-0002): a narrow interface, a declared-capability
model, a registry with a single selection point, contract tests that any
implementation must satisfy, and one development double so that higher-level
behaviour can be tested without weights. See [`ACTIVE_TASK.md`](ACTIVE_TASK.md).