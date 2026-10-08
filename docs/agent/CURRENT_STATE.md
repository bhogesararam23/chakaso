# Current state

Last updated: 2026-10-08, at commit `feat: generalize source identity beyond URLs`.

This file is the authority on what exists. If it disagrees with any other
document, this file is right and the other document is a defect.

A private planning note is never evidence that something is implemented.

## What exists

| Thing | Where | Notes |
| --- | --- | --- |
| Repository conventions | `.gitignore`, `.gitattributes`, `.editorconfig` | Private docx pack excluded by `.gitignore` and guarded by a test |
| License | `LICENSE` | Apache-2.0; rationale in ADR-0004 |
| Public documentation | `README.md`, `docs/*.md`, `docs/research/` | Specifications with mandatory status labels |
| Decision records | `docs/decisions/` | ADR-0001 to ADR-0010 |
| Contribution guide | `CONTRIBUTING.md` | |
| Agent contract | `AGENTS.md`, `docs/agent/` | |
| Python package | `src/chakaso/` | Installs; typed; `py.typed` ships |
| CLI | `src/chakaso/cli.py` | `--version`, `info`, `config show`, `chat` |
| Configuration | `src/chakaso/config/`, `configs/default.toml` | Typed schema, explicit loading, per-value provenance |
| Core primitives | `src/chakaso/core/` | Content-derived identifiers, SHA-256 hashing, error base |
| Model boundary | `src/chakaso/models/` | `LanguageModel` protocol, capabilities, registry, contract tests |
| Model implementations | `src/chakaso/models/deterministic.py` | A development double only. **No language model exists.** |
| Evidence records | `src/chakaso/evidence/` | `SourceRecord`, `EvidenceChunk`, `EvidencePack`, typed source references (web/file/text, ADR-0011), URL canonicalization, citation resolution |
| Chunking | `src/chakaso/retrieval/chunking.py` | Document text to evidence chunks: section-bounded, non-overlapping, deterministic (ADR-0010) |
| Conversation state | `src/chakaso/conversation/state.py` | Immutable, append-only turns with provenance; topic, entities and open questions |
| Conversation manager | `src/chakaso/conversation/manager.py` | Conducts one turn: context projection, model call, validation, evidence and citation recording. Transactional (ADR-0008) |
| Conversation shell | `chakaso chat` | Interactive and one-shot. States in its own output that the engine is a development double |
| Tests | `tests/` | 346 tests at the commit recorded above: package, CLI, configuration, primitives, model boundary, evidence, conversation state, conversation manager, chunking, repository hygiene. The count ages; the command does not. |
| CI | `.github/workflows/ci.yml` | Green on Python 3.11, 3.12, 3.13 |
| Retrieval, evidence store, correction | **do not exist** | Planned. Chunking is the only retrieval code |

## What works

- `python -m pip install -e ".[dev]"` installs the package and its dev tools.
- `python -m pytest` passes on Python 3.11, 3.12 and 3.13, which is the range CI
  enforces. Development on the newest interpreter alone is not sufficient: an
  unhashable dataclass default imported cleanly on 3.13 and failed at import on
  3.11, and CI found it rather than the local run.
- `python -m ruff check .`, `python -m ruff format --check .` and `python -m mypy src`
  pass under strict settings.
- `python -m chakaso --version` and `python -m chakaso info` report the version,
  Python version and platform.
- `python -m chakaso config show` prints every configuration value with the source
  it came from, and exits non-zero with a specific message when a configuration
  file is missing, malformed, or contains an unknown key or an out-of-range value.
- Source and chunk identifiers can be derived from content and validated from a
  string, which is the path a citation from generated text will take.
- `create_model(config)` builds the adapter named in configuration, and raises
  `UnknownModelError` listing the registered adapters when the name is wrong. The
  only adapter that exists is the deterministic development double.
- Asking the double to tokenize or produce structured output fails at the boundary
  with a message naming the model and the capability, rather than returning
  something that looks like an answer.
- A source record can be built from a URL and content, deriving its own identity and
  content hash, and can confirm later whether a given content still matches it.
- The same record type now covers sources that are not web pages: a local document is
  identified by a `file` reference and supplied text by a `text` reference (ADR-0011),
  while a web source's identifier is exactly what it was before. No document is read
  and nothing is fetched yet; a caller hands in the content.
- An evidence pack can be assembled from chunks and sources, and the references in an
  answer resolved against it: resolved citations, unknown identifiers and malformed
  identifiers are reported separately.
- A conversation can be held against the model boundary: `ConversationManager.send`
  records the user turn, projects the recent context, calls the model, validates the
  result, resolves evidence references, records the assistant turn and returns the new
  immutable state.
- A turn is transactional. If the model fails, returns nothing, or attributes its
  output to a different model, the conversation is unchanged and the failure keeps
  its own type (ADR-0008).
- Evidence supplied to `send` is what citations resolve against. A reference the
  model was not given is reported as unresolved and is never resolved to a source
  (ADR-0009).
- `python -m chakaso chat` holds a conversation: interactive on stdin, or one
  message with `--message`. It prints a notice on stderr naming the engine and
  saying that it is a development double, that nothing is retrieved, and that the
  conversation is not saved. The reply goes to stdout so the command stays
  scriptable.
- Document text can be split into evidence chunks with section paths, document
  positions and content-derived identifiers, and those chunks can be assembled into
  an `EvidencePack`. Nothing yet reads a document or ranks chunks.
- CI runs all of the above on three Python versions, with no secrets and no network
  access to a model provider.

**No language model is present in this repository.** The only implementation of the
model boundary is a development double declared as such in its own metadata, so a
conversation can be held but nothing intelligent is produced by it. Nothing is
retrieved, and no citation resolves unless a caller supplies evidence. A shell that
replies fluently invites the wrong conclusion, so `chat` says what it is before it
says anything else.

## What is tested

| Area | Coverage |
| --- | --- |
| Package installation | Version shape, distribution metadata agreement, console script entry point, `py.typed` |
| CLI | Version, help, unknown command, `info`, `config show` including failure paths |
| Configuration | Defaults, file layering and precedence, provenance, unknown keys, wrong types, boolean-vs-integer, ranges, name pattern, missing file, directory, invalid TOML, array values, immutability, schema/dataclass agreement, shipped file vs built-in defaults |
| Identifiers and hashing | Digest agreement with `hashlib`, UTF-8 handling, truncation bounds, part-separator ambiguity, derivation determinism, deduplication, content-change distinction, position and text sensitivity, malformed identifier rejection, ordering and hashing |
| Model boundary | The inherited contract suite (metadata, provenance of results, repeatability at zero temperature, empty-request rejection, length ceiling, unsupported-capability failures, declared capabilities being implemented), plus capability reconciliation, parameter validation, registry failure paths and the double's documented behaviour |
| Evidence | Canonicalization idempotence and non-merging, tracking-parameter removal, scheme and credential refusal, IPv6 handling, record immutability, naive-timestamp rejection, content-hash validation, change detection, pack validation, and citation resolution including fabricated and malformed references |
| Conversation manager | Construction and context bounds, dependency injection through the model boundary, first and follow-up turns, context projection limits, previous state preserved, failures leaving state unchanged for the next turn, model errors propagating untranslated, clock regressions, evidence and citation provenance per turn, evidence not leaking between turns, and end-to-end wiring against the deterministic double |
| Repository invariants | Private pack never tracked, `.gitignore` rule present, no secret-shaped files, no tracked file over 1 MiB, no commercial provider dependency, ADR numbering and indexing, documentation links resolve |

## What is experimental

Nothing. There is no code whose approach is unsettled; the code that exists does
one small thing each.

## What is not implemented

- Query planner: nothing decides whether retrieval would help
- Reading a document from a local path. The identity that blocked it is now settled
  (ADR-0011); what remains is the reader and its filesystem-safety rules.
- Ranking, indexing and embeddings: chunks cannot be scored or searched yet
- Fetching and parsing web content
- Claim-level support checking. Citation *reference* validation exists; whether cited
  evidence actually supports a claim does not.
- Reassessment and the correction loop
- Evaluation harness, benchmarks and metrics
- Tokenizer, dataset pipeline, model code, training loop
- A local inference adapter, and any model weights of any size
- Persistence of any kind. Conversation state lives in memory for the duration of the
  process.

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
| ADR-0008 | A turn either completes or the conversation is unchanged |
| ADR-0009 | A reference the model was not given is recorded, not fatal |
| ADR-0010 | Chunks are section-bounded and do not overlap |
| ADR-0011 | Source identity is a typed reference, not a URL |

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

P2 is complete: the conversation manager works and a shell exercises it end to end.
The blocking question for retrieval — what identifies a source with no URL — is now
answered (ADR-0011), so a local document and supplied text have an identity and a
record. The next unit is the first of P3, **local document ingestion and a lexical
retrieval baseline**: normalizing content deterministically, reading documents that are
already on disk into the records now defined, splitting them into evidence chunks with
stable identifiers, and ranking them for a query without embeddings. That is what will
supply the `EvidencePack` that `ConversationManager.send` already accepts, and it is
deliberately the smallest retrieval step — no network, no model, no index server.

Before live fetching (P4), the fetch policy has to be written. It is a blocking open
question in [`../research/open-questions.md`](../research/open-questions.md), and the
security posture in [`../retrieval.md`](../retrieval.md) is a list of threats with no
corresponding limits.

See [`ACTIVE_TASK.md`](ACTIVE_TASK.md) for the definition of done.