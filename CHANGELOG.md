# Changelog

Notable changes to Chakaso, newest first. Nothing has been released; every entry
below is pre-release.

Entries describe what changed and why it matters to someone using or evaluating
the project. They do not list edited files, and they do not describe progress
without a result. Where an entry records a capability, it states whether that
capability is implemented or still planned.

## Unreleased

### 2026-10-08 — Generalized source identity

**Added**

- `chakaso.evidence.identity`: a `SourceReference` and a `SourceKind` (`web`, `file`,
  `text`) that decide what a source *is*, kept deliberately separate from whether it
  can be fetched. `web_reference`, `file_reference` and `text_reference` build one.
- `SourceRecord.create_file` and `SourceRecord.create_text`, so a local document and
  supplied text are records now rather than a later step.
- ADR-0011 records the decision, and answers the open question that had blocked local
  ingestion. The distinction it turns on: a source reference is an identity, a fetchable
  URL is a permission, and the two were living in one function.

**Changed**

- `SourceRecord.canonical_url` became `canonical_reference`, and the record carries a
  `kind`. `derive_source_id` now takes a canonical reference. **The identifier formula
  and every web-source identifier are unchanged** (ADR-0007 is untouched): a web
  reference's canonical form is exactly `canonicalize_url`'s output, and a `file` URI or
  a `text` reference can never equal a canonical URL, so folding the three kinds into one
  formula cannot collide across them.
- `canonicalize_url` is now used only for web references. Identifying a local source no
  longer routes through it, so widening source identity no longer widens what the future
  fetcher is permitted to retrieve.

**Notes**

- Still no ingestion and no retrieval. A caller hands the content to
  `create_file`/`create_text`; nothing opens a file, reads a path from disk, or touches
  the network. The record can name a local source before anything can read one.
- File identity is machine-local (a `file` URI names a path on one filesystem), and a
  relative path resolves against the working directory, so absolute paths give the
  stablest identity.

### 2026-10-07 — Document chunking

**Added**

- `chakaso.retrieval.chunk_document`: splits document text belonging to an
  already-identified source into `EvidenceChunk` records. Chunks carry a section
  path, a position in document order, and a content-derived identifier.
- `split_sections`: Markdown-heading splitting with nested heading paths, so a
  citation into a subsection says where the subsection sits.
- `ChunkingConfig`, with a target size and a hard ceiling, in characters.
- ADR-0010 records the chunking decision. Chunks never cross a section boundary and
  do not overlap. Overlap was declined deliberately: with content-derived
  identifiers, overlapping chunks would report one sentence under several
  identifiers, which inflates every retrieval metric that counts chunks and makes
  "how many chunks mention this" meaningless. The boundary failure that overlap hides
  is now something an evaluation can measure.

**Notes**

- Sizes are characters, not tokens. `token_count` on a chunk stays unset, because
  there is no tokenizer and a character count in a field named `token_count` would be
  a lie a later reader would trust.
- Chunking does not read files, does not fetch and does not rank. Nothing here has
  touched the network.
- Reading a document from a local path turned out to be blocked: a document with no
  URL has no identity under ADR-0007, and `canonicalize_url`'s scheme check currently
  doubles as the fetch permission list. Widening it in a file-reading module would
  quietly widen what the future fetcher may retrieve, so the question is recorded in
  `docs/research/open-questions.md` and no ingestion code was written.

### 2026-10-07 — Local conversation shell

**Added**

- `chakaso chat`: a conversation shell that runs the whole stack end to end —
  configuration, context projection, the model boundary, validation, evidence and
  citation recording. Interactive on standard input, or one message and exit with
  `--message`, which is what a script wants. `:quit`, `:exit` and end of input all
  end a session; blank lines are ignored rather than sent; a failed turn is reported
  and the session continues, because a failure leaves the conversation unchanged.
- The shell prints a notice on standard error before the first reply, naming the
  engine, saying that it is a development double rather than a language model, and
  saying that nothing is retrieved, nothing is corrected and the conversation is not
  saved. It is derived from the model's own `development_double` metadata rather than
  from a hard-coded string, so it cannot drift out of date while the engine stays
  fake.
- The reply goes to standard output and everything else to standard error, so
  `--message` is usable in a pipeline.
- The shell reports two conditions a reader would otherwise not know about: a reply
  truncated at the length ceiling, and a reference to evidence the model was not
  given, which is listed and never resolved.

**Changed**

- The CLI's own description no longer says there is nothing to run, because there is
  now something to run.
- `--config` is defined once and shared by `config show` and `chat`, so the two cannot
  drift apart in behaviour or in help text.

**Note on scope**

`chakaso chat` is not a chatbot. Every reply it produces is the development double's
fixed text, and the command says so in its own output. There is no language model, no
retrieval, no fetching, no correction loop and no evaluation.

### 2026-10-07 — Conversation manager

**Added**

- `chakaso.conversation.ConversationManager`: the orchestration boundary over the
  existing conversation state. One turn rejects an empty message, appends the user
  turn to a working copy, projects the recent conversation into model messages, calls
  the model through the `LanguageModel` boundary, rejects a response that is empty or
  attributed to a different model, resolves evidence references, appends the
  assistant turn, and adopts the new state.
- A turn is transactional (ADR-0008). If any step fails, the conversation is
  unchanged, so a user turn is never left stranded without a reply and a retry is a
  first attempt rather than a follow-up to a turn that never happened.
- Distinguishable failures: `InvalidUserInputError` for an empty message,
  `InvalidGenerationError` for an unusable response, and model errors propagated
  untranslated rather than folded into a generic conversation error.
- `Reply`, carrying the new state, the assistant turn, the generation result (model
  identity and whether it was truncated) and the citation resolution.
- A reference to evidence the model was not given is recorded, never resolved, and
  reported to the caller (ADR-0009). Resolved citations, unknown references and
  malformed references stay three separate results.
- Context projection bounded by `model.max_context_turns`, a new configuration
  setting. It bounds growth; it is a turn ceiling, not a token budget, and is
  documented as the stopgap it is.
- `new_conversation_id()`, and conversation-layer error types in their own module
  alongside the other packages' error modules.

**Changed**

- `ConversationManager.send` accepts an `EvidencePack`. Nothing produces one yet —
  there is no retrieval — but the parameter exists now because adding it later would
  touch every call site, and every call site is somewhere the citation rule could be
  forgotten.
- The configuration test that restated the `ModelConfig` field names now checks them
  against the schema, so adding a setting is no longer a two-place edit that trains
  people to update a test without reading it.

**Verification**

- A new repository-hygiene test reads the syntax tree of every module under `src/`
  and fails if any module outside the model registry imports a concrete adapter,
  which is ADR-0002's guarantee. It was confirmed to fail against a deliberate
  violation before being kept.
- Argument-unused linting is relaxed for `tests/` only, because a test double that
  ignores the messages it was given is usually the point, and per-line suppressions
  would outnumber the code they annotate.

**Note on scope**

There is still no language model. The only implementation of the boundary is the
deterministic development double, so a conversation can be held and nothing
intelligent is produced by it. No retrieval, no fetching, no correction, no
evaluation, no tokenizer and no training.

### 2026-10-07 — Python 3.11 compatibility fix

**Fixed**

- `SourceRecord.metadata` used a shared `mappingproxy` as a dataclass field
  default. Python 3.11 rejects *any* unhashable default outright, and a
  mappingproxy is unhashable, so `chakaso.evidence` failed at import on 3.11 while
  working on 3.12 and 3.13. It now uses `default_factory`.

**Why this is recorded**

It was caught by CI rather than locally, because development was happening on
Python 3.13 and the failure was an import-time error rather than a wrong result.
Three consecutive pushes went out with red CI before it was noticed, which is a
process failure as much as a code one: a push is not finished until the run is
checked.

Two things changed as a result. Version-suffixed virtual environments (`.venv311`,
`.venv312`) are now documented for checking the floor locally, and `.gitignore`
covers them, so the suite runs on all three supported versions before a push rather
than only in CI.

### 2026-10-07 — Conversation state

**Added**

- `chakaso.conversation`: immutable, append-only conversation state. Turns record
  when they happened, which evidence was supplied for them and which sources the
  answer actually referenced — kept separate, because "which of those did you use?"
  is unanswerable if only the union is stored.
- `Conversation.to_model_messages()` is the single mapping from conversation state
  to what a model sees, with an optional turn limit. A model is given roles and text
  and nothing else.
- State carries the active topic, entities and open questions, and exposes the
  sources the conversation has touched, most recent first, which is what a follow-up
  question has to resolve against.
- Enforced invariants: timestamps must be timezone-aware, history is append-only for
  a given conversation, and an answer cannot cite a source it was not given. That
  last rule is the ADR-0003 rule applied to the recorded state rather than to the
  text.

**Not added, deliberately**

Claims and an answer record. Both belong to the reassessment path, which does not
exist yet; adding them here would bake the shape of correction into conversation
before anything has decided what a claim is.

### 2026-10-07 — Evidence records and citation resolution

**Added**

- `chakaso.evidence.SourceRecord`: an immutable record of one canonical URL with
  one content body, carrying a content-derived identifier, host, retrieval time,
  optional publication time, content hash and extraction metadata. Built through
  `create()`, so an identifier cannot disagree with the content it names.
- `EvidenceChunk`: an immutable evidence unit with a derived identifier, position,
  optional section, and retrieval and rerank scores recorded but never presented as
  confidence.
- URL canonicalization, documented as part of the identifier contract because
  identifiers are derived from it (ADR-0007). It removes fragments, default ports,
  trailing root dots, trailing slashes and campaign parameters, and refuses
  non-HTTP schemes and URLs with embedded credentials.
- `EvidencePack`: the bounded set of evidence supplied to one generation call,
  which validates that every chunk's source is present and that identifiers are
  unique, and drops sources no chunk refers to.
- `resolve_citations()`: a validation gate rather than a lookup. A well-formed
  identifier that was supplied resolves to its chunk and source; one that was not
  supplied is recorded as an unknown reference; an identifier-shaped string that is
  not valid is recorded as a malformed reference. Nothing outside the pack is ever
  resolved, which is the rule ADR-0003 exists to enforce.

**Fixed**

- `canonicalize_url` dropped the brackets from IPv6 literals, producing a string
  that parses as a different URL.
- `EvidenceChunk.create` raised an identifier error rather than a record error for
  a negative position, so the same defect had two different exception types
  depending on which entry point was used.
- Citation resolution classified references as unknown before checking whether they
  were valid, which made the malformed bucket unreachable. Inventing a source and
  fumbling the identifier format are different defects with different fixes, and
  they are now counted separately.

**Note on scope**

Nothing has been fetched or indexed. These records are built from content a caller
already has, so the evidence layer is implemented without any retrieval.

### 2026-10-07 — Language-model boundary

**Added**

- `chakaso.models`: the boundary application code depends on, implementing
  ADR-0002. `LanguageModel` requires only `metadata` and `generate`; optional
  behaviour is declared as a `Capability` and implemented on a separate protocol,
  with `tokenize()` and `generate_structured()` reconciling the declaration and the
  implementation. A model that declares a capability it does not implement is
  reported as a defective implementation rather than as caller error.
- `ModelRegistry` and `create_model(config)`: the single place where configuration
  becomes a model. Re-registering a name is rejected rather than allowed to shadow.
- `chakaso.models.deterministic`: a development double that produces fixed text,
  declares no capabilities, and marks itself with
  `metadata.development_double = True`. It is not a language model and says so in
  its own metadata, so a report can state plainly that no model was involved.
- Contract tests that every implementation of the boundary inherits, so a new
  adapter cannot pass its own tests while violating the interface.

**Note on scope**

The only implementation of the boundary is the development double. There is no
trained Chakaso model and no local inference adapter, so nothing in this repository
generates language. Retrieval, correction and evaluation still do not exist.

### 2026-10-07 — Core primitives

**Added**

- `chakaso.core.identifiers`: `SourceId` and `ChunkId`, validated on construction
  and derived from content rather than assigned at random (ADR-0007). The same URL
  with the same content yields the same identifier, so repeated retrieval
  deduplicates; the same URL with different content yields a different identifier,
  so an earlier answer keeps pointing at the bytes it actually used.
- `chakaso.core.hashing`: SHA-256 over text or bytes, with a truncated form for
  identifiers and an explicit part separator so that concatenating parts before
  hashing cannot be ambiguous.
- `chakaso.core.errors`: a base error type for deliberately raised failures, with
  validation errors also deriving from `ValueError` so untrusted input can be
  caught with the built-in type.

**Note on scope**

Still no model interface, retrieval, correction loop, tokenizer or trained model.

### 2026-10-07 — Engineering foundation

**Added — package**

- `src/chakaso/` package with a `src` layout, installable and typed, shipping a
  `py.typed` marker (ADR-0005). The layout is deliberate: it makes an uninstalled
  import fail rather than silently testing the working tree.
- A thin CLI: `--version`, `info`, and `config show`. Argument parsing only; the
  CLI holds no behaviour of its own.
- No runtime dependencies. The foundation uses the standard library only.

**Added — configuration**

- Typed configuration schema with declared types, ranges and name patterns,
  loaded explicitly from TOML with the standard library (ADR-0006).
- Per-value provenance: every configuration value reports the file it came from,
  or that it came from the built-in default. `chakaso config show` prints it.
- Unknown keys, wrong types, out-of-range values, arrays and unsupported TOML
  types are errors naming the file, the key and the expectation, rather than
  warnings. A typo that is silently ignored produces a run that does not use the
  settings its author believes it uses.
- `configs/default.toml`, whose values a test asserts match the built-in defaults
  so the two cannot drift.

**Added — verification**

- CI running formatting, linting, type checking and tests on Python 3.11, 3.12 and
  3.13, with no secrets and no access to a model provider.
- Repository-hygiene tests enforcing rules that are otherwise broken by accident:
  the private planning pack never becomes tracked, the ignore rule stays present,
  no secret-shaped or oversized files are tracked, no dependency is a commercial
  inference or search provider, decision records are numbered and indexed
  contiguously, and documentation links resolve.

**Fixed**

- Two documentation links pointed at paths that do not exist. Both were found by
  the new link check rather than by reading: `docs/agent/CONVENTIONS.md` pointed
  at `../../decisions/ADR-0006-toml-configuration.md` instead of `../decisions/`,
  and `docs/getting-started.md` linked to the `configs/` directory.

**Note on scope**

There is still no conversational system: no model interface, no retrieval, no
correction loop, no tokenizer and no trained model. No benchmark has been run and
no result is reported anywhere in this repository. Configuration is the only
component implemented so far.

### 2026-10-07 — Project foundation

**Added — repository**

- Apache-2.0 license for code, documentation and configuration. Trained weights,
  tokenizers and datasets are explicitly not covered; they will carry their own
  terms when they exist.
- `.gitignore` that excludes the author's private planning pack (`docs/*.docx`)
  along with virtual environments, tool caches, model weights, dataset shards and
  local experiment outputs.
- `.gitattributes` normalizing text to LF in the repository and marking binary
  formats, so line-ending conversion cannot corrupt weights or documents.
- `.editorconfig` for consistent indentation and line endings across editors.

**Added — decision records**

- ADR-0001: the initial runtime has no required commercial inference or search
  dependency. Network access is allowed for fetching public web content, which is
  processed locally. External providers may exist later only as optional adapters.
- ADR-0002: application code depends on a narrow language-model interface, not a
  concrete model. Implementations declare capabilities; unsupported requests fail
  at the boundary.
- ADR-0003: the retrieval layer owns source and chunk identifiers. Generated text
  may only reference identifiers it was given; an unknown identifier is recorded
  as an error and never resolved to a URL.
- ADR-0004: Apache-2.0, chosen for compatibility with the intended dependency stack
  and for its explicit patent grant.

**Added — public documentation**

- `README.md` stating what Chakaso is, what is implemented, and what is not.
- Architecture, transparency, retrieval, correction, training, evaluation and
  roadmap documents. These are specifications: they describe intended behaviour and
  label the status of every component.
- Research foundation: a research log, six falsifiable hypotheses, an open-questions
  list, and a literature file whose URLs were each fetched and verified before
  publication.
- Contribution guide.

**Added — internal and agent documentation**

- Agent contract (`AGENTS.md`) and agent context: current state, conventions, active
  task and known issues.
- Internal engineering notes: repository history, artifact policy and tooling
  rationale.
- Experiment record format.

**Note on scope**

There is no runnable system, no retrieval, no correction loop, no tokenizer and no
model. No benchmark has been run and no result is reported anywhere in this
repository.

**Note on history**

The repository's previous history, whose head tree was already empty, was replaced
with a fresh root commit. The previous head is preserved as tag
`archive/bootstrap-attempt-1`. See `docs/internal/engineering/repository-history.md`.
