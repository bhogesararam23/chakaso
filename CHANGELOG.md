# Changelog

Notable changes to Chakaso, newest first. Nothing has been released; every entry
below is pre-release.

Entries describe what changed and why it matters to someone using or evaluating
the project. They do not list edited files, and they do not describe progress
without a result. Where an entry records a capability, it states whether that
capability is implemented or still planned.

## Unreleased

### 2026-10-08 — Reproducible experiments and a regression baseline

**Added**

- `chakaso.experiments`: an `Experiment` (an identifier, a name, a **hypothesis**, the development
  benchmark it runs, and a configuration) and a content-fingerprinted `ExperimentResult` that keeps
  environment metadata separate from the measured numbers. `run_experiment` drives a real
  development benchmark and records its metrics, dataset identity/version and the evaluator(s) that
  decided; `result_id` fingerprints everything that determines the number and ignores the timestamp
  and host, so the same experiment reproduces identically on the same code. `compare_results` is a
  deterministic regression baseline that refuses to compare results defined differently (different
  benchmark version, configuration, evaluator or experiment) rather than subtracting unlike metrics.
  `baseline_experiment` supplies the canonical retrieval and correction baselines.
- `chakaso experiment run <retrieval|correction> [--json]` prints a reproducible experiment result.

**Notes**

- No experiment tracker, database or external service is introduced (ADR-0020, ADR-0001): results
  are serializable records a person commits, not persisted by the runtime. No statistical
  significance machinery — a delta over a handful of synthetic fixtures is not a confidence interval.

### 2026-10-08 — Semantic grounding boundary (fixture only)

**Added**

- `chakaso.grounding` gains a semantic grounding *boundary*: a per-claim `SemanticJudge` protocol
  (`judge(claim, evidence) -> SemanticEvaluation` over supported / contradicted / uncertain /
  not_evaluated), a `SemanticGroundingEvaluator` that adapts a judge onto the existing
  `GroundingEvaluator`, and a `FixtureSemanticJudge` that replays caller-supplied relations
  (ADR-0019). Answer evaluation and reassessment accept a semantic evaluator with no signature
  change, and grounding stays a separate dimension from citation.

**Notes**

- The only judge is a fixture, named `fixture-semantic-0.1` so a report cannot read it as
  understanding. No real semantic capability is claimed, no numeric confidence is introduced, and a
  named conflict still records no winner — resolving a conflict stays correction's job (ADR-0015).

### 2026-10-08 — Correction development benchmark

**Added**

- `chakaso.benchmark.correction`: `CorrectionCase` / `CorrectionDataset` apply the retrieval
  benchmark's versioning discipline (ADR-0013) to correction — content-derived case versions and a
  dataset fingerprint make a judgement edit a visible version change. Six synthetic cases cover every
  reachable decision (retain, correct, qualify, needs_review, abstain, and an irrelevant-evidence
  retain). `CorrectionBenchmarkRunner` builds real evidence packs from the shared fixture corpus and
  runs the same `reassess` the application uses, choosing structural or fixture-semantic purely from
  what a case declares; reports (text and deterministic JSON) give per-case expected-vs-actual and the
  aggregate metrics, and a pinned-fingerprint regression guards the curated set.
- `chakaso benchmark` now takes a subject: `retrieval` or `correction` (`--json` for both).

### 2026-10-08 — Persistent answers and application-level reassessment

**Added**

- `chakaso.answers`: a first-class `AnswerRecord` — model identity, evidence supplied, cited
  sources/chunks, claims, evaluation and a `correction_of` link — referencing the other layers by
  identifier rather than duplicating them. `AnswerId` is a per-event typed identifier, not a content
  hash, so a corrected answer is always distinguishable from the one it supersedes (ADR-0017).
  `AnswerStore` is the persistence boundary with an in-memory implementation that is append-only and
  refuses to overwrite a saved answer or to link a correction to a missing or cross-conversation
  answer (ADR-0018).
- `ConversationManager` optionally records a turn's `AnswerRecord` and saves it *before* adopting
  the new conversation, so a persistence failure aborts the turn and leaves the conversation
  unchanged — extending ADR-0008's atomicity to the answer history. Without a store, behaviour is
  unchanged.
- `chakaso.correction` gains an application-level operation, `reassess_stored_answer` /
  `reassess_and_record`, that reads a prior answer by identifier, re-grounds its recorded claims
  against caller-composed evidence, and pairs the decision with a `CorrectionRecord` linked to it.
  `reassess_followup` / `merge_evidence` compose standing plus new evidence deterministically.

**Notes**

- Nothing generates revised prose (there is no model), nothing runs reassessment automatically, and
  nothing is written to disk — the store is in-memory and lost on process exit. A prompt-injection and
  untrusted-text identity guard and a no-network import check were extended to cover the new layers.

### 2026-10-08 — Benchmark from the command line

**Added**

- `chakaso benchmark [--json] [--top-k N]`: runs the retrieval development benchmark over its
  synthetic fixtures from the command line and prints the report — human-readable text, or the
  deterministic machine-readable JSON with `--json`.

**Notes**

- The command is thin wiring: it calls `load_development_benchmark`, `RetrievalService` and
  `RetrievalBenchmarkRunner` and prints the result; all report wording and the "development
  benchmark" labelling live in `chakaso.benchmark.report`. It scores retrieval only, generates no
  answer (there is no language model), reads no files and touches no network, and its output is
  deterministic.

### 2026-10-08 — Correction foundation (reassessment, decision, record, metrics)

**Added**

- `chakaso.correction`: the reassessment foundation `docs/correction.md` specifies, built on the
  claim and grounding boundaries rather than a parallel one (ADR-0016).
  - `decide_claim` maps a claim's prior status and its status under new evidence onto a per-claim
    **retain / qualify / correct** disposition under a rule written in full: a claim is corrected
    only when the supplied evidence *contradicts* it, never because it was produced first or has
    aged — ADR-0015's refusal to rank by recency or authority is inherited.
  - `decide_answer` aggregates the dispositions into one answer-level `CorrectionDecision` under a
    fixed precedence — a contradiction outranks all; an unadjudicated conflict or an unsettled
    required claim goes to **needs_review** rather than guessing; lost support **qualifies**; total
    loss of grounding **abstains**; otherwise **retain**.
  - `reassess` re-runs a `GroundingEvaluator` over a prior answer's claims against a caller-composed
    pack and returns the `Reassessment`.
  - `record_reassessment` fixes a reassessment into an immutable, content-derived `CorrectionRecord`
    naming the answer it supersedes; `is_revision` reports a supersession only when the decision
    changed *and* a prior is named.
  - `correction_metrics` reports correction success, unjustified-persistence and unnecessary-revision
    rates over a caller-labelled set, exposing the raw counts so a rate cannot hide a thin
    denominator; a rate with no cases is `None`, never a fabricated `0.0` or `1.0`.
  - `reassess_followup` composes the standing evidence a prior answer rested on with the new
    evidence now in hand (`merge_evidence` unions packs de-duplicated and order-independent) and
    hands the claims to `reassess`. It is a convenience over the existing boundaries, not a new
    loop, and nothing in the runtime calls it automatically.

**Notes**

- Correction decides and records; it does not rewrite the answer. With no language model, `summary`
  is a structural tally rather than a revised sentence, the revised prose stays a model step that
  does not exist, nothing is persisted (there is no `AnswerRecord`), and the loop is not automatic —
  a caller invokes reassessment on evidence it supplies.
- The metrics are decision-agreement rates against a supplied label, not a claim that a revised
  answer is correct. No correction benchmark has been run and no measured correction result exists.
- ADR-0016 records the decision rule and its honesty constraints.

### 2026-10-08 — Claim, citation, grounding and answer-evaluation foundation

**Added**

- `chakaso.claims`: an immutable `Claim` with a content-derived `ClaimId` and a `ClaimStatus`
  — supported / unsupported / contradicted / uncertain / not_evaluated — that records how the
  *supplied evidence* bears on a claim, never whether it is true (ADR-0014). A replaceable
  `ClaimExtractor` boundary decomposes an answer: `StructuredClaimExtractor` builds claims a
  caller supplies and `SentenceClaimExtractor` is a documented deterministic heuristic, so a
  trained model can implement the same boundary later without touching callers.
  `ClaimEvidenceLink`/`links_from_claims` project claims into inspectable (claim, chunk)
  citations.
- `chakaso.citation`: structural citation validation. Each citation is classified **valid**
  (points at supplied evidence), **unknown** (names a chunk that was not supplied — a
  fabrication counted, never resolved, per ADR-0003), or **irrelevant** (present but outside a
  case's expected evidence); a required claim with no valid citation is **uncited**.
  `citation_metrics` reports structural citation precision and recall alongside those counts.
- `chakaso.grounding`: a `GroundingEvaluator` boundary returning a per-claim `GroundingResult`
  (ADR-0015). `StructuralGroundingEvaluator` emits only supported / unsupported / not_evaluated
  and cannot invent a contradiction; `ManualGroundingEvaluator` is the only path on which
  contradicted or uncertain appear and stands in for a human label or a future model. A
  `Contradiction` names no winner, and `SourceAuthority`/`most_recent_wins` are caller-supplied
  precedence hooks the engine never applies automatically.
- `chakaso.evaluation.evaluate_answer`: combines the citation and grounding results into one
  answer view that keeps the dimensions **separate** — citation metrics, grounding statuses,
  `evidence_coverage` (the grounded-claim support rate) and any contradictions — and
  deliberately computes no single merged "quality score." `is_grounded` holds only when every
  claim is supported, nothing is contradicted and no citation points outside the pack.

**Notes**

- These are structural primitives, not an understanding of meaning: "supported" and "valid"
  are defined by citation presence and a case's expected evidence, never by a judgement that a
  source proves a claim. Contradiction and uncertainty require a supplied judgement and are
  never detected automatically. No language model is in the loop — claims and evidence are
  supplied by a caller or fixture.
- No answer-quality result is reported. There is still no grounded-answer or correction
  benchmark and no measured model-quality number; this is the evaluation machinery correction
  will later build on, not a scored outcome. The reassessment and correction loop itself does
  not exist: nothing compares a previous answer against new evidence or records a revision.
- ADR-0014 records the claim decision and ADR-0015 the grounding boundary.

### 2026-10-08 — Retrieval development benchmark

**Added**

- `chakaso.benchmark`: a versioned benchmark the retrieval metrics can actually be run
  against. `BenchmarkCase` names gold (acceptable) and forbidden (unacceptable) evidence by
  the same `ChunkId`/`SourceId` retrieval returns, plus required/forbidden claims, an
  expected behaviour (answer/qualify/abstain) and a citation requirement; a forbidden set is
  what catches citation laundering. A strict loader parses JSONL cases plus a `manifest.toml`,
  refusing unknown/missing/malformed fields by name.
- Identity and versioning (ADR-0013): a readable `CaseId`/`DatasetId` slug that survives
  edits, and a content-derived version/fingerprint that cannot drift from what it labels.
- A curated synthetic fixture corpus and `development_benchmark`, whose gold evidence is
  resolved by ingesting the same corpus the runner searches, so identifiers always match and
  the build is fully deterministic.
- `RetrievalBenchmarkRunner` scores the retrieval service over a dataset into recall@k,
  precision@k, MRR, hit rate, forbidden hits, false retrievals and duplicates, keeping
  scored and abstention cases distinct. `format_report`/`report_json` render a run with its
  provenance, labelled a development instrument.

**Notes**

- This measures retrieval only, against a small hand-built synthetic corpus. It is a
  development instrument, not a scientific benchmark: no confidence interval or significance
  test is computed, no answer is evaluated, and no model-quality number is reported.
- A pinned-fingerprint regression fails if any fixture or case judgement changes, forcing a
  deliberate dataset-version bump rather than a silent edit.

### 2026-10-08 — Retrieval evaluation metric functions

**Added**

- `chakaso.evaluation`: pure, deterministic metric functions — `recall_at_k`,
  `precision_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`, `duplicate_count`,
  `unresolved_reference_count`, and a `measure_latency` observation helper.

**Notes**

- These are measurement *hooks*, not a benchmark. There is still no evaluation dataset, no
  task version and no measured number anywhere in the repository; inventing a figure would
  be the fabrication the project forbids. The functions compute well-defined quantities from
  a ranked result and a relevance judgement, ready for when a benchmark supplies the
  judgements.
- Metrics that need a ground-truth definition first — evidence coverage, NDCG — are not
  invented into a formula; they wait for the benchmark.

### 2026-10-08 — Fetched web content becomes evidence

**Added**

- `chakaso.retrieval.html.parse_html`: a narrow HTML reader (standard-library
  `HTMLParser`, not a browser) that extracts readable text, captures the `<title>`, maps
  headings to Markdown so the section-bounded chunker applies, decodes entities, and drops
  `<script>`/`<style>`/`<noscript>`/`<template>`/`<svg>`.
- `chakaso.retrieval.web.ingest_acquired`: turns an `AcquiredSource` into the same
  `IngestedDocument` a local file produces — decode, parse, normalize, chunk — so web and
  local evidence share one representation. HTML and plain text are accepted; any other
  media type is refused rather than mangled.
- `chakaso.retrieval.cache`: an in-memory `AcquisitionCache` keyed by canonical URL (with
  an optional age limit) and a `CachingFetcher` that wraps any `Fetcher` to avoid
  refetching.

**Security posture**

- Fetched text is data, never a directive. A prompt-injection line inside a page is stored
  verbatim as an evidence chunk and is never executed, never changes policy, and never
  assigns identity — a web source is named by the validated canonical URL, not by anything
  the page claims about itself (ADR-0003). Regression-tested.
- Decoding is strict: a declared `charset` is honoured, a UTF-8 byte-order mark is
  stripped, and bytes invalid in the expected encoding are refused, not replaced.

**Notes**

- Still opt-in and offline. Nothing on a default path fetches; the reader is not a browser
  and does no boilerplate removal, and PDF and full main-content extraction remain planned
  (K-008).

### 2026-10-08 — Bounded, opt-in web fetcher

**Added**

- `chakaso.retrieval.HttpFetcher` behind a `Fetcher` protocol: given an
  `AcquisitionRequest` and a `FetchPolicy` it returns an `AcquiredSource` (bytes, media
  type, requested URL, final URL after redirects, retrieval time). The real network call is
  an injectable `Transport`; `make_urllib_transport` provides the standard-library one.
- `FetchPolicy`: a validated value object for the limits that were missing — scheme
  allowlist, response-size cap, redirect depth, timeout, and a user-agent that cannot carry
  a newline. A policy can restrict to `https` but can never widen past http/https.
- `is_blocked_destination`: refuses loopback, private, link-local (including the cloud
  metadata address), unique-local, multicast and reserved address literals, plus known
  loopback hostnames.
- A typed acquisition error family: `InvalidSchemeError`, `BlockedDestinationError`,
  `ResponseTooLargeError`, `FetchTimeoutError`, `UnsupportedContentTypeError`,
  `NetworkFailureError`, under `AcquisitionError`.
- ADR-0012 records the decision and its security posture.

**Security posture, stated honestly**

- Redirects are followed by the fetcher, not the transport, so every hop's scheme and
  destination are re-validated: a public URL that redirects into a private network is
  blocked, and redirect loops and over-long chains are stopped.
- The destination screen is bounded: it blocks address literals, but a hostname that
  resolves to a private address is **not** caught by a pre-connect string check. Closing
  that needs IP-pinned resolution in the transport, which is not implemented. This is
  recorded as K-008, not hidden.
- Fetching is **opt-in and off every default path**: nothing in the library, `chakaso
  retrieve`, the tests or CI reaches the network, so the ADR-0001 no-network runtime
  guarantee still holds, and CI stays offline and deterministic.

**Notes**

- This is acquisition only. Decoding fetched bytes and turning them into evidence (HTML →
  text, then normalize/chunk) is a separate, still-pending stage; the fetcher never assigns
  an identifier and never treats page text as instructions.

### 2026-10-08 — Command-line retrieval

**Added**

- `chakaso retrieve --query TEXT --file PATH [--file PATH ...] [--top-k N]`: the local
  pipeline from the command line. It ingests the named files into an in-memory corpus,
  runs lexical BM25 retrieval, and prints the ranked chunks with their provenance —
  score, section, matched terms and the source's canonical reference.

**Notes**

- The command calls the same application layer the library exposes (`ingest_file`,
  `Corpus`, `RetrievalService`) and holds no retrieval logic of its own. Errors (a missing
  or unsupported file, a bad limit) are reported readably with no traceback.
- It is local and offline, reads only the files given, and generates no answer: there is
  still no language model. An empty retrieval is reported plainly, not filled with
  invented evidence. Only `.txt`, `.md` and `.markdown` are read.

### 2026-10-08 — Retrieval orchestration

**Added**

- `chakaso.retrieval.RetrievalService`: the component that runs the pipeline for one
  query — corpus, index, retriever, ranked chunks, evidence pack — and returns a
  `RetrievalOutcome` carrying a validated `EvidencePack` and the full ranked result list.
  It is deliberately not the conversation manager, so retrieval can be called on its own
  and a turn can decide whether to retrieve at all.

**Provenance guarantees**

- Sources come from the corpus, never invented; a chunk whose source the corpus lacks is
  refused.
- The retriever's score is recorded onto a copy of each chunk; the stored chunk and its
  identifier are unchanged, so a citation still points at the exact evidence used.
- An unmatched query produces an empty pack, never fabricated evidence — the failure
  ADR-0003 exists to prevent.
- The pack records which retriever and `top_k` produced it, so an answer's provenance
  includes the pipeline, not only the model.

**Notes**

- `ConversationManager.send` can now be handed a pack that something actually produced.
  No query planner exists, so nothing decides on its own that a turn needs retrieval;
  retrieval is still called explicitly. No network, no embeddings, no model.
- The index is built once from the corpus the service is given; a corpus that grows later
  needs a fresh service, because an index is a cache, never the source of truth.

### 2026-10-08 — Deterministic lexical retrieval

**Added**

- `chakaso.retrieval.build_lexical_index` and `LexicalRetriever`: an inverted index over
  a corpus's chunks, scored with Okapi BM25, returning the best `top_k` as
  `RetrievalResult` records — the chunk, a score, a rank, and the matched terms. This is
  the smallest thing that turns a corpus into ranked evidence, and the baseline a later
  dense retriever must beat.
- `Retriever`: a protocol, so the lexical implementation can be replaced without touching
  a caller. `source_ids` filters candidates.

**Notes**

- Retrieval is lexical and says so: it matches shared words and no meaning — no
  embeddings, no vector index, no model, no network. Its limitations (synonyms,
  morphology, paraphrase) are exactly what the project wants measured rather than assumed.
- Determinism is enforced, not hoped for: scores are summed in sorted term order so
  floating-point accumulation cannot drift, ties break by chunk identifier so the order is
  total, and a query that matches nothing returns nothing rather than filling `top_k` with
  invented evidence. A result's score is a ranking artefact, never a confidence.
- `k1` and `b` are documented per-call defaults, not yet configuration fields.

### 2026-10-08 — In-memory corpus

**Added**

- `chakaso.retrieval.Corpus`: the in-memory set of evidence retrieval reads from. It
  holds sources and their chunks, adds documents idempotently, enumerates deterministically
  (sources by identifier, chunks by source and position, independent of insertion order),
  and looks records up by identifier.
- `CorpusError`: raised when a chunk is added under a source that does not own it.

**Notes**

- No persistent storage and no index. A corpus is deliberately a plain in-memory
  container: a database would be maintained against a guess about what needs storing,
  and an index is a later cache over this, not its source of truth.
- Duplicate documents do not grow the corpus, because identifiers are content-derived.
  Nothing searches or ranks it yet.

### 2026-10-08 — Local document ingestion

**Added**

- `chakaso.retrieval.ingest_file` and `ingest_text`: the first stage that turns a source
  into evidence records. Each identifies the source (ADR-0011), reads it, normalizes it,
  splits it into sections and chunks, and returns an `IngestedDocument` — a
  `SourceRecord` plus its `EvidenceChunk` records — which is exactly the shape
  `ConversationManager.send` already consumes. No parallel representation was invented.
- A typed ingestion error family: `DocumentUnavailableError`, `UnsupportedDocumentError`,
  `DocumentTooLargeError`, `ContentDecodingError`, under `IngestionError`.

**Filesystem safety**

- Ingestion reads the single file a caller names and nothing else. It never crawls a
  directory, never walks a filesystem, and never reads a file because it happens to
  exist. A directory, an absent path, an empty path, a non-regular file, an unsupported
  extension, a file over the byte limit and non-UTF-8 bytes are each refused with a
  specific error, and every one of those rejections is exercised by a test.
- Only `.txt`, `.md` and `.markdown` are read, so a binary — including the private
  `.docx` pack — is refused on format rather than decoded into mangled text. A `..` in a
  path is normalized away, so a winding path cannot fork one document into two sources.
- The byte limit is checked against the file's size before its bytes are read, so a
  pathological file cannot exhaust memory; a bad file is a hard failure, never a silent
  truncation. An empty file is a valid source with zero chunks.

**Notes**

- Still no retrieval, no ranking and no network. This is explicit-source ingestion, not
  a query planner or a fetcher; the caller chooses what to ingest.
- The ingestable size limit is a documented parameter of each call, not a hidden
  constant, and not yet a field in the configuration file — nothing reads it per run
  yet, matching how chunking sizes are handled.

### 2026-10-08 — Deterministic content normalization

**Added**

- `chakaso.retrieval.normalize_document`: puts supplied text into a deterministic form
  before it is chunked. CRLF and lone CR become LF, text is composed to Unicode NFC,
  trailing spaces and tabs are stripped per line, and runs of blank lines collapse to
  one. It normalizes representation and not substance: no re-wrapping, no case change,
  no word removed, leading indentation preserved.

**Why**

- Normalization feeds the content hash and the chunk identifiers, so two formatting
  variants of one document — a Windows file and a paste of the same body — must produce
  one set of identifiers, or an evaluation compares formatting rather than content. The
  function is idempotent and total.

**Notes**

- Two limits are stated, not hidden: it does not parse Markdown fences (blank lines
  inside a code block collapse like any others, matching the chunker), and its Unicode
  form depends on the interpreter's normalization tables, so reproducible evaluation
  pins the interpreter version.
- Normalization is a stage the pipeline calls; nothing is normalized automatically yet,
  because nothing reads a document. It is exercised through text a caller supplies.

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
