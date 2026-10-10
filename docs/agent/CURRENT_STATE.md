# Current state

Last updated: 2026-10-10, at the commit adding durable SQLite answer storage (ADR-0021).

This file is the authority on what exists. If it disagrees with any other
document, this file is right and the other document is a defect.

A private planning note is never evidence that something is implemented.

## What exists

| Thing | Where | Notes |
| --- | --- | --- |
| Repository conventions | `.gitignore`, `.gitattributes`, `.editorconfig` | Private docx pack excluded by `.gitignore` and guarded by a test |
| License | `LICENSE` | Apache-2.0; rationale in ADR-0004 |
| Public documentation | `README.md`, `docs/*.md`, `docs/research/` | Specifications with mandatory status labels |
| Decision records | `docs/decisions/` | ADR-0001 to ADR-0025 |
| Contribution guide | `CONTRIBUTING.md` | |
| Agent contract | `AGENTS.md`, `docs/agent/` | |
| Python package | `src/chakaso/` | Installs; typed; `py.typed` ships |
| CLI | `src/chakaso/cli.py` | `--version`, `info`, `config show`, `chat`, `retrieve`, `benchmark retrieval|correction`, `experiment run`, `storage check|migrate` |
| Configuration | `src/chakaso/config/`, `configs/default.toml` | Typed schema in sections (`model`, `storage`), explicit loading, per-value provenance; storage defaults to the in-memory backend so nothing is written |
| Core primitives | `src/chakaso/core/` | Content-derived identifiers, SHA-256 hashing, error base |
| Model boundary | `src/chakaso/models/` | `LanguageModel` protocol, capabilities, registry, contract tests |
| Model implementations | `src/chakaso/models/deterministic.py` | A development double only. **No language model exists.** |
| Evidence records | `src/chakaso/evidence/` | `SourceRecord`, `EvidenceChunk`, `EvidencePack`, typed source references (web/file/text, ADR-0011), URL canonicalization, citation resolution |
| Normalization | `src/chakaso/retrieval/normalize.py` | Deterministic representation normalization (line endings, Unicode NFC, blank lines) before chunking |
| Chunking | `src/chakaso/retrieval/chunking.py` | Document text to evidence chunks: section-bounded, non-overlapping, deterministic (ADR-0010) |
| Ingestion | `src/chakaso/retrieval/ingest.py` | Reads one explicitly named local text/Markdown file, or supplied text, into a source record and its chunks; never crawls |
| Corpus | `src/chakaso/retrieval/corpus.py` | In-memory set of sources and chunks for retrieval: idempotent add, deterministic enumeration, provenance guard |
| Lexical retrieval | `src/chakaso/retrieval/lexical.py` | Deterministic BM25 index and retriever behind a `Retriever` protocol: ranking, top-k, source filter, inspectable explanations |
| Dense retrieval | `src/chakaso/retrieval/embeddings.py`, `dense.py` | An `EmbeddingModel` boundary with a deterministic **non-semantic** fixture double, an exact in-memory dense index, and a `DenseRetriever` behind the same `Retriever` protocol; results carry the strategy that produced them (ADR-0023). No real embedding model, no vector store on disk |
| Hybrid retrieval | `src/chakaso/retrieval/hybrid.py` | A `HybridRetriever` that composes the lexical and dense retrievers with transparent reciprocal-rank fusion, returning one ranking tagged `hybrid` where each result carries `HybridProvenance` — the lexical and dense ranks/scores that fed it (ADR-0024). Deterministic and offline; over the fixture embedding it re-ranks token overlap, not meaning |
| Retrieval orchestration | `src/chakaso/retrieval/service.py` | `RetrievalService.search`: query → ranked `EvidencePack` with recorded scores and preserved provenance; an empty result makes an empty pack, never invented evidence |
| Query planning | `src/chakaso/planning/` | A deterministic `QueryPlanner` boundary (ADR-0022): a typed `QueryPlan` (mode, normalized + original query, source constraints, top-k, reuse flag, explanation, planner version), meaning-preserving `normalize_query`, and `DeterministicQueryPlanner` deciding per turn whether to retrieve lexically, reuse the conversation's prior sources for a follow-up, or retrieve nothing. Not an LLM; lexical, dense and hybrid retrieval are all backed by retrievers (ADR-0022/ADR-0024) |
| Conversation state | `src/chakaso/conversation/state.py` | Immutable, append-only turns with provenance; topic, entities and open questions |
| Conversation manager | `src/chakaso/conversation/manager.py` | Conducts one turn: context projection, model call, validation, evidence and citation recording. Transactional (ADR-0008) |
| Retrieval-aware runtime | `src/chakaso/runtime/` | `RetrievalAwareConversation` composes planner → retrieval (lexical/dense/hybrid) → model → claims → citation validation → grounding → `AnswerRecord` → store as a thin coordinator, not a monolith (ADR-0025): retrieval optional, turn atomicity preserved, the planner decision and retrieval strategy recorded as provenance/metadata, and a structured, citation-safe evidence context rendered for the model. Runs the deterministic double; no model, no semantic claim |
| Conversation shell | `chakaso chat` | Interactive and one-shot. States in its own output that the engine is a development double |
| Retrieval CLI | `chakaso retrieve` | Ingests named local files, runs lexical retrieval, prints ranked evidence with provenance. Local and offline |
| Benchmark CLI | `chakaso benchmark` | Runs the retrieval development benchmark over its synthetic fixtures and prints a text or deterministic JSON report that labels itself a development instrument. Retrieval only; local and offline |
| Web fetcher | `src/chakaso/retrieval/acquire.py`, `policy.py`, `netguard.py` | Opt-in, bounded HTTP fetch under a `FetchPolicy`; scheme/size/redirect/timeout/destination rules. Off every default path (ADR-0012) |
| Web ingestion & cache | `src/chakaso/retrieval/html.py`, `web.py`, `cache.py` | Narrow HTML reader turns fetched bytes into the same evidence records as a local file; an opt-in in-memory cache avoids refetching |
| Evaluation | `src/chakaso/evaluation/` | Pure metric functions (recall@k, precision@k, MRR, duplicate + unresolved-reference counts, latency observation), and `evaluate_answer`, which combines citation and grounding results into separate answer-evaluation dimensions (`evidence_coverage`, `is_grounded`). No merged quality score and no model-quality number |
| Benchmark | `src/chakaso/benchmark/` | Versioned case schema, strict JSONL/manifest loader, synthetic development fixtures, a retrieval runner and reports; a correction benchmark (cases, dataset versioning, a runner over `reassess`, per-case + aggregate metrics, text/JSON reports, pinned fingerprint). Development instruments, not scientific benchmarks |
| Claims | `src/chakaso/claims/` | Immutable `Claim` with a content-derived `ClaimId` and a `ClaimStatus` that records what the supplied evidence supports — never truth. A replaceable `ClaimExtractor` boundary (structured + a documented sentence heuristic) and claim/evidence links (ADR-0014). No trained extractor |
| Citation validation | `src/chakaso/citation/` | Structural citation checks against a supplied pack (valid / unknown / irrelevant / uncited) plus citation precision/recall. Decides presence and expected-match, never that a source proves a claim |
| Grounding | `src/chakaso/grounding/` | `GroundingEvaluator` boundary: a structural evaluator (supported / unsupported / not_evaluated) and a manual evaluator that is the only path to contradicted / uncertain; a `Contradiction` names no winner (ADR-0015). A `SemanticJudge` / `SemanticGroundingEvaluator` boundary exists (ADR-0019) but its only implementation is a caller-supplied fixture — no real semantic judgement |
| Answers & persistence | `src/chakaso/answers/` | `AnswerId` is a per-event typed identifier, not a content hash (ADR-0017). An immutable `AnswerRecord` ties turn + model + evidence + claims + evaluation + a `correction_of` link by reference. `AnswerStore` boundary with two append-only, integrity-checked backends: in-memory `InMemoryAnswerStore` (the default) and a durable `SQLiteAnswerStore` built on the standard library (ADR-0018, ADR-0021). The durable store is proven by a test to survive a process restart, and is opt-in, so nothing writes to disk unless it is selected |
| Correction foundation | `src/chakaso/correction/` | The reassessment decision rule (retain / qualify / correct / abstain / needs_review, ADR-0016), `reassess` over the grounding boundary, an append-only `CorrectionRecord`, structural correction metrics, a follow-up helper (`reassess_followup`), and an application-level `reassess_stored_answer` / `reassess_and_record` over a stored answer. Decides and records; produces no revised prose (there is no model) and is not wired into an automatic loop |
| Experiments | `src/chakaso/experiments/` | `Experiment` (hypothesis + benchmark + configuration) and a content-fingerprinted, environment-separated `ExperimentResult`; `run_experiment` drives a development benchmark; `compare_results` is a deterministic regression baseline that refuses incompatible versions (ADR-0020). No tracker, no database, no model |
| Tests | `tests/` | ~941 tests at the commit recorded above: package, CLI (incl. `retrieve`, `benchmark` and `experiment`), configuration, identifiers and hashing, source identity, model boundary, evidence, normalization, chunking, ingestion, corpus, lexical retrieval, dense retrieval (embedding double, exact index, retriever), hybrid retrieval (reciprocal-rank fusion, per-component provenance), retrieval orchestration, end-to-end pipeline, fetch policy and netguard, HTML/web ingestion, cache, evaluation metrics and answer evaluation, benchmark identity/cases/loader/fixtures/runner/report and the correction benchmark, claims/extract/links, citation validation, grounding and the semantic boundary, correction decision/reassess/record/metrics/follow-up/service, answer identity/record/store/serialization and the durable SQLite store (restart, rollback, schema, integrity) with a two-backend conformance suite and conversation integration, experiments (model/runner/compare), security/determinism/reproducibility regressions, conversation state and manager, query planning (mode/normalization/plan invariants/deterministic rules), retrieval-aware runtime (plan→retrieve→record, atomicity, provenance), repository hygiene. The count ages; `python -m pytest` does not. |
| CI | `.github/workflows/ci.yml` | Green on Python 3.11, 3.12, 3.13 |
| Retrieval wired into an interactive command, an automatic correction loop | **do not exist** | The retrieval-aware runtime now composes planner → retrieval → model → record → store (ADR-0025) and is tested, but no conversation command drives it yet (only tests do), there is no real semantic embedding, and no reassessment runs itself |

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
  an `EvidencePack`. Nothing yet ranks chunks.
- A local document the caller names explicitly (`.txt`, `.md` or `.markdown`) can be
  read into a `SourceRecord` and its chunks — identified by path, normalized, split.
  Supplied text is ingested the same way with no filesystem access. Ingestion reads only
  the one path it is given: a directory, an absent path, an unsupported format, an
  oversized file or non-UTF-8 bytes are each refused with a specific error, and an empty
  file yields a valid source with zero chunks.
- Supplied text can be normalized deterministically (line endings, Unicode NFC, trailing
  whitespace, blank-line runs) so two formatting variants of one document produce one
  set of chunk identifiers. It preserves leading indentation and changes no words; it
  does not parse Markdown fences.
- Ingested documents can be collected into an in-memory `Corpus`: adding is idempotent
  (duplicate documents do not grow it), a chunk cannot be filed under a source that does
  not own it, and enumeration is deterministic regardless of insertion order.
- Chunks in a corpus can be ranked for a query by a deterministic lexical retriever
  (BM25): `build_lexical_index` then `LexicalRetriever.retrieve` returns ranked
  `RetrievalResult` records carrying a score, a rank and the matched terms. It matches
  shared words only — no embeddings, no model, no network — ties break by chunk
  identifier, and a query that matches nothing returns nothing rather than inventing a
  result.
- Chunks can also be ranked by **dense similarity** behind the same `Retriever` protocol:
  `build_dense_index` then `DenseRetriever.retrieve` embeds the query and every chunk through an
  `EmbeddingModel` boundary and ranks by cosine or dot similarity, returning `RetrievalResult`
  records tagged `dense` with no invented matched terms (ADR-0023). Its only embedding is
  `FixtureEmbeddingModel`, a deterministic hashed bag-of-tokens double declared `is_semantic=False`
  — it exercises the boundary and reproduces exactly, but it is not semantic and a dense result from
  it means token overlap under a different metric, not understanding.
- Chunks can be ranked by **hybrid fusion**: `HybridRetriever` composes the lexical and dense
  retrievers with transparent reciprocal-rank fusion, returning one ranking tagged `hybrid` where each
  result records the lexical and dense ranks/scores that produced it (`HybridProvenance`, ADR-0024).
  It is deterministic and offline; over the fixture embedding it re-ranks token overlap, so it
  demonstrates the fusion mechanism, not any semantic advantage.
- A query against a corpus can be run through `RetrievalService`, which ranks chunks,
  records the retriever's score onto the pack's chunks, and assembles a valid
  `EvidencePack` with its sources; `ConversationManager.send(evidence=...)` can now be
  handed a pack that something actually produced. An unmatched query gives an empty pack,
  never fabricated evidence.
- `python -m chakaso retrieve --query ... --file ...` runs the whole local pipeline from
  the command line: it ingests the named files, retrieves, and prints ranked evidence with
  its provenance. It reads only the files given, touches no network and generates no
  answer; an empty retrieval says so.
- `python -m chakaso benchmark` runs the retrieval development benchmark end to end from the
  command line and prints its report (text, or `--json` for a deterministic machine-readable
  report). It scores retrieval over the synthetic fixtures, labels itself a development
  instrument, and generates no answer.
- A web source acquired by the opt-in fetcher can be turned into evidence:
  `ingest_acquired` decodes the bytes, parses HTML into text with headings via a narrow
  reader (not a browser), normalizes and chunks it into the same records a local file
  produces. Fetched text is stored as data and is never an instruction or a source
  identity (ADR-0003); a `CachingFetcher` avoids refetching a canonical URL.
- A retrieval development benchmark runs end to end (`chakaso.benchmark`): a versioned,
  hand-built synthetic corpus scored by `RetrievalBenchmarkRunner` into recall@k /
  precision@k / MRR / forbidden-hit / false-retrieval metrics, reported as text or
  deterministic JSON that labels itself a development instrument. It measures retrieval
  only; a pinned content fingerprint regresses any unintended change to the fixtures.
- An answer can be decomposed into claims (`chakaso.claims`): a `ClaimExtractor` builds
  immutable claims from a supplied decomposition or a documented sentence heuristic, each with
  a stable content-derived identifier and an evaluation status that means "the supplied
  evidence supports this", never "this is true" (ADR-0014). No language model is involved.
- A claim's citations can be validated structurally (`chakaso.citation`): a reference to a
  chunk that was supplied is *valid*, one outside the pack is *unknown*, one present but
  outside a case's expected evidence is *irrelevant*, and a required claim that cites nothing
  valid is *uncited*; precision/recall count these without claiming a source proves a claim.
- Grounding turns those citations into a per-claim status (`chakaso.grounding`): the structural
  evaluator marks supported / unsupported / not_evaluated and cannot invent a contradiction,
  while a manual evaluator is the only way *contradicted* or *uncertain* appear; a
  `Contradiction` records the disagreement without naming a winner, and precedence is a
  caller-supplied hook, never an automatic ranking (ADR-0015).
- `evaluate_answer` combines the citation and grounding results into one view that keeps the
  dimensions separate — citation precision, evidence coverage (the grounded-claim support
  rate), grounding status and contradictions — and deliberately reports no single merged
  quality score. Claims and evidence are supplied by a caller or fixture, not generated.
- An answer's claims can be reassessed against evidence and the outcome decided and recorded
  (`chakaso.correction`): `decide_claim` / `decide_answer` turn prior-vs-now statuses into a
  per-claim disposition and an answer-level decision under a fixed, written rule (ADR-0016) — a
  contradiction outranks everything and nothing corrects for recency or source order;
  `record_reassessment` fixes the outcome into an immutable, content-identified
  `CorrectionRecord` that supersedes an earlier answer; `correction_metrics` reports success,
  unjustified-persistence and unnecessary-revision rates over a caller-labelled set. With no
  model, nothing generates revised wording, and no turn triggers reassessment by itself.
- An answer can be recorded as a durable-by-design `AnswerRecord` and looked up: `chakaso.answers`
  gives a per-event `AnswerId` (not a content hash, ADR-0017), an immutable record that references
  the turn, model, evidence, claims, evaluation and a `correction_of` link, and an `AnswerStore`
  whose in-memory implementation is append-only and refuses to overwrite a saved answer or to link
  a correction to a missing or cross-conversation answer (ADR-0018). A second backend,
  `SQLiteAnswerStore`, persists the same history to a local SQLite file — an explicit schema version,
  transactional all-or-nothing appends, strict-read integrity checks, and a test proving it survives a
  process restart (ADR-0021). It is opt-in, so nothing is written to disk by default, and a stored
  answer's evaluation is recomputed rather than persisted.
- A completed turn can be persisted atomically: given a store and a claim extractor,
  `ConversationManager.send` builds and saves the `AnswerRecord` *before* adopting the new
  conversation, so a persistence failure aborts the turn and leaves the conversation unchanged
  (ADR-0008 extended to the record). Without a store the turn behaves exactly as before.
- Reassessment is an application operation over a *stored* answer: `reassess_stored_answer` /
  `reassess_and_record` take an `AnswerId` and new evidence, re-ground the recorded claims, and
  return a decision and a `CorrectionRecord` linked to the prior. The follow-up boundary is
  identifier-addressed — no natural-language reference resolver is pretended.
- A semantic grounding *boundary* exists (`SemanticJudge` / `SemanticGroundingEvaluator`,
  ADR-0019) and composes into answer evaluation and reassess without changing their signatures, so
  grounding stays a separate dimension from citation. Its only implementation is a caller-supplied
  `FixtureSemanticJudge`; it names no winner in a conflict and there is no real semantic judge.
- A correction development benchmark runs (`chakaso.benchmark.correction`): synthetic cases whose
  expected decision (retain / correct / qualify / needs_review / abstain) the written rule is
  checked against, scored per case and in aggregate, reported as deterministic text/JSON that labels
  itself a development instrument, with a pinned-fingerprint regression.
- Reproducible experiments (`chakaso.experiments`): `run_experiment` drives a development benchmark
  into a content-fingerprinted `ExperimentResult` whose `result_id` ignores the timestamp and host;
  `compare_results` gives a deterministic regression baseline that refuses incompatible benchmark or
  evaluator versions. `chakaso benchmark` and `chakaso experiment run` expose these from the CLI.
- The durable store is configurable and inspectable: `config.storage` carries the backend
  (`memory` default, or `sqlite`), the database path and the SQLite journal mode, validated by the
  existing typed schema (ADR-0006, ADR-0021). `chakaso storage check --path ...` reports a store's
  schema version and integrity read-only; `chakaso storage migrate --path ...` creates or confirms
  its schema. Local, offline, and nothing is created or changed unless a real path is given.
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
| CLI | Version, help, unknown command, `info`, `config show` including failure paths, and `storage check`/`migrate` (path resolution, absent file, healthy report, non-store and newer-schema refusal) |
| Configuration | Defaults, file layering and precedence, provenance, unknown keys, wrong types, boolean-vs-integer, ranges, name pattern, missing file, directory, invalid TOML, array values, immutability, schema/dataclass agreement, shipped file vs built-in defaults, and the `storage` section (defaults write nothing, backend/journal-mode patterns rejected) |
| Identifiers and hashing | Digest agreement with `hashlib`, UTF-8 handling, truncation bounds, part-separator ambiguity, derivation determinism, deduplication, content-change distinction, position and text sensitivity, malformed identifier rejection, ordering and hashing |
| Model boundary | The inherited contract suite (metadata, provenance of results, repeatability at zero temperature, empty-request rejection, length ceiling, unsupported-capability failures, declared capabilities being implemented), plus capability reconciliation, parameter validation, registry failure paths and the double's documented behaviour |
| Evidence | Canonicalization idempotence and non-merging, tracking-parameter removal, scheme and credential refusal, IPv6 handling, record immutability, naive-timestamp rejection, content-hash validation, change detection, pack validation, and citation resolution including fabricated and malformed references |
| Conversation manager | Construction and context bounds, dependency injection through the model boundary, first and follow-up turns, context projection limits, previous state preserved, failures leaving state unchanged for the next turn, model errors propagating untranslated, clock regressions, evidence and citation provenance per turn, evidence not leaking between turns, and end-to-end wiring against the deterministic double |
| Answers and durable storage | Answer identity and record invariants, the serialization codec (round-trip, evaluation-not-persisted, malformed-document rejection), the append-only invariants run over **both** backends (in-memory and SQLite conformance), and the SQLite store's restart durability (including a separate-process write), transaction rollback, schema/version refusal, and on-disk corruption detection |
| Query planning | Mode/retrieval-flag consistency and plan construction invariants (reuse needs sources, a retrieving plan needs a non-empty query, every plan explains itself), meaning-preserving normalization (whitespace/Unicode/trailing-punctuation, idempotent, total, interior punctuation kept), the deterministic rules over a conversational sequence (greeting → none, question → fresh, follow-up → reuse prior sources, topic change → no stale reuse), unbacked-mode refusal, and protocol conformance |
| Dense retrieval | The fixture embedding's honest metadata (`is_semantic=False`, batch order, determinism, unit-length, zero-vector empty text), vector/metadata validation, index invariants (uniform dimension, no duplicate chunk, empty index), and the dense retriever: `dense` strategy tagging with empty matched terms, ranking/determinism, `source_ids` filter, zero-query and empty-index returning nothing, dimension-mismatch rejection, and `Retriever`-protocol conformance |
| Hybrid retrieval | Reciprocal-rank fusion of the lexical and dense retrievers: `hybrid` strategy tagging with full per-component provenance (both ranks when a chunk appears in both, `None` for the component that missed it — never fabricated), exact RRF score arithmetic, weight-driven favoring, `candidate_k` pooling, `source_ids` pass-through, determinism, invalid-config refusal, and `Retriever`-protocol conformance |
| Conversation runtime | A grounded turn through the real pipeline: a greeting retrieves nothing but still records, a knowledge turn retrieves and records its evidence with the planner decision + retrieval strategy on the answer, a follow-up reuses prior sources, a topic change does not, dense/hybrid flow through, an unwired mode or bad top-k is refused at construction, a retriever failure leaves the conversation and store untouched, and the evidence-context renderer emits ids + reference + text (empty pack → nothing) |
| Repository invariants | Private pack never tracked, `.gitignore` rule present, no secret-shaped files, no tracked file over 1 MiB, no commercial provider dependency, ADR numbering and indexing, documentation links resolve |

## What is experimental

The semantic grounding boundary (ADR-0019) and the experiment/comparison layer (ADR-0020) are
new research scaffolding: implemented, typed and tested, but with only a fixture semantic judge and
no validated real-workload use. The correction development benchmark scores a decision rule over
synthetic cases, not measured capability. None of these is a claim that the approach works on real
data — that is K-001, and it stands. The code that exists still does one small thing each. The
dense retrieval boundary (ADR-0023) is the same kind of scaffolding: a real retriever and an exact
index, but over a non-semantic fixture embedding, so it proves the shape, not any dense quality.

## What is not implemented

- A full document processor: PDF, robust main-content extraction and boilerplate
  removal. The HTML reader is narrow and is not a browser — it runs no scripts, applies no
  CSS and strips no ads.
- The retrieval-aware runtime is not wired into an interactive command. `chakaso.runtime` composes
  planner → retrieval → model → record (ADR-0025) and is exercised by tests, but `chakaso chat` does
  not use it, so an interactive session still neither plans nor retrieves; and nothing decides *which*
  URLs to fetch — the fetcher is only ever handed an explicit URL.
- Dense embeddings, a real semantic embedding model, an approximate vector index and reranking:
  a dense *boundary* and an exact in-memory index exist over a deterministic non-semantic fixture
  embedding (ADR-0023), but there is no genuine embedding model, no approximate (FAISS-like) index,
  and no reranking
- A real semantic judge. A semantic grounding *boundary* exists (`chakaso.grounding`, ADR-0019)
  but its only implementation replays caller-supplied relations; nothing decides automatically that
  evidence proves or contradicts a claim. Whether a cited source actually *proves* a claim — and
  automatic contradiction or uncertainty detection — does not exist.
- An automatic correction loop and a revised answer. Reassessment is now an application operation
  over a stored answer and there is a correction development benchmark, but nothing decides on its
  own that a turn needs reassessing, nothing generates revised prose (there is no model), and no
  reassessment is wired into a conversation turn to run by itself. An answer can be persisted durably
  (ADR-0021), but that is a store a caller selects, not an automatic correction.
- A general evaluation harness and model benchmarks. Retrieval and correction development
  benchmarks, metric functions, and structural claim/citation/grounding/answer evaluation exist;
  a *real* grounded-answer benchmark and any model-quality measurement do not. All benchmark
  numbers are over synthetic fixtures and check structure, not meaning.
- Tokenizer, dataset pipeline, model code, training loop
- A local inference adapter, and any model weights of any size
- Automatic persistence of correction records and experiment results. Answer history can now be
  persisted durably to SQLite (ADR-0021), but a `CorrectionRecord` or an `ExperimentResult` is still
  only serializable — the runtime does not write it to disk unless a caller wires it to a store.

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
| ADR-0012 | The fetcher is an optional, explicitly-bounded adapter |
| ADR-0013 | Benchmark cases have a stable identity and a content-derived version |
| ADR-0014 | A claim is a first-class, content-identified unit; status is evaluation, not truth |
| ADR-0015 | Grounding is a boundary; structural evaluation never claims semantic support |
| ADR-0016 | The correction decision is an explicit, evidence-driven rule |
| ADR-0017 | Answer identity is a per-event identifier, not a content hash |
| ADR-0018 | Answer persistence is an append-only store boundary, in-memory for now |
| ADR-0019 | Semantic grounding is a boundary with only a fixture implementation today |
| ADR-0020 | Experiments are content-fingerprinted, environment-separated research artifacts |
| ADR-0021 | Durable answer storage is a SQLite backend behind the existing store boundary |
| ADR-0022 | Query planning is a deterministic decision boundary, not a model |
| ADR-0023 | Dense retrieval is an embedding boundary with an exact index; the only embedding is a non-semantic double |
| ADR-0024 | Hybrid retrieval fuses lexical and dense results by reciprocal rank fusion |
| ADR-0025 | The retrieval-aware conversation runtime is a thin coordinator, not a monolith |

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

Retrieval, the retrieval and correction development benchmarks, the claim / citation / grounding /
answer-evaluation primitives, the correction decision rule with an application-level reassessment
over stored answers, an answer store wired atomically into a turn — in-memory and, since ADR-0021, a
durable SQLite backend that survives a restart — a deterministic query planner that decides per turn
whether and how to retrieve (ADR-0022), a fixture-only semantic grounding boundary, and reproducible
experiment records with a deterministic regression baseline, and a retrieval-aware runtime composing
planner → retrieval → model → record → store (ADR-0025) are all built and tested. The next real units,
in order:

1. **Benchmark and experiment expansion** — compare lexical / dense / hybrid and top-k settings over
   the synthetic fixtures through the existing experiment framework, reporting only measured,
   honestly-labelled development numbers (dense/hybrid run over the fixture embedding).
2. **End-to-end and restart integration demonstration** — a deterministic offline test running a query
   through the whole runtime to a durable store, then proving the answer history survives a
   close/reopen, with deliberate "test-the-tests" corruption checks on atomicity, provenance and
   persistence, and a security review of the new runtime.

A **real semantic judge** (the `SemanticJudge` boundary is fixture-only today), wiring the runtime
into **`chakaso chat`/the CLI**, and an **automatic, triggered correction** follow these. The full
retrieval flow is now executable and tested in the runtime (ADR-0025), but no interactive command
drives it yet, and dense/hybrid run over a non-semantic fixture embedding (ADR-0023/ADR-0024) — a real
embedding model, and a real language model, are what would make the answers mean something. All of it
stays behind the existing boundaries. See [`ACTIVE_TASK.md`](ACTIVE_TASK.md) for the definition of done.