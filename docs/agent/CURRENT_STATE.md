# Current state

Last updated: 2026-10-08, at commit `docs: record the claim, citation, grounding and answer-evaluation foundation` (code last changed at `feat: combine citations and grounding into answer evaluation`).

This file is the authority on what exists. If it disagrees with any other
document, this file is right and the other document is a defect.

A private planning note is never evidence that something is implemented.

## What exists

| Thing | Where | Notes |
| --- | --- | --- |
| Repository conventions | `.gitignore`, `.gitattributes`, `.editorconfig` | Private docx pack excluded by `.gitignore` and guarded by a test |
| License | `LICENSE` | Apache-2.0; rationale in ADR-0004 |
| Public documentation | `README.md`, `docs/*.md`, `docs/research/` | Specifications with mandatory status labels |
| Decision records | `docs/decisions/` | ADR-0001 to ADR-0015 |
| Contribution guide | `CONTRIBUTING.md` | |
| Agent contract | `AGENTS.md`, `docs/agent/` | |
| Python package | `src/chakaso/` | Installs; typed; `py.typed` ships |
| CLI | `src/chakaso/cli.py` | `--version`, `info`, `config show`, `chat`, `retrieve` |
| Configuration | `src/chakaso/config/`, `configs/default.toml` | Typed schema, explicit loading, per-value provenance |
| Core primitives | `src/chakaso/core/` | Content-derived identifiers, SHA-256 hashing, error base |
| Model boundary | `src/chakaso/models/` | `LanguageModel` protocol, capabilities, registry, contract tests |
| Model implementations | `src/chakaso/models/deterministic.py` | A development double only. **No language model exists.** |
| Evidence records | `src/chakaso/evidence/` | `SourceRecord`, `EvidenceChunk`, `EvidencePack`, typed source references (web/file/text, ADR-0011), URL canonicalization, citation resolution |
| Normalization | `src/chakaso/retrieval/normalize.py` | Deterministic representation normalization (line endings, Unicode NFC, blank lines) before chunking |
| Chunking | `src/chakaso/retrieval/chunking.py` | Document text to evidence chunks: section-bounded, non-overlapping, deterministic (ADR-0010) |
| Ingestion | `src/chakaso/retrieval/ingest.py` | Reads one explicitly named local text/Markdown file, or supplied text, into a source record and its chunks; never crawls |
| Corpus | `src/chakaso/retrieval/corpus.py` | In-memory set of sources and chunks for retrieval: idempotent add, deterministic enumeration, provenance guard |
| Lexical retrieval | `src/chakaso/retrieval/lexical.py` | Deterministic BM25 index and retriever behind a `Retriever` protocol: ranking, top-k, source filter, inspectable explanations |
| Retrieval orchestration | `src/chakaso/retrieval/service.py` | `RetrievalService.search`: query → ranked `EvidencePack` with recorded scores and preserved provenance; an empty result makes an empty pack, never invented evidence |
| Conversation state | `src/chakaso/conversation/state.py` | Immutable, append-only turns with provenance; topic, entities and open questions |
| Conversation manager | `src/chakaso/conversation/manager.py` | Conducts one turn: context projection, model call, validation, evidence and citation recording. Transactional (ADR-0008) |
| Conversation shell | `chakaso chat` | Interactive and one-shot. States in its own output that the engine is a development double |
| Retrieval CLI | `chakaso retrieve` | Ingests named local files, runs lexical retrieval, prints ranked evidence with provenance. Local and offline |
| Web fetcher | `src/chakaso/retrieval/acquire.py`, `policy.py`, `netguard.py` | Opt-in, bounded HTTP fetch under a `FetchPolicy`; scheme/size/redirect/timeout/destination rules. Off every default path (ADR-0012) |
| Web ingestion & cache | `src/chakaso/retrieval/html.py`, `web.py`, `cache.py` | Narrow HTML reader turns fetched bytes into the same evidence records as a local file; an opt-in in-memory cache avoids refetching |
| Evaluation | `src/chakaso/evaluation/` | Pure metric functions (recall@k, precision@k, MRR, duplicate + unresolved-reference counts, latency observation), and `evaluate_answer`, which combines citation and grounding results into separate answer-evaluation dimensions (`evidence_coverage`, `is_grounded`). No merged quality score and no model-quality number |
| Benchmark | `src/chakaso/benchmark/` | Versioned case schema, strict JSONL/manifest loader, synthetic development fixtures, a retrieval runner and text/JSON reports. A development instrument, not a scientific benchmark |
| Claims | `src/chakaso/claims/` | Immutable `Claim` with a content-derived `ClaimId` and a `ClaimStatus` that records what the supplied evidence supports — never truth. A replaceable `ClaimExtractor` boundary (structured + a documented sentence heuristic) and claim/evidence links (ADR-0014). No trained extractor |
| Citation validation | `src/chakaso/citation/` | Structural citation checks against a supplied pack (valid / unknown / irrelevant / uncited) plus citation precision/recall. Decides presence and expected-match, never that a source proves a claim |
| Grounding | `src/chakaso/grounding/` | `GroundingEvaluator` boundary: a structural evaluator (supported / unsupported / not_evaluated) and a manual evaluator that is the only path to contradicted / uncertain; a `Contradiction` names no winner (ADR-0015) |
| Tests | `tests/` | ~665 tests at the commit recorded above: package, CLI (incl. `retrieve`), configuration, identifiers and hashing, source identity, model boundary, evidence, normalization, chunking, ingestion, corpus, lexical retrieval, retrieval orchestration, end-to-end pipeline, fetch policy and netguard, HTML/web ingestion, cache, evaluation metrics and answer evaluation, benchmark identity/cases/loader/fixtures/runner/report, claims/extract/links, citation validation, grounding, conversation state and manager, repository hygiene. The count ages; `python -m pytest` does not. |
| CI | `.github/workflows/ci.yml` | Green on Python 3.11, 3.12, 3.13 |
| Dense retrieval, persistence, correction | **do not exist** | An opt-in, bounded fetcher and a narrow HTML reader now exist (off the default path); there are no embeddings, no persistent store and no correction |

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
- A query against a corpus can be run through `RetrievalService`, which ranks chunks,
  records the retriever's score onto the pack's chunks, and assembles a valid
  `EvidencePack` with its sources; `ConversationManager.send(evidence=...)` can now be
  handed a pack that something actually produced. An unmatched query gives an empty pack,
  never fabricated evidence.
- `python -m chakaso retrieve --query ... --file ...` runs the whole local pipeline from
  the command line: it ingests the named files, retrieves, and prints ranked evidence with
  its provenance. It reads only the files given, touches no network and generates no
  answer; an empty retrieval says so.
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

- A full document processor: PDF, robust main-content extraction and boilerplate
  removal. The HTML reader is narrow and is not a browser — it runs no scripts, applies no
  CSS and strips no ads.
- Query planner: nothing decides whether retrieval would help, and nothing decides *which*
  URLs to fetch; the fetcher is only ever handed an explicit URL.
- Dense embeddings, a vector index and reranking: retrieval is lexical only, matching
  shared words and no meaning
- Semantic claim-support checking. Claims, structural citation validation and structural
  grounding exist (`chakaso.claims`, `chakaso.citation`, `chakaso.grounding`): a claim's
  citation can be checked for presence and expected-match and marked supported or unsupported.
  Whether the cited source actually *proves* the claim — and automatic contradiction or
  uncertainty detection — does not exist; those statuses appear only when a caller supplies a
  judgement.
- Reassessment and the correction loop. The claim, status and grounding vocabulary correction
  will consume now exists; nothing compares a previous answer against new evidence, decides
  retain / qualify / correct, or records a correction.
- A general evaluation harness and model benchmarks. A retrieval development benchmark and
  metric functions exist, as do structural claim, citation and grounding evaluation and an
  `evaluate_answer` combiner; the grounded-answer and correction *benchmarks* and any
  model-quality measurement do not. Structural evaluation checks references and set
  membership, not meaning.
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
| ADR-0012 | The fetcher is an optional, explicitly-bounded adapter |
| ADR-0013 | Benchmark cases have a stable identity and a content-derived version |
| ADR-0014 | A claim is a first-class, content-identified unit; status is evaluation, not truth |
| ADR-0015 | Grounding is a boundary; structural evaluation never claims semantic support |

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

Retrieval (local ingestion, lexical BM25 retrieval, an opt-in bounded fetcher and a narrow
HTML reader), the retrieval development benchmark, and the claim / citation / grounding /
answer-evaluation primitives are all built and tested. What the project does not yet have is
the **reassessment and correction foundation**: taking a previous answer's claims plus new
evidence, classifying the claims against that evidence (the supported / unsupported /
contradicted vocabulary now exists), deciding retain / qualify / correct, and recording the
change. That is the next unit, and it builds on the claim and grounding boundaries already in
place rather than inventing new ones.

There is still no language model, so nothing generates an answer to be corrected; the
correction path is built and tested against claims and evidence a caller supplies. See
[`ACTIVE_TASK.md`](ACTIVE_TASK.md) for the definition of done.