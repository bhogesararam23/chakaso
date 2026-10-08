# Architecture

Chakaso is a local-first, model-agnostic, retrieval-grounded conversational
system. This document describes the components, what each one owns, and where the
boundaries are. It describes the intended architecture; the current status of each
component is stated explicitly.

## Design rules

These are constraints on the code, not aspirations. A change that breaks one
needs a decision record explaining why.

1. **Typed contracts between components.** Components exchange explicit data
   types, not dictionaries of loosely-related values. Where a boundary is expected
   to be replaced, it is an interface with at least one implementation.
2. **The model is replaceable.** Application code depends on a narrow
   language-model interface. Exactly one place maps configuration to a concrete
   implementation. ([ADR-0002](decisions/ADR-0002-language-model-boundary.md))
3. **Retrieval storage is replaceable.** Nothing above the retrieval layer may
   assume a particular index, database or file layout.
4. **Every stored artifact has a stable identifier and, where applicable, a
   content hash.** This is what makes an answer re-examinable after the live page
   changed. ([ADR-0003](decisions/ADR-0003-evidence-identifier-ownership.md))
5. **Configuration is externalized.** No magic constants in component code.
   Configuration is loaded explicitly and is versioned.
6. **Prompts and templates are versioned**, because a prompt change changes
   measured behaviour and must be attributable in an evaluation record.
7. **Evaluation datasets are versioned**, because a benchmark number without a
   task version is not a result.
8. **CPU-only execution remains possible.** The test suite must run on a machine
   with no GPU and no network access to model providers.
9. **Retrieved content is untrusted input.** Webpage text is data. It is never an
   instruction, and it never overrides application policy.

## Component overview

```text
                        user turn
                            |
                            v
                  +-------------------+
                  | Conversation      |  turn history, active topic,
                  | Manager           |  entities, previous sources,
                  +-------------------+  open questions
                            |
             +--------------+---------------+
             |                              |
             v                              v
   +-------------------+          +-------------------+
   | Query Planner     |          | Direct context    |
   | needs retrieval?  |          | (answerable from  |
   | what queries?     |          |  conversation)    |
   +-------------------+          +-------------------+
             |
             v
   +-------------------------------------------------+
   | Retrieval Pipeline                              |
   |   fetch -> parse/clean -> chunk -> rank         |
   +-------------------------------------------------+
             |
             v
   +-------------------+
   | Evidence Pack     |  immutable source_ids and chunk_ids
   +-------------------+
             |
             v
   +-------------------+        +-------------------+
   | Model Adapter     | <----  | prompt template   |
   | (LanguageModel)   |        | (versioned)       |
   +-------------------+        +-------------------+
             |
             v
   +-------------------+
   | Grounding /       |  every cited identifier must exist in the
   | Citation Validator|  evidence pack, or it is recorded as an error
   +-------------------+
             |
             +--> answer text
             +--> resolved source references
             +--> correction notice (if this turn revised an earlier answer)
             |
             v
          response

  Any turn may be challenged, which enters the reassessment path:

   previous AnswerRecord + new Evidence Pack
        -> claim comparison
        -> supported / unsupported / contradicted
        -> retain / qualify / correct
        -> new AnswerRecord, recorded with correction_of
```

## Component responsibilities

Each component owns one thing and must not grow into its neighbour.

| Component | Responsibility | Must not own | Status |
| --- | --- | --- | --- |
| Configuration | Declare, load and validate versioned settings, and report where each value came from | Component behaviour | Implemented |
| Conversation Manager | Conduct one turn: accept the user's message, project context, call the model, validate, record the answer | Model-specific logic; retrieval; correction | Implemented (orchestration only) |
| Query Planner | Decide whether retrieval is useful; formulate retrieval queries while preserving intent | Source truth | Planned |
| Retriever | Find candidate documents and chunks | Generate the final answer | Implemented (lexical BM25: ADR-0010 chunking; dense is planned) |
| Fetcher | Retrieve permitted public content under an explicit policy | Interpret facts | Implemented (bounded, opt-in: ADR-0012) |
| Document Processor | Extract readable text, metadata and section structure | Invent missing text | Implemented (narrow HTML reader: text + headings, no browser; PDF planned) |
| Chunker | Produce stable evidence units with positions | Rank claims | Implemented (Markdown structure, no overlap: ADR-0010) |
| Ranker | Order candidate evidence | Generate the answer | Planned |
| Evidence Store | Persist source and chunk records, metadata and hashes | Produce user-facing prose | Planned |
| Model Adapter | Uniform interface to any local or future model | Search | Implemented (boundary and registry; no trained model) |
| Grounding / Citation Validator | Check that every cited identifier exists and, structurally, whether cited evidence bears on the claim | Rewrite the user's request | Implemented (structural: `chakaso.citation` presence + `chakaso.grounding` supported/unsupported, ADR-0015); semantic support checking planned |
| Reassessment Engine | Compare previous claims with new evidence and decide retain/qualify/correct | Silently rewrite history | Planned |
| Evaluation | Measure behaviour and detect regressions | Change production behaviour | Implemented (metric functions, a retrieval development benchmark, and structural claim/citation/grounding/answer evaluation via `evaluate_answer`; grounded/correction benchmarks planned) |

A component marked "Planned" has no code. Configuration, the model boundary, the
evidence records, conversation state and manager, local ingestion, lexical retrieval, the
bounded opt-in fetcher, the retrieval development benchmark and the claim / citation /
grounding / answer-evaluation primitives are implemented;
[`agent/CURRENT_STATE.md`](agent/CURRENT_STATE.md) is the authority, including on
the fact that the only implementation of the model boundary is a development double
rather than a language model, and that the grounding and citation checks are structural.

### What the conversation manager does, and what it does not

`chakaso.conversation.ConversationManager` is the first component with behaviour.
One turn is: reject an empty message, append the user turn to a working copy,
project the recent conversation into model messages, call the model through the
boundary, reject a response that is empty or attributed to a different model, resolve
evidence references, append the assistant turn, and adopt the new state. A failure
at any step leaves the conversation unchanged ([ADR-0008](decisions/ADR-0008-transactional-turns.md)).

It does not retrieve, fetch, rank, chunk or index; it does not decide whether
retrieval would help; it does not correct an earlier answer; and it does not know
what a confidence score is. Its dependencies are the conversation state type, the
evidence types and the model boundary.

**The seam for retrieval.** `send` accepts an `EvidencePack`. That parameter is the
insertion point for query planning and retrieval: they will run before `send` and
pass the pack in. Until they exist, a turn with no evidence resolves no citations, so
a reference in an answer is reported as unresolved rather than resolved to a source
([ADR-0009](decisions/ADR-0009-unresolved-references-are-recorded.md)). The manager
was written to accept evidence before anything can produce it, deliberately: adding
that parameter later would touch every call site, and every call site is a place the
citation rule could be forgotten.

**Context projection.** The manager bounds a request by a number of recent turns
(`model.max_context_turns`). This is a stopgap, not a token budget. Measuring tokens
requires a tokenizer, which does not exist, so nothing here can honestly claim to fit
a model's context window.

## Interfaces

The boundaries that exist, or that the project is committed to building:

| Boundary | Shape | Status |
| --- | --- | --- |
| Configuration schema | Typed fields with declared ranges and patterns, loaded explicitly from TOML with per-value provenance | Implemented |
| `LanguageModel` | `metadata` plus `generate` over a message list and explicit generation parameters | Implemented |
| Model registry | adapter name -> factory, with one creation entry point | Implemented |
| `SourceRecord` | immutable retrieved-source identity and metadata | Implemented |
| `EvidenceChunk` | immutable evidence unit referencing a source | Implemented |
| `EvidencePack` | the set of chunks supplied to one generation call | Implemented |
| Citation resolution | evidence identifier -> verified source metadata, with rejection of unknown identifiers | Implemented |
| `Claim` / `ClaimStatus` | an answer decomposed into content-identified units; status records supplied-evidence support, not truth | Implemented (`chakaso.claims`, ADR-0014) |
| Citation validation | claim citations -> valid / unknown / irrelevant, plus uncited required claims | Implemented (`chakaso.citation`; structural) |
| `GroundingEvaluator` | claims + pack -> a per-claim grounding status (structural today; a supplied judgement for contradicted/uncertain) | Implemented (`chakaso.grounding`, ADR-0015) |
| `Conversation` | turns with provenance, active topic, entities, open questions, prior sources | Implemented |
| `ConversationManager` | conduct one turn against the model boundary; accept evidence; report the generation and citation outcome | Implemented |
| `AnswerRecord` | answer text, cited identifiers, model and prompt versions, correction lineage | Planned |
| Retriever / Fetcher | pluggable retrieval and fetch mechanisms | Implemented (lexical retriever behind a `Retriever` protocol; bounded opt-in fetcher behind a `Fetcher` protocol) |
| Reassessment | previous answer plus new evidence -> retain/qualify/correct | Planned |

"Planned" here means there is no code, and the shape described is the specification
to build against rather than a description of something that exists. Configuration,
the model boundary, the evidence records, conversation state, the conversation manager,
local ingestion, lexical retrieval and the retrieval development benchmark are
implemented; fetching exists only behind the opt-in bounded fetcher, so a default run
neither fetches nor reaches the network, and there is still no trained model.

### The model boundary

`chakaso.models` implements ADR-0002. It is narrow by construction:

- `LanguageModel` requires exactly two things: `metadata` and `generate` over a
  message list with explicit `GenerationParams`.
- Optional behaviour is declared as a `Capability` (`tokenize`,
  `structured_generation`) and implemented on a separate protocol — `Tokenizer`,
  `StructuredGenerator`. An implementation that cannot do something is not forced
  to define a method that raises.
- The declaration and the method are reconciled by module-level helpers,
  `tokenize()` and `generate_structured()`, which check the declared capability and
  then delegate. A model that declares a capability it does not implement is
  reported as a defect in the implementation rather than as caller error, because
  a caller has no way to plan around a false declaration.
- `metadata.development_double` marks an implementation that is not a language
  model. It exists so that documentation and reports can say plainly that no model
  was involved in a result.

`create_model(config)` is the single place where configuration becomes a model.
`chakaso.models.deterministic` provides the development double selected by
`configs/default.toml`; it declares no capabilities and produces fixed text.

## Data flow for one turn

Steps 1, 3, 4, 5 and 6 exist. Steps 2 and 7 do not. The consequence: the retrieval
pipeline is built and produces an evidence pack, but nothing decides *for* a turn whether
to retrieve — a caller runs `RetrievalService` and passes the pack to `send`. `chakaso
chat` does not wire that in, so a chat turn resolves no citations on its own.

1. The Conversation Manager appends the user turn and projects the recent
   conversation into model messages. Reference resolution against state ("this",
   "why not", "and the second one") is not implemented; the projected messages carry
   the previous turns, and nothing more specific.
2. **Not implemented.** The Query Planner decides whether the turn can be answered
   from conversation context or needs fresh evidence, and produces zero or more
   retrieval queries. Until it exists, retrieval is invoked explicitly by a caller.
3. **Implemented (local; fetch opt-in).** `RetrievalService` retrieves candidate chunks
   from a corpus and ranks them with the lexical (BM25) retriever, preserving source and
   section boundaries through chunking. Documents reach the corpus through local
   ingestion or, behind the bounded fetcher and narrow HTML reader, from a web URL — but
   nothing chooses a URL on its own (that is step 2).
4. **Implemented.** The selected chunks become an `EvidencePack` with immutable
   identifiers, produced by `RetrievalService.search` and validated on construction —
   every chunk's source present, scores recorded, an empty result an empty pack rather
   than invented evidence. Only chunks in the pack are visible to generation.
5. The Model Adapter generates from the conversation plus the evidence, referencing
   evidence by identifier, and the conversation manager records the assistant turn.
   Prompt templates are not versioned yet, because there is no prompt template.
6. The validator checks every identifier cited against the pack, records unknown
   identifiers as errors, and attaches source metadata to the surviving
   references. This exists as `resolve_citations`, and the conversation manager runs
   it on every answer.
7. **Not implemented.** The answer and its metadata are recorded as an
   `AnswerRecord`. Nothing about the retrieval or model provenance is discarded.
   Today the model's identity is reported in the reply and the assistant turn records
   which evidence was supplied and which sources were cited, but no record persists
   across turns, so a model identifier is not recoverable from conversation state
   alone.

## Storage

Storage is deliberately undecided beyond the properties it must have, because
committing to an index or a database before there is data to store would be a
guess ([ADR-0001](decisions/ADR-0001-local-first-runtime.md)).

What is fixed:

- Source and chunk records are immutable once written. Re-retrieving a URL creates
  a new record rather than mutating an old one.
- Records are content-addressed where the content is stored, so the bytes behind
  an answer can be identified after the fact.
- Nothing in application code depends on a filesystem layout.
- Local indexes are rebuildable from stored records. An index is a cache, never
  the source of truth.

## What is not decided

Stated plainly, because pretending otherwise would make the architecture look
more finished than it is:

- **Vector index and embedding model.** Planned as dense local embeddings plus a
  local similarity index. No choice is committed and nothing is implemented.
- **Web fetch mechanism.** The *transport* exists as a bounded, opt-in adapter under a
  `FetchPolicy` ([ADR-0012](decisions/ADR-0012-optional-bounded-fetcher.md)): given a URL
  it fetches safely and offline-testably, and is off every default path. What remains
  undecided is *discovery* — whether the first corpus comes from a search endpoint, a
  curated seed list, sitemap/feed discovery or a hand-built set — which is a query-planning
  question, not a fetching one.
- **Reranking model.** Optional in the design. No model is chosen.
- **Local inference runtime.** Depends on which local model is chosen first,
  which depends on the tokenizer work.
- **Persistence format for conversation state.** Decided for this phase: nothing is
  persisted. The manager keeps immutable state in memory and returns a new
  conversation on each completed turn, so a session is lost on exit. A store is
  deferred until retrieval and correction have attached to the state and its shape is
  no longer a guess. See
  [`research/open-questions.md`](research/open-questions.md#where-should-conversation-state-be-persisted).
- **Whether the model writes citations inline or in a structured side channel.**
  This is a real design question with evaluation consequences, and it is
  unresolved.
