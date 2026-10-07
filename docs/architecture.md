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
| Conversation Manager | Turns, conversation state, active topic, reference resolution, session metadata | Model-specific logic | Planned |
| Query Planner | Decide whether retrieval is useful; formulate retrieval queries while preserving intent | Source truth | Planned |
| Retriever | Find candidate documents and chunks | Generate the final answer | Planned |
| Fetcher | Retrieve permitted public content under an explicit policy | Interpret facts | Planned |
| Document Processor | Extract readable text, metadata and section structure | Invent missing text | Planned |
| Chunker | Produce stable evidence units with positions | Rank claims | Planned |
| Ranker | Order candidate evidence | Generate the answer | Planned |
| Evidence Store | Persist source and chunk records, metadata and hashes | Produce user-facing prose | Planned |
| Model Adapter | Uniform interface to any local or future model | Search | Implemented (boundary and registry; no trained model) |
| Grounding / Citation Validator | Check that every cited identifier exists and that cited evidence supports the claim | Rewrite the user's request | Planned |
| Reassessment Engine | Compare previous claims with new evidence and decide retain/qualify/correct | Silently rewrite history | Planned |
| Evaluation | Measure behaviour and detect regressions | Change production behaviour | Planned |

A component marked "Planned" has no code. Configuration and the model boundary are
implemented; [`agent/CURRENT_STATE.md`](agent/CURRENT_STATE.md) is the authority,
including on the fact that the only implementation of the boundary is a
development double rather than a language model.

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
| `Conversation` | messages, active topic, entities, open questions, prior sources | Planned |
| `AnswerRecord` | answer text, cited identifiers, model and prompt versions, correction lineage | Planned |
| Retriever / Fetcher | pluggable retrieval and fetch mechanisms | Planned |
| Reassessment | previous answer plus new evidence -> retain/qualify/correct | Planned |

"Planned" here means there is no code, and the shape described is the specification
to build against rather than a description of something that exists. Configuration,
the model boundary and the evidence records are implemented; retrieval itself is
not, so nothing has been fetched or indexed.

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

1. The Conversation Manager appends the user turn and resolves references
   ("this", "why not", "and the second one") against conversation state.
2. The Query Planner decides whether the turn can be answered from conversation
   context or needs fresh evidence, and produces zero or more retrieval queries.
3. If retrieval is needed, the pipeline fetches candidate documents under the
   fetch policy, extracts main content, chunks it with source and section
   boundaries preserved, and ranks candidates.
4. The selected chunks become an Evidence Pack with immutable identifiers. Only
   chunks in the pack are visible to generation.
5. The Model Adapter generates from the conversation plus the evidence, under a
   versioned prompt template, referencing evidence by identifier.
6. The validator checks every identifier cited against the pack, records unknown
   identifiers as errors, and attaches source metadata to the surviving
   references.
7. The answer and its metadata are recorded as an `AnswerRecord`. Nothing about
   the retrieval or model provenance is discarded.

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
- **Web fetch mechanism.** Must be local and must not require a commercial search
  API. Whether the first implementation is a search endpoint, a curated seed list
  or a sitemap-driven crawl is unresolved.
- **Reranking model.** Optional in the design. No model is chosen.
- **Local inference runtime.** Depends on which local model is chosen first,
  which depends on the tokenizer work.
- **Persistence format for conversation state.** Undecided; the type exists
  without a store.
- **Whether the model writes citations inline or in a structured side channel.**
  This is a real design question with evaluation consequences, and it is
  unresolved.
