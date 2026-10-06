# Architecture

## Current phase

Chakaso is in the repository-foundation phase.

There is no runtime architecture beyond the package and CLI skeleton yet.

## Target boundaries

The planned system is organized around these boundaries:

```
User
  |
Conversation Manager
  |
Query / Task Planner
  |----------------------|
Direct Context        Retrieval
                           |
                     Evidence Pack
                           |
                       Local Model
                           |
                Grounding / Citation
                           |
                         Answer
```

A follow-up challenge should be able to route back through conversation state and a dedicated reassessment path.

## Boundary rules

### Language model

The model is responsible for language generation and model-local reasoning. It must not own source URLs or invent provenance records.

### Retrieval

Retrieval owns source discovery, fetching, parsing, normalization, deduplication, chunking, ranking, and source metadata.

### Evidence

Evidence is represented with stable source and chunk identifiers so generated answers can reference verified material without the model inventing URLs.

### Conversation

Conversation state owns prior messages, active context, relevant entities, previous sources, claims, and unresolved questions.

### Reassessment

When new information challenges an earlier answer, the system should compare claims against the new evidence and prefer correction over defending the earlier response.

### Evaluation

Evaluation must remain separate from runtime logic so benchmarks do not silently become product behavior.

## Non-goals for the foundation phase

- building a toy chatbot and calling it the final model
- hiding uncertainty behind fake numeric confidence
- coupling the core architecture to a commercial inference provider
- treating retrieval output as trusted instructions
