# Active task

## Current unit: P2 — the conversation manager

**Goal.** The orchestration boundary on top of the conversation state records that
already exist. It accepts a user turn, projects the relevant conversation context
into model-facing messages, invokes the model through the `LanguageModel` boundary,
validates the result, records the assistant turn, and returns the new immutable
state.

**Why this is next.** P1 built the pieces: configuration, identifiers, the model
boundary, evidence records and conversation state. None of them talk to each other
yet. The manager is the first component with behaviour — the thing that makes a
conversation happen — and it is where the application-level error semantics and the
seam for future retrieval get fixed. Retrofitting either after retrieval and
correction exist would be far more expensive.

**Why it is not a chatbot.** The response engine is still a deterministic
development double. The manager's job is to be correct and replaceable, not to
appear intelligent.

### Done

Everything below is implemented and covered by tests. Nothing in this list is a
claim about answer quality.

- [x] Repository baseline: `.gitignore` (including the private-pack exclusion),
      `.gitattributes`, `.editorconfig`
- [x] Apache-2.0 license with the decision recorded (ADR-0004)
- [x] Public documentation set: README, getting started, architecture,
      transparency, retrieval, correction, training, evaluation, roadmap
- [x] Decision records ADR-0001 through ADR-0007
- [x] Research foundation: log, hypotheses, open questions, literature
- [x] Contribution guide and agent contract
- [x] Python project metadata and package skeleton (ADR-0005)
- [x] CI running format, lint, types and tests on Python 3.11, 3.12 and 3.13
- [x] Typed configuration with explicit loading and per-value provenance (ADR-0006)
- [x] Core primitives: content-derived identifiers, hashing, error base (ADR-0007)
- [x] Language-model boundary, capabilities, registry and contract tests (ADR-0002)
- [x] Source and evidence records, evidence packs and citation resolution (ADR-0003)
- [x] Conversation state: immutable, append-only turns with provenance

### In progress

- [ ] The conversation manager: turn orchestration, context projection, error
      semantics, and the seam where evidence will later be supplied
- [ ] A minimal local conversation shell in the CLI, honest about what produces
      its replies

## Definition of done for this unit

- The manager depends on the `LanguageModel` boundary and imports no concrete
  adapter; a test enforces that structurally rather than by review.
- Sending a user turn appends a user turn and an assistant turn and returns the new
  immutable state, leaving the previous state untouched.
- Follow-up turns receive the projected conversation context, and the number of
  projected turns is bounded by configuration rather than by a constant in code.
- Failures are distinguishable: invalid user input, invalid generated output, and
  model failures each have their own type, and a failure leaves the conversation
  unchanged.
- Evidence, when supplied, determines what citations may resolve; a reference
  outside the supplied evidence is never resolved to a source and is reported to
  the caller.
- `pytest`, `ruff check`, `ruff format --check` and `mypy src` pass on Python 3.11,
  3.12 and 3.13.
- Documentation states what the manager does, and that the response engine is a
  development double.

## Not started, and the order they will be taken

1. Local document ingestion and a lexical retrieval baseline (P3), which is what
   will supply the evidence the manager already accepts.
2. Chunking and an evidence store behind the existing record types (P3).
3. A fetch policy, then a web fetch mechanism under it (P4). The policy comes
   first: it is a blocking open question.
4. Query planning: deciding when retrieval is needed at all (P4).
5. Claim-level support checking, then reassessment and the correction loop (P5).
6. The evaluation harness and the first hand-built benchmark (P6).
7. A local model adapter runnable on CPU (P2/P7), which is what makes the shell
   more than a plumbing demonstration.
8. Tokenizer, dataset pipeline and the tiny Transformer (P7–P8).

The ordering is deliberate: retrieval before fetching, because a fetch policy
cannot be written without knowing what the retrieval layer needs; evaluation before
training, because a model judged by nothing cannot be improved deliberately.

## Explicitly not in this unit

- Any retrieval, fetching, embedding or indexing. The manager accepts an evidence
  pack from a caller; it does not obtain one.
- Any correction or reassessment behaviour.
- Any real language model, tokenizer or training code.
- Any persistent store for conversation state.
- Any claim about answer quality, and any numeric confidence.

These are excluded because the unit is about fixing the orchestration boundary
before the components on either side of it exist. A manager that guessed at
retrieval would have to be rewritten when retrieval arrives, and one that dressed up
the development double as an answer engine would be worse than that.
