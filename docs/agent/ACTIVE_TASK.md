# Active task

## Current unit: P3 — local retrieval (not started)

**Goal.** Take documents that are already on disk, split them into evidence chunks
with stable identifiers, and rank them for a query — without embeddings, without a
network and without a model. Then assemble an `EvidencePack` from the result.

**Why this is next.** `ConversationManager.send` already accepts an `EvidencePack`,
and nothing can produce one. Until something can, citation resolution is exercised
only by tests, and the fastest way to find out whether the identifier and record
types are right is to build the smallest thing that feeds them.

**Why the smallest thing.** A lexical baseline needs no model, no index server and no
dependency, so it can be built and evaluated before any of the decisions that
retrieval would otherwise force — which embedding model, which index, whether to
rerank. Those are open questions precisely because nothing has yet established what
the retrieval layer needs from them.

### Done

Everything below is implemented and covered by tests. Nothing in this list is a claim
about answer quality.

- [x] Repository baseline: `.gitignore` (including the private-pack exclusion),
      `.gitattributes`, `.editorconfig`
- [x] Apache-2.0 license with the decision recorded (ADR-0004)
- [x] Public documentation set: README, getting started, architecture, transparency,
      retrieval, correction, training, evaluation, roadmap
- [x] Decision records ADR-0001 through ADR-0009
- [x] Research foundation: log, hypotheses, open questions, literature
- [x] Contribution guide and agent contract
- [x] Python project metadata and package skeleton (ADR-0005)
- [x] CI running format, lint, types and tests on Python 3.11, 3.12 and 3.13
- [x] Typed configuration with explicit loading and per-value provenance (ADR-0006)
- [x] Core primitives: content-derived identifiers, hashing, error base (ADR-0007)
- [x] Language-model boundary, capabilities, registry and contract tests (ADR-0002)
- [x] Source and evidence records, evidence packs and citation resolution (ADR-0003)
- [x] Conversation state: immutable, append-only turns with provenance
- [x] The conversation manager: turn orchestration, context projection, error
      semantics, and the seam where evidence is supplied (ADR-0008, ADR-0009)
- [x] A test enforcing that no module outside the composition point imports a
      concrete model adapter
- [x] A local conversation shell: `chakaso chat`, interactive and one-shot, which
      states in its own output that the engine is a development double

## Definition of done for the current unit

- Ingest a document from a local path into a `SourceRecord` with a derived
  identifier, a content hash and no network access.
- Split it into `EvidenceChunk` records that preserve section boundaries and
  positions, with identifiers that change when the text does.
- Rank chunks for a query with a lexical scorer, and record the result as a retrieval
  score rather than as anything resembling confidence.
- Assemble an `EvidencePack` from the result, so that `ConversationManager.send` can
  be given real evidence end to end.
- Retrieval is reachable through an interface, so that a later dense or hybrid
  retriever can replace the lexical one without changing callers.
- `pytest`, `ruff check`, `ruff format --check` and `mypy src` pass on Python 3.11,
  3.12 and 3.13.
- Documentation states what retrieval does, and does not imply that anything has been
  fetched from the web.

## Not started, and the order they will be taken

1. A fetch policy, then a web fetch mechanism under it (P4). The policy comes first:
   it is a blocking open question, and the security posture is currently a list of
   threats with no limits.
2. Dense embeddings and a vector index (P3/P4), once the lexical baseline shows what
   it cannot do.
3. Query planning: deciding when retrieval is needed at all (P4).
4. Claim-level support checking, then reassessment and the correction loop (P5).
5. The evaluation harness and the first hand-built benchmark (P6).
6. A local model adapter runnable on CPU (P2/P7), which is what makes the shell more
   than a plumbing demonstration.
7. Tokenizer, dataset pipeline and the tiny Transformer (P7–P8).

The ordering is deliberate: retrieval before fetching, because a fetch policy cannot
be written without knowing what the retrieval layer needs; evaluation before
training, because a model judged by nothing cannot be improved deliberately.

## Explicitly not in this unit

- Any network access. Ingestion reads local files; fetching is a later unit and needs
  a policy first.
- Embeddings, a vector index or a reranking model.
- Any correction or reassessment behaviour.
- Any real language model, tokenizer or training code.
- Any claim about retrieval quality, and any measured metric. There is no benchmark
  yet, so a number would be invented.

## What previous units knowingly left undone

Recorded here rather than discovered later:

- An answer's model identity is reported in the reply but is not stored in
  conversation state, so it is not recoverable from a transcript alone. That belongs
  to the `AnswerRecord` the correction unit introduces (K-007).
- Nothing persists how often a model invents an evidence reference, so one of the
  project's headline metrics is observable only in-process (K-006).
- Nothing resolves references to earlier turns ("that", "the second one") against
  conversation state; follow-ups carry the previous turns and nothing more specific.
