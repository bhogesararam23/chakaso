# Active task

## Current unit: P7 — measurement and wiring the foundations enable

**Goal.** Put the evaluation and correction machinery to work in the two ways that are honest
without a language model: (1) a first *correction development benchmark* — a small, labelled set
of prior-answer-plus-new-evidence cases run through `chakaso.correction.reassess` and scored by
`correction_metrics`, reporting correction success, unjustified persistence and unnecessary
revision as a development instrument; and (2) wiring reassessment into an explicit follow-up path
so a caller can hand a prior answer's claims and the standing evidence to `reassess_followup` and
obtain a recorded `CorrectionRecord`. Neither generates revised prose: there is no model.

**Why this is next.** The substrate, the retrieval development benchmark and the claim / citation
/ grounding / answer-evaluation / correction primitives all exist and are tested. What is missing
is using them as a loop and measuring the loop — the same move that turned the metric functions
into a benchmark. It stays offline, deterministic and CPU-only, and it advances K-001 for the
correction path the way the retrieval benchmark advanced it for retrieval: a labelled set, not a
claim about a real workload.

**Why not the model first.** A correction whose "revised answer" is produced by a model cannot be
measured until a model exists. Deciding and recording the correction can, and that decision rule
(ADR-0016) is what a benchmark should pin down before any generation is involved.

## What the previous unit completed (P5/P6 — evaluation and correction foundations)

All of the following is implemented, tested, documented and green on Python 3.11–3.13. None of
it is a claim about answer quality or a measured model-quality result.

- [x] Retrieval development benchmark (`chakaso.benchmark`): case schema with gold/forbidden
      evidence, strict JSONL/manifest loader, synthetic fixtures, a runner, text/JSON reports that
      label themselves a development instrument, and a pinned-fingerprint regression (ADR-0013)
- [x] Claims (`chakaso.claims`): immutable `Claim` with a content-derived id and an evaluation
      status that is never truth; a replaceable `ClaimExtractor` boundary; claim/evidence links
      (ADR-0014)
- [x] Structural citation validation and metrics (`chakaso.citation`): valid / unknown /
      irrelevant / uncited, and citation precision/recall — presence and set membership only,
      never semantic support
- [x] Grounding (`chakaso.grounding`): a `GroundingEvaluator` boundary with a structural evaluator
      (supported / unsupported / not_evaluated) and a manual evaluator as the only route to
      contradicted / uncertain; a `Contradiction` names no winner and precedence is caller-supplied
      (ADR-0015)
- [x] Answer evaluation (`chakaso.evaluation.evaluate_answer`): citation and grounding combined
      into separate dimensions (`evidence_coverage`, `is_grounded`) with no merged quality score
- [x] Correction foundation (`chakaso.correction`): the decision rule (`decide_claim` /
      `decide_answer`, ADR-0016), `reassess` over the grounding boundary, an append-only
      `CorrectionRecord` with content-derived lineage, structural `correction_metrics`, and the
      `reassess_followup` / `merge_evidence` composition
- [x] `chakaso benchmark` — run the development benchmark from the command line (text or JSON)
- [x] Security and determinism regressions: analysis layers import nothing networked; untrusted and
      prompt-injection text is inert data identified by the caller's reference, never by its body;
      the benchmark is path- and clock-independent; reports and record ids are byte-deterministic

## What this unit is not

- Not a model, a tokenizer or training. The only model is the deterministic development double.
- Not semantic grounding. `contradicted` and `uncertain` come only from a supplied judgement; a
  `SemanticGroundingEvaluator` is a named future implementation of the existing boundary, not a
  present capability.
- Not an automatic correction loop. Nothing decides on its own that a turn needs reassessment, and
  no correction produces new prose without a model.
- Not persistence. `CorrectionRecord` lives in memory; the `AnswerRecord` that would store a
  `correction_of` lineage durably does not exist (K-007).
- Not a dense retriever, a query planner or a live crawl. Retrieval is lexical and explicitly
  invoked; fetching stays opt-in and off every default path (ADR-0012).

## Ordering after this unit

1. Persistence: an `AnswerRecord` with `correction_of` lineage and a durable correction history.
2. A semantic grounding evaluator behind the `GroundingEvaluator` boundary, validated against the
   manual judgement the interface already accepts.
3. Query planning: when to retrieve, and which sources (P4).
4. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer (P7–P8).

## Known gaps that bound this unit

- K-001: nothing has been measured against a real workload. The retrieval development benchmark
  and the correction development benchmark this unit adds are labelled, synthetic instruments;
  they do not close K-001.
- K-008: the fetcher blocks private address literals but not a hostname resolving to a private
  address; fetching stays opt-in and offline by default. It does not block local measurement.

## What previous units knowingly left undone

- An answer's model identity is reported in a reply but not stored in conversation state, so it is
  not recoverable from a transcript alone — the `AnswerRecord` above addresses it (K-007).
- Nothing records across turns how often a model invents an evidence reference;
  `unresolved_reference_count` exists but no run feeds it (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); follow-ups carry the
  previous turns and nothing more specific.
