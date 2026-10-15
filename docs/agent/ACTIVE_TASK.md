# Active task

## Current unit: P20 — provenance-reference integrity in answer inspection

**Goal.** An answer's `evaluation_provenance` and its `evidence_source_ids` / cited-source
references name evidence that lived in the **ephemeral in-memory corpus**, not in the durable
store. Reading a stored answer after a restart must make that distinction explicit: the record can
show *which* evidence it relied on and *how* it judged it, but it must not imply the underlying
text can still be re-read from anywhere the durable store holds. Extend answer inspection to label
references as "recorded; the cited content is not retained durably" where that is true, and keep
`--json` machine-readable and the human view terminal-safe. Prefer stating the limitation precisely
over fabricating an evidence snapshot; do not copy evidence bodies into the answer store without a
demonstrated need (weighed against privacy and size).

**Why this is next.** P19 made the turn's *evaluation outcome* durable and auditable. The honest
boundary now is that provenance *references* evidence that is not itself durable. Making that
boundary visible — rather than letting `answers show` imply the evidence is one click away — is a
correctness-and-honesty gap, not new surface area.

## What the previous unit completed (P19 — answer-provenance enrichment)

Implemented, tested, documented and green on Python 3.11–3.13 (ADR-0028).

- [x] A typed, immutable, serializable `EvaluationProvenance` snapshot (`chakaso.answers.provenance`)
      persisting the citation-structure counts, the per-claim structural-grounding statuses, the
      evaluator name+version, and the unresolved/malformed answer-text references the turn reached.
- [x] Persisted on `AnswerRecord` as `evaluation_provenance` inside the JSON document — no SQLite
      schema change, backward-compatible decode to `None`, strict rejection of malformed provenance.
- [x] The live `AnswerEvaluation` stays recomputed and unpersisted (ADR-0021 preserved): a historical
      fact and a recomputable view are now two distinct fields, and reassessment never rewrites the
      original outcome (proven by an end-to-end restart + reassessment-distinctness test).
- [x] `chakaso answers show|list` report the snapshot, labelled *recorded at the turn, not recomputed*,
      dimensions kept separate, `is_grounded`/`evidence_coverage` stated as structural only; `--json`
      exact, human view terminal-safe against control characters in persisted provenance.
- [x] Both backends preserve equivalent record semantics (conformance + cross-process restart).

**What P19 did not do (and will not pretend to):** it computes no aggregate invented-reference rate
(K-006 advanced, not closed — there is no generating model to invent references and no real workload,
K-001); it stores no evidence bodies; it added no evaluator, model or dependency; the double gains no
fabricated capability (the default record names the honest `structural-0.1`).

## What this unit is not

- Not an immutable evidence snapshot or a second store — references name evidence the durable store
  does not hold; say so rather than duplicating content without a demonstrated need.
- Not an aggregate research metric — the invented-reference *rate* needs a real generating model.
- Not a real semantic evaluator or model (P17's boundary stands).

## Ordering after this unit

1. A triggered follow-up that decides *when* to reassess, wired into the runtime behind explicit
   composition (not a fake autonomous loop).
2. A local CPU model adapter behind the `LanguageModel` boundary, only if a genuinely available,
   maintainable dependency exists; then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: offline and synthetic; a real workload has still not been run.
- K-006: the invented-reference condition is persisted per answer, but nothing aggregates it into a
  rate, and it is only durable when the SQLite backend is selected.
- K-008: the fetcher's destination screen is partial; any acquisition work must confirm no default path
  reaches the network and record the limitation rather than fix it silently.

## What previous units knowingly left undone

- No automatic, triggered correction; the correction path is still an explicit application operation.
- The planner uses an anaphoric-cue heuristic, not a discourse resolver.
