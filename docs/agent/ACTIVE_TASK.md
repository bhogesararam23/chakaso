# Active task

## Current unit: P15 — end-to-end + restart integration demonstration, test-the-tests, security review

**Goal.** Prove the assembled system as integration, not just as units: a deterministic, offline
end-to-end demonstration that runs a user query through planner → retrieval → evidence pack → the
development model → claims → citation validation → grounding → `AnswerRecord` → durable SQLite store,
then closes and reopens the store to show the grounded answer history survives the process; a
follow-up/correction path recorded against the stored history. Add deliberate "test-the-tests"
corruption checks on the critical invariants (turn atomicity, persistence integrity, retrieval
provenance, hybrid ranking, the model boundary) — corrupt the behaviour, show the test fails, restore.
Run a security review of the new runtime surface: untrusted retrieved content, prompt-injection
through evidence, persisted serialized records, filesystem/SQLite paths, and confirmation that no
default path reaches the network.

**Why this is next.** Phases 2–7 built and unit-tested durable storage, the planner, dense and hybrid
retrieval, and the runtime that composes them, and the first measured strategy comparison (E-001) is
recorded. What has not been done is an *integration* demonstration that the whole flow holds together
across a restart, the deliberate-corruption checks that prove the guarantees can fail (so that passing
means something), and an explicit security pass over the new runtime. This unit is verification and
honesty, not new capability.

## What the previous unit completed (P14 — measured retrieval-strategy comparison)

Implemented, tested, documented and green on Python 3.11–3.13. It produces real numbers; it invents none.

- [x] `chakaso.experiments.compare_retrieval_strategies`: scores lexical/dense/hybrid over the same
      dataset and reports measured per-metric deltas against a baseline; hard-codes no winner.
- [x] `retrieval_strategy_experiment` and a runner that honours a configured `retriever` strategy.
- [x] `chakaso benchmark strategies` and `chakaso benchmark retrieval --strategy …`.
- [x] The first run (E-001) recorded in `docs/research/research-log.md` with its real numbers — a
      neutral-to-negative result (dense/hybrid match lexical on recall/precision/hit, regress MRR by
      0.071 over the non-semantic fixture embedding); H7 added and left explicitly untested.

## What this unit is not

- Not a new capability or feature. It is integration proof, corruption checks, and a security review.
- Not a real model or semantic claim: the demonstration runs the deterministic double and the
  non-semantic fixture embedding, so it proves the wiring, not answer quality.
- Not automatic correction or a triggered follow-up; those remain later.

## Ordering after this unit

1. Final synchronisation sweep across docs/ADRs/research/CHANGELOG, the full 3.11–3.13 validation, CI
   green, and a clean tree with `HEAD == origin/main`.
2. Wiring the runtime into `chakaso chat` / the CLI behind typed retrieval configuration.
3. A real semantic judge behind `SemanticJudge`; then a triggered follow-up that decides *when* to
   reassess.
4. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: the demonstration is offline and synthetic; a real workload still has not been run.
- K-007 (closed by ADR-0021): answers persist across a restart — this unit demonstrates it end to end.
- K-008: the fetcher's destination screen is partial; the security pass must confirm no runtime default
  path reaches the network and record any limitation in scope rather than fixing it silently.

## What previous units knowingly left undone

- Nothing records across turns how often a model invents an evidence reference; the runtime detects it
  per turn, but no store aggregates the rate across a session (K-006).
- Nothing resolves references to earlier turns; the planner uses an anaphoric-cue heuristic, not a
  discourse resolver.
