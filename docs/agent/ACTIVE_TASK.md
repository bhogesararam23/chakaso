# Active task

## Current unit: P19 — answer-provenance enrichment

**Goal.** Make a persisted answer self-explanatory about *how* it was produced, without copying
large evidence bodies or persisting anything volatile. Inspect `AnswerRecord` first; add only fields
that genuinely explain a turn and are stable across restart — the citation-validation outcome and the
structural-grounding outcome already computed during a turn are the leading candidates. A provenance
reference must stay meaningful after a process restart; where it points at data that is no longer
available (an in-memory corpus, a recomputed evaluation), the record must say so rather than imply
the evidence can still be reconstructed. Extend the store/serialization codec and its conformance
tests, keep the in-memory and SQLite backends equivalent, and update the CLI `answers show` view.

**Why this is next.** P18 made the evidence *input* side inspectable from the CLI and hardened the
display surface against hostile content. The matching gap on the *output* side is that a stored
answer records the planner decision and retrieval strategy (P16) but not the validation/grounding
verdicts a turn actually reached, so "why was this answer what it was" is only partly auditable from
the durable record alone.

## What the previous unit completed (P18 — evidence and source usability, with adversarial security)

Implemented, tested, documented and green on Python 3.11–3.13.

- [x] A dependency-free terminal-safety primitive, `chakaso.core.safe_for_terminal`, that renders
      control characters (ESC, C0/C1, DEL) and the bidirectional embedding/isolate controls in
      untrusted text as *visible* escapes; adopted on the CLI's human-readable paths, with `--json`
      left to the serializer.
- [x] Source and evidence inspection from the CLI over the in-memory corpus — `chakaso sources
      list|show` and `chakaso evidence show` — calling the existing `Corpus`/`ingest_file`/record
      APIs, with valid `--json`, useful exit codes, and honest statements that **no durable source
      index exists** (a source appears only while its `--file` document is passed).
- [x] Source-scoped retrieval, `chakaso retrieve … --source <src_…>`, via the retrieval service's
      existing `source_ids` filter; identifiers are validated before lookup and an unknown-but-valid
      one is refused, never silently filtered to an empty result.
- [x] Adversarial regression tests: a document embedding an ESC sequence and a bidirectional override
      reaches the terminal as inert visible escapes while its substance is preserved, and `--json`
      keeps the exact bytes — proving the application's structural separation, not model robustness.

**What P18 did not do (and will not pretend to):** it did not add a durable source registry, or
cross-process source removal/replacement. There is no consumer for them under the current in-memory
architecture, and inventing a second store would violate the no-empty-scaffolding rule. It added no
real model and no real semantic judge; the retrieved-content-stays-data property remains structural.

## What this unit is not

- Not a durable source index or a second database — the corpus stays in-memory unless a real need is
  shown; do not invent persistence to look complete.
- Not a web-frontend, crawler, or a URL chooser; the fetcher stays opt-in and off every default path.
- Not a real semantic evaluator or model (P17's boundary stands).

## Ordering after this unit

1. A triggered follow-up that decides *when* to reassess, wired into the runtime behind explicit
   composition (not a fake autonomous loop).
2. A local CPU model adapter behind the `LanguageModel` boundary, only if a genuinely available,
   maintainable dependency exists; then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: offline and synthetic; a real workload has still not been run.
- K-008: the fetcher's destination screen is partial; any acquisition work must confirm no default path
  reaches the network and record the limitation rather than fix it silently.

## What previous units knowingly left undone

- No automatic, triggered correction; the correction path is still an explicit application operation.
- Nothing records across turns how often a model invents an evidence reference (K-006); the planner
  uses an anaphoric-cue heuristic, not a discourse resolver.
