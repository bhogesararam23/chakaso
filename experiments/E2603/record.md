# E2603 — first grounded-answer evaluation baseline over the self-authored P27 workload

- Status: completed
- Date: 2026-10-19
- Hypothesis: informs the strong form of [K-001](../../docs/agent/KNOWN_ISSUES.md) and the H1–H7
  questions — it does **not** test a hypothesis about model capability. It answers a narrower,
  concrete question: does the P27 grounded-answer harness (ADR-0031) actually run the real
  pipeline, score the intended dimensions separately, and *surface real failures* rather than
  reporting only green?
- Owner: autonomous engineering session (P27)
- Git commit: `7d4814dcd150332617e80c93772eb579dca22352`
- Config: `chakaso benchmark grounded` defaults — lexical retrieval, `top_k=5`, self-authored
  workload `ga-kestrel-library` v1.0.0.

## Motivation

Every prior measurement (retrieval benchmark, correction benchmark, E-001, E2601/E2602) scored
machinery or a toy model over synthetic fixtures. P27 built the first workload whose units are
*grounded-answer cases* — a question, acceptable and unacceptable evidence, required facts,
prohibited claims, an expected disposition, provenance and licence. Before trusting any future
model against it, the harness itself must be shown to run the pipeline and to be honest about what
fails. This record is that validation and the first baseline numbers on the workload.

## Baseline

The comparison is between answer *providers* over the identical workload and the identical lexical
retrieval, which is what a harness validation needs:

- **reference oracle** (`ReferenceAnswerProvider`) — replays each case's own required facts and
  cites only gold chunks retrieval surfaced. This is an oracle, not a model: it exists to prove the
  harness scores a correct, well-cited answer as correct and an abstention as an abstention.
- **non-answering provider** (`NonAnsweringProvider`) — abstains on every case, the honest shape of
  today's runtime, where no model produces a grounded answer.

Retrieval is the shared, genuinely-measured layer in both: the same `RetrievalService` over the same
self-authored corpus. There is no trained or pretrained model in either arm.

## Configuration

Resolved settings, recorded in full: strategy `lexical`, `top_k=5`, split filters `all`,
`development`, `heldout`; provider ∈ {reference, none}. Dataset `ga-kestrel-library` v1.0.0,
content fingerprint `d2d2ee00d117b88a`, 13 self-authored passages, 14 cases (9 development /
5 held-out). No random component: the run is fully deterministic.

## Data

- Name/version: `ga-kestrel-library` v1.0.0 (fingerprint `d2d2ee00d117b88a`).
- Source: **self-authored** for Chakaso; hand-written policy prose about a fictional library. No
  external text copied; nothing fetched (the fetcher is off every default path, ADR-0012).
- Licence: Apache-2.0 (the project's own licence); no third-party terms apply. Redistribution: no
  restriction, the project owns the text. This is stated per-passage in `SELF_AUTHORED_PROVENANCE`
  and is enforced by the schema — a source without provenance fails to load (ADR-0031).
- Split and contamination: development/held-out separation is enforced as a structural invariant —
  no passage grounds a case on both sides. Selection is done on development; the held-out number is
  reported separately. Cases span direct, multi-source, distractor/forbidden, conflicting,
  ambiguous, missing-evidence and unsupported-premise situations.

## Model

No language model participates. The retrieval layer is BM25 (`LexicalRetriever`). The two answer
"arms" are an oracle and a deliberate abstainer, both in-process and numpy-free. This section is
deliberately thin: there is no model to describe, and saying so is the point.

## Environment

Windows, CPython 3.13.5, CPU-only, offline. Development install `.[dev]` (numpy **not** installed;
the training tests skip, K-009). The grounded pipeline is numpy-free and runs in the base install.

## Reproducibility

From a clean environment after `python -m pip install -e ".[dev]"`:

```text
python -m chakaso benchmark grounded --provider reference --split all   # human report
python -m chakaso benchmark grounded --provider reference --split heldout --json
python -m chakaso benchmark grounded --provider none      --split all --json
```

The JSON report is deterministic (sorted keys); rerunning the same command on the same machine yields
identical bytes. The report's `configuration` records reproducibility metadata — interpreter version,
implementation, platform and package version — alongside the dataset id/version/content fingerprint,
provider label and whether it was an oracle; the workload fingerprint is printed in every report.

## Metrics

Definitions and denominators (implemented in `chakaso.benchmark.grounded_metrics`, each with
independent hand-checkable tests):

- Retrieval **recall@k / precision@k / MRR / hit rate**: over *scored* cases (those naming gold
  evidence); no-gold cases are counted as abstentions, not folded into the mean.
- **Forbidden hits**: cases whose retrieved set contains a passage the case marked unacceptable.
- **False retrievals**: no-gold (abstention) cases where retrieval returned anything.
- **Evidence coverage** (`mean_evidence_coverage`): supported claims / all claims, per scored case.
- **Citation validity**: valid citations / citations issued (valid+unknown+irrelevant).
- **Unsupported claim rate**, **required-fact coverage** (present/required), **prohibited-claim
  rate** (appeared/forbidden), **appropriate-abstention rate** (disposition matched / decisions),
  **grounded-answer rate** (grounded answers / answered cases). A dimension with an empty
  denominator is reported `0.0`, never a fabricated success.

## Results (measured, commit `7d4814d`)

Reference oracle, split **all** (11 scored, 3 abstention cases):
- Retrieval: recall@k **0.8182**, precision@k 0.2045, MRR 0.8636, hit rate 0.9091, forbidden hits
  **3**, false retrievals **3**.
- Answer (over 11 answered cases): mean evidence coverage **0.8182**, citation validity 1.0,
  unsupported-claim rate 0.0909, required-fact coverage 1.0, prohibited-claim rate 0.0,
  appropriate-abstention rate 1.0, grounded-answer rate **0.8182**. Failures: 0.

Reference oracle, split **heldout** (4 scored, 1 abstention):
- Retrieval: recall@k **1.0**, precision@k 0.2125, MRR 0.875, hit rate 1.0, forbidden hits 1,
  false retrievals 1.
- Answer (over 4 answered): mean evidence coverage 1.0, citation validity 1.0, grounded-answer rate
  1.0, appropriate-abstention rate 1.0. Failures: 0.

Reference oracle, split **development** (7 scored, 2 abstention):
- Retrieval: recall@k 0.7143, MRR 0.8571, hit rate 0.8571, forbidden hits 2, false retrievals 2.
- Answer (over 7 answered): mean evidence coverage 0.7143, unsupported-claim rate 0.1429,
  grounded-answer rate 0.7143.

Non-answering provider, split **all**: retrieval identical (it is the same pipeline); answer
answered_cases **0**, appropriate-abstention rate **0.2143** (it over-abstains every answerable
case), grounded-answer rate 0.0.

## Interpretation

1. The harness runs the real pipeline and scores it end to end with zero stage failures, and its
   arithmetic is pinned by independent tests — so the numbers are trustworthy *as measurements of
   the harness and of lexical retrieval over this workload*.
2. The oracle arm proves the pass path: where retrieval surfaced gold and the answer carried the
   required facts and correct citations, the case scores grounded with appropriate abstention on
   the unanswerable cases. This validates the metrics, not any model.
3. The negative findings are the valuable part, and the dimensions keep them visible instead of
   merging them: recall@k 0.82 against precision@k 0.20 (top_k=5 over a 13-passage pool), **three
   forbidden hits** and **three false retrievals** — lexical retrieval returns its top-k even for
   the three abstention cases, so an abstention case that should retrieve nothing retrieves
   something. A merged score would hide this; separate dimensions and per-case inspection do not.
4. With the oracle, per-answer grounding tracks retrieval recall: over each split the
   grounded-answer rate and mean evidence coverage equal that split's recall (0.8182 / 1.0 /
   0.7143), because the oracle is fully grounded exactly on the cases whose required gold retrieval
   surfaced, and makes an unsupported claim on the ones where it did not. This equality is the
   expected cascade — it validates the metric wiring, and says nothing about model skill.
5. `mean_evidence_coverage` is averaged over answered cases only. An earlier aggregation folded the
   abstention cases' definitionally-zero coverage into the mean (reading 0.6429 for split=all),
   conflating "declined to answer" with "answered poorly"; the denominator was corrected before
   these numbers were taken, and a regression test now pins it.

## Limitations

- The workload is self-authored, tiny (14 cases), and author-written — it is easier than real text
  and cannot support any claim about answer quality on natural language. K-001 stands.
- There is **no model** in either arm; the "answer" side measures the harness via an oracle and a
  deliberate abstainer, not a generator. No number here is model quality.
- Small, fixed sample; no confidence interval, no significance test (there is no variance to test —
  the run is deterministic — and reporting one would be theatre).
- Timeouts and malformed real-model outputs are handled as failures but not exercised: no provider
  here is slow or malformed, so those paths are covered by unit tests, not by this run.

## Conclusion

The P27 grounded-answer evaluation capability is **implemented, tested, and executed**: it runs the
real retrieval pipeline over a versioned, provenance-bearing, held-out self-authored workload;
scores retrieval and answer dimensions separately; distinguishes retrieval/generation/evaluation
failures; and produces deterministic human and machine reports. It honestly reports both a passing
oracle path and genuine retrieval weaknesses (forbidden hits, false retrievals on abstention
cases). What it does **not** establish — and must never be quoted as establishing — is anything
about the quality of a real language model's grounded answers, because no such model exists.

## Follow-up

The concrete, evidence-driven next step this baseline points at is a **Workstream-E correctness
improvement**: lexical retrieval answering abstention cases it should leave empty (three false
retrievals) and surfacing forbidden passages (three forbidden hits). A retrieval score-floor /
evidence-sufficiency gate, justified by these specific failure cases and covered by regression
tests, is the next candidate — not a larger workload and not a model. A *useful* answer capability
still requires a real model behind the boundary (K-001), which remains out of scope.
