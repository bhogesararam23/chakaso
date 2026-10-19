# E2604 — retrieval evidence-sufficiency decision: candidate evaluation against the E2603 baseline

- Status: completed
- Date: 2026-10-19
- Hypothesis: tests the strong-form K-001/K-010 question narrowly — can a deterministic,
  corpus-statistical evidence-sufficiency policy (ADR-0032) distinguish "no adequate evidence" from
  "found k things" on the P27 workload **without** degrading recall on the answerable cases? It is a
  retrieval-policy measurement, not a model-capability test (no model answers; the oracle measures the
  harness).
- Owner: autonomous engineering session (P28)
- Git commit: `bfb88bc04ee89323bb4275307e3bfd5015143b98`
- Config: lexical retrieval, `top_k=5`, `DiscriminatingTermSufficiencyPolicy(max_document_fraction=0.25)`;
  workload `ga-kestrel-library` v1.0.0, fingerprint `d2d2ee00d117b88a`.

## Motivation

E2603 (the baseline, left immutable) measured that lexical retrieval produced a **false retrieval on all
three abstention cases** and surfaced forbidden evidence on three cases. The project cannot abstain
correctly if retrieval always hands the answerer a pack. P28 adds a separate, deterministic sufficiency
decision so the system can tell "there is nothing here that supports an answer" from "here are k
candidates." This record measures that decision against the E2603 baseline and the recall it must not
cost.

## Baseline

The baseline is **E2603 itself** — the same workload, the same `RetrievalService`, the same reference
oracle — run with the sufficiency policy disabled (the default CLI path and the `--split` runs recorded
in [`../E2603/record.md`](../E2603/record.md)). The candidate is the identical run with
`DiscriminatingTermSufficiencyPolicy` enabled (opt-in, `--sufficiency`). Only the sufficiency decision is
added; retrieval, providers and every other dimension are byte-for-byte the same code path.

## Configuration

Fully recorded: strategy `lexical`, `top_k=5`; policy `discriminating-term`,
`max_document_fraction=0.25`, a short closed-class function-word list (`STOP_WORDS`); dataset
`ga-kestrel-library` v1.0.0 (13 passages / 14 cases / 9 development / 5 held-out). No random component.

## Data

Same as E2603: self-authored, Apache-2.0, human-inspectable, held-out-split workload, provenance enforced
by the schema, no external text, offline. The original baseline record is not overwritten; the workload
fingerprint is identical across both records.

## Model

No language model participates. Retrieval is BM25; the answer arm is the reference oracle (an oracle,
explicitly not a model). The new component is a deterministic corpus-statistical decision, not a model.

## Environment

Windows, CPython 3.13.5, CPU-only, offline, numpy-free. The sufficiency module adds only stdlib
(`collections`, `math`-free); the base install is unchanged.

## Reproducibility

```text
python -m chakaso benchmark grounded --split all     --sufficiency   # candidate, human report
python -m chakaso benchmark grounded --split all     --sufficiency --json
python -m chakaso benchmark grounded --split heldout --sufficiency --json
python -m chakaso benchmark grounded --split all     --json          # baseline path (sufficiency off)
```

## Metrics

Beyond E2603's retrieval and answer dimensions, P28 adds a separately-reported sufficiency dimension
(formulas and denominators in `chakaso.benchmark.grounded_metrics` and the runner, each hand-tested):

- **Sufficiency accuracy** = cases whose decision matched the case's answerability / assessed cases.
- **False abstentions** = answerable cases judged insufficient (a recall loss; denominator the answerable
  set).
- **Missed abstentions** = unanswerable cases judged sufficient (the E2603 defect; denominator the
  abstention set).
- Retrieval `recall@k / MRR / forbidden_hits / false_retrievals` are reported unchanged, because the
  policy is additive and does not touch what retrieval returns (ADR-0032).

## Results (measured, commit `bfb88bc`)

Reference oracle + sufficiency policy, split **all** (11 answerable, 3 abstention):
- Retrieval (unchanged from E2603): recall@k 0.8182, MRR 0.8636, forbidden hits 3, false retrievals 3.
- Answer (unchanged): mean evidence coverage 0.8182, grounded-answer rate 0.8182, appropriate-abstention
  rate 1.0.
- **Sufficiency (new): assessed 14, accuracy 1.0, false abstentions 0, missed abstentions 0.**

Split **held-out** (4 answerable, 1 abstention): retrieval recall@k 1.0, forbidden hits 1, false
retrievals 1 (unchanged); sufficiency assessed 5, accuracy 1.0, false abstentions 0, missed 0.

Split **development** (7 answerable, 2 abstention): sufficiency assessed 9, accuracy 1.0, 0 false, 0
missed.

For contrast, the baseline path (`--sufficiency` off) reports `assessed_cases = 0` on every split — the
E2603 numbers are not disturbed.

## Interpretation

1. On this workload the policy separates answer from abstention **perfectly** (accuracy 1.0): every one
   of the three E2603 abstention cases is now judged insufficient (3 previously-false-positive retrievals
   corrected at the decision level), while **not one** of the eleven answerable cases is wrongly
   abstained — so recall is fully preserved at the decision level, which is the trade-off the brief
   demanded be measured on both sides.
2. The retrieval metrics are deliberately identical to E2603. The policy is additive: `RetrievalService`
   still returns top-k, so the *retrieval* `false_retrievals` count stays 3. What P28 changes is not what
   retrieval hands back but whether the system has a correct, inspectable signal that the pack is not
   adequate evidence. That signal is what a future answer path (P32) will consume to abstain.
3. `forbidden_hits` (3) is also unchanged and out of scope here: it is a ranking/precision problem (a
   genuinely-answerable query retrieving a passage the case forbids), not an evidence-sufficiency
   problem. Claiming P28 "fixed" it would be false.

## Limitations

- **The threshold's working band is narrow and corpus-specific.** Sweep evidence (recorded in ADR-0032):
  on this 13-document corpus the policy is correct only for `max_document_fraction ≈ 0.23–0.31`; below it
  the answerable `book` case (df 3) is wrongly abstained, above it the noise term `library` (df 4) becomes
  "discriminating" and the abstentions leak. A single number tuned to this corpus is **not** a calibrated,
  transferable threshold, and there is no evidence a working band exists for another corpus. This is the
  central risk P29 must test, not a result to generalise.
- **Single-document corpora break it.** With `max_document_fraction < 1`, ceiling < 1, so nothing counts
  as discriminating and every answerable case is a false abstention (asserted by test). A regression test
  exercises this on purpose.
- No model answers; the "answer" arm is an oracle. Nothing here is evidence of answer capability
  (K-001, strong form, stands).
- The runtime does not yet act on the decision; it is measured here before being wired (P32). The
  non-answering provider already abstains everywhere, so end-user behaviour is unchanged today.
- Tiny sample, deterministic; no confidence interval or significance test (none is meaningful).

## Conclusion

A deterministic, offline, inspectable evidence-sufficiency policy correctly identifies all three
E2603 unanswerable false-retrieval cases as insufficient with zero cost to recall on the answerable
cases, on the shipped self-authored workload. It is a real, measured improvement to the *decision* layer
and is honestly bounded: it does not change retrieval's candidate emission, does not address forbidden
hits, depends on a narrow corpus-specific threshold band, and proves nothing about model capability.

## Follow-up

P29 (calibration and robustness): attempt to calibrate `max_document_fraction` and the function-word set
under a strict development/held-out protocol, test whether a working band survives a modest, justified
extension of the workload with hard negatives, and add tests that prevent held-out labels from touching
selection — or record honestly that this corpus is too small to support calibration.
