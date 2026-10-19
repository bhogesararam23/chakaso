# E2602 — Under early stopping selected on validation, what is the held-out loss across seeds, and does E2601's overfitting reproduce?

- Status: completed
- Date: 2026-10-19
- Hypothesis: none — a controlled replication/protocol experiment, not a test of H1–H7. It turns E2601's
  single-seed point result into a mean-with-spread and checks, on a properly held-out test set, that
  early stopping is the right reading of the loop and that the overfitting E2601 measured reproduces.
- Owner: Chakaso autonomous engineering session (P26)
- Git commit: `20986f8` (the controlled driver `chakaso.training.controlled`; all layers identical unless noted)
- Config: `CONTROLLED_CONFIG` = context_size 4, embed_dim 8, hidden_dim 12, learning_rate 0.02,
  steps 300, eval_every 10; seeds (7, 21, 1337, 2024, 42); validation_fraction 0.2, test_fraction 0.2.

## Motivation

E2601 ([`../E2601/record.md`](../E2601/record.md)) produced the project's first measured training result,
but as a single seed on a two-way split, and its held-out curve overfit after step 50. Two decisions this
informs: (1) how the model should actually be *reported* — at its validation-selected optimum, not a
fixed overshooting budget; (2) whether the result survives contact with seeds and a genuinely untouched
test set, or was one lucky partition. An experiment that would not change the confidence we place in
E2601 would not be worth running; this one does.

## Baseline

Same as E2601, but computed on the **test** split of each seed's three-way split, so it is a valid lower
bound for the reported held-out number:

- **Uniform** over the (per-seed) vocabulary: `ln(V)`.
- **Unigram**: frequency of the *training* targets, Laplace-smoothed, scored on the test tokens.

## Configuration

A fixed protocol, identical architecture/optimizer to E2601; the only changes are the three-way split and
the early-stopping selection rule. The rule is stated before the result: **stop at the first evaluation
step that achieves a new minimum validation loss; report the test loss at that step.** The selection reads
validation only; test is touched once, for reporting. The five seeds were pre-declared, not chosen after
seeing outcomes.

## Data

- Corpus `chakaso-toy-char-v1` v1.0.0, self-authored, Apache-2.0 (`chakaso.training.corpus`).
- Split: `split_three_way` — each line unit goes wholly to exactly one of train/validation/test (units
  pairwise-disjoint and preserved exactly once, tested); vocabulary fit on **training text only**, so a
  validation- or test-only character is the reserved unknown id (tested as an invariant over 16 seeds).
- Per-seed sizes (train / validation / test windowed positions): seed 7 267/77/78, 21 262/77/83,
  1337 254/86/82, 2024 244/85/93, 42 258/81/83. Vocabulary size varies 26–28 by which characters the
  training partition happens to contain.

## Model

The `chakaso.training.model.CharCausalMLP` character model, initialized from random at each seed.
984 parameters. Not a Transformer; makes no claim to language.

## Environment

Windows 11 (10.0.26200), CPython 3.13.5, numpy 2.4.6, CPU-only (no CUDA used). Facts about one host at
one moment; no wall-clock/peak-memory number was measured to a stated precision, so none is reported.

## Reproducibility

`run_controlled_experiment()` (fixed seeds, fixed config, deterministic full-batch updates) reproduces
every number below exactly. With the `[train]` extra installed:

```
python -c "from chakaso.training.controlled import run_controlled_experiment as r; print(r())"
```

## Metrics

Per seed: best validation loss, the step it occurred at, and the **test** loss at that validation-selected
step; plus the test loss at the un-early-stopped final step (300). Aggregate across seeds: mean and sample
standard deviation of the test loss at the selected step. All mean cross-entropy in nats, via
`chakaso.training.model` forward — the same evaluator as E2601.

## Results

Per-seed measured values (nats):

| seed | V | selected step | best val | test @ step | test @ final | unigram | uniform |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 27 | 40 | 1.9667 | 1.7296 | 2.4940 | 3.0529 | 3.2958 |
| 21 | 26 | 40 | 1.8100 | 1.7986 | 3.2289 | 2.9680 | 3.2581 |
| 1337 | 28 | 40 | 1.8010 | 1.5784 | 2.2369 | 2.9250 | 3.3322 |
| 2024 | 27 | 30 | 1.9273 | 2.0655 | 3.1838 | 2.9957 | 3.2958 |
| 42 | 28 | 50 | 1.8189 | 1.8167 | 2.7582 | 2.9626 | 3.3322 |

Aggregate across the five seeds:

| Quantity | Value |
| --- | --- |
| Mean test loss at the validation-selected step | **1.7978 nats** |
| Sample stdev of that | **0.1767** |
| Mean test loss at the final (step-300) budget | 2.7804 nats |
| Mean unigram baseline (test) | 2.9808 nats |

## Interpretation

Under early stopping chosen on validation, the model's held-out **test** loss is **1.80 ± 0.18 nats**,
comfortably below both the unigram baseline (2.98) and the uniform baseline (~3.3), on **every** seed.
The result is not one lucky partition: all five seeds generalize below the baselines, and the spread is
small relative to the margin over the baseline (≈1.18 nats).

E2601's overfitting reproduces across seeds: the val-selected step is always early (30–50), and the
test loss at the un-early-stopped step 300 is worse on every seed (mean 2.78, i.e. ~1.0 nat worse than
the early-stopped 1.80). This confirms that a fixed 300-step budget overshoots the held-out optimum and
that early stopping — not more training — is the correct reading of this data.

## Limitations

- Still the same 344-line-scale synthetic toy: five seeds reduce sampling noise but not the corpus's
  unreality. The absolute loss and the near-constant selected step are properties of this tiny data, not
  general language-modeling findings.
- Three-way split leaves small validation (77–86) and test (78–93) sets; both the selected step and the
  reported test number are individually high-variance, which is why five (not one) are reported.
- Early stopping selects on validation; with a 20%-validation set that selection is itself noisy. A test
  loss below unigram on every seed is a strong signal here but not a guarantee at this scale.
- No wall-clock/memory measurement; no comparison to a different architecture or tokenizer (that would
  be H6-adjacent and needs its own design).

## Conclusion

Moderate strength, and now with a spread: the from-scratch training loop genuinely reduces held-out loss
below trivial baselines, reproducibly across seeds, when the budget is chosen by a stated early-stopping
rule; E2601's single point was not an artifact of its seed. Stronger than E2601 (mean ± sd, proper test
set), and still bounded to a toy — no useful language model, no real workload; K-001 stands.

## Follow-up

The learning loop is now established with an error bar. The evidence no longer points at more of the same:
the next valuable step is *breadth*, not repetition — either the adapter path (make a trained checkpoint
servable behind `LanguageModel`, which P25 has begun) exercised on a held-out generation-quality
measurement, or designing the small, human-inspectable, legally-usable grounded-answer evaluation corpus
the project's actual hypotheses (H1–H7) require but do not yet have. A Transformer/subword tokenizer is
justified only when there is a question the character model cannot be used to ask.
