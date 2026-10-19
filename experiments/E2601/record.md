# E2601 — Does a from-scratch character model, trained by a real optimizer, lower held-out next-token loss below a trivial baseline?

- Status: completed
- Date: 2026-10-19
- Hypothesis: none — this is a capability-establishing experiment, not a test of H1–H7. Its question is
  whether Chakaso can *train and honestly measure* a language model at all (the gap K-001 records), the
  precondition for every later hypothesis that needs a real model.
- Owner: Chakaso autonomous engineering session (P24, gated by ADR-0030)
- Git commit: `a50dfa70e808308d5010c16bf781ae3db28fe17c` (the experiment driver; all layers identical
  unless noted)
- Config hash: dataset digest `8fcd73edf5583474`; vocabulary fingerprint `048217843a73e089`;
  checkpoint fingerprint `e62eebdcd0bba531ad00edaa3ae8e3cb525d44d9a5edb26e9b32242f2838ebe6`

## Motivation

Chakaso has, until now, never trained a model or produced a measured learning result
([`../../docs/research/research-log.md`](../../docs/research/research-log.md), K-001). P22 assessed
feasibility and ADR-0030 admitted `numpy` as an optional `[train]` dependency; this experiment is the
smallest honest thing that turns that decision into the project's *first genuinely measured number*. The
decision it informs: whether the from-scratch path can establish a reproducible learning loop before the
project considers anything larger. An experiment that would not change any decision should not be run —
this one decides whether the whole Option-B direction is credible.

## Baseline

Two predictors, measured on the *same held-out tokens* as the model, in nats of mean cross-entropy:

- **Uniform** over the vocabulary: a constant, `ln(V)`. The floor of knowing only the alphabet.
- **Unigram**: the frequency distribution of the *training* targets, Laplace-smoothed (alpha = 1),
  evaluated on held-out tokens. The floor of knowing how often characters occur but not their order.

Beating only "uniform" would be weak; the load-bearing comparison is against the **unigram** baseline,
because a model that merely reproduces character frequencies has learned nothing about context.

## Configuration

The run is fully pinned by `chakaso.training.experiment.DEFAULT_CONFIG`:

```
context_size   = 4       # previous characters the model conditions on
embed_dim      = 8
hidden_dim     = 12
learning_rate  = 0.02    # Adam
seed           = 7       # master seed: initialization and split selection
steps          = 300     # full-batch optimization steps
eval_every     = 25      # held-out evaluation cadence (final step always recorded)
validation_fraction = 0.2
optimizer      = Adam (in-project, chakaso.training.optimizer)
loss           = softmax cross-entropy (chakaso.training.model), gradients derived by hand
```

## Data

- Corpus: `chakaso-toy-char-v1` version `1.0.0` — **self-authored for the project**
  (`chakaso.training.corpus`), licence Apache-2.0, no external text, so no third-party terms and no
  public-benchmark contamination.
- Split: deterministic line-unit split (`chakaso.training.data`); units go wholly to one side, so no
  example is shared across the boundary. The vocabulary is fit on the **training** text only, so a
  held-out-only character maps to the reserved unknown id rather than widening the identifier space.
- Size: 344 training positions, 78 held-out positions (windowed), over a 28-symbol vocabulary.
- Contamination check: train and validation units are disjoint by construction (enforced and tested);
  the leak-free property is a test, not an assumption.
- **Limitation:** 78 held-out positions is a very small evaluation set. The numbers are real but the
  estimate is high-variance; this is a learning-loop demonstration, not a quality measurement.

## Model

A tiny causal character model (`chakaso.training.model.CharCausalMLP`): embedding lookup over a window
of 4 previous characters → concatenated window (32) → one `tanh` hidden layer (12) → softmax over the
28-symbol vocabulary. **984 trainable parameters**, initialized at random (uniform scaled by fan-in)
from `default_rng(seed)`. It is not a Transformer and makes no claim to model natural language.

## Environment

Observed on the run host (`platform`, `sys`, `numpy`), CPU-only:

- OS / architecture: Windows 11 (10.0.26200), AMD64
- Interpreter: CPython 3.13.5
- numpy: 2.4.6 (within the ADR-0030 `>=1.26,<2.5` bound)
- Accelerators: none used. The host has Intel Arc graphics and no CUDA device; the run is pure CPU.

These are facts about one host at one moment, not a hardware benchmark. Wall-clock and memory were not
measured to a stated precision and are therefore not reported — see Limitations.

## Reproducibility

Seed 7, the configuration above, and the code at the cited commit determine the result completely
(the training tests assert two runs are byte-identical). Reproduce with an environment that has the
`[train]` extra:

```
python -m pip install -e ".[dev,train]"
python -c "import json; from chakaso.training.experiment import measure; print(json.dumps(measure(), indent=2, sort_keys=True))"
```

The command prints the report below. Every value in "Results" is taken from that printout, not typed.

## Metrics

- **Held-out cross-entropy (nats):** mean `-ln P(target)` over the 78 held-out positions, computed by
  `chakaso.training.evaluation.evaluate`. Derived: perplexity `= exp(loss)`, bits/char `= loss / ln 2`.
- **Next-character accuracy:** fraction of positions whose argmax prediction equals the true next symbol.
- **Baselines:** `uniform_next_token_loss(V)` and `UnigramBaseline.fit(...).cross_entropy(heldout)`.
- The gradients that produce the model are validated against central finite differences for *every*
  scalar parameter (`tests/test_training_gradient.py`) before any number here is trusted.

## Results

Measured values from the run (rounded to 6 decimals in the report):

| Quantity | Value (nats unless noted) |
| --- | --- |
| Uniform baseline | 3.332205 |
| Unigram baseline | 3.053080 |
| Held-out loss, step 0 (random init) | 3.311822 |
| Held-out loss, minimum (step 50) | **1.664985** |
| Held-out loss, final (step 300) | 2.494577 |
| Final bits/char | 3.598913 |
| Final perplexity | 12.116603 |
| Final next-character accuracy | 0.602564 |
| Beats uniform | yes |
| Beats unigram | yes |

Full recorded curve (step: train / held-out):

```
  0: 3.336175 / 3.311822
 25: 1.363502 / 2.018858
 50: 0.648386 / 1.664985   <- held-out minimum
 75: 0.399168 / 1.757743
100: 0.317193 / 1.915931
125: 0.286629 / 2.084347
150: 0.273514 / 2.182068
175: 0.266301 / 2.257861
200: 0.261461 / 2.322418
225: 0.258272 / 2.377783
250: 0.256839 / 2.425481
275: 0.254429 / 2.461370
300: 0.253067 / 2.494577
```

The checkpoint at the final step has content fingerprint
`e62eebdcd0bba531ad00edaa3ae8e3cb525d44d9a5edb26e9b32242f2838ebe6`. Following the artifact policy, the
checkpoint bytes are **not** committed (the tiny model is fully reproducible from the seed and config);
the fingerprint and the reproduction command are the durable identity.

## Interpretation

Training from random initialization with a real optimizer lowers held-out next-character cross-entropy
**below both baselines** — from 3.31 nats at initialization to 1.66 at its best and 2.49 at the end, all
under the 3.05-nat unigram floor. This is the project's first measured evidence that its training
machinery actually learns structure (character-to-character dependencies), not just token frequency.

The curve is non-monotonic on held-out data: it reaches its minimum at step 50 and then *rises* while
training loss keeps falling. That is textbook overfitting of the 344-token training set by a 984-parameter
model, and it is reported as measured rather than smoothed away. The final checkpoint is past the held-out
optimum; a run with early stopping would score better on held-out loss, and that is a finding for the next
experiment, not a correction to this one.

## Limitations

- Single seed, single tiny corpus, 78 held-out positions — a high-variance estimate, not a general
  result about language modelling.
- Full-batch optimization and a fixed budget (no early stopping, no regularization) — the chosen
  300-step run overshoots its own held-out minimum.
- Held-out tokens include characters unseen in training, folded to the unknown id; the model's handling
  of them is scored, not assumed away.
- No wall-clock or peak-memory measurement was taken to a stated precision, so none is claimed here.
- This measures next-token prediction on a synthetic toy. It says nothing about grounding, citation,
  correction or answer quality — the project's actual research questions (H1–H7) remain untested.

## Conclusion

Moderate strength: Chakaso can now initialize, train, checkpoint, and **honestly measure** a character
language model, and that model beats trivial baselines on held-out data. The learning loop is real and
reproducible. What this does *not* establish is any language capability, any grounding ability, or that
the model generalizes beyond this toy corpus. K-001 is narrowed for the training loop — there is now a
measured, real model — but the broader gap (a real workload) stands.

## Follow-up

The next justified experiment, given the measured overfitting: vary the training budget and capacity
with early stopping on held-out loss (a controlled protocol, not a cherry-pick), and report held-out
loss against the same baselines across a few seeds so the result has an error bar rather than a point.
After that, the `LanguageModel` adapter over a trained checkpoint (P25) can be considered, because this
experiment produced a real artifact behind the boundary.
