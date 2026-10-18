# ADR-0030: A minimal, optional numerical dependency for a from-scratch training experiment

- Status: Accepted
- Date: 2026-10-19
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0002](ADR-0002-language-model-boundary.md), [ADR-0004](ADR-0004-apache-2.0-license.md), [ADR-0005](ADR-0005-src-layout.md), [ADR-0020](ADR-0020-experiment-artifacts.md)

## Context

The project has spent its life so far building deterministic research infrastructure: retrieval,
evidence identity, the model boundary, correction, evaluation and reproducible experiment records.
There is still **no language model** — the only implementation behind `LanguageModel` is the
development double ([ADR-0002](ADR-0002-language-model-boundary.md)), and there is no tokenizer, no
dataset pipeline, no model code and no training loop. K-001 records the honest consequence: nothing
here has been measured against a real workload or a real model.

The founding goal ([`../../README.md`](../../README.md)) is to *train Chakaso's own model from random
initialization*. The P22 feasibility assessment
([`../research/model-feasibility.md`](../research/model-feasibility.md)) concluded — from a direct
inspection of the developer machine, not from a run — that both a small pretrained adapter (Option A)
and a tiny from-scratch training experiment (Option B) fit the boundary cleanly, and that the gate on
either is not architecture but **the willingness to admit a numerical runtime as an optional,
non-default dependency**, plus a memory budget. It recommended deferring Option A until a real
grounded-answer evaluation exists, and pointing at a scoped, out-of-CI Option-B experiment as the
smallest honest next step — explicitly gated behind *this* decision.

This is P23: the dependency decision that must be written down **before** a single line of model,
tokenizer or training code exists, exactly as ADR-0001 and the roadmap require for consequential
dependencies.

## Problem

Name the minimal offline numerical stack the project is willing to admit as an optional dependency,
and pin how it is packaged, installed, tested and reproduced — such that:

- the base install stays dependency-free,
- the core CI matrix (Python 3.11–3.13, offline, CPU-only, no model downloads) is unchanged and does
  not install it,
- a tiny from-scratch language-model experiment can be built and *numerically verified* on this
  machine and in CI on a separate, clearly-optional path,
- the choice is honest about what it buys and what it costs, and states when it should be revisited.

## Requirements

- **Offline after install.** No runtime network access; no weight downloads. Reproducibility is a
  property of the code plus a seed, not of a remote artifact
  ([ADR-0001](ADR-0001-local-first-runtime.md)).
- **CPU-only.** No GPU is present (Intel Arc, no CUDA) and none is required
  ([`../research/model-feasibility.md`](../research/model-feasibility.md)).
- **Optional and isolated.** A clearly named extra that never enters `dependencies` and never enters
  the base editable install used by the main CI jobs.
- **Compatible with the declared floor.** The package targets Python 3.11+ (CI enforces 3.11, 3.12,
  3.13), so the dependency must install on all three, or the extra must declare a bound that keeps the
  floor working.
- **Deterministic and seedable.** Zero-temperature reproducibility is part of the model contract; the
  numerical stack must give a stable, explicitly-seeded PRNG.
- **Licence-compatible.** Apache-2.0 code ([ADR-0004](ADR-0004-apache-2.0-license.md)); a bundled
  dependency's licence must be compatible for redistribution through an extra.
- **Verifiable.** The project's engineering standard ("make failure modes explicit and testable",
  "verify numerical correctness") means gradients and losses must be checkable against an independent
  computation, not merely trusted.
- **Bounded memory.** The observed machine had ~1.4 GB free RAM at P22 inspection (a transient,
  historical observation, not a guarantee); the experiment must be trivially small.

## Options considered

- **Option 1 — Pure-Python, no numerical dependency.** Implement array math, a small autodiff or
  manual backprop, and a tiny model entirely in Python.
- **Option 2 — NumPy as a sole, optional numerical dependency; hand-written backprop.** A dense
  ndarray library (`numpy`) supplies array math and a seeded PRNG; the gradient of the tiny model is
  derived by hand and verified with finite-difference tests.
- **Option 3 — An automatic-differentiation framework (PyTorch) in an optional `[train]` extra.**
  `torch` (CPU wheel) supplies arrays, autograd, optimizers and checkpointing; the model is defined in
  its API.
- **Option 4 — A source-compiled or heavier alternative** (e.g. JAX, TensorFlow, a C++/Rust extension).

## Evidence gathered

Package metadata was read from PyPI's authoritative JSON API on 2026-10-19 (cited below); **no package
was installed or run as part of this decision** (that begins only under the roadmap's training sprint):

- **NumPy** — latest release `2.5.4` declares `Requires-Python >=3.12`; the newest release that still
  ships a `cp311` wheel (and so honours the project's floor) is `2.4.6`. SPDX licence expression:
  `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`. CPU-only binary wheels exist for `win_amd64`,
  `manylinux` and macOS for `cp311`–`cp313`; the wheel is a few megabytes of C extension, no transitive
  Python dependencies. Its random number generation is explicitly seedable
  (`numpy.random.default_rng(seed)`).
  Source: <https://pypi.org/pypi/numpy/json> (release files and `requires_python`, read 2026-10-19).
- **PyTorch** — latest release `2.14.1`, `Requires-Python >=3.10`. SPDX licence expression is a large
  mixed set (`Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND
  BSL-1.0 AND MIT`). The CPU `win_amd64` wheel is hundreds of megabytes with further native
  dependencies. Autograd is a black box relative to this project: correct only as far as the framework
  is trusted.
  Source: <https://pypi.org/pypi/torch/json> (release files, `requires_python`, license expression, read
  2026-10-19).

No claim here is a measured training result. These are packaging facts used to make a dependency
choice, and the memory/footprint statements are metadata-and-size observations, not this project's own
benchmark; the free-RAM figure is P22's historical, labelled observation of one machine at one moment.

## Decision

**Adopt Option 2: `numpy` as the single numerical dependency, isolated in a new optional `[train]`
extra, with the model's gradients derived by hand and verified by finite-difference tests.**

- Add to `pyproject.toml`, under `[project.optional-dependencies]`:

  ```toml
  train = [
      "numpy>=1.26,<2.5",
  ]
  ```

  The `<2.5` upper bound keeps `pip install -e ".[train]"` working on the Python 3.11 floor (NumPy
  2.5 dropped 3.11); the `>=1.26` floor predates 3.11 support being universal but is the point from
  which the modern seeded `Generator` API and the wheels above are all available. `dependencies`
  stays empty.
- Model-training code lives in **`chakaso.training`** — the module ADR-0005 already reserved for it —
  inside the one distribution, importable and testable like everything else. `chakaso.training` imports
  `numpy` only where a trainable component genuinely needs it.
- The base package keeps **zero runtime dependencies**; the whole existing application, the double, and
  every current test are unaffected. `import chakaso` must still not require `numpy`.
- The first experiment is a deliberately tiny **character-level causal model** run as an
  `experiments/<id>/` artifact ([`../internal/experiments/README.md`](../internal/experiments/README.md)),
  with its reusable mechanics (tokenizer, split, model, training step, checkpoint, evaluation) in
  `chakaso.training`.

## Reasoning

- **Verifiability beats convenience for a research first step.** The single most important thing the
  first experiment must establish is that the project can tell whether a learning loop *actually
  learns*. Hand-derived gradients checked against finite differences make the update rule a tested,
  inspectable property of *this* code; an autograd framework would make "the loss connects to the
  trainable parameters" something inherited on trust from a third party. For a few-thousand-parameter
  char model the manual derivation is small and worth the independence.
- **A minimal footprint the reproducibility story can carry.** One wheel, no transitive Python
  dependencies, seeded PRNG, CPU-only, offline after install — a another developer reproduces the
  experiment from `pip install -e ".[train]"` plus a seed and the committed config, with no external
  weights and no licence entanglement beyond permissive components.
- **It honours the floor deliberately.** The `numpy` 2.5 → 3.11 drop is recorded as the reason for the
  `<2.5` cap, so the bound is an explained decision rather than a pinned-and-forgotten number.
- **The boundary is untouched.** This adds a *build-time* dependency of an experiment, not a *run-time*
  dependency of the product. Nothing in `chakaso` above the model boundary learns that `numpy` exists.

## Trade-offs

- Manual backprop is more code to write and test than a framework call would be, and it caps how large
  a model is reasonable to train this way. That cap is acceptable — the first experiment is meant to be
  tiny, and scaling is explicitly deferred until reproducibility exists
  ([`../../docs/roadmap.md`](../../roadmap.md)).
- `numpy` is a compiled dependency: an exotic platform without a wheel would have to build from source.
  Windows AMD64, manylinux and macOS all ship wheels, so this is a documented edge, not a live one.
- Choosing hand-written gradients now means a future real model (or Option A) may prefer a framework.
  That is a *later* decision with its own ADR, not a reason to carry a heavyweight dependency today.

## Testing strategy

- **`chakaso.training` unit tests are `numpy`-gated.** They `pytest.importorskip("numpy")` so that the
  **core CI matrix (which installs only `.[dev]`) neither downloads nor requires `numpy`** and simply
  skips them — the 3.11/3.12/3.13 jobs stay exactly as they are.
- A **separate, clearly-named optional CI job** installs `.[dev,train]` and runs the gated training
  tests, so the numerical path *is* exercised remotely without coupling it to the core matrix. The
  tests verify: tensor shapes, a loss value against a hand-computed example, **finite-difference
  gradient checks**, that an optimizer step actually changes the intended parameters, seeded
  reproducibility, checkpoint round-trips, and train/validation split separation.
- The **full experiment** (a real training curve) runs outside CI on a machine that installs the extra;
  CI carries only a deterministic, tiny mechanics test, which is explicitly *not* presented as
  reproducing the experiment.

## Memory and performance risks

- The char-level model, its vocab, context and corpus are sized to fit comfortably within the ~1.4 GB
  free RAM P22 observed; the model is a few thousand parameters, so array math is trivially in-memory.
  These are the *same transient observations* as P22 — treated as a budget to respect, not a guarantee.
- A larger corpus or context could grow memory without bound; the experiment pins sizes in its config
  and the split is taken before any vocabulary is fit, so the memory footprint is a documented input,
  not an emergent surprise.
- Nothing here claims a measured throughput or memory figure; any such number belongs in the
  experiment record only if actually measured.

## Licensing

- `numpy` resolves to `BSD-3-Clause` plus bundled `0BSD`/`MIT`/`Zlib`/`CC0-1.0` components — all
  permissive and compatible with the project's Apache-2.0
  ([ADR-0004](ADR-0004-apache-2.0-license.md)), including redistribution through an extra.
- No model weights, datasets or generated artifacts are added by this decision, so ADR-0004's reserved
  policy for weights/datasets is untouched.

## Rejected alternatives

- **Option 1 (pure Python, no dependency).** Rejected for the *first experiment* because the whole
  point of the first measured result is a learning loop; hand-rolling array math in Python is slower
  and its error-prone, and `numpy` is permissively licensed, tiny, offline and already the standard
  substrate — so avoiding it buys reproducibility the project already has with it. (A pure-Python
  fallback is retained as a *revisit* option, not dismissed as silly; see below.)
- **Option 3 (PyTorch in `[train]`).** Rejected for now on evidence and fit: a several-hundred-MB wheel
  with further native dependencies against an environment P22 observed at ~1.4 GB free RAM; autograd as
  a black box against a project standard that wants gradients *verified*; and a dependency weight that
  a few-thousand-parameter char model does not earn. It is the most likely choice for a *later, larger*
  model — which needs its own ADR, a concrete evaluation workload, and a reason a tiny numpy model is
  insufficient.
- **Option 4 (JAX / TensorFlow / a compiled extension).** Rejected: heavier, less essential, and no
  advantage for the first learning loop; they inherit the same "carry a big runtime" objection as PyTorch
  with less familiarity.

## Consequences

- `pyproject.toml` gains a `train` extra with the bounded `numpy` pin; the base `dependencies` list and
  the `dev` extra are unchanged, so a plain install and the existing three-version CI matrix behave
  exactly as before.
- `chakaso.training` becomes the importable, tested home of the tokenizer, split, model, training step,
  checkpoint and evaluation the first experiment needs, and (later) the source of a trained artifact for
  a `LanguageModel` adapter (Phase D).
- The next unit — the smallest honest from-scratch experiment — is now unblocked; it may not begin
  before this ADR is committed, and this ADR is committed before any training code
  ([`ACTIVE_TASK.md`](../agent/ACTIVE_TASK.md)).

## Conditions to revisit this decision

Reopen ADR-0030 and write a successor when any of the following is true:

1. A **concrete evaluation workload** exists that a larger model would improve, and a tiny hand-derived
   model demonstrably cannot address it — the trigger to consider a framework (Option 3) for scaling.
2. The manual-gradient approach becomes **demonstrably a correctness risk** — e.g. the model grows past
   the point where hand-derived gradients can be validated by finite differences in reasonable time.
3. A **pretrained adapter (Option A)** is taken up, which carries its own dependency, licence and
   redistribution decision and is explicitly deferred by P22 until a real grounded-answer evaluation
   exists.
4. `numpy` loses compatibility with the Python floor or platform set the project commits to, or its
   licence/redistribution posture changes.
5. A pure-Python route (Option 1) is ever justified — only if even the `numpy` extra proves unacceptable
   for some target install, which no current requirement indicates.
