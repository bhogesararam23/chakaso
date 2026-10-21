# ADR-0034: PyTorch is admitted as an optional `[nn]` extra for the model path

- Status: Accepted
- Date: 2026-10-21
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0002](ADR-0002-language-model-boundary.md),
  [ADR-0004](ADR-0004-apache-2.0-license.md), [ADR-0005](ADR-0005-src-layout.md),
  [ADR-0030](ADR-0030-minimal-optional-training-dependency.md),
  [ADR-0033](ADR-0033-model-centred-product-contract.md)
- Amends: [ADR-0030](ADR-0030-minimal-optional-training-dependency.md), whose decision stands for the
  character-level experiment it was written for; what changes is the option it rejected "for now".

## Context

[ADR-0030](ADR-0030-minimal-optional-training-dependency.md) admitted exactly one numerical dependency —
`numpy`, in an optional `[train]` extra — and rejected PyTorch **for now**, on three grounds: a wheel
"hundreds of megabytes" against a machine P22 observed at ~1.4 GB free RAM; autograd as a black box
against a project standard that wants gradients *verified*; and a dependency weight "that a
few-thousand-parameter char model does not earn". It then wrote down five conditions under which the
decision should be reopened, including *"a concrete evaluation workload exists that a larger model would
improve"* and *"the manual-gradient approach becomes demonstrably a correctness risk"*.

[ADR-0033](ADR-0033-model-centred-product-contract.md) made the model the product. The architecture that
follows from the roadmap's M1/M2 — a subword tokenizer and a decoder-only Transformer with causal
attention ([`../training.md`](../training.md)) — is not expressible in the existing hand-derived layer:
`CharCausalMLP` has no attention and no positional mechanism, and its loop is full-batch.

[`../research/next-model-experiment.md`](../research/next-model-experiment.md) re-measured the
environment and the candidates rather than re-arguing them. Its measurements are the evidence for this
record; the numbers quoted below come from that note and are reproducible from its commands.

## Problem

Decide whether the model path may import an automatic-differentiation framework, and if so: under which
extra, with which isolation boundary, with which verification discipline in place of hand-derived
gradients, with which checkpoint-security rules, and with what happens to the dependency-free base
install and the three-version offline CI matrix.

## Evidence (measured 2026-10-21, this machine)

| Fact | Value |
| --- | --- |
| `torch` CPU wheel, `cp311-win_amd64` (PyPI, `torch` 2.14.1) | 124.1 MB |
| `torch` wheel, `cp311-manylinux_2_28_x86_64` (PyPI) | 554.6 MB — the Linux default wheel is not the CPU build, so CPU installs must use the CPU index |
| `torch` `Requires-Python` | `>=3.10`; `cp311`, `cp312`, `cp313` wheels all present, so the declared floor is honoured |
| Cold `import torch` on CPython 3.13.5 | 2.96 s (numpy: 0.25 s) |
| `torch` already installed on the developer interpreter | 2.13.0+cpu, 14 intra-op threads, `cuda.is_available()` → `False` |
| Transformer step, batch 32 × ctx 128, V 1024 / d 128 / 3 layers, 774,656 params, CPU | 0.444 s per step (9,230 tokens/s); 40 epochs over ~250 k tokens ≈ 18 min |
| Same-shape float32 matmul, NumPy vs torch | 0.159 s vs 0.137 s — **1.16×**, i.e. raw array speed is *not* an argument here |
| Existing full-batch loop at vocab 200 | 0.230 s/step at 20 k examples, 1.868 s at 100 k, and a `(batch × vocab)` probability array: 1.04 GB float64 at 650 k positions |
| Finite-difference verification cost | measured 984 scalars in ~1.13 s; extrapolated with the measured forward cost, all 774,656 scalars of the candidate model ≈ 45 h, a 200-scalar sample ≈ 42 s |
| Free RAM at inspection | 3.70 GB of 16.57 GB |
| Peak resident set of a training process | **not measured** — the Windows process-memory APIs and `tasklist` PID filtering returned nothing from this session's shell, so no RSS figure is claimed |

## Decision

**Admit `torch` as a second, separately named optional extra — `[nn]` — for the model path only.**

1. `pyproject.toml` gains:

   ```toml
   nn = [
       "torch>=2.6,<3",
   ]
   ```

   `dependencies` stays empty; `[dev]` stays as it is; `[train]` keeps *exactly* `numpy>=1.26,<2.5` and
   is unchanged, which the existing hygiene test enforces. The floor `>=2.6` is chosen because
   `torch.load` defaults to `weights_only=True` from 2.6, and the checkpoint rule below depends on that
   default. `<3` bounds the major because the serialization and `TransformerEncoderLayer` argument
   surfaces have changed incompatibly before.

2. **Isolation boundary.** `torch` may be imported only by modules that define or train a Transformer, or
   adapt a Transformer checkpoint: `chakaso.training.transformer*`, `chakaso.training.nn_training*`,
   `chakaso.training.nn_checkpoint*`, `chakaso.training.subword_experiment*` and
   `chakaso.models.nntransformer`. Nothing else in the package may import it. Enforced by test, not by
   review, in the same way the `[train]` numpy isolation is.

3. **The base install and the core CI matrix are unchanged.** `import chakaso`, `import chakaso.models`
   and `import chakaso.training` must stay free of `torch` (and of `numpy` where they already are); the
   new adapters are imported lazily by the registry, so reading the registry stays dependency-free.
   `torch`-requiring tests are `pytest.importorskip("torch")`-gated, exactly like the numpy-gated training
   tests, so the dependency-free matrix skips them and stays offline, CPU-only, and download-free.

4. **Verification discipline replaces hand-derived gradients; it does not disappear.** Where the char
   model verified *every* scalar by finite differences, the Transformer path is verified by:
   hand-computed losses on tiny controlled inputs; explicit causal-mask tests that a future token cannot
   affect an earlier prediction; output-shape and parameter-count assertions; gradient finiteness and
   parameter-movement assertions after an optimizer step; seeded-initialization reproducibility; and a
   **sampled** finite-difference cross-check against autograd, whose sample size is declared in the test
   rather than implied. This is a weaker guarantee than full scalar coverage and is recorded as such.

5. **Checkpoint security is a hard rule.** A Transformer artifact is written as tensors plus a JSON
   metadata block with content checksums, and read with `weights_only=True`. A call that would unpickle
   arbitrary objects is not used anywhere in the package. Load refuses an unknown schema version, a
   checksum mismatch, a vocabulary or architecture disagreement with the stored configuration, and a path
   that is not a file — loudly, as a typed error, never as a silent fallback to another model.

6. **Reproducibility claims are scoped honestly.** A run is reproducible on this project's stated
   interpreter and CPU build from its recorded seed, configuration, corpus digest and tokenizer
   fingerprint. Bit-identical replay across `torch` versions or across GPU hardware is **not** claimed,
   because it was not measured.

7. **CPU installation is documented, not assumed.** The Linux CPU build requires the CPU wheel index
   (`--index-url https://download.pytorch.org/whl/cpu`); a plain `pip install ".[nn]"` on Linux pulls the
   CUDA-bearing wheel. The docs state this so an extra does not silently become a multi-gigabyte install.

## Reasoning

- **ADR-0030's own revisit conditions are met, on measurements rather than appetite.** Condition 1: a
  concrete workload exists — a real 724 KB corpus with a held-out split and a bits-per-character metric —
  that the existing 984-parameter, no-attention, full-batch model demonstrably cannot address: its own loop
  costs 1.87 s/step at a 100 k batch and materializes a 1.04 GB probability array at corpus scale, so
  mini-batching is required before scaling is even discussable. Condition 2: full-scalar finite-difference
  verification of a hand-derived Transformer backward pass extrapolates to ~45 hours at the measured
  forward cost, which is the point where "validated by finite differences in reasonable time" fails.
- **The wheel-and-RAM objection that carried most of ADR-0030's rejection is now measured, and is weaker
  than it looked.** The Windows CPU wheel is 124.1 MB, `torch` is already present on the development
  interpreter, and the machine shows 3.70 GB free. That is not a licence to grow dependencies: the Linux
  wheel is 554.6 MB and imports in ~3 s, which is why rule 7 and the isolation rule exist.
- **The honest speed case is not about speed.** The measured same-dtype matmul comparison is 1.16×, which
  does not justify a framework. What justifies it is that the alternative is writing and maintaining a
  private autodiff engine for attention, layer norm, GELU and cross-entropy — a component this project
  would then be responsible for getting right, with only sampled verification available.
- **A separate extra rather than widening `[train]`.** The hygiene test that pins `[train]` to exactly
  numpy is a real guarantee: the dependency-free parts of the training layer (corpus, tokenizer, split,
  config, baselines) run in the core matrix. Folding torch into `[train]` would make every numpy-gated
  test torch-gated too and enlarge the untested surface. Two named extras keep the tiers honest.

## Trade-offs

- A second optional dependency class becomes untested in remote CI. K-009 already records that the
  `[train]` job cannot be pushed because this session's credential has no `workflow` scope; the `[nn]`
  tests inherit that gap, and it is widened. The mitigation is the same as K-009's: local execution with
  the extra installed, and the gap recorded rather than claimed away.
- `torch` is a large, fast-moving dependency. Pinning `<3` defers breakage rather than preventing it, and
  checkpoint format changes across torch versions are a live risk (rule 6 scopes the claim).
- Sampling the finite-difference check gives up a property the project currently has in full. This is the
  cost the record exists to make visible: from here on, gradient correctness rests partly on a framework
  plus a sampled cross-check, not entirely on this repository's own arithmetic.
- Import cost (~3 s cold) makes `torch` unacceptable on any default path; rule 2 keeps it out of them.

## Rejected alternatives

- **Stay NumPy-only and hand-derive the Transformer backward pass.** Rejected on the measured verification
  cost and the maintenance burden. Its real advantage — no new dependency — is preserved in part by
  keeping the char path and `[train]` exactly as they are.
- **Put `torch` in `[train]`.** Rejected: it would silently redefine an extra a hygiene test exists to
  keep minimal, and it would gate the dependency-free-tier tests behind a 124 MB+ wheel.
- **Make `torch` a runtime dependency of the model adapter.** Rejected by
  [ADR-0001](ADR-0001-local-first-runtime.md) and by the base-install rule; the adapter is lazy and its
  absence is a named error, mirroring how the `trained_char` adapter already behaves.
- **Use a pretrained model as the next experiment.** Rejected by
  [ADR-0033](ADR-0033-model-centred-product-contract.md) rule 7 and by the founding goal: the project's
  claim is its own weights from random initialization, and a borrowed model's number measures someone
  else's training run.
- **JAX / TensorFlow.** Rejected for the same reasons ADR-0030 rejected them, with less familiarity and no
  CPU-wheel advantage here.

## Consequences

- `pyproject.toml` gains `[nn]`; `docs/getting-started.md`, `README.md` and
  [`../training.md`](../training.md) document the extra and the CPU wheel index.
- New modules that import torch live in `chakaso.training` (the module
  [ADR-0005](ADR-0005-src-layout.md) reserved for training code) plus one adapter under `chakaso.models`,
  added to the hygiene test's adapter allowlists in the same commit as the adapter.
- **Amended 2026-10-22, under the rule above:** `chakaso.training.decoder` (the release-path decoder: rotary
  positions, an explicit attention block with a key/value cache, and masked targets) is admitted as a
  torch-importing module. It is a second architecture beside `transformer`, not a replacement of it, so the
  measured E2605/E2606/E2607 checkpoints keep their own schema and their own meaning.
- `mypy src` in the dependency-free matrix must keep excluding the torch-importing modules, for the same
  documented reason `chakaso.training` is already excluded: their type aliases resolve through an optional
  dependency the matrix does not install. They are fully type-checked where the extra is present.
- The `trained_char` adapter and the numpy char model remain valid: E2601/E2602's results are not
  restated, and the new path is an additional artifact kind with its own checkpoint schema version.
- Creates one follow-up decision it does not make: whether a second checkpoint format should also carry
  the tokenizer's merge table inline, or reference it by fingerprint. That is a checkpoint-schema decision
  for ADR-0035, recorded with the format.

## Conditions to revisit this decision

1. If a `torch`-free route becomes necessary for a target install (a platform with no CPU wheel, or a
   distribution that cannot carry it), revisit C3 of the research note with the verification cost restated.
2. If the project ever requires bit-identical replay across machines or versions, this decision is wrong
   for that requirement and must be re-made — reproducibility here is scoped to interpreter, CPU build and
   seed.
3. If a `torch`-based adapter (Option A / C4) is taken up, it needs its own record: licence, redistribution
   and the honesty boundary between borrowed and own weights.
4. If the `[nn]` CI job becomes pushable (a credential with the `workflow` scope), the gap K-009 records is
   closed and this record's testing strategy should be updated to point at the job rather than at local
   runs.
