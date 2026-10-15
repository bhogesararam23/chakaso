# Model feasibility: a runnable local model, or a from-scratch training experiment?

**Type:** Research — design and environmental analysis. **No experiment was run and nothing here is a
measured model result.** It follows the entry discipline in [`README.md`](README.md): verified
observations are separated from assumptions and from estimates, and no number that was not measured is
presented as one.

**Date:** 2026-10-15. **Milestone:** P22 (assumes the P21 runtime composition).

## Question

Chakaso's strategic frontier is crossing from deterministic research infrastructure into a project that
can experiment with a *real* language behaviour. The development double is not a language model, and the
fixture embeddings are not semantic (K-001). Before any weight is downloaded or any dependency is added,
which of two directions is actually feasible *here*, and what is the smallest honest next step?

- **Option A** — integrate a genuinely runnable, small *pretrained* local model behind the existing
  `LanguageModel` boundary ([ADR-0002](../decisions/ADR-0002-language-model-boundary.md)).
- **Option B** — build a minimal *from-scratch* training pipeline that demonstrates the learning loop:
  data preparation, a tokenization strategy, a model definition, training, checkpointing, held-out
  evaluation.

## Method — what was and was not done

**Done.** The available machine was inspected directly (Python, CPU, memory, display adapters, the
package environment, free disk). The existing boundaries and the local-first / dependency rules were read.

**Deliberately not done.** No model weights were downloaded, no numerical or deep-learning dependency was
installed, and no model was trained or run. The project's rules forbid inventing measurements
(`AGENTS.md`: "if it was not measured, it is not written down"), so every figure below that is *not* from
the inspection is explicitly labelled an assumption or an estimate, and the recommendation stops short of
executing an experiment it cannot yet do honestly.

## Verified environment (observed on this machine)

| Fact | Value | How verified |
| --- | --- | --- |
| OS / architecture | Windows 11, AMD64 | `platform` |
| Interpreter | CPython 3.13.5 | `sys.version` |
| Logical CPUs | 18 | `os.cpu_count()` |
| Total physical RAM | 15.4 GB | `Win32_ComputerSystem.TotalPhysicalMemory` |
| Free physical RAM at inspection | ~1.4 GB | `Win32_OperatingSystem.FreePhysicalMemory` |
| Display adapter | Intel(R) Arc(TM) Graphics | `Win32_VideoController` |
| CUDA-capable NVIDIA GPU | none observed | no NVIDIA adapter present |
| ML/runtime libraries in the project venv | none (`torch`, `tensorflow`, `transformers`, `numpy`, `jax`, `onnx`, `faiss`, `sentence-transformers`, `huggingface_hub`, `accelerate` all absent) | `pip list` |
| Free disk on the project volume | ~160 GiB | `Get-PSDrive` |
| CI hardware | no GPU, offline, three Python versions (3.11/3.12/13) | `.github/workflows/ci.yml` |

These are **facts about one developer machine at one moment**, not about CI and not a benchmark. The
`1.4 GB` free-RAM reading is transient — it reflects whatever else was running and is the single most
binding practical constraint on the *near-term, local* feasibility of both options.

## Constraints the decision must respect

- **Local-first** ([ADR-0001](../decisions/ADR-0001-local-first-runtime.md)): the runtime must not *require*
  a commercial inference or search API. An optional local dependency is permitted, but it becomes a
  consequential, documented decision (the repository has no runtime dependencies at all today;
  `pyproject.toml` keeps the base install dependency-free and heavy components optional).
- **Model boundary** ([ADR-0002](../decisions/ADR-0002-language-model-boundary.md)): application code depends
  on `LanguageModel`; configuration selects exactly one concrete adapter. Any real model is an *adapter*,
  never an import into application code.
- **Identifier ownership** ([ADR-0003](../decisions/ADR-0003-evidence-identifier-ownership.md)): a real model
  may generate text and reference identifiers; it may not produce source identity. This does not change
  with either option.
- **No fabricated capability**: a toy model is a learning experiment, not competitive general AI, and must
  never be described as understanding anything.

## Option A — a small pretrained local model

**What it is.** Run an existing small causal model (roughly 0.5–3B parameters; *estimate*, not measured
here) through a local inference stack behind a new adapter.

**Fit.** Clean: a `LanguageModel` implementation that loads weights, tokenizes, generates, and returns a
`GenerationResult` declaring its own `model_id` (and `development_double=False`). No application change
beyond configuration and the registry.

**Feasibility on this machine (assumptions + estimates, not measured).**
- CPU inference of a ~1–3B model in a quantized format is *possible* in principle on 18 threads, but
  tokens/second are low without a GPU and memory-bound. *Estimate:* a ~1B int4 model occupies roughly
  ~0.5–1 GB plus runtime overhead; the framework import alone is typically ~1–2 GB resident (*estimate*).
  With only ~1.4 GB free RAM observed, a real run here would likely fail until other memory is freed — a
  practical blocker, not a product limitation.
- The GPU present is Intel Arc, not CUDA. A CUDA-first stack gives no acceleration here; Intel paths
  (OpenVINO / IPEX / oneDNN) exist but add platform-specific complexity (*assumption: would work, untested*).

**Costs / risks.**
- A heavy third-party dependency (a torch-family runtime) against a currently dependency-free base; it
  must stay optional and needs its own ADR before it enters `pyproject.toml`.
- **Licence and redistribution**: weights carry their own terms ([ADR-0004](../decisions/ADR-0004-apache-2.0-license.md)
  reserves weights/datasets for separate policy); many "small open" models are non-commercial or
  usage-restricted, complicating reproduction and redistribution.
- **Reproducibility**: a downloaded external artifact is a moving target (version drift, cache state), and
  the base install / CI cannot require it.
- A genuinely *useful* answer still needs a retrieval workload and a real evaluation to measure against
  (K-001), which do not exist — so an A adapter alone buys a capability that cannot yet be honestly scored.

## Option B — a from-scratch training experiment

**What it is.** Train a deliberately tiny model (character- or small-BPE-level, a few million parameters,
a few hundred KB–MB of weights; *estimate*) end to end, producing the project's *own* weights and a real
held-out number (e.g. validation loss / next-token accuracy).

**Fit.** Also clean, and more faithful to the founding goal ("train its own model from random
initialization"). It yields a `LanguageModel` adapter whose weights this project owns.

**Feasibility on this machine (assumptions + estimates).**
- A toy model trains on CPU in a laptop-sized budget: tiny vocab, tiny context, minutes-to-hours for a
  meaningful curve (*estimate, unmeasured*). Memory pressure is far below Option A because the model is
  small and the corpus is self-contained — the observed ~1.4 GB free RAM is workable for a char-level model
  and is *not* workable for a 1–3B pretrained run.
- It needs a numerical training stack (at minimum array math; in practice an autograd framework). That is
  the same dependency decision as Option A, but it is a *build-time* dependency of an experiment, not a
  *run-time* dependency of the product — it never has to enter the base install or CI matrix.

**Costs / risks.**
- A toy model answers nothing useful; its whole value is a demonstrated, *measured* learning loop and a
  self-owned artifact. It must be labelled exactly that.
- Discipline required: held-out evaluation and reproducibility (seed, config, tokenizer version) must be
  recorded per the artifact policy, or it becomes an uninterpretable toy.

## Side-by-side

| Dimension | Option A (pretrained local) | Option B (from-scratch toy) |
| --- | --- | --- |
| RAM to run *here* (15.4 GB total, ~1.4 GB free) | Tight/blocked for a 1–3B model (estimate) | Comfortable for a few-M-param model (estimate) |
| Download / storage | Large external weights, MB–GB | None external; own weights, KB–MB (estimate) |
| CPU feasibility | Slow but runnable on 18 threads | Runnable, small budgets |
| GPU help here | None (no CUDA; Intel Arc path untested) | None needed |
| Runtime / dependency complexity | High (inference stack, weight loading) | Moderate (autograd + tokenizer), build-time only |
| Licence / redistribution | External, often restricted (ADR-0004) | Self-owned, trivial |
| Offline reproducibility | Weaker (external artifact drift) | Strong (seed + small data + own weights) |
| Scientific usefulness *now* | A useful model, but unmeasurable without a real eval (K-001) | The first *real measured number* the project lacks |
| Fit with `LanguageModel` boundary | Direct adapter | Direct adapter |
| Maintenance burden | Ongoing external-model coupling | Contained, self-describing |
| Likely next milestone it unlocks | Grounded-answer benchmark on a real workload | A measured training curve; the tokenizer/pipeline the roadmap wants |

## Conclusion (weak, pending a dependency decision)

Both options are *architecturally* clean behind `LanguageModel`; neither forces a boundary change, and both
require the same prerequisite: a documented decision to admit a numerical/deep-learning dependency as an
**optional, non-default** extra (its own ADR, consistent with ADR-0001/ADR-0005). No model, and no such
dependency, should enter the base install or the CI matrix.

Given the evidence on this machine — **no CUDA, ~1.4 GB free RAM observed, and no real evaluation workload
yet (K-001)** — the recommendation is:

1. **Do not pursue Option A yet.** A pretrained adapter's value is gated on a grounded-answer benchmark and
   a real corpus that do not exist, and its external-weights dependency conflicts most with the project's
   reproducibility and licence constraints. Deferring it is honest, not timid.
2. **The smallest justified follow-up is a scoped Option-B experiment, run outside the package and CI** as
   an `experiments/<id>/` artifact (`docs/internal/experiments/README.md` format): a char-level model, a
   fixed seed and config, a held-out set, reporting the first genuinely *measured* result the project will
   have — validation loss over training steps — with the explicit limitation that it is a learning-loop
   demonstration and not a claim of language capability.
3. Before any code, **write the dependency-feasibility ADR** that names the minimal offline numerical stack
   and keeps it an optional extra. That ADR is the gate, exactly as the roadmap requires for model choices.

This advances the open question [Which local model should the shell use?](open-questions.md#which-local-model-should-the-shell-use)
from unexamined to *evidence-bounded*: the constraint that binds near-term work is not architecture but
an admitted numerical dependency plus a memory budget, and it is now recorded rather than assumed.

## Limitations of this assessment

- It inspects **one** machine at **one** moment; the free-RAM figure is transient and CI hardware differs.
- No model was installed, downloaded, trained, or run. Every non-inspected figure is an **estimate** or an
  **assumption**, labelled as such; there is no measured number here, and none should be quoted from this
  note as if there were.
- It does not choose a specific model or framework — that is the deferred dependency ADR's job.
