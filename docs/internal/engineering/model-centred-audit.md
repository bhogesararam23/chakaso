# Model-centred audit

Internal engineering audit of what this repository actually is, against what it says it is for.
Written 2026-10-21 from direct inspection of the working tree at `eaee98d` and of the executed gates on
that commit. Nothing here is a proposal document: every row is either a checked fact, a measured number,
or is labelled as an assumption or an open question.

> **This is a snapshot, and it has aged.** It describes the tree at `eaee98d` on 2026-10-21, and it is kept
> unchanged because its value is the reasoning that moved the project, not its inventory. Read it against
> [`../../agent/CURRENT_STATE.md`](../../agent/CURRENT_STATE.md), which is the authority on what exists now.
> What has happened since, in the order a reader of this file would notice it: the model path went from one
> measured toy to eight experiment records — E2605's Transformer, E2606's five-seed spread, E2607 on licensed
> external prose, and E2608's two-architecture comparison whose decoder arm is released as
> `chakaso-base@0.1.0` at Level 2 of [ADR-0039](../../decisions/ADR-0039-three-level-model-release-gate.md);
> there are now three pinned corpora — the repository's own prose, three Project Gutenberg texts, and a
> 24-text corpus with a machine-readable catalogue and a measured data card each — plus a corpus-recovery path
> verified from git
> history, a checkpoint format per architecture family, a release contract, a streaming and cancellation
> runtime, a standard-library HTTP adapter, a generation-degeneration suite, and a three-tier CI matrix read
> green off the runners. Several rows below are therefore marked *verified* about things that have since been
> built — a missing component in this file is evidence about 2026-10-21, not about today. The
> 2026-10-24 documentation pass over every document is recorded in
> [`../../research/research-log.md`](../../research/research-log.md) and in the corrections it produced; this
> file was deliberately left as written.

Evidence labels used throughout:

| Label | Means |
| --- | --- |
| **verified** | Read or executed in this checkout; the cited file or command is the evidence. |
| **measured** | A number produced by a run recorded in `experiments/` or by a command quoted here. |
| **assumption** | Reasoned but not checked. Do not build on it without checking. |
| **open** | A question with no committed answer. |

## 1. Observed product identity

**verified.** The repository is, today, a retrieval-grounded conversational *system* with a language-model
interface and a model *experiment* inside it — not a language-model project with a system around it. The
evidence is structural, not tonal:

| Measurement | Value | How obtained |
| --- | --- | --- |
| Python modules in `src/chakaso/` | 18 subsystems, 131 modules, 20,697 lines | `find src -name '*.py' \| xargs wc -l` |
| Largest subsystems | `benchmark/` 19 modules, `retrieval/` 19 modules, `training/` 14, `correction/` 9, `answers/` 8 | module counts per directory |
| Model implementations behind `LanguageModel` | 2: a development double and an adapter over a 984-parameter character toy | `src/chakaso/models/` |
| Test lines | 16,036 across 93 `test_*.py` modules | `find tests -name '*.py' \| xargs wc -l` |
| Decision records | 32 ADRs; **one** (ADR-0030) concerns training; the rest concern retrieval, evidence, conversation, correction, storage, evaluation and planning | `docs/decisions/README.md` |
| CLI surface | 11 top-level commands behind 26 argument parsers in a 1,903-line `cli.py` | `python -m chakaso --help`, `grep -c add_parser` |
| Measured model results | 2 of 4 committed experiments are about a model (E2601, E2602); the other 2 are retrieval/evaluation instruments (E2603, E2604) | `experiments/` |

The project's own founding statement is that it "intends to train its own language model from random
initialization" ([`../../../AGENTS.md`](../../../AGENTS.md),
[`../../architecture.md`](../../architecture.md)). The distribution above does not match that statement: the
centre of mass of the engineering is in the layers that *serve* a model which does not exist.

This is drift, not a mistake to revert. The infrastructure is real and tested; the mistake would be to
keep expanding it at the same rate while the product — a trained model — stays a toy.

## 2. Actual model capability

**measured.** What a Chakaso-trained model can do today, stated at the strength the evidence supports:

- A 984-parameter character-level causal MLP, trained from random initialization on a 344-token
  self-authored corpus, reaches held-out next-character cross-entropy **1.6650 nats** at its
  validation-selected optimum (E2601 single seed; E2602 across five seeds: mean **1.7978 ± 0.1767
  nats**, below both the uniform 3.3322 and unigram 3.0531 baselines on every seed). The step-300 loss
  is worse than the optimum on every seed — the overfitting is systematic, not one lucky partition.
- It generates **incoherent text**. It is a demonstrated learning loop, not a language model in any
  useful sense, and no document may describe it as one.
- **No model answers a grounded question.** E2603 scored its grounded-answer workload with a reference
  oracle and a non-answering provider; the runtime's default adapter is still the double.

**verified.** The capability limits that the experiments did *not* test: no subword tokenization, no
Transformer, no attention, no sequence packing, no mini-batching, no sampling or decoding strategy, no
resume-from-checkpoint, no generation-quality measurement, no real corpus, no model on a real workload.
The rows in `docs/training.md` that still read "Not started" for tokenizer, model code and training loop
are stale — E2601/E2602 executed the character-level versions — and are corrected as part of this
realignment.

## 3. Current training and inference paths

**verified.**

```text
chakaso.training.corpus      self-authored 344-token toy text (dependency-free)
chakaso.training.tokenizer   CharVocabulary: codepoint-ordered ids, reserved UNK_ID (dependency-free)
chakaso.training.data        line-unit split, two-way and three-way, vocabulary fit on train only
chakaso.training.config      TrainingConfig (validated, frozen, dependency-free)
chakaso.training.model       CharCausalMLP, hand-derived gradients (numpy)
chakaso.training.optimizer   Adam over the Params bundle (numpy)
chakaso.training.loop        full-batch training with a held-out curve (numpy)
chakaso.training.checkpoint  JSON schema/2, content checksum, embedded vocabulary (numpy)
chakaso.training.evaluation  held-out loss / perplexity / bits-per-char / accuracy (numpy)
chakaso.training.experiment  E2601 driver; .controlled E2602 five-seed protocol (numpy)
        ↓
chakaso.models.trainedchar   the only consumer of a checkpoint above the boundary ([train] extra, lazy)
```

Two properties of this layering are load-bearing and must survive any realignment:

1. **The dependency direction is already correct.** Nothing in `chakaso.training` imports application
   code, and only `models/trainedchar.py` reaches from the boundary down into training (grep over
   `src/` confirms it). The application never imports the model implementations
   (`tests/test_repository_hygiene.py` enforces that structurally).
2. **The training path is already a Python API, not a CLI.** No training code lives in `cli.py`; the CLI
   reaches experiments through `chakaso.experiments`. The mission's "training must not be a terminal
   command" requirement is met for the toy path and must stay met for the real one.

What is *not* there: a path from a checkpoint to a Python caller that does not want a terminal, a
configuration file, or an application runtime. Loading a trained model today means constructing
`ModelConfig` or `TrainedCharacterModel(checkpoint_path=…)` by hand and driving the
`generate(messages, params)` boundary — there is no `load_model(...) → generate(prompt)`.

## 4. What is worth keeping

**verified.** These subsystems are sound, tested, and do not need to be rebuilt or removed. They are the
reason the realignment can be narrow:

- The model boundary and its contract suite (`chakaso.models.base`) — ADR-0002 works, and the trained
  adapter inherits the same suite as the double.
- The `src` layout, typed configuration with per-value provenance, error taxonomy, content-derived
  identifiers, and the repository-hygiene tests.
- Retrieval, evidence, claims, citation, grounding, correction, answers and their benchmarks — a real
  optional application layer, tested offline on CPU, which a future model plugs into rather than
  replaces.
- The experiment-record discipline (`docs/internal/experiments/README.md`), which is what lets section 2
  be stated in measured numbers at all.

**assumption.** That keeping this layer *optional and composable* costs nothing today. It costs the
documented coupling checks in section 7.

## 5. Architectural drift

| # | Drift | Evidence | Cost |
| --- | --- | --- | --- |
| D1 | The product surface is the terminal, not the library | 11 top-level CLI commands; the README's "Getting started" leads with `chakaso chat`; no documented Python entry point for a trained checkpoint | A developer cannot use the model without a terminal session |
| D2 | Model work is gated behind application work in the roadmap | `docs/roadmap.md` sequences M0 (system) before M1/M2 (tokenizer, Transformer); P27/P28 built evaluation workloads and a sufficiency policy while the model stayed a toy | Every application milestone is a decision made without a model to test it against |
| D3 | Evaluation instruments measure the harness, not the model | E2603/E2604 score an oracle and a deterministic policy; `describe_semantic_capability()` reports no real judge | A growing set of numbers that cannot answer "is the model getting better?" |
| D4 | The measured model is architecturally the wrong shape | `CharCausalMLP` is a concatenated-window MLP: no attention, no positional mechanism beyond the window, 4-character context | Its gradients are hand-derived and finite-difference-verified; that property does not scale to a Transformer (ADR-0030's revisit condition 2) |
| D5 | Data provenance exists only as an embedded string constant | `training/corpus.py` holds 344 tokens of text in a module; there is no manifest, no file-sourced corpus, no licence record beyond the project's own licence | A real corpus has no home, and no reproducible preparation path |
| D6 | Documentation disagrees with itself about the model layer | `docs/training.md` status table says tokenizer/model/training loop "Not started" while its own header cites a measured E2601 | A reader cannot tell what has been executed |
| D7 | The heavy numerics decision was made for a toy and never revisited | ADR-0030 rejected PyTorch "for now" with explicit revisit conditions and predicted a later ADR would carry a larger model | The next model step is blocked on re-deciding the dependency, not on writing code |

D7 is the one that must be resolved before any model code: the architecture decision that gates the next
experiment is a dependency decision, exactly as it was in P23.

## 6. Target state

**verified** as intent, **open** as implementation. Chakaso is a language-model project. The product is a
trained model and the reproducibility machinery around it; the conversation system is one optional
consumer of it.

The dependency direction the code must keep:

```text
corpus manifest → tokenizer → model → training → checkpoint → inference API → evaluation
                                                                          ↘
                                                             optional application layer
                                                             (retrieval, conversation, correction)
                                                                          ↘
                                                                     optional CLI
```

Rules this imposes:

1. A trained checkpoint is loadable and usable from Python with no terminal, no `Config` file and no
   application import.
2. Model code never imports CLI code; the CLI holds thin handlers over the same Python services
   (`chakaso.cli` already does this for experiments — the pattern extends, it is not invented).
3. Training never needs an interactive session; inference never needs the training loop alive.
4. A checkpoint carries or verifies everything needed to reproduce inference: tokenizer identity,
   architecture, vocabulary size, context length, dataset digest, seed, training configuration.
5. Model evaluation is runnable independently of the application runtime, and distinguishes a model
   failure from an evaluation failure.
6. Application subsystems stay importable and testable without any numerical dependency; the model
   layer must not become a load-bearing import for retrieval, storage or correction.
7. Heavy dependencies stay optional and isolated behind the module that needs them (ADR-0030's
   isolation rule extends to whatever ADR admits a framework).
8. The default test suite stays CPU-only, offline, no weight downloads, no network.

## 7. Compatibility risks

**verified.** The interfaces that must not break silently:

- `LanguageModel` / `ModelMetadata` / `GenerationParams` (`chakaso.models.base`) — every adapter
  inherits a shared contract suite (`tests/model_contracts.py`). A convenience layer over `generate`
  must not create a second model abstraction.
- `configs/default.toml` + `chakaso.config.schema` — `model.adapter` and `model.checkpoint_path` are the
  documented selection points; a new adapter is a new name, not a change of meaning.
- `chakaso.training.checkpoint` schema/2 — read by the `trained_char` adapter and by the E2601/E2602
  reproducibility path. A new artifact format must be a *new versioned schema* with explicit
  incompatibility behaviour, never an in-place edit of schema/2.
- `tests/test_repository_hygiene.py` allowlists — `CONCRETE_MODEL_MODULES` and
  `COMPOSITION_POINT_MODULES` enumerate adapter modules by name. A new adapter module must be added to
  both, in the same commit, or the ADR-0002 guard silently stops covering it.
- Dependency-policy tests (`test_the_base_install_declares_no_runtime_dependency`,
  `test_the_training_extra_is_numpy_and_bounded_for_the_python_floor`) pin `[train]` to exactly numpy. A
  new framework belongs in a *separate* extra, not in `[train]`.
- Documentation-link and ADR-numbering hygiene tests — every new ADR must be contiguous and indexed;
  every relative link in every committed markdown file must resolve.

**open.** Whether a `trained_char`-style adapter should remain a distinct registry name once a real model
exists, or become one adapter that dispatches on checkpoint schema. Decide when the second checkpoint
format exists, not before.

## 8. Test and CI baseline

**measured** on 2026-10-21 at `eaee98d`, this machine (Windows 11, CPython 3.13.5, CPU only):

| Gate | Result |
| --- | --- |
| `python -m pytest` (dependency-free `.venv`) | 1268 passed, 8 skipped in 16.76 s — the 8 skips are the numpy-gated training/adapter modules |
| CI on `main` | green for Python 3.11/3.12/3.13 (runs 39296297766 and predecessors) |
| `.github/workflows/ci.yml` | installs `.[dev]` only; format, lint, types, tests, entry-point check |
| numpy-gated training tests | run locally under `[train]`; **not** in remote CI — K-009 |

**measured.** The same gates after the model path landed, all three run on 2026-10-21 against the working
tree at `8c5feff` on this machine:

| Gate | Result |
| --- | --- |
| `python -m pytest` (dependency-free `.venv`) | 1,391 passed, 16 skipped in 13.74 s (1,407 collected; the skips are the numpy- and torch-gated modules) |
| `python -m pytest` (`.venv-train`, `.[dev,train]`) | 1,472 passed, 7 skipped in 17.34 s |
| `python -m pytest` (`.venv-nn`, `.[dev,train,nn]`) | 1,605 passed, 1 skipped in 56.66 s (1,606 collected) |
| `mypy src` (dependency-free) | clean over 117 files, with `chakaso/training` and the two trained adapters excluded as `pyproject.toml` documents |
| `mypy --config-file configs/mypy-model-path.toml` (`[nn]`) | **no issues in 142 source files** — the whole package, numpy and torch modules included, under the same strict settings. In `[train]` the same command reports 7 torch `import-not-found` errors, which is the correct answer: it requires both extras |
| `ruff check .` / `ruff format --check .` | clean, 317 files formatted |
| `.github/workflows/ci.yml` | unchanged: installs `.[dev]` only, so the two extra matrices above are still not run remotely |

**found while checking the gates, and it is a defect that was fixed.** `pyproject.toml` excluded the
optional-dependency modules from `mypy src`, and the documented way of checking them anyway —
`mypy src --exclude '_never_match_'` — does not work: mypy *appends* a command-line exclude to the
configured list, and a directory argument is subject to it too. The result was that 27 modules — the whole
`[train]` and `[nn]` path — had never been type-checked in any environment, in this session's runs included.
Type-checking them for real reported three dead `# type: ignore` comments and one genuine typing hole in the
byte-pair encoder's pair selection. The fix is `configs/mypy-model-path.toml` (same strictness, no
exclusions, requires both extras) plus two hygiene tests that keep the configurations from drifting and stop
the exclusion list changing shape quietly. It is recorded here rather than smoothed over because the previous
sentence in this document's plan — "mypy src is clean" — was, in the narrow sense, true and in the important
sense misleading.

**verified.** K-009's cause is still true and still binds this work: the session's GitHub credential has
scopes `gist, read:org, repo` and no `workflow` scope, so any commit touching
`.github/workflows/*` is refused by GitHub — re-checked on 2026-10-21 against `gh api user`. Nothing in this
realignment may claim a CI job that cannot be pushed; new optional-dependency tests are gated by
`importorskip` and their non-execution in remote CI is recorded, not hidden.

**measured.** Local environment for the next model step, observed the same day (and re-inspected rather
than recalled from P22, which observed a different state):

| Fact | Value | How observed |
| --- | --- | --- |
| Logical CPUs | 18 | `os.cpu_count()` |
| Total / free physical RAM | 16.57 GB / 3.70 GB | `GlobalMemoryStatusEx` |
| Free disk on `D:` | 168 GB | `shutil.disk_usage` |
| CUDA device | none | `torch.cuda.is_available()` → False |
| `torch` on the interpreter | 2.13.0+cpu, 14 intra-op threads | `import torch` |
| `numpy` on the interpreter | 2.4.0 | `import numpy` |
| Transformer step cost, CPU (batch 32, ctx 128, vocab 4096, d 256, 4 layers, 5.26 M params) | 1.82 s/step, 33 steps/min | executed forward+backward+step ×3, wall-clock |

P22's "~1.4 GB free RAM, no ML library installed" is **superseded as a current observation** by this
table; the earlier note stays as the record of what was true when ADR-0030 was written, and ADR-0030's
decision is judged against its own revisit conditions rather than rewritten.

## 9. Subsystem classification

Per-module verdict against the model-centred contract. "Consumer" means something in this repository or a
documented workflow actually uses it.

| Subsystem | Class | Consumer | Verdict |
| --- | --- | --- | --- |
| `core/` | Public interface (primitives) | everywhere | Keep |
| `config/` | Public interface | CLI, `app` | Keep; model configuration grows a real home, not more application sections |
| `models/` | **Core model** + public interface | runtime, CLI, tests | Keep and extend: a checkpoint loader as the primary entry point |
| `training/` | **Core model**, training and data infrastructure | `models.trainedchar`, experiments | Keep, extend into the real model path; its dependency isolation is the template |
| `experiments/` | Model evaluation (currently retrieval evaluation) | CLI, benchmarks | Keep; add model experiments as first-class, distinct from retrieval-strategy comparison |
| `retrieval/` | Optional application capability | runtime, CLI, benchmarks | Keep as optional; must never become a training dependency |
| `evidence/`, `citation/`, `claims/`, `grounding/` | Optional application capability | runtime, evaluation | Keep |
| `planning/`, `conversation/`, `runtime/` | Optional application capability | `app`, CLI | Keep |
| `answers/` | Optional application capability (persistence) | runtime, CLI | Keep |
| `correction/` | Optional application capability | `app`, benchmark | Keep |
| `evaluation/`, `benchmark/` | Model evaluation *and* application evaluation, currently conflated | CLI, experiments | Keep, but separate model metrics from harness metrics in documentation and naming |
| `cli.py` | Public interface (secondary) | humans, tests | Keep thin; do not add model behaviour here |
| `app.py` | Public interface (composition root) | CLI | Keep |
| Development doubles and fixtures (`models/deterministic`, `retrieval/embeddings` fixture, benchmark fixtures) | Development-only fixtures | tests, default runtime | Keep, and keep them visibly labelled as doubles |
| Semantic judge (`grounding/semantic`) | Research proposal, boundary only | tests, benchmark | Keep as documented scaffolding; no new scaffolding before a model exists |

**verified.** No module is currently dead. The cross-layer edges were
`models/trainedchar → training` (the intended adapter direction) when this was first written, and after P31
they are `models/trainedchar`, `models/nntransformer` and `models/registry` → `training`, plus a lazy import
inside the CLI's model handler — all of them pointing the one way the contract allows. No dependency
inversion is needed to make training independent of the application — it already is (§11 re-checks this with
the command). The realignment therefore adds a model path rather than rewiring the existing one, and the
coupling guard is a test, not a refactor.

## 10. What this audit changes

As written on 2026-10-21 before any of it existed. §11 records what was delivered and what was not.

1. The product contract becomes enforceable text: ADR-0033 (model-centred product, library API primary,
   CLI secondary), reflected in `AGENTS.md`, the README and `docs/architecture.md`.
2. The dependency decision that gates the next model step is re-made on measured evidence, as an
   amendment rather than a reversal: ADR-0034.
3. The model path is built out in dependency order — corpus manifest, subword tokenizer, packing,
   causal Transformer, training, versioned checkpoint, Python inference API, model metrics — each with
   tests that fail if the guarantee is dropped.
4. `E2605` measures the first comparison the project has never been able to run: two tokenizations and
   two architectures, the same corpus, the same held-out text, reported in bits per character so the
   number is tokenizer-independent.
5. Documentation states what exists at the strength the evidence supports, including the parts that stay
   toy, and the parts this realignment does not deliver.

## 11. Delivered state, read back against this audit (2026-10-21)

The audit's §10 list is now history rather than a plan, and the two claims it made about the existing tree
were re-checked rather than trusted:

- **verified.** `grep -rl "chakaso.training" src/chakaso/` names exactly four non-training modules:
  `models/trainedchar.py`, `models/nntransformer.py`, `models/registry.py` and `cli.py`. The first three are
  the intended adapter direction (ADR-0002's single selection point imports lazily behind it); `cli.py`
  imports the experiment driver *inside* the handler, so a dependency-free install never loads torch. No
  application module — `app`, `runtime`, `conversation`, `retrieval`, `evidence`, `answers`, `correction`,
  `evaluation`, `benchmark` — imports the training package, and no training module imports application code
  in either direction. The model layer and the harness layer are independent, and the §9 verdicts
  ("Keep", "Keep as optional", "Keep thin") hold unchanged: nothing was deleted by this realignment, and the
  subsystems it classified as optional application capability are still exactly that.
- **verified, and it is a limitation rather than a win.** §5's drift list is repaired in the sense that the
  documents now describe the code, but the drift the manifest was built to catch has recurred one level up:
  the corpus E2605 pinned is 71 documents of *this* documentation, and filling in the record and re-aligning
  the docs after the run changed 13 of them. `verify_corpus` refuses to re-prepare that corpus against the
  current tree, which is the designed behaviour and is recorded in the experiment record — with the follow-up
  it exposes: the driver must write the snapshot it can be re-trained from. A project that trains on its own
  documentation has to treat that documentation as a dataset version, not as a living file, and this
  repository does not yet have that discipline.
- **not delivered by this realignment, and not claimed:** any use of the trained checkpoint by the runtime
  (P32), an external corpus, multi-seed reproduction (P33/P34), a CI job for the optional extras (K-009, and
  the credential that prevents it), a peak-memory figure for any run, and any capability claim. The single
  measured model result is 1.8657 bits per character on one author's prose, with the generations that loop
  committed beside it.
