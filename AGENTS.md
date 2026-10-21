# AGENTS.md

Instructions for coding agents working in this repository. Humans are welcome to
read it too; it is the same contract.

## What this repository is

Chakaso is a **language-model research project**. The product is a model this project trained itself —
its weights, its tokenizer, its data provenance, its training reproducibility and its measurable
behaviour — together with the Python interface that lets a developer use a trained checkpoint. That is a
decision, not a slogan: [ADR-0033](docs/decisions/ADR-0033-model-centred-product-contract.md).

Everything else in the repository serves that product. The local-first conversational system — retrieval,
evidence identity, citation, grounding, correction, durable answers — is real, tested, and is the reason a
grounded model could later be measured honestly. It is one consumer of the model, not the project's
centre, and it is not the primary interface either: the Python library is. A CLI, a web frontend or a
hosted API may exist only with a technical justification that does not displace model work.

What exists today, precisely: a conversation can be held, but the model implementations behind the boundary
are a deterministic development double, an adapter over a measured 984-parameter character toy, and an
adapter over a measured 643,840-parameter subword Transformer (E2605, reproduced and put on a spread by
E2606). None of the three answers a grounded question, and neither trained model is useful text. Local
lexical retrieval exists (ingest, corpus, BM25, evidence
pack); web fetching is opt-in and off every default path — see
[`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md), which is the authority on what is real.
[`docs/internal/engineering/model-centred-audit.md`](docs/internal/engineering/model-centred-audit.md)
records how the repository got that way and what the realignment changes.

## Read order

1. [`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md) — what exists right now.
2. [`docs/agent/ACTIVE_TASK.md`](docs/agent/ACTIVE_TASK.md) — the current unit of work and what follows it.
3. [`docs/agent/CONVENTIONS.md`](docs/agent/CONVENTIONS.md) — how code and docs are written here.
4. [`docs/architecture.md`](docs/architecture.md) — components, boundaries and the dependency direction.
5. [`docs/decisions/`](docs/decisions/) — why the architecture is shaped this way. Read before proposing a change to any boundary.
6. [`docs/training.md`](docs/training.md) and [`docs/research/next-model-experiment.md`](docs/research/next-model-experiment.md) — the model path and the measured evidence behind the next experiment.

## Non-negotiables

**Never commit the private documentation pack.** `docs/*.docx` is the author's
private planning material. It is git-ignored, must stay where it is, and must not
be deleted, moved, copied into the repository, or summarised wholesale into a
tracked file. `tests/test_repository_hygiene.py` fails if any of it becomes
tracked. If you find it staged, unstage it and say so.

**The model is the unit of work while the model is a toy.** A change whose value is "another command" or
"another application subsystem" is not the next unit while no useful model exists. Model-path work means
data, tokenization, architecture, training, checkpoints, inference, model evaluation.
[ADR-0033](docs/decisions/ADR-0033-model-centred-product-contract.md).

**A trained checkpoint must be usable from Python.** Loadable and generatable with no terminal, no
configuration file and no import of the conversation runtime. CLI handlers stay thin and call the same
Python services; no core model behaviour may live only inside a CLI handler.

**A measured result must stay re-trainable.** A record that trains on tracked text names the revision its
corpus came from, and the run reads a checksum-verified snapshot — never the current working tree
([ADR-0036](docs/decisions/ADR-0036-corpus-snapshot-recovery-from-history.md)). The drift guard is never
weakened to let a re-run proceed, a reproduction is a new record with its own deviations rather than an edit
to the numbers it reproduces, and an experiment's artifacts (manifest, snapshot, checkpoint, report) are
written atomically and refused from being clobbered unasked. Seeds, budget, metric definitions and the primary
comparison are declared in the record before any run, and a result is aggregated from the reports on disk.

**Numerical dependencies stay optional and confined.** `numpy` belongs to `[train]`; `torch` belongs to
`[nn]` and may be imported only by the modules
[ADR-0034](docs/decisions/ADR-0034-optional-pytorch-model-extra.md) names. `import chakaso`,
`import chakaso.models` and `import chakaso.training` must not require either, and a missing extra is a
named error, never a silent fallback to the double.

**Do not describe planned work as done.** Every capability claim in documentation
carries one of: Implemented, Experimental, Planned, Research, Unknown. Definitions
are in [`docs/README.md`](docs/README.md). A change that implements something
updates that status in the same commit.

**Do not invent numbers, datasets, citations or experiments.** If it was not
measured, it is not written down. The measured results in this repository are
executed comparisons over small synthetic or self-authored development fixtures — the retrieval-strategy
measurement E-001, E2601 and E2602 (a tiny character LM whose held-out next-token loss beats uniform and
unigram baselines, recorded in
[`experiments/E2601/record.md`](experiments/E2601/record.md)), E2603/E2604 (retrieval and
evidence-sufficiency measurements over a self-authored workload), and E2605/E2606/E2607 (a 643,840-parameter
subword Transformer trained from random initialization: E2605 on this repository's own pinned prose, E2606 the
same protocol across five pre-declared seeds, reproducing E2605 exactly, and E2607 on an externally licensed
Project Gutenberg corpus). All of them check the shape of a mechanism or a learning loop on hand-built or
public-domain text, never meaning on a real workload. There
is no measurement of a real workload and no result from any *useful* language model —
the trained artifact is an experiment, not a deployed model — and it should stay obvious
that this is true. Implementing a training loop is not a measured run; a measured run on a toy is not
capability; an oracle scoring a harness is not a model.

**Distinguish model measurement from harness measurement.** A retrieval benchmark, a reference oracle, a
citation checker or a deterministic policy is evidence about the plumbing. It is never evidence of
language-model competence, and it is labelled that way wherever it is reported.

**The runtime must not require a commercial inference or search API.**
[ADR-0001](docs/decisions/ADR-0001-local-first-runtime.md). External providers may
only exist as optional adapters behind an existing interface.

**Application code must not import a concrete model implementation.**
[ADR-0002](docs/decisions/ADR-0002-language-model-boundary.md). Configuration maps
to a concrete model in exactly one place.

**The model may not produce source identity.**
[ADR-0003](docs/decisions/ADR-0003-evidence-identifier-ownership.md). Identifiers
come from the retrieval layer; an identifier the model was not given is an error
to record, never a URL to render.

**Do not create empty scaffolding.** No placeholder packages, no TODO-only
modules, no directories created because a roadmap mentions them. A module appears
when there is a real reason for it, with tests and a stated responsibility. A
directory listing future work belongs in [`docs/roadmap.md`](docs/roadmap.md).

**Do not weaken or delete a test to get CI green.** If it fails, the code or the
expectation is wrong.

**A changed interface needs a decision record or an update to the relevant
specification, in the same commit.**

## Commands

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

python -m pytest                   # tests
python -m ruff check .             # lint
python -m ruff format --check .    # format check
python -m mypy src                 # types
```

All four must pass before a commit is pushed. CI runs exactly these.

The four gates above are the dependency-free matrix: `numpy` and `torch` are absent there, so the
numerical and Transformer tests skip and `mypy src` does not resolve their types. Anything that touches
the model path is also run with the optional extras installed, and that is a separate command, not a
substitute for the four:

```bash
python -m pip install -e ".[dev,train]"        # numpy: the character model and its tests
python -m pip install -e ".[dev,train,nn]"     # + torch: the Transformer path
python -m pytest                               # now runs the gated modules instead of skipping them
python -m mypy --config-file configs/mypy-model-path.toml   # now type-checks every module
```

`--exclude` on the command line *adds* to the exclusion configured in `pyproject.toml`, so it cannot lift
it; the second configuration file is what checks the model path, and it needs both extras installed.

On Linux, install the CPU wheel index explicitly (`--extra-index-url
https://download.pytorch.org/whl/cpu`); PyPI's default `torch` wheel is not the CPU build
([ADR-0034](docs/decisions/ADR-0034-optional-pytorch-model-extra.md)).

Reproducing a measured run needs the corpus it consumed, which is not the text on disk once the repository
has edited it. The full sequence is in [`docs/training.md`](docs/training.md); the shape is:

```bash
chakaso corpus restore --manifest experiments/E2605/corpus-manifest.json --revision 9b8a478 \
  --output data/experiments/E2605/chakaso-repo-text-v1-snapshot.json   # reads history, verifies, writes a snapshot
chakaso corpus verify  --manifest experiments/E2605/corpus-manifest.json \
  --snapshot data/experiments/E2605/chakaso-repo-text-v1-snapshot.json
chakaso experiment model --work-dir … --corpus-snapshot … --expected-corpus-fingerprint e45993f9badf706c
chakaso experiment study --work-dir … --experiment E2606 --seeds 7,11,13,17,19 --split-seed 7 …
chakaso experiment study … --collect-only     # rebuild an aggregate from the reports, training nothing
```

`corpus restore` reads a local revision with `git archive`; it needs a clone with history, and it refuses a
revision whose tree is not the pinned corpus rather than substituting one.

## Repository map

```text
src/chakaso/        package (src layout — you must install it or set PYTHONPATH)
tests/              unit, contract and repository-hygiene tests
configs/            versioned configuration, TOML
experiments/        committed experiment records, one directory per experiment
data/               prepared, git-ignored training artifacts (corpora, checkpoints) — never committed
docs/               public documentation
docs/decisions/     architecture decision records
docs/research/      research log, hypotheses, literature, open questions
docs/agent/         this directory's context: current state, conventions, active task
docs/internal/      engineering notes that are not for external readers
```

Experiment records live at `experiments/<experiment-id>/record.md`; seven are committed (E2601–E2607) and
they are the project's only measured results. The format is defined in
[`docs/internal/experiments/README.md`](docs/internal/experiments/README.md), which requires the record to
be written **before** the run. Generated artifacts — corpora, checkpoints, logs — belong in `data/` and are
never committed ([`docs/internal/engineering/artifact-policy.md`](docs/internal/engineering/artifact-policy.md)).

The `src` layout is deliberate: it makes an uninstalled import fail rather than
silently testing the working tree
([ADR-0005](docs/decisions/ADR-0005-src-layout.md)).

## Working method

Inspect → understand → decide → implement one coherent unit → test → review →
document → commit → push → verify → next unit.

The next unit is chosen from the model path — data, tokenization, architecture, training, checkpoint,
inference API, model evaluation — while the model remains a toy
([ADR-0033](docs/decisions/ADR-0033-model-centred-product-contract.md)). Application subsystems are still
maintained, and a measured failure in one is still a reason to fix it; what changes is that "add another
command or another application layer" is no longer a default next unit.

One commit is one coherent change that can be understood on its own. Types are
`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:`, `research:`. Do
not split a single change into several commits, and do not bundle unrelated
changes into one.

Before every commit:

- the change is meaningful and the code is actually needed,
- tests cover the behaviour and pass,
- formatting, lint and types pass,
- documentation matches the code, including status labels,
- a decision record exists if a decision was made,
- `docs/agent/CURRENT_STATE.md` is updated if what exists has changed,
- `CHANGELOG.md` is updated if the change is notable for a reader,
- no private material and no secrets are staged.

## Traps

- **`docs/*.docx` is not repository content.** It is the private pack. Ignore it,
  and never assume it is absent because it is not in `git status`.
- **`git status` will not show the private pack** because it is ignored. Check
  tracked files instead if you need to be sure: `git ls-files | grep docx`.
- **The package is not importable without installing it** when using the
  repository as the working directory. That is intentional.
- **No GPU is available in CI.** Tests must be CPU-only and offline. A test that
  needs the network must be marked and skipped by default.
- **`numpy` and `torch` are absent from the four default gates.** A test that imports them at module level
  fails the dependency-free matrix; numerical tests are `pytest.importorskip(...)`-gated so that matrix
  skips them. `torch` costs ~3 s to cold-import, so it must never be reachable from a default path —
  the allowlist in [ADR-0034](docs/decisions/ADR-0034-optional-pytorch-model-extra.md) and the hygiene
  tests are what keep that true. Run the gated suite locally with the extras installed before claiming a
  numerical behaviour is verified.
- **Workflow files cannot be pushed from an OAuth token without the `workflow` scope.** GitHub refuses the
  whole push, naming the file. K-009 records this project's consequence: `[train]` and `[nn]` jobs do not
  run in remote CI. Never document a CI job that has not landed.
- **Checkpoint and corpus bytes stay out of Git.** They live in `data/`, are git-ignored, and are cited by
  fingerprint in the experiment record.
- **The research log is not a place for opinions.** Entry types are defined in
  [`docs/research/README.md`](docs/research/README.md).
