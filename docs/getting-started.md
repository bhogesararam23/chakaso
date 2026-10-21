# Getting started

## What you can actually run

**A trained model, from Python.** `chakaso.models.load_model(path)` loads a checkpoint this project
trained — the character toy of E2601, or the subword Transformer of E2605 — and `model.sample(...)`
generates from it. That path needs the optional extras (`.[train]`, `.[nn]`); the base install stays
dependency-free, and the trained model is a small one measured on this repository's own text, which its
own metadata says out loud.

**The training path itself**, as a library call and as a thin command: `chakaso.training.manifest`,
`.subword`, `.packing`, `.transformer`, `.nn_training`, `.nn_evaluation` and
`chakaso.training.subword_experiment.run_experiment`, or `python -m chakaso experiment model
--work-dir ...` for the whole E2605 comparison.

**Several local, offline commands:** `chakaso chat`, a conversation shell that grounds each turn
through the retrieval-aware runtime; `chakaso retrieve`, which ingests named local documents and runs
lexical retrieval over them (and `--source` restricts a query to named sources); `chakaso sources
list|show` and `chakaso evidence show`, which inventory and inspect the in-memory corpus of sources
and chunks built from those documents; `chakaso capabilities`, which reports what configuration
selects; and `chakaso answers list|show|lineage`, which inspects persisted answers and their
correction lineage; `chakaso benchmark retrieval` and `chakaso benchmark correction`, which
run the development benchmarks and print their reports; `chakaso experiment run <subject>`, which
produces a reproducible experiment result; `chakaso storage check` / `chakaso storage migrate`, which
inspect or initialize a durable answer store; and `chakaso config show` / `chakaso --version`.

**There is no useful language model, and no model in the default runtime path.** The default
implementation of the model boundary is a deterministic development double that returns fixed text
describing the request it received; it does not answer questions, and the shell says so before it produces
anything. What the trained adapters do produce is measured: a 984-parameter character model (E2601/E2602)
and a subword Transformer of roughly 0.6M parameters (E2605), both trained on this repository's own
documentation on CPU. They beat the uniform and unigram baselines on held-out text and they do not
constitute a capable assistant — that distinction is the point of the experiment records.

Retrieval exists but is **local**: it reads the files you name, normalizes and chunks
them, and ranks them for a query. `chakaso retrieve` ranks lexically (BM25 — shared words,
no embeddings, no meaning); `chakaso chat --strategy` can additionally rank with a dense or
hybrid retriever, but their only embedding is a deterministic **non-semantic** fixture
(ADR-0023/ADR-0024), so no strategy here retrieves by meaning. Below it, claim/citation/grounding evaluation, a correction decision rule with
reassessment, an answer store (in-memory and, since ADR-0021, a durable SQLite backend that
survives a process restart) and reproducible experiment records exist as tested code —
but there is still no model that answers a grounded question, no web fetching on any default path (the
fetcher is opt-in) and no automatic correction loop. Tokenizers (character and byte-pair subword), a
document-pinned corpus, a Transformer, mini-batch training, checkpoints and held-out model metrics all
exist and are tested; none of it has met a real workload (K-001). The benchmark and experiment
commands report development metrics computed over small synthetic fixtures, not model-quality or
real-world results. `chakaso config show` inspects configuration, and `chakaso --version` verifies an
install.

This document describes how to install, verify and run what exists. It will grow as
the system does, and it will not describe anything that does not exist.

## Requirements

- Python 3.11 or later. (Chakaso reads TOML configuration with the standard
  library `tomllib`, which arrived in 3.11. See
  [`decisions/ADR-0006`](decisions/ADR-0006-toml-configuration.md).)
- `git`
- No GPU. Nothing in the test suite requires one, and CI does not have one. Training the Transformer is
  CPU-only work: the recorded E2605 run completes in under half an hour on 18 logical CPUs.

## Setup

```bash
git clone https://github.com/bhogesararam23/chakaso.git
cd chakaso

python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install the package with its development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`.[dev]` installs the runtime dependencies plus the tools used by CI: `pytest`,
`ruff` and `mypy`. Nothing else is required, and no dependency is a commercial
inference or search provider. The base install declares **no** runtime dependency at all.

For the model path, add the optional tiers ([ADR-0030](decisions/ADR-0030-minimal-optional-training-dependency.md)
for numpy, [ADR-0034](decisions/ADR-0034-optional-pytorch-model-extra.md) for torch):

```bash
python -m pip install -e ".[dev,train]"        # the character model and its tests
python -m pip install -e ".[dev,train,nn]"     # + the Transformer path
```

On Linux, point pip at the CPU wheel index — PyPI's default `torch` wheel is not the CPU build:

```bash
python -m pip install "torch>=2.6,<3" --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,train]"
```

`import chakaso`, `import chakaso.models` and `import chakaso.training` must keep working with neither
extra installed; selecting a trained adapter is what requires one, and it fails by name instead of
quietly substituting the double.

## Verify the install

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m chakaso --version
```

All five should succeed. If `mypy` is slow on a first run, that is normal; it
caches under `.mypy_cache/`.

The test suite is designed to run offline. If a test needs the network, it is
marked and skipped by default.

### The three matrices

| Extras installed | What runs | What skips |
| --- | --- | --- |
| `.[dev]` (the CI matrix) | every dependency-free module, the application layer, the corpus manifest, both tokenizers, packing, the model metrics arithmetic | the numpy character model, the Transformer, its training, checkpoint and evaluation tests |
| `.[dev,train]` | + the character model, its hand-derived gradients verified against finite differences, its checkpoint and held-out evaluation | the torch modules |
| `.[dev,train,nn]` | everything, including the Transformer, mini-batch training, artifacts, evaluation and the inference adapter | nothing |

The skips are deliberate and are the same gate on every machine: `pytest.importorskip`. Because this
session's GitHub credential has no `workflow` scope, the `[train]` and `[nn]` jobs cannot be added to CI
(K-009), so the extras tiers are run locally and the gap is recorded rather than claimed away.

## Load and use a trained checkpoint

```bash
python -m pip install -e ".[dev,train,nn]"
python -m chakaso experiment model --work-dir data/experiments/E2605   --write-report experiments/E2605/results.json     # trains; see experiments/E2605/record.md
```

```python
from chakaso.models import load_model

model = load_model("data/experiments/E2605/E2605-transformer.pt")
result = model.sample("The retrieval layer owns", max_new_tokens=64)

print(result.text)
print(model.metadata.development_double)          # False: a trained checkpoint, stated
print(model.artifact.corpus_fingerprint)          # which pinned corpus produced it
print(model.score(open("docs/training.md").read()))  # held-out cross-entropy, in nats
```

`load_model` reads the artifact's own declared schema to choose the adapter and never falls back: a
missing extra, an absent file or a corrupt checkpoint is a `ModelError` naming the path, because a silent
substitution of the double would turn a configuration mistake into a result that looks like a model
answered. The loaded object is a `LanguageModel`, so `model.generate(messages, params)` works through the
same boundary the application uses.

### Testing the declared Python floor

CI runs the suite on Python 3.11, 3.12 and 3.13. Running only the interpreter you
happen to have is not enough: the floor is a real constraint, and a construct that
works on 3.13 can fail at import on 3.11.

A version-suffixed virtual environment checks the floor without disturbing your
main one. The `.gitignore` covers `.venv*` for this reason.

```bash
# with uv, which can also fetch the interpreter
uv python install 3.11
uv venv .venv311 --python 3.11
uv pip install --python .venv311/Scripts/python.exe -e ".[dev]"
.venv311/Scripts/python.exe -m pytest

# or with the standard library and an interpreter you already have
python3.11 -m venv .venv311
```

On Linux and macOS the paths are `.venv311/bin/python` instead.

## Holding a conversation

```bash
python -m chakaso chat
```

`chat` conducts turns through the retrieval-aware runtime, assembled by `chakaso.app` from the
resolved configuration (ADR-0026): the query planner decides each turn, retrieval runs over any
documents you supply with `--file`, and every answer is recorded in the configured store. The shell
reads messages from standard input and prints replies to standard output. Two commands end it:
`:quit` and `:exit`, and end of input works too. Blank lines are ignored rather than sent. A failed
turn is reported and the session continues, because a failure leaves the conversation unchanged
([ADR-0008](decisions/ADR-0008-transactional-turns.md)).

To ground answers in local documents, name them with `--file` (repeat the flag):

```bash
python -m chakaso chat --file docs/roadmap.md --file notes/spec.md
```

With no `--file` the corpus is empty, so a retrieving turn honestly returns no evidence — the
planner still runs and nothing is invented. To send one message and exit, which is what a script
wants:

```bash
python -m chakaso chat --message "What does the specification say?"
```

Configuration is chosen the same way `config show` chooses it, and a few `[retrieval]` settings can
be overridden per run — `--strategy lexical|dense|hybrid`, `--top-k N`, and `--no-retrieval` to
conduct turns without grounding. Each override is named on standard error so a run never looks like
it used the file's values when it did not:

```bash
python -m chakaso chat --config configs/default.toml --strategy hybrid
```

`dense` and `hybrid` run over a non-semantic fixture embedding (ADR-0023/ADR-0024): they exercise
the retrieval boundary and reproduce exactly, but they are not semantic understanding.

### Read this before believing a reply

The shell prints a notice on **standard error** before the first reply. With the default
configuration (retrieval on, no documents, in-memory store) it reads:

```
engine: deterministic (Deterministic double (not a language model))
This is a development double, not a language model. It returns fixed text that
exercises the conversation plumbing, and it does not answer anything.
retrieval: lexical selected, but no documents were supplied, so a retrieving turn
returns no evidence (pass --file to ground answers).
answers are recorded in memory for this process only and are not saved to disk.
```

That is not a disclaimer bolted on at the end. The `development_double` flag is part of the model's
own metadata, and the retrieval and persistence lines describe what the composition actually wired
from configuration — so the notice is derived from what the components say about themselves rather
than a hard-coded string that could drift out of date.

The reply goes to standard output and every report line to standard error, so `--message` stays
usable in a pipeline. Scripting it is fine; reading its output as an answer is not.

Two more things the shell will tell you about, on standard error, when they happen:

- **A truncated reply.** Generation stopped at the configured length ceiling, so the reply is
  incomplete. A caller that cannot tell that from a finished answer would present one as the other.
- **An unresolved evidence reference.** The reply cited evidence it was not given. Nothing was
  resolved to a source, and the reference is listed
  ([ADR-0009](decisions/ADR-0009-unresolved-references-are-recorded.md)). Grounding is structural:
  a resolved citation shows a source was referenced, never that it proves the answer, and a
  grounded turn is reported with the planner's decision, the evidence offered and the sources cited.

## Retrieving from local documents

```bash
python -m chakaso retrieve --query "what are the deadlines" --file docs/roadmap.md
```

`retrieve` ingests each `--file` (repeat the flag for multiple sources) into an
in-memory corpus, runs lexical BM25 retrieval for `--query`, and prints the ranked
chunks with their provenance — score, section, matched terms and the source's canonical
reference. `--top-k` bounds how many are returned.

It is local and offline: it reads only the files you name, never crawls a directory, and
touches no network and no model. It prints evidence, not an answer — there is no language
model to turn retrieved chunks into prose — and an empty result says so plainly rather
than inventing evidence. Only plain text and Markdown (`.txt`, `.md`, `.markdown`) are
read; any other format is refused rather than mangled into text.

To restrict retrieval to particular sources, pass their identifiers with `--source` (repeat the
flag). An identifier that is malformed, or that names a source not among the `--file` documents, is
refused with a clear error rather than silently filtering to an empty result:

```bash
python -m chakaso retrieve --query "deadlines" --file notes/spec.md --source src_3f2a91c4d5e6b708
```

## Inspecting sources and evidence

To see what the corpus actually holds before trusting an answer over it, list and inspect the
sources built from the documents you pass with `--file`:

```bash
# inventory the sources and how many chunks each produced
python -m chakaso sources list --file notes/spec.md --file notes/plan.md
python -m chakaso sources list --file notes/spec.md --json

# show one source and the chunks ingested from it (--text also prints each body)
python -m chakaso sources show <src_...> --file notes/spec.md --text

# show a single evidence chunk and the source it came from
python -m chakaso evidence show <chk_...> --file notes/spec.md
```

The corpus is **in-memory and rebuilt for the invocation**, exactly as `retrieve` and `chat` build
it: there is no durable source index, so a source or chunk appears only while its document is passed
with `--file`. Each command says so when asked about something that is not present, rather than
implying a persistent store exists. Identifiers are validated before lookup and are never normalised
into a different object; `--json` gives a valid machine-readable view whose text fields keep the exact
stored bytes, while the human-readable output renders source text (which is untrusted) inert so a
hostile document cannot emit terminal-control sequences. These commands are read-only and offline.

## Running the development benchmarks and an experiment

```bash
# score the lexical retriever over the retrieval development benchmark
python -m chakaso benchmark retrieval            # add --json for machine-readable output
python -m chakaso benchmark retrieval --top-k 5

# check the correction decision rule over its development benchmark
python -m chakaso benchmark correction           # add --json for machine-readable output

# run a baseline experiment and print its reproducible, content-fingerprinted result
python -m chakaso experiment run correction --json
```

These run the same library code the tests cover. Every report labels itself a **development**
instrument: the corpora are small and synthetic, the numbers are computed over those fixtures at run
time, and none is a claim about real-world retrieval, grounding, correction or model quality. Output
is deterministic and offline. Because there is no language model, the correction benchmark scores a
decision rule over labelled cases, not generated answers, and an experiment result is a
reproducible record of that measurement.

## Inspecting durable storage

The answer history can be persisted to a local SQLite file (ADR-0021), but only when the durable
backend is selected and pointed at a path — the default backend is in-memory and writes nothing. A
developer can create or confirm a store's schema, and inspect it without changing it:

```bash
# create (or confirm) a durable store's schema at an explicit path
python -m chakaso storage migrate --path ./answers.db

# report its schema version and integrity, read-only
python -m chakaso storage check --path ./answers.db --json
```

Both commands touch only the path given, and either refuses a database whose schema it cannot safely
interpret rather than repairing it. There is still no automatic persistence: a normal conversation
writes nothing to disk unless a caller wires a durable store in.

## Running without installing

The package uses a `src/` layout, so `python -c "import chakaso"` from the
repository root fails until the package is installed or the source directory is
on the path. That is deliberate: it prevents tests from passing against a
half-installed tree. To work without installing:

```bash
# Linux / macOS
PYTHONPATH=src python -m pytest

# Windows PowerShell
$env:PYTHONPATH = "src"; python -m pytest
```

## Inspecting capabilities and stored answers

Before running anything, ask what the configuration would wire — read-only, and without creating a
store or reading a file:

```bash
python -m chakaso capabilities            # the selected model, strategy and store backend
python -m chakaso capabilities --config configs/default.toml --json
```

The model line reports whether the engine is a development double, straight from its own metadata,
so the output cannot quietly claim a model that does not exist.

Answers recorded to a durable store can be inspected back through the same store `chat` wrote to.
With the default in-memory backend there is nothing to read after the process exits, and the command
says so:

```bash
python -m chakaso answers list --config configs/default.toml
python -m chakaso answers show <ans_...> --config configs/default.toml --json
python -m chakaso answers lineage <ans_...> --config configs/default.toml
```

This reads the configured SQLite store only (`initialize=False`, so it never creates or mutates a
database) and prints the answer's provenance — model identity, retrieval strategy, evidence and cited
sources, and any `correction_of` link — so a grounded turn is auditable after the fact. `answers
show` also reports the turn's **historical evaluation provenance** (ADR-0028): the citation counts,
the per-claim structural-grounding statuses, the evaluator name and version that decided them, and
any evidence references in the answer that did not resolve. It is labelled *recorded at the turn,
not recomputed on read* — the durable record keeps what the original turn decided while the live
evaluation view remains recomputed and is never frozen into history. Structural grounding is exactly
that: `is_grounded` and `evidence_coverage` report citation structure, never that the evidence proves
the answer.

## Configuration

Configuration is TOML, parsed with the standard library and loaded explicitly. No
file is discovered by scanning, and no environment variable is substituted for a
file value: a caller passes the files it wants, in order.

```bash
# The built-in defaults, with no file read at all
python -m chakaso config show

# The repository's versioned defaults
python -m chakaso config show --config configs/default.toml

# Versioned defaults plus a machine-local override, later files winning
python -m chakaso config show --config configs/default.toml --config configs/local.toml
```

Every value is printed with the source it came from, so "temperature was 0.0" can
be distinguished from "temperature was 0.0 because a file that has since been
edited said so".

An unknown key is an error, not a warning. A typo in a configuration file produces
a run that does not use the settings its author believes it uses, and that is worse
than a startup failure. The error names the file, the key, and the keys that do
exist.

## Repository layout

```text
src/chakaso/           package source
  config/              typed configuration, loading and provenance
  core/                identifiers, hashing, terminal-safe text, error base
  models/              language-model boundary, capabilities, registry
  evidence/            source records, evidence chunks, packs, citation resolution
  retrieval/           normalization, chunking, ingestion, corpus, lexical index and retrieval
  conversation/        immutable conversation state, conversation manager
  runtime/             retrieval-aware runtime: plan → retrieve → model → record → store
  app.py               application composition root: configuration → assembled dependencies
  claims/              claim representation and extraction boundary
  citation/            structural citation validation and metrics
  grounding/           grounding evaluators (structural, manual, semantic boundary)
  evaluation/          metric functions and the answer-evaluation combiner
  correction/          decision rule, reassessment, records, metrics, application service
  benchmark/           versioned cases, fixtures, retrieval and correction runners and reports
  answers/             answer identity, records and the append-only answer store
  experiments/         experiment and result records, the runner, regression comparison
  cli.py               thin command-line entry point
tests/                 unit, contract and repository-hygiene tests
configs/               versioned configuration files
docs/                  documentation (see docs/README.md)
```

Only modules that exist appear here. `docs/architecture.md` describes the modules
that are planned and their status; a directory is created when it contains
something real, not in anticipation of it. Experiment records will live at
`experiments/<experiment-id>/record.md`, and that directory appears with the first
experiment.

## Contributing

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md). The short version: every
behavioural change needs a test, claims in documentation must match measured
reality, and a change in architecture needs a decision record.
