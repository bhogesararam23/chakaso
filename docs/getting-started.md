# Getting started

## What you can actually run

Several things, all local and offline: `chakaso chat`, a conversation shell that exercises the
conversation plumbing end to end; `chakaso retrieve`, which ingests named local documents and runs
lexical retrieval over them; `chakaso benchmark retrieval` and `chakaso benchmark correction`, which
run the development benchmarks and print their reports; `chakaso experiment run <subject>`, which
produces a reproducible experiment result; `chakaso storage check` / `chakaso storage migrate`, which
inspect or initialize a durable answer store; and `chakaso config show` / `chakaso --version`.

**There is no language model in Chakaso.** The only implementation of the model
boundary is a deterministic development double, which returns fixed text describing
the request it received. It does not answer questions, and the shell says so before
it produces anything. The reply is evidence that the conversation boundary works —
context projection, the model call, validation, evidence and citation recording —
not an answer.

Retrieval exists but is **local and lexical**: it reads the files you name, normalizes
and chunks them, and ranks them for a query with BM25 — shared words only, no embeddings
and no meaning. Below it, claim/citation/grounding evaluation, a correction decision rule with
reassessment, an answer store (in-memory and, since ADR-0021, a durable SQLite backend that
survives a process restart) and reproducible experiment records exist as tested code —
but there is still no language model, no web fetching on any default path (the fetcher is opt-in),
no automatic correction loop, no tokenizer and no training code. The benchmark and experiment
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
- No GPU. Nothing in the test suite requires one, and CI does not have one.

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
inference or search provider.

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

The shell reads messages from standard input and prints replies to standard output.
Two commands end it: `:quit` and `:exit`, and end of input works too. Blank lines are
ignored rather than sent. A failed turn is reported and the session continues, because
a failure leaves the conversation unchanged
([ADR-0008](decisions/ADR-0008-transactional-turns.md)).

To send one message and exit, which is what a script wants:

```bash
python -m chakaso chat --message "What does the specification say?"
```

Configuration is chosen the same way `config show` chooses it:

```bash
python -m chakaso chat --config configs/default.toml
```

### Read this before believing a reply

The shell prints a notice on **standard error** before the first reply:

```
engine: deterministic (Deterministic double (not a language model))
This is a development double, not a language model. It returns fixed text that
exercises the conversation plumbing, and it does not answer anything.
No retrieval, no citations, no correction, and no confidence. The conversation is
not saved and is lost when this process exits.
```

That is not a disclaimer bolted on at the end. The `development_double` flag is part
of the model's own metadata, so the notice is derived from what the model says about
itself rather than from a hard-coded string that could drift out of date.

The reply goes to standard output and everything else to standard error, so
`--message` stays usable in a pipeline. Scripting it is fine; reading its output as
an answer is not.

Two more things the shell will tell you about, on standard error, when they happen:

- **A truncated reply.** Generation stopped at the configured length ceiling, so the
  reply is incomplete. A caller that cannot tell that from a finished answer would
  present one as the other.
- **An unresolved evidence reference.** The reply cited evidence it was not given.
  Nothing was resolved to a source, and the reference is listed
  ([ADR-0009](decisions/ADR-0009-unresolved-references-are-recorded.md)). The shell
  supplies no evidence, so this is only reachable with a model that invents identifiers.

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
  core/                identifiers, hashing, error base
  models/              language-model boundary, capabilities, registry
  evidence/            source records, evidence chunks, packs, citation resolution
  retrieval/           normalization, chunking, ingestion, corpus, lexical index and retrieval
  conversation/        immutable conversation state, conversation manager
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
