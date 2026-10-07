# Getting started

## What you can actually run

One thing: `chakaso chat`, a conversation shell that runs end to end.

**There is no language model in Chakaso.** The only implementation of the model
boundary is a deterministic development double, which returns fixed text describing
the request it received. It does not answer questions, and the shell says so before
it produces anything. The reply is evidence that the conversation boundary works —
context projection, the model call, validation, evidence and citation recording —
not an answer.

There is also no retrieval, no fetching, no correction loop, no tokenizer and no
training code. `chakaso config show` inspects configuration, and `chakaso --version`
verifies an install.

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
  ([ADR-0009](decisions/ADR-0009-unresolved-references-are-recorded.md)). Nothing can
  produce evidence yet, so this should only be reachable with a model that invents
  identifiers.

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
  conversation/        immutable conversation state
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
