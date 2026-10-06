# Contributing to Chakaso

Chakaso is a research project. The most valuable contributions are the ones that
make a claim in this repository either verifiable or false.

## What is most useful

- **A measurement.** A benchmark, a baseline, a metric implementation.
- **A counterexample.** A case where the architecture's stated behaviour does not
  hold: an answer that cites evidence it does not use, a follow-up that fails to
  resolve, a correction that does not happen.
- **A negative result.** An approach that did not work, with the configuration
  that produced it. This is a result and belongs in the research log.
- **A correction to documentation.** If a document describes behaviour the code
  does not have, that is a defect in the documentation, not a detail.
- **An implementation unit** with tests, matching an interface that already
  exists.

Less useful, though not unwelcome: stylistic refactors of code that has not yet
been exercised. There is a lot of unbuilt system and very little built system, and
polish on the small part is not where the leverage is.

## Ground rules

**Do not describe planned work as done.** The status vocabulary is defined in
[`docs/README.md`](docs/README.md). A pull request that implements a component
should update that component's status in the documentation in the same change.

**Do not invent numbers.** No benchmark result, loss curve, latency figure or
percentage appears in this repository unless it was measured, and it is reported
with the configuration, versions and seed that produced it.

**Do not weaken a test to make it pass.** If a test fails, either the code is
wrong or the test's expectation was wrong. Both are fine outcomes; deleting the
assertion is not.

**Documentation tracks code in the same commit.** If a change alters an interface,
the specification, the ADR (if a decision changed) and
[`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md) are updated together
with it.

**A new dependency needs a reason.** Check the standard library first. State what
the dependency does that the standard library does not, and keep heavy components
out of the minimal test environment.

## Development setup

See [`docs/getting-started.md`](docs/getting-started.md). The short version:

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Before opening a pull request, all of these must pass:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

CI runs the same commands. If you cannot run them locally, say so in the pull
request rather than assuming.

## What a change should contain

| Change | Required |
| --- | --- |
| Bug fix | A regression test that fails before the fix |
| New component | Tests, and a status update in the docs |
| Changed interface | Updated specification, updated contract tests, updated status |
| New architectural decision | A record in [`docs/decisions/`](docs/decisions/) |
| New dependency | Justification in the pull request, and a note in [`docs/internal/engineering/tooling.md`](docs/internal/engineering/tooling.md) if it is a tool |
| Experiment or benchmark | An entry in [`docs/research/research-log.md`](docs/research/research-log.md) and an experiment record |

## Commits

Commits are small and coherent. One commit is one idea that can be understood
without reading the next one. The convention is a type prefix and a direct
description:

```text
feat: add evidence pack identifier validation
fix: reject evidence identifiers absent from the pack
test: cover citation resolution with unknown identifiers
docs: record the decision to use TOML configuration
refactor: isolate configuration loading from validation
chore: pin the test dependency set
```

Do not split one change into several commits to look busy, and do not merge
unrelated changes into one commit to look efficient. History is part of the
documentation; somebody will read it to understand why the code is shaped the way
it is.

## Writing style

Documentation in this repository is written by people, for people who want to know
what is true. Write directly and technically:

- Say what the thing does, where it is implemented, and what it does not do.
- Prefer a specific sentence to a general one.
- Avoid hype. "Cutting-edge", "revolutionary", "game-changing" and "seamless" mean
  nothing here and will be removed.
- Avoid corporate phrasing that implies more maturity than exists.
- Where something is unknown, write that it is unknown.

Consistency matters more than any individual document's polish. The repository
should read as though one person wrote it.

## Code expectations

- Type annotations on public functions and on data that crosses a boundary.
- Comments explain *why*, not *what*. Code that needs a comment to explain what it
  does should usually be rewritten.
- No bare `except Exception` without a specific reason, and never a silent one.
  Failures must be diagnosable.
- Configuration is explicit. No magic constants in component code.
- Retrieved external content is untrusted data. It never becomes an instruction.

## Review

Reviews look for: does this do what it says, is it tested, does the documentation
match, is there a simpler way, and does it introduce a claim the project cannot
support. A pull request that is honest about its limitations will be merged faster
than one that is not.

## License

Contributions are accepted under the Apache License 2.0, the same terms as the
project ([`LICENSE`](LICENSE), [ADR-0004](docs/decisions/ADR-0004-apache-2.0-license.md)).
By contributing you confirm you have the right to submit the work under those
terms. Third-party code must be license-compatible and its origin recorded.
