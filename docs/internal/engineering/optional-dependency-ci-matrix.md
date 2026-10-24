# The optional-dependency CI matrix, as shipped

**Status: shipped, and read back green.** The three tiers this document used to describe as
prepared-and-unappliable are now the content of
[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml). Landing them took three runs, and each of the
two failures was a defect the old single-tier CI had no way to see:

| Run | Result | What it showed |
| --- | --- | --- |
| [`39702923101`](https://github.com/bhogesararam23/chakaso/actions/runs/39702923101) | 8 jobs, all failed `Types` | `mypy src` on `core/process_memory.py`, which reached for `ctypes.WinDLL` outside any platform guard — a Windows-only name described to a Linux runner, passed by every gate on the author's machine. Fixed in `5e1b641`. |
| [`39703950809`](https://github.com/bhogesararam23/chakaso/actions/runs/39703950809) | 7 jobs green, `model path 3.13` failed `Tests` | batched and unbatched scoring compared to nine decimals, finer than the float32 the arithmetic happens in. The expectation was the defect. Fixed in `b99ffee`. |
| [`39704678913`](https://github.com/bhogesararam23/chakaso/actions/runs/39704678913) | **all 8 jobs success** | core 3.13: 1,644 passed / 24 skipped in 17.39 s. model path 3.13: 2,002 passed / 2 skipped in 43.53 s. |
| [`39706259071`](https://github.com/bhogesararam23/chakaso/actions/runs/39706259071) | **all 8 jobs success** | the then-current head: core 3.13 1,644 / 24 in 17.39 s, model path 3.13 2,003 / 2 in 79.18 s, model-path jobs taking 1m58s–2m31s including the torch download. |
| [`40119588460`](https://github.com/bhogesararam23/chakaso/actions/runs/40119588460) | **all 8 jobs success** | the head after the 2026-10-24 documentation audit (`ad7ccc6..733b628`, seven commits: three architecture rules given the tests their own document claimed, four measured numbers corrected against their records, every public document reconciled, and one false provenance claim removed from `metadata.display_name`): core 3.13 **1,655 passed / 25 skipped** in 18.23 s, train 3.13 **1,736 / 16** in 21.84 s, model path 3.13 **2,028 / 2** in 87.53 s. That the audit's own changes were checked by the tiers they depend on is the point of this row. |
| [`40120242998`](https://github.com/bhogesararam23/chakaso/actions/runs/40120242998) | **all 8 jobs success** | the ledger-and-evidence commit on top of it (`733b628..4d6c2f0`), whose only difference from the previous run is documentation: the same counts — core 3.13 1,655 / 25, train 3.13 1,736 / 16, model path 3.13 2,028 / 2 — and a model-path wall clock of 48.48 s against 87.53 s an hour earlier. Recorded as an observation rather than a finding: two runs of identical test code disagreeing by nearly 2× on time is why no record in this repository presents a wall clock as a property of a model, and why the tier's *counts* are the thing worth citing. |

The runner reports one more skip than the same head does locally, and the difference is accounted for rather
than smoothed: the hygiene test that checks the private documentation pack is untracked skips where that pack
does not exist at all.

## What each tier is for

| Job | Python | Installs | Runs |
| --- | --- | --- | --- |
| `checks` | 3.11, 3.12, 3.13 | `.[dev]` | `ruff format --check .`, `ruff check .`, `mypy src`, `pytest`, entry-point smoke |
| `train` | 3.11, 3.12, 3.13 | `.[dev,train]` | `pytest`, `mypy src` |
| `model-path` | 3.11, 3.13 | `.[dev,train,nn]` from the CPU wheel index | `pytest`, `mypy --config-file configs/mypy-model-path.toml` |

Why this shape and not more:

- **`checks` is the dependency-free guarantee, and it is not a smoke test.** `numpy` and `torch` are absent
  there, so the numerical and Transformer modules skip — that skipping is the point being tested: a claim that
  `import chakaso` needs nothing is only verified where nothing is installed. It is also where the retrieval,
  conversation, CLI and **HTTP serving** surfaces run, because `chakaso.serve` is standard library by decision
  ([ADR-0038](../../decisions/ADR-0038-local-http-serving-adapter-from-the-standard-library.md)) and a hygiene
  test refuses a serving module that imports a numerical library.
- **`model-path` runs on the floor and the latest, not all three.** `torch` is a ~200 MB dependency and the
  failure modes it can introduce across minor Python versions are a wheel that does not exist or an operator
  signature that changed; both show up at an end of the supported range. Running it on 3.12 as well would
  triple the download for a case the two ends already cover. ADR-0034's `<3` bound is what this job tests.
- **Versions come from `pyproject.toml`, not from CI.** Each job installs the extras as the package declares
  them (`numpy>=1.26,<2.5`, `torch>=2.6,<3`), so CI tests what a user gets rather than a pinned snapshot that
  can rot. The reproducibility policy in [ADR-0030](../../decisions/ADR-0030-minimal-optional-training-dependency.md)
  is bounds, and a CI-only pin would be a second, untested dependency set.
- **CPU wheels come from the index ADR-0034 names.** PyPI's default `torch` wheel is the CUDA build on Linux.
- **Nothing is allowed to fail quietly.** No `continue-on-error`, no `|| true`, no marker that deselects a
  real test. `pytest -m 'not network'` is already the default in `pyproject.toml`, so these jobs run the
  documented commands, not a weaker subset of them.

## How the workflow is kept honest

Five tests in [`tests/test_repository_hygiene.py`](../../../tests/test_repository_hygiene.py) read the shipped
file rather than a copy of it:

- the jobs are exactly `checks`, `train`, `model-path`, and each installs exactly the extras its name claims,
  checked against `pyproject.toml` so a renamed extra fails loudly;
- every `run: python -m …` command in `checks` is one that `AGENTS.md` and `CONTRIBUTING.md` tell a
  contributor to run, which is what stops CI from quietly becoming a weaker contract;
- nothing in the file can hide a failure;
- `src/chakaso/serve/` imports neither `numpy` nor `torch`, so the dependency-free tier really is coverage of
  the serving adapter;
- this document names every job that is shipped, and is refused while it still describes the matrix as
  un-applied — the assertion that used to guard the opposite state is now the one that keeps this page from
  going stale.

## What this buys, and what it still does not prove

Applied, the matrix runs 3 + 3 + 2 = 8 jobs: the dependency-free suite, the suite with numpy, the suite with
torch, plus a strict type check over every source file in the `[nn]` environment. The local counts are
measured and recorded in [K-009](../../agent/KNOWN_ISSUES.md), and the table above carries the counts the
runners themselves reported, so the two are not conflated.

It does not prove the model is good at anything. **CI verifies the code path; it does not train a model, and
no experiment in this repository becomes a measured result by running in CI.** The experiment records stay
where they are, with their own per-run timings and fingerprints, and
[K-001](../../agent/KNOWN_ISSUES.md) stays open for the reason it is open.

## History: why this took until 2026-10-22

The matrix was designed on 2026-10-21 and could not be delivered: the session's GitHub credential then had
scopes `gist, read:org, repo`, and GitHub refuses a push that creates or updates any file under
`.github/workflows/` — the *whole* push, naming the file — which would have blocked every other commit in the
same push. It was therefore committed as a document. The credential's scopes were re-checked on 2026-10-22 and
now include `workflow`; the file is the delivery, and [K-009](../../agent/KNOWN_ISSUES.md) is closed with the
run identifiers above rather than left as a promise about a future one.
