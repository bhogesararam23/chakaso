# The optional-dependency CI matrix, as shipped

**Status: shipped, and read back once.** The three tiers this document used to describe as prepared-and-
unappliable are now the content of
[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml). The first push of that file produced run
[`39702923101`](https://github.com/bhogesararam23/chakaso/actions/runs/39702923101): all eight jobs reached
GitHub Actions, both `model-path` jobs installed the PyTorch CPU wheel on a Linux host for the first time, and
every one of the eight then failed the same step — `mypy src` on `core/process_memory.py`, which reached for
`ctypes.WinDLL` outside any platform guard and so described a Windows-only name to a Linux runner. Nothing in
this document claims the matrix passes: the verdict on the repaired code belongs to the next run, recorded
when it is read. What changed on 2026-10-22 is that the model path is now something the runners are asked
about, and they answered with a real defect on their first attempt.

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
torch, plus a strict type check over every source file in the `[nn]` environment. The local counts at the
time of writing are measured and recorded in [K-009](../../agent/KNOWN_ISSUES.md); the runner counts will
differ slightly because the working tree moves, and they will be read from Actions rather than assumed.

It does not prove the model is good at anything. **CI verifies the code path; it does not train a model, and
no experiment in this repository becomes a measured result by running in CI.** The experiment records stay
where they are, with their own per-run timings and fingerprints, and
[K-001](../../agent/KNOWN_ISSUES.md) stays open for the reason it is open.

## History: why this took until 2026-10-22

The matrix was designed on 2026-10-21 and could not be delivered: the session's GitHub credential then had
scopes `gist, read:org, repo`, and GitHub refuses a push that creates or updates any file under
`.github/workflows/` — the *whole* push, naming the file — which would have blocked every other commit in the
same push. It was therefore committed as a document. The credential's scopes were re-checked on 2026-10-22 and
now include `workflow`; the file is the delivery, and [K-009](../../agent/KNOWN_ISSUES.md) records both the
change and the remaining gap until the first run is read.
