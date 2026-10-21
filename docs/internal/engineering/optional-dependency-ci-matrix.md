# The optional-dependency CI matrix, prepared and why it is not running

**Status: designed, written, locally validated, and blocked from delivery.** The workflow below is complete
and ready to apply. It has not run on GitHub Actions, and nothing in this document claims it has —
[ADR-0034](../../decisions/ADR-0034-optional-pytorch-model-extra.md) and K-009 record why.

## What CI covers today, measured

`origin/main` at the time of writing runs
[`ci.yml`](https://github.com/bhogesararam23/chakaso/actions/workflows/ci.yml) with one job on Python 3.11,
3.12 and 3.13, installing `.[dev]` only. That is the dependency-free matrix: `numpy` and `torch` are absent,
so the numerical and model-path tests skip and `mypy src` excludes the modules that import them. The last
three runs on `main` — `39544899511`, `39544651281`, `39534871321` — each completed success on those three
Pythons, and each of them ran the same 16 skips. **The model path — the product — has never been tested on a
GitHub runner.** Everything that is green about it is green locally.

## Why the workflow file is not simply committed

The session's GitHub credential has scopes `gist, read:org, repo` and not `workflow`. GitHub refuses a push
that creates or updates any file under `.github/workflows/`, naming the file, and it refuses the *whole*
push — so committing the change here would block every other commit in the same push. Verified again on
2026-10-21 with `gh auth status`; the branch itself is not protected, and Actions is enabled with
`allowed_actions: all`, so the only missing thing is that one scope.

What a person with a `workflow`-scoped token has to do is copy the block below into
`.github/workflows/ci.yml` — the rest of the file is unchanged — and push. Two tests in
`tests/test_repository_hygiene.py` keep this document honest: one asserts that the jobs, install commands
and run commands in the block are what the tiers documented in `AGENTS.md` say they are, and one asserts the
block contains no `continue-on-error` and no skipped-assertion escape hatch. They do not claim it ran.

## The matrix

Three tiers, because that is what the package actually declares — the base install, `[train]` and `[nn]` —
and each job installs exactly one of them.

| Job | Python | Installs | Runs |
| --- | --- | --- | --- |
| `checks` (existing) | 3.11, 3.12, 3.13 | `.[dev]` | `ruff format --check .`, `ruff check .`, `mypy src`, `pytest`, entry-point smoke |
| `train` | 3.11, 3.12, 3.13 | `.[dev,train]` | `pytest`, `mypy src` |
| `model-path` | 3.11, 3.13 | `.[dev,train,nn]` from the CPU wheel index | `pytest`, `mypy --config-file configs/mypy-model-path.toml` |

Why this shape and not more:

- **`model-path` runs on the floor and the latest, not all three.** `torch` is a ~200 MB dependency and the
  failure mode it can introduce across minor Python versions is a wheel that does not exist or an operator
  signature that changed; both show up at an end of the supported range. Running it on 3.12 as well would
  triple the download for a case the two ends already cover. ADR-0034's `<3` bound is what this job tests.
- **`checks` is unchanged and stays free of `torch`.** The dependency-free matrix is the guarantee that
  `import chakaso` needs nothing, and that claim is only tested if some job installs nothing but `.[dev]`.
- **Versions come from `pyproject.toml`, not from CI.** Each job installs the extras as the package declares
  them (`numpy>=1.26,<2.5`, `torch>=2.6,<3`), so CI tests what a user gets rather than a pinned snapshot that
  can rot. The reproducibility policy in [ADR-0030](../../decisions/ADR-0030-minimal-optional-training-dependency.md)
  is bounds, and a CI-only pin would be a second, untested dependency set.
- **CPU wheels come from the index ADR-0034 names.** PyPI's default `torch` wheel is the CUDA build on Linux.
- **Nothing is allowed to fail quietly.** No `continue-on-error`, no `|| true`, no marker that deselects a
  real test. `pytest -m 'not network'` is already the default in `pyproject.toml`, so these jobs are the
  documented commands, not a weaker subset of them.

## The workflow block

```yaml
jobs:
  checks:
    name: core python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"
      - name: Format
        run: python -m ruff format --check .
      - name: Lint
        run: python -m ruff check .
      - name: Types
        run: python -m mypy src
      - name: Tests
        run: python -m pytest
      - name: Verify the installed entry point
        run: |
          python -m chakaso --version
          chakaso info

  train:
    name: train python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev,train]"
      - name: Types
        run: python -m mypy src
      - name: Tests
        run: python -m pytest

  model-path:
    name: model path python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.13"]
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev,train,nn]" --extra-index-url https://download.pytorch.org/whl/cpu
      - name: Types
        run: python -m mypy --config-file configs/mypy-model-path.toml
      - name: Tests
        run: python -m pytest
```

## What this buys, and what it still does not prove

Applied, the matrix would run 3 + 3 + 2 = 8 jobs: the 1,431 dependency-free tests, the 1,472 that include
numpy, and the 1,605 that include torch, plus a strict type check over all 145 source files in the
`[nn]` environment. Locally those are the numbers this session measured; on runners they would be measured
for the first time, and the counts will differ slightly because the working tree moves.

It still does not prove the model is good at anything. **CI verifies the code path; it does not train a
model, and no experiment in this repository becomes a measured result by running in CI.** The experiment
records stay where they are, with their own per-run timings and fingerprints, and
[K-001](../KNOWN_ISSUES.md) stays open for the reason it is open.
