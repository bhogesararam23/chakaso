# Known issues

Real problems with the current state, including the ones that are only
inconvenient. An empty file here would mean nobody has looked.

Last reviewed: 2026-10-19.

## Open

### K-001 — Nothing in this repository has been executed against a real workload

**Impact:** High, and it is the honest summary of the project's position.

The architecture is a specification. It has never handled a live web page, a real
question, or a useful model. Decisions recorded in ADRs 0001–0007 rest on reasoning, not
on measurement, and some of them may turn out to be wrong when they meet data.
The contract in ADR-0003 in particular assumes that models can be made to reference
opaque evidence identifiers reliably; that assumption is untested.

**Narrowed, not closed, on 2026-10-19.** E2601 trained a real model from random initialization and
measured it against baselines, so the *training loop* is no longer unexecuted — but that model is a
tiny synthetic-corpus character experiment, not a useful language model, and nothing has met a real
question or a real corpus. The stronger form of the gap stands: no real workload, no grounded answer,
no measured model quality.

**What would close it:** a run against a real corpus or a useful model that produces and grounds an
answer. A local retrieval implementation, a synthetic development benchmark, the evaluation / correction
foundations, and now a measured from-scratch training loop all exist and are exercised in tests or in a
committed experiment; none of them is a real workload with a real answer, so the underlying gap remains.

**Not a defect to fix now.** It is recorded so that no document reads as though the
architecture has been validated.

### K-002 — Documentation describes components that do not exist

**Impact:** Medium.

Some components specified in `docs/architecture.md` still have no code — the query planner, a
dense ranker, persistence, semantic grounding and the revised answer a model would write. The
evaluation and correction *foundations* now have code, so this risk is narrower than it was, but a
reader skimming for mechanism rather than status can still take a Planned component for a real one.

**Mitigation in place:** status labels are mandatory and the vocabulary is
defined in `docs/README.md`; `CURRENT_STATE.md` lists the gaps explicitly.

**Residual risk:** accepted, because writing the specification before the code is
the point of this phase. It becomes a real problem if a status label is ever left
at Experimental after the code is deleted.

### K-003 — Private planning material lives inside the working tree

**Impact:** Low but severe if it fails.

`docs/*.docx` is the author's private pack, kept in place deliberately. It is
excluded by `.gitignore` and never moved, so `git status` does not mention it,
which also means a mistake here is invisible to a casual check. A single
`git add -f` would publish it.

**Mitigation:** `tests/test_repository_hygiene.py` fails if any tracked file
matches the private-pack patterns. This is the one repository rule enforced by a
test rather than by review.

### K-004 — The `docs/` directory mixes public documents with private ones

**Impact:** Low.

`docs/` holds public markdown and, alongside it, the git-ignored private pack.
The pack cannot be moved (the project owner requires it to stay), so the
separation is enforced by ignore rules and a test rather than by directory
structure.

### K-008 — The fetcher's SSRF screen is partial and fetching is not polite yet

**Impact:** Medium, and bounded by the fetcher being off every default path.

The fetcher (ADR-0012) refuses non-global address *literals*, but a hostname that resolves
to a private address is not caught by a pre-connect string check; closing that needs
connecting to a pre-resolved, re-validated IP inside the transport, which is not
implemented. There is also no robots.txt handling and no per-host rate limiting, so a
caller that enables fetching must do so in an environment it controls, against hosts it is
entitled to visit.

**Why it is not more urgent:** no default code path, CLI command or CI test reaches the
network. The limitation is recorded rather than oversold, and it is the reason fetching is
opt-in.

**What would close it:** IP-pinned resolution in the transport, plus a politeness layer
(robots, per-host spacing) before the fetcher is used against the live web.

### K-009 — The training tests do not run in remote CI

**Impact:** Medium, and it is a coverage gap rather than a correctness one.

`chakaso.training` depends on `numpy`, which is an optional `[train]` extra (ADR-0030). The core CI
matrix installs only `.[dev]`, so the numerical tests `pytest.importorskip("numpy")` and **skip** there;
`mypy src` also skips `chakaso.training` in that matrix (its type aliases resolve through numpy). The
tests and the strict type-check of the training package therefore run only where the extra is installed
(locally, and by any maintainer who runs `.[dev,train]`) — not on every push in CI.

**Why it is not closed in the obvious way.** The clean fix is a dedicated CI job that installs
`.[dev,train]` and runs the full suite and `mypy src`; the code for it exists but **cannot be pushed**
because the session's GitHub credentials are an OAuth app without the `workflow` scope, and GitHub
refuses to let it create or update `.github/workflows/*`. ADR-0030's testing strategy anticipates that
job; delivering it needs a token with the `workflow` scope (or a human applying the one-job change).

**Mitigation in place now:** the dependency-free parts of the training layer (corpus, tokenizer, split,
config, baselines) run in the core matrix as ordinary tests; only the numpy-requiring modules skip. The
numerics are verified locally and the whole training package is clean under strict `mypy` with numpy
present.

**What would close it:** a CI workflow change (one extra job installing `.[dev,train]`) made with a
credential that has the `workflow` scope.

### K-006 — The invented-reference condition is persisted; the rate is not yet aggregated

**Impact:** Lower than it was, and it is now a reporting gap rather than a data-loss gap.

The conversation manager detects a reference to evidence that was not supplied, never resolves
it to a source, and reports it in `Reply.citation_resolution`
([ADR-0009](../decisions/ADR-0009-unresolved-references-are-recorded.md)). Since
[ADR-0028](../decisions/ADR-0028-answer-evaluation-provenance.md) every recorded answer also
persists those unresolved and malformed references in its `evaluation_provenance`, so the
condition is no longer something that exists only in a reply that is already gone — it is a
durable fact on each answer.

**What still does not exist.** There is no aggregation step that turns the per-answer
`evaluation_provenance` across a durable history into the *rate* `docs/evaluation.md` names as a
headline measurement, and the facts are only persisted when a caller selects the durable store
(the default in-memory backend keeps nothing across processes). So the number is computable from
what the repository now keeps, but nothing computes it yet.

**Why an aggregation is not written speculatively.** A rate over a handful of development-double
replies — which do not invent references, because the double emits fixed text — would be a
measurement of nothing. The honest next step is to compute it over a real workload or a model
that actually generates references, which is K-001.

**What would close it:** a report over durable answer history that counts `has_unresolved_references`
against total answers, run against a real generating model.

## Resolved

### Retired — no package, no tests and no CI existed

Closed on 2026-10-07. The repository now has an installable typed package, a test
suite, and CI running formatting, linting, type checking and tests on Python 3.11,
3.12 and 3.13. The entry is kept rather than deleted because the risk it recorded
— that the absence of verification would stop being obvious — is the kind that
returns whenever CI is disabled for a while, and it is worth being able to point at
the last time it was true.

It originally carried the number K-002, which was reused by a live issue when this
list was renumbered. Resolved entries are now unnumbered: a closed issue does not
need to be referenced, and reusing its number for something else is a way to make
two different problems look like one.

### Retired — no fetch policy existed

Closed on 2026-10-08. A `FetchPolicy` now declares and validates the operational limits —
scheme allowlist, response-size cap, redirect bound, timeout and a header-safe user-agent —
and the fetcher enforces each, tested offline (ADR-0012). The security posture in
`docs/retrieval.md` is no longer a list of threats with no corresponding limits. What
remains — the DNS-rebinding gap in the destination screen, and politeness (robots, rate
limiting) — is carried forward honestly as K-008 rather than claimed as solved.

### Retired — answer records did not survive a process

Closed on 2026-10-10 (this was K-007). `chakaso.answers.sqlite.SQLiteAnswerStore` now persists the
append-only answer history to a local SQLite file behind the same `AnswerStore` boundary ADR-0018 set,
with an explicit schema version, transactional all-or-nothing appends, strict-read integrity checks,
and a test that writes an answer from a *separate process* and reads it back after restart (ADR-0021).
The default backend stays in-memory, so the runtime still writes nothing to disk unless a caller
selects the durable store. What this does **not** close: a `CorrectionRecord` or `ExperimentResult` is
still only serializable rather than auto-persisted, and simultaneous multi-process write concurrency
is deliberately neither built nor claimed.
