# Known issues

Real problems with the current state, including the ones that are only
inconvenient. An empty file here would mean nobody has looked.

Last reviewed: 2026-10-21.

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

**Narrowed again, still open, on 2026-10-19 (P27/E2603).** There is now a grounded-answer *evaluation
workload* and a runner that scores the real retrieval + answer boundaries (`chakaso benchmark
grounded`, ADR-0031), so the pipeline can finally be pointed at a question and measured. But the
workload is self-authored and tiny, and E2603 scored it through an *oracle* and a non-answering
provider — not a model that answers. No real corpus and no useful answering model exist, so the
underlying gap is unchanged: nothing has been executed against a real workload by a model.

**Narrowed again, still open, on 2026-10-21 (P31/E2605).** There is now a trained Transformer behind the
`LanguageModel` boundary, measured on a real corpus — 71 documents and 780,223 characters of this
repository's own prose, pinned by content digest — and it beats both trivial baselines and the project's
previous architecture on held-out documents (1.8657 against 3.1343 bits per character). So "no model has ever
been trained or measured" is no longer true. What is still true is the stronger claim in the title: that
corpus is one author's technical writing about this project, no question was asked of the model, no answer was
grounded in retrieved evidence, and no turn used it. The gap K-001 names is a *workload*, and a measured
language model on self-authored text is not one.

**What would close it:** a run against a real corpus or a useful model that produces and grounds an
answer. A local retrieval implementation, a synthetic development benchmark, the evaluation / correction
foundations, a measured from-scratch training loop and a trained subword Transformer that loads from Python
all exist and are exercised in tests or in a committed experiment; none of them is a real workload with a real
answer, so the underlying gap remains.

**Not a defect to fix now.** It is recorded so that no document reads as though the
architecture has been validated.

### K-010 — Lexical retrieval returns its top-k even when nothing is relevant

**Impact:** Medium, measured by E2603; **partially addressed by P28 (ADR-0032, E2604)** — a decision now
exists, but retrieval itself is unchanged and the runtime does not yet consume it.

Discovered by E2603 (the grounded-answer workload run): `RetrievalService.search` always returns up to
`top_k` BM25-ranked chunks, including for questions whose answer is absent from the corpus. Over the
held-out and full splits every abstention case produced a **false retrieval** (3 of 3), and 3 cases
surfaced a passage they had explicitly marked **forbidden**. A retrieval-aware runtime cannot abstain
correctly if the retriever hands it evidence unconditionally — the "no evidence" signal a grounded
abstention depends on is never produced.

**Why it was not caught before.** The retrieval development benchmark scored only cases that name gold
evidence; abstention cases were checked for "did retrieval return nothing," but the fixture set was
small and the failure did not aggregate visibly. The P27 workload made it measurable per category.

**What P28 changed.** `chakaso.retrieval.sufficiency` adds a deterministic, additive evidence-sufficiency
decision (a discriminating-term policy, ADR-0032). E2604 measured it at **100% accuracy on the workload —
all three unanswerable cases judged insufficient, zero answerable cases wrongly abstained** — while
leaving retrieval's candidate emission, its `false_retrievals`/`forbidden_hits` metrics, and the runtime
behaviour untouched (the signal is opt-in via `--sufficiency` and consumed only by the evaluation harness
today).

**What still does not close it.** (1) The runtime does not yet *act* on the decision — wiring the answer
path to abstain on insufficient evidence is P32. (2) The policy's correct operating band is narrow and
corpus-specific (`max_document_fraction ≈ 0.23–0.31` on 13 documents); whether a working threshold exists
at all on a real corpus is untested — that is P29. (3) `forbidden_hits` is a ranking/precision problem the
sufficiency decision does not address.

**What would close it fully:** the P32 runtime consumption plus a P29 demonstration that the policy
survives calibration on data it was not tuned on, with recall preserved and abstention measured on both
sides.

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

### K-009 — The training and model-path tests do not run in remote CI

**Impact:** Medium, and it is a coverage gap rather than a correctness one — but it is now the largest gap in
the repository, because the model path is the product.

`chakaso.training` splits across two optional extras: `numpy` in `[train]` (ADR-0030) and `torch` in `[nn]`
(ADR-0034). The core CI matrix installs only `.[dev]`, so the numerical and model tests
`pytest.importorskip("numpy")` / `("torch")` and **skip** there; `mypy src` excludes the modules that import
them in that matrix (their types resolve through the absent package). The tests and the strict type-check of
the training and model packages therefore run only where the extras are installed — locally, and by any
maintainer who runs `.[dev,train,nn]` — not on every push in CI.

**Why it is not closed in the obvious way.** The clean fix is dedicated CI jobs installing `.[dev,train]` and
`.[dev,train,nn]` and running the full suite plus `mypy src --exclude '_never_match_'`; the commands exist but
the workflow **cannot be pushed** because the session's GitHub credentials are an OAuth app without the
`workflow` scope, and GitHub refuses to let it create or update `.github/workflows/*`. ADR-0030's testing
strategy anticipated that job and ADR-0034 inherited the gap. Delivering it needs a token with the `workflow`
scope (or a human applying the job change), and a `torch` wheel download in CI, which the offline-CI rule
wants reviewed first.

**Mitigation in place now:** the dependency-free parts of both layers — corpus, character tokenizer, split,
config, baselines, and now the corpus manifest, document split, byte-pair subword tokenizer, sequence packing
and model metrics — run in the core matrix as ordinary tests. Only the numpy- and torch-requiring modules
skip. The torch import surface is confined by a hygiene test to a named module list, so the gap is bounded
and visible rather than spread through the package.

**Type checking had the same gap, and it is now closed.** `pyproject.toml` excludes the optional-dependency
modules from `mypy src`, and the previously documented escape — `mypy src --exclude '_never_match_'` — does
not work: a command-line exclude is *appended* to the configured list, and directory arguments are subject to
it as well. So the `[train]` and `[nn]` path (27 modules) had never been type-checked anywhere. Checking it
properly found three dead `# type: ignore` comments and one real typing hole in the byte-pair encoder's pair
selection, both fixed, and `configs/mypy-model-path.toml` is now the command that checks all 142 source files
under the same strict settings — clean where both extras are installed, and an honest `import-not-found`
failure where torch is absent. Two hygiene tests keep the two configurations' strictness identical and pin
the exclusion list to the modules whose dependency it names.

**Re-verified again (2026-10-21, P33–P36).** The token's scopes are still `gist, read:org, repo` — no
`workflow` — so a push that touches `.github/workflows/*` is still refused (GitHub refuses the *whole* push,
naming the file, which is why the prepared matrix is delivered as a document rather than committed as a
workflow: it would block every other commit in the same push). Checked while at it: `main` is **not**
protected, and Actions is enabled with `allowed_actions: all` — the missing scope is the only blocker.

Locally all three matrices still run, and the counts grew with the new tests: **1,479 pass / 21 skip**
dependency-free, **1,560 pass / 12 skip** with `[train]`, and **1,739 pass / 1 skip** with `[train,nn]`;
strict typing is clean over all 147 source files in the `[nn]` environment (and over the 118 the
dependency-free `mypy src` reaches). Measured on 2026-10-21 against this working tree; the figures for the
delivered head are restated in the delivery ledger.

**The fix is now written and partly tested, which is new.**
[`docs/internal/engineering/optional-dependency-ci-matrix.md`](../internal/engineering/optional-dependency-ci-matrix.md)
carries a complete three-tier workflow — core on 3.11/3.12/3.13, `[dev,train]` on the same three, and
`[dev,train,nn]` on the floor and the latest with the CPU wheel index ADR-0034 names — ready to paste into
`.github/workflows/ci.yml` by anyone holding a `workflow`-scoped token. Four tests in
`tests/test_repository_hygiene.py` keep it from being decoration: the jobs install exactly the tiers they
claim, every `run:` command is one `AGENTS.md` and `CONTRIBUTING.md` document, nothing can fail quietly
(no `continue-on-error`, no deselection), and the **shipped** workflow still installs only `.[dev]` — that
last assertion is written to fail the moment someone applies the matrix, which is exactly when this entry
should be rewritten rather than left to rot. None of it has run on a runner, and no claim is made that it
will pass there: the `[nn]` job would download a torch wheel on a GitHub host for the first time.

**What would close it:** a CI workflow change (jobs installing `.[dev,train]` and `.[dev,train,nn]`) made with
a credential that has the `workflow` scope, plus a decision on fetching the CPU wheel index in CI.

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
