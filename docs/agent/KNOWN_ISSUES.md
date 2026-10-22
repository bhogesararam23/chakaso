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

### K-009 — The model path is in CI's definition; the first runner verdict is unread

**Impact:** Medium, and it is a coverage gap rather than a correctness one — smaller than it was, because the
gap is now "unverified on a runner" rather than "not covered at all".

`chakaso.training` splits across two optional extras: `numpy` in `[train]` (ADR-0030) and `torch` in `[nn]`
(ADR-0034). Until 2026-10-22 the CI matrix installed only `.[dev]`, so the numerical and model tests
`pytest.importorskip("numpy")` / `("torch")` and **skipped** there, and `mypy src` excluded the modules that
import them — the model path, which is the product, ran only on a maintainer's machine. That is fixed in the
workflow: three jobs now install `.[dev]`, `.[dev,train]` and `.[dev,train,nn]`. What is not fixed is the
part a repository cannot fix by itself — no result has been read back from Actions for the new jobs, so the
claim standing is "CI is defined to test the model path", not "CI has tested it".

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
selection, both fixed, and `configs/mypy-model-path.toml` is now the command that checks all 147 source files
under the same strict settings — clean where both extras are installed, and an honest `import-not-found`
failure where torch is absent. Two hygiene tests keep the two configurations' strictness identical and pin
the exclusion list to the modules whose dependency it names.

**Re-verified again (2026-10-21, P33–P36).** The token's scopes were `gist, read:org, repo` — no
`workflow` — so a push that touches `.github/workflows/*` was refused (GitHub refuses the *whole* push,
naming the file, which is why the prepared matrix was delivered as a document rather than committed as a
workflow: it would block every other commit in the same push). Checked while at it: `main` is **not**
protected, and Actions is enabled with `allowed_actions: all` — the missing scope was the only blocker.

**The scope blocker is gone (measured 2026-10-22).** `gh auth status` now reports `gist, read:org, repo,
workflow`, so this entry's premise changed under the documentation that repeated it. Every statement above this
paragraph was true of the credential that existed when it was written and is kept as dated history; the earlier
sessions' refusals were real, and the prepared matrix was written against a token that could not apply it. What
still has not happened is the thing that actually closes K-009: a workflow file carrying the `[train]` and `[nn]`
tiers **pushed and read back from the runner**. The scope is evidence that it is now possible, not evidence that
it passed — and applying the matrix still needs the decision ADR-0034 left open, whether a runner should fetch
the torch CPU wheel index on every push.

Locally all three matrices still run, and the counts grew with the new tests: **1,644 pass / 23 skip**
dependency-free, **1,724 pass / 14 skip** with `[train]`, and **2,001 pass / 1 skip** with `[train,nn]`;
strict typing is clean over all 157 source files in the `[nn]` environment (and over the 125 the
dependency-free `mypy src` reaches). Measured on 2026-10-22 against the working tree that shipped the matrix;
the figures for the delivered head are restated in the delivery ledger.

**The matrix is now the workflow, not a document about one (2026-10-22).**
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) carries three jobs — `checks` on
3.11/3.12/3.13 with `.[dev]`, `train` on the same three with `.[dev,train]`, and `model-path` on the floor and
the latest with `.[dev,train,nn]` from the CPU wheel index — and
[`docs/internal/engineering/optional-dependency-ci-matrix.md`](../internal/engineering/optional-dependency-ci-matrix.md)
describes that file instead of replacing it. Five hygiene tests read the shipped file rather than a copy: the
jobs and their installed extras match `pyproject.toml`, every `run:` command is one the contributor contract
documents, nothing can fail quietly, `src/chakaso/serve/` imports neither `numpy` nor `torch` (so the
dependency-free tier is genuinely coverage of the HTTP surface), and the document is refused while it still
describes the matrix as un-applied. The assertion that used to guard the *blocked* state — "the shipped
workflow installs only `.[dev]`" — was deleted, as its own text said it should be, and the test that replaces
it fails if the document falls behind the file.

**What still keeps this entry open:** the first runner result has not been read. A workflow file in the
repository is not evidence that Actions executed it; the `[nn]` job downloads a torch wheel on a GitHub host
for the first time, and its outcome is unknown until `gh run list` says otherwise. Recording the run
identifiers and their verdicts is what closes K-009 — and ADR-0034's open question, whether a runner should
fetch the CPU wheel index on every push, is now answered in the affirmative by the file itself, which should
be revisited if the download time becomes the reason a contributor's check is slow.

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
