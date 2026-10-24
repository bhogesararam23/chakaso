# Known issues

Real problems with the current state, including the ones that are only
inconvenient. An empty file here would mean nobody has looked.

Last reviewed: 2026-10-24, as the whole-documentation audit of every doc against the code and the committed
records. K-011 and K-012 are that pass's findings turned into entries; the rest of its output was corrected
in the documents themselves.

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

**Narrowed again, still open, on 2026-10-24 (E2606/E2607/E2608 and the release).** Three things this entry said
did not exist now do. There is a *spread*: E2606 ran the same protocol over five pre-declared seeds
(1.8794 ± 0.0077 bits per character). There is a *real corpus* in the sense of text the project did not write:
E2607 measured on 105 documents of licensed public-domain prose and E2608 on 919 documents from 24 sources, with
a machine-readable catalogue, per-document digests and a measured data card. And there is a *released* artifact:
`chakaso-base@0.1.0`, loadable, verifiable, serveable, at Level 2 of
[ADR-0039](../decisions/ADR-0039-three-level-model-release-gate.md). None of that is this entry's title claim,
which is about a **workload**: a public-domain book is not a task anybody needed doing, no question was asked of
any of these models, no answer was grounded in retrieved evidence and checked, and Level 3 — the level that
would mean the model is good for something — is withheld on the measurement. E2608's samples are less degenerate
than anything before them and still not text anyone would publish, so the gap moves rather than closes: the
instruments are now real, and what is missing is still a use.

**What would close it:** a run against a real corpus or a useful model that produces and grounds an
answer. A local retrieval implementation, a synthetic development benchmark, the evaluation / correction
foundations, a measured from-scratch training loop with a seed spread and a multi-seed external-corpus study, a
trained subword Transformer and a rotary decoder that load from Python, and one released artifact at Level 2 all
exist and are exercised in tests or in a committed experiment; none of them is a real workload with a real
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

**Impact:** Medium, and narrower than when it was written.

Some components specified in `docs/architecture.md` still have no code: a real **semantic judge** (the
grounding boundary exists, its only implementation is a caller-supplied fixture — ADR-0019/ADR-0027), a genuine
**embedding model** (the dense boundary and its exact index exist over a deterministic non-semantic fixture —
ADR-0023), an **approximate/vector index**, a **durable source index** (the corpus is rebuilt in memory each
run), the **revised answer prose** a reassessment would write, an **autonomous correction loop** that decides
unbidden to reassess, and any model that could be called useful.

**Corrected on 2026-10-24.** The list this entry carried was written before those components were built and had
never been reconciled: it named the query planner, a dense ranker, persistence and semantic grounding as having
no code, and all four now do — `chakaso.planning` (ADR-0022, run by the runtime on every `chakaso chat` turn),
`chakaso.retrieval.dense`/`hybrid` over the fixture embedding (ADR-0023/ADR-0024, measured by E-001),
`chakaso.answers.SQLiteAnswerStore` surviving a process restart (ADR-0018/ADR-0021), and the semantic grounding
boundary with evaluator name, version and an explicit error outcome. An issue register that cites a stale
component list is worse than no register, because it looks like current knowledge: what remains true of the
original claim is the *shape* of the risk — a reader skimming for mechanism rather than status can take a
Planned component for a real one — and the honest form of the list is the one above.

**Mitigation in place:** status labels are mandatory and the vocabulary is
defined in `docs/README.md`; `CURRENT_STATE.md` lists the gaps explicitly.

**Residual risk:** accepted, because writing the specification before the code is
the point of this phase. It becomes a real problem if a status label is ever left
at Experimental after the code is deleted — and this entry is the recorded instance of the *reverse*, a gap
description left standing after the gap was closed.

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

### K-009 — Closed: the model path now runs in remote CI, and the runners found two defects

**Closed on 2026-10-22.** Read back from GitHub Actions rather than inferred: run
[`39704678913`](https://github.com/bhogesararam23/chakaso/actions/runs/39704678913) completed success with
all eight jobs — `core python 3.11/3.12/3.13` in 27–36 s, `train python 3.11/3.12/3.13` in 37–39 s, and
`model path python 3.11/3.13` in 2m14s and 1m30s. The runner's own counts, from its logs: core 3.13
**1,644 passed / 24 skipped in 16.92 s**, model path 3.13 **2,002 passed / 2 skipped in 43.53 s**. Getting
there took two fixes the previous single-tier CI could not have found:

- run `39702923101` (the first push of the new file) reached all eight jobs and failed the same step in every
  one: `mypy src` on a Windows-only `ctypes.WinDLL` in `core/process_memory.py`, which had passed every gate
  on the author's machine. Fixed in `5e1b641`.
- run `39703950809` then failed one test on `model path python 3.13`: batched and unbatched scoring were
  compared to nine decimal places, finer than the float32 the computation happens in. The expectation was the
  defect; the bound is now stated in units of that format. Fixed in `b99ffee`, after which `39704678913` went
  green.

The entry is kept with its number because roughly twenty documents cite it, including dated ones that should
not be rewritten. What it recorded was true and is now false: the model path is not locally-only.

**Impact (historical):** Medium, and it was a coverage gap rather than a correctness one — larger than it
sounded, because the model path is the product.

`chakaso.training` splits across two optional extras: `numpy` in `[train]` (ADR-0030) and `torch` in `[nn]`
(ADR-0034). Until 2026-10-22 the CI matrix installed only `.[dev]`, so the numerical and model tests
`pytest.importorskip("numpy")` / `("torch")` and **skipped** there, and `mypy src` excluded the modules that
import them — the model path, which is the product, ran only on a maintainer's machine.

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
selection, both fixed, and `configs/mypy-model-path.toml` is now the command that checks every module in the
package — the three the default configuration excludes included — under the same strict settings: clean where
both extras are installed, and an honest `import-not-found` failure where `torch` is absent. Two hygiene tests
keep the two configurations' strictness identical and pin the exclusion list to the modules whose dependency it
names. This paragraph used to carry a source-file count in it; the count was removed rather than updated,
because it is exactly the kind of number K-013 describes drifting, and `mypy` prints the live one on every run.

**Re-verified again (2026-10-21, P33–P36).** The token's scopes were `gist, read:org, repo` — no
`workflow` — so a push that touches `.github/workflows/*` was refused (GitHub refuses the *whole* push,
naming the file, which is why the prepared matrix was delivered as a document rather than committed as a
workflow: it would block every other commit in the same push). Checked while at it: `main` is **not**
protected, and Actions is enabled with `allowed_actions: all` — the missing scope was the only blocker.

**The scope blocker went (measured 2026-10-22).** `gh auth status` began reporting `gist, read:org, repo,
workflow`, so this entry's premise changed under the documentation that repeated it. Every statement above
this paragraph was true of the credential that existed when it was written and is kept as dated history; the
earlier sessions' refusals were real, and the prepared matrix was written against a token that could not apply
it. ADR-0034's open question — whether a runner should fetch the torch CPU wheel index on every push — is now
answered by the shipped file in the affirmative, and the measured cost is 1m30s–2m14s per model-path job.

Locally all three matrices still run, and the counts grew with the new tests: **1,644 pass / 23 skip**
dependency-free, **1,724 pass / 14 skip** with `[train]`, and **2,001 pass / 1 skip** with `[train,nn]`;
strict typing is clean over all 157 source files in the `[nn]` environment (and over the 125 the
dependency-free `mypy src` reaches). Measured on 2026-10-22 against the working tree that shipped the matrix;
the figures for the delivered head are restated in the delivery ledger. The same three tiers were then run by
GitHub's own runners, which is what closed this entry — and their counts differ from these by a handful,
which is expected when the tree moves between the two measurements.

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

**Why the gap mattered, in one line:** the two defects the new tiers found — a Windows-only attribute and an
assertion finer than the datatype — were both invisible to every gate the repository had ever run, on the
machine that wrote them. A check that runs only where the extras happen to be installed, and only on one
operating system, is not a check on the model path.

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

### K-011 — Three dependency-direction rules were claimed to be test-enforced and were not

**Impact:** Medium, and it is a credibility problem rather than a behaviour bug. Found by the
2026-10-24 documentation audit of the whole doc surface against the code.

`docs/architecture.md` stated that "six rules follow from the direction, and each is enforced by a test
rather than by review". Four of the six had nothing behind them — three in whole, one in part:

* *Model code does not import CLI code* — nothing checked it, in either direction, so a handler gaining
  library logic or a module gaining `from chakaso.cli import …` would have passed every gate.
* *No module may produce source identity* (ADR-0003) — true of the code, unenforced. The model path is
  exactly where the tempting shortcut lives: a model that mints an identifier for text it likes produces
  an answer no pack can resolve, and citation validation can only report it afterwards.
* *`numpy` stays inside the admitted training modules* — [ADR-0034](../decisions/ADR-0034-optional-pytorch-model-extra.md)'s
  isolation rule pointed at "the same way the `[train]` numpy isolation is" enforced. The torch half had a
  named list and two tests; the numpy half had prose. `import chakaso.models` pulling in numpy through a
  new module would have been invisible until a user on a minimal install hit an ImportError.
* *Model evaluation is separate from harness evaluation* — no check at all, and one remains open below.

**Fixed on 2026-10-24** by four checks in `tests/test_repository_hygiene.py` —
`test_no_library_module_imports_the_command_line_interface`,
`test_no_model_path_module_mints_source_identifiers`, and
`test_only_the_admitted_model_path_modules_import_numpy` with its staleness guard — and `docs/architecture.md`
and ADR-0030/ADR-0034 now name the enforcing test for each rule instead of claiming blanket enforcement.

**What still does not close it.** The sixth rule, *model evaluation is separate from harness evaluation*,
still has no mechanical enforcer. The two do live in separate modules and neither imports the other's
metrics, so a structural check on the imports is possible; what is not checkable that way is the reporting
rule — that a number in a record is labelled for what it actually measures. That half belongs with the
status labels and review, and `docs/architecture.md` now says so in the open rather than implying a test
exists.

**What would close it:** either an import-direction check between `chakaso.training.model_metrics` /
`chakaso.training.nn_evaluation` and `chakaso.evaluation` / `chakaso.benchmark`, or a recorded decision that
this rule is a reporting convention and the sentence in `docs/architecture.md` should stop listing it among
the tested rules.

### K-012 — The model-facing evidence block is not versioned, so a change to it is not attributable

**Impact:** Medium for grounded-evaluation work; low today, because no model consumes the block usefully.

Design rule 6 of `docs/architecture.md` says prompts and templates are versioned "because a prompt change
changes measured behaviour and must be attributable in an evaluation record". The template half holds:
`chakaso.models.formatting` defines `chakaso-plain-v1` and `chakaso-role-labelled-v1`, an adapter is built
with one named by version, the version is reported by `describe()`, and the registry refuses an unknown
version rather than defaulting. The evidence half does not. `chakaso.runtime.context.render_evidence_context`
puts `EVIDENCE_CONTEXT_HEADER` — a sentence that tells the model to cite only the bracketed ids shown — in
front of one block per chunk, and nothing labels that rendering. Its header is an instruction about format
sitting outside the versioned prompt surface, so editing it would silently change what a measured grounded
run fed the model, which is exactly what rule 6 exists to prevent.

**Why it was not fixed in the audit.** The cheap move — a `EVIDENCE_CONTEXT_VERSION = "…-v1"` constant nobody
reads — would be scaffolding, and the honest move is to decide where the version is *recorded*: on the
`AnswerRecord`'s provenance beside the retrieved pack, or in the release/evaluation payload. That is an
interface change to the answer record and needs a decision record, not a rename.

**What would close it:** a version label on the evidence rendering that reaches the same place the prompt
template version already reaches (`describe()`, the answer's `evaluation_provenance`, and an experiment
report), with the existing records stating which label they ran under.

### K-013 — Counts are copied into six documents and drift on the next commit

**Impact:** Medium, and it is how most of this audit's findings were produced.

Every quantity that changes as the project grows — records, pinned corpora, CLI commands, source files, test
totals — is written out in more than one place. `README.md`, `docs/getting-started.md`, `docs/roadmap.md`,
`docs/internal/experiments/README.md`, `docs/agent/CURRENT_STATE.md` and `AGENTS.md` each carried some version
of "seven experiment records" or "142 source files", and `docs/agent/CONVENTIONS.md` says plainly: "Do not
duplicate a specification across documents. Link to it. Duplicated text goes stale in one place and stays
correct in the other."

The 2026-10-24 pass found the drift in the open: seven records where there are eight, two corpora where three
are pinned, 12 commands where `chakaso --help` prints 13, and 142/147 source files where `mypy` now counts 157.
Some copies were corrected; the counts in `docs/roadmap.md` and the test census were replaced by a pointer to
`CURRENT_STATE.md` plus the command that reproduces them, because a number in a phase table is a number nobody
re-reads before citing it.

**Why it keeps happening.** Nothing checks a prose count. The hygiene tests verify links, tracked-file sizes,
allowlists and the workflow document — a sentence can say "eight records" next to a directory holding nine and
every gate stays green.

**What would close it:** one owner per count (`CURRENT_STATE.md` for census figures, `--help` for the command
list, the `experiments/` directory listing for records) with every other document linking rather than
restating; optionally a hygiene check that a count stated in a document equals the thing counted, which is easy
for records and commands and awkward for test totals, since those depend on which extras are installed.

### K-014 — This arc delivered straight to `main` while the conventions require a branch and a PR

**Impact:** High for the process, none for the code.

`docs/agent/CONVENTIONS.md` states: "Bootstrap-scale changes may go directly to `main`. Anything that changes a
documented interface, adds a dependency, or spans more than one component goes on a branch and through a pull
request, so that CI runs before it lands." The foundation release did all three of those things — new optional
extras, new checkpoint schemas, a new serving adapter, a new release contract — in roughly forty commits pushed
directly to `main`.

**What actually protected the work, and what did not.** Every commit ran the four gates locally before it was
pushed, and the three CI tiers read back green from the runner after the push — but the rule's stated purpose is
that *CI runs before it lands*, and here CI ran after. That ordering is what found two defects on 2026-10-22
(a Windows-only attribute and an assertion finer than float32): they were caught on the runner, which in a
branch-and-PR flow would have caught them before `main` moved. Direct delivery also means a bad commit is
`main`'s history the moment it lands, so a revert is a public act rather than a closed pull request.

**Why it was done anyway.** The mission was executed as an autonomous single-operator run, and the branch flow
assumes a reviewer. That is a reason to change the rule or to state its exception, not to keep breaking it
quietly — an unenforced convention is a convention that teaches contributors to ignore it.

**What would close it:** either an amendment to `CONVENTIONS.md` naming when direct-to-`main` is permitted (for
example: solo research arcs with three-tier CI and a documented rollback, never with an unreviewed interface
change), or a decision record that the branch rule is unconditional and a note in `ACTIVE_TASK.md` that this
arc's method was out of contract. This entry does not propose which; it records that both cannot be true.

### K-015 — A loaded release reports the adapter's name, not the identity its own contract states

**Impact:** Medium, and it is the identity surface a consumer actually reads.

`load_model_release(directory)` verifies the contract, then calls `load_model` on the checkpoint file the
contract names. The returned model reports `metadata.model_id == "subword_transformer"` — which is the
*registry key* of the adapter — while `model.json` beside it says the model is `chakaso-base`.
`GenerationResult.model_id`, `describe()["model_id"]` and `/health`'s `model_id` field all carry the adapter
name. Nothing here is exactly false — `subword_transformer` is what reads the file, `describe()["family"]` does
say `decoder`, and the serving layer already prefers the contract when it can: `ServingApplication.for_path`
composes `release.model_id@version` as its `model_name`, so `/v1/models` and a completion body answer
`chakaso-base@0.1.0` while `/health` prints both identities side by side. The disparity is that the *primary*
interface — the Python library ADR-0033 makes the product — gives a caller no way to ask a loaded release what
it is called except by loading the contract as a second object and reconciling the two by hand.

**Found alongside a defect that was fixed rather than filed.** The same adapter used to hard-code a display
name claiming the model was "trained on the project's own documentation" — which is not true of the released
decoder, nor of any checkpoint since E2607. Since `model inspect`, `/v1/models` and an answer's
`model_display_name` provenance all print that string, a false data-provenance claim was living inside the
product. It now derives from the artifact's declared family and names no corpus at all
(`test_the_display_name_names_the_architecture_without_inventing_provenance`). `model_id` is a separate
question because changing it changes what callers key on.

**What would close it:** a decision about which identity governs a loaded release — the release's `model_id`,
the adapter's, or both as distinct fields — written as a record, because every option is visible in served
output and in stored answer provenance. The cheap variant (pass the contract's `model_id` through
`load_model`) would make `chakaso model inspect dist/chakaso-base` answer `chakaso-base` and needs the slice's
identity assertion updated in the same commit; the honest variant adds a `trained_as`/`adapter` pair so an
integrator can tell what the artifact is from what read it.

### K-016 — `chakaso model serve` is the one lifecycle command no test runs as a command

**Impact:** Low-to-medium, and it is a hole in the claim that the CLI is thin over the library.

`tests/test_serve_server.py` and `tests/test_serve_app.py` exercise the transport and the request logic over a
real socket (49 tests), and `tests/test_vertical_slice.py` starts a `ServingService` in front of a checkpoint
trained inside that test. What no test does is invoke the `serve` *action*: nothing calls the handler that
parses `--host`, `--port`, `--max-concurrent` and `--model-path`, installs the signal handling, and blocks
until Ctrl-C. The vertical slice's terminal-lifecycle test drives export, verify, inspect, score and generate
through `main([...])`, and stops there because a blocking command cannot be asserted on without a subprocess
and a timeout. CI likewise never starts the server as a process.

**Why it is not simply fixed by a test.** A subprocess serving test needs a trained artifact, an available
port, and a reliable kill path; the failure modes it would catch (a flag name that does not reach the
application, a default that binds more widely than documented) are exactly the ones a unit test of the
transport cannot see. The `--host` default in particular is the safety property ADR-0038 rests on, and today
nothing proves the CLI does not override it.

**What would close it:** one subprocess test that runs `python -m chakaso model serve <fixture release> --port 0`,
reads the bound address off the health endpoint, asserts `127.0.0.1`, and terminates the process; landed in the
dependency-free tier only if a double can be substituted, otherwise in `model path`.

### K-017 — Half the measured records have no machine-readable evidence in the repository

**Impact:** Medium for the early records, and inconsistent with what the project now does.

`experiments/E2601`–`E2604` contain `record.md` and nothing else. `E2605` and `E2607` committed their corpus
manifests, `E2606` and `E2607` gained their aggregates (`results.json`, digests cited in the records) on
2026-10-24 precisely so a reader could re-derive four corrected numbers without the git-ignored `data/` tree,
and `E2608` — the record behind the released artifact — still has only `record.md`: its two arm reports, two
generation-suite outputs and snapshot live under `data/` and are cited by digest alone.

**What that costs.** `AGENTS.md` requires that "a result is aggregated from the reports on disk", and the E2606
and E2607 corrections were found *by* recomputing against a report. For E2601–E2604 the arithmetic in the prose
cannot be recomputed at all, and for E2608 a reader without the author's `data/` directory cannot check the
1.6865 against 1.9043 comparison, the token budget, or the generation statistics that withheld Level 3.

**Why nothing was copied in during the audit.** Committing a byte-identical copy of an artifact changes what the
record cites (the provenance table names a `data/…` path) and would apply inconsistently across nine records
unless decided once. It is a policy question for `docs/internal/experiments/README.md` and
[`artifact-policy.md`](../internal/engineering/artifact-policy.md), not a drive-by file move.

**What would close it:** a stated rule about which generated evidence is committed (the plausible one: any
machine-readable input to a number printed in a record, up to the 1 MiB tracked-file bound, with the record's
provenance table naming the committed path), applied to E2605–E2608 first since their statistics are cited in
the model card, and to E2601–E2604 by regenerating their aggregates from the reports the records already quote.

### K-018 — `--samples-per-prompt` counts tokens, not samples

**Impact:** Low, and it misleads a reader of a measured record.

The flag on `chakaso experiment model` names a quantity that is not what it holds: `samples_per_prompt` is the
number of *tokens generated* per continuation, and each prompt yields exactly one greedy sample. E2608's
protocol section says so, its deviation 7 records the misnomer and explains why the name survived — both arms'
commands are quoted verbatim in the record, and renaming after the fact would make the quoted command line no
longer the one that ran.

**Why it is filed rather than renamed.** `samples_per_prompt` is a field of the experiment plan, and the plan is
serialized into every run's report under that key. Renaming it therefore splits the artifact history: reports
already on disk and quoted by three records name one key, reports after the rename name another. Checking
`chakaso.training.seed_study` shows the study's enforced protocol identity does *not* include this field
(`_PROTOCOL_FIELDS` covers corpus, split, tokenizer, architecture, parameter count and the budget), so nothing
would refuse a study that mixed the two spellings — which is precisely why a rename needs a stated plan version
rather than a find-and-replace.

**What would close it:** `--continuation-tokens` (or `--max-new-tokens-per-prompt`) with the old name kept as a
deprecated alias until the next record that declares a plan, plus a schema version bump for the plan block so a
reused report cannot silently mean the other thing.

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
