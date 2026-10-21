# Active task

## Current unit: nodes A–R — Chakaso Scratch LM Foundation v0.1

**Goal.** Turn the measured research implementation into a complete, versioned, integration-ready
language-model foundation: an artifact contract, a config-driven training engine, a Python inference runtime
with bounded and streaming decoding, a versioned prompt-formatting contract, an optional local HTTP interface,
a generation-quality suite that separates training / language-model / behavioural evidence, one reproducible
reference model trained from random initialization on verified licensed data, and an honest release gate.
Domain-specific development is deliberately deferred to the handoff (node Q).

**Environment, measured 2026-10-22 at `a664477`.** Python 3.13.5; `torch 2.13.0+cpu` (14 threads), `numpy
2.4.0`; 18 logical CPUs; **no CUDA device**; 166.8 GB free disk; **16.6 GB RAM with 1.5 GB free (91 % load)**
— the last number is the binding constraint, so the new data path must not materialise a corpus as Python
objects the way the present one does. Disk free and RAM were read from the OS (`shutil.disk_usage`,
`GlobalMemoryStatusEx`); `psutil` is not installed and no RSS figure is claimed anywhere in this repository.

**The starting audit found four things that change what this unit can promise.**

1. **The `workflow` scope is present on this session's credential** (`gh auth status`: `gist, read:org, repo,
   workflow`). K-009 records the opposite, and every doc since repeats it. The `[train]` / `[nn]` / serving CI
   matrix is therefore landable here — node M changes from "design and record the gap" to "land it and read the
   result from the runner". K-009 stays open until the pushed workflow actually runs the tiers.
2. **`CURRENT_STATE.md` still says no language model exists in the runtime** (`:324-333`), and
   `docs/architecture.md:217-219` / `config/schema.py:127-129` / `configs/default.toml:33-36` still say a
   tokenizer "does not exist". Both are false: three adapters are registered, `load_model` is public, and the
   subword tokenizer has a versioned serialisation with a fingerprint. The residual truth is that the *default*
   adapter is still the double and no trained checkpoint produces useful text.
3. **What is genuinely missing** (verified against the code, not assumed): no KV cache — `forward()` re-runs the
   whole prefix per token; no `ignore_index`, so padding is unrepresentable and only dense packed blocks train;
   no streaming anywhere; `GenerationParams` has no top-p, no stop sequences and no seed, and `top_k` is
   stranded as a constructor argument the registry never passes; `GenerationResult` carries no token ids; no
   chat/prompt template exists (the adapter joins message contents with `"\n"`, unversioned); no HTTP server
   (no `fastapi|starlette|uvicorn|http.server` match anywhere); no experiment-level resume —
   `resume_state_from` has no production caller; no tokens/s metric; `seed_study.RunOutcome.status` is a
   two-value free string, so "ran / failed / hit the ceiling / converged / improved-loss-but-degenerate" is not
   a recorded distinction; `generation_eval` is never invoked by a run, so no report has a generation section.
4. **The LR schedule has warmup and nothing after it** (`nn_training._learning_rate`), and E2607's three runs
   all selected the step ceiling. A decaying schedule is a first-class hypothesis for "the budget did not
   finish", not a side effect to hope for.

**Corrections owed by this unit, recorded here so they are not lost** (each lands in the workstream that
resolves it, per the brief): the stale-model and stale-tokenizer claims above; `AGENTS.md:75-81` and `:180`
("four records, E2601–E2604" — there are seven, and E2607's data is not self-authored);
`docs/agent/CURRENT_STATE.md:77` (five records → seven), `:92` (142 → 147 source files), `:410` ("no training on
an external corpus"), `:43`/`:65` (`benchmark grounded` missing); `KNOWN_ISSUES.md:30` (K-001 narrowing),
`:88-89` (K-002 lists implemented components as absent), `:171` (142 → 147); `README.md:23`/`:25`/`:29-30`/
`:129-130`/`:185` (record statuses, test counts, "one author's prose", 11 → 12 commands, ADR-0036 missing);
`docs/roadmap.md:144-146` (test counts), `:214-218` (first two "next" steps are measured); `docs/training.md:7`
and `:275` (E2606/E2607 are measured, not "running"), `:42` (stage M0 "In progress");
`docs/evaluation.md:205-208` and `:332-335` (error bars now exist; `1.8657` is the best of five draws and must
be quoted with the mean); `docs/research/README.md:42`/`:59`/`:61-62`; `docs/README.md:54`;
`docs/internal/experiments/README.md:1-3`; `docs/research/next-model-experiment.md:131-144`/`:195` (future tense
over completed studies); `ADR-0033.md:32-34` and `ADR-0036.md:12` (13 → 19 drifted documents);
`docs/architecture.md:174`/`:178`/`:225-247`/`:315` (model boundary under-documents `load_model`, `sample`,
`score`, seed study; "no prompt template" is now a rule to honour rather than a fact); and
`experiments/E2606/record.md:197` (E2605 selected step 1,380, not 1,440 — historical records otherwise stay
immutable, this one is a transcription error against E2605's own table).

**Definition of done.** Nodes B–L delivered as tested code, node G executed on verified licensed data with its
preregistration written before the run, node M's CI matrix landed and its remote result read from the runner,
node N's vertical slice reproduced from a clean installed environment, node O's release level awarded on
evidence only, nodes P–Q shipped as model card / data card / recipe / roadmap, and node R's full gate set green
in every applicable matrix with the pushed head and CI state independently verified. A checkpoint's weights must
originate from random initialization; no pretrained weights, hosted model API or remote inference provider may
stand in for the trained model. If generations still loop, Level 3 is recorded as **not achieved** and the
loops stay visible.

## Mission ledger — Chakaso Scratch LM Foundation v0.1 (nodes A–R)

| Node | Task | Depends on | Owns | Status | Evidence / commit | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| A | Live audit, reconcile, environment measurement | — | `docs/agent/ACTIVE_TASK.md` | in_progress | this section; `gh auth status` scope list; OS memory/disk reads | baseline gates re-run below |
| B | Model artifact release contract | A | `models/artifact.py` (new), `training/nn_checkpoint.py` | pending | — | — |
| C | Measured compute budget and architecture decision | A | `experiments/E2608/`, ADR-0037 | pending | — | — |
| D | Data pipeline and a broader licensed corpus | A | `training/source_corpus.py`, `training/dataset_card.py` | pending | — | — |
| E | Tokenizer and model path (special tokens, KV cache, `ignore_index`) | B, D | `training/subword.py`, `training/transformer.py` | pending | — | — |
| F | Reusable training engine (config, decay, accumulation, resume, throughput) | E | `training/nn_training.py`, `configs/*.toml` | pending | — | — |
| G | Reference model trained from random initialization | F | `experiments/E2608/` | pending | — | — |
| H | Checkpoint lifecycle: verify, load, resume, export | B, G | `models/`, `training/nn_checkpoint.py` | pending | — | — |
| I | Inference runtime and decoding controls | B, H | `models/base.py`, `models/nntransformer.py` | pending | — | — |
| J | Streaming and versioned prompt formatting | I | `models/formatting.py` (new) | pending | — | — |
| K | Optional local HTTP serving adapter | I, J | `serve/` (new) | pending | — | — |
| L | Generation-quality evaluation wired into a run | G, I | `training/generation_eval.py`, report shape | pending | — | — |
| M | CI dependency matrix (`[train]`, `[nn]`, serving) | K | `.github/workflows/ci.yml` | pending | — | — |
| N | End-to-end vertical slice from a clean environment | G, K | integration tests | pending | — | — |
| O | Release gate level awarded on evidence | N, L | `MODEL_CARD.md` | pending | — | — |
| P | Packaging, model card, data card, recipe, release note | O | `docs/`, `CHANGELOG.md` | pending | — | — |
| Q | Handoff roadmap, phases 0–6 | P | `docs/roadmap.md` | pending | — | — |
| R | Final gates, history review, push, remote verification | A–Q | — | pending | — | — |

A node moves to complete only when its acceptance criteria are evidenced here. Nothing in this table is marked
from the existence of a file.

## Previous unit (delivered): nodes A–N — reproducibility, multi-seed validation and generalization (P33–P36)

**Goal.** The reproducibility arc is built, measured and recorded: E2605 is reproducible from a corpus
recovered out of git history and its five-seed study (E2606) is measured. What remains in this unit is the last
node of the research arc (E2607's three seeds on the external corpus), then every gate in every dependency
matrix, the delivery ledger, and the push.

**Definition of done.** `pytest` passes in the dependency-free, `[train]` and `[train,nn]` matrices; `ruff
check`, `ruff format --check` and `mypy src` pass, and `mypy --config-file
configs/mypy-model-path.toml` passes over every source file where both extras are installed; the package builds
and imports from a clean install; the E2607 aggregate is rebuilt from its per-run reports and independently
recomputed; the history reads as a deliverable (each subject matching its diff, no rewritten pushed history);
the work is pushed to `main`, the remote SHA verified independently, and CI status read from the runner.
`.github/workflows/*` is not touched: the session's credential has no `workflow` scope, so the prepared
three-tier matrix ships as a tested document instead and the gap stays recorded (K-009).

**What this unit is not.** Not a bigger model. E2607's budget stays at E2605's 2,400 steps deliberately, which
means a smaller gap there has two candidate causes — the data, or the under-scaled budget — and the record says
so rather than picking after the fact.

**Delivery record (2026-10-21).** The arc began with `HEAD` == `origin/main` == `35e40f7` and a clean tree. Its
content head is `470bb5dc2253714b9bd9000063491174c135e21a`: thirty-three commits, the number `git rev-list
--count 35e40f7..470bb5d` returns, all pushed normally — no force-push, no branch deleted, no published history
rewritten, no hook bypassed, no `.github/workflows/*` file touched. One bookkeeping commit follows it, the
correction that names `470bb5d` as the content head: an earlier version of this paragraph named `1ded45b` and
counted thirty-two, which was true until the commit writing it landed. A ledger cannot be the last line of the
delivery it counts, so read the range between the two endpoints rather than `HEAD`.
Remote CI run `39633056951` completed **success** on Python 3.11, 3.12 and 3.13 for `470bb5d`, as did run
`39632675804` for `1ded45b` — the three jobs of the dependency-free matrix, which is still the only matrix CI
has (K-009), so the model path's two tiers remain locally verified and remotely untested. The commit count is
what the work supported: a snapshot format, its recovery, the leakage tests, the driver and CLI changes, the
seed study, the ingestion module, the generation suite, the atomic writer, two executed studies and the
documentation that describes them, each landed as one reviewable change rather than split to reach a number.

Two measurement notes that belong with the delivery, not in a footnote. E2605's corpus had to be recovered
from git history because the repository has since edited 19 of the documents it pinned — which is the drift
guard working, and the reason ADR-0036 exists. And E2607's three runs never early-stopped: on 2.13× the data
the declared 2,400-step budget truncates the curve, so its reported loss is a floor rather than a converged
level, stated in the record rather than smoothed.

## Mission ledger — reproducibility, multi-seed validation and generalization (P33–P36)

Durable status for this arc. Nodes are the delivery graph of the session brief; a node is complete only when
its acceptance criteria are met and evidenced here.

| Node | Task | Depends on | Owns | Status | Evidence / commit | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| A | Verify the starting state | — | (read-only) | validated | HEAD == `origin/main` == `35e40f7`, clean tree, token scopes `gist, read:org, repo`, branch unprotected, Actions enabled; baseline gates 1,391/16 dependency-free, 1,472 `[train]`, 1,605 `[train,nn]` | the recorded gates re-run before any edit |
| B | E2605 provenance, and a snapshot that can be recovered | A | `training.{manifest,snapshot}`, ADR-0036 | pushed | `bdd6e53` (self-verifying snapshot format), `592b3f4` (recovery from a revision) + ADR-0036 | 71/71 documents recovered from `9b8a478` and re-hashed; fingerprint `e45993f9badf706c` reproduced; 21 acquisition tests, 12 new snapshot-format tests |
| C | Data and leakage contract strengthened | B | `tests/test_training_{manifest,snapshot,subword_experiment}.py` | pushed | `55f3c81` | fingerprint stable across CRLF and lone-CR; verification freezes the bytes a run uses; selection shown to be a function of the validation curve alone |
| D | Snapshot in the driver, declared corpus, split seed, per-experiment naming | B | `training/subword_experiment.py`, `cli.py` | pushed | `cf4b8ca`, `561c7c9`, `8de19be` (record naming), `261814a` (overwrite/collect guards) | 20 driver tests incl. snapshot origin, refusal on a foreign corpus, artifact naming |
| E | Seed study runner | D | `training/seed_study.py` | pushed | `0ba1744`, `ca6e619` | 29 study tests: declaration rules, aggregation from disk, reuse refusal, failure accounting, no interval published |
| F | E2606 measured | E | `experiments/E2606/` | measured | `fc5f8ff` (preregistration, before the runs), `aacc863` (results), `e41d7e8` (E2605 addendum) | 5/5 seeds; mean 1.8794 ± 0.0077 bits/char against 3.0937 ± 0.0267; seed-7 reproduction exact, all 41 tensors equal |
| G | Independent licensed corpus | B | `training/source_corpus.py`, `experiments/E2607/` | pushed | `7fe88a3` (ingestion), `b765d37` (pinned corpus + preregistration) | 16 ingestion tests; 3 Project Gutenberg texts, 105 documents, 1,660,274 characters, fingerprint `931b07a635ca5a0c`, byte digests equal to `sha256sum` |
| H | E2607 executed | E, G | `experiments/E2607/` | measured | `b765d37` (preregistration, before the runs); the results land in this same commit's record | 3/3 seeds, mean 2.0176 ± 0.0118 bits/char against 3.2218 ± 0.0153 (gap 1.2042 vs 1.2143 internally); no run early-stopped, all selected the 2,400-step ceiling |
| I | Generation and generalization evaluation | D | `training/generation_eval.py` | pushed | `3654f92`, `0ad1604` | 22 dependency-free tests; all five E2606 and all three E2607 checkpoints measured with raw samples retained (external prompts taken from held-out documents of that corpus) |
| J | Correctness, security, resource discipline | B–G | `core/atomic.py`, `nn_checkpoint.py`, tests | pushed | `0a48ca8` (staged writes), `d2051ac` (environment block) | interrupted-write, corruption, path-traversal and revision-shape tests; wall clock recorded per run, peak memory not measurable from this shell (stated, not guessed) |
| K | Documentation and research integrity | B–J | `README.md`, `docs/*`, `CHANGELOG.md`, `docs/agent/*` | pushed | `53896b8`, `74a2a04`, `c3923e6`, `bbd350f`, `c0c9ae5`, `d057bfe`, `1ded45b`, `470bb5d` | link + status hygiene (23 tests) |
| L | Optional-dependency CI design | J | `docs/internal/engineering/optional-dependency-ci-matrix.md` | blocked (scope) | `9b47724` | 4 hygiene tests assert the matrix's parity with the documented commands and that the shipped workflow still covers only `.[dev]` |
| M | Full regression and package verification | K | whole tree | validated | gates recorded below | three matrices, both mypy configs, wheel build, clean-interpreter import, checkpoint load from the installed copy |
| N | Delivery | M | — | pushed | content head `470bb5dc2253714b9bd9000063491174c135e21a` (fast-forward from `35e40f7`, 33 commits, nothing forced or rewritten), followed by the one ledger correction this row is part of; CI run `39633056951` completed **success** on Python 3.11, 3.12 and 3.13 | the dependency-free matrix only (K-009) |

**Gate and package record for row M (measured 2026-10-21 against this tree).**

| Check | Result |
| --- | --- |
| `pytest`, dependency-free `.venv` | 1,479 passed, 21 skipped in 21.44 s (1,500 collected; the skips are the numpy- and torch-gated modules) |
| `pytest`, `.venv-train` (`.[dev,train]`) | 1,560 passed, 12 skipped in 25.58 s |
| `pytest`, `.venv-nn` (`.[dev,train,nn]`) | 1,739 passed, 1 skipped in 99.50 s |
| `ruff check .` / `ruff format --check .` | clean / 331 files already formatted |
| `mypy src` (dependency-free) | clean over 118 source files |
| `mypy --config-file configs/mypy-model-path.toml` (`[nn]`) | clean over all 147 source files |
| `pip wheel --no-deps .` | `chakaso-0.1.0.dev0-py3-none-any.whl`: 147 modules, `py.typed`, `LICENSE`, no `.pt`/corpus bytes, no runtime dependency, extras `dev`/`train`/`nn` with the declared bounds |
| Unpacked wheel, clean interpreter (`python -I -S`) | `chakaso 0.1.0.dev0`; `chakaso.models`, `chakaso.training`, and the dependency-free `manifest` / `snapshot` / `source_corpus` / `generation_eval` all import; `import numpy` fails, which is the point; 43 training exports, `restore_corpus_snapshot` among them |
| CLI from the unpacked wheel | `--version` exits 0, `capabilities` / `config show` / `sources list` exit 0, `corpus` with no action returns 1, `experiment study` without its required arguments exits 2 |
| `load_model` on a trained E2607 checkpoint, from the installed copy | `model_id=subword_transformer`, `development_double=False`, `context_window=128`; a 16-token greedy sample returns `\nAnd now the dust of the dark`; `score()` over `docs/architecture.md` returns 5.3878 nats (an out-of-domain smoke observation on a file the model never trained on, **not** a recorded result); an absent path is refused with a typed `ModelError` naming it |
| Corpus recovery, end to end | `chakaso corpus restore` recovered E2605's 71 documents from `9b8a478` (fingerprint `e45993f9badf706c`, snapshot checksum `84ecbc61cf1e634b…`); `corpus verify` accepts the snapshot and refuses the moved working tree, naming 19 changed files and one newly selected document |
| E2606 / E2607 aggregates | rebuilt from per-run reports by `--collect-only`, then recomputed independently over the reported values; both agree. Digests `3f8f0175c510ff68…` and `efe541727c46c369…` |
| Tracked-file guards | `test_repository_hygiene` 23 passed: private pack untracked, no secret-shaped files, no tracked file over 1 MiB, no commercial provider dependency, torch confined to the admitted modules, both mypy configurations non-drifting, the CI document consistent with the documented commands and with what the shipped workflow actually installs |

**Peak memory is not stated anywhere.** The process-memory interfaces reachable from this shell return
nothing, so no RSS figure is claimed for any run, in any record, in this session.

**Brief-node mapping (this session).** Brief §3 (verify the repository) → row A. §4 (E2605 provenance and
snapshot) → rows B and D. §5 (data and leakage contract) → row C. §6 (predeclare the multi-seed study) →
row E's declaration rules and the `fc5f8ff` preregistration. §7 (trustworthy experiment runner) → row E. §8
(execute the study) → row F. §9 (independent corpus) → row G. §10 (generation and generalization) → row I. §11
(optional-dependency CI) → row L. §12 (correctness, security, resources) → row J. §13 (documentation) → row K.
§14 (commit history) → the table's commit column. §15 (delivery) → row N.

## Mission ledger — model-centred realignment (prior arc, delivered)

Durable status for the realignment arc (nodes are the delivery graph; a node is complete only when its
acceptance criteria are met and recorded here). Statuses: `pending`, `ready`, `in_progress`, `blocked`,
`validated`, `committed`, `pushed`.

| Node | Task | Depends on | Owns | Status | Evidence / commit | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| A | Repository + architecture audit | — | `docs/internal/engineering/` | committed | `538a47d` | hygiene 14 passed then; 17 at this head |
| B | Product contract decision | A | `docs/decisions/`, `AGENTS.md` | committed | `033edde` (ADR-0033), `760af7a` (AGENTS.md, `.gitignore`) | hygiene 14 passed then; 17 at this head |
| C | Model + compute feasibility, measured | A | `docs/research/` | committed | `1c4172d` (measurements), `a75cccc` (ADR-0034) | executed probes |
| D | Target architecture + dependency decision | B, C | `docs/decisions/`, `pyproject.toml`, `docs/architecture.md` | committed | `a75cccc` (ADR-0034), `bb885d7` (`[nn]` extra + isolation tests), `ed3e156` (normative dependency direction) | hygiene gates the torch surface |
| E | Data and tokenization (P30) | D | `chakaso.training.{manifest,subword,packing,data}` | committed | `dcf8355`, `5a01714`, `ed97d55`, `eddb3e3`, `c1225d8` | dependency-free: 38 + 10 + 31 + 17 tests |
| F | Model architecture and initialization | D, E | `chakaso.training.transformer` | committed | `232f673` | 39 torch-gated tests, incl. parameter-count agreement |
| G | Training, optimization, checkpoints | F | `chakaso.training.nn_*`, `training.loop` | committed | `1219c3c` (checkpoint), `ac46cf6` (training + resume), `875477f` (mini-batch character) | 20 + 28 torch-gated tests |
| H | Inference and Python API | G | `chakaso.models` | committed | `c0e5f64` (`load_model` + adapter), `625f08a` (artifact scoring) | 12 + 30 tests, adapter inherits the contract suite |
| I | Model evaluation and baselines | E, G | `chakaso.training.model_metrics`, `nn_evaluation` | committed | `3fa9f12`, `625f08a` | 10 hand-computable metric tests + 7 evaluation tests |
| J | Experiment E2605, measured | E–I | `experiments/E2605/` | committed | driver `0cbed79`, CLI `d878828`, pre-run record `9b8a478`, `bcdfa7d`, measured record `490ac1f`, reproduction note `00cbd78` | 11 driver tests; the run itself measured 513.6 s + 69.9 s + 13.2 s |
| K | Subsystem reassessment + documentation | J | `README.md`, `docs/*`, `CHANGELOG.md`, `docs/agent/*` | committed | audit §11 re-reads the subsystem verdicts against the delivered tree; `760af7a`, `2470af6`, `691a319`, `adc9e77`, `cc816af`, `6fa370a`, `e59a8bd`, `e183a87` | hygiene 19 passed (links + status guards), and the dependency direction re-checked by grep |
| L | Full regression, review, push | K | whole tree | pushed | gates recorded below; delivered by `e183a87` and `9701bc7` | three matrices green, both mypy configurations green, wheel built and smoke-tested, CI green |

**Delivery record (2026-10-21).** Every row above is `pushed`: the arc began at `eaee98d`, its first 36
commits reach `9701bc7b841018245b09d35c257463df65bded17` on `origin/main`, and the commit that records this
line follows it onto the same branch (it is the only commit that post-dates the SHA it names). Remote CI run
`39544651281`
completed success on Python 3.11, 3.12 and 3.13 — the dependency-free matrix, which is the only matrix CI
has (K-009). No branch was forced, deleted or rewritten, and no `.github/workflows/*` file was touched.
Next eligible node: **P33** (corpus snapshot reproducibility), then **P34** (E2606 across pre-declared seeds).

**Brief-node mapping.** The delivery brief numbers its own nodes; this ledger's rows are the same work at a
finer grain. Brief A (inspect the repository) → row A. Brief B (model-centred target architecture) → rows B
and D. Brief C (choose the next experiment on evidence) → row C. Brief D (training data and tokenization) →
row E. Brief E (model implementation) → row F. Brief F (training, optimization, checkpoints) → row G. Brief G
(checkpoint-backed Python API) → row H. Brief H (model evaluation and experiment quality) → rows I and J.
Brief I (reassess the surrounding subsystems) → audit §9 and §11. Brief J (reconcile documentation and
project identity) → row K. Brief K (testing, CI, security, reproducibility) and brief L (professional git
history) → row L.

**Gate and smoke-test record for row L (measured 2026-10-21 at `e183a87`).**

| Check | Result |
| --- | --- |
| `pytest` dependency-free | 1,391 passed, 16 skipped in 13.37 s |
| `pytest` with `[train]` | 1,472 passed, 7 skipped in 17.67 s |
| `pytest` with `[train,nn]` | 1,605 passed, 1 skipped in 55.21 s |
| `ruff check .` / `ruff format --check .` | clean / 317 files already formatted |
| `mypy src` (dependency-free) and `mypy --config-file configs/mypy-model-path.toml` (`[nn]`) | 117 files clean; 142 files clean |
| `pip wheel --no-deps .` | built `chakaso-0.1.0.dev0-py3-none-any.whl`, 142 modules + `py.typed`, no runtime dependency, extras `dev`/`train`/`nn` |
| Clean-interpreter import of the unpacked wheel (`python -I -S`) | `chakaso 0.1.0.dev0`; `chakaso.models.load_model` and `chakaso.training` import; `chakaso --version`, `capabilities`, `config show` and `sources list` all exit 0, with `config show` reporting "No configuration file was read; these are the built-in defaults" |
| Checkpoint load / generate / score from the installed library | `load_model` on E2605's artifact reports `model_id=subword_transformer`, `development_double=False`; a 24-token greedy sample reproduces the recorded looping style; `score()` over `docs/training.md` returns 2.4996 nats (a smoke observation on a file that has since been edited, **not** a recorded result); scoring a text shorter than one 128-token window raises a typed `ModelError` naming both counts |
| `git diff --check` over the arc | one new blank line at EOF in the audit, fixed before push |
| Tracked-file guards | `test_repository_hygiene` 19 passed: private pack untracked, no secret-shaped files, no tracked file over 1 MiB, no commercial provider dependency, torch confined to the named allowlist, mypy configurations non-drifting |

**Blockers and standing constraints.** Remote CI cannot gain a `[train]`/`[nn]` job: this session's GitHub
credential has scopes `gist, read:org, repo` and no `workflow` scope (K-009, re-verified 2026-10-21), so CI
files stay untouched and the gap is recorded rather than claimed away. Peak process memory is not
measurable from this shell (the Windows process-memory APIs and `tasklist` filtering return nothing), so no
RSS number is stated anywhere. Starting `HEAD` for this arc was `eaee98d`; work is on `main`, which was
ahead of `origin/main` by one commit before this arc began.

## What the units before this completed (P30 and P31 — the model path, measured)

Decided (ADR-0033/0034/0035), built, tested, run and recorded (E2605). The model path is now: pinned corpus →
document split → learned subword tokenizer → packed next-token examples → decoder-only causal Transformer →
mini-batch training with validation-selected early stopping and bit-identical resume → one checksummed,
self-describing checkpoint → bits-per-character metrics → `chakaso.models.load_model`.

- [x] **P30, dependency-free:** `training.manifest` (per-document SHA-256 over normalized text, licence,
      source, typed `CorpusDriftError`), `data.split_text_units` (whole-document three-way split, deduplicated
      by content before selection), `training.subword` (byte-level BPE, named pre-tokenizer, deterministic
      tie-breaking, self-describing checksum-verified record), `training.packing` (targets shifted by one,
      dropped tail reported) and `training.model_metrics` (bits per character / nats / perplexity / accuracy
      through one arithmetic, with an `IdentifierDecoder` protocol so two architectures and two baselines are
      measured the same way). All of it runs in the dependency-free CI matrix.
- [x] **P31, behind `[nn]`:** `training.transformer` (parameter count derived from the configuration and
      cross-checked against the built model), `training.nn_training` (`plan_batch` as a pure function so a
      resumed run replays its batches; the loop's incremental selection is checked against
      `choose_validation_best` and refuses to return if they disagree), `training.nn_checkpoint`
      (ADR-0035), `training.nn_evaluation`, `models.load_model` + `models.nntransformer` (adapter inherits the
      shared `LanguageModel` contract; sampling is seeded from the request), and `chakaso experiment model`
      as a thin handler that imports torch lazily.
- [x] **E2605 measured result** ([`../../experiments/E2605/record.md`](../../experiments/E2605/record.md),
      written and committed *before* the run, with `results.json` and the corpus manifest committed after it):
      held-out **1.8657 bits per character** for the 643,840-parameter Transformer against **3.1343** for
      E2601's character architecture, **3.3224** unigram and **4.8010** uniform on the same positions;
      early stopping selected step 1380 of 2400 and the un-stopped model was **0.1434 nats worse**; three
      committed generations loop, and the record says that is the point. Single seed, single author's prose,
      no error bar, no runtime wiring.
- [x] **The honest residue, recorded rather than smoothed over:** the pre-run record's corpus figure
      (754,789 characters) was superseded by the run's own pin (780,223) and that is written into the record;
      re-preparing the pinned corpus in this checkout now refuses because the tree moved on, so reproducibility
      needs a snapshot the driver does not yet write (that is **P33**).

## What the previous unit completed (P28 — the retrieval evidence-sufficiency decision)

Designed (ADR-0032), built, tested, measured and recorded (E2604). Runtime consumption deliberately deferred
to P32 — the decision is measured before it is wired.

- [x] `chakaso.retrieval.sufficiency` — a deterministic, offline, numpy-free **evidence-sufficiency**
      decision layered **additively** over retrieval (the retriever still returns top-k; `Retriever` and
      `RetrievalService` are unchanged). `EvidenceSufficiency` (outcome + the discriminating terms +
      counts, so a verdict is inspectable), the `EvidenceSufficiencyPolicy` boundary, and
      `DiscriminatingTermSufficiencyPolicy` (a pack is sufficient iff a retrieved chunk shares a
      **non-function-word** query term that is rare in the corpus). The function-word exclusion was a
      *corrected* hypothesis: pure document-frequency flagged `to`/`on`/`and` (df=1 on a 13-doc corpus) as
      discriminating and leaked every abstention — the E2603 investigation caught it and the design changed.
- [x] `GroundedEvaluationRunner` scores sufficiency as a **separate dimension** (accuracy, false- and
      missed-abstentions) per case, opt-in so the E2603 baseline path is byte-identical; reports and
      `chakaso benchmark grounded --sufficiency` expose it (off by default, `configuration.sufficiency`).
- [x] **E2604 measured result:** sufficiency **accuracy 1.0** on the workload (14 assessed) — all three
      unanswerable cases judged insufficient, **zero** answerable cases wrongly abstained (recall
      preserved), retrieval's own recall@k/MRR/forbidden-hits/false-retrievals unchanged (additive).
      Honest limits recorded: a narrow corpus-specific working band, a broken single-doc case, no runtime
      effect, no model. [`../../experiments/E2604/record.md`](../../experiments/E2604/record.md).

## What the unit before that completed (P27 — the grounded-answer evaluation workload)

Designed, built, tested, run and recorded (ADR-0031; E2603).

- [x] `chakaso.benchmark.grounded_*` — a grounded-answer evaluation sub-module: a provenance-bearing
      `GroundedCase`/`GroundedSource`/`GroundedProvenance` schema (question, expected disposition,
      required facts, prohibited claims, gold/forbidden evidence by label, rationale, split), a
      `GroundedDataset` that refuses dangling evidence, duplicate cases and development/held-out split
      leakage, and a strict manifest+JSONL loader with the same validation.
- [x] A **self-authored Apache-2.0** workload (`ga-kestrel-library`, 13 passages / 14 cases, 9 dev /
      5 held-out, pinned content fingerprint `d2d2ee00d117b88a`) spanning direct, multi-source,
      distractor, conflicting, ambiguous, missing-evidence and unsupported-premise situations. Every
      required fact is asserted (by test) to be literally present in its gold evidence.
- [x] `GroundedEvaluationRunner` composing the real `RetrievalService` and `evaluate_answer` over an
      injected answer provider, classifying retrieval / generation / evaluation failures distinctly
      (a failure is recorded, never scored as success); a **reference oracle** (an oracle, explicitly
      not a model) and a non-answering provider; hand-checkable metric primitives over separate
      dimensions (no merged score); deterministic text and JSON reports that name the provider and
      whether it was an oracle; and `chakaso benchmark grounded` (`--provider`, `--split`, `--json`).
- [x] **E2603 measured result:** reference-oracle run over the workload — retrieval recall@k 0.8182 /
      MRR 0.8636 (held-out recall@k 1.0), zero stage failures, grounded-answer rate 0.8182, and honest
      negatives kept visible: 3 forbidden hits and 3 false retrievals. Recorded as a harness/retrieval
      measurement, never answer capability (no model answers; K-001 stands).

## What the unit before that completed (P26 — E2602, the training result reproduced across seeds)

Implemented, run, and recorded as a measured result
([`../../experiments/E2602/record.md`](../../experiments/E2602/record.md)).

- [x] `data.split_three_way` + `ThreeWaySplit` — a deterministic, disjoint train/validation/test
      separation (units wholly on one side, every unit exactly once) so early-stopping selection on
      validation never touches the reported test number; `fit_vocabulary` made structural to serve both
      split shapes. 5 numpy-free split tests.
- [x] `chakaso.training.controlled.run_controlled_experiment` — a fixed early-stopping protocol over
      pre-declared seeds, aggregating mean + sample stdev of held-out test loss against baselines.
- [x] **E2602 measured result:** mean test loss at the validation-selected step **1.7978 ± 0.1767 nats**,
      below both the unigram (mean 2.9808) and uniform (~3.3) baselines on **all five** seeds; the
      un-early-stopped step-300 loss (mean 2.7804) is worse on every seed — E2601's overfitting is
      systematic, not one lucky partition. Reported honestly as a small-corpus, high-variance toy result.

## What the unit before that completed (P25 — a trained artifact behind the `LanguageModel` boundary)

Implemented, tested, and honest about its limits; it integrates E2601's artifact without distorting the
interface.

- [x] `chakaso.models.trainedchar.TrainedCharacterModel` — a `trained_char` adapter selected through
      `model.adapter`/`model.checkpoint_path`: loads a checkpoint, rebuilds its bound vocabulary, generates
      greedily within `max_new_tokens` (reports `LENGTH` honestly — no learned stop token), declares
      `development_double=False` and implements `TOKENIZE`, and fails explicitly on an empty/missing/
      corrupt checkpoint rather than falling back to the double.
- [x] The checkpoint now embeds the vocabulary characters (schema/2, validated against the stored
      fingerprint) so a checkpoint can decode on its own — the measured E2601 losses are unchanged.
- [x] Lazy registry import keeps `import chakaso.models` and the base install numpy-free; the adapter
      inherits the full shared `LanguageModel` contract over a real trained checkpoint (numpy-gated tests).
- [x] Honest scope: the model is a tiny toy that produces incoherent text; the value is a real
      training-and-inference path, not usefulness, and that is stated in the adapter, the docs and the
      changelog.

## What the unit before that completed (P24 — the from-scratch training experiment, E2601)

Implemented, tested, run, and documented as the project's first measured result
([`../../experiments/E2601/record.md`](../../experiments/E2601/record.md)).

- [x] `chakaso.training` built end to end (the module ADR-0005 reserved): a self-authored Apache-2.0
      corpus, a deterministic character vocabulary (codepoint-ordered ids, reserved unknown id, content
      fingerprint), a **leak-free** line-unit split with the vocabulary fit on training text only, a
      validated `TrainingConfig`, a tiny causal character model with **hand-derived gradients verified
      against central finite differences for every scalar** (this caught and fixed a real numpy
      fancy-index scatter bug), an in-project Adam optimizer, a full-batch training loop that records a
      held-out curve, a content-checksummed JSON checkpoint that refuses corruption and incompatibility,
      uniform/unigram baselines, held-out evaluation, and an `E2601` experiment driver.
- [x] **E2601 measured result:** from random initialization, held-out next-character cross-entropy falls
      from 3.3118 nats (step 0) to **1.6650** (best, step 50) and ends at **2.4946** (step 300) — below
      **both** the unigram (3.0531) and uniform (3.3322) baselines; final bits/char 3.5989, perplexity
      12.12, next-char accuracy 0.6026; 984 parameters, seed 7, Windows/CPython 3.13.5/numpy 2.4.6, CPU
      only. The measured **overfitting** after step 50 is reported, not hidden.
- [x] The dependency-free layers (corpus/tokenizer/split/config/baselines) run in the core CI matrix;
      the numpy layers are `importorskip`-gated and pass with `[train]` installed. **A dedicated `[train]`
      CI job was NOT delivered** — the session's GitHub token lacks the `workflow` scope, so the
      workflow file cannot be pushed; recorded as K-009 rather than claimed as done.

## What the unit before that completed (P23 — the dependency decision, ADR before code)

Delivered as a decision record; no model code was written and no dependency was installed.

- [x] [ADR-0030](../decisions/ADR-0030-minimal-optional-training-dependency.md) names the minimal offline
      numerical stack: **`numpy`, isolated in a new optional `[train]` extra** (`numpy>=1.26,<2.5`; the
      `<2.5` cap keeps the Python 3.11 floor working), with **hand-derived gradients verified by finite
      differences** rather than a framework's black-box autograd. It evaluates the candidate stacks on
      the Python floor, Windows, offline reproducibility, CPU-only operation, memory, footprint,
      licensing, determinism/seed control, verifiability, and CI isolation, and cites PyPI metadata for
      the packaging facts.
- [x] A decision that **the base install stays dependency-free and the core CI matrix is unchanged** —
      training tests skip unless the extra is installed. A heavyweight autograd framework (PyTorch) is
      **rejected for now** with explicit
      revisit conditions (a concrete larger-model workload, or the manual-gradient route becoming a
      correctness risk). Training code is placed in `chakaso.training`, the module ADR-0005 reserved.

## Unit before that (P22 — a real-model feasibility assessment)

Delivered as a research note; no experiment was run and no dependency was added.

- [x] `docs/research/model-feasibility.md` inspects the available machine directly (Python 3.13.5,
      18 logical CPUs, ~15.4 GB RAM with ~1.4 GB free at inspection, Intel Arc — no CUDA, no ML library
      installed, ~160 GiB free disk) and assesses a runnable pretrained model (Option A) against a
      from-scratch training experiment (Option B) across hardware, licence, reproducibility, fit with the
      boundary, and maintenance.
- [x] A recommendation that separates verified facts from labelled estimates and assumptions: defer
      Option A until a real grounded-answer evaluation exists; the smallest honest next step is a scoped,
      out-of-CI from-scratch experiment producing the project's first genuinely measured number — and it
      was gated behind this P23 dependency ADR, now accepted. Logged in the research log; it narrowed the
      "which local model" open question without answering it.

## Unit before that (P21 — a triggered, explicitly-composed reassessment)

Implemented, tested, documented and green on Python 3.11–3.13 (recorded in
[ADR-0029](../decisions/ADR-0029-explicit-reassessment-trigger.md)).

- [x] A deterministic, inspectable **reassessment trigger** (`chakaso.correction.trigger`) reads a
      follow-up for an explicit, validated `ans_…` reference. It detects intent only — it reads no store,
      runs no evaluator, retrieves nothing and mutates nothing, so intent detection is independently
      testable from execution. A vague reference is never inferred; two distinct identifiers are
      *ambiguous* and refused, not guessed.
- [x] The trigger is **composed at the application root** (`Application.reassess_followup`) with the
      existing `reassess_and_record` boundary: a named, recorded answer is re-grounded against its
      standing evidence (the corpus chunks still present for the sources it named) merged with whatever
      the follow-up retrieves through a new read-only `gather_evidence` seam on the runtime. Unknown and
      ambiguous targets raise the boundary's typed errors (`UnknownAnswerError`,
      `AmbiguousReassessmentTargetError`); a follow-up that names nothing is an ordinary turn.
- [x] **Reachable from the supported runtime interface**: `chakaso chat` routes an answer-naming
      follow-up into reassessment and reports the decision honestly, and surfaces each recorded answer's
      identifier so a later turn can name it. The original `AnswerRecord` and its historical provenance
      are never rewritten; a reassessment is a distinct append-only `CorrectionRecord`; no revised prose
      is generated (the model in that path is a development double).

**What P21 did not do (and will not pretend to):** no autonomous decision (nothing revises an answer
*unbidden*), no revised wording, no correction record auto-persisted to disk, and the development double
gained no capability.

## Ordering after this unit

1. **P31 — the model itself: done.** `TransformerConfig`, the decoder-only causal Transformer, the
   mini-batch loop with validation-driven early stopping and bit-identical resume, `chakaso-transformer-checkpoint/1`
   (ADR-0035) and `chakaso.models.load_model` all exist, are tested in the `[nn]` matrix and were exercised by
   the run that produced the project's first Transformer measurement.
2. **P31's measurement — E2605: done and measured.** 1.8657 bits per character against 3.1343 for the
   character model, 3.3224 unigram and 4.8010 uniform on the same held-out documents, early stopping
   load-bearing, overfitting reported where it appeared, generations quoted because they loop.
3. **P33 — reproducibility of a pinned corpus.** A measured result must be trainable again: the driver writes
   the corpus snapshot beside the manifest, and preparation can consume it. Without this, a later commit
   either refuses or silently retargets a newer corpus.
4. **P34 — E2606: E2605 across pre-declared seeds**, mean ± sample SD, so the gap has an error bar.
5. **An external corpus with a recorded licence** — the data decision that bounds everything above — and only
   after it a capacity question.
6. **Then, and only then,** the application-side items this realignment deferred: P32 (the runtime acting on
   the sufficiency decision), a real `SemanticJudge`, and a grounded answer produced by a model rather than an
   oracle. Each is gated on its own dependency-feasibility decision, and none of them substitutes for the
   model path again.

All of it stays behind the existing boundaries. K-001 — that nothing has met a real workload — still stands
and is the reason E2605 was designed as a bounded comparison on documented self-authored text rather than a
capability claim.

## Known gaps that bound this unit

- K-001: offline and synthetic; a *useful* model on a real workload has still not been run. E2601 trained
  and measured a real (but toy) model, and E2605 trained and measured a 643,840-parameter Transformer on a
  real corpus — this repository's own 780,223 characters of pinned prose — which narrows the training half
  further. No real question was asked and no answer was grounded by a model, so the workload half is
  untouched and remains the frontier of the whole project.
- K-009: the numpy- and torch-gated tests do not run in remote CI, because the session's GitHub token
  lacks the `workflow` scope and the `[train]`/`[nn]` CI jobs cannot be pushed. Verified locally instead, in
  all three matrices: 1,391 passed / 16 skipped dependency-free, 1,472 with `[train]`, 1,605 with
  `[train,nn]`, and strict typing clean over all 142 source files in the `[train,nn]` environment. ADR-0034's
  `[nn]` tests inherit the gap by construction; no CI job for them is claimed. The type-checking half of this
  gap was closed as a defect found during delivery: the documented way of lifting the mypy exclusion did not
  work, so 27 modules had never been checked — see `configs/mypy-model-path.toml` and K-009.
- K-006: the invented-reference condition is persisted per answer, but nothing aggregates it into a
  rate, and it is only durable when the SQLite backend is selected.
- K-008: the fetcher's destination screen is partial; any acquisition work must confirm no default path
  reaches the network and record the limitation rather than fix it silently.

## What previous units knowingly left undone

- Automatic persistence of `CorrectionRecord`s and `ExperimentResult`s: produced and surfaced, not
  written to disk unless a caller wires a store in.
- The planner uses an anaphoric-cue heuristic, not a discourse resolver; and the reassessment trigger
  reads an explicit identifier, not natural-language intent (P21 was explicit composition, not autonomy).
