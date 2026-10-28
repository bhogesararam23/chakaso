# Changelog

Notable changes to Chakaso, newest first. Nothing has been *published*: the artifact described under 2026-10-22
is released **as an artifact under a contract and a gate** (Level 2 of ADR-0039), while its weights, its
corpus and its tokenizer stay out of Git and no version has been put on an index. Every entry below is
therefore pre-distribution, not pre-release in the sense of having no release at all.

Entries describe what changed and why it matters to someone using or evaluating
the project. They do not list edited files, and they do not describe progress
without a result. Where an entry records a capability, it states whether that
capability is implemented or still planned.

## Unreleased

### 2026-10-29 — a character split across two generated tokens was destroyed

**Fixed**

- Generation decoded each token on its own with `errors="replace"`. A character whose UTF-8 bytes span two
  byte-level tokens (a pound sign is `C2 A3`) came out as two U+FFFD characters in `generate`, in `stream`
  chunks and in anything computed from them. Found by sampling the released checkpoint (1 of 180 sampled
  continuations: `"ordifferent \ufffd\ufffd 1600"` against the correct `"£ 1600"` from a bulk decode). The adapter
  now decodes incrementally (`SubwordVocabulary.incremental_decoder`): a token that only starts a character
  yields an empty chunk, and the final stream chunk carries any flush. Greedy output was never affected on the
  recorded probes; a model that emits non-ASCII text was. Joined streamed text now equals the bulk decode.

### 2026-10-24 — the documentation was audited against the code, and the records were audited against their own reports

**Corrected**

- Four numbers in two committed experiment records did not match the reports those records aggregate from.
  E2606 said E2605 "stopped at step 1440 of 2,400" when the step E2605 **selected** is **1,380** (1,440 was the
  curve's argmin, which lost to `min_delta`), and called its observed range "11 SDs wide" where 0.018695 over
  a sample SD of 0.007702 is **2.4**. E2607's gap sample standard deviation was **0.004425**, not the
  transposed 0.004642 it printed, and its unigram-baseline gap range is **1.5832 to 1.6066**, not "1.5918 to
  1.6066". Each record now carries a dated *Corrections applied after the run* table that quotes the wording
  it replaced, because a silent edit to a measured record is exactly what the reproducibility rules exist to
  prevent. The aggregates themselves — `results.json` for E2606 and E2607 — are now committed beside the
  records that cite them, so the arithmetic is checkable without re-running either study.
- Three architectural rules that `docs/architecture.md` claimed were test-enforced were not: the CLI-import
  direction, ADR-0003's identifier ownership over the model path, and the `numpy` admission ADR-0030 states.
  They have tests now, each falsified by adding a violating module, and ADR-0030/ADR-0034 carry dated
  amendments naming them. The fourth rule the audit could not check mechanically — model metrics reported
  separately from harness metrics — stays open as **K-011**, and the unlabelled evidence-context rendering
  that design rule 6 implies is versioned becomes **K-012**.
- `docs/architecture.md` said prompt templates "are not versioned yet, because there is no prompt template",
  and `chakaso.runtime.context` said the same about there being no real model to prompt. Both are false since
  E2605: `chakaso.models.formatting` defines two versioned templates, the default being the shape every
  measured sample used. The sentences now say which half of a prompt is attributable in a record and which
  is not.
- [ADR-0039](docs/decisions/ADR-0039-three-level-model-release-gate.md)'s Level 1 evidence claimed a load of
  the release directory produces "the same model" as a load of the file it names; that rested on an equal
  `model_id` string. A vertical-slice test now compares what the two loads *emit* — the same bounded greedy
  continuation, identifier for identifier — and the record says when the claim got its evidence.
- **A released decoder no longer calls itself a Transformer trained on Chakaso's own documentation.**
  `chakaso.models.nntransformer` held one fixed display string for both checkpoint families, and it claimed the
  model was "trained on the project's own documentation" — false of `chakaso-base@0.1.0`, which is a rotary
  decoder trained on 919 documents of licensed public-domain prose. The string is printed by
  `chakaso model inspect`, served at `/v1/models` and stored in an answer record's `model_display_name`, so a
  false data-provenance claim was shipping inside the product. `metadata.display_name` now derives from the
  architecture the artifact declares (`Rotary Decoder` / `Subword Transformer`) and names no corpus at all: the
  corpus is the fingerprint inside the file and the licence line in its contract. The test was written first and
  failed against the old constant.
- The public documents were reconciled with what is measured: the record count, the third pinned corpus,
  E2608's result and its confound, the fact that a Level-2 release exists while its bytes are unpublished,
  the command count, and the limitations a release carries (now the artifact's own five, not a paraphrase).
- Six more entries went into [`docs/agent/KNOWN_ISSUES.md`](docs/agent/KNOWN_ISSUES.md), and the register's own
  stale entries were corrected. **K-013** — counts are copied into six documents and drift on the next commit,
  which is how "seven experiment records" and "12 commands" ended up in a README beside eight records and
  thirteen commands. **K-014** — this arc delivered to `main` directly while `CONVENTIONS.md` requires a branch
  and a pull request for anything that changes a documented interface, so either the rule or the practice is
  wrong. **K-015** — a loaded release reports the adapter's `model_id` (`subword_transformer`) while its own
  contract says `chakaso-base`; filed rather than changed, because that string is served and stored and the
  choice between the two identities is a decision. **K-016** — `chakaso model serve` is the one lifecycle
  command no test runs as a command and no CI job starts. **K-017** — E2601–E2604 commit no machine-readable
  evidence, and E2608's two arm reports are cited by digest alone. **K-018** — `--samples-per-prompt` counts
  tokens, not samples. K-001 was narrowed by E2606–E2608 and the release without pretending the workload gap
  had closed, and K-002 was found to be describing as missing the query planner, a dense ranker, durable
  persistence and the semantic-grounding boundary — all of which have had code for months, which is the reverse
  failure mode and the reason this entry was rewritten rather than ticked off.

### 2026-10-22 — the foundation release: `chakaso-base@0.1.0`, gated at Level 2 (nodes N–R)

**Released as an artifact, at Level 2 — see [MODEL_CARD.md](MODEL_CARD.md).**

- The reference model is **arm A of [E2608](experiments/E2608/record.md)**: a 3,413,760-parameter rotary
  decoder, trained from random initialization on 14,490,005 characters of public-domain English prose pinned by
  content digest (`e9fc2e92f59d88ea`), reaching **1.6865 bits per character** on held-out documents against
  unigram 3.2263 and uniform 5.5661, in 6,024 seconds on 14 CPU threads. The artifact directory —
  `E2608-decoder.pt` (SHA-256 `9e832fadf277e848…`), the run's `report.json` and a
  `chakaso-model-release/1` contract (`9aebe601dc1f12c5…`) — lives under `data/`, is cited by digest, and is not
  committed, per the artifact policy.
- **[ADR-0039](docs/decisions/ADR-0039-three-level-model-release-gate.md)** makes the gate explicit: Level 1
  (artifact contract) and Level 2 (reproducible integration) are **met with named evidence**; Level 3 (useful
  behaviour) is **not met and not claimed**. The measurement that withholds it is in the card and in the
  experiment record: five greedy continuations with mean bigram repetition 0.5538, zero tail loops, zero
  periodic runs of three or more, every sample truncated at its token bound, and no sample anyone would
  publish. Levels do not imply each other, and this release is the reason that sentence had to be written down.
- **Verified from a clean install, not from the working tree.** A wheel built from this head was installed into
  a fresh Python 3.13.5 environment and exercised through the packaged distribution: `chakaso model verify`
  checks the contract with no numerical library installed at all; `model inspect` in that state names the
  missing `[nn]` extra and exits 1 instead of answering from the double; after installing
  `.whl[train,nn]` (torch 2.14.1+cpu, numpy 2.4.6 — a minor version ahead of the development environment,
  within the declared bounds) the release loads, generates (` I am not a good man to be a good man.`), scores a
  held-out text at 5.0280 nats per token, and serves `/v1/models` and `/v1/chat/completions` over a real socket
  with its identity and the not-instruction-tuned note in the body. The corpus snapshot verifies through the
  installed CLI as well: 919 documents, 14,490,005 characters, matching its pinned fingerprint.
- The end-to-end sequence is now an automated test — `tests/test_vertical_slice.py`, eleven stages from a run
  inside the test through load, bounded generation, seeded sampling, streamed cancellation, scoring, release,
  verification and HTTP, plus a fresh-interpreter check that reaching a checkpoint does not import the
  conversation runtime — and it runs in CI's `model path` tier, so the joins between the parts are tested on a
  runner instead of re-checked by hand at release time.
- The handoff is a dated, prioritized list rather than a wish list: seven phases in
  [`docs/roadmap.md`](docs/roadmap.md) under *Handoff phases*, each with the exit criteria that would close it
  and the specific mistake it exists to prevent — including the note that E2608's 0.2178-bit gap between the
  arms is **not** a controlled architecture result, because the arms differ 3.3× in parameters and 2× in context.

### 2026-10-22 — a run now measures its own generations, its throughput and its memory (node L)

**Implemented**

- `chakaso.training.subword_experiment.run_experiment` writes three blocks it previously left to whoever came
  after it. `generation` is the output of `chakaso.training.generation_eval` run over the boundary against the
  artifact the run just saved — repetition ratios, longest repeated block, periodic runs, tail loops, length,
  termination and refusal behaviour, with the raw text retained and the interpretation limit carried inside the
  payload. `latency` records how many samples were produced, in how many seconds, at what character rate, and
  states the conditions of that measurement rather than implying a hardware benchmark. `resources` records the
  process's high-water resident memory from the operating system, or a named absence where the platform will
  not say.
- The suite distinguishes **refused** from **crashed**. It used to record any exception as `raised`, which made
  a model that correctly refuses to continue nothing look like a model that fell over: the boundary's typed
  `ModelError` is now a refusal, and anything else is a crash. Both are counted in the automated summary and
  neither is folded into the repetition statistics, because a sample that produced no text is not evidence of
  a fluent one. The empty-prompt probe's stated purpose was corrected to match the boundary's documented
  contract, which is that an empty request is refused by name.
- Measured with this wiring on the two E2608 arms: the decoder arm produced 5 continuation samples with
  **0 tail-looping and 0 periodic runs of 3 or more** (mean bigram repetition 0.5538), the transformer arm
  **0 tail-looping but 1 periodic run of 3 or more** (mean bigram repetition 0.7012); one edge probe per arm
  was a refusal, zero crashed. Both numbers and their raw text are in
  `data/experiments/E2608/generation-E2608-<family>.json`.

**Not claimed:** these are surface statistics. A lower repetition ratio is not a model that means anything, and
neither arm's samples are sentences anyone would publish.

### 2026-10-22 — the release lifecycle has a command line (node H)

**Implemented**

- `chakaso model inspect|verify|export|generate|stream|score|serve` — one group for the whole life of an
  artifact, every action a thin handler over the Python API rather than a second implementation of it.
  `export` turns a run's work directory into a distributable release directory: it copies the checkpoint and
  the report the run wrote, digests the copies against the originals, and assembles `model.json` from them, so
  provenance is read rather than typed. `verify` re-hashes a release against its own contract and needs no
  numerical library at all — a release can be audited on a machine that cannot run it. `generate` and `stream`
  take the boundary's decoding parameters as flags; `stream` cancels the run at its next step boundary on
  Ctrl-C rather than finishing an answer nobody is reading. `score` returns the same held-out cross-entropy the
  experiments report and refuses an adapter that has no measurement to make. `serve` starts the local HTTP
  endpoint on loopback and warns on the terminal when an operator names an address beyond it.
- `chakaso.models.export_release` and a `TextScorer` protocol at the boundary, so "this model can measure a
  text" is a shape a caller asks for instead of an attribute it hopes for.
- The first use of `model export` on this repository's own pilot artifact produced a refusal rather than a
  release: the pilot's report named the transformer schema for a decoder file, and the contract builder now
  rejects provenance that contradicts itself. That is the check working on the first real input, and it is the
  bug `8fc0c0d` fixed for every run after the pilot.
- **Not claimed:** `chakaso model serve` was verified by hand against arm A's artifact on this machine
  (`/health` reporting `chakaso-base@0.1.0`, a completion returned over a real socket), not in CI.

### 2026-10-22 — the developer-facing documentation catches up with the runtime (nodes N, P)

**Implemented**

- `README.md` now describes what a developer can actually do with a trained artifact — bounded decoding,
  streaming with cancellation, scoring, a release directory that verifies before it loads, and the optional
  local HTTP endpoint — each with a status label and a code block that was **executed** against the E2608
  pilot checkpoint rather than written freehand. Two things the snippets got wrong were found that way:
  `stop_sequences` is a tuple, not a list, and a work directory is not a release until a contract is written
  into it, which is exactly the distinction the release section now states.
- `tests/test_vertical_slice.py` walks the whole sequence in one run: prepare a corpus, train from random
  initialization, save, load through `chakaso.models.load_model`, generate under a bound, repeat at zero
  temperature, honour a seed, stream and cancel, score, build and verify a release, and answer over a real
  socket — plus a fresh-interpreter check that loading a model does not import the conversation runtime
  (ADR-0033). It is the test that would have caught the release-builder field-name defect and the serving
  loader's wrong function before they were written down as fixed.
- **What the slice does not claim:** the model in it is twelve steps of a one-layer decoder over five
  hand-written documents. Shape, not capability.

### 2026-10-22 — CI gained the two tiers that test the product (node M)

**Implemented**

- `.github/workflows/ci.yml` now runs three jobs instead of one: `checks` on Python 3.11/3.12/3.13 with
  `.[dev]` (the dependency-free guarantee, including the HTTP serving surface), `train` on the same three with
  `.[dev,train]`, and `model-path` on the floor and the latest with `.[dev,train,nn]` from the PyTorch CPU
  wheel index plus the strict type check over every module.
- The matrix existed before, as a document, because the session's GitHub token lacked the `workflow` scope
  and GitHub refuses a whole push that touches a workflow file. The scope is present now, so the file is the
  delivery and the document describes it; five hygiene tests read the shipped workflow rather than a copy —
  jobs and installed extras against `pyproject.toml`, every `run:` command against what `AGENTS.md` and
  `CONTRIBUTING.md` promise contributors, no mechanism that can hide a failure, `src/chakaso/serve/` free of
  `numpy` and `torch`, and the document refused while it still describes the matrix as un-applied.
- **Read back from the runner:** the first push of this workflow produced run `39702923101`, in which all
  eight jobs reached GitHub Actions — including both `model-path` jobs, which fetched the PyTorch CPU wheel on
  a Linux host for the first time — and all eight then failed the same step. What they found is the fix below,
  and a second run found an assertion finer than float32. Runs `39704678913` and `39706259071` are green across
  all eight jobs (core python 3.13: 1,644 passed / 24 skipped in 17.39 s; model path python 3.13: 2,003 passed
  / 2 skipped in 79.18 s), so the model path is now verified where the product runs rather than only where it
  is developed. K-009 is closed with those identifiers.

**Fixed**

- `chakaso.core.process_memory` reached for `ctypes.WinDLL`, a name that exists only in a Windows build of
  CPython, outside any platform guard. The runner is Linux, and all eight jobs failed the type check on it
  while every Windows gate passed it — found by CI on the first push, not by the author. The Windows reading
  is now behind a `sys.platform == "win32"` check that both a Linux and a Windows type checker can prune, the
  wrong-platform case returns a stated absence instead of raising, and its test patches one layer out so it
  runs on either operating system. `AGENTS.md` and `CONTRIBUTING.md` now tell a contributor to type-check the
  runner's platform from whichever machine they are on.

### 2026-10-22 — a trained checkpoint can be served locally over HTTP (node K)

**Implemented**

- `chakaso.serve` puts an OpenAI-shaped HTTP interface in front of one loaded model: `GET /health`,
  `GET /v1/models`, `POST /v1/completions` and `POST /v1/chat/completions`, with server-sent events for
  `stream: true`. It is the standard library's threaded HTTP server and nothing else — no framework, no new
  extra to install, and no third party between the request and the weights.
- It binds `127.0.0.1` unless an operator names another address, caps concurrent generations and answers a
  request over the cap with `429` and `Retry-After` rather than queueing it invisibly, refuses a body larger
  than the stated ceiling before reading it, and stops producing tokens when the client hangs up mid-stream —
  the disconnect cancels the generation through the same `AbortSignal` the Python API exposes.
- The adapter refuses the development double, and refuses the parts of the API shape it cannot honour
  (`n`, `best_of`, `logprobs`, `echo`, `user`, `suffix`) with the offending field named, instead of accepting
  them and answering a different question. A release directory is checksum-verified before a weight is read.
- **What is not claimed:** this has been exercised over a real socket against a scripted model and a real
  checkpoint loader, not yet end to end against a trained artifact; that is the vertical slice. There is no
  authentication, so a non-loopback bind is an operator's stated decision, not a supported deployment mode.

### 2026-10-21 — A measured result can be reproduced, and it was: E2605 across five seeds, and on prose nobody wrote about this project (ADR-0036, P33–P36)

**Measured**

- **E2606 puts a spread on the project's headline number.** Five pre-declared seeds (E2605's own seed, then the
  next four primes) over the identical corpus, split, tokenizer, architecture and budget: **1.8794 ± 0.0077
  bits per character** for the 643,840-parameter subword Transformer against **3.0937 ± 0.0267** for E2601's
  character architecture, with the unigram (3.3224) and uniform (4.8010) baselines identical across seeds
  because they depend on fixed data. Every seed beat every baseline; the gap on every seed was between 1.1828
  and 1.2685 bits. The seed effect is roughly 1/158 of the measured difference — so E2605's result was not the
  luck of one draw, but it *was* the most favourable of five, and the spread belongs beside the number.
- **E2605 was reproduced exactly.** Its seed-7 run returned the same corpus, split and tokenizer fingerprints,
  the same 643,840 parameters, the same selected step (1380), and held-out metrics identical to the last
  decimal on both architectures and both baselines. Loading both artifacts and comparing all 41 stored tensors
  returns equal, and their state digest matches: the model was reproduced, not merely its score. The one
  deviation is named rather than smoothed — the artifact *metadata* checksum differs because the environment
  block gained `numpy` and `torch_threads` between the two runs.
- **E2607 answers the question the internal corpus could not.** The same protocol, budget, tokenizer rules and
  metric code, run on three seeds of independently licensed prose — Project Gutenberg's Alice, Frankenstein and
  The Iliad, 105 documents, 1,660,274 characters — give **2.0176 ± 0.0118 against 3.2218 ± 0.0153 bits per
  character**, a gap of 1.2042 against 1.2143 internally. The advantage is not a property of Chakaso's own
  documentation. Absolute levels must not be compared across corpora (the subword tokenizer compresses 2.0829
  characters per identifier on the internal text and 1.8782 on the external one); gaps may, and they agree.
- **The text is still not usable, and now that is measured rather than asserted.** `chakaso.training.generation_eval`
  ran over all five internal and all three external checkpoints with the raw output retained: mean bigram
  repetition 0.4870–0.6904 internally and 0.5618–0.6956 externally, most samples stopping only because the
  token budget ended, and on the external corpus one sample degenerating into 38 characters of `OF` repeated
  and another coining words the corpus does not contain. An empty prompt is a typed `ModelError` naming the
  model and the reason on every checkpoint.

**Decided**

- **ADR-0036**: a recorded corpus is recovered from the revision that pinned it, never re-pinned from the
  working tree. The alternative — re-pinning when the tree moves — was rejected because it destroys the result
  rather than reproducing it, and "the corpus is what the fingerprint says" was rejected because a checksum
  that points at bytes nobody holds is not reproducible. The consequence is accepted: this project's
  reproducibility now depends on its history, which raises the cost of rewriting pushed commits.

**Added — reproducibility and measurement, in the dependency-free tier where possible**

- **Implemented** `chakaso.training.snapshot`: recover the exact corpus a manifest pins from a named revision
  of the local history, materialize it, run the same `verify_corpus` check a fresh preparation runs, and write a
  snapshot whose acquisition block names the commit object the bytes came out of. A revision one documentation
  edit away from the pin is refused with the same drift error the stale working tree gets, so this is a way to
  *find* the bytes and not a way to talk the guard out of its job. `chakaso corpus restore` and
  `chakaso corpus verify` reach it from the terminal; both are offline.
- **Implemented** the snapshot format at schema/2: per-document digests beside the bodies, a checksum over the
  file's own content, an acquisition block that distinguishes historical bytes from the current tree, and
  refusal of a document set that is missing, duplicated, unexpected or altered. ADR-0035's overwrite guard now
  covers corpus artifacts and experiment reports too, and `chakaso.core.atomic` stages every one of those
  writes so a process that dies mid-write leaves no half artifact — and no newline translation, which is what
  makes a written report's bytes the text its cited digest was computed over.
- **Implemented** `chakaso.training.seed_study`: declared seeds over one protocol template, one directory per
  run, aggregation rebuilt from the reports on disk (mean, sample standard deviation, median, range, with no
  confidence interval published and the reason stated), failed and missing runs named rather than averaged away,
  and a reused report accepted only after its configuration, corpus fingerprint and seed match the study that
  claims it. `chakaso experiment study --collect-only` rebuilds a published summary without training anything,
  which is how an outside reader checks one.
- **Implemented** `chakaso.training.source_corpus`: turn a downloaded licensed text into a pinned corpus
  offline — cut by the source's own wrapper markers, chunked at whole-paragraph boundaries, recorded with URL,
  byte digest, retrieval date and the licence in the file's own words. Nothing in the training path reaches the
  network; E2607's corpus bytes live in `data/` per the artifact policy, and what is committed is the
  acquisition.
- **Implemented** `chakaso.training.generation_eval`, over the `LanguageModel` boundary rather than behind it,
  so the same probes run against the development double, the character checkpoint and the Transformer. It
  carries its interpretation limit inside its payload: repetition ratios, cycles and termination are surface
  statistics, and judging whether a continuation means anything is a separate, human act.
- **Made explicit in the experiment driver**, because a study needs them to be comparable: `split_seed`, so a
  seed study varies initialization without re-dealing the held-out documents (without it, "the effect of a
  seed" would be the effect of a seed *and* a different test set); `expected_corpus_fingerprint`, so a run
  cannot train on merely *a* pinned corpus; and per-run experiment and corpus naming, so an E2607 directory
  does not hold files named for the repository's own text.

**Fixed**

- **A measured result could not be re-trained at all** — the driver never wrote a snapshot, so the moment the
  repository edited its own documentation the pinned corpus became unreachable and `verify_corpus` correctly
  refused. Now refused-or-reproduced are the only two outcomes.
- **`chakaso experiment model` let a typed corpus failure escape as a traceback.** `TrainingError` does not
  derive from `ChakasoError`, so the drift refusal and the overwrite guard that were already reachable from
  that command were never reported as the refusals they are.
- **A resumed seed could not replace its own interrupted run**: the retry was recorded as a failed seed because
  it refused to overwrite a checkpoint its earlier pass had left behind. A seed with no report now may; a seed
  with one still needs an explicit declaration, and the summary line no longer crashes on a study that has
  completed nothing.
- **The tail-only loop rule missed the case it exists to catch**, found by running it on E2605's own output: a
  sample that repeats a block eight times in its middle and ends elsewhere reported no loop. The suite now
  records both the tail verdict and the longest cycle anywhere, because they answer different questions.

**Verification and CI**

- All three dependency tiers are green locally at this head (dependency-free, `[train]`, `[train,nn]`), with
  strict typing clean over every source file in the `[nn]` environment. The counts and the delivery record are
  in `docs/agent/ACTIVE_TASK.md`.
- **The optional-dependency CI matrix is written, tested against the documented commands, and still not
  running.** `docs/internal/engineering/optional-dependency-ci-matrix.md` carries a complete three-tier
  workflow; four hygiene tests assert that its jobs install exactly the tiers they claim, that every command it
  runs is one `AGENTS.md` documents, that nothing can fail quietly, and that the *shipped* workflow still covers
  only the dependency-free tier. It cannot be pushed: this session's token has no `workflow` scope, and GitHub
  refuses the whole push rather than part of it. K-009 records the scope, the fact that the branch is
  unprotected and Actions enabled, and the exact step a person with that scope has to take. No claim is made
  that any of it ran on a runner.

**What this does not change**

Nothing here is a capable language model, a workload, or an answer to a question. Two corpora are now measured
on instead of one, which is a real widening of the evidence and not a step toward usefulness: no number in this
repository comes from a task anybody needed doing (K-001), the runtime's default model is still the
development double, and the next budget question is now constrained by a measured fact — on the larger corpus,
nothing converged within 2,400 steps.

### 2026-10-21 — There is a trained model: a subword Transformer, loadable from Python, measured against the character model (ADR-0035, P31, E2605)

**Decided**

- **ADR-0035** fixes the checkpoint format: one file holding tensors, canonical metadata and a SHA-256 over
  both, with the tokenizer stored inside it so a checkpoint cannot be loaded against a vocabulary that did
  not produce it. It separates a *usable* artifact (parameters plus metadata, what `load_model` reads) from a
  *resumable* one (parameters plus optimizer moments plus step), refuses to overwrite an existing artifact
  unless told to, and extends the `torch` module allowlist to exactly the modules that need it.

**Added — the model, its training, its checkpoint (the optional `[nn]` extra)**

- **Implemented** `chakaso.training.transformer`: a decoder-only causal Transformer (pre-norm blocks, GELU,
  learned positional embeddings, multi-head causal self-attention with an additive mask, input/output
  embeddings tied) whose parameter count is computed from the configuration and cross-checked against the
  built model, because tied weights make a byte-level sum over stored tensors overstate it. Initialization is
  named and seedless-by-default: zero-mean normal (std 0.02) weights, zero biases, unit normalization scales.
- **Implemented** `chakaso.training.nn_training`: mini-batch AdamW training with linear warmup, gradient
  clipping that records the pre-clip norm, and validation-selected early stopping. The batch order is a pure
  function of `(total, update_index, batch_size, seed)`, so a resumed run replays the same batches and
  reaches bit-identical parameters; the loop applies the selection rule incrementally and then refuses to
  return if it disagrees with `choose_validation_best`, the pure form of the same rule. Training loss and
  held-out loss are always separate numbers, and the held-out side is supplied by the caller, so this module
  cannot score the documents it trained on as unseen.
- **Implemented** `chakaso.training.nn_checkpoint`: save and load for the format ADR-0035 defines. Loading
  uses `weights_only=True`, verifies the checksum, and refuses an artifact whose schema, tokenizer,
  architecture or tensor shapes do not match what it claims — an incompatible checkpoint is an error naming
  the mismatch, never a silently substituted model.
- **Implemented** `chakaso.training.model_metrics`: a model's held-out loss expressed three ways — nats per
  identifier, perplexity, and **bits per character**, whose denominator is the decoded whole target stream.
  Bits per character is the only figure that compares a 116-symbol character model to a 1,024-symbol subword
  model at all, so it is the headline and perplexity is labelled tokenizer-bound. One function computes all
  three for both, against an `IdentifierDecoder` protocol rather than a class.
- **Implemented** `chakaso.training.nn_evaluation` (score a built model or a saved artifact over packed
  examples) and `chakaso.training.subword_experiment`, the driver that turns the model path into a measured
  comparison: pin the corpus, train both tokenizations, split by document, pack, train the Transformer and
  the character baseline, score both plus the trivial predictors on the same test positions, sample
  generations through the public loading path, and write a deterministic report with its own digest.
- **Implemented** the primary interface ADR-0033 demands: `chakaso.models.load_model(path)` returns a
  `LanguageModel` from a checkpoint, with no terminal, no configuration file and no application import. The
  concrete adapter (`chakaso.models.nntransformer`) is reached through the registry's one selection point,
  declares itself not a development double, generates greedily or by sampling, honours the document separator
  as a stop condition, and scores a text in nats.
- **Reachable from the terminal as a thin handler**: `chakaso experiment model --work-dir … [--max-steps …
  --json --write-report …]`, which delegates to the library and prints what the run measured. Heavy
  dependencies are imported lazily, so the dependency-free install reports what is missing instead of
  failing at import.
- **Test totals measured on 2026-10-21**: 1,407 collected in the dependency-free matrix (1,391 passed, 16
  skipped — up from 1,276 collected before this work), 1,479 with `[train]` (1,472 passed, 7 skipped) and
  1,606 with `[train,nn]` (1,605 passed, 1 skipped). The torch-gated ones inherit K-009's gap — they skip in
  the dependency-free matrix that CI runs, and no CI job installs `torch`.

**Fixed — the model path had never been type-checked, and now it is**

- `pyproject.toml` excludes the optional-dependency modules from `mypy src` so the dependency-free matrix
  stays green, and the documented way of checking them anyway — `mypy src --exclude '_never_match_'` — does
  not work: mypy appends a command-line exclusion rather than replacing the configured one, and directory
  arguments are subject to it too. So 27 modules — the entire `[train]` and `[nn]` path — were unchecked in
  every environment, including this project's own gate runs. Checking them for real reported three dead
  `# type: ignore` comments and one genuine typing hole in the byte-pair encoder's pair selection (a value
  only the code guaranteed to be an `int`), both corrected. `configs/mypy-model-path.toml` is now the command
  that type-checks all 142 source files under the project's strict settings — clean where both extras are
  installed, an honest `import-not-found` failure where torch is absent — and two hygiene tests keep the two
  configurations' strictness identical and pin the exclusion list to the modules it names.

**Measured**

- **E2605 result** ([`experiments/E2605/record.md`](experiments/E2605/record.md), with
  [`results.json`](experiments/E2605/results.json) and the corpus manifest committed beside it): on held-out
  documents of the pinned 71-document corpus, the Transformer trained from random initialization reaches
  **1.8657 bits per character** against **3.1343** for E2601's character architecture, **3.3224** for the
  unigram baseline and **4.8010** for uniform over the identifier space. Early stopping was load-bearing: the
  selected step was 1380 of a 2400-step budget and the un-stopped model was 0.1434 nats worse on held-out
  text. The three generation samples in the record loop ("a model that is a model that is a model") and are
  quoted as evidence of what the model is *not*. One seed, one author's prose, no error bar.

**Status and honesty**

- The record was committed before the run, and its pre-run text quoted a corpus figure (754,789 characters)
  that the run's own pinning superseded (780,223). That correction is written into the record rather than
  quietly edited, and the README and this changelog are fixed in the same commit; the research-log entry that
  measured the earlier figure stays as the dated observation it was, with a new entry recording the growth.
- A model now exists, and nothing about the application changes: no runtime path uses it as a default, P29
  (calibration) and P32 (runtime consumption) stay deferred, and the strong form of K-001 — no model has ever
  produced a grounded answer here — still stands.

### 2026-10-21 — Chakaso is a language-model project, and now has a real corpus to train on (ADR-0033/0034, P30)

**Decided**

- **ADR-0033** states the product contract: the product is a trained model plus the reproducibility
  machinery around it; the primary interface is the Python library — a checkpoint must be loadable and
  generatable with no terminal, no config file and no runtime import; the CLI is a secondary developer
  utility whose handlers stay thin; model evaluation is separate from harness evaluation, so a retrieval
  benchmark or a reference oracle can never be cited as language-model competence. It inverts the
  roadmap's "system before model" rule and, on the record, defers P29 and P32 rather than completing them.
- **ADR-0034** admits `torch` as a second optional extra, `[nn]`, for the model path only. It reopens
  ADR-0030 on that record's own revisit conditions with measurements: the existing full-batch loop costs
  1.87 s/step at a 100 k batch and materializes 1.04 GB of probabilities at corpus scale, and verifying
  every scalar of the target model by finite differences extrapolates to ~45 hours. It records honestly
  that the speed argument did **not** survive measurement (1.16× on a same-shape float32 matmul), that the
  Windows process-memory APIs gave no peak-RSS number from this shell, and that the `[nn]` tests inherit
  K-009's gap: they skip in the dependency-free matrix and no CI job for them is claimed.
- The measured environment and candidate comparison are in
  [`docs/research/next-model-experiment.md`](docs/research/next-model-experiment.md), and the diagnosis of
  how the repository drifted is in
  [`docs/internal/engineering/model-centred-audit.md`](docs/internal/engineering/model-centred-audit.md).

**Added — the model path's data layer, dependency-free**

- **Implemented** `chakaso.training.manifest`: a corpus named by path and content digest, with licence,
  source, normalization step and size bound recorded. `verify_corpus` checks the corpus *as a set* — an
  edited, deleted or newly-matching file is drift and is named, so a saved result cannot be restated
  against text the model never saw. Digests are taken after CRLF→LF normalization, so a contributor's git
  line-ending setting cannot move a corpus fingerprint. Snapshots keep a run's exact bytes after the
  tracked text moves; duplicate content is reported.
- **Implemented** `chakaso.training.data.split_text_units`: a document-level three-way split that
  deduplicates by content *before* selecting sides, because a byte-identical document otherwise trains
  twice and can straddle the boundary it is supposed to sit behind. Sides, an identifier map and a
  fingerprint over the exact assignment are carried on the result.
- **Implemented** `chakaso.training.subword`: a byte-pair subword vocabulary trained by this project on
  the text it is given. Byte-level base, so encoding is total and no character is unknown; a named
  pre-tokenizer, so merges never cross a word boundary and segmentation is reviewable; a reserved
  document-separator identifier allocated before any merge; deterministic tie-breaking, so the same
  training text always produces the same fingerprint; and a self-describing, checksum-verified record whose
  load refuses an altered merge table or a foreign format.
- **Implemented** `chakaso.training.packing`: one side of a split becomes fixed-length next-token
  examples, with every document terminated by the separator, targets shifted by exactly one token, long
  documents truncated into blocks rather than discarded, and the dropped tail reported rather than hidden.
- **Measured on the real corpus** (this checkout, 2026-10-21, as re-measured by the run that pinned it): 71
  tracked documents, 780,223 characters, manifest fingerprint `e45993f9badf706c`; a 49/11/11 document split
  with no duplicate content dropped, split fingerprint `87067cca96275def`; a 1,024-entry vocabulary (767
  merges, fingerprint `2605bf8f50183a6e`) trained on the training side alone, tokenizing the sides into
  240,418 / 82,376 / 46,577 identifiers at 2.0829 characters per identifier on the test side. Pinning,
  tokenizing and packing took 13.2 s. An earlier scratch preparation of the same include spec measured 69
  documents and 754,789 characters; the corpus grew as this work's own documentation was committed, which is
  exactly why the manifest pins content rather than a path list.
- **98 new tests**, all of them in the dependency-free matrix (1366 passed, 8 skipped there, up from 1268;
  1418 passed with `[train]` installed), including leakage tests written so they can actually fail: a
  held-out document carries a token no other document has, and its appearance anywhere in the training side
  is a failure.

**Corrected**

- `docs/training.md` ended with a status table saying the tokenizer, dataset pipeline, model code, training
  loop and checkpoints were "Not started" and that "Nothing in this document has been executed", while its
  own header cited the measured E2601. The table now names what is implemented, what is experimental and
  what is still P31. The README's status rows and the roadmap's ordering rule are realigned the same way.
- The `chakaso.training` package docstring still said "No trained Chakaso model exists yet"; it now states
  the three dependency tiers and that the measured result is a learning loop on a toy, not a useful model.

### 2026-10-19 — Retrieval can now tell "no evidence" from "k candidates" (P28, E2604)

**Added / measured**

- **Implemented** `chakaso.retrieval.sufficiency` (ADR-0032): a deterministic, offline, numpy-free
  **evidence-sufficiency decision** at an additive seam over retrieval — `Retriever`/`RetrievalService`
  keep returning top-k; a policy decides whether the pack is *adequate*. The one policy, discriminating-term
  overlap, calls a pack sufficient only if a retrieved chunk shares a **non-function-word** query term that
  is rare in the corpus — the signal that actually separated E2603's failures (junk matches on `the`/`to`/
  `library`) from real ones.
- The grounded evaluation runner, reports and `chakaso benchmark grounded --sufficiency` now score sufficiency
  as a **separate dimension** (accuracy, false-abstentions, missed-abstentions), reported per case and never
  merged with retrieval or answer metrics.
- **E2604 result** ([`experiments/E2604/record.md`](experiments/E2604/record.md)): on the self-authored
  workload the policy is **100% accurate** — all three unanswerable false-retrieval cases judged insufficient,
  **zero** answerable cases wrongly abstained (recall preserved) — while retrieval's `false_retrievals`/
  `forbidden_hits` metrics and the runtime are intentionally unchanged (the decision is measured, not yet wired).

**Status and honesty**

- This narrows K-010, it does not close it: the runtime does not yet act on the decision (P32), the policy
  depends on a narrow corpus-specific `max_document_fraction` band (~0.23–0.31 on 13 documents) that P29 must
  stress-test rather than generalise, and forbidden-evidence hits are a ranking problem it does not address. It
  is not a calibrated confidence, and with no model answering, K-001 stands.

### 2026-10-19 — A grounded-answer evaluation workload exists and is measured (P27, E2603)

**Added / measured**

- **Implemented** a grounded-answer evaluation sub-module in `chakaso.benchmark` (ADR-0031): a
  provenance-bearing case/source schema, a dataset that refuses dangling evidence, duplicate cases and
  development/held-out split leakage, and a strict manifest+JSONL loader.
- A **self-authored, Apache-2.0, human-inspectable** workload (`ga-kestrel-library`, 13 passages / 14
  cases, pinned fingerprint `d2d2ee00d117b88a`) spanning direct, multi-source, distractor, conflicting,
  ambiguous, missing-evidence and unsupported-premise questions — each with required facts, prohibited
  claims, acceptable/unacceptable evidence, a rationale and a documented licence. Sources are authored
  for the project, not scraped, so redistribution is provably permitted and the pipeline stays offline.
- `GroundedEvaluationRunner` runs the real retrieval + `evaluate_answer` boundaries over an injected
  answer provider, classifies retrieval / generation / evaluation failures separately, reports retrieval
  and answer dimensions separately (no merged score), and emits deterministic text/JSON reports.
  Reachable as `chakaso benchmark grounded --provider … --split … [--json]`.
- **E2603 result** ([`experiments/E2603/record.md`](experiments/E2603/record.md)): with a *reference
  oracle* (an oracle, not a model) over the workload — retrieval recall@k 0.8182 / MRR 0.8636 (held-out
  recall@k 1.0), zero stage failures — and honest negative findings kept visible: 3 forbidden hits and 3
  false retrievals, because lexical retrieval returns its top-k even on questions it should leave empty.

**Status and honesty**

- This is a **development instrument**, not a claim about answer capability. The reference provider is an
  oracle that validates the harness; the runtime still has no model that produces a grounded answer, so
  the strong form of K-001 stands. E2603's negatives set the next unit (a retrieval evidence-sufficiency
  decision), and its per-case report prevents any aggregate from hiding a systematic failure.

### 2026-10-19 — The training result is reproduced across seeds under a controlled protocol (E2602, P26)

**Added / measured**

- `chakaso.training.data.split_three_way` — a deterministic, disjoint train/**validation**/**test**
  separation (each line unit wholly on one side; every unit preserved exactly once), so a
  model-selection rule can read validation without contaminating the reported test number. The vocabulary
  stays fit on training text only; `fit_vocabulary` is now structural (works for the two- and three-way
  shapes via a read-only `train_text` protocol).
- `chakaso.training.controlled.run_controlled_experiment` — a fixed protocol (same model/optimizer as
  E2601, pre-declared seeds (7, 21, 1337, 2024, 42)) with **early stopping**: select the first step
  achieving a new minimum *validation* loss and report the test loss there; aggregate mean + sample
  standard deviation across seeds against the same uniform/unigram baselines.
- **E2602 result** ([`experiments/E2602/record.md`](experiments/E2602/record.md)): mean held-out **test**
  cross-entropy **1.7978 ± 0.1767 nats**, below both baselines on every seed (unigram mean 2.9808, uniform
  ~3.3); the un-early-stopped step-300 test loss (mean 2.7804) is worse on every seed — E2601's
  overfitting is systematic, not one lucky partition.

**Status and honesty**

- The result is a **language-modelling toy with an error bar**, not a capability claim: five seeds cut
  sampling noise, not the corpus's unreality, and the validation/test sets are small (77–93 positions). No
  wall-clock/memory measured. It advances the roadmap's training sprint and narrows K-001's training-loop
  half further (a *reproducible, multi-seed* result), but K-001 stands and no hypothesis (H1–H7) is
  tested. The evidence now points toward breadth (a real evaluation corpus) rather than more of the same.

### 2026-10-19 — A trained model is integrated behind the boundary (P25) and checkpoints carry their vocabulary

**Added**

- `chakaso.models.trainedchar.TrainedCharacterModel` — a second `LanguageModel` adapter (selected by
  `model.adapter = "trained_char"` with a `model.checkpoint_path`) that loads an E2601-class trained
  character checkpoint, rebuilds the vocabulary bound to those weights, and generates greedily one
  character at a time within `max_new_tokens`. It declares `development_double = False` and implements
  the `TOKENIZE` capability, and it fails explicitly — never silently falling back to the double — when
  the path, the `[train]` dependency, or the checkpoint is missing or invalid.
- The registry imports this adapter **lazily**, so `import chakaso.models` and a plain install stay
  dependency-free; only selecting `trained_char` needs `numpy`. `configs/default.toml` and the typed
  schema gain `model.checkpoint_path` (empty by default), so nothing loads weights unless asked.
- `chakaso.training.checkpoint` now embeds the vocabulary characters (schema `chakaso-checkpoint/2`),
  validating on load that they recompute to the stored fingerprint — a checkpoint that could only be
  checked, not decoded, was unfit for inference. This changes no measured value: E2601's held-out losses
  are byte-identical after re-running; only its checkpoint content fingerprint moved (the record updated).

**Status and honesty**

- The integrated model is still a **tiny toy** — its generated text is incoherent, and the adapter is
  never the default. Its value is a real training-and-inference path behind the boundary, not usefulness;
  the adapter's own display name and this documentation say so. The adapter satisfies the same shared
  `LanguageModel` contract as every implementation (tested over a real trained checkpoint). No useful
  language model exists; K-001's stronger form (a real workload) stands, and K-009 still records that
  the numpy-gated training/adapter tests do not run in remote CI (the token lacks the `workflow` scope).

### 2026-10-19 — The first model is trained and measured (experiment E2601, P24)

**Added / measured**

- `chakaso.training` — the from-scratch training pipeline the module ADR-0005 reserved, now real: a
  self-authored Apache-2.0 corpus (`chakaso-toy-char-v1`); a deterministic character vocabulary (ids by
  codepoint order, a reserved unknown id, a content fingerprint); a **leak-free** line-unit split that
  fits the vocabulary on training text only; a validated `TrainingConfig`; a tiny causal character model
  (embedding → `tanh` hidden → softmax) with **hand-derived gradients verified against central finite
  differences for every scalar**; an in-project Adam optimizer; a full-batch training loop that records a
  held-out curve; a content-checksummed JSON checkpoint that refuses corruption and an incompatible
  schema/shape; uniform and unigram baselines; held-out evaluation (loss / perplexity / bits-per-char /
  accuracy); and an experiment driver. The dependency-free layers run in core CI; the numpy layers are
  gated on the optional `[train]` extra (ADR-0030).
- **E2601 — the project's first genuinely measured result** ([`experiments/E2601/record.md`](experiments/E2601/record.md)):
  training the 984-parameter character model from random initialization lowers **held-out** next-character
  cross-entropy from 3.3118 nats (random init) to **1.6650 nats at its best (step 50)** and **2.4946 nats
  at the end (step 300)** — below both the uniform baseline (3.3322) and the unigram baseline (3.0531);
  final bits/char 3.5989, perplexity 12.12, next-character accuracy 0.6026. Environment: Windows 11,
  CPython 3.13.5, numpy 2.4.6, CPU-only, seed 7; reproducible from the seed and config.

**Status and honesty**

- This is a **learning-loop demonstration**, not a useful language model: a 344-token synthetic corpus,
  one seed, 78 held-out positions (a high-variance point), and the model **genuinely overfits** after
  step 50 — the held-out curve turns up and the record reports that as a measured finding, not a hidden
  one. The trained artifact is **not** the default model — the runtime's default remains the development
  double — though the P25 `trained_char` adapter (later the same day) can now load such a checkpoint
  behind the boundary, still optional and still a toy. No wall-clock or peak-memory number is claimed
  (none was measured to stated precision).
- `chakaso.training` is a numpy-dependent, optional-extra module; importing `chakaso` or `chakaso.training`
  still does **not** require numpy. The `[train]` CI job ADR-0030 anticipated could **not** be pushed (the
  session's GitHub token lacks the `workflow` scope), so the numerical tests run locally / where the extra
  is installed but not in remote CI — recorded as **K-009**. K-001 is narrowed for the training loop but
  stands in its stronger form: no real workload, no useful model.

### 2026-10-19 — The training-dependency decision is made (ADR-0030, P23 — decision, no code)

**Decision / research**

- [ADR-0030](docs/decisions/ADR-0030-minimal-optional-training-dependency.md) records the decision P22
  identified as the gate on any real model: admit **NumPy as a single, optional `[train]` extra**
  (`numpy>=1.26,<2.5` — the upper bound keeps the Python 3.11 floor installable) and derive the tiny
  model's gradients **by hand, verified with finite-difference tests**, rather than adopting a heavy
  autograd framework. The base install stays dependency-free; the extra never enters the core CI matrix,
  and training tests `importorskip("numpy")` so they are exercised remotely only by a separate optional
  CI job. Training code is placed in `chakaso.training`, the module ADR-0005 reserved for it.
- It evaluates the candidate stacks against the project's constraints (Python floor, Windows, offline
  reproducibility, CPU-only, memory, footprint, licensing, determinism/seed control, verifiability, CI
  isolation), cites PyPI packaging metadata for the facts it relies on, rejects PyTorch **for now** with
  explicit revisit conditions, and keeps a pure-Python route as a documented fallback. **No dependency was
  installed and no code was written** — this is the ADR-before-code gate the roadmap requires. It advances
  the project toward its first genuinely measured training result without claiming any result yet; K-001
  stands.

### 2026-10-19 — Repository documentation reconciled against the actual head (P23 pre-work)

**Documentation**

- Corrected stale and contradictory status text before extending the system: `AGENTS.md` no longer claims
  "no measured results of any kind" (it now distinguishes the executed synthetic E-001 measurement from the
  still-absent real-workload and trained-model results); `docs/roadmap.md` re-assesses P1–P7 against
  objective exit criteria rather than file presence; `docs/agent/CURRENT_STATE.md` fixes its aged test
  count and points its next-step at the true current milestone; `docs/internal/experiments/README.md` is
  aligned with the fact that E-001 ran while no experiment *record* or hypothesis test exists. No behaviour
  changed.

### 2026-10-15 — Model-feasibility assessment recorded (research, no experiment run)

**Research**

- `docs/research/model-feasibility.md` inspects the available machine directly (Python 3.13.5, 18 logical
  CPUs, ~15.4 GB RAM with ~1.4 GB free at inspection, Intel Arc — no CUDA, no ML library installed) and
  assesses two directions for putting a real model behind the `LanguageModel` boundary: a runnable small
  pretrained model (Option A) versus a minimal from-scratch training experiment (Option B). It separates
  verified observations from explicitly labelled estimates and assumptions, and **runs no experiment, adds
  no dependency and reports no measured result**.
- Recommendation (weak, pending a dependency ADR): defer Option A until a real grounded-answer evaluation
  exists; the smallest honest next step is a scoped, out-of-CI from-scratch experiment that produces the
  project's first genuinely measured number, gated behind a documented decision to admit an *optional*
  numerical dependency. Logged in `docs/research/research-log.md` and it narrows (does not answer) the
  "which local model" open question. K-001 stands.

### 2026-10-15 — A follow-up that names a stored answer can trigger an explicit reassessment (P21)

**Added**

- `chakaso.correction.trigger`: a deterministic, inspectable **reassessment trigger** that reads a
  follow-up for an explicit, validated `ans_…` reference and classifies the turn as naming zero, one, or
  several distinct stored answers. It detects *intent only* — it reads no store, runs no evaluator,
  retrieves no evidence and mutates nothing — so intent detection is independently testable from
  execution. A vague reference is never inferred, and two distinct identifiers are treated as **ambiguous**
  rather than resolved by a guess.
- `chakaso.correction.errors.AmbiguousReassessmentTargetError`, alongside the existing `UnknownAnswerError`,
  as the typed refusals the composition raises instead of guessing a target.
- `Application.reassess_followup` composes the trigger with the existing `reassess_and_record` boundary at
  the application root (ADR-0026/ADR-0029): a named, recorded answer is re-grounded against its **standing
  evidence** (the corpus chunks still present for the sources it named) merged with whatever the follow-up
  retrieves through a new **read-only** `RetrievalAwareConversation.gather_evidence` seam (no model call,
  no answer recorded, no conversation or store mutation).
- `chakaso chat` now routes a follow-up that explicitly names a stored answer into reassessment and
  reports the decision honestly, and after each recorded turn prints the answer's `ans_…` identifier on
  standard error so a later turn can name it — making the path reachable from the supported runtime
  interface, not only callable in a test.

**Notes**

- A reassessment is a **distinct append-only `CorrectionRecord`**; the original `AnswerRecord` and its
  historical `evaluation_provenance` (ADR-0028) are never rewritten, and **no revised prose is generated**
  because there is no language model — the record's summary is a structural tally. This is explicit
  composition, **not** an autonomous correction loop: nothing decides on its own that a turn needs
  revising, and a correction record is surfaced, not auto-persisted. Recorded in
  [ADR-0029](docs/decisions/ADR-0029-explicit-reassessment-trigger.md).

### 2026-10-15 — Public documentation reconciled with what the code actually does

**Changed**

- `README.md` no longer describes the shipped system as lexical-only retrieval with an
  in-memory-only answer store. It now states what exists and is tested: a dense retriever and a
  hybrid reciprocal-rank-fusion retriever behind the same boundary (over a **non-semantic** fixture
  embedding, ADR-0023/ADR-0024 — labelled Experimental), the retrieval-aware runtime and
  `chakaso.app` composition root that ground `chakaso chat` (ADR-0022/ADR-0025/ADR-0026), and the
  durable SQLite answer store selected by configuration (ADR-0021). `Dense embeddings, vector index
  and reranking` was split into what is implemented (the dense/hybrid boundary and exact index) and
  what remains Planned (a real semantic embedding, an approximate index, reranking).
- `docs/correction.md` and `docs/getting-started.md` corrected their stale persistence and
  retrieval claims; `docs/agent/CURRENT_STATE.md`'s ADR range was fixed to ADR-0001–ADR-0028.

**Notes**

- No behaviour changed; this is a documentation-truth pass. Every command and output shown in the
  README was run against the repository before being written down, and the E-001 strategy comparison
  was reverified (dense/hybrid regress MRR from 1.000 to 0.929 over the synthetic fixtures — a −0.071
  delta, reported as a measured development result, not a model-quality claim).

### 2026-10-15 — Answer inspection states that recorded references are not retained evidence

**Added**

- `chakaso answers show` now discloses, in the human-readable view and as an
  `evidence_retention` object in `--json`, that an answer's evidence and citation references are
  **identifiers only**: the durable store keeps what the turn decided and which evidence it named,
  not the evidence text, which lived in the ephemeral in-memory corpus and is not re-readable from
  the store. A recorded reference is not the source content, and the output says so plainly
  (ADR-0028, `docs/retrieval.md`).

**Notes**

- This is a disclosure of an existing boundary, not new persistence: no evidence bodies are copied
  into the answer store (weighed against size and privacy, and unjustified without a real consumer),
  and no capability is claimed. It closes the honest-representation gap so inspection cannot mislead
  a reader into thinking a cited document is recoverable from the answer database.

### 2026-10-15 — A persisted answer records the evaluation its turn actually reached

**Added**

- `chakaso.answers.provenance.EvaluationProvenance`: a typed, immutable, serializable snapshot of
  the evaluation a completed turn produced — the citation-structure counts (valid / unknown /
  irrelevant / uncited-of-required), the per-claim **structural-grounding** status the turn assigned
  and which claims were left not-evaluated, the **evaluator name and version** that decided them,
  and the evidence **references in the answer text that did not resolve** (unresolved and malformed).
  It is persisted on every `AnswerRecord` as `evaluation_provenance`, and captured by the
  conversation manager from the evaluation and citation resolution the turn already computed — so no
  new evaluator runs and the development double gains no fabricated capability (the evaluator a
  default record names is the honest `structural-0.1`).
- `chakaso answers show|list` now report that historical snapshot, explicitly labelled *recorded at
  the turn — not recomputed on read*, with the dimensions kept separate and `is_grounded` /
  `evidence_coverage` stated as structural, never a truth verdict. `--json` carries the exact stored
  values; the human-readable view renders any untrusted string (an evaluator name, an answer text,
  a reference id) terminal-safe.

**Changed**

- The answer codec persists the new field **inside the JSON document**, so the SQLite table and
  `PRAGMA user_version` are unchanged and no schema migration is required. A document written before
  this feature simply lacks the key and decodes to `None` (an honest "no provenance was captured",
  never an empty snapshot that reads as "evaluated and found lacking"); a present-but-malformed
  snapshot is rejected as corruption, consistent with the strict-read storage rule.

**Notes**

- The **live** `AnswerEvaluation` stays recomputed and unpersisted, exactly as ADR-0021 required —
  ADR-0028 preserves that principle rather than reversing it. A historical fact (what the turn
  decided, by whom) and a recomputable view (what a current evaluator decides now) are now two
  distinct fields, so reassessment and a later evaluator never silently rewrite the original
  outcome; a reassessment is a separate append-only event. Recorded in
  [ADR-0028](docs/decisions/ADR-0028-answer-evaluation-provenance.md).
- This advances **K-006**: the invented-reference condition is now persisted per answer, so the
  rate becomes computable from durable history — though nothing aggregates it yet, and only when the
  durable store is selected. No measured rate is reported, because there is no generating model to
  invent references (the double emits fixed text). Both backends preserve equivalent record
  semantics, proven by conformance and cross-process restart tests.

### 2026-10-15 — Source and evidence inspection from the CLI, and source-scoped retrieval

**Added**

- `chakaso sources list|show` and `chakaso evidence show` — inventory and inspect the in-memory
  corpus built from the documents passed with `--file`: source identities with kind, canonical
  reference and chunk count; one source and the chunks ingested from it (`sources show --text`
  prints the bodies); and a single chunk with its source. Each has a valid `--json` view, useful
  exit codes, and is covered by tests. They call the existing `Corpus`, `ingest_file` and
  `SourceRecord`/`EvidenceChunk` APIs and add no behaviour of their own.
- `chakaso retrieve … --source <src_…>` (repeatable) restricts a query to named sources through the
  retrieval service's existing `source_ids` filter. Identifiers are validated before lookup
  (`SourceId`/`ChunkId`); an unknown-but-valid identifier is a clear error rather than a silent
  empty result, and it is never normalised into a different object.

**Notes**

- This is the honest completion of the evidence-usability surface (P18): the corpus stays in-memory
  and is rebuilt per invocation, and every command states that **no durable source index exists**
  rather than pretending otherwise. A durable source registry, and cross-process source
  removal/replacement, are deliberately **not** added — there is no consumer for them under the
  current architecture, and inventing one would violate the no-empty-scaffolding rule. Human-
  readable output renders untrusted document text inert (`chakaso.core.safe_for_terminal`); `--json`
  keeps the exact bytes. Adversarial tests prove a hostile document's control characters reach a
  terminal as visible escapes, not as live sequences, while the structured data is unchanged.

### 2026-10-15 — Untrusted text is made inert before it reaches a terminal

**Added**

- `chakaso.core.safe_for_terminal`: a dependency-free rendering primitive that makes
  control characters (the `ESC` byte, the C0/C1 blocks and `DEL`) and the bidirectional
  embedding/isolate controls inert in text shown to a person, replacing each with a
  *visible* escape (`\x1b`, `\u202e`) rather than dropping it silently. Ordinary text,
  newline, tab, accents and emoji pass through unchanged; the function is total and
  idempotent on already-safe output.
- The CLI's human-readable paths — the `retrieve` evidence listing (title, reference,
  matched terms, chunk body) and the `chat` reply — now render untrusted content through
  it. `--json` output is left to the serializer, which keeps the exact bytes.

**Notes**

- This is a structural display boundary against terminal-output injection (ESC sequences,
  carriage-return overwrite, Trojan-Source reordering), not a claim that a language model
  resists prompt injection — no model exists, and the retrieved-content-stays-data property
  was already tested elsewhere (ADR-0003). The threat is recorded honestly: the primitive
  neutralises the characters Chakaso emits; a reader's own terminal remains their own.

### 2026-10-15 — Semantic-evaluation development dataset and deterministic scorer

**Added**

- `chakaso.evaluation.semantic_benchmark`: a versioned, hand-labelled synthetic dataset of
  claim/evidence cases — direct support, contradiction, partial support, irrelevance, negation,
  numbers/units, entity mismatch, conflicting evidence, ambiguity, multi-sentence evidence and
  citation/evidence mismatch — each with a human-authored expected relation and a content fingerprint.
- A deterministic scorer that runs any `SemanticJudge` over the dataset and reports accuracy,
  per-class precision/recall, a confusion matrix, and how unresolved and evaluator-error cases are
  handled, as text or JSON. Two named probes (support-all, always-uncertain) exist so the scorer can
  be shown to distinguish an over-claiming judge from an abstaining one.
- `chakaso benchmark semantic [--json]` runs the instrument end to end from the command line.

**Notes**

- The instrument measures the **scoring harness, not semantic understanding.** The only judge is the
  caller-supplied `FixtureSemanticJudge`, which is handed the expected relations and therefore scores
  perfectly; that perfect score is explicitly labelled structure-replay, never a capability claim (the
  honesty rule in §2.1 and ADR-0027). No general language-understanding number is reported, and the
  dataset is a fixed seam for measuring a real evaluator when one exists.

### 2026-10-15 — Semantic-evaluation contract hardened; no real evaluator simulated

**Changed**

- `SemanticEvaluation` and the grounding result now carry an evaluator **version** alongside the
  evaluator name, so changing which evaluator judged a claim is visible in evaluation provenance.
  Judges may declare a `version`; `FixtureSemanticJudge` does.
- A dedicated `SemanticJudgeError` marks an evaluator that fails or abstains on a claim. The
  `SemanticGroundingEvaluator` records it per claim as `not_evaluated` — never as support — and one
  failing claim no longer aborts the whole answer. Structural grounding stays usable with no semantic
  evaluator wired at all.

**Added**

- `describe_semantic_capability()` reports honestly that no real semantic evaluator exists and that
  the only judge is `FixtureSemanticJudge` (which replays caller-supplied relations). `chakaso
  capabilities` now surfaces this, so a run states plainly that its grounding is structural.

**Notes**

- No capability was invented: the project forbids presenting a fixture or token overlap as semantic
  understanding (§2.1), and ADR-0001 keeps the runtime dependency-free and offline. There is no
  dependency-feasible, CPU-only real semantic evaluator to ship yet, so the boundary, capability
  detection and the honest limitation are delivered instead. A real evaluator plugs into the same
  `SemanticJudge` boundary later without touching `evaluate_answer`, `reassess` or the correction rule
  (ADR-0027).

### 2026-10-15 — CLI capability and answer-history inspection

**Added**

- `chakaso capabilities [--json]` — a read-only report of what the resolved configuration wires: the
  model adapter and whether it is a development double (read from the model's own metadata, not a
  hard-coded string), the retrieval strategy and limits, and the answer-store backend and path. It
  opens no store and reads no files, so inspecting capabilities never creates a database.
- `chakaso answers list|show|lineage [--json]` — inspects persisted answers through the same durable
  store `chakaso chat` writes to (ADR-0021): list recorded answers with their provenance, show one
  by `ans_…` identifier, or walk a correction chain oldest-first. It opens the store read-only
  (`initialize=False`, no journal write), validates identifiers before lookup, and is explicit that
  the in-memory default persists nothing across processes, so there is nothing to inspect unless
  `[storage]` selects `sqlite` with a real file.

**Notes**

- Both commands report only capabilities that genuinely exist and structure that genuinely happened.
  A resolved citation is shown as a source that was referenced, never as one that proves a claim, and
  no confidence, quality or model capability is invented.

### 2026-10-15 — Chat wired to the retrieval-aware runtime behind a typed composition root

**Added**

- `chakaso.app` — the application composition root: the single, injectable place a resolved
  `Config` becomes a running system. It builds the model through the registry, the **configured**
  answer store (in-memory or durable SQLite — choosing `sqlite` with no path is an explicit error,
  never a silent downgrade), the retrieval services over a corpus, the query planner at the
  configured strategy and limits, and the conversation manager. Every dependency can be injected
  in tests (ADR-0026).
- A typed `[retrieval]` configuration section — `enabled`, `strategy` (`lexical`/`dense`/`hybrid`),
  `top_k`, `candidate_k`, `rrf_k` and `max_ingest_bytes` — validated by the same specs/dataclass
  parity mechanism as `[model]` and `[storage]`, with offline defaults and no silent strategy
  fallback.
- `chakaso chat` now conducts turns through the retrieval-aware runtime: the planner decides each
  turn, retrieval runs over any `--file` documents, and every answer is recorded in the configured
  store. An honest notice names the engine (a development double), the active strategy, whether the
  corpus has documents, and whether answers persist. `--message` runs one turn for scripts; new
  `--file`, `--strategy`, `--top-k` and `--no-retrieval` flags are real overrides, reported on
  stderr.

**Notes**

- This wires components that already existed and were tested in isolation; it adds no language
  model and makes no semantic claim. With the deterministic double and the non-semantic fixture
  embedding, `chat` grounds, cites and records honestly but produces no real answer, and with no
  `--file` a retrieving turn returns no evidence. Tests assert that a configuration change actually
  changes runtime behaviour, not merely that a config object serialises; the offline no-network
  import guard now covers `chakaso.app`.

### 2026-10-10 — End-to-end integration and security verification

**Added**

- An end-to-end integration demonstration that runs a query through the whole assembled runtime to a
  durable SQLite store and proves, on reopen, that the grounded answer and its `correction_of` lineage
  survive the process — the answer's substance persisted, its volatile evaluation recomputed.
- A deliberate "test-the-tests" check that a failing persistence save aborts the turn and leaves the
  conversation unchanged (no phantom answer), and an untrusted-content check that text inside a
  retrieved document stays inert data while an identifier never supplied resolves to nothing (ADR-0003).

**Notes**

- These verify the wiring built across the run; they add no capability and make no semantic claim — the
  flow still runs the deterministic double and the non-semantic fixture embedding. The no-network import
  guard already covers the planning and runtime packages; no default runtime path reaches the network.

### 2026-10-10 — Measured retrieval-strategy comparison

**Added**

- `chakaso.experiments.compare_retrieval_strategies`: runs the retrieval development benchmark with
  each retriever (lexical, dense, hybrid) over the *same* synthetic dataset and reports the measured
  metrics plus per-metric deltas against a baseline strategy — the strategy being the deliberate
  variable, which is why this is a separate tool from `compare_results` (a regression baseline that
  correctly refuses differing configurations). It hard-codes no winner.
- `retrieval_strategy_experiment` (a measured experiment per strategy), and the experiment runner now
  honours a configured `retriever` strategy when running a retrieval experiment.
- `chakaso benchmark strategies [--top-k N] [--json]` prints the measured comparison; `chakaso
  benchmark retrieval --strategy lexical|dense|hybrid` scores a single strategy.

**Notes**

- The first run of this (E-001, recorded in `docs/research/research-log.md`) is a neutral-to-negative
  result: over the non-semantic fixture embedding, dense and hybrid match lexical on recall@k/precision
  @k/hit-rate and *regress* MRR by 0.071. It tests the mechanism and the instrument, not the
  semantic-retrieval hypothesis (H7), which needs a real embedding model. No number was invented; the
  regression is reported as the measured outcome it is.

### 2026-10-10 — Retrieval-aware conversation runtime

**Added**

- `chakaso.runtime`: a `RetrievalAwareConversation` that composes the whole pipeline for one turn —
  the planner decides, the chosen retriever (lexical, dense or hybrid) produces an evidence pack, and
  the existing `ConversationManager` generates, validates citations, evaluates, records an
  `AnswerRecord` and persists it (ADR-0025). It is a thin coordinator, not a monolith: it delegates
  to every layer rather than reimplementing one.
- Retrieval is genuinely optional: a greeting plans no retrieval and produces an ordinary turn, while
  a knowledge question retrieves against the plan's prior-source constraints. Because planning and
  retrieval run before the manager's transactional `send`, a failed plan, unwired mode or raising
  retriever leaves the conversation and store untouched — no phantom answer, no half-recorded turn.
- `build_retrieval_services(corpus)` wires one service per mode over a corpus, each labelled with the
  strategy it actually uses; `render_evidence_context(pack)` renders the pack into a structured,
  citation-safe model-facing context (identifiers and text, never instructions, ADR-0003).
- The planner's decision and the retrieval strategy are recorded on each answer as provenance and
  metadata, so a grounded turn is inspectable afterwards; `ConversationManager.send` gained one
  optional `provenance` parameter to carry it.

**Notes**

- This makes the target flow executable and tested end to end offline — but still around a model that
  does not exist: generation is the deterministic double and dense/hybrid use the non-semantic fixture
  embedding, so the runtime grounds, cites and persists honestly and makes no semantic claim.
  `chakaso chat` is not yet wired to the runtime; no CLI command drives it yet.

### 2026-10-10 — Hybrid retrieval (reciprocal rank fusion)

**Added**

- `chakaso.retrieval.hybrid`: a `HybridRetriever`, a third implementation of the `Retriever`
  protocol, that composes the lexical and dense retrievers into one ranking through transparent,
  deterministic **reciprocal rank fusion** (ADR-0024). The formula is written out —
  `fused(c) = Σ weight / (rrf_k + rank)` over the components that returned a chunk — chosen because
  it needs only each component's *ranks*, not a shared score scale a BM25 score and a cosine
  similarity do not have.
- A `HybridConfig` (method, `rrf_k`, `candidate_k`, `lexical_weight`, `dense_weight`) validates at
  construction (only RRF, positive `rrf_k`, non-negative weights not both zero), and is the typed
  tuning object — deliberately not a second configuration system.
- `RetrievalResult` gained an optional `provenance`; a hybrid result carries a `HybridProvenance`
  recording the lexical and dense rank/score that fed it (or `None` for a component that missed the
  chunk) and the fused score that ordered it, so a hybrid hit stays auditable rather than an opaque
  number.
- With a hybrid retriever present, the query planner now backs `hybrid`: lexical, dense and hybrid are
  all selectable modes.

**Notes**

- Over the non-semantic fixture embedding (ADR-0023), hybrid mostly re-ranks token overlap; it
  demonstrates the fusion mechanism, and no claim of dense/hybrid superiority on meaning is made or
  supported. It stays in-memory and offline.

### 2026-10-10 — Dense retrieval boundary and an exact index

**Added**

- `chakaso.retrieval.embeddings`: an `EmbeddingModel` boundary (`metadata` plus a batch
  `embed(texts) -> vectors`) with `EmbeddingVector` (fixed-length, finite) and `EmbeddingMetadata`
  that carries a decisive `is_semantic` flag. Its only implementation is `FixtureEmbeddingModel`, a
  deterministic hashed bag-of-tokens double declared `is_semantic=False` and `development_double=True`
  — a development double, **not** a semantic embedding (ADR-0023).
- `chakaso.retrieval.dense`: an exact in-memory `DenseIndex` (uniform-dimension and duplicate-chunk
  validated) and a `DenseRetriever`, the second implementation of the `Retriever` protocol, ranking by
  cosine or dot similarity. A CPU-only, dependency-free exact scan; an approximate (FAISS-like) index
  is deferred until a measured corpus needs it.
- Every `RetrievalResult` now carries a `strategy` (`lexical` / `dense` / `hybrid`) so evaluation can
  attribute a hit to its mechanism; a dense result reports no matched terms rather than inventing some.
- Because a dense retriever exists, the query planner now backs the `dense` mode: `DeterministicQueryPlanner`
  accepts `lexical` and `dense`, and still refuses `hybrid` until the fusion layer is built.

**Notes**

- This makes the dense *shape* real, not dense *quality*. With the fixture embedding, a dense result is
  token overlap under a different metric; no benchmark here may claim dense beats lexical on meaning.
  There is still no real embedding model and the retrieval corpus and its indexes remain in-memory
  (nothing persists vectors to disk).

### 2026-10-10 — Deterministic query planner

**Added**

- `chakaso.planning`: a `QueryPlanner` boundary (ADR-0022) with one implementation,
  `DeterministicQueryPlanner` — a typed rule system, not a language model and not a discourse
  resolver. Given a turn and the conversation state it returns an inspectable `QueryPlan`: the
  retrieval mode, whether retrieval is required, the original and a meaning-preserving normalized
  query, prior-source constraints for a follow-up, `top_k`, the context that informed the decision,
  the reasons that actually fired, and a `planner_version`.
- Retrieval stays optional: a greeting plans no retrieval, a knowledge question plans fresh lexical
  retrieval, and a follow-up that shares the conversation's topic or uses an anaphoric cue plans
  retrieval constrained to the sources already in play — a topic change does not reuse stale evidence.
- `normalize_query` collapses whitespace, composes Unicode NFC and trims trailing terminal punctuation
  without rewriting words, case or interior punctuation; both the original and the normalized form are
  kept on the plan.
- `QueryMode` names `none`, `lexical`, `dense` and `hybrid`, but the planner refuses `dense`/`hybrid`
  at construction because no retriever backs them yet — the vocabulary is stable without promising a
  capability that does not exist. Construction enforces that a plan never misreports itself (a mode
  cannot disagree with its `retrieval_required` flag, a retrieving plan needs a query, reuse needs a
  source, and every plan must explain itself). The no-network import guard now covers the package.

**Notes**

- The planner is a standalone decision layer; the conversation runtime does not yet call it, so
  retrieval is still invoked explicitly by a caller. Its decisions are documented keyword rules, not
  semantic understanding, and are echoed verbatim in each plan so nothing about the reasoning is
  fabricated.

### 2026-10-10 — Durable SQLite answer storage

**Added**

- `chakaso.answers.sqlite.SQLiteAnswerStore`, a durable implementation of the existing `AnswerStore`
  boundary (ADR-0018) on the standard library's `sqlite3` — no new dependency and no third-party or
  hosted database (ADR-0001). Answer history, including the `correction_of` lineage, now survives the
  process that wrote it: a test writes an answer from a *separate process* and reads it back after
  restart. The default backend stays in-memory, so nothing is written to disk unless a caller opts in.
- An explicit schema version (`PRAGMA user_version`) with a migration boundary: a database from a newer
  or unsupported schema version is refused rather than opened for writing, and a store whose schema
  table is missing is reported as corrupt instead of being silently recreated over the top of it.
- Transactional appends: every save runs in one `BEGIN`/`COMMIT`, and a failure anywhere in a batch
  rolls the whole batch back, so a correction chain is never half-written. `save_many` appends a batch
  atomically on both backends, enforced by a new conformance suite run against in-memory and SQLite alike.
- Strict-read integrity: a row's indexed columns are cross-checked against its serialized document, and a
  malformed or divergent stored record raises rather than being repaired. An answer's `evaluation` is
  deliberately not persisted — it is recomputed against the current evaluator, not frozen into history.
- Typed configuration for the store: a `storage` section (`config.storage`) in the existing schema
  selects the backend (`memory` default | `sqlite`), the database path and the SQLite journal mode,
  validated by the same mechanism as every other setting (ADR-0006). The default is in-memory, so the
  built-in configuration writes nothing anywhere.
- `chakaso storage check` reports a durable store's schema version and integrity without writing to it;
  `chakaso storage migrate` creates or confirms its schema. Both act only on an explicit `--path` (or
  the configured `storage.path`), and both refuse a database they cannot safely interpret.

**Notes**

- Recorded as ADR-0021; this closes K-007 (answer records did not survive a process). Concurrency is
  scoped honestly: the store is single-thread, cross-process *restart* is tested, but simultaneous
  multi-process writing is neither built nor claimed.

### 2026-10-08 — Reproducible experiments and a regression baseline

**Added**

- `chakaso.experiments`: an `Experiment` (an identifier, a name, a **hypothesis**, the development
  benchmark it runs, and a configuration) and a content-fingerprinted `ExperimentResult` that keeps
  environment metadata separate from the measured numbers. `run_experiment` drives a real
  development benchmark and records its metrics, dataset identity/version and the evaluator(s) that
  decided; `result_id` fingerprints everything that determines the number and ignores the timestamp
  and host, so the same experiment reproduces identically on the same code. `compare_results` is a
  deterministic regression baseline that refuses to compare results defined differently (different
  benchmark version, configuration, evaluator or experiment) rather than subtracting unlike metrics.
  `baseline_experiment` supplies the canonical retrieval and correction baselines.
- `chakaso experiment run <retrieval|correction> [--json]` prints a reproducible experiment result.

**Notes**

- No experiment tracker, database or external service is introduced (ADR-0020, ADR-0001): results
  are serializable records a person commits, not persisted by the runtime. No statistical
  significance machinery — a delta over a handful of synthetic fixtures is not a confidence interval.

### 2026-10-08 — Semantic grounding boundary (fixture only)

**Added**

- `chakaso.grounding` gains a semantic grounding *boundary*: a per-claim `SemanticJudge` protocol
  (`judge(claim, evidence) -> SemanticEvaluation` over supported / contradicted / uncertain /
  not_evaluated), a `SemanticGroundingEvaluator` that adapts a judge onto the existing
  `GroundingEvaluator`, and a `FixtureSemanticJudge` that replays caller-supplied relations
  (ADR-0019). Answer evaluation and reassessment accept a semantic evaluator with no signature
  change, and grounding stays a separate dimension from citation.

**Notes**

- The only judge is a fixture, named `fixture-semantic-0.1` so a report cannot read it as
  understanding. No real semantic capability is claimed, no numeric confidence is introduced, and a
  named conflict still records no winner — resolving a conflict stays correction's job (ADR-0015).

### 2026-10-08 — Correction development benchmark

**Added**

- `chakaso.benchmark.correction`: `CorrectionCase` / `CorrectionDataset` apply the retrieval
  benchmark's versioning discipline (ADR-0013) to correction — content-derived case versions and a
  dataset fingerprint make a judgement edit a visible version change. Six synthetic cases cover every
  reachable decision (retain, correct, qualify, needs_review, abstain, and an irrelevant-evidence
  retain). `CorrectionBenchmarkRunner` builds real evidence packs from the shared fixture corpus and
  runs the same `reassess` the application uses, choosing structural or fixture-semantic purely from
  what a case declares; reports (text and deterministic JSON) give per-case expected-vs-actual and the
  aggregate metrics, and a pinned-fingerprint regression guards the curated set.
- `chakaso benchmark` now takes a subject: `retrieval` or `correction` (`--json` for both).

### 2026-10-08 — Persistent answers and application-level reassessment

**Added**

- `chakaso.answers`: a first-class `AnswerRecord` — model identity, evidence supplied, cited
  sources/chunks, claims, evaluation and a `correction_of` link — referencing the other layers by
  identifier rather than duplicating them. `AnswerId` is a per-event typed identifier, not a content
  hash, so a corrected answer is always distinguishable from the one it supersedes (ADR-0017).
  `AnswerStore` is the persistence boundary with an in-memory implementation that is append-only and
  refuses to overwrite a saved answer or to link a correction to a missing or cross-conversation
  answer (ADR-0018).
- `ConversationManager` optionally records a turn's `AnswerRecord` and saves it *before* adopting
  the new conversation, so a persistence failure aborts the turn and leaves the conversation
  unchanged — extending ADR-0008's atomicity to the answer history. Without a store, behaviour is
  unchanged.
- `chakaso.correction` gains an application-level operation, `reassess_stored_answer` /
  `reassess_and_record`, that reads a prior answer by identifier, re-grounds its recorded claims
  against caller-composed evidence, and pairs the decision with a `CorrectionRecord` linked to it.
  `reassess_followup` / `merge_evidence` compose standing plus new evidence deterministically.

**Notes**

- Nothing generates revised prose (there is no model), nothing runs reassessment automatically, and
  nothing is written to disk — the store is in-memory and lost on process exit. A prompt-injection and
  untrusted-text identity guard and a no-network import check were extended to cover the new layers.

### 2026-10-08 — Benchmark from the command line

**Added**

- `chakaso benchmark [--json] [--top-k N]`: runs the retrieval development benchmark over its
  synthetic fixtures from the command line and prints the report — human-readable text, or the
  deterministic machine-readable JSON with `--json`.

**Notes**

- The command is thin wiring: it calls `load_development_benchmark`, `RetrievalService` and
  `RetrievalBenchmarkRunner` and prints the result; all report wording and the "development
  benchmark" labelling live in `chakaso.benchmark.report`. It scores retrieval only, generates no
  answer (there is no language model), reads no files and touches no network, and its output is
  deterministic.

### 2026-10-08 — Correction foundation (reassessment, decision, record, metrics)

**Added**

- `chakaso.correction`: the reassessment foundation `docs/correction.md` specifies, built on the
  claim and grounding boundaries rather than a parallel one (ADR-0016).
  - `decide_claim` maps a claim's prior status and its status under new evidence onto a per-claim
    **retain / qualify / correct** disposition under a rule written in full: a claim is corrected
    only when the supplied evidence *contradicts* it, never because it was produced first or has
    aged — ADR-0015's refusal to rank by recency or authority is inherited.
  - `decide_answer` aggregates the dispositions into one answer-level `CorrectionDecision` under a
    fixed precedence — a contradiction outranks all; an unadjudicated conflict or an unsettled
    required claim goes to **needs_review** rather than guessing; lost support **qualifies**; total
    loss of grounding **abstains**; otherwise **retain**.
  - `reassess` re-runs a `GroundingEvaluator` over a prior answer's claims against a caller-composed
    pack and returns the `Reassessment`.
  - `record_reassessment` fixes a reassessment into an immutable, content-derived `CorrectionRecord`
    naming the answer it supersedes; `is_revision` reports a supersession only when the decision
    changed *and* a prior is named.
  - `correction_metrics` reports correction success, unjustified-persistence and unnecessary-revision
    rates over a caller-labelled set, exposing the raw counts so a rate cannot hide a thin
    denominator; a rate with no cases is `None`, never a fabricated `0.0` or `1.0`.
  - `reassess_followup` composes the standing evidence a prior answer rested on with the new
    evidence now in hand (`merge_evidence` unions packs de-duplicated and order-independent) and
    hands the claims to `reassess`. It is a convenience over the existing boundaries, not a new
    loop, and nothing in the runtime calls it automatically.

**Notes**

- Correction decides and records; it does not rewrite the answer. With no language model, `summary`
  is a structural tally rather than a revised sentence, the revised prose stays a model step that
  does not exist, nothing is persisted (there is no `AnswerRecord`), and the loop is not automatic —
  a caller invokes reassessment on evidence it supplies.
- The metrics are decision-agreement rates against a supplied label, not a claim that a revised
  answer is correct. No correction benchmark has been run and no measured correction result exists.
- ADR-0016 records the decision rule and its honesty constraints.

### 2026-10-08 — Claim, citation, grounding and answer-evaluation foundation

**Added**

- `chakaso.claims`: an immutable `Claim` with a content-derived `ClaimId` and a `ClaimStatus`
  — supported / unsupported / contradicted / uncertain / not_evaluated — that records how the
  *supplied evidence* bears on a claim, never whether it is true (ADR-0014). A replaceable
  `ClaimExtractor` boundary decomposes an answer: `StructuredClaimExtractor` builds claims a
  caller supplies and `SentenceClaimExtractor` is a documented deterministic heuristic, so a
  trained model can implement the same boundary later without touching callers.
  `ClaimEvidenceLink`/`links_from_claims` project claims into inspectable (claim, chunk)
  citations.
- `chakaso.citation`: structural citation validation. Each citation is classified **valid**
  (points at supplied evidence), **unknown** (names a chunk that was not supplied — a
  fabrication counted, never resolved, per ADR-0003), or **irrelevant** (present but outside a
  case's expected evidence); a required claim with no valid citation is **uncited**.
  `citation_metrics` reports structural citation precision and recall alongside those counts.
- `chakaso.grounding`: a `GroundingEvaluator` boundary returning a per-claim `GroundingResult`
  (ADR-0015). `StructuralGroundingEvaluator` emits only supported / unsupported / not_evaluated
  and cannot invent a contradiction; `ManualGroundingEvaluator` is the only path on which
  contradicted or uncertain appear and stands in for a human label or a future model. A
  `Contradiction` names no winner, and `SourceAuthority`/`most_recent_wins` are caller-supplied
  precedence hooks the engine never applies automatically.
- `chakaso.evaluation.evaluate_answer`: combines the citation and grounding results into one
  answer view that keeps the dimensions **separate** — citation metrics, grounding statuses,
  `evidence_coverage` (the grounded-claim support rate) and any contradictions — and
  deliberately computes no single merged "quality score." `is_grounded` holds only when every
  claim is supported, nothing is contradicted and no citation points outside the pack.

**Notes**

- These are structural primitives, not an understanding of meaning: "supported" and "valid"
  are defined by citation presence and a case's expected evidence, never by a judgement that a
  source proves a claim. Contradiction and uncertainty require a supplied judgement and are
  never detected automatically. No language model is in the loop — claims and evidence are
  supplied by a caller or fixture.
- No answer-quality result is reported. There is still no grounded-answer or correction
  benchmark and no measured model-quality number; this is the evaluation machinery correction
  will later build on, not a scored outcome. The reassessment and correction loop itself does
  not exist: nothing compares a previous answer against new evidence or records a revision.
- ADR-0014 records the claim decision and ADR-0015 the grounding boundary.

### 2026-10-08 — Retrieval development benchmark

**Added**

- `chakaso.benchmark`: a versioned benchmark the retrieval metrics can actually be run
  against. `BenchmarkCase` names gold (acceptable) and forbidden (unacceptable) evidence by
  the same `ChunkId`/`SourceId` retrieval returns, plus required/forbidden claims, an
  expected behaviour (answer/qualify/abstain) and a citation requirement; a forbidden set is
  what catches citation laundering. A strict loader parses JSONL cases plus a `manifest.toml`,
  refusing unknown/missing/malformed fields by name.
- Identity and versioning (ADR-0013): a readable `CaseId`/`DatasetId` slug that survives
  edits, and a content-derived version/fingerprint that cannot drift from what it labels.
- A curated synthetic fixture corpus and `development_benchmark`, whose gold evidence is
  resolved by ingesting the same corpus the runner searches, so identifiers always match and
  the build is fully deterministic.
- `RetrievalBenchmarkRunner` scores the retrieval service over a dataset into recall@k,
  precision@k, MRR, hit rate, forbidden hits, false retrievals and duplicates, keeping
  scored and abstention cases distinct. `format_report`/`report_json` render a run with its
  provenance, labelled a development instrument.

**Notes**

- This measures retrieval only, against a small hand-built synthetic corpus. It is a
  development instrument, not a scientific benchmark: no confidence interval or significance
  test is computed, no answer is evaluated, and no model-quality number is reported.
- A pinned-fingerprint regression fails if any fixture or case judgement changes, forcing a
  deliberate dataset-version bump rather than a silent edit.

### 2026-10-08 — Retrieval evaluation metric functions

**Added**

- `chakaso.evaluation`: pure, deterministic metric functions — `recall_at_k`,
  `precision_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`, `duplicate_count`,
  `unresolved_reference_count`, and a `measure_latency` observation helper.

**Notes**

- These are measurement *hooks*, not a benchmark. There is still no evaluation dataset, no
  task version and no measured number anywhere in the repository; inventing a figure would
  be the fabrication the project forbids. The functions compute well-defined quantities from
  a ranked result and a relevance judgement, ready for when a benchmark supplies the
  judgements.
- Metrics that need a ground-truth definition first — evidence coverage, NDCG — are not
  invented into a formula; they wait for the benchmark.

### 2026-10-08 — Fetched web content becomes evidence

**Added**

- `chakaso.retrieval.html.parse_html`: a narrow HTML reader (standard-library
  `HTMLParser`, not a browser) that extracts readable text, captures the `<title>`, maps
  headings to Markdown so the section-bounded chunker applies, decodes entities, and drops
  `<script>`/`<style>`/`<noscript>`/`<template>`/`<svg>`.
- `chakaso.retrieval.web.ingest_acquired`: turns an `AcquiredSource` into the same
  `IngestedDocument` a local file produces — decode, parse, normalize, chunk — so web and
  local evidence share one representation. HTML and plain text are accepted; any other
  media type is refused rather than mangled.
- `chakaso.retrieval.cache`: an in-memory `AcquisitionCache` keyed by canonical URL (with
  an optional age limit) and a `CachingFetcher` that wraps any `Fetcher` to avoid
  refetching.

**Security posture**

- Fetched text is data, never a directive. A prompt-injection line inside a page is stored
  verbatim as an evidence chunk and is never executed, never changes policy, and never
  assigns identity — a web source is named by the validated canonical URL, not by anything
  the page claims about itself (ADR-0003). Regression-tested.
- Decoding is strict: a declared `charset` is honoured, a UTF-8 byte-order mark is
  stripped, and bytes invalid in the expected encoding are refused, not replaced.

**Notes**

- Still opt-in and offline. Nothing on a default path fetches; the reader is not a browser
  and does no boilerplate removal, and PDF and full main-content extraction remain planned
  (K-008).

### 2026-10-08 — Bounded, opt-in web fetcher

**Added**

- `chakaso.retrieval.HttpFetcher` behind a `Fetcher` protocol: given an
  `AcquisitionRequest` and a `FetchPolicy` it returns an `AcquiredSource` (bytes, media
  type, requested URL, final URL after redirects, retrieval time). The real network call is
  an injectable `Transport`; `make_urllib_transport` provides the standard-library one.
- `FetchPolicy`: a validated value object for the limits that were missing — scheme
  allowlist, response-size cap, redirect depth, timeout, and a user-agent that cannot carry
  a newline. A policy can restrict to `https` but can never widen past http/https.
- `is_blocked_destination`: refuses loopback, private, link-local (including the cloud
  metadata address), unique-local, multicast and reserved address literals, plus known
  loopback hostnames.
- A typed acquisition error family: `InvalidSchemeError`, `BlockedDestinationError`,
  `ResponseTooLargeError`, `FetchTimeoutError`, `UnsupportedContentTypeError`,
  `NetworkFailureError`, under `AcquisitionError`.
- ADR-0012 records the decision and its security posture.

**Security posture, stated honestly**

- Redirects are followed by the fetcher, not the transport, so every hop's scheme and
  destination are re-validated: a public URL that redirects into a private network is
  blocked, and redirect loops and over-long chains are stopped.
- The destination screen is bounded: it blocks address literals, but a hostname that
  resolves to a private address is **not** caught by a pre-connect string check. Closing
  that needs IP-pinned resolution in the transport, which is not implemented. This is
  recorded as K-008, not hidden.
- Fetching is **opt-in and off every default path**: nothing in the library, `chakaso
  retrieve`, the tests or CI reaches the network, so the ADR-0001 no-network runtime
  guarantee still holds, and CI stays offline and deterministic.

**Notes**

- This is acquisition only. Decoding fetched bytes and turning them into evidence (HTML →
  text, then normalize/chunk) is a separate, still-pending stage; the fetcher never assigns
  an identifier and never treats page text as instructions.

### 2026-10-08 — Command-line retrieval

**Added**

- `chakaso retrieve --query TEXT --file PATH [--file PATH ...] [--top-k N]`: the local
  pipeline from the command line. It ingests the named files into an in-memory corpus,
  runs lexical BM25 retrieval, and prints the ranked chunks with their provenance —
  score, section, matched terms and the source's canonical reference.

**Notes**

- The command calls the same application layer the library exposes (`ingest_file`,
  `Corpus`, `RetrievalService`) and holds no retrieval logic of its own. Errors (a missing
  or unsupported file, a bad limit) are reported readably with no traceback.
- It is local and offline, reads only the files given, and generates no answer: there is
  still no language model. An empty retrieval is reported plainly, not filled with
  invented evidence. Only `.txt`, `.md` and `.markdown` are read.

### 2026-10-08 — Retrieval orchestration

**Added**

- `chakaso.retrieval.RetrievalService`: the component that runs the pipeline for one
  query — corpus, index, retriever, ranked chunks, evidence pack — and returns a
  `RetrievalOutcome` carrying a validated `EvidencePack` and the full ranked result list.
  It is deliberately not the conversation manager, so retrieval can be called on its own
  and a turn can decide whether to retrieve at all.

**Provenance guarantees**

- Sources come from the corpus, never invented; a chunk whose source the corpus lacks is
  refused.
- The retriever's score is recorded onto a copy of each chunk; the stored chunk and its
  identifier are unchanged, so a citation still points at the exact evidence used.
- An unmatched query produces an empty pack, never fabricated evidence — the failure
  ADR-0003 exists to prevent.
- The pack records which retriever and `top_k` produced it, so an answer's provenance
  includes the pipeline, not only the model.

**Notes**

- `ConversationManager.send` can now be handed a pack that something actually produced.
  No query planner exists, so nothing decides on its own that a turn needs retrieval;
  retrieval is still called explicitly. No network, no embeddings, no model.
- The index is built once from the corpus the service is given; a corpus that grows later
  needs a fresh service, because an index is a cache, never the source of truth.

### 2026-10-08 — Deterministic lexical retrieval

**Added**

- `chakaso.retrieval.build_lexical_index` and `LexicalRetriever`: an inverted index over
  a corpus's chunks, scored with Okapi BM25, returning the best `top_k` as
  `RetrievalResult` records — the chunk, a score, a rank, and the matched terms. This is
  the smallest thing that turns a corpus into ranked evidence, and the baseline a later
  dense retriever must beat.
- `Retriever`: a protocol, so the lexical implementation can be replaced without touching
  a caller. `source_ids` filters candidates.

**Notes**

- Retrieval is lexical and says so: it matches shared words and no meaning — no
  embeddings, no vector index, no model, no network. Its limitations (synonyms,
  morphology, paraphrase) are exactly what the project wants measured rather than assumed.
- Determinism is enforced, not hoped for: scores are summed in sorted term order so
  floating-point accumulation cannot drift, ties break by chunk identifier so the order is
  total, and a query that matches nothing returns nothing rather than filling `top_k` with
  invented evidence. A result's score is a ranking artefact, never a confidence.
- `k1` and `b` are documented per-call defaults, not yet configuration fields.

### 2026-10-08 — In-memory corpus

**Added**

- `chakaso.retrieval.Corpus`: the in-memory set of evidence retrieval reads from. It
  holds sources and their chunks, adds documents idempotently, enumerates deterministically
  (sources by identifier, chunks by source and position, independent of insertion order),
  and looks records up by identifier.
- `CorpusError`: raised when a chunk is added under a source that does not own it.

**Notes**

- No persistent storage and no index. A corpus is deliberately a plain in-memory
  container: a database would be maintained against a guess about what needs storing,
  and an index is a later cache over this, not its source of truth.
- Duplicate documents do not grow the corpus, because identifiers are content-derived.
  Nothing searches or ranks it yet.

### 2026-10-08 — Local document ingestion

**Added**

- `chakaso.retrieval.ingest_file` and `ingest_text`: the first stage that turns a source
  into evidence records. Each identifies the source (ADR-0011), reads it, normalizes it,
  splits it into sections and chunks, and returns an `IngestedDocument` — a
  `SourceRecord` plus its `EvidenceChunk` records — which is exactly the shape
  `ConversationManager.send` already consumes. No parallel representation was invented.
- A typed ingestion error family: `DocumentUnavailableError`, `UnsupportedDocumentError`,
  `DocumentTooLargeError`, `ContentDecodingError`, under `IngestionError`.

**Filesystem safety**

- Ingestion reads the single file a caller names and nothing else. It never crawls a
  directory, never walks a filesystem, and never reads a file because it happens to
  exist. A directory, an absent path, an empty path, a non-regular file, an unsupported
  extension, a file over the byte limit and non-UTF-8 bytes are each refused with a
  specific error, and every one of those rejections is exercised by a test.
- Only `.txt`, `.md` and `.markdown` are read, so a binary — including the private
  `.docx` pack — is refused on format rather than decoded into mangled text. A `..` in a
  path is normalized away, so a winding path cannot fork one document into two sources.
- The byte limit is checked against the file's size before its bytes are read, so a
  pathological file cannot exhaust memory; a bad file is a hard failure, never a silent
  truncation. An empty file is a valid source with zero chunks.

**Notes**

- Still no retrieval, no ranking and no network. This is explicit-source ingestion, not
  a query planner or a fetcher; the caller chooses what to ingest.
- The ingestable size limit is a documented parameter of each call, not a hidden
  constant, and not yet a field in the configuration file — nothing reads it per run
  yet, matching how chunking sizes are handled.

### 2026-10-08 — Deterministic content normalization

**Added**

- `chakaso.retrieval.normalize_document`: puts supplied text into a deterministic form
  before it is chunked. CRLF and lone CR become LF, text is composed to Unicode NFC,
  trailing spaces and tabs are stripped per line, and runs of blank lines collapse to
  one. It normalizes representation and not substance: no re-wrapping, no case change,
  no word removed, leading indentation preserved.

**Why**

- Normalization feeds the content hash and the chunk identifiers, so two formatting
  variants of one document — a Windows file and a paste of the same body — must produce
  one set of identifiers, or an evaluation compares formatting rather than content. The
  function is idempotent and total.

**Notes**

- Two limits are stated, not hidden: it does not parse Markdown fences (blank lines
  inside a code block collapse like any others, matching the chunker), and its Unicode
  form depends on the interpreter's normalization tables, so reproducible evaluation
  pins the interpreter version.
- Normalization is a stage the pipeline calls; nothing is normalized automatically yet,
  because nothing reads a document. It is exercised through text a caller supplies.

### 2026-10-08 — Generalized source identity

**Added**

- `chakaso.evidence.identity`: a `SourceReference` and a `SourceKind` (`web`, `file`,
  `text`) that decide what a source *is*, kept deliberately separate from whether it
  can be fetched. `web_reference`, `file_reference` and `text_reference` build one.
- `SourceRecord.create_file` and `SourceRecord.create_text`, so a local document and
  supplied text are records now rather than a later step.
- ADR-0011 records the decision, and answers the open question that had blocked local
  ingestion. The distinction it turns on: a source reference is an identity, a fetchable
  URL is a permission, and the two were living in one function.

**Changed**

- `SourceRecord.canonical_url` became `canonical_reference`, and the record carries a
  `kind`. `derive_source_id` now takes a canonical reference. **The identifier formula
  and every web-source identifier are unchanged** (ADR-0007 is untouched): a web
  reference's canonical form is exactly `canonicalize_url`'s output, and a `file` URI or
  a `text` reference can never equal a canonical URL, so folding the three kinds into one
  formula cannot collide across them.
- `canonicalize_url` is now used only for web references. Identifying a local source no
  longer routes through it, so widening source identity no longer widens what the future
  fetcher is permitted to retrieve.

**Notes**

- Still no ingestion and no retrieval. A caller hands the content to
  `create_file`/`create_text`; nothing opens a file, reads a path from disk, or touches
  the network. The record can name a local source before anything can read one.
- File identity is machine-local (a `file` URI names a path on one filesystem), and a
  relative path resolves against the working directory, so absolute paths give the
  stablest identity.

### 2026-10-07 — Document chunking

**Added**

- `chakaso.retrieval.chunk_document`: splits document text belonging to an
  already-identified source into `EvidenceChunk` records. Chunks carry a section
  path, a position in document order, and a content-derived identifier.
- `split_sections`: Markdown-heading splitting with nested heading paths, so a
  citation into a subsection says where the subsection sits.
- `ChunkingConfig`, with a target size and a hard ceiling, in characters.
- ADR-0010 records the chunking decision. Chunks never cross a section boundary and
  do not overlap. Overlap was declined deliberately: with content-derived
  identifiers, overlapping chunks would report one sentence under several
  identifiers, which inflates every retrieval metric that counts chunks and makes
  "how many chunks mention this" meaningless. The boundary failure that overlap hides
  is now something an evaluation can measure.

**Notes**

- Sizes are characters, not tokens. `token_count` on a chunk stays unset, because
  there is no tokenizer and a character count in a field named `token_count` would be
  a lie a later reader would trust.
- Chunking does not read files, does not fetch and does not rank. Nothing here has
  touched the network.
- Reading a document from a local path turned out to be blocked: a document with no
  URL has no identity under ADR-0007, and `canonicalize_url`'s scheme check currently
  doubles as the fetch permission list. Widening it in a file-reading module would
  quietly widen what the future fetcher may retrieve, so the question is recorded in
  `docs/research/open-questions.md` and no ingestion code was written.

### 2026-10-07 — Local conversation shell

**Added**

- `chakaso chat`: a conversation shell that runs the whole stack end to end —
  configuration, context projection, the model boundary, validation, evidence and
  citation recording. Interactive on standard input, or one message and exit with
  `--message`, which is what a script wants. `:quit`, `:exit` and end of input all
  end a session; blank lines are ignored rather than sent; a failed turn is reported
  and the session continues, because a failure leaves the conversation unchanged.
- The shell prints a notice on standard error before the first reply, naming the
  engine, saying that it is a development double rather than a language model, and
  saying that nothing is retrieved, nothing is corrected and the conversation is not
  saved. It is derived from the model's own `development_double` metadata rather than
  from a hard-coded string, so it cannot drift out of date while the engine stays
  fake.
- The reply goes to standard output and everything else to standard error, so
  `--message` is usable in a pipeline.
- The shell reports two conditions a reader would otherwise not know about: a reply
  truncated at the length ceiling, and a reference to evidence the model was not
  given, which is listed and never resolved.

**Changed**

- The CLI's own description no longer says there is nothing to run, because there is
  now something to run.
- `--config` is defined once and shared by `config show` and `chat`, so the two cannot
  drift apart in behaviour or in help text.

**Note on scope**

`chakaso chat` is not a chatbot. Every reply it produces is the development double's
fixed text, and the command says so in its own output. There is no language model, no
retrieval, no fetching, no correction loop and no evaluation.

### 2026-10-07 — Conversation manager

**Added**

- `chakaso.conversation.ConversationManager`: the orchestration boundary over the
  existing conversation state. One turn rejects an empty message, appends the user
  turn to a working copy, projects the recent conversation into model messages, calls
  the model through the `LanguageModel` boundary, rejects a response that is empty or
  attributed to a different model, resolves evidence references, appends the
  assistant turn, and adopts the new state.
- A turn is transactional (ADR-0008). If any step fails, the conversation is
  unchanged, so a user turn is never left stranded without a reply and a retry is a
  first attempt rather than a follow-up to a turn that never happened.
- Distinguishable failures: `InvalidUserInputError` for an empty message,
  `InvalidGenerationError` for an unusable response, and model errors propagated
  untranslated rather than folded into a generic conversation error.
- `Reply`, carrying the new state, the assistant turn, the generation result (model
  identity and whether it was truncated) and the citation resolution.
- A reference to evidence the model was not given is recorded, never resolved, and
  reported to the caller (ADR-0009). Resolved citations, unknown references and
  malformed references stay three separate results.
- Context projection bounded by `model.max_context_turns`, a new configuration
  setting. It bounds growth; it is a turn ceiling, not a token budget, and is
  documented as the stopgap it is.
- `new_conversation_id()`, and conversation-layer error types in their own module
  alongside the other packages' error modules.

**Changed**

- `ConversationManager.send` accepts an `EvidencePack`. Nothing produces one yet —
  there is no retrieval — but the parameter exists now because adding it later would
  touch every call site, and every call site is somewhere the citation rule could be
  forgotten.
- The configuration test that restated the `ModelConfig` field names now checks them
  against the schema, so adding a setting is no longer a two-place edit that trains
  people to update a test without reading it.

**Verification**

- A new repository-hygiene test reads the syntax tree of every module under `src/`
  and fails if any module outside the model registry imports a concrete adapter,
  which is ADR-0002's guarantee. It was confirmed to fail against a deliberate
  violation before being kept.
- Argument-unused linting is relaxed for `tests/` only, because a test double that
  ignores the messages it was given is usually the point, and per-line suppressions
  would outnumber the code they annotate.

**Note on scope**

There is still no language model. The only implementation of the boundary is the
deterministic development double, so a conversation can be held and nothing
intelligent is produced by it. No retrieval, no fetching, no correction, no
evaluation, no tokenizer and no training.

### 2026-10-07 — Python 3.11 compatibility fix

**Fixed**

- `SourceRecord.metadata` used a shared `mappingproxy` as a dataclass field
  default. Python 3.11 rejects *any* unhashable default outright, and a
  mappingproxy is unhashable, so `chakaso.evidence` failed at import on 3.11 while
  working on 3.12 and 3.13. It now uses `default_factory`.

**Why this is recorded**

It was caught by CI rather than locally, because development was happening on
Python 3.13 and the failure was an import-time error rather than a wrong result.
Three consecutive pushes went out with red CI before it was noticed, which is a
process failure as much as a code one: a push is not finished until the run is
checked.

Two things changed as a result. Version-suffixed virtual environments (`.venv311`,
`.venv312`) are now documented for checking the floor locally, and `.gitignore`
covers them, so the suite runs on all three supported versions before a push rather
than only in CI.

### 2026-10-07 — Conversation state

**Added**

- `chakaso.conversation`: immutable, append-only conversation state. Turns record
  when they happened, which evidence was supplied for them and which sources the
  answer actually referenced — kept separate, because "which of those did you use?"
  is unanswerable if only the union is stored.
- `Conversation.to_model_messages()` is the single mapping from conversation state
  to what a model sees, with an optional turn limit. A model is given roles and text
  and nothing else.
- State carries the active topic, entities and open questions, and exposes the
  sources the conversation has touched, most recent first, which is what a follow-up
  question has to resolve against.
- Enforced invariants: timestamps must be timezone-aware, history is append-only for
  a given conversation, and an answer cannot cite a source it was not given. That
  last rule is the ADR-0003 rule applied to the recorded state rather than to the
  text.

**Not added, deliberately**

Claims and an answer record. Both belong to the reassessment path, which does not
exist yet; adding them here would bake the shape of correction into conversation
before anything has decided what a claim is.

### 2026-10-07 — Evidence records and citation resolution

**Added**

- `chakaso.evidence.SourceRecord`: an immutable record of one canonical URL with
  one content body, carrying a content-derived identifier, host, retrieval time,
  optional publication time, content hash and extraction metadata. Built through
  `create()`, so an identifier cannot disagree with the content it names.
- `EvidenceChunk`: an immutable evidence unit with a derived identifier, position,
  optional section, and retrieval and rerank scores recorded but never presented as
  confidence.
- URL canonicalization, documented as part of the identifier contract because
  identifiers are derived from it (ADR-0007). It removes fragments, default ports,
  trailing root dots, trailing slashes and campaign parameters, and refuses
  non-HTTP schemes and URLs with embedded credentials.
- `EvidencePack`: the bounded set of evidence supplied to one generation call,
  which validates that every chunk's source is present and that identifiers are
  unique, and drops sources no chunk refers to.
- `resolve_citations()`: a validation gate rather than a lookup. A well-formed
  identifier that was supplied resolves to its chunk and source; one that was not
  supplied is recorded as an unknown reference; an identifier-shaped string that is
  not valid is recorded as a malformed reference. Nothing outside the pack is ever
  resolved, which is the rule ADR-0003 exists to enforce.

**Fixed**

- `canonicalize_url` dropped the brackets from IPv6 literals, producing a string
  that parses as a different URL.
- `EvidenceChunk.create` raised an identifier error rather than a record error for
  a negative position, so the same defect had two different exception types
  depending on which entry point was used.
- Citation resolution classified references as unknown before checking whether they
  were valid, which made the malformed bucket unreachable. Inventing a source and
  fumbling the identifier format are different defects with different fixes, and
  they are now counted separately.

**Note on scope**

Nothing has been fetched or indexed. These records are built from content a caller
already has, so the evidence layer is implemented without any retrieval.

### 2026-10-07 — Language-model boundary

**Added**

- `chakaso.models`: the boundary application code depends on, implementing
  ADR-0002. `LanguageModel` requires only `metadata` and `generate`; optional
  behaviour is declared as a `Capability` and implemented on a separate protocol,
  with `tokenize()` and `generate_structured()` reconciling the declaration and the
  implementation. A model that declares a capability it does not implement is
  reported as a defective implementation rather than as caller error.
- `ModelRegistry` and `create_model(config)`: the single place where configuration
  becomes a model. Re-registering a name is rejected rather than allowed to shadow.
- `chakaso.models.deterministic`: a development double that produces fixed text,
  declares no capabilities, and marks itself with
  `metadata.development_double = True`. It is not a language model and says so in
  its own metadata, so a report can state plainly that no model was involved.
- Contract tests that every implementation of the boundary inherits, so a new
  adapter cannot pass its own tests while violating the interface.

**Note on scope**

The only implementation of the boundary is the development double. There is no
trained Chakaso model and no local inference adapter, so nothing in this repository
generates language. Retrieval, correction and evaluation still do not exist.

### 2026-10-07 — Core primitives

**Added**

- `chakaso.core.identifiers`: `SourceId` and `ChunkId`, validated on construction
  and derived from content rather than assigned at random (ADR-0007). The same URL
  with the same content yields the same identifier, so repeated retrieval
  deduplicates; the same URL with different content yields a different identifier,
  so an earlier answer keeps pointing at the bytes it actually used.
- `chakaso.core.hashing`: SHA-256 over text or bytes, with a truncated form for
  identifiers and an explicit part separator so that concatenating parts before
  hashing cannot be ambiguous.
- `chakaso.core.errors`: a base error type for deliberately raised failures, with
  validation errors also deriving from `ValueError` so untrusted input can be
  caught with the built-in type.

**Note on scope**

Still no model interface, retrieval, correction loop, tokenizer or trained model.

### 2026-10-07 — Engineering foundation

**Added — package**

- `src/chakaso/` package with a `src` layout, installable and typed, shipping a
  `py.typed` marker (ADR-0005). The layout is deliberate: it makes an uninstalled
  import fail rather than silently testing the working tree.
- A thin CLI: `--version`, `info`, and `config show`. Argument parsing only; the
  CLI holds no behaviour of its own.
- No runtime dependencies. The foundation uses the standard library only.

**Added — configuration**

- Typed configuration schema with declared types, ranges and name patterns,
  loaded explicitly from TOML with the standard library (ADR-0006).
- Per-value provenance: every configuration value reports the file it came from,
  or that it came from the built-in default. `chakaso config show` prints it.
- Unknown keys, wrong types, out-of-range values, arrays and unsupported TOML
  types are errors naming the file, the key and the expectation, rather than
  warnings. A typo that is silently ignored produces a run that does not use the
  settings its author believes it uses.
- `configs/default.toml`, whose values a test asserts match the built-in defaults
  so the two cannot drift.

**Added — verification**

- CI running formatting, linting, type checking and tests on Python 3.11, 3.12 and
  3.13, with no secrets and no access to a model provider.
- Repository-hygiene tests enforcing rules that are otherwise broken by accident:
  the private planning pack never becomes tracked, the ignore rule stays present,
  no secret-shaped or oversized files are tracked, no dependency is a commercial
  inference or search provider, decision records are numbered and indexed
  contiguously, and documentation links resolve.

**Fixed**

- Two documentation links pointed at paths that do not exist. Both were found by
  the new link check rather than by reading: `docs/agent/CONVENTIONS.md` pointed
  at `../../decisions/ADR-0006-toml-configuration.md` instead of `../decisions/`,
  and `docs/getting-started.md` linked to the `configs/` directory.

**Note on scope**

There is still no conversational system: no model interface, no retrieval, no
correction loop, no tokenizer and no trained model. No benchmark has been run and
no result is reported anywhere in this repository. Configuration is the only
component implemented so far.

### 2026-10-07 — Project foundation

**Added — repository**

- Apache-2.0 license for code, documentation and configuration. Trained weights,
  tokenizers and datasets are explicitly not covered; they will carry their own
  terms when they exist.
- `.gitignore` that excludes the author's private planning pack (`docs/*.docx`)
  along with virtual environments, tool caches, model weights, dataset shards and
  local experiment outputs.
- `.gitattributes` normalizing text to LF in the repository and marking binary
  formats, so line-ending conversion cannot corrupt weights or documents.
- `.editorconfig` for consistent indentation and line endings across editors.

**Added — decision records**

- ADR-0001: the initial runtime has no required commercial inference or search
  dependency. Network access is allowed for fetching public web content, which is
  processed locally. External providers may exist later only as optional adapters.
- ADR-0002: application code depends on a narrow language-model interface, not a
  concrete model. Implementations declare capabilities; unsupported requests fail
  at the boundary.
- ADR-0003: the retrieval layer owns source and chunk identifiers. Generated text
  may only reference identifiers it was given; an unknown identifier is recorded
  as an error and never resolved to a URL.
- ADR-0004: Apache-2.0, chosen for compatibility with the intended dependency stack
  and for its explicit patent grant.

**Added — public documentation**

- `README.md` stating what Chakaso is, what is implemented, and what is not.
- Architecture, transparency, retrieval, correction, training, evaluation and
  roadmap documents. These are specifications: they describe intended behaviour and
  label the status of every component.
- Research foundation: a research log, six falsifiable hypotheses, an open-questions
  list, and a literature file whose URLs were each fetched and verified before
  publication.
- Contribution guide.

**Added — internal and agent documentation**

- Agent contract (`AGENTS.md`) and agent context: current state, conventions, active
  task and known issues.
- Internal engineering notes: repository history, artifact policy and tooling
  rationale.
- Experiment record format.

**Note on scope**

There is no runnable system, no retrieval, no correction loop, no tokenizer and no
model. No benchmark has been run and no result is reported anywhere in this
repository.

**Note on history**

The repository's previous history, whose head tree was already empty, was replaced
with a fresh root commit. The previous head is preserved as tag
`archive/bootstrap-attempt-1`. See `docs/internal/engineering/repository-history.md`.
