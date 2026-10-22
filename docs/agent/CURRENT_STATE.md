# Current state

Last updated: 2026-10-22, at the foundation release work (nodes A–R, ADR-0037 and ADR-0038). What changed
since the E2606 measurement is that the trained artifact gained a **runtime**, not a new claim of quality: a
loaded model can be asked for bounded decoding (temperature, top-k, top-p, a seed, stop sequences, a repeat
penalty), can stream a generation as it is produced and be cancelled mid-stream, formats a conversation
through a versioned prompt template, and is distributed as a **release directory** whose `model.json` states
what it is and whose checkpoint is checksum-verified before a weight is read
(`chakaso.models.release`, `chakaso.models.decoding`, `chakaso.models.formatting`). An optional local HTTP
adapter serves the same object over an OpenAI-shaped interface with no third party and no framework
([ADR-0038](../decisions/ADR-0038-local-http-serving-adapter-from-the-standard-library.md)).

On the model path itself there is now a second architecture family beside the measured E2605 transformer: a
rotary-position decoder with RMSNorm, tied embeddings and a key/value cache for incremental decoding
(`chakaso.training.decoder`), written under its own checkpoint schema so the artifacts of three measured
records keep their meaning. The training engine learned from it: cosine decay to a stated floor, gradient
accumulation with an effective batch size, throughput measured over gradient time only, a run that reports
*why* it ended (`converged`, `budget_ceiling`, `stopped_by_request`), experiment-level resume that refuses a
changed identity and allows a longer budget, and a tokenizer cache keyed to the inputs that produced it so a
seed study pays that cost once. A dataset catalogue pins each external source by URL, digest, licence and
retrieval date, and an auditing pass over the corpus reports its own composition, length distribution and
near-duplicate findings instead of assuming them (`chakaso.training.dataset_catalog`,
`chakaso.training.dataset_report`), which is how the 919-document, 14,490,005-character
`gutenberg-english-prose-v2` (fingerprint `e9fc2e92f59d88ea`) that E2608 trains on was checked before use.

**E2608 is the experiment that decides whether any of this produced a useful model.** Its pilot ran and
reported (8 steps at the reference configuration, 3,413,760 parameters, 4,561 positions per second of gradient
work, every held-out sample degenerate — as expected at that budget), the arms are declared in
[`experiments/E2608/record.md`](../../experiments/E2608/record.md) before the runs, and **no measured
comparison between the two architectures exists yet**. Until that record has numbers in it, the statement
below stands.

What does **not** change: the runtime's default model is still the development double, no model answers a
grounded question (K-001), nothing here has met a *workload* — a corpus that is not Chakaso's own writing is
not a task anybody needed doing — the generations of every checkpoint trained so far repeat a block back on
itself, and P29 (calibration) and P32 (runtime consumption of the sufficiency decision) stay deferred by the
same ADR that deferred them.

Remote CI is no longer one tier. `.github/workflows/ci.yml` runs `checks` dependency-free, `train` with
numpy, and `model-path` with torch from the CPU wheel index plus the strict type check over every module, and
all eight jobs have been **read back green** from GitHub's runners (run `39704678913`, 2026-10-22). Getting
there is the part worth keeping in view: the first push of that file failed on a Windows-only attribute that
every local gate had passed, and the second failed an assertion finer than float32 can express — two defects
the old single-tier CI could not have found because they lived in the tiers and the operating system it did
not use (K-009 records both, closed).

This file is the authority on what exists. If it disagrees with any other
document, this file is right and the other document is a defect.

A private planning note is never evidence that something is implemented.

## What exists

| Thing | Where | Notes |
| --- | --- | --- |
| Repository conventions | `.gitignore`, `.gitattributes`, `.editorconfig` | Private docx pack excluded by `.gitignore` and guarded by a test |
| License | `LICENSE` | Apache-2.0; rationale in ADR-0004 |
| Public documentation | `README.md`, `docs/*.md`, `docs/research/` | Specifications with mandatory status labels |
| Decision records | `docs/decisions/` | ADR-0001 to ADR-0038 (0037: the reference model is sized by measured throughput, with a token-matched comparison arm; 0038: the local serving adapter is standard-library HTTP, bound to loopback, and serves only a trained model) |
| Contribution guide | `CONTRIBUTING.md` | |
| Agent contract | `AGENTS.md`, `docs/agent/` | |
| Python package | `src/chakaso/` | Installs; typed; `py.typed` ships |
| CLI | `src/chakaso/cli.py` | `--version`, `info`, `capabilities`, `config show`, `chat`, `retrieve` (with `--source` to scope a query), `sources list|show`, `evidence show`, `benchmark retrieval|correction|strategies|semantic`, `experiment run`, `experiment model` and `experiment study` (both model-path actions need the optional `[train,nn]` extras and are thin handlers over `chakaso.training.subword_experiment` / `chakaso.training.seed_study`, importing torch lazily so a dependency-free install reports what is missing instead of failing at import; `study` also takes `--collect-only`, which rebuilds an aggregate from the reports on disk without training), and `corpus restore|verify` (dependency-free: recover a pinned corpus from the revision that recorded it, or check a snapshot or the tracked tree against its manifest, exit non-zero on drift), `storage check|migrate`, `answers list|show|lineage`, and the release lifecycle `model inspect|verify|export|generate|stream|score|serve` (every one a thin handler over the Python API: `inspect` prints `describe()` plus the release contract when the path is a release directory, `verify` re-hashes a release's artifact against its own document, `export` assembles a distributable directory from a run's work directory, `generate`/`stream` take the boundary's own decoding parameters (`--temperature --top-k --top-p --seed --stop --repeat-penalty --max-new-tokens`) and `stream` cancels the run at a step boundary on Ctrl-C, `score` returns the held-out cross-entropy and refuses an adapter that cannot measure it, and `serve` starts the local endpoint and blocks); `chat` builds its graph through `chakaso.app` and drives the runtime, and a follow-up that explicitly names a stored `ans_…` identifier routes into a reassessment rather than an ordinary reply (ADR-0029); after each recorded turn the shell prints the answer's identifier on stderr so a later turn can name it; `sources`/`evidence` inspect the in-memory corpus built from `--file` documents and are explicit that no durable source index exists. Untrusted document text is rendered inert on the human-readable paths via `chakaso.core.safe_for_terminal` |
| Application composition | `src/chakaso/app.py` | `build_application(loaded, …)` is the single place a resolved `Config` becomes the runtime's dependencies: model via the registry, the configured answer store (`memory` or durable `sqlite`, never a silent downgrade), retrieval services over a corpus, the planner at the configured strategy/limits, and the manager. Since ADR-0029 the application also composes a deterministic reassessment trigger with the existing reassessment boundary (`Application.reassess_followup`), gathering a follow-up's evidence read-only through the runtime; a named-but-unknown or ambiguous target raises the boundary's typed errors. Every dependency is injectable for tests (ADR-0026) |
| Configuration | `src/chakaso/config/`, `configs/default.toml` | Typed schema in sections (`model`, `storage`, `retrieval`), explicit loading, per-value provenance; `model` selects the adapter and (for the trained adapter) a `checkpoint_path` that defaults to empty so nothing loads by default; storage defaults to the in-memory backend so nothing is written; `retrieval` selects the strategy (`lexical`/`dense`/`hybrid`), `top_k`, hybrid fusion knobs and an ingest size bound, and whether grounding is on at all |
| Core primitives | `src/chakaso/core/` | Content-derived identifiers, SHA-256 hashing, terminal-safe rendering of untrusted text (`safe_for_terminal`), staged all-or-nothing artifact writes (`atomic.staged_write` / `write_text_atomically`, which also guarantees a written file's bytes are the characters a digest was computed over), error base |
| Model boundary | `src/chakaso/models/` | The interface a developer actually uses, and the runtime around it. `LanguageModel` protocol and capabilities (`TOKENIZE`, `STREAMING`); a registry that selects an adapter by name — and, for a trained one, by the checkpoint it is given — in exactly one place (ADR-0002); `chakaso.models.load_model(path)`, which takes a checkpoint **or a release directory** and verifies the release's checksum before reading a weight; `GenerationParams` as the whole bounded-decoding surface (temperature over a stated range, top-k, top-p, a seed, `stop_sequences`, `stop_token_ids`, repeat penalty, `max_new_tokens`) with one validation point so a request and a configuration cannot disagree about what a number means; `GenerationResult` carrying `token_ids`, `new_tokens`, `prompt_tokens`, `truncated` and `stopped_by`, and `FinishReason` distinguishing end-of-text, budget, context ceiling and caller cancellation; `decoding.py` (`TokenSelector`, `StopScanner`, `AbortSignal`, `truncate_prompt`) as pure dependency-free policy the adapters compose rather than reimplement; `formatting.py` with **versioned** prompt templates — `chakaso-plain-v1` is the default because that is the shape the measured runs were produced with, and `chakaso-role-labelled-v1` is opt-in, not a claim about instruction following; `release.py`, the `chakaso-model-release/1` document that states identity, capabilities, licence, limitations and the artifact's own digest, with `CorruptedReleaseError` and `IncompatibleReleaseError` separating "unreadable" from "not this schema"; and `describe()`, so a caller can ask a loaded runtime what it is. Every adapter inherits the same contract suite |
| Model implementations | `src/chakaso/models/deterministic.py`, `trainedchar.py`, `nntransformer.py` | Three adapters behind the boundary. `deterministic`: a development double only — the default, the only one `configs/default.toml` selects, and the one the serving adapter refuses outright. `trained_char` (P25): loads an E2601-class trained character checkpoint, rebuilds its bound vocabulary, generates greedily within `max_new_tokens`, declares `development_double=False` with a real `TOKENIZE` capability, and its model is a 984-parameter toy. `subword_transformer` (P31, extended for the second family): loads either checkpoint schema — `chakaso-transformer-checkpoint/1` or `chakaso-decoder-checkpoint/1` — rebuilds the architecture named inside it, decodes with the tokenizer stored in the same artifact, generates greedily or by temperature/top-k/top-p sampling with a seed, stops on a stop sequence or the document separator, streams through the boundary's `stream()` as a real step loop that observes cancellation, scores a text in nats with the same arithmetic the experiment uses, and declares itself not the double. What has been measured through it is E2605's 643,840-parameter Transformer (1.8794 ± 0.0077 bpc over five seeds, generations looping) and E2607's runs on externally licensed prose; the rotary decoder family exists and its comparison arm is running as E2608, so **no measured difference between the two architectures is recorded yet**. Both trained adapters are opt-in behind their extras (lazy registry imports, so `import chakaso.models` stays dependency-free; a missing extra or an incompatible checkpoint is a named `ModelError`, never a silent fall back to the double). **No useful language model exists.** |
| Local serving adapter | `src/chakaso/serve/` | **Implemented.** `app.py` is request logic with no socket in it — `GET /health`, `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions`, OpenAI-shaped bodies, and a refusal naming the field for every part of that shape it cannot honour (`n`, `best_of`, `logprobs`, `echo`, `user`, `suffix`, an over-long prompt, an out-of-range temperature, `stream: true` against a model that cannot stream). `server.py` is the transport: `http.server.ThreadingHTTPServer` from the standard library, no framework, `127.0.0.1` unless an operator names another address, a semaphore that turns a generation over the ceiling into a 429 with `Retry-After` rather than a silent queue, chunked server-sent events, and a client that hangs up mid-stream cancelling the generation through the boundary's own `AbortSignal`. `load_serving_model` accepts a checkpoint or a verified release directory and **refuses the development double** — no configuration makes this endpoint answer from canned text. Tested over a real socket: 49 tests covering event order, the `[DONE]` marker, cancellation observed as tokens actually stopping, an oversized body refused before it is read, and `stop()` on a service that never started. **It has not yet been run against a trained artifact end to end.** No authentication is offered, and none is implied: binding beyond loopback is a decision an operator states |
| Evidence records | `src/chakaso/evidence/` | `SourceRecord`, `EvidenceChunk`, `EvidencePack`, typed source references (web/file/text, ADR-0011), URL canonicalization, citation resolution |
| Normalization | `src/chakaso/retrieval/normalize.py` | Deterministic representation normalization (line endings, Unicode NFC, blank lines) before chunking |
| Chunking | `src/chakaso/retrieval/chunking.py` | Document text to evidence chunks: section-bounded, non-overlapping, deterministic (ADR-0010) |
| Ingestion | `src/chakaso/retrieval/ingest.py` | Reads one explicitly named local text/Markdown file, or supplied text, into a source record and its chunks; never crawls |
| Corpus | `src/chakaso/retrieval/corpus.py` | In-memory set of sources and chunks for retrieval: idempotent add, deterministic enumeration, provenance guard |
| Lexical retrieval | `src/chakaso/retrieval/lexical.py` | Deterministic BM25 index and retriever behind a `Retriever` protocol: ranking, top-k, source filter, inspectable explanations |
| Dense retrieval | `src/chakaso/retrieval/embeddings.py`, `dense.py` | An `EmbeddingModel` boundary with a deterministic **non-semantic** fixture double, an exact in-memory dense index, and a `DenseRetriever` behind the same `Retriever` protocol; results carry the strategy that produced them (ADR-0023). No real embedding model, no vector store on disk |
| Hybrid retrieval | `src/chakaso/retrieval/hybrid.py` | A `HybridRetriever` that composes the lexical and dense retrievers with transparent reciprocal-rank fusion, returning one ranking tagged `hybrid` where each result carries `HybridProvenance` — the lexical and dense ranks/scores that fed it (ADR-0024). Deterministic and offline; over the fixture embedding it re-ranks token overlap, not meaning |
| Retrieval orchestration | `src/chakaso/retrieval/service.py` | `RetrievalService.search`: query → ranked `EvidencePack` with recorded scores and preserved provenance; an empty result makes an empty pack, never invented evidence |
| Evidence sufficiency | `src/chakaso/retrieval/sufficiency.py` | A deterministic, offline, numpy-free **decision** layered additively over retrieval (ADR-0032): the retriever still returns top-k, and a policy decides whether the pack is adequate evidence. `DiscriminatingTermSufficiencyPolicy` reports `sufficient` only when a retrieved chunk shares a query term that is a **non-function-word** and rare in the corpus (document frequency ≤ `max_document_fraction`), naming the discriminating terms so the verdict is inspectable; empty pack/corpus is `insufficient`. It is **not a calibrated confidence** — it holds only over a narrow corpus-specific band (E2604) — it does not change retrieval's candidates, forbidden-hit ranking, or the runtime (which does not yet act on it, P32); measured today via `chakaso benchmark grounded --sufficiency` |
| Query planning | `src/chakaso/planning/` | A deterministic `QueryPlanner` boundary (ADR-0022): a typed `QueryPlan` (mode, normalized + original query, source constraints, top-k, reuse flag, explanation, planner version), meaning-preserving `normalize_query`, and `DeterministicQueryPlanner` deciding per turn whether to retrieve lexically, reuse the conversation's prior sources for a follow-up, or retrieve nothing. Not an LLM; lexical, dense and hybrid retrieval are all backed by retrievers (ADR-0022/ADR-0024) |
| Conversation state | `src/chakaso/conversation/state.py` | Immutable, append-only turns with provenance; topic, entities and open questions |
| Conversation manager | `src/chakaso/conversation/manager.py` | Conducts one turn: context projection, model call, validation, evidence and citation recording. Transactional (ADR-0008) |
| Retrieval-aware runtime | `src/chakaso/runtime/` | `RetrievalAwareConversation` composes planner → retrieval (lexical/dense/hybrid) → model → claims → citation validation → grounding → `AnswerRecord` → store as a thin coordinator, not a monolith (ADR-0025): retrieval optional, turn atomicity preserved, the planner decision and retrieval strategy recorded as provenance/metadata, and a structured, citation-safe evidence context rendered for the model. Runs the deterministic double; no model, no semantic claim |
| Conversation shell | `chakaso chat` | Interactive (`--message` for one-shot). Grounds turns through the retrieval-aware runtime over any `--file` documents, records each answer to the configured store, and states in its own output that the engine is a development double, names the active strategy, and says whether answers persist. `--no-retrieval` runs it ungrounded |
| Retrieval CLI | `chakaso retrieve` | Ingests named local files, runs lexical retrieval, prints ranked evidence with provenance. Local and offline |
| Benchmark CLI | `chakaso benchmark` | Runs the retrieval, correction, strategy-comparison, or semantic-evaluation development benchmark over its synthetic fixtures and prints a text or deterministic JSON report that labels itself a development instrument (`strategies` measures lexical/dense/hybrid and reports measured deltas; `semantic` scores the only available judges — a fixture and two probes — over the labelled semantic dataset, measuring the harness, never semantic capability). Local and offline |
| Web fetcher | `src/chakaso/retrieval/acquire.py`, `policy.py`, `netguard.py` | Opt-in, bounded HTTP fetch under a `FetchPolicy`; scheme/size/redirect/timeout/destination rules. Off every default path (ADR-0012) |
| Web ingestion & cache | `src/chakaso/retrieval/html.py`, `web.py`, `cache.py` | Narrow HTML reader turns fetched bytes into the same evidence records as a local file; an opt-in in-memory cache avoids refetching |
| Evaluation | `src/chakaso/evaluation/` | Pure metric functions (recall@k, precision@k, MRR, duplicate + unresolved-reference counts, latency observation), and `evaluate_answer`, which combines citation and grounding results into separate answer-evaluation dimensions (`evidence_coverage`, `is_grounded`). No merged quality score and no model-quality number. `semantic_benchmark` adds a versioned, hand-labelled synthetic dataset and a deterministic scorer (accuracy, per-class precision/recall, confusion, unresolved + error handling) — a development instrument that measures the harness; the only judge is a fixture and its perfect score is structure-replay, not semantic capability (ADR-0027) |
| Benchmark | `src/chakaso/benchmark/` | Versioned case schema, strict JSONL/manifest loader, synthetic development fixtures, a retrieval runner and reports; a correction benchmark (cases, dataset versioning, a runner over `reassess`, per-case + aggregate metrics, text/JSON reports, pinned fingerprint). Development instruments, not scientific benchmarks. A grounded-answer evaluation sub-module (ADR-0031, P27) adds a *distinct* workload: provenance-bearing `GroundedCase`/`GroundedSource`/`GroundedProvenance` schema, a `GroundedDataset` that refuses dangling evidence, duplicate cases and development/held-out split leakage, a strict loader (`grounded_loader`), a self-authored Apache-2.0 workload (`ga-kestrel-library`, 13 passages / 14 cases, pinned content fingerprint `d2d2ee00d117b88a`), an evaluation runner (`grounded_runner`) composing `RetrievalService` + `evaluate_answer` through injected answer providers (an oracle that measures the harness and a non-answering provider that reflects that no model answers today), per-stage failure classification (retrieval / generation / evaluation), hand-checkable metric primitives (`grounded_metrics`, dimensions never merged), deterministic text/JSON reports (`grounded_report`), and `chakaso benchmark grounded`. It evaluates the pipeline and retrieval, not answer capability — there is still no model that answers (K-001) |
| Claims | `src/chakaso/claims/` | Immutable `Claim` with a content-derived `ClaimId` and a `ClaimStatus` that records what the supplied evidence supports — never truth. A replaceable `ClaimExtractor` boundary (structured + a documented sentence heuristic) and claim/evidence links (ADR-0014). No trained extractor |
| Citation validation | `src/chakaso/citation/` | Structural citation checks against a supplied pack (valid / unknown / irrelevant / uncited) plus citation precision/recall. Decides presence and expected-match, never that a source proves a claim |
| Grounding | `src/chakaso/grounding/` | `GroundingEvaluator` boundary: a structural evaluator (supported / unsupported / not_evaluated) and a manual evaluator that is the only path to contradicted / uncertain; a `Contradiction` names no winner (ADR-0015). A `SemanticJudge` / `SemanticGroundingEvaluator` boundary exists (ADR-0019) carrying evaluator name + version, a rationale, named conflicts and an explicit `SemanticJudgeError` recorded as not-evaluated — never support (ADR-0027) — but its only implementation is a caller-supplied fixture; `describe_semantic_capability()` reports that no real semantic judge exists |
| Answers & persistence | `src/chakaso/answers/` | `AnswerId` is a per-event typed identifier, not a content hash (ADR-0017). An immutable `AnswerRecord` ties turn + model + evidence + claims + evaluation + a `correction_of` link by reference, and now persists a typed, immutable `evaluation_provenance` snapshot — the citation/grounding/reference outcomes and evaluator name+version the turn itself reached (ADR-0028) — while the live `AnswerEvaluation` stays recomputed and unpersisted. `AnswerStore` boundary with two append-only, integrity-checked backends: in-memory `InMemoryAnswerStore` (the default) and a durable `SQLiteAnswerStore` built on the standard library (ADR-0018, ADR-0021). The durable store is proven by a test to survive a process restart, and is opt-in, so nothing writes to disk unless it is selected |
| Correction foundation | `src/chakaso/correction/` | The reassessment decision rule (retain / qualify / correct / abstain / needs_review, ADR-0016), `reassess` over the grounding boundary, an append-only `CorrectionRecord`, structural correction metrics, a follow-up helper (`reassess_followup`), an application-level `reassess_stored_answer` / `reassess_and_record` over a stored answer, and a deterministic `DeterministicReassessmentTrigger` that reads a follow-up for an explicit `ans_…` reference (intent only — it never reassesses anything itself, ADR-0029). Decides and records; produces no revised prose (there is no model) and is not an autonomous loop — a follow-up must name a stored answer, and the composition is explicit |
| Experiments | `src/chakaso/experiments/` | `Experiment` (hypothesis + benchmark + configuration) and a content-fingerprinted, environment-separated `ExperimentResult`; `run_experiment` drives a development benchmark (the retrieval runner honours a configured `retriever` strategy); `compare_results` is a deterministic regression baseline that refuses incompatible versions, and `compare_retrieval_strategies` measures lexical vs dense vs hybrid over the same fixtures (ADR-0020). No tracker, no database, no model |
| Training (experimental) | `src/chakaso/training/` | The from-scratch training pipeline ADR-0005 reserved, in three dependency tiers. **Dependency-free**: a self-authored corpus (`chakaso-toy-char-v1`), a deterministic character vocabulary, leak-free splits, a validated `TrainingConfig`, uniform/unigram baselines, and since P30 the whole data layer for a real corpus — `manifest` (per-document SHA-256 over normalized text, licence, source, typed drift refusal), `data.split_text_units` (whole-document three-way split, deduplicated by content before selection), `subword` (byte-level BPE with a named pre-tokenizer, deterministic tie-breaking, self-describing checksum-verified record), `packing` (fixed-length next-token examples, dropped tail reported), `model_metrics` (bits per character / nats / perplexity / accuracy through one arithmetic for any identifier space), `snapshot` (recover a pinned corpus from a revision of the local history through a `RevisionReader` seam, materialize it, and run the same `verify_corpus` check a fresh preparation runs before writing a checksum-verified snapshot with an acquisition block), `source_corpus` (turn a downloaded licensed text into a pinned corpus offline: strip by the source's own markers, chunk at whole-paragraph bounds, record URL + byte digest + retrieval date + licence statement) and `generation_eval` (repetition ratios, longest repeated block, cycles anywhere and periodic tails, length/termination/refusal, over the `LanguageModel` boundary, with its interpretation limit carried in the payload). **`[train]`** (numpy, ADR-0030): the tiny causal character model with hand-derived, every-scalar-verified gradients, Adam, a full-batch loop recording a held-out curve, a mini-batch loop over the same model, the JSON checkpoint (schema/2), held-out evaluation and the E2601/E2602 drivers. **`[nn]`** (torch, ADR-0034): `transformer` (decoder-only causal, pre-norm, GELU, tied head, parameter count derived from the config), `decoder` (the second family ADR-0037 chose by measurement: rotary positions, RMSNorm, explicit q/k/v/o without biases, GELU feed-forward, tied embeddings, depth-scaled residual init, an ignore-index target mask and a key/value cache that makes incremental decoding the same computation as a full forward pass), `nn_training` (mini-batch AdamW with warmup and cosine decay to a stated floor, gradient accumulation over an effective batch size, clipping, validation-selected early stopping, throughput measured over gradient time only, a completion class saying whether the run converged, hit its budget or was stopped by request, a graceful stop signal and an experiment-level resume that refuses a changed identity and allows a longer budget), `nn_checkpoint` (ADR-0035: one checksummed file, tokenizer inline, `weights_only=True`, usable vs resumable kinds, and two architecture families each with its own schema), `nn_evaluation` (batched scoring shared with the model boundary's `score`), `subword_experiment` (the experiment driver: trains either family from a declared corpus snapshot with a required corpus fingerprint and its own split seed, reuses a fitted tokenizer from a cache keyed to the inputs that produced it, and writes a report that states its own budget outcome) and `seed_study` (declared seeds over one protocol, one directory per run, aggregation rebuilt from the reports on disk, protocol identity enforced on reuse and on rebuild, mean / sample SD / median / range with no interval). `dataset_catalog` and `dataset_report` (dependency-free) sit in front of all of it: a catalogue pins each external source by URL, digest, licence and retrieval date and ingests through a digest-keyed cache, and an auditing pass reports composition, length percentiles and exact plus near-duplicate groups with the candidate pairs it actually examined. None of it is re-exported, so `import chakaso.training` stays dependency-free and only named modules import torch (enforced by a hygiene test). No application logic imports the training package; the only readers of its checkpoints are the two optional adapters, and the E2605 driver generates through `chakaso.models.load_model` so the public path is the one under test |
| Experiment records | `experiments/E2601/record.md`, `experiments/E2602/record.md`, `experiments/E2603/record.md`, `experiments/E2604/record.md`, `experiments/E2605/record.md` | Five committed experiments and the project's only measured results. E2601: a randomly-initialised 984-parameter character model whose held-out next-char cross-entropy beats both uniform (3.3322) and unigram (3.0531) baselines — 1.6650 nats best, 2.4946 final — recorded honestly with the overfitting after step 50. E2602: the same protocol across five seeds with a three-way split and early stopping selected on validation → mean test loss **1.7978 ± 0.1767 nats**, below both baselines on every seed, and the step-300 final loss (mean 2.7804) worse on every seed — confirming the overfitting is systematic, not one lucky partition. E2603: the first grounded-answer evaluation baseline over the self-authored P27 workload — retrieval recall@k 0.8182 / MRR 0.8636 with the reference oracle over all 14 cases (held-out recall@k 1.0), zero stage failures, and honest negative findings (3 forbidden hits, 3 false retrievals on abstention cases); it measures the harness and retrieval, never a model, since no model answers in that path. E2604: the P28 evidence-sufficiency decision measured against the E2603 baseline — **100% accurate on the workload** (all 3 unanswerable cases judged insufficient, **0** answerable cases wrongly abstained → recall preserved) with retrieval's own metrics intentionally unchanged, and a documented narrow corpus-specific threshold band (~0.23–0.31). E2605: the first Transformer, trained from random initialization on the pinned 71-document corpus and recorded with its manifest and machine-readable report — held-out **1.8657 bits per character** (643,840 parameters, 2.6936 nats/token, perplexity 14.79, token accuracy 0.4501) against **3.1343** for E2601's character architecture, 3.3224 unigram and 4.8010 uniform; early stopping selected step 1380 of 2400 and the un-stopped model was 0.1434 nats worse; three committed generations loop. One seed, one author's prose, no error bar. Format per `docs/internal/experiments/README.md` |
| Tests | `tests/` | 1,668 tests collected at the current head (measured 2026-10-22 on this tree: **1,645 pass / 23 skip** dependency-free — the skips are the numpy- and torch-gated training/adapter tests — **1,726 pass / 14 skip** with `[train]`, **2,004 pass / 1 skip** with `[train,nn]`). The runner reports one more skip and one fewer pass than the same head does locally, and the difference is accounted for: the hygiene test that checks the private documentation pack is *untracked* skips where the pack does not exist at all. The suite covers package, CLI (incl. `retrieve`, `benchmark` and `experiment`, and the retrieval-aware `chat` wiring — corpus grounding, `--no-retrieval`, strategy/top-k overrides, durable-store recording, and `benchmark grounded` across providers/splits/`--sufficiency` with JSON determinism), configuration (incl. the `retrieval` section: defaults, strategy pattern, numeric limits, boolean `enabled`, provenance), application composition (`chakaso.app`: the configured store is used and never silently downgraded, strategy/top-k/enabled change real runtime behaviour, SQLite-backed answers survive a reopen, sqlite-without-path is refused, corpus building bounds file size, close is idempotent), identifiers and hashing, source identity, model boundary, evidence, normalization, chunking, ingestion, corpus, lexical retrieval, **retrieval evidence sufficiency** (ADR-0032: discriminating-term policy over empty/single-doc/all-stopword/morphology-miss cases, protocol conformance, and a read-only-ness check that it never mutates the pack), dense retrieval (embedding double, exact index, retriever), hybrid retrieval (reciprocal-rank fusion, per-component provenance), retrieval orchestration, end-to-end pipeline, fetch policy and netguard, HTML/web ingestion, cache, evaluation metrics and answer evaluation, benchmark identity/cases/loader/fixtures/runner/report and the correction benchmark, claims/extract/links, citation validation, grounding and the semantic boundary, correction decision/reassess/record/metrics/follow-up/service, answer identity/record/store/serialization and the durable SQLite store (restart, rollback, schema, integrity) with a two-backend conformance suite and conversation integration, experiments (model/runner/compare), security/determinism/reproducibility regressions (the no-network import guard covers `chakaso.app`), conversation state and manager, query planning (mode/normalization/plan invariants/deterministic rules), retrieval-aware runtime (plan→retrieve→record, atomicity, provenance), retrieval-strategy comparison (measured, deterministic lexical/dense/hybrid), end-to-end grounded-turn + restart integration and untrusted-content security checks, the explicit reassessment trigger (deterministic intent detection over valid / none / ambiguous / malformed references) and the composed `Application.reassess_followup` (valid, unknown-target, ordinary-follow-up and ambiguous routes; append-only integrity where the original answer is never mutated; SQLite restart conformance) and its reachability from `chakaso chat`, the grounded-answer evaluation workload (schema invariants, provenance refusal, dataset integrity and split-leakage, the pinned workload content fingerprint, strict-loader validation, hand-checkable metric primitives, runner pass/abstain/fail paths incl. per-stage failure classification and the answered-cases-only coverage denominator, evidence-sufficiency measurement, deterministic reports, and CLI integration), the training pipeline (dependency-free corpus/tokenizer/split/config/baselines, and numpy-gated model/finite-difference-gradient/checkpoint/evaluation/loop/experiment incl. that a trained model reproducibly beats both baselines), **the model path** (dependency-free: manifest discovery/verification/drift refusal and snapshot round-trips, document split with dedup and leakage refusal, BPE training determinism with per-occurrence merge counting and totality over Unicode, packing with shifted targets and dropped tails, bits-per-character arithmetic against hand-computable cases including a separator that costs a prediction and decodes to nothing; `[nn]`: transformer config validation and parameter-count agreement, mask causality, forward-loss against a hand-computed value and sampled finite differences, checkpoint round-trip plus checksum/schema/tokenizer/shape refusal, `weights_only` load, overwrite guard, resume reaching bit-identical parameters, plan/step scheduling, the loop's incremental selection agreeing with the pure rule and refusing when it does not, artifact-scoped evaluation, `load_model` adapter selection, the Transformer adapter's full `LanguageModel` contract incl. sampling and stop semantics, and the E2605 driver end to end on a synthetic root — pinning, splitting, both tokenizations, both baselines, report digest determinism, drift refusal on a re-run and overwrite refusal), repository hygiene. The count ages; `python -m pytest` does not. |
| CI | `.github/workflows/ci.yml` | Three tiers, and all of them verified **remotely**: `core` on 3.11/3.12/3.13 with `.[dev]`, `train` on the same three with `.[dev,train]`, `model path` on the floor and the latest with `.[dev,train,nn]` from the CPU wheel index plus `mypy --config-file configs/mypy-model-path.toml`. Run `39706259071` (2026-10-22) is green across all eight jobs — core 3.13 1,644 passed / 24 skipped in 17.39 s, model path 3.13 2,003 passed / 2 skipped in 79.18 s, model-path jobs taking 1m58s–2m31s including the torch download. The tier that installs nothing is the guarantee that the default path needs nothing, and it is also where the HTTP serving adapter is tested. [`optional-dependency-ci-matrix.md`](../internal/engineering/optional-dependency-ci-matrix.md) describes the shipped file; K-009 is closed, with the two defects the new tiers found on the way |
| An automatic correction loop that decides on its own when to reassess | **does not exist** | Since ADR-0029 a follow-up that *explicitly names* a stored answer composes into a reassessment through the runtime (a deterministic trigger + the existing boundary), but nothing decides unbidden that a turn needs revising, a vague reference is never resolved by a guess, there is no real semantic embedding, and the model any default path answers with is still the development double |

## What works

- `python -m pip install -e ".[dev]"` installs the package and its dev tools.
- `python -m pytest` passes on Python 3.11, 3.12 and 3.13, which is the range CI
  enforces. Development on the newest interpreter alone is not sufficient: an
  unhashable dataclass default imported cleanly on 3.13 and failed at import on
  3.11, and CI found it rather than the local run.
- `python -m ruff check .`, `python -m ruff format --check .` and `python -m mypy src`
  pass under strict settings. `mypy src` excludes the modules whose types resolve through the optional
  extras, so the model path is checked by `mypy --config-file configs/mypy-model-path.toml`, which passes
  over all 142 source files where `.[dev,train,nn]` is installed.
- `python -m chakaso --version` and `python -m chakaso info` report the version,
  Python version and platform.
- `python -m chakaso config show` prints every configuration value with the source
  it came from, and exits non-zero with a specific message when a configuration
  file is missing, malformed, or contains an unknown key or an out-of-range value.
- Source and chunk identifiers can be derived from content and validated from a
  string, which is the path a citation from generated text will take.
- `create_model(config)` builds the adapter named in configuration, and raises
  `UnknownModelError` listing the registered adapters when the name is wrong. Three adapters are
  registered; the default configuration selects the deterministic development double, and the two trained
  ones need their extra installed and a checkpoint path given — a missing extra is a named error, never a
  silent substitution.
- `chakaso.models.load_model(path)` returns a `LanguageModel` from a checkpoint without a terminal, a config
  file or an application import: it selects the adapter from the artifact's extension and declared schema,
  verifies the checksum, and refuses an artifact whose tokenizer, architecture or tensor shapes do not match
  what it claims. `model.generate(messages, params)` then satisfies the shared boundary contract,
  `model.sample(prompt, max_new_tokens=…)` is the bare-string shape an inference caller usually wants, and
  `model.score(text)` returns held-out cross-entropy in nats.
- Asking the double to tokenize or produce structured output fails at the boundary
  with a message naming the model and the capability, rather than returning
  something that looks like an answer.
- A source record can be built from a URL and content, deriving its own identity and
  content hash, and can confirm later whether a given content still matches it.
- The same record type now covers sources that are not web pages: a local document is
  identified by a `file` reference and supplied text by a `text` reference (ADR-0011),
  while a web source's identifier is exactly what it was before. No document is read
  and nothing is fetched yet; a caller hands in the content.
- An evidence pack can be assembled from chunks and sources, and the references in an
  answer resolved against it: resolved citations, unknown identifiers and malformed
  identifiers are reported separately.
- A conversation can be held against the model boundary: `ConversationManager.send`
  records the user turn, projects the recent context, calls the model, validates the
  result, resolves evidence references, records the assistant turn and returns the new
  immutable state.
- A turn is transactional. If the model fails, returns nothing, or attributes its
  output to a different model, the conversation is unchanged and the failure keeps
  its own type (ADR-0008).
- Evidence supplied to `send` is what citations resolve against. A reference the
  model was not given is reported as unresolved and is never resolved to a source
  (ADR-0009).
- `python -m chakaso chat` holds a conversation through the retrieval-aware runtime: interactive
  on stdin, or one message with `--message`. It builds its dependencies with `chakaso.app` from the
  resolved configuration, so the planner decides each turn, lexical/dense/hybrid retrieval runs over
  any `--file` documents, and every answer is recorded in the configured store (in-memory by
  default; durable SQLite when `[storage]` selects it). It prints a notice on stderr naming the
  engine as a development double, the active retrieval strategy, whether the corpus has documents
  (with none, a retrieving turn honestly returns no evidence), and whether answers persist. The
  reply and per-turn grounding go to stderr for the report lines so stdout stays scriptable;
  `--no-retrieval` conducts turns without grounding or planning.
- Document text can be split into evidence chunks with section paths, document
  positions and content-derived identifiers, and those chunks can be assembled into
  an `EvidencePack`. Nothing yet ranks chunks.
- A local document the caller names explicitly (`.txt`, `.md` or `.markdown`) can be
  read into a `SourceRecord` and its chunks — identified by path, normalized, split.
  Supplied text is ingested the same way with no filesystem access. Ingestion reads only
  the one path it is given: a directory, an absent path, an unsupported format, an
  oversized file or non-UTF-8 bytes are each refused with a specific error, and an empty
  file yields a valid source with zero chunks.
- Supplied text can be normalized deterministically (line endings, Unicode NFC, trailing
  whitespace, blank-line runs) so two formatting variants of one document produce one
  set of chunk identifiers. It preserves leading indentation and changes no words; it
  does not parse Markdown fences.
- Ingested documents can be collected into an in-memory `Corpus`: adding is idempotent
  (duplicate documents do not grow it), a chunk cannot be filed under a source that does
  not own it, and enumeration is deterministic regardless of insertion order.
- Chunks in a corpus can be ranked for a query by a deterministic lexical retriever
  (BM25): `build_lexical_index` then `LexicalRetriever.retrieve` returns ranked
  `RetrievalResult` records carrying a score, a rank and the matched terms. It matches
  shared words only — no embeddings, no model, no network — ties break by chunk
  identifier, and a query that matches nothing returns nothing rather than inventing a
  result.
- Chunks can also be ranked by **dense similarity** behind the same `Retriever` protocol:
  `build_dense_index` then `DenseRetriever.retrieve` embeds the query and every chunk through an
  `EmbeddingModel` boundary and ranks by cosine or dot similarity, returning `RetrievalResult`
  records tagged `dense` with no invented matched terms (ADR-0023). Its only embedding is
  `FixtureEmbeddingModel`, a deterministic hashed bag-of-tokens double declared `is_semantic=False`
  — it exercises the boundary and reproduces exactly, but it is not semantic and a dense result from
  it means token overlap under a different metric, not understanding.
- Chunks can be ranked by **hybrid fusion**: `HybridRetriever` composes the lexical and dense
  retrievers with transparent reciprocal-rank fusion, returning one ranking tagged `hybrid` where each
  result records the lexical and dense ranks/scores that produced it (`HybridProvenance`, ADR-0024).
  It is deterministic and offline; over the fixture embedding it re-ranks token overlap, so it
  demonstrates the fusion mechanism, not any semantic advantage.
- A query against a corpus can be run through `RetrievalService`, which ranks chunks,
  records the retriever's score onto the pack's chunks, and assembles a valid
  `EvidencePack` with its sources; `ConversationManager.send(evidence=...)` can now be
  handed a pack that something actually produced. An unmatched query gives an empty pack,
  never fabricated evidence.
- `python -m chakaso retrieve --query ... --file ...` runs the whole local pipeline from
  the command line: it ingests the named files, retrieves, and prints ranked evidence with
  its provenance. It reads only the files given, touches no network and generates no
  answer; an empty retrieval says so. `--source src_…` (repeatable) restricts the query to
  named sources; an identifier that is malformed or not among the ingested documents is
  refused with a clear error rather than filtering silently to nothing.
- `python -m chakaso sources list|show` and `python -m chakaso evidence show` inspect the
  in-memory corpus built from the `--file` documents a caller passes: source inventory with
  kind, reference and chunk count; one source and its chunks (`sources show --text` prints
  the bodies); and one chunk with its source. There is **no durable source index** — a source
  or chunk appears only while its document is given, and each command says so rather than
  implying a persistent store. Identifiers are validated before lookup, never normalised into a
  different object. `--json` keeps the exact stored bytes; the human-readable output renders
  untrusted document text inert so a hostile file cannot emit terminal-control sequences.
- `python -m chakaso benchmark` runs the retrieval development benchmark end to end from the
  command line and prints its report (text, or `--json` for a deterministic machine-readable
  report). It scores retrieval over the synthetic fixtures, labels itself a development
  instrument, and generates no answer.
- A web source acquired by the opt-in fetcher can be turned into evidence:
  `ingest_acquired` decodes the bytes, parses HTML into text with headings via a narrow
  reader (not a browser), normalizes and chunks it into the same records a local file
  produces. Fetched text is stored as data and is never an instruction or a source
  identity (ADR-0003); a `CachingFetcher` avoids refetching a canonical URL.
- A retrieval development benchmark runs end to end (`chakaso.benchmark`): a versioned,
  hand-built synthetic corpus scored by `RetrievalBenchmarkRunner` into recall@k /
  precision@k / MRR / forbidden-hit / false-retrieval metrics, reported as text or
  deterministic JSON that labels itself a development instrument. It measures retrieval
  only; a pinned content fingerprint regresses any unintended change to the fixtures.
- A grounded-answer evaluation workload runs end to end (`chakaso benchmark grounded`, ADR-0031 /
  P27): a versioned, **self-authored, Apache-2.0, human-inspectable** set of grounded-answer cases
  (each a question, acceptable and unacceptable evidence by passage label resolved to content-derived
  `ChunkId`s, required facts, prohibited claims, an expected disposition, a rationale, a provenance
  record and a development/held-out split), scored by `GroundedEvaluationRunner` through the real
  `RetrievalService` and `evaluate_answer` over an injected answer provider. It reports retrieval and
  answer dimensions separately (recall@k / precision@k / MRR / forbidden hits / false retrievals;
  evidence coverage / citation validity / unsupported-claim / required-fact coverage /
  prohibited-claim / appropriate-abstention / grounded-answer rates), classifies retrieval vs
  generation vs evaluation failures distinctly, and its text and `--json` reports label themselves a
  development instrument stating whether the provider was an oracle. The `--split heldout` path
  reports a number on cases that never influenced selection. **It measures the harness and retrieval,
  not answer capability** — the reference provider is an oracle (not a model) and the runtime still
  has no model that answers (K-001).
- An answer can be decomposed into claims (`chakaso.claims`): a `ClaimExtractor` builds
  immutable claims from a supplied decomposition or a documented sentence heuristic, each with
  a stable content-derived identifier and an evaluation status that means "the supplied
  evidence supports this", never "this is true" (ADR-0014). No language model is involved.
- A claim's citations can be validated structurally (`chakaso.citation`): a reference to a
  chunk that was supplied is *valid*, one outside the pack is *unknown*, one present but
  outside a case's expected evidence is *irrelevant*, and a required claim that cites nothing
  valid is *uncited*; precision/recall count these without claiming a source proves a claim.
- Grounding turns those citations into a per-claim status (`chakaso.grounding`): the structural
  evaluator marks supported / unsupported / not_evaluated and cannot invent a contradiction,
  while a manual evaluator is the only way *contradicted* or *uncertain* appear; a
  `Contradiction` records the disagreement without naming a winner, and precedence is a
  caller-supplied hook, never an automatic ranking (ADR-0015).
- `evaluate_answer` combines the citation and grounding results into one view that keeps the
  dimensions separate — citation precision, evidence coverage (the grounded-claim support
  rate), grounding status and contradictions — and deliberately reports no single merged
  quality score. Claims and evidence are supplied by a caller or fixture, not generated.
- An answer's claims can be reassessed against evidence and the outcome decided and recorded
  (`chakaso.correction`): `decide_claim` / `decide_answer` turn prior-vs-now statuses into a
  per-claim disposition and an answer-level decision under a fixed, written rule (ADR-0016) — a
  contradiction outranks everything and nothing corrects for recency or source order;
  `record_reassessment` fixes the outcome into an immutable, content-identified
  `CorrectionRecord` that supersedes an earlier answer; `correction_metrics` reports success,
  unjustified-persistence and unnecessary-revision rates over a caller-labelled set. With no
  model, nothing generates revised wording, and no turn triggers reassessment by itself.
- An answer can be recorded as a durable-by-design `AnswerRecord` and looked up: `chakaso.answers`
  gives a per-event `AnswerId` (not a content hash, ADR-0017), an immutable record that references
  the turn, model, evidence, claims, evaluation and a `correction_of` link, and an `AnswerStore`
  whose in-memory implementation is append-only and refuses to overwrite a saved answer or to link
  a correction to a missing or cross-conversation answer (ADR-0018). A second backend,
  `SQLiteAnswerStore`, persists the same history to a local SQLite file — an explicit schema version,
  transactional all-or-nothing appends, strict-read integrity checks, and a test proving it survives a
  process restart (ADR-0021). It is opt-in, so nothing is written to disk by default, and a stored
  answer's evaluation is recomputed rather than persisted.
- A completed turn can be persisted atomically: given a store and a claim extractor,
  `ConversationManager.send` builds and saves the `AnswerRecord` *before* adopting the new
  conversation, so a persistence failure aborts the turn and leaves the conversation unchanged
  (ADR-0008 extended to the record). Without a store the turn behaves exactly as before.
- Reassessment is an application operation over a *stored* answer: `reassess_stored_answer` /
  `reassess_and_record` take an `AnswerId` and new evidence, re-ground the recorded claims, and
  return a decision and a `CorrectionRecord` linked to the prior. The follow-up boundary is
  identifier-addressed — no natural-language reference resolver is pretended. Since ADR-0029 the
  runtime composes it into a conversation: `chakaso chat` reads a follow-up for an explicit `ans_…`
  reference through a deterministic trigger and, on exactly one valid target, re-grounds the named
  answer against its standing evidence plus whatever the follow-up retrieves (a read-only pass that
  runs no model and records nothing), then prints the decision and states plainly that no revised
  wording was produced because there is no model and the original answer is untouched. A named-but-
  unknown target is refused (`UnknownAnswerError`) and a follow-up naming two distinct answers is
  refused as ambiguous (`AmbiguousReassessmentTargetError`); a follow-up that names nothing is an
  ordinary conversational turn.
- A semantic grounding *boundary* exists (`SemanticJudge` / `SemanticGroundingEvaluator`,
  ADR-0019) and composes into answer evaluation and reassess without changing their signatures, so
  grounding stays a separate dimension from citation. Its verdict now carries the evaluator's name
  **and version** (so a change of judge is visible in provenance), and an evaluator that fails or
  abstains raises `SemanticJudgeError`, which the adapter records as *not evaluated* — never as
  support (ADR-0027). Its only implementation remains a caller-supplied `FixtureSemanticJudge`; it
  names no winner in a conflict and there is no real semantic judge, as `describe_semantic_capability()`
  and `chakaso capabilities` report. A versioned synthetic semantic-evaluation dataset and a
  deterministic scorer exist (`chakaso.evaluation.semantic_benchmark`, `chakaso benchmark semantic`)
  as a development instrument measuring the harness — the fixture's perfect score is structure-replay,
  never semantic capability.
- A correction development benchmark runs (`chakaso.benchmark.correction`): synthetic cases whose
  expected decision (retain / correct / qualify / needs_review / abstain) the written rule is
  checked against, scored per case and in aggregate, reported as deterministic text/JSON that labels
  itself a development instrument, with a pinned-fingerprint regression.
- Reproducible experiments (`chakaso.experiments`): `run_experiment` drives a development benchmark
  into a content-fingerprinted `ExperimentResult` whose `result_id` ignores the timestamp and host;
  `compare_results` gives a deterministic regression baseline that refuses incompatible benchmark or
  evaluator versions. `chakaso benchmark` and `chakaso experiment run` expose these from the CLI.
- The durable store is configurable and inspectable: `config.storage` carries the backend
  (`memory` default, or `sqlite`), the database path and the SQLite journal mode, validated by the
  existing typed schema (ADR-0006, ADR-0021). `chakaso storage check --path ...` reports a store's
  schema version and integrity read-only; `chakaso storage migrate --path ...` creates or confirms
  its schema. Local, offline, and nothing is created or changed unless a real path is given.
- `python -m chakaso capabilities` reports, read-only and offline, what the resolved configuration
  selects: the model adapter and whether it is a development double, the retrieval strategy and
  limits, and the answer-store backend (with its path). It builds only the model to read its own
  metadata — it opens no store and reads no files — so inspecting capabilities never creates a
  database. `--json` gives a machine-readable report.
- `python -m chakaso answers list|show|lineage` inspects the persisted answer history from the
  configured durable store (ADR-0021): list recorded answers with their provenance, show one by
  `ans_…` identifier, or walk a correction chain oldest-first. It is read-only (`initialize=False`,
  no journal write) and honest about the in-memory default — which persists nothing across
  processes, so there is nothing to inspect unless `[storage]` selects `sqlite` with a real file.
  `answers show` now also reports the turn's **historical evaluation provenance** (ADR-0028) — the
  citation counts, per-claim structural-grounding statuses, the evaluator name and version, and the
  unresolved/malformed references in the answer text — explicitly labelled *recorded at the turn,
  not recomputed on read*; `--json` carries the exact stored values while the human view renders any
  untrusted string terminal-safe. It keeps the dimensions separate and states that `is_grounded` and
  `evidence_coverage` are structural, never a truth verdict. It also states that an answer's evidence
  references are **identifiers only** — the durable store does not retain the cited content, which
  lived in the ephemeral in-memory corpus and is not re-readable from the store.
- CI runs all of the above on three Python versions, with no secrets and no network
  access to a model provider.
- Untrusted text (retrieved evidence, source titles/references, model replies) is rendered
  inert on the human-readable CLI paths: control characters and bidirectional overrides are
  replaced with visible escapes by `chakaso.core.safe_for_terminal` before they reach a
  terminal, so the application does not emit terminal-control sequences from hostile content.
  `--json` output is not passed through it — the serializer keeps the exact bytes. This is a
  display-boundary guarantee, not a claim a model resists prompt injection.

**The runtime's default model is still a development double.** Three implementations sit behind the model
boundary: `deterministic` (declared as a double in its own metadata, and the default in `configs/default.toml`),
`trained_char` behind `[train]`, and `subword_transformer` behind `[nn]`, the last two loaded by
`chakaso.models.load_model(path)`. Neither trained adapter is ever a default, neither falls back to the double,
and both generate text that loops and means nothing — so a conversation can still be held, but nothing
intelligent is produced by it, nothing is retrieved unless a caller supplies evidence, and no citation resolves
on its own. A shell that replies fluently invites the wrong conclusion, so `chat` says what it is before it says
anything else. Training a toy model is not the same as the product having a useful language model, and this file
keeps those apart.

## What is tested

| Area | Coverage |
| --- | --- |
| Package installation | Version shape, distribution metadata agreement, console script entry point, `py.typed` |
| CLI | Version, help, unknown command, `info`, `config show` including failure paths, `capabilities` (model/strategy/storage report, JSON, unknown-adapter failure), the retrieval-aware `chat` (grounding over `--file`, `--no-retrieval`, `--strategy`/`--top-k` overrides, durable-store recording, missing-file and bad-top-k failure), `retrieve` (ranked output, `--top-k`, missing-file and unsupported-format failure, `--source` scoping and refusal of an absent/malformed source), `sources list|show` and `evidence show` (inventory, `--json`, empty-corpus honesty, `--text`, malformed and unknown identifiers, hostile document text rendered inert while `--json` keeps exact bytes), `answers list|show|lineage` (memory and absent-file inspection say there is nothing to read, a chat-written durable record is listed/shown/walked, malformed and unknown identifiers fail), and `storage check`/`migrate` (path resolution, absent file, healthy report, non-store and newer-schema refusal) |
| Configuration | Defaults, file layering and precedence, provenance, unknown keys, wrong types, boolean-vs-integer, ranges, name pattern, missing file, directory, invalid TOML, array values, immutability, schema/dataclass agreement, shipped file vs built-in defaults, and the `storage` section (defaults write nothing, backend/journal-mode patterns rejected) |
| Identifiers and hashing | Digest agreement with `hashlib`, UTF-8 handling, truncation bounds, part-separator ambiguity, derivation determinism, deduplication, content-change distinction, position and text sensitivity, malformed identifier rejection, ordering and hashing |
| Model boundary | The inherited contract suite (metadata, provenance of results, repeatability at zero temperature, empty-request rejection, length ceiling, unsupported-capability failures, declared capabilities being implemented), plus capability reconciliation, parameter validation, registry failure paths and the double's documented behaviour |
| Evidence | Canonicalization idempotence and non-merging, tracking-parameter removal, scheme and credential refusal, IPv6 handling, record immutability, naive-timestamp rejection, content-hash validation, change detection, pack validation, and citation resolution including fabricated and malformed references |
| Conversation manager | Construction and context bounds, dependency injection through the model boundary, first and follow-up turns, context projection limits, previous state preserved, failures leaving state unchanged for the next turn, model errors propagating untranslated, clock regressions, evidence and citation provenance per turn, evidence not leaking between turns, and end-to-end wiring against the deterministic double |
| Answers and durable storage | Answer identity and record invariants, the serialization codec (round-trip, evaluation-not-persisted, malformed-document rejection), the historical `evaluation_provenance` snapshot (round-trip, backward-compatible decode to `None` for pre-provenance documents, corrupt-count/non-object rejection, unknown evaluator identity stays visible), the append-only invariants run over **both** backends (in-memory and SQLite conformance, including provenance surviving save/read identically on both), and the SQLite store's restart durability (including a separate-process write), transaction rollback, schema/version refusal, and on-disk corruption detection. The `answers show`/`list` CLI reports the historical provenance (human + `--json`) and renders it terminal-safe against control characters embedded in a persisted evaluator name or answer text |
| Query planning | Mode/retrieval-flag consistency and plan construction invariants (reuse needs sources, a retrieving plan needs a non-empty query, every plan explains itself), meaning-preserving normalization (whitespace/Unicode/trailing-punctuation, idempotent, total, interior punctuation kept), the deterministic rules over a conversational sequence (greeting → none, question → fresh, follow-up → reuse prior sources, topic change → no stale reuse), unbacked-mode refusal, and protocol conformance |
| Dense retrieval | The fixture embedding's honest metadata (`is_semantic=False`, batch order, determinism, unit-length, zero-vector empty text), vector/metadata validation, index invariants (uniform dimension, no duplicate chunk, empty index), and the dense retriever: `dense` strategy tagging with empty matched terms, ranking/determinism, `source_ids` filter, zero-query and empty-index returning nothing, dimension-mismatch rejection, and `Retriever`-protocol conformance |
| Hybrid retrieval | Reciprocal-rank fusion of the lexical and dense retrievers: `hybrid` strategy tagging with full per-component provenance (both ranks when a chunk appears in both, `None` for the component that missed it — never fabricated), exact RRF score arithmetic, weight-driven favoring, `candidate_k` pooling, `source_ids` pass-through, determinism, invalid-config refusal, and `Retriever`-protocol conformance |
| Conversation runtime | A grounded turn through the real pipeline: a greeting retrieves nothing but still records, a knowledge turn retrieves and records its evidence with the planner decision + retrieval strategy on the answer, a follow-up reuses prior sources, a topic change does not, dense/hybrid flow through, an unwired mode or bad top-k is refused at construction, a retriever failure leaves the conversation and store untouched, and the evidence-context renderer emits ids + reference + text (empty pack → nothing) |
| Application composition | `chakaso.app` turns typed configuration into a running system: the built-in config wires the runtime over the in-memory store; the configured store is used and never silently replaced; `sqlite` with no path is refused rather than downgraded; changing `strategy`/`top_k`/`enabled` changes what the assembled application actually does on a turn; a durable-backed answer survives closing and reopening the store; corpus building reads only named files and bounds their size; `close()` is idempotent |
| End-to-end & security | The whole flow as integration: a grounded turn persists to the durable SQLite store and survives a reopen (substance kept, evaluation recomputed), a `correction_of` lineage survives a restart, a failing save aborts the turn and leaves the conversation unchanged, and untrusted retrieved text stays data with a never-supplied identifier resolving to nothing; the no-network import guard covers the planning and runtime packages |
| Training and the inference adapter | Dependency-free: corpus provenance, deterministic character vocabulary (codepoint ordering, unknown-id handling, fingerprint), leak-free split (determinism, unit disjointness, a non-tautological anti-leakage invariant with a vacuous-pass guard, window padding), `TrainingConfig` validation, uniform/unigram baseline arithmetic. Numpy-gated (`[train]`): two analytically-known forward losses, gradient shapes/finiteness, **central-difference verification of every parameter scalar**, Adam updates and loss reduction, seeded reproducibility, checkpoint bit-for-bit round-trip and corruption/incompatibility/vocabulary-mismatch refusals, held-out evaluation matching the loop, the E2601 run being deterministic and beating both baselines with its overfitting reported; the `trained_char` adapter inherits the full shared `LanguageModel` contract over a real checkpoint, is greedy-deterministic, declares itself not-the-double, and refuses empty/missing/malformed checkpoints without falling back. The dependency policy (empty base deps, numpy only in `[train]`, bounded for the 3.11 floor) is a hygiene test |
| Repository invariants | Private pack never tracked, `.gitignore` rule present, no secret-shaped files, no tracked file over 1 MiB, no commercial provider dependency, base install declares no runtime dependency and numpy is only in `[train]` bounded for the Python floor, ADR numbering and indexing, documentation links resolve |

## What is experimental

The semantic grounding boundary (ADR-0019) and the experiment/comparison layer (ADR-0020) are
new research scaffolding: implemented, typed and tested, but with only a fixture semantic judge and
no validated real-workload use. The correction development benchmark scores a decision rule over
synthetic cases, not measured capability. None of these is a claim that the approach works on real
data — that is K-001, and it stands. The code that exists still does one small thing each. The
dense retrieval boundary (ADR-0023) is the same kind of scaffolding: a real retriever and an exact
index, but over a non-semantic fixture embedding, so it proves the shape, not any dense quality.
The training layer (`chakaso.training`, ADR-0030/0034/0035) is experimental in a stronger sense: it trains
and measures *real* models — E2601's 984-parameter character MLP over a 344-token synthetic corpus, and
E2605's 643,840-parameter subword Transformer over the repository's pinned 780 KB of prose — and neither is
wired into the runtime. Both prove the learning loop works and are honest about their scale; neither
establishes anything useful. The `[nn]` path is experimental in a second, narrower way: its gradients are
autograd cross-checked against hand-computed losses and verified by *sampled* finite differences, a weaker
guarantee than the character model's every-scalar check. That gap is documented rather than hidden, and it is
now at least checked where it runs: a CI job does install `torch` (the `model path` tier, green on the
runner), which the three tiers shipped on 2026-10-22 made true for the first time.

## What is not implemented

- A full document processor: PDF, robust main-content extraction and boilerplate
  removal. The HTML reader is narrow and is not a browser — it runs no scripts, applies no
  CSS and strips no ads.
- The retrieval-aware runtime is wired into `chakaso chat` through the composition root
  (`chakaso.app`, ADR-0025/ADR-0026), but nothing yet decides *which* URLs to fetch — the fetcher is
  only ever handed an explicit URL, and the CLI corpus comes only from `--file` documents a caller
  names. There is no durable source index: the corpus is rebuilt in memory each run.
- Dense embeddings, a real semantic embedding model, an approximate vector index and reranking:
  a dense *boundary* and an exact in-memory index exist over a deterministic non-semantic fixture
  embedding (ADR-0023), but there is no genuine embedding model, no approximate (FAISS-like) index,
  and no reranking
- A real semantic judge. A semantic grounding *boundary* exists (`chakaso.grounding`, ADR-0019)
  but its only implementation replays caller-supplied relations; nothing decides automatically that
  evidence proves or contradicts a claim. Whether a cited source actually *proves* a claim — and
  automatic contradiction or uncertainty detection — does not exist.
- An automatic correction loop and a revised answer. Reassessment is an application operation
  over a stored answer, there is a correction development benchmark, and since ADR-0029 a follow-up
  that *explicitly names* a stored answer composes into a reassessment through the runtime. But
  nothing decides on its own that a turn needs reassessing (the trigger fires only on an explicit
  `ans_…` reference, never a guessed one), a vague reference like "my last answer" is not resolved,
  and nothing generates revised prose (the model in that path is a development double). An answer can be
  persisted durably
  (ADR-0021), but that is a store a caller selects, not an automatic correction.
- A general evaluation harness and external model benchmarks. Retrieval and correction development
  benchmarks, metric functions, and structural claim/citation/grounding/answer evaluation exist, and model
  metrics are now measured too (held-out loss, perplexity, bits per character and token accuracy against
  uniform/unigram baselines — E2605). What does not exist is any *external* benchmark, a real
  grounded-answer benchmark, or a comparison against another model: every model number is measured on this
  repository's own text, and every harness number is over synthetic fixtures that check structure, not
  meaning.
- Training and model code *beyond the measured scale*: `chakaso.training` has two tokenizers (character
  and learned byte-pair subword), a content-pinned corpus manifest, a checksummed corpus snapshot that can be
  recovered from the revision that pinned it, a document-level split, sequence packing, a character MLP and a
  decoder-causal Transformer, a full-batch and a mini-batch loop with warmup, clipping, early stopping and
  resume, and two checkpoint formats — and E2605 measured them end to end on the repository's own 780 KB of
  text, E2606 across five seeds, E2607 on an external licensed corpus. What is *not* built: gradient
  accumulation in a production path (resume is wired into tests, not into an experiment run), distributed or GPU
  execution, a key/value cache during decoding, streaming generation, and any model at a scale where a number
  could be called capability.
- A *useful* inference model, and committed/released model weights: none. Two trained adapters exist behind
  the boundary — `chakaso.models.trainedchar` (an E2601-class character checkpoint, behind `[train]`) and
  `chakaso.models.nntransformer` (E2605's subword Transformer, behind `[nn]`, loaded by
  `chakaso.models.load_model`) — and both generate text that loops, repeats and means nothing. Neither is
  ever the default, neither checkpoint is committed (both are reproducible from seed, config and the pinned
  manifest, and are recorded by checksum), and there is no pretrained or provider-backed adapter at all.
  What is *not* built: beam/nucleus sampling, a tokenizer with normalization options beyond the named
  pre-tokenizer, or any model that produces a useful grounded answer.
- Automatic persistence of correction records and experiment results. Answer history can now be
  persisted durably to SQLite (ADR-0021), but a `CorrectionRecord` or an `ExperimentResult` is still
  only serializable — the runtime does not write it to disk unless a caller wires it to a store.

## Decisions made

| Record | Decision |
| --- | --- |
| ADR-0001 | No required commercial inference or search dependency at runtime |
| ADR-0002 | Application code depends on a language-model boundary, not a model |
| ADR-0003 | The retrieval layer owns evidence identifiers; the model only references them |
| ADR-0004 | Apache-2.0 for code; weights and datasets need their own policy before release |
| ADR-0005 | `src` layout, with training code as a module inside one distribution |
| ADR-0006 | TOML configuration parsed with the standard library, read-only |
| ADR-0007 | Evidence identifiers are derived from content and the canonical URL, not assigned at random |
| ADR-0008 | A turn either completes or the conversation is unchanged |
| ADR-0009 | A reference the model was not given is recorded, not fatal |
| ADR-0010 | Chunks are section-bounded and do not overlap |
| ADR-0011 | Source identity is a typed reference, not a URL |
| ADR-0012 | The fetcher is an optional, explicitly-bounded adapter |
| ADR-0013 | Benchmark cases have a stable identity and a content-derived version |
| ADR-0014 | A claim is a first-class, content-identified unit; status is evaluation, not truth |
| ADR-0015 | Grounding is a boundary; structural evaluation never claims semantic support |
| ADR-0016 | The correction decision is an explicit, evidence-driven rule |
| ADR-0017 | Answer identity is a per-event identifier, not a content hash |
| ADR-0018 | Answer persistence is an append-only store boundary, in-memory for now |
| ADR-0019 | Semantic grounding is a boundary with only a fixture implementation today |
| ADR-0020 | Experiments are content-fingerprinted, environment-separated research artifacts |
| ADR-0021 | Durable answer storage is a SQLite backend behind the existing store boundary |
| ADR-0022 | Query planning is a deterministic decision boundary, not a model |
| ADR-0023 | Dense retrieval is an embedding boundary with an exact index; the only embedding is a non-semantic double |
| ADR-0024 | Hybrid retrieval fuses lexical and dense results by reciprocal rank fusion |
| ADR-0025 | The retrieval-aware conversation runtime is a thin coordinator, not a monolith |
| ADR-0026 | The application composition root is a thin, injectable layer separate from the runtime |
| ADR-0027 | The semantic-evaluation contract carries evaluator version and an explicit error outcome; no real evaluator is simulated |
| ADR-0028 | An answer persists the historical evaluation it reached; the live view stays recomputed |
| ADR-0029 | Reassessment is triggered by an explicit identifier and composed at the application root |
| ADR-0030 | A minimal, optional numerical dependency (numpy in a `[train]` extra) gates any from-scratch training |
| ADR-0031 | The grounded-answer evaluation workload is self-authored, provenance-bearing data |
| ADR-0032 | Evidence sufficiency is a deterministic, corpus-statistical decision at the retrieval boundary |
| ADR-0033 | The product is a trained model; the Python library is the primary interface and the CLI a secondary developer utility |
| ADR-0034 | `torch` is admitted as a separate optional `[nn]` extra for the model path only, confined to named modules |
| ADR-0035 | A Transformer checkpoint is one checksummed, self-describing file carrying its tokenizer, with usable and resumable kinds separated |

Two process facts are recorded outside the ADR series because they are repository
history rather than architecture:

- The repository history was restarted from an empty tree; the previous head is
  preserved as tag `archive/bootstrap-attempt-1`
  ([`../internal/engineering/repository-history.md`](../internal/engineering/repository-history.md)).
- The private planning pack lives in `docs/*.docx`, is git-ignored, and is never
  committed.

## Known gaps and risks

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md). K-001 still stands and is the important
one: nothing in this repository has been executed against a real workload. The model
path is now measured on real text (E2605, on this repository's own documentation), but a
workload means a grounded question answered by a model and checked, and no turn here has
ever done that. K-009 closed on 2026-10-22: the `[train]` and `[nn]` tiers are CI jobs now, read back green
from GitHub's runners, and the two defects they found on the way — a Windows-only attribute and an assertion
finer than float32 — are the evidence that the gap was worth closing rather than just documenting.

## Next step

Retrieval, the retrieval and correction development benchmarks, the claim / citation / grounding /
answer-evaluation primitives, the correction decision rule with an application-level reassessment
over stored answers, an answer store wired atomically into a turn — in-memory and, since ADR-0021, a
durable SQLite backend that survives a restart — a deterministic query planner that decides per turn
whether and how to retrieve (ADR-0022), a fixture-only semantic grounding boundary, and reproducible
experiment records with a deterministic regression baseline, and a retrieval-aware runtime composing
planner → retrieval → model → record → store (ADR-0025) are all built and tested. The measured
strategy comparison (E-001) and the end-to-end/restart integration and security demonstration are done.
Source and evidence inspection from the CLI over the in-memory corpus, source-scoped retrieval,
adversarial terminal-safety of untrusted content, answer-provenance enrichment (ADR-0028) and a
triggered, explicitly-composed reassessment (ADR-0029) are all also built and tested — the two items
this section previously listed as "next" are now shipped.

**P23 (ADR-0030) and P24 (the experiment) are now complete.** ADR-0030 admitted `numpy` as a single
optional `[train]` extra with hand-derived, finite-difference-verified gradients, keeping it out of the
base install and the core CI matrix. `chakaso.training` then built the pipeline and **E2601 produced the
project's first genuinely measured result**: a 984-parameter character model trained from random
initialization whose held-out next-character cross-entropy (best 1.6650 nats, final 2.4946) beats both
the uniform (3.3322) and unigram (3.0531) baselines — recorded honestly with the measured overfitting.
Full record: [`../../experiments/E2601/record.md`](../../experiments/E2601/record.md).

**P25 is now complete:** an optional `trained_char` adapter (`chakaso.models.trainedchar`) loads an
E2601-class checkpoint behind the `LanguageModel` boundary — rebuilding the checkpoint's own vocabulary,
generating greedily within `max_new_tokens`, declaring `development_double=False` with a real `TOKENIZE`
capability, and failing explicitly (never falling back to the double) when the path, the dependency, or
the checkpoint is missing or corrupt. It is selected only through `model.adapter`/`checkpoint_path`, is
imported lazily so the base install and `import chakaso.models` stay numpy-free, and satisfies the same
inherited contract as every other adapter. The model behind it is a tiny toy that generates incoherent
text; its value is a real training-and-inference path, not usefulness — stated plainly.

**P27 is now complete:** `chakaso.benchmark` grew a grounded-answer evaluation sub-module (ADR-0031) —
a provenance-bearing `GroundedCase`/`GroundedSource` schema, a `GroundedDataset` that refuses dangling
evidence, duplicate cases and development/held-out split leakage, a strict loader, a **self-authored
Apache-2.0** workload (`ga-kestrel-library`, 13 passages / 14 cases, pinned fingerprint
`d2d2ee00d117b88a`), an evaluation runner over the real retrieval + `evaluate_answer` boundaries with
per-stage failure classification, hand-checkable metric primitives over separate dimensions,
deterministic reports, and `chakaso benchmark grounded`. **E2603 ran it** and recorded honest numbers —
retrieval recall@k 0.8182 / MRR 0.8636 over all cases (held-out recall@k 1.0), zero stage failures, and
genuine negatives (3 forbidden hits, 3 false retrievals on abstention cases) that the separate
dimensions keep visible ([`../../experiments/E2603/record.md`](../../experiments/E2603/record.md)).
It is a development instrument over a self-authored workload; it evaluates the harness and retrieval,
not answer capability — there is still no model that produces a grounded answer.

**P28 is now complete:** E2603's measured false retrievals drove a deterministic evidence-sufficiency
decision (`chakaso.retrieval.sufficiency`, ADR-0032) layered additively over retrieval — the retriever still
returns top-k; a separate policy says whether the pack is adequate, judged by whether a retrieved chunk
shares a *discriminating* (non-function-word, low-document-frequency) query term, and it names the terms so
the verdict is inspectable. Measured by **E2604**: **100% sufficiency accuracy on the workload** (all three
unanswerable cases judged insufficient, **zero** answerable cases wrongly abstained — recall preserved),
with retrieval's own `false_retrievals`/`forbidden_hits` metrics and the runtime intentionally unchanged
([`../../experiments/E2604/record.md`](../../experiments/E2604/record.md)). It is honestly bounded: it holds
only over a narrow corpus-specific threshold band, is not a calibrated confidence, and does not close K-010.

**P30 and P31 are now complete, and they re-ordered the project.** [ADR-0033](../decisions/ADR-0033-model-centred-product-contract.md)
made the model the product and the Python library the primary interface; P30 then built the model path's data
layer with no new dependency at all (`training.manifest`, `data.split_text_units`, `training.subword`,
`training.packing`, `training.model_metrics` — all in the dependency-free CI matrix), and
[ADR-0034](../decisions/ADR-0034-optional-pytorch-model-extra.md) re-opened ADR-0030 on its own written
revisit conditions to admit `torch` as a separate `[nn]` extra, recording that the speed argument did not
survive measurement (1.16× on a same-shape matmul) and that the gradient guarantee became sampled rather than
exhaustive. P31 built `training.transformer`, `training.nn_training`,
`training.nn_checkpoint` ([ADR-0035](../decisions/ADR-0035-transformer-checkpoint-format.md)),
`training.nn_evaluation`, the `training.subword_experiment` driver, `models.load_model` and the
`subword_transformer` adapter, plus `chakaso experiment model` as a thin handler.

**E2605 produced the project's first Transformer measurement**
([`../../experiments/E2605/record.md`](../../experiments/E2605/record.md), record committed before the run):
on held-out documents of the pinned 71-document / 780,223-character corpus, **1.8657 bits per character**
(643,840 parameters; 2.6936 nats/token, perplexity 14.79, token accuracy 0.4501) against **3.1343** for
E2601's character architecture, 3.3224 for the unigram and 4.8010 for the uniform predictor of its own
identifier space. Early stopping selected step 1380 of a 2400-step budget and the un-stopped model was 0.1434
nats worse. Wall clock 513.6 s + 69.9 s + 13.2 s on CPU, Python 3.13.5, torch 2.13.0+cpu. The record also
documents its own correction: the pre-run text quoted a corpus figure (754,789 characters) that the run's
pinning superseded, and three committed generation samples that loop.

E2605's follow-up list is now executed, so the next unit is set by what its results left open. **E2606
measured five pre-declared seeds** — 1.8794 ± 0.0077 bits per character against 3.0937 ± 0.0267 for the
character architecture, every seed below both baselines, and the seed-7 run reproducing E2605 to the last
decimal with all 41 stored tensors equal. **E2607 measured the same protocol on externally licensed prose**
(three Project Gutenberg texts, 105 documents, 1,660,274 characters): 2.0176 ± 0.0118 against 3.2218 ±
0.0153, a gap of 1.2042 against 1.2143 internally, so the advantage is not a property of this repository's own
writing — while the generations on that corpus degenerate into repeated function words and words the corpus
does not contain.

The next real unit is the one E2607's own caveat makes unavoidable: **the same external corpus at a budget that
converges.** No run early-stopped there — all three selected the 2,400-step ceiling with validation still
falling — so its reported level is truncated, and a capacity question (context length, layers, vocabulary)
asked against an unconverged baseline cannot be answered. Only after that does a capacity comparison become
interpretable, and only after a corpus and a budget that both converge does anything in this project have a
claim to make about a workload. **P29** (calibrating the sufficiency policy) and **P32** (runtime consumption
of that decision) stay deferred by [ADR-0033](../decisions/ADR-0033-model-centred-product-contract.md), and
nothing in either study changes their content: the trained checkpoints are loadable from Python and still
answer nothing grounded. A genuinely useful pretrained adapter (Option A of
[`../research/model-feasibility.md`](../research/model-feasibility.md)) and a real `SemanticJudge` remain
gated on their own dependency-feasibility decisions. All of it stays behind the existing boundaries, and
K-001 — that nothing has met a real workload — still stands. See
[`ACTIVE_TASK.md`](ACTIVE_TASK.md) for the definition of done.