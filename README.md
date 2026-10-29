# Chakaso

Chakaso is a **language-model research project**. It trains its own model weights from random
initialization, on data it can name and re-verify, and documents what the trained model measurably does —
including the things it cannot do. The long-term artifact is a distributed model with a reproducible
training pipeline, a documented architecture, controlled data provenance and a usable inference interface.

[ADR-0033](docs/decisions/ADR-0033-model-centred-product-contract.md) makes that the product definition,
and everything else in the repository serves it. A local-first conversational system — retrieval, evidence
identity, citation validation, grounding, correction, durable answers — exists here too, tested and
useful, because a grounded model can only be measured against infrastructure that can check it. It is a
consumer of the model, not the other way round, and it is not the primary interface: **the Python library
is.**

## What exists today

| | |
| --- | --- |
| **A trained model you can load from Python** | `chakaso.models.load_model(path)` returns a `LanguageModel` for a checkpoint this project trained, or for a **release directory** whose contract is verified before a weight is read. No terminal, no configuration file, no runtime import. |
| **The runtime around it** | **Implemented.** Bounded decoding through one validated parameter object (`temperature` over a stated range, `top_k`, `top_p`, a seed, `stop_sequences`, `stop_token_ids`, `max_new_tokens`), a generation result that says how it ended and how much it produced, `stream()` that yields as the model steps and stops when the caller cancels, a `score()` that returns the same held-out arithmetic the experiments use, `describe()` for the facts a caller cannot inspect from outside, and **versioned** prompt templates — the default is the shape the measured runs used, and a role-labelled template is opt-in rather than a claim about instruction following. |
| **A release contract** | **Implemented.** `chakaso.models.release` writes a model as one directory: `model.json` assembled from the run's own report plus the checkpoint's declared facts, the artifact's SHA-256 checked at load, licence and redistribution stated, and `known_limitations` required to be non-empty. A release that does not verify is a typed error, never a plausible generation from somebody else's weights. |
| **An optional local HTTP endpoint** | **Implemented.** `chakaso.serve` puts a loaded model behind `/health`, `/v1/models`, `/v1/completions` and `/v1/chat/completions` with server-sent events, using only the standard library: loopback by default, bounded concurrency, a client that hangs up cancelling the generation, and a refusal to serve the development double or to honour a request field it cannot ([ADR-0038](docs/decisions/ADR-0038-local-http-serving-adapter-from-the-standard-library.md)). Not authenticated, and not a hosted API. |
| **A tokenizer trained here** | A byte-pair subword vocabulary (`chakaso.training.subword`): byte-level base, so encoding is total; a named pre-tokenizer, so merges never cross a word boundary; deterministic tie-breaking, so the same text yields the same fingerprint; self-describing and checksum-verified on load. A deterministic character vocabulary also exists and is what E2601/E2602 measured. |
| **A corpus with provenance** | `chakaso.training.manifest` pins a document set by path and content digest, records its licence, and *refuses* to prepare a corpus that has drifted. `chakaso.training.snapshot` recovers a pinned corpus from the revision that recorded it and writes a checksum-verified snapshot, so a measured result stays re-trainable after the repository edits its own text. Three corpora are pinned. `chakaso-repo-text-v1` — Chakaso's own tracked prose, 71 documents, 780,223 characters, Apache-2.0, fingerprint `e45993f9badf706c`, the one E2605/E2606 trained on. `gutenberg-english-prose-v1` — three Project Gutenberg texts, 105 documents, 1,660,274 characters, public domain in the US under the Project Gutenberg License, fingerprint `931b07a635ca5a0c`, the one E2607 trained on; it exists because one author's house style bounds what the first can establish. `gutenberg-english-prose-v2` — 24 Project Gutenberg texts, 919 documents, 14,490,005 characters, fingerprint `e9fc2e92f59d88ea`, the one E2608 trained on, chosen so that no single source supplies more than 14.6 % of the text and a measured data card reports its composition and its duplication screen ([`docs/data/gutenberg-english-prose-v2.md`](docs/data/gutenberg-english-prose-v2.md)). Its bytes stay in `data/`; the catalogue and the per-document digests are committed, so it is rebuildable from its own URLs. |
| **Two model architectures, trained from scratch** | `chakaso.training.transformer` (decoder-only, causal attention, pre-norm, GELU, tied head), measured by E2605/E2606/E2607 and re-run as E2608's comparison arm, and `chakaso.training.decoder` (**Implemented**, the family ADR-0037 chose from measured throughput, measured against the transformer by E2608 and again at matched capacity and context by E2611): rotary positions, RMSNorm, biases-free attention, tied embeddings, an ignore-index target mask so padding is visibly not a label, and a key/value cache that makes stepped decoding equal to a full forward pass. Each writes its own checkpoint schema and the loader picks the class from the file, never a guess. `chakaso.training.nn_training` runs either: mini-batches, warmup, cosine decay to a stated floor, gradient accumulation over an effective batch size, gradient clipping, validation-selected early stopping, throughput measured over gradient time only, a completion class that names why the run ended, and resume that refuses a changed identity while allowing a longer budget. Requires the optional `[nn]` extra. |
| **Measured results** | Ten experiment records, all measured and closed. E2601/E2602: a 984-parameter character model whose held-out loss beats uniform and unigram baselines across five seeds (mean 1.7978 ± 0.1767 nats). E2603/E2604: retrieval and evidence-sufficiency measurements over a self-authored workload. [E2605](experiments/E2605/record.md): the first subword Transformer trained from random initialization on the pinned corpus reaches **1.8657 bits per character** on held-out documents, against 3.1343 for the character architecture, 3.3224 unigram and 4.8010 uniform. [E2606](experiments/E2606/record.md): five pre-declared seeds, with the document split held fixed so only initialization moves — **1.8794 ± 0.0077 bits per character against 3.0937 ± 0.0267**, every seed below every baseline, and the seed-7 run reproducing E2605 **exactly** (all 41 tensors equal) from a corpus recovered out of git history. [E2607](experiments/E2607/record.md): the same protocol on the external corpus, three seeds — **2.0176 ± 0.0118 against 3.2218 ± 0.0153**, so the gap survives the change of data (1.2042 against 1.2143), but no run early-stopped and all three selected the 2,400-step ceiling, which makes that level a floor rather than a converged loss. Every generation those runs produced is an exact repeating cycle. [E2608](experiments/E2608/record.md) compared the two Transformer-family architectures at a matched token budget (19,660,800 positions each, one seed, one split, one tokenizer) on the 919-document licensed corpus: the 3,413,760-parameter rotary decoder reaches **1.6865 bits per character** against **1.9043** for the 1,037,056-parameter transformer, both arms beating unigram (3.2263 / 3.2265) and uniform (5.5661 / 5.5666), and **both still truncated at the step ceiling with validation loss falling**. Its three preregistered hypotheses split: the schedule did not fix the truncation (H1 failed), the decoder beat the transformer **with the confound stated** — 3.3× the parameters and 2× the context, so this is not a controlled architecture answer (H2 succeeded on its declared condition, with the confound), and the exact-cycle degeneration stopped (H3 failed in the direction nobody expected: zero periodic runs of ≥ 3, zero tail loops, for the first time a checkpoint whose samples are phrase-level repetition — "*the Error of the Emperor of the Emperor of the Soule of the Soule*" — rather than a metronome). Mean bigram repetition is still 0.5538, every sample hit its token bound, and no sample is a sentence anyone would publish. That measurement is what withholds Level 3 from `chakaso-base@0.1.0` in [ADR-0039](docs/decisions/ADR-0039-three-level-model-release-gate.md); the artifact's own statement of it is [`MODEL_CARD.md`](MODEL_CARD.md). [E2610](experiments/E2610/record.md) then calibrated the repetition against human continuations and found it is greedy decoding of a human-diverse distribution, not a pipeline defect, while no decoding setting produces coherent text. [E2611](experiments/E2611/record.md) removed E2608's confound at ~1M parameters: with the control's width, depth, heads, feed-forward, vocabulary, context, data, split, tokenizer, seed, batch shape, token budget and schedule, the decoder reached **1.8077** bits per character against the transformer's **1.9043** — a **0.0966-bit** bundle effect outside the preregistered ±0.02 band, both arms still at the 2,400-step ceiling. About 44% of E2608's gap survives the match; it is still not a capability result. |
| **An application layer a model can plug into** | BM25/dense/hybrid retrieval over an in-memory corpus, claims, citation validation, structural and semantic-grounding boundaries, a correction decision rule with application-level reassessment, an append-only answer store (in-memory by default, durable SQLite when configured), and development benchmarks. |
| **Verification** | Tests, lint, format and strict typing on Python 3.11–3.13 in CI, offline, CPU-only, no secrets and no provider dependency. **Three tiers run on the runners**: `core` with `.[dev]`, `train` with `.[dev,train]`, and `model path` with `.[dev,train,nn]` from the PyTorch CPU wheel index plus the strict type check over every module — read back green across all eight jobs, most recently run `40119588460` on 2026-10-24. Per-tier pass counts live in [`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md), which is this repository's authority for them; they are not repeated here because a count copied into a README is a count that goes stale the next commit (K-013). The dependency-free tier is what proves the default install needs no numerical library, and it is also where the HTTP serving adapter is tested. Landing the tiers found two defects no local gate could see, recorded in [`docs/internal/engineering/optional-dependency-ci-matrix.md`](docs/internal/engineering/optional-dependency-ci-matrix.md). |

## What does not exist

* **No useful language model.** The trained models are small and their data is narrow — one author's
  technical prose (71 documents, 780,223 characters) for E2601/E2605/E2606, three Project Gutenberg texts
  (105 documents) for E2607, and 24 Project Gutenberg texts (919 documents, 14,490,005 characters) for E2608,
  which is the widest corpus this project has trained on. None of them is a capable assistant, and none is
  wired into anything as a default: the runtime's default adapter is a development double that says so, and no
  grounded question in this repository is answered by a trained model. E2605–E2607's greedy samples are exact
  repeating cycles; E2608's decoder arm is the first artifact measured here with **zero periodic runs of three
  or more and zero tail loops**, and its mean bigram repetition is still 0.5538 with every sample truncated at
  its token bound — less degenerate, still not text anyone would publish.
* **No real-workload evaluation, and no claim that an external corpus closes it.** E2605 and E2606 are
  measured on this repository's own prose; E2607 and E2608 measure licensed public-domain prose for exactly
  that reason. Neither is a workload: no number here comes from a question asked of the model, an
  answer checked against evidence, or a task anybody needed doing. That is gap K-001 in
  [`docs/agent/KNOWN_ISSUES.md`](docs/agent/KNOWN_ISSUES.md) and it is the project's frontier.
* **No published weights, and no Level 3.** A release *contract* exists and one artifact satisfies it:
  `chakaso-base@0.1.0`, built by `chakaso model export` from E2608's decoder arm and assessed at **Level 2**
  of [ADR-0039](docs/decisions/ADR-0039-three-level-model-release-gate.md) — reproducible integration, with
  **Level 3 (useful behaviour) withheld on the measurement above**, not on modesty. What is published is the
  contract, the card and the commands; the bytes are not. A checkpoint is reproducible from its recorded seed,
  config and corpus fingerprint, is cited by checksum, and stays out of Git
  ([ADR-0004](docs/decisions/ADR-0004-apache-2.0-license.md) still reserves weights and datasets for their own
  distribution decision, which has not been made).
* **No real semantic judge, no genuine embedding model, no autonomous correction loop, no durable source
  index.** Each boundary exists; what would fill it honestly does not. See
  [`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md) for the authoritative list.

Every capability claim in this repository carries a status label — Implemented, Experimental, Planned,
Research, Unknown — defined in [`docs/README.md`](docs/README.md). A sentence that implies a capability
without one is a defect.

## Use a trained checkpoint from Python

```bash
git clone https://github.com/bhogesararam23/chakaso.git
cd chakaso
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,train,nn]"          # [nn] pulls torch; on Linux add
                                                    # --extra-index-url https://download.pytorch.org/whl/cpu
```

```python
from chakaso.models import load_model

model = load_model("data/experiments/E2605/E2605-transformer.pt")
result = model.sample("The retrieval layer owns", max_new_tokens=64)

print(result.text)
print(model.metadata.model_id, model.metadata.development_double)
print(model.artifact.corpus_fingerprint, model.artifact.config.as_dict())
```

`load_model` never falls back: a missing extra, an absent path or a corrupt artifact is a named error, and
the loaded object states what it is through `metadata`. It also satisfies the `LanguageModel` boundary the
application depends on, so `model.generate(messages, params)` works the same way — one abstraction, not
two. Scoring a text is equally direct:

```python
from pathlib import Path

text = Path("docs/training.md").read_text(encoding="utf-8")
nats = model.score(text)  # mean held-out cross-entropy, in nats
```

The explicit `encoding` is not decoration. This repository's prose is UTF-8 with em dashes and non-ASCII
quotes in it, and `open(...).read()` on a Windows default codec reads a different string than the one the
runner measured — which is exactly the kind of unreproducible difference the rest of this project pins by
digest.

Generation can be streamed, bounded and cancelled. The same object answers the `LanguageModel` boundary the
application uses, so a turn is a list of messages and a policy is one parameter object:

```python
from chakaso.models import GenerationParams, Message, Role
from chakaso.models.decoding import AbortSignal

params = GenerationParams(
    max_new_tokens=64, temperature=0.7, top_k=40, seed=7, stop_sequences=("\n\n",)
)
messages = [Message(role=Role.USER, content="The retrieval layer owns")]
signal = AbortSignal()

for index, chunk in enumerate(model.stream(messages, params, signal=signal)):
    print(chunk.text, end="", flush=True)
    if chunk.final:
        print("\nhow it ended:", chunk.finish_reason.value)
        break
    if index == 8:
        signal.abort()  # a caller that has seen enough: the step loop stops at the next boundary
```

A release directory is what a consumer is handed, and building one is one command: it assembles `model.json`
from the run's own report and the checkpoint's own facts, copies the two files beside it, and every later load
verifies the artifact's digest before reading a weight.

```bash
chakaso model export data/experiments/E2608/decoder --output dist/chakaso-base \
  --model-id chakaso-base --model-version 0.1.0 \
  --summary "A 3,413,760-parameter rotary decoder trained from random initialization on 14,490,005 characters of public-domain English prose, reaching 1.6865 bits per character held out." \
  --capability tokenize --capability generate --capability stream --capability score \
  --limitation "Generations repeat phrases: mean bigram repetition 0.5538 over five greedy continuations; every sample reached its token bound. No sample is usable text." \
  --limitation "One seed (7). Seed luck is measured at ± 0.0077 bits on the transformer family; nothing here measures it for this architecture." \
  --limitation "Selected step 2,400 equals the budget ceiling with validation still falling: this is a floor, not a converged loss." \
  --limitation "Trained on 19th-century and classical public-domain prose. No instruction data, no questions, no answers, and no grounded question is answered." \
  --limitation "No semantic quality is measured anywhere in this release; the generation statistics are surface statistics only." \
  --licence-model "Apache-2.0" --licence-data "Public domain in the United States (Project Gutenberg)" \
  --redistribution "Weights may be redistributed with model.json attached; the corpus is redistributed from its own source, not from this repository." \
  --reproduce "chakaso experiment model --work-dir data/experiments/E2608/decoder"
chakaso model verify dist/chakaso-base           # re-hash the artifact against its own contract
chakaso model inspect dist/chakaso-base --json   # what the weights, the runtime and the contract state
chakaso model generate dist/chakaso-base --prompt "The whale" --max-new-tokens 24
chakaso model stream dist/chakaso-base --prompt "The whale" --temperature 0.7 --seed 7
chakaso model score dist/chakaso-base --text-file docs/training.md
chakaso model quality dist/chakaso-base --snapshot <corpus snapshot> --split-seed 7   # repetition vs human text, gated
chakaso model serve dist/chakaso-base --port 8765   # loopback by default, no auth, Ctrl-C to stop
```

Those five strings are the ones `chakaso-base@0.1.0` carries, copied from
[`MODEL_CARD.md`](MODEL_CARD.md) rather than written for this page; a release whose limitations read like
marketing is not this contract's kind of release.

The same operations are Python calls, because the terminal is a thin handler over them and a trained
checkpoint must be usable without one:

```python
from pathlib import Path

from chakaso.models import export_release, load_model
from chakaso.serve.server import serve

work = Path("data/experiments/E2608/decoder")
release = export_release(  # copies the checkpoint and report, then writes model.json beside them
    work,
    Path("dist/chakaso-base"),
    model_id="chakaso-base",
    model_version="0.1.0",
    summary=(
        "A 3,413,760-parameter rotary decoder trained from random initialization on 14,490,005 characters "
        "of public-domain English prose, reaching 1.6865 bits per character held out."
    ),
    capabilities=("tokenize", "generate", "stream", "score"),
    known_limitations=(  # required, and non-empty: the card's five, verbatim
        "Generations repeat phrases: mean bigram repetition 0.5538 over five greedy continuations; "
        "every sample reached its token bound. No sample is usable text.",
        "One seed (7). Seed luck is measured at ± 0.0077 bits on the transformer family; nothing here "
        "measures it for this architecture.",
        "Selected step 2,400 equals the budget ceiling with validation still falling: this is a floor, "
        "not a converged loss.",
        "Trained on 19th-century and classical public-domain prose. No instruction data, no questions, "
        "no answers, and no grounded question is answered.",
        "No semantic quality is measured anywhere in this release; the generation statistics are surface "
        "statistics only.",
    ),
    licence_model="Apache-2.0",
    licence_training_data="Public domain in the United States (Project Gutenberg)",
    redistribution=(
        "Weights may be redistributed with model.json attached; the corpus is redistributed from its own "
        "source, not from this repository."
    ),
    created_utc="2026-10-22T00:00:00Z",
    reproduce="chakaso experiment model --work-dir data/experiments/E2608/decoder",
)

model = load_model("dist/chakaso-base")  # the directory: verified against the contract, then loaded
service = serve(str(work / "E2608-decoder.pt")).start()
print(service.base_url, service.loopback_only)  # http://127.0.0.1:<port>, True by default
service.stop()
```

`serve` speaks `/health`, `/v1/models`, `/v1/completions` and `/v1/chat/completions`, streaming over
server-sent events, and it refuses the development double and any request field it cannot honour. It is not
authenticated, so binding beyond loopback is a decision an operator states.

## Train one

```bash
# the whole E2605 comparison: manifest -> split -> tokenizer -> both models -> held-out metrics
python -m chakaso experiment model \
  --work-dir data/experiments/E2605 \
  --write-report experiments/E2605/results.json
```

That command is a thin handler over
[`chakaso.training.subword_experiment`](src/chakaso/training/subword_experiment.py); the same call is
available from Python, and the training path has no dependency on the terminal. The smallest version of the
from-scratch loop (character model, no `torch`) is documented in
[`docs/training.md`](docs/training.md), and
[`experiments/E2605/record.md`](experiments/E2605/record.md) shows what a measured run records.

## Evaluate a model

```bash
python -m pytest                                   # default matrix: offline, CPU-only, no extra needed
python -m pytest                                   # with .[dev,train] / .[dev,train,nn] the gated modules run
python -m chakaso experiment model --work-dir data/scratch --max-steps 12 --eval-every 4 --json
```

Model metrics and harness metrics are different things and are kept apart: bits per character, perplexity
and token accuracy describe a model; the retrieval and correction benchmarks, the reference oracle in the
grounded evaluation, and the evidence-sufficiency policy describe the plumbing. `chakaso benchmark`
numbers are never evidence of language-model competence, and the reports say so.

## Why this direction

A general model can produce a fluent answer while telling you nothing about whether it is grounded, how
current it is, what was inferred versus read, or what would change its answer. Chakaso treats those as
architectural properties rather than prompt instructions — but they can only be *measured* against a model
whose parameters, data and training procedure are known. That is why the model comes first:

- **Language modelling and next-token prediction** are the core measurements
  ([`docs/training.md`](docs/training.md), [`docs/evaluation.md`](docs/evaluation.md)).
- **Tokenization and vocabulary design** are research variables the project trains and versions itself,
  not a borrowed component ([`docs/decisions/ADR-0035`](docs/decisions/ADR-0035-transformer-checkpoint-format.md)).
- **Data provenance and reproducibility** are enforced by content digests, pinned manifests and experiment
  records written before the run.
- **Checkpoints carry their own identity** — architecture, tokenizer, corpus fingerprint, seed, training
  configuration — and refuse to load otherwise.
- **Efficient experimentation** under real constraints: this is CPU-only work on one machine, sized so a
  run finishes inside a session and the numbers stay checkable.
- **Evidence-grounded behaviour** is the research direction the application layer exists to test, once a
  model can be pointed at it.

## Optional interfaces, and what they are for

The CLI (`python -m chakaso …`) is a secondary developer utility: 13 commands — `info`, `config`, `chat`,
`retrieve`, `benchmark`, `experiment`, `corpus`, `model`, `storage`, `capabilities`, `answers`, `sources`,
`evidence` — for inspecting configuration, retrieving over supplied documents, holding a conversation,
running benchmarks, recovering a pinned corpus, and driving the whole release lifecycle of a trained
checkpoint. Its handlers are thin and call the same Python services the library exposes; no core model
behaviour lives in a terminal handler, and a hygiene test fails if a library module ever imports one.
`chakaso chat` is honest about its own engine — it prints what the model adapter is before it prints
anything else.

There is deliberately no web frontend, hosted API or desktop application: an interface in front of a small
model advertises a capability that does not exist, and
[ADR-0001](docs/decisions/ADR-0001-local-first-runtime.md) already rules out the hosted dependency that
would make one convenient.

```bash
python -m chakaso capabilities     # what the current configuration actually selects, offline and read-only
python -m chakaso retrieve --query "how does chunking work" --file docs/retrieval.md
python -m chakaso chat --file docs/retrieval.md
```

## Repository layout

```text
src/chakaso/        the Python package
  training/         the model path: corpus manifests, catalogues and snapshots, external-source
                    ingestion, the character and learned subword tokenizers, document splits and
                    packing, the character model (numpy), the Transformer and the rotary decoder
                    with their training engine and checkpoint schemas (torch), held-out metrics,
                    the generation-evaluation suite, the seed study, and the E2605–E2608 drivers
  models/           the language-model boundary, the adapter registry and `load_model`,
                    the development double, the character and subword adapters, bounded decoding,
                    versioned prompt templates, and the release contract
  serve/            the optional standard-library HTTP adapter over one loaded model
  retrieval/ evidence/ claims/ citation/ grounding/ correction/ answers/
                    the optional application layer (local-first, offline, tested)
  evaluation/ benchmark/ experiments/   measurement instruments and their fixtures
  config/ core/ runtime/ conversation/ planning/ app.py cli.py
tests/              unit, contract and repository-hygiene tests
configs/            versioned configuration, TOML — including the corpus catalogues
experiments/        committed experiment records — the project's measured results
data/               prepared corpora, tokenizers, checkpoints and releases: git-ignored, cited by digest
docs/               specifications, decisions, research log, data cards, agent context
```

Experiment records live at `experiments/<experiment-id>/record.md`
([format](docs/internal/experiments/README.md)); generated artifacts never enter version control
([artifact policy](docs/internal/engineering/artifact-policy.md)). The `src/` layout is deliberate: it
makes an uninstalled import fail rather than silently test the working tree
([ADR-0005](docs/decisions/ADR-0005-src-layout.md)).

## Documentation

| Document | What it covers |
| --- | --- |
| [`MODEL_CARD.md`](MODEL_CARD.md) | The reference artifact: what it is, what it measured, what it cannot do, and which release-gate level it earned |
| [`docs/training.md`](docs/training.md) | Architecture stages, tokenizer and data pipeline, sizing policy, statuses |
| [`docs/evaluation.md`](docs/evaluation.md) | Model metrics, harness metrics, benchmarks, regression gates |
| [`docs/getting-started.md`](docs/getting-started.md) | Install (including the optional extras), running tests, running experiments |
| [`docs/architecture.md`](docs/architecture.md) | Components, boundaries and the normative dependency direction |
| [`docs/research/open-model-landscape.md`](docs/research/open-model-landscape.md) | The sourced DeepSeek/Qwen/OLMo/SmolLM/TinyLlama comparison and what transfers to a CPU-scale model |
| [`docs/research/chakaso-next-training-design.md`](docs/research/chakaso-next-training-design.md) | The recommended next model and the data-before-architecture argument |
| [`docs/research/chakaso-data-curriculum.md`](docs/research/chakaso-data-curriculum.md) | The staged licensed-corpus plan, mixture weights and contamination controls |
| [`docs/research/chakaso-capability-benchmarks.md`](docs/research/chakaso-capability-benchmarks.md) | What each capability claim would require, and what exists today |
| [`docs/research/chakaso-compute-and-scaling-plan.md`](docs/research/chakaso-compute-and-scaling-plan.md) | The measured throughput and the memory/time arithmetic for 3.4M–100M parameters |
| [`docs/research/chakaso-integration-contract.md`](docs/research/chakaso-integration-contract.md) | The stable consumer contract and its known gaps |
| [`docs/research/next-model-experiment.md`](docs/research/next-model-experiment.md) | The measured environment and candidate comparison behind the current model step |
| [`docs/research/model-feasibility.md`](docs/research/model-feasibility.md) | The earlier feasibility assessment (a runnable local model vs from scratch) |
| [`docs/internal/engineering/model-centred-audit.md`](docs/internal/engineering/model-centred-audit.md) | How the repository drifted and what the realignment changed |
| [`docs/retrieval.md`](docs/retrieval.md), [`docs/correction.md`](docs/correction.md), [`docs/transparency.md`](docs/transparency.md) | The optional application layer |
| [`docs/roadmap.md`](docs/roadmap.md) | Phases and exit criteria |
| [`docs/decisions/`](docs/decisions/) | Why the architecture is shaped this way (ADR-0001 … ADR-0041) |
| [`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md) | The current state, precisely |

## Contributing

Corrections to claims, measurements that contradict something here, and negative results are all welcome —
the project's value depends on them. See [`CONTRIBUTING.md`](CONTRIBUTING.md); coding agents start at
[`AGENTS.md`](AGENTS.md).

## License

Apache License 2.0 ([`LICENSE`](LICENSE)). Model weights, tokenizers and datasets are not published yet
and are not covered by this license; they will carry their own terms when they exist
([ADR-0004](docs/decisions/ADR-0004-apache-2.0-license.md)).
