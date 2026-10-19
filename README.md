# Chakaso

Chakaso is an open research project exploring how to build language models and AI
systems that are more transparent about what they know, what they retrieved, what
they inferred, and when they should change their answer.

Two things are being built in this repository, because neither is useful alone:

1. **A local-first conversational system** that answers from retrieved public
   evidence, keeps source metadata separate from generated text, resolves
   follow-up questions against conversation state, and can revise an earlier
   answer when new evidence contradicts it.
2. **A training pipeline** that will eventually produce Chakaso's own model
   weights, so the system's model layer can be replaced by a model this project
   trained rather than one it borrowed.

The second is the point of the first. A conversational system is only worth
building if its behaviour can be measured, and its behaviour can only be measured
against a model whose parameters, data and training procedure are known.

## Status

**There is a conversation shell, and no language model.** `chakaso chat` holds a
conversation end to end — configuration, context projection, the model boundary,
evidence and citation recording — but the only model implementation is a
deterministic development double that returns fixed text. It answers nothing, and it
says so when you start it.

There is a **local retrieval pipeline**: normalization, section-bounded chunking,
ingestion of explicitly supplied text and Markdown, an in-memory corpus, and a deterministic
lexical (BM25) retriever that ranks into an evidence pack, reachable through `chakaso retrieve`.
Behind the same retriever boundary a **dense** retriever ranks over a deterministic, **non-semantic**
fixture embedding, and a **hybrid** retriever fuses the two by reciprocal-rank fusion; the shapes are
real and tested, but no genuine embedding model exists, so dense and hybrid re-rank token overlap,
not meaning. A retrieval-aware runtime composes a deterministic query planner with these retrievers
so `chakaso chat` grounds each turn and records every answer. An opt-in, bounded web fetcher and a
narrow HTML reader turn a fetched page into the same evidence records, but fetching is off every
default path. Below retrieval the system has a measurement and correction spine: claim / citation /
grounding evaluation and an answer-evaluation combiner, a correction decision rule with an
application-level reassessment over a stored answer, and an append-only answer store that is
in-memory by default and durably **SQLite-backed when configured** — a history that survives a
process restart. There is also a fixture-only semantic grounding boundary, a correction development
benchmark, and reproducible experiment records with a deterministic regression baseline.
There is still **no language model in the product**, no subword tokenizer, and no automatic correction
loop. The runtime's only model is the development double. But the project has now **trained and measured
a model of its own**: experiment [E2601](experiments/E2601/record.md) trains a 984-parameter character
model from random initialization (`chakaso.training`, gated by
[ADR-0030](docs/decisions/ADR-0030-minimal-optional-training-dependency.md)) and shows its held-out
next-character loss falls below both uniform and unigram baselines — the project's first genuinely
measured result. It is a learning-loop demonstration on a tiny synthetic corpus, not a useful language
model, and it is not wired into the runtime; the retrieval and correction numbers remain development
metrics over synthetic fixtures, not model-quality or real-workload results.

The status vocabulary used across this documentation is:

| Label | Meaning |
| --- | --- |
| **Implemented** | Code exists in this repository and is covered by tests that pass. |
| **Experimental** | Code exists and runs, but the approach is not settled and may change incompletely. |
| **Planned** | Designed in enough detail to build; not built. |
| **Research** | An open question. There is no committed design. |
| **Unknown** | Not yet determined. |

| Area | Status |
| --- | --- |
| Repository conventions, licensing, decision records | Implemented |
| Public and internal documentation | Implemented |
| Tests and CI | Implemented (Python 3.11–3.13, no secrets, no network dependency) |
| Python package, configuration, CLI | Implemented |
| Language-model interface and model registry | Implemented (interface only; no trained model) |
| Source, evidence and citation records | Implemented |
| Conversation state | Implemented |
| Conversation manager, and a CLI conversation shell | Implemented (`chakaso chat` conducts turns through the retrieval-aware runtime; replies still come from a development double) |
| Normalization, chunking, local ingestion and corpus | Implemented |
| Lexical retrieval (BM25), orchestration into an evidence pack, and `chakaso retrieve` | Implemented |
| Deterministic query planning, the retrieval-aware runtime, and the `chakaso.app` composition root | Implemented (ADR-0022/ADR-0025/ADR-0026; rules, not a model) |
| Web fetch (opt-in, bounded) and narrow HTML ingestion | Implemented (off every default path; not a browser) |
| Evaluation metric functions | Implemented (recall@k, precision@k, MRR, duplicate + citation counts) |
| Claim / citation / grounding evaluation and the answer-evaluation combiner | Implemented (structural; status is supplied-evidence support, never truth. Semantic grounding is a boundary with a fixture-only judge) |
| Correction decision rule, reassessment, append-only records and a correction development benchmark | Implemented (foundation: scores the rule over synthetic cases; a deterministic, identifier-triggered reassessment is composed into `chakaso chat` — an explicit reference, never an autonomous guess, ADR-0029; no revised prose, no automatic loop) |
| Persistent answers and application-level reassessment | Implemented (append-only store, ADR-0017/0018; durable SQLite backend when configured survives a restart, ADR-0021; reassessment over a stored, identifier-addressed answer) |
| Reproducible experiments and a regression baseline | Implemented (development, ADR-0020); no model-quality result |
| Dense retrieval (embedding boundary + exact index) and hybrid reciprocal-rank fusion | Experimental (ADR-0023/ADR-0024) — the only embedding is a deterministic, non-semantic fixture double, so these re-rank token overlap, not meaning |
| A real semantic embedding model, an approximate (FAISS-like) vector index, and reranking | Planned |
| Character tokenizer, leak-free split, tiny causal model, Adam, checkpoint, training loop, baselines, held-out evaluation (`chakaso.training`) | Experimental (ADR-0030; runs under the optional `[train]` extra, out of core CI per K-009; not wired into the runtime) |
| Experiment E2601: a from-scratch character model trained from random init and measured against baselines | Implemented (Experimental; measured held-out loss beats uniform + unigram; toy scale, single seed, overfits after step 50) |
| Subword/BPE tokenizer, a Transformer, a real corpus, mini-batching and any scaling | Planned |
| A trained Chakaso model wired behind the `LanguageModel` boundary (an inference adapter) | Does not exist yet — the next milestone (P25) |
| Committed or released model weights | Does not exist (the E2601 checkpoint is reproducible from the seed and recorded by fingerprint, not committed) |
| Real evaluation harness and model benchmarks | Planned (retrieval and correction development benchmarks exist; no real-workload or model-quality measurement) |

Nothing in this list is aspirational language. If a row says Planned, there is no
code for it. "Implemented" means the code exists and is covered by tests; in several
rows it means only the data model exists, and the row says so. See
[`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md) for the precise current
state, including what is deliberately missing.

## Why this exists

A general language model can produce a fluent answer while telling you nothing
about whether that answer is grounded, how current it is, what it inferred versus
what it read, or what would change its mind. Retrieval helps, but retrieval does
not by itself produce provenance, calibrated uncertainty, or a disciplined way to
be corrected.

Chakaso treats those as architectural properties rather than prompt instructions:

- **Source identity belongs to the retrieval layer.** The model may reference
  evidence identifiers; it may not invent one. A reference the model was not
  given is an error to be recorded, not a URL to be rendered.
  ([ADR-0003](docs/decisions/ADR-0003-evidence-identifier-ownership.md))
- **The application does not depend on one model.** It depends on a narrow
  interface, with a single place where configuration selects an implementation.
  ([ADR-0002](docs/decisions/ADR-0002-language-model-boundary.md))
- **The runtime is local-first.** No commercial inference or search API is
  required for core functionality, so the pipeline stays reproducible and the
  measurements stay measurements of this project.
  ([ADR-0001](docs/decisions/ADR-0001-local-first-runtime.md))
- **Correction beats consistency.** An answer contradicted by better evidence is
  revised, and the revision is recorded. Defending an earlier answer because it
  was earlier is a failure mode with a metric.

## What Chakaso deliberately does not do

- It does not fabricate citations, source titles or URLs.
- It does not present a numeric confidence score unless that score has been
  calibrated and measured. Until then, uncertainty is expressed qualitatively
  ("the sources conflict", "the available evidence does not answer this").
- It does not expose hidden chain-of-thought as a transparency feature. It
  exposes evidence, assumptions, provenance, uncertainty and corrections.
- It does not claim a model exists. None has been trained yet.
- It does not depend on a commercial model or search provider for core
  functionality.

## Getting started

There is one thing to run, and it is honest about itself:

```bash
git clone https://github.com/bhogesararam23/chakaso.git
cd chakaso
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

python -m pip install -e ".[dev]"
python -m pytest
python -m chakaso --version

# see what the current configuration actually selects (offline, read-only)
python -m chakaso capabilities

# run the local retrieval pipeline over a document
python -m chakaso retrieve --query "how does chunking work" --file docs/retrieval.md

# hold a conversation, grounded over a document
python -m chakaso chat --file docs/retrieval.md
```

`capabilities` reports what is real without running anything:

```
model: deterministic (Deterministic double (not a language model))
development double: yes — returns fixed text, not answers
retrieval: on (strategy=lexical, top_k=5)
storage: memory
semantic evaluation: no real evaluator — only a caller-supplied fixture judge exists; grounding is structural
```

A grounded conversation turn is honest about its own engine:

```
$ python -m chakaso chat --message "What does the specification say?" --file docs/retrieval.md
engine: deterministic (Deterministic double (not a language model))
This is a development double, not a language model. It returns fixed text that
exercises the conversation plumbing, and it does not answer anything.
retrieval: lexical over 1 source(s) and 38 chunk(s). This grounds and cites structurally
only — no semantic judgement, and dense/hybrid run over a non-semantic fixture embedding.
answers are recorded in memory for this process only and are not saved to disk.
Deterministic double (not a language model). It received 1 message(s); the last
was 32 characters of 'user' text.
```

The notice and the per-turn grounding report go to standard error and the reply to
standard output, so the last line above is what a script would capture. Read it as
evidence that the conversation-and-grounding boundary works end to end, not as an
answer to the question.

Full instructions, including how to work without installing anything and how to test
the declared Python floor, are in [`docs/getting-started.md`](docs/getting-started.md).

## Repository layout

```text
chakaso/
├── src/chakaso/        # the Python package
├── tests/              # unit, contract and repository-hygiene tests
├── docs/               # documentation
│   ├── decisions/      # architecture decision records
│   ├── research/       # research log, hypotheses, open questions
│   ├── agent/          # durable context for coding agents
│   └── internal/       # engineering notes, artifact policy, experiment record format
├── configs/            # versioned configuration
├── CONTRIBUTING.md
└── pyproject.toml
```

Experiment records live at `experiments/<experiment-id>/record.md`, with any configuration alongside
them and never large artifacts. The first, [`experiments/E2601/record.md`](experiments/E2601/record.md),
is committed — the project's first measured training result. The record format is defined in
[`docs/internal/experiments/README.md`](docs/internal/experiments/README.md).

The package uses a `src/` layout so that tests exercise the installed package
rather than whatever happens to be in the working directory
([ADR-0005](docs/decisions/ADR-0005-src-layout.md)).

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/getting-started.md`](docs/getting-started.md) | Local setup, running tests, checking the install |
| [`docs/architecture.md`](docs/architecture.md) | Components, boundaries and data flow |
| [`docs/transparency.md`](docs/transparency.md) | What Chakaso exposes and what it withholds |
| [`docs/retrieval.md`](docs/retrieval.md) | Retrieval, evidence and source records |
| [`docs/correction.md`](docs/correction.md) | Reassessment when evidence contradicts an answer |
| [`docs/training.md`](docs/training.md) | Tokenizer, data pipeline and model development plan |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics, benchmarks and regression gates |
| [`docs/roadmap.md`](docs/roadmap.md) | Milestones from foundation to a scratch-trained model |
| [`docs/research/`](docs/research/) | Research log, hypotheses and open questions |
| [`docs/decisions/`](docs/decisions/) | Why the architecture is shaped the way it is |
| [`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md) | The current state, precisely |

## Contributing

Corrections to claims, measurements that contradict a statement in this
documentation, and negative results are all welcome — the project's value depends
on them. See [`CONTRIBUTING.md`](CONTRIBUTING.md). Coding agents should start at
[`AGENTS.md`](AGENTS.md).

## License

Apache License 2.0 ([`LICENSE`](LICENSE)). Model weights, tokenizers and datasets
are not published yet and are not covered by this license; they will carry their
own terms when they exist ([ADR-0004](docs/decisions/ADR-0004-apache-2.0-license.md)).
