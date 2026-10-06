# Literature

Sources that informed Chakaso's initial design, and what each one is used to
support. These are background reading. **No paper listed here has been
reproduced, extended or benchmarked by this project.**

A reference appears here only because a specific design decision rests on it. If a
decision changes, the reference supporting it should be re-examined rather than
kept for decoration.

Every URL below was checked and resolves. Titles are given exactly as published.

---

## Retrieval-augmented generation

**Lewis et al. (2020), "Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks."** NeurIPS 2020. <https://arxiv.org/abs/2005.11401>

Supports: separating parametric model memory from non-parametric external memory,
and treating provenance and knowledge updating as problems that a purely
parametric model does not solve. This is the basis for Chakaso tracking observed
evidence as a distinct category from model knowledge
([`../transparency.md`](../transparency.md)).

What it does not support: that retrieval improves factuality by itself. The paper
is about a specific retrieval-augmented architecture and its results, not a
general claim that adding retrieval to any model makes it reliable.

## Transformer architecture

**Vaswani et al. (2017), "Attention Is All You Need."**
<https://arxiv.org/abs/1706.03762>

Supports: the decoder-only Transformer with causal self-attention as the starting
architecture for the model work ([`../training.md`](../training.md)), and the
choice to keep the conceptual architecture simple while using optimized attention
primitives underneath.

**PyTorch, "Implementing High-Performance Transformers with Scaled Dot Product
Attention (SDPA)."**
<https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>

Supports: the claim that an optimized attention implementation can be adopted
without changing the conceptual architecture.

## Tokenization

**Hugging Face, Tokenizers documentation.** <https://huggingface.co/docs/tokenizers/main/index>

**Hugging Face, "Training from memory."**
<https://huggingface.co/docs/tokenizers/main/training_from_memory>

Supports: training a tokenizer from scratch over an iterator, which is what makes
a corpus-versioned tokenizer practical, and what makes it possible to freeze
tokenizer versions per model family ([`../training.md`](../training.md)).

## Data streaming

**Hugging Face, Datasets — "Stream."** <https://huggingface.co/docs/datasets/main/stream>

Supports: processing data without downloading an entire corpus first, which
matters under disk and compute constraints.

## Model and tokenizer extension

**Hugging Face, Transformers — "Customizing models."**
<https://huggingface.co/docs/transformers/en/custom_models>

**Hugging Face, Transformers — "Customizing tokenizers."**
<https://huggingface.co/docs/transformers/custom_tokenizers>

Supports: the assumption that both the model and the tokenizer can be defined by
this project rather than adopted wholesale from a pretrained checkpoint, which is
what a scratch-trained model requires.

## Semantic retrieval

**Sentence Transformers, "Semantic Search."**
<https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html>

**FAISS.** <https://github.com/facebookresearch/faiss>

Supports: dense local embeddings plus a local similarity index as the first
retrieval stack, and the claim that this stack can run locally without a hosted
service.

Not yet a decision: no embedding model and no index have been chosen. These
references establish that the approach is available, not that it is selected.

## Evaluation

**EleutherAI, lm-evaluation-harness.**
<https://github.com/EleutherAI/lm-evaluation-harness>

Supports: using an existing harness for standard language-model benchmarks rather
than building one, and the finding that a general harness does not cover
retrieval, grounding, citation or correction — which is why Chakaso needs its own
suite ([`../evaluation.md`](../evaluation.md)).

## Preference optimization

**Rafailov et al. (2023), "Direct Preference Optimization: Your Language Model is
Secretly a Reward Model."** <https://arxiv.org/abs/2305.18290>

Supports: DPO as a simpler preference-optimization option than a full PPO-based
RLHF pipeline, which is why it is named as a possible later stage rather than a
required one ([`../training.md`](../training.md)).

## Hallucination and uncertainty

**Huang et al. (2023), "A Survey on Hallucination in Large Language Models:
Principles, Taxonomy, Challenges, and Open Questions."**
<https://arxiv.org/abs/2311.05232>

Supports: treating hallucination as a reliability problem that retrieval mitigates
rather than eliminates, and therefore evaluating evidence support and abstention
as behaviours in their own right rather than assuming retrieval solves factuality.

This is the reference that most directly motivates a hypothesis rather than a
mechanism: see [H4](hypotheses.md). A survey is background, not evidence about
this system.

## Engineering

**Python, `venv`.** <https://docs.python.org/3/library/venv.html>

**pytest.** <https://docs.pytest.org/en/stable/>

Supports: the deliberately unambitious tooling choices — an isolated virtual
environment and a test framework — so that project complexity stays concentrated
on retrieval and model research rather than on build machinery.

---

## How to add a reference

State the specific decision it supports and, where relevant, what it does *not*
support. A reference that supports nothing in particular will be removed. Do not
add a citation that has not been read.
