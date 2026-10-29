# Model card — Chakaso Base 0.1.0

**Model:** `chakaso-base@0.1.0` · **Family:** `decoder` · **Schema:** `chakaso-decoder-checkpoint/1`
**Release contract:** `chakaso-model-release/1`, checksum `9aebe601dc1f12c5025ad012b88a6d0c54c376e982b9c54526924a4dd7314089`
**Created:** 2026-10-22 · **Status: Experimental** (a measured research artifact, not a deployed model)

## The one thing to know before using it

This model predicts the next subword of 19th-century and classical public-domain English prose, and it does
that better than anything this repository has measured before. **It does not answer questions, follow
instructions, or produce text anyone should publish.** Its own contract carries five limitations, and the
release gate awarded on this evidence is **Level 2** — see [Release gate](#release-gate) below.

## Summary

A 3,413,760-parameter rotary decoder trained from random initialization on 14,490,005 characters of English
prose from 24 Project Gutenberg texts, pinned by content digest. It reaches **1.6865 bits per character** on
held-out documents — below the unigram baseline (3.2263), below the uniform baseline (5.5661), and below the
1,037,056-parameter transformer architecture this project measured previously (1.9043) trained on the identical
token budget. Its generations do not contain the exact repeating cycles every earlier checkpoint in this
repository produced, and they still repeat phrases badly enough that mean bigram repetition is 0.5538.

## Release gate

Levels are defined in [ADR-0039](docs/decisions/ADR-0039-three-level-model-release-gate.md) and assessed here
against evidence rather than against the fact that the earlier levels passed.

| Level | What it claims | Verdict | Evidence |
| --- | --- | --- | --- |
| **1 — Artifact contract** | A named, versioned artifact that states its own provenance and cannot be silently swapped | **Met** | `chakaso-model-release/1` directory; `chakaso model verify` re-hashes `E2608-decoder.pt` against `model.json` (SHA-256 `9e832fadf277e848…`); loading a directory verifies before a weight is read; the builder refuses a report whose family contradicts its artifact schema |
| **2 — Reproducible integration** | Loadable, runnable and reproducible from the record, end to end, without a pretrained dependency | **Met** | [E2608's record](experiments/E2608/record.md) states the pinned corpus (`e9fc2e92f59d88ea`), the split, the tokenizer (`b597eb34f13d9058`), the budget and the exact commands; a clean vertical slice runs train → save → load → generate → stream → cancel → score → release → serve in CI's `model path` tier; weights originate from random initialization, and no provider or third-party model is reachable from any of it |
| **3 — Useful behaviour** | The model does something a caller would want done | **Not met, and not claimed** | Five greedy continuations: 0 tail loops, 0 periodic runs of ≥ 3, mean bigram repetition 0.5538, every sample truncated at its token bound, no sample a reader would keep; no grounded question answered anywhere in the repository (K-001); no semantic judge exists |

Level 3 is withheld on measurement, not modesty. What would earn it is named in
[the roadmap](docs/roadmap.md#handoff-phases).

## Measured facts

| | |
| --- | --- |
| Parameters | 3,413,760 (`decoder` family: rotary positions, RMSNorm, biases-free attention, GELU feed-forward, tied embeddings, depth-scaled residual init) |
| Context length | 256 positions · Vocabulary 4,096 byte-pair subwords, 3,839 merges, minimum frequency 3 · 2.1559 characters per token on test |
| Training data | `gutenberg-english-prose-v2`: 24 texts, 919 documents, 14,490,005 characters, corpus fingerprint `e9fc2e92f59d88ea`, split 643 train / 138 validation / 138 test documents (18,252 / 3,981 / 3,908 blocks at context 256) |
| Budget | seed 7, split seed 7 · 2,400 steps at batch 32 (8,192 positions/step, 19,660,800 positions total) · AdamW lr 6e-4, weight decay 0.01, warmup 200, **cosine decay to 0.1** · early stopping patience 3, min delta 5e-4 |
| Outcome class | `budget_ceiling` — selected step 2,400, which **is** the ceiling, with validation loss still falling at the last evaluation (2.5457 → 2.5436 nats over the final 100 steps). This is a floor, not a converged level |
| Held-out (test) | **1.6865 bits/character**, 2.5203 nats/token, perplexity 12.43, token accuracy 0.5057, over 1,000,448 scored positions (2,156,862 characters) |
| Baselines, same packing | unigram 3.2263 bits/character · uniform 5.5661 |
| Throughput | 4,821 positions/second over gradient time only; 6,024.06 s wall clock (100.4 min) on 14 CPU threads; 13.24 s preparation with the tokenizer reused from cache |
| Environment | Python 3.13.5, torch 2.13.0+cpu, numpy 2.4.0, device cpu, `Windows-11-10.0.26200-SP0`. No GPU was used and none is required by anything in this repository |
| Random initialization | Yes. Weights come from the seeded initialization in `chakaso.training.nn_training`; no pretrained weights, hosted model API or remote inference provider appears anywhere on the path from corpus to served token |

## Degeneration, measured

`chakaso.training.generation_eval`, run over this artifact through `chakaso.models.load_model` — five fixed
continuation prompts at greedy decoding (bounds 32, 48, 48, 48 and 64 new tokens) plus four hostile inputs.
Raw text is retained in `data/experiments/E2608/generation-E2608-decoder.json`.

| Probe | Bigram repetition | Longest repeated block | Periodic run |
| --- | --- | --- | --- |
| `document-heading` | 0.541 | `'Emperor of the'` × 2 | 2 |
| `status-section` | 0.570 | `' of the South Testament'` × 2 | 1 |
| `because-clause` | 0.677 | `'same as the same as '` × 2 | 2 |
| `list-continuation` | 0.584 | `' the world of the'` × 2 | 1 |
| `table-row` | 0.397 | `' of the world'` × 2 | 2 |

Verbatim, unedited: ` very unfortunate to the Error of the Emperor of the\nEmperor of the Soule of the Soule`
and ` of the world.\n\nI am sure you are not in the world of the world`.

Hostile inputs: a single character, non-ASCII text and an oversized prompt all generated; the empty prompt
produced a typed refusal (`ModelError`), which is the boundary's documented contract and is counted as a
refusal, not a crash. **Zero crashes across nine probes.**

### Calibrated against human text (E2610, added 2026-10-29)

The table above reports *character*-bigram ratios, which a reader cannot interpret without a reference:
the same statistic on real held-out prose of the same length is **0.3355**, and the five probes above were
markdown about this repository, which this model never trained on. The calibrated suite
(`chakaso model quality`, [E2610](experiments/E2610/record.md)) draws 24 prompts from held-out *books'* own
text, keeps the true continuation as a human reference, and measures word distinct-n. On 24 prompts:

| Decoding | Word distinct-1 | Distinct-2 | Distinct-3 | Passes the declared gates |
| --- | --- | --- | --- | --- |
| human continuation | 0.853 | 0.969 | 0.982 | (the reference) |
| greedy | 0.505 | 0.688 | 0.794 | No — fails all three diversity gates and the periodic gate |
| temperature 1.0, 3 seeds | 0.890 | 0.996 | 0.999 | Yes, on both prompt sets |

The repetition is therefore a property of **greedy decoding of a distribution that is itself human-diverse**,
not a defect in training data, packing, tokenization, positions or the cache (each checked; see the
[research log](docs/research/research-log.md)). It is *not* evidence the model writes well: settings that pass
every gate produce invented words (`Meshiciety`, `Puhaberlap`) and no argument, and the preregistered claim that
a decoding setting repairs repetition was **inconclusive** (K-020). Level 3 stays **not met**.

The suite's own limit travels with the data: these are *surface statistics* — repetition, length, termination
and refusal behaviour computed from emitted characters. No semantic quality, usefulness or groundedness is
measured here, and none may be inferred from these numbers.

## Known limitations

Stated in the release contract, verbatim:

1. Generations repeat phrases: mean bigram repetition 0.5538 over five greedy continuations; every sample
   reached its token bound. No sample is usable text. (Calibrated: that is a character statistic whose human
   value is 0.3355; the word-level gap is 0.505 against 0.853 distinct-1 for greedy. The sentence quoted here
   is the release contract's own and is not edited; the section above is the correction.)
2. One seed (7). Seed luck is measured at ± 0.0077 bits on the transformer family; nothing here measures it for
   this architecture.
3. Selected step 2,400 equals the budget ceiling with validation still falling: this is a floor, not a
   converged loss.
4. Trained on 19th-century and classical public-domain prose. No instruction data, no questions, no answers,
   and no grounded question is answered.
5. No semantic quality is measured anywhere in this release; the generation statistics are surface statistics
   only.

## How to use it

```python
from chakaso.models import GenerationParams, Message, Role, load_model

model = load_model("data/experiments/E2608/release")  # verified against model.json, then loaded
result = model.generate(
    [Message(role=Role.USER, content="Call me Ishmael.")],
    GenerationParams(max_new_tokens=32, temperature=0.0),
)
print(result.text, result.finish_reason.value)
print(model.score(open("docs/training.md", encoding="utf-8").read()), "nats per token")
```

```bash
chakaso model verify   data/experiments/E2608/release
chakaso model inspect  data/experiments/E2608/release --json
chakaso model generate data/experiments/E2608/release --prompt "Call me Ishmael." --max-new-tokens 32
chakaso model stream   data/experiments/E2608/release --prompt "Call me Ishmael." --temperature 0.7 --seed 7
chakaso model serve    data/experiments/E2608/release --port 8765   # loopback, unauthenticated, Ctrl-C stops
```

Requires the optional extras: `pip install -e ".[train,nn]"` (on Linux, add
`--extra-index-url https://download.pytorch.org/whl/cpu`). Everything above is the same through the library and
the terminal, because the handlers are thin and the model lives in the library.

## Intended use, and not

**Intended:** research and evaluation — a checkpoint whose data, tokenizer, architecture, budget and
degeneration are all known, so a claim about it can be checked; a target for the next experiment; a way to
exercise the release, runtime and serving surfaces end to end.

**Not intended:** answering anything, chat, summarization, code, translation, or any task where the reader
depends on the text being true or coherent. It is a small model trained on old books, and it is presented as
one.

## Provenance and licences

| | |
| --- | --- |
| Model weights | Apache-2.0 |
| Training data | Public domain in the United States, retrieved from Project Gutenberg; each source's own licence statement is recorded per-document in the dataset catalogue and its data card ([gutenberg-english-prose-v2](docs/data/gutenberg-english-prose-v2.md)) |
| Redistribution | Weights may be redistributed with `model.json` attached; the corpus is redistributed from its own source, not from this repository |
| Checkpoint | `E2608-decoder.pt`, 17,906,053 bytes, SHA-256 `9e832fadf277e848e286f5d5270725c6f9f51c04dbeda00d483b8998b4a713ab` |
| Run report | `report.json`, digest recorded in the contract as `training.report_checksum` |
| Experiment record | [experiments/E2608/record.md](experiments/E2608/record.md) — preregistered before either arm ran |
| Reproduction | `chakaso experiment model --work-dir data/experiments/E2608/decoder`, full command line in the record's *Commands* section, reading the pinned snapshot rather than the working tree |
| Bytes in Git | None. Checkpoints and corpora live in `data/`, are git-ignored, and are cited by digest |

## What would change this card

A seed spread on this configuration (E2609, declared in the record), a token budget at which validation loss
actually turns, and a generation measurement that a reader disagrees with on the text rather than on the ratio.
The capacity-matched arm this card previously listed as future work has now run: **E2611** compared a
1,018,240-parameter decoder with the 1,037,056-parameter transformer at the same context, data, split, seed,
batch and token budget, and the decoder reached **1.8077** bits per character against **1.9043** — a 0.0966-bit
bundle effect at ~1M parameters, both arms still at the 2,400-step ceiling. That answers E2608's confound at a
smaller scale; it does not change this artifact's numbers or award Level 3. Until the spread and the
behaviour measurement exist, this card's Level 3 row stays "not met", and the number in its headline stays the
interesting part.
