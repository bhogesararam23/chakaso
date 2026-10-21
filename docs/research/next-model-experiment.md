# Next model experiment: which architecture, on which data, under which measured constraints?

**Type:** Research — candidate comparison for the first model-centred experiment (P31/E2605). Every figure
below is either read from this checkout, produced by a command quoted here, or labelled an estimate or an
assumption. Nothing here is a measured model result; the measurement is
[E2605](../../experiments/E2605/record.md), now run, and where this note estimated it says so beside the
measured number.

**Date:** 2026-10-21. **Follows:** [ADR-0033](../decisions/ADR-0033-model-centred-product-contract.md)
(product contract), [`../internal/engineering/model-centred-audit.md`](../internal/engineering/model-centred-audit.md)
(diagnosis), [ADR-0030](../decisions/ADR-0030-minimal-optional-training-dependency.md) (the dependency
decision this note helps re-open).

## Question

ADR-0033 makes the model the product. The next experiment must therefore be chosen from the *measured*
state of the machine, the dependencies, the data that is actually usable, and the model code that already
exists — not from appetite for a bigger number. Which candidate is the strongest experiment that can be
completed, honestly measured and reproduced here?

Candidates compared:

- **C1 — continue the character model.** Keep `CharCausalMLP` and grow its corpus/context.
- **C2 — subword tokenizer + small causal Transformer, trained from random initialization.**
- **C3 — the same Transformer in hand-written NumPy** (no new dependency; manual backprop).
- **C4 — a pretrained external model** as a reference baseline (Option A of
  [`model-feasibility.md`](model-feasibility.md)), or as the product itself.

## Observations (measured 2026-10-21)

### Machine

| Observation | Value | Command |
| --- | --- | --- |
| Logical CPUs | 18 | `os.cpu_count()` |
| Physical RAM total / free at inspection | 16.57 GB / 3.70 GB | `GlobalMemoryStatusEx` |
| CUDA device | none (`torch.cuda.is_available()` → `False`) | executed |
| Free disk on the project volume | 168 GB | `shutil.disk_usage` |
| Interpreter | CPython 3.13.5 (the newest of the CI range; the floor 3.11 is also installed) | `python --version` |

P22's [`model-feasibility.md`](model-feasibility.md) observed ~1.4 GB free RAM and **no ML library
installed**. Both readings were true then; both are false now. This note supersedes them *as a current
observation* and does not alter that note's record of what was seen on 2026-10-15.

### Numerical stack already present

| Observation | Value | Command |
| --- | --- | --- |
| `torch` importable on the interpreter | 2.13.0+cpu, 14 intra-op threads | `import torch` |
| `numpy` importable | 2.4.0 | `import numpy` |
| Cold `import numpy` in a fresh process | 0.25 s | executed |
| Cold `import torch` in a fresh process | 2.96 s | executed |
| `import chakaso.training` | 0.15 s — it does not pull either library | executed |
| PyPI `torch` 2.14.1 CPU wheel, `cp311-win_amd64` | 124.1 MB | <https://pypi.org/pypi/torch/json>, read 2026-10-21 |
| PyPI `torch` 2.14.1 CPU wheel, `cp311-manylinux_2_28_x86_64` | 554.6 MB | same source |
| PyPI `torch` `Requires-Python` | `>=3.10`, with `cp311`/`cp312`/`cp313` wheels present | same source |
| PyPI `numpy` 2.5.4 `Requires-Python` | `>=3.12` — no `cp311` wheel, which is why ADR-0030 caps `[train]` at `<2.5` | <https://pypi.org/pypi/numpy/json> |

**Not measured:** the peak resident set of a training process. The Windows process-memory APIs
(`K32GetProcessMemoryInfo`, `GetProcessWorkingSetSize`) and `tasklist` PID filtering all returned no data
from the shell this session runs in, so no RSS number is claimed here or in ADR-0034. The free-RAM figure
above and the fact that the recorded run completes are what is available; a peak-memory measurement
requires a shell where those APIs answer.

### Compute cost, executed on CPU

A Transformer step measured directly (batch 32 × context 128 = 4,096 tokens, `AdamW` step, forward +
backward + update, best of the sampled iterations):

| Vocab | d_model | Layers | Heads | FFN | Parameters | Step (s) | Forward-only (s) | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 128 | 3 | 4 | 384 | 774,656 | 0.444 | 0.106 | 9,230 |
| 1024 | 192 | 4 | 6 | 576 | 1,901,568 | 0.851 | 0.243 | 4,814 |
| 1024 | 256 | 4 | 8 | 768 | 3,190,784 | 1.252 | 0.407 | 3,271 |
| 4096 | 256 | 4 | 8 | 1024 | 5,288,960 | 1.642 | 0.574 | 2,495 |

Arithmetic on those measured numbers, for the candidate E2605 configuration (row 1) and the corpus below:
~250,000 packed tokens is ~61 steps per epoch, so 40 epochs ≈ 2,450 steps ≈ **18 minutes** on this CPU.
A 3.19 M-parameter model over the same tokens costs ~51 minutes for the same 40 epochs. Both are honest
single-session budgets; neither is a claim that the resulting model is useful.

Same-shape, same-dtype array multiplication, for the C3 comparison: 20 × (8192×256 @ 256×256) took
**0.159 s in NumPy float32** and **0.137 s in torch float32** — a factor of 1.16. Raw matmul throughput is
therefore *not* an argument for a framework on this machine; the arguments are in the next two rows.

### The existing loop's scaling limit, measured

`CharCausalMLP` trains full-batch (ADR-0030 chose that deliberately for a 344-token corpus). Timed on
synthetic batches at vocabulary 200:

| Batch (examples) | Step (s) | Projected step at 650 k examples |
| --- | --- | --- |
| 20,000 | 0.230 | ~7.5 s |
| 100,000 | 1.868 | ~12.1 s |

The loop materializes a `(batch × vocab)` probability array every step: at 650 k positions that is
**1.04 GB float64** for the probability matrix alone, before gradients. So on a corpus of the size below,
the *existing* design needs mini-batching regardless of architecture — and mini-batching plus attention
puts the hand-written-gradient route (C3) in the position ADR-0030 predicted:

- Verifying **every** scalar by central differences costs 2 forward passes per scalar. Measured on the
  current model: 984 parameters in ~1.13 s. Extrapolated to the 774,656-parameter candidate at the
  measured torch forward cost of 0.106 s: **~45 hours** to verify all scalars, ~42 s to verify a 200-scalar
  sample. A sampled check is feasible but is a weaker guarantee than the one the current test suite
  enforces, and writing the backward pass being sampled is itself ~400 lines of delicate new code.

### Data that is actually usable

| Observation | Value | Command |
| --- | --- | --- |
| Tracked text in the repository (`*.md` + `LICENSE`) | 67 files, 724,471 bytes | `git ls-files '*.md' 'LICENSE' \| xargs wc -c` |
| Characters / whitespace words / distinct words | 722,080 / 102,914 / 15,255 | counted from those bytes |
| Non-ASCII characters | 1,204 | counted |
| Licence | Apache-2.0, the project's own ([ADR-0004](../decisions/ADR-0004-apache-2.0-license.md)) | `LICENSE` |
| Existing training corpus | `chakaso-toy-char-v1`: 344 tokens of self-authored text in a module constant | `src/chakaso/training/corpus.py` |
| Third-party corpora committed | none | `git ls-files` |

An *estimate* on the observed character count and typical English subword expansion: ~250,000–320,000
tokens at a 1,024-entry vocabulary. **Measured later by E2605: 369,371 identifiers over the pinned 780,223
characters** (240,418 / 82,376 / 46,577 by side, 2.0829 characters per identifier on the test side). The
estimate assumed 2.26–2.89 characters per identifier; the measured figure is 2.11, so the shortfall is the
compression assumption rather than the 8% the corpus grew by between the two counts. Which is what an
unmeasured estimate is for, and why the record quotes the manifest's own numbers and not this paragraph.

**Open.** No external corpus is committed, and committing one is refused by the artifact policy
([`../internal/engineering/artifact-policy.md`](../internal/engineering/artifact-policy.md)) and by
[ADR-0004](../decisions/ADR-0004-apache-2.0-license.md) until weights/datasets carry their own terms. The
repository text is legally usable and reproducible offline; it is also one author's technical prose about
one project, which bounds what any model trained on it can establish.

## Candidate comparison

| Dimension | C1 character model | C2 subword + Transformer (torch) | C3 subword + Transformer (NumPy) | C4 pretrained reference |
| --- | --- | --- | --- | --- |
| Parameters reachable in one session | ~10³ (measured 984) | 10⁵–10⁶ (measured 0.44 s/step at 775 k) | 10⁵ feasible, but no autograd | n/a |
| Sequence length / attention | none — a 4-character window | causal self-attention over 128 tokens | same, hand-derived | same |
| Tokenizer requirement | deterministic char vocab (exists) | learned BPE, new | learned BPE, new | the external model's own |
| Data requirement | any text | benefits from ≥10⁵ tokens | same | ≥10⁶ for anything resembling capability |
| CPU / GPU | CPU | CPU (measured); GPU unused, no CUDA here | CPU | would want acceleration; none available |
| Reproducibility | seed + config (measured in E2601/E2602) | seed + config, framework-version dependent | seed + config, fewest moving parts | external artifact, licence and version drift |
| Gradient correctness | hand-derived, **every** scalar FD-verified | autograd, cross-checked against hand-computed losses and sampled FD | hand-derived; full FD verification ~45 h (arithmetic above) | not this project's gradients |
| Evaluation metric quality | next-char CE, perplexity, bits/char | bits/char is tokenizer-independent, so comparable across C1/C2 | same | same, but confounded by the pretraining corpus |
| Dependency complexity | none beyond NumPy | +1 optional extra (124 MB wheel, 2.96 s import) | none added | framework + weights + licence terms |
| Maintenance cost | low | contained to `chakaso.training` modules that import torch | high — a private autodiff to keep correct | highest: coupling to an external model |
| Is it Chakaso's own weights? | yes | yes | yes | **no** |

## Decision taken here

**C2, on the repository's own tracked text, with C1 as the same-corpus baseline.** Concretely, E2605
answers one question with numbers: on an identical corpus, identical held-out split and identical budget,
does a subword-tokenized causal Transformer beat (a) the character-level model and (b) uniform/unigram
baselines, measured in **bits per character** so the comparison does not depend on the tokenizer?

Bits per character is the right unit because it is the only one that makes C1 and C2 comparable at all:
cross-entropy per token measures different things for different segmentations. `H_char = H_token ×
(tokens/characters)` — the token loss scaled by the measured compression rate. *(Implemented as
`chakaso.training.model_metrics.bits_per_character`, whose denominator is the decoded whole target stream
rather than a per-token character count; `tests/test_training_model_metrics.py` tests exactly that,
including a separator target that costs a prediction and contributes no characters.)*

**C3 rejected** — not for speed (the measured matmul comparison above is 1.16× and does not support a
speed argument), but because it trades a verified-gradient property the project currently holds *fully*
for one it can only hold *by sample*, and adds several hundred lines of private autodiff to maintain.
**C4 rejected as the product** per ADR-0033 rule 7 and ADR-0001, and rejected as the *headline* baseline
because its number would measure someone else's training run; it stays available later as a labelled
optional adapter.

## Limitations of this note

- It observes one machine on one day; the free-RAM figure is transient and CI hardware differs.
- The step timings measure one batch shape with `dropout=0.0`; real training adds dropout and data
  loading, so the wall-clock estimates are order-of-magnitude, labelled as arithmetic on measurements.
- The token-count and compression figures for the repository corpus are **estimates** until the tokenizer
  exists; the corpus is also single-author, single-domain technical prose.
- No peak-memory figure was obtained, and none is claimed.
- Nothing in this note is a model result. The first result is E2605.
