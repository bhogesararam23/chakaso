# Starting state — 2026-10-29

A dated snapshot taken before this session changed anything. Numbers here are measurements from this
machine on this date; where a count ages, [`../../agent/CURRENT_STATE.md`](../../agent/CURRENT_STATE.md) is
the live authority and this file is the observation.

## Repository

| Item | Value |
| --- | --- |
| verified `HEAD` at the start | `a2416d184c18fcd04f5690dea3683e634fa4bc03` |
| branch | `main`, tracking `origin/main`, no divergence |
| working tree | clean (the private `docs/*.docx` pack is ignored and was checked with `git ls-files`, not `git status`) |
| remote | `https://github.com/bhogesararam23/chakaso.git` |
| latest remote CI | run `40778307808`, **success**, 2m51s, on `a2416d1` |
| tracked files | 378 |

The previous session's E2610 work is the remote head: preregistration `132839b`, results `92d3120`, and the
diagnosis `a2416d1`.

## Measured results already in the repository

| Record | What it measured | Headline |
| --- | --- | --- |
| E2601 | 984-parameter character model, from random initialization | held-out 1.6650 nats best, below unigram 3.0531 and uniform 3.3322 |
| E2602 | the same protocol over five seeds | 1.7978 ± 0.1767 nats, below both baselines on every seed; overfitting systematic |
| E2603 | grounded-answer workload with a reference oracle | retrieval recall@k 0.8182 / MRR 0.8636; 3 forbidden hits, 3 false retrievals; harness measurement, not a model |
| E2604 | evidence-sufficiency decision | 100% accuracy on the workload, zero answerable cases wrongly abstained; narrow corpus-specific band |
| E2605 | first subword Transformer, 71 documents of this repository's prose | 1.8657 bits/character against 3.1343 character architecture, 3.3224 unigram, 4.8010 uniform; early stopping selected step 1380 of 2400 |
| E2606 | E2605's protocol over five pre-declared seeds | 1.8794 ± 0.0077 against 3.0937 ± 0.0267; seed 7 reproduced E2605 tensor for tensor |
| E2607 | same protocol on 105 documents of licensed Gutenberg prose | 2.0176 ± 0.0118 against 3.2218 ± 0.0153; no run early-stopped, so the level is a floor |
| E2608 | rotary decoder vs transformer, token-matched | 1.6865 against 1.9043 bits/character; 3.3× parameters and 2× context confound; both arms at the 2,400-step ceiling; decoder arm released as `chakaso-base@0.1.0` at Level 2 |
| E2610 | calibrated decoding sweep over the released checkpoint | repetition is a property of greedy decoding of a human-diverse distribution, not a pipeline defect; no decoding setting produces coherent text; H1 inconclusive, H2 failed as worded, H3 supported |

E2610's follow-up list is the current open unit: a converged run or an honest statement that convergence is out
of reach here; a capacity/context-matched architecture comparison; a seed spread on the decoder family; a
workload. This session takes the capacity-matched comparison as **E2611**, preregistered at `e957167` before its
arm ran.

## Open issues and roadmap gaps

[`../../agent/KNOWN_ISSUES.md`](../../agent/KNOWN_ISSUES.md) is the register. Open at the start: **K-001** (no
real workload), **K-002** (components still absent: real semantic judge, real embedding model, vector index,
durable source index, revised prose, autonomous correction, useful model), **K-008** (partial SSRF screen,
no politeness), **K-010** (lexical retrieval over-returns; sufficiency decision not consumed by the runtime),
**K-011** (one dependency rule has no mechanical enforcer), **K-012** (evidence context unversioned),
**K-013** (counts duplicated across documents), **K-014** (direct-to-`main` against the branch convention),
**K-015** (a loaded release reports the adapter's identity), **K-016** (`model serve` never run as a command),
**K-017** (early records have no machine-readable evidence), **K-018** (`--samples-per-prompt` misnomer),
**K-019** (release does not name the code that trained it), **K-020** (gates measure repetition, not coherence),
**K-021** (U+FFFD gate cannot tell a mid-text error from a bound-truncated character). K-009 is closed, with
the three-tier CI read back green.

Roadmap phases P0–P3 and P7 are done; P4, P5 and P6 are partial; P8 is achieved as measured experiments; P9 is
half met (loss is below baselines, useful behaviour is not); P10–P12 are not started; P13 is met as a contract,
not a weight download. The handoff phases 1–6 ("make the curve turn", "put a spread on the reference model",
"answer the architecture question", "attack degeneration", "put a trained model in front of a workload", "domain
selection") are all open; Phase 0 is closed. E2611 addresses Phase 3.

## Test and CI baseline

Measured on this machine at `a2416d1` before any edit:

| Matrix | Command | Result |
| --- | --- | --- |
| dependency-free | `.venv/Scripts/python.exe -m pytest -q` | **1,692 passed, 25 skipped** in 67.29 s |
| `[train,nn]` | `.venv-nn/Scripts/python.exe -m pytest -q` | **2,069 passed, 1 skipped** in 281.24 s |
| lint / format | `.venv/Scripts/python.exe -m ruff check .` and `ruff format --check .` | clean; 363 files already formatted |
| types, default config | `.venv/Scripts/python.exe -m mypy src` | clean over 125 source files |
| types, model-path config | `.venv-nn/Scripts/python.exe -m mypy --config-file configs/mypy-model-path.toml` | run this session; result recorded below |

The runner's counts differ from these by the private-pack hygiene skip, as every earlier record states. CI's
three tiers (`checks`, `train`, `model-path`) are shipped in `.github/workflows/ci.yml` and were green on the
starting head.

## Environment

Read from the OS on 2026-10-29:

| Quantity | Value |
| --- | --- |
| OS | Windows 11 (10.0.26200), x64 |
| Python | CPython 3.13.5 (`.venv`, `.venv-nn`, `.venv311`, `.venv312` exist) |
| `torch` | 2.13.0+cpu, 14 intra-op threads |
| `numpy` | 2.4.0 |
| logical CPUs | 18 |
| CUDA | none (`torch.cuda.is_available()` is `False`); an Intel Arc iGPU is present and unused |
| RAM | 15.43 GB total, **0.68 GB available, 95% load** at inspection |
| free disk | 154.1 GB on the project volume |
| peak process memory interface | answers (`windows:k32`), so a run can report its own peak RSS |

The RAM figure is the binding constraint and the reason a corpus is never materialized as Python objects. The
machine is shared with the author's session; wall clock is reported as observed, never as intrinsic.

## Existing artifacts and checksums

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| E2608 decoder checkpoint (released weights) | `data/experiments/E2608/decoder/E2608-decoder.pt` | `9e832fadf277e848e286f5d5270725c6f9f51c04dbeda00d483b8998b4a713ab` |
| E2608 transformer checkpoint (control) | `data/experiments/E2608/transformer/E2608-transformer.pt` | `db0c52f800390a879ee3946b39ea5228e04159882bed5bc9f37f269400996dd6` |
| E2608 release contract | `data/experiments/E2608/release/model.json` | `5e03da59b647ffaa6f11bd4232cad88fb7222c459efc84f2aebb66eb24be3a8d` |
| E2608 released artifact copy | `data/experiments/E2608/release/E2608-decoder.pt` | `9e832fad…` (same bytes as the checkpoint) |
| E2608 corpus snapshot (file digest) | `data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json` | `b44eb53f586008803539351cb4559486ceea191ef8b1396ab25143b3365906a1` |
| E2608 corpus fingerprint (in-record) | `gutenberg-english-prose-v2` | `e9fc2e92f59d88ea`; snapshot's own checksum `8eacb49f8d244e65…` |
| tokenizer cache | `data/experiments/tokenizers/gutenberg-english-prose-v2-tokenizer-f99386c7f5be11a7.json` | recorded by the run; fingerprint `b597eb34f13d9058` |

The snapshot verifies before this session's runs: `chakaso corpus verify` reports 919 documents / 14,490,005
characters at `e9fc2e92f59d88ea`.

## Work already complete; do not duplicate

- The three-tier CI matrix, the release contract and the Level-2 gate (ADR-0038/0039).
- The model path: manifest, snapshot, document split, BPE tokenizer, packing, two architecture families, the
  training loop with warmup/cosine/accumulation/clipping/early stopping/resume, two checkpoint schemas, the
  adapters, the serving adapter, the vertical slice.
- The calibrated quality suite (`chakaso.training.quality_eval`, `chakaso model quality`) and E2610's sweep.
- The grounded-answer workload and sufficiency decision (E2603/E2604), which measure the harness, not a model.
- The repository's documentation discipline: every capability has a status label; every measured number has a
  record; every decision has an ADR.

What remains genuinely missing is named by K-001 and the roadmap: a workload a model can answer, a converged
model, a capacity-matched architecture answer, a decoder seed spread, and a coherence measure that is not a
repetition statistic. This session's research package, E2611 and its follow-ups are aimed at the middle three.
