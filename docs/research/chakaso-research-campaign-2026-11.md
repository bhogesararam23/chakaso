# Chakaso research campaign, 2026-11-01 — index, decision matrix and roadmap

**Status:** Research. This document is an index and a plan, not a result. Nothing here is Implemented except
the one code change the campaign produced, and that change has its own commit (`cfdcd1a`) and its own issue
([K-024](../agent/KNOWN_ISSUES.md)). Capability claims carry the labels defined in
[`docs/README.md`](../README.md); every number below is either measured in this repository or quoted from a
source that is named.

**Position when this was written.** The largest trained artifact is the 3,413,760-parameter rotary decoder
([ADR-0039](../decisions/ADR-0039-three-level-model-release-gate.md) Level 2, Level 3 withheld). It produces no
usable text, and no measurement in the repository is evidence of language-model competence
([K-001](../agent/KNOWN_ISSUES.md)). Eleven experiment records exist (E2601–E2608, E2610–E2612), and E2612 was
undocumented when this campaign began.

---

## 1. What the campaign produced, and where each thing lives

| # | Deliverable | Document |
| --- | --- | --- |
| 1 | Codebase and experiment audit | [model-path-audit](../internal/engineering/model-path-audit-2026-11-01.md), [measurement-harness-audit](../internal/engineering/measurement-harness-audit-2026-11-01.md) |
| 2 | Comparative open-model research | [frontier families](../internal/engineering/open-model-research-llama-mistral-deepseek-2026-11-01.md), [small-model families](../internal/engineering/open-model-research-qwen-olmo-gemma-phi-smollm-2026-11-01.md), and 18 annotations in [open-model-landscape.md](open-model-landscape.md) |
| 3 | Architecture and implementation gap analysis | §3 below, plus the two audits |
| 4 | Data and licensing research | [data-and-licensing-research](../internal/engineering/data-and-licensing-research-2026-11-01.md), 14 annotations in [chakaso-data-curriculum.md](chakaso-data-curriculum.md) |
| 5 | Training and scaling research | [foundations-research](../internal/engineering/foundations-research-2026-11-01.md), 12 annotations in [chakaso-compute-and-scaling-plan.md](chakaso-compute-and-scaling-plan.md) |
| 6 | Tokenizer and context-length assessment | L-10 to L-12, C-11, H11, H12; the tokenizer defect in §2 |
| 7 | Capability and benchmark strategy | 10 annotations in [chakaso-capability-benchmarks.md](chakaso-capability-benchmarks.md) |
| 8 | Hardware and compute feasibility | "Hardware and compute feasibility, measured 2026-11-01" in [chakaso-compute-and-scaling-plan.md](chakaso-compute-and-scaling-plan.md) |
| 9 | Research annotations, with a count | **70**, indexed in §4 |
| 10 | Prioritised roadmap with milestones | §5 |
| 11 | Reconciled state, task ledger and issues | [CURRENT_STATE](../agent/CURRENT_STATE.md), [ACTIVE_TASK](../agent/ACTIVE_TASK.md), [KNOWN_ISSUES](../agent/KNOWN_ISSUES.md) (K-024 to K-029 added) |
| 12 | Final campaign report | [research-campaign-report-2026-11-01](../internal/engineering/research-campaign-report-2026-11-01.md) |

**The trust layer is [the verification ledger](../internal/engineering/research-verification-ledger-2026-11-01.md).**
33 claims were independently re-checked there — how each was checked, what came back, and which first readings
were wrong. Read it before relying on any finding in this index.

---

## 2. The campaign's one hard engineering result

The byte-pair merge loop had been indexed for speed in `6f3274f`, and the indexed heap discarded a pair whose
count fell as an earlier merge rewrote the shapes holding it. Such a pair left the candidate set permanently, so
the loop stopped choosing the most frequent pair — the rule its own docstring states.

Measured, not argued:

- Against a brute-force reference, the merge sequences disagree from **rank 8** and differ on **25 of 43**
  merges on a 43-piece fixture.
- Re-fitting the exact E2608 training side produced **`a1ef04f64739cd4b`** instead of the record's
  **`b597eb34f13d9058`**, diverging at merge rank 5 over 3,692 of 3,839 merges, and encoding 20 of its own
  training documents as 152,939 identifiers against the cached vocabulary's 152,053 (+0.583 %). E2608, E2611 and
  the released `chakaso-base@0.1.0` were not re-trainable from `main`, which is what
  [ADR-0036](../decisions/ADR-0036-corpus-snapshot-recovery-from-history.md) forbids.
- `cfdcd1a` fixes it and reproduces `b597eb34f13d9058` exactly, with the token count back to 152,053. Two new
  tests compare the indexed loop against the reference; **both fail against the pre-fix loop**, and nothing in
  the previous 1,705-test suite caught the change, because no test stated the rule independently of the
  implementation.
- The cost is measured too: an exact fit of the 12,663,402-character E2612 sample takes **2,116.4 s (35.3 min)**,
  against 29.2 s at 4.0M for the fast-but-wrong loop and 145.55 s for E2612's entire preparation.
- **The fix moves a boundary rather than erasing it.** E2612's own vocabulary `434debe82f949347` was fitted *by
  the defect*, and the fixed loop yields `3abe4df4b123d4c5` on the same sample. E2612's *comparison* is valid —
  both arms scored one shared vocabulary, so its hypotheses stand — but it is now reproducible only through a
  git-ignored cache.

Everything else in this campaign is documentation of state; this is the one place where the repository was not
what its records said.

---

## 3. Gap analysis: what Chakaso has, what it lacks, what is misleadingly claimed

| Area | What exists, verified | What is missing | Status of any claim about it |
| --- | --- | --- | --- |
| **Measurement integrity** | Bits-per-character arithmetic is principled; baselines are fitted on training targets only; the tokenizer sample is drawn from the training side only; snapshots re-hash every document body; E2612's arms are equal on ten fields checked individually | Nothing that invalidates a published number | The harness is honest. This was the campaign's most useful negative finding — the plumbing does not need rebuilding |
| **Split hygiene** | Whole-work splitting with `works_on_multiple_sides` reported (E2612: 312/56/62, zero overlap) | The field is **reported, not enforced**; a run without `--whole-work-splits` can leak within a work and say so in its artifact | E2607/E2608/E2611 carry dated corrections (K-023); their absolute numbers stay within-work |
| **Tokenizer** | Deterministic, cached, content-digested, total round-trip | Vocabulary size never swept; `target_size` is not asserted after a fit; exact fitting is expensive | Post-fix, the loop matches its documented rule. The 4,096 choice has **no local evidence** behind it (H12) |
| **Architecture** | 3.4M rotary decoder, reproduced parameter arithmetic three ways, RoPE/RMSNorm/GELU/tied embeddings verified against the built model | No measured reason to change any of it | Frontier mechanisms are **0 adopt / 1 adapt / 2 test / 8 reject** at this scale (L-01). A change whose value is "a modern architecture" is not supported by any evidence in the repository or the literature at 3.4M |
| **Data** | 430 works, 149,000,499 characters, five registers, licensed and pinned at the work level | A modern register, code, and a pinned catalogue input | The record itself says results do not transfer to modern text. **No partition in any run is modern**, so the corpus's age has never been measured, only disclaimed |
| **Budget** | 20M identifiers per arm, ~90 min CPU | Any converged run: all four decoder experiments completed `budget_ceiling` with `stopped_early: false` | Every absolute bits-per-character in the repository is a **floor**. Stated in E2611 and E2612; still not stated in CURRENT_STATE's headline numbers |
| **Evaluation** | Held-out per-register scoring with baselines and a declared band | A seed spread for the decoder at any size; an instrument that could detect a modernity shift | E2609 was never run, so the 0.02 materiality band is a convention inherited from E2611, not a measured noise floor |
| **Reproduction** | Snapshot restore/verify, atomic artifact writes, refuse-to-clobber | Digests that survive a different machine; tokenizer caches inside the reproducibility story | Report digests equal file hashes over bodies containing timings (K-028). A re-run can reproduce every metric and no digest |

---

## 4. Annotation index — 70 entries

Numbered so a future agent can cite one without rereading a file. Count is verified by pattern-matching each
document, not by counting on the way through writing.

| Prefix | Entries | Document | Subject |
| --- | --- | --- | --- |
| **L-01 … L-18** | 18 | [open-model-landscape.md](open-model-landscape.md) | Frontier and small-model mechanisms, what transfers below 10M, citation failures |
| **D-01 … D-14** | 14 | [chakaso-data-curriculum.md](chakaso-data-curriculum.md) | Licensing reaching text versus packaging, corpus candidates, provenance gaps, mixture weighting |
| **C-01 … C-12** | 12 | [chakaso-compute-and-scaling-plan.md](chakaso-compute-and-scaling-plan.md) | Measured throughput and memory, exact-fit cost, what scaling laws do and do not license, GPU as a data-quantity need |
| **B-01 … B-10** | 10 | [chakaso-capability-benchmarks.md](chakaso-capability-benchmarks.md) | What a legitimate instrument looks like at 3.4M, floors, weighting traps, benchmark licences |
| **H8 … H15** | 8 | [hypotheses.md](hypotheses.md) | Eight new falsifiable hypotheses with the run that would overturn each |
| **Q (8 questions)** | 8 | [open-questions.md](open-questions.md) | What the campaign could not close, and why |

Supporting, not counted as annotations: 33 verification rows in
[the ledger](../internal/engineering/research-verification-ledger-2026-11-01.md); the ranked findings of the two
audits; six new issues (K-024 … K-029).

---

## 5. Decision matrix

Ranked by value per unit of this machine's time. "Evidence" is the strength of the *case for trying it*, not of a
result already obtained here. Hardware column assumes the measured host: 18 CPU cores, 16.6 GB RAM, no GPU.

| Rank | Candidate | Expected benefit | Evidence | Cost | Hardware | Risk | Required experiment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **E2613 — exposure control (P½ arm)** | Settles whether the mixture costs prose or prose *dose* does; converts E2612's declared confound into a number | High that the confound is real — measured at 50.51 % pool share and ~2× exposure difference | ~90 min train + ~35 min shared tokenizer prep, amortised | Enough | Low. The worst outcome is a clean null | H8. Three arms at one cap, `--sample-prompts held_out` |
| 2 | **Seed spread at the reference configuration** | Gives the 0.02 band a measured noise floor; every "inside the band" claim currently rests on a convention | High that it is needed — E2602 has a five-seed spread for characters and the decoder has none | 3 × 90 min | Enough | Low; it is arithmetic after the runs | E2609-shape study on E2612-T's configuration, seeds 7/11/13 |
| 3 | **Register-reweighting sweep** | Finds whether the mixture has an interior optimum; the lever with the largest measured effect Chakaso has ever moved | Medium-high — E2612 measured the trade-off at one point; Shukor et al. (arXiv:2507.09404) claim 10–20 runs fit mixture laws, at their scale | 4–5 × 90 min, one shared vocabulary | Enough | Medium — at 5 prose shares, multiple comparisons need a declared primary | H9. Prose share 25/50/75/100 % at fixed tokens |
| 4 | **Era partition plus a licence-clean modern slice** | The only instrument that can detect the archaic-versus-modern shift every record currently disclaims | High that the current instrument is blind to it (B-01); high on the CC0 newspaper licence | ~1 day to acquire and prepare, then runs | Enough for 150M characters | Medium — OCR boilerplate needs a stripping rule written *before* selection | B-09. `us-pd-press-v1` held out by edition date |
| 5 | **Vocabulary sweep at fixed parameters** | Tests whether 4,096 is binding; 30.7 % of parameters sit in the table today | Medium — the published law points the other way and stops at 33M | 4 × 90 min, but 4 tokeniser fits ≈ 4 × 35 min | Enough | Medium — superlinear fit cost may make 16K/32K unaffordable | H12. {2k,4k,8k,16k} with feedforward narrowed to hold parameters |
| 6 | **Context-length arm (64/128/256)** | Challenges an assumption nobody has tested; Shi et al. (2502.01481) is the closest-scale source found | Medium, and it is a *reversal* of the roadmap's direction | 3 × 90 min | Enough | Low | H11 |
| 7 | **Schedule arms: WSD, warmup ∝ N, decay fraction** | Cheap — changes no other variable; both sources' evidence sits nearer this scale than any architecture claim | Medium — 2410.05192 at 0.1B–1.2B; 2406.19146 at 55M–901M | 2 × 90 min | Enough | Low | Not yet a hypothesis; preregister before spending |
| 8 | **Document-boundary masking arm** | The mechanism exists untested since ADR-0040 | Low-medium — with 4 KB documents and 256 context, windows rarely span a boundary | 2 × 90 min | Enough | **Blocked**: the audit found masking applied in training/validation loss but not in the held-out scorer, so a result would be uninterpretable until fixed | H13, after the scorer asymmetry is fixed |
| 9 | **9.84M-parameter capacity step** | Would test ADR-0041's next declared unit | Low *from this campaign* — E2612's own rule 2 says reweight before scaling, and no architecture or capacity result here supports spending it | ~4–6 h, unmeasured at that size | RAM is the risk, not time | High — costs a day and answers a question the mixture result says is not yet next | Defer behind ranks 1–4 |
| — | **GQA / MLA / MoE / SWA / MTP / FP8 / soft-capping / big vocabulary** | — | **Rejected**: 8 of 11 frontier mechanisms, with Meta's own "worse than the baseline for small model sizes" on MTP and Llama 2's own table putting GQA at 70B | — | — | Adopting any of these is cargo-culting, documented in L-01…L-09 | No experiment. Reopen only with a source measured below 10M |
| — | **Teacher-generated data (TinyStories, Cosmopedia, FinePhrase, Nemotron synthetic)** | Would likely improve surface text | High that it works; **contractually excluded from the released line** | Cheap | Enough | Silently displacing scratch-trained weights breaks ADR-0033 | A separately labelled research arm at most, never merged into the base |
| — | **Code register** | The design wanted 10 % code | Low — no licence-clean route found: The Stack v2 is gated and a curator's permissive filter is not a licence | — | — | A released model trained on it needs a licence decision the repository does not have | Not before a written licence decision |

---

## 6. Roadmap, with measurable milestones

One unit at a time, in the repository's method. Each milestone names the artifact that proves it.

**Phase A — make the current result mean something. (≈ 1 machine-day, no new hardware)**

1. **A1 E2613 three-arm exposure control.** *Milestone:* a record with H8 answered one way or the other, and a
   generation block using `--sample-prompts held_out`. *Falsifier:* if T's prose equals P½'s, the mixture is
   near-free and E2612's H2 failure was dose.
2. **A2 decoder seed spread (A1's config, 3 seeds).** *Milestone:* the standard deviation of test bits per
   character at 3.4M, published, and the 0.02 band either justified or replaced by a measured threshold.
3. **A3 fix the two measurement defects before they contaminate a result:** the scorer asymmetry under document
   masking (blocks rank 8) and `dropped_duplicates` under whole-work splits (K-025). *Milestone:* tests that fail
   against the pre-fix code, as `cfdcd1a`'s now do.

**Phase B — widen what the instrument can see. (≈ 2–3 machine-days)**

4. **B1 acquire `us-pd-press-v1`** under the preregistration list in
   [the data review](../internal/engineering/data-and-licensing-research-2026-11-01.md): licence quoted in both
   halves, per-object SHA-256, seeded selection rule, split grouped by edition/date, boilerplate rule written
   before slicing. *Milestone:* a pinned snapshot, a data card, and a corpus fingerprint in a record.
5. **B2 era partition.** *Milestone:* the first measurement of archaic-versus-modern, which no current artifact
   can produce.
6. **B3 decide where a cited tokenizer lives.** *Milestone:* a written answer to K-024(c) — either vocabulary
   files that a measured record cites enter the repository, or every record states the fingerprint it could not
   reproduce and how it knows.

**Phase C — spend the lever that has evidence. (≈ 4–6 machine-days)**

7. **C1 prose-share sweep** (H9), one shared vocabulary, four prose shares, declared primary comparison.
   *Milestone:* a mixture weight chosen on measurement rather than on the catalogue's accident.
8. **C2 vocabulary sweep at fixed parameters** (H12) and **C3 context arms** (H11), scheduled as cheap paired
   runs. *Milestone:* a tokenizer and context choice with local evidence, which today neither has.
9. **C4 only then: the capacity step.** *Milestone:* E2612's rule 1 becomes licensed by a reweighted mixture, not
   by a wish.

**Explicitly not in this roadmap:** any frontier architecture mechanism; code data; teacher-derived data in the
released line; a CLI, web frontend or hosted API; a benchmark score presented as capability; weakening a
reproduction guard to let a re-run proceed.

---

## 7. What remains genuinely uncertain, stated plainly

1. **Whether any of this reaches usefulness.** E2610 found no decoding setting that writes coherent text from the
   released checkpoint, and E2612's better arm produced *more* repetitive text than its control. Held-out loss and
   text quality moved in opposite directions in the campaign's own data. No instrument currently in the repository
   is known to detect a real improvement in what the model has learned.
2. **Whether 20M tokens can show anything.** Every decoder record is truncated at the step ceiling, and the
   literature says repetition is not a substitute for unique data (arXiv:2305.16264). The truncation gap cannot be
   closed on this machine by more steps alone.
3. **Whether the band means anything yet.** It is inherited, not measured; A2 exists to fix that.
4. **Whether the mixture's prose cost is real.** H8 is the campaign's single highest-value unresolved question and
   costs 90 minutes.
5. **What the corpus's age costs the model.** Never measured; no partition is modern. B1/B2 exist to measure it.
6. **Whether the project can release weights trained on anything modern.** Every web-scale candidate licenses its
   packaging, not its text, and two open legal questions (does share-alike reach weights; is a packaging-only
   grant enough for a release) remain unanswered in writing.
