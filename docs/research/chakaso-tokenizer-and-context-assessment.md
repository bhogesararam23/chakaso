# Tokenizer and context-length assessment

**Status:** Research. Assessment, not result: it consolidates what is **measured in this repository** about the
identifier space and the position window, separates it from what is **quoted from a source at another scale**, and
names what is **derived arithmetic** rather than observation. No experiment described here has been run.
Campaign context: [the index and decision matrix](chakaso-research-campaign-2026-11.md); verification status of
each external claim: [the ledger](../internal/engineering/research-verification-ledger-2026-11-01.md).

## 1. What the tokenizer actually is, measured

| Property | E2608 / E2611 arm | E2612 arms | Source |
| --- | ---: | ---: | --- |
| Kind | byte-pair subword, trained by `chakaso.training.subword` | same | `tokenizer.kind` |
| Vocabulary | 4,096 identifiers | 4,096 | `tokenizer.size` |
| Learned merges | 3,839 | 3,839 | `tokenizer.merges` |
| `min_frequency` | 3 | 3 | `tokenizer.min_frequency` |
| Characters per token, test side | **2.1559** | **2.0993** | `tokenizer.characters_per_token_test` |
| Test identifiers | 1,004,444 | 256,125 | `tokenizer.tokens_per_side.test` |
| Fit sample | whole training side, 10,129,113 characters | seeded 12,000,000-character sample (747 documents) | `plan.tokenizer_sample_characters` |
| Fingerprint | `b597eb34f13d9058` | `434debe82f949347` | `tokenizer.fingerprint` |

The 4,096 decomposes exactly: **256 byte values + 1 document-end + 3,839 learned merges = 4,096**
(`FIRST_MERGED_ID` is 257). So 93.7 % of the identifier space is learned and 6.3 % is the byte base — which is
why a vocabulary change is a *training* change, not a configuration change.

**T-01 — Observation.** Chakaso's compression is **2.0993–2.1559 characters per token** on English prose. For
reference, Llama 3 reports improving compression on an English sample from **3.17 to 3.94 characters per token**
when it moved to a 128,256-entry vocabulary (arXiv:2407.21783 §3.2 and Table 3, quoted in
[the frontier review](../internal/engineering/open-model-research-llama-mistral-deepseek-2026-11-01.md)). The
comparison is informative about the *procedure*, not the target: Meta justified its vocabulary by a measured
compression ratio on its own corpus, and Chakaso makes the same measurement but has never compared across
vocabulary sizes. Note also that the review found **no vocabulary-size sweep reported at all** in the Llama 3
text — 128K is presented as a decision with a measured rationale, not a measured optimum.

**T-02 — Limitation of the current numbers.** The two rows above are not comparable as compression, because they
are different corpora (a 24-work prose corpus against a 430-work five-register corpus) **and** the E2612
vocabulary was fitted by the merge loop between `6f3274f` and `cfdcd1a`, which did not implement the rule it
documents ([K-024](../agent/KNOWN_ISSUES.md)). Any statement of the form "the tokenizer achieves X" for E2612 is
a statement about an artifact current code cannot re-fit — refitting its exact sample today yields
`3abe4df4b123d4c5`.

## 2. Why 4,096 is defensible at this scale, and why nobody knows

**T-03 — Derived arithmetic, and the binding constraint.** With tied input/output embeddings at `d_model` 256,
the embedding table costs **4,096 × 256 = 1,048,576 parameters — 30.7 %** of the 3,413,760 total. Every further
4,096 identifiers cost another 1,048,576, i.e. **another 30.7 % of the whole current model**. A 32,768-identifier
vocabulary would need a table roughly **eight times the size of the entire model today**, and a 128K-identifier
table about 39 times it. This is arithmetic over the declared configuration, not a measurement.

**T-04 — Result from the literature, with its scale stated.** Vocabulary optima grow with model size: Tao et
al. (arXiv:2407.13623) report **N_vopt ∝ N^γ with γ ≈ 0.83** across 33M–3B parameters, and Takase et al.
(arXiv:2406.16508) find larger vocabularies better from 5k to 500k at a fixed 680M parameters — but explicitly
note that "**restricted corpora suppress these advantages**". **Neither tests below 33M**, roughly ten times
Chakaso's parameters, and the ≈37K–43K figure quoted for 33M is the authors' own projection from a low-cost
measurement, not a measured 33M result.

**T-05 — Interpretation, and the reason this is a legitimate local experiment.** The published direction says a
small model wants a small vocabulary, which supports staying at 4,096. The published *absolute* optimum points
much higher, which would destroy the parameter budget. **No source resolves that collision below 10M parameters**,
so the question is genuinely open rather than unanswered-by-negligence. The sweep that resolves it is H12:
{2k, 4k, 8k, 16k} fitted at **fixed total parameters**, narrowing the feedforward to pay for a larger table,
scored in **bits per character** so the unit does not confound the tokenizer change, and reporting
characters-per-token per arm.

**T-06 — Practical blocker, measured.** Exact fitting is now expensive: on the multi-register corpus,
1.0M characters fit in 6.6 s, 4.0M in 494 s and the 12.7M-character E2612 sample in **2,116.4 s (35.3 min)**.
A four-point vocabulary sweep therefore pays four fits, and the two largest may not be affordable at all on
16.6 GB. Mitigation: shrink the *fitting* sample below the training sample and declare it — the vocabulary is a
property of its sample, so this changes comparability and must be preregistered rather than discovered.

## 3. The separator, the byte base and what they cost

**T-07 — Observation.** The vocabulary reserves identifier 0–255 for bytes and 256 for the document end, so any
character outside the pre-tokenizer's piece alphabet degrades to raw bytes, and decoding is total by
construction. The E2611 record quotes a generated U+FFFD in the model's own output; that is a decoding artifact of
byte fallback, not a tokenizer corruption, and [K-021](../agent/KNOWN_ISSUES.md) already tracks the gate that
cannot tell the two apart.

**T-08 — Limitation.** Archaic-public-domain English is a *narrow* alphabet for tokenizer purposes. A vocabulary
fitted on it has no pressure to learn the symbols that modern technical registers need: code punctuation, set
theory and category-theory notation, SI units, URL fragments, arrow and big-operator glyphs. The evidence is
indirect and clear: the mathematics partition holds **3.04361** bits per character against prose's 1.80926
[E2612](../../experiments/E2612/record.md), and a tokenizer trained mostly on narrative is one plausible cause —
but it is **not separated** from "the model is under-trained on a thin register", because mathematics has only 5
held-out works and no arm exists that holds the vocabulary fixed while changing the register.

## 4. Context length: the assumption nobody has tested

**T-09 — Observation.** Every trained artifact uses **context 256**. It has never been varied at the reference
configuration. The roadmap assumed context should grow; the campaign found the opposite direction in the nearest-
scale source.

**T-10 — Result, and the most scale-appropriate citation found.** Shi et al. (arXiv:2502.01481) ran a six-layer
transformer over 12M / 25M / 50M-token slices and conclude that "**deploying expansive context windows on
constrained datasets guarantees suboptimal convergence**". At 20M training tokens and 3.4M parameters, Chakaso is
in the regime that paper measured, not in the 7B/128K regime the frontier reports describe. **H11** tests 64/128/256
at matched parameters and tokens. This is a hypothesis to overturn an assumption, not a conclusion.

**T-11 — Derived, and the reason document-boundary masking is nearly a no-op here.** E2612 packed
2,737 documents into 20,000,402 identifiers, a mean of **~7,300 identifiers per document** — about **28× the
256-position window**. A window therefore almost always lies wholly inside one document, so
[ADR-0040](../decisions/ADR-0040-document-boundary-attention-masking.md)'s masking can change only a small
minority of windows (the derived order of magnitude is a few per cent, computed as ~2 × 256 / 7,300; not
measured). **This is arithmetic over reported fields, not an experiment.** Its consequence for the roadmap is
that H13 has a low prior at this configuration and should not consume a run before the documents get shorter or
the window gets longer.

**T-12 — Blocker, independent of the prior.** The model-path audit found document masking applied in
training and validation loss but **not** in the held-out scorer
(`src/chakaso/training/nn_evaluation.py:72` against `nn_training.py:688-694`), so a masked run's two numbers use
different attention semantics. **H13 could not be tested honestly while that asymmetry stood.** It was fixed on 2026-11-01 (ACTIVE_TASK T5,
[ADR-0040](../decisions/ADR-0040-document-boundary-attention-masking.md) amendment): H13 is now testable and
still unrun.

**T-13 — Observation.** RoPE `theta = 10,000` at context 256 is well inside the regime the original formulation
targets. The long-context base-frequency scaling the frontier reports discuss (10,000 → 500,000 and beyond in
Llama 3 and DeepSeek) addresses extrapolation past training length, which this model neither attempts nor needs.
Adopting a larger theta would change nothing measurable here; it is on the reject list of
[the frontier review](../internal/engineering/open-model-research-llama-mistral-deepseek-2026-11-01.md) for that
reason, and reopening it requires a context arm, not a theta change.

## 5. What to change, in order, and what not to

| Item | Verdict | Reason |
| --- | --- | --- |
| Sweep vocabulary at **fixed parameters** | **TEST** (H12) | 30.7 % of the model sits in the table; the published law and this budget disagree below 10M and no source resolves it |
| Sweep context 64 / 128 / 256 | **TEST** (H11) | The nearest-scale literature points the opposite way to the roadmap's assumption |
| Import a 32K/128K/152K vocabulary | **REJECT at this scale** | The table alone would exceed the current model several times over (T-03) |
| Adopt RoPE base-frequency scaling | **REJECT** | It solves long-context extrapolation, which is not this configuration's problem (T-13) |
| Turn on document masking now | **DEFER** | Nearly a no-op at ~7,300-identifier documents against a 256 window (T-11), and the scorer asymmetry makes any result uninterpretable (T-12) |
| Keep `min_frequency` 3 as an unexamined default | **CHANGE** | It was mis-declared in E2612's record before the run, and it has never been varied; it interacts with the vocabulary sweep, so declare it as a swept parameter or pin it explicitly |
| Re-fit vocabularies per arm | **CHANGE — process** | 35.3 min per fit at the current sample; a multi-arm programme should fit **once** and hold the vocabulary across arms, which costs no methodological clarity and buys the runs back |

## 6. What this assessment does not establish

- **No capability claim** follows from any compression ratio here
  ([K-001](../agent/KNOWN_ISSUES.md)). Characters per token measures a tokenizer, not a model.
- The mathematics-register result (T-08) is **not** attributed to the tokenizer. Separating that requires an arm
  that holds the vocabulary fixed and changes the register, which does not exist.
- T-03, T-11 and the 256-window arithmetic are **derived from reported fields**, not measured. They are stated so
  a reader can recompute them, and every input is named.
- Nothing was run against modern English, code, mathematics in modern notation, or any non-English text; the
  entire identifier space is calibrated on archaic public-domain prose, and the 12.7M-character fit sample is
  652 documents out of a 6,512-document training side.
- The cost figures are **one machine**: 18 CPU cores, 16.6 GB RAM, no CUDA device, shared with an interactive
  session. They are conditions, not properties.
