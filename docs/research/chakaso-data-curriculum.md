# Chakaso data curriculum: what to train on, in what order, and under which licence

**Status:** Research and proposal. No corpus described here beyond `gutenberg-english-prose-v2` has been
acquired, pinned or measured. Every quantity labelled *estimate* is arithmetic or a literature figure, not a
measurement from this repository.
**Date:** 2026-10-29.
**Companions:** [`open-model-landscape.md`](open-model-landscape.md),
[`chakaso-next-training-design.md`](chakaso-next-training-design.md),
[`chakaso-compute-and-scaling-plan.md`](chakaso-compute-and-scaling-plan.md).

## The starting point, measured

The only external corpus this project has pinned is `gutenberg-english-prose-v2`: 24 Project Gutenberg texts,
919 documents, 14,490,005 characters, corpus fingerprint `e9fc2e92f59d88ea` (E2608). Its tokenizer
(`b597eb34f13d9058`, 4,096 identifiers) encodes it to **6,718,455 identifiers** — 4,690,865 train, 1,023,146
validation, 1,004,444 test (E2608 report). E2608 trained ~19.66M positions, about 4.2 epochs of the training
side.

Four properties of that corpus bound every model result in this repository:

1. **One register.** 19th- and early-20th-century English narrative prose, mostly novels. There is no
   technical, scientific, mathematical, encyclopedic or code text.
2. **One language.**
3. **A document-level split.** E2610's deviation 3 records what this means: held-out documents are unseen,
   but the same authors and works appear on both sides. A model can memorise a style or a work through
   another document of the same book.
4. **A small token budget.** 6.72M identifiers is about 5.8 tokens per parameter for E2608's 3.4M model, and
   about 0.067 tokens per parameter for a hypothetical 100M model. Chinchilla's compute-optimal point is
   roughly 20 tokens per parameter; the corpus is not remotely large enough for a model above ~300M
   parameters, and it is already repeated ~4× at 3.4M.

The data problem is therefore not "find a bigger web scrape". It is "build a licensed, documented, diverse
corpus large enough for the model size the compute can train, and hold out whole works so a score means
something."

## Curriculum stages

The stages below are ordered by what the project can afford and what the next measurement needs. Each stage
has an entry condition (data acquired, licensed and pinned) and an exit condition (a measured run that says
whether the stage helped). A stage that cannot be licensed is not entered.

### Stage 0 — the pinned prose corpus (done)

`gutenberg-english-prose-v2`, 24 books, 6.72M identifiers. Use it as the continuing control: every new
mixture must be compared on the same held-out documents where possible, and the old numbers stay valid.

**What it cannot support:** any claim about technical language, mathematics, code or instruction following;
a model above a few million parameters without heavy repetition; a held-out split by work.

### Stage 1 — a broad public-domain English library (next)

**Source.** Project Gutenberg, whose files are public domain in the United States and distributed under the
Project Gutenberg License; each file's preamble states that users outside the United States must check local
law. The existing catalogue mechanism (`chakaso.training.dataset_catalog`) already pins a URL, byte digest,
retrieval date and licence statement per source.

**Target.** 200–500 English books across fiction, essays, history, biography, philosophy, travel and drama,
selected by a declared rule (for example: the first N books by download count that are English, at least
50,000 characters, and not already in the v2 corpus), with no single author above a declared share (the v2
corpus used 14.6% as its observed maximum). A 200-book corpus is an *estimate* of 40–80M identifiers if the
average book resembles the v2 average (6.72M / 24 ≈ 280k identifiers per book); 500 books would be
100–200M. Measure the actual tokenizer compression; do not plan on the estimate.

**Why first.** It is the cheapest legal expansion: public domain, already supported by the catalogue, no
web scraping, and it directly widens register and topic. It also lets the project test the user's default
hypothesis — a general language foundation with a technical mixture — before buying a GPU or signing a data
agreement.

**Risks.** Gutenberg is still mostly Western, pre-1929 prose; it is not a general web distribution. Its
boilerplate must be stripped (the existing `source_corpus` module does this by the source's own markers).
OCR errors and archaic spellings are part of the distribution, not noise to be silently removed.

### Stage 2 — technical and scientific prose

**Sources, in order of legal simplicity:**

1. **Public-domain scientific classics** already on Project Gutenberg or elsewhere in the public domain:
   Darwin, Faraday, Maxwell, Kelvin, Huxley, Euclid translations, classic astronomy and natural philosophy.
   Same licence posture as Stage 1, different register.
2. **CC0 / public-domain open textbooks and references.** OpenStax and similar publishers use CC BY (not
   CC0); CC BY requires attribution and may have share-alike implications for a derived model that are not
   settled by this repository's current licence review. Treat CC BY as *usable only after a written
   licence decision*, not as equivalent to public domain.
3. **Wikipedia / Wiktionary dumps** are CC BY-SA 4.0 and GFDL. They are a large, clean, multilingual source,
   but the share-alike clause needs a legal decision before weights are released. Do not add them by default.

**Target.** 20–50% of the mixture by tokens, with per-source caps. The first experiment should be a
**technical-prose arm** at the same token budget as a prose-only control, so the effect of register is
measurable independently of size.

**What to measure.** Held-out bits per character on a *technical* held-out set (whole works, not documents),
and on the prose set. A mixture that improves technical loss while leaving prose loss unchanged is the
desired result; a mixture that improves one by damaging the other is a trade-off to report, not a win.

### Stage 3 — mathematics

**Sources.** Public-domain mathematics texts (Euclid, older algebra/calculus treatises) and CC0/CC BY
problem sets after licence review. Synthetic arithmetic and step-by-step derivations can be generated by the
project's own code with a declared generator; that is self-authored data, not teacher data, and must be
labelled as synthetic.

**Why after technical prose.** Mathematics text is a different distribution (symbols, notation, short
proofs) and a small model with a 4k byte-pair vocabulary will spend many tokens on notation. A 4k vocabulary
is probably too small for mathematics and code; a 16k–32k vocabulary is a hypothesis to test, not a default.
The tokenizer decision belongs with this stage, and it changes the model's embedding parameter count.

**Do not** download a large benchmark and train on it. That is contamination, and it makes the benchmark
worthless for the capability claim.

### Stage 4 — code

**Sources.** A curated set of repositories under permissive licences (MIT, Apache-2.0, BSD-2/3-Clause),
pinned by commit hash and file digest, with licence headers retained in the provenance record. The Stack and
similar corpora are large and licence-filtered, but they are not a licence review; do not treat a filter as
permission. Do not scrape repositories whose terms forbid it.

**Why last among the base stages.** Code is a high-value domain for the long-term goal (programming and
engineering), but it is the furthest from the current corpus and the most expensive to tokenize well. A
model that has not learned general English will not learn code from a small code mixture; a code mixture
added too early can consume capacity without producing a measurable code capability.

### Stage 5 — instruction and preference data (post-training, later)

Only after a base checkpoint exists at a measured level. Sources: self-authored instructions and
demonstrations (Apache-2.0), openly licensed instruction datasets after review, and — if a teacher model is
ever used — teacher-generated data with documented provenance, permitted use and its own experiment record.
ADR-0033 forbids substituting a hosted model's answers for Chakaso's; distillation is a separate,
labelled route, not the product.

## Mixture proposal (declared, not yet run)

A first technical mixture for a 5–10M-parameter model, expressed as token shares before any run:

| Component | Share | Source posture |
| --- | --- | --- |
| General English prose | 50% | Project Gutenberg, public domain |
| Technical / scientific prose | 25% | public-domain classics; CC BY only after a licence decision |
| Mathematics | 10% | public-domain texts + self-generated arithmetic/derivations |
| Code | 10% | permissive licences, pinned commits |
| Reference / encyclopedic | 5% | public domain or CC0; Wikipedia only after a share-alike decision |

These are **starting weights**, not findings. The experiment must declare them before the run and report
per-domain held-out loss. A mixture is not "better" because its average loss is lower; it is better only if
the domains it claims to serve improve without a measured regression elsewhere.

## Provenance and licence rules

Every source in a training corpus must have, in the catalogue:

- a stable identifier (Gutenberg number, repository + commit, edition identifier);
- the exact URL or local path it was retrieved from;
- a SHA-256 of the bytes actually read;
- a retrieval date;
- the licence statement as the source states it, not as a summary;
- a redistribution decision (the corpus bytes stay out of Git under the artifact policy);
- an exclusion reason when a candidate source is rejected.

A source with no licence statement is not admitted. A source whose licence is "free to read" but not
"free to reuse" is not admitted. A source whose licence is CC BY-SA is not admitted until a written decision
says what share-alike means for the weights; until then it is a candidate, not data.

## Quality filters

Applied deterministically and reported per source:

1. **Encoding and normalization** — UTF-8, Unicode NFC, line endings, blank-line runs (the existing
   `retrieval.normalize` rules; the training corpus has its own normalization path).
2. **Boilerplate removal** — Gutenberg headers/footers, licence blocks, tables of contents, page numbers.
   The rule is source-specific and recorded, not a generic regex.
3. **Length** — drop documents below a declared character floor (the v2 preparation used a document bound);
   report the dropped count.
4. **Language** — English only for the first curriculum; a language identifier's output is recorded, not
   trusted silently.
5. **Exact deduplication** — content digest; the manifest already records duplicate groups.
6. **Near-duplicate detection** — shingled similarity (the v2 report compared candidate pairs at 8-word
   shingles); report the pairs examined and the groups found, not only the count.
7. **Source caps** — no single source above a declared share of tokens; the v2 corpus's 14.6% is the
   precedent.
8. **Contamination** — see below.

## Contamination and split integrity

The current split is by document. E2610's deviation 3 is the warning: the same work can appear as two
documents. The next curriculum must:

- hold out **whole works and authors** for the evaluation sides, not just documents;
- keep every benchmark's items out of training by exact and n-gram overlap checks;
- record the overlap check and its result in the experiment report;
- never tune a gate after reading the result it gates;
- report a contamination finding as a limitation, not as a reason to delete the run.

For later benchmarks (GSM8K, HumanEval, MMLU, GPQA), the same rule applies: a benchmark score is only
interpretable if the benchmark's items were not in the training mixture and the model was not selected on
the test split. The project has no such benchmark yet, and should not acquire one by downloading a test set
into training.

## What is feasible now, and what is aspirational

| Stage | Feasible in this session? | What it needs |
| --- | --- | --- |
| v2 prose corpus | Already pinned | — |
| 200–500 Gutenberg books | **Yes, with network and time** | catalogue entries, byte digests, licence statements, tokenizer fit, disk |
| 20–50 public-domain science books | **Yes, with network and time** | same, plus a source-specific boilerplate rule |
| Mathematics / code mixture | **Not yet** | licence review, tokenizer decision, a code/math evaluation set |
| 1B+ token general corpus | **No on this machine** | a legal data pipeline, disk, and ~2 days of CPU at measured rates (see compute plan) |
| Instruction data | **No** | a base checkpoint worth instructing |

## Open questions this document does not answer

- Whether a 4k byte-pair vocabulary is adequate for technical prose; it is likely too small for mathematics
  and code, but the trade-off against embedding parameters at 5–10M total parameters is unmeasured.
- Whether Gutenberg's OCR/archaic distribution helps or hurts a modern technical target.
- Whether a CC BY-SA source can be used in a released model; this is a legal question, not an engineering
  one.
- Whether a mixture weight chosen before a run is better than a weight chosen by a small proxy experiment;
  the project's rule is to declare first and report the result, not to tune on the test set.

## References

1. Project Gutenberg. *The Project Gutenberg License.* <https://www.gutenberg.org/policy/license.html>,
   accessed 2026-10-29.
2. E2608. [`../../experiments/E2608/record.md`](../../experiments/E2608/record.md) — corpus, split,
   tokenizer and the measured 1.6865/1.9043 result.
3. E2610. [`../../experiments/E2610/record.md`](../../experiments/E2610/record.md) — the held-out-by-work
   limitation (deviation 3) and the decoding diagnosis.
4. Lee, K. et al. *Deduplicating Training Data Makes Language Models Better.* arXiv:2107.06499, 2021.
   <https://arxiv.org/abs/2107.06499>
5. Penedo, G. et al. *The FineWeb Datasets.* arXiv:2406.17557, 2024. <https://arxiv.org/abs/2406.17557>
6. Longpre, S. et al. *The Data Provenance Initiative.* arXiv:2310.16787, 2023.
   <https://arxiv.org/abs/2310.16787>
7. OLMo Team. *Olmo 3.* arXiv:2512.13961, 2025 — Dolma 3's mixture is a worked example of a documented
   multi-domain corpus. <https://arxiv.org/abs/2512.13961>
8. Hugging Face. *SmolLM3.* Blog, 8 Jul 2025 — a staged small-model mixture with published ratios.
   <https://huggingface.co/blog/smollm3>
