# ADR-0041: the next training target is a general base with a technical mixture, and the next unit is data, not architecture

**Status:** Accepted
**Date:** 2026-10-29
**Related:** [ADR-0033](ADR-0033-model-centred-product-contract.md) (the model is the product),
[ADR-0037](ADR-0037-reference-model-sized-by-measured-throughput.md) (the reference decoder),
[ADR-0039](ADR-0039-three-level-model-release-gate.md) (the release gate), E2608, E2610, E2611.

## Context

The project has a measured 3.4M-parameter rotary decoder that reaches 1.6865 bits per character on held-out
public-domain prose, a 1.04M-parameter transformer control at 1.9043, and a corpus of 6.72M subword
identifiers from 24 books in one register. Both E2608 arms selected the 2,400-step ceiling with validation
still falling. E2610 showed that the released checkpoint's repetition is a property of greedy decoding of a
human-diverse distribution, not a pipeline defect, and that no decoding setting produces coherent text. E2611
is the capacity- and context-matched architecture comparison that E2608's confound required.

The long-term direction is a model that can learn, explain and assist with language, mathematics, physics,
engineering and research documents, and that can be integrated into other software. The question this record
settles is narrower and comes first: **what should the next training unit be, and for which target?**

## Decision

1. **The target is a general English base model with a deliberately selected technical/scientific mixture.**
   It is not a science-only, mathematics-only, code-only or research-document-only model yet. The default
   hypothesis in the session brief — a general foundation with a technical mixture, followed by separately
   evaluated domain adaptation and instruction tuning — is adopted as the working target because it is the
   only one whose data can be licensed and measured at this scale, and because domain specialization without a
   general base would confound two unknowns.
2. **The next unit is an equal-token data experiment at the measured 3.4M decoder configuration.** The model,
   tokenizer (4,096), context (256), schedule (cosine for new runs, constant for controls) and evaluation
   protocol are held fixed. The only variable is the corpus: a prose control against a technical mixture, at
   the same token budget. The design is in
   [`chakaso-next-training-design.md`](../research/chakaso-next-training-design.md).
3. **Architecture and tokenizer changes wait for the data result.** MoE, MLA, FP8, multi-token prediction, a
   larger vocabulary, longer context and a new attention variant are each a separate controlled comparison
   after the data question. The decoder family E2608/E2611 measured is the reference until a measurement
   says otherwise.
4. **Scale only after the data lever is shown to work.** If the equal-token mixture experiment improves
   held-out loss without a prose regression, the recipe moves to a ~9.84M-parameter decoder on ~197M tokens;
   if not, the next unit is capacity or optimization, not a larger run of the same recipe.
5. **No post-training stage is entered yet.** Instruction tuning, preference optimization and RLVR need a base
   that beats baselines on a whole-work held-out set and a documented instruction/evaluation dataset. Neither
   exists. ADR-0033's order — base, then adaptation, then instruction — is kept.
6. **The product interface is unchanged.** The Python library remains the primary interface; any new
   checkpoint stays loadable through `chakaso.models.load_model`, and a release is gated by ADR-0039.

## Why this target, against the alternatives

The criteria are the ones the roadmap names: available compute and data, legal provenance, measurability,
what can be learned from scratch, integration, context/inference cost, and extensibility without destroying
existing capabilities.

| Direction | Data available and licensable now | Measurable at 3.4M–9.8M parameters | Learns from scratch at this scale | Integration value | Verdict |
| --- | --- | --- | --- | --- | --- |
| **General base + technical mixture** | Public-domain Gutenberg prose, public-domain science, self-authored mathematics, permissively licensed code after review | Bits per character per domain; repetition suite; later grounded QA | General structure is the prerequisite for every domain; the corpus can be expanded without a new architecture | Broadest; the API is the same for any later specialization | **Chosen** |
| Science/mathematics-first | Public-domain classics are small; modern textbooks are CC BY or restricted; notation needs a tokenizer decision | Exact-answer sets can be built, but a 3.4M model is unlikely to pass them; loss is measurable | The model would spend capacity on a narrow register before learning general English | Valuable later; not first | Deferred |
| Code-first | Permissively licensed repositories exist, but a licence review and a code tokenizer are prerequisites; execution sandbox needed | Pass@1 is measurable, but requires a code corpus and a sandbox not yet built | Code is a different distribution; a model that has not learned English will not learn code from a small mixture | High long-term value; wrong first step | Deferred |
| Research-document model | This repository's own prose is usable; external research PDFs are often not licensed for training | The current corpus already measures this register; a narrower corpus would not answer a new question | It would overfit one author/style and not generalize | Narrow; the current model already writes repetitive prose about this project | Rejected |
| Base, then modular domain specialization | Same as general; domain data added in later stages | Domain held-out sets can be built per stage | This is the chosen path with explicit stages; it avoids confounds | Best extensibility | **The chosen path's later stages** |

**Why not a narrow domain first.** A science or code model at this scale would be a model trained on a small,
hard distribution before it has learned general language. Its losses would not distinguish "the domain is
hard" from "the model is undertrained", and the project's data for those domains is not yet licensed,
tokenized or evaluated. The general base is the only target whose data, tokenizer, evaluation and compute are
all available now.

**Why not architecture-first.** E2608's 0.2178-bit gap is confounded by 3.3× parameters and 2× context. E2611
is the controlled test. Even if an architecture effect survives, it is a smaller lever than a 10× data and
token increase at this scale, and changing both would make the result uninterpretable.

## Consequences

- The roadmap's Phase 3 (capacity-matched architecture) is being addressed by E2611; Phase 4's data line becomes
  the next unit; Phase 1 (make the curve turn) is addressed by the token-budget increase that the data
  experiment requires.
- The data curriculum in [`chakaso-data-curriculum.md`](../research/chakaso-data-curriculum.md) is the
  acquisition plan; no corpus beyond `gutenberg-english-prose-v2` is pinned yet, and this record does not
  claim one is.
- The next experiment must be preregistered with the mixture weights, token counts, whole-work splits,
  baselines and success bands before it runs. If the data experiment fails, this record is amended or
  superseded, not quietly ignored.
- The model card and release gate for `chakaso-base@0.1.0` are unchanged. A new checkpoint is a new release
  candidate at Level 1/2; Level 3 still requires behaviour, not a lower loss.

## Conditions to revisit

1. E2611 shows a material architecture effect at matched capacity and context, in either direction. The
   winning family becomes the architecture for the data experiment; the ordering (data before scale) is
   unchanged.
2. The equal-token data experiment shows no improvement over the current corpus at fixed architecture, seed,
   budget and split. The next unit becomes capacity or optimization, and this decision's target is reassessed.
3. A licence review makes a larger or more diverse corpus available (for example, a permitted web corpus or
   an open textbook collection). The mixture weights are revisited with a new record.
4. The compute ceiling changes (a GPU becomes available). The same ordering holds; the budget table in
   [`chakaso-compute-and-scaling-plan.md`](../research/chakaso-compute-and-scaling-plan.md) is re-measured,
   not assumed.
