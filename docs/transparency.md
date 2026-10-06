# Transparency

Transparency in Chakaso means that the system's claims can be checked. It does not
mean that every answer is wrapped in a fixed template, and it does not mean
publishing the model's internal reasoning.

## Three epistemic categories

Every statement in an answer belongs to one of these, and the system tracks which:

| Category | Definition | Example |
| --- | --- | --- |
| **Observed evidence** | Information explicitly present in retrieved material or supplied by the user | "The paper reports 41.2 on the test split." |
| **Model knowledge** | Information produced from learned parameters without current retrieval | "That unit is named after its inventor." |
| **Inference** | A conclusion derived from evidence, model knowledge, or both | "Those two results are probably not comparable, because the splits differ." |

The distinction is architectural, not cosmetic. Each category is recorded
alongside the answer, so an evaluation can ask whether a claim was supported by
evidence the system actually had — rather than whether the answer sounded
confident.

Keeping the categories apart is also what makes correction possible. A claim
resting on observed evidence can be contradicted by better observed evidence. A
claim resting on inference has to be re-derived. Treating them as one kind of
statement makes "the answer changed" indistinguishable from "the answer was
wrong".

## What is exposed

- **Source provenance.** Which sources and chunks were used, with canonical URL,
  title, domain and retrieval time. This comes from the retrieval layer, never
  from generated text ([ADR-0003](decisions/ADR-0003-evidence-identifier-ownership.md)).
- **Support relationships.** Which evidence was treated as supporting which claim,
  where a validator established that.
- **Uncertainty, qualitatively.** Whether sources agree, conflict, or do not
  address the question.
- **Assumptions.** Where an answer depends on something the system assumed rather
  than established.
- **Limitations.** What the system could not check, or could not access.
- **Corrections.** When an answer replaced an earlier one, and why.

Presentation is decided per answer. A normal answer reads as a normal answer; a
"why this answer" section or expandable source cards appear when the answer
actually depends on evidence worth showing. A fixed
`Answer / Basis / Confidence / Sources` block is explicitly not the format,
because it manufactures the appearance of rigour and makes ordinary conversation
worse.

## What is withheld

- **Hidden chain-of-thought.** Internal reasoning traces are not a product
  feature. They are unfaithful to the computation that produced the answer,
  difficult to verify, and publishing them rewards plausible-sounding text over
  correct answers. What is exposed instead is evidence, assumptions, provenance,
  uncertainty and corrections.
- **Numeric confidence scores.** None are shown, because none have been
  calibrated. Until a calibration procedure exists and is measured, expressing
  uncertainty as a number would be a claim about the system that the system cannot
  support.

## Uncertainty language

Until calibration exists, uncertainty is expressed in terms of evidence:

| Situation | What is said |
| --- | --- |
| Retrieved sources agree and address the question | The sources agree; the answer follows from them |
| Retrieved sources conflict | The sources conflict, with what each says |
| Retrieval found nothing relevant | No relevant evidence was found; state what the answer rests on instead |
| Evidence is relevant but incomplete | What the evidence covers, and what it does not |
| The question rests on a false premise | The premise is challenged rather than answered |

These are behaviours to be implemented and then measured. In particular,
*appropriate abstention rate* — the fraction of insufficient-evidence cases in
which unsupported claims were avoided — is a metric, not a slogan
([`evaluation.md`](evaluation.md)).

## What transparency costs

- Storage. Keeping source text and hashes means keeping data that a
  generate-and-forget design would discard.
- Latency. Validation is a step between generation and presentation.
- Failure modes. A system that reports "no supporting evidence found" is less
  satisfying than one that always answers, and it will sometimes be wrong about
  having found nothing. It is still the better failure.
- Discipline. Every claim in the documentation must be checkable against code and
  measurements. Where it is not, the documentation says so.

## Status

| Behaviour | Status |
| --- | --- |
| Source provenance records | Implemented (data model) |
| Citation resolution from identifiers | Partly implemented |
| Qualitative uncertainty language | Planned |
| Assumption and limitation reporting | Planned |
| Correction notices | Planned |
| Calibrated confidence | Research — no calibration procedure exists |
| Claim-level support verification | Planned |
