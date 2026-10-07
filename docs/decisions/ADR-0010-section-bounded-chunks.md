# ADR-0010: Chunks are section-bounded and do not overlap

- Status: Accepted
- Date: 2026-10-07
- Relates to: [ADR-0007](ADR-0007-content-derived-identifiers.md)

## Context

A chunk is the unit an answer cites. Its boundaries decide what a citation can point
at, what retrieval returns, and — because chunk identifiers are derived from chunk
text ([ADR-0007](ADR-0007-content-derived-identifiers.md)) — what an identifier means.

The common approach in retrieval systems is a fixed-size window with overlap: cut
every N tokens, start the next window a little before the end of the previous one, so
a fact spanning a boundary is not lost. It is popular because it is easy and because
it makes a specific failure less likely.

Chakaso derives identifiers from content, so overlap has a consequence it does not
have elsewhere: the same sentence appears in several chunks, under several
identifiers, and a retrieval result counting chunks will count that sentence more than
once.

## Problem

How should a document be split into evidence chunks?

## Options considered

- **Fixed-size windows with overlap.** Simple, format-agnostic, tolerant of facts that
  straddle a boundary.
- **Section-bounded packing without overlap.** Split at headings, then paragraphs, and
  pack paragraphs into chunks up to a target size.
- **Sentence windows.** One or a few sentences per chunk, regardless of structure.
- **Semantic chunking.** Use an embedding model to find topic shifts.

## Decision

Split at Markdown headings first, then into paragraphs, then pack consecutive
paragraphs into chunks up to a target character count. Chunks never cross a section
boundary and do not overlap. A paragraph too large to fit is split at sentence
boundaries; a single sentence too large even for that is cut.

Sizes are counted in characters, and the reason is stated in the code: there is no
tokenizer in this project, and a field named `token_count` holding a character count
would be a lie a later reader would trust.

## Reasoning

- **A citation has to be pointable.** A chunk that spans two headings has no position
  a reader can follow: they open the source and cannot tell which part is being
  claimed. Section boundaries are the only structure boundary the source actually
  declares.
- **Overlap would corrupt the project's own metrics.** Retrieval recall@k counts
  whether relevant evidence appears in the top k. If one sentence occupies four
  chunks, a retriever that finds it four times looks better than one that finds it
  once, and "how many chunks mention this" stops being a meaningful quantity. The
  project's measurements matter more than a robustness trick.
- **Overlap would blur identity.** The same text under several identifiers means an
  answer citing one of them is citing a *copy* of the evidence it actually used.
  Provenance is supposed to be exact.
- **Structure is free here.** The documents this project ingests are its own
  documentation, which is Markdown. Using the structure that exists costs one regular
  expression.
- **Determinism is required.** Evaluation compares runs, so chunking has to produce
  identical output for identical input. A character-based rule is exactly reproducible
  in a way that anything model-driven is not.

## Trade-offs

- **A fact split across a heading boundary may be unanswerable from a single chunk.**
  A sentence under "Deadlines" that only makes sense with a sentence under
  "Appeals" is now in two chunks and neither is sufficient alone. Accepted: the
  alternative trades a visible retrieval limit for an invisible duplication.
- **No overlap makes boundary-adjacent claims harder to support.** This is the failure
  overlap exists to reduce, and this record chooses to have it out in the open, where
  an evaluation can measure it, rather than hidden in duplicated evidence.
- **Character sizes do not correspond to a model's token budget.** A chunk of 800
  characters is some unknown number of tokens. Fitting a context window honestly
  requires a tokenizer, and the ceiling exists to bound growth in the meantime.
- **Markdown only.** HTML and PDF need a document processor, and there is none.
  Ingestion of anything else will need one, and that is a separate decision.
- **Sentence splitting is a heuristic.** An abbreviation followed by a space will
  split a sentence. It only applies to paragraphs too large to fit, where the
  alternative is cutting mid-word.

## Rejected alternatives

**Fixed-size windows with overlap** was rejected for the metric and identity reasons
above. The robustness it buys is real, and it is being declined deliberately: an
evaluation that measures a boundary failure is more useful than a system that hides
one.

**Sentence windows** were rejected because they fragment structure regardless of how
the document is organised, and because a large document becomes thousands of
tiny chunks, which makes an evidence pack a sample rather than a selection.

**Semantic chunking** was rejected because it requires an embedding model, which does
not exist here, and because chunk boundaries would then depend on a model version —
so re-chunking with an upgraded model would silently change what every identifier
refers to.

## Consequences

- Chunking settings are identity-affecting. Two settings produce disjoint chunk sets
  for the same document, which is correct and is asserted by a test.
- An evidence pack cannot contain the same text twice from one source, because one
  piece of text belongs to exactly one chunk. Deduplication is structural rather than
  a step that could be forgotten.
- A future overlapping mode supersedes this record rather than quietly adding a
  parameter, because the metric consequences need to be re-examined.
- The `token_count` field on an evidence chunk stays unset until a tokenizer exists.
  It is not zero, and it is not a character count.
