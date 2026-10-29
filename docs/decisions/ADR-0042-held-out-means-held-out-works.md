# ADR-0042: a held-out score is held out by work, not by chunk

**Status:** Accepted
**Date:** 2026-10-30
**Related:** [ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md) (the corpus a run consumed),
[ADR-0037](ADR-0037-reference-model-sized-by-measured-throughput.md) (the reference model),
[ADR-0041](ADR-0041-next-training-target-general-base-with-technical-mixture.md) (the data experiment this
unblocks), E2607, E2608, E2611, K-022.

## Context

Every corpus built from books is a two-level object: a work is cut into chapter-sized documents of at most
16,384 characters (`chakaso.training.source_corpus.chunk_paragraphs`), and the split layer
(`chakaso.training.data.split_text_units`) assigns *documents* to train, validation and test. The split was
described as a document split because for the first corpus — this repository's own tracked files — a
document *is* a work.

For a corpus of books it is not. Measured on 2026-10-30 against the snapshot E2608 trained on
(`gutenberg-english-prose-v2`, 919 documents from 24 books, split seed 7, 15 % / 15 %):

| Quantity | Measured |
| --- | --- |
| Books represented in training | 24 |
| Books represented in the test side | 23 |
| Books on both sides | **23 of 23** |
| Test characters drawn from books also in training | **2,165,409 of 2,165,409 (100.0 %)** |

So E2608's 1.6865 bits per character and E2611's 1.8077 are *within-work* held-out numbers: the model was
scored on chapters of books whose other chapters it had read. E2607 has the same shape (3 books, 105
documents). E2605 and E2606 are unaffected in kind, because their corpus is a set of whole files.

What this does **not** invalidate is the comparison each record makes. Both arms of E2608 and both arms of
E2611 were split identically, so the *difference* between them is still a controlled measurement. What it
invalidates is the phrase "held out" as a claim about generalization to an unseen work, and any reading of
the absolute number as one.

The obvious fix — split the 24 books by work — is not available at that corpus size: 15 % of 24 books is
three or four test works, and a per-work estimate over three books is noise. The corpus has to grow before
its held-out number can mean what the record says it means.

## Decision

1. **A text unit may name the work it was cut from.** `TextUnit.group` carries it; `CorpusManifest.text_units()`
   fills it from the document's first path segment, which is the source slug for a corpus built from books
   and a repository directory for one built from tracked files. A document with no directory is its own work.
2. **The split can be a whole-work split.** `split_text_units(..., whole_works=True)` assigns every unit of
   a work to one side, using declared *character* targets rather than counting works, because works differ
   in length by an order of magnitude. Every unit must name a group, and fewer than three works is refused.
3. **The default stays the document split.** Recorded protocols must reproduce by name, and changing what
   `split_text_units` does by default would silently move the split under E2605–E2611. The new mode is opt-in
   per experiment (`ExperimentPlan.whole_work_splits`, `--whole-work-splits`).
4. **Every report states what its split did.** The corpus block carries `whole_work_splits`, `groups_per_side`
   and `works_on_multiple_sides`, so a reader does not have to trust a record's prose about whether "held out"
   meant a work. The count is reported, never enforced, because the historical records are what they are.
5. **A corpus built from books declares `whole_work_splits=True`.** The corpus that E2612 introduces does.
   A record that runs on a book corpus without it must say in its own text that its held-out set is
   within-work, in those words.
6. **The historical records are corrected with a dated note, not rewritten.** E2607, E2608 and E2611 keep
   their numbers and gain a correction naming the split unit, the measured overlap, and what survives
   (the comparisons) and what does not (the generalization claim).

## Why not the alternatives

| Alternative | Why not |
| --- | --- |
| Rewrite the historical records and re-run them with a work split | The 24-book corpus cannot support a work-level split; the re-run would measure a different, noisier thing and would replace a recorded number with a new one, which the experiment rules forbid |
| Make the whole-work split the only behaviour | It silently changes the split fingerprint of every recorded protocol, so a re-run of E2608 or E2611 would either refuse or measure a different held-out set under the same name |
| Group by content hash instead of by source | Two editions of one work hash differently and share a plot; content hashing catches verbatim copies, not the leakage this record is about |
| Infer the work from the title in the manifest | The manifest holds paths, not titles; inferring a grouping the corpus never declared is how a provenance claim becomes a guess |

## Consequences

- The split fingerprint of a whole-work run differs from a document run on the same corpus; that is the
  point, and it is what makes the two modes distinguishable in a report.
- Whole-work splits need many works, not many chunks. A corpus of 24 books yields three or four test works;
  a corpus of several hundred yields dozens. The data expansion is therefore a prerequisite for the
  evaluation, not only for the token budget.
- E2607's, E2608's and E2611's absolute bits-per-character figures carry the within-work qualifier from
  2026-10-30 onward. Their comparisons do not change, and no number in them is re-stated.
- `docs/training.md` and the experiment-record template describe the split mode as a field of the protocol.
