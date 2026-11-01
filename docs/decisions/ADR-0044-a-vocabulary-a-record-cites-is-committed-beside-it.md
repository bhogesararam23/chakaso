# ADR-0044: a vocabulary that a measured record cites, and that no command can refit, is committed beside the record

**Status:** Accepted
**Date:** 2026-11-01
**Closes:** the decision half of K-024(c) and ACTIVE_TASK T9.
**Related:** [ADR-0035](ADR-0035-transformer-checkpoint-format.md) (a checkpoint carries its tokenizer inline),
[ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md) (a recorded corpus is recoverable),
[`artifact-policy.md`](../internal/engineering/artifact-policy.md), E2612, E2613.

## Context

On 2026-11-01 the byte-pair merge loop was found to have chosen less frequent pairs than its documented rule
(K-024). The fix reproduces E2608's vocabulary exactly (`b597eb34f13d9058`), but E2612's vocabulary
(`434debe82f949347`) was fitted *by the defect*; the corrected loop yields `3abe4df4b123d4c5` on the same sample.
E2612 therefore survived only as a file in a git-ignored cache, which is exactly the load-bearing-cache condition
the artifact policy says a result must not rest on. A day later E2613 needed that vocabulary as a fixed input, and
the only way to supply it was to rely on the cache key still matching — which it did not, after an unrelated
change moved the split fingerprint the key is derived from.

The artifact policy's rule of thumb is "if it can be regenerated from something committed plus a recorded
environment, it is not committed." That rule is the reason this decision exists: **E2612's vocabulary can no longer
be regenerated from anything committed.**

## Decision

1. **A vocabulary is committed beside the record that cites it when both hold:** the record's measured result
   depends on that exact vocabulary, **and** the committed code cannot refit it (a vocabulary that is a pure
   function of committed inputs and fitting code that reproduces it is regenerated, not stored).
2. **It is committed as the file the loader reads**, named `experiments/<id>/vocabulary-<fingerprint>.json`. The
   format is the one `SubwordVocabulary.save` writes, which carries its own content checksum; loading re-checks it
   and the merge-table invariants, so a corrupted or edited file is refused.
3. **It is an input, not a cache.** A run uses it through `--tokenizer-file`, together with
   `--expected-tokenizer-fingerprint`, and fits and caches nothing. A test pins the committed bytes to the
   fingerprint, size, frequency floor and merge count the record reports.
4. **Size bound.** The file is a 4,096-identifier merge table of ~38 KB, under the repository's 1 MiB per-file
   bound with a factor of ~27 to spare. A vocabulary that exceeded the bound would not be committed; the record
   would instead have to state the fingerprint it cannot reproduce and how it knows, which is the other half of
   K-024(c).
5. **Licence.** A merge table is a list of byte-pair rules derived from the training text, not the text. It is
   committed under the repository's licence with the record, on the same reasoning as the per-document digests
   the corpus manifests already carry; this is stated, not assumed, because it is the one place a derived artifact
   from third-party text enters the tree. E2612's corpus is public-domain Project Gutenberg text.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| **Leave it in the cache and document the fingerprint** | A cache is keyed by an identity derived from other fingerprints; E2613 showed that an unrelated change can silently turn a hit into a refit |
| **Re-fit and re-record E2612** | A different vocabulary is a different record. It would answer a new question; it would not make E2612 reproducible |
| **Commit every vocabulary every record uses** | Those a committed fit reproduces exactly (E2608's) are regenerated; storing them would create a second source of truth that can drift from the code |
| **Store it in an object store** | There is no destination yet (artifact-policy: "Where do released weights live?"), and a 38 KB file does not justify one |
| **Embed the merges in the record's Markdown** | 3,839 rules in prose is unreadable and unchecked; the JSON carries its own checksum |

## Consequences

* `experiments/E2612/vocabulary-434debe82f949347.json` is added. It is byte-identical to the cache file the E2612
  record already cites (SHA-256 `d1061205f91289cc00cb873dbaff5f338dc97effe66b32b7bcfb48469ce2aed3`).
* The artifact policy gains the "committed" case above.
* E2613's first launch is the cautionary example; its valid arm reads the same vocabulary from the cache, which is
  byte-identical to the committed file, and the next arm on this corpus should pass `--tokenizer-file`.
* If the merge-loop rule changes again, this ADR does not change: committed files stay valid inputs because
  they are loaded, not refit.
