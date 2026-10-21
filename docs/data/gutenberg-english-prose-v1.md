# Data card: gutenberg-english-prose-v1 1.0.0

- **Catalogue:** `gutenberg-english-prose-v1`
- **Licence:** Public domain in the United States. Distributed by Project Gutenberg under the Project Gutenberg License, which permits copying, giving away and re-use; each source file states in its own preamble that users outside the United States must check local law before use.
- **Source:** Three Project Gutenberg plain-text ebooks, downloaded 2026-10-21 over TLS with certificate verification: pg11 'Alice's Adventures in Wonderland' (Lewis Carroll), pg84 'Frankenstein; or, the modern prometheus' (Mary Wollstonecraft Shelley), pg6130 'The Iliad' (Homer; the file's header names no translator, and none is claimed here). (retrieved 2026-10-21)
- **Documents:** 105 over 1,660,274 characters (1,687,349 bytes)
- **Character alphabet present:** 177 distinct symbols
- **Document length (characters):** min 5,757, p10 15,372, p25 15,758, p50 16,083, p75 16,239, p90 16,322, p100 16,382

## Composition

| Source | Documents | Characters |
| --- | ---: | ---: |
| `pg11` | 9 | 144,486 |
| `pg6130` | 69 | 1,096,668 |
| `pg84` | 27 | 419,120 |

## Duplication

- Exact duplicate document groups (same content digest): 0.
- Near-duplicate screening: 0 candidate pair(s) compared at 8-word shingles; threshold 0.5.
- Groups above the threshold: 0.

## Provenance

- Corpus snapshot checksum: `776afa22a2d54f49a2ff9d25e276c10b67fd362c802311e61ecabefeb3905040`

## What this corpus does not show

- `pg6130` supplies 66.1% of the characters, so a model trained on this corpus inherits that source's house style as much as the collection's.
- `pg84` supplies 25.2% of the characters, so a model trained on this corpus inherits that source's house style as much as the collection's.
- Continuous prose only: no instruction examples, no dialogue turns and no question-answer pairs, so nothing in this corpus can teach a model to answer anything.
- Split assignment belongs to an experiment, not to the corpus: two runs with different split seeds hold out different documents from the same bytes.
- Characters are counted after the manifest's line-ending normalization, and nothing else is normalized — no case folding, no diacritic removal, no deduplication across sources.
- This report was built from a snapshot alone, so it carries no catalogue: the provider, retrieval date and per-source urls are not stated.
