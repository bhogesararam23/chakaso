# Data card: gutenberg-english-prose-v2 1.0.0

- **Catalogue:** `gutenberg-english-prose-v2`
- **Licence:** Public domain in the United States; distributed by Project Gutenberg under the Project Gutenberg License, which permits copying, giving away and re-use; users outside the United States must check local law before use
- **Source:** Project Gutenberg (retrieved 2026-10-22)
- **Retrieved:** 2026-10-22
- **Documents:** 919 over 14,490,005 characters (14,729,790 bytes)
- **Character alphabet present:** 146 distinct symbols
- **Document length (characters):** min 882, p10 15,215, p25 15,771, p50 16,117, p75 16,289, p90 16,355, p100 16,384

## Composition

| Source | Documents | Characters |
| --- | ---: | ---: |
| `pg11` | 9 | 144,486 |
| `pg120` | 23 | 364,042 |
| `pg1260` | 64 | 1,022,067 |
| `pg1342` | 46 | 727,348 |
| `pg1400` | 62 | 994,234 |
| `pg1661` | 35 | 562,012 |
| `pg174` | 27 | 429,167 |
| `pg1871` | 46 | 727,264 |
| `pg2000` | 134 | 2,110,030 |
| `pg214` | 10 | 157,600 |
| `pg25424` | 26 | 417,990 |
| `pg2591` | 33 | 520,360 |
| `pg2701` | 77 | 1,218,377 |
| `pg30422` | 8 | 126,973 |
| `pg32` | 19 | 296,792 |
| `pg3207` | 78 | 1,209,162 |
| `pg345` | 54 | 843,883 |
| `pg35` | 12 | 179,589 |
| `pg36` | 21 | 337,251 |
| `pg4395` | 11 | 161,296 |
| `pg5200` | 8 | 118,392 |
| `pg768` | 41 | 645,377 |
| `pg84` | 27 | 419,120 |
| `pg98` | 48 | 757,193 |

## Duplication

- Exact duplicate document groups (same content digest): 0.
- Near-duplicate screening: 2 candidate pair(s) compared at 8-word shingles; threshold 0.5.
- Groups above the threshold: 0.

## Sources, as declared

| Slug | Title | Bytes SHA-256 | URL |
| --- | --- | --- | --- |
| `pg11` | Alice's Adventures in Wonderland | `01b38ea4c710a84b…` | https://www.gutenberg.org/cache/epub/11/pg11.txt |
| `pg120` | Treasure Island | `5bc08275eacea864…` | https://www.gutenberg.org/cache/epub/120/pg120.txt |
| `pg1260` | Jane Eyre: An Autobiography | `13414dee2951c3ee…` | https://www.gutenberg.org/cache/epub/1260/pg1260.txt |
| `pg1342` | Pride and Prejudice | `3f6bb9d6f78e0293…` | https://www.gutenberg.org/cache/epub/1342/pg1342.txt |
| `pg1400` | Great Expectations | `9a637118af8e953e…` | https://www.gutenberg.org/cache/epub/1400/pg1400.txt |
| `pg1661` | The Adventures of Sherlock Holmes | `922e2a12ccb43a4c…` | https://www.gutenberg.org/cache/epub/1661/pg1661.txt |
| `pg174` | The Picture of Dorian Gray | `38f36b510417177a…` | https://www.gutenberg.org/cache/epub/174/pg174.txt |
| `pg1871` | The Deputy of Arcis | `f604cc291496ae4e…` | https://www.gutenberg.org/cache/epub/1871/pg1871.txt |
| `pg2000` | Don Quijote | `534f41d59f714216…` | https://www.gutenberg.org/cache/epub/2000/pg2000.txt |
| `pg214` | In the Days When the World Was Wide, and Other Verses | `dae08e7ad618b825…` | https://www.gutenberg.org/cache/epub/214/pg214.txt |
| `pg25424` | In the Mayor's Parlour | `af12d836f7086d74…` | https://www.gutenberg.org/cache/epub/25424/pg25424.txt |
| `pg2591` | Grimms' Fairy Tales | `377ebb8ca06cd8bc…` | https://www.gutenberg.org/cache/epub/2591/pg2591.txt |
| `pg2701` | Moby Dick; Or, The Whale | `b5e88b71a1cda068…` | https://www.gutenberg.org/cache/epub/2701/pg2701.txt |
| `pg30422` | The Internet and Languages [around the year 2000] | `74c4dcd20f616d2d…` | https://www.gutenberg.org/cache/epub/30422/pg30422.txt |
| `pg32` | Herland | `d2887a5e9cd99e09…` | https://www.gutenberg.org/cache/epub/32/pg32.txt |
| `pg3207` | Leviathan | `3de1e492641d9395…` | https://www.gutenberg.org/cache/epub/3207/pg3207.txt |
| `pg345` | Dracula | `96cd16eacdbfebae…` | https://www.gutenberg.org/cache/epub/345/pg345.txt |
| `pg35` | The Time Machine | `2892e919000e17c8…` | https://www.gutenberg.org/cache/epub/35/pg35.txt |
| `pg36` | The war of the worlds | `8417469e3ab66474…` | https://www.gutenberg.org/cache/epub/36/pg36.txt |
| `pg4395` | In Wicklow and West Kerry | `3fe6c6b045837cea…` | https://www.gutenberg.org/cache/epub/4395/pg4395.txt |
| `pg5200` | Metamorphosis | `0933283a19a4daae…` | https://www.gutenberg.org/cache/epub/5200/pg5200.txt |
| `pg768` | Wuthering Heights | `e533fe750589f042…` | https://www.gutenberg.org/cache/epub/768/pg768.txt |
| `pg84` | Frankenstein; or, the modern prometheus | `7810cd483cffcf2c…` | https://www.gutenberg.org/cache/epub/84/pg84.txt |
| `pg98` | A Tale of Two Cities | `d54c2b80d40a40b9…` | https://www.gutenberg.org/cache/epub/98/pg98.txt |

## Provenance

- Corpus snapshot checksum: `8eacb49f8d244e65db63f80796d76a026f532252899b8f85c7631ac3b9a2415f`

## How this corpus was prepared

```text
chakaso corpus prepare --catalog configs\corpora\gutenberg-english-prose-v2.toml --raw-dir data\corpora\gutenberg-v2 --cache-dir data\corpora\gutenberg-english-prose-v2 --snapshot data\experiments\E2608\gutenberg-english-prose-v2-snapshot.json
```

## What this corpus does not show

- Continuous prose only: no instruction examples, no dialogue turns and no question-answer pairs, so nothing in this corpus can teach a model to answer anything.
- Split assignment belongs to an experiment, not to the corpus: two runs with different split seeds hold out different documents from the same bytes.
- Characters are counted after the manifest's line-ending normalization, and nothing else is normalized — no case folding, no diacritic removal, no deduplication across sources.
