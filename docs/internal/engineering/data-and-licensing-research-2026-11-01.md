# Training data and licensing research, 2026-11-01

**Status:** Research. Nothing in this document is Implemented. It records what was **verified from primary
sources on 2026-11-01**, what was measured on this machine, and what could not be verified. No training run
was performed against any corpus described here, and no capability claim follows from any entry.

**Companions:** [`../../docs/research/chakaso-data-curriculum.md`](../../research/chakaso-data-curriculum.md),
[`../../../experiments/E2612/record.md`](../../../experiments/E2612/record.md),
[ADR-0033](../../decisions/ADR-0033-model-centred-product-contract.md),
[ADR-0036](../../decisions/ADR-0036-corpus-snapshot-recovery-from-history.md),
[ADR-0039](../../decisions/ADR-0039-three-level-model-release-gate.md),
[ADR-0042](../../decisions/ADR-0042-held-out-means-held-out-works.md).

## Three claims that must not be conflated

Every entry below separates three things, because the repository's rules (a source with no licence statement
is not admitted; "free to read" is not "free to reuse") depend on keeping them apart:

1. **The dataset exists** — the page resolved, the card was read.
2. **The dataset is licensed for this use** — the licence *as the source states it*, plus what that licence
   legally reaches. This is where the important asymmetry is: `ODC-By` and `CC0` on a web corpus licence the
   curator's **packaging of the database**, not the copyrighted text inside it. The ODC-By v1.0 text says so
   itself: "this license only governs the rights over the Database, and not the contents of the Database
   individually" ([opendatacommons.org/licenses/by/1-0/](https://opendatacommons.org/licenses/by/1-0/),
   read 2026-11-01). No web corpus found in this research closes that gap. Only public-domain *content* does.
3. **I could download it** — verified on 2026-11-01 by an anonymous HTTPS range read of a real shard, with
   row counts and character lengths measured from the bytes, not from a card.

A fourth column, **curation reported**, quotes what the curators say they did. It is not evidence that they
did it.

## The machine these costs are stated for

`D:` 200 GB total, 47 GB used, **154 GB free** (measured 2026-11-01 with `df`). 16.6 GB RAM, 18 logical CPUs,
no CUDA device, Python 3.13 with `pyarrow` 24.0.0 and `fsspec` 2026.6.0 installed. Training-side rates are
the measured E2612 smoke numbers: **4,493.7 positions/s**, peak RSS **1,710 MB**, preparation (tokenizer fit
on a 12M-character sample plus packing) **70.4 s**, one 2,400-step arm ≈ **73 minutes**.

Target slice size used throughout: **150,000,000 characters ≈ 50M subword identifiers** at the compression
the pinned corpus showed (149,000,499 characters → roughly 48M identifiers on the training side, E2612).

## The current corpus, and what the acquisition route can and cannot do (measured 2026-11-01)

| Fact | Value |
| --- | --- |
| Catalogue object | <https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz> |
| Catalogue size / last-modified | 5,597,116 bytes / Sun, 25 Oct 2026 21:41:56 GMT (HTTP headers) |
| Catalogue columns | `Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves` |
| Rows / English `Text` rows | 79,674 / 61,900 |
| English works by LoCC class | `P` 32,971 · `Q` excluding `QA` 2,151 · `QA` 127 · `T` 1,514 · `R` 624 |
| Bulk plain-text archive | `txt-files.tar.zip`, 11,322,987,679 bytes (≈11.3 GB), last-modified 25 Oct 2026 |
| `robots.txt` | single rule: `Disallow: /ebooks/search` |

Three consequences, all measured rather than assumed:

* **The catalogue route is stable and dated.** The CSV is served, gzipped, ~5.6 MB, refreshed within the last
  week, and there is a single-object full-text archive (11.3 GB) that replaces ~650 polite requests at
  2 s spacing with one download. `scripts/acquire_gutenberg_corpus.py` is therefore not blocked; it is slow.
* **The provenance of the catalogue input is not recorded.** E2612's reproduction block names a local
  `pg_catalog.csv` with no URL, no digest and no retrieval date, and the file is not in the repository. Under
  ADR-0036 that is a gap: the selection rule is reproducible only against a snapshot nobody pinned. Fixing
  this costs one line in the record and one digest.
* **Gutenberg cannot fix the age of the register, by construction.** `Issued` is the Project Gutenberg
  release date, not the text's date: 60,360 of the 61,900 English works were issued in 2000 or later, so that
  field carries no information about when the English was written. There is no field in the catalogue that
  does. Register *balancing* is available (the 430 pinned works sample a pool of 61,900); register
  *modernisation* is not.

**Licence as Project Gutenberg states it** (<https://www.gutenberg.org/policy/license.html>, read 2026-11-01):
"This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no
cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online at [www.gutenberg.org]. If you are not
located in the United States, you'll have to check the laws of the country where you are located before
using this ebook." The page separates books not restricted under U.S. copyright from copyrighted books, and
has a distinct section on the Project Gutenberg trademark. **Verdict: usable now** for the public-domain
majority, with the per-file statement retained verbatim in the catalogue as the curriculum requires; the
trademark paragraph is the reason a redistribution decision must stay recorded rather than assumed.

| Candidate | Source URL (verified 2026-11-01) | Licence as stated | Contents / registers | Access and measured size | Curation reported | Cost of a 150M-character slice | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Project Gutenberg (current)** | <https://www.gutenberg.org/policy/license.html>; feeds <https://www.gutenberg.org/cache/epub/feeds/> | Project Gutenberg License (quoted above); non-U.S. users "have to check the laws of the country where they are located" | prose, science, technology, medicine, mathematics. **No code. No modern English.** | `pg_catalog.csv.gz` 5.6 MB; full text `txt-files.tar.zip` 11.3 GB; 61,900 English works | Boilerplate delimited by the source's own markers; project does no filtering of its own beyond that | Already acquired (149,000,499 characters pinned). A rebalanced corpus: one 11.3 GB archive or ~650 polite requests, <1 GB of working disk | **Usable now**, but it cannot change the age of the register |
| **PG-19 (DeepMind)** | `deepmind/pg19` on Hugging Face | tag `apache-2.0` (a curator claim over public-domain text) | Gutenberg books "published before 1919"; train/valid/test by book | 10K–100K rows; not downloaded | "removal of boilerplate license text, and the mapping of offensive discriminatory words as specified by Ofcom to placeholder tokens" | n/a — same distribution Chakaso already holds | **Not useful**: strictly a subset of the current corpus's age problem |
| **Wikipedia dumps (en)** | <https://dumps.wikimedia.org/enwiki/latest/> | "Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply"; GFDL also listed ([license page](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_4.0_International_License), page dated 15 July 2025) | encyclopedic, all registers, modern English | `enwiki-latest-pages-articles.xml.bz2` = 25,800,511,704 bytes (dump dated 08-Oct-2026); also shipped as parts: `pages-articles1…27`, e.g. part 1 = 299,613,153 bytes, typical part ≈ 350–420 MB | none by Wikimedia for LLM purposes; dumps are raw wikitext with revision history | **One part ≈ 0.3 GB download**; XML→text extraction needs a streaming parser (Wikipedia-Extractor or a hand-written one) and CPU time not yet measured here. 150M characters is a few percent of one part | **Not usable under the current rule**: CC BY-SA share-alike is unresolved for released weights ([curriculum §Provenance](../../research/chakaso-data-curriculum.md) refuses it as a candidate until a written decision exists). It is the *legally* harder source, not the *operationally* harder one |
| **FineWeb** | [`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | "Open Data Commons Attribution License (ODC-By) v1.0… The use of this dataset is also subject to CommonCrawl's Terms of Use" | modern English web, all registers, no code as a register | >18.5T tokens (gpt2), 96 CC dumps 2013–Apr 2024; `sample-10BT` 27.6 GB, `sample-100BT` 277.4 GB, `sample-350BT` 388 GB | datatrove pipeline; per-dump processing; quality filters and dedup explored in their blogpost; **v1.3.0 (31-01-2025) "removed specific domains in response to a C&D notice"** ([notice](https://huggingface.co/datasets/huggingface-legal/takedown-notices/blob/main/2025/2025-01-22-Torstar.md)) | A `sample-10BT` shard is ~2.15 GB; 150M characters ≈ one tenth of a shard. Disk ≤ 3 GB | **Usable with work** for a research arm. ODC-By ⇒ attribution, no share-alike, so it passes the *packaging* test; content rights are untouched by it |
| **FineWeb-Edu** | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | tag `odc-by`; card: released under ODC-By v1.0 "and also subject to CommonCrawl's Terms of Use" | modern English web filtered for educational quality → science/technical prose present, code absent | 1.3T tokens; `sample-10BT` = 14 parquet files, **28,518,193,415 bytes measured**; shard `000_00000.parquet` = 2.15 GB, **726,000 rows, mean 4,218 characters → ≈3.06 billion characters per shard (measured)** | educational classifier (`HuggingFaceFW/fineweb-edu-classifier`) "using annotations generated by LLama3-70B-Instruct"; per-document `score`, `int_score`, `language`, `language_score`, `token_count`, `url`, `dump` fields retained | **One shard download (2.15 GB) then subsample**, or range-read a row group. 150M characters = 4.9 % of one shard. Preparation ≈ minutes at E2612's measured 70.4 s per 12M-character tokenizer sample | **Usable now for research, usable with a recorded attribution for release.** Caveat to record: the *filter* is teacher-model-guided (Llama-3-70B annotations), so "educational" is a distilled judgement, not a property of the text |
| **RefinedWeb (Falcon)** | [`tiiuae/falcon-refinedweb`](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | "released under an ODC-By 1.0 license"; "users should also abide to the CommonCrawl ToU" | modern English web | 100B<n<1T rows bucket; ~5T tokens per the paper; not downloaded here | heavy filtering and dedup, per the FineWeb card FineWeb was built to replicate | larger than needed; same slicing arithmetic as FineWeb | **Usable with work**, strictly dominated by FineWeb/FineWeb-Edu for this project (older crawl, same licence posture) |
| **Dolma / Dolma 3** | [`allenai/dolma`](https://huggingface.co/datasets/allenai/dolma), [`allenai/dolma3_mix-6T`](https://huggingface.co/datasets/allenai/dolma3_mix-6T), `allenai/dolma3.5_pool` | `odc-by`; card: "2024-04-15: License Change… to ODC-BY" and "**By using this dataset, you are also bound any license agreements and terms of use of the original data sources**"; Dolma 3: "It is intended for research and educational use" | labelled per-source components: Common Crawl 875.2B tokens, CC News 22.0B, peS2o 38.8B, arXiv 1.5B, Wikipedia/Wikibooks 6.2B, MegaWiki 3.2B, Flan 52.1B | whole corpus >1T tokens; components are individually addressable, so a component-level slice is possible | full Dolma pipeline documented (quality filtering, global dedup); tags each document with its source and filter decisions | Download **one component** (e.g. arXiv 1.5B tokens, or CC News 22B) — 150M characters is well under 1 % of any of them. Disk a few GB | **Usable with work — and the best-documented multi-source corpus of the ones tested.** But the CC BY-SA Wikipedia component and the Springer-licensed peS2o component must be *excluded by prefix*, not inherited |
| **The Pile** | [`EleutherAI/pile`](https://huggingface.co/datasets/EleutherAI/pile) | tag `license: other`; the card's Licensing section names exactly one subset licence ("PubMed Central: MIT License") | 22 components incl. **Books3** and **BookCorpus2** (no evidence of author consent in either curator statement), GitHub, ArXiv, PG-19, DM Mathematics, FreeLaw, USPTO | 100B<n<1T bucket; README still points to `https://the-eye.eu/public/AI/pile/` for the bytes | none re-stated on the card beyond the original paper | n/a | **Not usable.** Per-component licence silence plus two components that are not cleared. **Could not verify** the widely reported 2024 withdrawal: [pile.eleuther.ai](https://pile.eleuther.ai/) and the GitHub README, both read 2026-11-01, carry no retirement notice, and the download link is the one that went offline. Do not write that it was withdrawn; write that its licence basis is not stated and its canonical URL is dead |
| **CC-News** | — | **could not verify** | — | — | — | — | **Unverifiable at the source.** `github.com/google-research-datasets/cc_news` returns 404 from the GitHub API and a repository search on 2026-11-01 found no matching repo. The only *verifiable* route to that data is Dolma's CC News component (22.0B tokens, ODC-By plus original-source terms), which is where it should be taken from if wanted |
| **The Stack v2 / StarCoder data** | [`bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2) | tag `license: other`. Raw card fetch returned: "**Access to dataset bigcode/the-stack-v2 is restricted. You must have access to it and be authenticated to access it. Please log in.**" API reports `gated=auto` | permissively filtered source code in many languages | not downloadable anonymously (verified 2026-11-01) | the paper's governance plan is real: "3.1 TB dataset consisting of permissively licensed source code in 30 programming languages", a "permissively licensed subset", an opt-out ("Am I in The Stack") and a removal process (arXiv:2211.15533) | n/a — access first | **Not usable now, and usable-with-work only after an account, an agreement, and a written decision.** Answering the question asked: a permissive *filter* is a curation rule the curator applied, not a licence it holds; BigCode itself frames it as governance with takedown obligations. [ADR-0039](../../decisions/ADR-0039-three-level-model-release-gate.md) release of weights trained on it needs a licence decision the repository does not have. **Code stays out of the mixture** — E2612 already declares this limit |
| **OpenWebMath** | [`open-web-math/open-web-math`](https://huggingface.co/datasets/open-web-math/open-web-math) | "OpenWebMath is made available under an ODC-By 1.0 license; users should also abide by the CommonCrawl ToU… **We do not alter the license of any of the underlying data**" | mathematics, physics, statistics, CS — the only candidate here with real math | 6.3M documents, 14.7B tokens | filtered from >200B CC HTML files; math-token heuristic | one shard-level slice is far more than 150M characters; disk ≈ GB | **Usable with work for the mathematics register — with a named warning.** The card's own top-domain table is 9.55 % stackexchange.com (CC BY-SA) and 3.14 % nature.com (publisher-owned), 1.27 % wikipedia.org. The curators state plainly that they do not alter those terms. So OpenWebMath is a *mixed-licence* corpus that inherits the share-alike problem in ~11 % of its characters |
| **ProofPile / proof-pile** | [`hoskinson-center/proof-pile`](https://huggingface.co/datasets/hoskinson-center/proof-pile), [`EleutherAI/proof-pile-2`](https://huggingface.co/datasets/EleutherAI/proof-pile-2), `nampdn-ai/mini-proofpile` | proof-pile tag `apache-2.0`; proof-pile-2 and mini-proofpile carry **no licence tag** (`mini-proofpile` is `gated=auto`) | informal LaTeX mathematics + Lean 3, Isabelle, Coq, HOL Light, Metamath, Mizar | arXiv.math subset described as "10GB"; not downloaded | heuristic file selection inside arXiv source folders; language filter; a noise analysis notebook | n/a | **Not usable.** The Apache tag sits over an arXiv subset, and arXiv full text is licensed per author (below). **Could not verify** a "ProofPile-A/B" pair as released dataset objects: the author searches returned `hoskinson-center/proof-pile`, `EleutherAI/proof-pile-2` and gated successors, nothing named ProofPile-A or ProofPile-B |
| **arXiv (full text and source dumps)** | [arXiv licence help](https://info.arxiv.org/help/license/index.html), [metadata](https://huggingface.co/api/datasets/librarian-bots/arxiv-metadata-snapshot) | authors choose the licence on deposit — CC BY 4.0, CC BY-SA 4.0, CC BY-NC-SA 4.0, CC BY-NC-ND 4.0, the arXiv perpetual non-exclusive licence, or CC0 — and arXiv states this "limits re-use"; separately "**a Creative Commons CC0 1.0 Universal Public Domain Dedication will apply to all metadata**" | modern scientific text, LaTeX, code fragments | metadata snapshots exist and are CC0-tagged (`librarian-bots/arxiv-metadata-snapshot`, updated 2026-10-27) | arXiv curates the licence list; no LLM curation | abstracts/metadata: trivially cheap. Full source: not obtainable through a route verified here | **Full text: not usable.** Per-article licences are heterogeneous and mostly non-commercial or no-derivatives, and the project's rule ("a source whose licence is 'free to read' but not 'free to reuse' is not admitted") excludes the arXiv default licence. **Metadata and abstracts (CC0): usable now** — that is a legitimate, licence-clean source of modern scientific *title/abstract* English, at a scale far below 150M characters |
| **DeepMind Mathematics Dataset** | [github.com/google-deepmind/mathematics_dataset](https://github.com/google-deepmind/mathematics_dataset) | GitHub `license.spdx_id` = **Apache-2.0** (this is the repository licence); README describes pre-generated data in a public GCS bucket (`console.cloud.google.com/storage/browser/mathematics-dataset`) | template-generated school-level Q&A: algebra, calculus, probability, number theory; **no prose** | "2 million (question, answer) pairs per module"; version 1.0 released with the paper | generator-based: questions are sampled from templates, so contamination risk is different from a scrape | a few modules is a few hundred MB; 150M characters is available but would be ~150M characters of *the same template* | **Usable now, with an honest limit noted.** The text is owned by the generator's licence, so it is clean; but it is synthetic arithmetic strings, which teaches symbol patterns and nothing about register. It is Stage-3 material only, and its arXiv-linked module (arXiv:1904.01557, "Mathematics as a Second Language", cited by the README) inherits the arXiv problem above |
| **NuminaMath** | [`project-numina/NuminaMath-CoT`](https://huggingface.co/datasets/project-numina/NuminaMath-CoT), `NuminaMath-TIR` | "The dataset is available under the Apache License, Version 2.0" | ~860k math problems with chain-of-thought solutions | not downloaded | card states the provenance plainly: "collected from online exam paper PDFs and mathematics discussion forums", then OCR, segmentation, translation, re-alignment | n/a | **Not usable as a licence basis.** The card documents that the content is OCR'd exam papers and forum posts; an Apache tag on the packaging does not clear a third party's exam paper. This is the clearest illustration in this research of why "curator tag" ≠ "licence", and the reason [the curriculum](../../research/chakaso-data-curriculum.md) requires the *source's own* statement. **MathMC / CommonMath: could not verify** — no dataset object found on Hugging Face or GitHub in the searches run (2026-11-01); they are paper-side splits of the Numina/Proof-Assistant line, and would need a primary page before being listed as data |
| **TinyStories** | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) | tag `cdla-sharing-1.0` — a **reciprocal** data licence (CDLA Sharing: derivatives must be shared back) | synthetic short English stories from a restricted vocabulary; modern simple prose | files are single large `.txt` (the card names `TinyStories-train.txt`, `TinyStories-valid.txt`, `TinyStoriesV2-GPT4-train.txt`); 150M characters is a fraction of the training file | "synthetically generated (by GPT-3.5 and GPT-4) short stories"; the paper is arXiv:2305.07759 | one download, a few hundred MB, no parquet handling | **Usable with work and two recorded reservations.** (i) CDLA-Sharing is reciprocal, so it belongs in the same drawer as CC BY-SA until a written decision exists; (ii) it is teacher-model output, which [ADR-0033](../../decisions/ADR-0033-model-centred-product-contract.md) treats as a separate, labelled route rather than the product. Its *register* is exactly the simple-modern-English gap, so it is worth keeping in view as a Level-2-irrelevant research arm only |
| **Cosmopedia / SmolLM-Corpus** | [`HuggingFaceTB/cosmopedia`](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia), [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | cosmopedia tag `apache-2.0`; smollm-corpus tag `odc-by` | synthetic textbooks/posts/table-exercises, plus deduplicated FineWeb and FineMath subsets | not downloaded | model-generated from prompts over web and textbook seeds; the FineWeb-Edu card notes its deduplicated re-release in SmolLM-Corpus | per-config slicing | **Usable with work.** Same teacher-provenance caveat as TinyStories, and the two licence tags (Apache over a corpus that contains ODC-By web data) are a conflict to record rather than resolve here |
| **HPLT 2.0 cleaned** | [`HPLT/HPLT2.0_cleaned`](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | "We license the **actual packaging** of these text data under the Creative Commons CC0 license ('no rights reserved')", plus: "**It is your responsibility that any use of the data complies with any applicable legal framework, such as… the EU Copyright Directive 2019/790 and the General Data Protection Regulation 2018**" | modern English web, 193 languages; monolingual portion 8T tokens (ACL 2025, Burchell et al., doi `10.18653/v1/2025.acl-long.854`) | **measured twice, disagreeing — see V-20/V-21/V-22 of the [verification ledger](research-verification-ledger-2026-11-01.md)**: configs `eng_Latn_1…eng_Latn_160`; `eng_Latn_1/train-00000-of-00356.parquet` = 239 MB, **104,293 rows**, **1,000-document mean 4,122.9 characters → ≈430 million characters per shard** (an earlier single-document read of 8,853 characters projected to ≈0.9 billion and was wrong). Anonymous HTTPS range read succeeded, no token, no agreement | per-document columns: `f` WARC path, `o`/`s`/`rs` offsets and sizes, `u` **full URL**, `c` MIME content-type, `ts` crawl timestamp, `collection`, `lang`/`prob` per-language list + probability, `robotstxt` (`allowed`), `filter` (`keep`), `pii`, `doc_scores` (nine 0–10 quality scores), `id` — the filter decisions, the robots status and a usable quality signal ship with the data | **One shard, 239 MB, is ~2.9× the target slice.** Working disk < 1 GB; no XML parsing, no extraction pipeline, no GPU, no account | **Usable now for a research arm — but its register is unproven and must be measured before a run.** See the verification section: this shard is ArchiveTeam `archivebot` blog/feed text from a 2020-08-12 crawl, not technical prose |
| **FinePhrase** | [`HuggingFaceFW/finephrase`](https://huggingface.co/datasets/HuggingFaceFW/finephrase) (last modified 2026-03-31) | tag `odc-by`; card lists source dataset `HuggingFaceFW/fineweb-edu/sample-350BT` and annotators `machine-generated` | modern English rewritten into four registers: **faq, math, table, tutorial** | **measured**: `faq/000_00000_0.parquet` = 270 MB, **67,000 rows, mean 4,938 characters → ≈331 million characters per shard**; anonymous range read 14.7 s. Card totals: 1,354,044,711 samples, 486,367,076,933 completion tokens | generator documented precisely: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `temperature=1.0`, `top_p=1.0`, `top_k=50`, `max_tokens=2048`, `model_max_context=8192`, suffix speculative decoding with 32 tokens, four prompts quoted in full on the card | **One shard = 2.2× the target slice, 270 MB.** Cheapest modern-register acquisition found | **Usable with work, and it is a distillation.** This is a 2026 release of a *teacher model's rewrites* of modern web text. It would give the corpus exactly the structured modern registers E2612 lacks (FAQ/tutorial/table/math), and its licence is the same ODC-By posture as FineWeb. But per ADR-0033 it must be labelled teacher-generated, and a model trained on it measures the mixture, not Chakaso's own reach. Honest use: a separate arm, never merged silently into the base corpus |
| **Nemotron-CC / -CC-v2** | [`nvidia/Nemotron-CC-v2`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | gated. Card: "This repository is publicly accessible, but you have to accept the conditions to access its files and content"; "By clicking 'Agree' I confirm… I intend to use this data for **model training purposes only**"; and, on the synthetic parts: "If this dataset is used to create, train, fine-tune, or otherwise improve an AI model, which is distributed… such AI model **may be subject to redistribution and use requirements in the Qwen License Agreement and the DeepSeek License Agreement**" | 6,585.8B tokens total: English CC 3,359.8B, English Synthetic CC 1,257.3B, Diverse QA 692.9B, Translated Diverse QA 558.2B, Math 206.2B, Math SFT 190.6B, Synthetic Code 174.9B, MMLU SFT 81.6B, Code SFT 58.5B, General SFT 5.7B | raw card fetch returned HTTP 401 anonymously (verified); `gated=manual` in the API | "filtered for English and globally deduplicated"; code subset: "license-based removal (BigCode-inspired, with a stricter license set), exact and fuzzy deduplication, heuristic quality filters" | n/a — a manual approval gate is not a slice | **Not usable.** Two independent reasons: a manual agreement that binds the *weights* to teacher-model licences (Qwen/DeepSeek), and a training-only use restriction. Under [ADR-0039](../../decisions/ADR-0039-three-level-model-release-gate.md) a released `chakaso-base` cannot carry that obligation, and under ADR-0033 teacher-model data is not the product |
| **Ultra-FineWeb** | [`openbmb/Ultra-FineWeb`](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | tag `apache-2.0` | ~1T English + 120B Chinese tokens filtered from FineWeb; `Ultra-FineWeb-L3` adds "Q&A Pair Generation and Multi-style Rewriting" (400B+ English) | not downloaded | "efficient verification-based high-quality filtering pipeline"; L1 = cleaning + heuristic filters + sensitive-field replacement + dedup | slicing arithmetic as FineWeb | **Usable with work — and a licence conflict to record, not resolve:** an Apache tag over data derived from an ODC-By web corpus whose content rights were never cleared. Same content-side objection as FineWeb, with an extra inherited-terms question |
| **Zyda / Zyda-2** | [`Zyphra/Zyda-2`](https://huggingface.co/datasets/Zyphra/Zyda-2) | "released under the terms of ODC-BY… you are also bound by any license agreements and terms of use of the original data sources" | curated web + synthetic + code + math mixes (per the card's component list) | not downloaded | re-filtered and re-deduplicated Common Search | component-level slicing possible | **Usable with work**, identical posture to Dolma and no better documented for Chakaso's registers than FineWeb-Edu |
| **US Public-Domain Newspapers** | [`iknowyoursecret/US-PD-Newspapers`](https://huggingface.co/datasets/iknowyoursecret/US-PD-Newspapers) | tag `cc0-1.0`; card: "The composition of the dataset adheres to the US criteria for public domain (any publication without a copyright removal). In agreement with the shorter term rules, the dataset is in the public domain for all countries with a Berne author-right model", quoting the Library of Congress's own "we generally do not own the rights to materials in our collections… you should determine for yourself"; and, stated for a purpose this project shares: "**The text can be used for model training and republished without restriction for reproducibility purposes**" | **1690–1963 American newspaper text**: reportage, editorial, advertisement, legal notice, serialized fiction. Card figures: 21 million unique editions, 98,742,987,471 words. Language caveat on the card: "most of the collection is in English", with 600k German and 400k Spanish editions | **measured**: 1,000 parquet files listed (2,618 original dump files exist), **151.5 GB** across the listed objects, largest 626 MB (`dlc_nosebleed_ver01.parquet`). Compiled by Pierre-Carl Langlais from the Library of Congress [Chronicling America OCR dumps](https://chroniclingamerica.loc.gov/data/ocr/); each parquet carries "the full text of a few thousand selected at random and a few core metadatas (edition id, date, word counts…)" | composition and provenance documented; **no LLM filtering claimed**, and the card is candid about it: OCR errors, headers/page counts and multi-column layout are "unlikely to be well formatted" and correction is future work | **One ~150–630 MB parquet is within the target slice several times over.** English filter + date filter + length floor applied locally; no account, no agreement, no pipeline beyond reading a column; disk ≪ 154 GB | **Usable now — and the only large source in this research whose licence reaches the *text*, not the packaging.** Its English is 1690–1963 journalism, so it does not deliver 2020s technical prose; it delivers a *later register* and, with the `date` metadata, a **time axis** the current corpus cannot have |
| **American Stories** | [`dell-research-harvard/AmericanStories`](https://huggingface.co/datasets/dell-research-harvard/AmericanStories) (last modified 2025-03-26) | tag **`cc-by-4.0`**; card Licensing Information: "The dataset has a CC-BY 4.0 license"; and "This dataset is drawn entirely from image scans in the public domain that are freely available from the Library of Congress" | "nearly 20 million scans from the public domain Chronicling America collection", rebuilt as **full article texts** rather than page-level OCR dumps | **measured**: 184 top-level objects, **340.38 GB total**, per-year tarballs of ~6.9–7.6 GB (e.g. `faro_1910.tar.gz` 7.6 GB) — the tarballs are scan-plus-text, so text extraction is a real step | a deep-learning pipeline: "layout detection, legibility classification, custom OCR, and the association of article texts spanning multiple bounding boxes", mobile-efficient architectures for scalability; article structure enables transformer-based reproduced-content detection | **One year-tarball (~7 GB) download, then unpack the article text and slice.** Feasible on this disk; extraction cost is unmeasured here, and it is the difference between this row and the row above | **Usable with work.** CC BY-4.0 is attribution-only — no share-alike, no non-commercial clause — over content that is public domain, so it clears the same bar as the newspapers with better text quality. Prefer it over `US-PD-Newspapers` if OCR quality matters and the extra 7 GB of unpacking is acceptable; prefer `US-PD-Newspapers` if the smallest possible acquisition matters |
| **Caselaw Access Project (HF mirror)** | [`endomorphosis/Caselaw_Access_Project_JSON`](https://huggingface.co/datasets/endomorphosis/Caselaw_Access_Project_JSON) | "The Caselaw Access Project dataset is licensed under the CC0 License" | US case law — formal modern legal English | **measured: the repo contains 942 files totalling 14 MB** — a sample, not the corpus | none read | would need the real CAP dump, whose distribution was not verified here | **Unverifiable as a corpus at this URL.** The licence statement is CC0 and US court opinions are the strongest public-domain content class available, but what sits at this address is a fragment; recording it as "available" would be inventing an acquisition route |
| **Open Math Courses** | [`KokunoYumeto/open-math-courses`](https://huggingface.co/datasets/KokunoYumeto/open-math-courses) | tag `cc0-1.0`; card: "All text in this dataset is marked CC0 1.0 (public domain)", with a per-row `license` column and an `EXCLUDED.jsonl` for parts under other licences | self-published mathematics lessons; a real, small, licence-clean modern math register | root listing measured: `.gitattributes` + `EXCLUDED.jsonl` only — **data size not measured** | self-authored, per-row licence field, exclusions published | unknown until the config is read | **Usable with work, size unknown.** Structurally the most Chakaso-shaped source in this list: a small, self-authored, per-item-licensed corpus. Worth one follow-up read before it is counted |
| **openwebtext (CC0 re-release)** | [`Skylion007/openwebtext`](https://huggingface.co/datasets/Skylion007/openwebtext) | tag `cc0-1.0` | the OpenWebText scrape | not downloaded | not read | — | **Not usable as a licence basis.** A CC0 tag on a re-release of somebody else's scrape dedicates a right the releaser does not hold. Included here because its tag is exactly the failure mode the curriculum's rule is written against |

## Contamination and evaluation hygiene: what a legitimate instrument would look like

The project's rule is that harness measurement is never evidence of language-model competence. So the
question is not "which benchmark should Chakaso score" but "what instrument could produce a reading a
reader may trust at 3.4M parameters". Six properties, each tied to something verified above or already
practised in this repository.

**1. The instrument must be able to detect the thing the experiment changed.** E2612 could detect a
register effect because both arms were scored on held-out works of every register, including the 0.27584-bit
technical gain and the 0.03026-bit prose cost. It could *not* detect the archaic-vs-modern shift, because no
side of the split is modern. That is the single strongest argument for a modern held-out partition, and it is
a property of the instrument, not of the model.

**2. Held out means held out by work.** [ADR-0042](../../decisions/ADR-0042-held-out-means-held-out-works.md)
and the measured E2612 split (312/56/62 works, zero on more than one side) already implement this. For a web
corpus the same rule must be applied **by domain or URL host**, not by document: a scraped corpus contains
many pages from one site, and document-level splitting of web text re-creates the two-level leak K-023
recorded for books. HPLT, FineWeb and FineWeb-Edu all ship a `url`/`file_path` column (measured), so
group-by-domain is available rather than invented.

**3. Report overlap, do not claim decontamination.** Exact-digest plus n-gram overlap between the new
corpus and every evaluation side, recorded as counts and as examined pairs in the report — the curriculum's
quality-filter list already demands this, and E2612's near-duplicate rule (8-word shingles) is the precedent.
The published detection methods are known to be brittle: watermarking-based detection (arXiv:2502.17259,
2025-02-24), paired-confidence testing (arXiv:2406.18326, 2024-06-26), the temporal-signal argument
(arXiv:2509.00072, 2025-08-26) and an explicit fragility study for contamination detection in reasoning
models (arXiv:2510.02386, 2025-09-30) — all read on arXiv's own API on 2026-11-01. A project this size should
not be in the business of running detectors; it should be in the business of knowing what it trained on.

**4. Every score needs a floor and a band.** Uniform and unigram baselines refitted per arm (E2612 does
this), plus the task's chance level, plus the 0.02 bits-per-character materiality band, plus declared seeds.
On multiple-choice instruments the floor is the whole story: HellaSwag is 4-way (25 %), BLiMP is a 2-way
minimal pair (50 %), and a 3,413,760-parameter model trained on archaic prose will sit at or within noise of
those floors. A floor score is not a measurement of a weakness in the model; it is a statement that the
instrument has no dynamic range at this size. That is why the honest instruments here are loss-per-character
on held-out works and per-paradigm deltas, not accuracies.

**5. Seeds before spread.** E2606 reproduced E2605 and put it on a spread; a single-draw difference outside
the band is still a single draw (E2612 declares this). Any data comparison should be preregistered with the
seed list, and a reproduction is a new record, never an edit
([ADR-0036](../../decisions/ADR-0036-corpus-snapshot-recovery-from-history.md)).

**6. Instrument licences are a real constraint on evaluation too.**

| Instrument | Primary source read 2026-11-01 | Licence as stated | What that means here |
| --- | --- | --- | --- |
| **LAMBADA** | arXiv:1606.06031 ("The LAMBADA dataset: Word prediction requiring a broad discourse context", 2016-06-20); [`cimec/lambada`](https://huggingface.co/datasets/cimec/lambada); [`EleutherAI/lambada_openai`](https://huggingface.co/datasets/EleutherAI/lambada_openai) | `cimec/lambada`: "The dataset is released under the CC BY 4.0… license"; card's Source Data: "The passages are extracted from **novels from Book Corpus**". `EleutherAI/lambada_openai`: tag `mit`, card "License: **Modified MIT** (openai/gpt-2)" | Word-final-clause perplexity on contemporary novel prose — the right shape for a 3.4M model, and it is loss-based, so no chance floor problem. But a CC-BY tag over third-party novels is the packaging-not-content fallacy again, and the "Modified MIT" route carries the GPT-2 non-commercial clause. **Usable as a locally-scored diagnostic; not redistributable, and it is BookCorpus-derived, which circulates in web scrapes — i.e. it is the contaminated-by-default class.** If used, it must be excluded from training by exact and n-gram overlap and its overlap result recorded |
| **HellaSwag** | [`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag); `github.com/rowanz/hellaswag` | card: "MIT https://github.com/rowanz/hellaswag/blob/master/LICENSE"; rows carry `source_id` values of the form `activitynet~v_…`; **the GitHub repository is DMCA-blocked** — API response on 2026-11-01: `{"message":"Repository access blocked","block":{"reason":"dmca","created_at":"2026-09-14T16:23:18Z","html_url":"https://github.com/github/dmca/blob/master/2026/09-14-wikihow.md"}}` | I could not read the LICENSE file the card points at, because the repository it lives in is not served. So the only licence claim for the instrument is a link to a page that returns nothing. Combined with a 25 % floor and video-caption provenance, **this is not an instrument Chakaso should build a claim on**. It is a good example to document, not to use |
| **BLiMP** | [`nyu-mll/blimp`](https://huggingface.co/datasets/nyu-mll/blimp) | tag `cc-by-4.0`; card: "BLiMP is distributed under a CC-BY license. Source: github.com/alexwarstadt/blimp#license"; 10K–100K rows; frontmatter `annotations_creators: crowdsourced`, `language_creators: machine-generated` | Minimal-pair acceptability across named paradigms (the config list includes NPI-licensor paradigms), 50 % chance floor. Shape is right for a small model **as a per-paradigm diagnostic of where the grammar signal lives**, and CC-BY is a real licence. Two cautions to preregister: many items originate in published linguistics example sentences, and parts of the paradigm content is template-generated, so a paradigm-level result can reflect generator overlap rather than competence |
| **WikiText-103** | [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) | "The dataset is available under the Creative Commons Attribution-ShareAlike License (CC BY-SA 4.0)" | Loss-based, clean to run, and modern-English. Blocked for *training* by the same share-alike decision Wikipedia dumps are blocked by; usable as an **eval-only side** if the split never enters a mixture — a distinction worth writing into the record explicitly, because the current rule speaks of admission into corpora, not of scoring |
| **GTDB** (Grounded Technologies Attribution Dataset, GEM) | — | **could not verify.** The Hugging Face search for `gtdb gem benchmark` returned only genomics corpora named GTDB (`AllTheBacteria/GTDB`, CC BY-SA 4.0), and `github.com/GEM-benchmark/NLGs` returned no repository object on 2026-11-01 | Do not list it as available. If wanted, it needs its primary source located first |
| **ChildCraft / Data-Dogey** (developmental English) | — | **could not verify either as a released, licensed dataset.** Searches run 2026-11-01: Hugging Face `childcraft` and `data-dogey` returned zero datasets; the nearest verified literature is "Is Child-Directed Speech Effective Training Data for Language Models?" (arXiv:2408.03617; EMNLP 2024 `2024.emnlp-main.1231`) with its code/data repository `styfeng/TinyDialogues`, whose **data licence I did not verify** | The *question* these corpora ask — can a small model learn from a small, natural, simple corpus — is exactly Chakaso's. The answer here is not to chase an unverifiable corpus but to build the equivalent from what the project owns |
| **BabyLM corpora** | `babyllm` Hugging Face author query returned **no datasets**; per-user 2026 mirrors exist (`openhonest/babylm-2026-fr-corpus`, `miguelcsx/babylm-2026-mosaic-bpe-corpus` tagged `other`, `drooryck/babylm-macaroni-corpus` tagged `mit`) | not verifiable at source | The official challenge corpora were not located at a primary address in the searches run. Not listed as available |

**The instrument that is actually available is the one already built.** E2612's held-out-works, per-register
bits-per-character scoring, with uniform/unigram baselines, a preregistered materiality band and a
whole-work split, is a legitimate instrument: it detects a data effect, it has floors, and it cannot be
contaminated by a benchmark because it is not a benchmark. What it lacks is a **later partition** — text whose
date is known and whose grammar is not 19th-century literary English. Adding one (held-out era buckets from a
public-domain press corpus, plus self-authored modern technical documents held out of training entirely) is
the smallest change that makes the archaic/transfer question measurable. Self-authored text has no licence
question at all, which is why it is the right shape for a project that releases weights, and the era buckets
have a date field to hang the claim on — which no amount of register rebalancing inside Gutenberg can supply,
because the catalogue carries no field that dates the language.

## Recommendation

**Acquire `us-pd-press-v1`: 150,000,000 characters sliced from
[`iknowyoursecret/US-PD-Newspapers`](https://huggingface.co/datasets/iknowyoursecret/US-PD-Newspapers) — CC0
over text the card states is public domain, spanning 1690–1963, with the edition `date` retained as a field —
and run it as a matched-token, matched-model comparison against the pinned `gutenberg-multiregister-v1`,
scored on a **held-out-era partition** so that the archaic penalty E2612 could only disclaim becomes a
measured number. Add a modern-English web slice (`HPLT2.0_cleaned`, or `fineweb-edu sample-10BT`) as a
**research-only arm, explicitly not releasable**. Do not acquire code; do not acquire Nemotron or The Stack;
do not treat any packaging-licensed web corpus as a release basis.

Why this is the smallest honest upgrade that changes what Chakaso learns:

* **It is the only upgrade whose licence reaches the text, not just the packaging.** Gutenberg (public domain
  under the Project Gutenberg License), the newspapers (CC0, "All the materials are now part of the public
  domain and have no intellectual property rights remaining", and explicitly "The text can be used for model
  training and republished without restriction for reproducibility purposes") and American Stories (CC BY-4.0
  over public-domain scans) are content-side clean. Every other modern candidate — FineWeb, FineWeb-Edu,
  RefinedWeb, Dolma, HPLT, Zyda, Ultra-FineWeb, OpenWebMath — licenses *the database* and its own card says
  the user stays bound by the sources' terms. That is a research-arm basis; it is not the basis on which
  [ADR-0039](../../decisions/ADR-0039-three-level-model-release-gate.md) releases weights.
* **It attacks the documented weakness as far as affordable, licence-clean data allows.** The register moves
  from literary prose to reportage, editorial, advertisement and legal notice, and the date span moves the
  corpus forward from "19th and early 20th century" into the mid-20th century. It does **not** deliver 2020s
  technical prose, and the record must say that; nothing found in this research delivers it under a licence
  that reaches the content. What it does deliver is the time axis: with `date` in the catalogue, "held out
  means held out by work" ([ADR-0042](../../decisions/ADR-0042-held-out-means-held-out-works.md)) can be
  extended to "and by era", which is the only instrument that can measure the transfer claim the current
  record can only disclaim.
* **Acquisition mechanism, verified rather than assumed.** One anonymous HTTPS GET of a single parquet object
  — 1,000 objects measured at 151.5 GB total, largest 626 MB, so one object is several times the target
  slice. No account, no agreement, no gate, no WARC/XML parsing. The card names the upstream source (the
  Library of Congress [Chronicling America OCR dumps](https://chroniclingamerica.loc.gov/data/ocr/), 2,618
  original dump files), so a second independent route exists if the mirror moves.
* **It fits the machine.** Well under 1 GB downloaded for the slice, ≈150 MB snapshot, ≪ 154 GB free disk;
  the measured E2612 preparation (70.4 s for a 12M-character tokenizer sample plus packing) and peak RSS
  (1,710 MB) already bound a corpus of this size. The real new work is a source-specific boilerplate rule, and
  the card warns about exactly what it must strip: OCR errors, headers, page counts, multi-column layout.
* **The cheap secondary lever, measured.** Register rebalancing inside Gutenberg has large headroom —
  `P` 32,971 / `Q` excluding `QA` 2,151 / `QA` 127 / `T` 1,514 / `R` 624 English works available against
  220 / 90 / 35 / 60 / 25 pinned — and it costs nothing new to licence. Pair it with a **self-authored modern
  technical register** from a declared deterministic project-side generator, labelled synthetic: the only
  data class in this research that is simultaneously modern, uncontaminated and unambiguously owned, and the
  natural held-out side for the transfer question.

**Research-only and fallback arms, in order.** (1) `HPLT/HPLT2.0_cleaned` `eng_Latn_1` — the cheapest modern
English per unit of effort measured here (239 MB, 104,293 rows, **≈430 million characters** at the measured
1,000-document mean of 4,122.9 characters, re-read here in 15.1 s; an earlier projection of ≈0.9 billion from a
single 8,853-character document was wrong — see V-20 of the
[verification ledger](research-verification-ledger-2026-11-01.md), which also recovers the column semantics in
V-21 and the register caveat in V-22), carrying per-document `lang`,
`robotstxt`, `pii`, `filter` and `doc_scores` fields; its own card
declines responsibility for EU Copyright Directive 2019/790 and GDPR compliance, which is the reason it is an
arm and not a base. (2) `HuggingFaceFW/fineweb-edu` `sample-10BT` — ODC-By + CommonCrawl ToU, 2.15 GB per
shard, 726,000 rows and ≈3.06 billion characters measured, per-document educational score; record that the
classifier's annotations were generated by Llama-3-70B-Instruct, i.e. the "educational" property is a
distilled judgement. (3) `dell-research-harvard/AmericanStories` — CC BY-4.0 over public-domain scans with
article-level OCR, at the cost of a ~7 GB year tarball and an extraction step. (4) `allenai/dolma3` one
component at a time, with its Wikipedia/Wikibooks and peS2o components excluded by prefix. (5) If even one
newspaper object is judged too expensive, the honest outcome is register rebalancing plus synthetic text only
— and the record should then state that the modern-transfer question remains unmeasured, rather than
re-describing E2612 as if it had answered it.

**What must be preregistered before any run** (the [ADR-0036](../../decisions/ADR-0036-corpus-snapshot-recovery-from-history.md)
discipline, applied to a download rather than to git history):

1. Corpus identifier and version, the exact URL of every fetched object, the SHA-256 of the bytes read and
   the retrieval date — for each newspaper parquet object (and, for the web arm, each HPLT shard), not just
   for the resulting snapshot. This is also the fix for the existing gap in E2612: its reproduction block names
   a local `pg_catalog.csv` with no URL, digest or date.
2. The licence statement copied verbatim from the source, in full: for the newspapers the CC0 line **and** the
   public-domain sentence **and** the Library of Congress's own "you should determine for yourself whether or
   not an item is protected by copyright"; for a web arm, the CC0/ODC-By **packaging** sentence **and** the
   TDM/GDPR or original-source-terms disclaimer. A licence record that quotes only the convenient half is not
   a licence record.
3. The selection rule written before downloading: seed, shuffle over the object's documents, the character
   floor and ceiling, an explicit **language filter** (the card states 600k German and 400k Spanish editions
   are in the collection), a boilerplate rule for headers/page counts, the era bucketing rule, and every
   rejection with its reason.
4. Split **by whole edition-and-date group** — i.e. by newspaper title and issue run, not by article — with
   the grouping key stated, an era partition (pre-1900 / 1900–1940 / 1940–1963) declared, and a
   zero-group-on-two-sides assertion like E2612's 312/56/62.
5. The tokenizer decision: a new 4,096-merge vocabulary refitted on a declared seeded sample (the sample is
   part of the cache identity, so this is a new tokenizer, not the E2612 one), with `min_frequency` declared
   and the fitted identifier count measured rather than estimated.
6. Both arms' token budget, step count, seed list, evaluation packs, the uniform/unigram baselines, and the
   0.02 bits-per-character materiality band.
7. The contamination check and its result: exact-digest and n-gram overlap between the new corpus and both
   the Gutenberg test side and any externally-licensed held-out set, recorded as pairs examined.
8. An explicit statement of what the result cannot be: no capability claim, no Level 3
   ([ADR-0039](../../decisions/ADR-0039-three-level-model-release-gate.md)), and no benchmark score. If
   LAMBADA-style or BLiMP-style instruments are attached, they are labelled harness-side diagnostics and
   their training-exclusion is verified, not asserted.
9. A redistribution decision per corpus: which arm may reach a released artifact (the public-domain newspaper
   arm: yes, with attribution recorded) and which may not (any packaging-licensed web arm: research only),
   until the content-versus-packaging question is answered in writing.

## What this research did not establish

* **Could not verify**: a retirement/withdrawal notice for The Pile at any primary page; a source repository
  for CC-News; dataset objects for "ProofPile-A/B", MathMC, CommonMath, GTDB (GEM), ChildCraft, Data-Dogey,
  or the BabyLM official corpora; the HellaSwag `LICENSE` file itself (its repository is DMCA-blocked); the
  size of `KokunoYumeto/open-math-courses`; the full contents behind the CC0-tagged Caselaw mirror; the
  meaning of the four numeric columns in the HPLT per-language table (the header row was not located before
  this document was written).
* **Not measured**: any throughput for the slicing pipeline (only the input sizes and download costs, which
  were read from real bytes), the OCR quality of the newspaper corpus (the card itself states it is uncorrected
  and that headers/page counts/multi-column layout are likely unwanted for training — a boilerplate rule is
  therefore mandatory, not optional), the text yield of one American Stories year tarball, any tokenizer
  compression on new text (a 4,096-vocabulary figure on a new corpus is unknown until fitted), and the quality
  of any corpus beyond what its curators report.
* **Not a legal opinion.** ODC-By, CC0, CC BY-SA, CDLA-Sharing and the NVIDIA/Qwen/DeepSeek chains above are
  reported as the sources state them. The two open legal questions this document surfaces — whether a
  share-alike or reciprocal data licence reaches released weights, and whether a packaging-only CC0/ODC-By
  grant is enough for a released artifact — are the same two the curriculum already lists as unanswered, and
  they are for a written licence decision, not for an engineering ticket.

## Sources read on 2026-11-01

* Project Gutenberg: [license policy](https://www.gutenberg.org/policy/license.html),
  [feeds index](https://www.gutenberg.org/cache/epub/feeds/), `robots.txt` — plus HTTP headers of
  `pg_catalog.csv.gz` and `txt-files.tar.zip`.
* Wikimedia: [enwiki latest dump index](https://dumps.wikimedia.org/enwiki/latest/),
  [CC BY-SA 4.0 text page](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_4.0_International_License).
* arXiv: [license help page](https://info.arxiv.org/help/license/index.html); arXiv API for
  1606.06031, 2107.06499, 2211.15533, 2304.01373, 2305.07759, 2310.06786, 2406.01574, 2406.18326,
  2408.03617, 2502.17259, 2509.00072, 2510.02386, 1904.01557.
* Hugging Face API and dataset cards: `HuggingFaceFW/fineweb`, `fineweb-edu`, `finephrase`,
  `HuggingFaceTB/cosmopedia`, `smollm-corpus`, `tiiuae/falcon-refinedweb`, `allenai/dolma`,
  `dolma3_mix-6T`, `EleutherAI/pile`, `lambada_openai`, `bigcode/the-stack-v2`,
  `open-web-math/open-web-math`, `hoskinson-center/proof-pile`, `project-numina/NuminaMath-CoT`,
  `roneneldan/TinyStories`, `deepmind/pg19`, `nyu-mll/blimp`, `cimec/lambada`, `Rowan/hellaswag`,
  `Salesforce/wikitext`, `HPLT/HPLT2.0_cleaned`, `Zyphra/Zyda-2`, `openbmb/Ultra-FineWeb`,
  `nvidia/Nemotron-CC-v2`, `iknowyoursecret/US-PD-Newspapers` (card read in full),
  `dell-research-harvard/AmericanStories`, `endomorphosis/Caselaw_Access_Project_JSON`,
  `KokunoYumeto/open-math-courses`, `librarian-bots/arxiv-metadata-snapshot`, `Skylion007/openwebtext`;
  model card `HuggingFaceFW/fineweb-edu-classifier`.
* Open Data Commons: [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/).
* Library of Congress: [Chronicling America OCR dumps](https://chroniclingamerica.loc.gov/data/ocr/), named as
  the upstream source by the CC0 corpus above (the LC page itself was not re-read for this document; the
  quotation used is the one carried on the dataset card).
* GitHub API: `google-deepmind/mathematics_dataset` (Apache-2.0), `rowanz/hellaswag` (blocked, DMCA notice
  [2026-09-14-wikihow](https://github.com/github/dmca/blob/master/2026-09-14-wikihow.md)), repository search
  for `cc_news`, `proof-pile`, `GEM`.
* Parquet objects range-read for this document: `HPLT/HPLT2.0_cleaned` `eng_Latn_1/train-00000-of-00356.parquet`,
  `HuggingFaceFW/finephrase` `faq/000_00000_0.parquet`, `HuggingFaceFW/fineweb-edu` `sample/10BT/000_00000.parquet`.
* Repository: [AGENTS.md](../../../AGENTS.md),
  [`docs/research/chakaso-data-curriculum.md`](../../research/chakaso-data-curriculum.md),
  [`experiments/E2612/record.md`](../../../experiments/E2612/record.md),
  [`scripts/acquire_gutenberg_corpus.py`](../../../scripts/acquire_gutenberg_corpus.py).
