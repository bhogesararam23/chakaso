# Multilingual tokenization and what it would cost a 3.4M-parameter model

**Status:** Research. Three kinds of statement are kept apart and labelled: **Published** (an abstract read on
2026-11-01), **Measured here** (computed in this repository from a committed file, with the code below), and
**Computed** (arithmetic on numbers given, shown). Nothing here is a model result, and the measurement is an
illustration on one short sentence per language, not a benchmark.
**Why this note exists.** The brief asks for multilingual understanding as a general-purpose capability. The
campaign's tokenizer assessment ([T-01…T-13](../../research/chakaso-tokenizer-and-context-assessment.md)) was about
the English identifier space; nothing in the repository had measured what the committed vocabulary does to other
scripts, or what a multilingual vocabulary would cost at this parameter count.

## Published

| Source | What the abstract states (paraphrased; read 2026-11-01) |
| --- | --- |
| Petrov, La Malfa, Torr, Bibi, [*Language Model Tokenizers Introduce Unfairness Between Languages*, arXiv:2305.15425](https://arxiv.org/abs/2305.15425) (2023-05-17) | The same text translated into different languages can tokenize to lengths differing by up to 15×, even for tokenizers trained for multilingual support; character- and byte-level models still show over 4× differences for some pairs |
| Ahia et al., [*Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models*, arXiv:2305.13707](https://arxiv.org/abs/2305.13707) (2023-05-23) | Token counts for the same information vary widely across 22 languages, so usage-priced APIs overcharge speakers of many of them while giving poorer results |
| Rust et al., [*How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models*, arXiv:2012.15613](https://arxiv.org/abs/2012.15613) (2020-12-31) | Across nine languages, a designated monolingual tokenizer matters about as much as pretraining data size for downstream performance; languages well represented in a multilingual vocabulary lose little |
| Liang et al., [*XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models*, arXiv:2301.10472](https://arxiv.org/abs/2301.10472) (2023-01-25) | A single vocabulary shared across 100+ languages limits capacity; a one-million-token vocabulary that de-emphasises sharing between lexically distant languages beats XLM-R on every task tested, most on low-resource languages |
| Xue et al., [*ByT5: Towards a token-free future with pre-trained byte-to-byte models*, arXiv:2105.13626](https://arxiv.org/abs/2105.13626) (2021-05-28) | A standard Transformer can process raw bytes with minimal changes, with the trade-offs stated in parameters, FLOPs and inference speed; byte-level models are competitive and more robust to noise |

Only abstracts were read. None of these studies a model below hundreds of millions of parameters, so **none is
evidence about 3.4M**; they establish the mechanisms (the tokenization premium, the vocabulary-capacity trade-off,
the byte-level alternative) that the arithmetic below applies.

## Measured here: what the committed vocabulary does to other scripts

The vocabulary is `experiments/E2612/vocabulary-434debe82f949347.json` (4,096 identifiers, 3,839 merges, fitted on
Project Gutenberg English). One sentence, written for this note and translated into seven languages by the author of
this note (translations are plain and unproofed by a native speaker), was encoded with it:

> The river runs through the old town, and in the evening the market fills with people who sell bread, fruit and cloth.

| Language | characters | UTF-8 bytes | tokens | tokens / character | tokens ÷ English tokens | bytes / token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 117 | 117 | 51 | 0.44 | 1.00 | 2.29 |
| Spanish | 106 | 107 | 68 | 0.64 | 1.33 | 1.57 |
| French | 121 | 122 | 68 | 0.56 | 1.33 | 1.79 |
| German | 120 | 122 | 77 | 0.64 | 1.51 | 1.58 |
| Russian | 108 | 197 | 197 | 1.82 | 3.86 | 1.00 |
| Arabic | 97 | 179 | 179 | 1.85 | 3.51 | 1.00 |
| Hindi | 103 | 259 | 259 | 2.51 | 5.08 | 1.00 |
| Chinese | 28 | 84 | 84 | 3.00 | 1.65 | 1.00 |

Every row round-trips exactly (`decode(encode(text)) == text`), so nothing is lost; it is only expensive. For the four
non-Latin scripts the vocabulary contains **no applicable merge**: exactly one token per UTF-8 byte (1.00 bytes per
token), which is the byte fallback working as designed. The Latin-script languages pay 1.33–1.51× English, because the
merges are English's. (Chinese's smaller ratio to English is an artefact of characters carrying more meaning each, not
a cheaper encoding: 3.00 tokens per character is the highest in the table.)

*Reproduce:* load the file with `SubwordVocabulary.load`, call `encode` on each sentence and compare lengths. The
byte-fallback finding is pinned as an executable fact in
`tests/test_training_subword.py::test_the_committed_english_vocabulary_has_no_merge_for_four_non_latin_scripts`.

**Computed — what that does to a 256-position window.** Characters that fit in 256 tokens at each language's measured
rate: English ≈ 587, Spanish ≈ 399, French ≈ 456, German ≈ 399, Russian ≈ 140, Arabic ≈ 139, Hindi ≈ 102, Chinese ≈
85. The context a Hindi or Chinese speaker gets is about a sixth or a seventh of an English speaker's, for the same
architecture — the *unfairness* Petrov et al. describe, reproduced in miniature at the byte-fallback end.

## Computed — what a vocabulary that served those languages would cost

With tied embeddings the table is `vocabulary × d_model` parameters, and the current model's `d_model` is 256:

| Vocabulary | Table parameters | Share of the current 3,413,760 | As a multiple of the whole current model |
| ---: | ---: | ---: | ---: |
| 4,096 (now) | 1,048,576 | 30.7 % | 0.31× |
| 16,384 | 4,194,304 | 122.9 % | 1.23× |
| 32,768 | 8,388,608 | 245.7 % | 2.46× |

(The shares exceed 100 % because the *rest* of the model would not shrink: the table alone at 32K identifiers is
larger than today's entire network.) This is the same collision the landscape document records for vocabulary
scaling: the published direction is a larger vocabulary for a larger model, and the parameter budget here does not
afford it. A model that served several languages through a subword vocabulary would not be a 3.4M-parameter model
with a different tokenizer; it would be a different model.

## What follows for Chakaso

1. **The current model is English-only by construction, not by choice of data alone.** The vocabulary encodes
   non-Latin scripts as bytes, and a model trained on English-only text would never learn those byte sequences
   (E2612's corpus is English). Multilingual capability is therefore not "less trained"; it is **absent**, and it is
   not reachable by any fine-tuning of this base.
2. **Adding a language is a stage-B decision, not a later stage.** A new vocabulary invalidates the parent's
   embeddings, so by [ADR-0043](../../decisions/ADR-0043-training-stages-are-separated-by-what-changes-and-entered-by-gates.md)
   rule 6 it is a new base run. The roadmap's ordering (data before architecture; scale before breadth) is unchanged.
3. **No multilingual experiment is licensed at this size.** The cost table makes the point arithmetically: serving even
   16K identifiers more than doubles the model before a layer is added. The honest options are (a) stay English-only
   until a parameter budget exists where a larger vocabulary is a minority of the model, or (b) a byte-level model
   (ByT5's direction), whose price here is a context window measured in bytes — about 2.3× fewer characters than
   today's English window — and which no experiment in this repository has tried.
4. **What to measure first, if the question reopens:** the same table on a *representative* parallel sample (hundreds of
   sentences per language, native-checked), and the vocabulary sweep H12 extended with a second language. Neither is
   run; both are cheap next to a training arm.

## Limits

* One sentence per language, author-translated: the ratios have no confidence interval and the Latin-script ratios in
  particular would move with the sentence. The byte-fallback finding (1.00 bytes per token for the four non-Latin
  scripts) does not depend on the sentence.
* The abstracts are paraphrased from the arXiv records; their bodies were not read, so the specific figures inside them
  (which languages, which pairs) are not reproduced here.
* A vocabulary fitted on English tells nothing about how a vocabulary *fitted on* a multilingual corpus would behave at
  4,096 identifiers; that is the unrun experiment in item 4.
