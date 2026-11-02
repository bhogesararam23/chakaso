# ADR-0048: the pre-tokenizer is a named part of a vocabulary's identity, and a space-prefixed rule is available

- Status: Accepted (2026-11-02). Implemented in three layers, each its own commit: the vocabulary (this one), the checkpoint, the experiment plan.
- Date: 2026-11-02
- Related: [H17](../research/hypotheses.md), [ADR-0035](ADR-0035-transformer-checkpoint-format.md), [ADR-0046](ADR-0046-checkpoint-records-the-tokenizer-frequency-floor-as-an-optional-field.md),
  [ADR-0044](ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md).

## Context

Every vocabulary this project has trained uses one pre-tokenizer, `\w+|\s+|[^\w\s]`: a run of word characters, a run of whitespace, or one other
symbol, with merges applied only inside a piece. A whitespace run is therefore always its own piece. Counted on 2026-11-02 over 1.5M characters of each
register with the committed vocabulary `434debe82f949347`, 34–39 % of the identifiers decode to whitespace only and 30–36 % are a single space
(`scripts/measure_token_classes.py`, research-log entry of that date). That is the observation behind H17. Testing H17 needs a second pre-tokenizer
that a vocabulary can be fitted under, and that a checkpoint, a report and a cache key can tell apart from the first.

The constraint is the same as ADR-0046's: nothing already on disk or already cited may change. The fingerprint of the committed vocabulary
(`434debe82f949347`) is cited by eight experiment records (E2612-E2619); every checkpoint embeds a merge table under it.

## Decision

1. **A vocabulary names its pre-tokenizer.** `SubwordVocabulary.pre_tokenizer` is one of a small registry of named rules (`DEFAULT_PRE_TOKENIZER`, the
   existing rule, and `SPACE_PREFIXED_PRE_TOKENIZER`). An unknown name is refused on construction, on load and by `train_subword_vocabulary`.
2. **The name enters the fingerprint and the checksum, and for the default rule they are unchanged.** The fingerprint already hashed the pre-tokenizer's
   name as one of its parts; it now hashes the vocabulary's own, which for the default is the string it always hashed. A test loads the committed
   vocabulary and asserts `434debe82f949347`. The same merge table under the other rule is a different tokenizer with a different fingerprint, because it
   encodes the same text differently.
3. **The space-prefixed rule** is `[ ]?\w+ | [ ]?[^\w\s] | \s*[^\S ] | \s+(?!\S) | \s+`. A word or one other symbol carries a single preceding space; whitespace that ends in a
   newline, tab or other non-space character stays one piece (a paragraph break is one merge target); in a run of spaces the last space goes to the
   next word. Every character falls in exactly one alternative, so the pieces always rejoin to the input, which a test asserts over multi-script text,
   tabs, CRLF and trailing space. This is the byte-level convention of GPT-2 (Radford et al., 2019) adapted to this project's `\w` word class; it is an
   established method and nothing here claims otherwise.
4. **The default rule stays the default.** Nothing in the repository switches to the new rule; it exists to be measured (H17).

## What this does not decide

* Whether the new rule is better. That is H17's experiment, preregistered separately.
* Anything about the checkpoint or the experiment plan, which are the next two layers: a checkpoint written under the non-default rule must record it (an
  optional metadata key, written only when non-default, as ADR-0046 did for the frequency floor) and an older reader must fail rather than mis-encode. The
  fingerprint check already guarantees the latter: a reader that does not know the key rebuilds a default-rule vocabulary, whose fingerprint differs from the
  recorded one, and refuses the file.

## Consequences

One field, one registry, one fitting parameter. No committed vocabulary, checkpoint, fingerprint or recorded number changes.
