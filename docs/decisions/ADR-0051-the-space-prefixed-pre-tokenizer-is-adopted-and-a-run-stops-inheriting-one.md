# ADR-0051: the space-prefixed pre-tokenizer is adopted for new fits, and a measured run stops inheriting one

- Status: Accepted (2026-11-04). Implemented in `chakaso.training.subword`, `chakaso.training.nn_checkpoint`,
  `chakaso.training.protocol` and `chakaso.training.subword_experiment`. The released artifact is untouched: its
  vocabulary is cited by fingerprint and its protocol pins the file.
- Date: 2026-11-04
- Related: [ADR-0048](ADR-0048-the-pre-tokenizer-is-a-named-part-of-a-vocabulary-s-identity.md) (which named the rule
  and explicitly declined to choose between them), [ADR-0049](ADR-0049-a-runs-optimizer-settings-are-declared-not-inherited.md)
  (the same argument one layer down: a recipe a run inherits is a recipe no re-run controls),
  [ADR-0044](ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md),
  [E2620](../../experiments/E2620/record.md), [E2624](../../experiments/E2624/record.md),
  [E2625](../../experiments/E2625/record.md), [E2628](../../experiments/E2628/record.md),
  [E2630](../../experiments/E2630/record.md), [E2632](../../experiments/E2632/record.md),
  [H17](../research/hypotheses.md), [K-001](../agent/KNOWN_ISSUES.md).

## Context

ADR-0048 decided that a pre-tokenizer is part of a vocabulary's identity — its name is recorded, its fingerprint
changes with it, and a checkpoint that cannot name the rule cannot be re-encoded honestly — and then said, in its own
words, that it **did not decide which rule to use**. Four measured records have since decided the question on loss, and
one decided the objection that no number of loss seeds could answer:

| Record | Reading | What it established |
| --- | --- | --- |
| [E2620](../../experiments/E2620/record.md) | **V-a**, one seed | O1 −0.0264 and whole pack −0.0321 bits per character at equal updates and positions, confounded with 40 % more text per update |
| [E2624](../../experiments/E2624/record.md) | **W-b** | at equal *text*, the loss is the update count: 0.058 bits per character of the gap is compute, not tokenizer |
| [E2625](../../experiments/E2625/record.md) | **X-a** | at equal updates *and* documents, O1 −0.0229 and whole pack −0.0286: the gain is the rule's at fixed compute |
| [E2628](../../experiments/E2628/record.md) | **Y-a** | a second initialisation on a second host, every register improving |
| [E2630](../../experiments/E2630/record.md) | **Y2-a** | five paired same-host seeds at E2614's own protocol: Δ(O1) **+0.026431**, 95 % Student-*t* interval **[+0.023212, +0.029650]**, all five draws positive, and the treatment's own seed deviation *smaller* than the control's (0.001595 against 0.002240) |
| [E2632](../../experiments/E2632/record.md) | **R-a** | the mechanism is refit-stable: four ~4.01M-character fitting samples × two rules, eight distinct vocabularies, within-sample compression ratios 1.405175–1.409317 with a relative spread of **0.30 %** |

E2630 answered "is it more than one draw". E2632 answered the question seeds cannot: whether the advantage belongs to
the rule or to the particular 12,000,000-character sample that produced `4a47f0281c81a9e5`. Two fits of one rule agree on
10.9 % of their merge *lists* and still compress the same to the third decimal, so the answer is the rule.

What stood between that evidence and a default was not more measurement. It was the discovery, made while writing this
record, that **the ambient default is load-bearing in the artifact format**. `nn_checkpoint` writes
`tokenizer_pre_tokenizer` only when the rule differs from `DEFAULT_PRE_TOKENIZER`, and a reader uses
`DEFAULT_PRE_TOKENIZER` when the key is absent. That is sound while the default never moves and silently wrong after it
moves once: `chakaso-base@0.1.0`'s checkpoint names no rule, because at the time that meant "the word-run rule", and
under a moved default the same bytes would mean "the space-prefixed rule" and the released model would be re-tokenised
against a vocabulary that does not encode its text.

## Decision

1. **Adopt the rule for new fits.** `DEFAULT_PRE_TOKENIZER` becomes `SPACE_PREFIXED_PRE_TOKENIZER` — the rule a caller
   gets when it fits a vocabulary without saying how. The gain is 0.026 bits per character on the objective five records
   measure, the cost is nothing at inference, and E2632 shows it does not depend on which text the rule was fitted on.
2. **Separate the format's constant from the library's default.** A new named constant,
   `LEGACY_DEFAULT_PRE_TOKENIZER = "word-run|whitespace-run|single-symbol"`, is what the absence of the key means.
   `nn_checkpoint`'s write condition and its read fallback, and the report's decision to omit the key, all read that
   constant, so **the absence of a rule names a date, not a choice**. A test pins its value literally, and a second test
   asserts that an artifact which writes no key reads back as the word-run rule.
3. **A measured run cannot inherit a rule.** `chakaso.training.protocol.load_plan` refuses a plan that fits a vocabulary
   (no `tokenizer_file`) without naming `tokenizer_pre_tokenizer`. Every protocol that reaches a measured record either
   pins a committed vocabulary — ADR-0044 — or states its rule; E2624 and E2625 already declared theirs even while
   pinning a file, which is the precedent this generalises. A run whose tokenizer is a named input cannot silently
   change identity; a run that *learns* a tokenizer must say how.
4. **Nothing released, and nothing measured, is restated.** `chakaso-base@0.1.0` stays at vocabulary
   `434debe82f949347`; the thirty-plus committed records keep their numbers, their fingerprints and their protocol
   digests. The adoption changes what happens the next time a vocabulary is fitted without instructions. Where a future
   record trains under the adopted rule, its digest differs from the E2612-era runs **because its alphabet differs**,
   which is the comparison refusing to be false rather than a defect in the digest.

## Consequences

- Code that constructs a `SubwordVocabulary` or calls `train_subword_vocabulary` without naming a rule now gets the
  adopted one. Tests that compare against a *specific* fingerprint must name the rule they mean; that is the change
  doing its job, not breakage, and the affected fixtures are updated in the same commit as the move.
- `scripts/measure_refit_sensitivity.py`'s arm labelled `default` is bound to the legacy constant rather than the ambient
  one, because E2632's "default" names the rule its committed reference vocabularies were fitted under. A measurement
  whose arm labels follow the ambient default would relabel its own arms without changing a byte.
- The comparison thresholds do not move. E2614's 0.02-bit band is the reference configuration's seed spread under the
  *old* rule; E2630 measured the adopted rule's own deviation at 0.001595, smaller, so reading a future difference
  against the old band is conservative rather than wrong, and the record that first trains under the adopted rule should
  say which band it used.
- A protocol that fits its own vocabulary now fails fast with a named missing field instead of producing a run whose
  alphabet depends on which release of this library was installed.
- **This decides nothing about capability.** A pre-tokenizer that spends 41 % more characters per identifier makes the
  same model predict slightly better over the same archaic English text. K-001 is unchanged by all six records: no model
  here answers a grounded question, and none produces useful text.

## Alternatives considered

- **Require declaration, keep the old default.** Rejected: it leaves a developer who fits a vocabulary without reading
  the ADRs with the measurably worse rule, and "worse on every register, on every seed, on both hosts, under every fit"
  is an odd thing to leave as the path of least resistance.
- **Flip the default without separating the format constant.** Rejected, and it is the reason this ADR exists at all:
  it would reinterpret every released checkpoint that omits the key, silently changing what their weights are read
  through. ADR-0048's own compatibility note — an artifact under the old default serialises exactly as before — depends
  on the default being a fixed thing.
- **Refit at 12M, retrain the reference configuration, and only then decide.** Rejected on cost and on need. E2632
  measured the mechanism at 4M and found it saturated — a 12M fit does not beat a 4M fit at 4,096 identifiers — so a
  third 2,116-second fit would buy a fingerprint, not information. The loss evidence is five paired seeds across two
  hosts. What the campaign lacks is useful text, and no tokenizer choice moves that.
