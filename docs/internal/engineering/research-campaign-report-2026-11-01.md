# Research campaign report — 2026-11-01

**Status:** Research. This is the closing document of the 2026-11-01 deep-research phase. It says what was
verified, what was found wrong in the campaign's own work, what remains uncertain, and what should happen next.
The durable findings are **not** in this file — they are in the documents it points at, so that an agent without
this conversation can recover them. Start with
[the campaign index](../../research/chakaso-research-campaign-2026-11.md) and
[the verification ledger](research-verification-ledger-2026-11-01.md).

## Where the phase actually began

Not at the brief's step 1. `main` was at `a7fe8ad`, and the first genuinely incomplete task was sitting in the
working tree: **E2612 had run both arms on 2026-10-30, produced 12 artifacts, answered its preregistered
hypotheses, and had no results section.** The record's own first line promised results "added, never substituted",
and nobody had added them. The experiment's outcome existed only in git-ignored JSON.

Verifying it before writing anything was the right instinct: both arms completed the declared 2,400-step ceiling,
shared corpus fingerprint `312571f9fa8351a8`, split fingerprint `adf650435c3a9068`, tokenizer
`434debe82f949347` and one 254,976-position test pack, and differed in exactly one field —
`training_group_prefixes`. The preregistered outcome is **decision rule 2**: H1 true by 13.8× the band
(**−0.27584** bits per character on the technical aggregate), H2 false by 1.5× (**+0.03026** on prose). Recorded
in [`experiments/E2612/record.md`](../../../experiments/E2612/record.md) with a committed
[`results.json`](../../../experiments/E2612/results.json), so the comparison survives without the ignored tree.

## What was verified, and is now safe to rely on

1. **The measurement harness is honest where it matters most.** The unigram baseline is fitted on the arm's own
   training targets and scored on test (`subword_experiment.py:952-973`); the tokenizer sample is drawn from the
   training side only, by a seeded shuffle designed so catalogue order cannot fit one register (`:636-654`); the
   snapshot loader re-checks the file checksum, the manifest, every document body and the document set
   (`manifest.py:681-777`); artifacts are written atomically and refuse to clobber. **The campaign looked for a
   leak in the comparison machinery and did not find one.** That is worth stating as positively as the defects.
2. **E2612's arms are matched on ten fields checked individually**, not assumed — including all five per-register
   token counts. Its verdict survives **every** available weighting of the technical aggregate (−0.27584,
   −0.27884, −0.18650), which is why the headline can be trusted even though the effect *size* depends on a field
   choice.
3. **Whole-work splitting worked**: 312/56/62 works, `works_on_multiple_sides: 0`. The K-023 leak is absent by
   construction in this record.
4. **Three external claims were re-read from paper bodies after the abstracts failed to confirm them** —
   the vocabulary scaling exponent γ ≈ 0.83 (arXiv:2407.13623), Gemma 3's replacement of soft-capping with
   QK-norm (arXiv:2503.19786), and Meta's own sign reversal on multi-token prediction (arXiv:2404.19737). Two of
   these underpin the campaign's central REJECT verdicts, so an abstract-level check would not have been enough.
5. **One dataset recommendation was measured rather than accepted**: an anonymous range read of
   `HPLT/HPLT2.0_cleaned` `eng_Latn_1` confirmed 104,293 rows and recovered all 17 column meanings — including a
   full URL and nine per-document quality scores, which is what makes domain-grouped splitting and declarative
   quality filtering possible instead of invented.

## What was found wrong, including in this campaign's own work

1. **A measured result was not re-trainable, and a tracked commit asserted the opposite.**
   `6f3274f` claimed "the merge sequence on a reference text is byte-identical to the previous implementation, so
   no vocabulary moves." Measured against a brute-force reference, the indexed loop disagrees from merge rank 8 on
   a 43-piece fixture and from merge rank 5 on E2608's own training side, producing a different fingerprint and a
   0.583 % worse encoding on that corpus's own text. Fixed in `cfdcd1a`, which reproduces the recorded fingerprint
   exactly, with two regression tests that fail against the pre-fix code. **The 1,705-test suite had passed this
   behaviour change because no test stated the rule independently of the implementation** — the gap, not the bug,
   is the finding. Full evidence chain: ledger V-24 … V-29 and [K-024](../../agent/KNOWN_ISSUES.md).
2. **The fix relocated a boundary instead of erasing it.** E2612's vocabulary was fitted *by the defect*, so the
   corrected code cannot re-fit it (`3abe4df4b123d4c5` against the record's `434debe82f949347`, measured). Its
   arm-to-arm comparison is untouched, but its artifacts are now cache-dependent, and tokenizer caches are
   excluded from the repository. That is an unresolved policy question, not a closed bug.
3. **Exactness has a price, and it is now measured**: a 4,096-identifier fit over the 12.7M-character E2612 sample
   takes **35.3 minutes**, against ~29 s at 4.0M for the fast-but-wrong loop. Any plan that inherits preparation
   timings from pre-fix records is wrong by an order of magnitude on that line item.
4. **Three tracked sentences were untrue when read against the code, and are now issues.** The acquisition script
   documents a catalogue digest it never records (K-029); whole-work runs report `dropped_duplicates: 0` while
   excluding documents (K-025); and the generation block's own docstring warns against exactly the probe default
   that E2612 used (K-026). None distorts a published number; all three degrade provenance, and two make a claim
   in tracked text false.
5. **A first reading of the campaign's own artifacts was wrong in three places, and each is recorded as
   corrected rather than quietly reworded.** E2612's `samples_refused: 1` is a designed empty-prompt edge probe,
   not a model failure — all five continuations generated. A projected dataset yield of ≈0.9 billion characters
   was really ≈430 million, because one 8,853-character document stood in for a 1,000-document mean of 4,122.9.
   And five arXiv identifiers circulating in this project's own drafts resolved to unrelated papers, including
   `2404.07944`, which is a quantum-bundles paper rather than evidence for Llama 3's schedule.
6. **The brief's own premises were wrong in three places and were corrected rather than satisfied.** Qwen never
   switched word-piece to byte-level BPE; Phi-1 and Phi-1.5 contain no tokenizer-vocabulary ablation; Gemma 2 used
   GeGLU, not PerToken-Fn. A research phase that confirmed its starting assumptions would have been the more
   suspicious outcome.

## What remains uncertain

**The one that matters most: nothing here is evidence of capability, and the campaign made that sharper.**
E2612's arm with 0.108 better held-out bits per character produced the **more repetitive** text
(`looping_fraction` 0.4 → 0.6, mean bigram repetition 0.675527 → 0.712437). Read with E2610's finding that no
decoding setting writes coherent text from the released checkpoint, the conclusion is not that the loss result is
weak — it is that **no instrument currently in this repository is known to detect a real improvement in what the
model has learned.** Held-out loss and surface quality are not merely loosely correlated here; in the campaign's
own data they moved in opposite directions.

The other open items are enumerated as eight questions in
[open-questions.md](../../research/open-questions.md) and eight falsifiable hypotheses in
[hypotheses.md](../../research/hypotheses.md). The highest-value one costs 90 minutes: E2612's arms differed in
prose *exposure* by roughly 2× as well as in register (measured: the training pool is 50.51 % prose), so the
narrow H2 failure cannot be attributed to the mixture rather than to the dose. **A third arm decides it, and until
it runs, E2612's most interesting number has two explanations.**

Also unresolved: the 0.02-bit materiality band is inherited from E2611's declaration and has never been
calibrated against a seed spread — E2609 was never run, so the decoder has no measured noise floor at any size.
And no partition in any run so far is modern English, which means the archaic-transfer disclaimer every record
carries has never once been measured.

*Update, later the same day:* the band's noise floor was then measured ([E2614](../../../experiments/E2614/record.md), five seeds): s = 0.00226 on prose, 0.00337 on the whole pack, 0.01232 on mathematics. The band holds for prose and is replaced by 0.035 for mathematics. The modern-English gap above is unchanged.

## What should happen next

In order, with the reasoning already argued in
[the decision matrix and roadmap](../../research/chakaso-research-campaign-2026-11.md):

1. **E2613 — the P½ exposure control** (H8), run with `--sample-prompts held_out`. ~90 minutes, no new machinery,
   and it is the only thing that turns E2612's declared confound into a conclusion.
2. **A decoder seed spread at the reference configuration**, so the materiality band stops being a convention.
3. **Fix the two measurement defects before they contaminate a future result**: the document-masking scorer
   asymmetry (which currently makes H13 untestable) and `dropped_duplicates` under whole-work splits.
4. **Acquire one licence-clean modern corpus** (`us-pd-press-v1`, CC0 over public-domain text) **with an era
   partition**, because that is the smallest change that makes the instrument able to detect the shift every
   record currently only disclaims.
5. **Then reweight the mixture; then consider capacity.** ADR-0041's next declared unit — the ~9.84M-parameter
   step — is **not** licensed by E2612, which said so in its own rule 2 before anyone read the result.

**Not next, and this should be said as plainly as the recommendations:** no frontier architecture mechanism.
Eleven mechanisms examined from their primaries produced **0 adopt, 1 adapt, 2 test and 8 reject** at this scale,
for reasons that are measured rather than stylistic — Llama 2's own Table 1 puts GQA at 70B, Meta states that
multi-token models "are worse than the baseline for small model sizes", Gemma 3 dropped soft-capping a generation
after Gemma 2 introduced it, and Mistral's 4,096-token sliding window exceeds Chakaso's entire 256-token context.
Meanwhile the one data experiment this project ran moved held-out loss by more than any of those mechanisms claims
at 3.4M. **The internal measurement and the external reading now point the same way, which is the strongest
argument this repository has for doing data work and the weakest possible argument for an architecture change.**

## Scope honesty

- **Not run:** any experiment. This phase produced one code fix, one completed record, six issues, 70 annotations,
  33 verification rows and five review documents. No model trained on anything new.
- **Not verified:** the specific claims listed in each review's "could not verify" register — including
  DeepSeek-V3's §6 limitations and R1's Appendix F (fetch-truncated), a distinct per-batch OLMo "traces" artifact,
  the retirement of The Pile at any primary page, and the sizes of several small candidate corpora.
- **Not measured:** any throughput for a new corpus slicing pipeline; tokenizer compression on modern English;
  exact-fit cost beyond 12.7M characters; anything on hardware other than this machine.
- **Not a legal opinion.** Licences are reported as their sources state them. Whether a share-alike or
  packaging-only grant reaches released weights is the same two-question gap the curriculum already lists as
  unanswered, and it needs a written decision rather than an engineering ticket.
