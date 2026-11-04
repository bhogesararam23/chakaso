# E2631 — does withholding weight decay from the norm scales and biases change anything?

**Status:** measured — **N-b.** Preregistered at 2026-11-03T22:41:57Z (`a5cde40`) and the arm started 38 seconds later
at 22:42:35Z, so the rules were committed before the run existed.
**Type:** Research experiment, **preregistered**. One training arm; its control is already on disk.
**Protocol file:** [`configs/experiments/E2631-vector-decay-exemption.toml`](../../configs/experiments/E2631-vector-decay-exemption.toml),
read by `chakaso experiment study --protocol … --seeds 7`. It differs from
[E2626's](../../configs/experiments/E2626-curve-ladder.toml) in exactly one key, `weight_decay_excludes_vectors`, and a
test in [`tests/test_experiment_protocol_files.py`](../../tests/test_experiment_protocol_files.py) holds it to that claim.

## Question

[ADR-0049](../../docs/decisions/ADR-0049-a-runs-optimizer-settings-are-declared-not-inherited.md) declared the optimizer recipe every measured
run had inherited from `torch`'s defaults, and added an option — `weight_decay_excludes_vectors` — that withholds
decoupled weight decay from the one-dimensional parameters: the RMSNorm scales and the feed-forward biases. ADR-0049
explicitly did **not** claim the exemption helps; it made the choice expressible and recorded, and left the question to
an arm. That arm is this record, and it is the only half of
[K-030](../../docs/agent/KNOWN_ISSUES.md) item 3 that is still open.

The practice the option copies is widespread — Llama- and Gemma-style recipes, and AdamW's own paper, keep decay off
parameters where shrinking the scale changes only the effective learning rate of the layer. The published evidence for
it is at scales two and three orders of magnitude above this one. At 3,413,760 parameters the exempt tensors are
**5,888 elements in 17 tensors — 0.17 % of the model** (measured in the control's checkpoint, not estimated), and the
hypothesis that decaying them matters at all is correspondingly weak. That is the null this record exists to test
rather than assume.

## The arms, and why the control needs no rerun

| Arm | Configuration | Whole pack | What it is |
| --- | --- | --- | --- |
| control | [E2626](../E2626/record.md) seed 7, this host, decay on every parameter | 2.05544739454246 | the reference configuration; its step-2,400 artifact is bit-identical to E2614 seed 7 and to E2612 arm T |
| treatment | E2631 seed 7, this host, decay withheld from 1-D parameters | to be measured | the same protocol with `weight_decay_excludes_vectors = true` |

E2626 ran on this host with the recipe ADR-0049 now declares, wrote a rung ladder, and its P4 check established that
its weights are E2614 seed 7's weights to the bit. The pair therefore differs in the exemption and nothing else: same
corpus (`312571f9fa8351a8`), same whole-work split (`adf650435c3a9068`), same committed vocabulary
(`434debe82f949347`), same 2,400 updates × 8,192 positions, same seed, same denominators (254,976 test positions,
535,282 characters).

## Metrics, margins and rules, declared before the run

Primary **O1** (equal-weight mean of the five register partitions), secondary the **whole pack**. Δ = control − treatment,
positive when the exemption is better. Margins are E2614's, as in every record of this series: O1 tie **0.0062**, O1
separable **0.02**; whole-pack tie **0.00933** (2.77 times its measured seed deviation of 0.00337).

* **N-a — the exemption is materially better.** Δ(O1) > 0.02 **and** Δ(whole pack) > 0.00933. Consequence: the option
  becomes a candidate default, and what stands between it and one is a second seed and a decision record — the pattern
  H17 went through, not an assertion.
* **N-b — nothing is measurable.** |Δ(O1)| ≤ 0.0062 **and** |Δ(whole pack)| ≤ 0.00933. Consequence: K-030 item 3
  closes with the exemption left off by default, on a measurement rather than on the size of the tensor group, and
  ADR-0049's option stays available to anyone who trains at a scale where it might matter.
* **N-c — the exemption is materially worse.** Δ(O1) < −0.02 **and** Δ(whole pack) < −0.00933. Consequence: same
  closure, opposite sign, and the record says which way the point estimate falls so the next person does not re-ask it.
* **N-d — anything else**: one measure material and the other not, or a mixed sign. Reported with both Δs and both
  numbers beside them, and the default does not move.

**Secondary, descriptive and declared as such.** The paired test E2633 built
([`scripts/paired_loss_test.py`](../../scripts/paired_loss_test.py)) is run over the two checkpoints: the character-weighted
mean difference across the 25 held-out works of the pack, with its 2,000-resample cluster interval and its
leave-one-out range. It answers a different question from the band — how well this difference is known on *this text*,
not whether it beats a re-initialisation — and no outcome letter turns on it.

The reading is `chakaso.training.arm_comparison.compare_arms` over the two committed reports, with
`decay_applied_to` newly among the conditions an arm is *allowed* to differ in: until this record it was in neither the
matched list nor the declarable list, which is the gap K-030 item 3's report should have named and did not. A
comparison of two arms that differ in the optimizer's decay now says so in its `declared_differences`, and every other
condition still has to match or the call refuses.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** **N-b.** 5,888 parameters of 3,413,760, decayed at 0.01 with a cosine schedule over 2,400 updates, are not
  plausibly worth 0.02 bits per character at this budget.
* **P2.** If the point estimates are not exactly zero, they favour the exemption: the exempt group is 2,048 norm-scale
  elements whose decay shrinks toward zero and a norm that shrinks is a layer that learns more slowly.
* **P3.** The rung curves stay within ±0.01 bits per character of each other at all twelve evaluations. A pair of arms
  that differ this little should not cross over, and a crossover at one rung would be the interesting result.
* **P4.** `protocol_digest` differs between the two runs, and the only field that differs inside it is
  `decay_applied_to`. Equal digests would mean the option is invisible to the protocol identity, which is the defect
  ADR-0049 avoided by recording it.

## Known limits, declared in advance

* One seed per arm. A difference inside the band cannot be resolved by more analysis of the same two runs; E2614's
  spread is what the band means, and this record reads against it.
* The 0.17 % figure is the *count* of exempt elements, not their influence: a norm scale multiplies an entire layer, so
  a small group can in principle matter. That is why the arm is measured rather than dismissed.
* Both arms are truncated at the step ceiling with validation still falling, so both numbers are floors
  (E2612's reading, unchanged), and the exemption could matter at a schedule that reaches a minimum.
* A single architecture at a single budget. Published recipes that exclude norm decay do so at scales where the
  embedding and the attention stack dominate; nothing here reaches them.
* K-001 stands: a loss figure, on archaic English prose, with no capability claim and no useful text.

## Cost

One arm at 2,400 updates × 8,192 positions with twelve rung writes, on this host, alongside other work: E2626's own
arm took 83 minutes at 3,766 positions/s and the three E2630 arms ran in 3,081–3,536 s. **Roughly 1–2.5 hours of CPU**,
plus a scoring pass of two checkpoints through `work_losses` (under a minute each, measured in E2633).

## Reproduction

```bash
QPYTHON=.venv-nn/Scripts/python.exe
"$QPYTHON" -m chakaso experiment study --work-dir data/experiments --experiment E2631 --seeds 7 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2631-vector-decay-exemption.toml \
  --write-report data/experiments/E2631/study.json

# Then, from committed files: one extract of the treatment report beside this record, `compare_arms` over it and
# E2626's committed report, and the paired secondary reading.
"$QPYTHON" scripts/compare_arms.py --arm "E2626=experiments/E2626/E2626-report.json" \
  --arm "E2631=experiments/E2631/E2631-report.json" --reference E2626 \
  --aggregate "chars=prose-,science-,technology-,medicine-,mathematics-" \
  --equal-aggregate "o1=prose-,science-,technology-,medicine-,mathematics-" \
  --out experiments/E2631/results.json
"$QPYTHON" scripts/paired_loss_test.py \
  --protocol configs/experiments/E2631-vector-decay-exemption.toml \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --work-dir data/experiments --root . --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --arm "exemption=data/experiments/E2631/seed-7/E2631-decoder.pt" \
  --arm "control=data/experiments/E2626/seed-7/E2626-decoder.pt" \
  --report "exemption=experiments/E2631/E2631-report.json" \
  --report "control=experiments/E2626/E2626-report.json" \
  --pair control/exemption --seed 20261104 --resamples 2000 \
  --out experiments/E2631/paired-loss.json
```

The first command differs from the one preregistered above in its aggregate flags only, corrected on 2026-11-04 for the
reason deviation 1 gives: `--aggregate` names the character-weighted reading `chars`, and the objective O1 is asked for
with `--equal-aggregate`.

The second command is the descriptive reading: it refuses to compute anything until each arm reproduces the
denominators and the pooled bits per character its own committed report publishes, so the interval it prints is
measured on the text the runs were scored on.

## Results (added 2026-11-04)

**Outcome: N-b — nothing is measurable.** Both readings fall inside their own tie margins, on the same side, and the
exemption stays off by default on a measurement rather than on the size of the tensor group. K-030 item 3 closes.

The arm ran 2026-11-03T22:42:35Z → 23:38:28Z on this host — 3,353 s, inside the 3,081–3,536 s the three E2630 arms
measured on this machine and well under E2626's 83 minutes, which ran alongside other work. It truncated at the
2,400-update ceiling with validation still falling, exactly as its control did; both numbers are floors.

| Reading | control (E2626 seed 7) | exemption (E2631 seed 7) | Δ = control − exemption | tie margin | verdict |
| --- | --- | --- | --- | --- | --- |
| **O1** — equal-weight mean of the five partitions (primary) | 2.021357 | 2.024857 | **−0.003499** | 0.0062 | inside |
| **Whole pack** (secondary) | 2.055447 | 2.059054 | **−0.003606** | 0.00933 | inside |
| Five-partition aggregate weighted by scored characters, which is what the declared `--aggregate o1=` computes | 2.049605 | 2.054224 | −0.004619 | — | descriptive; see deviation 1 and the addendum below |

Both point estimates put the exemption *worse*, by about a third of the margin on each measure, and neither reaches
its margin. N-b is the rule that fires, and it fires without needing the third rule: no measure is material, so
nothing has to be weighed against anything.

### Per partition, because the mean hides where the movement is

| Partition | control | exemption | Δ (control − exemption) |
| --- | --- | --- | --- |
| mathematics | 2.577769 | 2.597970 | **−0.020200** |
| medicine | 1.838141 | 1.834759 | +0.003383 |
| prose | 1.839520 | 1.842023 | −0.002503 |
| science | 1.948624 | 1.946925 | +0.001699 |
| technology | 1.902732 | 1.902607 | +0.000125 |

The entire O1 movement is carried by mathematics: −0.0202 there, with the other four inside ±0.0034 and two of them
positive. 0.0202 is 1.64 times the 0.01232 seed deviation E2614 measured for the mathematics partition, so a single
seed does not resolve it — but it is the only register where anything moved at all, and a future test of this option
should be a test on mathematics rather than another reading of the mean.

### The paired reading, declared descriptive

The tool refused nothing: both arms reproduced their committed denominators (254,976 positions, 535,282 characters)
and their pooled metric to better than 1e-9, and the pack is the runs' own — corpus `312571f9fa8351a8`, split
`adf650435c3a9068`, vocabulary `434debe82f949347`, 996 blocks over 25 held-out works.

Paired mean **−0.003606**, 95 % cluster interval over works **[−0.014208, +0.001682]** — includes zero. Thirteen of
the 25 works favour the control and twelve the exemption, exact sign test **p = 0.50000**. Leave-one-out range
**[−0.004653, +0.000460]** — removing one work flips the sign. The interval is four times the point estimate, and one
book does most of the work: `mathematics-pg302` is 0.133149 bits per character better under the control, while the
largest swing the other way is `science-pg39975` at +0.007572.

That is the signature E2633 named for a difference that is a draw rather than a property of the arm: a pair separated
by its data keeps its sign under every leave-one-out, and a pair separated by its initialisation does not. Two arms
that differ by 5,888 exempted parameters behave like the second.

### The four predictions

* **P1 — held.** N-b fired, and not narrowly: 0.0035 and 0.0036 against margins of 0.0062 and 0.00933.
* **P2 — failed.** The point estimates favour the *control*, not the exemption, on both readings and on the paired
  mean. The mechanism P2 reached for — a decayed norm scale shrinks, and a smaller norm learns more slowly — is not
  disproved at 0.17 % of the parameters, it is unmeasurable there, and the sign of what little moved runs the other
  way.
* **P3 — failed, twice.** Converted at the runs' own 2.0993 characters per identifier (identical in both reports,
  because both score the same pack), the validation curves differ by more than ±0.01 bits per character at two of the
  twelve evaluations — −0.010910 at step 400 and −0.010561 at step 2,200 — and the difference changes sign at step 600
  (+0.001480) and step 1,200 (+0.003460). The curves cross, which is the result this record called interesting if it
  happened. Step 0 is identical to the last digit (8.3717606 both), as it must be for one seed and one initialisation.
* **P4 — held, with a caveat about the digest rather than the run.** The protocol digests differ
  (`753cf5b64581` against `caa7db0a72ab`), and walking the digest's own input finds three differing keys: `experiment`
  and `plan.experiment_id`, which are the record's *name* and are inside the digest, and then
  `plan.weight_decay_excludes_vectors` with `transformer.budget.optimizer.decay_applied_to` — the exemption in its two
  places. So the option is visible to the protocol identity, which is what ADR-0049 asked for. The first two say
  something a reader comparing two records needs: a digest equality is only meaningful *within* one experiment id,
  because the id is part of it.

### What closes, and what does not

K-030 item 3 closes with the exemption **left off by default**, on this measurement. The option is not removed and not
recommended: `weight_decay_excludes_vectors` stays expressible and declared, for anyone training at a scale where the
embedding and attention stack do not dwarf 5,888 parameters. ADR-0049's careful refusal to claim the exemption helps is
now a claim tested once, at 3,413,760 parameters, and the test found nothing.

The comparison itself required one change to the tool: `decay_applied_to` had to be moved into the list of conditions
an arm is *allowed* to differ in (commit d970e8b). It had been in neither the matched list nor the declarable list, so
before this record a comparison of two arms that differ in decay could only refuse or lie. With it declarable,
`compare_arms` reports `all_conditions_matched: true` and names the single declared difference.

### Deviations

1. **The declared command's `--aggregate o1=` does not compute O1.** `compare_arms` weights a named aggregate by
   scored characters — the rule K-027 established after a population-weighted aggregate moved an effect size by four
   and a half margins — while O1 is the *equal-weight* mean of the five partition figures. The tool's aggregate gives
   2.049605 and 2.054224, the declared objective gives 2.021357 and 2.024857; both are inside their margins, so the
   outcome letter is unaffected, but the record named a quantity after a tool flag that computes a different one. The
   verdict is read on O1 as the preregistration declared it. What should follow is an equal-weight aggregate in
   `compare_arms`, so the primary objective of six records is computable from the reports rather than by hand.

   **Addendum, same day.** The follow-up landed: `compare_arms` now takes `--equal-aggregate`, which averages the named
   partitions with equal weight and labels its output `"equal weight over the named partitions (the O1 objective)"`,
   while `--aggregate` keeps the character-weighted reading and says so. `results.json` was regenerated with both —
   `o1` = 2.021357 and 2.024857 with a difference of 0.0035, `chars` = 2.049605 and 2.054224 with 0.004619 — so the
   primary objective is now computed by the tool rather than by hand, and the two quantities sit beside each other in
   the same file with their weightings named. The numbers this record quotes are unchanged by the regeneration; the
   preregistration's error is left in the text above because it is the finding.
2. **The treatment's committed extract cannot state its sample.** `experiments/E2631/E2631-report.json` carries no
   `evaluation_sample` block, because the arm was launched 90 minutes before that field landed in `prepare_corpus`. The
   composition of this pack is E2633's measured read of the identical pack — science 46.6 %, prose 41.5 %, mathematics
   6.0 %, technology 5.9 %, **medicine 0 %** — since both arms score vocabulary `434debe82f949347` over the same capped
   sample at the same seed. It is recorded as a deviation rather than patched into a report that was not written by the
   code that would write it.
3. Nothing was substituted. The pair, the seed, the resample count, the denominators, the margins and the four rules
   are the ones committed before the run; the arm ran the command in the Reproduction block and nothing else.
