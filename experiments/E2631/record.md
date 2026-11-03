# E2631 — does withholding weight decay from the norm scales and biases change anything?

**Status:** preregistered — committed before the arm runs.
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
  --aggregate "o1=prose-,science-,technology-,medicine-,mathematics-" \
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

The second command is the descriptive reading: it refuses to compute anything until each arm reproduces the
denominators and the pooled bits per character its own committed report publishes, so the interval it prints is
measured on the text the runs were scored on.
