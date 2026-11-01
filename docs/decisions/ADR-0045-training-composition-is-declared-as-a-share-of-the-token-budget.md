# ADR-0045: a training mixture is declared as a share of the token budget, drawn as nested prefixes of seeded shuffles

**Status:** Accepted
**Date:** 2026-11-01
**Related:** [ADR-0041](ADR-0041-next-training-target-general-base-with-technical-mixture.md) (data before architecture),
[ADR-0042](ADR-0042-held-out-means-held-out-works.md) (held out by work), E2612, E2613, E2614, hypothesis H9.

## Context

E2612 and E2613 varied the training mixture by filtering the training side on a group prefix and capping the packed
stream. That produced two compositions — every register in its natural proportion (prose came to 51.03 % of the 20M
identifiers, measured by replaying the shuffle) and prose alone — and nothing in between. H9 asks for the shape of
the trade-off between them, and a shape needs a dial: the share of the budget a register supplies has to be a number
the run is told, not the consequence of a cap applied to a shuffled stream.

## Decision

1. **A plan may declare `share_group_prefix` and `share_of_tokens` together.** Works whose group starts with the
   prefix supply that fraction of `max_training_tokens` identifiers; every other training work supplies the rest.
   The two fields are set together or not at all, require a token budget and `shuffle_training_units`, and exclude
   `training_group_prefixes`, which they would contradict. The share is of **identifiers**, the unit the budget and
   the step count are in, not of characters.
2. **Each class is a prefix of its own seeded shuffle.** A larger share of a class therefore contains every
   document a smaller share contained, so the arms of a sweep are nested and a difference between two arms is the
   added text, not a different sample of it. The two selections are then shuffled together with the same seed so
   registers interleave in the stream.
3. **The endpoints are the runs filtering already produced.** A share of 1 or 0 returns exactly the filtered,
   shuffled list that `training_group_prefixes` does, and the interleaving shuffle is skipped when one class is
   empty. E2612's prose control is therefore the share-1 endpoint of any sweep on the same split and budget, and a
   test holds the equality.
4. **A quota the pool cannot supply is an error.** A run that asked for a share and could not draw it would
   otherwise train on a different mixture under the same label.
5. **The realised composition is reported, not the declared one.** The report's corpus block gains
   `training_composition` (declared and realised share, identifiers by register and by work count), and the packed
   works replace the whole training side in `training_groups`. These keys, and the two budget keys, appear **only
   for a share-composed run**. A key emitted as `None` by every other run would change the protocol digest of
   E2612, E2613 and E2614 when their reports are re-read, and the digest exists so a faithful re-run can reproduce it.
6. **An arm comparison treats the share keys as optional declared differences.** A report written before the
   feature has no such key, and "absent" is a true value there.

## Consequences

- H9 and any later mixture study can be run from a committed protocol file that differs from its siblings in one
  line, and a test holds that.
- The realised share is within one document of the declared one per class; the report states it.
- The mechanism governs **what is drawn**, not how often a document is repeated: every arm of the first sweep stays
  under one pass over its class's pool (the pool inventory is in the E2615 record), which the mechanism does not
  enforce and the record does.
- Interface change: two `ExperimentPlan` fields, three report keys, one `arm_comparison` rule, and a module
  (`chakaso.training.mixture`) with no numerical dependency.

## Alternatives considered

- **Sampling documents with probabilities.** Cheaper to write, but two arms then differ by the sample as well as the
  share, and the endpoint equality in (3) is lost.
- **A weight per register.** Needed only when more than two classes must be independently controlled; the sweep this
  decision serves varies one. The general form can replace this one without changing (2) or (3).
- **Measuring the share in characters.** Characters per identifier differ by register (about 2.1–2.3 here), and the
  budget, the schedule and the step count are all counted in identifiers.
