# Measurement harness audit — 2026-11-01

**Scope.** A skeptical audit of how measured results are *produced and guarded*: bits-per-character
arithmetic, baseline fitting and leakage, held-out partition sampling, corpus fingerprint/snapshot
verification, seed and determinism coverage, the generation/quality gates, and any place a metric could be
reported for a different object than the record says. Evidence is `file.py:line` against the working tree and
against the on-disk reports for E2608, E2611 and E2612.

**Status labels.** *Verified-correct* = checked in code and, where a number was involved, recomputed from the
committed artifact. *Risk* = a mechanism that can distort a published number without a guard firing.
*Necessary-not-sufficient* = the guard is real but weaker than its headline reads. Nothing here is a
capability claim; all numbers are the surface/loss measurements the records already describe as such.

Two documents frame this: [ADR-0036](../../decisions/ADR-0036-corpus-snapshot-recovery-from-history.md)
(a measured corpus stays re-trainable), [ADR-0042](../../decisions/ADR-0042-held-out-means-held-out-works.md)
("held out" must mean held-out *works*), and [K-011](../../agent/KNOWN_ISSUES.md)/[K-020](../../agent/KNOWN_ISSUES.md).

---

## Summary of the arithmetic (the load-bearing formula)

Bits per character is **not** `nats/token · log2(e) / chars-per-token` applied per token. The single shared
implementation (`training/model_metrics.py:109-156`) computes it as a *total* description length over a
*total* character count:

```
nats_total      = fsum(per-position -ln p(target))          # model_metrics.py:139
characters      = len(decoder.decode(concatenated targets)) # model_metrics.py:141-142
bpc             = (nats_total / ln 2) / characters          # model_metrics.py:96-106
```

This is algebraically `loss_nats/ln2 · positions/characters`, i.e. the principled form with
`characters-per-token = characters/positions` measured over the *whole scored stream*, which is what the
docstring at `model_metrics.py:9-24` claims and what the tests check. Recomputed against the committed reports:

- E2612 prose arm: positions 254,976, characters 535,282, nats/token 3.1485 → bpc 2.1637 (report
  2.163693). `characters_per_token_test` 2.0993 = 535,282/254,976. **Verified-correct.**
- E2611: positions 996,608, characters 2,148,397 → 2.1557 chars/token, bpc 1.8077 (report 1.8077426492).
  **Verified-correct.**

`characters_scored` and `positions_scored` are therefore mutually consistent by construction — both are read
off the *same* `targets` sequence in the same call (`model_metrics.py:140-142`, `subword_experiment.py:1213`),
so a divergence would require the decoder to be handed a different target list than the loss was, which the
signature does not permit.

---

## Finding 1 — Separator/punctuation is charged once and identically to every arm — Verified-correct

**Observation.** The document separator `DOCUMENT_END` (id 256, `subword.py:67-71`) is emitted after every
document by the packer (`packing.py:173-176`) and is a scored target, but `SubwordVocabulary.decode` skips it
(`subword.py:274-275`), so it costs a prediction and adds zero characters — exactly as `model_metrics.py:21-24`
states. The uniform and unigram baselines are scored through the *same* `metrics_from_scores` with the *same*
`targets` and the *same* decoder (`subword_experiment.py:961-969`), so they inherit the same separator penalty
and the same denominator. The reports prove it: for E2612 both arms show identical
`positions_scored` 254,976 / `characters_scored` 535,282 / `separator_targets` 33 for the model, uniform and
unigram blocks.

**Distortion it prevents.** A subword model could look better than a character model by splitting one
sentence into more tokens; the shared decoded-character denominator is what makes that impossible. A
separate-rule baseline could dodge the boundary cost; the shared function denies it.

**One caveat (minor).** A second, different bits-per-char formula exists in the `[train]` character path:
`evaluation.py:64-70` reports `bits_per_char = loss/ln2`, implicitly one character per token. For a character
vocabulary that holds except that an `UNK`/reserved target decodes to nothing (`subword_experiment.py:515-528`
uses the shared arithmetic instead), so E2601/E2602's `bits_per_char` field (`experiment.py:143`) is not the
same instrument as the subword reports' `bpc`. The records head-line nats for those experiments, so no
published subword-vs-character table mixes the two; but the field is a trap for a future reader.

---

## Finding 2 — Baselines and tokenizer are fit on the arm's own training side, never on test — Verified-correct

**Observation.**
- Unigram: `UnigramBaseline.fit(train_targets, …)` then scored on `test_targets`
  (`subword_experiment.py:965-969`; partition baselines `:1035-1036,1063-1066`; E2601 `experiment.py:77-79`;
  E2602 `controlled.py:144-146`). Laplace α=1.0 stated (`baselines.py:51-66`).
- Uniform: a constant `ln(vocab_size)` (`baselines.py:28-37`), data-independent.
- BPE vocabulary: fitted on `split.train` only, optionally on a seeded *sub-sample of the training side*
  (`subword_experiment.py:636-654`), never on validation/test. Character vocabulary likewise
  `CharVocabulary(tuple(unit.text for unit in split.train))` (`:773`), and `data.fit_vocabulary` is
  `train_text`-only (`data.py:520-526`).
- Packing cap: `max_tokens` truncates *within* a side (`packing.py:166-177`); there is no code path that
  reaches across a side boundary to fill a capped pack.

**Consequence worth naming (not a leak).** The unigram baseline is fit on `prepared.train.examples`, which is
the *capped, shuffled* training pack, not the whole training side. When a run caps training
(`max_training_tokens`), the baseline sees the same capped subset the model does — still no test leakage, and
the model/baseline comparison stays fair, but the baseline is now conditioned on the sample the run happened
to draw. For E2612 the two arms therefore have *different* unigram baselines (prose 3.5678, mixture 3.5030)
because they train on different compositions; the honest cross-arm number is model-vs-model on the identical
held-out sample, which is what the record's H1/H2 rule uses. This is correct, but a "beats the baseline by X"
sentence is per-arm and must not be cross-subtracted.

**Distortion it would otherwise allow.** Test-derived frequencies in the unigram, or a test-only character
widening the vocabulary, would make the model and the baseline both cheaper on held-out data and turn a
memorization artifact into a "learning" result.

---

## Finding 3 — Held-out partitions are deterministic and, when declared, held out by work — Verified-correct with a data caveat

**Observation.** Each side's pack is drawn by `_sampled_units`, a seeded shuffle (`effective_split_seed`) of
that side's documents before the `max_test_tokens` cap (`subword_experiment.py:1011-1021`), then packed by
`pack_units` (`:778-781,1048-1054`). The partition stream is filtered from `prepared.split.test`
(`:1040-1046`), so every partition document is a held-out document; a partition can never contain a training
document. Sides are disjoint whole documents (or whole works) by `split_text_units`/`_split_whole_works`
(`data.py:139-339`), deduplicated by content digest first (`data.py:203-208`).

**Verified against E2612.** Both arms: `split_fingerprint adf650435c3a9068`, `whole_work_splits true`,
`works_on_multiple_sides 0`, groups-per-side 312/56/62, identical `positions_scored` 254,976 → the shared test
sample is bit-identical across arms, so the cross-arm bpc difference is model, not sampling. The register
partitions are held out by *work* because ADR-0042's `whole_work_splits` is on.

**Risk 3a (invisible leak when the flag is off).** `works_on_multiple_sides` is *reported, not enforced*
(`data.py:261-271`). E2608/E2611 ran with document splits and their reports show `whole_work_splits`/`works_on_multiple_sides`
as **absent** (field post-dates them). The within-work leak — 23/23 test books also in training, 100% of test
characters from books the model read other chapters of — was *not visible in the artifact* and was only
recovered by a manual recomputation recorded in E2611's 2026-10-30 correction. A distorted result looks like:
a "generalises to unseen prose" sentence attached to a number that is really within-work continuity, and the
report would not have contradicted it.

**Risk 3b (thin held-out registers).** E2612's `medicine-` partition is **one** held-out work
(`works 1`, 46,592 tokens); the report exposes it but the aggregate is not guarded against a per-register
claim that is really n=1 book. A distorted result looks like a confident "medicine improved 0.16 bpc" read
off a single unseen title.

---

## Finding 4 — Register-partition weighting: a same-named field counts the wrong object

**Observation.** `_test_partition_metrics` writes a top-level `"characters"` = sum over **all** matched
held-out documents (`subword_experiment.py:1060`), but only ~250k tokens of that population are actually
scored; the scored denominator lives nested at `metrics["characters_scored"]`. They differ by an order of
magnitude per register (prose `characters` 12,414,225 vs `characters_scored` 539,534; mathematics
1,125,529 vs 510,290). E2612 declares its technical aggregate "weighted by each partition's scored
characters."

**Risk.** Weighting the H1 technical aggregate by the natural-looking top-level `characters` versus the
correct `characters_scored` gives materially different magnitudes from the *same* artifacts:

- by population characters: prose arm 2.1946, mixture arm 2.0081 → Δ ≈ −0.19 bpc
- by scored characters:       prose arm 2.3922, mixture arm 2.1164 → Δ ≈ −0.28 bpc

The sign (H1 true) is stable, but the published **magnitude shifts by ~0.09 bpc — ~4.5× the declared 0.02
materiality band** — depending only on which "characters" the writer grabs. This is a concrete instance of Q7
(a metric attributable to a different object than the record says). The `works` field has the same
population-vs-scored ambiguity.

**Recommendation.** When E2612's results section is written, aggregate on `metrics["characters_scored"]` and
rename or drop the partition top-level `characters` so it cannot be mistaken for the denominator.

---

## Finding 5 — Corpus fingerprint and snapshot verification — Verified-correct

**Observation.** The manifest fingerprint is a digest over (schema, name, version, normalization, document
count, and each document's *path + SHA-256 of normalized text*) (`manifest.py:251-264`); line-ending
normalization to `\n` precedes hashing (`manifest.py:71-78,424-435`) so CRLF/LF checkouts give one
fingerprint. `read_snapshot` runs the trust order in `manifest.py:681-777`: the snapshot's self checksum,
then the embedded manifest checksum, then **every document body re-hashed against the manifest digest**
(`:388-395`, `CorpusManifest.from_document` with `texts=`), then document-set equality against the manifest
path set, then the manifest fingerprint against the stored one. A run never consumes bytes the manifest does
not vouch for.

`--expected-corpus-fingerprint` is enforced before any step: `prepare_corpus` raises `CorpusError` when the
resolved `manifest.fingerprint` differs from the declared one (`subword_experiment.py:747-756`), and a seed
study *refuses to construct* if the plan omits it (`seed_study.py:161-166`). A snapshot and its directory's
pinned manifest that name different corpora raise `CorpusDriftError` (`subword_experiment.py:575-582`).

**Atomic / refuse-to-clobber.** `write_snapshot`, `save_manifest` and `write_report` all raise when the
destination exists unless `allow_overwrite` is passed (`manifest.py:552-567`,
`subword_experiment.py:1388-1393`), and route through `write_text_atomically` which guarantees the bytes on
disk are the bytes the digest was computed over (the reason the JSON is written `indent=2` in binary-safe
mode — `subword_experiment.py:1380-1384`). `save_artifact` has the same guard plus a rebuild-before-write
check (`nn_checkpoint.py:221-255`).

**Risk 5a (guard is opt-in).** Nothing *requires* `--expected-corpus-fingerprint` for a single
`experiment model` run (`cli.py:2267` passes it straight through and it may be `None`). A run that omits it
trains on whatever snapshot the work-directory resolves, then records the actual fingerprint. So a *record
hand-written* with a stale fingerprint would not be contradicted automatically — only the report's own
`corpus.fingerprint` would name the truth. A distorted result looks like: prose in the record, a different
corpus in the artifact, and no failure; the defense is reading `corpus.fingerprint` from the report, not from
the prose.

---

## Finding 6 — Seeds are comprehensive; CPU bit-reproducibility is tolerance-based, not guaranteed

**Observation — where seeds enter.**
- Model init: `torch.manual_seed(config.seed)` **only when building fresh** (`nn_training.py:578-580`); a
  resume loads weights and restores Adam moments instead.
- Batch order: `plan_batch` is a pure function of `f"{seed}:{epoch}"` (`nn_training.py:289-319`), so a
  resumed run draws exactly the batches an uninterrupted one would — the reason resume is equivalence.
- Split: `effective_split_seed` (`subword_experiment.py:433-436`, `data.py:139-339`).
- Tokenizer sub-sample shuffle: `random.Random(effective_split_seed)` (`:646`).
- Training-unit shuffle and val/test sampling shuffle: same seed (`:727-729`, `:1019-1021`).
- Generation: recorded runs use `temperature=0.0` (`:1179`; probes `generation_eval.py:78`), i.e. greedy →
  seed-independent. Sampled quality configs carry an explicit seed (`quality_eval.py:105-114,406`).

**Risk 6a — no deterministic-algorithm / thread pinning.** There is no
`torch.use_deterministic_algorithms(True)` and no `torch.set_num_threads()` anywhere in the training path
(grep: only `manual_seed` exists). `describe_environment` **observes** `torch.get_num_threads()`
(`nn_checkpoint.py:371-385`) but never sets it. CPU float reduction order therefore depends on the thread
count of the machine; two identical-seed runs on 14 vs 18 threads can diverge numerically. The project
acknowledges this structurally: `compare_with_reference` treats loss as "within tolerance" rather than exact
and skips wall-clock/best-step from the identity set (`seed_study.py:445-499`). So "reproduced exactly,
tensor-for-tensor" claims hold only at a fixed thread count. **Necessary-not-sufficient.** A distorted result
looks like: a reproduction declared failed (or its tolerance quietly widened) purely because the runner had a
different CPU topology.

**Risk 6b — cited digests are not reproducible by re-running.** `write_report` hashes the whole report dict
(`subword_experiment.py:1394-1396`), and that dict carries `wall_clock_s`, `positions_per_second`,
`timings`, `latency`, `resources.peak_resident_megabytes` and per-curve `elapsed_seconds`
(`:1334-1336,1361-1363,1342-1351`). The artifact checksum likewise covers `run_metadata`
(which includes `wall_clock_seconds`/`positions_per_second`, `nn_training.py:279-280`) and `environment`
(includes `torch_threads`) (`nn_checkpoint.py:130-159`). So the report/checkpoint SHA-256 that E2611 and
E2612 cite (e.g. treatment report `bdcb4986…`, checkpoint `27bfb69…`) **cannot be regenerated** on any other
machine or session. The *numbers* are reproducible within tolerance; the *digests* are identities of one
particular execution. The `write_report` docstring's promise ("a reader who regenerates the report can compare
hashes without reading every number", `:1378-1381`) overstates what the digest actually guarantees. A
distorted result looks like: a faithful re-run reported as "does not match the record" because its hash
differs only in a timing field.

---

## Finding 7 — Reported test metrics are the validation-selected step, not the final step — Verified-correct

**Observation.** `train_transformer` tracks `best_state`/`best_step` under the declared patience/min_delta
rule, reloads `best_state` into the model before returning, cross-checks its incremental pick against the
pure `choose_validation_best` and **refuses to return if the two disagree**, then hands `state=best_state` to
the outcome (`nn_training.py:610-624,711-714,746-770,751-764`). `run_experiment` scores the test and each
partition from `outcome.artifact().to_model()` — i.e. the selected step — and reports validation and test as
separate blocks (`subword_experiment.py:1163-1169`). The character baseline re-runs to `best_step` and scores
those parameters, reporting `final`/`selected` distinctly (`:893-913`). Early-stopping selection reads only
`validation_examples`; `test` is never passed into the loop (`nn_training.py:530-544`).

**Why it can look confusing (not a defect).** E2608/E2611/E2612 arms all select `best_step = max_steps`
because validation is still falling at the ceiling (`completion budget_ceiling`, `stopped_early false` in all
four reports), so best == final and the two would be indistinguishable. That coincidence does not generalize:
for a converged run (E2605 selected step 1380) the test number is the step-1380 model, and the record says
"the un-stopped model was worse," which this code path supports. **Verified-correct**, and the self-check at
`nn_training.py:754-764` is the guard.

**Risk 7a — summarise() conflates sample denominators.** `summarise` prints `samples_refused` from
`generation.automated_summary`, which counts refusals across continuation **and** edge probes
(`generation_eval.py:376`, used at `subword_experiment.py:1429`). For E2611 the single "refused" is the
`empty-prompt` **edge** probe behaving to contract (`ModelError: prompt formatting received no message with
content`), while all five continuation samples generated. A distorted read looks like: "the model refused to
generate" when in fact the designed empty-prompt refusal is what tripped the counter. The per-sample blocks
disambiguate it; the one-line summary does not.

---

## Finding 8 — Generation/quality gates measure what they say, but are necessary-not-sufficient by design

**Definitions (from code, cross-checked against E2611's report).**
- `looping_fraction` — share of *continuation* samples whose **tail** is one block repeated ≥3 times;
  `detect_loop` inspects only the tail, `min_repeats=3` (`generation_eval.py:222-247`, `:361-362,384`). E2611:
  5/5, looping_fraction 0.0.
- "tail-looping" = `looping_samples` (same). "cycling" = `periodic_run_samples`, a ≥3 **consecutive** repeat
  *anywhere* via `longest_periodic_run` (`:250-277,379-383`). E2611: none (max run 2).
- `mean_bigram_repetition` — mean over continuation samples of the share of character bigrams already seen;
  `None` (<4 grams) is excluded, never flattered to 0 (`:157-180,385`). E2611: 0.654745.
- `refused` = a boundary `ModelError`; `crashed` = any other exception (`:299-306`). A model that declines to
  continue nothing is a refusal, not a defect.

**How they can pass misleadingly (acknowledged, guard is the label not the arithmetic).** A phrase-level
loop ("…the house and the king's room. I was going to see the door…") is highly repetitive yet contains no
≥3 *exact consecutive* block and no tail cycle, so it scores `looping_fraction 0.0, periodic 0` while
`mean_bigram_repetition 0.655` and quality distinct-1 0.721 (vs human 0.853) say the opposite. Both suites
carry `interpretation_limit` inside the payload so the number cannot be separated from the disclaimer
(`generation_eval.py:52-56,393`; `quality_eval.py:457-461`), and the record states "less repetitive ≠ usable."
**Verified-correct** as an instrument; the risk is a headline that drops the band, not a computation error.

**Quality gates two-sided, and one empty-input edge.** `assess_gates` fails a config whose distinct-n sits
*above* the human band too (`over_diverse`, `quality_eval.py:483-493`), so a repetition penalty cannot buy a
pass by being unnaturally novel (E2611 shows `greedy_rp1.2` and `t1.0` failing as over-diverse).
`looping`/`periodic`/`replacement` gates default a `None` aggregate to `0.0` and thus pass
(`:494-508`), but `all_passed` additionally requires every distinct-n to be *defined and in band*
(`:509-511`), so a degenerate all-refused/tiny-output config cannot pass (E2611 `greedy` fails all three
distinct gates). This closes the empty-pass hole. **Verified-correct.**

**Minor.** `quality_eval` `bound_fraction 0.0` for greedy (24 held-out prompts) coexists with the generation
suite's `all bound-reached` (5 fixed probes) — different prompt sets, both greedy. Not a contradiction, but a
reader comparing the two tables should know the inputs differ (`quality_eval.py:428-431` vs
`generation_eval.py:97-126`).

---

## Finding 9 — Retrieval/quality-side experiment machinery is harness, correctly fenced — Verified-correct (context)

`experiments/runner.py`, `compare.py`, `strategies.py` measure BM25/dense/hybrid and the correction policy over
synthetic fixtures: the result identity excludes the timestamp (`runner.py:52-61,128-129`), environment is
recorded separately from metrics so a host change does not move `result_id`, and `compare_results`/`compare_retrieval_strategies`
refuse across differing dataset id/version/content-fingerprint, configuration or evaluator set
(`compare.py:144-161`, `strategies.py:150-161`). These are labelled development instruments; per the AGENTS.md
contract they are **never** evidence of language-model competence, and the code makes no such claim. They do
not touch the model bpc path. **Verified-correct** as scoped.

---

## What I did not verify

- Bit-level CPU reproduction across thread counts: *not verified* (no deterministic-algorithm call; would
  require running arms on a different topology).
- Whether the committed `results.json` aggregates for E2608/E2611 exactly equal the git-ignored `data/` reports
  end-to-end: *partially verified* (spot-checked the E2611 control block against the E2608 transformer report —
  parameters 1,037,056, bpc 1.9043151128, split fingerprint `df603dc033c015dd`, tokenizer `b597eb34f13d9058` all
  match). Full aggregate equality: *not verified*.
- Whether `near_duplicate_groups` (`dataset_report.py:158-204`) found any cross-work near-duplicate in the
  Gutenberg corpora that could straddle a whole-work split: *not verified* — the data cards should state
  `pairs_examined`; near-duplicate removal is a **reporting** step, and the split only drops *exact* content
  duplicates (`data.py:203-208`), so a cross-work near-duplicate is not guarded in the training path.

## Top risks, ranked by potential to distort a published number

1. **F4 partition weighting ambiguity** — `characters` vs `metrics["characters_scored"]`; ~0.09 bpc swing on
   E2612 H1 from the artifacts as written. (`subword_experiment.py:1060`, `:1040-1067`)
2. **F3a within-work leak invisible when `whole_work_splits` off** — the field was absent from E2608/E2611
   reports; caught only by manual re-computation. (`data.py:261-271`)
3. **F6b cited report/checkpoint digests embed timing/threads** — not reproducible; contradicts the
   regeneration promise, though not the numbers. (`subword_experiment.py:1394`, `nn_checkpoint.py:130-159`)
4. **F5a `--expected-corpus-fingerprint` is opt-in for single runs** — mismatch caught only by reading the
   report, not by a mandatory guard. (`cli.py:2267`, `subword_experiment.py:747-756`)
5. **F3b thin held-out registers** — E2612 medicine n=1 work; per-register claims need n stated.
6. **F6a no thread/determinism pinning** — "exact reproduction" holds only at a fixed `torch` thread count.
7. **F7a `summarise` conflates edge and continuation refusals** — one-line reads overstate a model refusal.
8. **F8 repetition gates pass on phrase-level loops** — necessary-not-sufficient; mitigated by the
   in-payload `interpretation_limit` and the two-sided distinct bands.
9. **F1 caveat: two different bpc formulas** (character `loss/ln2` vs shared total/characters) — only the
   nats-headlined E2601/E2602 are affected; do not mix the fields in one table.
