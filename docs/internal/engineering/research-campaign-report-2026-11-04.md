# Research campaign report — 2026-11-04, the evidence-and-decision phase

Scope: the campaign brief of 2026-11-03 (`bf55e12` was `main` when it started) named K-032, H17 with ADR-0048, K-030
with ADR-0049, K-031, K-001, stronger capability measurement, and engineering for training efficiency, orchestration,
checkpoint integrity and automated verification. It asked for **at least 100 meaningful commits, pushed progressively**.

**This report states what was done, what was measured, what failed, and what was not reached. The commit target was not
reached: 35 commits landed, all of them verified, none manufactured. The reasons are in §7, and the honest number is in
§6.**

---

## 1. Initial state, measured rather than recalled

`main` == `origin/main` == `bf55e12`. 32 experiment records on disk (E2601–E2608, E2610–E2629), all measured.
Baseline on this machine at the session's first run: **2,085 pass / 34 skip** dependency-free. 18 logical CPUs, no CUDA
device, 15.4 GB RAM, ~153 GB free on `D:`. A 3,413,760-parameter arm at 2,400 updates costs 3,045–3,536 s here when the
machine is otherwise idle.

### Audit findings that changed the plan

Reading the implementations rather than the documentation produced five findings the brief's premises did not assume.

1. **K-032 was worse than "evidence is bulky."** `scripts/extract_report.py` replaced each per-item outcome array with
   the SHA-256 of its canonical JSON, because 1.45–2.28 MB of rows do not fit the 1 MiB tracked-file bound. The
   consequence was concrete and already paid for once: E2629 had to re-score four models because the extract it wanted
   to re-pair could not be re-paired. The bulk was the *item identity repeated per model*, not the verdicts.
2. **H17 had two seeds, not the four the brief assumed, and its refit half did not exist.** E2628 (**Y-a**) is one
   treatment arm on one host plus one control on another; ADR-0048's "does not decide" clause was still the only
   decision record. No tool for refit sensitivity existed.
3. **K-030 item 3's arm was genuinely unrun.** ADR-0049 had made `weight_decay_excludes_vectors` expressible; E2626's
   seed-7 run was the control and needed no re-run — which meant the arm was one training job, not a study.
4. **The whole-pack number and the register table are different text.** Found while building the paired test:
   `max_test_tokens` caps a *seeded sample* of a side, so E2612 arm T's headline 2.0554 is measured over 34 documents
   containing **no medicine at all**, while O1 — the primary objective of six records — gives medicine a fifth of its
   weight. Filed as K-033, closed later the same day (§4).
5. **A default in the library was load-bearing in the artifact format.** `nn_checkpoint` writes
   `tokenizer_pre_tokenizer` only when it differs from `DEFAULT_PRE_TOKENIZER`, and a reader uses
   `DEFAULT_PRE_TOKENIZER` when the key is absent. So adopting a better pre-tokenizer by moving the default would have
   silently re-tokenised every released checkpoint. Nothing in the brief or ADR-0048 said so; the tests proved it (a
   first attempt at the move made the registry collapse to one entry and the committed `434debe82f949347` vocabulary
   refuse to load).

## 2. Sources, and how the three kinds were kept apart

The brief asked for published evidence, local measurements and hypotheses to be separated. They are, in the records:

* **Published practice** is used only as *rationale for building an option*, never as a result. GPT-2's byte-level
  pre-tokenizer (a word piece carrying its preceding space) is why the space-prefixed rule exists at all; AdamW's paper
  and Llama/Gemma-style recipes are why norm-and-bias decay exemption was made expressible. No new external citation was
  introduced this phase, and no published number is quoted as if it were measured here.
* **Local measurement** is anything in this report with a record behind it: 33 records now, five of them written by this
  phase, each with committed machine-readable evidence and each preregistered before its run.
* **Hypotheses** stay in `docs/research/hypotheses.md` with their status lines. H17 is now confirmed at this scale and
  adopted by ADR-0051; H18 (a probe tracks loss monotonically) remains falsified once at one seed; the probe-ranking
  question E2634 bounded is not a hypothesis, it is a measured ceiling.

## 3. Implementations

Thirteen `feat:` commits, each with tests that fail against the previous code:

| Area | What landed | Why it is there |
| --- | --- | --- |
| Statistics | `student_t_quantile` (regularised incomplete beta, bisection), `mean_confidence_interval`, `sign_test_p_value` | A four-draw paired comparison needs 3.182, not 1.96; the difference is the whole content of an "interval excludes zero" claim |
| Loss measurement | `nn_evaluation.position_losses` / `document_losses` on one forward-pass path, `chakaso.training.paired_loss` (character-weighted mean, work-clustered bootstrap, leave-one-out), `subword_experiment.work_losses`, `scripts/paired_loss_test.py` | Every arm difference in the series had been read against a threshold. This is the test, and the tool refuses until each arm reproduces its own committed denominators |
| Evidence | `chakaso.training.outcome_vectors` (one bit per item, one shared item listing, digest of the array replaced), extractor and reader wiring, three refusal paths | K-032, closed by re-running the reading that had been refused |
| Sample disclosure | `packing.target_segments` / `sample_composition`, the report's `evaluation_sample`, `work_losses.composition` | K-033: a number whose denominator is not legible cannot be audited |
| Provenance | `chakaso.training.corpus_archive` + `chakaso corpus archive\|verify-archive\|extract-archive` | K-031's mechanism: a provider edits its files, so the raw bytes are a separate artifact from the snapshot |
| Comparability | `decay_applied_to` declarable in `arm_comparison`; `--equal-aggregate` computing O1 | A comparison of two arms differing in decay could only refuse or lie; and the tool's `o1` was not the objective's O1 |
| Instruments | `scripts/measure_refit_sensitivity.py`; `--expected-items-digest` / `--expected-pair-set-digest` on the three probe runners | H17's refit half; and a declared item set that the runner enforces instead of the author remembering |
| Tokenizer decision | `LEGACY_DEFAULT_PRE_TOKENIZER`, the `load_plan` declaration gate, the moved `DEFAULT_PRE_TOKENIZER` | ADR-0051, and the two things that make the move non-silent |
| Machine facts | `process_memory.available_memory()` → `resources.available_at_launch` | Cost paragraphs were typed from an operator's recollection of a task-manager window |
| Documentation gates | census-vs-directories and census-vs-parser; record status vs its own sections | K-013 asked for the mechanism after four hand-fixes of the same drift |

## 4. Measured outcomes

| Record | Outcome | The number |
| --- | --- | --- |
| [E2630](../../../experiments/E2630/record.md) | **Y2-a** | Space-prefixed tokenizer at E2614's five same-host seeds: Δ(O1) **+0.026431**, 95 % Student-*t* **[+0.023212, +0.029650]**, every draw positive; treatment deviation **0.001595** against the control's 0.002240 |
| [E2631](../../../experiments/E2631/record.md) | **N-b** | Vector-decay exemption vs E2626 seed 7: Δ(O1) **−0.003499** (tie 0.0062), Δ(whole pack) **−0.003606** (tie 0.00933); paired interval [−0.014208, +0.001682], 13:12 sign test p = 0.50000. K-030 item 3 closed |
| [E2632](../../../experiments/E2632/record.md) | **R-a** | Four fitting samples × two rules, eight distinct vocabularies: within-sample ratios **1.405175 / 1.409317 / 1.409002 / 1.405079**, relative spread **0.30 %**; merge-list Jaccard 0.109 with rank agreement 0.741/0.810 |
| [E2633](../../../experiments/E2633/record.md) | **L-a** | First paired test on held-out loss: material pairs [−0.209, −0.037] and [−0.217, −0.039]; inside-band pair [−0.0149, +0.0026] includes zero; one book moves −0.108 to −0.072 |
| [E2634](../../../experiments/E2634/record.md) | **G-a**, qualified | 745 continuation items: `context_pmi_permuted` separates both material pairs (p = 0.013718, 0.003658) — and its untrained floor is **0.3812**; floor-passing readings' best p = 0.031539 against a 0.016667 step; `easy` saturated at 0.9879 vs 0.9758 |

Issues closed by measurement: **K-032** (evidence recomputes — all eight of E2629's instrument rows reproduced
byte-identically from E2627's committed bytes), **K-033** (the sample is stated in the report; E2612 arm T's pack
recounted to 254,976 positions / 25 of 62 works / medicine 0 %, matching E2633's manual read), **K-030 item 3**, and the
checkable half of **K-013**.

## 5. What failed, including this campaign's own work

* **E2631's P2 and P3 failed.** The point estimates favour the *control*, not the exemption; and the validation curves
  cross at steps 600 and 1,200 and exceed ±0.01 bits per character at 400 and 2,200. Recorded as failing.
* **E2632's P2 failed on its first half**, and the failure is the useful finding: a 12M-character fit does not beat a
  4M fit at 4,096 identifiers (2.126855 against the best small fit's 2.126521). Compression is saturated at the
  identifier count, not the sample size.
* **E2634's P5 failed** in the useful direction: the separations' discordant items spread over 24 and 26 of the pack's
  works, no work above 12 % — unlike E2633's loss effect, which one book carried a third of. **P2 half-failed**: two
  readings put ρ at +0.500 with 3.47× the items, where E2629 at 215 had ranked all three arms correctly.
* **E2633's reproduction command could not have run** as committed (four options the script never had, the required
  `--protocol` missing). Corrected, then executed: byte-identical `results.json`, SHA-256
  `03d8f17f1a41ecbb4909581acfe9554733b0f962eadd645bc65047a23e6eb345`.
* **The first reconstruction of a held-out pack was wrong** — 256,256 positions where every report says 254,976, and
  1.2 bits per character worse on models that had already published their number. Caught by the tool's own guard; it
  produced `work_losses` and then the `evaluation_sample` block.
* **Two cost estimates were an order of magnitude off in the same direction.** E2632 budgeted 65–75 minutes from
  K-024's per-run figure and took 459 s for eight fits; E2634 budgeted 10–35 minutes and took 2 m 33 s. Records quoting
  K-024's fit costs should re-measure them inside one process.
* **The report's own first commit failed CI, and the reason is worth keeping in the record.** Its five record links
  pointed at `../../experiments/…` from `docs/internal/engineering/`, which is one directory short. Locally the gates
  passed, because `test_markdown_relative_links_resolve` walks the *tracked* files and the report was still untracked
  when they were run — so a new document is unchecked until it is staged. The fix is procedural: stage the file, then
  run the gates. This paragraph is in the phase's own report rather than somewhere less visible because the failure mode
  (a gate that silently does not see the thing you just wrote) is exactly the class of thing K-013 and the two census
  gates were built against.
* **The default move broke the rule registry** on its first attempt (both rules collapsed onto one key, the committed
  vocabulary refused to load). Eleven test fixtures that had been reading their identifier counts off an ambient default
  were rewritten to name the rule they mean; none was weakened, and the suite is why this was caught in minutes.

## 6. Verification

**Local, at the final head.** 2,103 pass / 34 skip dependency-free; 2,191 / 25 with `[train]` only; **2,704 pass / 1
skip** with `[train,nn]`; `mypy src` clean over 125 files; `mypy --config-file configs/mypy-model-path.toml` clean over
171. Ruff check and format clean.

**On the runner, read back after the pushes** (gh's `createdAt` is printed in this machine's local time, so run
identifiers and heads are cited instead of timestamps). Eight jobs — core / train / model path on 3.11, 3.12, 3.13, with
the model-path tier also running the strict typing configuration — and every campaign push read back green:

| Head | Run | Result |
| --- | --- | --- |
| `94b1df4` | 41698684010 | success |
| `4b576b7` | 41707179041 | success |
| `da69e06`, `d03f27b` | 41713002168, 41723778521 | success |
| `2905054` | 41730821152 | success |
| `a5cde40`, `08cfc76` | 41733074460, 41733882205 | success |
| `9628877` | 41740408984 | success |
| `09916cb` | 41746935106 | success — core 3.13 2,085/35, train 3.13 2,174/25, model path 3.13 2,678/2 |
| `dc58b09`, `6ccd428` | 41750403143, 41751899095 | success |
| `5cf40a7` | 41759227462 | success |
| `0082c1b`, `0a1b752` | 41761040465, 41763524817 | success |
| `a004e70` | 41800134350 | success |
| `d18c41f` | 41803174650 | success — core 3.13 2,102/35, train 3.13 2,191/25, model path 3.13 2,703/2 |
| `00525e5` | 41805564655 | **failure** — all eight tiers, one cause: the closing report's five relative record links were one directory short (§5). Tests, lint, format and both typing configurations passed on every tier |
| `4c63dae` | 41806985294 | success — the phase's final verified head; 36 commits through it, and the table that states this is that head's own successor |

The runner reports one more skip and one fewer pass than the same head does locally on the dependency-free tier, and the
difference is the documented hygiene test that skips where the private documentation pack does not exist at all.

**Commits: 35**, from `c86c248` to `9c9f916` — the count is pinned to `9c9f916` so it cannot go stale as the report
that states it lands, which makes **this report's own commit the 36th**, the commit that fixed its links the 37th, and
the one recording this verification the 38th. All on `main`, each milestone pushed as it landed rather than held for the
end (the `## Head` column above is the push sequence, and every row was read back from the runner); no force-push, no
history rewrite, no rebase, author identity unchanged, no co-author added. By type through `9c9f916`:
**13 feat, 12 research (5 preregistrations, 5 results, 2 evidence/research-log), 8 docs, 2 test**. Nothing was split to
inflate a count and no documentation-only batch was padded: the eight `docs:` commits each carry a correction or a
reconciliation of a real drift, and two of them (E2633's erratum, E2631's addendum) exist because a measurement
contradicted what had been written.

## 7. What was not reached, and why

* **The ≥100-commit target.** 35 landed through `9c9f916` (36 with this report). The binding constraints were
  verification cost and serialisation, not ideas: every commit in this phase ran the four dependency-free gates (~45 s)
  and, when it touched the model path, the gated tier (~2.5 min) plus strict typing — and the machine has one CPU pool,
  no GPU, and a single-process budget for training arms. Reaching 100 would have meant either splitting coherent changes
  into fragments or landing unverified commits; the brief forbids both, so the count is reported short rather than met
  dishonestly.
* **K-001 — usable generation.** Unmoved, and this phase did not pretend otherwise: no training-budget experiment was
  started, because the brief's own priority list (evidence, tokenizer decision, optimizer decision, corpus provenance,
  measurement) was all answerable on artifacts already on disk, and the two arms that did run cost ~56 minutes each.
  Every model in the repository is still Level 2 of ADR-0039 with Level 3 withheld, and nothing here answers a grounded
  question.
* **K-031's second half.** The archive, verifier and extractor exist and were exercised on the real 430-source corpus
  (162,017,394 raw bytes, 59,220,202 archived, digest stable across two builds, 24 s to build and 3.5 s to verify). The
  **430-URL drift re-scan has not been run** — it is 430 network fetches, which no CI tier can do and which the runtime
  is explicitly not allowed to require — and where an archive is *kept* is the corpus owner's decision, not a repository
  policy this project can make for them.
* **A memory floor that refuses a launch.** Deliberately not built. `available_at_launch` is now in every report;
  a guard that refuses a run below a declared figure is a guess about what the machine needs until the figure has been
  observed failing, and no run in this phase died that way.

## 8. Next, in order

Mirrored in `docs/agent/ACTIVE_TASK.md`'s ledger:

1. The first record that **names the adopted pre-tokenizer in its own protocol** and quotes E2630's measured deviation
   (0.001595) instead of inheriting E2614's band from the rule it replaced.
2. **A work-clustered statistic for paired verdicts** — E2633's bootstrap ported from loss to binary item outcomes —
   with K-035's work identity in the item rows landing first, so the claim is recomputable from the committed artifact.
3. **K-031's drift re-scan** as a sampled measurement (a named subset of sources, digests compared, the method stated)
   rather than 430 silent fetches, plus the storage decision.
4. **K-034**: the probe analysis takes its outcome vocabulary from the record rather than embedding another record's.
5. **K-001**: the budget question E2608 left — a run that converges, or an explicit statement that convergence is out
   of reach at this scale — which is the only item on this list that needs a training arm rather than an afternoon.
