# Hypotheses

Each hypothesis is written so that it could be shown to be false, and each names
what would falsify it. A hypothesis that cannot fail is a preference.

None of these has been tested. None has supporting data. They are listed because
they determine what gets built: a hypothesis that nothing in the repository is
designed to test is a hypothesis that will never be tested.

---

## H1 — Explicit evidence tracking reduces unsupported claims

**Hypothesis.** Tracking which evidence supports which claim, as structured state
rather than as prose the model produces, reduces the rate of unsupported claims
compared with asking the same model to be careful in a prompt.

**Falsified if.** On a fixed question set with a fixed model and retrieval
configuration, unsupported claim rate is not lower with structured evidence
tracking than with prompt-level instruction alone.

**Why it matters.** This is the project's central engineering bet. If prompt
instructions achieve the same result, the architecture is unnecessary overhead.

**Depends on.** Claim decomposition, claim-level support judgement, and a question
set with evidence where support is known. **Amended 2026-10-24:** all three exist. Claims are a
first-class unit (`chakaso.claims`, ADR-0014), support is judged per claim structurally (`chakaso.grounding`,
ADR-0015) with a semantic boundary whose only judge is a caller-supplied fixture (ADR-0019/ADR-0027), and
E2603's workload is a labelled question set with gold evidence
([`../../experiments/E2603/record.md`](../../experiments/E2603/record.md)). The condition this hypothesis
actually needs is still absent: it compares structured tracking against **the same model asked to be careful
in a prompt**, and no model in this repository answers a grounded question, so the prompt-instruction arm
cannot be run. What has been measured is the mechanism, not the comparison.

---

## H2 — A dedicated reassessment path reduces unjustified persistence

**Hypothesis.** Separating reassessment from generation — taking the previous
answer and new evidence as explicit inputs and classifying claims against the
evidence — reduces unjustified persistence below what a single model pass achieves
when asked to reconsider.

**Falsified if.** Unjustified persistence rate is not lower with the dedicated path
than with single-pass reconsideration, at equal rates of unnecessary revision.

**Why it matters.** Correction is the project's second central claim. A model that
agrees with every challenge would score well on persistence and badly on
unnecessary revision, which is why both metrics are required.

**Depends on.** A correction benchmark with both contradiction and non-contradiction cases. **Amended
2026-10-24:** the cases exist — `chakaso.benchmark.correction` carries six (retain-supported,
correct-contradiction, qualify-lost-support, review-unsettled-conflict, abstain-total-grounding-loss and an
irrelevant-evidence control) and a runner that scores the decision rule against them. What this hypothesis
still lacks is a challenge corpus that came from an interaction rather than from this file: the existing cases
score a deterministic policy, which is harness evidence, not evidence about a model's persistence under
correction.

---

## H3 — Small models with strong retrieval can produce useful grounded answers

**Hypothesis.** A small local model can produce answers that a user finds useful
when retrieval quality is high, even though its parametric knowledge is weak.

**Falsified if.** On the grounded-answer benchmark, answer usefulness does not
exceed a retrieval-only baseline (returning ranked evidence with no generated
answer) by a margin a reader would consider meaningful.

**Why it matters.** It determines whether the project should spend its effort on
the model or on retrieval. If retrieval-only is nearly as good, that is a finding
worth publishing rather than a failure.

**Depends on.** A retrieval implementation, a usefulness judgement procedure, and
a baseline. **Amended 2026-10-24:** retrieval exists (lexical BM25, dense and hybrid over an in-memory
corpus), a trained model exists and is loadable through the boundary, and the grounded-answer workload of
E2603/E2604 is a labelled question set with gold evidence. What does not exist is the two thirds of this
hypothesis that make it testable: a usefulness judgement that is not a structural citation check, and a
retrieval-only baseline to compare a generated answer against. The models measured so far produce no usable
answer to be judged, so the hypothesis is untestable here for a reason that is not the missing instrument.

---

## H4 — Retrieval does not remove the need for uncertainty behaviour

**Hypothesis.** Adding retrieval to a model reduces, but does not eliminate,
unsupported claims; appropriate abstention remains a distinct behaviour that must
be built and measured separately.

**Falsified if.** With retrieval in place, unsupported claim rate on
insufficient-evidence cases falls to a level where abstention training adds no
measurable improvement.

**Why it matters.** It is the reason uncertainty is treated as its own evaluation
layer rather than as a property of retrieval. The hallucination survey cited in
[`literature.md`](literature.md) reports that retrieval mitigates but does not
solve the problem; that is background, not evidence about this system.

**Depends on.** A retrieval implementation and an abstention benchmark. **Amended 2026-10-24:** both exist in
development form — lexical/dense/hybrid retrieval over an in-memory corpus, and E2603's unanswerable cases,
which measured that retrieval returned false evidence for **3 of 3** abstention questions and surfaced passages
its own fixture had marked forbidden (K-010). E2604 added a deterministic sufficiency decision that judged all
three unanswerable cases insufficient while wrongly abstaining on none. What is still missing is the model arm:
uncertainty behaviour is a property of what generates the answer, and nothing here generates one.

---

## H5 — Interaction logs can be used as training data without teaching self-imitation

**Hypothesis.** Logged interactions, filtered by measured outcome (supported
claims, successful corrections, appropriate abstentions), can be used as training
data without the model learning to reproduce the mistakes that remain in the log.

**Falsified if.** A model trained on unfiltered logs performs no worse — or
performs better — on unsupported claim rate than one trained on outcome-filtered
logs.

**Why it matters.** It decides whether the feedback loop in the roadmap is
realistic or whether it quietly degrades the model while appearing to improve it.

**Depends on.** An evaluation harness that can label outcomes, a trained model to
train further, and enough logged interactions to matter. **Amended 2026-10-24:** the first two now exist —
`chakaso.benchmark.grounded_*` labels claims, citations and grounding per answer (E2603/E2604), and there is a
trained model to continue-train, released at Level 2 (`chakaso-base@0.1.0`, with
[ADR-0037](../decisions/ADR-0037-reference-model-sized-by-measured-throughput.md)'s engine supporting resume
under a longer budget). The third does not: no logged interactions exist, because no one has used the system,
so the hypothesis remains untestable and this line is the reason the roadmap's feedback loop is a phase rather
than a plan.

---

## H6 — Tokenizer choice measurably affects small-model quality

**Hypothesis.** Vocabulary size and segmentation rule (BPE versus Unigram) change
the measured quality of a small model on a fixed corpus and evaluation, by more
than run-to-run variation at a fixed seed.

**Falsified if.** Quality differences between tokenizer configurations fall within
seed-to-seed variance.

**Why it matters.** It determines whether tokenizer work is a research variable or
a settled engineering default. If it is not a research variable, the project
should stop treating it as one.

**Depends on.** A tokenizer, a model, a corpus, and enough compute to run
configurations more than once. **Amended 2026-10-24:** four of the five prerequisites exist — a learned
tokenizer (`chakaso.training.subword`), trained models, four pinned corpora, and a seed-study runner that has
already run a protocol five times over one corpus (E2606) and twice more over another (E2607, E2608). What is
missing is the comparison this hypothesis names: it contrasts **BPE with Unigram**, and only BPE is
implemented. The character-versus-subword gap E2605 measured (3.1343 against 1.8657 bits per character) is
about segmentation, but it moves identifier space, model architecture and training budget together, so it
cannot be read as H6's answer. A tokenizer-arm study at fixed architecture, fixed context and fixed token
budget — the design rule E2608 wrote down after its own confound — is what would make this testable.

---

## H7 — A real dense (semantic) retriever exceeds lexical retrieval on paraphrased evidence

**Hypothesis.** A genuine embedding model, dense retrieval over it, retrieves paraphrased gold
evidence that lexical BM25 misses, raising recall@k above the lexical baseline on a corpus whose
gold chunks match the query in meaning but not in surface tokens.

**Falsified if.** On such a corpus, dense recall@k is not above lexical's — or any gain disappears
once gold chunks share no surface token with the query.

**Why it matters.** It is the research reason to build dense and hybrid retrieval at all. If a real
embedding does not beat lexical on paraphrase, the added machinery earns nothing and the project
should stop treating dense retrieval as a path to better grounding.

**Depends on.** A real (semantic) embedding model and a paraphrase-rich gold corpus. Neither exists.
The fixture embedding built in [ADR-0023](../decisions/ADR-0023-dense-retrieval-boundary.md) is
explicitly non-semantic, so the retrieval-strategy comparison run today
([research-log E-001](research-log.md)) does **not** test this hypothesis — over the fixture it
showed no gain and a small MRR regression. H7 remains untested.

## Hypotheses the 2026-11-01 campaign adds, with what would falsify each

Written after E2612 rather than before it, and each stated so that a single affordable run could overturn it.
None has been tested; none is a restatement of H1–H7.

**H8 — E2612's prose cost is exposure, not mixture.** *Statement:* at a fixed 20,000,000-identifier budget, an
arm that sees ~10M prose identifiers and nothing else scores its prose partition within 0.02 bits per character
of an arm that sees ~10M prose plus ~10M technical identifiers. *Why it is live:* E2612 declared in advance that
its 20M cap covers ~80 % of the prose pool but ~42 % of the mixture pool, so the control saw roughly twice the
prose of the treatment — the measurement now shows the pool is 50.51 % prose by characters, and the prose
partition differs by 0.03026, only 1.5× the band. *Falsified by:* a third arm (P½) landing materially better on
prose than the mixture arm at equal prose exposure. *Cost:* one 90-minute run, no new machinery.
*Status 2026-11-01:* **falsified as worded** by [E2613](../../experiments/E2613/record.md) (outcome H8-b): the prose-only arm at ~10M identifiers scored prose 1.80621, 0.03331 below the mixture arm's 1.83952, and 0.00305 below the 20M prose arm's 1.80926. The arm made about twice the mixture's prose passes, so the finding is that a step budget partly spent on technical text costs prose, not that technical text harms prose; a one-pass arm would separate the two: preregistered as [E2616](../../experiments/E2616/record.md). *Update 2026-11-02:* E2616 is **measured, outcome R-a with its short-schedule flag**: the one-pass arm scored prose 1.93944, 0.0999 *worse* than the mixture's 1.83952 (five times the separable margin), so halving the updates is a sufficient explanation of the whole gap and the arm cannot separate passes from updates; the composition question stays open. One seed.

**H9 — The mixture has an interior optimum in prose share.** *Statement:* held-out bits per character over a
mixed register corpus is minimised at some prose share strictly between 0 % and 100 %, and that share is
measurable with a handful of points. *Why:* E2612 moved composition once, at one point (≈42 % of the pool
reachable), and got −0.27584 technical against +0.03026 prose. The trade-off exists; its shape does not.
*Falsified by:* a monotone sweep whose optimum sits at a declared boundary. *Status 2026-11-01:* **restated and preregistered** as [E2615](../../experiments/E2615/record.md): "minimised" needs a weighting, and the repository's whole test pack weights the registers by an accident of one seeded sample (prose 41.9 %, science 45.3 %, technology 5.8 %, mathematics 7.0 %, medicine 0 %), so the primary objective is the equal-weight mean of the five register partitions, judged against the seed noise E2614 measured; the grid is 0/25/50/75/100 % with E2612-P as the 100 % endpoint; no arm has run. *Status 2026-11-02:* **measured as H9-c, unresolved** ([E2615](../../experiments/E2615/record.md)). The equal-weight mean is lowest at 25 % prose (2.00981) but 0 % and 50 % are within 0.02 of it (0.00911 and 0.01181), so 0 %, 25 % and 50 % cannot be told apart; 75 % and 100 % are separably worse. Prose improves and every technical register worsens with prose share; no arm dominates another. No interior optimum is shown, and none is refuted. One seed per arm; the 100 % endpoint is E2612 arm P's committed numbers from another machine. *Cheap because:* Shukor et al.
(arXiv:2507.09404) report that only 10–20 runs are needed to fit mixture scaling laws — a claim about *their*
scale that this project should test rather than assume. *Status 2026-11-02 (later):* **resolved on O1 as H9-a′** by [E2618](../../experiments/E2618/record.md): five seeds on each of the 0, 25 and 50 % arms put the 25 % arm lowest (2.00928 against 2.01858 and 2.02339) and ordered against both (0.0093 and 0.0141 bits per character, Bonferroni intervals clear of zero); 0 % and 50 % are unresolved from each other. The minimum is interior, small (under 1 % of O1), and a statement about the equal-weight convention: the whole-pack column and a prose-weighted objective name other shares. 25 % is the candidate working mixture, not yet a replacement for the natural mixture.

**H10 — Budget extension on a fixed pool does not close the truncation gap.** *Statement:* raising the step
ceiling at a fixed 20M-identifier pool leaves held-out loss above what a same-token run on 20M *unique*
identifiers reaches. *Why:* every artifact so far completed `budget_ceiling` with validation still falling, so
"train longer" looks free — and Muennighoff et al. (arXiv:2305.16264) report that up to 4 epochs of repeated data
yields negligible change versus unique data, while Xue et al. (arXiv:2305.13230) make degradation depend on
dataset size and parameter count. E2612's arms sat at **0.98 epochs**. *Falsified by:* an epoch-sweep whose held-out
loss keeps falling at 4 epochs at the same rate it fell from 0 to 1. *Status 2026-11-02:* **first measured point** from [E2619](../../experiments/E2619/record.md) (prose only, one seed, fixed 2,400 updates): reading a 10M pool about twice scores the same on held-out prose as reading a 20M pool once (1.8057 against 1.8093, a tie), while reading a 5M pool about four times scores 0.011–0.015 bits per character worse (1.8205). Repetition up to two passes is free here and four is not; this is not the budget-extension experiment H10 names (that varies updates over a fixed pool) and does not answer it.

**H11 — Context 256 is too long for a 20M-token budget.** *Statement:* at fixed parameters and tokens, a
shorter context reaches lower held-out bits per character than 256. *Why:* Shi et al. (arXiv:2502.01481) ran a
six-layer transformer over 12M/25M/50M-token slices and conclude that "deploying expansive context windows on
constrained datasets guarantees suboptimal convergence" — the most scale-appropriate source the campaign found,
and it points the opposite way to the roadmap's assumption that context should grow. *Falsified by:* a matched
64/128/256 sweep where 256 wins outright.

**H12 — Vocabulary 4,096 is not the binding constraint at 3.4M.** *Statement:* sweeping the fitted vocabulary
over {2k, 4k, 8k, 16k} **with total parameters held fixed** (narrowing the feedforward to pay for a larger table)
moves held-out bits per character by less than 0.02. *Control it needs (found 2026-11-02):* the held-out packs are capped in identifiers counted with each arm's own vocabulary, so a sweep over vocabulary sizes would score each arm on different documents; set `evaluation_selection_tokenizer_file` (ADR-0048) so every arm is scored on the same text. *Why:* the published optimum grows with model size
(γ ≈ 0.83, arXiv:2407.13623) and would put 33M near 37K+, but 4,096 identifiers already consume 1,048,576
parameters — 30.7 % of this model — so the published direction and this budget collide, and no source resolves it
below 10M. *Falsified by:* any arm outside the band.

**H13 — Document-boundary masking earns its cost at this context length.** *Statement:* enabling
`document_masking` lowers held-out bits per character on the shared pack at matched tokens. *Why:* the mechanism
exists ([ADR-0040](../decisions/ADR-0040-document-boundary-attention-masking.md)) and has never been run; packing
concatenates documents inside a 256-position window, so attention currently crosses document boundaries.
*Falsified by:* a masked arm indistinguishable from its unmasked control — a plausible outcome given that
E2612's documents are ~4 KB chapter slices and a 256-token window rarely spans a boundary. *Note:* the audit
found masking applied in training and validation loss but **not** in the held-out scorer
(`nn_evaluation.py:72` against `nn_training.py:688-694`). That asymmetry is fixed (2026-11-01, ADR-0040
amendment, ACTIVE_TASK T5), so the hypothesis is testable; it has still not been run.

**H14 — Technical gains concentrate where control loss is largest.** *Statement:* across register partitions, the
improvement from adding a register is monotonically related to that partition's control bits per character. *Why:*
E2612's one observation fits it — mathematics, the partition with the highest control loss (3.04361 against
1.99522–2.18647), improved most (−0.46584). *Falsified by:* any sweep where a well-modelled register gains more
than a poorly-modelled one. *Limit:* one experiment, one corpus, and the partitions differ 5× in held-out support.

**H15 — A measured record cannot be re-trained without its git-ignored cache.** *Statement:* with the tokenizer
and corpus caches removed, a re-run of E2608 or E2612 either fails or produces a different artifact. *Why:* the
campaign proved the sharp half of this — refitting E2608's train side under the pre-fix loop yielded
`a1ef04f64739cd4b` instead of the recorded `b597eb34f13d9058`, and after the fix E2612's own vocabulary
`434debe82f949347`, fitted by the defective loop, can no longer be re-fit at all. *Falsified by:* a clean re-run
from a fresh clone that reproduces the recorded fingerprints. *This one is testable in an hour and decides whether
tokenizer caches belong in the repository.*
*Status 2026-11-01:* the decision half is made. E2608's vocabulary refits exactly from current code; E2612's cannot,
so it is committed as an input ([ADR-0044](../decisions/ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md))
and a run loads it with `--tokenizer-file`. The clean-clone re-run itself has **not** been done, so the hypothesis is
neither confirmed nor falsified.

**H16 — Orthogonalised updates on the hidden 2-D weights beat AdamW at matched steps, beyond seed noise, at 3.4M
parameters.** *Statement:* training the E2612-T configuration with the Muon update on its hidden-layer matrices (AdamW
for the embedding table, norms and scalars) lowers held-out bits per character by more than the seed noise E2614
measures. *Why it is live, and why it is only a hypothesis:* Moonlight reports "~2× computational efficiency compared to
AdamW with compute optimal training" ([Liu et al., arXiv:2502.16982](https://arxiv.org/abs/2502.16982), first version
2025-02-24, a 3B/16B-parameter MoE on 5.7T tokens), and Keller Jordan's write-up of the optimiser
([2024-12-08](https://kellerjordan.github.io/posts/muon/)) reports a 1.35× speed-record improvement on a NanoGPT
FineWeb task and runs to 774M and 1.5B parameters, with the embedding, output head and scalar/vector parameters left to
AdamW. *Nothing read tests it below 774M parameters or at a 20M-token budget.* At Chakaso's shape the embedding table
is 1,048,576 of 3,413,760 parameters (30.7 %), so Muon would govern only the remaining ~69 %. *Falsified by:* an arm whose
whole-pack bits per character is within 2.77·s of the AdamW arm's, where s is E2614's sample deviation (whole pack 0.00337, so 2.77·s = 0.0093, and 0.0268 at the interval's upper end). *Why it is not
being built:* the noise floor now exists (E2614) and that reason is gone; the others stand: a new optimiser is an interface change that needs a decision
record, and the repository's rule is to implement only what its evidence justifies; the sources read are two orders of
magnitude above this scale. *Cost if pursued:* ~40 lines of PyTorch plus one ~90-minute arm.

**H17 — A space-prefixed pre-tokenizer lowers held-out bits per character at equal positions, because a third of the committed
vocabulary's identifiers are whitespace.** *Statement:* a vocabulary fitted on the same sample at the same 4,096 identifiers with a
pre-tokenizer in which a word piece carries its single preceding space (the convention of GPT-2's byte-level BPE, Radford et al. 2019)
reaches lower held-out bits per character than the committed vocabulary, at the same positions per step, the same parameters and the
same updates, by more than E2614's seed noise. *Why it is live (measured, 2026-11-02):* with the committed vocabulary
(`434debe82f949347`), 34.1–39.4 % of the identifiers in a 1.5M-character sample of each register decode to whitespace only, and
30.2–35.6 % are a single space, because the pre-tokenizer `\w+|\s+|[^\w\s]` makes every whitespace run its own piece and merges never
cross a piece. Characters per identifier are 1.99–2.31. *Arithmetic, not a measurement:* if each single space were folded into the
word after it, 30–36 % of identifiers would disappear and characters per identifier would rise by roughly 43–55 %, so the same 19.7M
positions would read that much more text and a 256-position window would span that many more characters. Whether that lowers bits per
character is the question: a single space is cheap to predict but not free (it is the word-boundary decision), and a larger share of
distinct word forms would compete for the same 4,096 identifiers. *Falsified by:* an arm within 2.77·s of the control on the whole
pack (0.0093; 0.0268 at the interval's upper end) or worse. *Controls it needs:* the same 12.7M-character fitting sample and
`min_frequency` 3; bits per character as the unit (so the tokenizer change does not move the metric); an arm at equal *characters* as
well as at equal positions, because the two readings answer different questions; the committed vocabulary's merge loop is the one
fixed by K-024. *Cost if pursued:* a new vocabulary schema (the loader already refuses a foreign pre-tokenizer, so this is a new
named pre-tokenizer and a decision record), one exact fit (35 minutes), and one or two ~50-minute arms. *Measured by:*
`scripts/measure_token_classes.py` (the sole measurement above; it needs no training). *Status 2026-11-02 (later):* **measured, one seed: V-a** ([E2620](../../experiments/E2620/record.md)). The fitted vocabulary spends 5.8–8.6 % of identifiers on whitespace (from 34–39 %) and gets 2.93 characters per identifier on held-out text (from 2.10); at equal updates and positions the equal-weight register mean falls 2.0207 → 1.9943 and the whole pack 2.0555 → 2.0235, every partition improving. The gain confounds a better tokenizer with 40 % more text per update; [E2624](../../experiments/E2624/record.md) holds the text fixed. Not yet a replacement for the default. *Status 2026-11-03 (later):* **E2624 and E2625 separate the confounds.** At equal text read with 30 % fewer updates the arm is worse by 0.0351 (W-b), but that is the updates: E2625 (the control's 2,400 updates over the control's own documents, 1.40 passes) gains **0.0229** on O1 and **0.0286** on the whole pack, X-a. 87 % of E2620's gain survives with 25 % less distinct text, and 30 % fewer updates cost 0.058 bits per character at this scale. One seed, one vocabulary, one budget; the treatment's seed spread, a refit sensitivity check and a decision record are what replacing the default would need. *Status 2026-11-03 (later):* the second seed is in flight as [E2628](../../experiments/E2628/record.md) — E2625's design at seed 11 on the Windows host, paired with E2614's seed-11 control on the same machine, with its confound (seed and host move together, bounded by E2617's measured 0.0025) declared in the record before the run.

**H18 — a capability probe's separations track held-out loss monotonically along a training curve.** *Statement:* for rungs of one training curve, ordered
by held-out bits per character, a probe that separates a pair with the better-loss model favoured will separate every pair whose loss gap is larger. *Why it
is live:* it is the assumption underneath every sentence of the form "the loss improved, and the probe agrees" — E2622's withdrawn arm ordering and E2623's
ceiling both rest on a probe being a monotone function of what the model learned. *Falsified by:* one pair where the probe significantly favours the
worse-loss model. *Status 2026-11-03:* **falsified, once, at one seed** ([E2626](../../experiments/E2626/record.md)). On the three usable phenomena of the
minimal-pair probe, rungs 1600 and 1800 of E2614's seed-7 curve separate significantly in the direction the loss does not favour (0.8800 against 0.8667-style
non-monotonicity, with the 1600/1800 pair at 24 discordant items to 2 and p = 1.0 × 10⁻⁵, while the loss prefers 1800 by 0.0233 bits per character). The
continuation probe produced no contradiction; its failure is a different one — it resolves no pair below 0.294 bits per character and none at all among the 32
material pairs whose rungs both lie past update 800. *What this does not establish:* that the contradiction reproduces across seeds or between independently
trained arms, which is the comparison the project actually makes; E2626's own limits section says rungs are not arms. *What would follow from confirming it:*
held-out loss and grammatical preference are different measurements of this model at the 0.02-bit scale, so a loss improvement cannot be reported as a
capability improvement without the capability number being measured on the arms themselves.
