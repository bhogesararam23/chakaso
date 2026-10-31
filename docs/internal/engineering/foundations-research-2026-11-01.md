# Foundational literature research — 2026-11-01

**Status: Research.** This is a reading of other people's primary sources, not a measurement by this
project. Nothing here is a Chakaso result. Every number below was either read from a fetched source in this
pass or is arithmetic done here on numbers read in this pass, and each is labelled with the reading that
produced it. Where a fetch failed or a paper could not be located, that is written as *could not verify*.

**Vocabulary** (per [`../../research/README.md`](../../research/README.md)): Observation / Hypothesis /
Experiment / Result / Interpretation / Limitation / Conclusion.

**Confidence scale used on every finding.**

| Label | Meaning |
| --- | --- |
| High | The claim was read from the primary source's own text or verbatim abstract in this pass |
| Medium | Only bibliographic metadata or an abstract summary was read; the mechanism is from the title |
| Low | Recalled and could not be verified in this pass — never a basis for a decision |

**Provenance classes used on every finding.** *Authors' reported* = what the paper's own experiments found.
*Independently re-analysed* = a different group re-derived it from the same or extracted data.
*Independently replicated* = a different group re-ran the training. *Folklore* = widely repeated, no
primary source read here.

**Reference configuration this is read against** (E2608 arm A / E2611 / E2612,
[`../../../experiments/E2612/record.md`](../../../experiments/E2612/record.md)): 3,413,760-parameter rotary
decoder, `d_model` 256, 4 layers, 8 heads, feedforward 640, tied embeddings, RoPE θ = 10,000, RMSNorm, GELU,
context 256, 4,096 BPE identifiers, AdamW lr 6e-4, weight decay 0.01, 200 warmup steps, cosine to 0.1 of
peak, batch 32 × 256 = 8,192 positions/step, 2,400 steps = 19,660,800 positions, float32, CPU, measured
4,493.7 positions/s at the smoke run, host 18 logical cores / 16.6 GB RAM. Tokens per parameter:
19,660,800 / 3,413,760 = **5.76**.

---

## 1. Scaling laws

### 1.1 Kaplan et al. 2020 — what it says and what its own authors bound

**Source:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott
Gray, Alec Radford, Jeffrey Wu, Dario Amodei, "Scaling Laws for Neural Language Models",
<https://arxiv.org/abs/2001.08361>, v1 23 January 2020. Full text read at
<https://ar5iv.labs.arxiv.org/html/2001.08361>. **Confidence: High** (body text read; individual quotations
were returned by the fetcher under a length cap, so some are short fragments rather than whole paragraphs).

*Observation (Result, authors' reported).* "The loss scales as a power-law with model size, dataset size, and
the amount of compute used for training, with some trends spanning more than seven orders of magnitude."
Reported exponents: `αN ∼ 0.076`, `αD ∼ 0.095`, `αCmin ∼ 0.050`; compute-optimal allocation
`N ∝ Cmin^0.73`, with `B ∝ Cmin^0.24` and `S ∝ Cmin^0.03`.

*Observation.* The abstract's prescription is "optimally compute-efficient training involves training very
large models on a relatively modest amount of data and stopping significantly before convergence" — i.e.
the truncated-loss-curve state Chakaso is in is, under this paper, the *expected* shape of an
compute-optimal run, not a defect.

*Observation (stated limits, verbatim as read).* Three bounds matter here: architecture has "minimal effects
within a wide range", and "Only models with fewer than 2 layers or with extreme depth-to-width ratios deviate
significantly from the trend"; the laws "break down at or before we reach this point, which is still many
orders of magnitude away"; and the fitted exponents "vary by an order of magnitude in either direction"
depending on how the fit is made. On learning-curve extrapolation: "fits are best late in training, when the
loss may be approximately quadratic", and the relations are "only valid in the power-law region of the
learning curve after initial transient effects".

*Interpretation.* Kaplan's 0.73 exponent says: at a fixed budget, put the compute into parameters, not
tokens. That is the opposite of Chakaso's current posture (small model, low tokens/parameter), and it is the
exponent that Hoffmann's paper later set out to overturn.

*Limitation.* The fetched text of Kaplan et al. gave me no statement that its laws may be projected downward
to arbitrary smallness; what it does say is that the trends span "more than seven orders of magnitude", that
they "break down at or before we reach this point, which is still many orders of magnitude away", and that the
exponents "vary by an order of magnitude in either direction". The specific lower bound of the model sizes they
trained was **not read in this pass**, so it is not stated here.

*Conclusion.* Kaplan's law is a description of a fitted range, with the authors' own caveat that it breaks
down outside it. **Confidence: High.**

### 1.2 Hoffmann et al. 2022 (Chinchilla) — the ratio, precisely

**Source:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland,
Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich
Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre, "Training Compute-Optimal Large Language Models",
<https://arxiv.org/abs/2203.15556>, v1 29 March 2022. Body read at
<https://ar5iv.labs.arxiv.org/html/2203.15556>. **Confidence: High** for the sentences quoted; see the ratio
note below.

*Observation (Result, authors' reported).* Scope: "By training over 400 language models ranging from 70
million to over 16 billion parameters on 5 to 500 billion tokens". Fitted form
`L̂(N,D) = E + A/N^α + B/D^β` with `E = 1.69`, `A = 406.4`, `B = 410.7`, `α = 0.34`, `β = 0.28`. Compute-optimal
allocation: "We find that a = 0.50 and b = 0.50" (`N_opt ∝ C^a`, `D_opt ∝ C^b`), and the three estimation
approaches give (a,b) = (0.50, 0.50), (0.49, 0.51), (0.46, 0.54), while "Kaplan et al. report a = 0.73 and
b = 0.27". Rule as stated in the paper: "for every doubling of model size the number of training tokens
should also be doubled"; "both quantities should scale at roughly the same rate".

*Observation (the ratio).* The sentence "approximately 20 tokens per parameter" **was not found** in the
fetched rendering — the phrase "tokens per parameter" does not appear in the text I read. The ratio is
nonetheless directly readable from the paper's own Table 3, which I did read: a 400M-parameter model
requires "1.92e+19 FLOPs and 8.0 Billion tokens", a 1B model "1.21e+20 FLOPs and 20.2 Billion tokens", a 10B
model "1.23e+22 FLOPs and 205.1 Billion tokens". 8.0B/400M = 20.0; 20.2B/1B = 20.2; 205.1B/10B = 20.5.
**So the ~20 tokens/parameter figure is arithmetic on the paper's own table, and it is constant precisely
because a ≈ b ≈ 0.5.** Confidence: High for the table entries; the ratio is derived here, not quoted.

*Observation (validity at small N, as written).* Two sentences bound the use of the fit. "projections from
very small models lead to different predictions than those from larger models." And: "While there is
significant uncertainty extrapolating out many orders of magnitude, our analysis clearly suggests that given
the training compute budget for many current LLMs, smaller models should have been trained on more tokens to
achieve the most performant model." Also: "Due to the cost of training large models, we only have two
comparable training runs at large scale". Their optimizer also moved with scale: "We use a maximum learning
rate of 2×10⁻⁴ for the smallest models and 1.25×10⁻⁴ for the largest models" (Appendix D.1).

**What the literature predicts for Chakaso's point, and what it does not license.**

- *Prediction if the relation is transported:* at N = 3.41M, a 20-tokens-per-parameter frontier implies
  D_opt ≈ 68.3M tokens ≈ 8,340 steps at 8,192 positions/step. At the measured 4,493.7 positions/s that is
  ~4.2 h of CPU training. **Confidence: High in arithmetic, Low as a prediction** (see the next line).
- *Not licensed:* Chakaso's N is 20× below the 70M smallest model Hoffmann trained and ~118× below the 400M
  smallest entry of Table 3, and its D (19.66M tokens) is ~250× below the 5B smallest token count in the same
  corpus of runs. Both axes are outside the fitted box, and the paper's own sentence says small-model
  projections disagree with large-model ones. Therefore the ~20 figure is a *prior to be tested*, not a
  target with authority. What the literature does license is directional and weaker: at fixed budget,
  spending more tokens on this model is expected to pay, because a = b ≈ 0.5 means the ratio is flat in C
  within the fitted range.
- *Not licensed at all:* any claim about *when* Chakaso's loss would converge, any claim that 20 tokens/param
  is Chakaso's optimum, and any use of `A/N^α + B/D^β` with Hoffmann's coefficients to predict Chakaso's
  absolute loss. Those coefficients were fitted to Pile cross-entropy on 70M+ models; plugging 3.41M and
  19.66M into them is exactly the extrapolation the authors warn against.

### 1.3 The critique and the re-analyses

*Observation (Independently re-analysed).* Tamay Besiroglu, Ege Erdil, Matthew Barnett, Josh You, "Chinchilla
Scaling: A replication attempt", <https://arxiv.org/abs/2404.10102>, v1 15 April 2024 / v2 15 May 2024. Full
text read at <https://arxiv.org/html/2404.10102v2>. Their abstract: "We find that reported estimates are
inconsistent with prior methods, fail at fitting the extracted data, report implausibly narrow confidence
intervals compared to actual trial counts. In contrast, our rederivation using the third approach yields
results compatible with prior methods." What I read in the body: they re-extract Hoffmann's Figure 4 data,
fit `α = 0.3478`, `β = 0.3658`, obtaining a compute-optimal allocation exponent of ~0.513 — "corresponding to
approximately twenty tokens per parameter"; they attribute the failure of Hoffmann's Approach 3 to the
published procedure averaging Huber losses and terminating the optimizer prematurely, aggravated by rounded
published coefficients; and they argue the published confidence bands would require roughly six hundred
thousand independent trials to justify, against a few hundred actual runs. Their scope is identical to
Hoffmann's: "seventy million to over sixteen billion parameters trained on five to five hundred billion
tokens", and they make **no** statement about reliability below that. **Confidence: High.** Verdict: the
*criticism is of the reported uncertainty, not of the ratio* — a re-derivation from extracted plot data lands
on ~20 tokens/parameter.

*Observation (Independently re-analysed, defending).* Rylan Schaeffer, Noam Levi, Andreas Kirsch, Theo
Guenais, Brando Miranda, Elyas Obbad, Sanmi Koyejo, "Evaluating the Robustness of Chinchilla Compute-Optimal
Scaling", <https://arxiv.org/abs/2509.23963>, 28 September 2025. Abstract read verbatim: they find "relative
differences between different interpretations of model parameters as high as 15.2 %" (the embedding-counting
ambiguity), that the choice "do[es] not meaningfully affect key results: the scaling law estimates and the
compute-optimal tokens-to-parameter ratio", and that "Chinchilla's key results withstand sizable
perturbations", with sensitivity concentrated in "additive or systematic errors". Reported ratio: "≈20 tokens
per parameter". Reported data: "400 models ranging from 44M to 16B parameters on 5B to 500B tokens". Note the
44M: under an embedding-inclusive count, the smallest Chinchilla model is 44M — still ~13× Chakaso.
**Confidence: High.**

*Observation (the reconciliation).* Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, Yair
Carmon, "Resolving Discrepancies in Compute-Optimal Scaling of Language Models",
<https://arxiv.org/abs/2406.19146>, v3 read 28 October 2024. They identify "three technical misalignments in
earlier work: exclusion of the final decoding layer from FLOP counts, excessively long initial learning rate
phases, and failure to adjust optimizer configurations per scale. Fixing these realigns predictions with the
Hoffmann model." Their evidence base is "over 900 training runs" spanning "55M to 901M" parameters; they
"consistently measure an exponent near zero-point-five", which "inherently produces a stable data-per-
parameter proportion across scaling tiers"; and they show that with a long fixed warmup "models are
suboptimally trained" before decay, and that "compute-capped trials rarely reach the decay phase". Their
concrete rule: "set the number of warmup tokens to be identical to the model size N". **Confidence: High.**
This is the most directly actionable of the scaling-law papers for Chakaso (§2.4, §5.4).

*Observation (inference-aware extension).* Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle,
"Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws",
<https://arxiv.org/abs/2401.00448>, v3 14 April 2025. Abstract read verbatim: they "conduct our analysis both
in terms of a compute budget and real-world costs and find that LLM researchers expecting reasonably large
inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal"; and "we
train 47 models of varying sizes and parameter counts to validate our formula and find that model quality
continues to improve as we scale tokens per parameter to extreme ranges (up to 10,000)"; and crucially "we
ablate the procedure used to fit the Chinchilla scaling law coefficients and find that developing scaling
laws only from data collected at typical token/parameter ratios overestimates the impact of additional tokens
at these extreme ranges". Body sentences read: "very small models, trained on enough data, can match larger
Chinchilla-style ones". **Confidence: High.** Implication: Chakaso's product is an inference-heavy local
runtime, which in this framework argues for staying small and spending the budget on tokens — the opposite
direction from Kaplan, and the same direction as its measured 5.76 → 20 gap. Authors' reported; I found no
independent replication of the 10,000-tokens-per-parameter claim in this pass.

### 1.4 "Beyond neural scaling laws" — correct attribution

**The paper the task brief associated with Besiroglu is not his.** Ben Sorscher, Robert Geirhos, Shashank
Shekhar, Surya Ganguli, Ari S. Morcos, "Beyond neural scaling laws: beating power law scaling via data
pruning", <https://arxiv.org/abs/2206.14486> (v1 29 June 2022, v6 read). Abstract read verbatim: "in theory
we can break beyond power law scaling and potentially even reduce it to exponential scaling instead if we
have access to a high-quality data pruning metric that ranks the order in which training examples should be
discarded"; "we performed the first large-scale benchmarking study of ten different data pruning metrics on
ImageNet. We find most existing high performing metrics scale poorly to ImageNet, while the best are
computationally intensive and require labels for every image"; they contribute "a new simple, cheap and
scalable self-supervised pruning metric that demonstrates comparable performance to the best supervised
metrics". **Confidence: High.** *Limitation for Chakaso:* the evidence is vision classification (ResNets on
CIFAR-10, SVHN, ImageNet), not autoregressive text; it licenses the hypothesis that a good pruning metric
steepens the data-scaling slope, and nothing about language.

*Besiroglu et al., "Scaling Laws by Deriving Optimal Hyperparameters" and "Compute MiniModels": **could not
verify.*** I searched for both titles several ways (including through an author-publication page and the arXiv
title index) and never obtained an arXiv ID for either; the hits I got were the Meta hyperparameter-optimizer
paper (2609.01431, quoted in §2.4) and unrelated astronomy entries whose IDs I had guessed. Neither title is
cited in the rest of this document. "Bespoke Models" *is* locatable as a record: "Establishing Task Scaling Laws via
Compute-Efficient Model Selection: Bespoke Models as a Case Study" appears at
<https://arxiv.org/abs/2412.04403> (title and ID from search results; **authors and abstract not read →
Confidence: Low; do not use**). "Loss-to-learn": a title search returned no matching entry in this pass —
*could not verify.*

---

## 2. Loss still descending at the step ceiling

Chakaso's state (from E2608/E2611/E2612 records): both arms hit 2,400 steps with the curve still falling and
`stopped_early: False`. The literature has four separate things to say about this.

### 2.1 Truncation at the horizon is not necessarily a bug

Kaplan's abstract (read, §1.1) prescribes "stopping significantly before convergence" for compute-optimal
training, and the compute-optimal interpretation is that a loss curve still descending at the budget end is
what an optimally budgeted run looks like. **Confidence: High** as a reading of the sentence; **Low** as an
instruction for Chakaso, because Chakaso's ceiling is a wall-clock convenience, not a declared compute-optimal
budget. The honest statement is that Chakaso does not know whether it is at the frontier or below it, and
that is exactly the question a scaling probe answers (§2.4).

### 2.2 The schedule is what makes "still descending" comparable across runs

*Observation (Authors' reported, High).* Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, Tengyu
Ma, "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective",
<https://arxiv.org/abs/2410.05192>, 7 October 2024. Mechanism as stated: a sustained high step size drives
progress along flat directions while oscillating perpendicularly; a rapid final annealing "suppress[es] these
deviations, uncover[ing] true convergence", whereas cosine "continuously compress[es] updates". Reported
numbers from the body: the decay window is fixed at ten percent of total steps and is robust — the result
"matches tightly within the range of 8% to 12%"; decaying immediately after the early transient dip costs
roughly two hundredths of validation score; maximum divergence from an independently tuned cosine baseline is
6e-3, and the advantage over cycling schedules reaches 4e-2; models span 0.1B–1.2B parameters at 50B, 100B
and 200B tokens. Practical consequence as stated: "intermediate states captured during uninterrupted training
reliably approximate independently tuned long-horizon runs" — the "elastic mid-tuning" property.

*Observation (Authors' reported, High).* Siqi Wang, Zhengyu Chen, Teng Xiao, Zheqi Lv, Jinluan Yang, Xunliang
Cai, Jingang Wang, Xiaomeng Li, "Scaling and Transferability of Annealing Strategies in Large Language Model
Training", <https://arxiv.org/abs/2512.13705>, 5 December 2025. They define the annealing horizon as "the
fraction of total training steps spent in the decay phase", find optimal ratios "highly transferable" with
"smaller models can serve as reliable proxies", and report for a 1B MoE model a power law in step count with
λ_T ≈ 5.987×10⁵ and α_T ≈ −0.946 — i.e. longer runs want proportionally shorter decay phases — with cross-
horizon prediction error "MAPE (<2%)". Their models: dense 50M–7B, MoE 100M–7B, "3 random seeds". Their
recommendation as read: "monitor loss against training steps rather than data volume for greater stability"
and preset annealing ratios instead of grid-searching.

*Interpretation for Chakaso.* Its cosine-to-0.1-of-peak over 2,400 steps is a schedule whose decay horizon is
tied to the step ceiling. Comparing "loss at step 2,400" across arms is sound; comparing Chakaso's truncated
curve against a *converged* reference, or reading "still descending" as "the model is limited by tokens
rather than capacity", is not established by anything read here. Two cheap design consequences: (a) if the
budget is ever raised, the decay fraction should be re-derived rather than stretched; (b) the loss-vs-steps
curve is the object to model, and both papers above say the shape transfers across horizons — which is the
enabling fact for a minimal probe.

*Limitation.* Both papers train ≥0.1B/≥50M parameters and ≥50B/100B-token-scale schedules. Neither reports a
3.4M-parameter run, so the transfer of their annealing laws down to Chakaso is untested here. Mark: authors'
reported, **not** independently replicated at this scale.

### 2.3 The ceiling that Chakaso will actually hit first is data, not steps

This is the finding that most changes the next decision.

*Observation (Authors' reported, High).* Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao,
Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel, "Scaling Data-Constrained
Language Models", <https://arxiv.org/abs/2305.16264>, v1 25 May 2023 (revised 28 June 2025). Abstract read
verbatim: "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having
unique data. However, with more repetition, the value of adding compute eventually decays to zero", over
experiments "ranging up to 900 billion training tokens and 9 billion parameter models", and they "propose and
empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated
tokens and excess parameters". They also report that "augmenting the training dataset with code data or
removing commonly used filters" extends the usable unique-data budget.

*Observation (Authors' reported, High, and in partial tension).* Fuzhao Xue, Yao Fu, Wangchunshu Zhou,
Zangwei Zheng, Yang You, "To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis",
<https://arxiv.org/abs/2305.13230>, v1 22 May 2023. Abstract read verbatim: repeating pre-training data makes
"the model … susceptible to overfitting, leading to multi-epoch degradation"; "significant factors include
dataset size, model parameters, and training objectives, while less influential factors consist of dataset
quality and model FLOPs"; and "Most regularization techniques do not yield significant improvements, except
for dropout, which demonstrates remarkable effectiveness but requires careful tuning when scaling up the
model size."

*Result of reading the two together, and of the repository's own arithmetic.* The two papers disagree about
how benign repetition is, and neither is settled by a primary source I read. But both make the quantity that
decides Chakaso's next run measurable, and Chakaso's records already contain it: E2612 caps the packed pool
at **20,000,000 training identifiers** and trains **19,660,800 positions**, i.e. **~0.98 epochs**; the
training loop (`src/chakaso/training/loop.py:198-203`, `nn_training.py:299-317`) iterates epochs over that
pool with a per-epoch seeded shuffle and wraps, so **any step increase beyond 2,460 repeats data**. At
4 epochs the same pool is 80M tokens = **~23.4 tokens per parameter** — which lands on the Chinchilla ratio
Chakaso is currently 3.5× below, at ~4× the wall clock (≈ 4.9 h per arm at the measured 4,493.7 positions/s).

*Interpretation.* Chakaso can move from 5.76 to ~20 tokens/parameter **without new data and without new
architecture**, purely by ~8,340 steps, and the strongest read primary result says the first three extra
epochs are close to free. That is a data-budget experiment it can actually afford.

*Limitation (important).* Muennighoff et al.'s "up to 4 epochs" was measured on corpora of hundreds of
billions of unique tokens and models to 9B parameters; Chakaso has 20M identifiers and 3.4M parameters. Xue
et al. list *dataset size and model parameters* as significant drivers of multi-epoch degradation, which
predicts that Chakaso's tolerance may be **lower**, not equal. Nobody read here measured a 3.4M-parameter
model over 4 epochs of a 20M-identifier pool. Mark: authors' reported, **not** independently replicated at
this scale, and the extrapolation is the opposite direction from the one Hoffmann's authors warn about. The
cheap mitigation is the one Chakaso already has infrastructure for: hold out works, and watch the
train/valid gap per epoch (§6.4 for the mixture-specific confound).

### 2.4 Is a compute-frontier sweep feasible on Chakaso's CPU, and how many runs?

*Feasibility arithmetic (High: all inputs read from the E2612 record; the products are computed here).*
One step = 8,192 positions at 4,493.7 positions/s ≈ 1.82 s/step ⇒ 2,400 steps ≈ 73 min. A 3-point frontier
probe at the *same* N with budgets of 2,400 / 8,340 / 20,000 steps is ≈ 1.2 + 4.2 + 10.1 ≈ **15.5 h**; a
proper (N, D) grid — say three sizes (the choice of sizes is **this document's design, not a prescription from
any source read here**) × three token budgets — is on the order of **60–80 h**
of CPU time. **That grid is an estimate, not a measurement:** throughput per position is only measured for the
current 3.4M configuration (4,493.7 positions/s), and it does not scale linearly with parameter count at these
sizes, so the smaller and larger arms have no measured rate. Days of machine occupancy rather than weeks. So:
yes, a real two-axis sweep is affordable at Chakaso's scale in principle, and it is *not* affordable as a
byproduct of ordinary experiment runs.
[`chakaso-compute-and-scaling-plan.md`](../../research/chakaso-compute-and-scaling-plan.md) is where measured and
estimated rows are kept separate; the numbers added here belong in
that document's estimated rows, not in a record.

*Minimum run count, from sources that actually measured it.*

- **Frontier slope in N** requires runs at differing N *and* differing D with the schedule horizon tied to
  the budget (Porian et al., §1.3: without per-scale optimizer and warmup alignment you get the 0.73-vs-0.5
  discrepancy). Their own study is "over 900 training runs" — the floor implied by that is "many", not "six".
- **Mixture slope** is the cheap case: Mustafa Shukor, Louis Bethune, Dan Busbridge, David Grangier, Enrico
  Fini, Alaaeldin El-Nouby, Pierre Ablin, "Scaling Laws for Optimal Data Mixtures",
  <https://arxiv.org/abs/2507.09404>, 12 July 2025, report "**Only 10-20 runs are needed to fit the scaling
  laws**" (§6.2). **Confidence: High.**
- **Hyperparameter scaling laws** are cheaper still: Zhiliang Chen, Sebastian Ament, David Eriksson,
  Maximilian Balandat, Eytan Bakshy, Jihao Andreas Lin, "Efficiently Estimating Optimal Hyperparameter
  Scaling Laws through Power-Law Entropy Search", <https://arxiv.org/abs/2609.01431>, 1 September 2026 —
  Bayesian optimisation over cheap trials: "< one-tenth of the computational budget", proxy scales "under six
  percent of target workload expenses, with the majority clustering near the two percent threshold",
  "approximately six evaluation steps dramatically shrink confidence intervals", and — directly useful here —
  "targeting uncertainty reduction for learning rate alone automatically allows practitioners to recover an
  accurate scaling law for batch size". **Confidence: High** (body read). Authors' reported; no independent
  replication read.
- **Elastic schedules** let one long run supply several frontier points: Wen et al.'s mid-tuning property
  (§2.2) and Wang et al.'s transferable annealing ratios (§2.2) are the two primary sources for "train once,
  read off several horizons".

*Conclusion.* The design that the literature actually supports at Chakaso's compute is not a Kaplan-style
isoFLOP sweep of 400 models; it is (i) one token-budget ladder at fixed N with the decay horizon re-derived
per run, (ii) a 10–20-run mixture fit, and (iii) ~6 cheap runs at 2–6% of target cost to locate lr.
**Confidence: High** that these are what the sources report; **Low** that any of them yields a *valid* slope
when the fitted range spans 3.4M parameters and 20–80M tokens — that range is narrower in absolute terms than
a single Chinchilla model, and no source read here covers it.

---

## 3. Tokenization

### 3.1 The founding papers (mechanism, correctly attributed)

- **"Neural Machine Translation of Rare Words with Subword Units",
  <https://arxiv.org/abs/1508.07909>** (v5 10 June 2016; ACL P16-1162) — the BPE-for-NMT paper. The arXiv and
  ACL Anthology listings returned by the search show Sennrich and Haddow as the first two authors; the full
  author list was not read in this pass, so it is not reproduced here. *Note the ID:*
  the task brief's usual shorthand `1508.07921` is **wrong** — that ID resolves to a graph-drawing paper
  ("Simultaneous Embeddings with Few Bends and Crossings"), which I fetched and read.
  **Confidence: High** for the corrected ID and title; the BPE algorithm's behaviour in translation (the
  subword split of rare/compound words into a fixed vocabulary built from a training corpus) is what the
  title states, and the BLEU numbers in the paper were **not read in this pass → Confidence: Low** for any
  specific score.
- **Taku Kudo, John Richardson, "SentencePiece: A simple and language independent subword tokenizer and
  detokenizer for Neural Text Processing", <https://arxiv.org/abs/1808.06226>**, v1 19 August 2018. Abstract
  read verbatim: "While existing subword segmentation tools assume that the input is pre-tokenized into word
  sequences, SentencePiece can train subword models directly from raw sentences, which allows us to make a
  purely end-to-end and language independent system"; and "it is possible to achieve comparable accuracy to
  direct subword training from raw sentences". **Confidence: High.** The unigram-LM segmentation algorithm
  itself is in Kudo's next paper, not this abstract.
- **Taku Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword
  Candidates", <https://arxiv.org/abs/1804.10959>**, v1 29 April 2018 (ICLR 2019). Read via the arXiv index:
  the mechanism is treating segmentation ambiguity as noise — probabilistically sampling multiple
  segmentations at training time, with a unigram-LM-based segmentation — for robustness, with gains reported
  as largest for low-resource and out-of-domain settings. **Confidence: Medium** (I read the abstract's
  content through the index summary rather than the verbatim abstract text). *Implication:* subword
  regularization is a *training-time* regularizer Chakaso could add without touching the vocabulary, and it
  is the one read mechanism whose reported benefit concentrates where data is scarce.
- **Byte-level:** the GPT-2 line cannot be cited from arXiv in this pass — I did not verify an arXiv posting of
  Radford et al. "Language Models are Unsupervised Multitask Learners", so I do not cite an ID for it
  (**could not verify**; do not cite `1901.11504` for GPT-2 without reading it). The verified byte-level
  primary source I did locate by ID is **Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin
Raffel, "ByT5: Towards a token-free future with pre-trained byte-to-byte models",
<https://arxiv.org/abs/2105.13626>**, v1 28 May 2021 (TACL 2022) — **ID, title, full author list and date
verified from the arXiv API metadata; the abstract's content and its reported sequence-length cost were not
read in this pass → Confidence: Medium, no numbers quoted.** The Rust attribution sometimes given for ByT5 is
**not supported by the metadata read here.**

### 3.2 Measured evidence on vocabulary size

*Observation (Authors' reported, High).* Sho Takase, Ryokan Ri, Shun Kiyono, Takuya Kato, "Large Vocabulary
Size Improves Large Language Models", <https://arxiv.org/abs/2406.16508>, v1 24 June 2024. Body read:
subword vocabularies of **5k, 10k, 50k, 100k and 500k** with model parameters **fixed at 680 million**,
trained on **one trillion tokens**; reported average accuracy gains relative to the 5k baseline:
**+0.4, +0.8, +1.4, +2.5 points** in English and **+3.3, +6.0, +7.7, +9.2** in Japanese; conclusion as read:
"larger vocabulary sizes lead to better performance", with the gains strongest for generative benchmarks and
**ample training volume**, while **restricted corpora suppress these advantages**.

*Interpretation for Chakaso (this is the honest answer to "is 4,096 a real bottleneck at 3.4M parameters?").*
The only read study that varies vocabulary at fixed model size reports an effect whose size depends on the
token budget — and its smallest model is 680M with 1T tokens. Chakaso has 3.4M parameters and 19.66M tokens,
i.e. the regime that study says *suppresses* the vocabulary advantage. There is therefore **no read evidence
that a 4,096-identifier vocabulary is the binding constraint at Chakaso's point**, and there is a read reason
to expect enlarging it costs embedding parameters (at `d_model` 256, tied, each +1,024 identifiers buys
262,144 parameters ≈ 7.7 % of the current model) more than it buys loss. *This is an interpretation of read
numbers, not a measured result, and it is a hypothesis: the test is a two-arm run at 4,096 vs 8,192 with the
parameter count matched by narrowing the feedforward, not a bare vocab change.*

*Observation (Authors' reported, High — and the reason to prefer bytes-in).* Théo Gigant, Bowen Peng,
Jeffrey Quesnelle, "Decoupling the Benefits of Subword Tokenization for Language Model Training via
Byte-level Simulation", <https://arxiv.org/abs/2604.27263>, v1 read 14 May 2026. Mechanism: rebuild the
benefits of subwords inside a pure byte pipeline — fixed byte-chunking to simulate compression/throughput,
binary boundary markers and subword-distance positional encodings to simulate the morphological prior — so
the two candidate explanations separate. Reported: measured compression **4.75 bytes per token**; the primary
model is a 1.7B LLaMA-3 architecture trained on FineWeb-Edu with a byte input layer carrying ~71M extra
parameters (a virtual 35k-token embedding space); chunking that quadruples samples-per-step gives large
validation-loss gains, and **the 68M-parameter configuration showed an even wider gap than 1.7B** under the
same simulation; optimizing for per-subword cross-entropy or multi-token prediction "delivered negligible
gains or degraded metrics". Their conclusion as read: sequence-length/throughput effects are "a primary,
highly accessible driver of the subword advantage" and small models are **more** sensitive to them.
*Limitation, as they state it:* the interventions were applied only in 50k-step windows before reverting,
synergies between the decoupled variables are excluded by construction, English-centric, and "findings lack
validation beyond the tested parameter scales" (i.e. nothing at 3.4M).

*Observation (Medium).* **Soufiane Hayou, Liyuan Liu, "Optimal Embedding Learning Rate in LLMs: The Effect of
Vocabulary Size", <https://arxiv.org/abs/2506.15025>**, v1 17 June 2025. Abstract read verbatim: a
limitation of μP theory "is the fact that input dimension (vocabulary size in LLMs) is considered fixed when
taking the width to infinity. This is unrealistic since vocabulary size is generally much larger than width
in practice"; they show training dynamics "interpolate between the μP regime and another regime that we call
Large Vocab (LV) Regime", in which "the optimal embedding LR to hidden LR ratio should roughly scale as
Θ(√width) … different from the Θ(width) ratio predicted by μP", validated by pretraining a 1B model.
Relevance: Chakaso has V=4,096 vs `d_model`=256 — a 16:1 vocab-to-width ratio, which is the axis this paper
says μP mishandles, and its embeddings are *tied*, so the embedding matrix is both input and output layer and
both are inside the same weight-decay and lr regime. **Confidence: High** for the abstract's claims; the
experimental magnitudes were not read.

*Not verified.* The "Mixture of Tokens" vocabulary-shrinking line: the ID I expected (2310.12485) resolves to
an unrelated statistics paper, and the paper that actually carries the title "Mixture of Tokens" is
**Szymon Antoniak, Michał Krutul, Maciej Pióro, Jakub Krajewski, Jan Ludziejewski, Kamil Ciebiera, Krystian
Król, Tomasz Odrzygóźdź, Marek Cygan, Sebastian Jaszczur, "Mixture of Tokens: Continuous MoE through
Cross-Example Aggregation", <https://arxiv.org/abs/2310.15961>**, v1 24 October 2023 — whose abstract I read
and which is a continuous-MoE architecture ("assigns mixtures of tokens from different examples to each
expert"; "3x increase in training speed over dense Transformer models in language pretraining"), **not** a
tiny-vocabulary result. The specific claim "a vocabulary of 8 identifiers reaches competitive perplexity" is
therefore **unverified; do not write it**. Likewise "The Tokenization Penalty" (IBM line): two searches
returned no matching primary source — **could not verify**. "Finding the Optimal Vocabulary Size for Neural
Machine Translation" (Thamme Gowda, Jonathan May, <https://arxiv.org/abs/2004.02334>, v1 5 April 2020) was
located with correct ID/title/authors; only its first abstract sentence was read (framing NMT as a
classification problem whose class balance depends on vocabulary size) — **Confidence: Medium**, and I found
no vocabulary-size study at all below 680M parameters.

---

## 4. Data quality and filtering

All numbers below were read this pass unless marked otherwise.

*Observation (High).* **Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell,
Colin Raffel, Leandro Von Werra, Thomas Wolf, "The FineWeb Datasets: Decanting the Web for the Finest Text
Data at Scale", <https://arxiv.org/abs/2406.17557>**, v1 25 June 2024. Body read:
- Yield: the pipeline produces a "**15T-token FineWeb dataset**", versus RefinedWeb's "500B tokens" and C4's
  "172B tokens"; extraction alone retained roughly "36 trillion tokens".
- **Deduplication can hurt.** Per-crawl (per-snapshot) MinHash dedup left "20 trillion tokens"; applying
  global MinHash across all snapshots collapsed the pool to "4 trillion tokens" and "actively hurt results by
  preserving lower-quality older archive segments while discarding viable material". Custom heuristics over
  the baseline filtered corpus gave a "~one percent" relative benchmark jump.
- **A filter that removed data and hurt:** the standalone terminal-punctuation rule "stripped nearly thirty
  percent of vocabulary" and was removed.
- **FineWeb-Edu:** "Llama-3-70B-Instruct to label 460,000 random pages", those labels fine-tuning "a linear
  regressor atop frozen Snowflake-Arctic-Embed-M vectors using 410,000 training samples", a "minimum threshold
  of 3", producing a "1.3-trillion token collection"; training on it moved "MMLU score increases from 33% to
  37%, a relative improvement of approximately 12% … and ARC score goes from 46% to 57%, an improvement of
  about 24%".
- Their own limitation: validation runs were confined to "sub-1.71B parameter models", the corpus is
  Common Crawl only, and "aggressive repetition/whitespace scrubbing likely diminishes programming code
  prevalence".

*Observation (High).* **Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris
Callison-Burch, Nicholas Carlini, "Deduplicating Training Data Makes Language Models Better",
<https://arxiv.org/abs/2107.06499>**, v1 14 July 2021. Abstract read verbatim: datasets "contain many
near-duplicate examples and long repetitive substrings"; "over 1% of the unprompted output of language models
trained on these datasets is copied verbatim from the training data"; deduplication removes e.g. "from C4 a
single 61 word English sentence that is repeated over 60,000 times"; deduplicated models "emit memorized text
ten times less frequently and require fewer train steps to achieve the same or better accuracy"; and
train-test overlap "affects over 4% of the validation set of standard datasets". Code released at
`google-research/deduplicate-text-datasets`. Note this is the paper the brief calls "exact vs MinHash": its
two tools are **Suffix Array** (exact substring) and **MinHash** (near-duplicate) — that mapping is stated in
the paper's method, and I read the abstract rather than the method section, so mark the tool-to-task mapping
**Confidence: Medium**.

*Observation (Medium — metadata only).* **Luca Soldaini, Rodney Kinney, Akshita Bhagia, … Kyle Lo, "Dolma: an
Open Corpus of Three Trillion Tokens for Language Model Pretraining Research",
<https://arxiv.org/abs/2402.00159>**, v1 31 January 2024 (author list read in full; the corpus size is in the
title; the internal filtering-ablation results were **not read**).
**Guilherme Penedo, Quentin Malarticle, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza
Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay, "The RefinedWeb Dataset for Falcon LLM:
Outperforming Curated Corpora with Web Data, and Web Data Only", <https://arxiv.org/abs/2306.01116>**, v1
1 June 2023 (title claims the result; the supporting numbers were **not read** — the "500B tokens" figure
above comes from FineWeb's comparison, not from RefinedWeb itself).
**Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li,
Peter J. Liu, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
<https://arxiv.org/abs/1910.10683>**, v1 23 October 2019 — the T5 paper that introduces C4; **the individual
C4 filter thresholds were not read in this pass**, so no C4 number is stated here.

*Result: what transfers to a 149,000,499-character Gutenberg corpus.*
1. **Nothing in the web-filter literature transfers as a filter.** Every read ablation is about removing
   boilerplate, navigation, code-mix and machine-translated junk from crawled HTML. The Gutenberg pipeline has
   none of those failure modes; importing C4-style thresholds would be copying a solution to a problem this
   corpus does not have. (Chakaso's own E2612 selection already rejects by length and duplicate work:
   "54 download failures, 29 ingest failures, 20 too short, 1 too long, 1 duplicate work".)
2. **Two transferable warnings, both specific.** (a) FineWeb's negative result on global cross-corpus MinHash
   is a general one: aggressive near-duplicate removal across sources of *unequal quality* systematically
   deletes the better material; a Gutenberg corpus has verbatim overlap risk (multi-edition texts, shared
   front matter, project gutenberg headers) that dedup would handle differently. (b) FineWeb's terminal-
   punctuation filter shows a "quality" rule can silently delete a large fraction of the vocabulary. Chakaso's
   `docs/data/gutenberg-multiregister-v1.md` and its acquisition script are where such a rule would live.
3. **The quality-classifier pattern does transfer, with a contract cost.** FineWeb-Edu's mechanism is: a strong
   model rates 460k samples → a cheap regressor over frozen embeddings learns to imitate the rating → the
   regressor filters the whole corpus. That is the highest-value read result in this section for a 150M-
   character corpus, because 150M characters is *small enough to rate exhaustively* with the cheap model. The
   cost: the rater is Llama-3-70B-Instruct. Chakaso's contract forbids importing weights, not importing
   *labels*, so the pattern is available as a **data-provenance question**: a teacher-scored corpus is a
   teacher in the data pipeline and must be recorded as such (§8.3).
4. **No deduplication implementation exists in the training path.** A grep for dedup in `src/chakaso` finds it
   only in the retrieval/identity and benchmark layers; `src/chakaso/training/dataset_report.py` computes a
   sketch-based duplicate *estimate* over candidate texts. There is no exact- or MinHash-level corpus
   de-duplication of training documents, and no read source here shows a loss benefit from dedup at 3.4M
   parameters. **Confidence: High** for the code observation; **Unknown** for whether it would help.

---

## 5. Optimization

### 5.1 AdamW

**Ilya Loshchilov, Frank Hutter, "Decoupled Weight Decay Regularization",
<https://arxiv.org/abs/1711.05101>**, v1 14 November 2017 (v3 verified: ID, title, authors, date). This is the
paper that names AdamW: weight decay applied to the parameters rather than added to the gradient, so it is
not rescaled by the adaptive denominator. **Confidence: High** for the identification; the paper's
generalization numbers were **not read**, so none are quoted. Chakaso uses AdamW with decay 0.01 —
consistent, not justified by this reading.

### 5.2 Learning-rate schedule shape

Read in §2.2: cosine-vs-WSD is not a free choice at Chakaso's budget. The two load-bearing facts are Wen et
al.'s "decay window at ten percent of steps … matches tightly within the range of 8% to 12%" and the ~2e-2
penalty for decaying right after the early transient, and Wang et al.'s finding that the optimal annealing
fraction itself scales with step count (α_T ≈ −0.946 for a 1B MoE). Chakaso's current decay reaches 0.1 of
peak at the 2,400-step ceiling, i.e. its decay fraction is ~98% after the 200-step warmup — it is a *truncated
cosine*, not an annealing phase, and by Wen et al.'s mechanism that is the configuration most likely to be
reporting an optimistic loss and a pessimistic "still descending" signal at the same time. MiniCPM's abstract
(<https://arxiv.org/abs/2404.06395>, Shengding Hu, Yuge Tu, Xu Han, … Zhiyuan Liu, Maosong Sun, v1 9 April
2024; WSD use verified verbatim) states the related design claim directly: it "introduce[s] a
Warmup-Stable-Decay (WSD) learning rate scheduler" and thereby derives a "much higher compute optimal
data-model ratio than Chinchilla Optimal" — i.e. the *schedule*, not the size, is what let a 1B-scale model
consume more tokens per parameter. **Confidence: High** for the quoted abstract phrases; the schedule-
comparison tables were not read.

### 5.3 muP at 3 million parameters

**"Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer",
<https://arxiv.org/abs/2203.03466>** — ID, title and abstract verified through the arXiv API; the author list
was not captured by the fetch that returned the abstract, so it is **not** reproduced here (cite by title and
ID). What was verified verbatim from the abstract: the paper shows that "in … Maximal Update Parametrization
(μP), many optimal HPs remain stable even as model size changes", defines μTransfer — "parametrize the target
model in μP, tune the HP indirectly on a smaller model, and zero-shot transfer them to the full-sized model" —
and reports "1) by transferring pretraining HPs from a model of 13M parameters, we outperform published
numbers of BERT-large (350M parameters), with a total tuning cost equivalent to pretraining BERT-large once;
2) by transferring from 40M parameters, we outperform published numbers of the 6.7B GPT-3 model, with tuning
cost only 7% of total pretraining cost". **Confidence: High** for the abstract; **Low** for any attribution
beyond title and ID.

*Is it usable at Chakaso's point?* Read evidence says the *direction* is available and cheap: every verified
transfer runs from small to large, and Chakaso's target *is* the small model, so the honest use is inverted —
tune at ~0.5–1M parameters and transfer up to 3.4M. **No run time is stated for that proxy: throughput is only
measured at the current 3.4M configuration, and it does not scale linearly with parameter count at these
sizes, so a proxy's cost here is Unknown, not estimated.** Two cautions
straight from sources read: Hayou & Liu (§3.2) show μP's fixed-vocabulary assumption is violated in LLMs and
predict a different embedding-lr rule (Θ(√width) rather than Θ(width)) in the "Large Vocab" regime, which is
Chakaso's regime (V/d_model = 16); and Shen et al., "Power Scheduler: A Batch Size and Token Number Agnostic
Learning Rate Scheduler" (Yikang Shen, Matthew Stallone, Mayank Mishra, Gaoyuan Zhang, Shawn Tan, Aditya
Prasad, Adriana Meza Soria, David D. Cox, Rameswar Panda, <https://arxiv.org/abs/2408.13359>, v1 23 August
2024) report that "the zero-shot transfer from small corpus to large corpus is underexplored", derive a
power-law relation between optimal lr, batch size and token count under WSD that they "demonstrate[d] …
transferability across model sizes", and combine it with μP so that one hyperparameter set works "regardless
of the number of training tokens, batch size, model size, and even model architecture" in 3B dense and MoE
runs. **Confidence: High** for both abstracts. Conclusion: μP is *plausibly* worth ~an hour of Chakaso's
machine, but no read source tests it at 3M parameters or at 20M-token budgets, so it is a hypothesis here,
not a recommendation. Note also that μP changes the parametrization (output-layer scaling, per-layer lr
multipliers), so adopting it is an architecture/interface change, which per AGENTS.md needs a decision record.

### 5.4 Batch size, warmup, clipping, weight decay placement

- **Critical batch.** **Sam McCandlish, Jared Kaplan, Dario Amodei, the OpenAI Dota Team, "An Empirical Model
  of Large-Batch Training", <https://arxiv.org/abs/1812.06162>**, v1 14 December 2018. Abstract read
  verbatim: "a simple and easy-to-measure statistic called the gradient noise scale predicts the largest
  useful batch size across many domains"; "We find that the noise scale increases as the loss decreases over a
  training run and depends on the model size primarily through improved model performance"; and it "provides a
  rough model of the benefits of adaptive batch-size training". Relevance: Chakaso trains at 8,192 positions
  per step on a 3.4M-parameter model that has reached only ~1.69 bits/char — the noise scale is exactly the
  quantity that tells it whether that batch is already past the useful limit. It is measurable from the
  existing loop with two gradient evaluations. **Confidence: High** for the abstract; no read number
  establishes B_crit for a 3.4M model.
- **lr-vs-batch for Adam is not linear.** **Shuaipeng Li, Penghao Zhao, Hailin Zhang, Xingwu Sun, Hao Wu, Dian
  Jiao, Weiyan Wang, Chengjun Liu, Zheng Fang, Jinbao Xue, Yangyu Tao, Bin Cui, Di Wang, "Surge Phenomenon in
  Optimal Learning Rate and Batch Size Scaling",
  <https://arxiv.org/abs/2405.14578>**, v1 23 May 2024. Abstract verbatim: the linear-scaling rule "is not
  applicable to Adam style optimizers"; they "prove that the optimal learning rate first rises and then falls
  as the batch size increases. Moreover, the peak value of the surge will gradually move toward the larger
  batch size as training progresses", verified across CV and NLP tasks. **Confidence: High.** Direct
  consequence: raising Chakaso's batch *and* keeping lr 6e-4 is not automatically better, and the direction of
  the error is known (past the peak, lr should go down).
- **Warmup length.** Porian et al. (§1.3) give the rule "set the number of warmup tokens to be identical to
  the model size N", and identify "excessively long initial learning rate phases" as one of the three causes
  of the Kaplan/Chinchilla exponent disagreement. Chakaso: N = 3.41M parameters; warmup = 200 steps × 8,192 =
  **1.64M tokens ≈ 0.48 N**, i.e. already under the rule, not over it. **Confidence: High** for the rule and
  the arithmetic; *the rule's own validation was at 55M–901M parameters*, so applying it at 3.4M is an
  extrapolation.
- **Gradient clipping.** **could not verify.** My targeted search for the E-LDS line (Zhang et al., "An
  Empirical Law of Deviation on Optimizer Stochasticity") returned only the theory literature (U-Clip,
  <https://arxiv.org/abs/2302.02971>; "Generalized Gradient Norm Clipping & Non-Euclidean Trust Regions",
  <https://arxiv.org/abs/2506.01913>) and no LLM-magnitude result. I therefore state **no** clipping number,
  and no recommendation about Chakaso's clipping setting: it is unlit by this pass.
- **Weight decay on embeddings and norms.** No primary source read. The near-universal practice of exempting
  embeddings, norms and biases from decay is, in this document's own vocabulary, **folklore — Low confidence,
  not evidence**. If Chakaso wants it, it needs either a source read end-to-end or its own two-arm measurement.
  (Tied embeddings make the question non-trivial here: the same matrix is both input and softmax output, so
  exempting it is exempting the output layer too.)

---

## 6. Curriculum and mixture weights

### 6.1 DoReMi — the strongest read mixture result

**Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu
Ma, Adams Wei Yu, "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining",
<https://arxiv.org/abs/2305.10429>**, v1 17 May 2023. Abstract read verbatim: a small proxy model trained with
"group distributionally robust optimization (Group DRO) over domains" produces domain weights "without
knowledge of downstream tasks", which are then used to resample and train a larger model; "we use DoReMi on a
280M-parameter proxy model to set the domain weights for training an 8B-parameter model (30x larger) more
efficiently"; "On The Pile, DoReMi improves perplexity across all domains, even when it downweights a
domain"; "DoReMi improves average few-shot downstream accuracy by 6.5% points over a baseline model trained
using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps"; and
on GLaM it "matches the performance of using domain weights tuned on downstream tasks". **Confidence: High.**
Authors' reported; I read no independent replication.

### 6.2 Mixture laws, and how few runs they need

**Mustafa Shukor, Louis Bethune, Dan Busbridge, David Grangier, Enrico Fini, Alaaeldin El-Nouby, Pierre Ablin,
"Scaling Laws for Optimal Data Mixtures", <https://arxiv.org/abs/2507.09404>**, v1 12 July 2025. Body read:
the functional form is a weight-dependent rational term added to Chinchilla-style power laws —
`L = E + 1/(Σᵢ Cᵢ hᵢ^γᵢ) + A/N^α + B/D^β` — with a "joint" variant coupling the coefficients to the weights,
`A_h = (Σᵢ Cᵢ Aᵢh)^γA` and likewise `B_h`; fitting uses small models (LLMs 412M–1.4B, multimodal 106M–932M,
vision-language 89M–531M) and "**Only 10-20 runs are needed to fit the scaling laws**"; the additive form
yields an invariant optimal mixture while the joint form allows "the optimal h is scale-dependent"; errors
reported as validation Mean Relative Error, with the joint forms "frequently achieving sub-0.5% error on
text/caption tasks" on Slimepajama/Obelics/DFN-style corpora. Their stated limitations: pretraining-focused,
and they warn that assuming "**no data repetition** … is unrealistic when training with very scarce,
high-quality sources" and that the mixture is assumed "fixed throughout training". **Confidence: High.**

**Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, Xipeng Qiu, "Data Mixing Laws: Optimizing Data
Mixtures by Predicting Language Modeling Performance", <https://arxiv.org/abs/2403.16952>**, v1 25 March 2024.
Abstract verbatim: they discover "the quantitative predictability of model performance regarding the mixture
proportions in function forms", fit on sample mixtures to predict unseen mixtures "before actual runs", and
propose "nested use of the scaling laws of training steps, model sizes, and our data mixing law … with only
small-scale training"; validated on "a 1B model trained for 100B tokens in RedPajama, reaching a performance
comparable to the one trained for 48% more steps on the default mixture"; the extension to continual training
"accurately predicts the critical mixture proportion that avoids catastrophic forgetting". **Confidence:
High.**

**Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, Min Lin,
"RegMix: Data Mixture as Regression for Language Model Pre-training", <https://arxiv.org/abs/2407.01492>**, v1
1 July 2024 — ID, title, full author list verified; the abstract fetch returned only its first line, so the
mechanism (train many small proxy models on random mixtures, regress, optimise the regression) is stated here
from the title plus that first sentence. **Confidence: Medium**; no numbers quoted.

### 6.3 Curriculum ordering: the honest finding is that I found no read evidence

The task asks "what evidence exists that a curriculum ordering helps a small model". Located-but-unread
primary sources: **Daniel Campos, "Curriculum learning for language modeling",
<https://arxiv.org/abs/2108.02170>**, v1 4 August 2021 (title/ID/authors verified; content not read —
**Confidence: Medium, and it is a single-author NMT-era paper, not a pretraining-scale result**); and **Arzu
Burcu Güven, Anna Rogers, Rob van der Goot, "Do Syntactic Categories Help in Developmentally Motivated
Curriculum Learning for Language Models?", <https://arxiv.org/abs/2511.08199>**, v1 11 November 2025 — the
title itself is a question. **Zhaolin Gao, Joongwon Kim, Wen Sun, Thorsten Joachims, Sid Wang, Richard
Yuanzhe Pang, Liang Tan, "Prompt Curriculum Learning for Efficient LLM Post-Training",
<https://arxiv.org/abs/2510.01135>** is *post-training*, not pretraining. Shukor et al. explicitly *exclude*
curriculum by assuming "mixture is fixed throughout training".

**Result of this pass: no read primary source establishes that a curriculum ordering helps a small pretrained
model.** Everything I verified supports *static mixture weights*, which is a different claim. Conclusion: the
curriculum lever for Chakaso is **Unverified**, and the correct next step is not a curriculum run but a
mixture run, because that is where the evidence is.

### 6.4 What the literature says about Chakaso's own measured mixture result

Chakaso measured (E2612, 2026-10-30, one seed per arm, matched token budget): technical held-out bits/char
**−0.27584** versus the prose-only control, prose **0.03026 worse**.

- *Interpretation, contrast with DoReMi.* DoReMi's headline is that reweighting "improves perplexity across all
  domains, even when it downweights a domain". Chakaso's prose arm got worse, which points in the opposite
  direction. But the two are not comparable, and the reason is visible in Chakaso's own record: **the arms are
  not exposure-matched, they are budget-matched.** The E2612 record states the 20M cap "covers about 80 %" of
  the 25M-token prose pool for arm P and "about 42 %" of the 48M-token mixture pool for arm T, "roughly 10M
  prose and 10M technical tokens". So arm T saw about **half as many prose tokens** as arm P — the prose
  degradation is confounded with prose-token dose, not cleanly attributed to register competition. Under
  Hoffmann's own additive form the D-term alone would predict a prose-loss penalty from halving prose tokens;
  *I do not compute its size, because plugging 70M+-fitted coefficients into a 3.4M model is the extrapolation
  §1.2 forbids.* Mark: **Interpretation, Confidence: Medium** (the confound is High — it is stated in the
  record; the direction of the explanation is inference).
- *Interpretation, contrast with FineWeb-Edu.* FineWeb-Edu's classifier "intentionally suppresses commerce,
  legal, and entertainment topics while boosting academic encyclopedia and programming repository coverage" —
  i.e. technical-register weighting trades general-domain breadth for technical gain, and its reported cost
  shows up on MMLU-style breadth, which is exactly the shape of Chakaso's trade-off. Both are authors'
  reported; neither is replicated here.
- *Design implication, and it is cheap.* The mixture question at Chakaso's scale is answerable with the
  Shukor et al. recipe: **10–20 runs** with weights varied on a grid, one seed, fixed budget, and the mixture
  law fitted over `h` — and Chakaso's 2026-10-30 result was one pair. Its record's own declared decision rule
  put the next unit at "~9.84M-parameter scale step with the mixture recipe held"; this reading argues the
  10–20-run mixture fit is a *lower*-cost next unit than the capacity step, because the mixture slope is the
  one thing the read literature says is measurable at small scale with few runs.

---

## 7. Context length

### 7.1 The one read source that is at Chakaso's scale

**Jingzhe Shi, Qinwei Ma, Hongyi Liu, Hang Zhao, Jeng-Neng Hwang, Lei Li — <https://arxiv.org/abs/2502.01481>,
v1 3 February 2025.** *Citation-integrity note:* the arXiv API metadata read for this ID gives the title
"**Intrinsic Entropy of Context Length Scaling in LLMs**" and the six authors above, while the rendered v2 of
the same record that I read carried the title "**Explaining Context Length Scaling and Bounds for Language
Models**" and listed Serge Belongie among the authors. The record is the same paper at different versions; the
difference is recorded rather than resolved, and anyone citing it should read the version they need. Body read: they partition error into Bayesian risk and approximation error; lengthening the context
raises the intrinsic dimension of the input manifold, "theoretically offering more predictive signals", but
under the assumption that "this dimension remains finite asymptotically and each added dimension contributes
fixed informational bits, marginal gains diminish rapidly. Once dimensionality surpasses what the finite
training corpus can support, the model's inability to generalize causes approximation error to dominate,
reversing any prior improvements." Saturation is stated as `lim_{l→∞} ∂R_Bayes/∂l = 0`, and validation curves
"consistently exhibit an inverted-U trajectory". **Their experimental scale is a modified six-layer transformer
trained on three OpenWebText slices of 12M, 25M and 50M tokens** (with LLaMA-3.1-8B as an external evaluator
and synthetic parity tasks). Practical rule as read: "**Input window sizing must scale proportionally with
training corpus volume. Deploying expansive context windows on constrained datasets guarantees suboptimal
convergence**, whereas adequate data reserves justify elongated receptive fields." Limitations as read: they
isolate monolithic long-context pretraining, exclude progressive finetuning schedules, and flag that their
entropy–dimension correspondence needs stronger statistical backing and that cross-architecture generalisation
is unverified. **Confidence: High** — and this is the single most scale-appropriate source in this entire
document: 12M–50M tokens is Chakaso's own order of magnitude (19.66M consumed from a ~104M-character training
side).

*Implication.* Chakaso's 256-token context is not obviously too short. On a 20M-identifier pool, doubling the
context halves the number of distinct windows per epoch and raises the effective dimension the corpus must
support; the read theory says that trade can go negative at this data volume. The test is a matched-budget
pair (context 256 vs 512, same tokens, same documents, same seed) — and note that doubling context at fixed
positions/step doubles the attention cost per position, so a 512-context arm is not the same 73-minute
experiment. **Unknown** whether it helps: no read source measures 256→512 at 3.4M parameters.

### 7.2 Document-boundary masking and truncation

**Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, Stefano Soatto, "Fewer
Truncations Improve Language Modeling", <https://arxiv.org/abs/2404.10830>**, v2 read 2 May 2024. Mechanism as
read: fixed-window training "inevitably breaks many documents into incomplete pieces"; *Best-fit Packing*
"avoids mid-document cuts by only segmenting texts that exceed the context limit, then grouping these intact
chunks via length-aware combinatorial optimization", and "to prevent the model from attending across unrelated
texts within a single training sequence, the pipeline applies causal masking". Reported: models 7B–13B,
context lengths 2,048 and 8,192, 300B–500B tokens; **+4.7 % reading comprehension, +16.8 % context following,
+15.0 % program synthesis**, and closed-domain hallucination measured as undefined-name errors "decreased by
up to 58.3 %". Ablation as read: applying attention masks to standard concatenated data "primarily improves
validation perplexity, while the packing strategy itself drives downstream task reliability". Cost as read:
Best-fit Packing "adds fewer than 0.003 % extra sequences and maintains near-zero padding", with linear
algorithmic scaling. Stated limitations: targets large decoders, not compact or short-context architectures;
"no method can completely eliminate the risk of generating false information"; truncation is unavoidable for
any document over the window. **Confidence: High** for the numbers above; the explicit perplexity deltas were
not in the text I fetched, so none are quoted.

*Implication for Chakaso.* Two separable levers, and the sources separate them differently: masking
cross-document attention is a *loss*-lever (Ding et al.: improves validation perplexity; and it is cheap —
per-sequence blocking, or a document-id-aware batch), while packing is a *downstream*-lever at 7B+. Chakaso's
E2612 pipeline chunks documents and joins paragraphs, so cross-document attention within a window is possible
where two chunks of different documents share a packed example; whether it does at `context_length` 256 with
per-document packing is a question about Chakaso's own code, **not** about the literature, and should be
answered by reading `src/chakaso/training/` before any run. **Hypothesis**, not conclusion.

### 7.3 RoPE and θ

Verified IDs/titles/authors/dates, mechanism as named by the titles and abstracts: **Jianlin Su, Yu Lu,
Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu, "RoFormer: Enhanced Transformer with Rotary Position
Embedding", <https://arxiv.org/abs/2104.09864>**, v1 20 April 2021 (five versions) — **Confidence: High** for
the record, **Medium** for the mechanism (I read metadata, not the derivation).
**Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian, "Extending Context Window of Large Language
Models via Positional Interpolation", <https://arxiv.org/abs/2306.15595>**, v1 27 June 2023 — the
interpolation mechanism, no numbers read (**Medium**).
**Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole, "YaRN: Efficient Context Window Extension of
Large Language Models", <https://arxiv.org/abs/2309.00071>**, v1 31 August 2023 — the frequency-scaled
extension that subsumes the "NTK-aware θ scaling" the brief asks about (**Medium**: ID/title/authors verified,
abstract content not read in this pass).

**Critical limitation for Chakaso, and it is a reading-of-the-sources result: every positional-extrapolation
paper I verified operates on models ≥1B that were trained at a shorter context and extended afterward.** None
of them measures whether θ = 10,000 is the right value for a 3.4M-parameter model trained *never to exceed*
256 positions. At 256 positions, RoPE's slowest rotation at θ = 10,000 covers a period far longer than the
window, so a large part of the rotary frequency budget is unused; shrinking θ would spend it better on the
256 positions the model actually sees. That is an inference from the mechanism, and I found **no** read
primary source measuring it. Status: **Hypothesis**, Confidence in the reasoning **Medium**, Confidence that
it has been measured anywhere **Low**. The test is a matched-budget θ ∈ {1, 100, 10,000} triple at 3.4M
parameters — three arms, ~3.7 hours.

---

## 8. Distillation and synthetic data — and what Chakaso cannot do

### 8.1 Teacher-in-weights distillation: the read numbers

**Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu, "TinyBERT:
Distilling BERT for Natural Language Understanding", <https://arxiv.org/abs/1909.10351>**, v1 23 September
2019 (ID correction: 1909.10357 is a lattice-QCD paper). Abstract read verbatim: "TinyBERT with 4 layers …
achieves more than 96.8% the performance of its teacher BERT_BASE on GLUE benchmark, while being 7.5x smaller
and 9.4x faster on inference", beats 4-layer distillation baselines at "about 28% parameters and about 31%
inference time of them", and "TinyBERT with 6 layers performs on-par with its teacher BERT_BASE".
**Confidence: High.** Note the correction to the brief's framing: TinyBERT is the *strongest* read result in
this section, but it is task-specific distillation of an encoder with a labelled-fine-tuning teacher — it is
not evidence that a from-scratch decoder can be made good by copying a teacher's logits.
**Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf, "DistilBERT, a distilled version of BERT:
smaller, faster, cheaper and lighter", <https://arxiv.org/abs/1910.01108>**, v1 2 October 2019, and **Wenhui
Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, Ming Zhou, "MiniLM: Deep Self-Attention Distillation for
Task-Agnostic Compression of Pre-Trained Transformers", <https://arxiv.org/abs/2002.10957>**, v1 25 February
2020 — IDs/titles/authors verified, abstract numbers **not read** (**Medium**).

### 8.2 Teacher-in-data: the result that actually bears on Chakaso

**Suriya Gunasekar, Yi Zhang, Jyoti Aneja, … Yin Tat Lee, Yuanzhi Li, "Textbooks Are All You Need",
<https://arxiv.org/abs/2306.11644>**, v1 20 June 2023. Abstract read verbatim: "phi-1 is a Transformer-based
model with 1.3B parameters, trained for 4 days on 8 A100s, using a selection of 'textbook quality' data from
the web (6B tokens) and synthetically generated textbooks and exercises with GPT-3.5 (1B tokens). Despite this
small scale, phi-1 attains pass@1 accuracy 50.6% on HumanEval and 55.5% on MBPP", and "phi-1-small, a smaller
model with 350M parameters trained with the same pipeline as phi-1 … still achieves 45% on HumanEval".
**Confidence: High.**

**Ronen Eldan, Yuanzhi Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?",
<https://arxiv.org/abs/2305.07759>**, v1 12 May 2023. Abstract read verbatim, in full: "Models with around
125M parameters such as GPT-Neo (small) or GPT-2 (small) can rarely generate coherent and consistent English
text beyond a few words even after extensive training. This raises the question of whether the emergence of
the ability to produce coherent English text only occurs at larger scales (with hundreds of millions of
parameters or more)…" … "we introduce TinyStories, a synthetic dataset of short stories that only contain
words that a typical 3 to 4-year-olds usually understand, generated by GPT-3.5 and GPT-4. We show that
TinyStories can be used to train and evaluate LMs that are much smaller than the state-of-the-art models
(**below 10 million total parameters**), or have much simpler architectures (**with only one transformer
block**), yet still produce fluent and consistent stories with several paragraphs that are diverse and have
almost perfect grammar, and demonstrate reasoning capabilities." They also introduce GPT-4-as-grader
evaluation across grammar, creativity and consistency. **Confidence: High.**

*This is the most important passage in §8 for Chakaso.* Chakaso's E2610 found its released checkpoint repeats
(mean bigram repetition 0.5538, every sample hitting its token bound) and that this is greedy decoding of a
human-diverse distribution rather than a pipeline defect. TinyStories is the read, measured existence proof
that **a model with fewer parameters than Chakaso's 3.4M can produce fluent multi-paragraph English**, and
that the lever there was the corpus — a synthetic dataset whose text is short, internally coherent and drawn
from a narrow vocabulary — not scale, not architecture, and not a teacher in the weights. It is also the
closest published analogue to Chakaso's own metric situation: they grade generations with a model rather than
with perplexity.

### 8.3 What Chakaso's contract permits, stated precisely

Chakaso's rule is that its **weights** are trained from random initialization; nothing in AGENTS.md or the
ADRs read prohibits a *text corpus* that a stronger model produced or selected. That splits the distillation
literature into three lanes:

1. **Prohibited for Chakaso (teacher in weights):** TinyBERT/DistilBERT/MiniLM-style logit or attention
   distillation, and any replacement of its initialization by a pretrained checkpoint. No source read here
   shows such a route being necessary for loss to fall — and none shows a from-scratch 3.4M decoder reaching
   coherent text either. **Recommendation: none.** This document does not propose touching the weights, and
   per the brief that is a hard constraint, not a preference.
2. **Available, with a provenance cost (teacher in *selection*):** FineWeb-Edu's pattern (§4) — a strong rater
   labels 460k samples, a cheap regressor learns from those labels, the regressor filters the corpus. On a
   149M-character corpus the filter step is affordable to run over everything. This is data provenance, must
   be pinned like any corpus (ADR-0036), and must be recorded as teacher-derived selection, not as a
   neutral fact.
3. **Available in principle, blocked in practice (teacher in *generation*):** TinyStories and phi-1. Both use a
   commercial model to *author* training text. For Chakaso this collides with ADR-0001's "the runtime must not
   require a commercial inference or search API" only at generation time, not at runtime, so it is contract-
   legal but raises licensing and provenance questions the repository has not answered. It also would move the
   measured object: a model trained on GPT-authored text measures imitation of that distribution, and
   Chakaso's current corpora are licensed public-domain text chosen for exactly that reason. **Status: Planned/
   Planned-only-as-a-question.** No recommendation to run it; the TinyStories fact is recorded because it
   bounds what "3.4M parameters cannot produce coherent text" would mean — it is **not** a theorem, and the
   published counterexample is at 3.4M-and-below on a different corpus.

**What Chakaso cannot do given no pretrained teacher is allowed:** none of §8.1's numbers can be borrowed.
TinyBERT's 96.8 %-of-teacher, phi-1's 50.6 % HumanEval and MiniLM's compression ratios are all measured on
students *distilled from a teacher*. There is no read evidence that from-scratch training at ≤10M parameters
reaches coherent English on a natural, uncurated or licensed-literary corpus — TinyStories' own comparison
point (125M GPT-Neo/GPT-2-small "can rarely generate coherent and consistent English text beyond a few words")
is on natural text and is a *negative* result for scale alone. So Chakaso's E2610 repetition finding is not
disproved by this literature, and it is not explained by it either.

---

## What changes the next decision, ranked

1. **Chakaso is at ~0.98 epochs of a 20,000,000-identifier pool, so reaching the Chinchilla-like ~20
   tokens/parameter needs only more steps, not more data** — ~8,340 steps, ~4.2 h/arm at the measured 4,493.7
   positions/s. Muennighoff et al. (<https://arxiv.org/abs/2305.16264>) report "up to 4 epochs of repeated
   data yields negligible changes to loss compared to having unique data". **Confidence: High** for both the
   arithmetic and the quote; **Medium** that ≤4-epochs-is-free holds at 3.4M parameters (Xue et al.,
   <https://arxiv.org/abs/2305.13230>, list dataset size and parameter count as the significant drivers of
   multi-epoch degradation, and never measured this scale).
2. **The "~20 tokens/parameter" number is real but has no authority at 3.4M parameters.** It is derived from
   Hoffmann's Table 3 (400M → 8.0B tokens; 1B → 20.2B), re-derived at ~20 by Besiroglu et al.
   (<https://arxiv.org/abs/2404.10102>) while *disputing their confidence intervals*, and defended by
   Schaeffer et al. (<https://arxiv.org/abs/2509.23963>) as "≈20 tokens per parameter". Chakaso's N is ≥20×
   below the smallest model in every one of those fits, and Hoffmann's own text says "projections from very
   small models lead to different predictions than those from larger models". **Confidence: High** — and the
   correct use is *direction*, not target.
3. **Sardana et al. (<https://arxiv.org/abs/2401.00448>) measured quality still improving out to 10,000
   tokens per parameter on 47 models, and found that laws fit only at typical ratios "overestimate the impact
   of additional tokens at these extreme ranges".** For a local-first, inference-heavy product this is the
   scaling result that points where Chakaso already stands. Authors' reported; not independently replicated.
   **Confidence: High** for what it says, **Low** that it transfers to 3.4M.
4. **The mixture result is confounded with prose-token dose, and a 10–20-run mixture law fit is the cheapest
   next experiment.** E2612's own record states the 20M cap covers ~80% of the prose pool but only ~42% of the
   mixture pool, so arm T received ~half the prose tokens of arm P; the "prose got 0.03026 worse" is not yet
   separable from "prose saw half the data". Shukor et al. (<https://arxiv.org/abs/2507.09404>) report "**Only
   10-20 runs are needed to fit the scaling laws**" for mixture weights, fitted on 412M–1.4B models, with the
   additive/joint forms and their stated "no data repetition" caveat. **Confidence: High** for both readings.
5. **Chakaso's schedule is a truncated cosine with no annealing phase, which makes "still descending at the
   ceiling" uninterpretable as a capacity statement.** Wen et al. (<https://arxiv.org/abs/2410.05192>): the
   decay window belongs at ~10% of steps (robust 8–12%), and early decay costs ~2e-2; Wang et al.
   (<https://arxiv.org/abs/2512.13705>): the optimal annealing fraction follows a transferable power law in
   step count (α_T ≈ −0.946, MAPE <2%). Porian et al. (<https://arxiv.org/abs/2406.19146>): warmup tokens
   should equal N, and their rule plus per-scale optimizer alignment resolves the Kaplan/Chinchilla exponent
   conflict across 900+ runs at 55M–901M. **Confidence: High** that these are what the sources say; **Medium**
   that they hold at 3.4M/20M-token scale.
6. **Vocabulary size is probably not the binding constraint, and there is a read reason why.** Takase et al.
   (<https://arxiv.org/abs/2406.16508>): at a fixed 680M model and 1T tokens, 5k→500k buys +0.4/+0.8/+1.4/+2.5
   English accuracy points — but "restricted corpora suppress these advantages". Chakaso is the suppressed
   regime. Gigant/Peng/Quesnelle (<https://arxiv.org/abs/2604.27263>) show the subword benefit is dominated by
   sequence-length/throughput (4.75 bytes per token), and that 68M models gain *more* from it than 1.7B ones —
   so if a tokenizer lever is pulled, the lever is compression per character, not merge count. **Confidence:
   High** for the numbers, **Medium** for the inference.
7. **Context 256 may already be too long for the corpus, and the read evidence is at Chakaso's own scale.**
   Shi et al. (<https://arxiv.org/abs/2502.01481>) ran a six-layer transformer on 12M/25M/50M-token slices and
   report inverted-U validation curves with the rule that "deploying expansive context windows on constrained
   datasets guarantees suboptimal convergence". Separately, Ding et al. (<https://arxiv.org/abs/2404.10830>)
   isolate the two confounded levers: document-boundary **masking** moves validation perplexity; packing moves
   downstream tasks (+4.7/+16.8/+15.0 points, ≤0.003% extra sequences) at 7B+. **Confidence: High**; the
   transfer to 3.4M parameters is still a hypothesis.
8. **The data-cleaning corpus is almost entirely non-transferable, with two exceptions.** FineWeb
   (<https://arxiv.org/abs/2406.17557>): global cross-snapshot MinHash collapsed 20T→4T tokens *and hurt*; the
   terminal-punctuation rule stripped ~30% of the vocabulary and was removed; FineWeb-Edu (Llama-3-70B
   labelling 460k pages, regressor over frozen Arctic embeddings, threshold 3, 1.3T tokens) moved MMLU 33→37%
   and ARC 46→57%. Lee et al. (<https://arxiv.org/abs/2107.06499>): dedup makes verbatim memorization
   10× rarer and reduces train-test overlap that affects >4% of standard validation sets. For a Gutenberg
   corpus: no web-noise filters apply, but **work-level near-duplicate control does** (multi-edition texts) and
   no training-path dedup exists in `src/chakaso/training/`. **Confidence: High** for the numbers; **High**
   that no C4/RefinedWeb-style filter is justified here.
9. **Coherent text at sub-10M parameters has been demonstrated — but only on a synthetic corpus.** Eldan & Li
   (<https://arxiv.org/abs/2305.07759>): models "below 10 million total parameters", even "with only one
   transformer block", produce fluent multi-paragraph stories on TinyStories, while 125M-parameter models on
   natural text "can rarely generate coherent and consistent English text beyond a few words". This is the
   strongest read argument that Chakaso's next lever is **corpus construction**, not capacity, and it is the
   boundary of what its no-teacher contract leaves available (teacher-in-data, not teacher-in-weights).
   **Confidence: High** for the abstract; the licensing question it opens is unresolved and is Chakaso's to
   decide.
10. **Optimization: one measurable and one unlit item.** The gradient noise scale (McCandlish/Kaplan/Amodei,
    <https://arxiv.org/abs/1812.06162>) is a two-gradient-evaluation measurement Chakaso can run to decide
    whether 8,192 positions/step is past the useful batch; and for Adam-style optimizers the optimal lr "first
    rises and then falls as the batch size increases" (Li et al., <https://arxiv.org/abs/2405.14578>), so
    batch and lr must move together, not independently. **Gradient clipping and weight-decay placement on
    embeddings: could not verify any primary source — folklore; do not act on this document for either.**

---

## Corrections to the brief's attributions (recorded because the project does not repeat unverified citations)

| Brief said | Correct reading |
| --- | --- |
| BPE is arXiv 1508.07921 | **1508.07909** (the BPE-for-NMT paper; author list not read in this pass). 1508.07921 is a graph-drawing paper by Frati, Hoffmann and Kusters, which the fetch read in full. |
| "Beyond neural scaling laws" is Besiroglu et al. | **Sorscher, Geirhos, Shekhar, Ganguli, Morcos — 2206.14486.** It is a vision data-pruning paper. |
| Hoffmann's Chinchilla data exists below ~70M parameters | No. "400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens"; smallest under an embedding-inclusive count is 44M (Schaeffer et al.). |
| The phrase "20 tokens per parameter" is a quoted sentence | The literal phrase was **not found** in the fetched full text. It is the ratio implied by Table 3 (400M → 8.0B; 1B → 20.2B tokens) and stated as "≈20 tokens per parameter" by Schaeffer et al. |
| GPT-2 as the byte-level BPE citation | **Could not verify** an arXiv record for Radford et al.'s GPT-2 report; not cited. Verified byte-level/subword-decoupling sources are used instead (2105.13626, 2604.27263). |
| Muennighoff's "Scaling Data-Constrained Language Models" | Exists and is verified: **2305.16264**. |
| BBPE as a distinct verified line | **Could not verify** the "byte-level subwords" paper I looked for (the guessed ID 2201.01526 is a gravitational-wave wavelet paper). |

## Could not verify (searched this pass; do not cite)

- Besiroglu et al., "Compute MiniModels" and "Scaling Laws by Deriving Optimal Hyperparameters" — no ID
  obtained despite several searches. Their substantive content is instead supplied from verified sources:
  2609.01431 (§2.4), 2406.19146 (§1.3), 2404.10102 (§1.3).
- "Loss-to-Learn" (downstream-performance-from-loss scaling laws) — no matching entry found.
- "The Tokenization Penalty" — no matching entry found.
- "Mixture of Tokens" as a *tiny-vocabulary* result — the real MoT paper (2310.15961) is a continuous-MoE
  architecture paper; the small-vocabulary claim is **not supported** by anything read here.
- The GPT-2 technical report as an arXiv record.
- Any quantitative gradient-clipping or weight-decay-placement result for language models.
- Any primary source measuring curriculum *ordering* in pretraining at small scale (2108.02170 located, not
  read; 2511.08199 located, not read).
- DS1000 / Soledad-Lopez et al. (guessed ID 2308.08972 is an econometrics paper; correct ID not obtained).
- RefinedWeb's own internal ablation numbers (only its identity and its "500B tokens" as reported by FineWeb).

---

## Method note, for reproducibility

Every source above was located with WebSearch and read with WebFetch, preferring
`https://arxiv.org/html/<id>` or `https://ar5iv.labs.arxiv.org/html/<id>` for body text and
`http://export.arxiv.org/api/query?id_list=…` for verbatim titles, authors, v1 dates and abstracts.
The API batch-check step is what caught the wrong IDs in the corrections table; several IDs that are
commonly repeated from memory (Sennrich, TinyBERT, 1907.11692 ≠ TinyBERT, 2310.12485 ≠ Mixture-of-Tokens,
2209.01649 ≠ μP) resolved to unrelated papers on other subjects. **No citation in this file was written
without its ID being resolved to a title in this pass.** Fetch failures and truncations are recorded as
*not read* rather than filled from recollection. Some long quotations were returned under the fetcher's own
length cap and appear as fragments; where a fragment carries a number, the number is what was on the page.

**Residual risk, stated.** Body quotations marked *as read* were returned by a page-reading tool rather than
transcribed from a page region confirmed by eye, so a fragment may be re-assembled from separated passages.
Anyone who is going to act on a load-bearing quotation — Hoffmann's small-model caveat, Porian et al.'s
warmup-tokens rule, Muennighoff's four-epoch sentence, Shukor et al.'s "10-20 runs", Eldan & Li's "below 10
million total parameters" — should open the linked source and read the sentence in its own context before it
enters an experiment record or an ADR. Every bibliographic fact in the corrections table was double-checked
against API metadata for exactly this reason.
