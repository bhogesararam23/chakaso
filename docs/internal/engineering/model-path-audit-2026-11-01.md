# Model path implementation audit — 2026-11-01

**Scope.** A skeptical read of the *implementation* of the model path: the byte-pair tokenizer, the rotary
decoder, the training mechanics, checkpointing and resumption, and the numerical risks that could silently
corrupt a measured result. Evidence is whole-file reads of `src/chakaso/training/{subword,tokenizer,
transformer,decoder,loop,nn_training,optimizer,packing,checkpoint,nn_checkpoint,data,corpus,snapshot,
manifest,config,seed_study,subword_experiment,baselines,model_metrics,nn_evaluation}.py` and
`src/chakaso/models/{base,nntransformer,decoding,load,release,trainedchar,deterministic}.py`, plus code I
executed against the pinned corpora and the committed reports under `data/experiments/`.

**Status labels.** *Verified-correct* = checked in code and, where a number was involved, recomputed.
*Risk* = a mechanism that can distort a measured number without a guard firing. *Observation* = what the code
does, which a reader would not get from the name or the docstring. *Not determined* = I could not establish it.

**Companion.** [measurement-harness-audit-2026-11-01.md](measurement-harness-audit-2026-11-01.md) audits how
numbers are *produced and guarded* (metric arithmetic, partition weighting, snapshot verification). This
document audits the machinery underneath. Where they overlap the two agree, and agreement is noted.

**Execution environment of everything I ran.** Python 3.13.5, `torch 2.13.0+cpu`, `numpy 2.4.0`, CPU — the
same block the committed reports record (`data/experiments/E2608/decoder/report.json`: `torch_threads "14"`,
`platform Windows-11-10.0.26200-SP0`).

---

## 1. The BPE tokenizer (`src/chakaso/training/subword.py`)

### 1.1 RISK (highest impact) — the merge loop no longer implements the rule its docstring states, and it changes fitted vocabularies on real prose

**Observation.** `train_subword_vocabulary` is documented as "the chosen rule is the most frequent adjacent
pair with a count of at least `min_frequency`, ties broken by the pair's own ordering"
(`subword.py:420-423`), and the in-loop comment repeats it (`subword.py:470-471`). Since commit `6f3274f`
("perf: index the byte-pair merge loop") the selection comes from a lazy-invalidated heap
(`subword.py:463-478`) and an index from pair to the shapes containing it (`subword.py:456-461`,
`subword.py:484-499`). The index is maintained correctly — I recomputed the true pair counts from the shapes
at every step of a 343-merge fit and the maintained `pair_counts` agreed with the recomputed truth at **all
343 steps**, zero disagreements. The *selection* is not: a new heap entry is pushed only for pairs of the
merged shape (`subword.py:496-499`), so when a pair's count **falls** because a shape containing it was merged
away and the new shape no longer contains it, that pair keeps only stale-high entries, which the search loop
pops and discards (`subword.py:474-478`) without ever re-pushing. The pair becomes invisible to the chooser
until its count rises again.

**Risk, measured.** Over the repository's own prose — 10 documents, **367,411 characters**, `target_size=600`,
`min_frequency=3`, 343 merges — the heap pick differed from the documented greedy argmax at **287 of 343
steps**, and at **286 of those the chosen pair had a strictly lower true count** than the pair the rule names
(the remaining one was a broken tie-break). This is not a rounding effect: it changes the fitted vocabulary.
The pre-`6f3274f` implementation (checked out to a temp module and run on the same text) is byte-for-byte the
brute-force greedy reference, and the two vocabularies differ:

| fit | fingerprint | held-out 20,000 chars → identifiers |
| --- | --- | --- |
| pre-`6f3274f` (documented rule) | `4193b946dc9bb514` | 11,412 |
| current `HEAD` | `dc00e781b4558328` | 11,452 |

At `target_size=1024` the same comparison gives `1f096b915be6b149` against `009b3ba69eb5fa86`, and **9,859
against 9,898 identifiers** for the same 20,000 characters. Across 300 randomized synthetic trials:
107 differ in merge sequence and 25 produce **fewer merges than the declared target** (never more).

**Risk, measured against a published result.** The commit message asserts "the merge sequence on a reference
text is byte-identical to the previous implementation, so no vocabulary moves." On the corpus that produced
E2608 and E2611 this is false. Those arms share the cached tokenizer
`data/experiments/tokenizers/gutenberg-english-prose-v2-tokenizer-f99386c7f5be11a7.json` — 4,096 identifiers,
3,839 merges, `min_frequency 3`, fingerprint `b597eb34f13d9058`, recorded as `reused_from_cache: true` in both
arm reports. Re-fitting the **same input** with `HEAD` — the same snapshot
(`data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json`) and the same split (my recomputed
`split_text_units(..., 0.15, 0.15, seed=7)` reproduces the recorded `split_fingerprint df603dc033c015dd` and
the identical per-side identifier lists, so the input is exactly the measured one) — yields **3,839 merges,
fingerprint `a1ef04f64739cd4b`, diverging from the measured table at merge rank 5**, and on the measured test
side it encodes the same 2,165,409 characters as **1,009,294 identifiers instead of 1,004,306**
(2.1455 against 2.1561 characters per identifier, −0.49 %). Because
`bpc = (nats/token) / (ln2 · chars/token)` (`model_metrics.py:96-106`, `:139-151`), a 0.49 % smaller
chars/token raises bpc by the same 0.49 % for unchanged per-token loss: E2608's 1.6865 would move to ≈1.695,
which is a **quarter of E2611's declared 0.02-bit materiality band** (`experiments/E2611/record.md`, H1–H3).
The full effect is not determined, because the segmentation of the *training* signal changes too.

**Consequence for the guards.** `target_size` is not asserted after a fresh fit: `_fit_or_reuse_vocabulary`
checks `stored.size` only on the **cache** path (`subword_experiment.py:681-685`); a freshly fitted vocabulary
is saved unchecked (`subword_experiment.py:694-700`), and `architecture_for` sizes the model from
`vocabulary.size` (`subword_experiment.py:495-512`). Since the loop can also break early (`subword.py:479-480`,
demonstrated in 25/300 synthetic trials), a run can silently train a **smaller vocabulary than the record
declares** with nothing firing. The measured corpus did not hit this (3,839 merges reached), so it is a
structural risk, not an observed one. Reproduction of E2608/E2611's tokenizer therefore depends on a
git-ignored cache file, not on the code.

### 1.2 Observation — what kind of BPE this is

- Byte-level base is real: identifiers 0–255 are UTF-8 bytes, 256 is the reserved document separator, 257
  starts the merges (`subword.py:65-71`). Encoding is total and `UNK` is absent; `decode` uses
  `errors="replace"` so a stream cut mid-codepoint yields U+FFFD rather than failing (`subword.py:280`).
- The pre-tokenizer is `\w+|\s+|[^\w\s]` (`subword.py:81`), and merges are learned and applied only inside a
  piece (`subword.py:233-234`, `subword.py:450-451`). Consequences a reader would not guess: a space is never
  part of a word token (no GPT-2-style `Ġthe`), a 1-byte piece can never merge at all because the encode loop
  requires `len(symbols) > 1` (`subword.py:239`), and every distinct whitespace run is its own piece.
  "Byte-pair encoding" is therefore a **word-boundary-respecting** BPE, which is a defensible choice but is not
  the segmentation GPT-2-style tokenizers produce, so its bits-per-character is not comparable to theirs.
- Encoding applies the lowest-rank pair **globally** per pass (`subword.py:241-248` then
  `_apply` at `subword.py:251-264`, non-overlapping left-to-right). That is the Sennrich reference behaviour,
  not HuggingFace's single-position merge, so Chakaso segmentations will differ from a HF `Bpe` tokenizer on
  the same merge table. Consistent between fit and encode (`subword.py:490` uses the same `_apply`), so no
  train/serve skew.
- `min_frequency` is a **count floor on the chosen pair**, applied inside the heap search
  (`subword.py:476`), not a floor on piece frequency. Because of 1.1 the floor is applied to a stale view of
  the counts. `min_frequency` is stored and round-tripped (`subword.py:320`, `:375`) but is **not** in the
  vocabulary `fingerprint` (`subword.py:165-175`) — harmless for encoding identity, and compensated by the
  cache key including it (`subword_experiment.py:626-633`) and the cache-path guard (`subword_experiment.py:687-692`).

### 1.3 Verified-correct

- **Merge-table integrity checks bite.** A merge may not forward-reference (`subword.py:139-144`: at rank *r*
  only ids `< 257+r` are legal), duplicates are refused (`subword.py:145-147`), and `bool` is refused where an
  int is required (`subword.py:126-131`, `:136`). `from_document` verifies the record's own checksum before
  reading it (`subword.py:343-352`), refuses a foreign schema/base/pre-tokenizer (`subword.py:353-370`) and a
  `vocab_size` that disagrees with the merge count (`subword.py:380-385`).
- **Determinism.** Same-process repeats, reversed input-document order, and three interpreter versions
  (3.11.16 / 3.12.14 / 3.13.5) all produce the identical fingerprint `dc00e781b4558328` on the same fit. The
  merge *sequence* is however sensitive to the iteration order of the `affected` shape set: replacing that
  iteration with `sorted()` changes the output from rank 33 onwards. So "the rule sequence is a property of the
  data alone" (`subword.py:23-25`, `:470-471`) holds empirically for CPython sets of int-tuples, not by
  construction.
- **The separator identifier never moves** (`subword.py:67`, `:71`) and `symbol_to_bytes(256)` returns `b""`
  (`subword.py:189-190`), so `decode` skips it and it contributes no characters while still costing its
  prediction — which is what `model_metrics.py:21-24` states and `model_metrics.py:141-146` implements.
- **Incremental decoding** is the correct instrument for generation: `IncrementalDecoder` holds a
  `codecs` UTF-8 incremental decoder with `errors="replace"` (`subword.py:98-115`), so a character split across
  tokens is emitted whole. Using `decode` per token would be wrong, and the docstring says so (`subword.py:282-288`).
- `encode(..., allowed=...)` refuses out-of-range identifiers by name (`subword.py:223-228`) — though it encodes
  the text twice to do so (`:224`, `:228`).

### 1.4 Observation / Risk (low) — leftovers and costs

- `_shift_counts` (`subword.py:528-548`) is **dead code**: a repository-wide search finds no caller. It
  documents the old per-occurrence count shift and does *not* maintain the pair→shapes index, so a future
  reader who assumes the live loop does the same bookkeeping as this helper would be wrong.
- `_ranks()` is rebuilt on every `_encode_all` call (`subword.py:231`), and `_encode_piece` rescans the whole
  symbol list per merge (`subword.py:241-245`) — quadratic in piece length. Correct, just costly at 4,096 rules.
- `symbol_to_bytes` is recursive (`subword.py:195`), depth bounded by the byte length of the symbol. With the
  512 KiB per-document bound (`manifest.py:66`) a pathological single 10,000-character word could exceed
  CPython's recursion limit. **Not verified** — I did not construct such a corpus.

### 1.5 Observation — the character vocabulary is a different, simpler contract

`CharVocabulary` assigns ids by codepoint order from **training text only** (`tokenizer.py:54-58`), reserves
id 0 for unknown (`tokenizer.py:31-33`), and is deliberately lossy on unseen characters (`tokenizer.py:93-104`).
`character_windows` left-pads early positions with `UNK_ID` (`data.py:549-552`), so the first characters of
each side are predicted from padding — identical treatment on both sides (`data.py:536-537`), which is the
stated intent. This is the E2601/E2602 path and is not used by the subword records.

---

## 2. The rotary decoder (`src/chakaso/training/decoder.py`)

### 2.1 Verified-correct — the architecture is what its name says

- **True causal multi-head attention.** q/k/v/o are four bias-free `nn.Linear`s (`decoder.py:312-315`), split
  by `view(batch, length, heads, head_width).transpose(1,2)` (`decoder.py:328-331`), scored by
  `scaled_dot_product_attention` with the default `1/sqrt(head_width)` scale (`decoder.py:342-348`) and merged
  back through `transpose().contiguous().view()` (`decoder.py:349`). Causality is an explicit additive mask
  (`decoder.py:256-280`, `decoder.py:597-601`) — including the non-square case, where query *i* sits at absolute
  position `offset + i` (`decoder.py:276-279`), which is what makes a cached step correct. `tests/
  test_training_decoder.py:150` asserts a later token cannot change an earlier logit; `:164` asserts stepped
  cached decoding equals the full forward pass; `:139` asserts the offset mask.
- **RoPE is standard and applied consistently.** `rope_angles` gives
  `angle = position · theta^(−2i/head_width)` (`decoder.py:219-221` — `scale = theta ** (band/pairs)` with
  `pairs = head_width/2`, so the exponent is `2i/head_width` exactly). `apply_rotary` is the interleaved form
  (`decoder.py:250-253`). Both q and k are rotated **before** the cache is concatenated (`decoder.py:333-340`),
  so a stored key carries its own position and the training and cached-decoding paths cannot disagree. The
  position-only-difference property is tested (`tests/test_training_decoder.py:115`). `head_width` is forced
  even at construction (`decoder.py:118-120`), so no dimension escapes rotation.
- **RMSNorm** is mean-of-squares only, with an `eps` inside and `rsqrt` outside, no bias, scaled by a learned
  weight (`decoder.py:288-301`, `decoder.py:62`). **Pre-norm** residual blocks (`decoder.py:385-387`).
  **GELU** is `functional.gelu` on the up-projection (`decoder.py:362`), the up/down pair carrying biases
  (`decoder.py:358-359`).
- **Tied embeddings** are a real tie: `self.head.weight = self.token_embedding.weight` (`decoder.py:427-428`),
  so `model.parameters()` counts the matrix once — verified empirically against the arithmetic (§2.2) and
  tested (`tests/test_training_decoder.py:359`).
- **Initialization**: every `nn.Linear`/`nn.Embedding` weight from `N(0, initialization_std)`, biases zeroed
  (`decoder.py:439-444`), then **only** the two residual branches scaled by `1/sqrt(2·layers)`
  (`decoder.py:446-450`) — tested (`tests/test_training_decoder.py:338`). RMSNorm weights stay at the `ones`
  they were constructed with (`decoder.py:297`). Parameters are float32 on CPU, and a test pins that
  (`tests/test_training_decoder.py:417`).
- **Loss**: cross-entropy with `ignore_index=-100` (`decoder.py:627-631`), and a batch in which *every* target
  is padding is refused rather than returning NaN (`decoder.py:621-626`, tested at `tests/test_training_decoder.py:266`).

### 2.2 Verified-correct — the parameter accounting reproduces 3,413,760 exactly

`DecoderConfig.parameter_count` (`decoder.py:187-199`) is
`vocab·d + layers·(4d² + 2·d·ff + ff + d + 2d) + d + (0 if tied else vocab·d)`. For
`d_model=256, num_layers=4, num_heads=8, feedforward_dim=640, vocab_size=4096, context_length=256` I built the
model and compared:

- formula → **3,413,760**; `count_parameters(built model)` → **3,413,760**; equal → `True` (torch 2.13.0+cpu).
- Per-layer term → 591,232; ×4 = 2,364,928; embedding 4,096×256 = 1,048,576; final norm 256. `head_width = 32`
  (even, `rope_pairs = 16`).
- `sum(state_dict tensor numel)` = **4,462,336**, i.e. 1,048,576 more, because the tied matrix is stored under
  both `token_embedding.weight` and `head.weight`. `ModelArtifact.parameter_count` deliberately reports the
  configuration number, not the byte sum (`nn_checkpoint.py:119-128`), and the measured reports carry both
  fields agreeing (`data/experiments/E2608/decoder/report.json`: `parameter_count` 3,413,760,
  `parameters_by_configuration` 3,413,760).
- The same check on the smaller arms: E2611's decoder (4096/128/128/3/4/384) formula = **1,018,240** = the
  reported 1,018,240; E2605's transformer formula
  (`transformer.py:141-170`) = **643,840**, which is the measured figure. Verified.

So the claimed size is not a claim about bytes; it is arithmetic that the built model agrees with.

### 2.3 Observation — dropout is present as a field and almost absent as a mechanism

`DecoderConfig.dropout` (`decoder.py:92`) reaches exactly one place: `dropout_p=self.dropout if self.training
else 0.0` inside `scaled_dot_product_attention` (`decoder.py:342-348`). There is **no** dropout on embeddings,
on the residual stream, or in the feed-forward (`decoder.py:353-364`), and no attention-probability dropout
beyond that flag. The docstring says "deliberately absent: dropout beyond the constructor field"
(`decoder.py:26-28`), which is accurate, but a reader who sets `dropout=0.1` is getting attention-only dropout
in a form that consumes the global torch RNG — see §5.5 for why that matters for reproducibility. All measured
runs used `dropout: 0.0` (`data/experiments/E2608/*/report.json`, `E2611/decoder-small`, `E2612/{prose,mixture}`).

### 2.4 Observation — the tied matrix is initialized twice, and the second draw wins

`_initialize` iterates `self.modules()` and calls `nn.init.normal_` on every `Linear`/`Embedding` weight
(`decoder.py:439-444`). Because `head.weight` **is** `token_embedding.weight` (`decoder.py:427-428`), the shared
tensor is drawn once when `token_embedding` is visited and again when `head` is visited last. Instrumenting
`nn.init.normal_` on a 2-layer/512-vocab decoder: the tied `(512, 64)` tensor receives **3** calls (torch's own
`nn.Embedding.__init__` default plus the two from `_initialize`) and the surviving values are the **`head`
draw**, not the embedding draw. Consequences: (i) the RNG stream consumes an extra `vocab×d` normal; (ii) any
future change that gives the head a different std silently changes the embedding; (iii) the statistical law is
unchanged today (both draws are `N(0, 0.02)`), so **no measured number is affected**. `CausalTransformer`
behaves identically (`transformer.py:228-246`, `:218-221`). **Not determined:** whether any published weight
identity depends on the draw order (nothing cites an initialization-order claim).

### 2.5 Observation / Risk — document-boundary masking is opt-in, and the held-out scorer does not use it

- The mechanism is correct and refused loudly: `_validity_mask` intersects causality, key validity and document
  membership (`decoder.py:634-660`, `:658`), `segment_ids` with a cache is refused (`decoder.py:557-563`), the
  training engine refuses a masked run over examples without `segment_ids`
  (`nn_training.py:352-369`), the transformer family is refused (`nn_training.py:174-180`,
  `subword_experiment.py:392-397`), and packing must be asked to track documents
  (`subword_experiment.py:799-818`, `packing.py:166-179`). Tests exist (`tests/test_training_decoder.py:424`,
  `:443`, `:455`). Every measured run has `document_masking` false or absent
  (`nn_training.py:501-513` treats a missing key as `False`).
- **Risk.** The *held-out* scoring path never passes `segment_ids`: `score_model_on_examples` calls
  `model(inputs)` with no mask (`nn_evaluation.py:72`), and `score_transformer`/`_test_partition_metrics`
  route through it (`subword_experiment.py:1001-1008`, `:1055`), as does the boundary's `score()`
  (`nntransformer.py:467-483`, which packs **without** `track_documents` at `:468-472`). Meanwhile the
  in-training validation number *does* mask (`nn_training.py:688-694`). So a masked run's validation loss and
  its reported test loss are computed under **different attention semantics**, and the two are placed in one
  table. Not triggered by any committed record; it would fire on the first masked measurement.

### 2.6 Observation — `context_length` is a budget for the decoder and a table for the transformer

`_validate_positions` refuses any position reaching past `context_length` (`decoder.py:493-521`), so a long
prompt is clipped by the caller instead of the model silently truncating (`nntransformer.py:249` →
`decoding.py:56-81`, right-clipping only, with `truncated` reported at `base.py:246-261`). The transformer
family instead has a real position table sized `context_length × d_model`
(`transformer.py:199`, refused at `transformer.py:299-304`) and its `parameter_count` charges for it
(`transformer.py:165-170`) — one of the reasons E2608's two arms were **not** capacity-matched.

---

## 3. Training

### 3.1 Optimizer

- The Transformer path uses `torch.optim.AdamW(parameters, lr=…, weight_decay=…)` over
  `list(model.parameters())` with **no parameter groups** (`nn_training.py:584-587`). Weight decay therefore
  applies to RMSNorm weights and to every feed-forward bias as well as to matrices — the standard nanoGPT-style
  decay/no-decay split is **not** made. `weight_decay=0.01` in every measured run
  (`data/experiments/E2608/*/report.json`), so this is a real property of the measured numbers, not a latent one.
- The `learning_rate` passed to AdamW is immediately overwritten per step from the schedule
  (`nn_training.py:641-643`), so the constructor value only matters for the first micro-second. Betas/eps are
  torch defaults (`β1=0.9, β2=0.999, ε=1e-8`) and are **not configuration and not recorded** — the schedule
  fields in `TransformerTrainingConfig.as_dict()` (`nn_training.py:187-191`) omit them. Not determined: nothing
  in the repository states them for the measured runs; they are torch-version defaults.
- `training/optimizer.py` is a **separate, numpy, float64 Adam for the character MLP only**
  (`optimizer.py:25-94`), with bias-corrected moments and no weight decay. Its docstring claims "the state is
  not part of the trained artifact (only the final parameters are checkpointed)" (`optimizer.py:10-12`) —
  accurate: `Checkpoint._content` carries arrays, config, vocabulary and dataset digest only
  (`checkpoint.py:65-74`). A reader who assumes the Transformer's optimizer is the in-project one is wrong; the
  Transformer's moments come from torch and are stored under `"optimizer"` (`nn_checkpoint.py:252`).

### 3.2 Learning-rate schedule — verified-correct

`_learning_rate` (`nn_training.py:413-429`) is a pure function of the step: linear warmup
`peak·step/warmup` for `step < warmup` (so step 0 records rate 0, visible in the committed curves, e.g.
`experiments/E2611/results.json` step 0 `learning_rate: 0.0`), then `constant` holds the peak (E2605–E2607 and
the E2611 arms), or `cosine` anneals from `warmup_steps` to `peak·decay_to` at `max_steps`
(`:425-429`, `progress` clamped to [0,1]). Because it is a function of the step rather than scheduler state, a
resumed run reads the same rate the uninterrupted one would — the claim at `nn_training.py:417-419` holds, and
`start_step` is honoured (`nn_training.py:639-641`). `decay_to ∈ [0,1)` and `warmup ≤ max_steps` are validated
(`nn_training.py:128-158`). Measured arms: warmup 200, cosine to 0.1 for the E2608 decoder, constant for the
transformer (`data/experiments/E2608/decoder/report.json` vs `transformer/report.json`) — i.e. E2608 compared
two schedules, which its own record now states.

### 3.3 Gradient clipping — verified-correct

`clip_grad_norm_` runs once per optimizer step **after** accumulation (`nn_training.py:680-685`), its return
value (the **pre-clip** norm) is recorded (`nn_training.py:678-679`, `:703`), and `gradient_clip_norm == 0`
disables clipping by passing `inf` rather than skipping the call, so the norm is still recorded
(`nn_training.py:683`). Tested (`tests/test_training_nn_training.py:395`).

### 3.4 Loss and the label shift

- **Verified-correct:** the shift is exactly one and there is no off-by-one. `packing.py:185-186` sets
  `input_ids = stream[i·w … i·w+L−1]` and `targets = stream[i·w+1 … (i+1)·w−1]` with `w = L+1`, so
  `targets[j]` is the successor of `input_ids[j]` and the last token of a block is a target, never an input
  (`packing.py:14-15`, `:181-194`). `PackedExample.__post_init__` refuses length disagreement
  (`packing.py:53-71`).
- **Observation (deliberate, and it is measured):** positions `i·w` for `i ≥ 1` — the first token of each block,
  one per block — are **never scored**, because they are only ever an input. On the measured test side that is
  1,004,444 scored positions out of `tokens_available` with one gap per block, and the bpc denominator is the
  decoded *concatenation of targets* (`model_metrics.py:141-142`), so numerator and denominator omit the same
  positions. Consistent, and cheap: for E2608's decoder arm, 1,000,448 scored of 1,004,444 packed identifiers.
- **Risk (low, quantified):** the cross-entropy is over *all* target positions, including the document
  separator, and the decoder has no `ignore_index` for 256 (`decoder.py:627-631`): separators are trained and
  scored. `model_metrics` charges them nats and gives them no characters (`model_metrics.py:21-24`,
  `:146`). Measured size: `separator_targets` 137 of 1,000,448 positions (0.014 %) for E2608's decoder, 134 of
  996,608 for E2611, 33 of 254,976 for E2612 (`data/experiments/.../report.json`). Negligible at these
  document lengths; it would not be for a corpus of short documents.
- **Verified-correct:** held-out aggregation is by position count, not a mean of batch means
  (`nn_training.py:394-406`), and evaluation is independent of batch size — tested
  (`tests/test_training_nn_training.py:243`, `:254`).
- **Risk (latent):** that weighting uses `count = int(inputs.numel())` (`nn_training.py:403`), i.e. **every**
  position of the block, while the loss it multiplies averages only the non-`IGNORE_INDEX` positions
  (`decoder.py:630`). `IGNORE_INDEX` is produced by no caller today (a repository search finds it only in
  `decoder.py` and its tests), so no published number is affected. Demonstrated on a synthetic padded pair:
  `evaluate_loss` returned 4.758717 nats over "32 positions" where 18 positions were actually scored, against
  a true per-scored-position mean of 4.729990 — **+0.0287 nats (+0.61 %)**, and an inflated position count. The
  decoder's padding capability is therefore only half-wired: representable in the model
  (`decoder.py:18-21`), unweighted-out in the evaluation helper.
- **Observation:** with `grad_accum_steps > 1` the gradient is the mean of per-micro-batch means
  (`nn_training.py:672`) and the recorded `train_loss` is `step_loss / grad_accum_steps`
  (`nn_training.py:677`), both **unweighted** by the number of positions in each micro-batch. `plan_batch`
  explicitly allows a short final batch (`nn_training.py:299-300`, `:319`), so the last update of an epoch is a
  slightly different estimator than "mean over positions". E2608's transformer arm and both E2611 arms ran with
  `grad_accum_steps=2` (`data/experiments/E2608/transformer/report.json`, `E2611/results.json`).

### 3.5 How packing builds sequences — verified-correct

`pack_units` (`packing.py:123-206`) encodes each unit and appends `DOCUMENT_END` (`packing.py:169-176`), so a
document's final content token predicts the separator and the separator predicts the next document's first
token — content is never predicted across a boundary. Long documents are cut into blocks, not discarded
(`packing.py:181-194`); a short final block is dropped **and reported** (`packing.py:204`,
`PackedBatch.dropped_tail`); zero usable blocks raises rather than handing back an empty loop
(`packing.py:195-200`). Sides are packed separately, which is where the leakage guarantee lives
(`packing.py:5-8`). `segment_ids` track per-position document index when asked (`packing.py:171-176`,
`:187-191`).

**`max_tokens` semantics — verified-correct and measured.** The break is at the *top* of the unit loop
(`packing.py:167-168`), so the stream is the first prefix of whole units whose length reaches `max_tokens`;
no document is half-packed, as `packing.py:141-143` claims. Measured: E2612 declared 20,000,000 and packed
**20,000,402** identifiers from **2,737** of its 6,512 training documents (`data/experiments/E2612/prose/report.json`).

**Observation worth stating in records.** `max_training_tokens` is not the amount of data trained on. E2612
packed 77,822 blocks of 257 but drew `32 × 2400 = 76,800` (`grad_accum_steps=1`), i.e. **98.7 % of one pass**,
leaving ~1,000 packed blocks never trained on. E2608's arms drew 4.21 and 4.22 epochs
(8,192 positions/step × 2,400 = 19,660,800 identifiers, identically for both — the "matched token budget" claim
in `experiments/E2608/record.md` is exactly true, verified by this arithmetic).

### 3.6 Split logic, whole-work splits and `training_group_prefixes`

- **Verified-correct:** `split_text_units` deduplicates by content digest before assigning sides
  (`data.py:203-208`), refuses empty/duplicate identifiers and fewer than three usable documents
  (`data.py:190-215`), assigns sides by a seeded shuffle of indices (`data.py:225-235`) and keeps whole units on
  one side (`data.py:236-249`). `_VALIDATION_FRACTION`/`_TEST_FRACTION` are module constants (0.15/0.15,
  `subword_experiment.py:101-102`) and the fractions are validated (`data.py:169-184`).
- **Verified-correct:** `whole_works=True` groups by `TextUnit.group`, requires every unit to name a group and
  at least three works, and fills sides against declared **character** targets so a few long works cannot
  decide the test set (`data.py:274-333`); an empty side is refused (`data.py:327-333`). Groups come from the
  first path segment of the document identifier (`manifest.py:276-296`) — `pg11/part-0007.md` → `pg11`.
- **Risk (verified numerically).** Two consequences of that grouping rule: a corpus whose documents are all
  directly under one directory collapses to a **single** group, so `whole_works=True` raises "needs at least
  three distinct works" — and for the repository corpus the group of every `docs/...` file is the literal
  string `docs`, i.e. the whole documentation tree is one "work". Nothing warns that the work granularity is
  the first path segment; `manifest.py:279-282` states it, and a caller choosing a corpus layout by directory
  depth gets either a valid work split or a hard failure, never a silently wrong one.
- **Risk — `dropped_duplicates` is lost on the whole-work path.** The document path returns
  `dropped_duplicates=tuple(dropped)` (`data.py:245-249`); `_split_whole_works` returns
  `UnitSplit(train=…, validation=…, test=…)` with **no** `dropped_duplicates` (`data.py:335-339`) even though
  duplicates were already removed from `usable` (`data.py:203-208`). Since the fingerprint digests the sides it
  can see (`data.py:122-132`) and the report prints `documents_per_side["dropped_duplicates"]`
  (`subword_experiment.py:1264-1269`), a whole-work run **reports 0 duplicates dropped while documents were
  silently excluded**. Measured reports all show `dropped_duplicates: 0`
  (`data/experiments/E2608/decoder/report.json`, `E2611/decoder-small`, `E2612/{prose,mixture}`), and I verified
  by digest that the E2608 corpus genuinely has zero duplicate documents (919 documents, 919 distinct sha256s),
  so no published number is wrong today. The accounting bug is live for the next corpus.
- **`training_group_prefixes` — verified-correct with one asymmetry.** Filtering is a `str.startswith(tuple)`
  on the unit's group (`subword_experiment.py:703-729`), an empty selection is refused before packing
  (`subword_experiment.py:719-724`), the tokenizer and the character vocabulary are deliberately fitted on the
  **whole** training side so two arms share one alphabet (`subword_experiment.py:770-773`), and validation/test
  are untouched by the prefix (`subword_experiment.py:778-781`). **Asymmetry:** the character baseline is fed
  the *unfiltered* `split.train`/`split.validation`/`split.test` (`subword_experiment.py:874-889`) while its
  reported `train_windows`/`test_windows` come from `prepared.char_windows`, which uses the filtered/capped sets
  (`subword_experiment.py:783-792`, `:941-943`). Where the character baseline runs (E2605/E2606), a prefix- or
  cap-filtered arm's report states window counts that are not the counts its model trained on. E2608+ skipped
  the baseline for memory (`subword_experiment.py:1113-1114`; the reason string is in
  `data/experiments/E2612/prose/report.json`), so the measured Transformer numbers are unaffected.
- **Seeded shuffling — verified-correct.** `plan_batch` (`nn_training.py:289-319`) is a pure function of
  `(total_examples, update_index, batch_size, seed)`: epoch = `update_index // ceil(total/batch)`, permutation =
  `random.Random(f"{seed}:{epoch}").shuffle(range(total))`, slice = `position*batch … +batch`, last batch may be
  short. A `str` seed goes through `random.seed`'s sha512 path, so it is independent of `PYTHONHASHSEED`. Tests
  cover purity, full per-epoch coverage, short tails and epoch distinctness
  (`tests/test_training_nn_training.py:189-224`). The corpus-level shuffles are separately seeded from
  `effective_split_seed`: training units (`subword_experiment.py:703-729`, `:765-769`), validation/test sampling
  (`subword_experiment.py:1011-1021`), tokenizer sample (`subword_experiment.py:636-654`). `split_seed` is
  distinct from the model seed by design (`subword_experiment.py:211-215`) and a study **requires** it
  (`seed_study.py:155-160`).

### 3.7 Selection and the two losses — verified-correct

`choose_validation_best` (`nn_training.py:322-349`) implements "improvement must exceed `min_delta`", counts
evaluations without it, and treats step 0 as a candidate so a run that learns nothing selects untrained weights
rather than a lucky early step; the loop applies the same rule incrementally and then **refuses to return if
the two disagree** (`nn_training.py:751-764`) — a self-check that is genuinely load-bearing and passes on every
measured run (`completion: budget_ceiling`, `best_step == max_steps` in all four recent reports).
`train_transformer` refuses empty sides and `start_step >= max_steps` (`nn_training.py:561-576`). `evaluate_loss`
puts the model in `eval()` and uses `no_grad` (`nn_training.py:390-393`); `model.train()` is re-asserted inside
the micro-batch loop (`nn_training.py:664`).

---

## 4. Checkpointing, resumption and artifact atomicity

### 4.1 Verified-correct — the torch artifact is the strong one

- **Refusal to clobber.** `save_artifact` raises `CheckpointError` if the target exists unless
  `allow_overwrite=True` (`nn_checkpoint.py:236-241`); the interruption writer passes it for its own resume file
  (`nn_training.py:719-737`). `run_experiment` forwards `plan.overwrite_artifacts`
  (`subword_experiment.py:1170-1171`, `:237-240`).
- **Atomicity.** The `.pt` file is written through `staged_write` (`nn_checkpoint.py:242-255`), which stages in
  the destination directory and swaps on success, unlinking on any `BaseException`
  (`core/atomic.py:30-50`); text artifacts additionally `fsync` and write with `newline=""` so a digest matches
  the bytes (`core/atomic.py:53-67`). Corpus artifacts add their own clobber guard
  (`manifest.py:552-567`), and reports refuse (`subword_experiment.py:1387-1393`). Honest about its limit: the
  directory entry is not fsynced (`core/atomic.py:9-12`).
- **Prevention over detection.** `save_artifact` first calls `artifact.to_model()` (`nn_checkpoint.py:234`) so an
  artifact that cannot be rebuilt from its own declared configuration never reaches disk; `to_model` compares
  tensor **names, shapes and dtypes** against a freshly built model and refuses by name
  (`nn_checkpoint.py:178-216`).
- **Integrity on load.** `torch.load(..., weights_only=True)` (`nn_checkpoint.py:271`), a schema allowlist
  (`nn_checkpoint.py:290-295`), a checksum over metadata **plus every tensor's bytes** recomputed and compared
  (`nn_checkpoint.py:316-329`), block-shape checks (`nn_checkpoint.py:282-288`), the inline tokenizer re-hashed
  to its recorded fingerprint (`nn_checkpoint.py:345-351`) and its size reconciled with the architecture's
  `vocab_size` (`nn_checkpoint.py:352-357`). ADR-0035's "tokenizer inline" is implemented, not aspirational.
- **Resumption is refused, not approximated.** `resume_state_from` and `verify_resume_compatibility` both
  require optimizer state (`nn_training.py:478-483`, `:800-806`) — a usable artifact is not
  a resume point. Corpus fingerprint, architecture dict and tokenizer fingerprint must agree
  (`nn_training.py:485-498`), and any *other* configuration difference is refused with the differing keys named;
  only a **larger** `max_steps` is allowed and it comes back as `budget_extended_from` for the record to print
  (`nn_training.py:500-527`, `:439-456`, `subword_experiment.py:1137-1150`, report field
  `budget_extended_from` at `subword_experiment.py:1320`). The `document_masking` key is back-filled to `False`
  for pre-feature checkpoints (`nn_training.py:504-506`) — correct, and the reason old artifacts still resume.
  `max_steps` is an absolute ceiling, not an increment (`nn_training.py:570-576`, `:639`).
- **The released model is verified before it is read.** `build_release` copies provenance out of the run report
  and the checkpoint rather than accepting typed values (`release.py:203-338`), refuses self-contradictory
  family/schema (`release.py:297-306`), requires non-empty limitations (`release.py:108-113`), `write_release`
  refuses to replace (`release.py:362-368`), `export_release` copies through `staged_write` and re-digests the
  copy against the origin (`release.py:432-441`), and `load_release` re-hashes the weights file and its size
  before a single weight is read (`release.py:462-502`, `load.py:53-68`, `:109-129`).

### 4.2 Risk — the JSON character checkpoint violates the atomicity rule

`Checkpoint.save` is `Path(path).write_text(...)` (`checkpoint.py:182-185`): **not** staged, **not** fsynced, and
**no refusal** when the destination already exists. `SubwordVocabulary.save` is the same
(`subword.py:388-394`) — it creates the parent directory and clobbers. AGENTS.md states artifact writes "are
written atomically and refused from being clobbered unasked"; the two oldest artifact layers do not comply, and
`Chakaso.checkpoint.load`'s checksum will catch the truncation only after the fact
(`checkpoint.py:106-114`). Impact on measured numbers: E2601/E2602 checkpoints are small and re-derivable, and
the tokenizer cache is re-fittable — but the *cache* being non-atomic means an interrupted fit can leave a
half-written vocabulary file whose name is exactly the identity key, and `SubwordVocabulary.load` would then
raise on JSON rather than re-fit (`subword_experiment.py:676-693`). Not determined: whether any such partial
file exists on disk.

### 4.3 Observation — round-trip losses in the artifact format

- `load_artifact` rebuilds the vocabulary from `tokenizer_merges` only (`nn_checkpoint.py:336-338`); the
  metadata document never carries `min_frequency` (`nn_checkpoint.py:142-159`). A loaded artifact therefore
  reports `min_frequency 2` regardless of the floor that fitted it, and a load→save round-trip writes a
  different checksum and a wrong floor. Behaviourally inert (`min_frequency` affects fitting only) but the
  artifact misreports one tokenizer-provenance field that the cache path does guard
  (`subword_experiment.py:687-692`).
- `ModelArtifact.checksum()` covers `run_metadata`, which includes `wall_clock_seconds` and
  `positions_per_second` (`nn_checkpoint.py:142-159`, `nn_training.py:279-280`). So the artifact digest is **not**
  a weight identity: two bit-identical reproductions get different checksums. `build_study_report` records
  `artifact_checksum` per run (`seed_study.py:555`) without comparing it — correct that it is not a protocol
  field, but a reader may mistake it for one. (The companion harness audit raises the same point as its risk 3.)

---

## 5. Numerical correctness risks that could silently corrupt a measured result

Ranked by how much they can move a published number.

**5.1 Padding in the loss.** Not present in the measured runs: packing produces dense, equal-length blocks and
never emits `IGNORE_INDEX` (`packing.py:181-194`; the constant is unused outside `decoder.py` and its tests).
The transformer family's cross-entropy has **no** `ignore_index` at all (`transformer.py:318-321`), so if a
caller ever fed padded blocks to `CausalTransformer` the padding would be trained and scored with an id-`-100`
target — which torch would treat as an out-of-range class rather than skip. Verified-correct for the shape the
pipeline actually produces; a trap for the next caller.

**5.2 Target-shift off-by-one.** None. Proven by construction and arithmetic in §3.4
(`packing.py:185-186`); `PackedExample.__post_init__` (`packing.py:53-61`) cannot catch a *wrong* shift, only a
length disagreement, so the guard here is the slice arithmetic itself plus
`tests/test_training_packing.py` and `test_evaluation_matches_a_manual_mean_over_every_position`
(`tests/test_training_nn_training.py:254`).

**5.3 RoPE applied inconsistently.** Not found. Training and cached stepping share one rotation call site
(`decoder.py:335-336`), keys are rotated before being cached, and the equivalence is tested rather than argued
(`tests/test_training_decoder.py:164`, `:115`, `:139`). `positions` defaults to `arange + already`
(`decoder.py:578-580`), and the scoring path never supplies `positions` (`nn_evaluation.py:72`), so every
block is scored from position 0 — matching training. The transformer family has no rotary path at all
(`transformer.py:199`, `:306`).

**5.4 float32 vs float64.** The Transformer path is float32 end to end (parameters pinned by
`tests/test_training_decoder.py:417`; dtype agreement enforced on load, `nn_checkpoint.py:188-192`), while the
**character** model and its Adam are float64 (`optimizer.py:25`, `:79-87`; `checkpoint.py:33`, `:157`). Both
feed the *same* bpc function (`model_metrics.py:109-156`, via `_CharacterDecoder` at
`subword_experiment.py:515-528`), so the E2605/E2606 tables compare a float64 estimator against a float32 one.
The per-position nats for the Transformer come from `torch.log_softmax` in float32 then `float()`-widened
(`nn_evaluation.py:73-79`), and are summed with `math.fsum` in float64 (`model_metrics.py:139`). Magnitude: far
below the reported 4 decimal places for 10⁶ positions; stated for honesty, **not quantified**.

**5.5 Non-determinism.** The only determinism control in the whole path is `torch.manual_seed(config.seed)`, and
only when no model is supplied (`nn_training.py:578-580`). There is no `set_num_threads` and no
`use_deterministic_algorithms` anywhere in `src/`. Consequences, in order of severity:

- **Checkpoint writes perturb the global RNG.** `save_artifact` calls `artifact.to_model()`
  (`nn_checkpoint.py:234`), which constructs a model and runs `_initialize` → `nn.init.normal_` draws. Verified:
  the torch RNG state after `save_artifact` differs from before, and the next `torch.randn(1000)` differs. Since
  `run_experiment` always passes a `checkpoint_path` (`subword_experiment.py:1159`) and the interim writer fires
  at every evaluation by default (`nn_training.py:718`, `checkpoint_every=0`), a run's RNG trajectory depends
  on how often it checkpointed. With `dropout=0` (all measured runs) nothing else consumes the stream, so the
  measured numbers are safe; the moment `dropout>0`, **`checkpoint_every` becomes a numerical setting**, and
  `verify_resume_compatibility` treats it as an ordinary configuration field (`nn_training.py:500-516`) while
  `to_model()` also allocates a second full model at every checkpoint.
- **Thread count.** `describe_environment` *records* `torch_threads` because "a run on 8 threads and a run on 18
  are not the same numerical procedure" (`nn_checkpoint.py:371-385`) — and nothing pins or enforces it. All
  four recent reports happen to record `"14"`. Reproduction is therefore tolerance-based, exactly as
  `REPRODUCTION_TOLERANCES` says (`seed_study.py:100-111`), and the AGENTS.md "reproduced E2605 exactly" claim
  is a claim about one machine configuration.
- **Resumed runs and dropout.** A resumed run re-seeds only if it builds its own model
  (`nn_training.py:578-580`); the batch plan and the rate schedule do line up (`nn_training.py:598-603`,
  `:655`). Not determined for dropout, because no measured run uses it.

**5.6 Study-level protocol checking has named gaps.** `SeedStudy` compares only the fields in
`_PROTOCOL_FIELDS` (`seed_study.py:74-95`) and `expected_identity` (`seed_study.py:168-192`): experiment,
corpus/split fingerprints, split seed, tokenizer fingerprint and size, architecture dict, parameter count,
batch/eval/patience/min_delta/lr/warmup/clip/weight_decay/max_steps, character steps, and test positions and
characters. **Absent: `schedule`, `decay_to`, `grad_accum_steps`, `document_masking`, `whole_work_splits`,
`max_training_tokens`, `tokenizer_sample_characters`, `shuffle_training_units`.** A report left in a run
directory by a sibling experiment that differed only in one of those would pass `_require_same_protocol`
(`seed_study.py:249-269`) and be aggregated — and `architecture` does not contain the schedule either
(`decoder.py:152-165`). Mitigating: seeds within one study come from one plan object
(`seed_study.py:198-217`), so this bites only on reuse of a foreign report, which is precisely the case the
function exists to catch. The measured studies are unaffected: the committed reports do carry these fields, and
`protocol_agreement` verified `split_fingerprint`, `tokenizer_fingerprint` and identical
`positions_scored 254,976` across E2612's arms (`experiments/E2612/record.md`; companion audit Finding 3).

**5.7 Baseline and metric wiring — verified-correct.** Uniform is `ln(V)` (`baselines.py:28-37`,
`subword_experiment.py:961`) and the unigram is Laplace-smoothed, **fitted on the arm's own packed training
targets** and scored on the same test targets as the model (`subword_experiment.py:952-973`,
`:1035-1066`); `metrics_from_scores` is the single arithmetic for model and both baselines. Two wording
issues: `baselines.py:12-14` says the comparison is over the "*validation* tokens" while the code scores the
**test** targets; and `subword_experiment.py:919-920` fits the character unigram on
`prepared.char_windows["train"]` targets while `run_minibatch_training` trained on the unfiltered split text
(§3.6). Neither affects a Transformer headline. The per-register partitions in E2612 are **independent capped
packs** (mathematics 250,624 · prose 252,672 · science 251,392 · technology 253,184 · medicine 46,592 scored
positions, summing to ~1.05M against the shared test pack's 254,976), so the partition numbers are not a
decomposition of the overall number and cannot be averaged into it as if they were — the code comment calls this
"a few tokens of tail" (`subword_experiment.py:1027-1032`), which understates it. The companion audit's Finding
4 covers the same ground with the weighting arithmetic.

**5.8 Inference-side numeric guards — verified-correct.** `TokenSelector.select` refuses a wrong-width row and
any **non-finite logit** rather than guessing an argmax (`decoding.py:223-236`) — a NaN cannot become a word.
Greedy ties go to the lowest identifier (`decoding.py:239-241`); the order of operations (penalty → temperature
→ top-k → top-p → draw) is stated and implemented (`decoding.py:157-171`, `:238-270`), with the repetition
penalty sign-correct and count-scaled (`decoding.py:196-213`), which is a deliberate deviation from the fixed
penalty of the standard formula. `StopScanner` matches only produced text and bounds its window
(`decoding.py:137-154`); `DOCUMENT_END` ends a run as `STOP`, never as content (`nntransformer.py:292-298`); a
cached run refuses to slide its window and reports `CONTEXT` instead of silently dropping the start of an
answer (`nntransformer.py:305-314`); a right-clipped prompt is reported (`decoding.py:56-81`, `base.py:246-261`).
Sampled requests without an explicit seed derive one from the prompt and parameters (`nntransformer.py:80-100`),
which makes a request replayable but means **E2610's decoding diagnosis is reproducible per request, not per
process** — the correct choice for a boundary, and worth remembering when a sample is cited.

---

## 6. Ranked summary: what could distort a measured number

1. **The merge-loop heap loses candidates** — `subword.py:463-499`; on real prose the pick differs from the
   documented rule at 287/343 steps (286 with a strictly lower true count). Vocabulary fingerprint changes;
   reproduction of E2608/E2611's tokenizer now depends on a git-ignored cache file.
2. **`6f3274f`'s "no vocabulary moves" claim is false on the measured corpus** — re-fitting `HEAD` on the exact
   E2608/E2611 input diverges at merge rank 5 (`a1ef04f64739cd4b` vs the measured `b597eb34f13d9058`) and costs
   0.49 % of characters-per-identifier on the measured test side, ≈ +0.008 bpc at unchanged per-token loss.
3. **`target_size` is not asserted after a fresh fit** — `subword_experiment.py:694-700`, `:495-512`; the loop
   can end short (25/300 synthetic trials), so a run can train a smaller vocabulary than its record declares.
4. **Document masking is not applied by the held-out scorer** — `nn_evaluation.py:72`,
   `subword_experiment.py:1001-1008`, `nntransformer.py:467-472` vs `nn_training.py:688-694`: a masked run's
   validation and test numbers use different attention semantics.
5. **Whole-work splits silently report 0 dropped duplicates** — `data.py:335-339` vs `data.py:245-249`.
6. **`evaluate_loss` weights by block width while the loss skips padding** — `nn_training.py:403-405` vs
   `decoder.py:630`; demonstrated +0.61 % on synthetic padding; latent today.
7. **Checkpoint writes consume the global RNG** — `nn_checkpoint.py:234` → verified state change; safe at
   `dropout=0`, makes `checkpoint_every` a numerical setting otherwise.
8. **Weight decay applied to norms and biases; betas/epsilon undeclared and unrecorded** —
   `nn_training.py:584-587`, `:187-191`.
9. **No thread/determinism pinning; reproduction is tolerance-based** — `nn_checkpoint.py:371-385`,
   `seed_study.py:100-111`.
10. **`_PROTOCOL_FIELDS` omits schedule, decay, accumulation, masking, work-splits and caps** —
    `seed_study.py:74-95`.
11. **JSON checkpoint and vocabulary saves are neither atomic nor clobber-guarded** — `checkpoint.py:182-185`,
    `subword.py:388-394`.
12. **`min_frequency` is dropped on artifact load** — `nn_checkpoint.py:336-338`, `:142-159`; the artifact
    misreports the fitting floor.

**Verified-correct and worth saying so plainly:** the parameter arithmetic (3,413,760 = formula = built model;
1,018,240 and 643,840 likewise); the one-token label shift with no off-by-one; per-position held-out
aggregation; the incremental/self-checking validation selection with a hard disagreement refusal; RoPE
consistency between training and cached decoding; the pre-norm/residual-scaled initialization; the tokenizer
inline + fingerprint re-check; resume refusal without optimizer state and the named single-field budget
extension; refusal of non-finite logits and of all-padding batches; digest-verified release loading.

## Not determined

- Adam betas/epsilon and torch reduction-order effects at a different thread count for the measured runs.
- The true bpc shift from a re-fitted tokenizer (only the chars-per-identifier change is measured; the
  re-segmented training signal is not).
- Whether `symbol_to_bytes` recursion depth can be exceeded by a real document (no such corpus constructed).
- Whether any partial `.partial` artifact exists on disk after an interrupted fit (§4.2).
- Whether any committed checkpoint was written before the `document_masking` field existed and later resumed
  (the back-fill at `nn_training.py:504-506` makes both paths legal and indistinguishable in the metadata).
