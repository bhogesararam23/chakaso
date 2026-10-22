# E2608 — the reference model: a rotary decoder at a budget declared to converge, against a token-matched transformer arm

**Type:** Research experiment, **preregistered** (this file was committed before either arm ran)
**Date preregistered:** 2026-10-22
**Decision record:** [ADR-0037](../../docs/decisions/ADR-0037-reference-model-sized-by-measured-throughput.md)
**Depends on:** E2605 (the measured architecture), E2606 (the seed-spreading method), E2607 (the external
corpus result and the ceiling finding), ADR-0036 (a measured corpus stays re-trainable)

## Motivation

E2607 measured the same protocol on prose this project did not write, and returned two findings. The one the
release wanted: the subword Transformer's advantage over the character architecture survives the change of
data (gap 1.2042 externally against 1.2143 internally). The one it did not want: **none of the three runs
early-stopped** — every one selected step 2,400, the ceiling of the declared budget, with validation loss
still falling at the last evaluation, which makes its 2.0176 bits per character a floor rather than a
converged level. And every checkpoint ever measured here generates text that repeats.

So two questions have to be answered before any capacity question is worth asking. First: was E2607's curve
still falling because the budget stopped it, because a constant learning rate never anneals, or because the
architecture is at its ceiling? Second: does the architecture the release path actually needs — rotary
positions, an incremental key/value cache, masked targets — learn better from the same text at the same
amount of text? This experiment is one protocol designed to answer both, on a corpus 8.8× larger than E2607's
and sized by measured throughput rather than by ambition.

## Hypotheses, stated before any run

- **H1 (schedule).** With cosine decay declared in advance and a budget the measured throughput says is
  affordable, at least one arm will select a step strictly below its ceiling. If both arms still select the
  ceiling, the schedule is not what truncated E2607, and that is a result.
- **H2 (architecture).** The decoder arm reaches lower held-out bits per character than the transformer arm
  when both consume the same number of tokens per optimizer step and the same number of steps.
- **H3 (behaviour).** The decoder arm's greedy generations still loop. This is stated as a hypothesis to be
  *confirmed*, not hoped away: no mechanism in this experiment is designed to fix degeneration, and a lower
  loss that still produces repeated text is the expected outcome of a 3.4M-parameter model on 4.7M training
  tokens.

## Dataset, licence and provenance

| Field | Value |
| --- | --- |
| name / version | `gutenberg-english-prose-v2` / `1.0.0` |
| declared sources | 24 Project Gutenberg texts, listed with their own titles, authors, URLs and byte digests in [`configs/corpora/gutenberg-english-prose-v2.toml`](../../configs/corpora/gutenberg-english-prose-v2.toml) |
| licence | Public domain in the United States, distributed under the Project Gutenberg License, which permits copying, giving away and re-use. Each file's own preamble states that users outside the United States must check local law; that sentence is part of the recorded licence. |
| retrieved | 2026-10-22, over TLS with certificate verification, by the operator step `curl -o data/corpora/gutenberg-v2/pg<id>.txt https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt` for each declared number |
| documents / characters | 919 documents, 14,490,005 characters (14,729,790 bytes) |
| corpus fingerprint | `e9fc2e92f59d88ea` |
| snapshot | `data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json`, checksum `8eacb49f8d244e65db63f80796d76a026f532252899b8f85c7631ac3b9a2415f` |
| data card | [`docs/data/gutenberg-english-prose-v2.md`](../../docs/data/gutenberg-english-prose-v2.md) — measured composition, document-length percentiles and duplication screen |
| redistribution | The corpus bytes are **not committed**; they live under `data/`, git-ignored, per [ADR-0020](../../docs/decisions/ADR-0020-experiment-artifacts.md). The catalogue and the per-document digests are committed, so the corpus is rebuildable from the URLs and verifiable against them. |

Preparation was performed by `chakaso corpus prepare`, which verifies every declared byte digest before
ingesting anything and caches each source's documents keyed by that digest. Near-duplicate screening over
919 documents at 8-word shingles compared 2 candidate pairs and found no group above 0.5 similarity; the
exact-duplicate count is 0. The largest single source supplies 14.6 % of the characters, so no one author
dominates this corpus the way one author dominates `chakaso-repo-text-v1`.

**Relation to E2607's corpus.** Three of the 24 titles (`pg11`, `pg84`, `pg6130`… of which `pg11` and `pg84`
are shared) also appear in `gutenberg-english-prose-v1`. This is recorded, not hidden: E2608 is not a
holdout for E2607's model, and no cross-experiment claim of independence is made from it. Documents are
content-addressed, so the shared titles appear here as their own digest pairs, and the split below assigns
them.

## Split

Document-granular three-way split by `chakaso.training.data.split_text_units`, validation fraction 0.15,
test fraction 0.15, **split seed 7 for every arm** (a study that moved the split would compare runs on
different text):

| Side | Documents | Purpose |
| --- | --- | --- |
| train | 643 | tokenizer fitting, gradient signal |
| validation | 138 | checkpoint selection and early stopping, nothing else |
| test | 138 | reported held-out metrics, read once per arm after selection |

Packing at each arm's own context length, terminated per document. For the decoder arm at context 256 this
gives 18,252 / 3,981 / 3,908 blocks, so **one epoch is 570 optimizer steps at batch 32**. The split
fingerprint and the per-side identifiers are recorded by the run itself; no document appears on two sides.

## Tokenizer

Byte-pair encoding (`chakaso.training.subword`), UTF-8 byte base, the named pre-tokenizer
`word-run|whitespace-run|single-symbol`, target 4,096 identifiers, `min_frequency` 3, fitted on the
**training side only**. Measured fingerprint `b597eb34f13d9058` at 3,839 merges, produced in 467 s on this
machine; the fitted file is shared between the arms through the plan's `tokenizer_cache_dir`, which is keyed
by corpus fingerprint, split fingerprint, target size and frequency floor, so a reuse cannot be a tokenizer
fitted from different data.

## Model and configuration — the two arms, matched in tokens

| Field | Arm A: decoder (reference) | Arm B: transformer (measured geometry) |
| --- | --- | --- |
| family / checkpoint schema | `decoder` / `chakaso-decoder-checkpoint/1` | `transformer` / `chakaso-transformer-checkpoint/1` |
| position encoding | rotary, `rope_theta` 10,000, no position table | learned table, 128 rows |
| normalization | RMSNorm, pre-norm | LayerNorm, `norm_first` |
| attention | explicit q/k/v/o, no biases, incremental KV cache | `nn.TransformerEncoderLayer`, fused in-proj, biases |
| feed-forward | up/GELU/down, biases | torch block default |
| embeddings | tied | tied |
| `d_model` / layers / heads / ff | 256 / 4 / 8 / 640 | 128 / 3 / 4 / 384 |
| context length | 256 tokens | 128 tokens |
| parameters | 3,413,760 declared and counted from the built model | 1,037,056 |
| batch × accumulation | 32 × 1 | 32 × 2 |
| **tokens per optimizer step** | **8,224** | **8,256** |

The token counts differ by 32 (one sequence), because a block is `context + 1` identifiers wide. This is
stated rather than rounded away: the arms are matched to within 0.4 % of the text per step, not exactly.

Both arms: AdamW, `learning_rate` 6e-4, `weight_decay` 0.01, `warmup_steps` 200, gradient clipping at 1.0,
`eval_every` 100, `patience` 3, `min_delta` 0.0005, `max_steps` **2,400** (about 4.2 epochs), **seed 7**,
float32 on CPU, 14 torch threads.

Arm A's schedule is **cosine, decaying to 0.1 of the peak rate**. Arm B keeps the **constant** schedule that
produced E2605–E2607, because changing both arms' schedules at once would confound the architecture question
with the schedule question. H1 is tested by whether either arm selects below its ceiling; the arms are not
schedule-matched and the record says so.

## Seeds and number of runs — declared, with the reason measured

**One seed per arm (seed 7), two runs.** The reason is measured cost, not preference: arm A costs 1.618 s per
step, so 2,400 steps is 65 minutes of gradient work plus about 20 minutes of validation evaluation inside
the loop and roughly the same again for the final held-out scoring. A three-seed study of arm A is
5 to 6 hours of a machine that has 1.5 GB free and is running nothing else. E2606 established the
multi-seed protocol on the architecture it belonged to; putting a spread on arm A is E2609, declared here as
the follow-up rather than attempted here as a fourth thing half-finished.

No seed is chosen by its score. Seed 7 is the seed every experiment in this repository has already used, so
it cannot have been selected for this one.

## Primary metric and baselines

Held-out **bits per character** on the test side (`chakaso.training.model_metrics`), computed over every
packed test position with the decoded-character denominator, identical arithmetic to E2605–E2607 so the
levels are comparable across records even though the corpora are not. Secondary: nats per token, perplexity,
token accuracy, the number of scored positions and the number of excluded separator targets — all recorded.

Baselines, both computed from this corpus's own statistics:

- **uniform** — every identifier equally likely, so `ln(4096)` nats per token, which becomes a bits-per-
  character figure through the measured characters-per-token of the test side. It is the floor a model must
  beat to have learned anything about the identifier distribution;
- **unigram** — training-side identifier frequencies with add-one smoothing (`UnigramBaseline`), the same
  form E2605 to E2607 reported.

The character-model baseline is **not run** on this corpus: it materializes one window per character position,
which at 14.5M characters exceeds the memory measured available. The report records the skip and its reason
rather than silently dropping the comparison. Consequently E2608 does **not** reproduce E2607's
architecture-to-architecture gap; it compares the two trained architectures against the two trivial
baselines and against each other.

## Generation evaluation protocol

Greedy (`temperature` 0.0) samples, 64 new tokens, from **prompts taken from this corpus's own test side**
(`--sample-prompts held_out`: the first 64 non-whitespace characters of the first test documents, recorded by
document identifier), because prompts quoting Chakaso's section titles would measure recall of a corpus the
model never saw. The suite is `chakaso.training.generation_eval`: periodic runs, tail loops, longest repeated
block, n-gram repetition ratio, plus the hostile inputs (empty, single character, non-ASCII, oversized). Raw
sample text is retained in `data/experiments/E2608/generation-*.json`; the aggregate never hides a sample.

**How the suite is run for this experiment, stated before the results.** The run report already carries the
raw samples, and wiring the generation suite *inside* the experiment driver would mean arm B is executed by
different code from arm A — a report-shape change is not a training change, but an experiment whose two arms
are produced by two versions of the driver is a comparison the record would have to qualify. So both arms are
run by the same commit, and `chakaso.training.generation_eval` is then executed over **both saved artifacts**
through `chakaso.models.load_model`, one JSON per family with the raw text retained, in the same way E2606 and
E2607's generation files were produced. The in-report wiring ships with this release and is exercised by the
next experiment (E2609) rather than retro-fitted onto this one. A run's own `samples` block and the suite's
output are the same generations, so no second sampling pass decides H3.

## Success, inconclusive and failure — fixed in advance

- **H1 success:** at least one arm's selected `best_step` is strictly below 2,400. **H1 failure:** both arms
  select the ceiling, which means the schedule was not the truncation and the next hypothesis is data or
  capacity.
- **H2 success:** arm A's test bits per character is below arm B's **and** below both trivial baselines.
  **Inconclusive:** arm A below baselines but not below arm B — a result about the corpus, not the
  architecture. **H2 failure:** arm A above arm B, i.e. the release architecture is worse on held-out loss
  than the one it is meant to replace, which would be reported as the reason to reconsider it.
- **H3 success (the uncomfortable one):** the decoder arm's samples contain a periodic run of ≥ 3 in the
  majority of prompts. **H3 failure:** they do not, which would be the first non-degenerate output this
  project has measured and would be reported as a surprise with its raw text attached.
- A missing or failed run is never aggregated. Both arms must complete; if one does not, the experiment is
  reported as incomplete and no comparison is drawn.

## Failure and missing-run policy

A run that raises is recorded with its error text and its partial artifacts, and the aggregate is not built
over it. Re-running an arm requires deleting its directory or declaring `--overwrite`; a report that already
exists is never silently replaced, per ADR-0035's rule applied to experiment artifacts. Resumption is
verified: `--resume-from` refuses a checkpoint whose corpus fingerprint, architecture, tokenizer fingerprint
or configuration disagrees, and a raised `max_steps` is reported as an extension, not absorbed.

## Expected compute, measured before this was written

From the throughput measurement in ADR-0037 (12 timed steps per configuration, this machine):

| Arm | Step time | Tokens/s | 2,400 steps | Peak RSS |
| --- | --- | --- | --- | --- |
| A decoder | 1.618 s | 5,081 | ~65 min + evaluation overhead inside the loop | 2,126 MB |
| B transformer ×2 micro-batches | ~0.60 s (0.300 s per 4,128 tokens) | ~13,760 | ~24 min + evaluation overhead | 928 MB |

Technical pilot `E2608SMOKE2` (same corpus, same tokenizer cache, same code path, 8 steps at the full
configuration, its own experiment id, run before either arm) measures the end-to-end cost including the
final held-out scoring, and its numbers are reported here rather than aggregated anywhere. Run 2026-10-22 on
`Windows-11-10.0.26200-SP0`, Python 3.13.5, torch 2.13.0+cpu, numpy 2.4.0, 14 torch threads, device cpu:

| What | Value |
| --- | --- |
| Configuration | d256, 4 layers, 8 heads, feed-forward 640, context 256, vocab 4096, RoPE θ 10,000, tied embeddings — **3,413,760 parameters** |
| Budget run | 8 steps, warmup 2, evaluation every 4, patience 2, cosine to 0.1, batch 32, no accumulation |
| Outcome class | `budget_ceiling` (the run was never allowed to converge; this is the expected label for a pilot) |
| Throughput | **4,561 positions/second** over gradient time only; 8 steps is 65,536 positions |
| Wall clock | **239.35 s** total, of which 12.98 s was corpus preparation and tokenizer work — so ~226 s for 8 steps, two held-out evaluations, one final evaluation and the samples |
| Held-out scoring | each scheduled evaluation scores the validation split (1,019,136 positions); the final pass scores validation and test (1,000,448 positions, 2,156,862 characters) — so the loop's cost is dominated by evaluation, not by the steps |
| Curve | step 0 train 8.3853 / val 8.3822 → step 4 6.3642 / 6.2774 → step 8 6.1286 / 6.1129 nats |
| Final test metrics | 4.0808 bits/character, 6.0982 nats/token, perplexity 445.06, token accuracy 0.3557 |
| Baselines on the same split | unigram 3.2263 bits/character, uniform 5.5661 — the model is still worse than a unigram after 8 steps, which is what 8 steps of a 3.4 M-parameter model means |
| Tokenizer | fingerprint `b597eb34f13d9058`, 4,096 identifiers, 3,839 merges, minimum frequency 3, **reused from the cache** (so an arm does not pay the ~467 s fit again) |
| Artifacts | `E2608SMOKE2-decoder.pt` (usable) and `E2608SMOKE2-decoder-resumable.pt`, both written atomically and refused on clobber |
| Generations | 3 continuations of held-out test-side prompts (`pg120/part-0002.md@954dc8227aca`, `part-0005`, `part-0016`), each 24 tokens, `finish_reason: length`, and each one **24 space characters** |

Three things this pilot was for, and what it said. **The pipeline works end to end** — snapshot verification,
cached tokenizer, either-family training, checkpoint writing, held-out scoring and generation through
`chakaso.models.load_model` all ran on the decoder family. It did **not** exercise experiment-level resume:
neither smoke run reports `resumed_from_step`, so the resume path is covered by unit tests only until an arm
needs it. **The evaluation, not the training, is the expensive part**: 8 steps
of gradient work are ~14 s at the measured rate, and the rest of the 239 s is four passes over a million
held-out positions (three scheduled validation evaluations, one final validation-and-test pass) plus the samples,
which is why the arms' declared `eval_every 100` is a cost decision as well as a statistical one. **The samples are degenerate at step 8**, which is expected and recorded rather than
hidden: a pilot is not evidence about capability, and the space-run output is the same failure mode the
generation suite has reported for every checkpoint in this repository so far.

Two deviations belong to the pilot itself, both fixed after it ran rather than smoothed over. Its report
stated `chakaso-transformer-checkpoint/1` for an artifact whose own header declares
`chakaso-decoder-checkpoint/1` — the report's artifact block hardcoded the schema string for every family,
which would have propagated into a release contract; the block now reads the schema the run wrote (commit
`ceb96a0`), and the pilot's report predates that fix. And the record's one-epoch arithmetic said 570 steps
where the run computes 571 (`ceil(train_examples / batch_size)` over 18,252 blocks at batch 32); the figure
used for the budget statements below is the run's own 571.

## Known limitations and threats to validity

1. **One seed per arm.** A single draw cannot separate a real difference from initialization luck. E2606
   measured the size of that luck for the transformer family at 0.0077 bits per character; nothing here
   measures it for the decoder family, and the comparison will be read with that gap in mind.
2. **Different schedules across arms.** H2 compares architectures that do not share a learning-rate policy.
   The alternative (cosine for both) would have answered a different question and left H1 untested.
3. **The corpus is literature.** 24 public-domain English texts, 19th-century and classical, no instruction
   data, no questions, no answers, and a documented character alphabet of 146 symbols. A model that learns
   their surface is not learning anything a product would call language ability.
4. **No character baseline**, for the memory reason stated, so E2607's gap-to-the-character-model number is
   not reproduced here.
5. **Timings are contaminated by the machine's other work.** Peak RSS 2.13 GB against 1.5 GB free at
   measurement time invites paging; a step time measured here is a property of this machine right now, and
   no timing number in this record will be presented as intrinsic to the model.
6. **Shared titles with E2607's corpus.** Stated above; no independence claim is made between the two.
7. **Prompt provenance is the test side of the same split** the reported metrics are computed on. The
   prompts are samples of text the model was scored on but never trained on, which is the honest available
   option for a single-seed run and is disclosed rather than smoothed.

## Planned outputs

- `data/experiments/E2608/` per arm: report, tokenizer (shared cache), resumable artifact, final artifact,
  generation JSON with raw samples.
- `experiments/E2608/record.md` — this file, completed with results, deviations, and a conclusion that names
  which of H1/H2/H3 held.
- `data/experiments/E2608/release/` — the reference checkpoint packaged under the release contract
  (`model.json` + digest verification) and its model card, only if arm A completes and beats the baselines.
- No weights, no corpus bytes, no checkpoints in Git.

## Commands, exactly as run

Nothing in this record is reproducible by description alone, so the invocations are recorded verbatim. Both
arms share these arguments; only the family, context width, architecture size, accumulation and schedule
differ, and they are listed per arm.

```bash
COMMON=(
  --root . --experiment E2608
  --corpus-snapshot data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json
  --expected-corpus-fingerprint e9fc2e92f59d88ea
  --tokenizer-cache-dir data/experiments/tokenizers
  --tokenizer-min-frequency 3 --vocab-size 4096
  --no-character-baseline --sample-prompts held_out --samples-per-prompt 64
  --batch-size 32 --learning-rate 0.0006 --weight-decay 0.01
  --warmup-steps 200 --eval-every 100 --max-steps 2400
  --patience 3 --min-delta 0.0005 --seed 7 --split-seed 7 --verbose
)

# arm A — the rotary decoder, cosine decay to 0.1
.venv-nn/Scripts/python.exe -m chakaso experiment model "${COMMON[@]}" \
  --work-dir data/experiments/E2608/decoder \
  --model-family decoder --context-length 256 --d-model 256 --num-layers 4 --num-heads 8 \
  --feedforward-dim 640 --rope-theta 10000 --grad-accum-steps 1 \
  --schedule cosine --decay-to 0.1 \
  --write-report data/experiments/E2608/decoder/report.json

# arm B — the measured transformer, constant schedule, token-matched by accumulation
.venv-nn/Scripts/python.exe -m chakaso experiment model "${COMMON[@]}" \
  --work-dir data/experiments/E2608/transformer \
  --model-family transformer --context-length 128 --d-model 128 --num-layers 3 --num-heads 4 \
  --feedforward-dim 384 --grad-accum-steps 2 --schedule constant \
  --write-report data/experiments/E2608/transformer/report.json
```

The pilot was the same `COMMON` block with `--experiment E2608SMOKE2 --work-dir
data/experiments/E2608-smoke2 --max-steps 8 --eval-every 4 --warmup-steps 2 --samples-per-prompt 24`. The
snapshot it points at is not the working tree: `chakaso corpus prepare` wrote it from the catalogue
`configs/corpora/gutenberg-english-prose-v2.toml`, and `chakaso corpus verify` re-checks it, so a re-run after
any repository edit trains on the same bytes this record names.

## Follow-up this record commits to

1. **E2609** — arm A across pre-declared seeds (5, split fixed), reporting mean ± sample SD bits per
   character, using E2606's method unchanged.
2. If H2 succeeds and H3 confirms looping: a generation-behaviour experiment, not a capacity one — the
   measured question becomes decoding and data diversity, since loss is already improving.
3. If H2 fails: the release path keeps the measured architecture and this decision is revisited with the
   curves attached.

## Results

Measured 2026-10-22 on this machine (`Windows-11-10.0.26200-SP0`, Python 3.13.5, torch 2.13.0+cpu,
numpy 2.4.0, 14 torch threads, device cpu), from the two reports on disk:
`data/experiments/E2608/decoder/report.json` and `data/experiments/E2608/transformer/report.json`. Both arms
completed; neither raised; neither early-stopped.

### The two arms

| | arm A — decoder | arm B — transformer |
| --- | --- | --- |
| Architecture | d256, 4 layers, 8 heads, FF 640, context 256, RoPE θ 10,000, tied embeddings | d128, 3 layers, 4 heads, FF 384, context 128, sinusoidal-free learned positions, tied head |
| Parameters | **3,413,760** | **1,037,056** |
| Checkpoint schema | `chakaso-decoder-checkpoint/1` | `chakaso-transformer-checkpoint/1` |
| Artifact digest (first 16) | `4ad9efc0fc89fdbe` (`E2608-decoder.pt`) | `09c1aee67fa990d7` (`E2608-transformer.pt`) |
| Schedule | cosine to 0.1 | constant (the measured E2605–E2607 protocol) |
| Optimizer batch | 32 × 256 = 8,192 positions/step, accumulation 1 | 32 × 128 × 2 = 8,192 positions/step, accumulation 2, effective batch 64 |
| **Total training positions** | **19,660,800** | **19,660,800** — the token budget matched exactly, as declared |
| Seed / split seed | 7 / 7 | 7 / 7 |
| Tokenizer | `b597eb34f13d9058`, 4,096 identifiers, 3,839 merges, reused from cache | identical (same cache, same corpus fingerprint) |
| Throughput (gradient time only) | 4,821.2 positions/s | 12,098.1 positions/s |
| Wall clock | 6,024.06 s (100.4 min) | 2,322.89 s (38.7 min) |
| Validation bits/character | 1.7100 | 1.9357 |
| **Test bits/character** | **1.6865** | **1.9043** |
| Test nats/token | 2.5203 | 2.8455 |
| Test perplexity | 12.43 | 17.21 |
| Test token accuracy | 0.5057 | 0.4613 |
| Scored test positions | 1,000,448 over 2,156,862 characters | 996,608 over 2,148,397 characters |
| Uniform baseline (same packing) | 5.5661 | 5.5666 |
| Unigram baseline (same packing) | 3.2263 | 3.2265 |
| Selected step / outcome class | 2,400 / `budget_ceiling`, early stopped: false | 2,400 / `budget_ceiling`, early stopped: false |

Both arms beat both trivial baselines by a wide margin, and the decoder arm is 0.2178 bits per character below
the transformer arm on the same tokens, the same split, the same tokenizer and the same seed.

### Curves, and what they were still doing at the end

Validation loss in nats, evaluated on the 1,019,136-position validation split every 100 steps:

| Step | arm A | arm B |
| --- | --- | --- |
| 0 | 8.3822 | 8.3496 |
| 100 | 4.3212 | 5.4460 |
| 500 | 3.0681 | 3.6721 |
| 1,000 | 2.7604 | 3.2546 |
| 1,500 | 2.6325 | 3.0610 |
| 2,000 | 2.5661 | 2.9409 |
| 2,300 | 2.5457 | 2.8871 |
| **2,400 (final)** | **2.5436** | **2.8790** |

Neither curve had turned up by its last evaluation. Arm A's final 100 steps still improved validation loss by
0.0021 nats; arm B's by 0.0081. The budget ended, the models did not.

### Generation evaluation

Run with `chakaso.training.generation_eval` over each saved artifact through `chakaso.models.load_model`, five
fixed continuation probes at greedy decoding plus four hostile probes, raw text retained in
`data/experiments/E2608/generation-E2608-decoder.json` and `…-transformer.json`. This is the procedure the
protocol section declared before the arms ran: both arms were trained by the same commit, and the in-report
wiring of this suite (`8244c6b`) landed after them, so the numbers below come from the suite executed over the
two artifacts rather than from a report field.

| | arm A — decoder | arm B — transformer |
| --- | --- | --- |
| Tail-looping samples | **0 of 5** | **0 of 5** |
| Samples with a periodic run of ≥ 3 | **0 of 5** | **1 of 5** (`list-continuation`, 3 repeats) |
| Mean bigram repetition ratio | **0.5538** | **0.7012** |
| Longest repeated block, worst sample | `' of the South Testament'` × 2 | `' the whale of the whale of '` × 2 |
| Empty outputs | 0 | 0 |
| Refusals (`ModelError`, documented) | 1 — `empty-prompt` | 1 — `empty-prompt` |
| Crashes (any other exception) | **0** | **0** |
| Samples that hit their token bound | 5 of 5 | 5 of 5 |

Two samples from each arm, verbatim, because the statistics above are not the evidence and a reader is entitled
to the text:

> ` very unfortunate to the Error of the Emperor of the\nEmperor of the Soule of the Soule` — arm A, `document-heading`
>
> ` of the world.\n\nI am sure you are not in the world of the world` — arm A, `table-row`
>
> ` man of the whale of the whale, and the same man of the\nwhale, and the whale of the whale of the whale` — arm B, `document-heading`
>
> ` in the same time of the same time to the same\ntime to the same time o` — arm B, `status-section`

The suite's own limit travels with these files: *surface statistics only — no semantic quality, usefulness or
groundedness is measured here, and none may be inferred from these numbers.*

## Interpretation — which hypothesis held

**H1 (the schedule was the truncation): FAILURE, as declared.** Both arms selected step 2,400, the ceiling, and
both were still improving at it. Cosine decay to 0.1 did not turn a truncated run into a converged one. What
this does *not* establish is that the schedule is useless: it establishes that at 19.7 million positions neither
architecture had finished learning, so the binding constraint is budget, capacity or data — not the shape of the
learning-rate curve. The declared failure consequence stands: the next hypothesis is data or capacity.

**H2 (the rotary decoder beats the measured transformer on held-out loss): SUCCESS on the pre-declared
condition, with a confound that must be read alongside it.** 1.6865 < 1.9043, and both beat the unigram
(3.2263) and uniform (5.5661) baselines, which is exactly the condition written before the runs. But the arms are
not capacity-matched or context-matched: arm A has 3.3× the parameters and 2× the context length of arm B. They
are *token*-matched, which was the declared design, and it means the honest sentence is "the larger rotary
decoder, trained on the same number of tokens, reaches lower held-out loss than the smaller measured
transformer" — not "rotary positions, RMSNorm and a key/value cache are better". A controlled architecture
answer needs an arm at matched parameters or matched context, and that is recorded as follow-up work rather
than claimed here.

**H3 (the decoder's output still loops): FAILURE of the declared success criterion — and that is the
surprising direction, so it is reported with its text attached.** The criterion was a periodic run of ≥ 3 in the
majority of prompts; the decoder arm produced **zero**, and no arm produced a tail loop. For the first time in
this project, a trained checkpoint's samples are not exact cycles: arm A writes *the Error of the Emperor of the
Emperor of the Soule of the Soule*, which is phrase-level repetition, not a metronome. Repetition has not gone
away, though: mean bigram repetition is still 0.5538, every sample hit its 64-token bound, and no sample is a
sentence anyone would publish. The measured claim is therefore narrow: **the decoder arm's generations are less
degenerate than any checkpoint this repository has produced, and they are still not usable text.**

## Deviations from the preregistration

1. **The character baseline did not run**, for the memory reason stated in the budget section, and the report
   records the skip with its cause rather than dropping the comparison silently. E2607's gap-to-the-character-arm
   number is therefore not reproduced here.
2. **One seed per arm**, as declared with its cost reason. The 0.2178 bits gap between arms is a single pair of
   draws; E2606 measured the size of seed luck on the transformer family at ± 0.0077 bits, which is roughly
   1/28 of this gap — suggestive, not a substitute for a spread.
3. **The two arms scored different position counts** (1,000,448 vs 996,608) because the packing tail depends on
   context length, so the trivial baselines differ in the fourth decimal between the arms. Metrics are per
   position, so this is a difference in which positions were averaged, not a bias in the comparison; the
   arm-specific baselines are the ones quoted above.
4. **The pilot's report named the wrong checkpoint schema.** Fixed before either arm ran (`ceb96a0`); both arm
   reports carry their file's actual schema. The release builder now refuses a report whose family and its
   artifact's schema contradict each other (`4aa8f0e`), which is what the pilot's report still triggers when
   `chakaso model export` is pointed at it — the guard catching the historical record rather than smoothing it
   over.
5. **Timings are from a shared machine.** Arm A measured 4,821 positions/s against ADR-0037's projected 5,081
   (-5.1%), and its wall clock 100.4 minutes against a projected 65 minutes plus evaluation overhead — the
   overhead was larger than estimated because three full passes over a million held-out positions accompany the
   25 scheduled evaluations. The operator session also ran test suites during parts of both arms. No timing
   figure here is presented as intrinsic to the model or the hardware.
6. **The arms were trained by pre-wiring code.** The in-report `generation`/`latency`/`resources` blocks
   (`8244c6b`) are not in E2608's reports by design, not by omission; the protocol section states the
   substitute procedure.
7. **`--samples-per-prompt` is a token count, not a sample count.** One greedy continuation per held-out prompt,
   of `samples_per_prompt` tokens — which is what the protocol section describes. The flag's name misleads and
   should be renamed; it was not renamed here because both arms' commands are cited verbatim above.

## Conclusion

The reference model this release ships is **arm A: a 3,413,760-parameter rotary decoder, trained from random
initialization on 14,490,005 characters of public-domain prose pinned by content digest, reaching 1.6865 bits
per character on held-out documents** — the lowest held-out loss this project has measured, below its own
predecessor arm and below both trivial baselines, reproducible with the commands above and loadable through
`chakaso.models.load_model`.

It is not a useful language model. Its generations repeat phrases, every sample reached its token bound, and
nothing it writes answers anything. The honest gate level for this artifact is therefore **Level 2**: the
contract, the integration and the measurement are complete and verified; the behaviour claim is not supported.
What would change that is named in [`docs/roadmap.md`](../../docs/roadmap.md): a larger token budget with
validation that has actually turned, a seed spread on this configuration, and a capacity-matched comparison arm
so the architecture question can be answered instead of hinted at.

