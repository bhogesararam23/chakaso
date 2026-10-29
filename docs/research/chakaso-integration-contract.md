# Chakaso integration contract: what a consumer may rely on

**Status:** Implemented and tested for the surfaces named below, with the gaps in [Known gaps](#known-gaps)
stated rather than papered over. This document defines the contract a future software consumer can depend on;
it does not connect any external repository.
**Date:** 2026-10-29.
**Primary interface:** the Python library ([ADR-0033](../decisions/ADR-0033-model-centred-product-contract.md)).
The CLI and the local HTTP adapter are secondary developer surfaces over the same library.

## What a consumer loads

A consumer has two legal entry points, and neither requires the conversation runtime, the training pipeline or
a terminal:

```python
from chakaso.models import GenerationParams, Message, Role, load_model

model = load_model("path/to/release-directory")  # or a checkpoint file
result = model.generate(
    [Message(role=Role.USER, content="Call me Ishmael.")],
    GenerationParams(max_new_tokens=32, temperature=0.0),
)
```

- **A release directory** is a `chakaso-model-release/1` document (`model.json`) plus the artifact it names.
  `load_model` verifies the artifact's SHA-256 against the contract **before a weight is read**; a corrupt
  directory raises `CorruptedReleaseError`, an incompatible schema raises `IncompatibleReleaseError`, and a
  missing extra raises a `ModelError` naming it. The release carries identity, capabilities, licence,
  limitations and the artifact digest.
- **A checkpoint file** is a self-describing, checksummed `chakaso-transformer-checkpoint/1` or
  `chakaso-decoder-checkpoint/1` file with the tokenizer inline. `load_model` selects the adapter from the
  file's declared schema, verifies the checksum, and refuses a tokenizer, architecture or tensor shape that
  does not match what the file claims.
- **The development double** is never loadable from a release path and is refused by the serving adapter. A
  consumer that needs a trained model must not treat the double as one; `model.metadata.development_double`
  is the flag to check.

## The model boundary

| Member | Contract |
| --- | --- |
| `model.metadata` | `ModelMetadata`: `model_id`, `display_name`, `context_window`, `capabilities`, `development_double`. `model_id` is the adapter/registry identity, not necessarily the release's contract identity (K-015). |
| `model.describe()` | A mapping that reports the runtime identity, family, context, tokenizer fingerprint, prompt-template version and capabilities. |
| `model.generate(messages, params)` | Returns `GenerationResult`: `text`, `token_ids`, `new_tokens`, `prompt_tokens`, `truncated`, `stopped_by`, `finish_reason`, `model_id`. Messages are `Message(role=Role.USER/ASSISTANT/SYSTEM, content=...)`. |
| `model.stream(messages, params)` | Yields chunks as tokens are produced; the concatenated text equals `generate(...).text` for the same request; a caller's `AbortSignal` stops the loop at a step boundary and reports `FinishReason.CANCELLED`. |
| `model.score(text)` | Mean next-token cross-entropy in nats for a text the model can window; an adapter that cannot measure it raises a named `ModelError` rather than returning a guess. |
| `model.sample(prompt, max_new_tokens=...)` | A convenience for a bare-string prompt; it is the same boundary request. |

### Generation parameters

`GenerationParams` is the whole bounded-decoding surface, validated in one place so a request and a
configuration cannot disagree about what a number means: `max_new_tokens`, `temperature` (0 means greedy),
`top_k`, `top_p`, `seed`, `stop_sequences`, `stop_token_ids`, `repeat_penalty`. A caller that needs
determinism sets `temperature=0`; a caller that samples sets a seed. `GenerationResult.finish_reason` is one
of `end_of_text`, `length`, `context` or `cancelled`; `truncated` says whether the prompt was shortened to the
context window. `stop_sequences` are matched on decoded text; a stop token id is matched on the token
stream.

### Errors

All model failures are typed and named: `ModelError` (base), `CorruptedReleaseError`, `IncompatibleReleaseError`,
`UnsupportedCapabilityError`, `UnknownModelError`. A consumer should catch `ModelError` at the boundary and
inspect the type; no path silently falls back to the development double.

## Versioning and identity

| Object | Version / identity |
| --- | --- |
| Release document | schema `chakaso-model-release/1`; its own SHA-256 is the contract checksum |
| Checkpoint | schema `chakaso-transformer-checkpoint/1` or `chakaso-decoder-checkpoint/1`; file checksum; tokenizer fingerprint and vocabulary inside |
| Tokenizer | self-describing BPE record with a fingerprint; the checkpoint's forward pass and the adapter's decode use the same vocabulary |
| Prompt template | `chakaso-plain-v1` (the measured default) or `chakaso-role-labelled-v1` (opt-in, not an instruction-following claim) |
| Package | `chakaso 0.1.0.dev0`; the artifact version `0.1.0` is a different series (K-019) |

A consumer must pin the artifact digest, not a path. A path can be replaced; a digest cannot. `chakaso model
verify <release>` re-hashes the artifact against the contract. The release document does not yet state the
code revision that trained it (K-019); the experiment record does, one document away.

## Streaming, cancellation and resource bounds

- Generation is a step loop, not a batch that is collected first. `stream()` yields as it produces; a
  cancelled stream stops the work rather than discarding already-computed tokens.
- `max_new_tokens` and the model's `context_window` are hard bounds. An over-long prompt is truncated (and the
  result says so) rather than silently dropped; an over-long request is refused at the boundary.
- The local HTTP adapter (`chakaso.model serve`) bounds concurrent generations with a semaphore and returns
  `429` with `Retry-After` rather than queueing without limit; it reads an oversized body only up to its bound
  and refuses it; a client hang-up cancels the generation through the boundary's own `AbortSignal`.
- The server binds `127.0.0.1` by default. Binding elsewhere is an operator's explicit choice. There is no
  authentication and none is implied; do not expose it to a network you do not control.

## Local HTTP shape

| Endpoint | Method | Contract |
| --- | --- | --- |
| `/health` | GET | process and model identity |
| `/v1/models` | GET | the loaded release identity and capabilities |
| `/v1/completions` | POST | OpenAI-shaped prompt completion; returns text and finish reason |
| `/v1/chat/completions` | POST | OpenAI-shaped messages; returns the model's text |

The adapter refuses every field of the OpenAI shape it cannot honour (`n`, `best_of`, `logprobs`, `echo`,
`user`, `suffix`, an out-of-range temperature, `stream: true` against a non-streaming model) with a message
naming the field. It refuses the development double outright. It is a local adapter, not a compatibility
promise for every OpenAI feature.

## Integration checklist for a future consumer

1. Pin the release directory by its artifact SHA-256 and contract checksum; do not load a mutable path.
2. Run `chakaso model verify <release>` (or `load_model`, which verifies) before the first request.
3. Read `model.metadata`: `development_double` must be `False`; the capabilities must include what the
   integration needs; `context_window` must fit the prompts.
4. Read the model card and the release's limitation list. Level 2 of ADR-0039 is integration, not usefulness;
   Level 3 is not met for `chakaso-base@0.1.0`.
5. Set `max_new_tokens`, temperature/top-p/top-k and a seed explicitly; do not rely on defaults for a
   reproducible result.
6. Handle `ModelError` and its subtypes; treat a refusal as a named outcome, not a crash.
7. Test streaming and cancellation on the integration's own event loop; do not assume `stream()` is
   non-blocking in a framework the adapter has not seen.
8. Record `model_id`, the release identity and the prompt-template version in the consumer's own logs, so a
   change of model is visible in its history.
9. Do not import `chakaso.training`, `chakaso.app` or `chakaso.runtime` to load a model; the public path is
   `chakaso.models`. (A fresh-interpreter probe in `tests/test_vertical_slice.py` checks that loading a
   checkpoint does not import the conversation runtime.)
10. If the consumer needs a capability the model does not declare, fail closed; do not prompt around it.

## What is tested

- `tests/test_models_release.py`: a directory that will not load without verifying its artifact, a builder
  that refuses a report whose family contradicts its checkpoint schema, a required non-empty limitation list,
  `export_release` re-digesting its copies, and typed refusals separating corrupt from incompatible documents.
- `tests/test_models_load.py` and the adapter contract suite: metadata, capabilities, prompt handling,
  generation bounds, stop semantics, seeded sampling, streaming equals generation, cancellation, scoring,
  and refusal of empty/missing/malformed checkpoints.
- `tests/test_serve_app.py` and `tests/test_serve_server.py` (49 tests): request logic without a socket,
  the four loader refusals, event order, `[DONE]`, cancellation observed as tokens stopping, the concurrency
  ceiling, 413 before read, and `stop()` on a service that never started.
- `tests/test_vertical_slice.py`: a decoder trained inside the test, then load → bounded generation → seeded
  sample → streamed cancellation → score → release → verify → serve, plus a fresh-interpreter probe and the
  same lifecycle through the CLI handlers.

## Known gaps

| Gap | Effect on a consumer |
| --- | --- |
| K-015: a loaded release reports the adapter's `model_id`, not the contract's `chakaso-base` | The primary Python interface has no single field that answers "what release is this?"; `/v1/models` and completion bodies use the contract identity, `/health` prints both. Reconcile the two or read the contract separately. |
| K-016: `chakaso model serve` is not run as a command by a test or CI | The transport and request logic are tested over a socket; the CLI action's flag/default wiring is not. |
| K-018: `--samples-per-prompt` counts tokens | A record or a consumer that treats it as a sample count is wrong. |
| K-019: the release does not name the code revision that trained it | A release directory that travels alone cannot tell a consumer which code trained the weights. |
| No authentication, no rate limiting beyond concurrency | Do not expose the server beyond loopback without adding a proxy. |
| No multi-model serving | One release per process; the registry selects an adapter, not a model zoo. |
| No instruction-following checkpoint | The chat endpoint answers with a base model's continuation; it is not an assistant. |

These are not integration blockers for a research consumer that reads the card. They are blockers for
presenting the release as a product API, which ADR-0039's Level 3 deliberately withholds.
