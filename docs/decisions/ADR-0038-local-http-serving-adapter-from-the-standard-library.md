# ADR-0038: the local serving adapter is standard-library HTTP, bound to loopback, and serves only a trained model

**Status:** Accepted
**Date:** 2026-10-22
**Amends:** [ADR-0033](ADR-0033-model-centred-product-contract.md) (the Python API is the primary interface —
this record decides what the *secondary* interface may and may not do), and it is the interface counterpart of
[ADR-0001](ADR-0001-local-first-runtime.md) (the runtime must not require a commercial API).
**Related:** ADR-0002 (the model boundary), ADR-0034 (torch stays optional), ADR-0035/ADR-0037 (the artifacts
this endpoint serves), and the release contract of node B.

## Context

The release mission asks for an optional local HTTP server exposing `/health`, `/v1/models`,
`/v1/completions` and `/v1/chat/completions` with streaming, so that a developer can point an existing
OpenAI-shaped client at a Chakaso checkpoint. The brief suggested "a minimal, well-supported framework".
The repository had no HTTP server at all, so this is a new interface and a new set of failure modes, on a
machine with no GPU, 16.6 GB of RAM most of it already used, and one CPU pool that every generation competes
for.

Two facts made the decision non-trivial. First, the model behind the endpoint is small and its generations
loop — an interface that looks like a hosted chat API invites a caller to assume it is one. Second, the
platform this project is developed on disagrees with POSIX about what a disconnected client looks like,
which had to be measured before the abort path could be written down.

## Problem

Decide what backs the endpoint, what it binds to, what it is allowed to answer, and what it must refuse — in
a way that does not add a dependency tier to every CI matrix, does not put an unauthenticated port on a
shared machine by default, and does not let a caller mistake the development double for a trained model over
the network.

## Evidence (measured 2026-10-22, this machine, Windows 11 26200, Python 3.13.5)

- **A disconnected client does not raise one exception type.** With a paced chunked response and a client that
  closes after the first read, the server-side `sendall` raised `ConnectionAbortedError` (WinError 10053) in
  one run and `ConnectionResetError` (WinError 10054) in another; a POSIX pipe gives `BrokenPipeError`. The
  first version of the handler caught only `(BrokenPipeError, ConnectionResetError)`, so on the 10053 path the
  exception escaped to `socketserver.handle_error`, printed a traceback into the server log for an event that
  is not an error, and **left the generation running** — the abort never fired. Catching the
  `ConnectionError` family plus `BrokenPipeError` covers all three.
- **The notice is not instant and not free:** the write that failed was the third chunk after the client
  closed, ~15 ms in. A response that finishes before the close is observed cannot be aborted at all, which is
  why the abort test uses a paced, long generation rather than a four-token one.
- **`BaseServer.shutdown()` deadlocks unless the serve loop is running.** The stdlib says so in its own
  docstring, and `ServingService.stop()` inherited it: stopping a service that was constructed but never
  started blocked forever, which hung the whole test session (observed: a pytest run that produced no result
  for seven minutes and left three stranded processes). `stop()` is now guarded by the liveness of the thread
  that would be running the loop.
- **Nothing about the surface needs a framework.** Four endpoints, one model, and a request schema small
  enough to validate by hand — `http.server.ThreadingHTTPServer` plus `socketserver` is in the standard
  library, is threaded, and served every test over a real socket.

## Decision

1. **The transport is the standard library.** `chakaso.serve.server` wraps `ThreadingHTTPServer`; the request
   logic in `chakaso.serve.app` holds no socket, so the contract is testable without a port, a thread or a
   framework, in the dependency-free CI matrix. Adding a framework would add a fifth install tier to every
   matrix and a supply chain to a service whose purpose is running a model on the user's own machine.
2. **Loopback by default, and honest about it.** `127.0.0.1:8765` unless an operator names an address. There
   is no authentication, and the service reports `loopback_only` rather than implying that a non-loopback
   bind is a supported deployment mode.
3. **It serves a trained model or refuses to start.** `load_serving_model` takes a checkpoint or a release
   directory — verifying the release's checksum before reading a single weight — and raises if what loaded is
   the development double. There is no flag that makes the endpoint answer from canned text.
4. **It refuses what it cannot honour, naming the field.** `n`, `best_of`, `logprobs`, `echo`, `user` and
   `suffix` are 400s with `error.param` set; `stream: true` against a model with no `stream` is refused before
   the response starts, so a 200 never contains an error event that a client cannot distinguish from a
   generation that began and failed. A body over the ceiling is refused before it is read.
5. **Concurrency is capped and visible.** A semaphore bounds simultaneous generations — each one is CPU-bound
   on this project's hardware — and a request over the cap gets `429` with `Retry-After` rather than joining a
   queue that makes every answer slower without saying so.
6. **A client that hangs up stops the work.** The transport owns an `AbortSignal`; the write failure described
   above calls `abort()`, and the model's step loop checks it between tokens. Cancellation is proven by the
   generation producing fewer tokens than it was asked for, not by a flag being set.
7. **The response carries its own provenance.** Every completion body includes a `chakaso` block with the
   model identity, the finish reason in this project's vocabulary, and the note that the model is not
   instruction tuned — so the shape being compatible with a hosted API is not mistaken for the behaviour of one.

## Reasoning

The OpenAI shape was chosen because it is the interface existing clients already speak, which is the only
reason a local endpoint is worth having; it was not chosen because the model earns that shape. That gap
between an familiar interface and an unfamiliar artifact is exactly what the refusals and the provenance
block are for: a caller asking for `logprobs` gets an explanation, not zeros.

Keeping the request logic socket-free is what makes the honesty testable. The contract — what is accepted,
what is refused, what each refusal says — runs in 0.2 s against a scripted model with no port involved; the
transport tests then cover only what exists once bytes move: event order, the `[DONE]` marker, cancellation,
the ceiling, an oversized body, and shutdown.

## Trade-offs

- **No middleware, no routing sugar, no async.** Every request is handled on a thread and blocks a generation;
  throughput is bounded by the CPU pool, which is the same bound the model has whether or not a framework
  hides it.
- **Chunked transfer with a hand-rolled writer.** A few dozen lines that must get framing right, in exchange
  for no dependency and full control of the abort path.
- **`HTTP/1.1` keep-alive on a stdlib server** means a client that neither reads nor closes holds a thread;
  the concurrency ceiling is what bounds that, not a per-connection timeout.

## Rejected alternatives

- **FastAPI/Flask/uvicorn.** Rejected: the surface is four endpoints; the cost is a new dependency tier in CI,
  a longer cold start, and a supply chain on the one path that should be runnable on a machine with nothing
  installed but the model extra.
- **Serving the development double behind a `--allow-double` flag.** Rejected: an endpoint that returns fixed
  strings in a chat-shaped envelope is a way to fool an integration test, and the double exists precisely so
  that no default path needs torch. If a caller wants a fake, they can write one in ten lines.
- **Binding `0.0.0.0` by default for convenience.** Rejected: no authentication plus a shared machine is an
  open door to someone else's prompts and someone else's CPU.
- **Queuing requests over the cap.** Rejected: an unexplained wait is worse than a `429` that says come back.

## Consequences

- `chakaso.serve` is a new module that imports no third-party library, so it runs in the dependency-free
  matrix; its tests over a real socket are part of that matrix too, which means CI covers the transport
  without torch installed. The gated end-to-end slice (a trained artifact answered over HTTP) is covered by
  the `[nn]` matrix instead.
- The release contract of node B becomes a runtime input: `load_serving_model` prefers a release directory and
  verifies it, so a server cannot be pointed at weights that disagree with their own document.
- The abort path is now a documented contract with a test that would fail if a future change narrowed the
  exception family back.
- Documentation must say what this endpoint is not: it is not a hosted API, not instruction tuned, not
  authenticated, and not evidence that the model is useful. Those statements belong in the model card, which
  is the gate that lets this endpoint ship at all.

## Conditions to revisit this decision

- If the surface grows past a handful of endpoints, or needs validation/routing that this hand-rolled layer is
  duplicating, the framework argument has to be made again with the new cost.
- If a serving mode beyond loopback is ever genuinely wanted, it needs authentication designed in, not a
  flag removed.
- If the model gets large enough that one generation saturates the machine, the ceiling becomes a queueing
  and back-pressure question rather than a semaphore size.
