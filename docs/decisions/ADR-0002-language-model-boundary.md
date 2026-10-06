# ADR-0002: The application depends on a language-model boundary, not a model

- Status: Accepted
- Date: 2026-10-07

## Context

Chakaso's stated destination is its own scratch-trained model. Everything before
that point is a stand-in: a local open-weights model for early experiments, tiny
randomly-initialised models for pipeline validation, and eventually Chakaso
checkpoints. The application that wraps those models — conversation state,
retrieval, evidence handling, citation validation, reassessment — is expected to
outlive all of them.

The tempting shortcut is to let application code import one concrete model
implementation directly, because at the start there is exactly one.

## Problem

How should application code obtain and call a language model, given that the
concrete model is expected to change repeatedly?

## Options considered

- **Direct import** — application modules import a specific local model wrapper
  and call it. Simplest possible code, and there is only one model today.
- **Abstract base class** — a `LanguageModel` ABC with concrete subclasses, and
  application code typed against the base class.
- **Narrow protocol/interface plus explicit capability metadata** — the boundary
  is a small structural interface; implementations declare what they support,
  and unsupported requests fail loudly at the boundary rather than deep inside a
  call stack.
- **Full model-serving layer** — a separate inference service behind HTTP or
  gRPC, with the application as a client.

## Decision

Application code depends on a narrow language-model interface and never imports
a concrete implementation from business logic. Concretes are selected
explicitly, at a single composition point, from configuration.

The boundary covers only what the application actually needs:

- generation from a message list, with explicit generation parameters,
- structured generation where the implementation can support it,
- tokenization,
- metadata describing the model and its capabilities.

Implementations declare their capabilities, and a caller requesting an
unsupported capability gets a clear error naming the model and the capability.
The interface may not grow to accommodate a feature of one specific model.

## Reasoning

- Chakaso's evaluation claims are system claims. If the application cannot swap
  the model without edits outside the composition point, those claims cannot be
  compared across models, which is the whole point of training one.
- A narrow interface is also a test seam: a deterministic in-process
  implementation makes conversation, retrieval and citation logic testable
  without downloading weights or needing a GPU.
- Capability metadata avoids the common failure mode where an interface pretends
  every model supports every feature, and the pretence is only discovered at
  runtime by a confusing downstream error.
- It keeps the local-first commitment honest. A model boundary is what makes
  "local now, Chakaso later, remote optionally" a configuration change instead
  of a rewrite.

## Trade-offs

- Slightly more code than a direct import, and one extra indirection when
  reading a call stack.
- A narrow interface will occasionally not expose something a specific model
  does well. The correct response is to decide whether the application genuinely
  needs it, not to widen the interface by default.
- Capability negotiation adds a failure mode (asking for something unsupported)
  that a permissive duck-typed call would have hidden.

## Rejected alternatives

**Direct import** was rejected because replacing the model is an explicit
project milestone (M2/M3 in the roadmap). A boundary introduced at that point
would be a rewrite of every call site under time pressure, which is exactly when
boundaries get drawn badly.

**Abstract base class** was rejected as the primary mechanism. Inheritance
forces implementations into a class hierarchy they may not want, and it does not
express capabilities. A structural interface is easier to satisfy and easier to
fake in tests. (A shared base may still be added later as a convenience for
implementations, without becoming the application-facing contract.)

**Full model-serving layer** was rejected as premature. A network hop between
the application and its own model adds serialization, lifecycle and failure
handling before there is any need to run inference in a separate process. It can
be introduced later *behind the same interface*, which is precisely the point of
having the interface.

## Consequences

- There is exactly one place in the codebase that maps configuration to a
  concrete model implementation. Adding a model means adding an adapter and a
  registry entry, not editing application logic.
- Tests must include contract tests that every implementation of the boundary
  must satisfy, plus a deterministic implementation used by higher-level tests.
- Generation parameters that are model-specific cannot be smuggled through the
  interface; if the application needs a parameter, it belongs in the interface
  with defined behaviour for implementations that ignore it.
- Tokenization is part of the boundary, so token counting and context-window
  budgeting do not require reaching into model internals.
- The boundary must stay transport-agnostic: whether a future implementation is
  in-process, an ONNX runtime or a remote service is an implementation detail.
