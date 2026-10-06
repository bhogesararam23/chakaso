# ADR-0001: Local-first runtime with no commercial inference or search dependency

- Status: Accepted
- Date: 2026-10-07

## Context

Chakaso intends to train its own language model and to answer questions from
retrieved public evidence. Both intentions are cheap to state and easy to
undermine: the fastest way to a convincing demo is to call a hosted model and a
hosted search API, and the fastest way to lose the project is to let that
dependency become structural.

The project has no funding, no privileged data and no compute allocation. What
it can accumulate is control over its own pipeline.

## Problem

Should the initial runtime be allowed to depend on commercial inference or
search APIs?

## Options considered

- **Hosted-first** — build on a commercial chat completion API and a commercial
  search API, and replace them later with local components. Fastest path to a
  working demo.
- **Local-first, commercial adapters later** — the runtime works with local
  components only; any external provider is an optional adapter behind the same
  interface.
- **Local-only, permanently** — refuse external providers entirely, forever.

## Decision

The initial runtime has no required commercial inference or search dependency.
Core functionality — conversation, retrieval, evidence handling, generation and
reassessment — must run with local components. Network access is permitted for
fetching public web content, which is processed locally.

External providers may be added later only as adapters behind an existing
interface, and they must remain optional: removing them cannot break the core.

## Reasoning

- The stated goal is a Chakaso-trained model. If application behaviour is tuned
  against a hosted model's outputs, that tuning is worthless once the model is
  replaced.
- A hosted dependency would make reproducible evaluation impossible without
  network access, credentials and a budget. This project needs a test suite that
  runs on a laptop with the network unplugged.
- A hosted search API would put source selection — the thing Chakaso claims to
  make transparent — behind a closed ranking system that cannot be inspected or
  versioned.
- The contract between the application and a model is the same contract whether
  the model is local or remote. Designing the interface first costs little and
  keeps the option open.

## Trade-offs

- Answer quality in the earliest phase will be visibly worse than a hosted model
  would produce. This is accepted and must not be hidden in documentation.
- More engineering is required up front: interfaces, configuration, adapter
  boundaries, local inference plumbing.
- Some experiments become slower to run, because small local models need
  iteration rather than a single prompt.

## Rejected alternatives

**Hosted-first** was rejected because it inverts the project's research
objective. The measurable claims Chakaso wants to make — about evidence support,
citation precision and correction behaviour — are claims about *its own*
pipeline. Building those on a rented model measures the wrong system, and the
resulting benchmark numbers would have to be thrown away.

**Local-only permanently** was rejected as premature absolutism. A future
comparison against a hosted model may be legitimate research, and refusing the
option outright would force an architectural rewrite to obtain it. The decision
that matters is that external providers are never *required*.

## Consequences

- Every model-shaped component must be expressed as an interface with at least
  one local implementation. See [ADR-0002](ADR-0002-language-model-boundary.md).
- Retrieval must be expressible behind a local search/fetch interface, and the
  choice of mechanism must not leak into business logic.
- CI must run without secrets and without network access to model providers.
- Documentation must be explicit about which local components are implemented
  and which are still planned, so "local-first" is not read as "already
  working".
- Heavy dependencies (inference runtimes, embedding stacks, vector indexes) are
  introduced only when a module needs them, and must stay optional so the test
  environment remains minimal.
