# ADR-0004: Apache-2.0 as the project license

- Status: Accepted
- Date: 2026-10-07

## Context

Chakaso is a public repository that intends to publish code, and eventually model
weights, tokenizers, datasets and evaluation suites. It also intends to depend on
a stack of open-source components for most of its heavy lifting.

A repository with no license is not "unlicensed and therefore free". It is
unlicensed and therefore unusable by anyone who needs to know the terms,
including the project's own future self when it wants to vendor something.

## Problem

Under what terms should Chakaso's code be published?

## Options considered

- **No license** — defer the decision until there is something worth licensing.
- **MIT** — short, permissive, universally understood.
- **Apache-2.0** — permissive, with an explicit patent grant and a
  contribution/notice structure.
- **AGPL-3.0** — copyleft with a network-use trigger, preventing closed hosted
  forks.
- **A custom research license** — restrict commercial use or require citation.

## Decision

Chakaso's source code is licensed under the Apache License, Version 2.0.

The license covers repository content: source code, documentation and
configuration. It does not automatically cover trained model weights, tokenizer
vocabularies or datasets, which are not yet published and will carry their own
terms recorded when they exist. This record does not invent those terms.

## Reasoning

- The intended dependency stack (PyTorch, Transformers, Tokenizers, Datasets,
  FAISS, sentence-transformers, pytest) is permissively licensed. Apache-2.0 is
  straightforwardly compatible with all of it, so no future component has to be
  rejected on license grounds.
- The explicit patent grant matters more here than in a typical application.
  Chakaso's value is in model architecture, training procedure and evaluation
  method — the kinds of contribution that attract patent questions — and an
  implicit-grant license leaves that ambiguous.
- Apache-2.0's requirement to state changes is useful for a research project
  where someone may fork the evaluation harness and report numbers from it. The
  notice requirement keeps provenance visible.
- Permissive licensing maximises the chance that the research is used and
  critiqued, which is the point of publishing it.

## Trade-offs

- Permissive licensing allows a closed derivative, including a hosted product
  built on Chakaso, with no obligation to contribute back. Accepted: the project
  is trying to establish a research record, not to extract rent from early code.
- Apache-2.0 is longer and more formal than MIT, and its NOTICE machinery is
  overhead for a project that currently has nothing to put in a NOTICE file.
- Because the license covers code and not weights, the artifact policy has to be
  answered separately before any model is released. That obligation is now
  visible rather than hidden.

## Rejected alternatives

**No license** was rejected because it blocks contribution and reuse by default.
Deferring the decision does not preserve optionality; it removes it.

**MIT** was rejected only on the patent-grant question. It is otherwise a fine
choice and would have been selected if Chakaso were a conventional application
rather than a model-and-method research project.

**AGPL-3.0** was rejected because it would restrict the use of the code by
researchers and small projects — the audience most likely to validate or refute
the project's claims — in order to deter a hypothetical closed competitor that
does not yet exist.

**A custom research license** was rejected as the worst option available. Drafted
restrictions on "commercial use" are ambiguous, unenforceable in practice, and
incompatible with the open-source dependencies the project relies on. A custom
license would also discourage exactly the scrutiny the project needs.

## Consequences

- Contributions are accepted under the same terms, as stated in
  [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md).
- Third-party code copied into the repository must be license-compatible, and its
  origin recorded.
- Model weights, tokenizers and datasets require an explicit artifact policy
  before release. Until then, the repository must not imply that they are
  covered.
- Documentation and benchmark results are covered by the same license, so a
  published evaluation can be reproduced and challenged freely.
