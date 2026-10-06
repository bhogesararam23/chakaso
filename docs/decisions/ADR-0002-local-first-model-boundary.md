# ADR-0002: Keep the model boundary provider-independent

- Status: accepted
- Date: 2026-10-07

## Context

The initial development environment should work without depending on commercial inference APIs. The eventual goal is to replace the initial development model with a Chakaso-trained model without rewriting the surrounding system.

## Decision

The application will communicate with language models through an internal model abstraction rather than calling a specific provider directly throughout the codebase.

The concrete implementation can change between:

- local/open-source development models
- experimental checkpoints
- Chakaso-trained weights

without changing the surrounding conversation, retrieval, or evaluation contracts.

## Why

A stable model boundary reduces provider lock-in and lets infrastructure evolve independently from model research.

## Consequences

The model adapter must remain deliberately small. Provider-specific behavior belongs behind the adapter rather than leaking into application logic.
