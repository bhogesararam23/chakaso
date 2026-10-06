# ADR-0001: Repository principles

- Status: accepted
- Date: 2026-10-07

## Context

Chakaso is intended to become a real language-model and grounded AI system. The project needs a repository structure that survives rapid experimentation without turning the codebase into a collection of disconnected prototypes.

## Decision

Chakaso will use incremental engineering with explicit documentation, tests, stable interfaces, and small meaningful commits.

The repository will distinguish:

- implementation
- agent-facing state
- architecture decisions
- experiments and measured results
- public technical documentation

## Why

This keeps the project understandable as it grows and makes the Git history useful as an engineering record.

## Consequences

There is some documentation overhead for meaningful changes, but future work becomes easier to inspect, reproduce, review, and extend.
