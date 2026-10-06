# Current state

Updated: 2026-10-07

## Repository

Chakaso is a newly initialized public repository.

The repository currently contains:

- Python packaging metadata
- a minimal importable `chakaso` package
- a minimal CLI entry point
- a smoke test
- agent instructions
- documentation structure

## What is not implemented

The following are intentionally not implemented yet:

- a language model
- tokenizer
- training pipeline
- retrieval system
- web search
- evidence pack
- conversation manager
- citation validation
- correction/reassessment engine
- production inference service
- model evaluation suite

## Current architectural position

The project is establishing interfaces and engineering discipline before building the model and runtime.

The long-term system is expected to keep the language model, retrieval, conversation state, grounding, evaluation, and storage as replaceable components.

## Truth rule

Do not describe planned components as existing components. Update this file whenever the repository's actual capabilities materially change.
