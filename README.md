# Chakaso

Chakaso is an open research project exploring how to build language models and AI systems that are more transparent about what they know, what they retrieved, what they inferred, and when they should correct themselves.

The long-term goal is to build the model itself, not just put a wrapper around an existing AI API.

## What exists right now

This repository is at the foundation stage.

Right now it contains the Python project skeleton, tests, engineering rules, architecture notes, decision records, and research structure.

The model, tokenizer, retrieval system, web search, evidence pipeline, correction engine, and training stack are not implemented yet.

## What we are trying to build

The target system is a local-first conversational AI stack with separate boundaries for:

- conversation state
- task and retrieval planning
- public-web or local retrieval
- evidence and provenance
- language-model generation
- grounding and citation validation
- reassessment and correction
- evaluation

A major design rule is simple:

> Correction is more important than consistency.

If new evidence shows that an earlier answer was wrong, the system should be able to change its answer and record that change instead of defending the old answer just because it said it earlier.

## Research direction

The initial model is expected to be a real trainable decoder-only language model built from scratch in this project. Early experiments will be intentionally small so that the training and evaluation pipeline can be tested before scaling.

The surrounding runtime will be designed so a development model can later be replaced by Chakaso-trained weights.

## Transparency

Chakaso is not intended to expose hidden chain-of-thought.

Transparency here means things such as:

- what evidence supports a claim
- where a source came from
- what the system inferred
- important assumptions
- limitations
- corrections when the evidence changes

Numeric confidence will not be presented as a meaningful product feature until it is actually calibrated and evaluated.

## Documentation

Start with `docs/README.md`.

Agent and repository rules are in `AGENTS.md`.

Architecture and current project truth live under `docs/agent/`.

Important technical decisions live under `docs/decisions/`.

Research and experiment records live under `docs/research/`.

Deeper public documentation lives under `docs/public/`.

## Status

This is an active research and engineering project. The repository will change substantially as the model, training pipeline, retrieval system, and evaluation framework are implemented.

Claims in this README describe the current repository or clearly marked research direction. They are not claims that unfinished components already work.
