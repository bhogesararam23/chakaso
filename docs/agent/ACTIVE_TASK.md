# Active task

## Current unit: P1 repository and engineering foundation

**Goal.** A repository that a new contributor or agent can clone, install, test,
lint and type-check, with no secrets, no network dependency on a model provider,
and documentation that matches the code.

**Why this is first.** Every later phase inherits these conventions. Retrieval,
training and evaluation all depend on typed configuration, stable identifiers and
a replaceable model boundary, and retrofitting those after the fact is how a
research repository becomes unreproducible.

### Done

- [x] Repository baseline: `.gitignore` (including the private-pack exclusion),
      `.gitattributes`, `.editorconfig`
- [x] Apache-2.0 license with the decision recorded (ADR-0004)
- [x] Public documentation set: README, getting started, architecture,
      transparency, retrieval, correction, training, evaluation, roadmap
- [x] Decision records ADR-0001 through ADR-0006
- [x] Research foundation: log, hypotheses, open questions, literature
- [x] Contribution guide and agent contract
- [x] Python project metadata and package skeleton (ADR-0005)
- [x] CI running format, lint, types and tests on Python 3.11, 3.12 and 3.13
- [x] Typed configuration with explicit loading and per-value provenance (ADR-0006)

### In progress

- [ ] Core primitives: identifiers, content hashing, errors
- [ ] Language-model boundary with contract tests
- [ ] Source, evidence and conversation records

### Not started, and in the order they will be taken

1. A local model adapter, runnable on CPU (P2).
2. Conversation manager behaviour on top of the state records (P2).
3. A CLI that holds a conversation (P2).
4. Local document ingestion and a lexical retrieval baseline (P3).
5. Evidence pack assembly with identifier validation (P3).
6. A web fetch mechanism under an explicit fetch policy (P4).
7. Citation resolution end to end (P4).
8. Reassessment and the correction loop (P5).
9. The evaluation harness and the first hand-built benchmark (P6).
10. Tokenizer, dataset pipeline and the tiny Transformer (P7–P8).

## Definition of done for the current unit

- A clean checkout installs with `python -m pip install -e ".[dev]"`.
- `pytest`, `ruff check`, `ruff format --check` and `mypy src` all pass locally and
  in CI.
- Configuration is typed, externalized and fails clearly on invalid input.
- The language-model boundary has contract tests and no concrete implementation is
  imported by application code.
- Source, evidence and conversation records exist with tests.
- Documentation status labels match the code, and `CURRENT_STATE.md` is accurate.

## Explicitly not in this unit

- Any retrieval implementation.
- Any model weights, tokenizer or training code.
- Any claim about answer quality.

These are excluded because the current unit is about making later work measurable,
not about producing a demo. A demo built before the measurement exists would have
to be thrown away or, worse, believed.
