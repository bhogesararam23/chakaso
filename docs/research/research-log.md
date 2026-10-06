# Research log

## 2026-10-07 — Repository foundation

### Observation

The project needs a software foundation before model training and retrieval experiments.

### Decision

Start with packaging, tests, agent context, architectural boundaries, and explicit decisions.

### Result

The repository is now bootstrapped as a small Python project. No model or retrieval functionality is claimed as implemented yet.

### Next question

What is the smallest useful set of typed internal contracts that lets the conversation, model, retrieval, evidence, and evaluation components evolve independently?
