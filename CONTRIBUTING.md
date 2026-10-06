# Contributing

Chakaso is built incrementally.

Before opening a change, read `AGENTS.md` and the relevant documentation under `docs/`.

## What a good change looks like

A change should do one coherent thing, include tests where behavior changes, and update documentation when the architecture or public behavior changes.

Avoid:

- cosmetic commits
- placeholder implementations presented as complete
- unrelated refactors mixed into feature work
- secrets, credentials, datasets, or model weights in Git

## Commit messages

Use small, meaningful conventional-style commits such as:

```
feat: add evidence contract
fix: reject invalid source ids
test: cover contradictory evidence
docs: explain reassessment flow
```

The commit history is part of the project's engineering record.
