# Changelog

Notable changes to Chakaso, newest first. Nothing has been released; every entry
below is pre-release.

Entries describe what changed and why it matters to someone using or evaluating
the project. They do not list edited files, and they do not describe progress
without a result. Where an entry records a capability, it states whether that
capability is implemented or still planned.

## Unreleased

### 2026-10-07 — Project foundation

**Added — repository**

- Apache-2.0 license for code, documentation and configuration. Trained weights,
  tokenizers and datasets are explicitly not covered; they will carry their own
  terms when they exist.
- `.gitignore` that excludes the author's private planning pack (`docs/*.docx`)
  along with virtual environments, tool caches, model weights, dataset shards and
  local experiment outputs.
- `.gitattributes` normalizing text to LF in the repository and marking binary
  formats, so line-ending conversion cannot corrupt weights or documents.
- `.editorconfig` for consistent indentation and line endings across editors.

**Added — decision records**

- ADR-0001: the initial runtime has no required commercial inference or search
  dependency. Network access is allowed for fetching public web content, which is
  processed locally. External providers may exist later only as optional adapters.
- ADR-0002: application code depends on a narrow language-model interface, not a
  concrete model. Implementations declare capabilities; unsupported requests fail
  at the boundary.
- ADR-0003: the retrieval layer owns source and chunk identifiers. Generated text
  may only reference identifiers it was given; an unknown identifier is recorded
  as an error and never resolved to a URL.
- ADR-0004: Apache-2.0, chosen for compatibility with the intended dependency stack
  and for its explicit patent grant.

**Added — public documentation**

- `README.md` stating what Chakaso is, what is implemented, and what is not.
- Architecture, transparency, retrieval, correction, training, evaluation and
  roadmap documents. These are specifications: they describe intended behaviour and
  label the status of every component.
- Research foundation: a research log, six falsifiable hypotheses, an open-questions
  list, and a literature file whose URLs were each fetched and verified before
  publication.
- Contribution guide.

**Added — internal and agent documentation**

- Agent contract (`AGENTS.md`) and agent context: current state, conventions, active
  task and known issues.
- Internal engineering notes: repository history, artifact policy and tooling
  rationale.
- Experiment record format.

**Note on scope**

There is no runnable system, no retrieval, no correction loop, no tokenizer and no
model. No benchmark has been run and no result is reported anywhere in this
repository.

**Note on history**

The repository's previous history, whose head tree was already empty, was replaced
with a fresh root commit. The previous head is preserved as tag
`archive/bootstrap-attempt-1`. See `docs/internal/engineering/repository-history.md`.
