# Known issues

Real problems with the current state, including the ones that are only
inconvenient. An empty file here would mean nobody has looked.

Last reviewed: 2026-10-07.

## Open

### K-001 — Nothing in this repository has been executed against a real workload

**Impact:** High, and it is the honest summary of the project's position.

The architecture is a specification. It has never handled a live web page, a real
question, or a model. Decisions recorded in ADRs 0001–0004 rest on reasoning, not
on measurement, and some of them may turn out to be wrong when they meet data.
The contract in ADR-0003 in particular assumes that models can be made to reference
opaque evidence identifiers reliably; that assumption is untested.

**What would close it:** a retrieval implementation and one end-to-end turn.

**Not a defect to fix now.** It is recorded so that no document reads as though the
architecture has been validated.

### K-002 — No package, no tests and no CI exist yet

**Impact:** Medium, and temporary.

The repository has no runnable code and therefore no automated verification of
anything. `README.md` says this; the risk is that it stops being true quietly.
`docs/agent/CURRENT_STATE.md` is the file that must be updated first when that
changes.

**What would close it:** the current unit (see `ACTIVE_TASK.md`).

### K-003 — Documentation describes components that do not exist

**Impact:** Medium.

`docs/architecture.md`, `docs/retrieval.md`, `docs/correction.md` and
`docs/evaluation.md` specify behaviour for components with no code. Every affected
row carries a status label, but a reader skimming for mechanism rather than status
could come away with the wrong impression.

**Mitigation in place:** status labels are mandatory and the vocabulary is
defined in `docs/README.md`; `CURRENT_STATE.md` lists the gaps explicitly.

**Residual risk:** accepted, because writing the specification before the code is
the point of this phase. It becomes a real problem if a status label is ever left
at Experimental after the code is deleted.

### K-004 — Private planning material lives inside the working tree

**Impact:** Low but severe if it fails.

`docs/*.docx` is the author's private pack, kept in place deliberately. It is
excluded by `.gitignore` and never moved, so `git status` does not mention it,
which also means a mistake here is invisible to a casual check. A single
`git add -f` would publish it.

**Mitigation:** `tests/test_repository_hygiene.py` fails if any tracked file
matches the private-pack patterns. This is the one repository rule enforced by a
test rather than by review.

### K-005 — The `docs/` directory mixes public documents with private ones

**Impact:** Low.

`docs/` holds public markdown and, alongside it, the git-ignored private pack.
The pack cannot be moved (the project owner requires it to stay), so the
separation is enforced by ignore rules and a test rather than by directory
structure.

### K-006 — No fetch policy has been written

**Impact:** Medium, and blocking for any retrieval work.

Retrieved web content is untrusted input, and the security posture in
`docs/retrieval.md` is a list of threats with no corresponding limits: no maximum
response size, no redirect depth, no rate limit, no robots handling, no
user-agent identification. Any fetch implementation must define these first.
Tracked in `docs/research/open-questions.md`.

## Resolved

Nothing yet. Entries move here with the commit that fixed them, or they are
deleted with a note in the changelog if the issue turned out not to be real.
