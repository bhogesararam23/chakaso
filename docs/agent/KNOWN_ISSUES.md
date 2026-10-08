# Known issues

Real problems with the current state, including the ones that are only
inconvenient. An empty file here would mean nobody has looked.

Last reviewed: 2026-10-08.

## Open

### K-001 — Nothing in this repository has been executed against a real workload

**Impact:** High, and it is the honest summary of the project's position.

The architecture is a specification. It has never handled a live web page, a real
question, or a model. Decisions recorded in ADRs 0001–0007 rest on reasoning, not
on measurement, and some of them may turn out to be wrong when they meet data.
The contract in ADR-0003 in particular assumes that models can be made to reference
opaque evidence identifiers reliably; that assumption is untested.

**What would close it:** a run against a real corpus or a real model. A local retrieval
implementation, a synthetic development benchmark and the evaluation / correction foundations now
exist and are exercised in tests; none of them is a real workload, and no model produces an
answer, so the underlying gap stands.

**Not a defect to fix now.** It is recorded so that no document reads as though the
architecture has been validated.

### K-002 — Documentation describes components that do not exist

**Impact:** Medium.

Some components specified in `docs/architecture.md` still have no code — the query planner, a
dense ranker, persistence, semantic grounding and the revised answer a model would write. The
evaluation and correction *foundations* now have code, so this risk is narrower than it was, but a
reader skimming for mechanism rather than status can still take a Planned component for a real one.

**Mitigation in place:** status labels are mandatory and the vocabulary is
defined in `docs/README.md`; `CURRENT_STATE.md` lists the gaps explicitly.

**Residual risk:** accepted, because writing the specification before the code is
the point of this phase. It becomes a real problem if a status label is ever left
at Experimental after the code is deleted.

### K-003 — Private planning material lives inside the working tree

**Impact:** Low but severe if it fails.

`docs/*.docx` is the author's private pack, kept in place deliberately. It is
excluded by `.gitignore` and never moved, so `git status` does not mention it,
which also means a mistake here is invisible to a casual check. A single
`git add -f` would publish it.

**Mitigation:** `tests/test_repository_hygiene.py` fails if any tracked file
matches the private-pack patterns. This is the one repository rule enforced by a
test rather than by review.

### K-004 — The `docs/` directory mixes public documents with private ones

**Impact:** Low.

`docs/` holds public markdown and, alongside it, the git-ignored private pack.
The pack cannot be moved (the project owner requires it to stay), so the
separation is enforced by ignore rules and a test rather than by directory
structure.

### K-008 — The fetcher's SSRF screen is partial and fetching is not polite yet

**Impact:** Medium, and bounded by the fetcher being off every default path.

The fetcher (ADR-0012) refuses non-global address *literals*, but a hostname that resolves
to a private address is not caught by a pre-connect string check; closing that needs
connecting to a pre-resolved, re-validated IP inside the transport, which is not
implemented. There is also no robots.txt handling and no per-host rate limiting, so a
caller that enables fetching must do so in an environment it controls, against hosts it is
entitled to visit.

**Why it is not more urgent:** no default code path, CLI command or CI test reaches the
network. The limitation is recorded rather than oversold, and it is the reason fetching is
opt-in.

**What would close it:** IP-pinned resolution in the transport, plus a politeness layer
(robots, per-host spacing) before the fetcher is used against the live web.

### K-006 — Nothing persists how often a model invents an evidence reference

**Impact:** Medium. It is a gap in one of the project's own headline measurements.

The conversation manager detects a reference to evidence that was not supplied,
never resolves it to a source, and reports it in `Reply.citation_resolution`
([ADR-0009](../decisions/ADR-0009-unresolved-references-are-recorded.md)). Nothing
writes that down. The reply is the only place the condition exists, so the rate at
which it happens — which `docs/evaluation.md` makes a headline metric — cannot be
computed from anything the repository keeps.

**Why a field on conversation state is not the fix.** A durable count of unresolved
references is evaluation data, not conversational data. Adding a field that nothing
reads would put the shape of an unbuilt component into a type that other components
already depend on, which is the retrofit the project has been avoiding.

**What would close it:** the evaluation layer, item 6 of the current plan.

### K-007 — An answer's model identity is not recoverable from the transcript

**Impact:** Low now, higher as soon as two models are compared.

The reply reports which model produced an answer and whether it was a development
double. The assistant turn records which evidence was supplied and which sources
were cited, but not the model. A transcript alone therefore cannot answer "which
model said this", which is the first question a comparison between two models
raises.

**Why it is deferred:** the correction *foundation* (decision, reassessment, an in-memory record)
exists, but the durable `AnswerRecord` it points to does not — that is the next unit, and
designing the persisted record before a second model compares against it is the retrofit the
project avoids.

**What would close it:** the `AnswerRecord` type and a run that uses two adapters.

## Resolved

### Retired — no package, no tests and no CI existed

Closed on 2026-10-07. The repository now has an installable typed package, a test
suite, and CI running formatting, linting, type checking and tests on Python 3.11,
3.12 and 3.13. The entry is kept rather than deleted because the risk it recorded
— that the absence of verification would stop being obvious — is the kind that
returns whenever CI is disabled for a while, and it is worth being able to point at
the last time it was true.

It originally carried the number K-002, which was reused by a live issue when this
list was renumbered. Resolved entries are now unnumbered: a closed issue does not
need to be referenced, and reusing its number for something else is a way to make
two different problems look like one.

### Retired — no fetch policy existed

Closed on 2026-10-08. A `FetchPolicy` now declares and validates the operational limits —
scheme allowlist, response-size cap, redirect bound, timeout and a header-safe user-agent —
and the fetcher enforces each, tested offline (ADR-0012). The security posture in
`docs/retrieval.md` is no longer a list of threats with no corresponding limits. What
remains — the DNS-rebinding gap in the destination screen, and politeness (robots, rate
limiting) — is carried forward honestly as K-008 rather than claimed as solved.
