# ADR-0032: Evidence sufficiency is a deterministic, corpus-statistical decision at the retrieval boundary

- Status: Accepted
- Date: 2026-10-19
- Relates to: [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0015](ADR-0015-grounding-boundary.md), [ADR-0022](ADR-0022-query-planner-boundary.md), [ADR-0025](ADR-0025-retrieval-aware-runtime.md), [ADR-0027](ADR-0027-semantic-evaluation-contract.md), [ADR-0031](ADR-0031-grounded-answer-evaluation-workload.md)

## Context

E2603 ran the P27 grounded-answer workload through the real retrieval pipeline and found, per case,
that **every unanswerable question produced a non-empty evidence pack** (3 false retrievals of 3) and
several answerable cases surfaced passages they had marked forbidden (3 forbidden hits). The project's
whole value proposition — ground an answer in evidence, and *decline* when the evidence is not there —
depends on a system that can tell "I found the supporting evidence" from "I found k documents that
merely share words with the question." Today the retriever cannot: `LexicalRetriever.retrieve` returns
any chunk that shares at least one query term, ranked by BM25, and `RetrievalService` wraps whatever
comes back into an `EvidencePack`. There is no notion of "nothing here is actually adequate evidence."

A failure analysis of the exact E2603 cases is the starting point, not a score threshold:

- `how much does it cost to renew a dinosaur skeleton` retrieved five passages whose only overlap with
  the query was the function words `a` and `to`; none contains `dinosaur`, `skeleton` or `cost` — none
  of those terms appears in **any** document (document frequency 0).
- `why does the library enforce silence on its fourth floor` and `what is the distance between the
  library and the moon` matched only `library` (and `is`/`the`/`on`) — `library` occurs in 4 of 13
  documents, while the discriminating terms (`silence`, `fourth`, `floor`, `distance`, `moon`) are
  absent everywhere.
- By contrast every *answerable* case's evidence carries at least one query term with document
  frequency **1 of 13** — `closed`, `overdue`, `licensed`, `borrower`, `period` — a word that appears
  in essentially one document and therefore actually discriminates.

So the separating signal is not the raw BM25 score (a `library`/`the` match on a junk passage scores
comparably to a real match), nor a fixed threshold (BM25 magnitudes are not comparable across query
lengths or across the lexical/dense/hybrid strategies). The separating signal is whether the retrieved
evidence shares a **discriminating term** with the question — a term whose corpus document frequency is
low — rather than only corpus-common words.

## Problem

How should Chakaso decide, deterministically and offline and without inventing calibrated probabilities,
that retrieved evidence is or is not *sufficient* to ground an answer — and how does that decision reach
the answer path and the evaluation harness without breaking the retrieval boundary or degrading recall?

## Options considered

- **A raw BM25 score floor.** Rejected: the analysis shows a spurious match and a genuine match can score
  similarly; BM25 scores are query-length- and corpus-dependent and are explicitly *not* probabilities
  (the retriever docstring says so). A single number would be uncalibrated magic tuned to 13 documents.
- **A per-query normalised score against a model's own maximum.** Rejected for now: it needs a
  reference "ideal" score that is itself a modelling assumption, and it is not comparable across
  strategies; premature before P29's calibration work.
- **A learned or embedding-based relevance classifier.** Rejected: there is no training signal, no real
  corpus, and the runtime must stay offline and dependency-light (ADR-0001/ADR-0030). A semantic judge
  is its own gated decision (ADR-0019/ADR-0027).
- **A deterministic corpus-statistical rule: evidence is sufficient when the retrieved set shares at least
  one *discriminating* query term** — a term whose document frequency in the corpus is at or below a
  configurable fraction of the documents — rather than only common words. Chosen: it is exactly the
  frequency signal BM25's own IDF already encodes, is deterministic, inspectable, offline, strategy-agnostic
  (it reads the returned chunks and the corpus, not a score), and directly targets the measured defect.

## Decision

Add a **retrieval-boundary evidence-sufficiency policy** in `chakaso.retrieval.sufficiency`:

- `EvidenceSufficiency` is the outcome value: an `outcome ∈ {sufficient, insufficient}`, the
  discriminating terms that justified it (so the decision is *inspectable*, not a bare boolean), and the
  counts behind it. There is no third "indeterminate" state on the core path: an empty pack or an empty
  corpus is **insufficient** (nothing to ground on), and a policy that cannot read the corpus is a
  construction error, not an answer.
- `DiscriminatingTermSufficiencyPolicy` is the one implemented policy. Given a query and the retrieved
  chunks, it recomputes document frequency deterministically from the corpus's chunks (the same corpus
  retrieval searched) and reports *sufficient* iff at least one retrieved chunk contains a query term
  that is (a) **not a closed-class function word** and (b) has document frequency between 1 and
  `max_document_fraction` of the corpus (default **0.25**), excluding terms that appear in no document
  (they cannot support anything). Requiring the term to be a content word is not an optional refinement:
  implementation showed that on a 13-document corpus even `to`, `on` and `and` occur in a single
  document, so document frequency **alone** flags them as "discriminating" and every abstention query
  leaks; the function-word exclusion (a short, inspectable closed-class list) is what makes the signal
  mean what it claims. The policy is pure, deterministic, numpy-free and offline.
- `max_document_fraction` is a **configuration parameter, explicitly documented as a development
  heuristic tuned to this corpus and NOT a calibrated, transferable threshold.** A corpus of a different
  size or term distribution will need it re-derived; P29 attempts that honestly (or records that the
  workload is too small to support it). No claim of universal calibration is made anywhere.
- The policy is an **additive seam**, not a change to the retriever or `RetrievalService` contract:
  `Retriever.retrieve` and `RetrievalService.search` keep their exact behaviour (candidates in, pack out —
  they still return top-k). A separate deterministic policy decides sufficiency over the result, which is
  the two-stage shape the design brief calls for. Nothing downstream silently rewrites a weak pack into a
  confident answer: the decision is a value callers must read.
- The evaluation runner consumes it as a **new, separately-reported dimension** (evidence-sufficiency
  accuracy, false-abstention rate, answerable cases affected) so the trade-off is measurable and visible
  per case, and never merged into one score (ADR-0031's discipline).

## Reasoning

- It measures the thing that actually differs in the failures — a shared *discriminating* word versus a
  shared *stopword* — using only corpus statistics the index already has, so it cannot drift with query
  phrasing the way a score threshold does.
- It is honest by construction: it names the terms behind each decision, so a report shows *why* a query
  was judged insufficient and a reviewer can disagree with the specific evidence.
- It preserves the retrieval boundary and recall: it never drops candidates from the pack (a differently
  phrased but genuine passage still reaches the answer path); it only reports an *additional* judgement.
  Filtering or re-ranking is a separate concern and is deliberately not smuggled in here.

## Trade-offs

- `max_document_fraction` is a knob on a tiny corpus; the choice 0.25 (discriminating iff a content term
  is in ≤ ~3 of 13 documents here) fits the shipped workload but is not defensible on a real corpus
  without P29. Worse, implementation measured that on this corpus the policy separates abstention from
  answer correctly only over a **narrow band (~0.23–0.31)**: below it the answerable `book` case is
  wrongly abstained (recall loss), above it the noise term `library` (df 4) becomes "discriminating" and
  the false-premise/planet abstentions leak. A working threshold exists here only because the corpus is
  small and its vocabulary sparse; nothing guarantees one exists for another corpus. This fragility is
  recorded, not hidden, and is the central thing P29 must test rather than assert away.
- A term appearing in exactly one document is not always the "answer" term (morphology, synonyms — the
  lexical limitation that made `renewal` fail to match `renewed`). The policy does not solve recall; a
  query whose real evidence is phrased differently can still be judged insufficient. This is disclosed,
  not hidden: it is measured as the false-abstention dimension and is the honest limit of a lexical
  signal.
- Two-stage adds a concept to reason about; but it keeps retrieval's "give me candidates" job separate
  from "is there enough to answer", which is the correct separation.

## Rejected alternatives

A score floor and a normalised-score policy were rejected because the failure analysis shows raw BM25
magnitudes do not separate the real defect (they rank spurious common-word matches comparably to genuine
ones) and are not comparable across queries or strategies. A learned/embedded judge was rejected for the
offline, no-training-signal, dependency-light constraints and because semantic judging is separately
gated (ADR-0019/ADR-0027).

## Consequences

- `chakaso.retrieval` grows a `sufficiency` module (result value + a deterministic policy protocol and one
  implementation), with its own tests, including empty-corpus, empty-pack, all-common-word and
  morphology-miss edge cases.
- `GroundedEvaluationRunner` records a per-case sufficiency decision and scores evidence-sufficiency
  accuracy, false-abstention and the recall trade-off separately; the report and `chakaso benchmark
  grounded --json` expose them. The baseline (E2603) is left immutable; the post-policy run is a new
  experiment record (E2604) so a reader can see exactly what moved and what did not.
- Retrieval recall on the answerable cases must be preserved and asserted by tests; a run that lowers
  false retrievals by abstaining on answerable questions is a regression, not a success, and must be
  reported as such.
- The runtime (P32) is a separate, later integration: the answer path will read the decision and abstain
  explicitly; this record authorises the boundary and the evaluation use first, measured before wired.
