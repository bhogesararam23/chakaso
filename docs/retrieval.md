# Retrieval and evidence

Retrieval is where Chakaso gets the right to make a claim about provenance. If
source identity is not established here, nothing downstream can be verified.

## Pipeline

```text
question
  -> decide whether fresh external information is needed
  -> formulate retrieval queries that preserve the user's intent
  -> fetch candidate documents under the fetch policy
  -> normalize URLs and deduplicate sources
  -> extract main content, metadata and section structure
  -> chunk, preserving source and section boundaries
  -> rank candidates
  -> assemble an Evidence Pack with immutable identifiers
  -> hand only the selected evidence to generation
```

The order matters. Deduplication before chunking prevents one page reachable at
three URLs from occupying half the evidence pack. Section boundaries survive
chunking because a chunk that straddles two sections has no defensible citation
position.

## Normalization

**Implemented.** `chakaso.retrieval.normalize_document` puts supplied text into a
deterministic form before it is chunked, so equivalent content reached a different way —
a file with CRLF endings and a paste of the same body, the same accent composed or
composed and decomposed — produces one set of chunk identifiers rather than two.

It normalizes representation only: CRLF and lone CR become LF, text is composed to
Unicode NFC, trailing spaces and tabs are stripped from each line, and runs of blank
lines collapse to one (blank lines at the edges are dropped). It never re-wraps lines,
never changes case, never removes a word, and preserves leading indentation, because
lists and code blocks mean something by their indentation. The function is idempotent
and total.

Two limits are part of the contract, not accidents: it does not understand Markdown
fences, so blank lines inside a code block collapse like any others (this matches the
dedicated chunker, ADR-0010), and its Unicode form depends on the interpreter's
normalization tables, so reproducible evaluation must pin the interpreter version.

## Local ingestion

**Implemented (local text and Markdown only).** `chakaso.retrieval.ingest_file` reads
one explicitly named local document and `ingest_text` takes one block of supplied text.
Each identifies the source (ADR-0011), reads it, normalizes it, splits it into sections
and chunks, and returns an `IngestedDocument` — a `SourceRecord` plus its chunks — which
is exactly the shape `ConversationManager.send` consumes. No parallel representation is
invented.

Ingestion reads only the single source it is handed. It never crawls a directory, never
walks a filesystem, and never reads a file because it happens to exist. The safety rules
are about what a document can be that is unusable or dangerous to load, and each is
enforced with a specific error:

| Condition | Handling |
| --- | --- |
| Path empty, absent, or not a regular file | `DocumentUnavailableError` |
| Extension not plain text or Markdown | `UnsupportedDocumentError` (so a binary, including the private `.docx` pack, is refused on format) |
| Larger than the byte limit | `DocumentTooLargeError`, checked against the file's size before its bytes are read |
| Not valid UTF-8 | `ContentDecodingError`, rather than decoded with replacement characters |
| Empty file | A valid source with zero chunks — nothing to cite, not an error |

A `..` in a path is normalized away when the reference is built, so a winding path
cannot fork one document into two sources. Only `.txt`, `.md` and `.markdown` are read
from a local path; PDF and full main-content extraction need a document processor that
does not exist yet. HTML is handled only on the web-fetch path, by the narrow
reader in `chakaso.retrieval.html` (reached via `ingest_acquired`) — never by `ingest_file`.

## Chunking

**Implemented.** `chakaso.retrieval.chunk_document` turns document text and an
already-identified source into `EvidenceChunk` records.

The rules, in order:

1. Split at Markdown headings, and keep each heading with its body. A heading's path
   is reported in full (`4. Deadlines > 4.1 Extensions`), so a citation into a
   subsection says where the subsection sits.
2. Split each section into paragraphs at blank lines.
3. Pack consecutive paragraphs into a chunk until adding the next would pass the
   target size. A paragraph that would overflow starts a new chunk.
4. A paragraph too large for one chunk is split at sentence ends; a single sentence
   too large even for that is cut.
5. Positions are assigned sequentially across the whole document, so position order
   is document order and a gap would mean text was dropped.

Chunks never cross a section boundary and do not overlap, for the reasons in
[ADR-0010](decisions/ADR-0010-section-bounded-chunks.md): a chunk spanning two
headings has no pointable citation position, and overlapping chunks would report one
sentence under several identifiers, which inflates every retrieval metric that counts
chunks.

Sizes are counted in **characters**, not tokens. There is no tokenizer, so
`token_count` on a chunk is left unset rather than filled with a character count that
a later reader would trust.

Chunking does not read files and does not fetch anything; it is handed text. The stage
that reads a file is local ingestion, described below — a source with no URL has had an
identity since [ADR-0011](decisions/ADR-0011-typed-source-references.md).

## Corpus

**Implemented (in-memory).** `chakaso.retrieval.Corpus` is the set of evidence available
to retrieval: sources and their chunks, with membership, enumeration and lookup by
identifier. It stores nothing on disk and runs no service — persistent storage is a
later decision made when there is a reason for it, and an index will be a cache, never
the source of truth.

Adding is idempotent and provenance-preserving. A chunk added under a source that does
not own it is refused (`CorpusError`), because retrieval returning such a chunk would
hand back evidence with nothing behind it. Re-adding a source or chunk that is already
present is a no-op — identifiers are content-derived, so duplicate documents simply do
not grow the corpus. Enumeration is deterministic and independent of insertion order:
sources by identifier, chunks by (source identifier, position).

### Corpus lifecycle and CLI inspection

**Implemented (in-memory, per invocation).** The corpus is rebuilt in memory for each
run from the documents a caller explicitly supplies; there is **no durable source index**
and no second store competing with the answer store. That is a deliberate boundary, not a
gap papered over: a database with no real consumer would be maintained against a guess
about what needs storing. Adding a source is idempotent, and because identity is content-
derived, re-ingesting the same bytes cannot fork a duplicate, while a content change yields
a *new* identifier rather than silently redefining an old one — so a replaced document's stale
chunks are never searchable under the old source. Removal and replacement across process
restarts are therefore not a user-facing need of the current architecture, and the CLI does
not pretend to offer them.

What the current model *does* support honestly is inspection and source-scoped retrieval,
exposed through commands that call the same `Corpus`, `ingest_file` and `RetrievalService`
apis the library exposes and add no behaviour of their own:

- `chakaso sources list --file …` — inventory the sources built from the given documents,
  their kind, canonical reference and chunk count; `--json` for a machine-readable view.
- `chakaso sources show <src_…> --file … [--text]` — inspect one source and the chunks
  ingested from it (bodies only with `--text`).
- `chakaso evidence show <chk_…> --file …` — inspect one chunk and its source.
- `chakaso retrieve … --source <src_…>` — restrict a query to named sources.

Identifiers are validated before lookup (`SourceId`/`ChunkId`) and never normalised into a
different object; an unknown-but-valid identifier is a clear error, not a silent empty result.
Text drawn from a document is untrusted: the human-readable output renders it through
`chakaso.core.safe_for_terminal` (see *Security: retrieved content is untrusted* below) so a
hostile file cannot emit terminal-control sequences, while `--json` keeps the exact bytes for a
consumer. Each command is explicit that a source appears only while its document is passed with
`--file`.

A persisted answer from the durable answer store references sources and chunks **by identifier
only**. The answer store keeps the record of what a turn decided and which evidence it named
(ADR-0028), but not the evidence text itself — that lived in the ephemeral in-memory corpus. Reading
a stored answer after a restart therefore shows the references and their provenance, and states
plainly that the cited content is not retained in, or re-readable from, the answer store. An
immutable evidence snapshot inside the answer store is deliberately **not** taken: it would duplicate
every cited body (a size and privacy cost) with no demonstrated consumer, which the no-empty-
scaffolding rule forbids.

## Lexical retrieval

**Implemented (lexical only).** `chakaso.retrieval.build_lexical_index` builds an
inverted index over a corpus's chunks, and `LexicalRetriever` scores a query against it
with Okapi BM25 and returns the best `top_k`. This is the smallest thing that turns a
corpus into ranked evidence, and it is the baseline a later dense or hybrid retriever
must beat in evaluation.

It is lexical on purpose: it matches shared words and nothing else — no embeddings, no
vector index, no model, no network. Its limitations are the ones the project expects and
wants measured rather than assumed: no synonyms, no morphology, no paraphrase, nothing
that needs meaning.

Guarantees that the rest of the system relies on:

- **Determinism.** The same corpus and query always give the same ranking. Scores are
  summed in sorted term order so floating-point accumulation cannot drift, and ties are
  broken by chunk identifier, so the order is total rather than dictionary-dependent.
- **No fabrication.** A query that matches nothing returns nothing. Retrieval never
  invents a result, a source, or a score to fill `top_k`.
- **An inspectable boundary.** `Retriever` is a protocol; replacing the lexical
  implementation with a dense one changes no caller. `source_ids` restricts candidates
  — a filter, not a re-ranking.
- **Honest explanation.** A `RetrievalResult` carries only what was computed: the chunk,
  a `score`, a `rank`, and the `matched_terms`. No confidence, no semantic justification,
  and the score is a ranking artefact, never a probability that a claim is true.

`k1` and `b` are documented defaults a caller may override, not yet configuration fields,
matching how chunking sizes are handled.

## Retrieval orchestration

**Implemented.** `chakaso.retrieval.RetrievalService` coordinates the pipeline for one
query — corpus, index, retriever, ranked chunks, evidence pack — and is deliberately
separate from the conversation manager, so retrieval can run on its own and a turn can
decide whether to retrieve at all. `search(query, top_k=...)` returns a
`RetrievalOutcome` carrying a validated `EvidencePack` and the full ranked
`RetrievalResult` list.

What it guarantees about provenance:

- **Sources come from the corpus, never from the retriever.** A chunk whose source the
  corpus does not hold is refused; the pack is built only from real records.
- **Scores are recorded, not invented.** The retriever's score is written onto a copy of
  each chunk as `retrieval_score`; the stored chunk and its identifier are unchanged.
- **An empty retrieval is an empty pack.** Nothing matching yields `pack.is_empty`, never
  fabricated evidence — the failure ADR-0003 exists to prevent.
- **The pack records its own provenance.** `retrieval_config` names the retriever and the
  `top_k`, so an answer's provenance includes the pipeline, not only the model.

The index is built once from the corpus the service is given; a corpus that grows
afterwards needs a fresh service, because an index is a cache over the corpus, never its
source of truth.

## Safe fetching

**Implemented (opt-in, bounded).** `chakaso.retrieval.HttpFetcher` turns an
`AcquisitionRequest` into an `AcquiredSource` under a `FetchPolicy`
([ADR-0012](decisions/ADR-0012-optional-bounded-fetcher.md)). It is an optional adapter
behind the `Fetcher` interface and is off every default path: no code in the library, the
CLI or CI reaches the network, which keeps the runtime requirement of
[ADR-0001](decisions/ADR-0001-local-first-runtime.md) true. The actual HTTP call is an
injectable `Transport`, so the whole policy is exercised offline against a stand-in.

What the fetcher enforces:

- **Scheme.** Only `http`/`https`; a policy may restrict to `https` but can never widen
  beyond the web schemes. `canonicalize_url` is *not* consulted for fetch permission
  (ADR-0011) — normalization and security are separate.
- **Response size.** Bounded, and the body is read up to the limit, not buffered whole.
- **Redirects.** Followed by the fetcher, not the transport, so every hop's scheme and
  destination are re-checked and a redirect into a private network is blocked; the count is
  capped and loops are broken.
- **Timeout and identity.** A positive timeout and a user-agent that cannot carry a
  newline (no header injection).
- **Destination.** Loopback, private, link-local, unique-local, multicast and the cloud
  metadata address are refused *when written as address literals*, along with known
  loopback hostnames.

**The destination screen is bounded and says so.** A hostname that resolves to a private
address is not caught by a pre-connect string check; closing that needs connecting to a
pre-resolved, re-validated IP inside the transport, which is not implemented. A deployment
that enables fetching owns the network environment. Fetching acquires bytes; a separate
stage (`ingest_acquired`, below) decodes, parses, normalizes and chunks them into the same
evidence records a local file produces.

Fetched text is data, never a directive. A line inside a page that reads like an
instruction — "ignore previous instructions", "call this URL" — is carried into an
`EvidenceChunk` verbatim and is never executed, never changes policy, and never assigns
identity: a web source is named by the validated canonical URL of the response, not by
anything the page claims about itself (ADR-0003).

## Source identity

Retrieval assigns identity; the model does not
([ADR-0003](decisions/ADR-0003-evidence-identifier-ownership.md)).

Identity is a canonical *reference*, not only a URL
([ADR-0011](decisions/ADR-0011-typed-source-references.md)). A web source is named by
its canonical URL exactly as before, a local document by a `file` URI, and content
supplied directly by a `text` reference. `derive_source_id` is indifferent to which, so
the ADR-0007 formula is unchanged and every identifier already derived from a URL does
not move.

```text
SourceRecord
    source_id            stable identifier assigned at retrieval
    kind                 web, file or text (ADR-0011)
    canonical_reference  the canonical reference: a URL, a file URI, or a text reference
    title                as extracted, not as generated
    domain               host of a web reference; empty for file and text
    retrieved_at         when this record's content was obtained
    published_at         when known; null otherwise, never guessed
    content_hash         hash of the retrieved content
    raw_text_path        where the stored text lives, if it is stored
    metadata             extraction-specific extras

EvidenceChunk
    chunk_id         stable identifier
    source_id        the source this chunk came from
    text             the chunk text, exactly as extracted
    section          section or heading path within the document
    position         ordinal position within the source
    token_count      measured with the project tokenizer where available
    retrieval_score  score from the retrieval stage
    rerank_score     score from reranking, if reranking ran
```

Properties that hold by construction:

- Records are immutable. Retrieving the same URL again produces a new record, so
  an old answer can still be explained against the evidence it actually used.
- `published_at` is null when unknown. A guessed publication date is worse than a
  missing one, because it lets a stale source look current.
- `content_hash` covers the retrieved content, so an answer can be re-examined
  against the bytes that supported it even after the page changed.
- Scores are recorded but never presented to a user as confidence. A retrieval
  score is a ranking artefact, not a probability that a claim is true.

## Evidence Pack

The Evidence Pack is the set of chunks supplied to one generation call. It is the
boundary that makes citation validation possible:

- Identifiers that were not in the pack are not resolvable, so a citation the
  model invented is caught rather than rendered.
- The pack records which retrieval and ranking configuration produced it, so an
  answer's provenance includes the pipeline, not just the model.
- Only chunks in the pack are visible to generation. Retrieval that is not used
  cannot be cited, because it was not part of the turn.

## Citation resolution

Resolving a citation is a validation gate, not a lookup:

1. Parse the identifiers referenced by the generated answer.
2. Reject any identifier that was not supplied in the evidence pack for that call,
   and record the rejection as a fabrication event.
3. For surviving identifiers, resolve `source_id` to stored metadata and produce
   the user-visible reference.
4. Where claim-level support is checked, compare the claim against the cited
   chunk and record whether it is supported, partially supported, unsupported, or
   contradicted.

Step 2 is why the rejections are recorded rather than dropped. The rate of
invented identifiers is a measurement of the system, and a system that silently
discards them cannot report it.

## Security: retrieved content is untrusted

A fetched page is adversarial input until proven otherwise. The fetch and
processing layers must handle:

| Threat | Handling |
| --- | --- |
| Prompt injection in page text | Retrieved text is placed in evidence, never in an instruction position. Instructions in a page are data, not directives. |
| Source spoofing | Identity comes from the validated canonical URL and assigned `source_id`, not from the page's own claims about itself. |
| Redirect chains | Redirects are followed within a bounded limit and the final URL is what is recorded. |
| Oversized responses | Fetch size is capped and truncation is recorded, not hidden. |
| Recursive fetching | Recursion depth is bounded by policy. |
| Unsafe or mislabelled content types | Content type is checked; HTML is not executed, and non-text content is not parsed as text. |
| Duplicated or boilerplate-heavy sources | Deduplication by canonical URL and content hash before chunking. |
| Misleading titles | Titles are recorded as extracted and are not treated as evidence of content. |
| Stale sources | `published_at` and `retrieved_at` are recorded so freshness can be reasoned about explicitly. |
| Terminal-output injection | When untrusted text reaches a human-readable terminal, control characters (the `ESC` byte, the C0/C1 blocks and `DEL`) and the bidirectional embedding/isolate controls are made inert as *visible escapes* by `chakaso.core.safe_for_terminal`. Machine-readable JSON goes through the serializer instead, which keeps the exact bytes. This is a structural display guarantee, not a claim that a model resists injection. |

The fetcher enforces the parts of this that a fetch can enforce at request time — the
scheme allowlist, the response-size cap, a bounded and re-validated redirect chain, a
timeout and a header-safe user-agent, and refusal of non-global address literals (see
*Safe fetching* above, and the documented SSRF limitation in
[ADR-0012](decisions/ADR-0012-optional-bounded-fetcher.md)). The remaining rows are
constraints on stages that are not yet wired to live servers.

No retrieved content is ever executed, and no instruction found inside retrieved
content can change application policy. This is a design constraint on the
generation step as well: the prompt template must separate instructions from
evidence textually and structurally.

## Evidence sufficiency

Retrieval ranks candidates; it does not decide whether any of them is *adequate evidence*. E2603
measured the gap: lexical retrieval returned a non-empty pack for every unanswerable question, because
BM25 hands back anything sharing even a function word with the query. `chakaso.retrieval.sufficiency`
(ADR-0032) adds that missing decision at a **separate, additive seam** — the retriever and
`RetrievalService.search` keep their exact behaviour and still return top-k; a deterministic policy reads
the returned chunks and the corpus and reports `sufficient` or `insufficient`.

The one implemented policy is **discriminating-term overlap**: a pack is sufficient when at least one
retrieved chunk shares a query term that is (a) not a closed-class function word and (b) has a low corpus
document frequency — a term whose presence actually discriminates, rather than an incidental `the`/`to`/
`library` overlap. This uses the same frequency signal BM25's IDF already encodes, is deterministic,
offline, numpy-free and strategy-agnostic (it reads chunks and corpus statistics, not a score), and names
the terms behind each decision so the outcome is inspectable.

It is **not a calibrated confidence** and not a raw score threshold (raw BM25 magnitudes are not
comparable across queries or strategies). On the small self-authored workload it separates answer from
abstention correctly only over a narrow `max_document_fraction` band — a documented, corpus-specific
limitation the project treats as the thing to stress-test, not to generalise (see E2604 and the P29
calibration work). The runtime does not yet consume the decision (wiring it into the answer path is P32);
today the signal is measured through `chakaso benchmark grounded --sufficiency` and recorded per case.

## Status

| Component | Status |
| --- | --- |
| `SourceRecord`, `EvidenceChunk` types | Implemented |
| URL canonicalization and host extraction | Implemented |
| Content normalization (representation, deterministic) | Implemented |
| Source reference identity for local and supplied content (ADR-0011) | Implemented |
| Evidence Pack with identifier validation | Implemented |
| Citation resolution, including rejection of unknown identifiers | Implemented |
| Local ingestion of explicit text and Markdown documents | Implemented |
| In-memory corpus (membership, enumeration, provenance guard) | Implemented |
| Source and evidence inspection from the CLI (in-memory, per invocation; no durable index) | Implemented (`chakaso sources list|show`, `chakaso evidence show`) |
| Source-scoped retrieval from the CLI (`retrieve --source`, identifiers validated before lookup) | Implemented |
| Durable source registry / cross-process source removal-replacement | Not justified — the corpus stays in-memory until a real consumer exists |
| Fetch policy and fetcher (opt-in, bounded; literal-address SSRF screen) | Implemented |
| HTML text extraction and web ingestion (narrow reader, not a browser) | Implemented |
| In-memory acquisition cache (opt-in, `CachingFetcher`) | Implemented |
| PDF and full main-content extraction | Planned |
| Chunking | Implemented |
| Lexical retrieval (BM25 index, deterministic ranking, explanations) | Implemented |
| Retrieval orchestration (query → ranked EvidencePack) | Implemented |
| Evidence sufficiency decision (deterministic, additive seam over retrieval) | Implemented (`chakaso.retrieval.sufficiency`, ADR-0032) — a discriminating-term policy, offline and inspectable; **not** a calibrated confidence, runtime consumption deferred to P32 (measured via `chakaso benchmark grounded --sufficiency`) |
| Dense retrieval: `EmbeddingModel` boundary, exact dense index, `DenseRetriever` behind `Retriever` | Implemented (`chakaso.retrieval.embeddings`/`dense`, ADR-0023) — the only embedding is a deterministic, **non-semantic** development double; results are tagged with their producing strategy |
| Hybrid retrieval: reciprocal-rank fusion of the lexical and dense retrievers, with per-component provenance | Implemented (`chakaso.retrieval.hybrid`, ADR-0024) — over the fixture dense embedding it re-ranks token overlap, not meaning |
| A real semantic embedding model, and an approximate (FAISS-like) vector index | Planned |
| Reranking | Planned |
| Claim-level support checking | Planned |
| Retrieval metric functions (recall@k, precision@k, MRR, duplicate + citation counts) | Implemented (`chakaso.evaluation`) |
| Retrieval metrics on a benchmark | Implemented on development benchmarks (retrieval fixtures, and the self-authored grounded-answer workload — E2603/E2604 measure recall@k, MRR, forbidden hits, false retrievals over synthetic/self-authored data); NDCG remains Planned |

The records, the pack and resolution are implemented and tested without any
network access: a record is built from content a caller already has. A lexical retriever
and an in-memory corpus now exist. A bounded, opt-in fetcher
*can* reach the web, but it is off every default path, so a default run neither fetches nor touches
the network — retrieval operates on content a caller ingests from disk or supplies.

Retrieval metrics *have* been measured — the retrieval-strategy comparison (E-001) and the
grounded-answer workload (E2603, E2604) — but only over synthetic or self-authored fixture corpora, never
a real-world corpus. Those numbers describe the retriever and the harness on hand-built data; they are
not evidence of retrieval quality on real documents (K-001). No number here is invented; every one names
the dataset and configuration that produced it.
