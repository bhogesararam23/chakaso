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
cannot fork one document into two sources. Only `.txt`, `.md` and `.markdown` are read;
HTML, PDF and every other format need a document processor that does not exist yet.

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

No retrieved content is ever executed, and no instruction found inside retrieved
content can change application policy. This is a design constraint on the
generation step as well: the prompt template must separate instructions from
evidence textually and structurally.

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
| Fetch policy and fetcher | Planned |
| Document processing and main-content extraction (HTML, PDF) | Planned |
| Chunking | Implemented |
| Lexical retrieval (BM25 index, deterministic ranking, explanations) | Implemented |
| Dense embeddings and vector index | Planned |
| Reranking | Planned |
| Claim-level support checking | Planned |
| Retrieval metrics (Recall@k, precision@k, MRR/NDCG) | Planned |

The records, the pack and resolution are implemented and tested without any
network access: a record is built from content a caller already has. No retriever
and no fetcher exist, so nothing has been retrieved from the web.

No retrieval metric has been measured. Any number appearing in this repository's
documentation would be invented; there are none.
