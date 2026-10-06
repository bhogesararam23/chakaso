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

## Source identity

Retrieval assigns identity; the model does not
([ADR-0003](decisions/ADR-0003-evidence-identifier-ownership.md)).

```text
SourceRecord
    source_id        stable identifier assigned at retrieval
    canonical_url    normalized, after redirects
    title            as extracted, not as generated
    domain           registrable domain of canonical_url
    retrieved_at     when this record's content was obtained
    published_at     when known; null otherwise, never guessed
    content_hash     hash of the retrieved content
    raw_text_path    where the stored text lives, if it is stored
    metadata         extraction-specific extras

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
| URL normalization and deduplication helpers | Implemented |
| Evidence Pack type and identifier validation | Partly implemented |
| Fetch policy and fetcher | Planned |
| Document processing and main-content extraction | Planned |
| Chunking | Planned |
| Local lexical ranking baseline | Planned |
| Dense embeddings and vector index | Planned |
| Reranking | Planned |
| Claim-level support checking | Planned |
| Retrieval metrics (Recall@k, precision@k, MRR/NDCG) | Planned |

No retrieval has been run against the live web, and no retrieval metric has been
measured. Any number appearing in this repository's documentation would be
invented; there are none.
