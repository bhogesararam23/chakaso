"""Retrieval.

What exists here is the local path of the pipeline: normalization, chunking, ingestion
of explicitly supplied documents into evidence records, an in-memory corpus that holds
them, and a deterministic lexical index and BM25 retriever that rank them for a query. A
bounded, opt-in fetcher acquires web bytes under a policy, and a narrow reader turns HTML
or plain text into the same evidence records as a local file.
:func:`ingest_file` reads one named local file; :func:`ingest_text` takes one block of
supplied text; :func:`ingest_acquired` turns a fetched web source into evidence. All of
them normalize deterministically and split into evidence chunks with stable identifiers,
section paths and positions.

Ingestion reads only the single file a caller names. It never crawls a directory, never
walks a filesystem, and never reads a file because it happens to exist. Retrieval is
lexical: it matches shared words and nothing more — no embeddings, no vector index, no
model and no query planning. Web content is reached only through an opt-in, bounded
fetcher (ADR-0012) that is off every default path: no code here touches the network, and
CI runs fully offline.
"""

from __future__ import annotations

from chakaso.retrieval.acquire import (
    AcquiredSource,
    AcquisitionRequest,
    Fetcher,
    HttpFetcher,
    Transport,
    TransportResponse,
    make_urllib_transport,
)
from chakaso.retrieval.cache import AcquisitionCache, CachingFetcher
from chakaso.retrieval.chunking import ChunkingConfig, Section, chunk_document, split_sections
from chakaso.retrieval.corpus import Corpus
from chakaso.retrieval.errors import (
    AcquisitionError,
    BlockedDestinationError,
    ChunkingError,
    ContentDecodingError,
    CorpusError,
    DocumentTooLargeError,
    DocumentUnavailableError,
    FetchTimeoutError,
    IngestionError,
    InvalidSchemeError,
    NetworkFailureError,
    ResponseTooLargeError,
    RetrievalError,
    UnsupportedContentTypeError,
    UnsupportedDocumentError,
)
from chakaso.retrieval.html import ParsedHtml, parse_html
from chakaso.retrieval.ingest import (
    DEFAULT_MAX_BYTES,
    SUPPORTED_FORMATS,
    IngestedDocument,
    ingest_file,
    ingest_text,
)
from chakaso.retrieval.lexical import (
    DEFAULT_B,
    DEFAULT_K1,
    LexicalIndex,
    LexicalRetriever,
    RetrievalResult,
    Retriever,
    build_lexical_index,
    tokenize,
)
from chakaso.retrieval.normalize import normalize_document
from chakaso.retrieval.policy import MAX_ALLOWED_SCHEMES, FetchPolicy
from chakaso.retrieval.service import DEFAULT_TOP_K, RetrievalOutcome, RetrievalService
from chakaso.retrieval.web import ingest_acquired

__all__ = [
    "DEFAULT_B",
    "DEFAULT_K1",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_TOP_K",
    "MAX_ALLOWED_SCHEMES",
    "SUPPORTED_FORMATS",
    "AcquiredSource",
    "AcquisitionCache",
    "AcquisitionError",
    "AcquisitionRequest",
    "BlockedDestinationError",
    "CachingFetcher",
    "ChunkingConfig",
    "ChunkingError",
    "ContentDecodingError",
    "Corpus",
    "CorpusError",
    "DocumentTooLargeError",
    "DocumentUnavailableError",
    "FetchPolicy",
    "FetchTimeoutError",
    "Fetcher",
    "HttpFetcher",
    "IngestedDocument",
    "IngestionError",
    "InvalidSchemeError",
    "LexicalIndex",
    "LexicalRetriever",
    "NetworkFailureError",
    "ParsedHtml",
    "ResponseTooLargeError",
    "RetrievalError",
    "RetrievalOutcome",
    "RetrievalResult",
    "RetrievalService",
    "Retriever",
    "Section",
    "Transport",
    "TransportResponse",
    "UnsupportedContentTypeError",
    "UnsupportedDocumentError",
    "build_lexical_index",
    "chunk_document",
    "ingest_acquired",
    "ingest_file",
    "ingest_text",
    "make_urllib_transport",
    "normalize_document",
    "parse_html",
    "split_sections",
    "tokenize",
]
