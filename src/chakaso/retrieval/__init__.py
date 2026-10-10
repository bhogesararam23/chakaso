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
walks a filesystem, and never reads a file because it happens to exist. The retriever
boundary has two implementations: lexical BM25 (shared words, nothing more), and a dense
exact similarity index over an `EmbeddingModel` boundary — whose only embedding today is a
deterministic development double that is *not* semantic (ADR-0023), so a dense result is
token overlap under a different metric, not understanding. Query planning lives outside this
package (`chakaso.planning`). Web content is reached only through an opt-in, bounded
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
from chakaso.retrieval.dense import DenseIndex, DenseRetriever, build_dense_index
from chakaso.retrieval.embeddings import (
    EmbeddingMetadata,
    EmbeddingModel,
    EmbeddingVector,
    EmbedModelError,
    FixtureEmbeddingModel,
    SimilarityMetric,
)
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
from chakaso.retrieval.hybrid import FusionMethod, HybridConfig, HybridRetriever
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
from chakaso.retrieval.strategy import HybridProvenance, RetrievalStrategy
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
    "DenseIndex",
    "DenseRetriever",
    "DocumentTooLargeError",
    "DocumentUnavailableError",
    "EmbedModelError",
    "EmbeddingMetadata",
    "EmbeddingModel",
    "EmbeddingVector",
    "FetchPolicy",
    "FetchTimeoutError",
    "Fetcher",
    "FixtureEmbeddingModel",
    "FusionMethod",
    "HttpFetcher",
    "HybridConfig",
    "HybridProvenance",
    "HybridRetriever",
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
    "RetrievalStrategy",
    "Retriever",
    "Section",
    "SimilarityMetric",
    "Transport",
    "TransportResponse",
    "UnsupportedContentTypeError",
    "UnsupportedDocumentError",
    "build_dense_index",
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
