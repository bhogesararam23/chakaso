"""Retrieval.

What exists here is the local path of the pipeline: normalization, chunking, ingestion
of explicitly supplied documents into evidence records, and an in-memory corpus that
holds them for retrieval.
:func:`ingest_file` reads one named local file; :func:`ingest_text` takes one block of
supplied text. Both normalize deterministically and split into evidence chunks with
stable identifiers, section paths and positions.

Ingestion reads only the single file a caller names. It never crawls a directory, never
walks a filesystem, and never reads a file because it happens to exist. There is no
ranking, no embedding, no index and no query planning. Nothing in this package has
touched the network, and web content is not fetched or parsed.
"""

from __future__ import annotations

from chakaso.retrieval.chunking import ChunkingConfig, Section, chunk_document, split_sections
from chakaso.retrieval.corpus import Corpus
from chakaso.retrieval.errors import (
    ChunkingError,
    ContentDecodingError,
    CorpusError,
    DocumentTooLargeError,
    DocumentUnavailableError,
    IngestionError,
    RetrievalError,
    UnsupportedDocumentError,
)
from chakaso.retrieval.ingest import (
    DEFAULT_MAX_BYTES,
    SUPPORTED_FORMATS,
    IngestedDocument,
    ingest_file,
    ingest_text,
)
from chakaso.retrieval.normalize import normalize_document

__all__ = [
    "DEFAULT_MAX_BYTES",
    "SUPPORTED_FORMATS",
    "ChunkingConfig",
    "ChunkingError",
    "ContentDecodingError",
    "Corpus",
    "CorpusError",
    "DocumentTooLargeError",
    "DocumentUnavailableError",
    "IngestedDocument",
    "IngestionError",
    "RetrievalError",
    "Section",
    "UnsupportedDocumentError",
    "chunk_document",
    "ingest_file",
    "ingest_text",
    "normalize_document",
    "split_sections",
]
