"""Retrieval.

What exists here is chunking: splitting document text into evidence chunks with
stable identifiers, section paths and positions. It does not read files and it does
not fetch anything.

There is no ranking, no embedding, no index and no query planning. Nothing in this
package has touched the network, and a document must be handed to it as text.
"""

from __future__ import annotations

from chakaso.retrieval.chunking import ChunkingConfig, Section, chunk_document, split_sections
from chakaso.retrieval.errors import ChunkingError, RetrievalError

__all__ = [
    "ChunkingConfig",
    "ChunkingError",
    "RetrievalError",
    "Section",
    "chunk_document",
    "split_sections",
]
