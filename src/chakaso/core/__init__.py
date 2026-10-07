"""Shared primitives: identifiers, content hashing and the error base.

Modules live here when they are used by more than one component and belong to no
single one. Everything in :mod:`chakaso.core` is standard-library only.
"""

from __future__ import annotations

from chakaso.core.errors import ChakasoError, IdentifierError
from chakaso.core.hashing import SHA256_HEX_LENGTH, sha256_hex, short_sha256_hex
from chakaso.core.identifiers import (
    CHUNK_ID_PREFIX,
    SOURCE_ID_PREFIX,
    ChunkId,
    Identifier,
    SourceId,
    derive_chunk_id,
    derive_source_id,
)

__all__ = [
    "CHUNK_ID_PREFIX",
    "SHA256_HEX_LENGTH",
    "SOURCE_ID_PREFIX",
    "ChakasoError",
    "ChunkId",
    "Identifier",
    "IdentifierError",
    "SourceId",
    "derive_chunk_id",
    "derive_source_id",
    "sha256_hex",
    "short_sha256_hex",
]
