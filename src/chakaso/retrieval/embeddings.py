"""The embedding-model boundary and a deterministic development double.

Dense retrieval needs vectors. This module defines *the interface a real embedding model
will implement* — an `EmbeddingModel` protocol with `metadata` and a batch `embed` — so
that dense indexing and retrieval can be built and tested now, and a genuine model dropped
in later behind the same boundary (the same discipline ADR-0002 applies to the language
model, and ADR-0021 applies to storage).

What is implemented today is **not** semantic. `FixtureEmbeddingModel` projects a chunk's
bag-of-tokens into a fixed-length vector by stable hashing. It is deterministic, offline and
dependency-free, which makes it a sound *development double* — but two texts that share no
surface token get orthogonal vectors, so its cosine similarity measures token overlap, not
meaning. It is declared `is_semantic=False` and `development_double=True`, and nothing here
must be allowed to imply that a hash vector is an embedding that understands language. A
real embedding model is what would close that gap; naming the boundary now is what makes it
a replacement rather than a rewrite.

Vectors are `EmbeddingVector` — a fixed-length tuple of finite floats — so a non-finite or
ragged vector is refused at construction rather than silently corrupting a ranking.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from chakaso.core.errors import ChakasoError
from chakaso.retrieval.lexical import tokenize

__all__ = [
    "EmbedModelError",
    "EmbeddingMetadata",
    "EmbeddingModel",
    "EmbeddingVector",
    "FixtureEmbeddingModel",
    "SimilarityMetric",
]


class EmbedModelError(ChakasoError, ValueError):
    """An embedding is unusable: wrong dimension, non-finite values, or a bad request."""


class SimilarityMetric(StrEnum):
    """How two vectors are compared. Both are deterministic; cosine is length-invariant."""

    COSINE = "cosine"
    DOT = "dot"


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """A fixed-length, finite embedding for one piece of text."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            message = "an embedding vector must have at least one dimension"
            raise EmbedModelError(message)
        for value in self.values:
            if not math.isfinite(value):
                message = f"embedding values must be finite, got {value!r}"
                raise EmbedModelError(message)

    @property
    def dimension(self) -> int:
        """How many components the vector has."""
        return len(self.values)


@dataclass(frozen=True, slots=True)
class EmbeddingMetadata:
    """What an embedding model can be trusted to have declared about itself.

    Attributes:
        model_id: Stable identity of the embedding implementation and version.
        dimension: The length every vector this model produces shares.
        normalized: Whether the model guarantees unit-length vectors.
        development_double: Whether this is a stand-in rather than a real model.
        is_semantic: Whether the vectors encode meaning. A hash/lexical projection is not
            semantic, and must set this False; it exists so no downstream claim can quietly
            treat a fixture as understanding.
    """

    model_id: str
    dimension: int
    normalized: bool
    development_double: bool
    is_semantic: bool

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            message = "embedding metadata must name a model_id"
            raise EmbedModelError(message)
        if self.dimension < 1:
            message = f"embedding dimension must be at least 1, got {self.dimension}"
            raise EmbedModelError(message)


@runtime_checkable
class EmbeddingModel(Protocol):
    """The replaceable boundary dense retrieval embeds through."""

    @property
    def metadata(self) -> EmbeddingMetadata:
        """The model's declared identity and vector properties."""
        ...

    def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        """Embed ``texts`` in order, one vector each, all of ``metadata.dimension``.

        Batch input, positional output: ``embed(texts)[i]`` corresponds to ``texts[i]``.
        The empty string is a valid input and yields a vector; it is never an error.
        """
        ...


class FixtureEmbeddingModel:
    """A deterministic, offline `EmbeddingModel` — a development double, not semantic.

    It hashes a text's tokens into a fixed-length bag-of-words vector and L2-normalizes it.
    That is enough to exercise the dense boundary end to end and to reproduce exactly, but
    similarity between two vectors is token overlap under the hood: no synonymy, no
    paraphrase, no meaning. Callers and reports must describe it as a fixture / development
    double, never as a semantic embedding (ADR-0023).
    """

    def __init__(self, *, dimension: int = 64, seed: str = "chakaso-fixture-embedding") -> None:
        if dimension < 1:
            message = f"dimension must be at least 1, got {dimension}"
            raise EmbedModelError(message)
        if not seed:
            message = "a fixture embedding model needs a non-empty seed"
            raise EmbedModelError(message)
        self._dimension = dimension
        self._seed = seed

    @property
    def metadata(self) -> EmbeddingMetadata:
        """Declare the double honestly: fixed dimension, normalized, not semantic."""
        return EmbeddingMetadata(
            model_id=f"fixture-hash-embedding-{self._dimension}",
            dimension=self._dimension,
            normalized=True,
            development_double=True,
            is_semantic=False,
        )

    def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        """Embed each text in ``texts`` as a normalized hashed bag-of-tokens vector."""
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> EmbeddingVector:
        vector = [0.0] * self._dimension
        for token in tokenize(text):
            vector[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An empty (or token-free) text has no direction; keep it a zero vector rather
            # than dividing by zero — the dense retriever treats a zero-norm query as a
            # non-match instead of inventing a ranking.
            return EmbeddingVector(tuple(vector))
        return EmbeddingVector(tuple(value / norm for value in vector))

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(
            f"{self._seed}\x00{token}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big") % self._dimension
