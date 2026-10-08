"""Evidence identifiers.

Source and chunk identity belongs to the retrieval layer, never to generated text
(ADR-0003), and it is derived from the content the identifier names rather than
assigned at random (ADR-0007). This module owns both the derivation and the
validation, so nothing else constructs an identifier string by hand.

The format is fixed: a four-character type prefix followed by sixteen hexadecimal
characters.

    src_3f2a91c4d5e6b708     a retrieved source
    chk_9c1d0e77aa4b2310     an evidence chunk

The prefix is not decoration. An identifier appears in log lines, in stored
metadata and in model output, and a validator needs to recognise a reference
without a lookup. Sixteen hexadecimal characters is 64 bits of digest, which keeps
identifiers short enough for a model to reproduce while making a collision
negligible at the scale this project operates at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Self

from chakaso.core.errors import IdentifierError
from chakaso.core.hashing import join_parts, sha256_hex, short_sha256_hex

__all__ = [
    "CHUNK_ID_PREFIX",
    "SOURCE_ID_PREFIX",
    "ChunkId",
    "Identifier",
    "SourceId",
    "derive_chunk_id",
    "derive_source_id",
]

SOURCE_ID_PREFIX = "src_"
CHUNK_ID_PREFIX = "chk_"

#: Hexadecimal characters in the digest portion of an identifier.
_DIGEST_LENGTH = 16


@dataclass(frozen=True, order=True)
class Identifier:
    """Base class for validated, prefixed identifiers.

    Frozen and ordered, so identifiers can be dictionary keys, set members and
    sorted output. Construction validates, so an ``Identifier`` instance is
    evidence that its string is well formed.
    """

    #: Type prefix, overridden by each subclass.
    PREFIX: ClassVar[str] = ""
    #: Human-readable type name, used in error messages.
    TYPE_NAME: ClassVar[str] = "identifier"

    value: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        pattern = self.pattern()
        if pattern.fullmatch(self.value) is None:
            message = (
                f"{self.value!r} is not a valid {type(self).TYPE_NAME}. "
                f"Expected {self.PREFIX!r} followed by {_DIGEST_LENGTH} lowercase "
                "hexadecimal characters, for example "
                f"{self.PREFIX}{'0' * _DIGEST_LENGTH}."
            )
            raise IdentifierError(message)

    @classmethod
    def pattern(cls) -> re.Pattern[str]:
        """Return the regular expression that a valid identifier must match."""
        return re.compile(rf"{re.escape(cls.PREFIX)}[0-9a-f]{{{_DIGEST_LENGTH}}}")

    @classmethod
    def parse(cls, value: str) -> Self:
        """Validate ``value`` and return an instance of this identifier type.

        This is the entry point for untrusted input, such as an identifier found
        in generated text, so that validation happens before any lookup.
        """
        return cls(value)

    def __str__(self) -> str:
        # Identifiers are interpolated into prompts, logs and user-facing source
        # references, where the dataclass repr would be noise.
        return self.value


@dataclass(frozen=True, order=True)
class SourceId(Identifier):
    """Identifies a retrieved source: one canonical reference with one content body."""

    PREFIX: ClassVar[str] = SOURCE_ID_PREFIX
    TYPE_NAME: ClassVar[str] = "source identifier"


@dataclass(frozen=True, order=True)
class ChunkId(Identifier):
    """Identifies one evidence chunk within one source."""

    PREFIX: ClassVar[str] = CHUNK_ID_PREFIX
    TYPE_NAME: ClassVar[str] = "chunk identifier"


def derive_source_id(canonical_reference: str, content: str | bytes) -> SourceId:
    """Derive a source identifier from a canonical reference and the source content.

    The same reference with the same content produces the same identifier, so repeated
    retrieval deduplicates. The same reference with different content produces a
    different identifier, so an earlier record is never overwritten and an earlier
    answer still points at the bytes it actually used (ADR-0007).

    The reference is the canonical string a :class:`~chakaso.evidence.identity.SourceReference`
    produces: a canonical URL for a web source, a ``file`` URI for a local document, a
    ``text`` reference for supplied content. This function is indifferent to which, so
    web identifiers are unchanged by the generalization (ADR-0011).

    Args:
        canonical_reference: The normalized reference. Its form is part of the
            identifier contract: changing the rule that produces it changes every
            future identifier derived from it.
        content: The source content. Bytes are hashed as given; text is hashed
            as UTF-8.

    Raises:
        IdentifierError: ``canonical_reference`` is empty. An identifier cannot be
            derived from nothing, and an empty reference means the source was never
            obtained rather than that it is empty.
    """
    if not canonical_reference:
        message = "cannot derive a source identifier from an empty canonical reference"
        raise IdentifierError(message)

    digest = short_sha256_hex(
        join_parts(canonical_reference, sha256_hex(content)), length=_DIGEST_LENGTH
    )
    return SourceId(f"{SOURCE_ID_PREFIX}{digest}")


def derive_chunk_id(source_id: SourceId, position: int, text: str) -> ChunkId:
    """Derive a chunk identifier from its source, its position and its text.

    Position alone would be enough to distinguish chunks within a source, but the
    text is included so that re-chunking a document produces new chunks rather than
    silently redefining what an existing identifier refers to.

    Raises:
        IdentifierError: ``position`` is negative.
    """
    if position < 0:
        message = f"chunk position must not be negative, got {position}"
        raise IdentifierError(message)

    digest = short_sha256_hex(
        join_parts(str(source_id), str(position), text), length=_DIGEST_LENGTH
    )
    return ChunkId(f"{CHUNK_ID_PREFIX}{digest}")
