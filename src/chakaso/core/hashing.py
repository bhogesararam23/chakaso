"""Content hashing.

Hashes in Chakaso are used for two things, and the distinction matters:

* **Integrity and reproducibility.** ``content_hash`` on a source record
  identifies the exact bytes that supported an answer, so the answer can be
  re-examined after the live page changed (ADR-0003).
* **Identity.** Source and chunk identifiers are derived from hashes of their
  content (ADR-0007), which is what makes repeated retrieval deduplicate while a
  changed page produces a distinguishable record.

Both uses want a cryptographic digest. A faster non-cryptographic hash would be
adequate for identity, but not for the integrity claim, and having two hashing
functions in a provenance system invites using the wrong one.
"""

from __future__ import annotations

import hashlib

from chakaso.core.errors import ChakasoError

__all__ = [
    "SHA256_HEX_LENGTH",
    "HashingError",
    "is_sha256_hex",
    "join_parts",
    "sha256_hex",
    "short_sha256_hex",
]

#: Length of a full SHA-256 digest in hexadecimal characters.
SHA256_HEX_LENGTH = 64

#: Separator used when hashing several parts as one input. A unit separator cannot
#: appear in a URL, a hexadecimal digest or a decimal integer, so its presence
#: makes the concatenation unambiguous: hashing ("ab", "c") must not produce the
#: same digest as hashing ("a", "bc").
PART_SEPARATOR = "\x1f"


class HashingError(ChakasoError, ValueError):
    """A hashing request was malformed."""


def sha256_hex(data: str | bytes) -> str:
    """Return the full SHA-256 hexadecimal digest of ``data``.

    Text is encoded as UTF-8. Bytes are hashed as given, which is how the raw
    response body of a fetch should be hashed: decoding and re-encoding first
    would hide differences that matter.
    """
    encoded = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(encoded).hexdigest()


def short_sha256_hex(data: str | bytes, *, length: int = 16) -> str:
    """Return the first ``length`` hexadecimal characters of the SHA-256 digest.

    Truncation is used only for identifiers, where a shorter string is easier to
    read in a log and cheaper for a model to reproduce. Full digests are used
    wherever the hash is a record's integrity claim.

    Raises:
        HashingError: ``length`` is not between 1 and :data:`SHA256_HEX_LENGTH`.
    """
    if not 1 <= length <= SHA256_HEX_LENGTH:
        message = f"digest length must be between 1 and {SHA256_HEX_LENGTH}, got {length}"
        raise HashingError(message)
    return sha256_hex(data)[:length]


def join_parts(*parts: str) -> str:
    """Join parts with :data:`PART_SEPARATOR` for use as a single hash input.

    Exposed because identifier derivation is specified in terms of it, and a test
    asserts that the separator actually prevents concatenation ambiguity.
    """
    return PART_SEPARATOR.join(parts)


def is_sha256_hex(value: object) -> bool:
    """Whether ``value`` is a full lowercase SHA-256 hexadecimal digest.

    Used to validate a stored ``content_hash`` field, where a truncated or
    upper-case digest would compare unequal to a freshly computed one and make a
    record look corrupt for no real reason.
    """
    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)
