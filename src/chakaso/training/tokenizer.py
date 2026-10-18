"""A deterministic character-level tokenizer.

The first experiment is character-level on purpose: it needs no external model or trained vocabulary,
its behaviour is fully inspectable, and every token is a real character a person can read. This module
holds the *contract* — a vocabulary built from training text only, an encode/decode pair, and a
stable fingerprint — without any dependency on a numerical library. That separation is deliberate: the
tokenizer decides how text becomes integer identifiers, and that decision must be reviewable and
testable independently of the model that consumes the integers.

Two properties are load-bearing and are enforced here rather than assumed:

* **Built on training data only.** A vocabulary must be fit before any held-out example is seen, so
  that no validation character can influence the identifier space the model is trained in. The
  constructor takes training text and nothing else; the split layer (``chakaso.training.data``) is
  responsible for handing it only the training portion.
* **Deterministic.** The same set of training characters always produces the same ``char → id`` map and
  the same fingerprint, because identifiers are assigned by codepoint order, not insertion order.

Unknown characters — anything not in the training vocabulary — are mapped to a reserved unknown id
rather than dropped or guessed, so encoding is total and reversible up to the lossy fact that several
unknown characters share one id. This is honest about its own limit rather than pretending a clean
round-trip it does not have.
"""

from __future__ import annotations

import hashlib

__all__ = ["UNK_ID", "CharVocabulary"]

#: The reserved identifier for a character that is not in the vocabulary. Id 0 is never assigned to a
#: real character, so a model's embedding row 0 is unambiguously the unknown token.
UNK_ID = 0


class CharVocabulary:
    """A fixed, deterministic character vocabulary fit from training text.

    Instances are immutable. Identifiers start at ``1``; ``0`` is reserved for :data:`UNK_ID`.
    """

    __slots__ = ("_char_to_id", "_chars", "_fingerprint")

    def __init__(self, training_texts: tuple[str, ...]) -> None:
        if not training_texts:
            message = "a vocabulary needs at least one training text; none was supplied"
            raise ValueError(message)

        distinct = {char for text in training_texts for char in text}
        if not distinct:
            message = "the training texts contain no characters; a vocabulary cannot be empty"
            raise ValueError(message)

        # Sorted by codepoint so identifier assignment is independent of iteration/insertion order.
        self._chars: tuple[str, ...] = tuple(sorted(distinct))
        self._char_to_id: dict[str, int] = {
            char: index + 1 for index, char in enumerate(self._chars)
        }
        self._fingerprint = _compute_fingerprint(self._chars)

    @property
    def chars(self) -> tuple[str, ...]:
        """The known characters, in identifier order (starting at id 1)."""
        return self._chars

    @property
    def size(self) -> int:
        """Total identifier count, including the reserved unknown id."""
        return len(self._chars) + 1

    @property
    def fingerprint(self) -> str:
        """A stable hex digest of the vocabulary; two identical vocabs share it exactly."""
        return self._fingerprint

    def id_of(self, char: str) -> int:
        """Return the identifier for ``char``, or :data:`UNK_ID` if it is unknown."""
        if len(char) != 1:
            message = f"id_of expects a single character, got {char!r}"
            raise ValueError(message)
        return self._char_to_id.get(char, UNK_ID)

    def char_of(self, token_id: int) -> str | None:
        """Return the character for ``token_id``, or ``None`` for the unknown id or a real gap."""
        if token_id == UNK_ID or token_id < 0 or token_id >= self.size:
            return None
        return self._chars[token_id - 1]

    def encode(self, text: str) -> tuple[int, ...]:
        """Map ``text`` to identifiers, sending any unknown character to :data:`UNK_ID`."""
        return tuple(self._char_to_id.get(char, UNK_ID) for char in text)

    def decode(self, ids: tuple[int, ...] | list[int]) -> str:
        """Reconstruct text from ``ids``.

        Unknown identifiers are skipped, so decoding is the inverse of encoding for known characters
        only; a round-trip that involved unknown characters is intentionally lossy.
        """
        out: list[str] = []
        for token_id in ids:
            char = self.char_of(int(token_id))
            if char is not None:
                out.append(char)
        return "".join(out)

    def unknown_fraction(self, text: str) -> float:
        """The fraction of ``text`` characters that are not in the vocabulary (0.0 if empty)."""
        if not text:
            return 0.0
        unknown = sum(1 for char in text if char not in self._char_to_id)
        return unknown / len(text)


def _compute_fingerprint(chars: tuple[str, ...]) -> str:
    # A content-derived identity for the vocabulary, matching how the rest of the project fingerprints
    # content (sha256 over a stable, delimited serialization). Codepoint-separated so no two distinct
    # character sets can collide by concatenation ambiguity.
    payload = "\x1f".join(chars).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
