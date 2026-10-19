"""The experiment corpus: a small, self-authored, redistributable text.

This module exists so the first training experiment has data it is allowed to use and can reproduce
from the repository alone. The corpus is **self-authored** for Chakaso — it is not copied from any
external source — so there is no licence to track down, no download, and no contamination risk from a
public benchmark. It is covered by the project's Apache-2.0 licence like the rest of the code
(ADR-0004); no third-party terms apply.

The text is deliberately tiny and repetitive on purpose. A first experiment is not trying to model
natural-language breadth; it is trying to establish, and to *measure*, that a model initialised at
random can lower its held-out next-character prediction loss against a trivial baseline. That needs a
sequence with real but learnable structure — recurring character patterns and a limited alphabet — so
that a genuine reduction in loss is achievable within a small, CPU-only budget and can be reproduced
exactly from a seed. Breadth would only make the first curve harder to interpret, not more honest.

Everything the record needs about the data — source, licence, size and the limitation that it is a
toy — is exposed here rather than left to be inferred from a string constant.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CORPUS", "Corpus", "corpus_text"]

# A compact original text. It reuses a small set of words and punctuation so a character-level model
# has learnable structure, while staying legible enough that a person can sanity-check what the model
# is being asked to predict. Authored for this project; see the module docstring.
_CORPUS_TEXT = """the quick red fox ran over the dry sand.
the red fox ran and the quick fox slept.
a quick brown owl flew over the quiet town.
the owl slept while the fox ran and ran.
sand and stone and dry brown grass.
the fox and the owl and the quiet town.
quick runs over the sand, slow runs over the stone.
the town slept under the quiet brown sky.
red fox, brown owl, quick and quiet.
over the sand and over the stone the river ran.
"""


@dataclass(frozen=True, slots=True)
class Corpus:
    """The experiment corpus with the provenance the record must cite.

    Attributes:
        name: A stable identifier for the corpus, used in the experiment configuration.
        version: The corpus version; a change to the text is a version bump, never an edit in place.
        source: Where the text came from — here, authored for the project.
        licence: The licence terms that permit its use and redistribution.
        text: The full corpus text.
    """

    name: str
    version: str
    source: str
    licence: str
    text: str

    def __post_init__(self) -> None:
        if not self.name:
            message = "corpus name must not be empty"
            raise ValueError(message)
        if not self.text:
            message = "corpus text must not be empty"
            raise ValueError(message)

    @property
    def char_count(self) -> int:
        """Number of characters in the corpus text."""
        return len(self.text)


#: The corpus the first experiment uses.
CORPUS = Corpus(
    name="chakaso-toy-char-v1",
    version="1.0.0",
    source="Authored for Chakaso (self-created; no external text was copied).",
    licence="Apache-2.0 (the project's own licence; no third-party terms apply).",
    text=_CORPUS_TEXT,
)


def corpus_text() -> str:
    """Return the corpus text.

    Provided as a function so callers depend on the accessor rather than importing a mutable module
    attribute; the underlying :class:`Corpus` is frozen.
    """
    return CORPUS.text
