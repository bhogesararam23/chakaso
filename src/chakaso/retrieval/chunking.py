"""Splitting document text into evidence chunks.

A chunk is the unit an answer cites, so its boundaries are part of the citation
contract rather than an implementation detail. Three properties follow from that and
are the reason this module is as opinionated as it is.

**Section boundaries are never crossed.** A chunk that spans two headings has no
defensible citation position: a reader following the reference cannot tell which part
of the source is being claimed. Sections are therefore split before anything else.

**Chunk identifiers are derived from content** (ADR-0007), so re-chunking a document
with different settings produces a different set of chunks rather than redefining what
an existing identifier refers to. That is a feature: an older answer keeps pointing at
the text it actually used.

**Splitting is deterministic.** The same text and settings always produce identical
chunks. Evaluation compares runs, and a chunker with any variability in it would make
retrieved evidence incomparable between two runs of the same configuration.

Sizes are counted in characters, not tokens. There is no tokenizer in this project, so
a token budget would be a guess wearing a unit. Character counts are exact and are
labelled as what they are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chakaso.core.identifiers import SourceId
from chakaso.evidence import EvidenceChunk
from chakaso.retrieval.errors import ChunkingError

__all__ = ["ChunkingConfig", "chunk_document", "split_sections"]

#: An ATX Markdown heading, capturing its level and text. Markdown is the format the
#: project's own documents use, and it is the only structure extractor implemented:
#: HTML needs a processor and a fetch policy, and neither exists.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

#: A paragraph break: one or more blank lines.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

#: Sentence ends, for splitting a paragraph that cannot fit in one chunk. Deliberately
#: crude: it exists so that pathological input degrades into readable pieces, not to
#: make a claim about sentence structure. An abbreviation splits a sentence here, and
#: the cost is one extra chunk.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """How much text one chunk should hold.

    The defaults are a starting point for a small model, not a measured optimum, and
    they are not configuration yet: nothing consumes a chunking setting from a
    configuration file, so adding one would be a field no code reads.

    Attributes:
        target_characters: Chunks are filled towards this size. A paragraph that would
            push a chunk past it starts a new chunk instead.
        max_characters: A hard ceiling. Text that cannot be split to fit — a single
            sentence longer than this — is cut, because an oversized chunk breaks an
            assumption the caller has made about what it will receive.
    """

    target_characters: int = 800
    max_characters: int = 2000

    def __post_init__(self) -> None:
        if self.target_characters < 1:
            message = f"target_characters must be at least 1, got {self.target_characters}"
            raise ChunkingError(message)
        if self.max_characters < self.target_characters:
            message = (
                f"max_characters ({self.max_characters}) must not be below "
                f"target_characters ({self.target_characters}); every chunk is expected "
                "to fit the ceiling, so a ceiling below the target would be unsatisfiable"
            )
            raise ChunkingError(message)


@dataclass(frozen=True, slots=True)
class Section:
    """A run of document text under one heading path.

    ``heading`` is the heading path with its levels, for example ``"4. Deadlines >
    Extensions"``, or ``None`` for text that appears before any heading.
    """

    heading: str | None
    body: str


def split_sections(text: str) -> tuple[Section, ...]:
    """Split ``text`` at Markdown headings, keeping each heading with its body.

    Text before the first heading becomes a section with ``heading=None``. Nothing is
    discarded: a document with no headings at all is one section, and its whole body is
    that section's body.
    """
    sections: list[Section] = []
    heading_path: str | None = None
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body or heading_path is not None:
            sections.append(Section(heading=heading_path, body=body))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is None:
            body_lines.append(line)
            continue

        flush()
        body_lines = []
        level = len(match.group(1))
        title = match.group(2).strip()
        # Deeper headings nest under shallower ones, so a chunk from a subsection
        # reports where it sits rather than only its immediate heading.
        prefix = heading_path.split(" > ")[: level - 1] if heading_path else []
        heading_path = " > ".join([*prefix, title]) if title else None

    flush()
    return tuple(section for section in sections if section.body)


def chunk_document(
    source_id: SourceId,
    text: str,
    *,
    config: ChunkingConfig | None = None,
) -> tuple[EvidenceChunk, ...]:
    """Split ``text`` belonging to ``source_id`` into evidence chunks.

    Positions are sequential across the whole document, so position order is document
    order and a gap would mean text was dropped.

    A document with no text produces no chunks. An empty chunk is not evidence and the
    record type rejects it, so returning one would be a worse answer than returning
    nothing.

    Raises:
        ChunkingError: ``config`` is unsatisfiable.
    """
    resolved = config if config is not None else ChunkingConfig()
    pieces: list[tuple[str | None, str]] = []

    for section in split_sections(text):
        for body in _pack_paragraphs(section.body, resolved):
            pieces.append((section.heading, body))

    return tuple(
        EvidenceChunk.create(
            source_id=source_id,
            text=body,
            position=position,
            section=heading,
            # Left unset rather than guessed: counting tokens needs a tokenizer, and a
            # character count in a field named token_count would be a lie that a later
            # reader would trust.
            token_count=None,
        )
        for position, (heading, body) in enumerate(pieces)
    )


def _pack_paragraphs(body: str, config: ChunkingConfig) -> list[str]:
    """Group a section's paragraphs into chunk-sized pieces, in order."""
    paragraphs: list[str] = []
    for raw in _PARAGRAPH_BREAK.split(body):
        paragraph = raw.strip()
        if not paragraph:
            continue
        paragraphs.extend(_split_oversized(paragraph, config))

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        # A separator costs two characters when paragraphs are joined, and ignoring it
        # would let a chunk exceed the target by exactly the amount it is joined with.
        projected = current_length + len(paragraph) + (2 if current else 0)
        if current and projected > config.target_characters:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
            continue
        current.append(paragraph)
        current_length = projected

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_oversized(paragraph: str, config: ChunkingConfig) -> list[str]:
    """Break a paragraph that cannot fit in one chunk.

    Sentences first, because a sentence boundary is at least a real boundary in the
    text. Anything still too long is cut, which can split mid-word: the alternative is
    a chunk larger than the ceiling the caller was promised, and a reader would rather
    have a truncated sentence than an unbounded one.
    """
    if len(paragraph) <= config.max_characters:
        return [paragraph]

    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_BREAK.split(paragraph):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= config.max_characters:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(sentence) <= config.max_characters:
            current = sentence
            continue
        pieces.extend(
            sentence[start : start + config.max_characters]
            for start in range(0, len(sentence), config.max_characters)
        )
        current = ""

    if current:
        pieces.append(current)
    return pieces
