"""The evidence pack, and citation resolution.

The pack is the boundary that makes provenance checkable. Only the chunks in it are
visible to one generation call, so an identifier the model produced that is not in
the pack is caught rather than rendered (ADR-0003).

Resolution is a validation gate, not a lookup. Three outcomes are distinguished,
because they are three different facts about the system:

* a **citation** — a well-formed identifier that was supplied, resolved to a stored
  source;
* an **unknown reference** — a well-formed identifier that was not supplied, which
  is a fabrication and is counted as one;
* a **malformed reference** — something shaped like an identifier that is not
  valid, which means the model attempted a citation and got the form wrong.

Dropping either of the last two would make the system's failure rate invisible, so
both are returned rather than logged and forgotten.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from chakaso.core.identifiers import (
    CHUNK_ID_PREFIX,
    SOURCE_ID_PREFIX,
    ChunkId,
    SourceId,
)
from chakaso.evidence.errors import SourceRecordError
from chakaso.evidence.records import EvidenceChunk, SourceRecord

__all__ = [
    "Citation",
    "CitationResolution",
    "EvidencePack",
    "find_references",
]

#: Anything shaped like an attempted identifier reference: the prefix followed by at
#: least four hexadecimal characters. Deliberately looser than the identifier
#: grammar, so a truncated digest or an upper-case one is reported as a malformed
#: attempt rather than being ignored.
#:
#: Deliberately *not* looser than that. A pattern accepting any word characters
#: would also match ordinary prose such as "src_file", and a false positive in the
#: fabrication metric is worse than a missed one, because it makes a real number
#: untrustworthy. The cost is that an attempt using non-hexadecimal characters,
#: such as `src_ZZZZZZZZ`, is not detected at all.
_REFERENCE_PATTERN = re.compile(rf"(?:{SOURCE_ID_PREFIX}|{CHUNK_ID_PREFIX})[0-9a-fA-F]{{4,}}")


@dataclass(frozen=True, slots=True)
class Citation:
    """One reference from an answer, resolved against the pack that was supplied."""

    identifier: str
    source: SourceRecord
    chunk: EvidenceChunk | None = None


@dataclass(frozen=True, slots=True)
class CitationResolution:
    """What an answer's references turned out to be."""

    citations: tuple[Citation, ...] = ()
    unknown_identifiers: tuple[str, ...] = ()
    malformed_identifiers: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """Whether every reference resolved.

        A false value is a measurement, not a cosmetic problem: it means the answer
        referenced evidence that was not in front of it.
        """
        return not self.unknown_identifiers and not self.malformed_identifiers

    @property
    def cited_source_ids(self) -> tuple[SourceId, ...]:
        """Distinct source identifiers referenced, in order of first appearance."""
        seen: dict[SourceId, None] = {}
        for citation in self.citations:
            seen.setdefault(citation.source.source_id, None)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class EvidencePack:
    """The evidence supplied to exactly one generation call.

    Construction validates the pack rather than trusting the retriever: a chunk
    whose source is missing, or two chunks sharing an identifier, would make
    citation resolution report nonsense, and a pack is cheap to check.
    """

    chunks: tuple[EvidenceChunk, ...]
    sources: Mapping[SourceId, SourceRecord]
    retrieval_config: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            message = "evidence pack contains duplicate chunk identifiers"
            raise SourceRecordError(message)

        for chunk in self.chunks:
            if chunk.source_id not in self.sources:
                message = (
                    f"evidence pack is missing the source record for {chunk.source_id}, "
                    f"which chunk {chunk.chunk_id} refers to"
                )
                raise SourceRecordError(message)

    @classmethod
    def of(
        cls,
        chunks: Iterable[EvidenceChunk],
        sources: Iterable[SourceRecord],
        *,
        retrieval_config: Mapping[str, str] | None = None,
    ) -> EvidencePack:
        """Build a pack from chunks and the sources they came from.

        Sources that no chunk refers to are dropped: a pack records the evidence
        that was supplied, and keeping an unused source would let an answer cite
        something the model never saw.
        """
        source_index = {source.source_id: source for source in sources}
        included_chunks = tuple(chunks)
        used = {chunk.source_id for chunk in included_chunks}
        return cls(
            chunks=included_chunks,
            sources=MappingProxyType(
                {
                    source_id: source
                    for source_id, source in source_index.items()
                    if source_id in used
                }
            ),
            retrieval_config=MappingProxyType(dict(retrieval_config or {})),
        )

    @property
    def is_empty(self) -> bool:
        """Whether no evidence was supplied."""
        return not self.chunks

    @property
    def supplied_identifiers(self) -> frozenset[str]:
        """Every identifier the model was given: all chunk ids and their source ids."""
        identifiers = {str(chunk.chunk_id) for chunk in self.chunks}
        identifiers.update(str(source_id) for source_id in self.sources)
        return frozenset(identifiers)

    def source_for(self, source_id: SourceId) -> SourceRecord | None:
        """The source record for ``source_id``, or ``None`` if it is not in the pack."""
        return self.sources.get(source_id)

    def chunk_for(self, chunk_id: ChunkId) -> EvidenceChunk | None:
        """The chunk for ``chunk_id``, or ``None`` if it is not in the pack."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None


def find_references(text: str) -> tuple[str, ...]:
    """Return every identifier-shaped reference in ``text``, in order, deduplicated.

    Deduplicated because a claim that names the same source three times is one
    citation, and counting it three times would inflate a citation metric.
    """
    seen: dict[str, None] = {}
    for match in _REFERENCE_PATTERN.finditer(text):
        seen.setdefault(match.group(0), None)
    return tuple(seen)


def resolve_citations(text: str, pack: EvidencePack) -> CitationResolution:
    """Resolve the references in ``text`` against ``pack``.

    Nothing is resolved that was not supplied. An identifier outside the pack is
    recorded as unknown and never looked up, which is the rule ADR-0003 exists to
    enforce: a fabricated citation must not become a rendered URL.

    The order of the three checks matters and is the point of this function.
    Validity is established first, then membership, then resolution. Testing
    membership first — which is the obvious way to write it — puts every malformed
    reference into the unknown bucket and makes the malformed bucket unreachable, so
    "the model invented a source" and "the model fumbled the format" become one
    number. They are different defects with different fixes.
    """
    supplied = pack.supplied_identifiers
    citations: list[Citation] = []
    unknown: list[str] = []
    malformed: list[str] = []

    for reference in find_references(text):
        if reference.startswith(CHUNK_ID_PREFIX):
            try:
                chunk_id = ChunkId.parse(reference)
            except ValueError:
                malformed.append(reference)
                continue
            if reference not in supplied:
                unknown.append(reference)
                continue
            chunk = pack.chunk_for(chunk_id)
            source = pack.source_for(chunk.source_id) if chunk is not None else None
            if chunk is None or source is None:
                # Unreachable while the pack validates itself, but resolving a
                # citation against evidence that is not there would be worse than
                # reporting it as unresolved.
                unknown.append(reference)
                continue
            citations.append(Citation(identifier=reference, source=source, chunk=chunk))
            continue

        try:
            source_id = SourceId.parse(reference)
        except ValueError:
            malformed.append(reference)
            continue
        if reference not in supplied:
            unknown.append(reference)
            continue
        source = pack.source_for(source_id)
        if source is None:
            unknown.append(reference)
            continue
        citations.append(Citation(identifier=reference, source=source))

    return CitationResolution(
        citations=tuple(citations),
        unknown_identifiers=tuple(unknown),
        malformed_identifiers=tuple(malformed),
    )
