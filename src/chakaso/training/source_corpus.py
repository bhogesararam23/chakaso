"""Turning a downloaded text into a pinned corpus, offline, with its provenance recorded.

The project's own documentation is one author's prose about one project, and E2605 measured a model on it.
Asking whether the same architecture learns from text it has no relationship to requires a corpus from
somewhere else — and an external corpus is exactly the case where provenance matters most, because nobody
can be asked to trust a file that appeared in a directory.

The rules this module encodes:

* **The source is named, not implied.** Each text carries its URL, the SHA-256 of the bytes as downloaded,
  the date they were taken, and the licence statement in its own words. None of that is inferred.
* **Boilerplate is cut by a stated marker, not by taste.** Project Gutenberg texts open and close with
  `*** START OF THE PROJECT GUTENBERG EBOOK … ***` lines; a file without both is refused rather than
  hand-edited, because a corpus you had to clean up by hand is not a corpus anyone can rebuild.
* **Documents are whole paragraphs grouped to a bound.** Chunking never splits mid-paragraph, so a held-out
  document cannot be a fragment of a trained-on sentence, and the chunking is a pure function of the text —
  the same bytes always give the same documents in the same order.
* **Nothing here touches the network.** Acquisition is an operator step (a `curl` recorded in the experiment
  directory); this module reads the files it is given and writes a snapshot that training then uses offline.

Standard library only, so the whole ingestion path imports, type-checks and runs in the dependency-free
matrix.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from chakaso.core.hashing import sha256_hex
from chakaso.training.errors import CorpusError
from chakaso.training.manifest import (
    CorpusDocument,
    CorpusManifest,
    normalize_text,
    write_snapshot,
)

__all__ = [
    "DEFAULT_MAX_CHARACTERS_PER_DOCUMENT",
    "GUTENBERG_BASE_URL",
    "IngestedSource",
    "build_corpus_manifest",
    "chunk_paragraphs",
    "ingest_project_gutenberg",
    "pin_project_gutenberg",
    "strip_gutenberg_wrappers",
]

GUTENBERG_BASE_URL = "https://www.gutenberg.org/cache/epub/"

#: A document is a chapter-sized unit, not a whole book: the split needs several independent units on each
#: side, and prose at this length is what the corpus is made of.
DEFAULT_MAX_CHARACTERS_PER_DOCUMENT = 16_384

_START = re.compile(r"^\*{3}\s*START OF\s+.{0,40}?PROJECT GUTENBERG.{0,80}?\*{3}\s*$", re.M)
_END = re.compile(r"^\*{3}\s*END OF\s+.{0,40}?PROJECT GUTENBERG.{0,80}?\*{3}\s*$", re.M)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True, slots=True)
class IngestedSource:
    """One downloaded text, with what was observed about it rather than what is assumed.

    Attributes:
        slug: Stable identifier used as the document path prefix, e.g. ``pg11``.
        ebook_number: The source's own identifier in the URL.
        title/author: Read from the file header; empty when the file does not state them.
        licence_statement: The file's own words about what may be done with it, verbatim.
        source_digest: SHA-256 of the downloaded bytes, before any normalization.
        raw_characters: Characters in the downloaded bytes as UTF-8 text.
        text: The body between the source's own start and end markers.
    """

    slug: str
    ebook_number: int
    title: str
    author: str
    licence_statement: str
    source_digest: str
    raw_characters: int
    text: str

    def __post_init__(self) -> None:
        if not self.slug or "/" in self.slug:
            message = f"source slug {self.slug!r} cannot be used as a path prefix"
            raise CorpusError(message)
        if not self.text.strip():
            message = f"source {self.slug!r} carries no text after its wrappers are removed"
            raise CorpusError(message)

    @property
    def url(self) -> str:
        return f"{GUTENBERG_BASE_URL}{self.ebook_number}/pg{self.ebook_number}.txt"

    def documents(self, max_characters: int) -> tuple[CorpusDocument, ...]:
        """The pinned documents this source contributes, in reading order."""
        return tuple(
            _document(f"{self.slug}/part-{index:04d}.md", chunk)
            for index, chunk in enumerate(
                chunk_paragraphs(self.text, max_characters=max_characters), 1
            )
        )


def _header_field(header: str, label: str) -> str:
    match = re.search(rf"^{label}:(.*)$", header, re.M)
    return match.group(1).strip() if match else ""


def strip_gutenberg_wrappers(raw: str, *, slug: str) -> tuple[str, str]:
    """Return the body between the source's own markers, and the file's header before them.

    Raises:
        CorpusError: either marker is absent or misplaced — the sign that this is not the file the record
            claims, or that it has already been edited.
    """
    text = normalize_text(raw)
    start = _START.search(text)
    end = _END.search(text)
    if start is None or end is None or end.start() <= start.end():
        message = (
            f"source {slug!r} does not carry both Project Gutenberg wrapper markers in order; the file is "
            "either not the one named or has been edited after download"
        )
        raise CorpusError(message)
    return text[: start.start()], text[start.end() : end.start()].strip("\n") + "\n"


def ingest_project_gutenberg(
    raw: str,
    *,
    slug: str,
    ebook_number: int,
    source_digest: str | None = None,
) -> IngestedSource:
    """Read one downloaded text into an :class:`IngestedSource`, recording what the file says about itself.

    ``source_digest`` is the SHA-256 of the bytes as they arrived, supplied by the caller that read them:
    decoding a file in text mode rewrites its line endings, and a provenance claim about "the downloaded
    bytes" has to survive someone running ``sha256sum`` on the same file. Left ``None``, the digest is taken
    over ``raw`` as given, which is the right thing for a caller holding a string.

    Raises:
        CorpusError: the wrappers are missing, or the body is empty.
    """
    header, body = strip_gutenberg_wrappers(raw, slug=slug)
    licence_lines = [
        line for line in header.split("\n")[:20] if line and not line.startswith("The Project")
    ]
    return IngestedSource(
        slug=slug,
        ebook_number=ebook_number,
        title=_header_field(header, "Title"),
        author=_header_field(header, "Author"),
        licence_statement="\n".join(licence_lines[:6]).strip(),
        source_digest=source_digest
        if source_digest is not None
        else sha256_hex(raw.encode("utf-8")),
        raw_characters=len(raw),
        text=body,
    )


def chunk_paragraphs(text: str, *, max_characters: int) -> tuple[str, ...]:
    """Group whole paragraphs into documents of at most ``max_characters``, in reading order.

    A paragraph longer than the bound becomes a document on its own rather than being cut: splitting
    mid-paragraph would put a sentence's halves on opposite sides of the split, which is the leak this
    layer exists to prevent.

    Raises:
        CorpusError: the bound is not positive, or the text has no paragraphs at all.
    """
    if max_characters < 1:
        message = f"max_characters must be positive, got {max_characters}"
        raise CorpusError(message)
    paragraphs = [block.strip() for block in _PARAGRAPH_BREAK.split(text) if block.strip()]
    if not paragraphs:
        message = "the source text has no paragraphs to chunk"
        raise CorpusError(message)

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if current and len(candidate) > max_characters:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return tuple(chunks)


def _document(path: str, text: str) -> CorpusDocument:
    encoded = text.encode("utf-8")
    return CorpusDocument(
        path=path,
        sha256=hashlib.sha256(encoded).hexdigest(),
        byte_count=len(encoded),
        char_count=len(text),
        text=text,
    )


def build_corpus_manifest(
    sources: Sequence[IngestedSource],
    *,
    name: str,
    version: str,
    licence: str,
    source: str,
    retrieved: str,
    max_characters_per_document: int = DEFAULT_MAX_CHARACTERS_PER_DOCUMENT,
) -> CorpusManifest:
    """Pin the documents several sources contribute, in slug then reading order.

    Args:
        retrieved: The date the source bytes were taken, as an ISO date. Provenance needs the version of a
            collection, not the collection's name.

    Raises:
        CorpusError: two sources share a slug, or a manifest field is empty.
    """
    if len({item.slug for item in sources}) != len(sources):
        message = "two sources share a slug, so their documents could not be told apart"
        raise CorpusError(message)
    documents: list[CorpusDocument] = []
    for item in sources:
        documents.extend(item.documents(max_characters_per_document))
    paths = [document.path for document in documents]
    if len(set(paths)) != len(paths):
        message = "the ingested documents are not uniquely identified by path"
        raise CorpusError(message)
    return CorpusManifest(
        name=name,
        version=version,
        licence=licence,
        source=f"{source} (retrieved {retrieved})",
        # The prefix rules are what would select these documents if they were ever tracked as files; the
        # snapshot is how they actually reach a run. Recording both keeps the manifest self-describing.
        include=tuple(f"{item.slug}/" for item in sources),
        extensions=(".md",),
        exact_paths=(),
        max_document_bytes=max_characters_per_document * 4,
        normalization="crlf-and-cr-to-lf",
        documents=tuple(sorted(documents, key=lambda document: document.path)),
    )


def pin_project_gutenberg(
    files: Mapping[str, tuple[int, Path]],
    *,
    name: str,
    version: str,
    licence: str,
    source: str,
    retrieved: str,
    snapshot: Path,
    max_characters_per_document: int = DEFAULT_MAX_CHARACTERS_PER_DOCUMENT,
    allow_overwrite: bool = False,
) -> tuple[CorpusManifest, str]:
    """Read downloaded texts from disk, pin them, and write the snapshot training will use.

    Args:
        files: slug → (ebook number, path to the downloaded file). The paths are read as UTF-8; a file that
            is not UTF-8 is refused, because a corpus of undecodable bytes is not a corpus.
        snapshot: Where to write the artifact. Under ``data/``, per the artifact policy.

    Returns:
        The pinned manifest and the snapshot's checksum.

    Raises:
        CorpusError: a source cannot be ingested, or a snapshot already exists and overwriting was not asked
            for.
    """
    sources: list[IngestedSource] = []
    digests: dict[str, str] = {}
    for slug, (ebook_number, path) in sorted(files.items()):
        raw_bytes = Path(path).read_bytes()
        raw = raw_bytes.decode("utf-8-sig")
        ingested = ingest_project_gutenberg(
            raw, slug=slug, ebook_number=ebook_number, source_digest=sha256_hex(raw_bytes)
        )
        sources.append(ingested)
        digests[slug] = ingested.source_digest
    manifest = build_corpus_manifest(
        sources,
        name=name,
        version=version,
        licence=licence,
        source=source,
        retrieved=retrieved,
        max_characters_per_document=max_characters_per_document,
    )
    checksum = write_snapshot(
        manifest,
        snapshot,
        acquisition={
            "method": "chakaso-project-gutenberg",
            "retrieved": retrieved,
            "urls": ";".join(item.url for item in sources),
            "source_digests": ";".join(f"{key}={value}" for key, value in sorted(digests.items())),
            "characters_per_document_bound": str(max_characters_per_document),
            "titles": ";".join(item.title or item.slug for item in sources),
        },
        allow_overwrite=allow_overwrite,
    )
    return manifest, checksum
