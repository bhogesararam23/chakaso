"""A corpus manifest: training data named by content, not by whatever is on disk.

The project's earlier data was a string constant in ``chakaso.training.corpus`` — honest about its
provenance because it was tiny and self-authored, but not a path to a real corpus. A model trained on
text that cannot be named, hashed and re-read is not reproducible, so this module makes the corpus an
artifact with the same discipline the rest of the repository applies to evidence:

* **Pinned content.** Every document is recorded by its relative path *and* the SHA-256 of its normalized
  bytes, so the corpus is exactly the text it says it is.
* **Refusal of drift.** Loading re-reads and re-verifies. A file that changed, disappeared, or a new file
  that joined the declared include set is a :class:`chakaso.training.errors.CorpusDriftError` naming the
  paths — never a silent substitution of a different dataset under a saved result.
* **Recorded licence and preparation.** Source, licence, normalization step and size bound live on the
  manifest, because "we trained on some markdown" is not provenance.
* **Snapshotting.** The bytes a run actually consumed can be written once into a digest-checked file, so a
  run stays replayable after the tracked text moves — with the drift guard still standing between the
  manifest and a fresh preparation. A snapshot carries its own checksum and an acquisition record;
  :mod:`chakaso.training.snapshot` explains how the bytes for a *past* revision are recovered.

Digests are taken over the **normalized** text (line endings to ``\\n``), so a CRLF checkout and an LF
checkout produce one fingerprint. That is the point of normalizing before hashing: a corpus identity must
not depend on a contributor's git configuration.

Standard library only, so the corpus layer is importable, testable and type-checkable in the
dependency-free CI matrix.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from chakaso.core.atomic import write_text_atomically
from chakaso.core.hashing import is_sha256_hex, join_parts, sha256_hex
from chakaso.training.data import TextUnit
from chakaso.training.errors import (
    CorpusDriftError,
    CorpusError,
    CorruptedCorpusArtifactError,
)

__all__ = [
    "DEFAULT_MAX_DOCUMENT_BYTES",
    "CorpusDocument",
    "CorpusManifest",
    "CorpusSnapshot",
    "TextUnit",
    "build_manifest",
    "normalize_text",
    "read_manifest",
    "read_snapshot",
    "save_manifest",
    "verify_corpus",
    "write_snapshot",
]

_MANIFEST_SCHEMA = "chakaso-corpus-manifest/1"
_SNAPSHOT_SCHEMA = "chakaso-corpus-snapshot/2"
_NORMALIZATION = "crlf-and-cr-to-lf"
#: Files larger than this are refused: a corpus document is a document, not a data blob, and an accidental
#: half-gigabyte file must not be read into memory by a preparation step.
DEFAULT_MAX_DOCUMENT_BYTES = 512 * 1024

_GLOB_CHARACTERS = "*?["


def normalize_text(raw: str) -> str:
    """Return ``raw`` with every line ending rewritten to ``\\n``.

    The only transformation applied. It exists so a corpus fingerprint is independent of the working-tree
    line endings a git checkout happened to use. Nothing else is altered — whitespace, punctuation and
    Unicode stay exactly as authored.
    """
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_hidden(relative: str) -> bool:
    """Whether a relative path lies under a dot-directory or is itself a dot-file.

    Virtual environments, tool caches and agent scratch space all live in dot-directories here, and each
    one contains markdown that is not the project's text (dependency licenses, cached reports). A corpus
    must not acquire them by accident.
    """
    return any(part.startswith(".") for part in PurePosixPath(relative).parts)


def _is_glob(entry: str) -> bool:
    return any(character in entry for character in _GLOB_CHARACTERS)


def _selects(relative: str, include: Sequence[str]) -> bool:
    """Whether the include spec selects ``relative``.

    Three entry forms, deliberately different because they answer different needs: an entry ending in
    ``/`` is a directory prefix (recursive, restricted to the declared extensions); an entry containing a
    glob character is matched against the whole relative path with :func:`fnmatch.fnmatch` (and ``*`` in
    fnmatch crosses ``/``, so ``*.md`` means "any markdown, anywhere" — stated here because it surprises
    people); any other entry is an exact relative path, taken whatever its extension, so a license file
    with no suffix can be named.
    """
    for entry in include:
        if entry.endswith("/"):
            if relative.startswith(entry):
                return True
        elif _is_glob(entry):
            if fnmatch.fnmatch(relative, entry):
                return True
        elif relative == entry:
            return True
    return False


def discover_documents(
    root: Path,
    *,
    include: Sequence[str],
    extensions: Sequence[str] = (".md",),
    exact_paths: Sequence[str] = (),
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
) -> tuple[Path, ...]:
    """Return the files under ``root`` the include spec selects, sorted by relative POSIX path.

    Dot-directories are never searched. Glob and directory entries only ever select the declared
    extensions, so an unexpected binary cannot enter a text corpus; a file listed in ``exact_paths`` is
    selected whatever its suffix, which is how ``LICENSE`` joins a markdown corpus.

    Raises:
        CorpusError: the spec selects nothing, a named exact path does not exist, or a selected file is
            larger than ``max_document_bytes``.
    """
    if not include:
        message = "a corpus needs at least one include entry; none was given"
        raise CorpusError(message)
    if not root.is_dir():
        message = f"corpus root {str(root)!r} is not a directory"
        raise CorpusError(message)

    exact = set(exact_paths)
    for entry in exact:
        if not (root / entry).is_file():
            message = f"corpus include names {entry!r} as an exact file, but it is not a file under {str(root)!r}"
            raise CorpusError(message)

    selected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = _relative_posix(path, root)
        if _is_hidden(relative):
            continue
        if relative in exact:
            selected.append(path)
            continue
        if not _selects(relative, include) or path.suffix not in extensions:
            continue
        selected.append(path)

    # `rglob` never descends into hidden directories only by the filter above, and an oversized document
    # is refused only once it is actually selected: a bound that applies to files nobody asked for would
    # break preparation for reasons unrelated to the corpus.
    for path in selected:
        size = path.stat().st_size
        if size > max_document_bytes:
            message = (
                f"corpus document {_relative_posix(path, root)!r} is {size} bytes, above the declared "
                f"bound {max_document_bytes}"
            )
            raise CorpusError(message)

    if not selected:
        message = (
            f"the include spec {list(include)!r} (extensions {list(extensions)!r}) selected no files "
            f"under {str(root)!r}"
        )
        raise CorpusError(message)
    # Sorted by *relative* POSIX path, not by the order rglob happened to yield: the manifest requires
    # documents in path order so that one corpus always serializes to one byte-identical record.
    return tuple(sorted(selected, key=lambda path: _relative_posix(path, root)))


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One document of a corpus, pinned by its path and the digest of its normalized text."""

    path: str
    sha256: str
    byte_count: int
    char_count: int
    text: str = ""

    def __post_init__(self) -> None:
        if not self.path or _is_hidden(self.path):
            message = f"a corpus document needs a non-hidden relative path, got {self.path!r}"
            raise CorpusError(message)
        if not is_sha256_hex(self.sha256):
            message = f"document {self.path!r} carries a digest that is not a lowercase sha256 hex"
            raise CorpusError(message)
        if self.byte_count < 0 or self.char_count < 0:
            message = f"document {self.path!r} records negative counts"
            raise CorpusError(message)

    @property
    def identifier(self) -> str:
        """A stable identifier for this exact content at this path."""
        return f"{self.path}@{self.sha256[:12]}"


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """A corpus as a recorded, verifiable object: what it is, where it came from, what it contains."""

    name: str
    version: str
    licence: str
    source: str
    include: tuple[str, ...]
    extensions: tuple[str, ...]
    exact_paths: tuple[str, ...]
    max_document_bytes: int
    normalization: str
    documents: tuple[CorpusDocument, ...]

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "licence", "source", "normalization"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                message = f"corpus manifest field {field_name!r} must be a non-empty string"
                raise CorpusError(message)
        if not self.documents:
            message = f"corpus manifest {self.name!r} has no documents"
            raise CorpusError(message)
        paths = [document.path for document in self.documents]
        if len(set(paths)) != len(paths):
            message = f"corpus manifest {self.name!r} records a path twice: {paths}"
            raise CorpusError(message)
        if paths != sorted(paths):
            message = f"corpus manifest {self.name!r} documents are not sorted by path: {paths}"
            raise CorpusError(message)
        if self.max_document_bytes < 1:
            message = f"max_document_bytes must be >= 1, got {self.max_document_bytes}"
            raise CorpusError(message)

    @property
    def fingerprint(self) -> str:
        """A short digest over the exact paths and content digests this manifest pins."""
        payload = join_parts(
            _MANIFEST_SCHEMA,
            self.name,
            self.version,
            self.normalization,
            str(len(self.documents)),
            join_parts(
                *(join_parts(document.path, document.sha256) for document in self.documents)
            ),
        )
        return sha256_hex(payload)[:16]

    @property
    def char_count(self) -> int:
        """Total characters across every document."""
        return sum(document.char_count for document in self.documents)

    @property
    def byte_count(self) -> int:
        """Total bytes of normalized text across every document."""
        return sum(document.byte_count for document in self.documents)

    def text_units(self) -> tuple[TextUnit, ...]:
        """The documents as split-ready units, in the manifest's order.

        Each unit's ``group`` is the document's first path segment, which is the source's directory in a
        corpus built from books (``pg1342/part-0007.md``) and a repository directory in one built from the
        tracked tree (``docs/architecture.md``). A document with no directory is its own group. Grouping is
        only consulted when a caller asks for a whole-work split; a document split ignores it.
        """
        units: list[TextUnit] = []
        for document in self.documents:
            if not document.text:
                continue
            head, separator, _ = document.identifier.partition("/")
            units.append(
                TextUnit(
                    identifier=document.identifier,
                    text=document.text,
                    group=head if separator else document.identifier,
                )
            )
        return tuple(units)

    @property
    def duplicate_paths(self) -> tuple[tuple[str, ...], ...]:
        """Groups of paths whose recorded content is byte-identical.

        Exposed because duplicated text is a data-hygiene problem, not a curiosity: a repeated document is
        trained on twice, and if its copies ever land on opposite sides of a split the held-out number
        stops being held out. The split layer drops these; a caller who wants to inspect the corpus can
        ask the manifest instead of guessing.
        """
        by_digest: dict[str, list[str]] = {}
        for document in self.documents:
            by_digest.setdefault(document.sha256, []).append(document.path)
        return tuple(tuple(paths) for _, paths in sorted(by_digest.items()) if len(paths) > 1)

    def combined_text(self) -> str:
        """Every document's text joined by a blank line, in manifest order."""
        return "\n\n".join(document.text for document in self.documents if document.text)

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": _MANIFEST_SCHEMA,
            "name": self.name,
            "version": self.version,
            "licence": self.licence,
            "source": self.source,
            "include": list(self.include),
            "extensions": list(self.extensions),
            "exact_paths": list(self.exact_paths),
            "max_document_bytes": self.max_document_bytes,
            "normalization": self.normalization,
            "documents": [
                {
                    "path": document.path,
                    "sha256": document.sha256,
                    "byte_count": document.byte_count,
                    "char_count": document.char_count,
                }
                for document in self.documents
            ],
        }

    @staticmethod
    def _canonical(content: dict[str, Any]) -> str:
        # One deterministic serialization is the checksum's input, so a reordered dict cannot change it.
        return json.dumps(content, sort_keys=True, separators=(",", ":"))

    def checksum(self) -> str:
        """The digest of the manifest record itself, excluding the checksum field."""
        return sha256_hex(self._canonical(self._content()))

    def to_document(self) -> dict[str, Any]:
        """The on-disk record: content plus its own checksum. Never carries the text."""
        document = dict(self._content())
        document["checksum"] = self.checksum()
        return document

    @classmethod
    def from_document(
        cls, document: dict[str, Any], *, texts: dict[str, str] | None = None
    ) -> CorpusManifest:
        """Rebuild a manifest from its record, verifying the record's own checksum.

        ``texts`` supplies document bodies when they arrive from elsewhere (a snapshot). When supplied,
        every body is re-hashed against the digest the manifest records, so a snapshot cannot smuggle
        altered text under a valid manifest checksum.

        Raises:
            CorruptedCorpusArtifactError: the checksum is missing or disagrees with the content, or a
                supplied body does not hash to the recorded digest.
            CorpusError: the schema is unknown or a required field is missing or mis-typed.
        """
        if "checksum" not in document:
            message = "corpus manifest has no checksum field"
            raise CorruptedCorpusArtifactError(message)
        content = {key: value for key, value in document.items() if key != "checksum"}
        stored = document["checksum"]
        if not isinstance(stored, str) or sha256_hex(cls._canonical(content)) != stored:
            message = "corpus manifest checksum does not match its content — the file was altered"
            raise CorruptedCorpusArtifactError(message)
        if content.get("schema_version") != _MANIFEST_SCHEMA:
            message = f"unsupported corpus manifest schema {content.get('schema_version')!r}"
            raise CorpusError(message)

        try:
            entries = content["documents"]
            rebuilt: list[CorpusDocument] = []
            for entry in entries:
                path = str(entry["path"])
                digest = str(entry["sha256"])
                text = "" if texts is None else texts.get(path, "")
                if texts is not None and path in texts:
                    computed = sha256_hex(text.encode("utf-8"))
                    if computed != digest:
                        message = (
                            f"document {path!r} does not hash to the digest the manifest records "
                            f"(stored {digest[:12]}…, computed {computed[:12]}…)"
                        )
                        raise CorruptedCorpusArtifactError(message)
                rebuilt.append(
                    CorpusDocument(
                        path=path,
                        sha256=digest,
                        byte_count=int(entry["byte_count"]),
                        char_count=int(entry["char_count"]),
                        text=text,
                    )
                )
            return cls(
                name=str(content["name"]),
                version=str(content["version"]),
                licence=str(content["licence"]),
                source=str(content["source"]),
                include=tuple(str(item) for item in content["include"]),
                extensions=tuple(str(item) for item in content["extensions"]),
                exact_paths=tuple(str(item) for item in content.get("exact_paths", ())),
                max_document_bytes=int(content["max_document_bytes"]),
                normalization=str(content["normalization"]),
                documents=tuple(sorted(rebuilt, key=lambda item: item.path)),
            )
        except CorruptedCorpusArtifactError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            message = f"corpus manifest is missing or mis-typed a required field: {exc}"
            raise CorpusError(message) from exc


def _document_from_file(path: Path, root: Path) -> CorpusDocument:
    relative = _relative_posix(path, root)
    raw = path.read_bytes()
    text = normalize_text(_decode(raw, relative))
    encoded = text.encode("utf-8")
    return CorpusDocument(
        path=relative,
        sha256=sha256_hex(encoded),
        byte_count=len(encoded),
        char_count=len(text),
        text=text,
    )


def _decode(raw: bytes, relative: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        message = f"corpus document {relative!r} is not valid UTF-8: {exc}"
        raise CorpusError(message) from exc


def build_manifest(
    root: Path,
    *,
    name: str,
    version: str,
    licence: str,
    source: str,
    include: Sequence[str] = ("docs/", "*.md"),
    extensions: Sequence[str] = (".md",),
    exact_paths: Sequence[str] = (),
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
) -> CorpusManifest:
    """Read the selected files under ``root`` and return the manifest that pins them.

    Raises:
        CorpusError: the spec selects nothing or a file is unreadable, not UTF-8 or oversized, or a
            manifest field is empty.
        CorpusDriftError: the selected files contain no text at all — an empty corpus is refused here
            rather than becoming a training run that silently evaluates a loss of zero.
    """
    paths = discover_documents(
        root,
        include=include,
        extensions=tuple(extensions),
        exact_paths=tuple(exact_paths),
        max_document_bytes=max_document_bytes,
    )
    documents = tuple(_document_from_file(path, root) for path in paths)
    if sum(document.char_count for document in documents) == 0:
        message = f"corpus {name!r} selected {len(documents)} file(s) and not one character of text"
        raise CorpusDriftError(message)
    return CorpusManifest(
        name=name,
        version=version,
        licence=licence,
        source=source,
        include=tuple(include),
        extensions=tuple(extensions),
        exact_paths=tuple(exact_paths),
        max_document_bytes=max_document_bytes,
        normalization=_NORMALIZATION,
        documents=documents,
    )


def verify_corpus(manifest: CorpusManifest, root: Path) -> CorpusManifest:
    """Re-discover and re-hash ``root``, returning a manifest that carries the freshly read text.

    The corpus is checked *as a set*, not file by file: an edited document, a deleted document and a new
    document that now matches the include spec are all drift, because each changes what a model was
    trained on.

    Raises:
        CorpusDriftError: any of the three cases above, naming every offending path.
        CorpusError: a selected file is unreadable or not UTF-8.
    """
    discovered = {
        _relative_posix(path, root): path
        for path in discover_documents(
            root,
            include=manifest.include,
            extensions=manifest.extensions,
            exact_paths=manifest.exact_paths,
            max_document_bytes=manifest.max_document_bytes,
        )
    }
    recorded = {document.path: document for document in manifest.documents}

    missing = sorted(set(recorded) - set(discovered))
    added = sorted(set(discovered) - set(recorded))
    changed: list[str] = []
    fresh: list[CorpusDocument] = []
    for relative in sorted(set(recorded) & set(discovered)):
        current = _document_from_file(discovered[relative], root)
        if current.sha256 != recorded[relative].sha256:
            changed.append(relative)
        fresh.append(current)

    if missing or added or changed:
        details = []
        if changed:
            details.append(f"content changed: {changed}")
        if missing:
            details.append(f"recorded files absent: {missing}")
        if added:
            details.append(f"files now selected that the manifest does not record: {added}")
        message = (
            f"corpus {manifest.name!r} (fingerprint {manifest.fingerprint}) no longer matches the files "
            f"under {str(root)!r} — " + "; ".join(details)
        )
        raise CorpusDriftError(message)

    return CorpusManifest(
        name=manifest.name,
        version=manifest.version,
        licence=manifest.licence,
        source=manifest.source,
        include=manifest.include,
        extensions=manifest.extensions,
        exact_paths=manifest.exact_paths,
        max_document_bytes=manifest.max_document_bytes,
        normalization=manifest.normalization,
        documents=tuple(sorted(fresh, key=lambda document: document.path)),
    )


def _atomic_write_text(payload: str, target: Path, *, allow_overwrite: bool, label: str) -> None:
    """Write ``payload`` so a reader never sees half a file, and refuse to clobber unless told.

    The swap is :func:`chakaso.core.atomic.write_text_atomically`, which also guarantees that the bytes on
    disk are the characters the digest was computed over. The guard is ADR-0035's rule applied to corpus
    artifacts: the file a measured run cites is not something a later run gets to replace silently, and a
    corpus artifact truncated by an interrupted write would otherwise be indistinguishable from one that was
    altered on purpose.
    """
    if target.exists() and not allow_overwrite:
        message = (
            f"{label} already exists at {str(target)!r}; a recorded run's corpus artifact is not replaced "
            "unless the caller passes allow_overwrite=True"
        )
        raise CorpusError(message)
    write_text_atomically(payload, target)


def save_manifest(manifest: CorpusManifest, path: Path, *, allow_overwrite: bool = False) -> None:
    """Write the manifest record (never the text) as deterministic UTF-8 JSON.

    Raises:
        CorpusError: a file already occupies ``path`` and ``allow_overwrite`` was not passed.
    """
    payload = json.dumps(manifest.to_document(), sort_keys=True, separators=(",", ":"))
    _atomic_write_text(
        payload, Path(path), allow_overwrite=allow_overwrite, label="corpus manifest"
    )


def read_manifest(path: Path) -> CorpusManifest:
    """Read a manifest written by :func:`save_manifest`.

    The result carries no document text: text only ever arrives through :func:`verify_corpus` or
    :func:`read_snapshot`, both of which check it against the pinned digests.

    Raises:
        CorruptedCorpusArtifactError: the file is not valid JSON or its checksum disagrees.
        CorpusError: the schema or a required field is unknown.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        message = f"corpus manifest at {str(path)!r} is not valid JSON: {exc}"
        raise CorruptedCorpusArtifactError(message) from exc
    if not isinstance(document, dict):
        message = f"corpus manifest at {str(path)!r} must be a JSON object"
        raise CorruptedCorpusArtifactError(message)
    return CorpusManifest.from_document(document)


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """A corpus whose text travels with it: the pinned manifest, its bodies, and how they were obtained.

    Attributes:
        manifest: The corpus, carrying every document's verified text.
        checksum: The digest the snapshot file records over its own content.
        acquisition: Where the bytes came from, as recorded when the snapshot was written. Empty means "the
            tracked files, as they stood when this process read them"; a restored snapshot names the
            revision instead, which is what distinguishes a historical corpus from the working tree.
    """

    manifest: CorpusManifest
    checksum: str
    acquisition: Mapping[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """The manifest fingerprint this snapshot's bytes reproduce."""
        return self.manifest.fingerprint


def write_snapshot(
    manifest: CorpusManifest,
    path: Path,
    *,
    acquisition: Mapping[str, str] | None = None,
    allow_overwrite: bool = False,
) -> str:
    """Write the corpus text the manifest pins into one checksum-verified artifact; return its digest.

    A snapshot exists so a run's exact input survives later edits: the manifest fingerprint identifies the
    tracked files it was built from, and the snapshot identifies the bytes this corpus actually contributed.
    It belongs under ``data/``, which the artifact policy and ``.gitignore`` keep out of version control.

    Every document carries its own digest here as well as in the embedded manifest, and the file as a whole
    carries a checksum over its content, so a reader can tell an altered document from a truncated file
    from a substituted manifest — the three ways a snapshot could otherwise stop being the corpus a record
    names.

    Raises:
        CorpusError: the manifest carries no text (call :func:`verify_corpus` first, so the bytes being
            written have been checked against their digests), or a file already occupies ``path``.
    """
    if not any(document.text for document in manifest.documents):
        message = (
            "cannot snapshot a manifest with no document text; run verify_corpus() first so the bytes "
            "being written have been checked against their digests"
        )
        raise CorpusError(message)
    content: dict[str, Any] = {
        "schema_version": _SNAPSHOT_SCHEMA,
        "manifest_fingerprint": manifest.fingerprint,
        "manifest": manifest.to_document(),
        "documents": [
            {
                "path": document.path,
                "sha256": document.sha256,
                "byte_count": document.byte_count,
                "char_count": document.char_count,
                "text": document.text,
            }
            for document in manifest.documents
        ],
        "acquisition": dict(acquisition or {}),
    }
    payload = json.dumps(content, sort_keys=True, separators=(",", ":"))
    checksum = sha256_hex(payload)
    _atomic_write_text(
        json.dumps({**content, "checksum": checksum}, sort_keys=True, separators=(",", ":")),
        Path(path),
        allow_overwrite=allow_overwrite,
        label="corpus snapshot",
    )
    return checksum


def read_snapshot(path: Path) -> CorpusSnapshot:
    """Load a snapshot written by :func:`write_snapshot`, re-checking every digest it carries.

    The order of checks is the order of trust: the file's own checksum first (so a truncated or edited file
    is named as corruption rather than as a corpus problem), then the manifest record inside it, then every
    document body against the digest the manifest pins, then the document set against the manifest's path
    set. Nothing is taken from the snapshot that the manifest does not vouch for.

    Raises:
        CorruptedCorpusArtifactError: the file is not valid JSON, its own checksum disagrees with its
            content, a body does not hash to its recorded digest, its document set is not the manifest's, or
            it does not carry the manifest fingerprint it claims.
        CorpusError: the schema is unknown or a required field is missing or mis-typed.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        message = f"corpus snapshot at {str(path)!r} is not valid JSON: {exc}"
        raise CorruptedCorpusArtifactError(message) from exc
    if not isinstance(document, dict):
        message = f"corpus snapshot at {str(path)!r} must be a JSON object"
        raise CorruptedCorpusArtifactError(message)
    if document.get("schema_version") != _SNAPSHOT_SCHEMA:
        message = f"corpus snapshot at {str(path)!r} has an unsupported schema"
        raise CorruptedCorpusArtifactError(message)

    if "checksum" not in document:
        message = f"corpus snapshot at {str(path)!r} has no checksum field"
        raise CorruptedCorpusArtifactError(message)
    content = {key: value for key, value in document.items() if key != "checksum"}
    stored_checksum = document["checksum"]
    recomputed = sha256_hex(json.dumps(content, sort_keys=True, separators=(",", ":")))
    if recomputed != stored_checksum:
        message = (
            f"corpus snapshot at {str(path)!r} does not match its own checksum (stored "
            f"{str(stored_checksum)[:12]}…, computed {recomputed[:12]}…) — the file was altered or truncated"
        )
        raise CorruptedCorpusArtifactError(message)

    try:
        manifest_document = document["manifest"]
        entries = document["documents"]
        stored_fingerprint = document["manifest_fingerprint"]
        acquisition = document["acquisition"]
    except KeyError as exc:
        message = f"corpus snapshot at {str(path)!r} is missing a required field: {exc}"
        raise CorpusError(message) from exc

    texts: dict[str, str] = {}
    for entry in entries:
        path_name = str(entry["path"])
        if path_name in texts:
            message = f"corpus snapshot at {str(path)!r} carries document {path_name!r} twice"
            raise CorruptedCorpusArtifactError(message)
        texts[path_name] = str(entry["text"])

    manifest = CorpusManifest.from_document(manifest_document, texts=texts)
    recorded = {document.path: document for document in manifest.documents}
    unexpected = sorted(set(texts) - set(recorded))
    if unexpected:
        message = (
            f"corpus snapshot at {str(path)!r} carries {len(unexpected)} document(s) the manifest does not "
            f"record: {unexpected}"
        )
        raise CorruptedCorpusArtifactError(message)
    missing = sorted(set(recorded) - set(texts))
    if missing:
        message = (
            f"corpus snapshot at {str(path)!r} is missing {len(missing)} document(s) the manifest records: "
            f"{missing}"
        )
        raise CorruptedCorpusArtifactError(message)
    for entry in entries:
        path_name = str(entry["path"])
        document_digest = str(entry["sha256"])
        if document_digest != recorded[path_name].sha256:
            message = (
                f"corpus snapshot document {path_name!r} records digest {document_digest[:12]}… where the "
                f"manifest pins {recorded[path_name].sha256[:12]}…"
            )
            raise CorruptedCorpusArtifactError(message)

    if manifest.fingerprint != stored_fingerprint:
        message = (
            f"corpus snapshot at {str(path)!r} does not carry the manifest fingerprint it claims "
            f"({stored_fingerprint!r} stored, {manifest.fingerprint!r} rebuilt)"
        )
        raise CorruptedCorpusArtifactError(message)
    if not isinstance(acquisition, dict):
        message = f"corpus snapshot at {str(path)!r} has an acquisition block that is not an object"
        raise CorpusError(message)
    return CorpusSnapshot(
        manifest=manifest,
        checksum=stored_checksum,
        acquisition={str(key): str(value) for key, value in acquisition.items()},
    )
