"""A checksummed archive of a pinned corpus's raw bytes.

K-031's open half. A catalogue pins each source's URL and the SHA-256 of the bytes it was pinned with, and
`corpus verify` refuses a mismatch — but a URL is not an archive. Gutenberg edits its files: on 2026-11-02 a
re-fetch of the 430 pinned URLs found one whose upstream `Last-Modified` was a day after the retrieval, and the
original bytes were nowhere else. The corpus identity survived because the change fell outside the ingested text,
which was luck rather than a property of the design.

:func:`write_archive` makes the bytes themselves an artifact: one gzip stream, one member per pinned source, and a
manifest naming the catalogue it was built from. Three decisions are deliberate.

* **Deterministic bytes.** Members are ordered by slug, every tar header field that could carry a machine's clock or
  user is fixed, and the gzip header stores no file name and no mtime. Two builds of one raw directory agree on the
  archive digest, so the check is "do we hold the same archive", not "do we hold similar files".
* **Verification reads the payload.** :func:`verify_archive` decompresses every member and digests it. A check that
  only re-reads the manifest proves the manifest is self-consistent, which is not the claim anyone needs.
* **Nothing is replaced quietly.** An existing archive, manifest or extracted file is refused unless the caller asks,
  and extraction refuses a member whose name would land outside its destination.

Standard library only: the person who needs to verify a corpus is often one who has not installed this package's
training extras.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from chakaso.core.atomic import staged_write, write_text_atomically
from chakaso.core.hashing import sha256_hex
from chakaso.training.dataset_catalog import CatalogSource, DatasetCatalog
from chakaso.training.errors import CorpusError

__all__ = [
    "ArchiveManifest",
    "extract_archive",
    "manifest_from_document",
    "read_manifest",
    "verify_archive",
    "write_archive",
    "write_manifest",
]

#: Every header field that could otherwise carry the building machine is fixed, so the stream is a function of its
#: contents alone.
_FIXED_MTIME = 0
_FIXED_MODE = 0o644
_READ_CHUNK = 1 << 20

Reader = Callable[[CatalogSource], bytes]


def _ordered(catalog: DatasetCatalog) -> tuple[CatalogSource, ...]:
    """The catalogue's sources in slug order — the order an archive's members and a digest depend on."""
    return tuple(sorted(catalog.sources, key=lambda item: item.slug))


def _default_reader(item: CatalogSource, raw_dir: Path) -> bytes:
    return (raw_dir / f"{item.slug}.txt").read_bytes()


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """What an archive holds and how to check it.

    The manifest is the object a reader verifies against: the archive's own digest, the catalogue it was built from,
    and one entry per source carrying the digest the catalogue declares and the bytes the archive holds.
    """

    corpus: str
    version: str
    catalog_path: str
    catalog_sha256: str
    archive_path: str
    archive_sha256: str
    archive_bytes: int
    raw_bytes: int
    sources: tuple[dict[str, Any], ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "version": self.version,
            "catalog": {
                "path": self.catalog_path,
                "sha256": self.catalog_sha256,
                "sources": len(self.sources),
            },
            "archive": {
                "path": self.archive_path,
                "sha256": self.archive_sha256,
                "bytes": self.archive_bytes,
                "raw_bytes": self.raw_bytes,
                "format": "tar.gz, deterministic: members sorted by slug, headers fixed",
            },
            "sources": list(self.sources),
        }


def _digest_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_member(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = _FIXED_MODE
    info.mtime = _FIXED_MTIME
    info.type = tarfile.REGTYPE
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def read_manifest(path: Path) -> dict[str, Any]:
    """The manifest as written.

    Raises:
        CorpusError: the file holds JSON that is not an object.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        message = f"{path} is a JSON {type(document).__name__}, expected an object"
        raise CorpusError(message)
    return document


def manifest_from_document(document: Mapping[str, Any]) -> ArchiveManifest:
    """The manifest object a committed manifest file describes, for a caller that wants to check it.

    Raises:
        CorpusError: a field the writer always records is missing or carries the wrong type.
    """
    try:
        catalog = document["catalog"]
        archive = document["archive"]
        return ArchiveManifest(
            corpus=str(document["corpus"]),
            version=str(document["version"]),
            catalog_path=str(catalog["path"]),
            catalog_sha256=str(catalog["sha256"]),
            archive_path=str(archive["path"]),
            archive_sha256=str(archive["sha256"]),
            archive_bytes=int(archive["bytes"]),
            raw_bytes=int(archive["raw_bytes"]),
            sources=tuple(dict(entry) for entry in document["sources"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        message = f"the manifest is missing or mis-types a field the writer always records: {error}"
        raise CorpusError(message) from error


def write_manifest(manifest: ArchiveManifest, path: Path, *, allow_overwrite: bool = False) -> Path:
    """Write the manifest atomically, refusing to replace one that already exists unless asked.

    Raises:
        CorpusError: ``path`` exists and ``allow_overwrite`` was not passed.
    """
    destination = Path(path)
    if destination.exists() and not allow_overwrite:
        message = (
            f"{destination} already holds a manifest; a corpus artifact is not replaced unasked, and two "
            "manifests of one archive would leave a reader choosing between them"
        )
        raise CorpusError(message)
    write_text_atomically(
        json.dumps(manifest.to_document(), indent=2, sort_keys=True) + "\n", destination
    )
    return destination


def write_archive(
    catalog: DatasetCatalog,
    destination: Path,
    *,
    catalog_path: Path,
    readers: Mapping[str, bytes] | None = None,
    raw_dir: Path | None = None,
    allow_overwrite: bool = False,
) -> ArchiveManifest:
    """Archive the bytes of every source the catalogue pins, refusing a byte that no longer matches its digest.

    The bytes come from ``readers`` when the caller supplies them (a re-fetch, a memory-resident set, a test) or
    from ``raw_dir`` under the catalogue's own naming. Either way the rule is the same and is the reason this
    function exists: an archive of drifted bytes carries a false claim about what was pinned, and the claim is what
    people trust — a catalogue can be re-read, a bad archive is believed by whoever holds it.

    Raises:
        CorpusError: the destination exists and ``allow_overwrite`` was not passed, neither ``readers`` nor
            ``raw_dir`` was given, a pinned source has no bytes, a source's digest does not match the catalogue, or
            ``readers`` holds a slug the catalogue does not name.
    """
    target = Path(destination)
    if readers is None and raw_dir is None:
        message = "an archive needs its bytes: pass readers or raw_dir"
        raise CorpusError(message)
    ordered = _ordered(catalog)
    if readers is not None:
        unknown = sorted(set(readers) - {item.slug for item in ordered})
        if unknown:
            message = (
                f"{len(unknown)} source(s) are not in the catalogue {catalog.name!r}: {unknown[:5]}"
            )
            raise CorpusError(message)
        missing = [item.slug for item in ordered if item.slug not in readers]
        if missing:
            message = f"{len(missing)} pinned source(s) have no bytes: {sorted(missing)[:5]}"
            raise CorpusError(message)
        read: Reader = lambda item: readers[item.slug]  # noqa: E731 - the caller's own mapping
    else:
        assert raw_dir is not None  # narrowed by the check above
        absent = [item.slug for item in ordered if not (raw_dir / f"{item.slug}.txt").is_file()]
        if absent:
            message = (
                f"{str(raw_dir)!r} is missing {len(absent)} pinned source(s): {sorted(absent)[:5]}"
            )
            raise CorpusError(message)
        read = lambda item: _default_reader(item, raw_dir)  # noqa: E731
    if target.exists() and not allow_overwrite:
        message = (
            f"{target} already exists; a corpus archive is not replaced unasked, and two archives of one "
            "corpus would leave a reader choosing between them"
        )
        raise CorpusError(message)

    entries: list[dict[str, Any]] = []
    raw_total = 0
    with staged_write(target) as staged:
        with (
            staged.open("wb") as handle,
            gzip.GzipFile(
                filename="", mode="wb", compresslevel=9, mtime=_FIXED_MTIME, fileobj=handle
            ) as stream,
            tarfile.open(fileobj=stream, mode="w", format=tarfile.GNU_FORMAT) as tar,
        ):
            # The three close in reverse order: the tar flushes its members, the gzip stream writes its trailer,
            # the file closes last. The digest below is taken after all three, or it hashes a half-written archive.
            for item in ordered:
                payload = read(item)
                digest = sha256_hex(payload)
                if digest != item.sha256:
                    message = (
                        f"{item.slug}: the bytes digest {digest}, the catalogue declares "
                        f"{item.sha256}; an archive of drifted bytes is a forged provenance"
                    )
                    raise CorpusError(message)
                name = f"{item.slug}.txt"
                tar.addfile(_tar_member(name, len(payload)), io.BytesIO(payload))
                raw_total += len(payload)
                entries.append(
                    {
                        "slug": item.slug,
                        "url": item.url,
                        "sha256": digest,
                        "bytes": len(payload),
                        "member": name,
                    }
                )
        archive_digest = _digest_file(staged)
        archive_size = staged.stat().st_size
    return ArchiveManifest(
        corpus=catalog.name,
        version=catalog.version,
        catalog_path=str(catalog_path),
        catalog_sha256=_digest_file(catalog_path),
        archive_path=str(target),
        archive_sha256=archive_digest,
        archive_bytes=archive_size,
        raw_bytes=raw_total,
        sources=tuple(entries),
    )


def _members(path: Path) -> dict[str, bytes]:
    """Every regular member of the archive, by name. A non-regular entry is refused, not skipped."""
    out: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                message = (
                    f"{member.name}: an archive of source bytes holds no {member.type!r} entries"
                )
                raise CorpusError(message)
            reader = tar.extractfile(member)
            if reader is None:  # pragma: no cover - a regular file always yields a reader
                message = f"{member.name}: the archive lists it and does not hold it"
                raise CorpusError(message)
            out[member.name] = reader.read()
    return out


def verify_archive(
    catalog: DatasetCatalog, archive_path: Path, *, manifest_path: Path | None = None
) -> list[str]:
    """Every problem with an archive, as sentences. Empty means the archive *is* the corpus the catalogue pins.

    Each pinned source must appear exactly once and digest to what the catalogue declares, and the archive must hold
    nothing else: a member the catalogue does not name is an unpinned source, which is a corpus-identity problem even
    when every declared digest matches. With ``manifest_path``, the archive's bytes are also compared with the digest
    its manifest records.
    """
    problems: list[str] = []
    path = Path(archive_path)
    if not path.is_file():
        return [f"{path}: no such archive"]
    try:
        members = _members(path)
    except (OSError, tarfile.TarError) as error:
        return [f"{path}: cannot be read as a gzip tar stream: {error}"]
    expected = {f"{item.slug}.txt": item for item in catalog.sources}
    for name in sorted(set(members) - set(expected)):
        problems.append(f"{name}: in the archive, not in the catalogue {catalog.name!r}")
    for name in sorted(set(expected) - set(members)):
        problems.append(f"{name}: the catalogue pins it and the archive does not hold it")
    for name in sorted(expected):
        if name not in members:
            continue
        item = expected[name]
        digest = sha256_hex(members[name])
        if digest != item.sha256:
            problems.append(
                f"{item.slug}: the archive's bytes digest {digest}, the catalogue declares {item.sha256}"
            )
    if manifest_path is not None:
        document = read_manifest(manifest_path)
        recorded = str(document["archive"]["sha256"])
        observed = _digest_file(path)
        if recorded != observed:
            problems.append(
                f"{path.name}: the file digests to {observed}, its manifest records {recorded}"
            )
    return problems


def extract_archive(
    archive_path: Path, destination_dir: Path, *, allow_overwrite: bool = False
) -> tuple[str, ...]:
    """Write every member under ``destination_dir`` and return the names extracted.

    Refuses an existing file unless asked, and refuses a member whose name would land outside the destination: an
    archive is untrusted input, and a corpus archive that could write anywhere the reader has permission is not a
    corpus archive.

    Raises:
        CorpusError: a member name escapes the destination, or a destination file exists and ``allow_overwrite`` was
            not passed.
    """
    path, destination = Path(archive_path), Path(destination_dir)
    members = _members(path)
    names = sorted(members)
    for name in names:
        target = destination / name
        if target.parent != destination:
            message = (
                f"{name}: an archive member that resolves outside the destination is not extracted"
            )
            raise CorpusError(message)
        if target.exists() and not allow_overwrite:
            message = f"{target} already exists; extracted bytes are not replaced unasked"
            raise CorpusError(message)
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        with staged_write(destination / name) as staged:
            staged.write_bytes(members[name])
    return tuple(names)
