"""Recovering the exact corpus a recorded experiment pinned, from history rather than from the disk.

:mod:`chakaso.training.manifest` can prove that the files on disk are the corpus a manifest names, and it
refuses the moment they are not. That refusal is the right behaviour, and it has a consequence: after any
documentation edit, a recorded experiment can no longer be re-prepared from the working tree at all. The
drift guard cannot be weakened — a corpus that quietly becomes a different corpus invalidates every number
built on it — so the only way to reproduce a measured result is to read the bytes from the revision that
produced them.

This module does that, and only that:

* it asks a version-control history for one revision's tracked files,
* materializes them into a staging tree that nothing else reads,
* runs the **same** :func:`chakaso.training.manifest.verify_corpus` check a fresh preparation runs, so an
  altered, missing or newly added document fails here exactly as it fails in the working tree,
* and writes the result as a checksum-verified snapshot recording which revision the bytes came from.

Nothing is downloaded and nothing is written outside the staging tree and the snapshot path. Once a snapshot
exists the training path needs only it, so every later run works offline.

The history is read through a :class:`RevisionReader` seam so acquisition is testable without a repository;
:class:`GitRevisionReader` is the implementation this project uses. Standard library only, so this layer
runs in the dependency-free CI matrix.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from chakaso.training.errors import CorpusAcquisitionError, CorpusError
from chakaso.training.manifest import (
    CorpusManifest,
    CorpusSnapshot,
    read_manifest,
    read_snapshot,
    verify_corpus,
    write_snapshot,
)

__all__ = [
    "ACQUISITION_METHOD",
    "AcquisitionRecord",
    "GitRevisionReader",
    "RevisionReader",
    "is_hidden_path",
    "materialize_revision",
    "restore_corpus_snapshot",
]

#: Written into a snapshot's acquisition block, so a reader can tell which mechanism produced the bytes.
ACQUISITION_METHOD = "chakaso-git-revision"

#: A corpus archive is held in memory while it is unpacked, so it needs a bound. This one is far above any
#: corpus this project trains on and far below what would exhaust a CPU-only machine.
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    """How a snapshot's bytes were obtained — provenance for the artifact, not for the corpus.

    Attributes:
        revision: The revision the caller named.
        resolved_revision: The immutable commit object that name resolved to.
        documents: How many files the revision supplied.
        characters: How much normalized text they carried.
    """

    revision: str
    resolved_revision: str
    documents: int
    characters: int

    def as_mapping(self) -> dict[str, str]:
        return {
            "method": ACQUISITION_METHOD,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "documents": str(self.documents),
            "characters": str(self.characters),
        }


class RevisionReader(Protocol):
    """A read-only view of one repository's history, sized to exactly what corpus acquisition needs."""

    def resolve(self, revision: str) -> str:
        """Return the full commit object name ``revision`` names.

        Raises:
            CorpusAcquisitionError: the name does not resolve here.
        """
        ...

    def listing(self, resolved_revision: str) -> tuple[str, ...]:
        """Return every tracked path in ``resolved_revision``, as relative POSIX paths.

        Raises:
            CorpusAcquisitionError: the revision is not available here.
        """
        ...

    def archive(self, resolved_revision: str, paths: Sequence[str]) -> Mapping[str, bytes]:
        """Return the stored bytes of ``paths`` at ``resolved_revision``, keyed by the same path names.

        A path the revision does not carry is absent from the result; the caller decides whether that is an
        error.

        Raises:
            CorpusAcquisitionError: the revision is unavailable, or the archive cannot be read.
        """
        ...


def _validate_revision(revision: str) -> None:
    """Reject anything that is not a plain revision name before it reaches a subprocess.

    Arguments go to ``git`` as a list with no shell, so a metacharacter is not a command-injection risk. A
    value beginning with ``-`` is a problem of a different kind: ``git`` would read it as an option of its
    own. Refusing the shape is cheaper than reasoning about each subcommand's option parser, and the
    ``--end-of-options`` marker is sent as well because it is what the marker is for.
    """
    if (
        not revision
        or revision.startswith("-")
        or any(character.isspace() for character in revision)
    ):
        message = f"revision name {revision!r} is not a usable revision"
        raise CorpusAcquisitionError(message)


class GitRevisionReader:
    """Read one revision's tracked files out of a local Git repository.

    Args:
        repo_root: A directory inside the repository to read from.

    Raises:
        CorpusAcquisitionError: ``repo_root`` is not a directory.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        self._root = Path(repo_root) if repo_root is not None else Path.cwd()
        if not self._root.is_dir():
            message = f"repository root {str(self._root)!r} is not a directory"
            raise CorpusAcquisitionError(message)

    def _git(self, arguments: Sequence[str]) -> bytes:
        executable = shutil.which("git")
        if executable is None:
            message = (
                "corpus acquisition reads history and `git` is not on PATH; install Git, or load a "
                "snapshot that was already acquired"
            )
            raise CorpusAcquisitionError(message)
        result = subprocess.run(
            [executable, "-C", str(self._root), *arguments],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()
            message = f"`git {arguments[0]}` in {str(self._root)!r} failed: {detail}"
            raise CorpusAcquisitionError(message)
        return result.stdout

    def resolve(self, revision: str) -> str:
        _validate_revision(revision)
        try:
            payload = self._git(
                ["rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"]
            )
        except CorpusAcquisitionError as exc:
            message = (
                f"revision {revision!r} is not available in {str(self._root)!r} ({exc}). A clone made with "
                "--depth holds no history; fetch the revision, or use a snapshot acquired elsewhere."
            )
            raise CorpusAcquisitionError(message) from exc
        resolved = payload.decode("utf-8", "replace").strip()
        if len(resolved) != 40:
            message = f"revision {revision!r} did not resolve to a commit object name"
            raise CorpusAcquisitionError(message)
        return resolved

    def listing(self, resolved_revision: str) -> tuple[str, ...]:
        payload = self._git(
            ["ls-tree", "-r", "-z", "--name-only", "--end-of-options", resolved_revision]
        )
        return tuple(name.decode("utf-8") for name in payload.split(b"\0") if name)

    def archive(self, resolved_revision: str, paths: Sequence[str]) -> dict[str, bytes]:
        if not paths:
            return {}
        # `git archive` takes `<tree-ish> [<path>…>` with no `--` separator: everything after one is parsed
        # as a revision, which turns the first path into a "not a valid object name" failure. The
        # `--end-of-options` marker above is what keeps a path-shaped value from being read as an option.
        payload = self._git(
            ["archive", "--format=tar", "--end-of-options", resolved_revision, *paths]
        )
        if len(payload) > _MAX_ARCHIVE_BYTES:
            message = (
                f"the archive of {resolved_revision[:12]} is {len(payload)} bytes, above the "
                f"{_MAX_ARCHIVE_BYTES} byte bound this layer holds in memory"
            )
            raise CorpusAcquisitionError(message)
        extracted: dict[str, bytes] = {}
        try:
            with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    handle = tar.extractfile(member)
                    if handle is not None:
                        extracted[member.name] = handle.read()
        except (tarfile.TarError, OSError) as exc:
            message = f"the archive of {resolved_revision[:12]} could not be read: {exc}"
            raise CorpusAcquisitionError(message) from exc
        return extracted


def is_hidden_path(relative: str) -> bool:
    """Whether a relative path lies under a dot-directory or is itself a dot-file.

    The corpus layer never selects these (see :mod:`chakaso.training.manifest`), so acquisition does not
    materialize text no manifest is allowed to name.
    """
    return any(part.startswith(".") for part in PurePosixPath(relative).parts)


def _candidate_paths(manifest: CorpusManifest, listing: Sequence[str]) -> tuple[str, ...]:
    """The revision files that could join this corpus: a declared extension, or a named exact path.

    Deliberately a superset of what the include spec selects. The exact set is decided by
    :func:`chakaso.training.manifest.verify_corpus` over the materialized tree, because that is the tested
    rule; duplicating its glob and prefix logic here would create a second, weaker definition of a corpus.
    """
    extensions = set(manifest.extensions)
    exact = set(manifest.exact_paths)
    return tuple(
        path
        for path in sorted(set(listing))
        if not is_hidden_path(path) and (PurePosixPath(path).suffix in extensions or path in exact)
    )


def _materialize(
    resolved_revision: str, destination: Path, *, paths: Sequence[str], reader: RevisionReader
) -> tuple[str, ...]:
    """Write ``paths`` as the resolved revision stores them into ``destination``.

    Bytes are copied exactly as the history holds them: normalization, decoding and digesting belong to
    :mod:`chakaso.training.manifest`, so a materialized tree is checked by the same code a working tree is.
    """
    archive = reader.archive(resolved_revision, paths)
    absent = sorted(set(paths) - set(archive))
    if absent:
        message = f"revision {resolved_revision[:12]} does not carry {len(absent)} requested path(s): {absent}"
        raise CorpusAcquisitionError(message)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for relative in sorted(archive):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            message = f"revision {resolved_revision[:12]} stores {relative!r}, which is not a relative path"
            raise CorpusAcquisitionError(message)
        outlet = target.joinpath(*pure.parts)
        outlet.parent.mkdir(parents=True, exist_ok=True)
        outlet.write_bytes(archive[relative])
        written.append(relative)
    return tuple(written)


def materialize_revision(
    revision: str,
    destination: Path,
    *,
    paths: Sequence[str],
    reader: RevisionReader | None = None,
) -> tuple[str, ...]:
    """Write ``paths`` as ``revision`` stores them into ``destination``, and return the names written.

    Raises:
        CorpusAcquisitionError: history cannot be read, a named path is absent from the revision, or a
            stored path would write outside ``destination``.
    """
    active = reader if reader is not None else GitRevisionReader()
    return _materialize(active.resolve(revision), destination, paths=paths, reader=active)


def restore_corpus_snapshot(
    *,
    manifest: CorpusManifest | Path,
    revision: str,
    destination: Path,
    reader: RevisionReader | None = None,
    allow_overwrite: bool = False,
) -> CorpusSnapshot:
    """Recover the corpus a pinned manifest names from ``revision`` and write it as a snapshot.

    The manifest is the trust anchor and a revision is only a source of bytes: every document recovered is
    re-hashed and the whole materialized tree goes through :func:`chakaso.training.manifest.verify_corpus`,
    so a revision one documentation edit away from the pinned corpus is refused with the same drift error a
    stale working tree gets. Nothing here is a way to talk the guard into accepting different text.

    Args:
        manifest: The pinned manifest, or the path to one as written by ``save_manifest``.
        revision: Any revision name whose tree holds the corpus — a commit, a tag, or a branch that still
            contains it. Paths are read relative to the repository root, which must be the root the manifest
            was built from.
        destination: Where to write the snapshot. It belongs under ``data/``, which the artifact policy keeps
            out of version control.
        reader: The history to read; defaults to :class:`GitRevisionReader` over the current directory.
        allow_overwrite: Replace an existing snapshot deliberately, or not at all.

    Returns:
        The snapshot as it was **re-read from disk**, so the object handed back is the artifact a later run
        loads rather than the one this call assembled in memory.

    Raises:
        CorpusAcquisitionError: history cannot be read, or a pinned document is absent from the revision.
        CorpusDriftError: the revision's corpus is not the pinned one — a document changed, disappeared, or
            a new one joined the include spec.
        CorruptedCorpusArtifactError: the manifest file does not match its own checksum.
        CorpusError: the snapshot already exists and ``allow_overwrite`` was not passed.
    """
    pinned = manifest if isinstance(manifest, CorpusManifest) else read_manifest(Path(manifest))
    active = reader if reader is not None else GitRevisionReader()
    resolved = active.resolve(revision)
    candidates = _candidate_paths(pinned, active.listing(resolved))
    absent = sorted({document.path for document in pinned.documents} - set(candidates))
    if absent:
        message = (
            f"revision {resolved[:12]} holds none of the {len(absent)} document(s) the manifest "
            f"{pinned.name!r} pins: {absent}. A shallow clone has no history to read; fetch the revision "
            "first, or load a snapshot acquired while it was available."
        )
        raise CorpusAcquisitionError(message)

    with tempfile.TemporaryDirectory(prefix="chakaso-corpus-staging-") as staging:
        _materialize(resolved, Path(staging), paths=candidates, reader=active)
        verified = verify_corpus(pinned, Path(staging))
        checksum = write_snapshot(
            verified,
            Path(destination),
            acquisition=AcquisitionRecord(
                revision=revision,
                resolved_revision=resolved,
                documents=len(verified.documents),
                characters=verified.char_count,
            ).as_mapping(),
            allow_overwrite=allow_overwrite,
        )

    written = read_snapshot(Path(destination))
    if written.fingerprint != pinned.fingerprint:
        message = (
            f"the snapshot written for {pinned.name!r} carries fingerprint {written.fingerprint!r}, not the "
            f"pinned {pinned.fingerprint!r}"
        )
        raise CorpusError(message)
    if written.checksum != checksum:
        message = f"the snapshot at {str(destination)!r} does not match the digest this call wrote"
        raise CorpusError(message)
    return written
