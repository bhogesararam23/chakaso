"""Tests for recovering a pinned corpus from history.

The property under test is that a *revision* is only a source of bytes, never an excuse: the same manifest
checks that refuse a stale working tree must refuse a revision whose text differs, and accept one whose text
is exactly what was pinned. A test against a fake reader proves the plumbing; the interesting cases run
against a repository built for the purpose, with Git doing the reading.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from chakaso.training.errors import (
    CorpusAcquisitionError,
    CorpusDriftError,
    CorpusError,
)
from chakaso.training.manifest import (
    CorpusManifest,
    CorpusSnapshot,
    build_manifest,
    read_snapshot,
    save_manifest,
    write_snapshot,
)
from chakaso.training.snapshot import (
    ACQUISITION_METHOD,
    GitRevisionReader,
    materialize_revision,
    restore_corpus_snapshot,
)

# ------------------------------------------------------------------ fakes


class _History:
    """A stand-in revision reader: one dict of trees, no repository involved."""

    def __init__(self, trees: Mapping[str, Mapping[str, bytes]]) -> None:
        self._resolved = {name: name.ljust(40, "0") for name in trees}
        self._trees = {name.ljust(40, "0"): dict(tree) for name, tree in trees.items()}

    def resolve(self, revision: str) -> str:
        if revision not in self._resolved:
            message = f"unknown revision {revision!r}"
            raise CorpusAcquisitionError(message)
        return self._resolved[revision]

    def listing(self, resolved_revision: str) -> tuple[str, ...]:
        return tuple(sorted(self._tree(resolved_revision)))

    def archive(self, resolved_revision: str, paths: Sequence[str]) -> dict[str, bytes]:
        tree = self._tree(resolved_revision)
        return {path: tree[path] for path in paths if path in tree}

    def _tree(self, resolved_revision: str) -> dict[str, bytes]:
        if resolved_revision not in self._trees:
            message = f"unknown revision {resolved_revision!r}"
            raise CorpusAcquisitionError(message)
        return self._trees[resolved_revision]


def _tree(**overrides: bytes) -> dict[str, bytes]:
    base = {
        "docs/a.md": b"Alpha document.\n",
        "docs/b.md": b"Beta document, longer than alpha.\n",
        "LICENSE": b"Apache License 2.0\n",
    }
    base.update(overrides)
    return base


def _pinned(root: Path) -> CorpusManifest:
    return build_manifest(
        root,
        name="test-corpus",
        version="1.0.0",
        licence="Apache-2.0",
        source="built by tests",
        include=("docs/", "*.md"),
        extensions=(".md",),
        exact_paths=("LICENSE",),
    )


def _staged_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "docs").mkdir(parents=True)
    for name, text in (
        ("docs/a.md", "Alpha document.\n"),
        ("docs/b.md", "Beta document, longer than alpha.\n"),
        ("LICENSE", "Apache License 2.0\n"),
    ):
        (root / name).write_text(text, encoding="utf-8")
    return root


def _restore(
    manifest: CorpusManifest | Path,
    revision: str,
    destination: Path,
    history: _History,
    **kwargs: Any,
) -> CorpusSnapshot:
    return restore_corpus_snapshot(
        manifest=manifest, revision=revision, destination=destination, reader=history, **kwargs
    )


# ------------------------------------------------------------------ restore, against a fake history


def test_restoring_a_pinned_corpus_writes_a_snapshot_that_verifies(tmp_path: Path) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))
    destination = tmp_path / "data" / "corpus-snapshot.json"

    snapshot = _restore(manifest, "deadbeef", destination, _History({"deadbeef": _tree()}))

    assert snapshot.fingerprint == manifest.fingerprint
    assert snapshot.acquisition["method"] == ACQUISITION_METHOD
    assert snapshot.acquisition["resolved_revision"] == "deadbeef".ljust(40, "0")
    assert snapshot.acquisition["documents"] == "3"
    reloaded = read_snapshot(destination)
    assert reloaded.manifest.combined_text() == manifest.combined_text()
    assert reloaded.checksum == snapshot.checksum


def test_restore_refuses_a_revision_whose_text_was_edited(tmp_path: Path) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))
    history = _History({"cafe1234": _tree(**{"docs/a.md": b"Alpha document, rewritten later.\n"})})

    with pytest.raises(CorpusDriftError, match="content changed"):
        _restore(manifest, "cafe1234", tmp_path / "s.json", history)


def test_restore_refuses_a_revision_that_added_a_document_to_the_include_rule(
    tmp_path: Path,
) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))
    history = _History(
        {"cafe5678": _tree(**{"docs/c.md": b"A document that joined after the pin.\n"})}
    )

    with pytest.raises(CorpusDriftError, match=r"does not record: \['docs/c.md'\]"):
        _restore(manifest, "cafe5678", tmp_path / "s.json", history)


def test_restore_names_the_documents_a_revision_does_not_hold(tmp_path: Path) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))
    history = _History(
        {"beef0001": {"docs/a.md": b"Alpha document.\n", "LICENSE": b"Apache License 2.0\n"}}
    )

    with pytest.raises(CorpusAcquisitionError, match="holds none of the 1 document"):
        _restore(manifest, "beef0001", tmp_path / "s.json", history)


def test_restore_refuses_a_revision_that_cannot_be_resolved(tmp_path: Path) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))

    with pytest.raises(CorpusAcquisitionError, match="unknown revision"):
        _restore(manifest, "12345678", tmp_path / "s.json", _History({"deadbeef": _tree()}))


def test_restore_reads_a_manifest_from_disk_when_given_a_path(tmp_path: Path) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))
    manifest_path = tmp_path / "corpus-manifest.json"
    save_manifest(manifest, manifest_path)

    snapshot = _restore(
        manifest_path, "deadbeef", tmp_path / "s.json", _History({"deadbeef": _tree()})
    )

    assert snapshot.fingerprint == manifest.fingerprint


def test_restore_does_not_replace_an_existing_snapshot_unasked(tmp_path: Path) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))
    history = _History({"deadbeef": _tree()})
    destination = tmp_path / "s.json"
    _restore(manifest, "deadbeef", destination, history)

    with pytest.raises(CorpusError, match="not replaced unless"):
        _restore(manifest, "deadbeef", destination, history)

    again = _restore(manifest, "deadbeef", destination, history, allow_overwrite=True)
    assert again.fingerprint == manifest.fingerprint


def test_a_restored_snapshot_is_byte_deterministic_for_the_same_revision(tmp_path: Path) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    _restore(manifest, "deadbeef", first, _History({"deadbeef": _tree()}))
    _restore(manifest, "deadbeef", second, _History({"deadbeef": _tree()}))

    assert first.read_bytes() == second.read_bytes()


def test_a_snapshot_from_history_is_distinguishable_from_one_taken_from_the_tree(
    tmp_path: Path,
) -> None:
    """The historical bytes and the current ones must not be silently interchangeable artifacts."""
    root = _staged_corpus(tmp_path)
    manifest = _pinned(root)
    from_tree = tmp_path / "tree.json"
    from_history = tmp_path / "history.json"

    write_snapshot(_pinned(root), from_tree)
    _restore(manifest, "deadbeef", from_history, _History({"deadbeef": _tree()}))

    assert read_snapshot(from_tree).acquisition == {}
    assert read_snapshot(from_history).acquisition["revision"] == "deadbeef"


# ------------------------------------------------------------------ materialize


def test_materialize_refuses_a_path_that_would_escape_the_staging_tree(tmp_path: Path) -> None:
    history = _History({"deadbeef": {"../outside.md": b"not inside the corpus\n"}})

    with pytest.raises(CorpusAcquisitionError, match="not a relative path"):
        materialize_revision(
            "deadbeef", tmp_path / "staging", paths=("../outside.md",), reader=history
        )

    assert not (tmp_path.parent / "outside.md").exists()


def test_materialize_writes_the_stored_bytes_exactly(tmp_path: Path) -> None:
    history = _History(
        {"deadbeef": {"docs/a.md": b"Alpha document.\r\nCRLF preserved as stored.\r\n"}}
    )

    written = materialize_revision("deadbeef", tmp_path, paths=("docs/a.md",), reader=history)

    assert written == ("docs/a.md",)
    assert (
        tmp_path / "docs/a.md"
    ).read_bytes() == b"Alpha document.\r\nCRLF preserved as stored.\r\n"


def test_restore_surfaces_a_document_that_is_not_valid_utf8(tmp_path: Path) -> None:
    manifest = _pinned(_staged_corpus(tmp_path))
    history = _History({"deadbeef": _tree(**{"docs/a.md": b"\xff\xfe not text at all"})})

    with pytest.raises(CorpusError, match="not valid UTF-8"):
        _restore(manifest, "deadbeef", tmp_path / "s.json", history)


# ------------------------------------------------------------------ the real thing: Git


def _git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _git_bytes(cwd: Path, *arguments: str) -> bytes:
    """Run git without newline translation, because some assertions are about the line endings."""
    result = subprocess.run(["git", *arguments], cwd=str(cwd), capture_output=True, check=True)
    return result.stdout


@pytest.fixture()
def repository(tmp_path: Path) -> Path:
    """A throwaway repository: one commit of corpus text, then an edit that moves it."""
    if shutil.which("git") is None:  # pragma: no cover - every supported runner has git
        pytest.skip("corpus acquisition reads history, and git is not on PATH")
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Corpus test")
    _git(root, "config", "core.autocrlf", "false")
    (root / "docs/a.md").write_text("Alpha document.\n", encoding="utf-8")
    (root / "docs/b.md").write_text("Beta document, longer than alpha.\n", encoding="utf-8")
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (root / "notes.txt").write_text("tracked but not a corpus document\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "the corpus as pinned")
    (root / "docs/a.md").write_text("Alpha document, edited afterwards.\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "a documentation edit")
    return root


def _manifest_as_of(repository: Path, revision: str, destination: Path) -> CorpusManifest:
    """Build the manifest the pinned tree would have produced, by materializing that revision."""
    paths = tuple(
        line
        for line in _git(repository, "ls-tree", "-r", "--name-only", revision).splitlines()
        if line.endswith(".md") or line == "LICENSE"
    )
    root = destination / "as-recorded"
    materialize_revision(revision, root, paths=paths, reader=GitRevisionReader(repository))
    return _pinned(root)


def test_git_restores_the_pinned_corpus_from_the_revision_that_recorded_it(
    repository: Path, tmp_path: Path
) -> None:
    first = _git(repository, "rev-parse", "HEAD~1")
    manifest = _manifest_as_of(repository, first, tmp_path)

    snapshot = restore_corpus_snapshot(
        manifest=manifest,
        revision=first[:8],
        destination=tmp_path / "s.json",
        reader=GitRevisionReader(repository),
    )

    assert snapshot.fingerprint == manifest.fingerprint
    assert snapshot.acquisition["resolved_revision"] == first
    assert snapshot.acquisition["documents"] == "3"
    assert snapshot.manifest.char_count == manifest.char_count


def test_git_refuses_the_revision_that_edited_the_corpus(repository: Path, tmp_path: Path) -> None:
    first = _git(repository, "rev-parse", "HEAD~1")
    head = _git(repository, "rev-parse", "HEAD")
    manifest = _manifest_as_of(repository, first, tmp_path)

    with pytest.raises(CorpusDriftError, match="content changed"):
        restore_corpus_snapshot(
            manifest=manifest,
            revision=head,
            destination=tmp_path / "s.json",
            reader=GitRevisionReader(repository),
        )


def test_git_refuses_a_revision_where_a_new_document_joined_the_rule(
    repository: Path, tmp_path: Path
) -> None:
    first = _git(repository, "rev-parse", "HEAD~1")
    manifest = _manifest_as_of(repository, first, tmp_path)
    (repository / "docs/c.md").write_text("A third document.\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "another document joins the include rule")

    with pytest.raises(CorpusDriftError, match=r"does not record: \['docs/c.md'\]"):
        restore_corpus_snapshot(
            manifest=manifest,
            revision="HEAD",
            destination=tmp_path / "s.json",
            reader=GitRevisionReader(repository),
        )


def test_git_refuses_a_revision_that_deleted_a_pinned_document(
    repository: Path, tmp_path: Path
) -> None:
    first = _git(repository, "rev-parse", "HEAD~1")
    manifest = _manifest_as_of(repository, first, tmp_path)
    _git(repository, "rm", "-q", "docs/b.md")
    _git(repository, "commit", "-q", "-m", "a pinned document disappears")

    with pytest.raises(CorpusAcquisitionError, match="holds none of the 1 document"):
        restore_corpus_snapshot(
            manifest=manifest,
            revision="HEAD",
            destination=tmp_path / "s.json",
            reader=GitRevisionReader(repository),
        )


def test_git_ignores_tracked_files_the_include_rule_does_not_select(
    repository: Path, tmp_path: Path
) -> None:
    first = _git(repository, "rev-parse", "HEAD~1")
    manifest = _manifest_as_of(repository, first, tmp_path)
    (repository / "notes.txt").write_text("edited, but never a corpus document\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "an edit outside the corpus")

    snapshot = restore_corpus_snapshot(
        manifest=manifest,
        revision=first,
        destination=tmp_path / "s.json",
        reader=GitRevisionReader(repository),
    )

    assert snapshot.fingerprint == manifest.fingerprint
    assert [document.path for document in snapshot.manifest.documents] == [
        "LICENSE",
        "docs/a.md",
        "docs/b.md",
    ]


def test_a_document_stored_with_crlf_is_still_the_corpus_the_pin_normalized(
    repository: Path, tmp_path: Path
) -> None:
    """A corpus identity must not depend on how a checkout stored the line endings.

    The pin is taken over normalized text, so a revision whose *blob* carries CRLF for the same characters
    is the same corpus — and restoring it has to succeed rather than report drift. This is the case an
    LF-only fingerprint exists to handle, so it is tested with CRLF actually inside the object store.
    """
    head = _git(repository, "rev-parse", "HEAD")
    manifest = _manifest_as_of(repository, head, tmp_path)

    (repository / ".gitattributes").write_text("docs/b.md -text\n", encoding="utf-8")
    (repository / "docs/b.md").write_bytes(b"Beta document, longer than alpha.\r\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "the same text, stored with a CRLF ending")
    stored = _git_bytes(repository, "show", "HEAD:docs/b.md")
    assert b"\r\n" in stored  # the blob really carries CRLF, which is what the pin normalizes away
    assert _git(repository, "rev-parse", "HEAD") != head

    snapshot = restore_corpus_snapshot(
        manifest=manifest,
        revision="HEAD",
        destination=tmp_path / "s.json",
        reader=GitRevisionReader(repository),
    )

    assert snapshot.fingerprint == manifest.fingerprint
    body = next(
        document.text for document in snapshot.manifest.documents if document.path == "docs/b.md"
    )
    assert "\r" not in body
    assert body == "Beta document, longer than alpha.\n"


def test_git_names_an_unusable_revision_shape(repository: Path) -> None:
    reader = GitRevisionReader(repository)

    for revision in ("", "-x", "HEAD~1 ", "a b"):
        with pytest.raises(CorpusAcquisitionError, match="not a usable revision"):
            reader.resolve(revision)


def test_git_names_a_revision_the_local_history_does_not_hold(repository: Path) -> None:
    reader = GitRevisionReader(repository)

    with pytest.raises(CorpusAcquisitionError, match="is not available in"):
        reader.resolve("0" * 40)


def test_a_restored_snapshot_loads_in_a_fresh_process(repository: Path, tmp_path: Path) -> None:
    """Reproduction means a later process can read the artifact, so this one is a separate interpreter."""
    first = _git(repository, "rev-parse", "HEAD~1")
    manifest = _manifest_as_of(repository, first, tmp_path)
    destination = tmp_path / "s.json"
    restore_corpus_snapshot(
        manifest=manifest,
        revision=first,
        destination=destination,
        reader=GitRevisionReader(repository),
    )
    program = (
        "from pathlib import Path\n"
        "from chakaso.training.manifest import read_snapshot\n"
        "snapshot = read_snapshot(Path(input()))\n"
        "print(snapshot.fingerprint, snapshot.manifest.char_count,\n"
        "      ','.join(sorted(snapshot.acquisition)))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        input=str(destination),
        capture_output=True,
        text=True,
        check=True,
    )

    fields = result.stdout.split()
    assert fields[0] == manifest.fingerprint
    assert fields[1] == str(manifest.char_count)
    assert "resolved_revision" in fields[2]
