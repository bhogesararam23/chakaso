"""Tests for the corpus manifest: provenance that refuses to drift.

The property under test is not "a JSON file round-trips" — it is that a recorded corpus cannot quietly
become a different corpus. Every failure path here is a way a saved training result could otherwise be
restated against text its model never saw, so the tests are written from the drift outward: change the
files and the load must say which ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chakaso.training.errors import (
    CorpusDriftError,
    CorpusError,
    CorruptedCorpusArtifactError,
)
from chakaso.training.manifest import (
    CorpusDocument,
    CorpusManifest,
    build_manifest,
    discover_documents,
    normalize_text,
    read_manifest,
    read_snapshot,
    save_manifest,
    verify_corpus,
    write_snapshot,
)


def _corpus(root: Path) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "a.md").write_text("Alpha document.\n", encoding="utf-8")
    (root / "docs" / "b.md").write_text("Beta document, longer than alpha.\n", encoding="utf-8")
    (root / "README.md").write_text("Top-level readme.\n", encoding="utf-8")
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")


def _build(root: Path, **overrides: object) -> CorpusManifest:
    settings: dict[str, object] = {
        "name": "test-corpus",
        "version": "1.0.0",
        "licence": "Apache-2.0",
        "source": "built by tests",
        "include": ("docs/", "*.md"),
        "extensions": (".md",),
        "exact_paths": ("LICENSE",),
    }
    settings.update(overrides)
    return build_manifest(root, **settings)  # type: ignore[arg-type]


# ------------------------------------------------------------------ discovery


def test_build_manifest_pins_every_selected_file_with_its_digest(tmp_path: Path) -> None:
    _corpus(tmp_path)
    manifest = _build(tmp_path)

    paths = [document.path for document in manifest.documents]
    assert paths == ["LICENSE", "README.md", "docs/a.md", "docs/b.md"], paths
    assert all(len(document.sha256) == 64 for document in manifest.documents)
    assert manifest.char_count == sum(len(document.text) for document in manifest.documents)
    assert manifest.byte_count == manifest.char_count  # these fixtures are pure ASCII


def test_discovery_is_deterministic_and_identical_across_rebuilds(tmp_path: Path) -> None:
    _corpus(tmp_path)
    first = _build(tmp_path)
    second = _build(tmp_path)

    assert first.fingerprint == second.fingerprint
    assert [document.path for document in first.documents] == [
        document.path for document in second.documents
    ]


def test_hidden_directories_are_never_selected(tmp_path: Path) -> None:
    _corpus(tmp_path)
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "dependency.md").write_text(
        "not the project's text\n", encoding="utf-8"
    )

    manifest = _build(tmp_path)

    assert not any(".venv" in document.path for document in manifest.documents)


def test_an_extensionless_file_is_only_selected_when_named_exactly(tmp_path: Path) -> None:
    _corpus(tmp_path)

    manifest = _build(tmp_path, exact_paths=())

    assert "LICENSE" not in [document.path for document in manifest.documents]


def test_discovery_refuses_to_select_a_missing_exact_path(tmp_path: Path) -> None:
    _corpus(tmp_path)

    with pytest.raises(CorpusError, match="not a file"):
        discover_documents(tmp_path, include=("docs/",), exact_paths=("NOTICE",))


def test_discovery_refuses_when_the_spec_selects_nothing(tmp_path: Path) -> None:
    _corpus(tmp_path)

    with pytest.raises(CorpusError, match="selected no files"):
        discover_documents(tmp_path, include=("nowhere/",))


def test_discovery_refuses_an_oversized_document(tmp_path: Path) -> None:
    _corpus(tmp_path)

    with pytest.raises(CorpusError, match="above the declared bound"):
        discover_documents(tmp_path, include=("docs/",), max_document_bytes=8)


def test_an_empty_include_set_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path)

    with pytest.raises(CorpusError, match="at least one include entry"):
        discover_documents(tmp_path, include=())


def test_a_nonexistent_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="not a directory"):
        discover_documents(tmp_path / "absent", include=("docs/",))


def test_non_utf8_bytes_are_refused_by_name(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "binary.md").write_bytes(b"\xff\xfe not utf-8 \x80")

    with pytest.raises(CorpusError, match="not valid UTF-8"):
        _build(tmp_path, include=("docs/",), exact_paths=())


# ------------------------------------------------------------------ normalization


def test_line_endings_are_normalized_before_hashing(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "crlf.md").write_bytes(b"one\r\ntwo\rthree\n")

    document = _build(tmp_path, include=("docs/",), exact_paths=()).documents[0]

    assert document.text == "one\ntwo\nthree\n"
    assert normalize_text("a\r\nb\rc\n") == "a\nb\nc\n"


def test_a_crlf_and_an_lf_copy_of_one_document_share_one_digest(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "lf.md").write_bytes(b"same text\nsecond line\n")
    lf = _build(tmp_path, include=("docs/",), exact_paths=())
    (tmp_path / "docs" / "lf.md").write_bytes(b"same text\r\nsecond line\r\n")
    crlf = _build(tmp_path, include=("docs/",), exact_paths=())

    assert lf.fingerprint == crlf.fingerprint


def test_a_corpus_of_no_text_at_all_is_refused(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "empty.md").write_text("", encoding="utf-8")

    with pytest.raises(CorpusDriftError, match="not one character"):
        _build(tmp_path, include=("docs/",), exact_paths=())


# ------------------------------------------------------------------ the record types


def test_a_document_refuses_a_digest_that_is_not_sha256_hex() -> None:
    with pytest.raises(CorpusError, match="not a lowercase sha256"):
        CorpusDocument(path="a.md", sha256="xyz", byte_count=2, char_count=2, text="ab")


def test_a_document_refuses_a_hidden_path() -> None:
    digest = "0" * 64
    with pytest.raises(CorpusError, match="non-hidden relative path"):
        CorpusDocument(path=".venv/a.md", sha256=digest, byte_count=1, char_count=1, text="a")


def test_a_manifest_refuses_duplicate_and_unsorted_documents() -> None:
    _document = CorpusDocument(path="b.md", sha256="1" * 64, byte_count=1, char_count=1, text="b")
    other = CorpusDocument(path="a.md", sha256="2" * 64, byte_count=1, char_count=1, text="a")

    with pytest.raises(CorpusError, match="records a path twice"):
        CorpusManifest(
            name="n",
            version="1",
            licence="Apache-2.0",
            source="s",
            include=("docs/",),
            extensions=(".md",),
            exact_paths=(),
            max_document_bytes=10,
            normalization="crlf-and-cr-to-lf",
            documents=(_document, _document),
        )
    with pytest.raises(CorpusError, match="not sorted by path"):
        CorpusManifest(
            name="n",
            version="1",
            licence="Apache-2.0",
            source="s",
            include=("docs/",),
            extensions=(".md",),
            exact_paths=(),
            max_document_bytes=10,
            normalization="crlf-and-cr-to-lf",
            documents=(_document, other),
        )


def test_a_manifest_refuses_an_empty_provenance_field() -> None:
    document = CorpusDocument(path="a.md", sha256="1" * 64, byte_count=1, char_count=1, text="a")
    with pytest.raises(CorpusError, match="licence"):
        CorpusManifest(
            name="n",
            version="1",
            licence="  ",
            source="s",
            include=("docs/",),
            extensions=(".md",),
            exact_paths=(),
            max_document_bytes=10,
            normalization="crlf-and-cr-to-lf",
            documents=(document,),
        )


def test_fingerprint_moves_when_content_moves_and_when_a_file_joins(tmp_path: Path) -> None:
    _corpus(tmp_path)
    before = _build(tmp_path)

    edited = _corpus_and_rebuild(tmp_path, edit=True)
    assert edited.fingerprint != before.fingerprint

    grown = _corpus_and_rebuild(tmp_path, add=True)
    assert grown.fingerprint != before.fingerprint


def _corpus_and_rebuild(root: Path, *, edit: bool = False, add: bool = False) -> CorpusManifest:
    if edit:
        (root / "docs" / "a.md").write_text("Alpha document, edited.\n", encoding="utf-8")
    if add:
        (root / "docs" / "c.md").write_text("A third document.\n", encoding="utf-8")
    return _build(root)


# ------------------------------------------------------------------ manifest save/load


def test_manifest_round_trips_without_carrying_text(tmp_path: Path) -> None:
    _corpus(tmp_path)
    manifest = _build(tmp_path)
    target = tmp_path / "data" / "manifest.json"

    save_manifest(manifest, target)
    loaded = read_manifest(target)

    assert loaded.fingerprint == manifest.fingerprint
    assert [document.path for document in loaded.documents] == [
        document.path for document in manifest.documents
    ]
    assert all(document.text == "" for document in loaded.documents)
    assert "text" not in target.read_text(encoding="utf-8")


def test_a_tampered_manifest_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path)
    target = tmp_path / "manifest.json"
    save_manifest(_build(tmp_path), target)
    document = json.loads(target.read_text(encoding="utf-8"))
    document["version"] = "9.9.9"
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="does not match its content"):
        read_manifest(target)


def test_a_manifest_without_a_checksum_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path)
    document = _build(tmp_path).to_document()
    document.pop("checksum")

    with pytest.raises(CorruptedCorpusArtifactError, match="no checksum"):
        CorpusManifest.from_document(document)


def test_an_unknown_manifest_schema_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path)
    document = _build(tmp_path).to_document()
    # Re-checksum the record so the schema, not the checksum, is what fails: a well-formed record from a
    # format this code does not understand must be reported as incompatible, not corrupt.
    document["schema_version"] = "chakaso-corpus-manifest/99"
    document["checksum"] = _rechecksum(document)

    with pytest.raises(CorpusError, match="unsupported corpus manifest schema"):
        CorpusManifest.from_document(document)


def _rechecksum(document: dict[str, object]) -> str:
    content = {key: value for key, value in document.items() if key != "checksum"}
    from chakaso.core.hashing import sha256_hex

    return sha256_hex(json.dumps(content, sort_keys=True, separators=(",", ":")))


def test_a_manifest_missing_a_required_field_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path)
    document = _build(tmp_path).to_document()
    document.pop("licence")
    document["checksum"] = _rechecksum(document)

    with pytest.raises(CorpusError, match="missing or mis-typed"):
        CorpusManifest.from_document(document)


def test_a_manifest_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("{not json", encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="not valid JSON"):
        read_manifest(target)


def test_a_manifest_file_that_is_json_but_not_an_object_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="must be a JSON object"):
        read_manifest(target)


# ------------------------------------------------------------------ drift detection


def test_verify_corpus_returns_the_text_when_nothing_changed(tmp_path: Path) -> None:
    _corpus(tmp_path)
    manifest = _build(tmp_path)
    target = tmp_path / "manifest.json"
    save_manifest(manifest, target)
    pinned = read_manifest(target)

    verified = verify_corpus(pinned, tmp_path)

    assert verified.fingerprint == manifest.fingerprint
    assert verified.combined_text() == manifest.combined_text()


def test_verify_corpus_names_an_edited_document(tmp_path: Path) -> None:
    _corpus(tmp_path)
    pinned = read_manifest_of(_build(tmp_path), tmp_path)
    (tmp_path / "docs" / "a.md").write_text("Alpha, rewritten.\n", encoding="utf-8")

    with pytest.raises(CorpusDriftError, match=r"content changed: \['docs/a.md'\]"):
        verify_corpus(pinned, tmp_path)


def test_verify_corpus_names_a_deleted_document(tmp_path: Path) -> None:
    _corpus(tmp_path)
    pinned = read_manifest_of(_build(tmp_path), tmp_path)
    (tmp_path / "docs" / "b.md").unlink()

    with pytest.raises(CorpusDriftError, match=r"recorded files absent: \['docs/b.md'\]"):
        verify_corpus(pinned, tmp_path)


def test_verify_corpus_names_a_document_that_newly_joins(tmp_path: Path) -> None:
    _corpus(tmp_path)
    pinned = read_manifest_of(_build(tmp_path), tmp_path)
    (tmp_path / "docs" / "c.md").write_text("Unannounced new document.\n", encoding="utf-8")

    with pytest.raises(CorpusDriftError, match=r"does not record: \['docs/c.md'\]"):
        verify_corpus(pinned, tmp_path)


def read_manifest_of(manifest: CorpusManifest, root: Path) -> CorpusManifest:
    target = root / "manifest.json"
    save_manifest(manifest, target)
    return read_manifest(target)


# ------------------------------------------------------------------ snapshots


def test_snapshot_round_trips_and_preserves_the_fingerprint(tmp_path: Path) -> None:
    _corpus(tmp_path)
    manifest = _build(tmp_path)
    snapshot = tmp_path / "data" / "corpus.json"

    digest = write_snapshot(manifest, snapshot)
    loaded = read_snapshot(snapshot)

    assert digest and snapshot.is_file()
    assert loaded.fingerprint == manifest.fingerprint
    assert loaded.combined_text() == manifest.combined_text()
    assert [document.text for document in loaded.documents] == [
        document.text for document in manifest.documents
    ]


def test_a_snapshot_refuses_a_manifest_with_no_text(tmp_path: Path) -> None:
    _corpus(tmp_path)
    pinned = read_manifest_of(_build(tmp_path), tmp_path)

    with pytest.raises(CorpusError, match="no document text"):
        write_snapshot(pinned, tmp_path / "snapshot.json")


def test_altered_text_in_a_snapshot_is_caught_by_the_digest(tmp_path: Path) -> None:
    _corpus(tmp_path)
    manifest = _build(tmp_path)
    snapshot = tmp_path / "snapshot.json"
    write_snapshot(manifest, snapshot)
    document = json.loads(snapshot.read_text(encoding="utf-8"))
    document["documents"][0]["text"] = "swapped in by someone else\n"
    snapshot.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="does not hash to the digest"):
        read_snapshot(snapshot)


def test_a_snapshot_carrying_a_foreign_manifest_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path)
    manifest = _build(tmp_path)
    snapshot = tmp_path / "snapshot.json"
    write_snapshot(manifest, snapshot)
    document = json.loads(snapshot.read_text(encoding="utf-8"))
    document["manifest_fingerprint"] = "0" * 16
    snapshot.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="manifest fingerprint"):
        read_snapshot(snapshot)


def test_a_snapshot_of_unknown_schema_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "snapshot.json"
    target.write_text('{"schema_version": "chakaso-corpus-snapshot/9"}', encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="unsupported schema"):
        read_snapshot(target)


def test_a_snapshot_missing_a_required_field_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path)
    snapshot = tmp_path / "snapshot.json"
    write_snapshot(_build(tmp_path), snapshot)
    document = json.loads(snapshot.read_text(encoding="utf-8"))
    document.pop("manifest")
    snapshot.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CorpusError, match="missing a required field"):
        read_snapshot(snapshot)


# ------------------------------------------------------------------ convenience views


def test_text_units_carry_content_derived_identifiers(tmp_path: Path) -> None:
    _corpus(tmp_path)
    manifest = _build(tmp_path)

    units = manifest.text_units()

    assert len(units) == len(manifest.documents)
    assert [unit.identifier for unit in units] == [
        document.identifier for document in manifest.documents
    ]
    assert all(
        unit.identifier.startswith(document.path + "@")
        for unit, document in zip(units, manifest.documents, strict=True)
    )
    assert len({unit.identifier for unit in units}) == len(units)


def test_duplicate_content_is_reported_by_path(tmp_path: Path) -> None:
    _corpus(tmp_path)
    (tmp_path / "docs" / "copy.md").write_text(
        (tmp_path / "docs" / "a.md").read_text(encoding="utf-8"), encoding="utf-8"
    )

    manifest = _build(tmp_path)

    assert manifest.duplicate_paths == (("docs/a.md", "docs/copy.md"),)


def test_a_manifest_supplied_with_text_validates_it_against_the_record(tmp_path: Path) -> None:
    _corpus(tmp_path)
    manifest = _build(tmp_path)
    document = manifest.to_document()

    with pytest.raises(CorruptedCorpusArtifactError, match="does not hash to the digest"):
        CorpusManifest.from_document(
            document, texts={entry.path: "different text\n" for entry in manifest.documents}
        )
