"""Tests for the archive that answers K-031: the pinned bytes, held somewhere a digest can prove.

The property is that an archive *is* the corpus its catalogue names, and that a reader with no network can tell.
Three fixtures do the work — three small sources with their digests declared, the operations a corpus maintainer
performs (build, verify, extract), and the ways each can go wrong.
"""

from __future__ import annotations

import io
import json
import tarfile
from hashlib import sha256
from pathlib import Path

import pytest

from chakaso.training.corpus_archive import (
    extract_archive,
    manifest_from_document,
    read_manifest,
    verify_archive,
    write_archive,
    write_manifest,
)
from chakaso.training.dataset_catalog import (
    CatalogSource,
    DatasetCatalog,
    load_catalog,
    write_catalog,
)
from chakaso.training.errors import CorpusError

_TEXTS = {"b-book": b"second book text\n", "a-book": b"first book text\n", "c-book": b"third\n"}
_DIGESTS = {slug: sha256(payload).hexdigest() for slug, payload in _TEXTS.items()}


def _catalog(slugs: tuple[str, ...] = ("a-book", "b-book", "c-book")) -> DatasetCatalog:
    return DatasetCatalog(
        name="fixture-corpus",
        version="1.0.0",
        licence="Public domain",
        source="A local fixture, not a provider",
        retrieved="2026-11-04",
        sources=tuple(
            CatalogSource(
                slug=slug,
                ebook_number=index + 1,
                url=f"https://example.org/{index + 1}",
                sha256=_DIGESTS[slug],
                title=f"Book {slug}",
            )
            for index, slug in enumerate(sorted(slugs))
        ),
    )


def _catalog_file(catalog: DatasetCatalog, tmp_path: Path) -> Path:
    """The catalogue as a pinned file, written by the catalog writer the project actually uses."""
    path = tmp_path / "catalog.toml"
    write_catalog(catalog, path, allow_overwrite=True)
    return path


def _archive(tmp_path: Path, catalog: DatasetCatalog | None = None) -> tuple[Path, Path]:
    subject = catalog or _catalog()
    destination = tmp_path / "corpus.tar.gz"
    manifest = write_archive(
        subject, destination, catalog_path=_catalog_file(subject, tmp_path), readers=dict(_TEXTS)
    )
    return destination, write_manifest(manifest, tmp_path / "manifest.json")


def test_an_archive_round_trips_to_the_bytes_the_catalogue_pinned(tmp_path: Path) -> None:
    archive, manifest_path = _archive(tmp_path)

    assert verify_archive(_catalog(), archive, manifest_path=manifest_path) == []
    document = read_manifest(manifest_path)
    assert document["archive"]["raw_bytes"] == sum(len(payload) for payload in _TEXTS.values())
    assert [entry["slug"] for entry in document["sources"]] == ["a-book", "b-book", "c-book"]

    out = tmp_path / "extracted"
    assert extract_archive(archive, out) == ("a-book.txt", "b-book.txt", "c-book.txt")
    assert (out / "b-book.txt").read_bytes() == _TEXTS["b-book"]


def test_two_builds_of_one_byte_set_agree_on_the_archive_digest(tmp_path: Path) -> None:
    catalog = _catalog()
    first = write_archive(
        catalog,
        tmp_path / "one.tar.gz",
        catalog_path=_catalog_file(catalog, tmp_path),
        readers=dict(_TEXTS),
    )
    second = write_archive(
        catalog,
        tmp_path / "two.tar.gz",
        catalog_path=_catalog_file(catalog, tmp_path),
        readers=dict(_TEXTS),
    )

    assert first.archive_sha256 == second.archive_sha256
    assert (tmp_path / "one.tar.gz").read_bytes() == (tmp_path / "two.tar.gz").read_bytes()


def test_the_member_order_is_the_slug_order_not_the_callers(tmp_path: Path) -> None:
    catalog = _catalog()
    shuffled = {"c-book": _TEXTS["c-book"], "b-book": _TEXTS["b-book"], "a-book": _TEXTS["a-book"]}
    subject = write_archive(
        catalog,
        tmp_path / "shuffled.tar.gz",
        catalog_path=_catalog_file(catalog, tmp_path),
        readers=shuffled,
    )
    ordered = write_archive(
        catalog,
        tmp_path / "ordered.tar.gz",
        catalog_path=_catalog_file(catalog, tmp_path),
        readers=dict(_TEXTS),
    )

    assert subject.archive_sha256 == ordered.archive_sha256


def test_an_archive_refuses_bytes_that_no_longer_match_the_pinned_digest(tmp_path: Path) -> None:
    catalog = _catalog()
    destination = tmp_path / "corpus.tar.gz"
    drifted = dict(_TEXTS)
    drifted["b-book"] = b"somebody edited the book\n"

    with pytest.raises(CorpusError, match="b-book: the bytes digest"):
        write_archive(
            catalog,
            destination,
            catalog_path=_catalog_file(catalog, tmp_path),
            readers=drifted,
        )
    assert not destination.exists()


def test_a_missing_pinned_source_is_reported_before_anything_is_written(tmp_path: Path) -> None:
    catalog = _catalog()
    destination = tmp_path / "corpus.tar.gz"
    readers = {slug: payload for slug, payload in _TEXTS.items() if slug != "a-book"}

    with pytest.raises(CorpusError, match="1 pinned source\\(s\\) have no bytes"):
        write_archive(
            catalog, destination, catalog_path=_catalog_file(catalog, tmp_path), readers=readers
        )
    assert not destination.exists()


def test_a_source_the_catalogue_does_not_name_is_refused(tmp_path: Path) -> None:
    catalog = _catalog(("a-book", "b-book"))
    with pytest.raises(CorpusError, match="not in the catalogue"):
        write_archive(
            catalog,
            tmp_path / "corpus.tar.gz",
            catalog_path=_catalog_file(catalog, tmp_path),
            readers=dict(_TEXTS),
        )


def test_an_archive_is_never_replaced_unasked(tmp_path: Path) -> None:
    catalog = _catalog()
    destination, _manifest = _archive(tmp_path, catalog)

    with pytest.raises(CorpusError, match="not replaced unasked"):
        write_archive(
            catalog,
            destination,
            catalog_path=_catalog_file(catalog, tmp_path),
            readers=dict(_TEXTS),
        )
    again = write_archive(
        catalog,
        destination,
        catalog_path=_catalog_file(catalog, tmp_path),
        readers=dict(_TEXTS),
        allow_overwrite=True,
    )
    assert again.archive_bytes == destination.stat().st_size


def test_the_raw_directory_is_a_second_way_to_supply_the_bytes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for slug, payload in _TEXTS.items():
        (raw / f"{slug}.txt").write_bytes(payload)
    catalog = _catalog()
    destination = tmp_path / "from-disk.tar.gz"

    from_readers = write_archive(
        catalog,
        tmp_path / "from-memory.tar.gz",
        catalog_path=_catalog_file(catalog, tmp_path),
        readers=dict(_TEXTS),
    )
    from_disk = write_archive(
        catalog, destination, catalog_path=_catalog_file(catalog, tmp_path), raw_dir=raw
    )
    assert from_disk.archive_sha256 == from_readers.archive_sha256

    (raw / "c-book.txt").unlink()
    with pytest.raises(CorpusError, match="is missing 1 pinned source"):
        write_archive(
            catalog,
            tmp_path / "absent.tar.gz",
            catalog_path=_catalog_file(catalog, tmp_path),
            raw_dir=raw,
        )


def test_an_archive_without_bytes_at_all_is_refused(tmp_path: Path) -> None:
    catalog = _catalog()
    with pytest.raises(CorpusError, match="needs its bytes"):
        write_archive(
            catalog, tmp_path / "none.tar.gz", catalog_path=_catalog_file(catalog, tmp_path)
        )


def test_verification_reads_the_payload_and_not_only_the_manifest(tmp_path: Path) -> None:
    archive, manifest_path = _archive(tmp_path)
    assert verify_archive(_catalog(), archive, manifest_path=manifest_path) == []

    tampered = tmp_path / "tampered.tar.gz"
    payload = archive.read_bytes()
    tampered.write_bytes(payload[: len(payload) // 2] + b"\x00" * 64)
    problems = verify_archive(_catalog(), tampered, manifest_path=manifest_path)
    assert problems
    assert any("cannot be read as a gzip tar stream" in line for line in problems)

    # A whole archive whose bytes differ: the manifest still records the old digest, so both halves report.
    partial = tmp_path / "partial.tar.gz"
    write_archive(
        _catalog(("a-book",)),
        partial,
        catalog_path=_catalog_file(_catalog(), tmp_path),
        readers={"a-book": _TEXTS["a-book"]},
    )
    problems = verify_archive(_catalog(), partial, manifest_path=manifest_path)
    # The two missing sources *and* the archive digest: the manifest belongs to the full archive, and a reader
    # checking a partial one against it is told both facts rather than one.
    assert [line.split(":")[0] for line in problems] == [
        "b-book.txt",
        "c-book.txt",
        "partial.tar.gz",
    ]
    assert problems[0] == "b-book.txt: the catalogue pins it and the archive does not hold it"


def test_an_unpinned_member_is_reported_even_when_every_declared_digest_matches(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "extra.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for slug, payload in _TEXTS.items():
            info = tarfile.TarInfo(f"{slug}.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        extra = b"nobody pinned this\n"
        info = tarfile.TarInfo("unpinned.txt")
        info.size = len(extra)
        tar.addfile(info, io.BytesIO(extra))

    assert verify_archive(_catalog(), archive) == [
        "unpinned.txt: in the archive, not in the catalogue 'fixture-corpus'"
    ]


def test_extraction_refuses_to_overwrite_and_to_escape(tmp_path: Path) -> None:
    archive, _manifest = _archive(tmp_path)
    out = tmp_path / "out"
    extract_archive(archive, out)
    (out / "a-book.txt").write_bytes(b"edited\n")

    with pytest.raises(CorpusError, match="already exists"):
        extract_archive(archive, out)
    assert extract_archive(archive, out, allow_overwrite=True) == (
        "a-book.txt",
        "b-book.txt",
        "c-book.txt",
    )
    assert (out / "a-book.txt").read_bytes() == _TEXTS["a-book"]

    hostile = tmp_path / "hostile.tar.gz"
    with tarfile.open(hostile, "w:gz") as tar:
        payload = b"plant\n"
        info = tarfile.TarInfo("../escaped.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(CorpusError, match="outside the destination"):
        extract_archive(hostile, tmp_path / "second")
    assert not (tmp_path / "escaped.txt").exists()


def test_the_manifest_is_a_document_a_reader_can_re_read(tmp_path: Path) -> None:
    _, path = _archive(tmp_path)
    document = read_manifest(path)
    assert document["corpus"] == "fixture-corpus"
    assert document["archive"]["format"].startswith("tar.gz, deterministic")
    assert (
        manifest_from_document(document).archive_sha256 == read_manifest(path)["archive"]["sha256"]
    )

    with pytest.raises(CorpusError, match="missing or mis-types"):
        manifest_from_document({"corpus": "x"})


def test_a_second_manifest_for_one_archive_is_refused_until_asked(tmp_path: Path) -> None:
    catalog = _catalog()
    subject = write_archive(
        catalog,
        tmp_path / "corpus.tar.gz",
        catalog_path=_catalog_file(catalog, tmp_path),
        readers=dict(_TEXTS),
    )
    path = write_manifest(subject, tmp_path / "manifest.json")

    with pytest.raises(CorpusError, match="already holds a manifest"):
        write_manifest(subject, path)
    write_manifest(subject, path, allow_overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8"))["catalog"]["sources"] == 3


def test_a_manifest_that_is_not_an_object_is_a_named_failure(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(CorpusError, match="expected an object"):
        read_manifest(path)


def test_a_loaded_catalogue_drives_the_same_checks(tmp_path: Path) -> None:
    archive, manifest_path = _archive(tmp_path)
    assert archive.is_file()
    pinned = _catalog_file(_catalog(), tmp_path)

    assert verify_archive(load_catalog(pinned), archive, manifest_path=manifest_path) == []
    assert verify_archive(load_catalog(pinned), tmp_path / "nope.tar.gz") == [
        f"{tmp_path / 'nope.tar.gz'}: no such archive"
    ]
