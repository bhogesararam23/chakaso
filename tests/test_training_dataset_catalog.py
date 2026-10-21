"""Tests for the declared dataset catalogue: what may become a corpus, and did it arrive unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chakaso.core.hashing import sha256_hex
from chakaso.training import dataset_catalog
from chakaso.training.dataset_catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogSource,
    DatasetCatalog,
    gutenberg_url,
    load_catalog,
    pin_catalog,
    write_catalog,
)
from chakaso.training.errors import CorpusAcquisitionError, CorpusError
from chakaso.training.manifest import read_snapshot

_BODY = (
    """First paragraph of the book, with enough words in it to chunk.\n\n"""
    """Second paragraph, also long enough to be a document of its own under a small bound.\n"""
)


def _raw(book: str) -> str:
    return (
        f"Title: {book}\nAuthor: A Person\n\n"
        f"*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\n{_BODY}"
        f"*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
    )


def _source(slug: str, number: int, raw_dir: Path, *, title: str | None = None) -> CatalogSource:
    path = raw_dir / f"{slug}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_raw(slug).encode("utf-8"))
    return CatalogSource(
        slug=slug,
        ebook_number=number,
        url=gutenberg_url(number),
        sha256=sha256_hex(path.read_bytes()),
        title=slug if title is None else title,
        author="A Person",
    )


def _book_title(slug: str) -> str:
    return f"The book called {slug}"


def _book_title(slug: str) -> str:
    return f"The book called {slug}"


def _write(tmp_path: Path, catalog: DatasetCatalog) -> Path:
    path = tmp_path / "catalog.toml"
    write_catalog(catalog, path, allow_overwrite=True)
    return path


def _catalog(raw_dir: Path) -> DatasetCatalog:
    return DatasetCatalog(
        name="test-corpus",
        version="1.0.0",
        licence="Public domain in the United States",
        source="Project Gutenberg",
        retrieved="2026-01-05",
        sources=(_source("pg1", 1, raw_dir), _source("pg2", 2, raw_dir)),
        max_characters_per_document=48,
    )


def test_a_source_declaration_refuses_a_slug_that_is_not_a_path_prefix() -> None:
    with pytest.raises(CorpusError, match="path prefix"):
        CatalogSource(slug="../escape", ebook_number=1, url="https://x/y", sha256="0" * 64)


def test_a_source_declaration_refuses_a_digest_that_is_not_a_digest() -> None:
    with pytest.raises(CorpusError, match="64 lowercase hex"):
        CatalogSource(slug="pg1", ebook_number=1, url="https://x/y", sha256="abc123")


def test_a_source_declaration_refuses_a_url_with_no_scheme() -> None:
    with pytest.raises(CorpusError, match="no scheme"):
        CatalogSource(slug="pg1", ebook_number=1, url="www.gutenberg.org", sha256="0" * 64)


def test_a_source_declaration_refuses_an_empty_licence_override() -> None:
    with pytest.raises(CorpusError, match="empty licence"):
        CatalogSource(slug="pg1", ebook_number=1, url="https://x/y", sha256="0" * 64, licence="   ")


def test_a_catalog_refuses_to_declare_the_same_source_twice(tmp_path: Path) -> None:
    first = _source("pg1", 1, tmp_path)
    with pytest.raises(CorpusError, match="same slug"):
        DatasetCatalog(
            name="x",
            version="1",
            licence="l",
            source="s",
            retrieved="2026-01-05",
            sources=(first, first),
        )


def test_a_catalog_refuses_a_retrieved_date_it_cannot_order(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="ISO"):
        DatasetCatalog(
            name="x",
            version="1",
            licence="l",
            source="s",
            retrieved="January 2026",
            sources=(_source("pg1", 1, tmp_path),),
        )


def test_a_catalog_with_no_sources_is_refused() -> None:
    with pytest.raises(CorpusError, match="no sources"):
        DatasetCatalog(
            name="x", version="1", licence="l", source="s", retrieved="2026-01-05", sources=()
        )


def test_a_catalog_round_trips_through_toml(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "raw")
    path = tmp_path / "catalog.toml"
    write_catalog(catalog, path)

    loaded = load_catalog(path)

    assert loaded == catalog
    assert "Public domain in the United States" in path.read_text(encoding="utf-8")


def test_a_catalog_file_refuses_a_schema_it_does_not_know(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('schema_version = "chakaso-dataset-catalog/9"\n', encoding="utf-8")

    with pytest.raises(CorpusError, match="unsupported dataset catalog"):
        load_catalog(path)


def test_a_catalog_source_with_an_unknown_field_is_refused(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "raw")
    document = json.loads(json.dumps(catalog.as_dict()))
    document["sources"][0]["mirrors"] = ["rsync://elsewhere"]

    with pytest.raises(CorpusError, match="not a mapping"):
        DatasetCatalog.from_mapping(document)


def test_a_catalog_is_not_written_over_unasked(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "raw")
    path = tmp_path / "catalog.toml"
    write_catalog(catalog, path)

    with pytest.raises(CorpusError, match="allow_overwrite"):
        write_catalog(catalog, path)

    write_catalog(catalog, path, allow_overwrite=True)
    assert load_catalog(path) == catalog


def test_verification_names_the_files_that_are_absent(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    (raw / "pg2.txt").unlink()

    assert catalog.missing_raw_files(raw) == ("pg2",)
    with pytest.raises(CorpusAcquisitionError, match="missing 1 declared source file"):
        catalog.verify_raw(raw)


def test_verification_refuses_bytes_that_are_not_what_was_declared(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    (raw / "pg1.txt").write_text(_raw("pg1") + "an inserted sentence\n", encoding="utf-8")

    with pytest.raises(CorpusAcquisitionError, match="do not match the digest"):
        catalog.verify_raw(raw)


def test_ingestion_reads_every_declared_source(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)

    sources = catalog.ingest(raw, tmp_path / "cache")

    assert [item.slug for item in sources] == ["pg1", "pg2"]
    assert all(item.source_digest for item in sources)
    assert sources[0].title == "pg1"


def test_a_second_preparation_reuses_the_cache_instead_of_re_ingesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    catalog.ingest(raw, tmp_path / "cache")
    calls: list[str] = []
    original = dataset_catalog.ingest_project_gutenberg

    def counting(raw_text: str, **kwargs: object) -> object:
        calls.append(str(kwargs["slug"]))
        return original(raw_text, **kwargs)  # type: ignore[operator]

    monkeypatch.setattr(dataset_catalog, "ingest_project_gutenberg", counting)

    again = catalog.ingest(raw, tmp_path / "cache")

    assert calls == []
    assert [item.slug for item in again] == ["pg1", "pg2"]


def test_changing_the_document_bound_re_ingests_rather_than_reusing_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    catalog.ingest(raw, tmp_path / "cache")
    calls: list[str] = []
    original = dataset_catalog.ingest_project_gutenberg

    def counting(raw_text: str, **kwargs: object) -> object:
        calls.append(str(kwargs["slug"]))
        return original(raw_text, **kwargs)  # type: ignore[operator]

    monkeypatch.setattr(dataset_catalog, "ingest_project_gutenberg", counting)
    wider = DatasetCatalog.from_mapping({**catalog.as_dict(), "max_characters_per_document": 200})

    wider.ingest(raw, tmp_path / "cache")

    assert calls == ["pg1", "pg2"]


def test_a_cache_file_holding_another_source_is_refused(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    catalog.ingest(raw, tmp_path / "cache")
    cache = catalog.cache_path(catalog.sources[0], tmp_path / "cache")
    document = json.loads(cache.read_text(encoding="utf-8"))
    document["slug"] = "pg99"
    cache.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CorpusAcquisitionError, match="holds a document set for"):
        catalog.ingest(raw, tmp_path / "cache")


def test_a_cache_entry_that_disagrees_with_the_bytes_is_refreshed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    catalog.ingest(raw, tmp_path / "cache")
    cache = catalog.cache_path(catalog.sources[0], tmp_path / "cache")
    document = json.loads(cache.read_text(encoding="utf-8"))
    document["source_digest"] = "f" * 64
    cache.write_text(json.dumps(document), encoding="utf-8")

    refreshed = catalog.ingest(raw, tmp_path / "cache")

    assert refreshed[0].source_digest == catalog.sources[0].sha256
    assert (
        json.loads(cache.read_text(encoding="utf-8"))["source_digest"] == refreshed[0].source_digest
    )


def test_a_file_whose_header_disagrees_with_the_catalogue_is_refused(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    declared = CatalogSource(
        slug="pg1",
        ebook_number=1,
        url=gutenberg_url(1),
        sha256=catalog.sources[0].sha256,
        title="A different book entirely",
    )
    edited = DatasetCatalog(
        name="x",
        version="1",
        licence="l",
        source="s",
        retrieved="2026-01-05",
        sources=(declared, catalog.sources[1]),
        max_characters_per_document=48,
    )

    with pytest.raises(CorpusAcquisitionError, match="the file's own header says"):
        edited.ingest(raw, tmp_path / "cache")


def test_a_source_without_its_wrapper_markers_is_refused(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    (raw / "pg1.txt").write_text("just a text file, no markers at all\n", encoding="utf-8")
    source = catalog.sources[0]
    declared = CatalogSource(
        slug=source.slug,
        ebook_number=source.ebook_number,
        url=source.url,
        sha256=sha256_hex((raw / "pg1.txt").read_bytes()),
    )
    edited = DatasetCatalog(
        name="x",
        version="1",
        licence="l",
        source="s",
        retrieved="2026-01-05",
        sources=(declared, catalog.sources[1]),
        max_characters_per_document=48,
    )

    with pytest.raises(CorpusError, match="wrapper markers"):
        edited.ingest(raw, tmp_path / "cache")


def test_a_pinned_catalog_writes_a_snapshot_that_reads_back_identically(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    path = tmp_path / "catalog.toml"
    write_catalog(catalog, path)

    manifest, checksum = pin_catalog(
        catalog,
        raw_dir=raw,
        cache_dir=tmp_path / "cache",
        snapshot=tmp_path / "snap.json",
        catalog_path=path,
    )

    loaded = read_snapshot(tmp_path / "snap.json")
    assert loaded.checksum == checksum
    assert loaded.manifest.fingerprint == manifest.fingerprint
    assert loaded.manifest.name == "test-corpus"
    assert loaded.manifest.char_count > 0
    assert loaded.acquisition["method"] == "chakaso-dataset-catalog"
    assert loaded.acquisition["catalog_sha256"] == sha256_hex(path.read_bytes())
    assert all(f"pg{number}=" in loaded.acquisition["source_digests"] for number in (1, 2))


def test_a_snapshot_of_a_pinned_catalog_is_not_replaced_unasked(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    catalog = _catalog(raw)
    path = tmp_path / "catalog.toml"
    write_catalog(catalog, path)
    pin_catalog(
        catalog,
        raw_dir=raw,
        cache_dir=tmp_path / "cache",
        snapshot=tmp_path / "snap.json",
        catalog_path=path,
    )

    with pytest.raises(CorpusError, match="not replaced unless"):
        pin_catalog(
            catalog,
            raw_dir=raw,
            cache_dir=tmp_path / "cache",
            snapshot=tmp_path / "snap.json",
            catalog_path=path,
        )


def test_the_catalog_records_the_schema_it_is_written_under(tmp_path: Path) -> None:
    assert _catalog(tmp_path / "raw").as_dict()["schema_version"] == CATALOG_SCHEMA_VERSION


def test_a_per_source_licence_override_reaches_the_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    base = _catalog(raw)
    sources = tuple(
        CatalogSource(**{**item.as_dict(), "licence": "CC0"}) if index == 0 else item
        for index, item in enumerate(base.sources)
    )
    catalog = DatasetCatalog(
        name=base.name,
        version=base.version,
        licence=base.licence,
        source=base.source,
        retrieved=base.retrieved,
        sources=sources,
        max_characters_per_document=base.max_characters_per_document,
    )

    manifest, _ = pin_catalog(
        catalog,
        raw_dir=raw,
        cache_dir=tmp_path / "cache",
        snapshot=tmp_path / "snap.json",
        catalog_path=_write(tmp_path, catalog),
    )

    assert manifest.licence == f"{base.licence}; exceptions: CC0"
