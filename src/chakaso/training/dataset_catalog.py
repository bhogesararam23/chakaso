"""A declared list of dataset sources, verified before any training run is allowed to see them.

The layer that pins a corpus (:mod:`chakaso.training.manifest`) answers "which bytes did this run train
on?". This module answers the question before it: "which files *may* become a corpus, and did the copy on
disk arrive unaltered?" A recorded corpus whose source list lives only in an operator's shell history is not
reproducible — the next person cannot tell what was deliberately excluded from what was accidentally
missing.

The rules this module encodes:

* **Every source is named with its URL, its retrieval date and the digest of the bytes as downloaded.** The
  catalogue is checked into the repository; the corpus bytes are not. A reviewer can therefore audit
  provenance without fetching anything.
* **A digest mismatch is a refusal, not a warning.** A source file that differs from what the catalogue
  declares is either a different edition or a corrupted download, and both change the corpus a measured
  result claims.
* **Ingestion is cached per source, and the cache is keyed by digest.** Re-preparing a corpus reuses the
  documents already derived from bytes that still hash the same, and re-ingests exactly those whose file
  changed. That is what makes "prepare incrementally" a property rather than a manual discipline.
* **Chunking is recorded, not remembered.** The document bound used to cut a source is stored beside its
  cached documents; a different bound invalidates the cache instead of silently reusing it.

Standard library only, so acquisition, verification and caching all run in the dependency-free matrix.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from chakaso.core.atomic import write_text_atomically
from chakaso.core.hashing import sha256_hex
from chakaso.training.errors import CorpusAcquisitionError, CorpusError
from chakaso.training.manifest import CorpusManifest, write_snapshot
from chakaso.training.source_corpus import (
    DEFAULT_MAX_CHARACTERS_PER_DOCUMENT,
    GUTENBERG_BASE_URL,
    IngestedSource,
    build_corpus_manifest,
    ingest_project_gutenberg,
)

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CatalogSource",
    "DatasetCatalog",
    "gutenberg_url",
    "load_catalog",
    "pin_catalog",
    "write_catalog",
]

CATALOG_SCHEMA_VERSION = "chakaso-dataset-catalog/1"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class CatalogSource:
    """One downloadable text a corpus may be built from, as declared rather than as observed.

    Attributes:
        slug: Path prefix for the documents this source contributes; also its cache filename.
        ebook_number: The Project Gutenberg identifier, which is what the URL is built from.
        url: The exact location the bytes came from.
        sha256: Digest of the downloaded bytes, before any decoding or normalization.
        title/author: Human-readable identity, recorded for the data card and never used to select data.
        licence: Overrides the catalogue-wide licence when one source is not under the same terms.
    """

    slug: str
    ebook_number: int
    url: str
    sha256: str
    title: str = ""
    author: str = ""
    licence: str | None = None

    def __post_init__(self) -> None:
        if not _SLUG.match(self.slug):
            message = f"source slug {self.slug!r} is not a usable path prefix"
            raise CorpusError(message)
        if isinstance(self.ebook_number, bool) or self.ebook_number < 1:
            message = f"ebook_number must be an integer >= 1, got {self.ebook_number!r}"
            raise CorpusError(message)
        if not _DIGEST.match(self.sha256):
            message = (
                f"source {self.slug!r} declares a digest that is not 64 lowercase hex characters"
            )
            raise CorpusError(message)
        if not self.url.startswith(("https://", "http://")):
            message = f"source {self.slug!r} has a url with no scheme: {self.url!r}"
            raise CorpusError(message)
        if self.licence is not None and not self.licence.strip():
            message = f"source {self.slug!r} declares an empty licence; omit the field to inherit"
            raise CorpusError(message)

    def as_dict(self) -> dict[str, object]:
        """The declaration as plain data, in a stable key order."""
        values: dict[str, object] = {
            "slug": self.slug,
            "ebook_number": self.ebook_number,
            "url": self.url,
            "sha256": self.sha256,
            "title": self.title,
            "author": self.author,
        }
        if self.licence is not None:
            values["licence"] = self.licence
        return values


@dataclass(frozen=True, slots=True)
class DatasetCatalog:
    """A named corpus and the list of sources it is built from.

    Attributes:
        name/version: The corpus identity a manifest records, so a run can declare which catalogue it used.
        licence: The terms that apply unless a source overrides them.
        source: The provider, in the words a reader would recognise.
        retrieved: ISO date the declared bytes were taken. Project Gutenberg renumbers and re-edits its
            collection, so "the current version" is not an identifier.
        sources: The declared texts, in slug order.
        max_characters_per_document: The chunking bound, recorded so a later run gets the same documents.
    """

    name: str
    version: str
    licence: str
    source: str
    retrieved: str
    sources: tuple[CatalogSource, ...]
    max_characters_per_document: int = DEFAULT_MAX_CHARACTERS_PER_DOCUMENT

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "licence", "source"):
            if not getattr(self, field_name).strip():
                message = f"catalog {field_name} must not be empty"
                raise CorpusError(message)
        if not _ISO_DATE.match(self.retrieved):
            message = f"catalog retrieved date {self.retrieved!r} is not an ISO YYYY-MM-DD date"
            raise CorpusError(message)
        if not self.sources:
            message = f"catalog {self.name!r} declares no sources"
            raise CorpusError(message)
        if (
            isinstance(self.max_characters_per_document, bool)
            or self.max_characters_per_document < 1
        ):
            message = (
                f"max_characters_per_document must be an integer >= 1, got "
                f"{self.max_characters_per_document!r}"
            )
            raise CorpusError(message)
        slugs = [item.slug for item in self.sources]
        if len(set(slugs)) != len(slugs):
            duplicated = sorted({slug for slug in slugs if slugs.count(slug) > 1})
            message = f"catalog {self.name!r} declares the same slug more than once: {duplicated}"
            raise CorpusError(message)
        numbers = [item.ebook_number for item in self.sources]
        if len(set(numbers)) != len(numbers):
            message = f"catalog {self.name!r} declares the same ebook number for two slugs"
            raise CorpusError(message)

    def licence_for(self, item: CatalogSource) -> str:
        """The terms that apply to one source — its own if it states them, otherwise the catalogue's."""
        return item.licence if item.licence is not None else self.licence

    def as_dict(self) -> dict[str, object]:
        """The catalogue as plain data."""
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "licence": self.licence,
            "source": self.source,
            "retrieved": self.retrieved,
            "max_characters_per_document": self.max_characters_per_document,
            "sources": [item.as_dict() for item in self.sources],
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> DatasetCatalog:
        """Rebuild a catalogue from stored data, refusing unknown or missing fields.

        Raises:
            CorpusError: the schema version is not this one, a required field is absent, or a source
                declaration is not a mapping of the known keys.
        """
        schema = values.get("schema_version")
        if schema != CATALOG_SCHEMA_VERSION:
            message = f"unsupported dataset catalog schema: {schema!r}"
            raise CorpusError(message)
        for field_name in ("name", "version", "licence", "source", "retrieved"):
            if not isinstance(values.get(field_name), str):
                message = f"catalog field {field_name!r} must be a string"
                raise CorpusError(message)
        bound = values.get("max_characters_per_document", DEFAULT_MAX_CHARACTERS_PER_DOCUMENT)
        if isinstance(bound, bool) or not isinstance(bound, int):
            message = f"catalog max_characters_per_document must be an integer, got {bound!r}"
            raise CorpusError(message)
        raw_sources = values.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            message = "catalog sources must be a non-empty list"
            raise CorpusError(message)
        known = set(CatalogSource.__dataclass_fields__)
        sources: list[CatalogSource] = []
        for entry in raw_sources:
            if not isinstance(entry, dict) or not known.issuperset(entry):
                message = f"a catalog source is not a mapping of {sorted(known)}: {entry!r}"
                raise CorpusError(message)
            sources.append(CatalogSource(**entry))
        return cls(
            name=str(values["name"]),
            version=str(values["version"]),
            licence=str(values["licence"]),
            source=str(values["source"]),
            retrieved=str(values["retrieved"]),
            sources=tuple(sources),
            max_characters_per_document=bound,
        )

    def raw_path(self, item: CatalogSource, raw_dir: Path) -> Path:
        """Where the downloaded bytes for one source are expected to live."""
        return raw_dir / f"{item.slug}.txt"

    def cache_path(self, item: CatalogSource, cache_dir: Path) -> Path:
        """Where the ingested document set for one source is cached."""
        return cache_dir / f"{item.slug}.json"

    def missing_raw_files(self, raw_dir: Path) -> tuple[str, ...]:
        """Slugs whose downloaded file is not present — reported, never assumed away."""
        return tuple(
            item.slug for item in self.sources if not self.raw_path(item, raw_dir).is_file()
        )

    def verify_raw(self, raw_dir: Path) -> dict[str, str]:
        """Check every declared source's bytes against its declared digest.

        Returns:
            slug → digest of the bytes actually read, for a caller that wants to record them.

        Raises:
            CorpusAcquisitionError: a file is absent, or its digest is not the declared one. Both lists the
                slugs and say what was expected, because "the corpus is not the one that was pinned" is the
                failure a reproducibility layer exists to make loud.
        """
        absent = self.missing_raw_files(raw_dir)
        if absent:
            message = (
                f"catalog {self.name!r} is missing {len(absent)} declared source file(s) under "
                f"{str(raw_dir)!r}: {list(absent)}; fetch them from the urls the catalog names"
            )
            raise CorpusAcquisitionError(message)
        observed: dict[str, str] = {}
        mismatched: list[str] = []
        for item in self.sources:
            digest = sha256_hex(self.raw_path(item, raw_dir).read_bytes())
            if digest != item.sha256:
                mismatched.append(f"{item.slug}: declared {item.sha256}, read {digest}")
            observed[item.slug] = digest
        if mismatched:
            message = (
                f"{len(mismatched)} source file(s) under {str(raw_dir)!r} do not match the digest the "
                f"catalog {self.name!r} declares — the bytes are a different edition or were altered after "
                f"download: {'; '.join(mismatched)}"
            )
            raise CorpusAcquisitionError(message)
        return observed

    def ingest(self, raw_dir: Path, cache_dir: Path) -> tuple[IngestedSource, ...]:
        """Ingest every declared source, reusing cached documents whose bytes have not moved.

        The cache is keyed by the digest of the downloaded file *and* the chunking bound, so a changed file
        or a changed bound re-ingests that source and nothing else.

        Returns:
            The ingested sources, in slug order.

        Raises:
            CorpusAcquisitionError: verification failed, a cached document set disagrees with the source it
                claims to hold, or a file's own header names a different work than the catalogue declares.
        """
        verified = self.verify_raw(raw_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        produced: list[IngestedSource] = []
        for item in self.sources:
            digest = verified[item.slug]
            cached = self._read_cache(item, cache_dir, digest)
            if cached is not None:
                produced.append(cached)
                continue
            raw_bytes = self.raw_path(item, raw_dir).read_bytes()
            ingested = ingest_project_gutenberg(
                raw_bytes.decode("utf-8-sig"),
                slug=item.slug,
                ebook_number=item.ebook_number,
                source_digest=digest,
            )
            _assert_declaration_matches_file(item, ingested)
            self._write_cache(item, cache_dir, ingested)
            produced.append(ingested)
        return tuple(produced)

    def _read_cache(
        self, item: CatalogSource, cache_dir: Path, digest: str
    ) -> IngestedSource | None:
        path = self.cache_path(item, cache_dir)
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("slug") != item.slug:
            message = f"cache file {str(path)!r} holds a document set for {document.get('slug')!r}"
            raise CorpusAcquisitionError(message)
        if document.get("source_digest") != digest:
            return None
        if document.get("max_characters_per_document") != self.max_characters_per_document:
            return None
        return IngestedSource(
            slug=item.slug,
            ebook_number=item.ebook_number,
            title=document["title"],
            author=document["author"],
            licence_statement=document["licence_statement"],
            source_digest=digest,
            raw_characters=document["raw_characters"],
            text=document["text"],
        )

    def _write_cache(
        self,
        item: CatalogSource,
        cache_dir: Path,
        ingested: IngestedSource,
    ) -> None:
        path = self.cache_path(item, cache_dir)
        # A cache entry is derived, reproducible data: it exists only to avoid re-ingesting bytes that still
        # hash the same, so a stale entry (different digest, different bound) is replaced rather than
        # quarantined. The overwrite guard lives on the snapshot and the manifest, which carry a record a
        # measured result cites; nothing here is cited by anything.
        payload = {
            "slug": item.slug,
            "ebook_number": item.ebook_number,
            "title": ingested.title,
            "author": ingested.author,
            "licence_statement": ingested.licence_statement,
            "source_digest": ingested.source_digest,
            "raw_characters": ingested.raw_characters,
            "max_characters_per_document": self.max_characters_per_document,
            "text": ingested.text,
        }
        write_text_atomically(
            json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=False) + "\n",
            path,
        )


def _assert_declaration_matches_file(item: CatalogSource, ingested: IngestedSource) -> None:
    """Refuse a file whose own header names a different work than the catalogue declares for that slug.

    The digest proves the bytes; this proves the *description*. Without it, a catalogue could pin
    ``pg11.txt`` under the title of another book and every downstream data card would repeat the error with
    the authority of a checked file.
    """
    for label, declared, observed in (
        ("title", item.title, ingested.title),
        ("author", item.author, ingested.author),
    ):
        if not declared or not observed:
            continue
        if " ".join(declared.lower().split()) != " ".join(observed.lower().split()):
            message = (
                f"source {item.slug!r} is declared as {label} {declared!r} but the file's own header says "
                f"{observed!r}; the bytes are a different work than the catalogue describes"
            )
            raise CorpusAcquisitionError(message)


def load_catalog(path: Path) -> DatasetCatalog:
    """Read a catalogue from a TOML file.

    Raises:
        CorpusError: the file is not a catalogue this code understands.
    """
    with path.open("rb") as handle:
        return DatasetCatalog.from_mapping(tomllib.load(handle))


def write_catalog(catalog: DatasetCatalog, path: Path, *, allow_overwrite: bool = False) -> None:
    """Write a catalogue as TOML, deterministically, so a diff of it is a diff of the corpus declaration.

    Raises:
        CorpusError: a file already occupies ``path`` and overwriting was not asked for.
    """
    if path.exists() and not allow_overwrite:
        message = (
            f"dataset catalog already exists at {str(path)!r}; it is replaced only when the caller "
            "passes allow_overwrite=True"
        )
        raise CorpusError(message)
    lines = [
        "# Declared sources of one corpus. The bytes stay out of Git; this file is the audit trail.",
        f"schema_version = {json.dumps(CATALOG_SCHEMA_VERSION)}",
        f"name = {json.dumps(catalog.name)}",
        f"version = {json.dumps(catalog.version)}",
        f"licence = {json.dumps(catalog.licence)}",
        f"source = {json.dumps(catalog.source)}",
        f"retrieved = {json.dumps(catalog.retrieved)}",
        f"max_characters_per_document = {catalog.max_characters_per_document}",
    ]
    for item in catalog.sources:
        lines.append("")
        lines.append("[[sources]]")
        for key, value in item.as_dict().items():
            rendered = value if isinstance(value, int) else json.dumps(value)
            lines.append(f"{key} = {rendered}")
    write_text_atomically("\n".join(lines) + "\n", path)


def pin_catalog(
    catalog: DatasetCatalog,
    *,
    raw_dir: Path,
    cache_dir: Path,
    snapshot: Path,
    catalog_path: Path,
    allow_overwrite: bool = False,
) -> tuple[CorpusManifest, str]:
    """Verify, ingest and pin a catalogue's sources into the snapshot a training run reads.

    Args:
        catalog_path: The catalogue file, digested into the acquisition block so a later reader can prove
            which declaration produced this corpus, not merely which name it had.

    Returns:
        The pinned manifest and the snapshot's checksum.

    Raises:
        CorpusAcquisitionError: a declared source is absent or its bytes are not what was declared.
        CorpusError: the snapshot already exists and overwriting was not asked for.
    """
    sources = catalog.ingest(raw_dir, cache_dir)
    manifest = build_corpus_manifest(
        sources,
        name=catalog.name,
        version=catalog.version,
        # A catalogue may declare per-source terms, so the manifest's single licence field records the
        # catalogue-wide term plus every override that actually applies; nothing is flattened away.
        licence=_combined_licence(catalog),
        source=catalog.source,
        retrieved=catalog.retrieved,
        max_characters_per_document=catalog.max_characters_per_document,
    )
    checksum = write_snapshot(
        manifest,
        snapshot,
        acquisition={
            "method": "chakaso-dataset-catalog",
            "catalog": catalog.name,
            "catalog_version": catalog.version,
            "catalog_sha256": sha256_hex(catalog_path.read_bytes()),
            "retrieved": catalog.retrieved,
            "urls": ";".join(f"{item.slug}={item.url}" for item in catalog.sources),
            "source_digests": ";".join(f"{item.slug}={item.sha256}" for item in catalog.sources),
            "characters_per_document_bound": str(catalog.max_characters_per_document),
            "titles": ";".join(
                f"{item.slug}={item.title or item.slug}" for item in catalog.sources
            ),
        },
        allow_overwrite=allow_overwrite,
    )
    return manifest, checksum


def _combined_licence(catalog: DatasetCatalog) -> str:
    overrides = sorted(
        {
            item.licence
            for item in catalog.sources
            if item.licence is not None and item.licence != catalog.licence
        }
    )
    if not overrides:
        return catalog.licence
    return catalog.licence + "; exceptions: " + " | ".join(overrides)


def gutenberg_url(ebook_number: int) -> str:
    """The canonical Project Gutenberg text URL for one ebook number."""
    return f"{GUTENBERG_BASE_URL}{ebook_number}/pg{ebook_number}.txt"
