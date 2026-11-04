"""A bounded drift re-scan, tested offline: what it samples, what it reports, and what it must never touch.

The fetcher is injected, so every row here comes from a fake that returns bytes or raises the error a policy limit
would raise. The properties that matter are the ones the run cannot be trusted without: the sample is fixed before
any request is made, a moved source is classified from the bytes fetched today, and neither the catalogue nor the
pinned raw directory is written to — asserted by digest and by an empty directory, not by reading the code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from chakaso.core.hashing import sha256_hex
from chakaso.retrieval.acquire import AcquiredSource, AcquisitionRequest
from chakaso.retrieval.errors import ResponseTooLargeError
from chakaso.retrieval.policy import FetchPolicy
from chakaso.training.dataset_catalog import CatalogSource, DatasetCatalog, write_catalog
from chakaso.training.source_corpus import ingest_project_gutenberg

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rescan_pinned_sources.py"

_WHEN = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)
_POLICY = FetchPolicy(
    allowed_schemes=frozenset({"https"}),
    timeout_seconds=20.0,
    max_redirects=3,
    max_response_bytes=1_048_576,
)


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("rescan_pinned_sources_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_DEFAULT_BODY = (
    "First paragraph of the book, with enough words in it to chunk.\n\n"
    "Second paragraph, long enough to be a document of its own under a small bound.\n"
)


def _raw(
    book: str, number: int, *, body: str | None = None, title_line: str | None = None
) -> bytes:
    """A minimal Project Gutenberg plain text: the fixture's shape, with the header line free to move alone."""
    return (
        f"Title: {book if title_line is None else title_line}\nAuthor: A Person\n\n"
        f"*** START OF THE PROJECT GUTENBERG EBOOK {number} ***\n"
        f"{_DEFAULT_BODY if body is None else body}"
        f"*** END OF THE PROJECT GUTENBERG EBOOK {number} ***\n"
    ).encode()


@dataclass
class _FakeFetcher:
    """A `Fetcher` that answers each URL with fixed bytes or a fixed policy refusal."""

    answers: dict[str, bytes | Exception]
    calls: list[str] = field(default_factory=list)
    policies: list[FetchPolicy] = field(default_factory=list)

    def fetch(self, request: AcquisitionRequest, policy: FetchPolicy) -> AcquiredSource:
        self.calls.append(request.url)
        self.policies.append(policy)
        answer = self.answers[request.url]
        if isinstance(answer, Exception):
            raise answer
        return AcquiredSource(
            requested_url=request.url,
            final_url=request.url,
            content=answer,
            media_type="text/plain; charset=utf-8",
            retrieved_at=_WHEN,
        )


def _catalog(tmp_path: Path, counts: dict[str, int] | None = None) -> DatasetCatalog:
    """A catalogue of ``counts[register]`` sources per register, every one pinned to a real ingestable text."""
    slugs = [
        f"{register}-pg{100 + number}"
        for register, total in (counts or {"prose": 3, "science": 3, "technology": 3}).items()
        for number in range(total)
    ]
    sources = []
    for number, slug in enumerate(slugs, start=1):
        declared = _raw(slug, number)
        sources.append(
            CatalogSource(
                slug=slug,
                ebook_number=number,
                url=f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt",
                sha256=sha256_hex(declared),
                title=slug,
                author="A Person",
                ingested_sha256=ingest_project_gutenberg(
                    declared.decode("utf-8-sig"), slug=slug, ebook_number=number
                ).documents_digest(16384),
            )
        )
    catalog = DatasetCatalog(
        name="rescan-fixture",
        version="1.0.0",
        licence="Public domain",
        source="A test archive",
        retrieved="2026-10-30",
        sources=tuple(sources),
    )
    write_catalog(catalog, tmp_path / "catalog.toml")
    return catalog


# ------------------------------------------------------------------ the sample


def test_the_sample_is_the_same_for_the_same_catalogue_on_any_day(tmp_path: Path) -> None:
    module = _script()
    catalog = _catalog(tmp_path)

    first = module.select_sources(catalog, per_register=2)
    second = module.select_sources(catalog, per_register=2)

    assert [item.slug for item in first] == [item.slug for item in second]
    assert len(first) <= 6


def test_every_register_in_the_catalogue_is_represented_in_the_sample(tmp_path: Path) -> None:
    module = _script()
    catalog = _catalog(tmp_path)

    chosen = module.select_sources(catalog, per_register=1)
    registers = {module.register_of(item.slug) for item in chosen}

    assert registers == {"prose", "science", "technology"}
    assert all(
        sum(1 for item in chosen if module.register_of(item.slug) == name) == 1
        for name in registers
    )


def test_a_register_shorter_than_the_quota_contributes_what_it_has_rather_than_a_pad(
    tmp_path: Path,
) -> None:
    module = _script()
    catalog = _catalog(tmp_path, {"prose": 5, "science": 1})

    chosen = module.select_sources(catalog, per_register=3)

    assert sum(1 for item in chosen if module.register_of(item.slug) == "science") == 1
    assert sum(1 for item in chosen if module.register_of(item.slug) == "prose") == 3


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("mathematics-pg16449", "mathematics"),
        ("prose-pg9403", "prose"),
        ("architecture.md", "architecture.md"),
    ],
)
def test_the_register_is_read_off_the_slug_the_way_the_split_names_it(
    slug: str, expected: str
) -> None:
    assert _script().register_of(slug) == expected


def test_a_quota_that_is_not_a_positive_integer_is_refused(tmp_path: Path) -> None:
    module = _script()
    catalog = _catalog(tmp_path)

    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="per_register must be"):
            module.select_sources(catalog, per_register=bad)  # type: ignore[arg-type]


# ------------------------------------------------------------------ the rows


def test_a_source_that_still_serves_the_pinned_bytes_is_reported_unchanged(tmp_path: Path) -> None:
    module = _script()
    catalog = _catalog(tmp_path)
    item = catalog.sources[0]
    fetcher = _FakeFetcher({item.url: _raw(item.slug, item.ebook_number)})

    report = module.rescan(
        catalog,
        (item,),
        fetcher,
        _POLICY,
        catalog_path=tmp_path / "catalog.toml",
        sleep=lambda _seconds: None,
    )

    row = report["rows"][0]
    assert row["status"] == "unchanged"
    assert row["observed_sha256"] == item.sha256
    assert row["retrieved_at"] == _WHEN.isoformat()
    assert row["media_type"] == "text/plain; charset=utf-8"
    assert report["counts"] == {"unchanged": 1}


def test_a_header_that_moved_while_the_text_stayed_is_classified_outside_the_ingested_text(
    tmp_path: Path,
) -> None:
    """The `science-pg3489` case, end to end: different raw digest, identical ingested documents.

    The classification is the whole point of ADR-0047's first half — a moved header is survivable and a moved body
    is not — so it is exercised through the catalogue's own classifier rather than restated.
    """
    module = _script()
    catalog = _catalog(tmp_path)
    item = catalog.sources[1]
    moved = _raw(item.slug, item.ebook_number, title_line=f"{item.slug} (revised edition note)")
    assert sha256_hex(moved) != item.sha256
    fetcher = _FakeFetcher({item.url: moved})

    report = module.rescan(
        catalog,
        (item,),
        fetcher,
        _POLICY,
        catalog_path=tmp_path / "catalog.toml",
        sleep=lambda _seconds: None,
    )

    row = report["rows"][0]
    assert row["status"] == "drifted-ingested-text-identical"
    assert row["observed_sha256"] == sha256_hex(moved)
    assert row["bytes_fetched"] == len(moved)


def test_text_that_moved_inside_the_ingested_documents_is_named_as_the_serious_case(
    tmp_path: Path,
) -> None:
    module = _script()
    catalog = _catalog(tmp_path)
    item = catalog.sources[2]
    changed = _raw(
        item.slug, item.ebook_number, body="A different paragraph entirely, with other words.\n"
    )
    fetcher = _FakeFetcher({item.url: changed})

    report = module.rescan(
        catalog,
        (item,),
        fetcher,
        _POLICY,
        catalog_path=tmp_path / "catalog.toml",
        sleep=lambda _seconds: None,
    )

    assert report["rows"][0]["status"] == "drifted-ingested-text-differs"


def test_a_fetch_the_policy_refuses_is_a_row_not_a_lost_source(tmp_path: Path) -> None:
    module = _script()
    catalog = _catalog(tmp_path)
    item = catalog.sources[0]
    refusal = ResponseTooLargeError(
        f"response from {item.url} exceeds the 1048576-byte fetch limit"
    )
    fetcher = _FakeFetcher({item.url: refusal})

    report = module.rescan(
        catalog,
        (item,),
        fetcher,
        _POLICY,
        catalog_path=tmp_path / "catalog.toml",
        sleep=lambda _seconds: None,
    )

    row = report["rows"][0]
    assert row["status"] == "fetch-refused"
    assert row["observed_sha256"] is None
    assert "exceeds the 1048576-byte fetch limit" in row["detail"]
    assert report["counts"] == {"fetch-refused": 1}


def test_every_request_carries_the_declared_limits(tmp_path: Path) -> None:
    module = _script()
    catalog = _catalog(tmp_path)
    chosen = module.select_sources(catalog, per_register=1)
    fetcher = _FakeFetcher({item.url: _raw(item.slug, item.ebook_number) for item in chosen})

    report = module.rescan(
        catalog,
        chosen,
        fetcher,
        _POLICY,
        catalog_path=tmp_path / "catalog.toml",
        sleep=lambda _seconds: None,
    )

    assert {policy.max_response_bytes for policy in fetcher.policies} == {1_048_576}
    assert {policy.timeout_seconds for policy in fetcher.policies} == {20.0}
    assert all(policy.allowed_schemes == frozenset({"https"}) for policy in fetcher.policies)
    assert len(fetcher.calls) == report["selection"]["sampled"]


def test_requests_are_spaced_and_the_first_of_them_is_not_sleeped_on(tmp_path: Path) -> None:
    module = _script()
    catalog = _catalog(tmp_path)
    chosen = module.select_sources(catalog, per_register=2)
    fetcher = _FakeFetcher({item.url: _raw(item.slug, item.ebook_number) for item in chosen})
    slept: list[float] = []

    module.rescan(
        catalog,
        chosen,
        fetcher,
        _POLICY,
        catalog_path=tmp_path / "catalog.toml",
        sleep=slept.append,
    )

    assert slept == [module.REQUEST_SPACING_SECONDS] * (len(chosen) - 1)


# ------------------------------------------------------------------ what it must not touch


def test_a_rescan_writes_nothing_where_the_pinned_bytes_live(tmp_path: Path) -> None:
    """The safety property, checked as an absence: the raw directory the corpus was built from stays as it was.

    The fetched bytes go to a temporary staging directory that the classifier's own read never leaves behind, so
    the pinned directory is empty before and empty after even though three sources came back different.
    """
    module = _script()
    catalog = _catalog(tmp_path, {"prose": 3})
    (tmp_path / "raw").mkdir()
    item = catalog.sources[0]
    fetcher = _FakeFetcher(
        {
            url: _raw("other", 1)
            for url in (item.url, catalog.sources[1].url, catalog.sources[2].url)
        }
    )

    report = module.rescan(
        catalog,
        (item, catalog.sources[1], catalog.sources[2]),
        fetcher,
        _POLICY,
        catalog_path=tmp_path / "catalog.toml",
        sleep=lambda _seconds: None,
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["catalog.toml", "raw"]
    assert list((tmp_path / "raw").iterdir()) == []
    assert report["catalog_untouched"] is True
    assert any(row["status"].startswith("drifted-") for row in report["rows"])


def test_the_report_carries_the_pin_it_compared_against_and_its_own_limits(tmp_path: Path) -> None:
    module = _script()
    catalog = _catalog(tmp_path)
    item = catalog.sources[0]
    fetcher = _FakeFetcher({item.url: _raw(item.slug, item.ebook_number)})
    before = sha256_hex((tmp_path / "catalog.toml").read_bytes())

    report = module.rescan(
        catalog,
        (item,),
        fetcher,
        _POLICY,
        catalog_path=tmp_path / "catalog.toml",
        sleep=lambda _seconds: None,
    )

    assert report["catalog_file_sha256"] == before
    assert report["catalog"] == {
        "name": "rescan-fixture",
        "version": "1.0.0",
        "sources": len(catalog.sources),
        "licence": "Public domain",
        "provider": "A test archive",
    }
    assert report["catalog_retrieved"] == "2026-10-30"
    assert "cannot bound the drift of the whole catalogue" in report["limitations"]
    assert report["policy"]["timeout_seconds"] == 20.0
    assert report["policy"]["max_response_bytes"] == 1_048_576
    assert report["policy"]["user_agent"] == _POLICY.user_agent
    assert module.RESCAN_POLICY.user_agent.startswith("chakaso-corpus-drift-rescan/")
    assert module.RESCAN_POLICY.allowed_schemes == frozenset({"https"})
    assert module.RESCAN_POLICY.max_redirects == 3


# ------------------------------------------------------------------ the command


def test_the_command_refuses_to_replace_a_drift_record_already_written(tmp_path: Path) -> None:
    """The refusal comes before the catalogue is read, so no scan is spent to learn it."""
    module = _script()
    out = tmp_path / "rescan.json"
    out.write_text("{}", encoding="utf-8")

    assert (
        module.main([f"--catalog={tmp_path / 'catalog.toml'}", "--per-register=1", f"--out={out}"])
        == 2
    )


def test_a_slug_the_catalogue_does_not_declare_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script()
    _catalog(tmp_path, {"prose": 1})

    status = module.main(
        [
            f"--catalog={tmp_path / 'catalog.toml'}",
            "--slugs=prose-pg999999",
            f"--out={tmp_path / 'rescan.json'}",
        ]
    )

    assert status == 1
    assert "declares none of" in capsys.readouterr().err


def test_the_written_report_is_valid_json_with_one_row_per_sampled_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script()
    module_catalog = _catalog(tmp_path)
    chosen = module.select_sources(module_catalog, per_register=1)
    fetcher = _FakeFetcher({item.url: _raw(item.slug, item.ebook_number) for item in chosen})
    out = tmp_path / "rescan.json"
    monkeypatch.setattr(module, "_fetcher", lambda: fetcher)

    assert (
        module.main([f"--catalog={tmp_path / 'catalog.toml'}", "--per-register=1", f"--out={out}"])
        == 0
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert len(report["rows"]) == len(chosen)
    assert report["counts"] == {"unchanged": len(chosen)}
