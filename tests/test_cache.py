"""Tests for the acquisition cache and the caching fetcher.

The cache is a bounded, deterministic convenience: it deduplicates by canonical URL, it
ages out under an injected clock, and it never re-fetches what it still holds. The
caching fetcher must delegate the parts it cannot key on, so a bad URL still surfaces the
inner fetcher's typed error rather than a normalization one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from chakaso.retrieval import (
    AcquiredSource,
    AcquisitionCache,
    AcquisitionRequest,
    CachingFetcher,
    Fetcher,
    FetchPolicy,
    HttpFetcher,
    InvalidSchemeError,
    TransportResponse,
)

RETRIEVED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def src(url: str, retrieved_at: datetime = RETRIEVED) -> AcquiredSource:
    return AcquiredSource(
        requested_url=url,
        final_url=url,
        content=b"x",
        media_type="text/html",
        retrieved_at=retrieved_at,
    )


class _CountingFetcher:
    def __init__(self, result: AcquiredSource) -> None:
        self.result = result
        self.calls: list[str] = []

    def fetch(self, request: AcquisitionRequest, policy: FetchPolicy) -> AcquiredSource:
        self.calls.append(request.url)
        return self.result


def _never_transport(_url: str, _policy: FetchPolicy) -> TransportResponse:
    message = "the transport must not be reached for a refused URL"
    raise AssertionError(message)


def test_cache_roundtrip_by_canonical_url() -> None:
    cache = AcquisitionCache()
    source = src("https://example.org/x")
    cache.put(source)

    assert cache.get("https://example.org/x") is source


def test_tracking_parameters_resolve_to_one_entry() -> None:
    cache = AcquisitionCache()
    cache.put(src("https://example.org/x"))

    assert cache.get("https://example.org/x?utm_source=news") is not None


def test_an_entry_older_than_max_age_is_dropped() -> None:
    later = RETRIEVED + timedelta(seconds=100)
    cache = AcquisitionCache(now=lambda: later, max_age_seconds=50)
    cache.put(src("https://example.org/x", retrieved_at=RETRIEVED))

    assert cache.get("https://example.org/x") is None
    assert len(cache) == 0


def test_without_max_age_an_entry_never_expires() -> None:
    much_later = RETRIEVED + timedelta(days=365)
    cache = AcquisitionCache(now=lambda: much_later)
    cache.put(src("https://example.org/x", retrieved_at=RETRIEVED))

    assert cache.get("https://example.org/x") is not None


def test_invalidate_and_clear() -> None:
    cache = AcquisitionCache()
    cache.put(src("https://example.org/x"))
    cache.invalidate("https://example.org/x")

    assert len(cache) == 0

    cache.put(src("https://example.org/y"))
    cache.clear()
    assert len(cache) == 0


def test_invalidating_an_absent_url_is_a_no_op() -> None:
    AcquisitionCache().invalidate("https://example.org/never")


def test_a_miss_is_recorded_as_absent() -> None:
    assert AcquisitionCache().get("https://example.org/missing") is None


def test_caching_fetcher_fetches_once_then_serves_the_cache() -> None:
    result = src("https://example.org/x")
    inner = _CountingFetcher(result)
    caching = CachingFetcher(inner, AcquisitionCache())
    policy = FetchPolicy()

    first = caching.fetch(AcquisitionRequest("https://example.org/x"), policy)
    second = caching.fetch(AcquisitionRequest("https://example.org/x?utm_source=n"), policy)

    assert first is result
    assert second is result
    # The tracking-parameter variant hit the same key, so the inner fetcher ran once.
    assert inner.calls == ["https://example.org/x"]


def test_a_bad_url_reaches_the_inner_fetcher_for_its_typed_error() -> None:
    # An ftp URL cannot be keyed by the canonical web reference; the caching fetcher must
    # delegate so the inner fetcher raises InvalidSchemeError, not a normalization error.
    caching = CachingFetcher(HttpFetcher(_never_transport), AcquisitionCache())

    with pytest.raises(InvalidSchemeError):
        caching.fetch(AcquisitionRequest("ftp://example.org/x"), FetchPolicy())


def test_caching_fetcher_satisfies_the_fetcher_protocol() -> None:
    caching: Fetcher = CachingFetcher(
        _CountingFetcher(src("https://example.org/x")), AcquisitionCache()
    )

    assert isinstance(caching, Fetcher)
