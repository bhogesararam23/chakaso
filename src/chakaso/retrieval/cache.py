"""A small, deterministic acquisition cache — a cache, never a source of truth.

Refetching the same URL on every query wastes time and, against a live server, is rude.
`AcquisitionCache` keeps acquired sources in memory keyed by their canonical URL, and
`CachingFetcher` wraps any `Fetcher` so a repeated request is served from it. It is a
baseline: there is no disk store, no distributed cache and no HTTP-conditional
revalidation. Those are later replacements behind the same interface (ADR-0012).

Freshness is honest and deterministic: an optional `max_age_seconds`, compared against an
injectable clock, decides whether an entry is still good. Nothing pretends the cached copy
is live; it is the copy taken at `retrieved_at`, and that timestamp is part of the record.

Keying uses the canonical web reference, so the same page reached through tracking
parameters or a trailing slash is one cache entry — the same normalization that makes it
one source (ADR-0011).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from chakaso.evidence import web_reference
from chakaso.evidence.errors import UrlError
from chakaso.retrieval.acquire import AcquiredSource, AcquisitionRequest, Fetcher
from chakaso.retrieval.policy import FetchPolicy

__all__ = ["AcquisitionCache", "CachingFetcher"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AcquisitionCache:
    """An in-memory cache of acquired sources keyed by canonical URL."""

    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        max_age_seconds: float | None = None,
    ) -> None:
        self._entries: dict[str, AcquiredSource] = {}
        self._now = now if now is not None else _utcnow
        self._max_age = max_age_seconds

    def _key(self, url: str) -> str:
        return web_reference(url).canonical

    def get(self, url: str) -> AcquiredSource | None:
        """Return a still-fresh cached source for ``url``, or ``None``.

        An expired entry is dropped rather than returned, so a caller cannot mistake a
        stale copy for a current one while the age limit is configured.
        """
        key = self._key(url)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._max_age is not None:
            age = self._now() - entry.retrieved_at
            if age > timedelta(seconds=self._max_age):
                del self._entries[key]
                return None
        return entry

    def put(self, acquired: AcquiredSource) -> None:
        """Store an acquired source under its request's canonical URL."""
        self._entries[self._key(acquired.requested_url)] = acquired

    def invalidate(self, url: str) -> None:
        """Drop the cached entry for ``url``, if any. Absent entries are ignored."""
        self._entries.pop(self._key(url), None)

    def clear(self) -> None:
        """Empty the cache."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class CachingFetcher:
    """A `Fetcher` that serves repeat requests from an `AcquisitionCache`."""

    def __init__(self, inner: Fetcher, cache: AcquisitionCache) -> None:
        self._inner = inner
        self._cache = cache

    def fetch(self, request: AcquisitionRequest, policy: FetchPolicy) -> AcquiredSource:
        """Return a cached source if one is fresh, otherwise fetch, cache and return.

        A URL that cannot be canonicalized is passed straight to the inner fetcher, which
        raises the correct typed error — the cache never turns a policy refusal into a
        normalization error.
        """
        try:
            cached = self._cache.get(request.url)
        except UrlError:
            return self._inner.fetch(request, policy)

        if cached is not None:
            return cached

        acquired = self._inner.fetch(request, policy)
        self._cache.put(acquired)
        return acquired
