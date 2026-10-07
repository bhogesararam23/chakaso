"""Tests for URL canonicalization.

Canonicalization is part of the identifier contract (ADR-0007), so the property
under test is mostly idempotence and the avoidance of false merges: two URLs must
not become one unless they name the same document.
"""

from __future__ import annotations

import pytest

from chakaso.evidence import UrlError, canonicalize_url, host_of


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Scheme and host case, and the scheme-less noise a page might contain.
        ("HTTPS://Example.ORG/Path", "https://example.org/Path"),
        ("https://example.org", "https://example.org/"),
        # Fragments never identify a document.
        ("https://example.org/a#section-3", "https://example.org/a"),
        # Default ports are the same authority.
        ("http://example.org:80/a", "http://example.org/a"),
        ("https://example.org:443/a", "https://example.org/a"),
        # Non-default ports are preserved, because they are a different service.
        ("https://example.org:8443/a", "https://example.org:8443/a"),
        # A trailing root dot resolves identically.
        ("https://example.org./a", "https://example.org/a"),
        # A single trailing slash on a non-root path is removed.
        ("https://example.org/a/", "https://example.org/a"),
        ("https://example.org/", "https://example.org/"),
    ],
)
def test_canonical_forms(raw: str, expected: str) -> None:
    assert canonicalize_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "https://example.org/a?utm_source=newsletter&utm_medium=email",
        "https://example.org/a?utm_campaign=x",
        "https://example.org/a?fbclid=abc123",
        "https://example.org/a?ref=homepage",
        "https://example.org/a?mc_cid=1&mc_eid=2",
    ],
)
def test_tracking_parameters_are_removed(raw: str) -> None:
    # Without this, one document reached from three links becomes three sources and
    # fills an evidence pack with copies of itself.
    assert canonicalize_url(raw) == "https://example.org/a"


def test_meaningful_query_parameters_are_kept() -> None:
    assert canonicalize_url("https://example.org/search?q=chakaso") == (
        "https://example.org/search?q=chakaso"
    )


def test_query_parameter_order_does_not_create_two_sources() -> None:
    assert canonicalize_url("https://example.org/a?b=2&a=1") == canonicalize_url(
        "https://example.org/a?a=1&b=2"
    )


def test_parameters_that_look_like_tracking_but_are_not_are_kept() -> None:
    # `source` is a tracking parameter; `source_id` is not, and dropping it would
    # silently merge different documents.
    assert canonicalize_url("https://example.org/a?source_id=42") == (
        "https://example.org/a?source_id=42"
    )


def test_blank_parameter_values_survive() -> None:
    assert canonicalize_url("https://example.org/a?q=") == "https://example.org/a?q="


def test_percent_escapes_are_preserved_and_normalization_is_idempotent() -> None:
    once = canonicalize_url("https://example.org/a%20b?q=x%20y")
    twice = canonicalize_url(once)

    assert once == twice
    assert "a%20b" in once


def test_whitespace_around_the_url_is_stripped() -> None:
    assert canonicalize_url("  https://example.org/a  ") == "https://example.org/a"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "ftp://example.org/a",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "https://",
        "https:///path",
    ],
)
def test_unusable_urls_are_refused(raw: str) -> None:
    # A scheme that reaches another protocol is a security question, not a
    # normalization one, so it is refused rather than ignored.
    with pytest.raises(UrlError):
        canonicalize_url(raw)


def test_embedded_credentials_are_refused() -> None:
    # A canonical URL is written into source provenance, so accepting credentials
    # would store a secret in a record meant to be published.
    with pytest.raises(UrlError, match="credentials"):
        canonicalize_url("https://user:secret@example.org/a")


def test_url_without_a_host_is_refused_with_the_url_in_the_message() -> None:
    with pytest.raises(UrlError, match="no host"):
        canonicalize_url("https://")


def test_ipv6_hosts_survive_canonicalization() -> None:
    assert canonicalize_url("https://[2001:db8::1]:8443/a") == "https://[2001:db8::1]:8443/a"


def test_host_of_returns_the_lowercase_host() -> None:
    assert host_of("https://Example.ORG:8443/a") == "example.org"


def test_host_of_is_the_host_not_a_registrable_domain() -> None:
    # Documented deliberately: reducing a public suffix needs a dependency and a
    # frequently-updated data file, so the host is recorded exactly.
    assert host_of("https://news.example.co.uk/a") == "news.example.co.uk"


def test_different_documents_are_not_merged() -> None:
    # The failure that matters most: over-eager normalization silently merges two
    # distinct pages, and a merged source is invisible in a way a duplicate is not.
    distinct = {
        "https://example.org/a",
        "https://example.org/a/b",
        "https://example.org/a?q=1",
        "http://example.org/a",
        "https://other.example.org/a",
        "https://example.org:8443/a",
    }
    canonical = {canonicalize_url(url) for url in distinct}

    assert len(canonical) == len(distinct)
