"""Tests for the fetch policy value object (ADR-0012).

A policy is where the fetch limits live, so the tests are about what it refuses: a policy
must not be able to widen fetching beyond http/https, and every limit must be positive and
header-safe.
"""

from __future__ import annotations

import pytest

from chakaso.retrieval import MAX_ALLOWED_SCHEMES, FetchPolicy


def test_defaults_are_bounded_and_fetch_https() -> None:
    policy = FetchPolicy()

    assert policy.allowed_schemes == MAX_ALLOWED_SCHEMES
    assert policy.permits_scheme("https")
    assert policy.permits_scheme("http")
    assert policy.timeout_seconds > 0
    assert policy.max_redirects >= 0
    assert policy.max_response_bytes >= 1


def test_a_policy_may_restrict_to_https_only() -> None:
    policy = FetchPolicy(allowed_schemes=frozenset({"https"}))

    assert policy.permits_scheme("https")
    assert not policy.permits_scheme("http")


def test_a_policy_can_never_widen_beyond_http_https() -> None:
    # The whole point of ADR-0011/ADR-0012: fetching stays bounded to web schemes.
    with pytest.raises(ValueError, match="only allow http/https"):
        FetchPolicy(allowed_schemes=frozenset({"https", "file"}))


def test_an_empty_scheme_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one scheme"):
        FetchPolicy(allowed_schemes=frozenset())


@pytest.mark.parametrize("timeout", [0.0, -1.0])
def test_timeout_must_be_positive(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        FetchPolicy(timeout_seconds=timeout)


def test_negative_redirect_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_redirects"):
        FetchPolicy(max_redirects=-1)


def test_zero_redirects_is_allowed() -> None:
    # A policy that follows no redirects is valid and meaningful.
    assert FetchPolicy(max_redirects=0).max_redirects == 0


def test_response_floor_below_one_byte_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_response_bytes"):
        FetchPolicy(max_response_bytes=0)


@pytest.mark.parametrize("agent", ["", "chakaso/0.1\r\nX-Injected: yes", "a\nb"])
def test_a_user_agent_that_could_inject_a_header_is_rejected(agent: str) -> None:
    with pytest.raises(ValueError, match="user-agent"):
        FetchPolicy(user_agent=agent)
