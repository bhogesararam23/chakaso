"""Tests for the HTTP fetcher, driven entirely through an injected transport.

No test here touches the network: the transport is a script, which is precisely why the
scheme rule, the destination screen, the redirect bound (and re-validation of each hop),
the size limit and the media-type handling can all be exercised offline and
deterministically (ADR-0012). A transport that must not be called is asserted not to be.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from chakaso.retrieval import (
    AcquisitionError,
    AcquisitionRequest,
    BlockedDestinationError,
    Fetcher,
    FetchPolicy,
    HttpFetcher,
    InvalidSchemeError,
    NetworkFailureError,
    ResponseTooLargeError,
    TransportResponse,
)

RETRIEVED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


class _ScriptedTransport:
    """Returns queued responses in order and records every URL it was asked for."""

    def __init__(self, *responses: TransportResponse | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, url: str, policy: FetchPolicy) -> TransportResponse:
        self.calls.append(url)
        if not self._responses:
            message = "transport called more times than scripted"
            raise AssertionError(message)
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok(body: bytes = b"hello", **headers: str) -> TransportResponse:
    lowered = {key.lower(): value for key, value in headers.items()}
    return TransportResponse(status=200, headers=lowered, body=body)


def redirect(location: str) -> TransportResponse:
    return TransportResponse(status=302, headers={"location": location}, body=b"")


def fetcher(*responses: TransportResponse | Exception, **policy_field: object):
    transport = _ScriptedTransport(*responses)
    policy = FetchPolicy(**policy_field)  # type: ignore[arg-type]
    service = HttpFetcher(transport, now=lambda: RETRIEVED)
    return service, policy, transport


# ---------------------------------------------------------------------------
# A successful fetch
# ---------------------------------------------------------------------------


def test_fetch_returns_bytes_and_provenance() -> None:
    service, policy, transport = fetcher(ok(b"page body", **{"content-type": "text/html"}))

    source = service.fetch(AcquisitionRequest(url="https://example.org/a"), policy)

    assert source.requested_url == "https://example.org/a"
    assert source.final_url == "https://example.org/a"
    assert source.content == b"page body"
    assert source.media_type == "text/html"
    assert source.retrieved_at == RETRIEVED
    assert transport.calls == ["https://example.org/a"]


def test_media_type_drops_the_charset_parameter() -> None:
    service, policy, _ = fetcher(ok(b"x", **{"content-type": "text/html; charset=utf-8"}))

    source = service.fetch(AcquisitionRequest(url="https://example.org/a"), policy)

    assert source.media_type == "text/html"


def test_headers_are_matched_case_insensitively() -> None:
    service, policy, _ = fetcher(
        TransportResponse(status=200, headers={"CONTENT-Type": "text/plain"}, body=b"ok")
    )

    source = service.fetch(AcquisitionRequest(url="https://example.org/a"), policy)

    assert source.media_type == "text/plain"


# ---------------------------------------------------------------------------
# Scheme and destination
# ---------------------------------------------------------------------------


def test_a_disallowed_scheme_is_refused_without_a_request() -> None:
    service, policy, transport = fetcher()

    with pytest.raises(InvalidSchemeError):
        service.fetch(AcquisitionRequest(url="ftp://example.org/a"), policy)

    assert transport.calls == []


def test_https_only_policy_refuses_http() -> None:
    service, policy, transport = fetcher(allowed_schemes=frozenset({"https"}))

    with pytest.raises(InvalidSchemeError):
        service.fetch(AcquisitionRequest(url="http://example.org/a"), policy)

    assert transport.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/secret",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.5/internal",
        "http://[::1]/loopback",
        "http://localhost/private",
    ],
)
def test_a_private_or_metadata_destination_is_blocked_before_any_request(url: str) -> None:
    service, policy, transport = fetcher()

    with pytest.raises(BlockedDestinationError):
        service.fetch(AcquisitionRequest(url=url), policy)

    assert transport.calls == []


def test_a_url_with_no_host_is_blocked() -> None:
    service, policy, transport = fetcher()

    with pytest.raises(BlockedDestinationError):
        service.fetch(AcquisitionRequest(url="http:///no-host"), policy)

    assert transport.calls == []


def test_an_empty_url_is_refused() -> None:
    service, policy, _ = fetcher()

    with pytest.raises(AcquisitionError, match="empty URL"):
        service.fetch(AcquisitionRequest(url="   "), policy)


# ---------------------------------------------------------------------------
# Redirects, re-validated at every hop
# ---------------------------------------------------------------------------


def test_a_redirect_updates_the_final_url() -> None:
    service, policy, transport = fetcher(redirect("https://example.org/final"), ok(b"landed"))

    source = service.fetch(AcquisitionRequest(url="https://example.org/start"), policy)

    assert source.final_url == "https://example.org/final"
    assert source.content == b"landed"
    assert transport.calls == ["https://example.org/start", "https://example.org/final"]


def test_a_relative_redirect_is_resolved_against_its_base() -> None:
    service, policy, _ = fetcher(redirect("/moved"), ok(b"here"))

    source = service.fetch(AcquisitionRequest(url="https://example.org/start"), policy)

    assert source.final_url == "https://example.org/moved"


def test_a_redirect_into_a_private_network_is_blocked_mid_chain() -> None:
    service, policy, transport = fetcher(redirect("http://127.0.0.1/secret"))

    with pytest.raises(BlockedDestinationError):
        service.fetch(AcquisitionRequest(url="https://example.org/start"), policy)

    # The first hop was fetched; the private target is screened before its request.
    assert transport.calls == ["https://example.org/start"]


def test_too_many_redirects_is_an_error() -> None:
    service, policy, _ = fetcher(
        redirect("https://example.org/1"),
        redirect("https://example.org/2"),
        redirect("https://example.org/3"),
        max_redirects=2,
    )

    with pytest.raises(AcquisitionError, match="more than 2 redirects"):
        service.fetch(AcquisitionRequest(url="https://example.org/0"), policy)


def test_a_redirect_loop_is_broken() -> None:
    service, policy, _ = fetcher(redirect("https://example.org/same"))

    with pytest.raises(AcquisitionError, match="redirect loop"):
        service.fetch(AcquisitionRequest(url="https://example.org/same"), policy)


def test_a_redirect_with_no_location_is_a_network_failure() -> None:
    service, policy, _ = fetcher(TransportResponse(status=302, headers={}, body=b""))

    with pytest.raises(NetworkFailureError, match="no Location"):
        service.fetch(AcquisitionRequest(url="https://example.org/a"), policy)


# ---------------------------------------------------------------------------
# Size and transport errors
# ---------------------------------------------------------------------------


def test_a_response_larger_than_the_limit_is_refused() -> None:
    service, policy, _ = fetcher(ok(b"x" * 100), max_response_bytes=50)

    with pytest.raises(ResponseTooLargeError):
        service.fetch(AcquisitionRequest(url="https://example.org/a"), policy)


def test_a_transport_failure_propagates_unchanged() -> None:
    # The fetcher does not swallow a transport's typed failure into a generic one.
    service, policy, _ = fetcher(NetworkFailureError("connection refused"))

    with pytest.raises(NetworkFailureError, match="connection refused"):
        service.fetch(AcquisitionRequest(url="https://example.org/a"), policy)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_http_fetcher_satisfies_the_fetcher_protocol() -> None:
    transport: Callable = _ScriptedTransport(ok(b"x"))
    fetcher_instance: Fetcher = HttpFetcher(transport)

    assert isinstance(fetcher_instance, Fetcher)
