"""Safe web acquisition, as a boundary over an injectable transport.

The fetcher turns an `AcquisitionRequest` into an `AcquiredSource` under a `FetchPolicy`:
it checks the scheme, screens the destination, follows a bounded number of redirects while
re-checking every hop, and refuses a body larger than the limit. It acquires bytes and
nothing more — it does not parse them, does not decide what identifies the source, and does
not treat anything in the content as an instruction (ADR-0003, ADR-0012).

The actual network call is a ``Transport`` injected into the fetcher. Nothing in this
module, the library, the CLI or CI performs one by default; `make_urllib_transport` exists
for a caller who explicitly opts into fetching, and it is the only place the standard
library's HTTP machinery is touched. Testing the fetcher against a stand-in transport means
the scheme rule, the redirect bound, the size bound, the timeout mapping and the
destination screen are all exercised offline and deterministically.

Redirects are followed here, not by the transport, so that each hop's scheme and
destination are re-validated: a permitted public URL that redirects into a private network
is blocked, which a transport that silently follows redirects would not be.
"""

from __future__ import annotations

import http.client
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from chakaso.retrieval.errors import (
    AcquisitionError,
    BlockedDestinationError,
    FetchTimeoutError,
    InvalidSchemeError,
    NetworkFailureError,
    ResponseTooLargeError,
)
from chakaso.retrieval.netguard import is_blocked_destination
from chakaso.retrieval.policy import FetchPolicy

__all__ = [
    "AcquiredSource",
    "AcquisitionRequest",
    "Fetcher",
    "HttpFetcher",
    "Transport",
    "TransportResponse",
    "make_urllib_transport",
]

#: HTTP status codes that mean "the real document is at Location."
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class AcquisitionRequest:
    """A request to fetch one URL."""

    url: str


@dataclass(frozen=True, slots=True)
class AcquiredSource:
    """Bytes acquired from a URL, with the metadata needed for provenance.

    ``final_url`` is where the content actually came from after redirects, which may
    differ from the requested URL; both are kept so provenance is not lost. The content
    is raw bytes — decoding and parsing are a separate stage, so a fetcher cannot silently
    mislabel a binary as text.
    """

    requested_url: str
    final_url: str
    content: bytes
    media_type: str
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """One HTTP response, without redirect following, as the fetcher sees it."""

    status: int
    headers: Mapping[str, str]
    body: bytes


#: Performs exactly one request and returns its response. Injected so the fetcher's policy
#: logic is testable and no default code path reaches the network.
Transport = Callable[[str, FetchPolicy], TransportResponse]


@runtime_checkable
class Fetcher(Protocol):
    """The acquisition boundary: turn a request into acquired content under a policy."""

    def fetch(self, request: AcquisitionRequest, policy: FetchPolicy) -> AcquiredSource:
        """Fetch ``request`` honouring ``policy``.

        Raises:
            AcquisitionError subclasses: the fetch was refused or failed, with the reason
            named by the specific type.
        """
        ...


class HttpFetcher:
    """A `Fetcher` that enforces a `FetchPolicy` over an injected `Transport`."""

    def __init__(self, transport: Transport, *, now: Callable[[], datetime] | None = None) -> None:
        self._transport = transport
        self._now = now if now is not None else _utcnow

    def fetch(self, request: AcquisitionRequest, policy: FetchPolicy) -> AcquiredSource:
        url = request.url.strip()
        if not url:
            message = "cannot fetch an empty URL"
            raise AcquisitionError(message)

        visited: set[str] = set()
        redirects = 0

        while True:
            self._check_target(url, policy)
            visited.add(url)

            response = self._transport(url, policy)

            if response.status in _REDIRECT_STATUSES:
                location = self._header(response.headers, "location")
                if location is None:
                    # A redirect with no target is a dead end, not a document.
                    message = f"redirect with no Location header from {url}"
                    raise NetworkFailureError(message)
                next_url = urljoin(url, location).strip()
                redirects += 1
                if redirects > policy.max_redirects:
                    message = f"more than {policy.max_redirects} redirects fetching {request.url}"
                    raise AcquisitionError(message)
                if next_url in visited:
                    message = f"redirect loop fetching {request.url}"
                    raise AcquisitionError(message)
                url = next_url
                continue

            if len(response.body) > policy.max_response_bytes:
                message = (
                    f"response from {url} exceeds the {policy.max_response_bytes}-byte fetch limit"
                )
                raise ResponseTooLargeError(message)

            content_type = self._header(response.headers, "content-type") or ""
            return AcquiredSource(
                requested_url=request.url,
                final_url=url,
                content=response.body,
                media_type=content_type.split(";", 1)[0].strip().lower(),
                retrieved_at=self._now(),
            )

    def _check_target(self, url: str, policy: FetchPolicy) -> None:
        """Validate a URL's scheme and destination before it is fetched."""
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if not policy.permits_scheme(scheme):
            allowed = ", ".join(sorted(policy.allowed_schemes))
            message = f"scheme {parts.scheme!r} is not fetchable; allowed: {allowed}"
            raise InvalidSchemeError(message)

        host = parts.hostname
        if host is None:
            message = f"URL has no host to fetch: {url!r}"
            raise BlockedDestinationError(message)
        if is_blocked_destination(host):
            message = f"destination {host!r} is not publicly routable and is blocked"
            raise BlockedDestinationError(message)

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        for key, value in headers.items():
            if key.lower() == name:
                return value
        return None


class _NoRedirectHandler(HTTPRedirectHandler):
    """Stop urllib from following redirects, so the fetcher can re-validate each hop."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> None:
        # Returning None makes urllib raise HTTPError for the redirect, which the
        # transport turns into a TransportResponse the fetcher inspects.
        return None


def make_urllib_transport() -> Transport:
    """Return the default transport: a single, non-redirecting HTTP GET via the standard library.

    This is the only code that touches the network, and it is reached only when a caller
    explicitly builds it and hands it to an :class:`HttpFetcher`. Nothing in the library,
    the CLI or CI does so by default (ADR-0001, ADR-0012).
    """
    opener = build_opener(_NoRedirectHandler)

    def transport(url: str, policy: FetchPolicy) -> TransportResponse:
        request = Request(url, headers={"User-Agent": policy.user_agent})
        limit = policy.max_response_bytes + 1
        try:
            with opener.open(request, timeout=policy.timeout_seconds) as response:
                return TransportResponse(
                    status=int(response.status),
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=response.read(limit),
                )
        except HTTPError as error:
            # HTTPError is a subclass of URLError and must be matched first: a 4xx/5xx is
            # a delivered response, not a transport failure, and the fetcher inspects it.
            body = error.read(limit) if error.fp is not None else b""
            headers = (
                {key.lower(): value for key, value in error.headers.items()}
                if error.headers is not None
                else {}
            )
            return TransportResponse(status=int(error.code), headers=headers, body=body)
        except TimeoutError as error:
            # socket.timeout is an alias of TimeoutError on 3.11+, so this covers a
            # timeout raised during connect or during body read.
            message = f"timed out fetching {url!r}: {error}"
            raise FetchTimeoutError(message) from error
        except URLError as error:
            # urllib wraps the underlying cause in .reason; a timeout surfaces there too.
            if isinstance(error.reason, TimeoutError):
                message = f"timed out fetching {url!r}: {error.reason}"
                raise FetchTimeoutError(message) from error
            message = f"network failure fetching {url!r}: {error}"
            raise NetworkFailureError(message) from error
        except http.client.HTTPException as error:
            message = f"malformed HTTP response from {url!r}: {error}"
            raise NetworkFailureError(message) from error

    return transport
