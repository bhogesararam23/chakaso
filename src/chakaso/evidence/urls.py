"""URL canonicalization.

Canonicalization is not cosmetic here. Source identity is derived from the
canonical URL (ADR-0007), so the rule below is part of the identifier contract:
changing it changes every identifier derived from a URL, and old records stop
matching new ones.

The rule is deliberately conservative. It removes differences that are certainly
meaningless — a fragment, a default port, a tracking parameter — and leaves
everything else alone. Aggressively "tidying" URLs would merge distinct pages, and
a silently merged source is worse than a duplicate one, because a duplicate is
visible in the evidence pack and a merge is not.

Canonicalization requires no network access. It does not follow redirects, because
that would make identity depend on the state of a remote server at the moment of
the fetch, which is the opposite of reproducible.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

from chakaso.evidence.errors import UrlError

__all__ = ["ALLOWED_SCHEMES", "TRACKING_PARAMETERS", "canonicalize_url", "host_of"]

#: Only these schemes are fetchable. Everything else is refused rather than
#: ignored, because a scheme that reaches a different protocol is a security
#: question, not a normalization question.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Query parameters that identify a campaign rather than a document. Removing them
#: is what stops one page reached from three links becoming three sources.
TRACKING_PARAMETERS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
        "referrer",
        "source",
        "spm",
        "yclid",
        "_ga",
        "_gl",
    }
)


def canonicalize_url(url: str) -> str:
    """Return the canonical form of ``url``.

    Applied, in order:

    * surrounding whitespace is stripped, and the input must be non-empty;
    * the scheme must be ``http`` or ``https``, and is lowercased;
    * embedded credentials (``user:password@host``) are refused, because a URL is
      recorded in provenance and would otherwise store a secret;
    * the host is lowercased and a trailing root dot is removed;
    * a default port (80 for http, 443 for https) is removed;
    * the fragment is dropped, since it does not identify a document;
    * ``utm_*`` and known campaign parameters are dropped from the query;
    * remaining query parameters are sorted, so ordering cannot create two sources;
    * an empty query is removed;
    * a single trailing slash on a non-root path is removed.

    Raises:
        UrlError: the URL is empty, has no host, has a scheme other than http or
            https, contains credentials, or has a query string that cannot be
            parsed.
    """
    candidate = url.strip()
    if not candidate:
        message = "cannot canonicalize an empty URL"
        raise UrlError(message)

    parts = urlsplit(candidate)

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        message = (
            f"URL scheme {parts.scheme!r} is not supported. "
            f"Supported schemes: {', '.join(sorted(ALLOWED_SCHEMES))}."
        )
        raise UrlError(message)

    if parts.username or parts.password:
        message = (
            "URL contains embedded credentials. A canonical URL is recorded in "
            "source provenance and must not carry a secret."
        )
        raise UrlError(message)

    host = (parts.hostname or "").lower()
    if not host:
        message = f"URL has no host: {url!r}"
        raise UrlError(message)
    if host.endswith("."):
        # A trailing dot is a legal way to write a fully qualified name and
        # resolves identically, so it must not create a second source.
        host = host[:-1]

    port = parts.port
    if port is not None and (scheme, port) in {("http", 80), ("https", 443)}:
        port = None

    # An IPv6 literal must keep its brackets, or the rebuilt URL is not a URL:
    # `urlsplit` returns the address without them, and dropping them produces
    # `https://2001:db8::1:8443/a`, which parses as something entirely different.
    authority = f"[{host}]" if ":" in host else host
    netloc = authority if port is None else f"{authority}:{port}"

    path = parts.path
    # Percent-encode characters that are legal in a URL but ambiguous in text, and
    # leave existing escapes untouched so normalization is idempotent.
    path = quote(path, safe="/%:@!$&'()*+,;=~-._")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"

    try:
        query_pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=False)
    except ValueError as exc:
        message = f"URL query string cannot be parsed: {url!r}: {exc}"
        raise UrlError(message) from exc

    kept = [
        (name, value)
        for name, value in query_pairs
        if name.lower() not in TRACKING_PARAMETERS and not name.lower().startswith("utm_")
    ]
    query = "&".join(f"{name}={value}" for name, value in sorted(kept))

    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    """Return the lowercase host of an already canonical URL.

    This is the host, not a registrable domain. Reducing ``news.example.co.uk`` to
    ``example.co.uk`` needs the public suffix list, which is a dependency and a
    frequently-updated data file. Recording the host is exact and needs neither;
    the distinction is documented rather than papered over.
    """
    return (urlsplit(url).hostname or "").lower()
