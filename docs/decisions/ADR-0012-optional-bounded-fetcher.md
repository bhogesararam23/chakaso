# ADR-0012: The fetcher is an optional, explicitly-bounded adapter

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0011](ADR-0011-typed-source-references.md)

## Context

The pipeline needs to be able to acquire web content, but several decisions were
deliberately held back until the local substrate existed. [ADR-0001](ADR-0001-local-first-runtime.md)
forbids a runtime that requires a commercial inference or search API, and permits an
external provider only as an optional adapter behind an existing interface.
[ADR-0011](ADR-0011-typed-source-references.md) separated source *identity* from fetch
*permission* and left `canonicalize_url` as the normalizer and the http/https policy for
web references specifically.

The security posture in `docs/retrieval.md` was, until now, a list of threats with no
corresponding limits (K-005): no maximum response size, no redirect bound, no timeout, no
identification. And SSRF — a URL that points the fetcher at a private network, a loopback
service or a cloud metadata address — is a real class of harm that a naive `urlopen` invites.

## Problem

How should Chakaso fetch web content without (a) making the runtime depend on the network
or a commercial provider, (b) turning URL normalization into the fetch security policy, or
(c) pretending that string checks on a hostname are a complete SSRF defence?

## Options considered

- **`urllib.request.urlopen(url)` directly.** One line, and wrong: unbounded response size,
  default redirect following with no limit, no timeout, no scheme policy, no host
  policy, and a caller cannot test it offline.
- **A third-party HTTP client.** Rejected under the dependency discipline: the standard
  library can honour the policy, and an added dependency must solve a problem the
  standard library cannot.
- **Treat `canonicalize_url`'s scheme check as the fetch permission.** Rejected: it is
  exactly the coupling ADR-0011 removed. Normalization and security policy are different
  responsibilities even when they happen to list the same schemes today.
- **A fetcher behind an explicit interface, driven by a validated policy, with the actual
  transport injectable so it can be tested offline and without a real network stack.**
  Chosen.

## Decision

Introduce a fetcher boundary in `chakaso.retrieval`:

- `Fetcher` is a protocol: given an `AcquisitionRequest` and a `FetchPolicy`, return an
  `AcquiredSource`. `HttpFetcher` is the one implementation, and it reaches the network
  only through an injectable transport, so no default code path touches the network and
  every test runs against a stand-in.
- `FetchPolicy` is a typed, validated value object carrying the operational limits that
  were missing: allowed schemes, a connect/read timeout, a maximum redirect count, a
  maximum response size, and a user-agent string. Every field has a bounded, non-dangerous
  default. The fetcher enforces them.
- `canonicalize_url` stays the URL normalizer and the web-scheme rule; it is not consulted
  for network security and does not become one.
- Acquisition produces an `AcquiredSource` (bytes, content type, final URL, retrieved
  timestamp). Turning it into evidence is a *separate* step: an HTML parser extracts text
  and a title, the content is normalized and chunked through the existing ingestion
  pipeline, and identity comes from the validated canonical URL — the fetcher never
  assigns an identifier and never treats page text as instructions.

### Network safety is bounded, not complete, and says so

The fetcher refuses the obvious hostile targets: it rejects any URL whose host is a
loopback, private, link-local or otherwise non-globally-routable literal (including the
cloud metadata address), and it refuses schemes other than the policy's allowed set. This
is a real, tested boundary for literal addresses.

It is explicitly **not** a complete SSRF defence. A hostname that resolves to a private
address defeats a pre-connect string check, and closing that gap properly means connecting
to a pre-resolved, re-validated IP — an operation that must live inside the transport, not
in a pure function. That is not implemented here. The honest posture:

- The fetcher is opt-in and off every default path. Nothing in the library, the CLI or CI
  invokes it against a live network. `chakaso retrieve` reads local files only.
- The default runtime still requires no network, which keeps ADR-0001 true.
- Callers that enable fetching are responsible for running it in an environment where a
  DNS-rebinding target is not reachable, until IP-pinning lands in a transport.

## Reasoning

- **The boundary is where the tests live.** Making the transport injectable means the
  scheme check, redirect limit, size bound, timeout, content-type handling and the host
  blocklist are all exercised offline and deterministically, without a server and without
  the internet.
- **A policy is a value, not a code path.** Putting the limits in one validated object
  means a bad value fails at construction, and the fetcher's behaviour is reproducible from
  a record of the policy it used.
- **Separation is preserved.** The fetcher acquires bytes; a parser produces text; the
  ingestion pipeline assigns identity. Fetched text is data, never an instruction, and the
  identity of a fetched source comes from the validated canonical URL, not the page's claim
  about itself (ADR-0003).

## Trade-offs

- **The host blocklist is best-effort.** It stops the common cases and is documented as
  insufficient against DNS rebinding, rather than oversold.
- **No robots, no rate limiting, no cache coherency against live servers yet.** A simple
  in-memory cache keyed by canonical URL avoids re-fetching within a process; honest
  politeness to real operators (robots parsing, per-host rate limits) is deferred and noted
  rather than claimed.
- **An extra indirection layer** for a component that is not on the default path. Accepted:
  the indirection is what makes it testable and replaceable, and what keeps the runtime
  offline.

## Rejected alternatives

`urlopen` with no policy, a third-party HTTP dependency, and folding fetch permission into
`canonicalize_url` were each rejected — for unbounded behaviour, for dependency cost, and
for re-creating the identity/permission coupling ADR-0011 removed.

## Consequences

- `chakaso.retrieval` gains `policy` (FetchPolicy), `acquire` (Fetcher, AcquisitionRequest,
  AcquiredSource, HttpFetcher), `cache` (an in-memory cache), and `html` (a narrow HTML
  text extractor).
- A new typed acquisition error family distinguishes invalid scheme, blocked destination,
  timeout, response-too-large, unsupported content type, decoding and network failure.
- K-005 (no fetch policy) is addressed: the limits exist, are enforced and are tested. The
  remaining DNS-rebinding gap is recorded as an explicit limitation, not silently closed.
- A query planner that decides *which* URLs to fetch is still out of scope and still an
  open question; this records *how* a named URL is fetched safely, not *whether* to.
