"""Destination screening for the fetcher — bounded, and honest about the bound.

A naive fetch will happily be pointed at a loopback service, a private network segment, or
a cloud metadata address (``169.254.169.254``). Screening the destination before the
request blocks those, for *address literals*.

This is a real but partial control, and saying so is the point (ADR-0012). It inspects the
host component of the URL; a hostname that *resolves* to a private address defeats it,
because a pure pre-connect string check cannot know where DNS will send the connection.
Closing that gap requires connecting to a pre-resolved, re-validated IP, which belongs in
the transport, not in a function like this one, and is not implemented here. Until it is,
fetching is opt-in, off every default path, and the caller owns the network environment it
runs in.

Hostnames are checked only against a small set of well-known loopback and metadata names;
everything that is not an IP literal is otherwise allowed to reach the transport, where a
real deployment is expected to have imposed a stronger boundary.
"""

from __future__ import annotations

import ipaddress
from typing import Final

__all__ = ["is_blocked_destination"]

#: Hostnames that mean "this machine" or a metadata endpoint regardless of what DNS says.
_BLOCKED_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.google.internal",
    }
)


def is_blocked_destination(host: str) -> bool:
    """Whether a URL's host component names a destination the fetch policy refuses.

    ``host`` is the bare host, brackets and any IPv6 zone identifier removed. An IP
    literal is refused unless it is globally routable, which covers loopback, private,
    link-local (including the metadata address), unique-local, reserved and multicast
    ranges in both address families. A non-literal hostname is refused only if it matches
    a known loopback/metadata name; a hostname that resolves to a private address is not
    caught here — that limitation is documented, not hidden.
    """
    candidate = host.lower().strip("[]")
    # An IPv6 zone identifier (`fe80::1%eth0`) is not part of the address; drop it.
    candidate = candidate.split("%", 1)[0]

    if candidate in _BLOCKED_HOSTNAMES:
        return True

    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        # Not an IP literal. Without resolving — which this function must not do, or it
        # would double as a network dependency — a hostname other than the known names is
        # left to the transport.
        return False

    return not address.is_global
