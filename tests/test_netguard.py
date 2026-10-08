"""Tests for the destination screen (ADR-0012).

The screen blocks address literals that a fetch must never reach, and its documented
limit is that a hostname is only checked against known loopback/metadata names — the
tests pin both what it catches and what it deliberately does not pretend to catch.
"""

from __future__ import annotations

import pytest

from chakaso.retrieval.netguard import is_blocked_destination


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",  # loopback v4
        "127.1.2.3",  # whole loopback /8
        "10.0.0.1",  # private
        "192.168.0.1",  # private
        "172.16.0.1",  # private
        "169.254.169.254",  # link-local, the cloud metadata address
        "::1",  # loopback v6
        "fc00::1",  # unique local
        "fd12::34",  # unique local
        "fe80::1",  # link-local v6
    ],
)
def test_non_global_address_literals_are_blocked(host: str) -> None:
    assert is_blocked_destination(host) is True


def test_ipv6_zone_identifier_is_still_screened() -> None:
    assert is_blocked_destination("fe80::1%eth0") is True


@pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1", "2001:4860:4860::8888"])
def test_globally_routable_literals_are_allowed(host: str) -> None:
    assert is_blocked_destination(host) is False


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "metadata.google.internal"])
def test_known_loopback_and_metadata_names_are_blocked(host: str) -> None:
    assert is_blocked_destination(host) is True


@pytest.mark.parametrize("host", ["example.com", "sub.example.org"])
def test_an_unknown_hostname_is_not_blocked_by_name(host: str) -> None:
    # The documented limitation: a hostname that resolves to a private address is not
    # caught here, because a pure string check cannot know where DNS will send it. The
    # screen returns False and the caller owns the network environment.
    assert is_blocked_destination(host) is False
