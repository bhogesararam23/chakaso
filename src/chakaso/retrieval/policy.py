"""The fetch policy: the operational limits a web fetch must respect.

`docs/retrieval.md` described the security threats of fetching for a long time without
any corresponding limits. This is those limits, as one validated value object, so a bad
policy fails at construction rather than mid-fetch, and so the behaviour of a fetch is
reproducible from a record of the policy that produced it (ADR-0012).

The policy is deliberately narrow. It can only ever restrict the fetchable schemes to a
subset of ``http`` and ``https`` — a policy can never widen fetching to ``file``, ``ftp``
or anything else, because widening the fetch permission is exactly the thing ADR-0011
separated source identity from. Defaults are bounded and non-dangerous: no unlimited
response, no unlimited redirects, no missing timeout, and a user-agent that cannot smuggle
a header through a newline.

Values here are component defaults, not yet fields in the TOML configuration: nothing
reads a fetch policy from a file per run yet, so a configuration field would be one no
code consumes. A caller that needs different limits constructs a different policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

__all__ = ["MAX_ALLOWED_SCHEMES", "FetchPolicy"]

#: The only schemes a policy may ever permit. A policy may restrict to a subset of these,
#: never add to them.
MAX_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    """Validated limits for one fetcher: schemes, timeout, redirects, size, identity."""

    allowed_schemes: frozenset[str] = field(default_factory=lambda: MAX_ALLOWED_SCHEMES)
    timeout_seconds: float = 10.0
    max_redirects: int = 5
    max_response_bytes: int = 2 * 1024 * 1024
    user_agent: str = "chakaso/0.1 (local research fetcher)"

    def __post_init__(self) -> None:
        if not self.allowed_schemes:
            message = "a fetch policy must allow at least one scheme"
            raise ValueError(message)
        if not self.allowed_schemes <= MAX_ALLOWED_SCHEMES:
            offending = sorted(self.allowed_schemes - MAX_ALLOWED_SCHEMES)
            message = (
                f"a fetch policy may only allow http/https; these are not permitted: "
                f"{offending}. Widening the fetchable schemes is not a policy choice"
            )
            raise ValueError(message)
        if self.timeout_seconds <= 0:
            message = f"timeout_seconds must be positive, got {self.timeout_seconds}"
            raise ValueError(message)
        if self.max_redirects < 0:
            message = f"max_redirects must not be negative, got {self.max_redirects}"
            raise ValueError(message)
        if self.max_response_bytes < 1:
            message = f"max_response_bytes must be at least 1, got {self.max_response_bytes}"
            raise ValueError(message)
        if not self.user_agent or any(character in self.user_agent for character in "\r\n"):
            message = (
                "a user-agent must be non-empty and contain no carriage return or line "
                "feed, which could inject a header"
            )
            raise ValueError(message)

    def permits_scheme(self, scheme: str) -> bool:
        """Whether ``scheme`` is one this policy allows to be fetched."""
        return scheme in self.allowed_schemes
