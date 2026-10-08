"""Source references: what a source *is*, kept separate from whether it can be fetched.

Source identity was the canonical URL plus a content hash (ADR-0007). That names a
web page precisely and nothing else. A document on disk, a passage supplied directly
by a caller, and a generated fixture have no URL, and inventing one — a fake
``http://`` string — would make provenance claim a source was fetched from a web
server when it never was, and would put a fabricated URL in reach of the citation
layer (ADR-0003).

This module generalizes the *reference* half of identity. A :class:`SourceReference`
carries a :class:`SourceKind` and a canonical string, and the derivation formula of
ADR-0007 is unchanged: only its input widens from "canonical URL" to "canonical
reference". Because a web reference's canonical string is exactly
:func:`~chakaso.evidence.urls.canonicalize_url`'s output, every identifier already
derived from a URL stays identical.

The separation is deliberate and is the whole reason this module exists.
``canonicalize_url`` is a URL normalizer whose ``http``/``https`` check also acts as
the list of schemes a future fetcher may retrieve. A ``file`` or ``text`` reference
must not be routed through it, or identifying a local source would quietly widen what
the network layer is allowed to reach. Each kind builds its own canonical form, and
only ``web`` consults the URL machinery.

A reference owns identity and provenance and nothing else. It never opens a file and
never makes a request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from chakaso.evidence.errors import SourceReferenceError
from chakaso.evidence.urls import canonicalize_url

__all__ = [
    "SourceKind",
    "SourceReference",
    "file_reference",
    "require_consistent",
    "text_reference",
    "web_reference",
]


class SourceKind(StrEnum):
    """How a source was obtained, which decides how it is identified.

    ``WEB`` is content retrieved from a URL, ``FILE`` is a document read from a local
    path, and ``TEXT`` is content supplied directly with no persistent location. A
    ``dataset`` or ``generated`` kind is added here, with its own reference builder,
    when a real need for it arrives — not before.
    """

    WEB = "web"
    FILE = "file"
    TEXT = "text"


#: The prefix a canonical string must begin with for each kind. Keeping the markers
#: mutually unreachable means one kind can never be mistaken for another inside the
#: identifier hash input, and lets a record reject a ``kind``/value mismatch.
_KIND_MARKERS: Final[dict[SourceKind, tuple[str, ...]]] = {
    SourceKind.WEB: ("http://", "https://"),
    SourceKind.FILE: ("file://",),
    SourceKind.TEXT: ("text:",),
}

#: A ``file://`` URI's drive letter on Windows, for case-normalizing it. ``Path`` may
#: report ``C:`` or ``c:`` depending on the platform and the input, and a difference
#: that is purely cosmetic must not split one document into two sources.
_WINDOWS_DRIVE = re.compile(r"^file:///[A-Za-z](?=:)")


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A source's identity-defining reference: its kind, canonical form and location.

    ``canonical`` is the string folded into the source identifier. ``locator`` is a
    human-facing location for provenance and display — the canonical URL for a web
    source, the filesystem path for a file, the label for text — and is ``None`` when
    there is nothing to point a reader at.
    """

    kind: SourceKind
    canonical: str
    locator: str | None = None

    def __post_init__(self) -> None:
        require_consistent(self.kind, self.canonical)


def require_consistent(kind: SourceKind, canonical: str) -> None:
    """Raise unless ``canonical`` begins with the marker for ``kind``.

    Prevents an inconsistent record — a ``kind`` of ``FILE`` beside an ``http://``
    value — from reaching the identifier derivation, where the mismatch would be
    invisible and would name a source as something it is not.
    """
    if not canonical:
        message = "a source reference must have a non-empty canonical value"
        raise SourceReferenceError(message)
    if not canonical.startswith(_KIND_MARKERS[kind]):
        message = (
            f"a {kind.value} reference's canonical value must begin with one of "
            f"{_KIND_MARKERS[kind]!r}, got {canonical!r}"
        )
        raise SourceReferenceError(message)


def web_reference(url: str) -> SourceReference:
    """Return a reference to a web source, identified by its canonical URL.

    The canonical form is :func:`~chakaso.evidence.urls.canonicalize_url` exactly, so
    identifiers of web sources are unchanged by this abstraction. A URL that is not
    fetchable web content — an unsupported scheme, no host, embedded credentials — is
    rejected by :func:`canonicalize_url` with a :class:`~chakaso.evidence.errors.UrlError`.
    """
    canonical = canonicalize_url(url)
    return SourceReference(kind=SourceKind.WEB, canonical=canonical, locator=canonical)


def file_reference(path: str | Path) -> SourceReference:
    """Return a reference to a local document, identified by a ``file`` URI.

    The path is expanded (``~``), resolved to an absolute form, and rendered as an
    RFC 8089 ``file`` URI with the standard library. It never passes through
    :func:`canonicalize_url`: a local file is not a fetchable web URL, and identifying
    it must not be made to depend on the fetcher's scheme policy.

    A relative ``path`` is resolved against the current working directory, so an
    absolute path gives the stablest identity. Whether the file exists or can be read
    is not this function's concern; ingestion checks that when it opens the file.
    """
    resolved = Path(path).expanduser().resolve()
    canonical = _WINDOWS_DRIVE.sub(lambda match: match.group(0).lower(), resolved.as_uri())
    return SourceReference(kind=SourceKind.FILE, canonical=canonical, locator=str(resolved))


def text_reference(label: str | None = None) -> SourceReference:
    """Return a reference to content supplied directly, with no persistent location.

    Identity is the content, which the caller folds in through
    :func:`~chakaso.core.identifiers.derive_source_id`; the reference itself only marks
    the kind, and an optional ``label`` distinguishes two passages that would otherwise
    be the same opaque source. The label is caller-supplied and trusted, not a URL.
    """
    canonical = f"text:{label}" if label else "text:"
    return SourceReference(kind=SourceKind.TEXT, canonical=canonical, locator=label)
