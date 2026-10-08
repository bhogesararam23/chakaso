"""Tests for the source-reference abstraction (ADR-0011).

A source reference decides identity for kinds that are not web pages, so the
assertions here are about the properties that make that safe: the web canonical form
is unchanged, each kind produces a stable and mutually distinct canonical string, and
a reference that contradicts its own kind cannot be built.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from chakaso.evidence import (
    SourceKind,
    SourceReference,
    SourceReferenceError,
    UrlError,
    file_reference,
    text_reference,
    web_reference,
)

URL = "https://example.org/specification"


# ---------------------------------------------------------------------------
# Web references
# ---------------------------------------------------------------------------


def test_web_reference_canonicalizes_like_the_url_layer() -> None:
    reference = web_reference(f"{URL}?utm_source=newsletter#top")

    assert reference.kind is SourceKind.WEB
    assert reference.canonical == URL
    assert reference.locator == URL


def test_web_reference_rejects_a_non_fetchable_scheme() -> None:
    # canonicalize_url owns the http/https policy; a local or odd scheme never
    # reaches the web reference path.
    with pytest.raises(UrlError):
        web_reference("ftp://example.org/a")


# ---------------------------------------------------------------------------
# File references
# ---------------------------------------------------------------------------


def test_file_reference_is_a_file_uri_not_a_web_url(tmp_path: Path) -> None:
    document = tmp_path / "notes.md"
    document.write_text("body", encoding="utf-8")

    reference = file_reference(document)

    assert reference.kind is SourceKind.FILE
    assert urlsplit(reference.canonical).scheme == "file"
    assert reference.canonical.startswith("file:///")
    assert "example.org" not in reference.canonical


def test_file_reference_locator_is_the_resolved_path(tmp_path: Path) -> None:
    reference = file_reference(tmp_path / "notes.md")

    assert reference.locator == str((tmp_path / "notes.md").resolve())


def test_file_reference_is_deterministic_for_one_path(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"

    assert file_reference(path).canonical == file_reference(path).canonical
    assert file_reference(str(path)).canonical == file_reference(path).canonical


def test_file_reference_normalizes_the_windows_drive_case(tmp_path: Path) -> None:
    # On Windows the authority carries a drive letter whose case is cosmetic and
    # could otherwise split one document into two sources; elsewhere there is no
    # drive to normalize. Either way the canonical form must not hold an uppercase
    # drive letter, and one path must give one identity.
    canonical = file_reference(tmp_path / "notes.md").canonical
    drive = re.match(r"^file:///([A-Za-z]:)", canonical)
    if drive is not None:
        assert drive.group(1) == drive.group(1).lower()
    assert canonical == file_reference(tmp_path / "notes.md").canonical


def test_file_reference_percent_encodes_special_characters(tmp_path: Path) -> None:
    path = tmp_path / "a b ünïcode.md"

    canonical = file_reference(path).canonical

    # A space and a non-ASCII letter must be escaped, not carried literally, so the
    # canonical value stays a valid URI and one file keeps one identity.
    assert " " not in canonical
    assert "%20" in canonical
    assert "ü" not in canonical
    assert "%C3%BC" in canonical


def test_file_reference_resolves_a_relative_path_against_the_cwd() -> None:
    relative = file_reference("docs/spec.md")
    absolute = file_reference(Path.cwd() / "docs/spec.md")

    assert relative.canonical == absolute.canonical


# ---------------------------------------------------------------------------
# Text references
# ---------------------------------------------------------------------------


def test_text_reference_without_a_label_is_an_opaque_source() -> None:
    reference = text_reference()

    assert reference.kind is SourceKind.TEXT
    assert reference.canonical == "text:"
    assert reference.locator is None


def test_text_reference_label_distinguishes_two_passages() -> None:
    assert text_reference("appeals").canonical == "text:appeals"
    assert text_reference("appeals").canonical != text_reference("deadlines").canonical


# ---------------------------------------------------------------------------
# Kind separation and validation
# ---------------------------------------------------------------------------


def test_kinds_are_mutually_unreachable_as_strings() -> None:
    # One cannot be confused for another inside the identifier hash input, so
    # folding the three kinds into one formula cannot collide across them.
    web = web_reference("https://example.org/x").canonical
    file_ = file_reference(Path("x")).canonical
    text = text_reference("x").canonical

    assert not web.startswith(("file://", "text:"))
    assert not file_.startswith(("http://", "https://", "text:"))
    assert not text.startswith(("http://", "https://", "file://"))


def test_a_reference_cannot_contradict_its_kind() -> None:
    with pytest.raises(SourceReferenceError, match="must begin with"):
        SourceReference(kind=SourceKind.WEB, canonical="file:///tmp/x")


def test_an_empty_canonical_value_is_rejected() -> None:
    with pytest.raises(SourceReferenceError, match="non-empty"):
        SourceReference(kind=SourceKind.TEXT, canonical="")
