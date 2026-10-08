"""Deterministic content normalization, before anything is chunked or identified.

Normalization makes equivalent content behave the same way without changing what it
says. It exists because two runs that reach the same text by different routes — a file
with Windows line endings and a paste of the same body, the same accent composed one way
and decomposed another — should produce the same chunks and therefore the same
identifiers, or an evaluation comparing runs is comparing formatting rather than
content.

It is deliberately shallow. It normalizes representation (line endings, Unicode form,
trailing whitespace, repeated blank lines) and never touches substance: no re-wrapping,
no case change, no word removed, no punctuation altered. A normalizer that "tidies"
prose would corrupt the very text an evidence chunk is meant to quote.

Two limits are stated rather than hidden.

* Blank-line handling does not understand Markdown fences, so blank lines inside a code
  block are collapsed the same as blank lines between paragraphs. This matches the
  chunker, which already splits on blank lines without parsing fences (ADR-0010); a
  fence-aware normalizer needs a parser, and there is none yet.
* Unicode normalization depends on the normalization algorithm shipped with the running
  interpreter. NFC is stable for ordinary text, but a code point whose decomposition
  changes between Unicode versions would normalize differently across Python builds.
  Reproducible evaluation must therefore pin the interpreter, exactly as it must pin
  everything else about the environment.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize_document"]


def normalize_document(text: str) -> str:
    """Return ``text`` in a deterministic, representation-normalized form.

    Applied, in order:

    * CRLF and lone CR line endings become ``\\n``;
    * the text is composed to Unicode NFC;
    * trailing spaces and tabs are removed from every line, but leading indentation is
      kept, because lists and code blocks depend on it;
    * a run of blank lines collapses to a single blank line, and blank lines at the very
      start and end are dropped;
    * nothing else changes: no re-wrapping, no case folding, no word removed.

    The function is idempotent — normalizing already-normalized text is a no-op — and
    total: every string has a normal form, so there is no input that raises.
    """
    with_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    composed = unicodedata.normalize("NFC", with_newlines)
    trimmed = [line.rstrip(" \t") for line in composed.split("\n")]
    return _collapse_blank_lines(trimmed).strip("\n")


def _collapse_blank_lines(lines: list[str]) -> str:
    """Join ``lines`` with runs of blank lines reduced to one blank line."""
    kept: list[str] = []
    for line in lines:
        if line == "" and (not kept or kept[-1] == ""):
            continue
        kept.append(line)
    return "\n".join(kept)
