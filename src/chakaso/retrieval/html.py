"""A narrow HTML-to-text extractor — a reader, not a browser.

Web pages must become text before the normalization and chunking stages can turn them into
evidence. This module does exactly that and nothing more, using the standard library's
`HTMLParser`: it pulls readable text, captures the `<title>`, and maps headings to the
Markdown ATX form (`## …`) so that the section-bounded chunker (ADR-0010) works on web
content the same way it works on Markdown files.

It explicitly does *not*: execute JavaScript, apply CSS, fetch subresources, or heuristically
strip navigation, ads or cookie banners. That boilerplate removal is the job of a real
document processor, which this deliberately is not — the goal is honest extraction of the
document's text and structure, not a claim to reconstruct every website (ADR-0012). What it
drops is only what must never be treated as content: `<script>`, `<style>`, `<noscript>`,
`<template>` and `<svg>`.

Everything produced here is *data*. Text found inside a page — including a line that reads
like an instruction to the system — is carried through as the page's content and is never
interpreted. Identity is assigned later from the validated canonical URL, never from
anything the page says about itself (ADR-0003).
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

__all__ = ["ParsedHtml", "parse_html"]

#: Elements whose text is never content, so they are skipped entirely.
_SKIP = frozenset({"script", "style", "noscript", "template", "svg"})

#: Heading level to its Markdown marker, so the chunker sees section boundaries.
_HEADINGS = {f"h{i}": "#" * i for i in range(1, 7)}

#: Elements that begin or end a block; a blank line is emitted around them so that the
#: normalizer and paragraph splitter see real boundaries rather than run-together text.
_BLOCKS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "dl",
        "dd",
        "dt",
        "fieldset",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)


@dataclass(frozen=True, slots=True)
class ParsedHtml:
    """The text extracted from one document and its declared title.

    ``title`` is the `<title>` element's text as extracted, not as generated; it is
    recorded as provenance and is never treated as evidence of what the page says.
    """

    title: str
    text: str


class _Extractor(HTMLParser):
    """Collects visible text with Markdown headings, skipping non-content elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title: list[str] = []
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, _attrs: object) -> None:
        if tag in _SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _HEADINGS:
            self._parts.append(f"\n\n{_HEADINGS[tag]} ")
        elif tag in _BLOCKS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _HEADINGS or tag in _BLOCKS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title.append(data)
        else:
            self._parts.append(data)

    def result(self) -> ParsedHtml:
        return ParsedHtml(title="".join(self._title).strip(), text="".join(self._parts))


def parse_html(markup: str) -> ParsedHtml:
    """Extract readable text (with Markdown headings) and the title from an HTML document.

    Malformed HTML is tolerated: `HTMLParser` does not raise on a document that browsers
    would repair, and text is collected as it is encountered. The result is raw enough that
    it should be passed through :func:`~chakaso.retrieval.normalize.normalize_document`
    before chunking, which is exactly what web ingestion does.
    """
    extractor = _Extractor()
    extractor.feed(markup)
    extractor.close()
    return extractor.result()
