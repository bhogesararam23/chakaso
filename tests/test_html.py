"""Tests for the narrow HTML reader.

The reader is a text-and-structure extractor, not a browser, so the assertions are about
what it keeps and what it must drop: the document's text and heading structure survive,
scripts and styles do not, and entities are decoded.
"""

from __future__ import annotations

from chakaso.retrieval import normalize_document, parse_html


def test_title_and_headings_are_extracted() -> None:
    markup = (
        "<html><head><title>My Page</title></head>"
        "<body><h1>Intro</h1><p>Hello world.</p><h2>More</h2><p>Detail.</p></body></html>"
    )

    parsed = parse_html(markup)

    assert parsed.title == "My Page"
    assert "# Intro" in parsed.text
    assert "## More" in parsed.text
    assert "Hello world." in parsed.text


def test_scripts_and_styles_are_dropped() -> None:
    markup = (
        "<style>.x { color: red }</style><script>alert('take that')</script><p>Real content.</p>"
    )

    parsed = parse_html(markup)

    assert "alert" not in parsed.text
    assert "color" not in parsed.text
    assert "Real content." in parsed.text


def test_entities_are_decoded() -> None:
    parsed = parse_html("<p>Tom &amp; Jerry &lt;3</p>")

    assert "Tom & Jerry <3" in parsed.text


def test_paragraphs_become_separated_blocks() -> None:
    parsed = parse_html("<p>one</p><p>two</p>")

    assert parsed.text.split() == ["one", "two"]
    # The raw extraction leaves a blank line between blocks; the normalizer collapses it to
    # exactly one, so the chunker sees two paragraphs.
    assert normalize_document(parsed.text) == "one\n\ntwo"


def test_an_empty_document_yields_nothing() -> None:
    parsed = parse_html("")

    assert parsed.title == ""
    assert parsed.text.strip() == ""


def test_malformed_markup_does_not_raise() -> None:
    # HTMLParser tolerates markup a browser would repair; extraction must not crash on it.
    parsed = parse_html("<p>unclosed <b>bold")

    assert "unclosed" in parsed.text
