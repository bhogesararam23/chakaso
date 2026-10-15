"""Tests for the terminal-safety rendering primitive.

The property under test is narrow and must not be oversold: :func:`safe_for_terminal`
removes the characters that mean something to a terminal emulator rather than to a
reader — control bytes that can move the cursor or introduce escape sequences, and the
bidirectional controls that let stored text read differently from what it is — while
changing nothing about ordinary evidence text. These are structural guarantees, not a
claim that a model resists prompt injection.
"""

from __future__ import annotations

import unicodedata

from chakaso.core.text import safe_for_terminal


def test_an_empty_string_is_returned_unchanged() -> None:
    assert safe_for_terminal("") == ""


def test_ordinary_text_is_preserved_exactly() -> None:
    text = "The window closes in April; café naïve.\n\nSecond paragraph."
    assert safe_for_terminal(text) == text


def test_accents_and_emoji_are_not_treated_as_controls() -> None:
    # Category-based detection must not sweep in printable non-ASCII text.
    text = "naïve 東京 🎯 ✅"
    assert safe_for_terminal(text) == text


def test_newline_and_tab_are_kept() -> None:
    text = "line one\nline two\tcolumn"
    assert safe_for_terminal(text) == text


def test_the_escape_byte_is_neutralised_as_a_visible_marker() -> None:
    # ESC is the byte that introduces a terminal escape sequence; it must not survive,
    # and it must leave a trace rather than vanish silently.
    rendered = safe_for_terminal("\x1b[2J clear the screen")
    assert "\x1b" not in rendered
    assert rendered.startswith("\\x1b[2J clear the screen")


def test_carriage_return_cannot_overwrite_a_displayed_line() -> None:
    rendered = safe_for_terminal("innocent\rEVIL")
    assert "\r" not in rendered
    assert "EVIL" in rendered
    assert "\\x0d" in rendered


def test_every_c0_and_c1_control_except_newline_and_tab_is_escaped() -> None:
    for codepoint in range(0x00, 0x20):
        character = chr(codepoint)
        if character in "\n\t":
            continue
        rendered = safe_for_terminal(f"a{character}b")
        assert character not in rendered
        assert rendered == f"a\\x{codepoint:02x}b"
    for codepoint in range(0x80, 0xA0):
        assert chr(codepoint) not in safe_for_terminal(chr(codepoint))


def test_del_is_escaped() -> None:
    assert "\x7f" not in safe_for_terminal("a\x7fb")
    assert safe_for_terminal("\x7f") == "\\x7f"


def test_bidirectional_controls_are_escaped_even_though_they_are_not_cc() -> None:
    # Trojan-Source-style overrides are category Cf, so they need explicit handling.
    for codepoint in (0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069):
        assert unicodedata.category(chr(codepoint)) != "Cc"
        rendered = safe_for_terminal(f"dir{chr(codepoint)}ection")
        assert chr(codepoint) not in rendered
        assert f"\\u{codepoint:04x}" in rendered


def test_the_transform_is_idempotent_on_safe_output() -> None:
    # Re-rendering already-safe text must not double-escape the visible markers, since
    # a backslash is itself printable and is left alone.
    once = safe_for_terminal("hostile\x1bsequence")
    assert safe_for_terminal(once) == once


def test_a_fully_hostile_string_loses_every_control_byte() -> None:
    hostile = "\x1b]0;title\x07\x1b[31mred\x1b[0m \u202e scam \x00"
    rendered = safe_for_terminal(hostile)
    for character in rendered:
        assert unicodedata.category(character) != "Cc"
        assert character not in "\u202a\u202b\u202c\u202d\u202e"
