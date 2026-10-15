"""Making untrusted text safe to write to a terminal.

Evidence text and model output are adversarial input until proven otherwise: a
retrieved document, a fetched page or a generated reply can carry characters that
mean something to a terminal emulator rather than to a reader. An escape byte can
move the cursor, clear a line, set the window title or — in some emulators — feed
input back; the bidirectional embedding and isolate controls can reorder or hide
text so what is displayed is not what was stored (the "Trojan Source" family). None
of that is a language-model concern the project can defensibly claim to solve, but
the *application* can refuse to hand raw control bytes to a terminal, which is a
structural guarantee that is testable without a model.

:func:`safe_for_terminal` neutralises that class of characters in text destined for
a human-readable terminal. It is deliberately narrow and honest about what it is:

* it keeps substance — no word, byte of ordinary text, newline or tab changes — and
  alters representation only for control and bidirectional-format characters;
* it renders a removed character as a *visible escape* (``\\x1b``, ``\\u202e``) rather
  than dropping it silently, because a hostile document that quietly loses its escape
  byte hides the very fact that it carried one. The project's rule that a bad input is
  a hard, visible failure and never a silent truncation is inherited here.

It is the display boundary, not a data transform: JSON output must not be passed
through it, because a serializer already escapes control bytes into a valid document
and rewriting them would corrupt the data it exists to describe. It is also not a
claim that a terminal cannot show anything dangerous — a reader is still responsible
for their own terminal's capabilities — only that Chakaso does not emit control or
reordering characters on a human-readable path.
"""

from __future__ import annotations

import unicodedata

__all__ = ["safe_for_terminal"]

#: Whitespace that is both ordinary in evidence text and harmless to render.
_KEEP_WHITESPACE = frozenset({"\n", "\t"})

#: Bidirectional embedding, override and isolate controls. They are category Cf, not
#: Cc, so the general control sweep below would miss them, yet they are exactly the
#: characters that let stored text read differently from what it is.
_BIDI_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _escape(codepoint: int) -> str:
    """Render one code point as the shortest unambiguous backslash escape."""
    if codepoint < 0x100:
        return f"\\x{codepoint:02x}"
    if codepoint < 0x10000:
        return f"\\u{codepoint:04x}"
    return f"\\U{codepoint:08x}"


def safe_for_terminal(text: str) -> str:
    """Return ``text`` with control and bidirectional-format characters made inert.

    Every Unicode control character (general category ``Cc`` — the C0 block, ``DEL``
    and the C1 block) except newline and tab, and every bidirectional embedding,
    override and isolate control, is replaced by a visible escape sequence. All other
    characters are passed through unchanged, so ordinary evidence — including accents,
    emoji and the surrounding layout a newline provides — is preserved exactly.

    The function is total: it accepts any string and never raises, because a display
    helper that fails on hostile input is the failure mode it exists to prevent.
    """
    if not text:
        return text

    pieces: list[str] = []
    for character in text:
        if character in _KEEP_WHITESPACE:
            pieces.append(character)
            continue
        if unicodedata.category(character) == "Cc" or character in _BIDI_CONTROLS:
            pieces.append(_escape(ord(character)))
        else:
            pieces.append(character)
    return "".join(pieces)
