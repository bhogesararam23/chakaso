"""Deterministic, meaning-preserving query normalization.

A query is normalized so the same question asked with different spacing or Unicode forms
plans the same way, and so a lexical retriever sees the words without terminal punctuation
that carries no lexical signal. Unlike document normalization
(`chakaso.retrieval.normalize`), a query is user input whose meaning must be preserved:
this collapses whitespace runs, composes to Unicode NFC, strips the ends, and removes a
trailing run of *terminal* punctuation (``? ! . ; : ,``) that a question mark or stray
period adds to the end of a turn. It changes no words, no case, and no interior
punctuation — "uses C++ in rankings" keeps its ``++``, and "What is X?" becomes "What is
X" only by dropping the trailing ``?``. The original text is always kept alongside the
normalized form (``QueryPlan`` stores both), so nothing a later reader needs is lost.

The function is total and idempotent: every string has a normal form, and normalizing an
already-normal form is a no-op.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize_query"]

#: Trailing characters trimmed because they signal sentence form, not retrieval content.
#: Only ever removed from the very end; interior punctuation is left untouched.
_TERMINAL_PUNCTUATION = "?!.;:,"


def normalize_query(text: str) -> str:
    """Return ``text`` deterministic-normalized for retrieval, preserving its words.

    Whitespace runs (including newlines and tabs) collapse to single spaces, the result is
    Unicode-NFC composed, the ends are stripped, and a trailing run of terminal punctuation
    is removed. Nothing interior is rewritten, and the transform never raises.
    """
    composed = unicodedata.normalize("NFC", text)
    collapsed = " ".join(composed.split())
    return collapsed.rstrip(_TERMINAL_PUNCTUATION)
