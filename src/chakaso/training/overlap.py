"""Near-duplicate leakage between a training side and held-out text, measured by shared word n-grams.

A whole-work split (ADR-0042) guarantees that no *work* is on both sides. It cannot guarantee that no *text* is:
two catalogue entries can be two editions of one book, an anthology can contain a novel that is also a work in its
own right, and a translation reuses a source's sentences. Exact deduplication by document digest (the split's own
screen) misses all of these, because the bodies differ by a header, a preface or a chapter boundary.

This module measures that second kind of leak. Every held-out text is cut into overlapping word n-grams, each hashed
to 64 bits, and the share of them that also occur somewhere in the training text is reported per held-out unit. A
unit sharing most of its n-grams with training is not held out in any sense a reader would accept, whatever its
catalogue name says.

The measure is deliberately plain: lower-cased word runs, a fixed n, an exact-match test. It has no threshold of
its own — what counts as contaminated is the record's decision — and it reports what it saw, including units too
short to contain a single n-gram.

Requires the optional ``[train]`` extra (``numpy``).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

__all__ = ["overlap_report", "word_ngram_hashes"]

_WORD = re.compile(r"\w+")


def word_ngram_hashes(text: str, n: int = 8) -> np.ndarray:
    """Sorted unique 64-bit hashes of every ``n``-word window of ``text``, lower-cased.

    BLAKE2b with an 8-byte digest is used instead of Python's ``hash`` so the same text gives the same hashes in
    every process; a 64-bit hash over tens of millions of windows makes a chance collision a ~1e-5 event, which is
    far below any share a record would act on.

    Raises:
        ValueError: ``n`` is not a positive integer.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        message = f"n must be a positive integer, got {n!r}"
        raise ValueError(message)
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    hashes = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    " ".join(words[index : index + n]).encode(), digest_size=8
                ).digest(),
                "big",
            )
            for index in range(len(words) - n + 1)
        ),
        dtype=np.uint64,
        count=len(words) - n + 1,
    )
    return np.unique(hashes)


def overlap_report(
    training_texts: Iterable[str],
    held_out: Mapping[str, str],
    *,
    n: int = 8,
) -> dict[str, Any]:
    """The share of each held-out unit's ``n``-grams that also occur in the training texts.

    Args:
        training_texts: Every training document's text (order is irrelevant).
        held_out: Unit name (a work, a document) to its text.
        n: Words per window.

    Returns:
        ``units`` (per unit: windows counted, windows found in training, and their ratio — ``None`` for a unit
        shorter than ``n`` words), ``training_windows`` (distinct windows on the training side), and ``pooled``
        (all held-out windows together, which weights long units more, as a scored sample would).
    """
    parts = [word_ngram_hashes(text, n) for text in training_texts]
    training = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)

    units: dict[str, dict[str, Any]] = {}
    shared_total = window_total = 0
    for name, text in held_out.items():
        hashes = word_ngram_hashes(text, n)
        if hashes.size == 0:
            units[name] = {"windows": 0, "shared": 0, "ratio": None}
            continue
        shared = int(np.isin(hashes, training, assume_unique=True).sum())
        units[name] = {
            "windows": int(hashes.size),
            "shared": shared,
            "ratio": round(shared / hashes.size, 6),
        }
        shared_total += shared
        window_total += int(hashes.size)
    return {
        "n": n,
        "training_windows": int(training.size),
        "units": units,
        "pooled": {
            "windows": window_total,
            "shared": shared_total,
            "ratio": None if window_total == 0 else round(shared_total / window_total, 6),
        },
    }
