"""Count what a subword vocabulary spends its identifiers on, for a piece of text.

Every identifier is decoded and put in one class: ``document_end`` (the reserved separator), ``whitespace`` (decodes to
whitespace only, with ``single_space`` counted separately because it is the case a space-prefixed pre-tokenizer would
fold into the next word), ``punctuation_or_symbol`` (no letter, digit or underscore) and ``word_or_subword``. The
measurement is of the vocabulary on the text as given; it says nothing about a re-fitted vocabulary.

Usage:
    python scripts/measure_token_classes.py --vocabulary experiments/E2612/vocabulary-434debe82f949347.json \\
        --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \\
        --prefix prose- --max-characters 1500000
    python scripts/measure_token_classes.py --vocabulary V.json --text-file some.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.training.subword import RESERVED_DOCUMENT_END_ID, SubwordVocabulary

_SYMBOL_ONLY = re.compile(r"[^\w\s]+")
CLASSES = ("word_or_subword", "whitespace", "punctuation_or_symbol", "document_end")


def token_classes(vocabulary: SubwordVocabulary, text: str) -> dict[str, Any]:
    """Class counts, shares and characters per token for ``text`` under ``vocabulary``."""
    identifiers = vocabulary.encode(text)
    counts: Counter[str] = Counter(dict.fromkeys(CLASSES, 0))
    single_space = 0
    for identifier in identifiers:
        if identifier == RESERVED_DOCUMENT_END_ID:
            counts["document_end"] += 1
            continue
        piece = vocabulary.decode([identifier])
        if piece.strip() == "":
            counts["whitespace"] += 1
            single_space += piece == " "
        elif _SYMBOL_ONLY.fullmatch(piece):
            counts["punctuation_or_symbol"] += 1
        else:
            counts["word_or_subword"] += 1
    tokens = len(identifiers)
    return {
        "characters": len(text),
        "tokens": tokens,
        "characters_per_token": len(text) / tokens if tokens else 0.0,
        "counts": dict(counts),
        "single_space_tokens": single_space,
        "shares": {name: counts[name] / tokens if tokens else 0.0 for name in CLASSES},
        "single_space_share": single_space / tokens if tokens else 0.0,
    }


def _snapshot_text(path: Path, prefix: str, max_characters: int) -> str:
    documents = json.loads(path.read_text(encoding="utf-8"))["documents"]
    parts: list[str] = []
    total = 0
    for document in documents:
        if document["path"].startswith(prefix):
            parts.append(document["text"])
            total += len(document["text"])
            if total >= max_characters:
                break
    return "\n\n".join(parts)[:max_characters]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--vocabulary", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-file", type=Path)
    source.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--prefix", default="", help="with --snapshot: only documents under this path prefix"
    )
    parser.add_argument("--max-characters", type=int, default=1_500_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        vocabulary = SubwordVocabulary.load(args.vocabulary)
        text = (
            args.text_file.read_text(encoding="utf-8")
            if args.text_file is not None
            else _snapshot_text(args.snapshot, args.prefix, args.max_characters)
        )
        result = {
            "vocabulary_fingerprint": vocabulary.fingerprint,
            **token_classes(vocabulary, text),
        }
    except (OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
