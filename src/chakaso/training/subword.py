"""A byte-pair subword vocabulary trained by this project, on this project's own text.

The character vocabulary in :mod:`chakaso.training.tokenizer` was the right first tokenizer: no external
artifact, every token a readable character, and a gradient path small enough to verify scalar by scalar.
It also fixes the unit of learning to one character, so a model must spend capacity on spelling while a
context window of four characters reaches barely past a word. A subword tokenizer is the step that makes a
Transformer worth training, and the roadmap treats the tokenizer as a research variable rather than a
borrowed component: vocabulary size and segmentation rule decide how much text fits in a context and how
much of the corpus one token covers ([`docs/training.md`](../../../docs/training.md)).

Choices, and why:

* **Byte-level base.** The 256 UTF-8 byte values are the atomic symbols, so encoding is *total*: no
  character is ever unknown, and the lossy ``UNK`` behaviour of the character vocabulary disappears.
  Decoding is exact except where a caller deliberately truncates a byte sequence mid-codepoint.
* **A named pre-tokenizer.** Text is cut into runs of word characters (letters, digits, underscore), runs
  of whitespace, and single other symbols; merges are only ever learned and applied *within* a piece. That
  keeps word boundaries visible in the vocabulary instead of letting a merge glue an arbitrary suffix of
  one word to a prefix of the next, and it makes segmentation reviewable by a person.
* **Reserved control identifier.** :data:`RESERVED_DOCUMENT_END_ID` is allocated before any merge, so
  document separators do not shift identifier values when the vocabulary is retrained at a different size,
  and every merged symbol has the same id for the same rule sequence.
* **Deterministic tie-breaking.** The next merge is the most frequent pair; ties go to the pair that sorts
  first. Nothing depends on dictionary iteration order, so the same training text always produces the same
  fingerprint.
* **Self-describing serialization.** The merge table plus its own checksum, verified on load, because a
  checkpoint silently paired with a different tokenizer is the failure mode that shows up as a model that
  suddenly cannot produce coherent text.

Standard library only, so a tokenizer and its segmentations are testable and type-checkable in the
dependency-free CI matrix, without a numerical framework in the way.
"""

from __future__ import annotations

import codecs
import heapq
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from chakaso.core.hashing import join_parts, sha256_hex
from chakaso.training.errors import (
    CorpusError,
    CorruptedCorpusArtifactError,
)

__all__ = [
    "BYTE_BASE",
    "DOCUMENT_END",
    "FIRST_MERGED_ID",
    "RESERVED_DOCUMENT_END_ID",
    "IncrementalDecoder",
    "SubwordVocabulary",
    "pre_tokenize",
    "train_subword_vocabulary",
]

#: Every atomic symbol is one UTF-8 byte, so identifiers 0-255 are fixed and need no training.
BYTE_BASE = 256
#: Allocated before any merge so a separator's identifier never moves when the vocabulary is retrained.
RESERVED_DOCUMENT_END_ID = 256
#: The first identifier a learned merge may claim.
FIRST_MERGED_ID = 257
#: The conventional token that marks the end of a document inside a packed stream.
DOCUMENT_END = RESERVED_DOCUMENT_END_ID

_SCHEMA_VERSION = "chakaso-subword-vocabulary/1"
_PRE_TOKENIZER = "word-run|whitespace-run|single-symbol"
_BASE = "utf-8-bytes"

# Three classes, chosen so the alternatives cover every character: `\w+` is a run of letters, digits and
# underscores (an underscore stays inside the word it belongs to, as in GPT-2's pre-tokenizer, so a code
# identifier segments as one piece rather than three); `\s+` is a whitespace run; `[^\w\s]` is one other
# symbol. A test asserts the join of the pieces equals the input, which is what proves nothing is dropped.
PIECE_PATTERN = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)


def pre_tokenize(text: str) -> tuple[str, ...]:
    """Split ``text`` into pieces that may be merged internally, never across a boundary."""
    return tuple(PIECE_PATTERN.findall(text))


class IncrementalDecoder:
    """Turns a stream of identifiers into text without splitting a multi-byte character.

    :meth:`feed` returns only the characters the bytes so far complete (possibly none, mid-character);
    :meth:`finish` flushes what is left, replacing a truly truncated character exactly as
    :meth:`SubwordVocabulary.decode` does. Feeding every identifier and then finishing yields the same text
    as one bulk ``decode``, which is the property the tests hold it to.
    """

    def __init__(self, vocabulary: SubwordVocabulary) -> None:
        self._vocabulary = vocabulary
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, symbol: int) -> str:
        """The text newly completed by ``symbol``; empty for the separator or an unfinished character."""
        if symbol == RESERVED_DOCUMENT_END_ID:
            return ""
        if symbol < 0 or symbol >= self._vocabulary.size:
            message = (
                f"identifier {symbol} is outside this vocabulary (size {self._vocabulary.size})"
            )
            raise CorpusError(message)
        return self._decoder.decode(self._vocabulary.symbol_to_bytes(symbol))

    def finish(self) -> str:
        """Flush a character the stream ended inside of, as a replacement character."""
        return self._decoder.decode(b"", final=True)


@dataclass(frozen=True, slots=True)
class SubwordVocabulary:
    """A trained byte-pair vocabulary: its merge table, and the encode/decode pair it defines."""

    merges: tuple[tuple[int, int], ...]
    min_frequency: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.min_frequency, bool) or not isinstance(self.min_frequency, int):
            message = f"min_frequency must be an integer, got {self.min_frequency!r}"
            raise ValueError(message)
        if self.min_frequency < 1:
            message = f"min_frequency must be >= 1, got {self.min_frequency}"
            raise ValueError(message)
        seen: set[tuple[int, int]] = set()
        for rank, pair in enumerate(self.merges):
            left, right = pair
            for value in (left, right):
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    message = f"merge {rank} contains a symbol id that is not a non-negative integer: {pair}"
                    raise ValueError(message)
                if value >= FIRST_MERGED_ID + rank:
                    message = (
                        f"merge {rank} references symbol {value}, which this vocabulary does not define "
                        f"before that rank (ids up to {FIRST_MERGED_ID + rank - 1} exist)"
                    )
                    raise ValueError(message)
            if pair in seen:
                message = f"merge pair {pair} appears twice; a vocabulary must apply each rule once"
                raise ValueError(message)
            seen.add(pair)

    @property
    def size(self) -> int:
        """Total identifier count: bytes, the reserved separator, and one per learned merge."""
        return FIRST_MERGED_ID + len(self.merges)

    @property
    def document_end_id(self) -> int:
        """The reserved separator identifier this vocabulary always uses."""
        return RESERVED_DOCUMENT_END_ID

    @property
    def merge_count(self) -> int:
        """How many merges were learned."""
        return len(self.merges)

    @property
    def fingerprint(self) -> str:
        """A stable digest of the exact rule sequence — the tokenizer half of a checkpoint identity."""
        payload = join_parts(
            _SCHEMA_VERSION,
            _BASE,
            _PRE_TOKENIZER,
            str(self.size),
            join_parts(*(join_parts(str(left), str(right)) for left, right in self.merges)),
        )
        return sha256_hex(payload)[:16]

    def _ranks(self) -> dict[tuple[int, int], int]:
        return {pair: rank for rank, pair in enumerate(self.merges)}

    def symbol_to_bytes(self, symbol: int) -> bytes:
        """Return the bytes a single identifier stands for.

        Byte identifiers return themselves; the reserved separator returns nothing, because it is control,
        not text; a merged symbol returns the concatenation of what its two parts stand for, which is the
        fact the merge table asserts.
        """
        if symbol < BYTE_BASE:
            return bytes((symbol,))
        if symbol == RESERVED_DOCUMENT_END_ID:
            return b""
        if symbol >= self.size:
            message = f"symbol {symbol} is outside this vocabulary (size {self.size})"
            raise CorpusError(message)
        left, right = self.merges[symbol - FIRST_MERGED_ID]
        return self.symbol_to_bytes(left) + self.symbol_to_bytes(right)

    def pieces(self) -> tuple[str, ...]:
        """Every identifier's readable form, in identifier order, for hand inspection.

        Control and split multi-byte characters are rendered as escapes rather than dropped: a person
        reviewing a vocabulary must see exactly what a token covers, including where a character is only
        half present.
        """
        out: list[str] = []
        for symbol in range(self.size):
            raw = self.symbol_to_bytes(symbol)
            if symbol == RESERVED_DOCUMENT_END_ID:
                out.append("<doc-end>")
                continue
            try:
                out.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                out.append("".join(f"\\x{byte:02x}" for byte in raw))
        return tuple(out)

    def encode(self, text: str, *, allowed: frozenset[int] | None = None) -> tuple[int, ...]:
        """Map ``text`` to identifiers.

        Encoding is total: any Unicode input produces byte symbols and whatever merges apply, and nothing is
        dropped or guessed. When ``allowed`` is given, identifiers outside it are refused instead of
        emitted - the guard a caller uses when a vocabulary must not exceed a model's embedding table.
        """
        if allowed is not None:
            unknown = sorted({symbol for symbol in self._encode_all(text) if symbol not in allowed})
            if unknown:
                message = f"encode produced identifiers outside the allowed set: {unknown}"
                raise CorpusError(message)
        return self._encode_all(text)

    def _encode_all(self, text: str) -> tuple[int, ...]:
        ranks = self._ranks()
        produced: list[int] = []
        for piece in pre_tokenize(text):
            produced.extend(self._encode_piece(piece, ranks))
        return tuple(produced)

    def _encode_piece(self, piece: str, ranks: dict[tuple[int, int], int]) -> tuple[int, ...]:
        symbols = list(piece.encode("utf-8"))
        while len(symbols) > 1:
            best: tuple[tuple[int, int], int] | None = None
            for index in range(len(symbols) - 1):
                pair = (symbols[index], symbols[index + 1])
                rank = ranks.get(pair)
                if rank is not None and (best is None or rank < best[1]):
                    best = (pair, rank)
            if best is None:
                break
            symbols = self._apply(symbols, best[0], FIRST_MERGED_ID + best[1])
        return tuple(symbols)

    @staticmethod
    def _apply(symbols: list[int], pair: tuple[int, int], merged: int) -> list[int]:
        """Replace every non-overlapping left-to-right occurrence of ``pair`` with ``merged``."""
        left, right = pair
        out: list[int] = []
        index = 0
        while index < len(symbols):
            if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
                out.append(merged)
                index += 2
            else:
                out.append(symbols[index])
                index += 1
        return out

    def decode(self, identifiers: Sequence[int]) -> str:
        """Reconstruct text from identifiers, skipping the reserved separator.

        Raises:
            CorpusError: an identifier is outside this vocabulary.
        """
        buffer = bytearray()
        for symbol in identifiers:
            if symbol == RESERVED_DOCUMENT_END_ID:
                continue
            if symbol < 0 or symbol >= self.size:
                message = f"identifier {symbol} is outside this vocabulary (size {self.size})"
                raise CorpusError(message)
            buffer.extend(self.symbol_to_bytes(symbol))
        return buffer.decode("utf-8", errors="replace")

    def incremental_decoder(self) -> IncrementalDecoder:
        """A decoder for text that arrives one identifier at a time, as it does during generation.

        :meth:`decode` on a single identifier cannot be joined with its neighbours: a character whose UTF-8
        bytes span two tokens is replaced in each half, and the replacement is permanent.
        """
        return IncrementalDecoder(self)

    def token_count(self, text: str) -> int:
        """How many identifiers ``text`` occupies — the denominator of every bits-per-token number."""
        return len(self._encode_all(text))

    def compression_ratio(self, text: str) -> float:
        """Characters per token for ``text``; 0.0 when it encodes to nothing.

        This is the quantity that turns a per-token cross-entropy into a per-character one, which is the
        only form comparable across tokenizations.
        """
        tokens = self.token_count(text)
        if tokens == 0:
            return 0.0
        return len(text) / tokens

    def to_document(self) -> dict[str, Any]:
        """The serializable record: schema, conventions, merge table, and its own checksum."""
        content = self._content()
        document = dict(content)
        document["checksum"] = self.checksum()
        return document

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "base": _BASE,
            "pre_tokenizer": _PRE_TOKENIZER,
            "reserved": {"document_end": RESERVED_DOCUMENT_END_ID},
            "vocab_size": self.size,
            "min_frequency": self.min_frequency,
            "merges": [[left, right] for left, right in self.merges],
        }

    @staticmethod
    def _canonical(content: dict[str, Any]) -> str:
        return json.dumps(content, sort_keys=True, separators=(",", ":"))

    def checksum(self) -> str:
        """Digest of the record itself, excluding the checksum field."""
        return sha256_hex(self._canonical(self._content()))

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> SubwordVocabulary:
        """Rebuild and verify a vocabulary record.

        Raises:
            CorruptedCorpusArtifactError: the checksum is absent or disagrees, or the JSON shape is wrong.
            CorpusError: the schema, base or pre-tokenizer is not one this code implements, or the declared
                vocabulary size disagrees with the merge table.
        """
        if not isinstance(document, dict):
            message = "a subword vocabulary record must be a JSON object"
            raise CorruptedCorpusArtifactError(message)
        if "checksum" not in document:
            message = "subword vocabulary has no checksum field"
            raise CorruptedCorpusArtifactError(message)
        content = {key: value for key, value in document.items() if key != "checksum"}
        stored = document["checksum"]
        if not isinstance(stored, str) or sha256_hex(cls._canonical(content)) != stored:
            message = (
                "subword vocabulary checksum does not match its content — the file was altered"
            )
            raise CorruptedCorpusArtifactError(message)
        for field_name, expected in (
            ("schema_version", _SCHEMA_VERSION),
            ("base", _BASE),
            ("pre_tokenizer", _PRE_TOKENIZER),
        ):
            if content.get(field_name) != expected:
                message = (
                    f"subword vocabulary declares {field_name}={content.get(field_name)!r}, "
                    f"this code implements {expected!r}"
                )
                raise CorpusError(message)
        reserved = content.get("reserved")
        if (
            not isinstance(reserved, dict)
            or reserved.get("document_end") != RESERVED_DOCUMENT_END_ID
        ):
            message = f"subword vocabulary reserves an unexpected identifier layout: {reserved!r}"
            raise CorpusError(message)

        try:
            merges = tuple((int(pair[0]), int(pair[1])) for pair in content["merges"])
            declared_size = int(content["vocab_size"])
            min_frequency = int(content["min_frequency"])
        except (KeyError, TypeError, ValueError) as exc:
            message = f"subword vocabulary is missing or mis-typed a required field: {exc}"
            raise CorpusError(message) from exc
        vocabulary = cls(merges=merges, min_frequency=min_frequency)
        if declared_size != vocabulary.size:
            message = (
                f"vocabulary declares vocab_size {declared_size} but its {len(merges)} merge(s) "
                f"imply {vocabulary.size}"
            )
            raise CorpusError(message)
        return vocabulary

    def save(self, path: Path) -> None:
        """Write the vocabulary as deterministic UTF-8 JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_document(), sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> SubwordVocabulary:
        """Read a vocabulary written by :meth:`save`.

        Raises:
            CorruptedCorpusArtifactError: the file is not valid JSON.
        """
        raw = Path(path).read_text(encoding="utf-8")
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            message = f"subword vocabulary at {str(path)!r} is not valid JSON: {exc}"
            raise CorruptedCorpusArtifactError(message) from exc
        return cls.from_document(document)


def train_subword_vocabulary(
    texts: Iterable[str],
    *,
    target_size: int,
    min_frequency: int = 2,
) -> SubwordVocabulary:
    """Learn byte-pair merges from ``texts`` until the vocabulary reaches ``target_size``.

    Counts are taken over *distinct pieces weighted by frequency*, and the chosen rule is the most frequent
    adjacent pair with a count of at least ``min_frequency``, ties broken by the pair's own ordering. A
    vocabulary trained on training-split text alone is what keeps a held-out document from influencing the
    identifier space, exactly as the character vocabulary's train-only rule does.

    Raises:
        ValueError: ``target_size`` is smaller than the base vocabulary, ``min_frequency`` is not a
            positive integer, or the texts yield no pieces at all.
    """
    if (
        isinstance(target_size, bool)
        or not isinstance(target_size, int)
        or target_size < FIRST_MERGED_ID
    ):
        message = f"target_size must be an integer >= {FIRST_MERGED_ID} (the byte base plus the separator), got {target_size!r}"
        raise ValueError(message)
    if isinstance(min_frequency, bool) or not isinstance(min_frequency, int) or min_frequency < 1:
        message = f"min_frequency must be an integer >= 1, got {min_frequency!r}"
        raise ValueError(message)

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(pre_tokenize(text))
    if not counts:
        message = (
            "a subword vocabulary needs at least one pre-tokenized piece; the texts produced none"
        )
        raise ValueError(message)

    shapes: dict[tuple[int, ...], int] = {
        tuple(piece.encode("utf-8")): frequency for piece, frequency in counts.items()
    }
    # An index from pair to the shapes that contain it, and a heap of pair counts. The first version of this
    # loop scanned every shape on every merge, which is O(merges x shapes x shape length): on a
    # multi-register corpus, where nearly every pre-tokenized piece is unique, that made a 4,096-merge fit
    # take hours. The index makes a merge touch only the shapes the pair actually occurs in.
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_shapes: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for shape, frequency in shapes.items():
        for pair, occurrences in _pair_multiset(shape).items():
            pair_counts[pair] += occurrences * frequency
            pair_to_shapes.setdefault(pair, set()).add(shape)
    # Entries are invalidated lazily: an entry is current only if its count still matches the counter.
    heap: list[tuple[int, tuple[int, int]]] = [
        (-count, pair) for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    while FIRST_MERGED_ID + len(merges) < target_size:
        # Most frequent pair wins; a tie breaks on the smaller pair so the rule sequence is a property of
        # the data alone, never of the order two equal counts happened to be stored in.
        pair: tuple[int, int] | None = None
        while heap:
            negative, candidate = heapq.heappop(heap)
            current = pair_counts.get(candidate, 0)
            if current == -negative and current >= min_frequency:
                pair = candidate
                break
        if pair is None:
            break
        merged = FIRST_MERGED_ID + len(merges)
        merges.append(pair)

        affected = pair_to_shapes.pop(pair, set())
        for shape in affected:
            frequency = shapes.pop(shape, None)
            if frequency is None:
                continue
            _remove_shape(pair_counts, pair_to_shapes, shape, frequency)
            new_shape = tuple(SubwordVocabulary._apply(list(shape), pair, merged))
            existing = shapes.pop(new_shape, 0)
            if existing:
                _remove_shape(pair_counts, pair_to_shapes, new_shape, existing)
                frequency += existing
            shapes[new_shape] = frequency
            for new_pair, occurrences in _pair_multiset(new_shape).items():
                pair_counts[new_pair] += occurrences * frequency
                pair_to_shapes.setdefault(new_pair, set()).add(new_shape)
                heapq.heappush(heap, (-pair_counts[new_pair], new_pair))

    return SubwordVocabulary(merges=tuple(merges), min_frequency=min_frequency)


def _pair_multiset(shape: Sequence[int]) -> Counter[tuple[int, int]]:
    return Counter(pairwise(shape))


def _remove_shape(
    pair_counts: Counter[tuple[int, int]],
    pair_to_shapes: dict[tuple[int, int], set[tuple[int, ...]]],
    shape: tuple[int, ...],
    frequency: int,
) -> None:
    """Subtract one shape's pair contributions from the counters and the index."""
    for pair, occurrences in _pair_multiset(shape).items():
        remaining = pair_counts[pair] - occurrences * frequency
        if remaining > 0:
            pair_counts[pair] = remaining
        else:
            pair_counts.pop(pair, None)
        holders = pair_to_shapes.get(pair)
        if holders is not None:
            holders.discard(shape)
            if not holders:
                pair_to_shapes.pop(pair, None)


def _shift_counts(
    pair_counts: Counter[tuple[int, int]],
    old: tuple[int, ...],
    new: tuple[int, ...],
    frequency: int,
) -> None:
    """Move one shape's pair contributions from its pre-merge form to its merged form.

    Counted per *occurrence*, not per shape: a piece containing the merged pair three times contributes
    three times its document frequency, and subtracting one would leave the next iteration choosing a
    merge on a stale count.
    """
    for pair, occurrences in _pair_multiset(old).items():
        remaining = pair_counts[pair] - occurrences * frequency
        if remaining > 0:
            pair_counts[pair] = remaining
        else:
            pair_counts.pop(pair, None)
    for pair, occurrences in _pair_multiset(new).items():
        pair_counts[pair] += occurrences * frequency
