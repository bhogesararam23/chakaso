"""What a corpus is made of, counted — and what in it repeats itself.

Two jobs, both required before a corpus can carry a model claim.

**Composition.** A corpus of "1,660,274 characters" says nothing about whether it is twenty pages or twenty
books, whether one author dominates it, or whether its documents are long enough to be independent units.
:func:`build_report` measures those things from the pinned manifest, so a data card states observed shape
rather than an impression, and a reader can see the skew instead of being told there is none.

**Duplication.** A corpus assembled from a public collection can contain the same text twice — different
editions of one book, an anthology that reprints a title already in the set. Exact duplication is already
caught: the split deduplicates by content digest and the manifest can list the duplicate groups. Near
duplication is the one that quietly inflates a held-out number, because a test document that is 80 % a
training document is not held out at all. :func:`near_duplicate_groups` finds candidate pairs with a
deterministic min-hash signature and reports the pairs that exceed a declared similarity, so the finding is
either empty or stated in the record.

Nothing here judges "quality". A small model on public-domain Victorian prose is what it is, and the report's
job is to make that visible.

Standard library only, so corpus auditing runs in the dependency-free matrix.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

from chakaso.core.atomic import write_text_atomically
from chakaso.training.dataset_catalog import DatasetCatalog
from chakaso.training.errors import CorpusError
from chakaso.training.manifest import CorpusManifest

__all__ = [
    "DEFAULT_MAX_BUCKET",
    "DEFAULT_MIN_SHARED",
    "DEFAULT_SHINGLE_WORDS",
    "DEFAULT_SIMILARITY",
    "DEFAULT_SKETCH_WIDTH",
    "DatasetReport",
    "bottom_k",
    "build_report",
    "candidate_pairs",
    "default_limitations",
    "near_duplicate_groups",
    "render_data_card",
    "report_to_file",
    "word_shingles",
    "write_data_card",
]

#: Words per shingle. Eight is long enough that a shared sentence rarely produces a shared shingle by
#: accident, and short enough that ordinary paraphrase and recurring chapter headings still differ.
DEFAULT_SHINGLE_WORDS = 8
#: How many smallest-hash values one document keeps. More values estimate overlap better and cost a
#: proportional amount of memory; 64 is ample for a corpus of thousands of documents.
DEFAULT_SKETCH_WIDTH = 64
#: Fewest sketch values two documents must share to be worth comparing exactly. A bottom-k sketch of 64
#: values that share 4 in common is already far from a coincidence, and the exact Jaccard is computed on
#: whatever survives this filter.
DEFAULT_MIN_SHARED = 4
#: Largest posting list treated as evidence. A sketch value held by more documents than this is shared
#: boilerplate — a page marker, a repeated heading — not a sign that two documents are near duplicates, and
#: comparing them pairwise would cost the square of the bucket for nothing.
DEFAULT_MAX_BUCKET = 64
#: Jaccard overlap of word shingles at which two documents are reported as near duplicates.
DEFAULT_SIMILARITY = 0.5

_SEPARATORS = " \t\r\n"


def _hash(token: str) -> int:
    """A stable 64-bit digest of one shingle.

    ``hash()`` is deliberately not used: CPython salts string hashing per process, so a "deterministic"
    duplicate scan that moved between runs would be a bug nobody could reproduce.
    """
    return int.from_bytes(blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def word_shingles(text: str, *, size: int = DEFAULT_SHINGLE_WORDS) -> frozenset[str]:
    """Every contiguous run of ``size`` whitespace-separated tokens, as a set.

    A document shorter than ``size`` tokens produces no shingles and is therefore never a near-duplicate
    candidate — a three-line fragment shares nothing measurable with anything, and saying so is honest.
    """
    if size < 1:
        message = f"shingle size must be >= 1, got {size}"
        raise CorpusError(message)
    tokens = text.split()
    if len(tokens) < size:
        return frozenset()
    return frozenset(
        " ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    )


def bottom_k(shingles: Iterable[str], *, width: int = DEFAULT_SKETCH_WIDTH) -> frozenset[int]:
    """The ``width`` smallest stable hashes of ``shingles`` — a bottom-k sketch.

    Two documents whose shingle sets overlap heavily keep many of the same smallest values, which is the
    property the candidate search below uses. This is the single-permutation form of the estimator: no
    permutations to seed, no seed to forget, and the same sketch on every machine.
    """
    if width < 1:
        message = f"sketch width must be >= 1, got {width}"
        raise CorpusError(message)
    values = {_hash(item) for item in shingles}
    if not values:
        return frozenset()
    return frozenset(sorted(values)[:width])


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _shared_values(
    sketches: Mapping[str, frozenset[int]], *, min_shared: int, max_bucket: int
) -> dict[tuple[str, str], int]:
    """Pairwise counts of shared sketch values, gathered through an inverted index.

    Every document posts its sketch values to a bucket; a pair is enumerated once per bucket it shares, so
    the work is the size of the postings rather than the square of the corpus. Pairs below ``min_shared``
    shared values are dropped, and buckets larger than ``max_bucket`` are skipped as boilerplate.

    Raises:
        CorpusError: ``min_shared`` or ``max_bucket`` is not usable.
    """
    if min_shared < 1:
        message = f"min_shared must be >= 1, got {min_shared}"
        raise CorpusError(message)
    if max_bucket < 2:
        message = f"max_bucket must be >= 2, got {max_bucket}"
        raise CorpusError(message)
    buckets: dict[int, list[str]] = {}
    for identifier in sorted(sketches):
        for value in sketches[identifier]:
            buckets.setdefault(value, []).append(identifier)

    shared: dict[tuple[str, str], int] = {}
    for members in buckets.values():
        if len(members) < 2 or len(members) > max_bucket:
            continue
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                pair = (left, right) if left < right else (right, left)
                shared[pair] = shared.get(pair, 0) + 1
    return {pair: count for pair, count in shared.items() if count >= min_shared}


def near_duplicate_groups(
    documents: Mapping[str, str],
    *,
    shingle_words: int = DEFAULT_SHINGLE_WORDS,
    sketch_width: int = DEFAULT_SKETCH_WIDTH,
    min_shared: int = DEFAULT_MIN_SHARED,
    max_bucket: int = DEFAULT_MAX_BUCKET,
    similarity: float = DEFAULT_SIMILARITY,
) -> tuple[tuple[tuple[str, ...], float], ...]:
    """Group documents whose word shingles overlap by at least ``similarity``.

    Args:
        documents: identifier → text. Identifiers are used only to name the findings.

    Returns:
        ``(group, similarity)`` pairs, strongest first, where ``group`` holds at least two identifiers.
        An empty result means no pair crossed the threshold — a measurement, not an absence of one.
    """
    if not documents:
        return ()
    if not 0.0 < similarity <= 1.0:
        message = f"similarity must be in (0, 1], got {similarity}"
        raise CorpusError(message)
    # Two passes, because a corpus of a thousand documents does not fit in memory as a thousand shingle
    # sets on the machine this project runs on. The first pass keeps only the 64-value sketch of each
    # document and throws the strings away; the second recomputes shingles for the documents the sketch
    # actually proposed a comparison for, which is a handful.
    sketches = {
        key: bottom_k(word_shingles(text, size=shingle_words), width=sketch_width)
        for key, text in documents.items()
    }
    candidates = _shared_values(sketches, min_shared=min_shared, max_bucket=max_bucket)
    del sketches

    remembered: dict[str, frozenset[str]] = {}

    def shingles_of(key: str) -> frozenset[str]:
        if key not in remembered:
            remembered[key] = word_shingles(documents[key], size=shingle_words)
        return remembered[key]

    matched: list[tuple[tuple[str, ...], float]] = []
    for left, right in sorted(candidates):
        value = _jaccard(shingles_of(left), shingles_of(right))
        if value >= similarity:
            matched.append(((left, right), value))
    return tuple(sorted(_merge(matched), key=lambda item: -item[1]))


def candidate_pairs(
    documents: Mapping[str, str],
    *,
    shingle_words: int = DEFAULT_SHINGLE_WORDS,
    sketch_width: int = DEFAULT_SKETCH_WIDTH,
    min_shared: int = DEFAULT_MIN_SHARED,
    max_bucket: int = DEFAULT_MAX_BUCKET,
) -> int:
    """How many pairs the sketch proposes to compare exactly — the denominator a finding needs.

    Reported alongside a duplication result because "no near duplicates" over three compared pairs and over
    three hundred are different statements, and only the number tells them apart.
    """
    if not documents:
        return 0
    sketches = {
        key: bottom_k(word_shingles(text, size=shingle_words), width=sketch_width)
        for key, text in documents.items()
    }
    return len(_shared_values(sketches, min_shared=min_shared, max_bucket=max_bucket))


def _merge(
    matched: Sequence[tuple[tuple[str, ...], float]],
) -> tuple[tuple[tuple[str, ...], float], ...]:
    """Collapse matched pairs into groups, keeping the strongest pairwise similarity as the group's."""
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        while parent.setdefault(key, key) != key:
            key = parent[key]
        return key

    groups: dict[str, list[str]] = {}
    scores: dict[str, float] = {}
    for (left, right), value in matched:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
        root = find(left)
        members = groups.setdefault(root, [])
        for member in (left, right):
            if member not in members:
                members.append(member)
        scores[root] = max(scores.get(root, 0.0), value)
    return tuple(
        (tuple(sorted(members)), scores[root])
        for root, members in groups.items()
        if len(members) > 1
    )


@dataclass(frozen=True, slots=True)
class DatasetReport:
    """The measured shape of one pinned corpus.

    Attributes:
        documents/characters/bytes: Totals over the pinned documents.
        documents_by_source/characters_by_source: Composition, so a one-title corpus cannot be read as a
            many-title one.
        document_characters: Length percentiles of individual documents.
        distinct_symbols: Size of the character alphabet actually present.
        exact_duplicate_groups: Document paths sharing a content digest, as the manifest sees them.
        near_duplicate_groups: Groups from :func:`near_duplicate_groups`, with their similarity.
        pairs_examined: Candidate pairs the sketch actually compared — the honest denominator.
    """

    name: str
    version: str
    licence: str
    source: str
    documents: int
    characters: int
    bytes: int
    documents_by_source: tuple[tuple[str, int], ...]
    characters_by_source: tuple[tuple[str, int], ...]
    document_characters: Mapping[str, int]
    distinct_symbols: int
    exact_duplicate_groups: tuple[tuple[str, ...], ...]
    near_duplicate_groups: tuple[tuple[tuple[str, ...], float], ...]
    pairs_examined: int
    similarity_threshold: float
    shingle_words: int

    def as_dict(self) -> dict[str, object]:
        """The report as plain data, ordered so two runs produce the same JSON."""
        return {
            "name": self.name,
            "version": self.version,
            "licence": self.licence,
            "source": self.source,
            "documents": self.documents,
            "characters": self.characters,
            "bytes": self.bytes,
            "documents_by_source": [list(item) for item in self.documents_by_source],
            "characters_by_source": [list(item) for item in self.characters_by_source],
            "document_characters": dict(self.document_characters),
            "distinct_symbols": self.distinct_symbols,
            "exact_duplicate_groups": [list(item) for item in self.exact_duplicate_groups],
            "near_duplicate_groups": [
                [list(group), round(value, 6)] for group, value in self.near_duplicate_groups
            ],
            "pairs_examined": self.pairs_examined,
            "similarity_threshold": self.similarity_threshold,
            "shingle_words": self.shingle_words,
        }


def _percentiles(lengths: Sequence[int]) -> dict[str, int]:
    """Nearest-rank percentiles of document lengths, including the extremes explicitly."""
    if not lengths:
        return {}
    ordered = sorted(lengths)
    keys = ("min", "p10", "p25", "p50", "p75", "p90", "p100")
    out: dict[str, int] = {}
    for name, fraction in zip(keys, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0), strict=True):
        index = -(-fraction * (len(ordered) - 1) // 1) if name != "min" else 0
        out[name] = ordered[int(index)]
    return out


def build_report(
    manifest: CorpusManifest,
    *,
    shingle_words: int = DEFAULT_SHINGLE_WORDS,
    similarity: float = DEFAULT_SIMILARITY,
    sketch_width: int = DEFAULT_SKETCH_WIDTH,
    min_shared: int = DEFAULT_MIN_SHARED,
    max_bucket: int = DEFAULT_MAX_BUCKET,
) -> DatasetReport:
    """Measure a pinned corpus: composition, document shape and duplication.

    The document texts come from the manifest, which is the only copy a measured run is allowed to trust.
    """
    by_source: dict[str, list[int]] = {}
    lengths: list[int] = []
    alphabet: set[str] = set()
    for document in manifest.documents:
        slug = document.path.split("/", 1)[0]
        counts = by_source.setdefault(slug, [0, 0])
        counts[0] += 1
        counts[1] += document.char_count
        lengths.append(document.char_count)
        alphabet.update(document.text)
    documents = {document.path: document.text for document in manifest.documents}
    near = near_duplicate_groups(
        documents,
        shingle_words=shingle_words,
        sketch_width=sketch_width,
        min_shared=min_shared,
        max_bucket=max_bucket,
        similarity=similarity,
    )
    return DatasetReport(
        name=manifest.name,
        version=manifest.version,
        licence=manifest.licence,
        source=manifest.source,
        documents=len(manifest.documents),
        characters=manifest.char_count,
        bytes=manifest.byte_count,
        documents_by_source=tuple(
            (slug, int(counts[0])) for slug, counts in sorted(by_source.items())
        ),
        characters_by_source=tuple(
            (slug, int(counts[1])) for slug, counts in sorted(by_source.items())
        ),
        document_characters=_percentiles(lengths),
        distinct_symbols=len(alphabet),
        exact_duplicate_groups=manifest.duplicate_paths,
        near_duplicate_groups=near,
        # Every candidate pair the sketch proposed, whether or not it crossed the threshold.
        pairs_examined=candidate_pairs(
            documents,
            shingle_words=shingle_words,
            sketch_width=sketch_width,
            min_shared=min_shared,
            max_bucket=max_bucket,
        ),
        similarity_threshold=similarity,
        shingle_words=shingle_words,
    )


def default_limitations(report: DatasetReport) -> tuple[str, ...]:
    """What the measured shape of a corpus already says about its own limits.

    These are computed from the report rather than written by hand, so a data card cannot keep a limitation
    after the corpus stops having it — a dominant source is named with its actual share, and disappears from
    the card when the share does.
    """
    total = report.characters or 1
    lines: list[str] = []
    for slug, characters in report.characters_by_source:
        share = 100.0 * characters / total
        if share >= 25.0:
            lines.append(
                f"`{slug}` supplies {share:.1f}% of the characters, so a model trained on this corpus "
                "inherits that source's house style as much as the collection's."
            )
    lines += [
        "Continuous prose only: no instruction examples, no dialogue turns and no question-answer pairs, "
        "so nothing in this corpus can teach a model to answer anything.",
        "Split assignment belongs to an experiment, not to the corpus: two runs with different split seeds "
        "hold out different documents from the same bytes.",
        "Characters are counted after the manifest's line-ending normalization, and nothing else is "
        "normalized — no case folding, no diacritic removal, no deduplication across sources.",
    ]
    return tuple(lines)


def render_data_card(
    report: DatasetReport,
    *,
    catalog: DatasetCatalog | None = None,
    snapshot_checksum: str | None = None,
    catalog_sha256: str | None = None,
    sources: Sequence[tuple[str, str, str, str]] = (),
    prepared_with: str = "",
    limitations: Sequence[str] = (),
) -> str:
    """Markdown for one dataset: what it is, where it came from, what is in it, what it cannot show.

    Args:
        catalog: The declared source list, when the caller has it; its name and retrieval date are printed.
        sources: ``(slug, title, url, digest)`` tuples, verbatim from the catalogue.
        prepared_with: The command and configuration that produced this corpus, so the card is runnable.
        limitations: Statements about what the corpus does not cover, written as limits rather than hopes.

    Returns:
        The card as text. Writing it is the caller's choice, because a card belongs beside an experiment or
        in ``docs/``, not inside a corpus artifact.
    """
    name = catalog.name if catalog is not None else report.name
    retrieved = catalog.retrieved if catalog is not None else ""
    lines = [
        f"# Data card: {report.name} {report.version}",
        "",
        f"- **Catalogue:** `{name}`",
        f"- **Licence:** {report.licence}",
        f"- **Source:** {report.source}",
    ]
    if retrieved:
        lines.append(f"- **Retrieved:** {retrieved}")
    lines += [
        f"- **Documents:** {report.documents:,} over {report.characters:,} characters "
        f"({report.bytes:,} bytes)",
        f"- **Character alphabet present:** {report.distinct_symbols} distinct symbols",
        "- **Document length (characters):** "
        + ", ".join(f"{key} {value:,}" for key, value in report.document_characters.items()),
        "",
        "## Composition",
        "",
        "| Source | Documents | Characters |",
        "| --- | ---: | ---: |",
    ]
    characters = dict(report.characters_by_source)
    for slug, count in report.documents_by_source:
        lines.append(f"| `{slug}` | {count:,} | {characters.get(slug, 0):,} |")
    lines += [
        "",
        "## Duplication",
        "",
        f"- Exact duplicate document groups (same content digest): {len(report.exact_duplicate_groups)}.",
        f"- Near-duplicate screening: {report.pairs_examined:,} candidate pair(s) compared at "
        f"{report.shingle_words}-word shingles; threshold {report.similarity_threshold:g}.",
        f"- Groups above the threshold: {len(report.near_duplicate_groups)}.",
    ]
    for group, value in report.near_duplicate_groups:
        lines.append(f"  - similarity {value:g}: {', '.join(f'`{item}`' for item in group)}")
    if sources:
        lines += [
            "",
            "## Sources, as declared",
            "",
            "| Slug | Title | Bytes SHA-256 | URL |",
            "| --- | --- | --- | --- |",
        ]
        for slug, title, url, digest in sources:
            lines.append(f"| `{slug}` | {title} | `{digest[:16]}…` | {url} |")
    if catalog_sha256 is not None or snapshot_checksum is not None:
        lines += ["", "## Provenance", ""]
        if catalog_sha256 is not None:
            lines.append(f"- Catalogue file SHA-256: `{catalog_sha256}`")
        if snapshot_checksum is not None:
            lines.append(f"- Corpus snapshot checksum: `{snapshot_checksum}`")
    if prepared_with:
        lines += [
            "",
            "## How this corpus was prepared",
            "",
            "```text",
            prepared_with.strip(),
            "```",
        ]
    lines += ["", "## What this corpus does not show", ""]
    if limitations:
        lines += [f"- {item}" for item in limitations]
    else:
        lines.append("- Nothing was measured about the texts beyond what is recorded above.")
    lines.append("")
    return "\n".join(lines)


def write_data_card(card: str, destination: Path, *, allow_overwrite: bool = False) -> None:
    """Write a data card atomically, refusing to clobber one that was already signed off.

    Raises:
        CorpusError: a card already exists at ``destination`` and overwriting was not asked for.
    """
    if destination.exists() and not allow_overwrite:
        message = (
            f"a data card already exists at {str(destination)!r}; it is replaced only when the caller "
            "passes allow_overwrite=True"
        )
        raise CorpusError(message)
    write_text_atomically(card, destination)


def report_to_file(
    report: DatasetReport, destination: Path, *, allow_overwrite: bool = False
) -> None:
    """Write the machine-readable report beside the card, so a claim about the corpus is checkable.

    Raises:
        CorpusError: the destination exists and overwriting was not asked for.
    """
    if destination.exists() and not allow_overwrite:
        message = (
            f"a dataset report already exists at {str(destination)!r}; it is replaced only when the caller "
            "passes allow_overwrite=True"
        )
        raise CorpusError(message)
    write_text_atomically(
        json.dumps(report.as_dict(), sort_keys=True, indent=1, ensure_ascii=False) + "\n",
        destination,
    )
