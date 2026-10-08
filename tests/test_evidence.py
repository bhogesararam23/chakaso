"""Tests for source and chunk records, and for citation resolution.

The properties that matter here are the ones ADR-0003 depends on: identity belongs
to the retrieval side, a changed page produces a distinguishable record, and a
reference outside the supplied evidence is never resolved.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta, timezone

import pytest

from chakaso.core.hashing import sha256_hex
from chakaso.core.identifiers import ChunkId, SourceId, derive_source_id
from chakaso.evidence import (
    Citation,
    CitationResolution,
    EvidenceChunk,
    EvidencePack,
    SourceKind,
    SourceRecord,
    SourceRecordError,
    SourceReferenceError,
    UrlError,
    canonicalize_url,
    file_reference,
    resolve_citations,
    text_reference,
)

RETRIEVED = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
PUBLISHED = datetime(2026, 4, 1, 9, 30, tzinfo=UTC)
URL = "https://example.org/specification"
TEXT = "The submission window closes on 14 April."
REVISED = "The submission window closes on 21 April."
OTHER_TEXT = "Section 4 covers appeals."


def make_source(
    *,
    url: str = URL,
    content: str = TEXT,
    title: str = "Specification",
    retrieved_at: datetime = RETRIEVED,
    published_at: datetime | None = PUBLISHED,
) -> SourceRecord:
    return SourceRecord.create(
        url=url,
        title=title,
        content=content,
        retrieved_at=retrieved_at,
        published_at=published_at,
    )


# ---------------------------------------------------------------------------
# SourceRecord
# ---------------------------------------------------------------------------


def test_create_derives_identity_host_and_hash() -> None:
    record = make_source()

    assert record.source_id == derive_source_id(canonicalize_url(URL), TEXT)
    assert record.canonical_reference == URL
    assert record.kind is SourceKind.WEB
    assert record.domain == "example.org"
    assert record.content_hash == sha256_hex(TEXT)
    assert record.published_at == PUBLISHED


def test_create_canonicalizes_the_url_before_deriving_identity() -> None:
    # Identity is derived from the canonical URL, so a tracking parameter must not
    # produce a second source for one page.
    noisy = make_source(url=f"{URL}?utm_source=newsletter#top")

    assert noisy.canonical_reference == URL
    assert noisy.source_id == make_source().source_id


def test_a_changed_page_produces_a_distinguishable_record() -> None:
    # ADR-0007: an earlier answer must keep pointing at the bytes it actually used.
    original = make_source(content=TEXT)
    revised = make_source(content=REVISED, retrieved_at=RETRIEVED + timedelta(days=30))

    assert original.source_id != revised.source_id
    assert original.content_hash != revised.content_hash


def test_matches_content_confirms_a_record_still_describes_its_bytes() -> None:
    record = make_source()

    assert record.matches_content(TEXT) is True
    assert record.matches_content(REVISED) is False


def test_matches_content_accepts_bytes_of_the_same_content() -> None:
    assert make_source().matches_content(TEXT.encode("utf-8")) is True


def test_records_are_frozen() -> None:
    record = make_source()

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.title = "something else"  # type: ignore[misc]


def test_metadata_is_read_only() -> None:
    record = SourceRecord.create(
        url=URL,
        title="t",
        content=TEXT,
        retrieved_at=RETRIEVED,
        metadata={"extractor": "test"},
    )

    assert record.metadata["extractor"] == "test"
    with pytest.raises(TypeError):
        record.metadata["extractor"] = "other"  # type: ignore[index]


def test_default_metadata_is_empty_and_shared_immutably() -> None:
    assert make_source().metadata == {}


def test_undated_source_reports_that_freshness_is_unknown() -> None:
    # A guessed publication date would let a stale source look current.
    record = make_source(published_at=None)

    assert record.published_at is None
    assert record.is_freshness_known is False
    assert make_source().is_freshness_known is True


def test_naive_retrieval_timestamp_is_rejected() -> None:
    # A naive timestamp silently means "whatever the machine's timezone was".
    with pytest.raises(SourceRecordError, match="timezone-aware"):
        make_source(retrieved_at=datetime(2026, 10, 7, 12, 0))


def test_naive_publication_timestamp_is_rejected() -> None:
    with pytest.raises(SourceRecordError, match="published_at"):
        make_source(published_at=datetime(2026, 4, 1, 9, 30))


def test_non_utc_offsets_are_accepted() -> None:
    # Only awareness is required. Requiring UTC would reject a correctly labelled
    # timestamp from a source that reports its own offset.
    offset = datetime(2026, 10, 7, 14, 0, tzinfo=UTC).astimezone(timezone(timedelta(hours=2)))

    assert make_source(retrieved_at=offset).retrieved_at.utcoffset() == timedelta(hours=2)


def test_empty_canonical_reference_is_rejected() -> None:
    with pytest.raises(SourceReferenceError, match="non-empty"):
        SourceRecord(
            source_id=derive_source_id(URL, TEXT),
            canonical_reference="",
            kind=SourceKind.WEB,
            title="t",
            domain="example.org",
            retrieved_at=RETRIEVED,
            published_at=None,
            content_hash=sha256_hex(TEXT),
        )


def test_record_kind_must_match_its_reference() -> None:
    # A FILE kind beside an http value would name a source as something it is not,
    # and no derived identifier could reveal the mismatch later.
    with pytest.raises(SourceReferenceError, match="file reference"):
        SourceRecord(
            source_id=derive_source_id(URL, TEXT),
            canonical_reference=URL,
            kind=SourceKind.FILE,
            title="t",
            domain="",
            retrieved_at=RETRIEVED,
            published_at=None,
            content_hash=sha256_hex(TEXT),
        )


@pytest.mark.parametrize("digest", ["", "abc", sha256_hex(TEXT).upper(), sha256_hex(TEXT) + "0"])
def test_malformed_content_hash_is_rejected(digest: str) -> None:
    # A truncated or upper-case digest would compare unequal to a freshly computed
    # one and make a record look corrupt for no real reason.
    with pytest.raises(SourceRecordError, match="content_hash"):
        SourceRecord(
            source_id=derive_source_id(URL, TEXT),
            canonical_reference=URL,
            kind=SourceKind.WEB,
            title="t",
            domain="example.org",
            retrieved_at=RETRIEVED,
            published_at=None,
            content_hash=digest,
        )


def test_create_rejects_an_unusable_url() -> None:
    with pytest.raises(UrlError):
        make_source(url="ftp://example.org/a")


# ---------------------------------------------------------------------------
# Non-web source records (ADR-0011)
# ---------------------------------------------------------------------------


def test_create_file_records_a_local_source(tmp_path) -> None:
    path = tmp_path / "spec.md"
    path.write_text(TEXT, encoding="utf-8")
    record = SourceRecord.create_file(
        path=path, title="Specification", content=TEXT, retrieved_at=RETRIEVED
    )

    assert record.kind is SourceKind.FILE
    assert record.canonical_reference.startswith("file:///")
    # A local file has no host, so the domain is empty rather than invented.
    assert record.domain == ""
    assert record.source_id == derive_source_id(record.canonical_reference, TEXT)
    assert record.matches_content(TEXT) is True
    assert record.matches_content(REVISED) is False


def test_create_text_uses_content_as_its_identity() -> None:
    record = SourceRecord.create_text(content=TEXT, retrieved_at=RETRIEVED, label="clause-4")

    assert record.kind is SourceKind.TEXT
    assert record.canonical_reference == text_reference("clause-4").canonical
    assert record.domain == ""
    assert record.source_id == derive_source_id("text:clause-4", TEXT)


def test_two_identical_passages_are_one_source() -> None:
    # Identity is the content, so the same supplied text deduplicates rather than
    # filling an evidence pack with copies.
    first = SourceRecord.create_text(content=TEXT, retrieved_at=RETRIEVED)
    second = SourceRecord.create_text(content=TEXT, retrieved_at=RETRIEVED)

    assert first.source_id == second.source_id


def test_a_label_distinguishes_two_identical_passages() -> None:
    first = SourceRecord.create_text(content=TEXT, retrieved_at=RETRIEVED, label="appeals")
    second = SourceRecord.create_text(content=TEXT, retrieved_at=RETRIEVED, label="deadlines")

    assert first.source_id != second.source_id


def test_web_local_and_text_sources_of_one_content_are_distinct(tmp_path) -> None:
    # Folding the three kinds into one identifier formula cannot collide, because a
    # canonical URL, a file URI and a text reference are distinct strings.
    web = make_source()
    local = SourceRecord.create_file(
        path=tmp_path / "a.md", title="t", content=TEXT, retrieved_at=RETRIEVED
    )
    pasted = SourceRecord.create_text(content=TEXT, retrieved_at=RETRIEVED)

    assert web.canonical_reference == URL
    assert local.canonical_reference.startswith("file:///")
    assert pasted.canonical_reference == "text:"
    assert len({web.source_id, local.source_id, pasted.source_id}) == 3


def test_a_file_source_identity_is_deterministic(tmp_path) -> None:
    # Same path, same content, same identifier, regardless of the reference built
    # through the record or the abstraction directly.
    path = tmp_path / "spec.md"
    record = SourceRecord.create_file(path=path, title="t", content=TEXT, retrieved_at=RETRIEVED)
    reference = file_reference(path)

    assert record.canonical_reference == reference.canonical
    assert record.source_id == derive_source_id(reference.canonical, TEXT)


# ---------------------------------------------------------------------------
# EvidenceChunk
# ---------------------------------------------------------------------------


def test_chunk_create_derives_its_identifier() -> None:
    source = make_source()
    chunk = EvidenceChunk.create(source_id=source.source_id, text=TEXT, position=0)

    assert chunk.source_id == source.source_id
    assert str(chunk.chunk_id).startswith("chk_")
    assert chunk.position == 0


def test_chunk_records_scores_without_presenting_them() -> None:
    chunk = EvidenceChunk.create(
        source_id=make_source().source_id,
        text=TEXT,
        position=0,
        retrieval_score=0.82,
        rerank_score=0.4,
        token_count=11,
        section="4. Deadlines",
    )

    assert chunk.retrieval_score == 0.82
    assert chunk.rerank_score == 0.4
    assert chunk.token_count == 11
    assert chunk.section == "4. Deadlines"


def test_chunk_without_a_heading_records_none_rather_than_empty() -> None:
    chunk = EvidenceChunk.create(source_id=make_source().source_id, text=TEXT, position=0)

    assert chunk.section is None


@pytest.mark.parametrize("position", [-1, -100])
def test_negative_chunk_position_is_rejected(position: int) -> None:
    with pytest.raises(SourceRecordError, match="negative"):
        EvidenceChunk.create(source_id=make_source().source_id, text=TEXT, position=position)


def test_empty_chunk_text_is_rejected() -> None:
    # An empty chunk cannot support a claim, so it is not evidence.
    with pytest.raises(SourceRecordError, match="empty"):
        EvidenceChunk.create(source_id=make_source().source_id, text="", position=0)


def test_non_positive_token_count_is_rejected() -> None:
    with pytest.raises(SourceRecordError, match="token_count"):
        EvidenceChunk.create(
            source_id=make_source().source_id, text=TEXT, position=0, token_count=0
        )


# ---------------------------------------------------------------------------
# EvidencePack
# ---------------------------------------------------------------------------


def make_pack(*, source: SourceRecord | None = None, extra_source: SourceRecord | None = None):
    primary = source or make_source()
    chunks = [
        EvidenceChunk.create(source_id=primary.source_id, text=TEXT, position=0),
        EvidenceChunk.create(source_id=primary.source_id, text=OTHER_TEXT, position=1),
    ]
    sources = [primary]
    if extra_source is not None:
        sources.append(extra_source)
    return EvidencePack.of(chunks, sources)


def test_pack_indexes_its_sources() -> None:
    source = make_source()
    pack = make_pack(source=source)

    assert len(pack.chunks) == 2
    assert pack.source_for(source.source_id) == source
    assert pack.is_empty is False


def test_pack_drops_sources_no_chunk_references() -> None:
    # A pack records the evidence that was supplied. Keeping an unused source would
    # let an answer cite something the model never saw.
    unused = make_source(url="https://example.org/unused", content="unused")
    pack = make_pack(extra_source=unused)

    assert unused.source_id not in pack.sources
    assert str(unused.source_id) not in pack.supplied_identifiers


def test_supplied_identifiers_cover_chunks_and_sources() -> None:
    source = make_source()
    pack = make_pack(source=source)

    supplied = pack.supplied_identifiers
    assert str(source.source_id) in supplied
    for chunk in pack.chunks:
        assert str(chunk.chunk_id) in supplied


def test_pack_rejects_a_chunk_whose_source_is_missing() -> None:
    orphan = EvidenceChunk.create(source_id=SourceId("src_0000000000000000"), text=TEXT, position=0)

    with pytest.raises(SourceRecordError, match="missing the source record"):
        EvidencePack.of([orphan], [])


def test_pack_rejects_duplicate_chunk_identifiers() -> None:
    chunk = EvidenceChunk.create(source_id=make_source().source_id, text=TEXT, position=0)

    with pytest.raises(SourceRecordError, match="duplicate"):
        EvidencePack.of([chunk, chunk], [make_source()])


def test_empty_pack_is_allowed_and_says_so() -> None:
    # Retrieval may legitimately find nothing, and the answer then has to be
    # honest about having no evidence rather than failed to construct.
    pack = EvidencePack.of([], [])

    assert pack.is_empty is True
    assert pack.supplied_identifiers == frozenset()


def test_pack_records_the_retrieval_configuration() -> None:
    pack = EvidencePack.of(
        [EvidenceChunk.create(source_id=make_source().source_id, text=TEXT, position=0)],
        [make_source()],
        retrieval_config={"index": "lexical", "top_k": "5"},
    )

    assert pack.retrieval_config["index"] == "lexical"


def test_pack_lookup_by_identifier_returns_none_when_absent() -> None:
    pack = make_pack()

    assert pack.chunk_for(ChunkId("chk_0000000000000000")) is None
    assert pack.source_for(SourceId("src_0000000000000000")) is None


# ---------------------------------------------------------------------------
# Citation resolution
# ---------------------------------------------------------------------------


def test_chunk_reference_resolves_to_its_chunk_and_source() -> None:
    source = make_source()
    pack = make_pack(source=source)
    chunk = pack.chunks[0]

    resolution = resolve_citations(f"The window closes in April [{chunk.chunk_id}].", pack)

    assert resolution.is_clean is True
    assert len(resolution.citations) == 1
    citation = resolution.citations[0]
    assert isinstance(citation, Citation)
    assert citation.chunk == chunk
    assert citation.source == source
    assert citation.identifier == str(chunk.chunk_id)


def test_source_reference_resolves_without_a_chunk() -> None:
    source = make_source()
    pack = make_pack(source=source)

    resolution = resolve_citations(f"See the specification [{source.source_id}].", pack)

    assert resolution.is_clean is True
    assert resolution.citations[0].chunk is None
    assert resolution.citations[0].source == source


def test_fabricated_identifier_is_recorded_and_never_resolved() -> None:
    # This is the rule ADR-0003 exists to enforce: an identifier outside the pack
    # must not become a rendered URL.
    pack = make_pack()
    invented = "src_aaaaaaaaaaaaaaaa"

    resolution = resolve_citations(f"The deadline is in March [{invented}].", pack)

    assert resolution.citations == ()
    assert resolution.unknown_identifiers == (invented,)
    assert resolution.is_clean is False


def test_malformed_reference_is_recorded_separately() -> None:
    # A model that attempted a citation and got the form wrong is a different
    # failure from one that invented a well-formed identifier, and the two should
    # not be averaged together.
    pack = make_pack()

    resolution = resolve_citations(
        "See [chk_0123456789ABCDEF], [src_0123] and [src_0123456789abcdef0].", pack
    )

    assert resolution.malformed_identifiers == (
        "chk_0123456789ABCDEF",
        "src_0123",
        "src_0123456789abcdef0",
    )
    assert resolution.unknown_identifiers == ()


def test_repeated_reference_is_counted_once() -> None:
    # Counting it three times would inflate a citation metric.
    source = make_source()
    pack = make_pack(source=source)

    resolution = resolve_citations(
        f"a [{source.source_id}] b [{source.source_id}] c [{source.source_id}]", pack
    )

    assert len(resolution.citations) == 1


def test_cited_source_ids_are_distinct_and_ordered() -> None:
    source = make_source()
    pack = make_pack(source=source)
    first, second = pack.chunks

    resolution = resolve_citations(f"[{first.chunk_id}] [{second.chunk_id}]", pack)

    assert resolution.cited_source_ids == (source.source_id,)


def test_ordinary_prose_is_not_mistaken_for_a_reference() -> None:
    # A false positive in this metric is worse than a missed one, so the pattern
    # requires enough hexadecimal characters to look like an attempted digest.
    pack = make_pack()

    for text in ("see src_file for details", "the srcdoc attribute", "chk_sum mismatch"):
        resolution = resolve_citations(text, pack)
        assert resolution == CitationResolution()


def test_text_with_no_references_resolves_to_nothing() -> None:
    assert (
        resolve_citations("A plain answer with no citations.", make_pack()) == CitationResolution()
    )


def test_resolution_is_clean_when_there_was_nothing_to_resolve() -> None:
    assert CitationResolution().is_clean is True
