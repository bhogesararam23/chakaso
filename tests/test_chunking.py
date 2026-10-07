"""Tests for chunking.

A chunk is what an answer cites, so the assertions here are about the properties a
citation depends on: boundaries that can be pointed at, identifiers that follow the
content, and output that is the same every time.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.core.identifiers import derive_source_id
from chakaso.evidence import SourceRecord
from chakaso.retrieval import ChunkingConfig, ChunkingError, chunk_document, split_sections

URL = "https://example.org/specification"
RETRIEVED = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)

DOCUMENT = """\
An introduction that appears before any heading.

It has two paragraphs.

# 4. Deadlines

The submission window closes on 14 April.

Extensions are granted only for documented medical reasons.

## 4.1 Extensions

Requests must be made in writing.

# 5. Appeals

Appeals are heard by a panel.
"""


def source() -> SourceRecord:
    return SourceRecord.create(
        url=URL, title="Specification", content=DOCUMENT, retrieved_at=RETRIEVED
    )


def chunks(text: str = DOCUMENT, *, config: ChunkingConfig | None = None):
    return chunk_document(source().source_id, text, config=config)


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------


def test_sections_follow_heading_boundaries() -> None:
    sections = split_sections(DOCUMENT)

    assert [section.heading for section in sections] == [
        None,
        "4. Deadlines",
        "4. Deadlines > 4.1 Extensions",
        "5. Appeals",
    ]


def test_a_heading_keeps_its_body_and_not_the_next_heading() -> None:
    sections = split_sections(DOCUMENT)
    deadlines = next(section for section in sections if section.heading == "4. Deadlines")

    assert "closes on 14 April" in deadlines.body
    assert "4.1 Extensions" not in deadlines.body


def test_nested_headings_report_their_path() -> None:
    # A citation pointing at a subsection has to say where the subsection sits.
    sections = split_sections(DOCUMENT)
    extensions = next(
        section for section in sections if section.heading == "4. Deadlines > 4.1 Extensions"
    )

    assert "in writing" in extensions.body


def test_text_before_the_first_heading_is_kept() -> None:
    sections = split_sections(DOCUMENT)

    assert sections[0].heading is None
    assert "before any heading" in sections[0].body


def test_a_document_with_no_headings_is_one_section() -> None:
    sections = split_sections("Just a paragraph.\n\nAnd another.")

    assert len(sections) == 1
    assert sections[0].heading is None


def test_heading_levels_can_skip() -> None:
    # Markdown allows a jump from # to ###. The path is shorter than the level would
    # imply, and that is better than inventing a heading nobody wrote.
    sections = split_sections("# Top\n\nbody\n\n### Deep\n\nmore\n")

    assert [section.heading for section in sections] == ["Top", "Top > Deep"]


def test_empty_heading_text_does_not_produce_an_empty_path() -> None:
    sections = split_sections("#\n\nbody\n")

    assert [section.heading for section in sections] == [None]


def test_a_trailing_heading_with_no_body_is_dropped() -> None:
    # A section with no text cannot produce a chunk, and an empty section would only
    # create an empty heading to reason about.
    assert split_sections("# Title\n") == ()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_positions_are_sequential_and_cover_the_document_in_order() -> None:
    result = chunks()

    assert [chunk.position for chunk in result] == list(range(len(result)))
    joined = " ".join(chunk.text for chunk in result)
    assert joined.index("introduction") < joined.index("closes on 14 April")
    assert joined.index("closes on 14 April") < joined.index("Appeals are heard")


def test_chunks_never_cross_a_section_boundary() -> None:
    # A chunk spanning two headings has no citation position a reader could follow.
    result = chunks()
    headings = {chunk.section for chunk in result}

    assert headings == {
        None,
        "4. Deadlines",
        "4. Deadlines > 4.1 Extensions",
        "5. Appeals",
    }
    deadlines = [chunk for chunk in result if chunk.section == "4. Deadlines"]
    assert all("in writing" not in chunk.text for chunk in deadlines)
    assert all("Appeals are heard" not in chunk.text for chunk in deadlines)


def test_paragraphs_are_packed_up_to_the_target() -> None:
    text = "\n\n".join(["a" * 40] * 4)
    result = chunk_document(
        source().source_id, text, config=ChunkingConfig(target_characters=100, max_characters=200)
    )

    # 40 + 2 + 40 fits in 100; a third would be 124, so two paragraphs per chunk.
    assert [chunk.text.count("a") for chunk in result] == [80, 80]


def test_a_single_paragraph_larger_than_the_ceiling_is_split_at_sentences() -> None:
    sentence = "This sentence is exactly long enough to matter in the arithmetic. "
    text = sentence * 8
    config = ChunkingConfig(target_characters=100, max_characters=200)

    result = chunk_document(source().source_id, text, config=config)

    assert len(result) > 1
    assert all(len(chunk.text) <= config.max_characters for chunk in result)


def test_a_sentence_larger_than_the_ceiling_is_cut() -> None:
    # The alternative is a chunk larger than the ceiling the caller was promised, and
    # an unbounded chunk is worse than a truncated sentence.
    config = ChunkingConfig(target_characters=50, max_characters=100)

    result = chunk_document(source().source_id, "x" * 250, config=config)

    assert [len(chunk.text) for chunk in result] == [100, 100, 50]


def test_chunk_text_is_trimmed_and_paragraph_breaks_are_normalised() -> None:
    result = chunks("  First paragraph.  \n\n\n\n  Second paragraph.  \n")

    assert result[0].text == "First paragraph.\n\nSecond paragraph."


def test_a_document_with_no_content_produces_no_chunks() -> None:
    # An empty chunk is not evidence and the record type rejects it, so returning one
    # would be a worse answer than returning nothing.
    assert chunks("") == ()
    assert chunks("   \n\n  \n") == ()


def test_the_same_input_produces_identical_chunks() -> None:
    # Evaluation compares runs; a chunker with any variability would make retrieved
    # evidence incomparable between two runs of one configuration.
    assert chunks() == chunks()


def test_chunk_identifiers_change_when_the_text_changes() -> None:
    # ADR-0007: an earlier answer must keep pointing at the text it actually used.
    original = chunks("A paragraph about deadlines.")
    revised = chunks("A paragraph about appeals.")

    assert original[0].chunk_id != revised[0].chunk_id


def test_chunk_identifiers_change_when_the_position_changes() -> None:
    first = chunks("one\n\ntwo")
    second = chunks("zero\n\none\n\ntwo")

    assert first[0].text != second[0].text
    assert {chunk.chunk_id for chunk in first}.isdisjoint({chunk.chunk_id for chunk in second})


def test_different_chunking_settings_produce_a_disjoint_set() -> None:
    coarse = chunks(config=ChunkingConfig(target_characters=10_000, max_characters=20_000))
    fine = chunks(config=ChunkingConfig(target_characters=40, max_characters=80))

    assert len(coarse) < len(fine)
    assert {chunk.chunk_id for chunk in coarse}.isdisjoint({chunk.chunk_id for chunk in fine})


def test_chunks_carry_no_token_count() -> None:
    # Counting tokens needs a tokenizer, and a character count in a field named
    # token_count would be a lie a later reader would trust.
    assert all(chunk.token_count is None for chunk in chunks())


def test_chunks_carry_no_retrieval_or_rerank_score() -> None:
    # Chunking does not rank. A score here would have to be invented.
    assert all(chunk.retrieval_score is None for chunk in chunks())
    assert all(chunk.rerank_score is None for chunk in chunks())


def test_chunking_does_not_change_the_text_it_was_given() -> None:
    # Every character of a document must survive into some chunk, or a citation could
    # point at a chunk whose text is not what the source said.
    result = chunks("Alpha beta.\n\nGamma delta.\n\nEpsilon zeta.")
    assert " ".join(" ".join(chunk.text.split()) for chunk in result) == (
        "Alpha beta. Gamma delta. Epsilon zeta."
    )


def test_chunks_reference_the_source_they_came_from() -> None:
    record = source()
    result = chunk_document(record.source_id, DOCUMENT)

    assert all(chunk.source_id == record.source_id for chunk in result)


def test_a_source_with_no_chunks_is_still_a_valid_source() -> None:
    # Nothing here produces an empty evidence pack; that is the pack's business.
    assert chunk_document(derive_source_id(URL, ""), "") == ()


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0, -1])
def test_target_size_must_be_positive(target: int) -> None:
    with pytest.raises(ChunkingError, match="at least 1"):
        ChunkingConfig(target_characters=target)


def test_ceiling_below_target_is_rejected() -> None:
    # Every chunk is expected to fit the ceiling, so a ceiling below the target is
    # unsatisfiable and would silently become the real target.
    with pytest.raises(ChunkingError, match="must not be below"):
        ChunkingConfig(target_characters=100, max_characters=50)


def test_settings_may_be_equal() -> None:
    assert ChunkingConfig(target_characters=100, max_characters=100).max_characters == 100


def test_default_settings_are_usable() -> None:
    assert chunks()


def test_document_matching_an_evidence_pack_flow() -> None:
    # The end of the path this unit exists for: text becomes chunks, and chunks become
    # an evidence pack that citation resolution can check against.
    from chakaso.evidence import EvidencePack

    record = source()
    result = chunk_document(record.source_id, DOCUMENT)
    pack = EvidencePack.of(result, [record])

    assert pack.chunks == result
    assert str(result[0].chunk_id) in pack.supplied_identifiers
