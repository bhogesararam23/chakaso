"""Tests for the lexical index and BM25 retriever.

The properties that matter: ranking is deterministic and reproducible, an unmatched
query returns nothing rather than something invented, ties fall to a total order, and
the explanation carries only what was computed. The retriever is checked through its
protocol too, because being replaceable is the point of the boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.evidence import EvidenceChunk
from chakaso.retrieval import (
    ChunkingConfig,
    LexicalRetriever,
    RetrievalError,
    RetrievalResult,
    Retriever,
    build_lexical_index,
    ingest_text,
    tokenize,
)

RETRIEVED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
ONE_CHUNK_PER_PARAGRAPH = ChunkingConfig(target_characters=1, max_characters=2000)


def chunk(text: str, label: str) -> EvidenceChunk:
    document = ingest_text(text, label=label, config=ONE_CHUNK_PER_PARAGRAPH, now=lambda: RETRIEVED)
    return document.chunks[0]


def index_of(*chunks: EvidenceChunk):
    return build_lexical_index(chunks)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_tokenize_folds_case_and_splits_punctuation() -> None:
    assert tokenize("The Deadline, is REAL!") == ("the", "deadline", "is", "real")


def test_tokenize_keeps_unicode_word_characters() -> None:
    assert tokenize("Café naïve") == ("café", "naïve")


def test_tokenize_of_no_words_is_empty() -> None:
    assert tokenize("... --- !!!") == ()


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_more_matches_rank_higher() -> None:
    heavy = chunk("apple apple apple", "heavy")
    light = chunk("apple banana", "light")
    retriever = LexicalRetriever(index_of(heavy, light))

    results = retriever.retrieve("apple", top_k=5)

    assert [result.chunk.chunk_id for result in results] == [heavy.chunk_id, light.chunk_id]
    assert results[0].score > results[1].score


def test_a_query_matching_nothing_returns_nothing() -> None:
    # Retrieval that found nothing must not fabricate a result.
    retriever = LexicalRetriever(index_of(chunk("only words here", "a")))

    assert retriever.retrieve("elephant", top_k=5) == ()


def test_ranks_are_sequential_from_one() -> None:
    retriever = LexicalRetriever(
        index_of(chunk("deadline", "a"), chunk("deadline appeal", "b"), chunk("deadline", "c"))
    )

    results = retriever.retrieve("deadline", top_k=5)

    assert [result.rank for result in results] == [1, 2, 3]


def test_matched_terms_only_lists_terms_present_in_the_chunk() -> None:
    doc = chunk("the appeal deadline", "d")
    retriever = LexicalRetriever(index_of(doc))

    (result,) = retriever.retrieve("deadline missingword", top_k=5)

    assert result.matched_terms == ("deadline",)


def test_top_k_limits_the_result_count() -> None:
    retriever = LexicalRetriever(
        index_of(chunk("alpha", "1"), chunk("alpha", "2"), chunk("alpha", "3"))
    )

    assert len(retriever.retrieve("alpha", top_k=2)) == 2


def test_top_k_larger_than_corpus_returns_every_match() -> None:
    retriever = LexicalRetriever(index_of(chunk("alpha", "1"), chunk("alpha", "2")))

    assert len(retriever.retrieve("alpha", top_k=100)) == 2


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_retrieval_is_deterministic_across_calls() -> None:
    retriever = LexicalRetriever(
        index_of(chunk("deadline appeals", "a"), chunk("deadline policy", "b"))
    )

    first = retriever.retrieve("deadline", top_k=5)
    second = retriever.retrieve("deadline", top_k=5)

    assert [(r.chunk.chunk_id, r.score, r.rank) for r in first] == [
        (r.chunk.chunk_id, r.score, r.rank) for r in second
    ]


def test_tied_scores_are_broken_by_chunk_identifier() -> None:
    # Two identical chunks in different sources have equal scores; their order must be
    # the total order of their identifiers, not dictionary insertion order.
    left = chunk("identical body", "l")
    right = chunk("identical body", "r")
    retriever = LexicalRetriever(index_of(left, right))

    results = retriever.retrieve("identical", top_k=5)

    assert [result.score for result in results] == [results[0].score, results[0].score]
    assert results[0].chunk.chunk_id < results[1].chunk.chunk_id


def test_index_order_does_not_change_the_ranking() -> None:
    a = chunk("alpha beta", "a")
    b = chunk("beta gamma", "b")
    forward = LexicalRetriever(index_of(a, b))
    backward = LexicalRetriever(index_of(b, a))

    assert [r.chunk.chunk_id for r in forward.retrieve("beta", top_k=5)] == [
        r.chunk.chunk_id for r in backward.retrieve("beta", top_k=5)
    ]


# ---------------------------------------------------------------------------
# Filtering and empties
# ---------------------------------------------------------------------------


def test_source_filter_restricts_candidates() -> None:
    doc_a = chunk("deadline", "a")
    doc_b = chunk("deadline", "b")
    retriever = LexicalRetriever(index_of(doc_a, doc_b))

    results = retriever.retrieve("deadline", top_k=5, source_ids=[doc_a.source_id])

    assert [r.chunk.chunk_id for r in results] == [doc_a.chunk_id]


def test_an_empty_index_retrieves_nothing() -> None:
    retriever = LexicalRetriever(build_lexical_index(()))

    assert retriever.retrieve("anything", top_k=5) == ()


def test_an_empty_query_retrieves_nothing() -> None:
    retriever = LexicalRetriever(index_of(chunk("alpha", "a")))

    assert retriever.retrieve("", top_k=5) == ()
    assert retriever.retrieve("!!!", top_k=5) == ()


# ---------------------------------------------------------------------------
# Boundary and validation
# ---------------------------------------------------------------------------


def test_lexical_retriever_satisfies_the_retriever_protocol() -> None:
    retriever: Retriever = LexicalRetriever(index_of(chunk("alpha", "a")))

    assert isinstance(retriever, Retriever)
    assert isinstance(retriever.retrieve("alpha", top_k=1)[0], RetrievalResult)


@pytest.mark.parametrize("top_k", [0, -3])
def test_top_k_below_one_is_rejected(top_k: int) -> None:
    retriever = LexicalRetriever(index_of(chunk("alpha", "a")))

    with pytest.raises(RetrievalError, match="at least 1"):
        retriever.retrieve("alpha", top_k=top_k)


def test_non_positive_k1_is_rejected() -> None:
    with pytest.raises(RetrievalError, match="k1"):
        LexicalRetriever(index_of(chunk("a", "a")), k1=0.0)


@pytest.mark.parametrize("b", [-0.1, 1.5])
def test_b_must_be_in_the_unit_interval(b: float) -> None:
    with pytest.raises(RetrievalError, match="between 0 and 1"):
        LexicalRetriever(index_of(chunk("a", "a")), b=b)
