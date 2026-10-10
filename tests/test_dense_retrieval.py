"""Tests for the dense retrieval boundary and the exact index built on it.

Three things are pinned here. First, the embedding double is honest about being a
development double: it is deterministic and batch-ordered, but it is not semantic, and that
is asserted, not just documented. Second, the dense index and retriever enforce their own
structural rules — uniform dimension, no duplicate chunk, the empty/zero cases returning
nothing rather than a fabricated ranking. Third, the retriever behaves like a retriever:
results carry the ``dense`` strategy with no invented matched terms, ranking is total and
reproducible, ``source_ids`` filters, and — because the fixture is a hashed bag-of-tokens —
the most token-overlapping chunk ranks first, which is exactly the non-semantic behaviour
ADR-0023 refuses to dress up as understanding.
"""

from __future__ import annotations

import math

import pytest

from chakaso.evidence import EvidenceChunk
from chakaso.retrieval import (
    DenseIndex,
    DenseRetriever,
    EmbeddingMetadata,
    EmbeddingModel,
    EmbeddingVector,
    EmbedModelError,
    FixtureEmbeddingModel,
    RetrievalError,
    RetrievalStrategy,
    Retriever,
    SimilarityMetric,
    build_dense_index,
    ingest_text,
)


def _norm(vector: EmbeddingVector) -> float:
    return math.sqrt(sum(value * value for value in vector.values))


# -- the embedding double --------------------------------------------------------


def test_the_fixture_declares_itself_a_non_semantic_double() -> None:
    metadata = FixtureEmbeddingModel().metadata

    assert metadata.is_semantic is False
    assert metadata.development_double is True
    assert metadata.normalized is True
    assert metadata.dimension == 64


def test_fixture_embedding_model_satisfies_the_protocol() -> None:
    assert isinstance(FixtureEmbeddingModel(), EmbeddingModel)


def test_embed_is_batch_ordered_and_deterministic() -> None:
    model = FixtureEmbeddingModel(dimension=32)
    texts = ["alpha beta", "gamma", "alpha beta"]

    vectors = model.embed(texts)

    assert len(vectors) == len(texts)
    assert all(vector.dimension == 32 for vector in vectors)
    # Same text in, same vector out — position 0 and position 2 are the same input.
    assert vectors[0] == vectors[2]
    # A deterministic double reproduces across separate models of the same configuration.
    assert FixtureEmbeddingModel(dimension=32).embed(["alpha beta"])[0] == vectors[0]


def test_distinct_texts_get_distinct_vectors_and_empty_gets_zero() -> None:
    model = FixtureEmbeddingModel(dimension=32)
    alpha, beta = model.embed(["alpha", "beta"])

    assert alpha != beta
    (empty,) = model.embed([""])
    assert _norm(empty) == 0.0


def test_normalized_vectors_have_unit_length() -> None:
    (vector,) = FixtureEmbeddingModel(dimension=16).embed(["several words here"])

    assert math.isclose(_norm(vector), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("dimension", [0, -1])
def test_fixture_rejects_bad_dimension(dimension: int) -> None:
    with pytest.raises(EmbedModelError, match="dimension"):
        FixtureEmbeddingModel(dimension=dimension)


def test_fixture_rejects_empty_seed() -> None:
    with pytest.raises(EmbedModelError, match="seed"):
        FixtureEmbeddingModel(seed="")


def test_vector_rejects_empty_and_non_finite_values() -> None:
    with pytest.raises(EmbedModelError, match="at least one dimension"):
        EmbeddingVector(())
    with pytest.raises(EmbedModelError, match="finite"):
        EmbeddingVector((float("nan"), 1.0))


def test_metadata_requires_a_model_id_and_positive_dimension() -> None:
    with pytest.raises(EmbedModelError, match="model_id"):
        EmbeddingMetadata(
            model_id=" ", dimension=4, normalized=True, development_double=True, is_semantic=False
        )
    with pytest.raises(EmbedModelError, match="dimension"):
        EmbeddingMetadata(
            model_id="m", dimension=0, normalized=True, development_double=True, is_semantic=False
        )


# -- the index -------------------------------------------------------------------


def _chunks() -> tuple[EvidenceChunk, ...]:
    docs = [ingest_text(text, label=f"doc{i}") for i, text in enumerate(["alpha beta", "delta"])]
    return tuple(doc.chunks[0] for doc in docs)


def test_index_is_built_once_and_aligned_with_its_chunks() -> None:
    chunks = _chunks()
    index = build_dense_index(chunks, FixtureEmbeddingModel(dimension=16))

    assert index.document_count == len(chunks)
    assert index.dimension == 16
    assert len(index.vectors) == len(index.chunks)


def test_index_rejects_mismatched_chunk_and_vector_counts() -> None:
    chunks = _chunks()
    model = FixtureEmbeddingModel(dimension=8)
    vectors = model.embed([chunks[0].text])

    with pytest.raises(RetrievalError, match="must match"):
        DenseIndex(chunks, vectors)


def test_index_rejects_non_uniform_dimensions_and_duplicates() -> None:
    chunks = _chunks()
    good = FixtureEmbeddingModel(dimension=8).embed([c.text for c in chunks])
    ragged = (good[0], EmbeddingVector((0.0, 0.0, 0.0)))

    with pytest.raises(RetrievalError, match="uniform"):
        DenseIndex(chunks, ragged)

    with pytest.raises(RetrievalError, match="duplicate"):
        DenseIndex((chunks[0], chunks[0]), (good[0], good[0]))


def test_empty_index_has_zero_dimension() -> None:
    assert DenseIndex((), ()).dimension == 0


# -- the retriever ---------------------------------------------------------------


def _corpus_chunks() -> tuple[EvidenceChunk, ...]:
    texts = ["alpha beta gamma", "alpha beta", "delta epsilon"]
    docs = [ingest_text(text, label=f"c{i}") for i, text in enumerate(texts)]
    return tuple(doc.chunks[0] for doc in docs)


def _retriever(*, metric: SimilarityMetric = SimilarityMetric.COSINE) -> DenseRetriever:
    model = FixtureEmbeddingModel(dimension=32)
    index = build_dense_index(_corpus_chunks(), model)
    return DenseRetriever(index, model, metric=metric)


def test_dense_retriever_satisfies_the_retriever_protocol() -> None:
    assert isinstance(_retriever(), Retriever)


def test_results_are_ranked_tagged_dense_and_free_of_invented_terms() -> None:
    results = _retriever().retrieve("alpha beta", top_k=3)

    assert [result.rank for result in results] == [1, 2, 3]
    assert all(result.strategy is RetrievalStrategy.DENSE for result in results)
    # A similarity search matched no terms; reporting an empty term list is honest.
    assert all(result.matched_terms == () for result in results)
    # The fixture is token overlap: the chunk sharing both query tokens ranks first.
    assert "alpha beta" in results[0].chunk.text


def test_a_disjoint_chunk_scores_zero_and_a_full_match_scores_one() -> None:
    results = {
        result.chunk.text: result.score for result in _retriever().retrieve("alpha beta", top_k=3)
    }

    assert math.isclose(results["alpha beta"], 1.0, rel_tol=1e-9)
    assert math.isclose(results["delta epsilon"], 0.0, abs_tol=1e-9)


def test_source_ids_filter_the_candidates() -> None:
    chunks = _corpus_chunks()
    model = FixtureEmbeddingModel(dimension=32)
    index = build_dense_index(chunks, model)
    retriever = DenseRetriever(index, model)

    results = retriever.retrieve("alpha beta", top_k=3, source_ids=[chunks[2].source_id])

    assert [result.chunk.text for result in results] == ["delta epsilon"]


def test_retrieval_is_deterministic() -> None:
    retriever = _retriever()

    assert retriever.retrieve("alpha beta", top_k=3) == retriever.retrieve("alpha beta", top_k=3)


def test_dot_and_cosine_agree_on_unit_vectors_and_both_rank_the_best_match_first() -> None:
    cosine = _retriever(metric=SimilarityMetric.COSINE).retrieve("alpha beta", top_k=1)
    dot = _retriever(metric=SimilarityMetric.DOT).retrieve("alpha beta", top_k=1)

    assert cosine[0].chunk.text == dot[0].chunk.text == "alpha beta"


def test_empty_query_and_empty_index_return_nothing() -> None:
    assert _retriever().retrieve("   ", top_k=3) == ()

    empty_index = DenseIndex((), ())
    retriever = DenseRetriever(empty_index, FixtureEmbeddingModel(dimension=8))
    assert retriever.retrieve("alpha", top_k=3) == ()


def test_top_k_below_one_is_an_error() -> None:
    with pytest.raises(RetrievalError, match="top_k"):
        _retriever().retrieve("alpha", top_k=0)


def test_retriever_rejects_an_embedder_of_a_different_dimension() -> None:
    model = FixtureEmbeddingModel(dimension=32)
    index = build_dense_index(_corpus_chunks(), model)
    mismatched = FixtureEmbeddingModel(dimension=8)

    with pytest.raises(RetrievalError, match="does not match"):
        DenseRetriever(index, mismatched)


def test_retriever_reports_the_embedding_model_identity() -> None:
    metadata = _retriever().embedding_metadata

    assert metadata.development_double is True
    assert metadata.is_semantic is False
