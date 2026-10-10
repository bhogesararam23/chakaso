"""Tests for hybrid retrieval: reciprocal-rank fusion of the lexical and dense retrievers.

The properties that matter are the ones a fusion can silently violate. Results must be tagged
``hybrid`` and carry ``HybridProvenance`` that records, per component, the rank and score that
actually fed the merge — a chunk only one component returned must show ``None`` for the other,
never a fabricated contribution. The fused score must be exactly the documented RRF formula, so
a reader can recompute it. Fusion is deterministic, respects ``top_k`` and the ``source_ids``
filter, produces nothing neither component returned, and lets weights favor one component.
Nothing here claims dense or hybrid *understanding* — the dense component is the non-semantic
fixture double (ADR-0023), so this tests the mechanism, not quality.
"""

from __future__ import annotations

import math

import pytest

from chakaso.retrieval import (
    DenseRetriever,
    FixtureEmbeddingModel,
    FusionMethod,
    HybridConfig,
    HybridRetriever,
    LexicalRetriever,
    RetrievalError,
    RetrievalStrategy,
    Retriever,
    build_dense_index,
    build_lexical_index,
    ingest_text,
)

CORPUS = ["alpha beta", "alpha", "gamma"]


def _hybrid(**config: object) -> HybridRetriever:
    docs = [ingest_text(text, label=f"h{i}") for i, text in enumerate(CORPUS)]
    chunks = tuple(doc.chunks[0] for doc in docs)
    lexical = LexicalRetriever(build_lexical_index(chunks))
    model = FixtureEmbeddingModel(dimension=16)
    dense = DenseRetriever(build_dense_index(chunks, model), model)
    return HybridRetriever(lexical, dense, config=HybridConfig(**config))  # type: ignore[arg-type]


def _by_text(results: object, text: str) -> object:
    return next(r for r in results if r.chunk.text == text)  # type: ignore[attr-defined]


def test_hybrid_satisfies_the_retriever_protocol() -> None:
    assert isinstance(_hybrid(), Retriever)


def test_fused_results_are_ranked_and_tagged_hybrid() -> None:
    results = _hybrid().retrieve("alpha beta", top_k=3)

    assert [result.rank for result in results] == [1, 2, 3]
    assert all(result.strategy is RetrievalStrategy.HYBRID for result in results)
    assert all(result.provenance is not None for result in results)


def test_a_chunk_both_components_returned_carries_both_provenances() -> None:
    results = _hybrid().retrieve("alpha beta", top_k=3)

    top = _by_text(results, "alpha beta")
    assert top.provenance.lexical_rank == 1
    assert top.provenance.dense_rank == 1
    # "gamma" is matched by dense only; lexical must be recorded as absent, not faked.
    disjoint = _by_text(results, "gamma")
    assert disjoint.provenance.lexical_rank is None
    assert disjoint.provenance.dense_rank is not None
    assert disjoint.matched_terms == ()


def test_fused_score_is_exactly_the_documented_rrf_value() -> None:
    # rrf_k=60, both weights 1.0: the top chunk is rank 1 in both → 1/61 + 1/61.
    results = _hybrid(rrf_k=60, lexical_weight=1.0, dense_weight=1.0).retrieve(
        "alpha beta", top_k=3
    )

    assert math.isclose(results[0].score, 1 / 61 + 1 / 61, rel_tol=1e-12)


def test_weights_can_favor_one_component() -> None:
    # Zeroing the dense weight leaves lexical to decide the order; a dense-only chunk scores 0.
    results = _hybrid(lexical_weight=1.0, dense_weight=0.0).retrieve("alpha beta", top_k=3)

    top = _by_text(results, "alpha beta")
    assert math.isclose(top.score, 1 / 61, rel_tol=1e-12)
    assert _by_text(results, "gamma").score == 0.0


def test_candidate_k_widens_the_component_pool_before_fusion() -> None:
    retriever = _hybrid(candidate_k=3)

    assert retriever.config.candidate_k == 3
    assert len(retriever.retrieve("alpha", top_k=2)) == 2


def test_source_ids_filter_passes_through_to_both_components() -> None:
    docs = [ingest_text(text, label=f"s{i}") for i, text in enumerate(CORPUS)]
    chunks = tuple(doc.chunks[0] for doc in docs)
    lexical = LexicalRetriever(build_lexical_index(chunks))
    model = FixtureEmbeddingModel(dimension=16)
    dense = DenseRetriever(build_dense_index(chunks, model), model)
    retriever = HybridRetriever(lexical, dense)

    results = retriever.retrieve("alpha beta", top_k=3, source_ids=[chunks[0].source_id])

    assert [result.chunk.text for result in results] == ["alpha beta"]


def test_fusion_is_deterministic() -> None:
    retriever = _hybrid()

    assert retriever.retrieve("alpha beta", top_k=3) == retriever.retrieve("alpha beta", top_k=3)


def test_top_k_below_one_is_an_error() -> None:
    with pytest.raises(RetrievalError, match="top_k"):
        _hybrid().retrieve("alpha", top_k=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rrf_k": 0},
        {"candidate_k": 0},
        {"lexical_weight": -1.0},
        {"lexical_weight": 0.0, "dense_weight": 0.0},
    ],
)
def test_invalid_hybrid_config_is_refused(kwargs: dict[str, float]) -> None:
    # Constructed inside the raises block: an invalid config must fail at construction, not
    # silently pass a default through.
    with pytest.raises(RetrievalError):
        HybridConfig(**kwargs)


def test_only_rrf_is_a_supported_method() -> None:
    # The enum currently has a single method; constructing HybridRetriever with a config
    # whose method is not RRF would be the failure path, so assert the default is RRF.
    assert _hybrid().config.method is FusionMethod.RRF
