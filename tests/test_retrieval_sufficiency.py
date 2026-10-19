"""Tests for the retrieval evidence-sufficiency policy (ADR-0032 / P28)."""

from __future__ import annotations

import pytest

from chakaso.benchmark.cases import ExpectedBehavior
from chakaso.benchmark.grounded_workload import load_development_grounded_workload
from chakaso.retrieval import Corpus, RetrievalService, ingest_text
from chakaso.retrieval.errors import RetrievalError
from chakaso.retrieval.sufficiency import (
    DiscriminatingTermSufficiencyPolicy,
    EvidenceSufficiencyPolicy,
    SufficiencyOutcome,
)


def _corpus(texts: list[str]) -> Corpus:
    corpus = Corpus()
    for i, text in enumerate(texts):
        corpus.add_document(ingest_text(text, label=f"doc-{i}"))
    return corpus


class TestDiscriminatingTermPolicy:
    def test_protocol_is_satisfied(self) -> None:
        policy = DiscriminatingTermSufficiencyPolicy(_corpus(["the cat sat"]))
        assert isinstance(policy, EvidenceSufficiencyPolicy)

    def test_rejects_out_of_range_fraction(self) -> None:
        corpus = _corpus(["some text"])
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(RetrievalError):
                DiscriminatingTermSufficiencyPolicy(corpus, max_document_fraction=bad)

    def test_unique_term_present_is_sufficient(self) -> None:
        corpus = _corpus(["the cat sat", "the dog ran", "the bird sang", "the fish swam"])
        policy = DiscriminatingTermSufficiencyPolicy(corpus, max_document_fraction=0.25)
        result = policy.assess("which dog ran", list(corpus.chunks))
        assert result.is_sufficient
        assert "dog" in result.discriminating_terms  # df=1, unique to one doc

    def test_only_common_word_matches_is_insufficient(self) -> None:
        corpus = _corpus(["the cat sat", "the dog ran", "the bird sang", "the fish swam"])
        policy = DiscriminatingTermSufficiencyPolicy(corpus, max_document_fraction=0.25)
        # 'the' is in all four docs (df=4, not discriminating); no query discriminating term is matched.
        result = policy.assess("what is the thing", list(corpus.chunks))
        assert result.outcome is SufficiencyOutcome.INSUFFICIENT
        assert result.discriminating_terms == ()

    def test_absent_term_is_never_sufficient(self) -> None:
        corpus = _corpus(["the cat sat", "the dog ran"])
        policy = DiscriminatingTermSufficiencyPolicy(corpus, max_document_fraction=0.5)
        result = policy.assess("a dinosaur skeleton", list(corpus.chunks))
        assert result.outcome is SufficiencyOutcome.INSUFFICIENT

    def test_empty_corpus_is_insufficient(self) -> None:
        policy = DiscriminatingTermSufficiencyPolicy(Corpus())
        result = policy.assess("anything at all", [])
        assert result.outcome is SufficiencyOutcome.INSUFFICIENT

    def test_empty_candidates_is_insufficient(self) -> None:
        corpus = _corpus(["the quick brown fox"])
        policy = DiscriminatingTermSufficiencyPolicy(corpus)
        result = policy.assess("quick fox", [])
        assert result.outcome is SufficiencyOutcome.INSUFFICIENT
        assert result.candidate_chunk_count == 0

    def test_decision_is_deterministic(self) -> None:
        corpus = _corpus(["alpha beta", "alpha gamma", "delta epsilon", "zeta eta"])
        policy = DiscriminatingTermSufficiencyPolicy(corpus, max_document_fraction=0.5)
        first = policy.assess("delta zeta", list(corpus.chunks))
        second = policy.assess("delta zeta", list(corpus.chunks))
        assert first == second


class TestWorkloadIntegration:
    """The policy against the exact workload that produced E2603's negative findings."""

    def test_abstention_cases_become_insufficient_and_answers_stay_sufficient(self) -> None:
        corpus, dataset = load_development_grounded_workload()
        service = RetrievalService(corpus)
        policy = DiscriminatingTermSufficiencyPolicy(corpus)  # default 0.25, in the working band
        false_abstentions = 0
        missed_abstentions = 0
        for case in dataset.cases:
            outcome = service.search(case.question)
            decision = policy.assess(case.question, outcome.pack.chunks)
            if case.expected_behavior is ExpectedBehavior.ABSTAIN:
                if decision.is_sufficient:
                    missed_abstentions += 1  # policy said "answer" on an unanswerable case
            elif not decision.is_sufficient:
                false_abstentions += 1  # policy said "abstain" on an answerable case (recall loss)
        # The whole point: no false abstention (recall preserved), no missed abstention (defect fixed).
        assert false_abstentions == 0
        assert missed_abstentions == 0

    def test_policy_never_mutates_the_retrieval_result(self) -> None:
        corpus, dataset = load_development_grounded_workload()
        service = RetrievalService(corpus)
        policy = DiscriminatingTermSufficiencyPolicy(corpus)
        case = dataset.cases[0]
        before = [str(r.chunk.chunk_id) for r in service.search(case.question).results]
        policy.assess(case.question, service.search(case.question).pack.chunks)
        after = [str(r.chunk.chunk_id) for r in service.search(case.question).results]
        assert before == after  # sufficiency is a read-only decision, not a filter or re-rank
