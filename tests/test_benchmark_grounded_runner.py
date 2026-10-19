"""Tests for the grounded-answer evaluation runner and its failure classification."""

from __future__ import annotations

import pytest

from chakaso.benchmark.cases import BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.grounded_cases import (
    CaseSplit,
    GroundedCase,
    GroundedProvenance,
    GroundedSource,
)
from chakaso.benchmark.grounded_dataset import GroundedDataset, build_corpus
from chakaso.benchmark.grounded_runner import (
    GeneratedAnswer,
    GroundedEvaluationRunner,
    NonAnsweringProvider,
    ReferenceAnswerProvider,
    StageFailure,
    environment_metadata,
)
from chakaso.benchmark.grounded_workload import load_development_grounded_workload
from chakaso.benchmark.identity import CaseId, DatasetId
from chakaso.retrieval import Corpus, RetrievalService

PROVENANCE = GroundedProvenance(
    origin="Authored for Chakaso.",
    licence="Apache-2.0.",
    collection_method="self-authored",
    redistribution="no restriction",
)


def _source(label: str, text: str) -> GroundedSource:
    return GroundedSource(label=label, title=label.title(), text=text, provenance=PROVENANCE)


class _RaisingService:
    """A retrieval service whose search always fails, to exercise the retrieval-failure path."""

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    def search(self, query: str, *, top_k: int = 5, source_ids: object = None) -> object:
        message = "retrieval exploded"
        raise RuntimeError(message)


class _ExplodingProvider:
    label = "exploding"
    is_oracle = False

    def answer(self, case, pack, *, gold, forbidden) -> GeneratedAnswer:
        message = "generation exploded"
        raise RuntimeError(message)


class _MalformedProvider:
    """Produces a claim with blank text, which the extractor rejects during evaluation."""

    label = "malformed"
    is_oracle = False

    def answer(self, case, pack, *, gold, forbidden) -> GeneratedAnswer:
        return GeneratedAnswer(text="oops", claims=(("", ()),), abstained=False, label=self.label)


def _single_case_dataset() -> tuple[Corpus, GroundedDataset]:
    source = _source("loan", "A loan lasts 28 days.")
    case = GroundedCase(
        case_id=CaseId("ga-loan"),
        category=BenchmarkCategory.DIRECT_RETRIEVAL,
        split=CaseSplit.DEVELOPMENT,
        question="how long does a loan last",
        rationale="single-source direct fact",
        gold_labels=("loan",),
        required_evidence_labels=("loan",),
        required_facts=("28 days",),
    )
    dataset = GroundedDataset(
        dataset_id=DatasetId("ga-test"), version="1.0.0", sources=(source,), cases=(case,)
    )
    return build_corpus(dataset.sources), dataset


def _answer_and_abstain_dataset() -> tuple[Corpus, GroundedDataset]:
    source = _source("loan", "A loan lasts 28 days.")
    cases = (
        GroundedCase(
            case_id=CaseId("ga-loan"),
            category=BenchmarkCategory.DIRECT_RETRIEVAL,
            split=CaseSplit.DEVELOPMENT,
            question="how long does a loan last",
            rationale="answerable",
            gold_labels=("loan",),
            required_facts=("28 days",),
        ),
        GroundedCase(
            case_id=CaseId("ga-ghost"),
            category=BenchmarkCategory.MISSING_EVIDENCE,
            split=CaseSplit.DEVELOPMENT,
            question="what is the moon made of",
            rationale="unanswerable",
            expected_behavior=ExpectedBehavior.ABSTAIN,
        ),
    )
    dataset = GroundedDataset(
        dataset_id=DatasetId("ga-test"), version="1.0.0", sources=(source,), cases=cases
    )
    return build_corpus(dataset.sources), dataset


class TestReferenceOracle:
    def test_correct_answer_scores_grounded(self) -> None:
        corpus, dataset = _single_case_dataset()
        runner = GroundedEvaluationRunner(corpus, provider=ReferenceAnswerProvider())
        run = runner.run(dataset)
        outcome = run.outcomes[0]
        assert outcome.failure is StageFailure.NONE
        assert run.failures.total == 0
        assert outcome.hit is True
        assert outcome.is_grounded is True
        assert outcome.evidence_coverage == pytest.approx(1.0)
        assert outcome.citation_valid == 1
        assert run.answer_metrics.required_fact_coverage == pytest.approx(1.0)
        assert run.answer_metrics.grounded_answer_rate == pytest.approx(1.0)

    def test_abstention_is_correct_for_unanswerable(self) -> None:
        corpus, dataset = _answer_and_abstain_dataset()
        runner = GroundedEvaluationRunner(corpus, provider=ReferenceAnswerProvider())
        run = runner.run(dataset)
        assert run.answer_metrics.appropriate_abstention_rate == pytest.approx(1.0)
        ghost = next(o for o in run.outcomes if o.case_id == "ga-ghost")
        assert ghost.abstention_observed is True
        assert ghost.abstention_correct is True

    def test_coverage_means_over_answered_cases_only(self) -> None:
        # The single answered case is fully grounded; the abstention (coverage 0 by definition)
        # must not dilute the mean. This pins the denominator fix.
        corpus, dataset = _answer_and_abstain_dataset()
        run = GroundedEvaluationRunner(corpus, provider=ReferenceAnswerProvider()).run(dataset)
        assert run.answer_metrics.answered_cases == 1
        assert run.answer_metrics.mean_evidence_coverage == pytest.approx(1.0)


class TestNonAnsweringProvider:
    def test_over_abstention_is_penalized(self) -> None:
        corpus, dataset = _answer_and_abstain_dataset()
        runner = GroundedEvaluationRunner(corpus, provider=NonAnsweringProvider())
        run = runner.run(dataset)
        # One answerable case is wrongly abstained; one unanswerable is correctly abstained.
        assert run.answer_metrics.answered_cases == 0
        assert run.answer_metrics.appropriate_abstention_rate == pytest.approx(0.5)


class TestRetrievalOnly:
    def test_no_provider_leaves_answer_metrics_empty(self) -> None:
        corpus, dataset = _single_case_dataset()
        runner = GroundedEvaluationRunner(corpus)
        run = runner.run(dataset)
        assert run.provider_label == "retrieval-only"
        assert run.retrieval_metrics.scored_cases == 1
        assert run.retrieval_metrics.mean_recall_at_k == pytest.approx(1.0)
        assert run.answer_metrics.answered_cases == 0
        assert run.outcomes[0].answer_text is None


class TestFailureClassification:
    def test_retrieval_failure_is_recorded_not_scored(self) -> None:
        corpus, dataset = _single_case_dataset()
        runner = GroundedEvaluationRunner(
            corpus, service=_RaisingService(corpus), provider=ReferenceAnswerProvider()
        )
        run = runner.run(dataset)
        assert run.failures.retrieval == 1
        assert run.outcomes[0].failure is StageFailure.RETRIEVAL

    def test_generation_failure_is_recorded(self) -> None:
        corpus, dataset = _single_case_dataset()
        runner = GroundedEvaluationRunner(
            corpus, service=RetrievalService(corpus), provider=_ExplodingProvider()
        )  # type: ignore[arg-type]
        run = runner.run(dataset)
        assert run.failures.generation == 1
        assert run.outcomes[0].failure is StageFailure.GENERATION

    def test_evaluation_failure_is_recorded(self) -> None:
        corpus, dataset = _single_case_dataset()
        runner = GroundedEvaluationRunner(corpus, provider=_MalformedProvider())  # type: ignore[arg-type]
        run = runner.run(dataset)
        assert run.failures.evaluation == 1
        assert run.outcomes[0].failure is StageFailure.EVALUATION


class TestSplitFilterAndDeterminism:
    def test_split_filter_selects_only_that_split(self) -> None:
        corpus, dataset = load_development_grounded_workload()
        runner = GroundedEvaluationRunner(
            corpus, service=RetrievalService(corpus), split_filter=CaseSplit.HELDOUT
        )
        run = runner.run(dataset)
        assert all(outcome.split is CaseSplit.HELDOUT for outcome in run.outcomes)
        assert len(run.outcomes) == len(dataset.cases_in(CaseSplit.HELDOUT))

    def test_run_is_deterministic(self) -> None:
        corpus, dataset = load_development_grounded_workload()
        first = GroundedEvaluationRunner(corpus, provider=ReferenceAnswerProvider()).run(dataset)
        second = GroundedEvaluationRunner(corpus, provider=ReferenceAnswerProvider()).run(dataset)
        assert first.dataset_content_fingerprint == second.dataset_content_fingerprint
        assert first.retrieval_metrics == second.retrieval_metrics
        assert first.answer_metrics == second.answer_metrics


class TestEnvironmentMetadata:
    def test_records_interpreter_platform_and_version(self) -> None:
        metadata = environment_metadata()
        for key in ("python_version", "python_implementation", "platform", "chakaso_version"):
            assert metadata[key], f"{key} must be recorded for reproducibility"

    def test_is_reproducible_within_a_machine(self) -> None:
        assert environment_metadata() == environment_metadata()
