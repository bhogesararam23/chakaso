"""Tests for the grounded-answer evaluation report: determinism, labels and provenance."""

from __future__ import annotations

import json

from chakaso.benchmark.grounded_report import (
    GROUNDED_EVALUATION_NOTE,
    format_grounded_report,
    grounded_report_json,
    grounded_report_to_dict,
)
from chakaso.benchmark.grounded_runner import (
    GroundedEvaluationRunner,
    GroundedRun,
    ReferenceAnswerProvider,
)
from chakaso.benchmark.grounded_workload import load_development_grounded_workload


def _oracle_run() -> GroundedRun:
    corpus, dataset = load_development_grounded_workload()
    return GroundedEvaluationRunner(
        corpus, provider=ReferenceAnswerProvider(), metadata={"code_revision": "test-rev"}
    ).run(dataset)


class TestReportShape:
    def test_json_is_deterministic(self) -> None:
        first = grounded_report_json(_oracle_run())
        second = grounded_report_json(_oracle_run())
        assert first == second

    def test_report_round_trips_as_json(self) -> None:
        loaded = json.loads(grounded_report_json(_oracle_run()))
        assert loaded["evaluation"]["kind"] == "development-grounded-answer"
        assert loaded["configuration"]["code_revision"] == "test-rev"

    def test_note_and_oracle_label_are_present(self) -> None:
        loaded = grounded_report_to_dict(_oracle_run())
        assert loaded["evaluation"]["note"] == GROUNDED_EVALUATION_NOTE
        assert loaded["configuration"]["provider_is_oracle"] is True
        assert loaded["configuration"]["provider"] == "reference-oracle (not a model)"

    def test_dimensions_are_kept_separate(self) -> None:
        loaded = grounded_report_to_dict(_oracle_run())
        # Retrieval and answer metrics are distinct sections; there is no merged score key anywhere.
        assert "retrieval_metrics" in loaded
        assert "answer_metrics" in loaded
        assert "quality_score" not in json.dumps(loaded)

    def test_per_case_entries_are_inspectable(self) -> None:
        loaded = grounded_report_to_dict(_oracle_run())
        cases = loaded["cases"]
        assert isinstance(cases, list) and len(cases) == 14
        by_id = {entry["case_id"]: entry for entry in cases}
        assert by_id["ga-missing-dinosaur"]["abstention_expected"] is True
        assert by_id["ga-direct-loan"]["category"] == "direct-retrieval"

    def test_human_report_renders_all_cases(self) -> None:
        text = format_grounded_report(_oracle_run())
        assert "Grounded-answer evaluation" in text
        assert GROUNDED_EVALUATION_NOTE in text
        assert "ga-direct-loan" in text
        assert "Per case:" in text
