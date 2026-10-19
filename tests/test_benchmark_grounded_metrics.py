"""Independent, hand-computed checks for the grounded-answer metric primitives."""

from __future__ import annotations

import pytest

from chakaso.benchmark.grounded_metrics import (
    NO_DENOMINATOR,
    citation_validity_rate,
    mean,
    prohibited_claim_rate,
    rate,
    required_fact_coverage,
    unsupported_claim_rate,
)


class TestMeanAndRate:
    def test_mean_of_empty_is_zero(self) -> None:
        assert mean([]) == 0.0

    def test_mean_hand_computed(self) -> None:
        # (0.2 + 0.4 + 0.9) / 3 = 1.5 / 3 = 0.5
        assert mean([0.2, 0.4, 0.9]) == pytest.approx(0.5)

    @pytest.mark.parametrize("denominator", [0, -1])
    def test_rate_guards_nonpositive_denominator(self, denominator: int) -> None:
        assert rate(3, denominator) == NO_DENOMINATOR

    def test_rate_hand_computed(self) -> None:
        assert rate(3, 4) == pytest.approx(0.75)


class TestCitationValidity:
    def test_denominator_excludes_uncited_claims(self) -> None:
        # valid=6, unknown=2, irrelevant=2, uncited=5 (ignored). 6 / (6+2+2) = 0.6
        value = citation_validity_rate(valid=6, unknown=2, irrelevant=2)
        assert value == pytest.approx(0.6)

    def test_no_citations_issued_is_zero(self) -> None:
        assert citation_validity_rate(valid=0, unknown=0, irrelevant=0) == 0.0

    def test_all_valid_is_one(self) -> None:
        assert citation_validity_rate(valid=4, unknown=0, irrelevant=0) == pytest.approx(1.0)


class TestUnsupportedClaims:
    def test_hand_computed(self) -> None:
        # 2 of 5 claims unsupported = 0.4
        assert unsupported_claim_rate(unsupported=2, total_claims=5) == pytest.approx(0.4)

    def test_no_claims_is_zero(self) -> None:
        assert unsupported_claim_rate(unsupported=0, total_claims=0) == 0.0


class TestRequiredFacts:
    def test_hand_computed(self) -> None:
        # 3 of 4 required facts present = 0.75
        assert required_fact_coverage(facts_present=3, facts_required=4) == pytest.approx(0.75)

    def test_case_requiring_no_facts_is_zero(self) -> None:
        assert required_fact_coverage(facts_present=0, facts_required=0) == 0.0


class TestProhibitedClaims:
    def test_hand_computed(self) -> None:
        # 1 prohibited claim appeared out of 2 forbidden = 0.5
        assert prohibited_claim_rate(prohibited_present=1, prohibited_total=2) == pytest.approx(0.5)

    def test_forbidding_nothing_is_zero(self) -> None:
        assert prohibited_claim_rate(prohibited_present=0, prohibited_total=0) == 0.0
