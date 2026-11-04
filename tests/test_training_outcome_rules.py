"""Tests for the outcome vocabulary: what a declaration may say, and the three ways it is refused.

The properties that matter are the three refusals (K-034) — an unknown fact, two letters that both match, no letter
that matches — because those are the cases a prose mapping hides. One test exists to prove the point of the whole
mechanism: two records with different letters over the same facts, neither of them touching shared code.
"""

from __future__ import annotations

import pytest

from chakaso.training.errors import TrainingError
from chakaso.training.outcome_rules import (
    OUTCOME_VOCABULARY_SCHEMA,
    OutcomeRule,
    OutcomeRuleError,
    OutcomeVocabulary,
    decide,
    load_vocabulary,
)

#: The facts `scripts/analyse_arm_pairs.py` computes for a family of arm pairs.
FACT_NAMES = [
    "material_pairs",
    "material_pairs_ranked",
    "every_material_pair_ranked",
    "some_material_pair_ranked",
    "any_separation",
    "any_contradiction",
    "any_inside_band_separated",
    "any_ordering_inverted",
]


def _arm_pair_document(letters: str = "Z") -> dict[str, object]:
    """The E2629 rules (``Z``) or the same shape under another record's letters, which is the whole mechanism."""
    return {
        "schema_version": OUTCOME_VOCABULARY_SCHEMA,
        "name": f"{letters} — arm-pair outcomes",
        "outcomes": [
            {
                "letter": f"{letters}-c",
                "reading": "an instrument ranked a pair the wrong way or separated the inside-band pair",
                "when": [
                    {
                        "any_of": [
                            {"flag": "any_contradiction"},
                            {"flag": "any_inside_band_separated"},
                        ]
                    }
                ],
            },
            {
                "letter": f"{letters}-a",
                "reading": "an instrument ranks every material pair the criterion's way and nothing is contradicted",
                "when": [
                    {"flag": "every_material_pair_ranked"},
                    {"not_flag": "any_contradiction"},
                    {"not_flag": "any_inside_band_separated"},
                ],
            },
            {
                "letter": f"{letters}-b",
                "reading": "an instrument ranks at least one material pair correctly and none contradicts",
                "when": [
                    {"flag": "some_material_pair_ranked"},
                    {"not_flag": "any_contradiction"},
                ],
            },
            {
                "letter": f"{letters}-d",
                "reading": "no decision-bearing instrument separated a material pair",
                "when": [{"not_flag": "any_separation"}],
            },
        ],
    }


def _facts(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "material_pairs": 2,
        "material_pairs_ranked": 2,
        "every_material_pair_ranked": True,
        "some_material_pair_ranked": True,
        "any_separation": True,
        "any_contradiction": False,
        "any_inside_band_separated": False,
        "any_ordering_inverted": False,
    }
    base.update(overrides)
    return base


def _vocabulary(letters: str = "Z") -> OutcomeVocabulary:
    return load_vocabulary(_arm_pair_document(letters), known_facts=FACT_NAMES)


# ------------------------------------------------------------------ deciding


def test_the_declaration_assigns_the_letter_the_record_meant() -> None:
    verdict = decide(_facts(), _vocabulary())

    assert verdict["letter"] == "Z-a"
    assert verdict["vocabulary"] == "Z — arm-pair outcomes"
    assert verdict["facts"]["material_pairs"] == 2
    assert verdict["letters_declared"] == ["Z-c", "Z-a", "Z-b", "Z-d"]


def test_a_rule_written_first_wins_only_because_the_later_one_does_not_also_match() -> None:
    """Contradiction is decided by the facts, not by the order the author typed them in."""
    verdict = decide(_facts(any_contradiction=True), _vocabulary())

    assert verdict["letter"] == "Z-c"


def test_the_catch_all_fires_when_nothing_else_can() -> None:
    verdict = decide(
        _facts(
            every_material_pair_ranked=False,
            some_material_pair_ranked=False,
            any_separation=False,
            material_pairs_ranked=0,
        ),
        _vocabulary(),
    )

    assert verdict["letter"] == "Z-d"


def test_a_number_can_be_bounded_as_well_as_a_flag() -> None:
    vocabulary = load_vocabulary(
        {
            "schema_version": OUTCOME_VOCABULARY_SCHEMA,
            "name": "coverage",
            "outcomes": [
                {
                    "letter": "N-a",
                    "reading": "rho at or above the tracking threshold",
                    "when": [{"fact": "material_pairs", "at_least": 1, "at_most": 4}],
                },
                {
                    "letter": "N-b",
                    "reading": "anything else",
                    "when": [{"always": True}],
                },
            ],
        },
        known_facts=FACT_NAMES,
    )

    assert decide(_facts(material_pairs=2), vocabulary)["letter"] == "N-a"
    assert decide(_facts(material_pairs=9), vocabulary)["letter"] == "N-b"


def test_a_bound_pair_that_cannot_both_hold_is_refused_as_an_unreachable_rule() -> None:
    with pytest.raises(OutcomeRuleError, match="can never match"):
        load_vocabulary(
            {
                "schema_version": OUTCOME_VOCABULARY_SCHEMA,
                "name": "impossible",
                "outcomes": [
                    {
                        "letter": "X-a",
                        "reading": "rho is both below and above itself",
                        "when": [{"fact": "material_pairs", "at_least": 5, "at_most": 2}],
                    }
                ],
            },
            known_facts=FACT_NAMES,
        )


# ------------------------------------------------------------------ the three refusals


def test_a_fact_the_analysis_does_not_compute_is_refused_when_the_vocabulary_is_loaded() -> None:
    document = _arm_pair_document()
    document["outcomes"][1]["when"] = [{"flag": "every_material_pair_rankd"}]

    with pytest.raises(OutcomeRuleError, match="does not compute"):
        load_vocabulary(document, known_facts=FACT_NAMES)


def test_the_default_reading_reports_the_first_match_and_names_the_letter_it_shadowed() -> None:
    """Ordered precedence is how E2629's letters are actually written, and an overlap is shown, not hidden.

    `Z-a` (every material pair ranked) implies `Z-b` (some ranked): the two are separated by precedence, so the
    artifact says both matched. A reader can then disagree with the order instead of with the arithmetic.
    """
    document = _arm_pair_document()
    document["outcomes"][2]["when"] = [{"flag": "some_material_pair_ranked"}]

    verdict = decide(_facts(), load_vocabulary(document, known_facts=FACT_NAMES))

    assert verdict["letter"] == "Z-a"
    assert verdict["precedence"] == "first match in declaration order"
    assert verdict["also_matched"] == ["Z-b"]


def test_a_declaration_that_says_its_letters_are_disjoint_refuses_when_two_match() -> None:
    document = _arm_pair_document()
    document["exclusive"] = True
    document["outcomes"][2]["when"] = [{"flag": "some_material_pair_ranked"}]

    with pytest.raises(OutcomeRuleError, match=r"\['Z-a', 'Z-b'\] both match"):
        decide(_facts(), load_vocabulary(document, known_facts=FACT_NAMES))


def test_facts_no_rule_covers_are_refused_with_the_facts_printed() -> None:
    vocabulary = load_vocabulary(
        {
            "schema_version": OUTCOME_VOCABULARY_SCHEMA,
            "name": "narrow",
            "outcomes": [
                {
                    "letter": "A-a",
                    "reading": "only the tidy case",
                    "when": [{"not_flag": "any_separation"}],
                }
            ],
        },
        known_facts=FACT_NAMES,
    )

    with pytest.raises(OutcomeRuleError, match="no outcome rule matches"):
        decide(_facts(), vocabulary)


# ------------------------------------------------------------------ malformed declarations


def test_a_letter_declared_twice_is_refused() -> None:
    document = _arm_pair_document()
    document["outcomes"].append(dict(document["outcomes"][0]))

    with pytest.raises(OutcomeRuleError, match="declares a letter twice"):
        load_vocabulary(document, known_facts=FACT_NAMES)


def test_a_rule_with_no_condition_must_say_it_is_the_catch_all() -> None:
    with pytest.raises(OutcomeRuleError, match="catch-all"):
        OutcomeRule(letter="Q-a", reading="whatever happens", when=())


def test_a_rule_needs_a_reading_because_a_letter_nobody_can_explain_is_not_a_contract() -> None:
    with pytest.raises(OutcomeRuleError, match="no reading"):
        OutcomeRule(letter="Q-a", reading="  ", when=({"always": True},))


@pytest.mark.parametrize(
    "constraint",
    [
        {"flag": "any_separation", "not_flag": "any_contradiction"},
        {"fact": "material_pairs"},
        {"fact": "material_pairs", "at_least": "two"},
        {"always": False},
        {"all_of": []},
        {"all_of": [{"always": True}], "any_of": [{"always": True}]},
        "any_separation",
    ],
)
def test_a_constraint_this_code_cannot_read_is_refused_by_shape(constraint: object) -> None:
    document = _arm_pair_document()
    document["outcomes"][0]["when"] = [constraint]

    with pytest.raises(
        OutcomeRuleError, match=r"must be one of|needs a non-empty|must be a number"
    ):
        load_vocabulary(document, known_facts=FACT_NAMES)


def test_a_declaration_of_another_schema_is_refused_by_name() -> None:
    document = _arm_pair_document()
    document["schema_version"] = "chakaso-outcome-vocabulary/2"

    with pytest.raises(OutcomeRuleError, match="names schema"):
        load_vocabulary(document, known_facts=FACT_NAMES)


def test_a_declaration_with_no_outcomes_is_refused() -> None:
    with pytest.raises(OutcomeRuleError, match="carries no outcomes"):
        load_vocabulary(
            {"schema_version": OUTCOME_VOCABULARY_SCHEMA, "name": "empty", "outcomes": []},
            known_facts=FACT_NAMES,
        )


# ------------------------------------------------------------------ the point of the mechanism


def test_a_second_record_gets_its_own_letters_from_the_same_facts_without_touching_shared_code() -> (
    None
):
    """E2634 preregistered G-letters and was reported a Z-a by the tool. That is the defect this proves closed.

    Two vocabularies, the same facts, one analysis: the letters differ because the *declarations* differ, and nothing
    between the two runs was edited in a library.
    """
    facts = _facts(
        every_material_pair_ranked=False,
        material_pairs_ranked=1,
        some_material_pair_ranked=True,
    )

    assert decide(facts, _vocabulary("Z"))["letter"] == "Z-b"
    assert decide(facts, _vocabulary("G"))["letter"] == "G-b"


def test_a_vocabulary_round_trips_through_its_own_document_form() -> None:
    vocabulary = _vocabulary()

    assert load_vocabulary(vocabulary.as_dict(), known_facts=FACT_NAMES) == vocabulary


def test_a_missing_fact_at_decision_time_is_refused_not_read_as_false() -> None:
    facts = _facts()
    del facts["any_contradiction"]

    with pytest.raises(OutcomeRuleError, match="supplied none"):
        decide(facts, _vocabulary())


def test_the_outcome_error_is_the_package_error_type() -> None:
    assert issubclass(OutcomeRuleError, TrainingError)
