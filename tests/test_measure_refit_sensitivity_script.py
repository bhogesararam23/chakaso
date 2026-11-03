"""Tests for the refit-sensitivity measurement: what moves when only the fitting sample moves.

The fixtures are small vocabularies built in this file, because the properties under test — that a compression figure
is the ratio the loss unit depends on, that a merge comparison reads rank and not frequency, that the ratio between two
rules is computed on the same evaluation text for both — are checkable on five documents and not on four million
characters.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

from chakaso.training.subword import (  # noqa: E402
    DEFAULT_PRE_TOKENIZER,
    SPACE_PREFIXED_PRE_TOKENIZER,
    train_subword_vocabulary,
)

_TEXTS = (
    "the quick red fox ran over the dry sand and kept running",
    "the quiet owl slept above the slow river and the dry bank",
    "a red fox and an owl kept the sand and the river in view",
)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "measure_refit_sensitivity_under_test", _SCRIPTS / "measure_refit_sensitivity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compression_counts_the_characters_a_position_buys() -> None:
    vocabulary = train_subword_vocabulary(
        _TEXTS, target_size=290, min_frequency=2, pre_tokenizer=DEFAULT_PRE_TOKENIZER
    )
    block = _script().compression(vocabulary, _TEXTS)

    assert block["documents"] == 3
    assert block["characters"] == sum(len(text) for text in _TEXTS)
    assert block["identifiers"] == sum(len(vocabulary.encode(text)) for text in _TEXTS)
    assert block["characters_per_identifier"] == pytest.approx(
        block["characters"] / block["identifiers"]
    )


def test_a_merge_comparison_reads_shared_pieces_and_the_order_they_came_in() -> None:
    module = _script()
    first = [(1, 2), (3, 4), (5, 6), (7, 8)]
    same = [(1, 2), (3, 4), (5, 6), (7, 8)]
    reordered = [(7, 8), (5, 6), (3, 4), (1, 2)]
    disjoint = [(9, 10), (11, 12), (13, 14), (15, 16)]

    identical = module.merge_overlap(first, same)
    assert identical["jaccard"] == 1.0
    assert identical["rank_agreement"] == pytest.approx(1.0)
    reversed_order = module.merge_overlap(first, reordered)
    assert reversed_order["jaccard"] == 1.0
    assert reversed_order["rank_agreement"] == pytest.approx(-1.0)
    apart = module.merge_overlap(first, disjoint)
    assert apart["jaccard"] == 0.0
    # Two shared merges carry no ordering information, so the module reports none rather than a confident 1.0.
    assert module.merge_overlap(first, [(1, 2), (3, 4), (99, 100)])["rank_agreement"] is None


def test_measure_fits_both_rules_on_the_same_sample_and_reports_the_ratio() -> None:
    module = _script()
    fits = [
        tuple(text for text in _TEXTS),
        tuple(text for text in reversed(_TEXTS)),
    ]

    result = module.measure(
        labels=[7, 11],
        fitting_samples=fits,
        evaluation_texts=_TEXTS,
        identifiers=290,
        min_frequency=2,
    )

    assert result["fit_seeds"] == [7, 11]
    assert len(result["fits"]) == 2
    for fit in result["fits"]:
        assert set(fit["compression"]) == {"default", "space-prefixed"}
        assert fit["fingerprints"]["default"] != fit["fingerprints"]["space-prefixed"]
        assert fit["ratio_space_over_default"] > 0
    assert set(result["merge_stability"]) == {"default", "space-prefixed"}
    assert result["evaluation"]["documents"] == 3


def test_measure_carries_a_committed_vocabulary_through_the_same_ruler() -> None:
    module = _script()
    reference = train_subword_vocabulary(
        _TEXTS, target_size=290, min_frequency=2, pre_tokenizer=SPACE_PREFIXED_PRE_TOKENIZER
    )

    result = module.measure(
        labels=[7],
        fitting_samples=[_TEXTS],
        evaluation_texts=_TEXTS,
        identifiers=280,
        min_frequency=2,
        references={"committed": reference},
    )

    block = result["references"]["committed"]
    assert block["fingerprint"] == reference.fingerprint
    assert block["pre_tokenizer"] == SPACE_PREFIXED_PRE_TOKENIZER
    assert block["compression"]["characters"] == sum(len(text) for text in _TEXTS)


def test_inputs_that_do_not_line_up_are_refused() -> None:
    module = _script()
    with pytest.raises(module.AnalysisError, match="one fitting sample"):
        module.measure(
            labels=[], fitting_samples=[], evaluation_texts=_TEXTS, identifiers=290, min_frequency=2
        )
    with pytest.raises(module.AnalysisError, match="do not describe one experiment"):
        module.measure(
            labels=[7],
            fitting_samples=[_TEXTS, _TEXTS],
            evaluation_texts=_TEXTS,
            identifiers=290,
            min_frequency=2,
        )


def test_the_seed_list_is_parsed_and_a_repeat_is_refused() -> None:
    module = _script()
    assert module._seeds("7, 11,13") == (7, 11, 13)
    with pytest.raises(module.AnalysisError, match="comma-separated"):
        module._seeds("7,eleven")
    with pytest.raises(module.AnalysisError, match="distinct"):
        module._seeds("7,7")


def test_the_summary_names_every_rule_and_the_ratio_they_agree_on() -> None:
    module = _script()
    result = module.measure(
        labels=[7, 11],
        fitting_samples=[_TEXTS, tuple(reversed(_TEXTS))],
        evaluation_texts=_TEXTS,
        identifiers=290,
        min_frequency=2,
    )
    text = module.summarise(result)
    assert "fit seeds [7, 11]" in text
    assert "default" in text and "space-prefixed" in text
    assert "characters per identifier" in text
