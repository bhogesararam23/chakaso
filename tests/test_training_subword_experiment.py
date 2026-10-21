"""End-to-end tests for the E2605 driver, on a synthetic root.

The driver is the piece that turns the model path into a measured number, so these run the real pipeline —
manifest, document split, both tokenizations, both training loops, held-out scoring, report writing —
against a small corpus the test builds. None of them asserts a capability; they assert that the measurement
is well-formed, reproducible, and cannot leak.

Gated on both optional extras: the experiment needs `torch` for the Transformer and `numpy` for the
character baseline.
"""

from __future__ import annotations

import json
from dataclasses import replace
from math import log2
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

from chakaso.training.errors import CheckpointError, CorpusDriftError
from chakaso.training.subword_experiment import (
    ExperimentPlan,
    prepare_corpus,
    run_experiment,
    summarise,
    write_report,
)

DOCUMENTS = {
    "docs/alpha.md": "# Alpha\n\nThe alpha report describes sand, foxes and dry river beds.\n",
    "docs/beta.md": "# Beta\n\nThe beta report describes owls, quiet towns and sleeping patterns.\n",
    "docs/gamma.md": "# Gamma\n\nThe gamma report describes stones, grass and the colour brown.\n",
    "docs/delta.md": "# Delta\n\nDelta repeats the sand and the river, then adds an owl in a town.\n",
    "docs/epsilon.md": "# Epsilon\n\nEpsilon describes flint, ember, granite and helium in prose.\n",
    "docs/zeta.md": "# Zeta\n\nZeta describes basalt, cedar, dolomite and alder in prose.\n",
    "docs/eta.md": "# Eta\n\nEta describes the river, the stone and the grass again and again.\n",
    "docs/theta.md": "# Theta\n\nTheta describes ember, flint and granite in a different order.\n",
}
_EXTRA_FILES = ("README.md", "AGENTS.md", "LICENSE")


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative, text in DOCUMENTS.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (root / "README.md").write_text(
        "# Reader\n\nA short reader document about foxes and quiet towns.\n", encoding="utf-8"
    )
    (root / "AGENTS.md").write_text(
        "# Agent notes\n\nShort notes about the repository.\n", encoding="utf-8"
    )
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    return root


def _plan(root: Path, work: Path) -> ExperimentPlan:
    return ExperimentPlan(
        root=root,
        work_dir=work,
        vocab_size=268,
        context_length=16,
        batch_size=4,
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        max_steps=12,
        eval_every=4,
        patience=3,
        min_delta=0.0,
        warmup_steps=2,
        character_context=4,
        character_embed_dim=4,
        character_hidden_dim=8,
        character_batch_windows=64,
        character_steps=40,
        character_eval_every=20,
        samples_per_prompt=8,
    )


def test_prepare_corpus_pins_documents_and_trains_both_tokenizations(tmp_path: Path) -> None:
    prepared = prepare_corpus(_plan(_root(tmp_path), tmp_path / "work"))

    assert prepared.manifest.char_count > 0
    assert len(prepared.split.train) >= 3
    assert prepared.vocabulary.size <= 268
    assert prepared.tokens["train"] > 0
    assert prepared.char_vocabulary.size > 1
    assert (tmp_path / "work" / "chakaso-repo-text-v1-manifest.json").is_file()
    assert (tmp_path / "work" / "chakaso-repo-text-v1-tokenizer.json").is_file()


def test_the_document_split_sides_are_disjoint_for_both_tokenizations(tmp_path: Path) -> None:
    prepared = prepare_corpus(_plan(_root(tmp_path), tmp_path / "work"))

    identifiers = {
        side: {unit.identifier for unit in getattr(prepared.split, side)}
        for side in ("train", "validation", "test")
    }

    assert identifiers["train"] & identifiers["validation"] == set()
    assert identifiers["train"] & identifiers["test"] == set()
    assert identifiers["validation"] & identifiers["test"] == set()
    assert prepared.char_windows["train"][0].shape[0] > 0


def test_a_second_run_refuses_a_corpus_whose_files_have_moved(tmp_path: Path) -> None:
    root = _root(tmp_path)
    plan = _plan(root, tmp_path / "work")
    prepare_corpus(plan)
    (root / "docs" / "alpha.md").write_text(
        "# Alpha\n\nRewritten after the manifest was pinned.\n", encoding="utf-8"
    )

    with pytest.raises(CorpusDriftError, match="content changed"):
        prepare_corpus(plan)


def test_run_experiment_produces_a_wellformed_measured_report(tmp_path: Path) -> None:
    report = run_experiment(_plan(_root(tmp_path), tmp_path / "work"))

    transformer = report["transformer"]["test"]
    assert transformer["positions_scored"] > 0
    assert transformer["characters_scored"] > 0
    assert transformer["bits_per_character"] > 0
    assert report["character_baseline"]["test"]["positions_scored"] > 0
    assert report["baselines_subword"]["uniform"]["bits_per_character"] > 0
    assert (
        report["transformer"]["parameter_count"]
        == report["transformer"]["parameters_by_configuration"]
    )
    assert report["corpus"]["documents"] == len(DOCUMENTS) + len(_EXTRA_FILES)
    assert report["samples"]
    assert all(sample["text"] is not None for sample in report["samples"])
    assert report["transformer"]["artifact"]["schema"] == "chakaso-transformer-checkpoint/1"


def test_the_uniform_baseline_matches_its_closed_form(tmp_path: Path) -> None:
    report = run_experiment(_plan(_root(tmp_path), tmp_path / "work"))
    uniform = report["baselines_subword"]["uniform"]

    # A uniform predictor spends log2(V) bits per token, so its bits-per-character is that cost scaled by
    # the tokens-per-character ratio. If this fails, the metric arithmetic or the baseline is wrong.
    expected = (
        log2(uniform["vocabulary_size"])
        * uniform["positions_scored"]
        / uniform["characters_scored"]
    )

    assert uniform["bits_per_character"] == pytest.approx(expected, rel=1e-6)


def test_two_runs_with_one_seed_measure_the_same_corpus_the_same_way(tmp_path: Path) -> None:
    root = _root(tmp_path)

    # Two independent output directories: a second run into the first one's directory is refused by
    # ADR-0035's overwrite guard, which is the point — a measured artifact must not be silently replaced.
    first = run_experiment(_plan(root, tmp_path / "work-a"))
    second = run_experiment(_plan(root, tmp_path / "work-b"))

    assert first["corpus"]["fingerprint"] == second["corpus"]["fingerprint"]
    assert first["tokenizer"]["fingerprint"] == second["tokenizer"]["fingerprint"]
    assert first["corpus"]["split_fingerprint"] == second["corpus"]["split_fingerprint"]
    assert first["transformer"]["parameter_count"] == second["transformer"]["parameter_count"]
    assert first["transformer"]["test"]["bits_per_character"] == pytest.approx(
        second["transformer"]["test"]["bits_per_character"], abs=1e-3
    )


def test_the_report_is_deterministic_json_and_carries_its_own_digest(tmp_path: Path) -> None:
    report = run_experiment(_plan(_root(tmp_path), tmp_path / "work"))
    target = tmp_path / "results.json"

    digest = write_report(report, target)
    reloaded = json.loads(target.read_text(encoding="utf-8"))

    assert reloaded["experiment"] == "E2605"
    assert len(digest) == 64
    assert digest == write_report(reloaded, tmp_path / "again.json")


def test_the_summary_names_every_model_and_both_baselines(tmp_path: Path) -> None:
    line = summarise(run_experiment(_plan(_root(tmp_path), tmp_path / "work")))

    assert "transformer" in line
    assert "character" in line
    assert "unigram" in line
    assert "uniform" in line
    assert "bits/char" in line


def test_a_second_run_into_the_same_directory_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    plan = _plan(root, tmp_path / "work")
    run_experiment(plan)

    with pytest.raises(CheckpointError, match="already holds an artifact"):
        run_experiment(plan)

    # And it proceeds when the caller says the earlier artifact may be replaced.
    assert run_experiment(replace(plan, overwrite_artifacts=True))


def test_a_plan_that_cannot_run_is_refused_before_any_work_starts(tmp_path: Path) -> None:
    root = _root(tmp_path)

    with pytest.raises(ValueError, match="root"):
        ExperimentPlan(root=root / "absent", work_dir=tmp_path / "work")
    with pytest.raises(ValueError, match="vocab_size"):
        ExperimentPlan(root=root, work_dir=tmp_path / "work", vocab_size=200)
    with pytest.raises(ValueError, match="step budget"):
        ExperimentPlan(root=root, work_dir=tmp_path / "work", max_steps=2, eval_every=8)


def test_a_short_character_budget_clamps_its_interval_instead_of_never_reporting(
    tmp_path: Path,
) -> None:
    plan = _plan(_root(tmp_path), tmp_path / "work")

    # The interval is clamped to a budget that can actually honour it: 30 steps evaluated every 200 would
    # select on nothing, and a plan that cannot report is worse than a plan that reports late.
    short = replace(plan, character_steps=30, character_eval_every=200)
    assert short.character_evaluation_interval() == 3

    # A budget that already fits its interval is left alone, so a long run is unchanged by the clamp.
    long = replace(plan, character_steps=4000, character_eval_every=200)
    assert long.character_evaluation_interval() == 200
