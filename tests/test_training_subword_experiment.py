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
import shutil
from dataclasses import replace
from math import log2
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

from chakaso.core.hashing import sha256_hex
from chakaso.training.errors import (
    CheckpointError,
    CorpusDriftError,
    CorpusError,
    CorruptedCorpusArtifactError,
)
from chakaso.training.nn_training import TrainingScheduleError
from chakaso.training.subword_experiment import (
    CorpusSpec,
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


def _plan(root: Path, work: Path, **overrides: object) -> ExperimentPlan:
    plan = ExperimentPlan(
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
    return plan if not overrides else replace(plan, **overrides)


def _other_root(tmp_path: Path) -> Path:
    """A second repository whose tracked text is a different corpus."""
    root = tmp_path / "other-repo"
    documents = {
        "docs/one.md": "# One\n\nHarbour cranes unload coal while the fog holds the quay.\n",
        "docs/two.md": "# Two\n\nA ledger of tonnage, tar and rope is kept in the office.\n",
        "docs/three.md": "# Three\n\nThe pilot signs for the tide and the bell buoy sounds twice.\n",
        "docs/four.md": "# Four\n\nCranes, coal and fog again, in a report the harbour also files.\n",
        "docs/five.md": "# Five\n\nTarred rope and a ledger of tonnage, kept by the same office.\n",
        "docs/six.md": "# Six\n\nThe quay is quiet when the fog lifts and the cranes stand still.\n",
    }
    for relative, text in documents.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (root / "README.md").write_text(
        "# Harbour\n\nA reader document about cranes and fog.\n", encoding="utf-8"
    )
    (root / "AGENTS.md").write_text("# Notes\n\nShort harbour notes.\n", encoding="utf-8")
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    return root


def test_prepare_corpus_pins_documents_and_trains_both_tokenizations(tmp_path: Path) -> None:
    prepared = prepare_corpus(_plan(_root(tmp_path), tmp_path / "work"))

    assert prepared.manifest.char_count > 0
    assert len(prepared.split.train) >= 3
    assert prepared.vocabulary.size <= 268
    assert prepared.tokens["train"] > 0
    assert prepared.char_vocabulary.size > 1
    assert (tmp_path / "work" / "chakaso-repo-text-v1-manifest.json").is_file()
    # The tokenizer is cached under the identity of the inputs it was fitted from, so a reuse can never be
    # a tokenizer fitted on different data.
    cached = list((tmp_path / "work").glob("chakaso-repo-text-v1-tokenizer-*.json"))
    assert len(cached) == 1
    identity = cached[0].stem.rsplit("-", 1)[1]
    assert len(identity) == 16 and all(character in "0123456789abcdef" for character in identity)
    assert not prepared.tokenizer_reused


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


def test_a_second_run_with_only_the_tree_refuses_a_corpus_whose_files_have_moved(
    tmp_path: Path,
) -> None:
    """The pre-P33 refusal still stands when there is no snapshot to fall back on."""
    root = _root(tmp_path)
    plan = _plan(root, tmp_path / "work")
    prepare_corpus(plan)
    plan.snapshot_path().unlink()
    (root / "docs" / "alpha.md").write_text(
        "# Alpha\n\nRewritten after the manifest was pinned.\n", encoding="utf-8"
    )

    with pytest.raises(CorpusDriftError, match="content changed"):
        prepare_corpus(plan)


def test_a_second_run_trains_on_the_pinned_bytes_after_the_text_moves(tmp_path: Path) -> None:
    """A recorded run's snapshot is the corpus its directory pins, so the drift guard has something to read.

    This is the whole point of writing a snapshot: the tracked text moves, and the run keeps training on the
    bytes the record names instead of refusing or, worse, silently re-targeting the newer text.
    """
    root = _root(tmp_path)
    plan = _plan(root, tmp_path / "work")
    first = prepare_corpus(plan)
    (root / "docs" / "alpha.md").write_text(
        "# Alpha\n\nRewritten after the manifest was pinned.\n", encoding="utf-8"
    )

    second = prepare_corpus(plan)

    assert second.corpus_origin == "snapshot from this work dir"
    assert second.manifest.fingerprint == first.manifest.fingerprint
    assert second.manifest.combined_text() == first.manifest.combined_text()
    assert "Rewritten after the manifest was pinned" not in second.manifest.combined_text()


def test_a_declared_snapshot_is_used_without_reading_the_tracked_tree(tmp_path: Path) -> None:
    root = _root(tmp_path)
    source = tmp_path / "work"
    prepare_corpus(_plan(root, source))
    snapshot = source / "chakaso-repo-text-v1-snapshot.json"
    assert snapshot.is_file()

    # A root that has moved on in a way the original pin cannot match, and a fresh work directory with no
    # manifest of its own: only the declared snapshot can supply the corpus here.
    (root / "docs" / "alpha.md").write_text("# Alpha\n\nDifferent now.\n", encoding="utf-8")
    elsewhere = _plan(root, tmp_path / "other", corpus_snapshot=snapshot, split_seed=3)

    prepared = prepare_corpus(elsewhere)

    assert prepared.corpus_origin == "declared snapshot"
    assert prepared.split.fingerprint != prepare_corpus(_plan(root, source)).split.fingerprint


def test_a_run_refuses_a_corpus_that_is_not_the_one_its_protocol_names(tmp_path: Path) -> None:
    root = _root(tmp_path)
    plan = _plan(root, tmp_path / "work")
    prepared = prepare_corpus(plan)
    other = _plan(
        root,
        tmp_path / "other",
        expected_corpus_fingerprint="0" * 16,
    )

    assert prepared.manifest.fingerprint != "0" * 16
    with pytest.raises(CorpusError, match="requires corpus"):
        prepare_corpus(other)


def test_a_snapshot_that_disagrees_with_its_directorys_pin_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    prepare_corpus(_plan(root, tmp_path / "work"))
    foreign = _plan(_other_root(tmp_path), tmp_path / "foreign")
    prepare_corpus(foreign)
    work = tmp_path / "work"
    shutil.copy(foreign.snapshot_path(), work / "chakaso-repo-text-v1-snapshot.json")

    with pytest.raises(CorpusDriftError, match="must not be pointed at one corpus"):
        prepare_corpus(_plan(root, work))


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
    # A record cites this digest so a reader can check it without rerunning anything, which only works if
    # the bytes on disk are the text that was hashed.
    assert b"\r\n" not in target.read_bytes()
    assert sha256_hex(target.read_bytes().decode("utf-8")) == digest
    assert digest == write_report(reloaded, tmp_path / "again.json")


def test_a_report_does_not_replace_a_recorded_one_unasked(tmp_path: Path) -> None:
    report = run_experiment(_plan(_root(tmp_path), tmp_path / "work"))
    target = tmp_path / "results.json"
    write_report(report, target)

    with pytest.raises(ValueError, match="already holds a report"):
        write_report(report, target)

    assert write_report(report, target, allow_overwrite=True)


def test_the_summary_names_every_model_and_both_baselines(tmp_path: Path) -> None:
    line = summarise(run_experiment(_plan(_root(tmp_path), tmp_path / "work")))

    assert "transformer" in line
    assert "character" in line
    assert "unigram" in line
    assert "uniform" in line
    assert "bits/char" in line


def test_a_run_names_its_artifacts_after_its_experiment_and_its_corpus(tmp_path: Path) -> None:
    """One driver serves several experiments and corpora, so nothing may be named by a module constant."""
    plan = _plan(
        _root(tmp_path),
        tmp_path / "work",
        experiment_id="E2699",
        corpus=CorpusSpec(
            name="harbour-text",
            version="1.0.0",
            licence="Apache-2.0",
            source="built by tests",
            include=("docs/",),
            extensions=(".md",),
            exact_paths=(),
        ),
    )

    report = run_experiment(plan)

    assert report["experiment"] == "E2699"
    assert report["corpus"]["name"] == "harbour-text"
    assert report["corpus"]["include"] == ["docs/"]
    written = sorted(path.name for path in (tmp_path / "work").iterdir())
    # The resumable file carries the family in its name, so a checkpoint written by one architecture can
    # never be picked up as the resume point of the other by a directory convention alone.
    assert [name for name in written if not name.startswith("harbour-text-tokenizer-")] == [
        "E2699-transformer-resumable.pt",
        "E2699-transformer.pt",
        "harbour-text-manifest.json",
        "harbour-text-snapshot.json",
    ]
    assert len([name for name in written if name.startswith("harbour-text-tokenizer-")]) == 1
    assert report["corpus"]["snapshot"]["acquisition"] == {"method": "chakaso-working-tree"}


def test_a_plan_that_cannot_name_its_artifacts_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)

    with pytest.raises(ValueError, match="experiment_id"):
        _plan(root, tmp_path / "work", experiment_id="")
    with pytest.raises(ValueError, match="experiment_id"):
        _plan(root, tmp_path / "work", experiment_id="nested/E2606")
    with pytest.raises(ValueError, match="corpus name"):
        _plan(
            root,
            tmp_path / "work",
            corpus=CorpusSpec(name="a/b", version="1", licence="Apache-2.0", source="tests"),
        )
    with pytest.raises(ValueError, match="expected_corpus_fingerprint"):
        _plan(root, tmp_path / "work", expected_corpus_fingerprint="too-short")


def test_selection_is_a_function_of_the_validation_curve_alone(tmp_path: Path) -> None:
    """The reported step comes from held-out *selection* text; the test side is scored, never chosen from.

    E2601 measured that this architecture overfits well before its budget ends, so a step selected against
    the test documents would report a number the model never had to earn. The curve the report carries
    contains only validation losses, and the selected step is its argmin under the declared rule.
    """
    plan = _plan(_root(tmp_path), tmp_path / "work")
    report = run_experiment(plan)

    curve = report["transformer"]["curve"]
    assert all(row["validation_loss"] is not None for row in curve)
    validation = [(row["step"], row["validation_loss"]) for row in curve]
    selected = report["transformer"]["budget"]["best_step"]
    assert validation[min(range(len(validation)), key=lambda index: validation[index][1])] == (
        selected,
        min(value for _, value in validation),
    )
    assert report["transformer"]["budget"]["patience"] == plan.patience
    assert report["transformer"]["budget"]["min_delta"] == plan.min_delta
    assert report["transformer"]["test"]["positions_scored"] > 0


def test_a_snapshot_of_another_corpus_is_recorded_under_its_own_name(tmp_path: Path) -> None:
    """A run pointed at an external snapshot must not leave files named for this repository's corpus.

    The plan's default corpus spec is what a fresh preparation uses; when the bytes come from a snapshot,
    every record has to be named after the corpus the snapshot actually pins, or the directory would
    describe the wrong corpus.
    """
    from chakaso.training.manifest import build_manifest, write_snapshot

    root = _root(tmp_path)
    foreign = build_manifest(
        root,
        name="harbour-text",
        version="1.0.0",
        licence="Apache-2.0",
        source="built by tests",
        include=("docs/", "*.md"),
        extensions=(".md",),
        exact_paths=("LICENSE", "AGENTS.md"),
    )
    snapshot = tmp_path / "harbour-snapshot.json"
    write_snapshot(foreign, snapshot)
    work = tmp_path / "work"

    prepared = prepare_corpus(_plan(root, work, corpus_snapshot=snapshot))

    assert prepared.manifest.name == "harbour-text"
    written = sorted(path.name for path in work.iterdir())
    assert [name for name in written if not name.startswith("harbour-text-tokenizer-")] == [
        "harbour-text-manifest.json",
    ]
    assert len([name for name in written if name.startswith("harbour-text-tokenizer-")]) == 1


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


# ------------------------------------------------------------------ architecture families
def test_a_decoder_family_run_records_the_family_and_its_own_artifact(tmp_path: Path) -> None:
    import torch

    root = _root(tmp_path)
    work = tmp_path / "work-decoder"

    report = run_experiment(
        _plan(root, work, model_family="decoder", character_baseline=False, vocab_size=268)
    )

    assert report["model_family"] == "decoder"
    assert report["transformer"]["architecture"]["rope_theta"] == 10_000.0
    budget = report["transformer"]["budget"]
    assert budget["completion"] in {"converged", "budget_ceiling"}
    assert budget["positions_per_second"] > 0
    assert budget["schedule"] == "constant"
    assert budget["effective_batch_size"] == budget["batch_size"]
    assert report["character_baseline"]["skipped"] is True
    assert (work / "E2605-decoder.pt").is_file()
    assert (work / "E2605-decoder-resumable.pt").is_file()
    # The report names the schema the file is actually written under. It used to state the transformer
    # schema for every family, which described a file the loader would have rebuilt as the wrong
    # architecture — so the claim is checked against the artifact, not against a constant.
    artifact = torch.load(work / "E2605-decoder.pt", map_location="cpu", weights_only=True)
    assert report["transformer"]["artifact"]["schema"] == artifact["schema_version"]
    assert artifact["schema_version"] == "chakaso-decoder-checkpoint/1"


def test_a_skipped_character_baseline_says_why_in_the_report(tmp_path: Path) -> None:
    root = _root(tmp_path)

    report = run_experiment(
        _plan(root, tmp_path / "work-skip", model_family="decoder", character_baseline=False)
    )

    reason = report["character_baseline"]["reason"]
    assert "memory" in reason
    assert f"{report['corpus']['characters']:,}" in reason


def test_an_unknown_model_family_is_refused_before_anything_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_family"):
        _plan(_root(tmp_path), tmp_path / "work", model_family="mamba")


def test_the_decoder_family_needs_a_head_width_it_can_rotate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="even integer"):
        _plan(_root(tmp_path), tmp_path / "work", model_family="decoder", d_model=14, num_heads=2)


def test_an_unsupported_schedule_name_is_refused_on_the_plan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schedule"):
        _plan(_root(tmp_path), tmp_path / "work", schedule="warmup-rest-warmup")


def test_a_resume_from_another_corpus_is_refused_before_a_step_is_taken(tmp_path: Path) -> None:
    root = _root(tmp_path)
    work = tmp_path / "work-resume"
    run_experiment(_plan(root, work, max_steps=8, eval_every=4))
    point = work / "E2605-transformer-resumable.pt"
    other = _other_root(tmp_path)

    with pytest.raises(TrainingScheduleError, match="corpus:"):
        run_experiment(
            _plan(
                other,
                tmp_path / "work-other",
                resume_from=point,
                max_steps=12,
                eval_every=4,
            )
        )


def test_a_verified_resume_continues_from_the_step_it_stopped_at(tmp_path: Path) -> None:
    root = _root(tmp_path)
    work = tmp_path / "work-partial"
    run_experiment(_plan(root, work, max_steps=8, eval_every=4))
    point = work / "E2605-transformer-resumable.pt"

    report = run_experiment(
        _plan(
            root,
            work,
            max_steps=12,
            eval_every=4,
            resume_from=point,
            overwrite_artifacts=True,
        )
    )

    curve = report["transformer"]["curve"]
    assert curve[0]["step"] == 8
    assert curve[-1]["step"] == 12
    assert report["transformer"]["budget"]["records"] == len(curve)


def test_a_second_run_reuses_the_tokenizer_a_previous_run_fitted(tmp_path: Path) -> None:
    root = _root(tmp_path)
    work = tmp_path / "work-tokenizer-cache"

    first = prepare_corpus(_plan(root, work))
    second = prepare_corpus(_plan(root, work))

    assert not first.tokenizer_reused
    assert second.tokenizer_reused
    assert second.vocabulary.fingerprint == first.vocabulary.fingerprint
    assert len(list(work.glob("*-tokenizer-*.json"))) == 1


def test_a_different_split_seed_fits_a_new_tokenizer_rather_than_reusing_one(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    work = tmp_path / "work-tokenizer-seeds"
    prepare_corpus(_plan(root, work))

    other = prepare_corpus(_plan(root, work, split_seed=11))

    assert not other.tokenizer_reused
    assert len(list(work.glob("*-tokenizer-*.json"))) == 2


def test_a_tampered_cached_tokenizer_is_refused_rather_than_trusted(tmp_path: Path) -> None:
    # The cache is keyed by identity, so a plan asking for a different size simply fits a new tokenizer.
    # The failure worth guarding is a file whose bytes no longer match its own record: that is refused.
    root = _root(tmp_path)
    work = tmp_path / "work-tokenizer-tamper"
    prepare_corpus(_plan(root, work))
    cached = next(iter(work.glob("*-tokenizer-*.json")))
    document = json.loads(cached.read_text(encoding="utf-8"))
    document["merges"] = document["merges"][:2]
    cached.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="checksum"):
        prepare_corpus(_plan(root, work))


def test_a_shared_cache_directory_is_used_instead_of_each_runs_own(tmp_path: Path) -> None:
    root = _root(tmp_path)
    shared = tmp_path / "shared-tokenizers"
    first = prepare_corpus(_plan(root, tmp_path / "work-a", tokenizer_cache_dir=shared))
    second = prepare_corpus(_plan(root, tmp_path / "work-b", tokenizer_cache_dir=shared))

    assert not first.tokenizer_reused
    assert second.tokenizer_reused
    assert second.vocabulary.fingerprint == first.vocabulary.fingerprint
    assert len(list(shared.glob("*-tokenizer-*.json"))) == 1
    assert not any((tmp_path / "work-a").glob("*-tokenizer-*.json"))


def test_held_out_prompts_come_from_the_test_side_and_are_recorded(tmp_path: Path) -> None:
    root = _root(tmp_path)
    work = tmp_path / "work-held-out"

    report = run_experiment(_plan(root, work, sample_prompts="held_out", samples_per_prompt=4))

    samples = report["samples"]
    assert samples
    test_side = set(report["corpus"]["identifiers_per_side"]["test"])
    trained = set(report["corpus"]["identifiers_per_side"]["train"]) | set(
        report["corpus"]["identifiers_per_side"]["validation"]
    )
    for sample in samples:
        # The prompt's provenance is a document identifier, and it is one the model never trained on. That
        # is the whole point of the option: a heading from the trained corpus would flatter the sample.
        assert sample["prompt_source"] in test_side
        assert sample["prompt_source"] not in trained
        assert sample["text"] != ""


def test_the_repository_prompt_set_is_named_as_such(tmp_path: Path) -> None:
    root = _root(tmp_path)

    report = run_experiment(_plan(root, tmp_path / "work-repo-prompts", samples_per_prompt=4))

    assert {sample["prompt_source"] for sample in report["samples"]} == {"chakaso-repo-text-v1"}


def test_an_unknown_prompt_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_prompts"):
        _plan(_root(tmp_path), tmp_path / "work", sample_prompts="whatever-the-model-likes")


def test_the_one_line_summary_survives_a_skipped_character_baseline(tmp_path: Path) -> None:
    root = _root(tmp_path)

    report = run_experiment(_plan(root, tmp_path / "work-summary", character_baseline=False))
    line = summarise(report)

    assert "character baseline skipped" in line
    assert "bits/char" in line
    assert report["character_baseline"]["skipped"] is True
