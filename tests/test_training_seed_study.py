"""Tests for the seed study: declared seeds, one protocol, an aggregate rebuilt from disk.

The failure these tests exist to prevent is a summary that quietly stops describing one experiment — a reused
report from a different corpus, a seed that crashed and got averaged out, a protocol field that moved between
runs. Each is a way a mean of five numbers can be computed over things that are not five runs of the same
thing.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

from tests.test_training_subword_experiment import _plan, _root

from chakaso.core.hashing import sha256_hex
from chakaso.training import seed_study
from chakaso.training.errors import StudyProtocolError, TrainingScheduleError
from chakaso.training.nn_checkpoint import load_artifact
from chakaso.training.nn_training import verify_resume_compatibility
from chakaso.training.seed_study import (
    DEFAULT_METRICS,
    REPRODUCTION_TOLERANCES,
    SeedStudy,
    aggregate_metric,
    aggregate_study,
    build_study_report,
    collect_run_reports,
    compare_with_reference,
    protocol_agreement,
    protocol_identity,
    run_seed_study,
    study_summary,
    write_study_report,
)
from chakaso.training.subword_experiment import (
    REPO_TEXT_CORPUS,
    ExperimentPlan,
    _training_config,
    prepare_corpus,
    run_experiment,
)

# ------------------------------------------------------------------ declaration


def _study_plan(tmp_path: Path) -> ExperimentPlan:
    """A protocol whose corpus and split are fixed, drawn from the same synthetic root the runs use."""
    root = _root(tmp_path)
    pin_dir = tmp_path / "pin"
    pinned = prepare_corpus(_plan(root, pin_dir))
    assert pinned.snapshot is not None
    return _plan(
        root,
        tmp_path / "study",
        corpus_snapshot=REPO_TEXT_CORPUS.snapshot_path(pin_dir),
        split_seed=7,
        expected_corpus_fingerprint=pinned.manifest.fingerprint,
    )


def test_a_study_refuses_to_run_without_a_fixed_split_and_corpus(tmp_path: Path) -> None:
    plan = _plan(_root(tmp_path), tmp_path / "study")

    with pytest.raises(StudyProtocolError, match="split_seed"):
        SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))
    without_corpus = replace(plan, split_seed=7)
    with pytest.raises(StudyProtocolError, match="corpus fingerprint"):
        SeedStudy(experiment="E2698", plan=without_corpus, seeds=(7, 11))


def test_a_study_refuses_seeds_that_are_not_a_declared_distinct_list(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)

    with pytest.raises(StudyProtocolError, match="no declared seeds"):
        SeedStudy(experiment="E2698", plan=plan, seeds=())
    with pytest.raises(StudyProtocolError, match="distinct"):
        SeedStudy(experiment="E2698", plan=plan, seeds=(7, 7))
    with pytest.raises(StudyProtocolError, match="integer"):
        SeedStudy(experiment="E2698", plan=plan, seeds=(7, True))  # type: ignore[arg-type]
    with pytest.raises(StudyProtocolError, match="directory name"):
        SeedStudy(experiment="nested/E2698", plan=plan, seeds=(7,))


def test_each_seed_gets_its_own_directory_and_keeps_the_shared_protocol(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))

    first = study.plan_for(7)
    second = study.plan_for(11)

    assert first.work_dir == tmp_path / "study" / "E2698" / "seed-7"
    assert second.work_dir == tmp_path / "study" / "E2698" / "seed-11"
    assert (first.seed, second.seed) == (7, 11)
    assert first.split_seed == second.split_seed == 7
    assert first.experiment_id == "E2698"
    assert first.expected_corpus_fingerprint == plan.expected_corpus_fingerprint
    assert first.corpus_snapshot == plan.corpus_snapshot


# ------------------------------------------------------------------ aggregation


def test_the_aggregate_reports_centre_spread_and_extent_and_invents_no_interval() -> None:
    summary = aggregate_metric([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["runs"] == 5
    assert summary["mean"] == 3.0
    assert summary["sample_stdev"] == pytest.approx(math.sqrt(2.5), abs=1e-6)
    assert summary["median"] == 3.0
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0
    assert summary["range"] == 4.0
    assert "confidence_interval" not in summary


def test_a_single_run_has_no_sample_spread_and_no_runs_has_nothing() -> None:
    assert aggregate_metric([2.0])["sample_stdev"] is None
    assert aggregate_metric([2.0])["runs"] == 1
    assert aggregate_metric([]) == {"runs": 0, "mean": None, "sample_stdev": None}


# ------------------------------------------------------------------ fake reports


def _run_report(seed: int, *, bits: float = 2.0, corpus: str | None = None) -> dict[str, Any]:
    """A report shaped like the driver's, carrying only the fields the study reads."""
    architecture = {
        "vocab_size": 268,
        "context_length": 16,
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "feedforward_dim": 32,
        "dropout": 0.0,
        "tie_embeddings": True,
        "initialization_std": 0.02,
    }
    return {
        "experiment": "E2698",
        "corpus": {
            "fingerprint": corpus or "a" * 16,
            "split_fingerprint": "s" * 16,
            "split_seed": 7,
            "model_seed": seed,
            "prepared_from": "declared snapshot",
        },
        "tokenizer": {"fingerprint": "t" * 16, "size": 268},
        "transformer": {
            "architecture": architecture,
            "parameter_count": 12345,
            "budget": {
                "batch_size": 4,
                "max_steps": 12,
                "eval_every": 4,
                "patience": 3,
                "min_delta": 0.0,
                "learning_rate": 0.0006,
                "weight_decay": 0.01,
                "warmup_steps": 2,
                "gradient_clip_norm": 1.0,
                "best_step": 8,
                "stopped_early": True,
                "wall_clock_s": 1.0,
            },
            "test": {
                "bits_per_character": bits,
                "loss_nats_per_token": round(bits * math.log(2), 6),
                "perplexity": round(2.0**bits, 6),
                "token_accuracy": 0.5,
                "positions_scored": 100,
                "characters_scored": 140,
            },
            "artifact": {"checksum": f"c{seed}" + "0" * (64 - len(str(seed)) - 1)},
        },
        "character_baseline": {
            "budget": {"max_steps": 40, "wall_clock_s": 0.5},
            "test": {"bits_per_character": 3.0, "loss_nats_per_token": 3.0 * math.log(2)},
        },
        "baselines_subword": {
            "unigram": {"bits_per_character": 3.4},
            "uniform": {"bits_per_character": 4.0},
        },
        "environment": {"device": "cpu"},
        # The keys a real report carries and the whole-protocol digest reads: the plan with the seed that varies
        # between runs, and the family. A fake without them would test an aggregator no real report can feed.
        "model_family": "transformer",
        "plan": {"seed": seed, "max_steps": 12, "work_dir": f"runs/seed-{seed}"},
    }


def _write_run(work: Path, experiment: str, seed: int, report: dict[str, Any]) -> Path:
    directory = work / experiment / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8", newline=""
    )
    return directory


def _seeded_dir(tmp_path: Path) -> Path:
    _write_run(tmp_path, "E2698", 7, _run_report(7, bits=2.0))
    _write_run(tmp_path, "E2698", 11, _run_report(11, bits=4.0))
    return tmp_path


def test_protocol_identity_names_the_fields_that_must_not_move() -> None:
    identity = protocol_identity(_run_report(7))

    assert identity["corpus_fingerprint"] == "a" * 16
    assert identity["split_seed"] == 7
    assert identity["parameter_count"] == 12345
    assert identity["test_positions"] == 100
    assert identity["character_steps"] == 40


def test_agreement_reports_the_shared_protocol_and_a_seed_that_broke_it() -> None:
    shared = protocol_agreement({7: _run_report(7), 11: _run_report(11)})

    assert shared["runs_compared"] == 2
    assert shared["disagreements"] == {}

    with_moved_corpus = protocol_agreement(
        {7: _run_report(7), 13: _run_report(13, corpus="b" * 16)}
    )

    # The named field is reported, and so is the whole-protocol digest that covers it: the digest is the net that
    # catches a difference nobody listed (K-030 item 5), so a listed difference must trip it too.
    assert with_moved_corpus["disagreements"] == {
        "corpus_fingerprint": [13],
        "protocol_digest": [13],
    }


def test_a_difference_nobody_listed_still_breaks_the_study_protocol() -> None:
    """K-030 item 5: the hand-kept field list names what someone thought of; the digest covers the rest."""
    unlisted = _run_report(13)
    unlisted["plan"]["max_steps"] = (
        99  # not in _PROTOCOL_FIELDS, so only the whole-protocol digest sees it
    )

    broken = protocol_agreement({7: _run_report(7), 13: unlisted})

    assert broken["disagreements"] == {"protocol_digest": [13]}
    # ...while the seed itself, which every run of a study differs in, is not a disagreement.
    assert protocol_agreement({7: _run_report(7), 11: _run_report(11)})["disagreements"] == {}


def test_a_missing_metric_path_is_refused_rather_than_averaged_as_zero() -> None:
    truncated = _run_report(7)
    del truncated["transformer"]["test"]["bits_per_character"]

    with pytest.raises(StudyProtocolError, match=r"has no 'transformer/test/bits_per_character'"):
        aggregate_study({7: truncated})

    no_protocol_field = _run_report(7)
    del no_protocol_field["transformer"]["budget"]["patience"]

    with pytest.raises(StudyProtocolError, match="stopped at 'patience'"):
        protocol_identity(no_protocol_field)


def test_the_aggregate_carries_every_declared_metric() -> None:
    aggregate = aggregate_study({7: _run_report(7, bits=2.0), 11: _run_report(11, bits=4.0)})

    assert set(aggregate) == set(DEFAULT_METRICS)
    assert aggregate["transformer_bits_per_character"]["mean"] == 3.0
    assert aggregate["uniform_bits_per_character"]["runs"] == 2


# ------------------------------------------------------------------ building from disk


def test_collect_names_the_seeds_that_have_no_report(tmp_path: Path) -> None:
    _write_run(tmp_path, "E2698", 7, _run_report(7))

    found, missing = collect_run_reports(experiment="E2698", work_dir=tmp_path, seeds=[7, 11, 13])

    assert sorted(found) == [7]
    assert missing == [11, 13]


def test_an_aggregate_refuses_runs_that_do_not_share_one_protocol(tmp_path: Path) -> None:
    _write_run(tmp_path, "E2698", 7, _run_report(7))
    _write_run(tmp_path, "E2698", 11, _run_report(11, corpus="z" * 16))

    with pytest.raises(StudyProtocolError, match="do not share one protocol"):
        build_study_report(experiment="E2698", work_dir=tmp_path, seeds=[7, 11])


def test_the_aggregate_is_rebuilt_from_disk_without_running_anything(tmp_path: Path) -> None:
    _write_run(tmp_path, "E2698", 7, _run_report(7, bits=2.0))
    _write_run(tmp_path, "E2698", 11, _run_report(11, bits=4.0))

    report = build_study_report(experiment="E2698", work_dir=tmp_path, seeds=[7, 11, 13])

    assert report["completed_seeds"] == [7, 11]
    assert report["missing_seeds"] == [13]
    assert report["failed_seeds"] == []
    assert report["aggregate"]["transformer_bits_per_character"]["mean"] == 3.0
    assert report["per_run"]["7"]["artifact_checksum"].startswith("c7")
    assert "confidence_interval" not in json.dumps(report)

    destination = tmp_path / "study.json"
    digest = write_study_report(report, destination)
    assert digest == sha256_hex(destination.read_bytes().decode("utf-8"))
    assert not list(tmp_path.glob("*.partial"))


def test_the_summary_line_names_the_sample_the_spread_and_the_missing_runs(tmp_path: Path) -> None:
    report = build_study_report(
        experiment="E2698", work_dir=_seeded_dir(tmp_path), seeds=[7, 11, 13]
    )

    line = study_summary(report)

    assert "E2698: 2/3 seeds" in line
    assert "sample stdev 1.4142" in line
    assert "missing [13]" in line


def test_the_summary_survives_a_study_with_nothing_completed(tmp_path: Path) -> None:
    """The line an operator reads after an hour of training must not crash on the failure case."""
    report = build_study_report(experiment="E2698", work_dir=tmp_path, seeds=[7, 11])

    line = study_summary(report)

    assert line == "E2698: 0/2 seeds, nothing measured yet, missing [7, 11]"


def test_a_single_completed_run_reports_no_spread_and_says_single_run(tmp_path: Path) -> None:
    _write_run(tmp_path, "E2698", 7, _run_report(7, bits=2.0))

    report = build_study_report(experiment="E2698", work_dir=tmp_path, seeds=[7, 11])
    line = study_summary(report)

    assert report["aggregate"]["transformer_bits_per_character"]["sample_stdev"] is None
    assert "sample stdev single run" in line
    assert "missing [11]" in line


# ------------------------------------------------------------------ reproduction comparison


def test_reproduction_agrees_exactly_on_identity_and_within_tolerance_on_loss() -> None:
    verdict = compare_with_reference(_run_report(7, bits=1.8659), _run_report(7, bits=1.8657))

    assert verdict["verdict"] == "reproduced"
    assert verdict["identity"]["corpus_fingerprint"]["match"] is True
    measured = verdict["metrics"]["transformer_bits_per_character"]
    assert measured["difference"] == pytest.approx(0.0002, abs=1e-9)
    assert measured["within_tolerance"] is True
    assert measured["tolerance"] == REPRODUCTION_TOLERANCES["transformer_bits_per_character"]


def test_a_reproduction_that_moves_an_identity_field_is_a_deviation() -> None:
    verdict = compare_with_reference(_run_report(7, corpus="b" * 16), _run_report(7))

    assert verdict["verdict"] == "deviated"
    assert "corpus_fingerprint" in verdict["deviations"]
    assert verdict["identity"]["corpus_fingerprint"]["recorded"] == "a" * 16


def test_a_metric_beyond_its_declared_tolerance_is_reported_as_a_deviation() -> None:
    verdict = compare_with_reference(_run_report(7, bits=1.87), _run_report(7, bits=1.8657))

    assert verdict["verdict"] == "deviated"
    assert "transformer_bits_per_character" in verdict["deviations"]
    assert verdict["metrics"]["transformer_bits_per_character"]["within_tolerance"] is False


def test_a_baseline_that_moves_at_all_is_a_deviation() -> None:
    recorded = _run_report(7)
    shifted = _run_report(7)
    shifted["baselines_subword"]["uniform"]["bits_per_character"] = 4.0001

    verdict = compare_with_reference(shifted, recorded)

    assert "uniform_bits_per_character" in verdict["deviations"]


def test_a_reference_needs_a_completed_seed(tmp_path: Path) -> None:
    _write_run(tmp_path, "E2698", 7, _run_report(7))

    with pytest.raises(StudyProtocolError, match="reference seed"):
        build_study_report(
            experiment="E2698",
            work_dir=tmp_path,
            seeds=[7, 11],
            reference=_run_report(11),
            reference_seed=11,
        )


def test_a_reference_comparison_is_part_of_the_report_it_belongs_to(tmp_path: Path) -> None:
    _write_run(tmp_path, "E2698", 7, _run_report(7, bits=1.8659))

    report = build_study_report(
        experiment="E2698",
        work_dir=tmp_path,
        seeds=[7],
        reference=_run_report(7, bits=1.8657),
        reference_seed=7,
    )

    assert report["reproduction"]["verdict"] == "reproduced"
    assert "reproduction reproduced" in study_summary(report)


# ------------------------------------------------------------------ running a study


def test_a_study_runs_its_declared_seeds_on_one_corpus(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))

    report = run_seed_study(study)
    rebuilt = build_study_report(experiment="E2698", work_dir=plan.work_dir, seeds=[7, 11])

    assert report["completed_seeds"] == [7, 11]
    assert report["failed_seeds"] == []
    assert report["aggregate"]["transformer_bits_per_character"]["runs"] == 2
    assert report["aggregate"]["transformer_bits_per_character"]["mean"] == pytest.approx(
        rebuilt["aggregate"]["transformer_bits_per_character"]["mean"]
    )
    # Only the seed moved: the corpus, split, tokenizer and test side are shared, and the aggregate says so.
    assert report["protocol"]["corpus_fingerprint"] == plan.expected_corpus_fingerprint
    assert report["protocol"]["split_seed"] == 7
    assert report["protocol"]["split_fingerprint"]
    assert report["protocol"]["test_positions"] > 0
    assert {run["corpus_origin"] for run in report["per_run"].values()} == {"declared snapshot"}
    for seed in (7, 11):
        assert (plan.work_dir / "E2698" / f"seed-{seed}" / "E2698-transformer.pt").is_file()


def test_two_seeds_measure_the_same_corpus_and_different_initializations(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)

    report = run_seed_study(SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11)))

    # Building the report at all proves the two runs agreed on every protocol field, because a
    # disagreement raises. What has to differ is the trained model itself.
    checksums = {run["artifact_checksum"] for run in report["per_run"].values()}
    assert len(checksums) == 2
    assert (
        report["per_run"]["7"]["metrics"]["uniform_bits_per_character"]
        == report["per_run"]["11"]["metrics"]["uniform_bits_per_character"]
    )
    assert report["runs"][0]["report_digest"] != report["runs"][1]["report_digest"]


def test_a_second_pass_reuses_completed_runs(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))

    first = run_seed_study(study)
    second = run_seed_study(study)

    assert first["runs"][0]["reused"] is False
    assert second["runs"][0]["reused"] is True
    assert (
        second["aggregate"]["transformer_bits_per_character"]["mean"]
        == first["aggregate"]["transformer_bits_per_character"]["mean"]
    )


def test_a_reused_report_must_belong_to_this_study(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))
    run_seed_study(study)

    stale = study.run_dir(7) / "report.json"
    payload = json.loads(stale.read_text(encoding="utf-8"))
    payload["corpus"]["fingerprint"] = "f" * 16
    stale.write_text(json.dumps(payload), encoding="utf-8", newline="")

    with pytest.raises(StudyProtocolError, match="corpus_fingerprint"):
        run_seed_study(study)


def test_collecting_with_a_study_checks_its_declaration_not_only_agreement(tmp_path: Path) -> None:
    """Five runs that share a *different* protocol still agree with each other.

    Cross-run agreement alone would happily aggregate them, so a collection made against a declared study
    has to check each report against what that study says its runs must look like.
    """
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))
    run_seed_study(study)

    rebuilt = build_study_report(experiment="E2698", work_dir=plan.work_dir, seeds=[7], study=study)
    assert rebuilt["completed_seeds"] == [7]

    wider = SeedStudy(
        experiment="E2698",
        plan=replace(plan, max_steps=plan.max_steps + 60),
        seeds=(7,),
    )
    with pytest.raises(StudyProtocolError, match="max_steps"):
        build_study_report(experiment="E2698", work_dir=plan.work_dir, seeds=[7], study=wider)


def test_rerunning_over_recorded_reports_needs_an_explicit_declaration(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))
    run_seed_study(study)

    with pytest.raises(StudyProtocolError, match="already have a completed report"):
        run_seed_study(study, reuse_completed=False)

    declared = SeedStudy(
        experiment="E2698", plan=replace(plan, overwrite_artifacts=True), seeds=(7,)
    )
    rerun = run_seed_study(declared, reuse_completed=False)
    assert rerun["completed_seeds"] == [7]
    assert rerun["runs"][0]["reused"] is False


def test_a_rerun_can_replace_the_artifacts_in_its_own_seed_directory(tmp_path: Path) -> None:
    """The report is what makes a run recorded, so a seed with no report has nothing to clobber."""
    study = SeedStudy(experiment="E2698", plan=_study_plan(tmp_path), seeds=(7,))

    assert study.plan_for(7).overwrite_artifacts is True
    assert study.plan.overwrite_artifacts is False


def test_a_reused_report_must_have_used_the_seed_its_directory_claims(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))
    _write_run(plan.work_dir, "E2698", 7, _run_report(11))

    with pytest.raises(StudyProtocolError, match="model_seed"):
        run_seed_study(study)

    # A budget that moved is the same defect: the run is not of this protocol.
    (study.run_dir(7) / "report.json").unlink()
    shifted = _run_report(7)
    shifted["transformer"]["budget"]["max_steps"] = 999
    _write_run(plan.work_dir, "E2698", 7, shifted)
    with pytest.raises(StudyProtocolError, match="max_steps"):
        run_seed_study(study)


def test_a_seed_that_raises_is_recorded_as_failed_and_the_others_still_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))

    def _fail_on_eleven(candidate: ExperimentPlan, **kwargs: Any) -> dict[str, Any]:
        if candidate.seed == 11:
            failure = "torch refused"
            raise RuntimeError(failure)
        return run_experiment(candidate, **kwargs)

    monkeypatch.setattr(seed_study, "run_experiment", _fail_on_eleven)

    report = run_seed_study(study)

    assert report["completed_seeds"] == [7]
    assert report["failed_seeds"] == [11]
    assert report["aggregate"]["transformer_bits_per_character"]["runs"] == 1
    assert report["aggregate"]["transformer_bits_per_character"]["sample_stdev"] is None
    assert "torch refused" in report["runs"][1]["error"]
    assert "failed [11]" in study_summary(report)


# ------------------------------------------------------------------ a decoder study without a character baseline


def _decoder_report(seed: int, *, prose: float, science: float) -> dict[str, Any]:
    """A report shaped like an E2612-style run: baseline skipped, two declared test partitions."""
    report = _run_report(seed, bits=(prose + science) / 2)
    report["character_baseline"] = {"skipped": True, "reason": "declared off"}
    report["transformer"]["test_partitions"] = {
        "prose-": {"metrics": {"bits_per_character": prose}},
        "science-": {"metrics": {"bits_per_character": science}},
    }
    return report


def test_a_skipped_character_baseline_is_a_declared_part_of_the_protocol_not_a_refusal() -> None:
    identity = protocol_identity(_decoder_report(7, prose=1.8, science=2.0))

    assert identity["character_steps"] is None
    assert identity["tokenizer_fingerprint"] == "t" * 16


def test_the_metrics_follow_the_plan_it_measured(tmp_path: Path) -> None:
    from chakaso.training.seed_study import metrics_for_plan

    plain = metrics_for_plan(_plan(_root(tmp_path), tmp_path / "work"))
    decoder = metrics_for_plan(
        _plan(
            _root(tmp_path),
            tmp_path / "work",
            character_baseline=False,
            test_partitions=("prose-", "science-"),
        )
    )

    assert "character_bits_per_character" in plain
    assert not any(name.startswith("partition_") for name in plain)
    assert "character_bits_per_character" not in decoder
    assert decoder["partition_prose-_bits_per_character"] == (
        "transformer",
        "test_partitions",
        "prose-",
        "metrics",
        "bits_per_character",
    )


def test_a_decoder_study_aggregates_each_partition_and_summarises_without_a_character_model(
    tmp_path: Path,
) -> None:
    from chakaso.training.seed_study import metrics_for_plan

    plan = _plan(
        _root(tmp_path),
        tmp_path / "work",
        character_baseline=False,
        test_partitions=("prose-", "science-"),
    )
    metrics = metrics_for_plan(plan)
    reports = {
        seed: _decoder_report(seed, prose=1.80 + step * 0.01, science=2.0 + step * 0.02)
        for step, seed in enumerate((7, 11, 13, 17, 19))
    }

    aggregate = aggregate_study(reports, metrics=metrics)

    assert aggregate["partition_prose-_bits_per_character"]["mean"] == pytest.approx(1.82)
    assert aggregate["partition_science-_bits_per_character"]["sample_stdev"] == pytest.approx(
        math.sqrt(0.0010), abs=1e-6
    )
    line = study_summary(
        {
            "study": "E2699",
            "declared_seeds": [7, 11, 13, 17, 19],
            "missing_seeds": [],
            "failed_seeds": [],
            "aggregate": {
                "transformer_bits_per_character": aggregate["transformer_bits_per_character"]
            },
        }
    )
    assert "character" not in line and "sample stdev" in line


def test_a_stdev_from_few_runs_comes_with_the_interval_that_few_runs_deserve() -> None:
    three = aggregate_metric([1.00, 1.01, 1.03])
    five = aggregate_metric([1.00, 1.01, 1.03, 1.02, 0.99])

    for summary, low, high in ((three, 0.521, 6.285), (five, 0.599, 2.874)):
        stdev = summary["sample_stdev"]
        assert summary["stdev_95_interval"] == pytest.approx(
            [stdev * low, stdev * high], rel=2e-3
        ), "the multipliers are the standard chi-square values for n-1 degrees of freedom"
    assert "stdev_95_interval" not in aggregate_metric([2.0])


@pytest.mark.parametrize(
    ("degrees", "lower", "upper"),
    [(2, 0.0506, 7.3778), (4, 0.4844, 11.1433), (6, 1.2373, 14.4494), (9, 2.7004, 19.0228)],
)
def test_the_chi_square_quantiles_match_the_published_table(
    degrees: int, lower: float, upper: float
) -> None:
    from chakaso.training.seed_study import chi_square_quantile

    assert chi_square_quantile(0.025, degrees) == pytest.approx(lower, abs=2e-4)
    assert chi_square_quantile(0.975, degrees) == pytest.approx(upper, abs=2e-4)
    with pytest.raises(ValueError, match="probability"):
        chi_square_quantile(1.0, degrees)


def test_a_reproduction_of_a_run_without_a_character_baseline_compares_what_both_measured() -> None:
    recorded = _decoder_report(7, prose=1.80, science=2.0)
    rerun = _decoder_report(7, prose=1.80, science=2.0)

    result = compare_with_reference(rerun, recorded)

    assert result["verdict"] == "reproduced"
    assert not any(name.startswith("character_") for name in result["metrics"])

    drifted = _decoder_report(7, prose=1.80, science=2.0)
    drifted["transformer"]["test"]["bits_per_character"] += 0.01
    assert compare_with_reference(drifted, recorded)["verdict"] == "deviated"


# ------------------------------------------------------------------ what recovery can address on disk
#
# The audit that filed K-036 established the layout a per-seed resume has to be able to rely on. These pin
# the parts of it that hold today, so the discovery that closes K-036 is built on a stated contract rather
# than on the same directory arrangement by luck.


def test_each_seed_writes_its_resume_point_inside_its_own_directory(tmp_path: Path) -> None:
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))

    run_seed_study(study)

    for seed in (7, 11):
        resumable = study.run_dir(seed) / "E2698-transformer-resumable.pt"
        assert resumable.is_file(), (
            f"seed {seed} left no resume point where only seed {seed} writes"
        )
        point = load_artifact(resumable)
        assert point.run_metadata["kind"] == "resumable"
        assert point.optimizer_state is not None
        assert point.run_metadata["training"]["seed"] == seed
    # The study shares one tokenizer cache across its seeds because the study fixes the corpus and the split,
    # and it shares nothing else: a resume point at the top level would belong to no seed and be discoverable
    # by all of them.
    assert not list(plan.work_dir.glob("*-resumable.pt"))


def test_the_usable_artifact_and_the_resume_point_are_two_files_of_one_seed(tmp_path: Path) -> None:
    """A finished seed holds both a model and the moments that built it, and only one continues a run.

    Discovery has to reach for the resumable name specifically. The usable artifact carries the
    validation-selected parameters rather than the last step's, so continuing from it would resume Adam's
    moments against weights they were never computed for.
    """
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))

    run_seed_study(study)

    usable = load_artifact(study.run_dir(7) / "E2698-transformer.pt")
    resumable = load_artifact(study.run_dir(7) / "E2698-transformer-resumable.pt")
    assert usable.run_metadata["kind"] == "usable"
    assert usable.optimizer_state is None
    assert resumable.step == plan.max_steps

    with pytest.raises(TrainingScheduleError, match="no optimizer state"):
        verify_resume_compatibility(
            usable,
            config=_training_config(study.plan_for(7), usable.config),
            corpus_fingerprint=usable.corpus_fingerprint,
            vocabulary=usable.vocabulary,
        )


def test_a_seed_cannot_continue_from_its_sibling_s_resume_point(tmp_path: Path) -> None:
    """The seed is part of the optimization settings, so the existing configuration check already names it.

    This is the guarantee Milestone B's discovery must not weaken: two seeds of one study share their corpus,
    their split, their tokenizer and every budget number, and differ in nothing but the seed, so a
    mis-associated checkpoint is a checkpoint that agrees with the run on everything except the one field that
    says which run it is.
    """
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))
    run_seed_study(study)
    point_7 = study.run_dir(7) / "E2698-transformer-resumable.pt"

    with pytest.raises(TrainingScheduleError, match=r"\['seed'\]"):
        run_experiment(
            replace(
                plan,
                seed=11,
                work_dir=tmp_path / "wrong-seed",
                resume_from=point_7,
                overwrite_artifacts=True,
            )
        )


# ------------------------------------------------------------------ K-036: discovery of an interrupted seed


def _converging_plan(tmp_path: Path) -> ExperimentPlan:
    """A protocol that stops by its own rule before its ceiling, so a checkpoint lands mid-curve.

    A run that reaches ``max_steps`` leaves a resume point the study has nothing to continue, so the
    interruption K-036 is about cannot be reproduced with the default budget. ``min_delta`` above any real
    improvement makes the declared patience rule stop the run at its second evaluation while the ceiling is
    still four steps away.
    """
    return replace(_study_plan(tmp_path), patience=1, min_delta=9.0)


def _resume_point(study: SeedStudy, seed: int) -> Path:
    return study.run_dir(seed) / "E2698-transformer-resumable.pt"


def test_an_interrupted_seed_continues_from_the_checkpoint_its_own_run_left(
    tmp_path: Path,
) -> None:
    plan = _converging_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))
    run_seed_study(study)

    stopped_at = load_artifact(_resume_point(study, 7)).step
    assert 0 < stopped_at < plan.max_steps
    (study.run_dir(7) / "report.json").unlink()

    recovered = run_seed_study(SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11)))
    body = json.loads((study.run_dir(7) / "report.json").read_text(encoding="utf-8"))

    assert recovered["runs"][0]["status"] == "completed"
    assert recovered["runs"][0]["reused"] is False
    assert recovered["runs"][1]["reused"] is True
    assert body["transformer"]["budget"]["resumed_from_step"] == stopped_at
    # The curve opens at the step the interrupted run reached: a clean start always opens at 0, so this is the
    # statement that the first eight steps were not trained a second time.
    assert next(row["step"] for row in body["transformer"]["curve"]) == stopped_at
    assert body["transformer"]["curve"][-1]["step"] == plan.max_steps


def test_a_recovered_seed_aggregates_with_the_sibling_that_never_stopped(
    tmp_path: Path,
) -> None:
    """K-037: the study has to form, not just the run.

    One seed resumed from its own checkpoint and one that ran uninterrupted are the same protocol by
    construction — that is the entire property the resume mechanism exists to give — so their reports must
    share a ``protocol_digest``. With the plan's ``resume_from`` path hashed as a condition, this is where the
    recovery died: ``StudyProtocolError`` naming the digest, and the aggregate missing a seed that had just
    been paid for.
    """
    plan = _converging_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))
    run_seed_study(study)
    (study.run_dir(7) / "report.json").unlink()

    recovered = run_seed_study(SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11)))

    assert recovered["completed_seeds"] == [7, 11]
    assert recovered["missing_seeds"] == []
    assert recovered["protocol"]["protocol_digest"]


def test_a_seed_with_nothing_on_disk_starts_clean(tmp_path: Path) -> None:
    plan = _converging_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))

    assert study.discover_resume_point(7) is None
    first = run_seed_study(study)
    body = json.loads((study.run_dir(7) / "report.json").read_text(encoding="utf-8"))

    assert first["runs"][0]["reused"] is False
    assert body["transformer"]["budget"]["resumed_from_step"] is None
    assert body["transformer"]["curve"][0]["step"] == 0


def test_a_checkpoint_at_the_ceiling_is_not_a_run_left_to_continue(tmp_path: Path) -> None:
    """A seed that trained to its ceiling and recorded nothing has no step to resume onto.

    ``train_transformer`` refuses a ``start_step`` already at ``max_steps``, and it is right to: continuing
    past the ceiling would train a schedule the configuration does not describe. Discovery reads that as
    "nothing left to continue" and starts clean, which reproduces the same curve from the same fixed seed,
    split and batch plan rather than a different one.
    """
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))
    run_seed_study(study)

    assert load_artifact(_resume_point(study, 7)).step == plan.max_steps
    (study.run_dir(7) / "report.json").unlink()
    assert study.discover_resume_point(7) is None

    rerun = run_seed_study(SeedStudy(experiment="E2698", plan=plan, seeds=(7,)))
    body = json.loads((study.run_dir(7) / "report.json").read_text(encoding="utf-8"))

    assert rerun["completed_seeds"] == [7]
    assert body["transformer"]["budget"]["resumed_from_step"] is None


def test_a_completed_seeds_report_beats_its_own_checkpoint(tmp_path: Path) -> None:
    """The report is what makes a run recorded, and it wins over anything else in the directory."""
    plan = _study_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7,))
    run_seed_study(study)
    digest_before = sha256_hex((study.run_dir(7) / "report.json").read_bytes().decode("utf-8"))

    again = run_seed_study(study)

    assert again["runs"][0]["reused"] is True
    assert _resume_point(study, 7).is_file()
    assert sha256_hex((study.run_dir(7) / "report.json").read_bytes().decode("utf-8")) == (
        digest_before
    )


def test_an_unreadable_seed_checkpoint_fails_that_seed_and_names_itself(tmp_path: Path) -> None:
    """Recovery refuses rather than retraining over evidence the operator can still look at.

    A staged swap leaves either the whole file or the previous one, so a resume point that will not load was
    altered after the run wrote it. Silently starting clean would discard bytes nobody can explain; this
    records the seed as failed with the loader's reason, and still runs the seeds that are fine.
    """
    plan = _converging_plan(tmp_path)
    study = SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11))
    run_seed_study(study)
    (study.run_dir(7) / "report.json").unlink()
    _resume_point(study, 7).write_bytes(b"not an artifact")

    result = run_seed_study(SeedStudy(experiment="E2698", plan=plan, seeds=(7, 11)))

    failed = next(run for run in result["runs"] if run["seed"] == 7)
    assert failed["status"] == "failed"
    assert "CorruptedCheckpointError" in failed["error"]
    assert result["failed_seeds"] == [7]
    assert result["completed_seeds"] == [11]
