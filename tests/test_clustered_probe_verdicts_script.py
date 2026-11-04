"""The clustered-versus-item-level reading over a probe's committed rows: what it reports and what it refuses.

The fixture is designed so both answers are derived before the script runs. Nine items in three works, one arm
passing every item and the other failing every one: McNemar sees nine independent coins and reports p = 2 / 2 ** 8,
while the randomization reading sees three works, of which four assignments exist once the first is held fixed, and
reports p = 0.25. That divergence is the whole reason the script exists, so it is asserted on the output line rather
than described.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from chakaso.training.item_clusters import WORK_CLUSTER_UNIT, ClusterListing

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "clustered_probe_verdicts.py"
_MODELS = ("arm-a", "arm-b", "arm-c")


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("clustered_probe_verdicts_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _items() -> list[tuple[str, str]]:
    return [(f"ctx-prose-{index:04d}", f"w{index // 3}") for index in range(9)]


def _verdicts(model: str) -> list[float]:
    if model == "arm-b":
        return [0.0] * 9
    return [1.0] * 9


def _probe() -> dict[str, Any]:
    rows = _items()
    return {
        "items": {
            "count": len(rows),
            "digest": "i" * 64,
            "clusters": ClusterListing(unit=WORK_CLUSTER_UNIT, rows=tuple(rows)).as_dict(),
        },
        "models": {
            model: {
                "result": {
                    "outcomes": {
                        "pmi_prior_span": [
                            [identifier, _verdicts(model)[index]]
                            for index, (identifier, _) in enumerate(rows)
                        ]
                    }
                }
            }
            for model in _MODELS
        },
    }


def _argv(probe_path: Path, out: Path) -> list[str]:
    return [
        "--probe",
        str(probe_path),
        "--reading",
        "pmi_prior_span",
        "--model",
        "arm-a",
        "--model",
        "arm-b",
        "--model",
        "arm-c",
        "--seed",
        "7",
        "--resamples",
        "500",
        "--out",
        str(out),
    ]


@pytest.fixture
def probe_file(tmp_path: Path) -> Path:
    path = tmp_path / "context.json"
    path.write_text(json.dumps(_probe()), encoding="utf-8")
    return path


# ------------------------------------------------------------------ the two readings


def test_the_clustered_reading_is_the_stricter_of_the_two_on_the_same_rows(
    probe_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "result.json"
    module = _script()

    assert module.main(_argv(probe_file, out)) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    block = result["readings"]["pmi_prior_span"]
    opposed = next(row for row in block["pairs"] if row["second"] == "arm-b")

    assert result["items"] == 9
    assert result["clusters"] == 3
    assert result["cluster_unit"] == WORK_CLUSTER_UNIT
    # Nine discordant coins under McNemar: 2 * (1/2 ** 9) * 1 = 0.00390625, and Holm's first step is 0.05/3.
    assert float(opposed["clustered"]["item_level_mcnemar"]["p_value"]) == pytest.approx(0.00390625)
    assert opposed["separated_item_level"] is True
    # Three works: of the four assignments with the first held, only the unflipped one reaches the observed sum.
    assert float(opposed["clustered"]["cluster_sign_flip"]["p_value"]) == pytest.approx(0.25)
    assert opposed["separated_clustered"] is False
    assert block["separated_item_level"] == 2
    assert block["separated_clustered"] == 0


def test_a_cluster_bootstrap_interval_that_cannot_move_is_still_reported_as_the_point_it_is(
    probe_file: Path, tmp_path: Path
) -> None:
    """Every item of every work points the same way, so resampling works cannot move the estimate at all.

    The interval excludes zero and the randomization p-value is 0.25 on the same rows. Both are correct and they
    answer different questions — how the estimate moves when works are resampled, versus whether three works can
    be three observations — which is why the artifact prints them next to each other instead of choosing.
    """
    out = tmp_path / "result.json"
    module = _script()

    assert module.main(_argv(probe_file, out)) == 0
    opposed = next(
        row
        for row in json.loads(out.read_text(encoding="utf-8"))["readings"]["pmi_prior_span"][
            "pairs"
        ]
        if row["second"] == "arm-b"
    )

    assert opposed["clustered"]["interval"] == [1.0, 1.0]
    assert opposed["clustered"]["excludes_zero"] is True
    assert opposed["clustered"]["design_effect"] is None


def test_two_arms_that_answer_every_item_alike_are_reported_as_no_difference(
    probe_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "result.json"
    module = _script()

    assert module.main(_argv(probe_file, out)) == 0
    alike = next(
        row
        for row in json.loads(out.read_text(encoding="utf-8"))["readings"]["pmi_prior_span"][
            "pairs"
        ]
        if {row["first"], row["second"]} == {"arm-a", "arm-c"}
    )

    assert alike["clustered"]["mean_difference"] == 0.0
    assert alike["clustered"]["signal_spread"]["discordant_items"] == 0
    assert alike["clustered"]["cluster_sign_flip"]["p_value"] == 1.0
    assert alike["separated_clustered"] is False


def test_a_compacted_extract_gives_the_same_reading_as_the_file_it_was_made_from(
    probe_file: Path, tmp_path: Path
) -> None:
    """K-032's compact form is the shape the committed evidence actually has, so this path is the real one."""
    from chakaso.training.outcome_vectors import compact_probe_result

    module = _script()
    compact = tmp_path / "extract.json"
    compact.write_text(json.dumps(compact_probe_result(_probe())), encoding="utf-8")
    plain_out, compact_out = tmp_path / "plain.json", tmp_path / "compact.json"

    assert module.main(_argv(probe_file, plain_out)) == 0
    assert module.main(_argv(compact, compact_out)) == 0
    assert json.loads(compact_out.read_text(encoding="utf-8")) == json.loads(
        plain_out.read_text(encoding="utf-8")
    )


# ------------------------------------------------------------------ refusals


def test_a_probe_written_before_the_cluster_block_says_what_would_supply_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe = _probe()
    del probe["items"]["clusters"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(probe), encoding="utf-8")
    module = _script()

    assert module.main(_argv(path, tmp_path / "out.json")) == 2
    assert "--clusters-from" in capsys.readouterr().err


def test_a_reading_the_probe_never_scored_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_probe()), encoding="utf-8")
    argv = _argv(path, tmp_path / "out.json")
    argv[argv.index("pmi_prior_span")] = "pmi_next_span"

    assert _script().main(argv) == 2
    assert "pmi_next_span" in capsys.readouterr().err


def test_two_arms_whose_item_sets_differ_are_not_a_paired_comparison(tmp_path: Path) -> None:
    probe = _probe()
    outcomes = probe["models"]["arm-b"]["result"]["outcomes"]["pmi_prior_span"]
    outcomes.pop()
    path = tmp_path / "p.json"
    path.write_text(json.dumps(probe), encoding="utf-8")

    assert _script().main(_argv(path, tmp_path / "out.json")) == 2


def test_a_register_entry_without_its_quota_is_refused_rather_than_defaulted() -> None:
    module = _script()

    with pytest.raises(module.ClusterAnalysisError, match="names no quota"):
        module.quotas("prose-:400,science")


def test_a_rebuild_with_no_snapshot_is_refused_before_anything_is_loaded(tmp_path: Path) -> None:
    module = _script()
    argv = _argv(tmp_path / "p.json", tmp_path / "out.json")
    argv += ["--clusters-from", "rebuild", "--builder", "context"]
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_probe()), encoding="utf-8")

    assert module.main(argv) == 2
