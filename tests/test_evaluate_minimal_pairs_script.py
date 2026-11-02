"""The probe runner scores artifacts with their own tokenizers and gives the instrument a floor."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.data import TextUnit  # noqa: E402
from chakaso.training.minimal_pairs import MinimalPair, PairSet  # noqa: E402
from chakaso.training.nn_checkpoint import load_artifact, save_artifact  # noqa: E402
from chakaso.training.nn_training import TransformerTrainingConfig, train_transformer  # noqa: E402
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.subword import train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import TransformerConfig  # noqa: E402

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_minimal_pairs.py"
_GOOD = "the quick red fox ran over the dry sand"
_PAIRS = (
    MinimalPair("p0", "order", _GOOD, " ".join(reversed(_GOOD.split()))),
    MinimalPair("p1", "order", "the quiet owl slept", "owl the slept quiet"),
)
_VOCABULARY = train_subword_vocabulary(
    [_GOOD, "the quiet owl slept", "owl the slept quiet"], target_size=280, min_frequency=2
)
_CONFIG = TransformerConfig(
    vocab_size=_VOCABULARY.size,
    context_length=64,
    d_model=8,
    num_layers=1,
    num_heads=2,
    feedforward_dim=16,
)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_minimal_pairs_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _trained_artifact(tmp_path_factory.mktemp("probe") / "trained.pt")


def _trained_artifact(path: Path) -> Path:
    units = tuple(TextUnit(identifier=f"u{i}", text=_GOOD) for i in range(40)) + tuple(
        TextUnit(identifier=f"o{i}", text="the quiet owl slept") for i in range(40)
    )
    examples = pack_units(units, _VOCABULARY, context_length=16).examples
    config = TransformerTrainingConfig(
        batch_size=4,
        max_steps=250,
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup_steps=2,
        gradient_clip_norm=1.0,
        seed=3,
        eval_every=250,
        patience=99,
        min_delta=0.0,
        architecture=_CONFIG,
    )
    outcome = train_transformer(
        train_examples=examples,
        validation_examples=examples,
        vocabulary=_VOCABULARY,
        config=config,
        corpus_fingerprint="probe-fixture",
    )
    save_artifact(outcome.artifact(), path)
    return path


def _pair_file(path: Path) -> Path:
    pair_set = PairSet(
        name="fixture", version="1", licence="test", provenance="written for tests", pairs=_PAIRS
    )
    path.write_text(json.dumps(pair_set.as_dict()), encoding="utf-8")
    return path


def test_a_trained_artifact_beats_its_untrained_control_on_the_probe(
    tmp_path: Path, trained: Path
) -> None:
    checkpoint = trained
    out = tmp_path / "out.json"

    status = _script().main(  # type: ignore[attr-defined]
        [
            f"--pairs={_pair_file(tmp_path / 'pairs.json')}",
            f"--checkpoint=trained={checkpoint}",
            "--control",
            f"--out={out}",
        ]
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert status == 0
    assert result["models"]["trained"]["result"]["accuracy"] == 1.0
    assert result["models"]["untrained-control"]["result"]["accuracy"] < 1.0
    assert result["models"]["trained"]["checkpoint_checksum"]
    # The token-count reading is reported with the model's own tokenizer, beside the character-length one.
    assert result["models"]["trained"]["result"]["token_count_only_accuracy"] is not None
    assert result["pair_set"]["name"] == "fixture"


def test_the_control_is_reproducible(trained: Path) -> None:
    artifact = load_artifact(trained)
    script = _script()

    first = script.untrained_control(artifact)  # type: ignore[attr-defined]
    second = script.untrained_control(artifact)  # type: ignore[attr-defined]

    for (name, left), (_, right) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=True
    ):
        assert torch.equal(left, right), name


def test_the_contamination_screen_reports_pairs_found_in_a_snapshot(
    tmp_path: Path, trained: Path
) -> None:
    checkpoint = trained
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "documents": [
                    {"path": "a", "text": "Long ago the quick red fox ran over the dry sand."}
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.json"

    status = _script().main(  # type: ignore[attr-defined]
        [
            f"--pairs={_pair_file(tmp_path / 'pairs.json')}",
            f"--checkpoint=trained={checkpoint}",
            f"--snapshot={snapshot}",
            "--ngram=5",
            f"--out={out}",
        ]
    )

    contamination = json.loads(out.read_text(encoding="utf-8"))["contamination"]
    assert status == 0
    assert contamination["flagged_ids"] == ["p0"]


def test_a_missing_checkpoint_is_an_error_not_a_crash(tmp_path: Path) -> None:
    status = _script().main(  # type: ignore[attr-defined]
        [f"--pairs={_pair_file(tmp_path / 'p.json')}", f"--checkpoint=x={tmp_path / 'absent.pt'}"]
    )

    assert status == 1


def test_extra_control_seeds_add_untrained_models_that_differ_from_one_another(
    tmp_path: Path, trained: Path
) -> None:
    out = tmp_path / "out.json"

    status = _script().main(  # type: ignore[attr-defined]
        [
            f"--pairs={_pair_file(tmp_path / 'pairs.json')}",
            f"--checkpoint=trained={trained}",
            "--control",
            "--control-seeds=11,13",
            f"--out={out}",
        ]
    )

    models = json.loads(out.read_text(encoding="utf-8"))["models"]
    assert status == 0
    assert {"untrained-control", "untrained-control-seed-11", "untrained-control-seed-13"} <= set(
        models
    )
    assert models["untrained-control-seed-11"]["seed"] == 11


def test_a_different_seed_gives_a_different_untrained_model(trained: Path) -> None:
    artifact = load_artifact(trained)
    script = _script()

    first = script.untrained_control(artifact, 11)  # type: ignore[attr-defined]
    second = script.untrained_control(artifact, 13)  # type: ignore[attr-defined]

    assert any(
        not torch.equal(left, right)
        for (_, left), (_, right) in zip(
            first.state_dict().items(), second.state_dict().items(), strict=True
        )
    )
