"""The continuation probe runner rebuilds the experiments' held-out split, refuses a mismatched one, and floors the probe with untrained models."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.data import split_text_units  # noqa: E402
from chakaso.training.manifest import read_snapshot, write_snapshot  # noqa: E402
from chakaso.training.nn_checkpoint import save_artifact  # noqa: E402
from chakaso.training.nn_training import TransformerTrainingConfig, train_transformer  # noqa: E402
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.source_corpus import IngestedSource, build_corpus_manifest  # noqa: E402
from chakaso.training.subword import train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import TransformerConfig  # noqa: E402

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_continuations.py"
_WORDS = ["river", "stone", "field", "bird", "horse", "door", "letter", "ship", "tree", "king"]


def _text(seed: int, length: int) -> str:
    return " ".join(
        _WORDS[int(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), 16) % len(_WORDS)]
        for i in range(length)
    )


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_continuations_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sources = []
    for register in ("prose", "science"):
        for work in range(20):
            body = "\n\n".join(
                _text(work * 100 + part + (5000 if register == "science" else 0), 70)
                for part in range(8)
            )
            sources.append(
                IngestedSource(
                    slug=f"{register}-w{work}",
                    ebook_number=work + 1 + (1000 if register == "science" else 0),
                    title="t",
                    author="a",
                    licence_statement="l",
                    source_digest="0" * 64,
                    raw_characters=len(body),
                    text=body,
                )
            )
    manifest = build_corpus_manifest(
        sources,
        name="fixture",
        version="1",
        licence="test",
        source="test",
        retrieved="2026-01-01",
        max_characters_per_document=900,
    )
    path = tmp_path_factory.mktemp("snapshot") / "snapshot.json"
    write_snapshot(manifest, path, acquisition={"method": "test"})
    return path


@pytest.fixture(scope="module")
def vocabulary_file(snapshot: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    vocabulary = train_subword_vocabulary(
        [unit.text for unit in read_snapshot(snapshot).manifest.text_units()],
        target_size=300,
        min_frequency=2,
    )
    path = tmp_path_factory.mktemp("vocabulary") / "vocabulary.json"
    vocabulary.save(path)
    return path


@pytest.fixture(scope="module")
def checkpoint(
    vocabulary_file: Path, snapshot: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    from chakaso.training.subword import SubwordVocabulary

    vocabulary = SubwordVocabulary.load(vocabulary_file)
    units = read_snapshot(snapshot).manifest.text_units()
    config = TransformerConfig(
        vocab_size=vocabulary.size,
        context_length=64,
        d_model=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
    )
    examples = pack_units(units[:40], vocabulary, context_length=32).examples
    outcome = train_transformer(
        train_examples=examples,
        validation_examples=examples[:8],
        vocabulary=vocabulary,
        config=TransformerTrainingConfig(
            batch_size=4,
            max_steps=20,
            learning_rate=1e-2,
            weight_decay=0.0,
            warmup_steps=2,
            gradient_clip_norm=1.0,
            seed=3,
            eval_every=20,
            patience=99,
            min_delta=0.0,
            architecture=config,
        ),
        corpus_fingerprint="continuation-fixture",
    )
    path = tmp_path_factory.mktemp("checkpoint") / "model.pt"
    save_artifact(outcome.artifact(), path)
    return path


def _run(snapshot: Path, vocabulary_file: Path, checkpoint: Path, out: Path, *extra: str) -> int:
    return _script().main(  # type: ignore[no-any-return, attr-defined]
        [
            f"--snapshot={snapshot}",
            f"--vocabulary={vocabulary_file}",
            f"--checkpoint=model={checkpoint}",
            "--registers=prose,science",
            "--per-register=3",
            "--prefix-length=12",
            "--continuation-length=5",
            "--hard-gap=20",
            f"--out={out}",
            *extra,
        ]
    )


def test_the_runner_scores_a_checkpoint_and_reports_the_item_digest(
    tmp_path: Path, snapshot: Path, vocabulary_file: Path, checkpoint: Path
) -> None:
    out = tmp_path / "out.json"

    status = _run(snapshot, vocabulary_file, checkpoint, out, "--control", "--control-seeds=11")

    result = json.loads(out.read_text(encoding="utf-8"))
    assert status == 0
    assert result["items"]["count"] == 6
    assert len(result["items"]["digest"]) == 64
    assert {"model", "untrained-control", "untrained-control-seed-11"} <= set(result["models"])
    reading = result["models"]["model"]["result"]
    assert reading["items"] == 6
    assert set(reading["by_register"]) == {"prose", "science"}


def test_the_items_do_not_depend_on_the_model_scored(
    tmp_path: Path, snapshot: Path, vocabulary_file: Path, checkpoint: Path
) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    _run(snapshot, vocabulary_file, checkpoint, first)
    _run(snapshot, vocabulary_file, checkpoint, second, "--control")

    assert (
        json.loads(first.read_text(encoding="utf-8"))["items"]["digest"]
        == json.loads(second.read_text(encoding="utf-8"))["items"]["digest"]
    )


def test_the_items_come_only_from_the_test_side_of_the_rebuilt_split(
    snapshot: Path,
) -> None:
    script = _script()
    units = script.test_side(snapshot, split_seed=7, expected_fingerprint=None)  # type: ignore[attr-defined]
    rebuilt = split_text_units(
        read_snapshot(snapshot).manifest.text_units(),
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=7,
        whole_works=True,
    )

    assert units == rebuilt.test
    assert {unit.group for unit in rebuilt.test}.isdisjoint({unit.group for unit in rebuilt.train})


def test_a_split_that_is_not_the_one_the_experiments_used_is_refused(
    tmp_path: Path, snapshot: Path, vocabulary_file: Path, checkpoint: Path
) -> None:
    status = _run(
        snapshot,
        vocabulary_file,
        checkpoint,
        tmp_path / "out.json",
        "--expected-split-fingerprint=0000000000000000",
    )

    assert status == 1


def test_a_checkpoint_with_another_tokenizer_than_the_items_is_refused(
    tmp_path: Path, snapshot: Path, checkpoint: Path
) -> None:
    other = train_subword_vocabulary(["a b c a b c a b c"] * 4, target_size=270, min_frequency=2)
    other_file = tmp_path / "other.json"
    other.save(other_file)

    assert _run(snapshot, other_file, checkpoint, tmp_path / "out.json") == 1
