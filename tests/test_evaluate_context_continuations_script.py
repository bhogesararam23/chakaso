"""The same-region probe runner: one held-out split shared with v1, a digest of its own, and a real floor.

The fixture trains a tiny model on a synthetic corpus so the runner is exercised end to end, and scores a
randomly initialised network of the same architecture beside it. That second score is the instrument's own
claim, measured on an actual random initialisation rather than a scripted scorer: a model that has learned
nothing must land on one half for every distractor kind, because the pointwise-mutual-information subtraction
removes each candidate's own predictability — including the far lower predictability of a permuted span, which
is the case v1 never had to face.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.data import split_text_units  # noqa: E402
from chakaso.training.item_clusters import (  # noqa: E402
    WORK_CLUSTER_UNIT,
    keys_for_items,
    load_cluster_block,
)
from chakaso.training.manifest import read_snapshot, write_snapshot  # noqa: E402
from chakaso.training.nn_checkpoint import save_artifact  # noqa: E402
from chakaso.training.nn_training import (  # noqa: E402
    TransformerTrainingConfig,
    train_transformer,
)
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.source_corpus import IngestedSource, build_corpus_manifest  # noqa: E402
from chakaso.training.subword import SubwordVocabulary, train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import TransformerConfig  # noqa: E402

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_context_continuations.py"
_WORDS = ["river", "stone", "field", "bird", "horse", "door", "letter", "ship", "tree", "king"]
_KINDS = ("prior_span", "next_span", "permuted", "reversed")
#: The identity E2623 cites for its own item set. A different construction must not produce it.
_V1_DIGEST = "9557a327bfe61e2d747a826777b748ebe285b2dd352ba6a769aca0262cfacdcd"


def _text(seed: int, length: int) -> str:
    return " ".join(
        _WORDS[int(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), 16) % len(_WORDS)]
        for i in range(length)
    )


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_context_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sources = []
    for register in ("prose", "science"):
        for work in range(30):
            body = "\n\n".join(
                _text(work * 100 + part + (5000 if register == "science" else 0), 90)
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
        max_characters_per_document=1200,
    )
    path = tmp_path_factory.mktemp("ctx-snapshot") / "snapshot.json"
    write_snapshot(manifest, path, acquisition={"method": "test"})
    return path


@pytest.fixture(scope="module")
def vocabulary_file(snapshot: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    vocabulary = train_subword_vocabulary(
        [unit.text for unit in read_snapshot(snapshot).manifest.text_units()],
        target_size=300,
        min_frequency=2,
    )
    path = tmp_path_factory.mktemp("ctx-vocabulary") / "vocabulary.json"
    vocabulary.save(path)
    return path


@pytest.fixture(scope="module")
def checkpoint(
    vocabulary_file: Path, snapshot: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    vocabulary = SubwordVocabulary.load(vocabulary_file)
    units = read_snapshot(snapshot).manifest.text_units()
    config = TransformerConfig(
        vocab_size=vocabulary.size,
        context_length=64,
        d_model=16,
        num_layers=1,
        num_heads=2,
        feedforward_dim=32,
    )
    examples = pack_units(units[:60], vocabulary, context_length=48).examples
    outcome = train_transformer(
        train_examples=examples,
        validation_examples=examples[:8],
        vocabulary=vocabulary,
        config=TransformerTrainingConfig(
            batch_size=4,
            max_steps=60,
            learning_rate=1e-2,
            weight_decay=0.0,
            warmup_steps=2,
            gradient_clip_norm=1.0,
            seed=3,
            eval_every=60,
            patience=99,
            min_delta=0.0,
            architecture=config,
        ),
        corpus_fingerprint="context-fixture",
    )
    path = tmp_path_factory.mktemp("ctx-checkpoint") / "model.pt"
    save_artifact(outcome.artifact(), path)
    return path


def _argv(snapshot: Path, vocabulary_file: Path, checkpoint: Path, out: Path) -> list[str]:
    return [
        "--snapshot",
        str(snapshot),
        "--vocabulary",
        str(vocabulary_file),
        "--registers",
        "prose-:6,science-:4",
        "--max-per-work",
        "5",
        "--prefix-length",
        "16",
        "--continuation-length",
        "6",
        "--checkpoint",
        f"model={checkpoint}",
        "--control",
        "--control-seeds",
        "11",
        "--out",
        str(out),
    ]


def _run(snapshot: Path, vocabulary_file: Path, checkpoint: Path, tmp_path: Path) -> dict[str, Any]:
    out = tmp_path / "context.json"
    assert _script().main(_argv(snapshot, vocabulary_file, checkpoint, out)) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_runner_scores_every_kind_for_every_model_and_keeps_the_per_item_record(
    snapshot: Path, vocabulary_file: Path, checkpoint: Path, tmp_path: Path
) -> None:
    result = _run(snapshot, vocabulary_file, checkpoint, tmp_path)

    assert result["items"]["count"] == 10
    assert result["items"]["kinds"] == list(_KINDS)
    assert len(result["items"]["digest"]) == 64
    assert result["items"]["test_documents"] > 0
    for name in ("model", "untrained-control", "untrained-control-seed-11"):
        readings = result["models"][name]["result"]
        assert set(readings["pmi"]) == set(_KINDS)
        assert set(readings["conditional"]) == set(_KINDS)
        assert set(readings["outcomes"]) == {f"pmi_{kind}" for kind in _KINDS}
        for kind in _KINDS:
            assert len(readings["outcomes"][f"pmi_{kind}"]) == 10, kind


def test_the_runner_records_which_work_every_scored_item_was_cut_from(
    snapshot: Path, vocabulary_file: Path, checkpoint: Path, tmp_path: Path
) -> None:
    """K-035: the clustering a record claims has to be readable out of the artifact it wrote.

    Every reading of one probe result asks the same items in the same order, so one listing serves all of them; what
    is checked here is that each reading's own per-item rows align to it, which is the property a paired clustered
    statistic depends on.
    """
    result = _run(snapshot, vocabulary_file, checkpoint, tmp_path)
    listing = load_cluster_block(result["items"]["clusters"])

    assert listing.unit == WORK_CLUSTER_UNIT
    assert len(listing.rows) == result["items"]["count"] == 10
    assert len(listing.clusters) < len(listing.rows), "ten items from 60 works is not a cluster"
    for name, entry in result["models"].items():
        for reading, rows in entry["result"]["outcomes"].items():
            keys = keys_for_items(listing, [str(row[0]) for row in rows])
            assert all(re.fullmatch(r"(prose|science)-w\d+", key) for key in keys), (name, reading)


def test_the_cluster_listing_is_the_same_pairing_on_every_run(
    snapshot: Path, vocabulary_file: Path, checkpoint: Path, tmp_path: Path
) -> None:
    first = _run(snapshot, vocabulary_file, checkpoint, tmp_path)["items"]["clusters"]
    second = _run(snapshot, vocabulary_file, checkpoint, tmp_path / "b")["items"]["clusters"]

    assert first["digest"] == second["digest"]
    assert first["rows"] == second["rows"]


def test_an_untrained_networks_floor_is_reported_per_kind_and_covers_chance(
    snapshot: Path,
    vocabulary_file: Path,
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    """A ten-word synthetic lexicon is not the corpus.

    With a handful of identifiers every span overlaps the passage heavily and a random initialisation can copy
    it, so the point estimate over ten items is not a floor claim. What this proves is that the runner produces
    a floor per kind, with an interval wide enough at this size not to pretend to a difference. Whether the
    floor is *at* chance on real held-out English is E2627's measurement over seven initialisations.
    """
    result = _run(snapshot, vocabulary_file, checkpoint, tmp_path)

    control = result["models"]["untrained-control"]["result"]["pmi"]
    for kind in _KINDS:
        assert control[kind]["interval_95"][0] <= 0.5 <= control[kind]["interval_95"][1], kind
        assert control[kind]["items"] == 10, kind


def test_the_digest_is_stable_across_runs_and_is_not_the_one_v1_cites(
    snapshot: Path, vocabulary_file: Path, checkpoint: Path, tmp_path: Path
) -> None:
    first = _run(snapshot, vocabulary_file, checkpoint, tmp_path / "a")
    second = _run(snapshot, vocabulary_file, checkpoint, tmp_path / "b")

    assert first["items"]["digest"] == second["items"]["digest"]
    assert first["items"]["digest"] != _V1_DIGEST


def test_a_split_that_is_not_the_experiments_one_is_refused(
    snapshot: Path,
    vocabulary_file: Path,
    checkpoint: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = _argv(snapshot, vocabulary_file, checkpoint, tmp_path / "out.json")
    argv += ["--expected-split-fingerprint", "0" * 16]

    assert _script().main(argv) == 1
    assert "these are not the held-out documents" in capsys.readouterr().err


def test_the_rebuilt_split_matches_the_one_the_experiments_used(
    snapshot: Path,
    vocabulary_file: Path,
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    """The runner must be able to check the fingerprint the records name, not only reject a wrong one."""
    fingerprint = split_text_units(
        read_snapshot(snapshot).manifest.text_units(),
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=7,
        whole_works=True,
    ).fingerprint
    out = tmp_path / "checked.json"
    argv = _argv(snapshot, vocabulary_file, checkpoint, out)
    argv += ["--expected-split-fingerprint", fingerprint]

    assert _script().main(argv) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["items"]["split_fingerprint"] == fingerprint


def test_a_declared_item_digest_is_enforced_before_a_checkpoint_is_loaded(
    snapshot: Path,
    vocabulary_file: Path,
    checkpoint: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The rule E2627 fixed by hand: a record declares the item set, and a run may not score another one.

    Both halves are checked, because a gate that always refuses is as useless as one that never does — and the refusal
    has to happen before the expensive part, so a mismatched run leaves no output file behind.
    """
    agreed = tmp_path / "agreed.json"
    digest = _run(snapshot, vocabulary_file, checkpoint, tmp_path)["items"]["digest"]
    argv = _argv(snapshot, vocabulary_file, checkpoint, agreed)

    assert _script().main([*argv, "--expected-items-digest", digest]) == 0
    assert agreed.is_file()

    refused = tmp_path / "refused.json"
    assert (
        _script().main(
            [
                *_argv(snapshot, vocabulary_file, checkpoint, refused),
                "--expected-items-digest",
                "0" * 64,
            ]
        )
        == 1
    )
    assert "item set other than the one it declared" in capsys.readouterr().err
    assert not refused.exists()


def test_a_checkpoint_cut_with_another_vocabulary_is_refused_by_name(
    snapshot: Path,
    checkpoint: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    foreign = train_subword_vocabulary(
        ["gliff glaff gloom plink plonk sorf sorf sorf treppe treppe"],
        target_size=260,
        min_frequency=1,
    )
    path = tmp_path / "foreign-vocabulary.json"
    foreign.save(path)
    argv = [
        "--snapshot",
        str(snapshot),
        "--vocabulary",
        str(path),
        "--registers",
        "prose-:2",
        "--prefix-length",
        "16",
        "--continuation-length",
        "6",
        "--checkpoint",
        f"model={checkpoint}",
    ]

    assert _script().main(argv) == 1
    assert "the items were cut with" in capsys.readouterr().err
