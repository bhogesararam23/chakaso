"""Tests for the released-model contract: what it must state, and what it refuses to accept.

The contract exists so a developer can tell what an artifact is, what it was trained on, and what it cannot
do — without reading an experiment record, and without trusting that the file named in a document is the file
on disk. Both halves are tested: a release that omits a required statement is refused at construction, and a
release whose bytes have been swapped is refused at load. The second property is the one a loading path can
quietly lose, because a plausible checkpoint from a different run loads perfectly well.
"""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from chakaso.models.errors import (
    CorruptedReleaseError,
    IncompatibleReleaseError,
    ModelError,
)
from chakaso.models.load import resolve_checkpoint
from chakaso.models.release import (
    RELEASE_DOCUMENT,
    ModelRelease,
    build_release,
    load_release,
    write_release,
)

CHECKPOINT_BYTES = b"\x80\x02 fake checkpoint payload for contract tests"
REPORT_DIGEST_STUB = "0" * 64


def _report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "experiment": "E2608",
        "model_family": "decoder",
        "corpus": {
            "name": "gutenberg-english-prose-v2",
            "version": "1.0.0",
            "licence": "Public domain in the United States",
            "fingerprint": "e9fc2e92f59d88ea",
            "documents": 919,
            "characters": 14_490_005,
            "split_seed": 7,
            "snapshot": {"checksum": "c" * 32},
        },
        "tokenizer": {"fingerprint": "a" * 16, "size": 4096, "merges": 3839},
        "transformer": {
            "architecture": {"d_model": 192, "num_layers": 4},
            "parameter_count": 2_167_232,
            "budget": {
                "seed": 7,
                "schedule": "cosine",
                "max_steps": 4000,
                "best_step": 3400,
                "completion": "converged",
                "batch_size": 32,
                "effective_batch_size": 64,
                "learning_rate": 0.0006,
                "warmup_steps": 200,
            },
            "test": {
                "bits_per_character": 1.71,
                "nats_per_token": 2.4,
                "perplexity": 11.2,
                "token_accuracy": 0.5,
                "positions": 480_000,
            },
        },
        "baselines_subword": {
            "unigram": {"bits_per_character": 3.6},
            "uniform": {"bits_per_character": 5.3},
        },
        "environment": {"python": "3.13.5", "torch": "2.13.0+cpu", "device": "cpu"},
    }
    return report


def _directory(tmp_path: Path, report: dict[str, Any] | None = None) -> Path:
    root = tmp_path / "release"
    root.mkdir(parents=True, exist_ok=True)
    (root / "E2608-decoder.pt").write_bytes(CHECKPOINT_BYTES)
    (root / "report.json").write_text(
        json.dumps(report if report is not None else _report()), encoding="utf-8"
    )
    return root


def _release(root: Path) -> ModelRelease:
    return build_release(
        root,
        model_id="chakaso-base",
        model_version="0.1.0",
        summary="A 2.2M-parameter decoder trained from random initialization on public-domain prose.",
        capabilities=("tokenize", "generate", "score"),
        known_limitations=(
            "Generations repeat: measured, not fixed.",
            "No grounded question is answered by this model.",
        ),
        checkpoint_name="E2608-decoder.pt",
        licence_model="Apache-2.0",
        licence_training_data="Public domain in the United States (Project Gutenberg)",
        redistribution="Weights may be redistributed with this document attached.",
        created_utc="2026-10-22T00:00:00Z",
        reproduce="chakaso experiment model --work-dir data/experiments/E2608",
    )


def test_a_release_is_built_from_the_run_report_not_from_typed_claims(tmp_path: Path) -> None:
    root = _directory(tmp_path)

    release = _release(root)

    assert release.model_id == "chakaso-base"
    assert release.family == "decoder"
    assert release.training["corpus"]["fingerprint"] == "e9fc2e92f59d88ea"
    assert release.training["held_out"]["bits_per_character"] == 1.71
    assert release.training["baselines"]["unigram_bits_per_character"] == 3.6
    assert release.checkpoint["sha256"] == sha256(CHECKPOINT_BYTES).hexdigest()
    assert release.checkpoint["parameter_count"] == 2_167_232
    assert release.software["torch"] == "2.13.0+cpu"


def test_a_release_round_trips_through_its_document(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    release = _release(root)

    path = write_release(release, root)
    loaded = load_release(root)

    assert path.name == RELEASE_DOCUMENT
    assert loaded == release
    assert loaded.checksum == release.compute_checksum()


def test_a_release_document_is_not_written_over_unasked(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    release = _release(root)
    write_release(release, root)

    with pytest.raises(IncompatibleReleaseError, match="allow_overwrite"):
        write_release(release, root)

    write_release(release, root, allow_overwrite=True)
    assert load_release(root) == release


def test_writing_a_release_requires_its_checkpoint_to_be_present(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    release = _release(root)
    (root / "E2608-decoder.pt").unlink()

    with pytest.raises(IncompatibleReleaseError, match="does not exist"):
        write_release(release, root)


def test_a_swapped_checkpoint_is_refused_before_it_is_loaded(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    write_release(_release(root), root)
    (root / "E2608-decoder.pt").write_bytes(b"\x80\x02 someone else's weights entirely")

    with pytest.raises(IncompatibleReleaseError, match="hashes to"):
        load_release(root)


def test_a_missing_checkpoint_file_is_named_by_the_verification(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    write_release(_release(root), root)
    (root / "E2608-decoder.pt").unlink()

    with pytest.raises(IncompatibleReleaseError, match="not in"):
        load_release(root)


def test_a_size_that_disagrees_with_the_contract_is_refused(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    release = _release(root)
    tampered = deepcopy(release.document())
    tampered["checkpoint"]["size_bytes"] = 999_999
    document = dict(tampered)
    content = {key: value for key, value in document.items() if key != "checksum"}
    document["checksum"] = sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (root / RELEASE_DOCUMENT).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(IncompatibleReleaseError, match="size on disk"):
        load_release(root)


def test_an_altered_release_document_fails_its_own_checksum(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    write_release(_release(root), root)
    document = json.loads((root / RELEASE_DOCUMENT).read_text(encoding="utf-8"))
    document["training"]["held_out"]["bits_per_character"] = 0.42
    (root / RELEASE_DOCUMENT).write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(CorruptedReleaseError, match="does not match its content"):
        load_release(root)


def test_a_document_without_a_checksum_is_not_a_release(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    write_release(_release(root), root)
    document = json.loads((root / RELEASE_DOCUMENT).read_text(encoding="utf-8"))
    document.pop("checksum")
    (root / RELEASE_DOCUMENT).write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(CorruptedReleaseError, match="must carry its own checksum"):
        load_release(root)


def test_a_directory_holding_no_document_reports_that_clearly(tmp_path: Path) -> None:
    with pytest.raises(CorruptedReleaseError, match="holds no release document"):
        load_release(tmp_path / "nothing-here")


def test_an_unknown_release_schema_is_refused(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    release = _release(root)
    document = release.to_document()
    document["schema_version"] = "chakaso-model-release/9"

    with pytest.raises(IncompatibleReleaseError, match="unsupported release schema"):
        ModelRelease.from_document(document)


def test_a_release_must_state_its_limitations(tmp_path: Path) -> None:
    root = _directory(tmp_path)

    with pytest.raises(IncompatibleReleaseError, match="known limitations"):
        build_release(
            root,
            model_id="m",
            model_version="1",
            summary="s",
            capabilities=("generate",),
            known_limitations=(),
            checkpoint_name="E2608-decoder.pt",
            licence_model="Apache-2.0",
            licence_training_data="PD",
            redistribution="yes",
            created_utc="2026-10-22T00:00:00Z",
            reproduce="chakaso …",
        )


def test_a_release_must_state_every_licence_field(tmp_path: Path) -> None:
    release = _release(_directory(tmp_path))
    incomplete = {**release.document(), "licence": {"model": "Apache-2.0"}}
    document = dict(incomplete)
    content = {key: value for key, value in document.items() if key != "checksum"}
    document["checksum"] = sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(IncompatibleReleaseError, match="licence block is missing"):
        ModelRelease.from_document(document)


def test_a_release_must_name_a_capability(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    document = _release(root).document()
    document["capabilities"] = []
    content = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    document["checksum"] = sha256(content).hexdigest()

    with pytest.raises(IncompatibleReleaseError, match="at least one capability"):
        ModelRelease.from_document(document)


def test_provenance_the_report_does_not_carry_cannot_be_invented(tmp_path: Path) -> None:
    report = _report()
    del report["transformer"]["budget"]["completion"]
    root = _directory(tmp_path, report)

    with pytest.raises(IncompatibleReleaseError, match="cannot state provenance"):
        build_release(
            root,
            model_id="m",
            model_version="1",
            summary="s",
            capabilities=("generate",),
            known_limitations=("loops",),
            checkpoint_name="E2608-decoder.pt",
            licence_model="Apache-2.0",
            licence_training_data="PD",
            redistribution="yes",
            created_utc="2026-10-22T00:00:00Z",
            reproduce="chakaso …",
        )


def test_a_release_directory_resolves_to_the_checkpoint_it_declares(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    write_release(_release(root), root)

    resolved = resolve_checkpoint(root)

    assert resolved == root / "E2608-decoder.pt"
    assert resolve_checkpoint(resolved) == resolved


def test_loading_a_model_through_a_release_directory_reaches_the_adapter(
    tmp_path: Path,
) -> None:
    from chakaso.models import load_model

    root = _directory(tmp_path)
    write_release(_release(root), root)

    # The payload is not a real artifact, so the failure must come from the checkpoint layer naming the
    # format problem — never from the release layer, and never a silent fallback to the double.
    with pytest.raises(ModelError):
        load_model(root)


def test_a_release_document_is_human_readable_json(tmp_path: Path) -> None:
    root = _directory(tmp_path)
    write_release(_release(root), root)

    text = (root / RELEASE_DOCUMENT).read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert "chakaso-model-release/1" in text
    assert "known_limitations" in text
    assert "Generations repeat" in text
