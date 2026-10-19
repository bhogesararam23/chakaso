"""Tests for the checkpoint's exact round-trip and its integrity/compatibility refusals.

A checkpoint must load back bit-for-bit and must loudly reject a corrupted or foreign file rather than
half-reading it. Numpy-gated (``[train]`` path).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

numpy = pytest.importorskip("numpy")

from chakaso.training.checkpoint import Checkpoint, checkpoint_filename  # noqa: E402
from chakaso.training.config import TrainingConfig  # noqa: E402
from chakaso.training.errors import (  # noqa: E402
    CorruptedCheckpointError,
    IncompatibleCheckpointError,
)
from chakaso.training.model import CharCausalMLP  # noqa: E402
from chakaso.training.tokenizer import CharVocabulary  # noqa: E402


def _vocab() -> CharVocabulary:
    # A real, self-consistent vocabulary: 5 characters + the reserved unknown id = size 6, and a
    # fingerprint the checkpoint loader will validate against.
    return CharVocabulary(("abcde",))


def _config() -> TrainingConfig:
    return TrainingConfig(
        context_size=3, embed_dim=4, hidden_dim=5, learning_rate=0.05, seed=7, steps=2, eval_every=1
    )


def _checkpoint(seed: int = 7) -> Checkpoint:
    vocab = _vocab()
    model = CharCausalMLP(vocab_size=vocab.size, context_size=3, embed_dim=4, hidden_dim=5)
    params = model.initialize(numpy.random.default_rng(seed))
    return Checkpoint(
        params=params,
        config=_config(),
        vocab_size=vocab.size,
        vocab_fingerprint=vocab.fingerprint,
        dataset_digest="0123456789abcdef",
        vocab_chars=vocab.chars,
    )


def _restamp(document: dict[str, Any]) -> dict[str, Any]:
    """Recompute the checksum over a tampered document's content, leaving only semantic faults."""
    content = {k: v for k, v in document.items() if k != "checksum"}
    document["checksum"] = hashlib.sha256(
        Checkpoint._canonical(content).encode("utf-8")
    ).hexdigest()
    return document


def test_the_conventional_filename_is_derived_from_the_experiment_id() -> None:
    assert checkpoint_filename("E2601") == "E2601-checkpoint.json"


def test_a_saved_checkpoint_loads_back_bit_for_bit(tmp_path: Path) -> None:
    original = _checkpoint()
    path = tmp_path / checkpoint_filename("E2601")
    original.save(path)

    loaded = Checkpoint.load(path)

    for (name, before), (_, after) in zip(
        original.params.items(), loaded.params.items(), strict=True
    ):
        assert numpy.array_equal(before, after), f"{name} changed across a round trip"
    assert loaded.config == original.config
    assert loaded.vocab_size == original.vocab_size
    assert loaded.vocab_fingerprint == original.vocab_fingerprint
    assert loaded.vocab_chars == original.vocab_chars
    assert loaded.dataset_digest == original.dataset_digest
    # The rebuilt vocabulary encodes/decodes the known characters identically to the original.
    rebuilt = loaded.vocabulary()
    assert rebuilt.decode(rebuilt.encode("abcde")) == "abcde"
    assert rebuilt.chars == original.vocab_chars


def test_the_fingerprint_is_stable_and_changes_with_a_parameter() -> None:
    first = _checkpoint()
    # The same parameters always produce the same fingerprint, and it is exactly the embedded checksum.
    assert first.fingerprint() == _checkpoint().fingerprint()
    assert (
        Checkpoint.from_document(first.to_document()).fingerprint()
        == first.to_document()["checksum"]
    )

    # A different parameter draw produces a different fingerprint.
    other = _checkpoint(seed=999)
    assert first.fingerprint() != other.fingerprint()


def test_a_fingerprint_validated_vocabulary_is_accepted() -> None:
    # The checkpoint's stored characters must hash to its stored fingerprint; a real one round-trips.
    loaded = Checkpoint.from_document(_checkpoint().to_document())
    assert loaded.vocabulary().fingerprint == loaded.vocab_fingerprint


def test_vocabulary_characters_that_do_not_match_the_fingerprint_are_incompatible() -> None:
    document = _checkpoint().to_document()
    # Replace the characters with a different set but keep the (now-wrong) fingerprint; the checksum is
    # re-stamped so the only remaining fault is the vocabulary/fingerprint disagreement.
    document["vocab_chars"] = ["x", "y", "z", "w", "v"]
    with pytest.raises(IncompatibleCheckpointError, match="do not hash to"):
        Checkpoint.from_document(_restamp(document))


def test_vocabulary_size_disagreeing_with_the_characters_is_incompatible() -> None:
    document = _checkpoint().to_document()
    document["vocab_chars"] = ["a", "b", "c"]  # 3 chars, but vocab_size still says 6
    with pytest.raises(IncompatibleCheckpointError, match="vocab_size"):
        Checkpoint.from_document(_restamp(document))


def test_altering_an_array_without_fixing_the_checksum_is_corruption(tmp_path: Path) -> None:
    original = _checkpoint()
    path = tmp_path / "ckpt.json"
    original.save(path)

    document = json.loads(path.read_text(encoding="utf-8"))
    document["arrays"]["hidden_bias"][0] += 1.0  # tamper; checksum no longer matches
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CorruptedCheckpointError, match="checksum mismatch"):
        Checkpoint.load(path)


def test_a_document_without_a_checksum_is_corrupted() -> None:
    document = _checkpoint().to_document()
    del document["checksum"]
    with pytest.raises(CorruptedCheckpointError, match="no checksum"):
        Checkpoint.from_document(document)


def test_a_file_that_is_not_json_is_corrupted(tmp_path: Path) -> None:
    path = tmp_path / "junk.json"
    path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(CorruptedCheckpointError, match="not valid JSON"):
        Checkpoint.load(path)


def test_a_top_level_json_array_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "arr.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CorruptedCheckpointError, match="must be a JSON object"):
        Checkpoint.load(path)


def test_an_unknown_schema_version_raises_incompatible() -> None:
    document = _checkpoint().to_document()
    document["schema_version"] = "chakaso-checkpoint/99"
    with pytest.raises(IncompatibleCheckpointError, match="unsupported checkpoint schema"):
        Checkpoint.from_document(_restamp(document))


def test_an_array_shape_disagreeing_with_the_config_is_incompatible() -> None:
    document = _checkpoint().to_document()
    # 99 embedding rows when the declared vocab_size is 6: the only fault left after re-stamping is
    # the shape disagreement, not the integrity check.
    document["arrays"]["embedding"] = [[0.0] * 4] * 99
    with pytest.raises(IncompatibleCheckpointError, match="expected"):
        Checkpoint.from_document(_restamp(document))


def test_a_missing_array_is_incompatible() -> None:
    document = _checkpoint().to_document()
    del document["arrays"]["output_weight"]
    with pytest.raises(IncompatibleCheckpointError, match="missing the"):
        Checkpoint.from_document(_restamp(document))
