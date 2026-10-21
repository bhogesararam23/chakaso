"""Torch-gated tests for the versioned Transformer artifact format.

The property under test is that an artifact loads *as the model it was saved as*, or not at all. Every
refusal path here is a way a checkpoint could otherwise produce a plausible-looking wrong model: a tensor
swapped for another shape, a tokenizer that no longer matches the weights, a configuration field this code
would silently ignore, or a file that tries to carry a Python object.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.errors import (  # noqa: E402
    CheckpointError,
    CorruptedCheckpointError,
    IncompatibleCheckpointError,
)
from chakaso.training.nn_checkpoint import (  # noqa: E402
    ModelArtifact,
    checkpoint_filename,
    describe_environment,
    load_artifact,
    save_artifact,
)
from chakaso.training.subword import train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import CausalTransformer, TransformerConfig  # noqa: E402

CONFIG = TransformerConfig(
    vocab_size=260,
    context_length=6,
    d_model=8,
    num_layers=1,
    num_heads=2,
    feedforward_dim=16,
)
VOCABULARY = train_subword_vocabulary(
    ["the quick red fox ran over the dry sand and slept"] * 3, target_size=260, min_frequency=2
)


def _artifact(step: int = 4, *, corpus: str = "abc123") -> ModelArtifact:
    torch.manual_seed(1)
    model = CausalTransformer(CONFIG)
    return ModelArtifact(
        config=CONFIG,
        state={name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        vocabulary=VOCABULARY,
        corpus_fingerprint=corpus,
        run_metadata={"note": "test"},
        step=step,
        environment=describe_environment(),
    )


class HostilePayload:
    """A picklable-by-default object that a weights-only loader must refuse."""

    value = 1


def _raw(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[no-any-return]


def _write_raw(path: Path, payload: dict[str, object]) -> None:
    torch.save(payload, path)


# ------------------------------------------------------------------ round trip


def test_an_artifact_round_trips_bit_for_bit(tmp_path: Path) -> None:
    artifact = _artifact()
    target = tmp_path / checkpoint_filename("TEST")

    save_artifact(artifact, target)
    loaded = load_artifact(target)

    assert loaded.config == artifact.config
    assert loaded.vocabulary.fingerprint == artifact.vocabulary.fingerprint
    assert loaded.vocabulary.merges == artifact.vocabulary.merges
    assert loaded.corpus_fingerprint == artifact.corpus_fingerprint
    assert loaded.step == artifact.step
    assert loaded.run_metadata == artifact.run_metadata
    assert sorted(loaded.state) == sorted(artifact.state)
    for name, tensor in artifact.state.items():
        assert torch.equal(loaded.state[name], tensor), name


def test_a_reloaded_artifact_builds_an_identical_model(tmp_path: Path) -> None:
    artifact = _artifact()
    target = tmp_path / "artifact.pt"
    save_artifact(artifact, target)
    inputs = torch.tensor([[1, 2, 3, 4]])

    original = artifact.to_model()
    reloaded = load_artifact(target).to_model()

    with torch.no_grad():
        assert torch.equal(original(inputs)[0], reloaded(inputs)[0])


def test_saving_refuses_to_overwrite_an_existing_artifact_silently(tmp_path: Path) -> None:
    target = tmp_path / "artifact.pt"
    save_artifact(_artifact(), target)
    before = target.read_bytes()

    with pytest.raises(CheckpointError, match="already holds an artifact"):
        save_artifact(_artifact(step=99), target)
    assert target.read_bytes() == before

    save_artifact(_artifact(step=99), target, allow_overwrite=True)
    assert load_artifact(target).step == 99


def test_saving_creates_a_missing_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "data" / "runs" / "artifact.pt"

    save_artifact(_artifact(), target)

    assert target.is_file()


def test_an_interrupted_artifact_write_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that dies mid-save must not leave a file that reads like an artifact.

    The artifact is what a record cites, and a half-written one is worse than a missing one: it has the
    right name, the right extension and a plausible size, and only verification would expose it.
    """
    import torch

    def _interrupt(*_ignored: object, **_also_ignored: object) -> None:
        failure = "device out of space"
        raise OSError(failure)

    target = tmp_path / "artifact.pt"
    monkeypatch.setattr(torch, "save", _interrupt)

    with pytest.raises(OSError, match="out of space"):
        save_artifact(_artifact(), target)

    assert not target.exists()
    assert [path.name for path in tmp_path.iterdir()] == []


def test_an_artifact_reports_the_architectures_parameter_count() -> None:
    artifact = _artifact()
    model = artifact.to_model()

    assert artifact.parameter_count == artifact.config.parameter_count
    assert model.parameter_count == artifact.config.parameter_count
    # The stored tensors hold the tied embedding matrix twice, which is why the count comes from the
    # architecture rather than from summing what is on disk.
    assert sum(int(tensor.numel()) for tensor in artifact.state.values()) > artifact.parameter_count


# ------------------------------------------------------------------ integrity


def test_a_swapped_tensor_is_caught_by_the_checksum(tmp_path: Path) -> None:
    target = tmp_path / "artifact.pt"
    artifact = _artifact()
    save_artifact(artifact, target)
    payload = _raw(target)
    tensors = dict(payload["tensors"])  # type: ignore[arg-type]
    name = sorted(tensors)[0]
    tensors[name] = tensors[name] + 1.0
    _write_raw(target, {**payload, "tensors": tensors})

    with pytest.raises(CorruptedCheckpointError, match="checksum does not match"):
        load_artifact(target)


def test_edited_metadata_is_caught_by_the_checksum(tmp_path: Path) -> None:
    target = tmp_path / "artifact.pt"
    save_artifact(_artifact(), target)
    payload = _raw(target)
    metadata = json.loads(str(payload["metadata"]))
    metadata["corpus_fingerprint"] = "someone-elses-run"
    _write_raw(target, {**payload, "metadata": json.dumps(metadata, sort_keys=True)})

    with pytest.raises(CorruptedCheckpointError, match="checksum does not match"):
        load_artifact(target)


def test_a_file_without_a_checksum_block_is_not_an_artifact(tmp_path: Path) -> None:
    target = tmp_path / "not-artifact.pt"
    _write_raw(target, {"schema_version": "chakaso-transformer-checkpoint/1", "tensors": {}})

    with pytest.raises(IncompatibleCheckpointError, match="not a Chakaso artifact"):
        load_artifact(target)


def test_an_unknown_schema_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "future.pt"
    artifact = _artifact()
    save_artifact(artifact, target)
    payload = _raw(target)
    _write_raw(target, {**payload, "schema_version": "chakaso-transformer-checkpoint/9"})

    with pytest.raises(IncompatibleCheckpointError, match="unsupported artifact schema"):
        load_artifact(target)


def test_a_file_that_is_not_a_chakaso_payload_is_reported_as_unreadable(tmp_path: Path) -> None:
    target = tmp_path / "random.pt"
    _write_raw(target, {"anything": torch.zeros(2)})

    with pytest.raises(IncompatibleCheckpointError, match="not a Chakaso artifact"):
        load_artifact(target)


def test_a_checkpoint_cannot_carry_a_python_object(tmp_path: Path) -> None:
    # The security rule: the loader uses weights_only=True, so a file that tries to reconstruct a class
    # instance fails at the boundary instead of executing the pickle.
    target = tmp_path / "hostile.pt"
    torch.save({"payload": HostilePayload()}, target)

    with pytest.raises(CorruptedCheckpointError, match="could not be loaded"):
        load_artifact(target)


# ------------------------------------------------------------------ incompatibilities


def test_a_tensor_shape_that_disagrees_with_the_declared_architecture_is_refused(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    tensors = dict(artifact.state)
    tensors["token_embedding.weight"] = torch.zeros(7, 8)
    broken = ModelArtifact(
        config=artifact.config,
        state=tensors,
        vocabulary=artifact.vocabulary,
        corpus_fingerprint=artifact.corpus_fingerprint,
    )

    with pytest.raises(IncompatibleCheckpointError, match="do not match the architecture"):
        broken.to_model()
    with pytest.raises(IncompatibleCheckpointError, match="do not match the architecture"):
        save_artifact(broken, tmp_path / "never-written.pt")


def test_a_missing_tensor_is_refused() -> None:
    artifact = _artifact()
    tensors = dict(artifact.state)
    tensors.pop("final_norm.weight")

    with pytest.raises(IncompatibleCheckpointError, match="missing tensors"):
        ModelArtifact(
            config=artifact.config,
            state=tensors,
            vocabulary=artifact.vocabulary,
            corpus_fingerprint=artifact.corpus_fingerprint,
        ).to_model()


def test_a_vocabulary_that_does_not_hash_to_its_recorded_fingerprint_is_refused(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.pt"
    artifact = _artifact()
    save_artifact(artifact, target)
    payload = _raw(target)
    metadata = json.loads(str(payload["metadata"]))
    merges = metadata["tokenizer_merges"]
    # Replace the last rule with another valid pair: the table keeps its length (so the declared size
    # still agrees) and the metadata checksum is repaired, so only the vocabulary's own fingerprint
    # can catch that the rules are not the ones that produced these weights.
    metadata["tokenizer_merges"] = [*merges[:-1], [200, 201]]
    repaired = _rechecksum({**payload, "metadata": json.dumps(metadata, sort_keys=True)})
    _write_raw(target, repaired)

    with pytest.raises(
        IncompatibleCheckpointError, match="does not hash to its recorded fingerprint"
    ):
        load_artifact(target)


def test_an_architecture_wider_than_the_vocabulary_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "artifact.pt"
    artifact = _artifact()
    save_artifact(artifact, target)
    payload = _raw(target)
    metadata = json.loads(str(payload["metadata"]))
    metadata["tokenizer_size"] = 4096
    repaired = _rechecksum({**payload, "metadata": json.dumps(metadata, sort_keys=True)})
    _write_raw(target, repaired)

    with pytest.raises(IncompatibleCheckpointError, match="vocabulary"):
        load_artifact(target)


def _rechecksum(payload: dict[str, object]) -> dict[str, object]:
    import hashlib

    metadata = str(payload["metadata"])
    tensors = payload["tensors"]
    assert isinstance(tensors, dict)
    digest = hashlib.sha256()
    digest.update(metadata.encode("utf-8"))
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {**payload, "checksum": digest.hexdigest()}


def test_an_artifact_naming_a_field_this_code_ignores_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "artifact.pt"
    artifact = _artifact()
    save_artifact(artifact, target)
    payload = _raw(target)
    metadata = json.loads(str(payload["metadata"]))
    metadata["config"] = {**metadata["config"], "rotary_base": 10000}
    repaired = _rechecksum({**payload, "metadata": json.dumps(metadata, sort_keys=True)})
    _write_raw(target, repaired)

    with pytest.raises(IncompatibleCheckpointError, match="cannot build"):
        load_artifact(target)


def test_an_artifact_without_a_corpus_fingerprint_is_refused() -> None:
    with pytest.raises(IncompatibleCheckpointError, match="corpus fingerprint"):
        ModelArtifact(
            config=CONFIG,
            state=dict(_artifact().state),
            vocabulary=VOCABULARY,
            corpus_fingerprint="",
        )


def test_a_negative_step_is_refused() -> None:
    with pytest.raises(IncompatibleCheckpointError, match="negative"):
        ModelArtifact(
            config=CONFIG,
            state=dict(_artifact().state),
            vocabulary=VOCABULARY,
            corpus_fingerprint="x",
            step=-1,
        )
