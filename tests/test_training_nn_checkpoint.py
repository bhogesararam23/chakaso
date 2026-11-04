"""Torch-gated tests for the versioned Transformer artifact format.

The property under test is that an artifact loads *as the model it was saved as*, or not at all. Every
refusal path here is a way a checkpoint could otherwise produce a plausible-looking wrong model: a tensor
swapped for another shape, a tokenizer that no longer matches the weights, a configuration field this code
would silently ignore, or a file that tries to carry a Python object.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.decoder import ChakasoDecoder, DecoderConfig  # noqa: E402
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

    def _interrupt(*_ignored: object, **_also_ignored: object) -> None:
        failure = "device out of space"
        raise OSError(failure)

    target = tmp_path / "artifact.pt"
    monkeypatch.setattr(torch, "save", _interrupt)

    with pytest.raises(OSError, match="out of space"):
        save_artifact(_artifact(), target)

    assert not target.exists()
    assert [path.name for path in tmp_path.iterdir()] == []


def test_the_environment_block_records_what_a_reproduction_needs_to_compare() -> None:
    """Versions and thread count as observed, because those are the inputs a number can disagree over."""
    environment = describe_environment()

    assert set(environment) == {
        "torch",
        "numpy",
        "python",
        "platform",
        "device",
        "torch_threads",
    }
    assert environment["device"] == "cpu"
    assert environment["torch"] == torch.__version__
    assert int(environment["torch_threads"]) >= 1
    # An absent optional dependency is reported as absent, never as a version this machine does not have.
    assert environment["numpy"] == importlib.metadata.version("numpy")


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


# ------------------------------------------------------------------ the decoder family
DECODER_CONFIG = DecoderConfig(
    vocab_size=260,
    context_length=6,
    d_model=8,
    num_layers=1,
    num_heads=2,
    feedforward_dim=16,
)


def _decoder_artifact(step: int = 4, *, corpus: str = "abc123") -> ModelArtifact:
    torch.manual_seed(1)
    model = ChakasoDecoder(DECODER_CONFIG)
    return ModelArtifact(
        config=DECODER_CONFIG,
        state={name: tensor.clone() for name, tensor in model.state_dict().items()},
        vocabulary=VOCABULARY,
        corpus_fingerprint=corpus,
        run_metadata={"kind": "usable", "best_step": step},
        step=step,
    )


def test_a_decoder_artifact_round_trips_as_the_model_it_was_saved_as(tmp_path: Path) -> None:
    artifact = _decoder_artifact()
    path = tmp_path / "decoder.pt"
    save_artifact(artifact, path)

    loaded = load_artifact(path)
    model = loaded.to_model()

    assert isinstance(model, ChakasoDecoder)
    assert loaded.config == DECODER_CONFIG
    assert loaded.checksum() == artifact.checksum()
    assert (
        json.loads(loaded.metadata_document())["schema_version"] == "chakaso-decoder-checkpoint/1"
    )


def test_a_decoder_checkpoint_records_the_rotary_base_it_was_built_with(tmp_path: Path) -> None:
    artifact = _decoder_artifact()
    path = tmp_path / "decoder.pt"
    save_artifact(artifact, path)

    recorded = json.loads(load_artifact(path).metadata_document())

    assert recorded["config"]["rope_theta"] == DECODER_CONFIG.rope_theta
    assert recorded["config"]["context_length"] == DECODER_CONFIG.context_length


def test_the_two_families_cannot_be_read_as_each_other(tmp_path: Path) -> None:
    # A transformer configuration happens to be a subset of a decoder one (rope_theta is optional), so a
    # swapped schema tag survives the configuration and is refused where it must be: the tensors. The
    # reverse swap is caught earlier, because a decoder configuration names a field the transformer code
    # does not have. Either way the file cannot produce a model that is not the one that was saved.
    transformer_path = tmp_path / "swapped-decoder.pt"
    save_artifact(_artifact(), transformer_path)
    payload = torch.load(transformer_path, map_location="cpu", weights_only=True)
    payload["schema_version"] = "chakaso-decoder-checkpoint/1"
    torch.save(payload, transformer_path)

    with pytest.raises(IncompatibleCheckpointError, match="tensors do not match"):
        load_artifact(transformer_path).to_model()

    decoder_path = tmp_path / "swapped-transformer.pt"
    save_artifact(_decoder_artifact(), decoder_path)
    payload = torch.load(decoder_path, map_location="cpu", weights_only=True)
    payload["schema_version"] = "chakaso-transformer-checkpoint/1"
    torch.save(payload, decoder_path)

    with pytest.raises(IncompatibleCheckpointError, match="cannot build"):
        load_artifact(decoder_path)


def test_an_unsupported_schema_version_is_named_and_refused(tmp_path: Path) -> None:
    artifact = _decoder_artifact()
    path = tmp_path / "future.pt"
    save_artifact(artifact, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = "chakaso-decoder-checkpoint/9"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="unsupported artifact schema"):
        load_artifact(path)


def test_a_checkpoint_of_the_wrong_dtype_is_refused_before_it_is_written() -> None:
    artifact = _decoder_artifact()
    doubled = {name: tensor.double() for name, tensor in artifact.state.items()}
    wrong_dtype = ModelArtifact(
        config=artifact.config,
        state=doubled,
        vocabulary=artifact.vocabulary,
        corpus_fingerprint=artifact.corpus_fingerprint,
        step=artifact.step,
    )

    with pytest.raises(IncompatibleCheckpointError, match="dtype disagreement"):
        save_artifact(wrong_dtype, Path("never-written.pt"))


def test_a_transformer_artifact_still_loads_under_its_own_schema(tmp_path: Path) -> None:
    path = tmp_path / "transformer.pt"
    save_artifact(_artifact(), path)

    loaded = load_artifact(path)

    assert isinstance(loaded.to_model(), CausalTransformer)
    assert json.loads(loaded.metadata_document())["schema_version"] == (
        "chakaso-transformer-checkpoint/1"
    )


def test_an_artifact_file_name_names_its_family() -> None:
    assert checkpoint_filename("E2608") == "E2608-transformer.pt"
    assert checkpoint_filename("E2608", "decoder") == "E2608-decoder.pt"
    with pytest.raises(IncompatibleCheckpointError, match="no artifact family"):
        checkpoint_filename("E2608", "mamba")


def test_writing_a_checkpoint_does_not_advance_the_random_stream_the_run_is_drawing_from(
    tmp_path: Path,
) -> None:
    """K-030: validating an artifact builds a model, and building one draws from the global generator."""
    import torch

    artifact = _artifact()
    torch.manual_seed(1234)
    before = torch.get_rng_state()

    save_artifact(artifact, tmp_path / "a.pt")

    assert torch.equal(torch.get_rng_state(), before)
    # The generator really would have moved: the stream a run keeps drawing from is the same one.
    torch.manual_seed(1234)
    expected = torch.rand(3)
    torch.manual_seed(1234)
    save_artifact(artifact, tmp_path / "b.pt")
    assert torch.equal(torch.rand(3), expected)


# ------------------------------------------------ the tokenizer's frequency floor (ADR-0046, K-030 item 6)


def _write_as_written_before_adr_0046(artifact: ModelArtifact, path: Path) -> dict[str, object]:
    """A file in the exact shape the code produced before the field existed.

    The metadata document is built here the way ``metadata_document`` built it then — no ``tokenizer_min_frequency``
    key — and sealed with the same checksum algorithm. (Checked once, on 2026-11-02, against a file written by the
    unmodified code: the metadata strings were identical. A binary fixture is not committed because checkpoint
    bytes stay out of Git.)
    """
    document = {
        "schema_version": "chakaso-transformer-checkpoint/1",
        "config": artifact.config.as_dict(),
        "tokenizer_fingerprint": artifact.vocabulary.fingerprint,
        "tokenizer_size": artifact.vocabulary.size,
        "tokenizer_merges": [[a, b] for a, b in artifact.vocabulary.merges],
        "corpus_fingerprint": artifact.corpus_fingerprint,
        "step": artifact.step,
        "run_metadata": dict(artifact.run_metadata),
        "environment": dict(artifact.environment),
        "tensor_names": sorted(artifact.state),
    }
    metadata = json.dumps(document, sort_keys=True, separators=(",", ":"))
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in artifact.state.items()}
    digest = hashlib.sha256(metadata.encode("utf-8"))
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensors[name].shape)).encode("utf-8"))
        digest.update(str(tensors[name].dtype).encode("utf-8"))
        digest.update(tensors[name].view(torch.uint8).numpy().tobytes())
    payload: dict[str, object] = {
        "schema_version": "chakaso-transformer-checkpoint/1",
        "metadata": metadata,
        "checksum": digest.hexdigest(),
        "tensors": tensors,
        "optimizer": None,
    }
    _write_raw(path, payload)
    return payload


def _fitted_at(floor: int):  # type: ignore[no-untyped-def]
    """A vocabulary fitted under the rule an artifact from before ADR-0048 would have used.

    Named explicitly, not left to the default: these fixtures exist to prove that an artifact which names **no** rule
    still loads and re-saves byte for byte, and under the adopted default (ADR-0051) a freshly fitted vocabulary does
    name one, which would make the fixture describe a different artifact than the one the test is about.
    """
    from chakaso.training.subword import LEGACY_DEFAULT_PRE_TOKENIZER

    return train_subword_vocabulary(
        ["the quick red fox ran over the dry sand and slept"] * 3,
        target_size=260,
        min_frequency=floor,
        pre_tokenizer=LEGACY_DEFAULT_PRE_TOKENIZER,
    )


def _with_floor(floor: int | None) -> ModelArtifact:
    base = _artifact()
    return ModelArtifact(
        config=base.config,
        state=base.state,
        vocabulary=_fitted_at(3),
        corpus_fingerprint=base.corpus_fingerprint,
        run_metadata=base.run_metadata,
        step=base.step,
        environment=base.environment,
        tokenizer_min_frequency=floor,
    )


def test_a_floor_the_vocabulary_was_fitted_at_survives_a_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "artifact.pt"
    save_artifact(_with_floor(3), target)

    loaded = load_artifact(target)

    assert loaded.tokenizer_min_frequency == 3
    assert loaded.vocabulary.min_frequency == 3


def test_an_artifact_written_before_the_field_existed_still_loads_and_says_unrecorded(
    tmp_path: Path,
) -> None:
    target = tmp_path / "old.pt"
    written = _write_as_written_before_adr_0046(_with_floor(None), target)

    loaded = load_artifact(target)

    assert loaded.checksum() == written["checksum"]
    assert loaded.tokenizer_min_frequency is None
    assert "tokenizer_min_frequency" not in json.loads(loaded.metadata_document())


def test_resaving_an_unrecorded_artifact_preserves_its_metadata_document_and_checksum(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old.pt"
    original = _write_as_written_before_adr_0046(_with_floor(None), old)
    loaded = load_artifact(old)
    target = tmp_path / "resaved.pt"

    save_artifact(loaded, target)

    resaved = _raw(target)
    assert resaved["metadata"] == original["metadata"]
    assert resaved["checksum"] == original["checksum"]


def test_the_new_key_is_additive_so_a_reader_built_before_it_finds_every_key_it_reads(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.pt"
    save_artifact(_with_floor(3), target)
    document = json.loads(str(_raw(target)["metadata"]))
    assert document["tokenizer_min_frequency"] == 3

    # What a reader built before the field does: take the named keys it knows and nothing else.
    known = {
        "config",
        "tokenizer_fingerprint",
        "tokenizer_size",
        "tokenizer_merges",
        "corpus_fingerprint",
        "step",
        "run_metadata",
        "environment",
        "tensor_names",
        "schema_version",
    }
    assert set(document) - known == {"tokenizer_min_frequency"}
    assert {"config", "tokenizer_merges", "corpus_fingerprint"} <= set(document)


def test_an_unrecorded_floor_is_not_written_as_a_key() -> None:
    assert "tokenizer_min_frequency" not in json.loads(_with_floor(None).metadata_document())


@pytest.mark.parametrize("bad", [0, -1, True, "3", 2.5])
def test_a_floor_that_is_not_a_positive_integer_is_refused(bad: object) -> None:
    with pytest.raises(IncompatibleCheckpointError, match="tokenizer_min_frequency"):
        _with_floor(bad)  # type: ignore[arg-type]


def test_a_stored_floor_that_is_not_a_positive_integer_is_a_corrupt_artifact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.pt"
    save_artifact(_with_floor(3), target)
    payload = _raw(target)
    document = json.loads(str(payload["metadata"]))
    document["tokenizer_min_frequency"] = "three"
    metadata = json.dumps(document, sort_keys=True, separators=(",", ":"))
    payload["metadata"] = metadata
    # Re-seal so the checksum is honest and the refusal is about the value, not about tampering.
    tensors: dict[str, torch.Tensor] = payload["tensors"]  # type: ignore[assignment]
    digest = hashlib.sha256(metadata.encode("utf-8"))
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    payload["checksum"] = digest.hexdigest()
    _write_raw(target, payload)

    with pytest.raises(IncompatibleCheckpointError, match="tokenizer_min_frequency"):
        load_artifact(target)


# ------------------------------------------------ the tokenizer's pre-tokenizer (ADR-0048)


def _with_rule(rule: str) -> ModelArtifact:
    base = _artifact()
    vocabulary = train_subword_vocabulary(
        ["the quick red fox ran over the dry sand and slept"] * 3,
        target_size=260,
        min_frequency=2,
        pre_tokenizer=rule,
    )
    return ModelArtifact(
        config=base.config,
        state=base.state,
        vocabulary=vocabulary,
        corpus_fingerprint=base.corpus_fingerprint,
        run_metadata=base.run_metadata,
        step=base.step,
        environment=base.environment,
    )


def test_a_non_default_pre_tokenizer_is_recorded_and_survives_a_round_trip(tmp_path: Path) -> None:
    from chakaso.training.subword import SPACE_PREFIXED_PRE_TOKENIZER

    artifact = _with_rule(SPACE_PREFIXED_PRE_TOKENIZER)
    target = tmp_path / "artifact.pt"
    save_artifact(artifact, target)

    loaded = load_artifact(target)

    assert json.loads(artifact.metadata_document())["tokenizer_pre_tokenizer"] == (
        SPACE_PREFIXED_PRE_TOKENIZER
    )
    assert loaded.vocabulary.pre_tokenizer == SPACE_PREFIXED_PRE_TOKENIZER
    assert loaded.vocabulary.fingerprint == artifact.vocabulary.fingerprint
    assert loaded.vocabulary.encode("the quick fox") == artifact.vocabulary.encode("the quick fox")


def test_the_legacy_default_pre_tokenizer_writes_no_key() -> None:
    from chakaso.training.subword import LEGACY_DEFAULT_PRE_TOKENIZER

    document = json.loads(_with_rule(LEGACY_DEFAULT_PRE_TOKENIZER).metadata_document())

    assert "tokenizer_pre_tokenizer" not in document


def test_an_artifact_that_names_no_rule_is_read_as_the_word_run_rule(tmp_path: Path) -> None:
    """The absence of the key records a **date**, not a choice, so it cannot track the ambient default.

    `DEFAULT_PRE_TOKENIZER` is the rule a new fit uses when its caller names none, and ADR-0051 is the record that would
    move it. The format's own constant is a different name, and it is pinned here to its literal value so that an edit
    swapping the fallback back onto the ambient default fails in this test rather than silently re-tokenising the text a
    released artifact was trained on.
    """
    from chakaso.training import nn_checkpoint
    from chakaso.training.subword import LEGACY_DEFAULT_PRE_TOKENIZER

    assert nn_checkpoint.LEGACY_DEFAULT_PRE_TOKENIZER == "word-run|whitespace-run|single-symbol"
    assert LEGACY_DEFAULT_PRE_TOKENIZER == nn_checkpoint.LEGACY_DEFAULT_PRE_TOKENIZER

    target = tmp_path / "unrecorded-rule.pt"
    artifact = _with_rule(LEGACY_DEFAULT_PRE_TOKENIZER)
    save_artifact(artifact, target)

    loaded = load_artifact(target)

    assert "tokenizer_pre_tokenizer" not in json.loads(artifact.metadata_document())
    assert loaded.vocabulary.pre_tokenizer == "word-run|whitespace-run|single-symbol"
    assert loaded.vocabulary.fingerprint == artifact.vocabulary.fingerprint


def test_a_reader_that_does_not_know_the_key_refuses_the_file_instead_of_mis_encoding() -> None:
    from chakaso.training.subword import (
        LEGACY_DEFAULT_PRE_TOKENIZER,
        SPACE_PREFIXED_PRE_TOKENIZER,
        SubwordVocabulary,
    )

    artifact = _with_rule(SPACE_PREFIXED_PRE_TOKENIZER)
    recorded = json.loads(artifact.metadata_document())

    # What a reader built before the key existed does: rebuild the vocabulary from the merge table alone, under the
    # rule that was the only one there was. Named explicitly, because the ambient default is a *choice about the
    # next fit* (ADR-0051) and this sentence is about a reader from the past.
    as_the_old_reader_sees_it = SubwordVocabulary(
        merges=tuple((int(a), int(b)) for a, b in recorded["tokenizer_merges"]),
        pre_tokenizer=LEGACY_DEFAULT_PRE_TOKENIZER,
    )

    assert as_the_old_reader_sees_it.fingerprint != recorded["tokenizer_fingerprint"]


def test_a_stored_pre_tokenizer_this_code_does_not_implement_is_refused(tmp_path: Path) -> None:
    from chakaso.training.subword import SPACE_PREFIXED_PRE_TOKENIZER

    target = tmp_path / "artifact.pt"
    save_artifact(_with_rule(SPACE_PREFIXED_PRE_TOKENIZER), target)
    payload = _raw(target)
    document = json.loads(str(payload["metadata"]))
    document["tokenizer_pre_tokenizer"] = "made-up"
    metadata = json.dumps(document, sort_keys=True, separators=(",", ":"))
    payload["metadata"] = metadata
    tensors: dict[str, torch.Tensor] = payload["tensors"]  # type: ignore[assignment]
    digest = hashlib.sha256(metadata.encode("utf-8"))
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    payload["checksum"] = digest.hexdigest()
    _write_raw(target, payload)

    with pytest.raises(IncompatibleCheckpointError, match="tokenizer_pre_tokenizer"):
        load_artifact(target)
