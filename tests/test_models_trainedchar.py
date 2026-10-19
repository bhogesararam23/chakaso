"""Contract and behavior tests for the trained-character inference adapter (P25).

The adapter must satisfy the same `LanguageModel` contract as any other implementation, and it must
load a *real* trained checkpoint — no stub weights, no silent fall back to the double. Numpy-gated:
these run only where the `[train]` extra is installed (K-009).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from tests.model_contracts import LanguageModelContract

from chakaso.models import (
    TRAINED_CHAR_MODEL_ID,
    Capability,
    GenerationParams,
    ModelError,
    create_model,
)
from chakaso.models.registry import (
    TRAINED_CHAR_MODEL_ID as REGISTRY_ID,
)
from chakaso.models.registry import default_registry
from chakaso.models.trainedchar import TrainedCharacterModel
from chakaso.training.checkpoint import Checkpoint
from chakaso.training.config import TrainingConfig
from chakaso.training.corpus import CORPUS
from chakaso.training.data import fit_vocabulary, split_corpus
from chakaso.training.loop import run_training


def _config() -> TrainingConfig:
    return TrainingConfig(
        context_size=4,
        embed_dim=8,
        hidden_dim=12,
        learning_rate=0.05,
        seed=1,
        steps=120,
        eval_every=60,
    )


def _write_checkpoint(directory: Path) -> Path:
    """Train a genuinely tiny model and save its checkpoint, returning the file path."""
    config = _config()
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=config.seed)
    trained = run_training(split, config)
    vocabulary = fit_vocabulary(split)
    checkpoint = Checkpoint(
        params=trained.final_params,
        config=config,
        vocab_size=vocabulary.size,
        vocab_fingerprint=vocabulary.fingerprint,
        dataset_digest=trained.dataset_digest,
        vocab_chars=vocabulary.chars,
    )
    path = directory / "adapter-checkpoint.json"
    checkpoint.save(path)
    return path


class TestTrainedCharacterModel(LanguageModelContract):
    expected_capabilities = frozenset({Capability.TOKENIZE})

    @pytest.fixture
    def model(self, tmp_path: Path) -> TrainedCharacterModel:
        return TrainedCharacterModel(checkpoint_path=str(_write_checkpoint(tmp_path)))

    def test_generate_is_greedy_and_deterministic(self, model: TrainedCharacterModel) -> None:
        from chakaso.models import Message, Role

        messages = [Message(role=Role.USER, content="the quick")]
        first = model.generate(messages, GenerationParams(max_new_tokens=16, temperature=0.0))
        second = model.generate(messages, GenerationParams(max_new_tokens=16, temperature=0.0))

        assert first.text == second.text
        assert first.finish_reason.name == "LENGTH"

    def test_tokenizer_matches_the_checkpoint_vocabulary(
        self, model: TrainedCharacterModel
    ) -> None:
        ids = model.tokenize("the")
        assert ids
        # Round-trips through the checkpoint's own vocabulary for known characters.
        assert model._vocabulary.decode(tuple(ids)) == "the"

    def test_metadata_declares_it_is_not_the_double(self, model: TrainedCharacterModel) -> None:
        assert model.metadata.development_double is False
        assert model.metadata.capabilities == frozenset({Capability.TOKENIZE})


def test_an_empty_checkpoint_path_fails_instead_of_falling_back() -> None:
    with pytest.raises(ModelError, match="does not fall back"):
        TrainedCharacterModel(checkpoint_path="")


def test_a_missing_checkpoint_file_fails_explicitly(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="no checkpoint"):
        TrainedCharacterModel(checkpoint_path=str(tmp_path / "absent.json"))


def test_a_malformed_checkpoint_is_refused(tmp_path: Path) -> None:
    junk = tmp_path / "junk.json"
    junk.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelError, match="could not load"):
        TrainedCharacterModel(checkpoint_path=str(junk))


def test_the_registry_selects_the_adapter_from_configuration(tmp_path: Path) -> None:
    from chakaso.config import Config, ModelConfig, RetrievalConfig, StorageConfig

    path = _write_checkpoint(tmp_path)
    model_config = ModelConfig(
        adapter=TRAINED_CHAR_MODEL_ID,
        max_new_tokens=16,
        temperature=0.0,
        max_context_turns=8,
        checkpoint_path=str(path),
    )
    config = Config(
        model=model_config,
        storage=StorageConfig(backend="memory", path="", journal_mode="delete"),
        retrieval=RetrievalConfig(
            enabled=True,
            strategy="lexical",
            top_k=5,
            candidate_k=10,
            rrf_k=60,
            max_ingest_bytes=1_048_576,
        ),
    )

    built = create_model(config)

    assert isinstance(built, TrainedCharacterModel)
    assert built.metadata.model_id == TRAINED_CHAR_MODEL_ID


def test_the_registry_name_literal_matches_the_adapter_constant() -> None:
    # The registry keeps a literal so it stays importable without numpy; this ties it to the
    # adapter's own authoritative constant.
    from chakaso.models.trainedchar import TRAINED_CHAR_MODEL_ID as ADAPTER_ID

    assert REGISTRY_ID == ADAPTER_ID == TRAINED_CHAR_MODEL_ID
    assert TRAINED_CHAR_MODEL_ID in default_registry().names()
