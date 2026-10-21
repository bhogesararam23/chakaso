"""Contract and behaviour tests for the subword-Transformer inference adapter.

The adapter must satisfy the same `LanguageModel` contract as every other implementation and must load a
*real* trained checkpoint — no stub weights, no silent fall back to the double. Torch-gated: these run only
where the `[nn]` extra is installed, and K-009 records that remote CI does not install it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.model_contracts import LanguageModelContract  # noqa: E402

from chakaso.models import (  # noqa: E402
    SUBWORD_TRANSFORMER_MODEL_ID,
    Capability,
    FinishReason,
    GenerationParams,
    Message,
    ModelError,
    Role,
    create_adapter,
    load_model,
)
from chakaso.models.nntransformer import (  # noqa: E402
    SUBWORD_TRANSFORMER_MODEL_ID as MODULE_ID,
)
from chakaso.models.nntransformer import SubwordTransformerModel  # noqa: E402
from chakaso.models.registry import default_registry  # noqa: E402
from chakaso.training.data import TextUnit, split_text_units  # noqa: E402
from chakaso.training.nn_checkpoint import checkpoint_filename, save_artifact  # noqa: E402
from chakaso.training.nn_training import TransformerTrainingConfig, train_transformer  # noqa: E402
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.subword import DOCUMENT_END, train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import TransformerConfig  # noqa: E402

_TEXT = (
    "the quick red fox ran over the dry sand and the quiet owl slept in the town. "
    "the fox ran again over the sand and the owl slept quietly. "
)
_CONTEXT = 8


def _checkpoint(directory: Path) -> Path:
    """Train a genuinely tiny model through the real pipeline and save its artifact.

    The seeds are tried in a fixed order and the first that yields a non-empty greedy completion is kept,
    because the inherited contract asserts an adapter returns text — a property of the trained artifact,
    not of the contract, so it is established here rather than relaxed there.
    """
    documents = tuple(
        TextUnit(identifier=f"doc-{index}", text=f"{_TEXT} line {index}\n") for index in range(6)
    )
    split = split_text_units(documents, validation_fraction=0.2, test_fraction=0.2, seed=3)
    vocabulary = train_subword_vocabulary([unit.text for unit in split.train], target_size=280)
    architecture = TransformerConfig(
        vocab_size=vocabulary.size,
        context_length=_CONTEXT,
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
    )
    train_batch = pack_units(split.train, vocabulary, context_length=_CONTEXT)
    validation_batch = pack_units(split.validation, vocabulary, context_length=_CONTEXT)

    for seed in (1, 2, 3, 4, 5):
        outcome = train_transformer(
            train_examples=train_batch.examples,
            validation_examples=validation_batch.examples,
            vocabulary=vocabulary,
            config=TransformerTrainingConfig(
                batch_size=4,
                max_steps=24,
                learning_rate=1e-2,
                weight_decay=0.0,
                warmup_steps=4,
                gradient_clip_norm=1.0,
                seed=seed,
                eval_every=12,
                patience=4,
                min_delta=0.0,
                architecture=architecture,
            ),
            corpus_fingerprint="adapter-fixture",
        )
        path = directory / checkpoint_filename("ADAPTER")
        save_artifact(outcome.artifact(), path)
        model = SubwordTransformerModel(str(path))
        if model.sample(_TEXT, max_new_tokens=8).text:
            return path
        path.unlink()

    pytest.fail(
        "no seed in the fixture produced a model that emits text; the contract test cannot run"
    )


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One trained artifact for the whole module: the fixture is a real run, not a stub."""
    return _checkpoint(tmp_path_factory.mktemp("adapter"))


class TestSubwordTransformerModel(LanguageModelContract):
    expected_capabilities = frozenset({Capability.TOKENIZE})

    @pytest.fixture
    def model(self, checkpoint: Path) -> SubwordTransformerModel:
        return SubwordTransformerModel(str(checkpoint))


def test_the_adapter_declares_itself_trained_and_names_its_context(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))

    assert model.metadata.model_id == SUBWORD_TRANSFORMER_MODEL_ID
    assert model.metadata.development_double is False
    assert model.metadata.context_window == _CONTEXT
    assert "not a capable assistant" in model.metadata.display_name


def test_the_module_and_registry_names_agree() -> None:
    assert MODULE_ID == SUBWORD_TRANSFORMER_MODEL_ID
    assert SUBWORD_TRANSFORMER_MODEL_ID in default_registry()


def test_tokenize_uses_the_vocabulary_bound_inside_the_checkpoint(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))

    assert model.tokenize("the fox") == list(model.artifact.vocabulary.encode("the fox"))
    assert model.tokenize("") == []


def test_sample_is_deterministic_at_zero_temperature(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))

    first = model.sample(_TEXT, max_new_tokens=12)
    second = model.sample(_TEXT, max_new_tokens=12)

    assert first == second
    assert first.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)


def test_sampling_repeats_an_identical_request_and_differs_across_requests(
    checkpoint: Path,
) -> None:
    # The seed is derived from the request, so a sampled call is replayable (the boundary's promise) while
    # two different prompts are not silently pinned to the same random stream.
    model = SubwordTransformerModel(str(checkpoint))

    same_a = model.sample("the quick red fox", max_new_tokens=16, temperature=0.9)
    same_b = model.sample("the quick red fox", max_new_tokens=16, temperature=0.9)
    other = model.sample("the quiet owl slept", max_new_tokens=16, temperature=0.9)

    assert same_a.text == same_b.text
    assert other.text != same_a.text or other.finish_reason != same_a.finish_reason


def test_generate_and_sample_share_one_decoding_path(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))
    messages = [Message(role=Role.USER, content=_TEXT)]

    boundary = model.generate(messages, GenerationParams(max_new_tokens=10, temperature=0.0))
    convenience = model.sample(_TEXT, max_new_tokens=10)

    assert boundary == convenience


def test_top_k_one_matches_the_greedy_choice(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))

    greedy = model.sample(_TEXT, max_new_tokens=6)
    restricted = model.sample(_TEXT, max_new_tokens=6, temperature=0.8, top_k=1)

    assert restricted.text == greedy.text


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_an_invalid_top_k_is_refused(checkpoint: Path, top_k: object) -> None:
    model = SubwordTransformerModel(str(checkpoint))

    with pytest.raises(ModelError, match="top_k"):
        model.sample(_TEXT, max_new_tokens=4, temperature=0.5, top_k=top_k)  # type: ignore[arg-type]


def test_an_empty_prompt_is_refused_rather_than_answered(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))

    with pytest.raises(ModelError, match="non-empty prompt"):
        model.sample("")


def test_a_long_prompt_slides_the_window_instead_of_failing(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))
    long_prompt = " ".join(f"line {index} of a very long prompt" for index in range(80))

    result = model.sample(long_prompt, max_new_tokens=4)

    assert len(result.text) >= 0
    assert result.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)


def test_generating_stops_at_the_document_separator(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The separator is a boundary, not content: predicting it must end generation and report STOP rather
    # than emitting an empty string as if the model had finished a sentence.
    model = SubwordTransformerModel(str(checkpoint))

    class AlwaysSeparator:
        def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
            batch, _ = input_ids.shape
            logits = torch.zeros(batch, model.artifact.config.vocab_size)
            logits[:, DOCUMENT_END] = 10.0
            return logits

    monkeypatch.setattr(model, "_model", AlwaysSeparator())

    result = model.sample(_TEXT, max_new_tokens=12)

    assert result.finish_reason is FinishReason.STOP
    assert result.text == ""


def test_score_is_lower_on_text_the_model_was_trained_on(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))
    shuffled = " ".join(reversed(_TEXT.split()))

    trained_text = model.score(_TEXT * 3)
    other_text = model.score(shuffled * 3)

    assert trained_text < other_text


def test_score_refuses_text_it_cannot_score(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))

    with pytest.raises(ModelError, match="empty text"):
        model.score("   ")
    with pytest.raises(ModelError, match="shorter than one"):
        model.score("a")


# ------------------------------------------------------------------ selection and refusal


def test_a_checkpoint_loads_and_generates_in_a_fresh_process(checkpoint: Path) -> None:
    """Reproduction means a later process can use the artifact, so this one is a separate interpreter.

    Greedy decoding is deterministic, so the two processes must agree character for character; a checkpoint
    that only worked inside the process that wrote it would still pass every other test in this module.
    """
    import subprocess
    import sys

    from chakaso.models import load_model

    in_process = load_model(checkpoint).sample(_TEXT, max_new_tokens=12, temperature=0.0).text
    program = (
        "from pathlib import Path\n"
        "from chakaso.models import load_model\n"
        f"model = load_model(Path(input()))\n"
        f"print(model.sample({_TEXT!r}, max_new_tokens=12, temperature=0.0).text)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        input=str(checkpoint),
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == in_process.strip()


def test_load_model_returns_the_transformer_adapter(checkpoint: Path) -> None:
    loaded = load_model(checkpoint)

    assert isinstance(loaded, SubwordTransformerModel)
    assert loaded.metadata.model_id == SUBWORD_TRANSFORMER_MODEL_ID


def test_create_adapter_selects_it_without_a_resolved_config(checkpoint: Path) -> None:
    model = create_adapter(SUBWORD_TRANSFORMER_MODEL_ID, checkpoint_path=str(checkpoint))

    assert model.metadata.development_double is False
    assert model.metadata.context_window == _CONTEXT


def test_an_empty_checkpoint_path_is_refused() -> None:
    with pytest.raises(ModelError, match="requires a checkpoint path"):
        SubwordTransformerModel("")


def test_an_absent_checkpoint_is_refused_without_falling_back(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="found no checkpoint"):
        SubwordTransformerModel(str(tmp_path / "absent.pt"))


def test_a_corrupt_checkpoint_names_the_path_and_is_not_replaced_by_the_double(
    tmp_path: Path,
) -> None:
    target = tmp_path / checkpoint_filename("CORRUPT")
    target.write_bytes(b"this is not a torch artifact")

    with pytest.raises(ModelError, match="could not load") as caught:
        SubwordTransformerModel(str(target))

    assert "CORRUPT" in str(caught.value)
    assert not isinstance(caught.value, type(None))


def test_the_adapter_is_never_the_default_of_the_shipped_configuration() -> None:
    from chakaso.config import load_config
    from chakaso.models import create_model

    shipped = Path(__file__).resolve().parent.parent / "configs" / "default.toml"
    loaded = load_config(shipped)
    assert loaded.config.model.adapter != SUBWORD_TRANSFORMER_MODEL_ID
    model = create_model(loaded.config)
    assert model.metadata.development_double is True
