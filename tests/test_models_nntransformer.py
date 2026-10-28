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
from chakaso.training.decoder import ChakasoDecoder, DecoderConfig  # noqa: E402
from chakaso.training.nn_checkpoint import (  # noqa: E402
    ModelArtifact,
    checkpoint_filename,
    save_artifact,
)
from chakaso.training.nn_training import TransformerTrainingConfig, train_transformer  # noqa: E402
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.subword import DOCUMENT_END, train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import TransformerConfig  # noqa: E402

_TEXT = (
    "the quick red fox ran over the dry sand and the quiet owl slept in the town. "
    "the fox ran again over the sand and the owl slept quietly. "
)
_CONTEXT = 8


def _messages(text: str = _TEXT) -> list[Message]:
    """The single-user-message request the boundary tests sample with."""
    return [Message(role=Role.USER, content=text)]


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


def test_the_display_name_names_the_architecture_without_inventing_provenance(
    checkpoint: Path, decoder_checkpoint: Path
) -> None:
    """A name the artifact cannot contradict, and no corpus claim the artifact did not earn.

    The string this replaced read "Subword Transformer (… trained on the project's own documentation …)". It
    was written when E2605 was the only Transformer-family checkpoint this project had measured, and it is false
    of the released rotary decoder and of every artifact trained on licensed prose since E2607 — while being
    printed by `chakaso model inspect`, served at `/v1/models` and stored in an answer's provenance. A display
    name is a provenance surface, so it either derives from the file or it says nothing about the corpus.
    """
    transformer = SubwordTransformerModel(str(checkpoint))
    decoder = SubwordTransformerModel(str(decoder_checkpoint))

    assert "Transformer" in transformer.metadata.display_name
    assert "decoder" in decoder.metadata.display_name.lower()
    for model in (transformer, decoder):
        assert "documentation" not in model.metadata.display_name
        assert "not a capable assistant" in model.metadata.display_name
    assert decoder.describe()["family"] == "decoder"
    assert transformer.describe()["family"] == "transformer"


@pytest.fixture(scope="session")
def decoder_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A decoder-family artifact from the same vocabulary, so the adapter reads a real second schema.

    The weights are freshly initialized rather than trained: this fixture exists to prove the adapter reports
    the architecture the file declares, and an untrained model answers that without a second training run.
    """
    directory = tmp_path_factory.mktemp("adapter-decoder")
    documents = tuple(
        TextUnit(identifier=f"doc-{index}", text=f"{_TEXT} line {index}\n") for index in range(6)
    )
    split = split_text_units(documents, validation_fraction=0.2, test_fraction=0.2, seed=3)
    vocabulary = train_subword_vocabulary([unit.text for unit in split.train], target_size=280)
    config = DecoderConfig(
        vocab_size=vocabulary.size,
        context_length=_CONTEXT,
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
    )
    torch.manual_seed(1)
    decoder = ChakasoDecoder(config)
    artifact = ModelArtifact(
        config=config,
        state={name: tensor.clone() for name, tensor in decoder.state_dict().items()},
        vocabulary=vocabulary,
        corpus_fingerprint="adapter-fixture",
        run_metadata={"kind": "usable", "best_step": 0},
        step=0,
    )
    path = directory / checkpoint_filename("ADAPTERD", "decoder")
    save_artifact(artifact, path)
    return path


class TestSubwordTransformerModel(LanguageModelContract):
    expected_capabilities = frozenset({Capability.TOKENIZE, Capability.STREAMING})

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

    # One validation point for generation parameters: the parameter type itself. The adapter used to check
    # the same field a second way, which let a bad value pass one gate and fail the other.
    with pytest.raises(ValueError, match="top_k must be an integer"):
        model.sample(_TEXT, max_new_tokens=4, temperature=0.5, top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, True, "3"])
def test_a_misconfigured_adapter_is_refused_at_construction(
    top_k: object, checkpoint: Path
) -> None:
    # The registry's own knob, checked where it is set: a configuration that cannot sample should not load.
    with pytest.raises(ModelError, match="configured with top_k"):
        SubwordTransformerModel(str(checkpoint), top_k=top_k)


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


# ------------------------------------------------------------------ streaming and decoding controls
def test_streaming_yields_one_chunk_per_token_and_ends_with_the_finish_reason(
    checkpoint: Path,
) -> None:
    from chakaso.models.base import GenerationParams
    from chakaso.models.decoding import AbortSignal

    model = SubwordTransformerModel(str(checkpoint))
    params = GenerationParams(max_new_tokens=6, temperature=0.0)

    chunks = list(model.stream(_messages(), params, signal=AbortSignal()))
    content = [chunk for chunk in chunks if not chunk.final]

    assert len(content) == 6
    assert chunks[-1].final and chunks[-1].finish_reason is not None
    assert not any(chunk.final for chunk in content)
    assert "".join(chunk.text for chunk in content) == model.generate(_messages(), params).text


def test_an_aborted_stream_stops_early_and_says_so(checkpoint: Path) -> None:
    from chakaso.models.base import FinishReason, GenerationParams
    from chakaso.models.decoding import AbortSignal

    model = SubwordTransformerModel(str(checkpoint))
    signal = AbortSignal()
    params = GenerationParams(max_new_tokens=24, temperature=0.0)

    collected = []
    for chunk in model.stream(_messages(), params, signal=signal):
        collected.append(chunk)
        signal.abort()

    # One token out, then the caller's abort is seen at the next step boundary: two chunks, the second one
    # the marker that says why the run ended.
    assert len(collected) == 2
    assert collected[0].final is False and collected[0].text
    assert collected[1].final and collected[1].finish_reason is FinishReason.CANCELLED


def test_a_stop_sequence_ends_generation_and_is_reported(checkpoint: Path) -> None:
    from chakaso.models.base import FinishReason, GenerationParams

    model = SubwordTransformerModel(str(checkpoint))
    greedy = model.generate(_messages(), GenerationParams(max_new_tokens=12, temperature=0.0))
    first_character = greedy.text[:1]

    stopped = model.generate(
        _messages(),
        GenerationParams(max_new_tokens=12, temperature=0.0, stop_sequences=(first_character,)),
    )

    assert stopped.finish_reason is FinishReason.STOP
    assert stopped.stopped_by == first_character
    assert len(stopped.token_ids) < len(greedy.token_ids)


def test_a_stop_token_identifier_ends_generation(checkpoint: Path) -> None:
    from chakaso.models.base import FinishReason, GenerationParams

    model = SubwordTransformerModel(str(checkpoint))
    greedy = model.generate(_messages(), GenerationParams(max_new_tokens=8, temperature=0.0))
    assert greedy.token_ids

    target = greedy.token_ids[-1]
    stopped = model.generate(
        _messages(),
        GenerationParams(max_new_tokens=8, temperature=0.0, stop_token_ids=(target,)),
    )

    assert stopped.finish_reason is FinishReason.STOP
    assert stopped.stopped_by == f"token:{target}"
    # A model that repeats the same identifier can hit the declared stop earlier than the run that chose it
    # as "last"; the guarantee is that the identifier ends the generation, not where it ends it.
    assert 0 < len(stopped.token_ids) <= len(greedy.token_ids)


def test_generation_reports_prompt_and_new_token_counts_and_truncation(checkpoint: Path) -> None:

    model = SubwordTransformerModel(str(checkpoint))
    long_prompt = " ".join(["the quick red fox"] * 60)

    result = model.sample(long_prompt, max_new_tokens=4)

    assert result.new_tokens == len(result.token_ids) == 4
    assert result.prompt_tokens == model.metadata.context_window
    assert result.truncated is True


def test_an_invalid_parameter_set_is_refused_once_at_the_parameter_type(
    checkpoint: Path,
) -> None:
    from chakaso.models.base import GenerationParams

    model = SubwordTransformerModel(str(checkpoint))

    with pytest.raises(ValueError, match="top_p must be in"):
        GenerationParams(temperature=1.0, top_p=0.0)
    with pytest.raises(ValueError, match="repeat_penalty must be in"):
        GenerationParams(repeat_penalty=9.0)
    with pytest.raises(ValueError, match="stop_sequences must be non-empty strings"):
        GenerationParams(stop_sequences=("",))
    with pytest.raises(ModelError, match="at least one message"):
        model.generate([], GenerationParams())


def test_the_repetition_penalty_changes_a_greedy_sample_without_changing_the_greedy_rule(
    checkpoint: Path,
) -> None:
    from chakaso.models.base import GenerationParams

    model = SubwordTransformerModel(str(checkpoint))
    plain = model.generate(_messages(), GenerationParams(max_new_tokens=8, temperature=0.0))
    penalised = model.generate(
        _messages(),
        GenerationParams(max_new_tokens=8, temperature=0.0, repeat_penalty=2.0),
    )

    assert plain.text != penalised.text
    # The penalty changes which token is best; it does not change the rule that the best token is taken.
    assert penalised.finish_reason == plain.finish_reason


def test_the_prompt_template_is_part_of_the_runtime_description(checkpoint: Path) -> None:
    from chakaso.models.formatting import ROLE_LABELLED_CHAT

    plain = SubwordTransformerModel(str(checkpoint))
    chat = SubwordTransformerModel(str(checkpoint), prompt_template=ROLE_LABELLED_CHAT.version)

    assert plain.describe()["prompt_template"] == "chakaso-plain-v1"
    assert chat.describe()["prompt_template"] == "chakaso-role-labelled-v1"
    assert chat.describe()["family"] == "transformer"
    assert chat.describe()["development_double"] is False


def test_a_chat_template_reaches_the_model_as_labelled_text(checkpoint: Path) -> None:
    from chakaso.models.base import GenerationParams
    from chakaso.models.formatting import ROLE_LABELLED_CHAT

    plain = SubwordTransformerModel(str(checkpoint))
    chat = SubwordTransformerModel(str(checkpoint), prompt_template=ROLE_LABELLED_CHAT.version)
    params = GenerationParams(max_new_tokens=2, temperature=0.0)

    labelled = chat.generate(_messages("hello there"), params)
    unlabelled = plain.generate(_messages("hello there"), params)

    # Observable end to end, without reaching into the adapter: the fixture checkpoint's window is 8
    # identifiers, and the labelled transcript ("user: hello there\nassistant: ") overflows it while the
    # plain one does not. The template the model was fed is therefore the one that was configured.
    assert labelled.truncated is True
    assert unlabelled.truncated is False
    assert labelled.prompt_tokens == unlabelled.prompt_tokens == 8


def _scripted(model: SubwordTransformerModel, identifiers: list[int]) -> None:
    """Make the model choose exactly ``identifiers``, one per step, whatever the weights say."""
    size = model.artifact.vocabulary.size
    queue = list(identifiers)

    def logits(_ids: object, _cache: object) -> list[float]:
        row = [0.0] * size
        row[queue.pop(0)] = 50.0
        return row

    model._logits = logits  # type: ignore[method-assign]


def test_a_character_split_across_two_generated_tokens_is_not_destroyed(checkpoint: Path) -> None:
    # Regression: each token used to be decoded on its own with errors="replace", so the two bytes of a
    # pound sign (C2 A3) became two replacement characters that no later join could repair. Found by a
    # sampled run over the released checkpoint, where it surfaced as "ordifferent \ufffd\ufffd 1600".
    model = SubwordTransformerModel(str(checkpoint))
    _scripted(model, [ord("a"), 0xC2, 0xA3, ord("b")])

    result = model.generate(_messages(), GenerationParams(max_new_tokens=4, temperature=0.0))

    assert result.text == "a\u00a3b"
    assert result.token_ids == (ord("a"), 0xC2, 0xA3, ord("b"))


def test_a_streamed_split_character_arrives_whole_and_chunks_join_to_the_result(
    checkpoint: Path,
) -> None:
    model = SubwordTransformerModel(str(checkpoint))
    _scripted(model, [ord("a"), 0xC2, 0xA3, ord("b")])

    chunks = list(model.stream(_messages(), GenerationParams(max_new_tokens=4, temperature=0.0)))

    assert all("\ufffd" not in chunk.text for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == "a\u00a3b"
    assert chunks[-1].final


def test_a_run_cut_off_inside_a_character_flushes_one_replacement(checkpoint: Path) -> None:
    model = SubwordTransformerModel(str(checkpoint))
    _scripted(model, [ord("a"), 0xC2])

    chunks = list(model.stream(_messages(), GenerationParams(max_new_tokens=2, temperature=0.0)))
    _scripted(model, [ord("a"), 0xC2])
    result = model.generate(_messages(), GenerationParams(max_new_tokens=2, temperature=0.0))

    assert "".join(chunk.text for chunk in chunks) == "a\ufffd"
    assert result.text == "a\ufffd"
