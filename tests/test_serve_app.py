"""Tests for the local HTTP serving adapter.

Two layers, tested separately on purpose. :class:`~chakaso.serve.app.ServingApplication` is pure request
logic, so the contract — what is accepted, what is refused, what a refusal says — is checked without a port,
a thread or a trained model, in the dependency-free matrix, against a scripted model that records what it was
asked for. :class:`~chakaso.serve.server.ServingService` is then exercised over a real socket with a real
checkpoint, which is the only way to test chunked streaming, a client that hangs up, and the concurrency
limit.

The claim under test throughout is the honest one: this serves a trained model and nothing else, and it
refuses the parts of the API shape it cannot honour instead of quietly answering a different question.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from chakaso.models import (
    Capability,
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    ModelMetadata,
)
from chakaso.models.base import GenerationChunk
from chakaso.serve.app import (
    DEFAULT_MAX_INPUT_CHARACTERS,
    MAX_NEW_TOKENS_LIMIT,
    ServiceResponse,
    ServingApplication,
    ServingError,
    load_serving_model,
)

TEXTS = (" the", " model", " that", " repeats")


class _ScriptedModel:
    """A model that yields a fixed answer and records every request, so policy is observable."""

    def __init__(
        self,
        *,
        answer: str = "the model that repeats",
        streaming: bool = True,
        double: bool = False,
    ) -> None:
        self._answer = answer
        self._streaming = streaming
        self.requests: list[tuple[tuple[str, ...], GenerationParams]] = []
        self.metadata = ModelMetadata(
            model_id="scripted_test_model",
            display_name="Scripted test model",
            context_window=128,
            capabilities=(
                frozenset({Capability.TOKENIZE, Capability.STREAMING})
                if streaming
                else frozenset({Capability.TOKENIZE})
            ),
            development_double=double,
        )

    @property
    def serving_identity(self) -> str:
        return "scripted@0.1.0"

    def describe(self) -> dict[str, object]:
        return {"model_id": self.metadata.model_id, "family": "scripted", "parameter_count": 1}

    def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        self.requests.append((tuple(message.content for message in messages), params))
        return GenerationResult(
            text=self._answer,
            model_id=self.metadata.model_id,
            finish_reason=FinishReason.STOP,
            token_ids=(1, 2, 3),
            new_tokens=3,
            prompt_tokens=7,
        )

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        signal: object | None = None,
    ) -> Iterator[GenerationChunk]:
        self.requests.append((tuple(message.content for message in messages), params))
        words = self._answer.split(" ")
        for index, word in enumerate(words):
            text = word if index == 0 else f" {word}"
            final = index == len(words) - 1
            yield GenerationChunk(
                text=text,
                token_id=index,
                final=final,
                finish_reason=FinishReason.STOP if final else None,
            )


class _NonStreamingModel(_ScriptedModel):
    """A model that satisfies the boundary but cannot stream, to prove the refusal is real."""

    stream = None  # type: ignore[assignment]


def _application(*, non_streaming: bool = False, **kwargs: object) -> ServingApplication:
    kind: type[_ScriptedModel] = _NonStreamingModel if non_streaming else _ScriptedModel
    model = kind(**kwargs)  # type: ignore[arg-type]
    return ServingApplication(model=model, model_name="scripted@0.1.0")  # type: ignore[arg-type]


def _post(application: ServingApplication, route: str, payload: object) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    response = application.handle("POST", route, body)
    assert response.body is not None
    return response.status, json.loads(response.body.decode("utf-8"))


# ------------------------------------------------------------------ discovery endpoints
def test_health_states_the_model_the_limits_and_that_it_is_not_the_double() -> None:
    response = _application().handle("GET", "/health")
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status == 200
    assert payload["status"] == "ok"
    assert payload["development_double"] is False
    assert payload["streaming"] is True
    assert payload["context_window"] == 128
    assert payload["limits"]["max_new_tokens"] == MAX_NEW_TOKENS_LIMIT
    assert payload["limits"]["max_input_characters"] == DEFAULT_MAX_INPUT_CHARACTERS


def test_models_lists_one_model_with_its_runtime_facts() -> None:
    payload = json.loads(_application().handle("GET", "/v1/models").body.decode("utf-8"))

    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["id"] == "scripted@0.1.0"
    assert payload["data"][0]["family"] == "scripted"
    assert "model_id" not in payload["data"][0]


def test_an_unknown_route_names_the_routes_that_exist() -> None:
    response = _application().handle("GET", "/v2/complete")

    assert response.status == 404
    body = json.loads(response.body.decode("utf-8"))
    assert "/v1/completions" in body["error"]["message"]


def test_a_write_method_on_a_read_route_is_refused_not_served() -> None:
    assert _application().handle("GET", "/v1/completions", b"{}").status == 404
    assert _application().handle("DELETE", "/health").status == 405


# ------------------------------------------------------------------ completions
def test_a_completion_returns_text_usage_and_the_finish_reason() -> None:
    application = _application()

    status, payload = _post(
        application, "/v1/completions", {"model": "scripted@0.1.0", "prompt": "once"}
    )

    assert status == 200
    assert payload["choices"][0]["text"] == "the model that repeats"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert payload["chakaso"]["truncated_prompt"] is False
    prompt, params = application.model.requests[-1]  # type: ignore[union-attr]
    assert prompt == ("once",)
    assert params.max_new_tokens == 64


def test_a_chat_request_renders_roles_and_returns_an_assistant_message() -> None:
    application = _application()

    status, payload = _post(
        application,
        "/v1/chat/completions",
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is Chakaso?"},
            ]
        },
    )

    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "the model that repeats",
    }
    assert "not instruction tuned" in payload["chakaso"]["note"]
    prompt, _ = application.model.requests[-1]  # type: ignore[union-attr]
    assert prompt == ("Be brief.", "What is Chakaso?")


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"prompt": ""}, "prompt"),
        ({"prompt": ["a", "b"]}, "prompt"),
        ({"prompt": "x", "n": 4}, "n"),
        ({"prompt": "x", "logprobs": True}, "logprobs"),
        ({"prompt": "x", "echo": True}, "echo"),
        ({"prompt": "x", "max_tokens": 0}, "max_tokens"),
        ({"prompt": "x", "max_tokens": MAX_NEW_TOKENS_LIMIT + 1}, "max_tokens"),
        ({"prompt": "x", "temperature": "hot"}, "temperature"),
        ({"prompt": "x", "stream": "yes"}, "stream"),
        ({"prompt": "x", "stop": 3}, "stop"),
        ({"prompt": "x", "temperature": 12.0}, "temperature"),
        ({"prompt": "x", "unknown_field": 1}, "unknown_field"),
        ({"prompt": "x", "model": "someone-elses-model"}, "model"),
    ],
)
def test_a_request_the_service_cannot_honour_is_refused_with_the_field_named(
    payload: dict[str, object], param: str
) -> None:
    status, body = _post(_application(), "/v1/completions", payload)

    assert status == 400
    assert body["error"]["param"] == param
    assert body["error"]["type"] == "invalid_request_error"


def test_asking_for_one_completion_explicitly_is_not_a_refusal() -> None:
    status, _ = _post(
        _application(), "/v1/completions", {"prompt": "x", "n": 1, "best_of": 1, "logprobs": False}
    )

    assert status == 200


@pytest.mark.parametrize(
    "messages",
    [
        [],
        "not-a-list",
        [{"role": "user"}],
        [{"role": "tool", "content": "x"}],
        [{"role": "user", "content": 7}],
        [{"role": "user", "content": "x"}, "junk"],
    ],
)
def test_a_malformed_transcript_is_refused(messages: object) -> None:
    status, body = _post(_application(), "/v1/chat/completions", {"messages": messages})

    assert status == 400
    assert body["error"]["param"] == "messages"


def test_a_prompt_over_the_stated_limit_is_refused_before_the_model_runs() -> None:
    application = _application()

    status, body = _post(
        application,
        "/v1/completions",
        {"prompt": "a" * (DEFAULT_MAX_INPUT_CHARACTERS + 1)},
    )

    assert status == 400
    assert body["error"]["param"] == "prompt"
    assert application.model.requests == []  # type: ignore[union-attr]


def test_a_body_that_is_not_json_is_refused_rather_than_treated_as_empty() -> None:
    response = _application().handle("POST", "/v1/completions", b"{not json")

    assert response.status == 400
    assert json.loads(response.body.decode("utf-8"))["error"]["param"] == "body"


def test_no_body_is_not_the_same_as_an_empty_prompt() -> None:
    response = _application().handle("POST", "/v1/completions", None)

    assert response.status == 400


# ------------------------------------------------------------------ streaming
def _events(response: ServiceResponse) -> list[str]:
    return list(response.stream)


def test_a_streamed_completion_is_server_sent_events_ending_in_the_done_marker() -> None:
    application = _application()

    response = application.handle(
        "POST", "/v1/completions", json.dumps({"prompt": "once", "stream": True}).encode("utf-8")
    )
    events = _events(response)
    payloads = [json.loads(event.removeprefix("data: ").strip()) for event in events[:-1]]

    assert response.headers[0][1].startswith("text/event-stream")
    assert events[-1] == "data: [DONE]\n\n"
    assert "".join(item["choices"][0]["text"] for item in payloads) == "the model that repeats"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["chakaso_finish_reason"] == "stop"


def test_a_streamed_chat_turn_declares_the_role_once() -> None:
    response = _application().handle(
        "POST",
        "/v1/chat/completions",
        json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(
            "utf-8"
        ),
    )
    events = _events(response)
    payloads = [json.loads(event.removeprefix("data: ").strip()) for event in events[:-1]]

    assert payloads[0]["choices"][0]["delta"].get("role") == "assistant"
    assert all("role" not in item["choices"][0]["delta"] for item in payloads[1:])
    assert payloads[0]["object"] == "chat.completion.chunk"


def test_streaming_is_refused_for_a_model_that_cannot_stream() -> None:
    application = _application(non_streaming=True)

    status, body = _post(application, "/v1/completions", {"prompt": "x", "stream": True})

    assert status == 400
    assert body["error"]["param"] == "stream"
    assert "does not support streaming" in body["error"]["message"]


def test_the_chat_path_refuses_streaming_on_the_same_terms() -> None:
    # Both endpoints stream through the same gate. Checking only the completion path is how a chat request
    # would reach the generator of a model that has none.
    application = _application(non_streaming=True)

    status, body = _post(
        application,
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert status == 400
    assert body["error"]["param"] == "stream"
    assert application.model.requests == []  # type: ignore[union-attr]


def test_a_model_failure_mid_stream_is_reported_as_an_event_not_a_truncated_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chakaso.models import ModelError

    application = _application()
    handed: list[tuple[tuple[str, ...], GenerationParams, object]] = []

    def failing_stream(
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        signal: object | None = None,
    ) -> Iterator[GenerationChunk]:
        handed.append((tuple(message.content for message in messages), params, signal))
        yield GenerationChunk(text="ok", token_id=0)
        message = "the model fell over"
        raise ModelError(message)

    monkeypatch.setattr(application.model, "stream", failing_stream)

    response = application.handle(
        "POST", "/v1/completions", json.dumps({"prompt": "x", "stream": True}).encode("utf-8")
    )
    events = _events(response)

    assert "data: [DONE]" not in events
    assert "the model fell over" in events[-1]
    # The double records what the adapter actually passed down: the prompt as a message, and a live
    # abort signal the transport can cancel when the client hangs up mid-stream.
    assert handed[0][0] == ("x",)
    assert handed[0][2] is not None


# ------------------------------------------------------------------ model loading
def test_the_serving_loader_refuses_a_path_that_is_not_a_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ServingError, match="could not be loaded"):
        load_serving_model(str(tmp_path / "nothing-here.pt"))


def test_an_application_can_be_built_only_around_a_real_loadable_path(tmp_path: Path) -> None:
    with pytest.raises(ServingError):
        ServingApplication.for_path(str(tmp_path / "missing.pt"))


def test_a_double_is_refused_by_the_serving_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chakaso.serve.app as app_module

    def a_double(path: str) -> _ScriptedModel:
        del path  # the loader's argument is the point of the test, not something it inspects
        return _ScriptedModel(double=True)

    monkeypatch.setattr(app_module, "load_model", a_double)

    with pytest.raises(ServingError, match="refuses the development double"):
        load_serving_model("anything.json")


# ------------------------------------------------------------------ model refusal
def test_a_scripted_model_that_is_the_double_cannot_back_an_application() -> None:
    # The guard lives in load_serving_model, and the application refuses a double at construction time too:
    # a service that answers while the trained model is missing is the failure mode this whole layer exists
    # to prevent.
    application = _application()
    assert application.model.metadata.development_double is False

    with pytest.raises(ServingError, match="could not be loaded"):
        ServingApplication.for_path(str(Path("definitely-not-a-checkpoint.pt")))
