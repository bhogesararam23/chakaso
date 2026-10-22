"""The request handling behind the local server: pure functions over data, no socket in sight.

Two reasons this file knows nothing about HTTP. The first is testability: an endpoint's contract — what it
accepts, what it refuses, what it returns when the model is missing — is checked by calling a function, so
the serving tests need no port, no thread and no client library. The second is the boundary itself: a
request handler that also decides what a request means is where transport bugs and policy bugs hide in each
other.

What this layer guarantees:

* **It serves one named model, loaded once, and never a stand-in.** A release directory or a checkpoint is
  resolved at construction; if it is missing, corrupt, or needs an extra that is not installed, the server
  refuses to start with the reason. There is no path here that produces text without a trained model, and
  the development double is refused by name if it is what was configured.
* **A field it cannot honour is refused, not ignored.** An OpenAI-compatible subset is only honest if the
  parts it does not implement come back as errors. ``n``, ``best_of``, ``logprobs``, ``echo``, ``user`` and
  ``suffix`` are refused with the field named, because a caller that asked for five completions and got one
  would otherwise read the response as agreement.
* **Every bound is stated before it is hit.** Input length, output ceiling and the model's context window
  are checked ahead of the model, and reported by ``/health``.
* **A cancelled client is a cancelled generation.** Streaming polls for a disconnect between tokens and
  aborts through the same :class:`~chakaso.models.decoding.AbortSignal` the library exposes, so a closed
  connection does not keep the machine busy producing the rest of an answer nobody is reading.

Nothing here imports ``torch``: the model arrives already loaded, which is what lets a caller audit a
service's behaviour without paying for a framework.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chakaso.models import (
    Capability,
    FinishReason,
    GenerationParams,
    LanguageModel,
    Message,
    ModelError,
    Role,
    load_model,
    load_model_release,
)
from chakaso.models.base import GenerationChunk, StreamingGenerator
from chakaso.models.decoding import AbortSignal

__all__ = [
    "DEFAULT_MAX_INPUT_CHARACTERS",
    "DEFAULT_MAX_NEW_TOKENS",
    "MAX_NEW_TOKENS_LIMIT",
    "ServiceUnavailableError",
    "ServingApplication",
    "ServingError",
    "load_serving_model",
]

#: Characters accepted in one request before the tokenizer is consulted. A bound, not a judgement: the
#: model cannot use a long prompt, and reading megabytes into a decode is how a local service becomes a
#: denial of service against its own machine.
DEFAULT_MAX_INPUT_CHARACTERS = 32_768
#: The output ceiling a request gets when it does not name one.
DEFAULT_MAX_NEW_TOKENS = 64
#: The largest ceiling a caller may ask for, so one request cannot reserve the process for an hour.
MAX_NEW_TOKENS_LIMIT = 2_048
#: Roles a chat request may use. ``tool`` and ``function`` are excluded because nothing here executes them.
CHAT_ROLES = frozenset({"system", "user", "assistant"})

#: The OpenAI-compatible subset of request fields this service can honour.
SUPPORTED_FIELDS = frozenset(
    {
        "model",
        "prompt",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "stream",
        "n",
        "best_of",
        "logprobs",
        "echo",
        "user",
        "suffix",
    }
)

#: For each refused field, the values that mean "the caller asked for the default anyway". A request that
#: sets ``n=1`` wants one completion, which is what this service produces; ``n=5`` wants something else and
#: is refused rather than quietly answered with a single choice.
_DEFAULTS_FOR: dict[str, tuple[object, ...]] = {
    "n": (None, 1),
    "best_of": (None, 1),
    "logprobs": (None, False),
    "echo": (None, False),
    "user": (None, ""),
    "suffix": (None, ""),
}
REFUSED_WHEN_SET = frozenset(_DEFAULTS_FOR)

#: ``FinishReason`` in the OpenAI subset. ``context`` maps to ``length`` because both mean the answer was
#: cut short and is not complete; the unmapped value is still on the wire as ``chakaso_finish_reason``.
FINISH_REASON_MAP = {
    FinishReason.STOP: "stop",
    FinishReason.LENGTH: "length",
    FinishReason.CONTEXT: "length",
    FinishReason.CANCELLED: "stop",
}


@dataclass(frozen=True, slots=True)
class ServingError(Exception):
    """A request this service will not carry out, with the HTTP status it deserves.

    Attributes:
        message: What the caller did, described in terms of the request rather than the internals.
        status: 400 for a bad request, 404 for an unknown route, 500 for a service that cannot serve.
        param: The field at fault, when there is one.
    """

    message: str
    status: int = 400
    param: str | None = None

    def body(self) -> dict[str, Any]:
        """The wire form: an ``error`` object shaped like the API the caller expects."""
        return {
            "error": {
                "message": self.message,
                "type": _error_type(self.status),
                "param": self.param,
            }
        }


class ServiceUnavailableError(ServingError):
    """The service is up but cannot take this request now — busy, or the model went away."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, status=503)
        self.retry_after = retry_after


def _error_type(status: int) -> str:
    if status == 400:
        return "invalid_request_error"
    if status == 404:
        return "not_found_error"
    if status == 503:
        return "service_unavailable"
    return "server_error"


def _bad(message: str, *, param: str | None = None) -> ServingError:
    """A 400 for a request this service will not carry out."""
    return ServingError(message, status=400, param=param)


def load_serving_model(path: str) -> tuple[LanguageModel, str | None]:
    """Load the model a server instance was pointed at, refusing every substitute.

    A release directory is verified before its checkpoint is read, so what is served is what was signed off.
    A missing path, an unreadable checkpoint or an absent optional extra fails here, at startup, with the
    reason — never as a quiet downgrade to something smaller.

    Returns:
        The model, and the release identity ``id@version`` when ``path`` was a release directory.

    Raises:
        ServingError: the model could not be loaded, or what loaded is the development double.
    """
    release_identity: str | None = None
    try:
        if Path(path).is_dir():
            release, model = load_model_release(Path(path))
            release_identity = f"{release.model_id}@{release.model_version}"
        else:
            model = load_model(path)
    except (ModelError, OSError, ValueError) as exc:
        message = f"the serving model at {path!r} could not be loaded: {exc}"
        raise ServingError(message, status=500, param="model") from exc
    if model.metadata.development_double:
        message = (
            "the serving adapter refuses the development double: a local inference endpoint must answer "
            "from a trained checkpoint"
        )
        raise ServingError(message, status=500, param="model")
    return model, release_identity


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    """A handled request: status, headers, and either bytes to send or an iterator to stream."""

    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes | None = None
    stream: Iterator[str] | None = None
    #: Asks a streaming generation to stop. The transport calls it when the client goes away mid-stream,
    #: which is the only way a closed connection becomes a cancelled model run rather than wasted work.
    abort: Callable[[], None] | None = None

    @classmethod
    def json(cls, payload: object, status: int = 200) -> ServiceResponse:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        return cls(
            status=status,
            headers=(("content-type", "application/json; charset=utf-8"),),
            body=text.encode("utf-8"),
        )


@dataclass
class _Limits:
    """The bounds one service instance enforces, stated rather than discovered by failure."""

    max_input_characters: int = DEFAULT_MAX_INPUT_CHARACTERS
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_new_tokens: int = MAX_NEW_TOKENS_LIMIT


@dataclass
class ServingApplication:
    """The endpoints of ``chakaso serve``, callable as methods.

    Attributes:
        model: A loaded, non-double language model.
        model_name: What the service calls itself on the wire — the release id when there is one.
        limits: The input and output bounds it enforces.
        created: Unix timestamp of this instance, reported as the model's ``created``.
    """

    model: LanguageModel
    model_name: str
    limits: _Limits = field(default_factory=_Limits)
    created: int = field(default_factory=lambda: int(time.time()))

    @classmethod
    def for_path(
        cls,
        path: str,
        *,
        max_input_characters: int = DEFAULT_MAX_INPUT_CHARACTERS,
        default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> ServingApplication:
        """Load ``path`` and build the service around it.

        Raises:
            ServingError: the model cannot be loaded, or is not a trained model.
        """
        model, release_identity = load_serving_model(path)
        return cls(
            model=model,
            model_name=release_identity or model.metadata.model_id,
            limits=_Limits(
                max_input_characters=max_input_characters,
                default_max_new_tokens=default_max_new_tokens,
            ),
        )

    # ------------------------------------------------------------------ dispatch
    def handle(self, method: str, path: str, body: bytes | None = None) -> ServiceResponse:
        """Route one request. A refused request becomes a status body, never an escaped exception."""
        try:
            return self._dispatch(method, path, body)
        except ServingError as exc:
            return ServiceResponse.json(exc.body(), status=exc.status)

    def _dispatch(self, method: str, path: str, body: bytes | None) -> ServiceResponse:
        route = path.split("?", 1)[0].rstrip("/") or "/"
        if method == "OPTIONS":
            return ServiceResponse(status=204, headers=(("allow", "GET, POST, OPTIONS"),))
        if method not in ("GET", "POST"):
            return ServiceResponse(status=405, headers=(("allow", "GET, POST"),), body=b"")
        if route == "/health":
            return self.health()
        if route == "/v1/models":
            return self.models()
        if method != "POST":
            return self._not_found(route)
        if route == "/v1/completions":
            return self.completions(body)
        if route == "/v1/chat/completions":
            return self.chat_completions(body)
        return self._not_found(route)

    def _not_found(self, route: str) -> ServiceResponse:
        message = (
            f"no route {route!r}; this service offers /health, /v1/models, /v1/completions and "
            "/v1/chat/completions"
        )
        error = ServingError(message, status=404)
        return ServiceResponse.json(error.body(), status=error.status)

    # ------------------------------------------------------------------ endpoints
    def health(self) -> ServiceResponse:
        """Liveness, plus everything a caller must know before sending a request."""
        metadata = self.model.metadata
        return ServiceResponse.json(
            {
                "status": "ok",
                "model": self.model_name,
                "model_id": metadata.model_id,
                "development_double": metadata.development_double,
                "context_window": metadata.context_window,
                "capabilities": sorted(capability.value for capability in metadata.capabilities),
                "streaming": Capability.STREAMING in metadata.capabilities,
                "limits": {
                    "max_input_characters": self.limits.max_input_characters,
                    "default_max_new_tokens": self.limits.default_max_new_tokens,
                    "max_new_tokens": self.limits.max_new_tokens,
                },
            }
        )

    def models(self) -> ServiceResponse:
        """The single model this instance serves, in the shape a client lists models with."""
        describe = getattr(self.model, "describe", None)
        runtime: dict[str, object] = describe() if callable(describe) else {}
        listed = {key: value for key, value in runtime.items() if key != "model_id"}
        return ServiceResponse.json(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": self.created,
                        "owned_by": "chakaso",
                        "context_window": self.model.metadata.context_window,
                        "development_double": False,
                        **listed,
                    }
                ],
            }
        )

    def completions(self, body: bytes | None) -> ServiceResponse:
        """``POST /v1/completions`` — one prompt string, one continuation."""
        payload = self._payload(body)
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            message = (
                "prompt must be a non-empty string; a list of prompts is not served, because each would "
                "need its own choice index and this service produces one"
            )
            raise _bad(message, param="prompt")
        self._check_input(prompt)
        params, stream = self._params(payload)
        messages = [Message(role=Role.USER, content=prompt)]
        if stream:
            streaming = self._require_streaming()
            return self._stream_response(
                lambda signal: self._completion_chunks(streaming, messages, params, signal)
            )
        result = self._generate(messages, params)
        return ServiceResponse.json(
            {
                "id": self._identifier(),
                "object": "text_completion",
                "created": self.created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": result.text,
                        "finish_reason": FINISH_REASON_MAP[result.finish_reason],
                        "logprobs": None,
                    }
                ],
                "usage": self._usage(result),
                "chakaso": {
                    "finish_reason": result.finish_reason.value,
                    "prompt_tokens": result.prompt_tokens,
                    "truncated_prompt": result.truncated,
                    "stopped_by": result.stopped_by,
                },
            }
        )

    def chat_completions(self, body: bytes | None) -> ServiceResponse:
        """``POST /v1/chat/completions`` — messages in, one assistant turn out."""
        payload = self._payload(body)
        messages = self._messages(payload.get("messages"))
        params, stream = self._params(payload)
        if stream:
            streaming = self._require_streaming()
            return self._stream_response(
                lambda signal: self._chat_chunks(streaming, messages, params, signal)
            )
        result = self._generate(messages, params)
        return ServiceResponse.json(
            {
                "id": self._identifier(),
                "object": "chat.completion",
                "created": self.created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": FINISH_REASON_MAP[result.finish_reason],
                    }
                ],
                "usage": self._usage(result),
                "chakaso": {
                    "finish_reason": result.finish_reason.value,
                    "prompt_tokens": result.prompt_tokens,
                    "truncated_prompt": result.truncated,
                    "stopped_by": result.stopped_by,
                    "note": (
                        "The base model is not instruction tuned. Message formatting is a convention this "
                        "service applies to the prompt, not evidence that the model was trained to follow "
                        "it."
                    ),
                },
            }
        )

    # ------------------------------------------------------------------ request checks
    def _payload(self, body: bytes | None) -> dict[str, Any]:
        if body is None or not body.strip():
            message = "a JSON request body is required"
            raise _bad(message, param="body")
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            message = f"request body is not UTF-8 JSON: {exc}"
            raise _bad(message, param="body") from exc
        if not isinstance(decoded, dict):
            message = "the request body must be a JSON object"
            raise _bad(message, param="body")
        requested = decoded.get("model")
        if requested not in (None, "", self.model_name, self.model.metadata.model_id):
            message = (
                f"this instance serves {self.model_name!r} only; {requested!r} is not loaded here"
            )
            raise _bad(message, param="model")
        for field_name in sorted(set(decoded) - SUPPORTED_FIELDS):
            message = f"unsupported request field {field_name!r}"
            raise _bad(message, param=field_name)
        for field_name in sorted(REFUSED_WHEN_SET & set(decoded)):
            if decoded[field_name] in _DEFAULTS_FOR[field_name]:
                continue
            message = (
                f"{field_name!r} is not supported: this service produces one completion from one prompt, "
                "and refuses rather than ignoring a field the caller asked for"
            )
            raise _bad(message, param=field_name)
        return decoded

    def _check_input(self, text: str) -> None:
        if len(text) > self.limits.max_input_characters:
            message = (
                f"prompt is {len(text):,} characters; this service accepts at most "
                f"{self.limits.max_input_characters:,}, and the model's own window is "
                f"{self.model.metadata.context_window} tokens"
            )
            raise _bad(message, param="prompt")

    def _messages(self, raw: object) -> list[Message]:
        if not isinstance(raw, list) or not raw:
            message = "messages must be a non-empty list"
            raise _bad(message, param="messages")
        built: list[Message] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                message = f"messages[{index}] is not an object"
                raise _bad(message, param="messages")
            role = item.get("role")
            content = item.get("content")
            if role not in CHAT_ROLES:
                message = (
                    f"messages[{index}].role {role!r} is not served; this endpoint accepts "
                    f"{sorted(CHAT_ROLES)}"
                )
                raise _bad(message, param="messages")
            if not isinstance(content, str):
                message = f"messages[{index}].content must be a string"
                raise _bad(message, param="messages")
            built.append(Message(role=Role(str(role)), content=content))
        self._check_input(" ".join(item.content for item in built))
        return built

    def _params(self, payload: dict[str, Any]) -> tuple[GenerationParams, bool]:
        raw_tokens = payload.get("max_tokens", self.limits.default_max_new_tokens)
        if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int) or raw_tokens < 1:
            message = "max_tokens must be an integer >= 1"
            raise _bad(message, param="max_tokens")
        if raw_tokens > self.limits.max_new_tokens:
            message = (
                f"max_tokens {raw_tokens} exceeds this service's ceiling of "
                f"{self.limits.max_new_tokens}"
            )
            raise _bad(message, param="max_tokens")
        temperature = payload.get("temperature", 0.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            message = "temperature must be a number"
            raise _bad(message, param="temperature")
        stop_sequences = self._stops(payload.get("stop"))
        seed = payload.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            message = "seed must be an integer"
            raise _bad(message, param="seed")
        top_k = payload.get("top_k")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int)):
            message = "top_k must be an integer"
            raise _bad(message, param="top_k")
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            message = "stream must be a boolean"
            raise _bad(message, param="stream")
        try:
            params = GenerationParams(
                max_new_tokens=raw_tokens,
                temperature=float(temperature),
                top_k=top_k,
                top_p=payload.get("top_p"),
                seed=seed,
                stop_sequences=stop_sequences,
            )
        except ValueError as exc:
            # The parameter type is the single validation point for these fields; its message already
            # names the one at fault, so it is carried rather than reworded.
            raise _bad(str(exc), param=_field_from_message(str(exc))) from exc
        return params, bool(stream)

    def _stops(self, stop: object) -> tuple[str, ...]:
        if stop is None:
            return ()
        if isinstance(stop, str) and stop:
            return (stop,)
        if isinstance(stop, list):
            if any(not isinstance(item, str) or not item for item in stop):
                message = "each stop sequence must be a non-empty string"
                raise _bad(message, param="stop")
            return tuple(stop)
        message = "stop must be a non-empty string or a list of non-empty strings"
        raise _bad(message, param="stop")

    # ------------------------------------------------------------------ generation
    def _require_streaming(self) -> StreamingGenerator:
        """Refuse ``stream: true`` as a 400 before any response starts, and hand back what can do it.

        Checked up front rather than at the first chunk: a caller that asked for a stream and got a 200
        with an error event inside it cannot tell whether the service streamed and failed or never could
        have. Returning the narrowed model is what keeps the call sites honest — the streaming path is only
        reachable with a model whose ``stream`` is a callable, so a claim the object cannot honour becomes a
        refusal rather than an ``AttributeError`` inside a generator.
        """
        model = self.model
        if not (isinstance(model, StreamingGenerator) and callable(model.stream)):
            message = "the loaded model does not support streaming"
            raise _bad(message, param="stream")
        return model

    def _generate(self, messages: Sequence[Message], params: GenerationParams) -> Any:
        try:
            return self.model.generate(messages, params)
        except ModelError as exc:
            message = f"the model could not serve this request: {exc}"
            raise ServingError(message, status=500, param="model") from exc

    def _usage(self, result: Any) -> dict[str, int]:
        completion = int(getattr(result, "new_tokens", 0))
        prompt = int(getattr(result, "prompt_tokens", 0))
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def _identifier(self) -> str:
        return f"chakaso-{self.created}-{abs(hash(self.model_name)) % 10_000_000:07d}"

    def _stream_response(
        self, produce: Callable[[AbortSignal], Iterator[dict[str, Any]]]
    ) -> ServiceResponse:
        """Wrap an event generator as a server-sent-events response.

        The signal is created here and handed to the producer, so the transport owns cancellation: when the
        client goes away mid-stream, the handler aborts the generation instead of running it out.
        """
        signal = AbortSignal()

        def events() -> Iterator[str]:
            try:
                for payload in produce(signal):
                    yield f"data: {json.dumps(payload, sort_keys=True, ensure_ascii=False)}\n\n"
            except ServingError as exc:
                body = json.dumps(exc.body(), sort_keys=True)
                yield f"data: {body}\n\n"
                return
            except ModelError as exc:
                message = f"the model stopped mid-generation: {exc}"
                body = json.dumps(ServingError(message).body(), sort_keys=True)
                yield f"data: {body}\n\n"
                return
            yield "data: [DONE]\n\n"

        return ServiceResponse(
            status=200,
            headers=(
                ("content-type", "text/event-stream; charset=utf-8"),
                ("cache-control", "no-cache"),
                ("x-accel-buffering", "no"),
            ),
            stream=events(),
            abort=signal.abort,
        )

    def _completion_chunks(
        self,
        streaming: StreamingGenerator,
        messages: Sequence[Message],
        params: GenerationParams,
        signal: AbortSignal,
    ) -> Iterator[dict[str, Any]]:
        for chunk in self._stream(streaming, messages, params, signal):
            yield self._chunk_payload(chunk, delta={"content": chunk.text})

    def _chat_chunks(
        self,
        streaming: StreamingGenerator,
        messages: Sequence[Message],
        params: GenerationParams,
        signal: AbortSignal,
    ) -> Iterator[dict[str, Any]]:
        first = True
        for chunk in self._stream(streaming, messages, params, signal):
            delta: dict[str, Any] = {"content": chunk.text}
            if first:
                delta["role"] = "assistant"
                first = False
            yield self._chunk_payload(chunk, delta=delta)

    def _stream(
        self,
        streaming: StreamingGenerator,
        messages: Sequence[Message],
        params: GenerationParams,
        signal: AbortSignal,
    ) -> Iterator[GenerationChunk]:
        yield from streaming.stream(messages, params, signal=signal)

    def _chunk_payload(self, chunk: Any, *, delta: dict[str, Any]) -> dict[str, Any]:
        finish_reason = getattr(chunk, "finish_reason", None)
        kind = "chat.completion.chunk" if "role" in delta else "text_completion.chunk"
        payload: dict[str, Any] = {
            "id": self._identifier(),
            "object": kind,
            "created": self.created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": delta, "text": delta.get("content", "")}],
        }
        if chunk.final and finish_reason is not None:
            payload["choices"][0]["finish_reason"] = FINISH_REASON_MAP[finish_reason]
            payload["chakaso_finish_reason"] = finish_reason.value
        return payload


def _field_from_message(message: str) -> str | None:
    for name in ("max_new_tokens", "temperature", "top_k", "top_p", "seed", "stop_sequences"):
        if name in message:
            return name
    return None
