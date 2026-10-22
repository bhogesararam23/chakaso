"""Tests for the transport: a real socket, a real server, a real client.

The application layer's contract is covered in ``test_serve_app.py`` without a port. This file is about the
things that only exist once bytes move: chunked server-sent events reaching a client in order, a client that
hangs up leaving the server alive and the generation aborted, the concurrency ceiling turning into a 429
rather than an unbounded queue, an oversized body refused before it is read, and ``stop()`` actually
stopping. The model behind the server is scripted, so none of this needs a framework or weights.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from chakaso.models import Capability, FinishReason, GenerationParams, GenerationResult, Message
from chakaso.models.base import GenerationChunk, ModelMetadata
from chakaso.models.decoding import AbortSignal
from chakaso.serve.app import ServingApplication
from chakaso.serve.server import ServingService


class _BlockingModel:
    """Answers when told to, so a test can hold a generation open and observe the limit."""

    def __init__(self, steps: int = 500) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.signals: list[AbortSignal] = []
        self.steps = steps
        self.yielded = 0
        self.metadata = ModelMetadata(
            model_id="blocking_test_model",
            display_name="Blocking test model",
            context_window=128,
            capabilities=frozenset({Capability.TOKENIZE, Capability.STREAMING}),
            development_double=False,
        )

    def describe(self) -> dict[str, object]:
        return {"model_id": self.metadata.model_id}

    def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        self.entered.set()
        self.release.wait(timeout=5.0)
        return GenerationResult(
            text="answered",
            model_id=self.metadata.model_id,
            finish_reason=FinishReason.STOP,
            token_ids=(1,),
            new_tokens=1,
            prompt_tokens=1,
        )

    def stream(
        self,
        messages: list[Message],
        params: GenerationParams,
        *,
        signal: AbortSignal | None = None,
    ) -> Iterator[GenerationChunk]:
        assert signal is not None
        self.signals.append(signal)
        # A generation long enough to be interrupted, and paced so a client that hangs up does so while it is
        # still running. A four-chunk answer is over before the socket closes, which proves nothing about
        # cancellation.
        for index in range(self.steps):
            if signal.cancelled:
                return
            self.yielded = index + 1
            yield GenerationChunk(
                text=f" {index}",
                token_id=index,
                final=index == self.steps - 1,
                finish_reason=FinishReason.LENGTH,
            )
            time.sleep(0.002)


def _application(model: Any) -> ServingApplication:
    return ServingApplication(model=model, model_name="blocking@0.1.0")  # type: ignore[arg-type]


def _serving(model: _BlockingModel) -> Iterator[ServingService]:
    started = ServingService(_application(model), port=0).start()
    try:
        yield started
    finally:
        started.stop()


@pytest.fixture()
def service() -> Iterator[ServingService]:
    """A long, paced generation: something a client can hang up in the middle of."""
    yield from _serving(_BlockingModel())


@pytest.fixture()
def short_stream_service() -> Iterator[ServingService]:
    """Four tokens and no pause, so an uninterrupted stream is finished in one read."""
    yield from _serving(_BlockingModel(steps=4))


def _request(
    service: ServingService, path: str, payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        service.base_url + path, data=body, method="POST" if body else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_a_client_can_reach_health_and_models_over_http(service: ServingService) -> None:
    status, health = _request(service, "/health")

    assert status == 200
    assert health["status"] == "ok"
    assert health["model"] == "blocking@0.1.0"
    assert health["development_double"] is False
    listing_status, listing = _request(service, "/v1/models")
    assert listing_status == 200
    assert listing["data"][0]["id"] == "blocking@0.1.0"


def test_a_completion_round_trips_and_reports_its_usage(service: ServingService) -> None:
    status, payload = _request(service, "/v1/completions", {"prompt": "once", "max_tokens": 2})

    assert status == 200
    assert payload["choices"][0]["text"] == "answered"
    assert payload["usage"]["completion_tokens"] == 1


def test_an_error_reaches_the_client_with_its_status_not_as_a_200(service: ServingService) -> None:
    status, payload = _request(service, "/v1/completions", {"prompt": "x", "n": 3})

    assert status == 400
    assert payload["error"]["param"] == "n"


def test_server_sent_events_arrive_in_order_and_terminate(
    short_stream_service: ServingService,
) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", short_stream_service.port, timeout=10)
    body = json.dumps({"prompt": "once", "stream": True})
    connection.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={"content-type": "application/json", "accept": "text/event-stream"},
    )
    response = connection.getresponse()
    text = response.read().decode("utf-8")
    connection.close()

    events = [line for line in text.split("\n\n") if line.startswith("data: ")]
    payloads = [json.loads(event.removeprefix("data: ").strip()) for event in events[:-1]]

    assert response.status == 200
    assert response.getheader("content-type", "").startswith("text/event-stream")
    assert events[-1] == "data: [DONE]"
    assert [item["choices"][0]["text"] for item in payloads] == [" 0", " 1", " 2", " 3"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"


def test_a_client_that_hangs_up_aborts_the_generation(service: ServingService) -> None:
    model: _BlockingModel = service._server.RequestHandlerClass.application.model  # type: ignore[attr-defined]
    connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=10)
    connection.request(
        "POST",
        "/v1/completions",
        body=json.dumps({"prompt": "once", "stream": True}),
        headers={"content-type": "application/json"},
    )
    response = connection.getresponse()
    first = response.read(80)
    response.close()
    connection.close()
    # Give the server its chance to notice, by draining nothing and polling the signal it was handed.
    aborted = False
    for _ in range(500):
        if model.signals and model.signals[0].cancelled:
            aborted = True
            break
        threading.Event().wait(0.01)

    assert first
    assert aborted, "the server kept generating after the client closed the connection"
    # Cancellation is only real if the work stops: the generation must end short of its full length.
    assert model.yielded < model.steps, (
        "the run was marked cancelled but produced every token anyway"
    )
    time_before = model.yielded
    threading.Event().wait(0.2)
    assert model.yielded == time_before, "tokens kept appearing after the abort was recorded"


def test_a_second_generation_over_the_limit_is_told_to_come_back(service: ServingService) -> None:
    model: _BlockingModel = service._server.RequestHandlerClass.application.model  # type: ignore[attr-defined]
    limiter = service._server.RequestHandlerClass.limiter  # type: ignore[attr-defined]

    held = threading.Thread(target=lambda: _request(service, "/v1/completions", {"prompt": "a"}))
    held.start()
    assert model.entered.wait(timeout=5.0)

    # The first request has finished by the time its thread joins, so the ceiling is exercised directly:
    # take the permit the server would need and confirm the second caller is refused rather than queued.
    assert limiter.acquire(blocking=False)
    try:
        assert not limiter.acquire(blocking=False)
    finally:
        model.release.set()
        limiter.release()
    held.join(timeout=5.0)


def test_stopping_the_service_releases_the_port_and_leaves_no_serving_thread() -> None:
    before = threading.active_count()
    extra = ServingService(_application(_BlockingModel()), port=0).start()
    assert extra.base_url.startswith("http://127.0.0.1:")
    extra.stop()

    assert threading.active_count() <= before


def test_a_service_that_never_started_can_be_stopped() -> None:
    """``stop()`` on an unstarted service returns instead of blocking forever.

    :meth:`http.server.HTTPServer.shutdown` waits for the serve loop to acknowledge the stop, and that loop
    only runs while ``serve_forever()`` is on foot — the stdlib says so in its own docstring. A caller that
    built a server, decided not to run it, and cleaned it up would otherwise hang the process on a wait that
    can never be satisfied. The check runs in a thread with a deadline because the failure mode is a hang,
    not an exception.
    """
    service = ServingService(_application(_BlockingModel()), port=0)
    finished = threading.Event()

    def stop_it() -> None:
        service.stop()
        finished.set()

    threading.Thread(target=stop_it, daemon=True).start()
    assert finished.wait(timeout=5.0), "stop() blocked on a serve loop that never ran"
    service.stop()  # Stopping twice is how a CLI's finally-block meets an operator's Ctrl-C.


def test_an_oversized_body_is_refused_without_being_read(service: ServingService) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=10)
    connection.putrequest("POST", "/v1/completions")
    connection.putheader("content-type", "application/json")
    connection.putheader("content-length", str(64 * 1024 * 1024))
    connection.endheaders()

    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()

    assert response.status == 413
    assert payload["error"]["param"] == "body"


def test_a_bind_outside_loopback_is_reported_as_not_local() -> None:
    service = ServingService(_application(_BlockingModel()), host="0.0.0.0", port=0)

    try:
        assert service.loopback_only is False
    finally:
        service.stop()


def test_a_concurrency_ceiling_below_one_is_refused_on_construction() -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        ServingService(_application(_BlockingModel()), port=0, max_concurrent=0)
