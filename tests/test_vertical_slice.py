"""The whole developer sequence, in one file, over a model this repository trained here.

Node N of the foundation release is not a unit test of any one part: it is the claim that a developer can
prepare a corpus, train from random initialization, save, load through the public API, generate under stated
bounds, stream and cancel that stream, score a text, verify a release and serve the result over HTTP — with no
terminal, no configuration file and no import of the conversation runtime (ADR-0033). Every one of those steps
has its own focused tests elsewhere; what this file adds is the sequence, because the ways a release fails in
practice are at the joins: a checkpoint the loader cannot rebuild, a report the release builder misreads, an
application that starts and then cannot answer.

The training run here is deliberately tiny — twelve steps of a one-layer model over five hand-written
documents — so what is asserted is *shape*: that numbers come back, that the artifact carries its own
tokenizer, that bounded decoding honours its bound, that cancellation stops the work, and that the HTTP
surface reports the model it loaded rather than a stand-in. Nothing in this file is evidence about capability,
and a passing slice does not make the model useful.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

from tests.test_training_subword_experiment import _plan, _root

from chakaso.models import (
    Capability,
    GenerationParams,
    Message,
    Role,
    load_model,
)
from chakaso.models.decoding import AbortSignal
from chakaso.models.release import build_release, load_model_release, write_release
from chakaso.serve.app import ServingApplication, ServingError
from chakaso.serve.server import ServingService
from chakaso.training.nn_checkpoint import checkpoint_filename
from chakaso.training.subword_experiment import run_experiment, write_report


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """One tiny real run, released and ready, shared by every stage of the slice."""
    root = _root(tmp_path_factory.mktemp("slice-repo"))
    work = tmp_path_factory.mktemp("slice-work")
    plan = _plan(root, work, model_family="decoder", character_baseline=False, vocab_size=268)

    report = run_experiment(plan)
    write_report(report, work / "report.json")
    checkpoint = work / checkpoint_filename(plan.experiment_id, "decoder")
    assert checkpoint.is_file(), "the run did not write the artifact its report names"

    release = build_release(
        work,
        model_id="chakaso-slice",
        model_version="0.1.0",
        summary="A twelve-step decoder trained on five hand-written documents, for the slice.",
        capabilities=("tokenize", "generate", "stream", "score"),
        known_limitations=(
            "Trained for twelve steps on synthetic fixtures; its output is not language.",
        ),
        checkpoint_name=checkpoint.name,
        licence_model="Apache-2.0",
        licence_training_data="Self-authored documents in this repository's test fixtures",
        redistribution="A test fixture; not a distribution.",
        created_utc="2026-10-22T00:00:00Z",
        reproduce="python -m pytest tests/test_vertical_slice.py",
    )
    write_release(release, work)

    return {
        "work": work,
        "checkpoint": checkpoint,
        "release": release,
        "report": report,
        "model": load_model(str(checkpoint)),
    }


def test_the_artifact_loads_and_declares_itself_not_the_double(trained: dict[str, object]) -> None:
    model = trained["model"]
    metadata = model.metadata  # type: ignore[union-attr]

    assert metadata.development_double is False
    assert Capability.TOKENIZE in metadata.capabilities
    assert Capability.STREAMING in metadata.capabilities
    # The tokenizer travels inside the artifact: a caller who has the file has everything inference needs.
    assert len(model.tokenize("the fox")) > 0  # type: ignore[union-attr]


def test_the_loaded_model_can_be_asked_what_it_is(trained: dict[str, object]) -> None:
    model = trained["model"]
    described = model.describe()  # type: ignore[union-attr]

    assert described["model_id"] == model.metadata.model_id  # type: ignore[union-attr]
    assert described["parameter_count"] == trained["report"]["transformer"]["parameter_count"]  # type: ignore[index]


def test_generation_honours_its_bound_and_is_repeatable_at_zero_temperature(
    trained: dict[str, object],
) -> None:
    model = trained["model"]
    messages = [Message(role=Role.USER, content="The fox")]
    params = GenerationParams(max_new_tokens=12, temperature=0.0)

    first = model.generate(messages, params)  # type: ignore[union-attr]
    second = model.generate(messages, params)  # type: ignore[union-attr]

    assert first.text == second.text
    assert first.new_tokens <= 12
    assert first.prompt_tokens > 0
    assert len(first.token_ids) == first.new_tokens
    assert first.finish_reason.value in {"stop", "length", "context"}
    assert first.model_id == model.metadata.model_id  # type: ignore[union-attr]


def test_a_sampled_generation_is_the_seed_of_its_run_and_no_one_elses(
    trained: dict[str, object],
) -> None:
    """One seed, drawn twice, gives the same text — which is what makes a sample reportable at all.

    What is *not* claimed here is that two different seeds give two different texts. At twelve steps this
    model's distribution is close to degenerate, so an inequality assertion would be a coin flip about
    fixture noise rather than a property worth pinning.
    """
    model = trained["model"]
    messages = [Message(role=Role.USER, content="The fox")]
    params = GenerationParams(max_new_tokens=8, temperature=0.8, seed=11)

    assert (
        model.generate(messages, params).text
        == model.generate(  # type: ignore[union-attr]
            messages, params
        ).text
    )


def test_a_stream_stops_producing_when_the_caller_cancels(trained: dict[str, object]) -> None:
    model = trained["model"]
    messages = [Message(role=Role.USER, content="The fox")]
    params = GenerationParams(max_new_tokens=32, temperature=0.0)
    signal = AbortSignal()

    seen = 0
    for chunk in model.stream(messages, params, signal=signal):  # type: ignore[union-attr]
        seen += 1
        assert chunk.text is not None
        if seen == 2:
            signal.abort()

    assert 2 <= seen < 32, "the stream ran to its ceiling after the caller cancelled it"


def test_scoring_a_text_returns_a_number_the_trainer_would_recognize(
    trained: dict[str, object],
) -> None:
    text = "The fox ran over the dry sand and the quiet owl slept in the town." * 6

    score = trained["model"].score(text)  # type: ignore[union-attr]

    assert isinstance(score, float)
    assert score == score and score > 0.0


def test_the_release_directory_is_the_thing_a_consumer_can_load(trained: dict[str, object]) -> None:
    release, model = load_model_release(trained["work"])  # type: ignore[arg-type]

    assert release.checksum == trained["release"].checksum  # type: ignore[attr-defined]
    assert model.metadata.model_id == trained["model"].metadata.model_id  # type: ignore[union-attr]
    assert release.known_limitations, "a release without a stated limitation is not a release"


def test_the_http_adapter_answers_from_the_trained_model_without_a_terminal(
    trained: dict[str, object],
) -> None:
    """The serving layer, over a real socket, in front of the artifact this file just trained.

    The serving unit tests script the model because they are about request logic, and a scripted model cannot
    show that a checkpoint loaded through the release path actually produces a completion body. This can.
    """
    application = ServingApplication.for_path(str(trained["work"]))
    service = ServingService(application, port=0).start()
    try:
        with urllib.request.urlopen(f"{service.base_url}/health", timeout=10) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["development_double"] is False

        request = urllib.request.Request(
            f"{service.base_url}/v1/completions",
            data=json.dumps({"prompt": "The fox", "max_tokens": 6, "temperature": 0.0}).encode(
                "utf-8"
            ),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))

        assert body["object"] == "text_completion"
        assert body["choices"][0]["index"] == 0
        assert isinstance(body["choices"][0]["text"], str)
        assert body["usage"]["completion_tokens"] <= 6
        assert body["model"] == application.model_name
        # The project's own facts travel beside the compatible ones, so a caller can tell what actually
        # ended the run rather than inferring it from the OpenAI-shaped label.
        assert body["chakaso"]["finish_reason"] in {"stop", "length", "context", "cancelled"}
        assert body["chakaso"]["prompt_tokens"] > 0
    finally:
        service.stop()


def test_a_path_that_is_not_a_loadable_model_is_refused_at_startup(tmp_path: Path) -> None:
    """An application is only buildable around something that loads — the refusal happens before a port exists.

    The companion claim, that a *loadable* development double is refused too, is made in the serving suite
    against a model that reports itself as the double; here the point is that a bad path is a named startup
    error rather than a server that answers 500s.
    """
    with pytest.raises(ServingError, match="could not be loaded"):
        ServingApplication.for_path(str(tmp_path / "absent.pt"))


def test_loading_a_model_does_not_import_the_conversation_runtime(
    trained: dict[str, object],
) -> None:
    """ADR-0033's library-first rule, checked in a fresh interpreter rather than asserted in prose.

    The claim worth testing is that reaching a checkpoint does not drag the conversation stack in with it:
    retrieval, the evidence layer and the application graph stay consumers of the model, not requirements of
    it, so an integrator who only wants generation inherits nothing else.
    """
    probe = (
        "import sys, chakaso.models;"
        f"m = chakaso.models.load_model({str(trained['checkpoint'])!r});"
        "assert not m.metadata.development_double;"
        "banned = [k for k in sys.modules"
        " if k.startswith(('chakaso.app', 'chakaso.retrieval', 'chakaso.conversation'))];"
        "print('BANNED:' + ','.join(banned))"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("BANNED:"), result.stdout
