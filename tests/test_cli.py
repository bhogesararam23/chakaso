"""Tests for the command line interface.

The CLI is thin and should stay thin. These tests cover the behaviour a user
depends on: the version is reported, help works, and unknown input fails loudly
rather than silently doing nothing.
"""

from __future__ import annotations

import io
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chakaso import __version__
from chakaso.answers import EvaluationProvenance, GroundedClaim, SQLiteAnswerStore
from chakaso.claims.model import Claim, ClaimStatus
from chakaso.cli import _format_provenance, _report_caveats, build_parser, main
from chakaso.config import load_config
from chakaso.conversation import Conversation, ConversationManager
from chakaso.core.identifiers import ChunkId
from chakaso.models import (
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    ModelMetadata,
)
from chakaso.retrieval import ingest_file

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs" / "default.toml"

SESSION = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


class _ScriptedModel:
    """A model that returns prepared text, for exercising CLI diagnostics.

    The CLI's only real adapter is the deterministic double, which never emits an
    evidence reference or a length-terminated reply. Those reporting paths still need
    covering, and a scripted model is how they are reached.
    """

    def __init__(self, answer: str, *, finish_reason: FinishReason = FinishReason.STOP) -> None:
        self._answer = answer
        self._finish_reason = finish_reason

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(model_id="scripted", display_name="Scripted", context_window=128)

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        return GenerationResult(
            text=self._answer, model_id="scripted", finish_reason=self._finish_reason
        )


def _reply(answer: str, *, finish_reason: FinishReason = FinishReason.STOP):
    manager = ConversationManager(
        _ScriptedModel(answer, finish_reason=finish_reason),
        Conversation(conversation_id="conv_cli"),
        now=lambda: SESSION,
    )
    return manager.send("hello")


def test_version_flag_prints_version_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"chakaso {__version__}"


def test_help_flag_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "usage: chakaso" in capsys.readouterr().out


def test_no_arguments_prints_help_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    # Running the tool with no arguments is a request to find out what it does,
    # not an error.
    assert main([]) == 0
    assert "usage: chakaso" in capsys.readouterr().out


def test_unknown_command_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["definitely-not-a-command"])

    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_info_reports_version_python_and_platform(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out

    assert f"chakaso {__version__}" in out
    assert platform.python_version() in out
    assert sys.implementation.name in out
    assert platform.system() in out


def test_parser_does_not_require_a_command() -> None:
    # Guards against a subcommand being made mandatory, which would break the
    # no-arguments behaviour the tests above rely on.
    parser = build_parser()
    assert parser.parse_args([]).command is None


def test_config_without_an_action_prints_its_own_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["config"]) == 0

    out = capsys.readouterr().out
    assert "usage: chakaso config" in out
    assert "show" in out


def test_config_show_prints_defaults_with_their_source(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["config", "show"]) == 0

    out = capsys.readouterr().out
    assert "model.adapter" in out
    assert "deterministic" in out
    assert "built-in default" in out
    assert "No configuration file was read" in out


def test_config_show_reports_the_file_a_value_came_from(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[model]\nmax_new_tokens = 42\n", encoding="utf-8")

    assert main(["config", "show", "--config", str(config_file)]) == 0

    out = capsys.readouterr().out
    assert "42" in out
    assert str(config_file) in out
    assert "built-in default" in out


def test_config_show_layers_repeated_config_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "base.toml"
    base.write_text("[model]\ntemperature = 0.25\n", encoding="utf-8")
    local = tmp_path / "local.toml"
    local.write_text("[model]\ntemperature = 0.75\n", encoding="utf-8")

    assert main(["config", "show", "--config", str(base), "--config", str(local)]) == 0

    out = capsys.readouterr().out
    assert "0.75" in out
    assert "0.25" not in out


def test_config_show_reports_a_bad_file_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[model]\nadaptor = 'typo'\n", encoding="utf-8")

    assert main(["config", "show", "--config", str(config_file)]) == 1

    captured = capsys.readouterr()
    assert "model.adaptor" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_config_show_reports_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.toml"

    assert main(["config", "show", "--config", str(missing)]) == 1


def test_config_show_accepts_the_shipped_default_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The file documented in configs/ must be loadable by the command documented
    # in docs/getting-started.md.
    assert main(["config", "show", "--config", str(DEFAULT_CONFIG_FILE)]) == 0

    out = capsys.readouterr().out
    assert load_config(DEFAULT_CONFIG_FILE).config.model.adapter in out


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


def test_chat_with_a_message_prints_the_reply_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The reply goes to stdout and nothing else does, so the command stays usable in
    # a pipeline.
    assert main(["chat", "--message", "What does the specification say?"]) == 0

    captured = capsys.readouterr()
    assert "Deterministic double" in captured.out
    assert "1 message(s)" in captured.out
    # The reply appears once, on stdout. The notice on stderr names the engine as
    # well, so the assertion is about the reply rather than about the word
    # "Deterministic".
    assert "1 message(s)" not in captured.err
    assert len(captured.out.strip().splitlines()) == 1


def test_chat_states_that_the_engine_is_not_a_language_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The command's honesty is a feature of it, not decoration: without this, a shell
    # that replies fluently invites exactly the wrong conclusion.
    main(["chat", "--message", "hello"])

    err = capsys.readouterr().err
    assert "not a language model" in err
    assert "does not answer anything" in err


def test_chat_reports_the_grounding_and_persistence_that_are_active(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The notice is honest about what the default configuration actually does: retrieval is
    # selected but, with no documents supplied, it grounds nothing; the in-memory store keeps
    # answers for this process only. Neither is mistaken for a capability that fired.
    main(["chat", "--message", "hello"])

    err = capsys.readouterr().err
    assert "retrieval: lexical" in err
    assert "no documents were supplied" in err
    assert "not saved to disk" in err


def test_chat_explains_why_a_turn_retrieved_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A greeting is not a knowledge request: the planner chooses no retrieval and says so, and
    # the reason the rules produced is surfaced rather than hidden.
    main(["chat", "--message", "hello"])

    err = capsys.readouterr().err
    assert "no retrieval this turn (planner chose none)" in err


def test_chat_grounds_a_turn_over_a_supplied_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # With a document in the corpus a knowledge turn retrieves it. The reply is still the
    # double's fixed text, but the grounding note reports the evidence actually offered and is
    # explicit that a citation is structure, not proof.
    document = tmp_path / "spec.md"
    document.write_text(
        "# Purpose\n\nChakaso grounds answers in retrieved evidence.\n", encoding="utf-8"
    )

    assert main(["chat", "--file", str(document), "--message", "What is Chakaso?"]) == 0

    err = capsys.readouterr().err
    assert "retrieval: lexical over 1 source(s)" in err
    assert "retrieved via lexical" in err
    assert "proves the answer" in err


def test_chat_no_retrieval_disables_grounding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["chat", "--no-retrieval", "--message", "hello"])

    err = capsys.readouterr().err
    assert "retrieval disabled" in err
    assert "overridden from configuration: retrieval.enabled=false" in err


def test_chat_reports_a_strategy_override(capsys: pytest.CaptureFixture[str]) -> None:
    # An explicit strategy is honoured and named; the notice is clear that dense runs over the
    # non-semantic fixture embedding rather than pretending semantic retrieval.
    assert main(["chat", "--strategy", "dense", "--message", "hello"]) == 0

    err = capsys.readouterr().err
    assert "retrieval.strategy='dense' (--strategy)" in err
    assert "retrieval: dense" in err


def test_chat_rejects_a_non_positive_top_k(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["chat", "--top-k", "0", "--message", "hello"]) == 1
    assert "--top-k must be at least 1" in capsys.readouterr().err


def test_chat_reports_a_missing_corpus_file_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.md"

    assert main(["chat", "--file", str(missing), "--message", "hello"]) == 1

    err = capsys.readouterr().err
    assert "does not exist" in err
    assert "Traceback" not in err


def test_chat_records_answers_to_a_durable_store(tmp_path: Path) -> None:
    # The configured store is the one actually used: a knowledge turn persists an answer to the
    # SQLite backend, and it survives closing the application (the durable store is not silently
    # replaced by the in-memory default).
    from chakaso.answers import SQLiteAnswerStore

    db = tmp_path / "answers.db"
    config = tmp_path / "config.toml"
    config.write_text(
        f'[storage]\nbackend = "sqlite"\npath = "{db.as_posix()}"\n',
        encoding="utf-8",
    )

    assert main(["chat", "--config", str(config), "--message", "What is Chakaso?"]) == 0

    with SQLiteAnswerStore(db, initialize=False) as store:
        assert store.list()
        assert store.list()[0].provenance["retrieval_strategy"] in {"lexical", "none"}


def test_chat_names_the_recorded_answer_identifier_so_it_can_be_reassessed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Reassessment addresses a stored answer by its ``ans_…`` identifier, so the shell must surface
    # it or the feature is unreachable in practice. The identifier is reported on stderr, keeping
    # stdout the reply alone.
    document = tmp_path / "spec.md"
    document.write_text("# P\n\nChakaso grounds answers.\n", encoding="utf-8")

    assert main(["chat", "--file", str(document), "--message", "What is Chakaso?"]) == 0

    err = capsys.readouterr().err
    assert "recorded answer ans_" in err
    assert "reassess" in err


def test_chat_routes_an_answer_naming_follow_up_into_a_reassessment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The whole P21 path is reachable from the supported runtime interface: a first turn records a
    # durable answer, and a later follow-up naming that identifier composes a reassessment instead of
    # an ordinary reply — and the original answer is still the only answer in the store.
    from chakaso.answers import SQLiteAnswerStore

    document = tmp_path / "spec.md"
    document.write_text("# P\n\nChakaso grounds answers in retrieved evidence.\n", "utf-8")
    db = tmp_path / "answers.db"
    config = tmp_path / "config.toml"
    config.write_text(
        f'[storage]\nbackend = "sqlite"\npath = "{db.as_posix()}"\n', encoding="utf-8"
    )

    assert (
        main(
            [
                "chat",
                "--config",
                str(config),
                "--file",
                str(document),
                "--message",
                "What is Chakaso?",
            ]
        )
        == 0
    )
    with SQLiteAnswerStore(db, initialize=False) as store:
        answer_id = str(store.list()[0].answer_id)
    capsys.readouterr()  # discard the first command's output

    assert (
        main(
            [
                "chat",
                "--config",
                str(config),
                "--file",
                str(document),
                "--message",
                f"reconsider {answer_id}",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"reassessment of {answer_id}" in captured.out
    assert "decision" in captured.out
    assert "no revised wording" in captured.err
    assert "no language model" in captured.err
    with SQLiteAnswerStore(db, initialize=False) as store:
        assert len(store.list()) == 1  # a reassessment never appended a rewritten answer


def test_chat_rejects_an_empty_message_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["chat", "--message", "   "]) == 1

    captured = capsys.readouterr()
    assert "must contain text" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_chat_reports_an_unknown_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[model]\nadapter = 'nope'\n", encoding="utf-8")

    assert main(["chat", "--config", str(config_file), "--message", "hello"]) == 1

    err = capsys.readouterr().err
    assert "nope" in err
    assert "deterministic" in err


def test_chat_reports_a_bad_config_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[model\nadapter =", encoding="utf-8")

    assert main(["chat", "--config", str(config_file), "--message", "hello"]) == 1
    assert "not valid TOML" in capsys.readouterr().err


def test_chat_accepts_the_shipped_default_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["chat", "--config", str(DEFAULT_CONFIG_FILE), "--message", "hello"]) == 0
    assert "Deterministic double" in capsys.readouterr().out


def test_chat_reads_messages_from_stdin_until_end_of_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("first question\nsecond question\n"))

    assert main(["chat"]) == 0

    out = capsys.readouterr().out
    assert out.count("chakaso> ") == 2


def test_chat_stops_at_a_quit_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("hello\n:quit\nunreachable\n"))

    assert main(["chat"]) == 0

    out = capsys.readouterr().out
    assert out.count("chakaso> ") == 1


def test_chat_ignores_blank_lines_rather_than_sending_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The double reports how many messages it received, which is how a test can tell
    # that the blank line never became a turn.
    monkeypatch.setattr("sys.stdin", io.StringIO("a\n\n\nb\n"))

    assert main(["chat"]) == 0

    captured = capsys.readouterr()
    assert "3 message(s)" in captured.out
    assert "must contain text" not in captured.err


def test_chat_survives_a_failed_turn_and_keeps_the_conversation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A failed turn leaves the conversation unchanged, so the session continues rather
    # than losing everything said so far.
    monkeypatch.setattr("sys.stdin", io.StringIO("   \nhello\n"))

    assert main(["chat"]) == 0
    assert "1 message(s)" in capsys.readouterr().out


def test_chat_reports_a_truncated_reply_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    _report_caveats(_reply("cut off here", finish_reason=FinishReason.LENGTH))

    captured = capsys.readouterr()
    assert "length ceiling" in captured.err
    assert captured.out == ""


def test_chat_reports_an_unresolved_reference_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The rule ADR-0003 and ADR-0009 enforce: a reference the model was not given is
    # never resolved, and the user is told rather than left to assume it was checked.
    _report_caveats(_reply("The deadline is in March [src_0123456789abcdef]."))

    captured = capsys.readouterr()
    assert "referenced evidence it was not given" in captured.err
    assert "src_0123456789abcdef" in captured.err


def test_chat_reports_malformed_references_separately(capsys: pytest.CaptureFixture[str]) -> None:
    _report_caveats(_reply("See [src_0123]."))

    err = capsys.readouterr().err
    assert "malformed evidence references" in err
    assert "referenced evidence it was not given" not in err


def test_chat_says_nothing_extra_about_a_clean_reply(capsys: pytest.CaptureFixture[str]) -> None:
    _report_caveats(_reply("An ordinary answer."))

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


THREE_SECTIONS = "# A\n\nalpha.\n\n# B\n\nbeta.\n\n# C\n\ngamma.\n"


def test_retrieve_prints_ranked_evidence_with_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Deadlines\n\nThe window closes in April.\n", encoding="utf-8")

    assert main(["retrieve", "--query", "closes April", "--file", str(document)]) == 0

    out = capsys.readouterr().out
    assert "retriever: lexical" in out
    assert "1. score=" in out
    assert "The window closes in April." in out
    # The source reference is a file URI, not a fabricated URL.
    assert "file:///" in out


def test_retrieve_says_so_when_nothing_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An empty retrieval is a clean outcome, not an error, and is reported as such.
    document = tmp_path / "notes.md"
    document.write_text("# Deadlines\n\nThe window closes in April.\n", encoding="utf-8")

    assert main(["retrieve", "--query", "quantum tunnelling", "--file", str(document)]) == 0
    assert "no evidence matched the query" in capsys.readouterr().out


def test_retrieve_honours_top_k(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    document = tmp_path / "many.md"
    document.write_text(THREE_SECTIONS, encoding="utf-8")

    assert (
        main(["retrieve", "--query", "alpha beta gamma", "--file", str(document), "--top-k", "1"])
        == 0
    )
    assert capsys.readouterr().out.count("   source:") == 1

    assert (
        main(["retrieve", "--query", "alpha beta gamma", "--file", str(document), "--top-k", "5"])
        == 0
    )
    assert capsys.readouterr().out.count("   source:") == 3


def test_retrieve_needs_at_least_one_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["retrieve", "--query", "anything"]) == 1
    assert "at least one --file" in capsys.readouterr().err


def test_retrieve_reports_a_missing_file_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.md"

    assert main(["retrieve", "--query", "x", "--file", str(missing)]) == 1

    err = capsys.readouterr().err
    assert "does not exist" in err
    assert "Traceback" not in err


def test_retrieve_refuses_an_unsupported_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = tmp_path / "plan.docx"
    document.write_bytes(b"binary content")

    assert main(["retrieve", "--query", "x", "--file", str(document)]) == 1
    assert "unsupported" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_benchmark_prints_a_development_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "retrieval"]) == 0

    out = capsys.readouterr().out
    assert "Retrieval benchmark report" in out
    # The report labels itself a development instrument so a number cannot be read as a
    # general-capability claim.
    assert "Development benchmark" in out
    assert "Recall@k:" in out


def test_benchmark_json_is_machine_readable_and_labels_itself_development(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "retrieval", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"]["kind"] == "development"
    assert payload["benchmark"]["note"]
    assert "recall_at_k" in payload["metrics"]


def test_benchmark_output_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    main(["benchmark", "retrieval", "--json"])
    first = capsys.readouterr().out
    main(["benchmark", "retrieval", "--json"])
    second = capsys.readouterr().out

    assert first == second


def test_benchmark_honours_top_k(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "retrieval", "--top-k", "1", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"]["top_k"] == 1


def test_benchmark_correction_reports_the_decision_rule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "correction"]) == 0

    out = capsys.readouterr().out
    assert "Correction benchmark report" in out
    assert "[PASS] retain-supported" in out


def test_benchmark_correction_json_is_deterministic_and_development(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "correction", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    main(["benchmark", "correction", "--json"])
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert first["benchmark"]["subject"] == "correction-decision-rule"
    assert first["metrics"]["decision_accuracy"] == 1.0


def test_benchmark_without_a_subject_asks_for_one(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark"]) == 1
    err = capsys.readouterr().err
    assert "retrieval" in err
    assert "correction" in err


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------


def test_experiment_run_correction_json_is_a_reproducible_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["experiment", "run", "correction", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    main(["experiment", "run", "correction", "--json"])
    second = json.loads(capsys.readouterr().out)

    # The reproducibility fingerprint is stable across runs; only the (excluded-from-id) wall-clock
    # timestamp and host differ, so the full records are not byte-identical but the result_id is.
    assert first["result_id"] == second["result_id"]
    assert first["experiment"]["experiment_id"] == "dev-correction-baseline"
    assert first["metrics"]["decision_accuracy"] == 1.0


def test_experiment_run_retrieval_prints_a_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["experiment", "run", "retrieval"]) == 0
    out = capsys.readouterr().out

    assert out.startswith("experiment dev-retrieval-baseline: result ")
    assert "on dev-retrieval" in out


def test_experiment_without_an_action_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["experiment"]) == 1
    assert "run" in capsys.readouterr().err


def test_experiment_model_action_is_documented(capsys: pytest.CaptureFixture[str]) -> None:
    # The CLI is a secondary interface, so the one thing it must get right is saying what it runs: a
    # small model on the project's own text, and not a claim of capability.
    with pytest.raises(SystemExit) as caught:  # argparse prints help and exits
        main(["experiment", "model", "--help"])
    assert caught.value.code == 0

    # argparse wraps help text, so the limitation sentence is read with its line breaks collapsed.
    printed = " ".join(capsys.readouterr().out.split())
    assert "--work-dir" in printed
    assert "not a capable language model, and not a real-world workload" in printed


def test_experiment_model_requires_a_work_directory() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["experiment", "model"])
    assert caught.value.code == 2


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="the model experiment lives behind [train] and [nn]",
)
def _model_experiment_root(tmp_path: Path) -> Path:
    """A synthetic repository the model driver can train on.

    Big enough that the default 128-token context yields at least one packed block per side, which is what
    the driver's packing layer refuses to do silently if it does not.
    """
    root = tmp_path / "repo"
    for index in range(40):
        target = root / "docs" / f"doc{index}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "# Document "
            + str(index)
            + " prose about topic "
            + str(index)
            + " that repeats, and repeats, and repeats again across the sentence.\n"
            + "A second line about the same topic with other words that repeat as well.\n",
            encoding="utf-8",
        )
    # The driver's include spec names these files exactly, so a corpus root has to carry them.
    (root / "AGENTS.md").write_text(
        "# Notes\n\nShort notes that repeat about the repository.\n", encoding="utf-8"
    )
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    return root


def _short_model_run_arguments(root: Path, work: Path, *extra: str) -> list[str]:
    return [
        "experiment",
        "model",
        "--root",
        str(root),
        "--work-dir",
        str(work),
        "--max-steps",
        "12",
        "--eval-every",
        "4",
        "--warmup-steps",
        "2",
        "--seed",
        "5",
        *extra,
    ]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="the model experiment lives behind [train] and [nn]",
)
def test_experiment_model_runs_the_pipeline_through_the_library_api(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through the CLI: the handler must delegate, not reimplement."""
    root = _model_experiment_root(tmp_path)
    work = tmp_path / "work"
    report_path = tmp_path / "results.json"

    assert main(_short_model_run_arguments(root, work, "--write-report", str(report_path))) == 0

    out = capsys.readouterr()
    assert "bits/char" in out.out
    assert "not language capability" in out.err
    assert report_path.is_file()
    assert "report digest" in out.err
    # A measured run leaves its input behind, or the next checkout cannot rerun it.
    assert (work / "chakaso-repo-text-v1-snapshot.json").is_file()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="the model experiment lives behind [train] and [nn]",
)
def test_experiment_model_trains_on_the_snapshot_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--corpus-snapshot` is the reproduction path: the run reads those bytes, not the tracked files."""
    from chakaso.training.manifest import build_manifest, write_snapshot

    root = _model_experiment_root(tmp_path)
    pinned = build_manifest(
        root,
        name="chakaso-repo-text-v1",
        version="1.0.0",
        licence="Apache-2.0",
        source="built by tests",
        include=("docs/", "*.md"),
        extensions=(".md",),
        exact_paths=("LICENSE", "AGENTS.md"),
    )
    snapshot = tmp_path / "recorded-snapshot.json"
    write_snapshot(pinned, snapshot)
    capsys.readouterr()

    assert (
        main(
            _short_model_run_arguments(
                root,
                tmp_path / "work",
                "--corpus-snapshot",
                str(snapshot),
                "--expected-corpus-fingerprint",
                pinned.fingerprint,
                "--split-seed",
                "3",
            )
        )
        == 0
    )

    # The declared snapshot is what ran, and no second copy of it was written into the work directory.
    assert not (tmp_path / "work" / "chakaso-repo-text-v1-snapshot.json").exists()
    assert (tmp_path / "work" / "chakaso-repo-text-v1-manifest.json").is_file()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="the model experiment lives behind [train] and [nn]",
)
def test_experiment_model_refuses_a_corpus_it_was_not_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A declared fingerprint is a requirement, checked before any training step is taken."""
    assert (
        main(
            _short_model_run_arguments(
                _model_experiment_root(tmp_path),
                tmp_path / "work",
                "--expected-corpus-fingerprint",
                "0" * 16,
            )
        )
        == 1
    )

    err = capsys.readouterr().err
    assert "requires corpus" in err
    assert not (tmp_path / "work" / "E2605-transformer.pt").exists()


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def _corpus_repository(root: Path) -> str:
    """Build a throwaway repository holding one pinned corpus, and return its first commit."""
    for name, text in (
        ("docs/a.md", "Alpha document.\n"),
        ("docs/b.md", "Beta document, longer than alpha.\n"),
    ):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# Notes\n\nShort notes.\n", encoding="utf-8")
    for arguments in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Corpus test"],
        ["config", "core.autocrlf", "false"],
        ["add", "."],
        ["commit", "-q", "-m", "the corpus as pinned"],
    ):
        subprocess.run(["git", *arguments], cwd=str(root), check=True, capture_output=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(root), check=True, capture_output=True, text=True
    ).stdout.strip()
    (root / "docs/a.md").write_text("Alpha document, edited afterwards.\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-aqm", "a documentation edit"], cwd=str(root), check=True)
    return head


@pytest.fixture()
def corpus_cli(tmp_path: Path) -> tuple[Path, Path, str]:
    """A pinned manifest, the repository it was built from, and the revision that recorded it."""
    if shutil.which("git") is None:  # pragma: no cover - every supported runner has git
        pytest.skip("corpus acquisition reads history, and git is not on PATH")
    from chakaso.training.manifest import build_manifest, save_manifest

    root = tmp_path / "repo"
    recorded = _corpus_repository(root)
    # Pin the manifest as the *recorded* revision held it, not as the edited tree holds it: that is the
    # situation the command exists for.
    archive = tmp_path / "as-recorded"
    for relative in ("docs/a.md", "docs/b.md", "LICENSE", "AGENTS.md"):
        blob = subprocess.run(
            ["git", "show", f"{recorded}:{relative}"],
            cwd=str(root),
            check=True,
            capture_output=True,
        ).stdout
        outlet = archive / relative
        outlet.parent.mkdir(parents=True, exist_ok=True)
        outlet.write_bytes(blob)
    pinned = build_manifest(
        archive,
        name="cli-corpus",
        version="1.0.0",
        licence="Apache-2.0",
        source="built by tests",
        include=("docs/", "*.md"),
        extensions=(".md",),
        exact_paths=("LICENSE", "AGENTS.md"),
    )
    manifest_path = tmp_path / "corpus-manifest.json"
    save_manifest(pinned, manifest_path)
    return root, manifest_path, recorded


def test_corpus_without_an_action_fails_loudly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["corpus"]) == 1
    assert "choose a corpus action" in capsys.readouterr().err


def test_corpus_restore_writes_a_snapshot_the_manifest_verifies(
    corpus_cli: tuple[Path, Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, manifest_path, recorded = corpus_cli
    output = tmp_path / "data" / "cli-corpus-snapshot.json"

    assert (
        main(
            [
                "corpus",
                "restore",
                "--manifest",
                str(manifest_path),
                "--revision",
                recorded,
                "--root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    err = capsys.readouterr()
    assert "chakaso-git-revision" in err.err
    assert output.is_file()
    assert (
        main(["corpus", "verify", "--manifest", str(manifest_path), "--snapshot", str(output)]) == 0
    )
    assert "verifies against" in capsys.readouterr().out


def test_corpus_restore_names_the_revision_it_cannot_read(
    corpus_cli: tuple[Path, Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, manifest_path, _ = corpus_cli

    assert (
        main(
            [
                "corpus",
                "restore",
                "--manifest",
                str(manifest_path),
                "--revision",
                "0" * 40,
                "--root",
                str(root),
                "--output",
                str(tmp_path / "s.json"),
            ]
        )
        == 1
    )

    assert "is not available in" in capsys.readouterr().err


def test_corpus_verify_refuses_a_moved_tree_and_a_mismatched_snapshot(
    corpus_cli: tuple[Path, Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from chakaso.training.manifest import build_manifest, save_manifest

    root, manifest_path, recorded = corpus_cli

    assert main(["corpus", "verify", "--manifest", str(manifest_path), "--root", str(root)]) == 1
    assert "content changed" in capsys.readouterr().err

    other = tmp_path / "other.json"
    assert (
        main(
            [
                "corpus",
                "restore",
                "--manifest",
                str(manifest_path),
                "--revision",
                recorded,
                "--root",
                str(root),
                "--output",
                str(other),
            ]
        )
        == 0
    )
    capsys.readouterr()

    # A manifest pinned from the moved tree is a different corpus, so the recorded snapshot no longer
    # answers for it: verify has to say so instead of reporting a fingerprint match.
    moved = build_manifest(
        root,
        name="cli-corpus",
        version="1.0.0",
        licence="Apache-2.0",
        source="built by tests",
        include=("docs/", "*.md"),
        extensions=(".md",),
        exact_paths=("LICENSE", "AGENTS.md"),
    )
    foreign = tmp_path / "foreign-manifest.json"
    save_manifest(moved, foreign)

    assert main(["corpus", "verify", "--manifest", str(foreign), "--snapshot", str(other)]) == 1
    assert "the manifest" in capsys.readouterr().err


def test_corpus_restore_requires_its_three_arguments() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["corpus", "restore", "--manifest", "x"])
    assert caught.value.code == 2


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


def test_storage_without_an_action_fails_loudly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["storage"]) == 1
    assert "choose a storage action" in capsys.readouterr().err


def test_storage_check_without_a_configured_path_reports_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The built-in configuration selects the in-memory backend with no path, so there is
    # nothing durable to inspect and the command says so rather than inventing a database.
    assert main(["storage", "check"]) == 0
    assert "no durable store path" in capsys.readouterr().out


def test_storage_migrate_creates_a_store_and_check_reports_it_healthy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "answers.db"
    assert main(["storage", "migrate", "--path", str(db)]) == 0
    capsys.readouterr()  # discard the migrate summary
    assert db.is_file()

    assert main(["storage", "check", "--path", str(db), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["schema_version"] == 1
    assert payload["answer_count"] == 0


def test_storage_check_on_an_absent_file_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "missing.db"

    assert main(["storage", "check", "--path", str(db)]) == 0

    assert "no durable store file" in capsys.readouterr().out
    assert not db.exists()  # a check must not create the database it inspects


def test_storage_check_refuses_a_database_that_is_not_an_answer_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "other.db"
    raw = sqlite3.connect(str(db))
    raw.execute("CREATE TABLE unrelated (x INTEGER)")
    raw.commit()
    raw.close()

    assert main(["storage", "check", "--path", str(db)]) == 1
    assert "not a Chakaso answer store" in capsys.readouterr().err


def test_storage_check_refuses_a_newer_schema_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "answers.db"
    SQLiteAnswerStore(db).close()
    raw = sqlite3.connect(str(db))
    raw.execute("PRAGMA user_version = 999")
    raw.commit()
    raw.close()

    assert main(["storage", "check", "--path", str(db)]) == 1
    assert "newer" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# benchmark strategies
# ---------------------------------------------------------------------------


def test_benchmark_strategies_reports_a_measured_comparison(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "strategies"]) == 0

    out = capsys.readouterr().out
    assert "retrieval strategy comparison" in out
    for strategy in ("lexical", "dense", "hybrid"):
        assert strategy in out
    # The honesty label is part of the output, not an afterthought.
    assert "non-semantic" in out


def test_benchmark_strategies_json_is_a_valid_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "strategies", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [row["strategy"] for row in payload["results"]] == ["lexical", "dense", "hybrid"]
    assert payload["baseline"] == "lexical"


def test_benchmark_retrieval_accepts_a_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "retrieval", "--strategy", "hybrid", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "recall_at_k" in payload["metrics"]


def test_benchmark_semantic_reports_a_development_instrument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "semantic"]) == 0

    out = capsys.readouterr().out
    assert "Semantic-evaluation development report" in out
    # The honesty label is part of the output, not an afterthought.
    assert "NOT semantic understanding" in out
    assert "fixture-semantic-0.1" in out
    assert "probe-support-all" in out


def test_benchmark_semantic_json_is_a_labeled_development_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "semantic", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) == 3
    for run in payload:
        assert run["benchmark"]["kind"] == "development"
        assert "accuracy" in run["metrics"]


def test_benchmark_without_a_subject_mentions_semantic(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark"]) == 1
    assert "semantic" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# benchmark grounded (P27)
# ---------------------------------------------------------------------------


def test_benchmark_grounded_reports_a_development_instrument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "grounded"]) == 0
    out = capsys.readouterr().out
    assert "Grounded-answer evaluation" in out
    assert "oracle" in out  # the default provider is labelled an oracle, not a model


def test_benchmark_grounded_json_separates_dimensions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "grounded", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evaluation"]["kind"] == "development-grounded-answer"
    assert payload["configuration"]["provider_is_oracle"] is True
    assert "retrieval_metrics" in payload and "answer_metrics" in payload
    assert "quality_score" not in json.dumps(payload)


def test_benchmark_grounded_json_records_reproducibility_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "grounded", "--json"]) == 0
    config = json.loads(capsys.readouterr().out)["configuration"]
    # Workstream F: a report names the interpreter, platform and package version behind its numbers.
    for key in ("python_version", "python_implementation", "platform", "chakaso_version"):
        assert config[key]


def test_benchmark_grounded_sufficiency_off_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "grounded", "--split", "all", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"]["sufficiency"] == "off"
    assert payload["sufficiency_metrics"]["assessed_cases"] == 0  # E2603 baseline path unchanged


def test_benchmark_grounded_sufficiency_on_measures_the_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "grounded", "--split", "all", "--sufficiency", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"]["sufficiency"] == "on"
    metrics = payload["sufficiency_metrics"]
    assert metrics["assessed_cases"] == 14
    # The policy fixes E2603's 3 missed abstentions with no false abstentions on the workload.
    assert metrics["missed_abstentions"] == 0
    assert metrics["false_abstentions"] == 0
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_benchmark_grounded_heldout_split_runs_only_heldout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "grounded", "--split", "heldout", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"]["split_filter"] == "heldout"
    assert all(case["split"] == "heldout" for case in payload["cases"])


def test_benchmark_grounded_none_provider_abstains_everywhere(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "grounded", "--provider", "none", "--split", "all", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"]["provider_is_oracle"] is False
    # A non-answering provider answers nothing, and over-abstains the answerable cases.
    assert payload["answer_metrics"]["answered_cases"] == 0
    assert payload["answer_metrics"]["appropriate_abstention_rate"] < 1.0


def test_benchmark_grounded_retrieval_only_has_no_answer_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "grounded", "--provider", "retrieval-only", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"]["provider"] == "retrieval-only"
    assert payload["retrieval_metrics"]["scored_cases"] >= 1
    assert payload["answer_metrics"]["answered_cases"] == 0


def test_benchmark_grounded_output_is_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["benchmark", "grounded", "--json"])
    first = capsys.readouterr().out
    main(["benchmark", "grounded", "--json"])
    second = capsys.readouterr().out
    assert first == second


def test_benchmark_without_a_subject_mentions_grounded(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark"]) == 1
    assert "grounded" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_reports_model_strategy_and_storage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["capabilities"]) == 0

    out = capsys.readouterr().out
    assert "model: deterministic" in out
    assert "development double: yes" in out
    assert "retrieval: on (strategy=lexical" in out
    assert "storage: memory" in out
    # The report is honest that no real semantic evaluator exists — only the fixture judge.
    assert "semantic evaluation: no real evaluator" in out


def test_capabilities_json_is_a_valid_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["capabilities", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["development_double"] is True
    assert payload["retrieval"]["strategy"] == "lexical"
    assert payload["retrieval"]["enabled"] is True
    assert payload["storage"]["backend"] == "memory"
    assert payload["semantic_evaluation"]["real_evaluator_available"] is False
    assert payload["semantic_evaluation"]["available_judges"] == ["fixture-semantic-0.1"]


def test_capabilities_reflects_configured_storage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[storage]\nbackend = "sqlite"\npath = "run/answers.db"\n'
        '[retrieval]\nstrategy = "hybrid"\ntop_k = 7\n',
        encoding="utf-8",
    )

    assert main(["capabilities", "--config", str(config), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieval"]["strategy"] == "hybrid"
    assert payload["retrieval"]["top_k"] == 7
    assert payload["storage"]["backend"] == "sqlite"
    assert payload["storage"]["path"] == "run/answers.db"


def test_capabilities_reports_an_unknown_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[model]\nadapter = 'nope'\n", encoding="utf-8")

    assert main(["capabilities", "--config", str(config)]) == 1
    assert "nope" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------


def _durable_chat(tmp_path: Path) -> tuple[Path, Path]:
    """Run one durable-backed chat turn and return the (config, database) paths it wrote."""
    db = tmp_path / "answers.db"
    config = tmp_path / "config.toml"
    config.write_text(
        f'[storage]\nbackend = "sqlite"\npath = "{db.as_posix()}"\n',
        encoding="utf-8",
    )
    assert main(["chat", "--config", str(config), "--message", "What is Chakaso?"]) == 0
    return config, db


def test_answers_without_an_action_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["answers"]) == 1
    assert "choose an answers action" in capsys.readouterr().err


def test_answers_list_reports_nothing_for_the_memory_backend(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The default backend keeps nothing across processes, so inspection says so rather than
    # inventing a store to read.
    assert main(["answers", "list"]) == 0
    assert "in-memory backend holds nothing" in capsys.readouterr().out


def test_answers_list_reports_when_the_durable_file_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[storage]\nbackend = "sqlite"\npath = "missing.db"\n', encoding="utf-8")

    assert main(["answers", "list", "--config", str(config)]) == 0

    out = capsys.readouterr().out
    assert "no durable store file" in out
    assert not (tmp_path / "missing.db").exists()  # listing must not create the database


def test_answers_list_shows_a_durable_record_written_by_chat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _db = _durable_chat(tmp_path)
    capsys.readouterr()  # discard the chat output

    assert main(["answers", "list", "--config", str(config)]) == 0

    out = capsys.readouterr().out
    assert "ans_" in out
    assert "conversation=" in out


def test_answers_list_json_is_valid_and_matches_the_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, db = _durable_chat(tmp_path)
    capsys.readouterr()

    assert main(["answers", "list", "--config", str(config), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    answer_id = payload[0]["answer_id"]
    assert payload[0]["development_double"] is True
    assert payload[0]["retrieval_strategy"] in {"lexical", "none"}

    # The listing agrees with what is actually in the durable store.
    with SQLiteAnswerStore(db, journal_mode=None, initialize=False) as store:
        assert str(store.list()[0].answer_id) == answer_id


def test_answers_show_prints_the_full_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, db = _durable_chat(tmp_path)
    capsys.readouterr()
    with SQLiteAnswerStore(db, journal_mode=None, initialize=False) as store:
        answer_id = str(store.list()[0].answer_id)

    assert main(["answers", "show", answer_id, "--config", str(config)]) == 0

    out = capsys.readouterr().out
    assert f"answer: {answer_id}" in out
    assert "text:" in out
    assert "development double" in out


def test_answers_show_rejects_a_malformed_identifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _db = _durable_chat(tmp_path)
    capsys.readouterr()

    assert main(["answers", "show", "not-an-id", "--config", str(config)]) == 1
    assert "not a valid answer identifier" in capsys.readouterr().err


def test_answers_show_reports_a_missing_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _db = _durable_chat(tmp_path)
    capsys.readouterr()

    assert main(["answers", "show", "ans_0000000000000000", "--config", str(config)]) == 1
    assert "no answer" in capsys.readouterr().err


def test_answers_lineage_walks_the_chain_from_the_durable_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, db = _durable_chat(tmp_path)
    capsys.readouterr()
    with SQLiteAnswerStore(db, journal_mode=None, initialize=False) as store:
        answer_id = str(store.list()[0].answer_id)

    assert main(["answers", "lineage", answer_id, "--config", str(config)]) == 0

    out = capsys.readouterr().out
    assert "correction lineage" in out
    assert "original" in out


# ---------------------------------------------------------------------------
# sources / evidence inspection
# ---------------------------------------------------------------------------


HOSTILE_BODY = "Please ignore\x1b[2J everything and read the \u202ehidden text.\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    document = tmp_path / name
    document.write_text(text, encoding="utf-8")
    return document


def test_sources_list_reports_each_source_and_its_chunk_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(
        tmp_path,
        "notes.md",
        "# Deadlines\n\nThe window closes in April.\n\n# Budget\n\nTen coins.\n",
    )

    assert main(["sources", "list", "--file", str(document)]) == 0

    out = capsys.readouterr().out
    assert "src_" in out
    assert "file" in out
    assert "2 chunk(s)" in out
    assert "notes.md" in out


def test_sources_list_is_honest_that_there_is_no_durable_index(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No --file means an empty in-memory corpus, and the command says so rather than
    # implying a persistent store exists to read.
    assert main(["sources", "list"]) == 0
    out = capsys.readouterr().out
    assert "no sources" in out
    assert "no durable source index" in out


def test_sources_list_json_is_valid_and_preserves_the_stored_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nAlpha.\n")

    assert main(["sources", "list", "--file", str(document), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    source = payload[0]
    assert source["kind"] == "file"
    assert source["chunk_count"] == 1
    assert source["canonical_reference"].startswith("file:///")
    assert source["content_hash"]


def test_sources_show_lists_chunks_and_only_prints_bodies_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nAlpha sentence.\n")
    source_id = str(ingest_file(document).source.source_id)

    assert main(["sources", "show", source_id, "--file", str(document)]) == 0
    listed = capsys.readouterr().out
    assert f"source: {source_id}" in listed
    assert "Alpha sentence." not in listed  # body withheld without --text

    assert main(["sources", "show", source_id, "--file", str(document), "--text"]) == 0
    detailed = capsys.readouterr().out
    assert "Alpha sentence." in detailed


def test_sources_show_json_matches_the_ingested_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nAlpha.\n")
    record = ingest_file(document)
    source_id = str(record.source.source_id)
    chunk_id = str(record.chunks[0].chunk_id)

    assert main(["sources", "show", source_id, "--file", str(document), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["source_id"] == source_id
    assert payload["chunks"][0]["chunk_id"] == chunk_id


def test_sources_show_rejects_a_malformed_identifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nAlpha.\n")

    assert main(["sources", "show", "not-an-id", "--file", str(document)]) == 1
    assert "not a valid source identifier" in capsys.readouterr().err


def test_sources_show_reports_a_valid_but_absent_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nAlpha.\n")

    # A well-formed identifier that is not among the given files is an error, not a
    # silently empty listing.
    assert main(["sources", "show", "src_0000000000000000", "--file", str(document)]) == 1
    err = capsys.readouterr().err
    assert "no source" in err
    assert "durable source index" in err


def test_evidence_show_prints_a_chunk_and_its_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nThe answer is April.\n")
    record = ingest_file(document)
    chunk_id = str(record.chunks[0].chunk_id)

    assert main(["evidence", "show", chunk_id, "--file", str(document)]) == 0

    out = capsys.readouterr().out
    assert f"chunk: {chunk_id}" in out
    assert "The answer is April." in out
    assert str(record.source.source_id) in out


def test_evidence_show_rejects_a_malformed_identifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nAlpha.\n")

    assert main(["evidence", "show", "chk_zzz", "--file", str(document)]) == 1
    assert "not a valid chunk identifier" in capsys.readouterr().err


def test_evidence_show_reports_an_absent_chunk(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nAlpha.\n")

    assert main(["evidence", "show", "chk_0000000000000000", "--file", str(document)]) == 1
    assert "no chunk" in capsys.readouterr().err


def test_sources_without_an_action_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sources"]) == 1
    assert "choose a sources action" in capsys.readouterr().err


def test_evidence_without_show_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evidence"]) == 1
    assert "evidence action" in capsys.readouterr().err


def test_retrieve_can_be_restricted_to_one_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    alpha = _write(tmp_path, "alpha.md", "# Alpha\n\nThe alpha fact about widget.\n")
    beta = _write(tmp_path, "beta.md", "# Beta\n\nThe beta fact about widget.\n")
    alpha_source = str(ingest_file(alpha).source.source_id)

    assert (
        main(["retrieve", "--query", "widget fact", "--file", str(alpha), "--file", str(beta)]) == 0
    )
    assert capsys.readouterr().out.count("   source:") == 2

    assert (
        main(
            [
                "retrieve",
                "--query",
                "widget fact",
                "--file",
                str(alpha),
                "--file",
                str(beta),
                "--source",
                alpha_source,
            ]
        )
        == 0
    )
    restricted = capsys.readouterr().out
    assert restricted.count("   source:") == 1
    assert "alpha.md" in restricted


def test_retrieve_refuses_a_source_that_is_not_among_the_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "a.md", "# One\n\nAlpha.\n")

    assert (
        main(
            [
                "retrieve",
                "--query",
                "alpha",
                "--file",
                str(document),
                "--source",
                "src_0000000000000000",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "among the ingested documents" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# adversarial source content stays inert on the terminal
# ---------------------------------------------------------------------------


def test_a_hostile_source_body_is_rendered_inert(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A document that embeds an ESC control sequence and a bidirectional override must not
    # reach the terminal raw; the display paths make those bytes inert while preserving the
    # substance of the text (the words are still shown).
    document = _write(tmp_path, "hostile.md", "# Hostile\n\n" + HOSTILE_BODY)
    chunk_id = str(ingest_file(document).chunks[0].chunk_id)

    assert main(["evidence", "show", chunk_id, "--file", str(document)]) == 0

    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "\u202e" not in out
    assert "\\x1b" in out and "\\u202e" in out  # visible escape markers, not silent drops
    assert "ignore" in out and "hidden text" in out


def test_retrieval_of_a_hostile_document_does_not_emit_control_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _write(tmp_path, "hostile.md", "# Hostile\n\n" + HOSTILE_BODY)

    assert main(["retrieve", "--query", "ignore everything", "--file", str(document)]) == 0

    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "\u202e" not in out
    assert "\\x1b[2J" in out


def test_json_output_keeps_the_exact_bytes_a_display_would_escape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The sanitiser is a display boundary only: machine-readable JSON preserves the raw
    # chunk text so a consumer can still see exactly what was stored.
    document = _write(tmp_path, "hostile.md", "# Hostile\n\n" + HOSTILE_BODY)
    record = ingest_file(document)
    chunk_id = str(record.chunks[0].chunk_id)

    assert main(["evidence", "show", chunk_id, "--file", str(document), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "\x1b" in payload["text"]
    assert str(ChunkId.parse(payload["chunk_id"])) == chunk_id


# ---------------------------------------------------------------------------
# answer inspection surfaces the historical provenance (P19)
# ---------------------------------------------------------------------------


def test_answers_show_reports_the_historical_evaluation_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _db = _durable_chat(tmp_path)
    capsys.readouterr()
    with SQLiteAnswerStore(_db_of(config), journal_mode=None, initialize=False) as store:
        answer_id = str(store.list()[0].answer_id)

    assert main(["answers", "show", answer_id, "--config", str(config)]) == 0

    out = capsys.readouterr().out
    # The inspection distinguishes the recorded outcome from a recomputation, names the evaluator,
    # and keeps the structural dimensions separate rather than showing one quality number.
    assert "recorded at the turn — historical, not recomputed" in out
    assert "evaluator: structural" in out
    assert "evidence_coverage (structural, not truth)" in out


def test_answers_show_json_includes_evaluation_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _db = _durable_chat(tmp_path)
    capsys.readouterr()
    with SQLiteAnswerStore(_db_of(config), journal_mode=None, initialize=False) as store:
        answer_id = str(store.list()[0].answer_id)

    assert main(["answers", "show", answer_id, "--config", str(config), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    prov = payload["evaluation_provenance"]
    assert prov["recorded_as"] == "historical"
    assert prov["grounding_evaluator"].startswith("structural")
    assert set(prov["grounding"]) == {
        "supported",
        "unsupported",
        "contradicted",
        "uncertain",
        "not_evaluated",
    }
    assert set(prov["citations"]) == {
        "made",
        "valid",
        "unknown",
        "irrelevant",
        "required_claims",
        "uncited_claims",
    }


def test_answers_list_json_carries_the_provenance_per_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _db = _durable_chat(tmp_path)
    capsys.readouterr()

    assert main(["answers", "list", "--config", str(config), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["evaluation_provenance"]["recorded_as"] == "historical"


def test_answers_show_states_that_evidence_content_is_not_retained(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A recorded reference is not the evidence itself: the durable store holds identifiers, not the
    # corpus text. Inspection must say so plainly rather than imply the cited content is re-readable.
    config, _db = _durable_chat(tmp_path)
    capsys.readouterr()
    with SQLiteAnswerStore(_db_of(config), journal_mode=None, initialize=False) as store:
        answer_id = str(store.list()[0].answer_id)

    assert main(["answers", "show", answer_id, "--config", str(config)]) == 0
    assert "does not retain the referenced content" in capsys.readouterr().out

    assert main(["answers", "show", answer_id, "--config", str(config), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence_retention"]["store_retains_evidence_content"] is False
    assert payload["evidence_retention"]["note"]


def test_persisted_provenance_is_rendered_inert_on_the_terminal() -> None:
    # A regression for control characters inside newly persisted provenance: an unknown evaluator
    # identity or a hostile answer text reaching the human-readable inspection must not emit live
    # terminal-control sequences. The historical fact is preserved as a visible escape.
    provenance = EvaluationProvenance(
        claims=1,
        citations_made=0,
        citations_valid=0,
        citations_unknown=0,
        citations_irrelevant=0,
        required_claims=0,
        uncited_claims=0,
        grounded_claims=(
            GroundedClaim(Claim("ans_s", 0, "A fact.").claim_id, ClaimStatus.NOT_EVALUATED),
        ),
        grounding_evaluator="evil\x1b[2Jjudge",
        grounding_evaluator_version="1\u202e0",
        unresolved_reference_ids=(),
        malformed_reference_ids=(),
    )

    rendered = "\n".join(_format_provenance(provenance))

    assert "\x1b" not in rendered
    assert "\u202e" not in rendered
    assert "\\x1b[2Jjudge" in rendered


def _db_of(config: Path) -> Path:
    """Derive the durable store path from a temp config file written by _durable_chat."""
    return config.parent / "answers.db"
