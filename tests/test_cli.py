"""Tests for the command line interface.

The CLI is thin and should stay thin. These tests cover the behaviour a user
depends on: the version is reported, help works, and unknown input fails loudly
rather than silently doing nothing.
"""

from __future__ import annotations

import io
import json
import platform
import sqlite3
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chakaso import __version__
from chakaso.answers import SQLiteAnswerStore
from chakaso.cli import _report_caveats, build_parser, main
from chakaso.config import load_config
from chakaso.conversation import Conversation, ConversationManager
from chakaso.models import (
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    ModelMetadata,
)

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


def test_capabilities_json_is_a_valid_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["capabilities", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["development_double"] is True
    assert payload["retrieval"]["strategy"] == "lexical"
    assert payload["retrieval"]["enabled"] is True
    assert payload["storage"]["backend"] == "memory"


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
