"""Tests for the command line interface.

The CLI is thin and should stay thin. These tests cover the behaviour a user
depends on: the version is reported, help works, and unknown input fails loudly
rather than silently doing nothing.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

from chakaso import __version__
from chakaso.cli import build_parser, main
from chakaso.config import load_config

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs" / "default.toml"


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
