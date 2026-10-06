"""Tests for the command line interface.

The CLI is thin and should stay thin. These tests cover the behaviour a user
depends on: the version is reported, help works, and unknown input fails loudly
rather than silently doing nothing.
"""

from __future__ import annotations

import platform
import sys

import pytest

from chakaso import __version__
from chakaso.cli import build_parser, main


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
