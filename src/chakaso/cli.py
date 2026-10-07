"""Command line entry point.

The CLI is deliberately thin. It exists so that the package can be verified after
installation (``chakaso --version``), so that configuration can be inspected
(``chakaso config show``), and so that future components have one obvious place to
expose themselves. It does not implement behaviour of its own, and it must not grow
logic that belongs in a component.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from chakaso import __version__
from chakaso.config import ConfigError, LoadedConfig, load_config

__all__ = ["build_parser", "main"]

PROGRAM = "chakaso"


def build_parser() -> argparse.ArgumentParser:
    """Build the root argument parser.

    Kept separate from :func:`main` so that tests can assert on the parser's
    behaviour without going through the process exit path.
    """
    root, _config_parser = _build_parsers()
    return root


def _build_parsers() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the root parser and the ``config`` sub-parser.

    Both are returned because ``chakaso config`` with no action prints the config
    parser's own help. Asking argparse for it by re-parsing ``["config", "--help"]``
    would raise ``SystemExit`` from inside a function that otherwise returns an exit
    code, which is a worse contract for callers and for tests.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Chakaso is a research project. There is no conversational system to run "
            "yet; see docs/agent/CURRENT_STATE.md in the repository."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROGRAM} {__version__}",
        help="print the installed version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.add_parser(
        "info",
        help="print runtime facts useful in a bug report",
        description=(
            "Print the package version, the Python version and the platform. "
            "This is the information a bug report needs first."
        ),
    )

    config_parser = subparsers.add_parser(
        "config",
        help="inspect configuration",
        description="Inspect the resolved configuration and where each value came from.",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", metavar="ACTION")
    show_parser = config_subparsers.add_parser(
        "show",
        help="print the resolved configuration with its provenance",
        description=(
            "Print every configuration value together with the source it came from. "
            "Nothing is loaded implicitly: pass the files to apply, in order."
        ),
    )
    show_parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "configuration file to apply; repeat to layer files, later ones winning. "
            "Omit to show the built-in defaults."
        ),
    )

    return parser, config_parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    parser, config_parser = _build_parsers()
    args = parser.parse_args(argv)

    if args.command is None:
        # No command is not an error; the user most likely wants to know what the
        # tool does. Argparse is asked for the same help text a failure would show.
        parser.print_help()
        return 0

    if args.command == "info":
        _print_info()
        return 0

    if args.command == "config":
        return _run_config(parser, config_parser, args)

    # Unreachable while every subcommand is handled above. argparse rejects
    # unknown commands before this point, so reaching here means a subcommand was
    # added to the parser and not to this dispatch.
    parser.error(f"unhandled command: {args.command!r}")
    return 2


def _run_config(
    parser: argparse.ArgumentParser,
    config_parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> int:
    if args.config_command is None:
        config_parser.print_help()
        return 0

    if args.config_command == "show":
        return _show_config(args.config)

    parser.error(f"unhandled config action: {args.config_command!r}")
    return 2


def _show_config(paths: list[str] | None) -> int:
    files = [Path(item) for item in paths] if paths else []
    primary = files[0] if files else None
    overrides = files[1:]

    try:
        loaded = load_config(primary, overrides=overrides)
    except ConfigError as exc:
        # A configuration problem is a user-facing error, not a crash. It is
        # printed without a traceback and exits non-zero.
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    print(_format_config(loaded))
    return 0


def _format_config(loaded: LoadedConfig) -> str:
    key_width = max(len(item.key) for item in loaded.resolved)
    value_width = max(len(_format_value(item.value)) for item in loaded.resolved)

    lines = [f"{'KEY':<{key_width}}  {'VALUE':<{value_width}}  SOURCE"]
    for item in loaded.resolved:
        value = _format_value(item.value)
        lines.append(f"{item.key:<{key_width}}  {value:<{value_width}}  {item.source}")

    if not loaded.sources:
        lines.append("")
        lines.append("No configuration file was read; these are the built-in defaults.")
    return "\n".join(lines)


def _format_value(value: object) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def _print_info() -> None:
    print(f"{PROGRAM} {__version__}")
    print(f"python {platform.python_version()} ({sys.implementation.name})")
    print(f"platform {platform.system()} {platform.release()} ({platform.machine()})")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
