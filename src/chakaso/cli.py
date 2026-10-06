"""Command line entry point.

The CLI is deliberately thin. It exists so that the package can be verified after
installation (``chakaso --version``) and so that future components have one
obvious place to expose themselves. It does not implement behaviour of its own,
and it must not grow logic that belongs in a component.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence

from chakaso import __version__

__all__ = ["build_parser", "main"]

PROGRAM = "chakaso"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Kept separate from :func:`main` so that tests can assert on the parser's
    behaviour without going through the process exit path.
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # No command is not an error; the user most likely wants to know what the
        # tool does. Argparse is asked for the same help text a failure would show.
        parser.print_help()
        return 0

    if args.command == "info":
        _print_info()
        return 0

    # Unreachable while every subcommand is handled above. argparse rejects
    # unknown commands before this point, so reaching here means a subcommand was
    # added to the parser and not to this dispatch.
    parser.error(f"unhandled command: {args.command!r}")
    return 2


def _print_info() -> None:
    print(f"{PROGRAM} {__version__}")
    print(f"python {platform.python_version()} ({sys.implementation.name})")
    print(f"platform {platform.system()} {platform.release()} ({platform.machine()})")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
