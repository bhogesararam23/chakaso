"""An experiment protocol declared as data, so a record can name the file instead of forty flags.

E2613's first launch went wrong in the space between a record that said what a run would be and a command line
that was supposed to say the same thing: the two are written separately and nothing compares them. A protocol
file closes that gap from one side. The record cites a committed TOML file, the run reads that file, and the
only things a command line can still change are the ones that name *where* and *which corpus* — never what is
measured.

The file holds one ``[plan]`` table of :class:`~chakaso.training.subword_experiment.ExperimentPlan` fields. It
does not hold ``root``, ``work_dir``, ``corpus`` or ``overwrite_artifacts`` (a location, a corpus spec the
snapshot supplies, and a run-time permission), and any
key the plan does not have is refused, because a typo in a protocol must fail loudly rather than fall back to a
default the operator did not choose.
"""

from __future__ import annotations

import tomllib
import types
import typing
from pathlib import Path
from typing import Any

from chakaso.training.errors import TrainingError
from chakaso.training.subword_experiment import ExperimentPlan

__all__ = ["ProtocolError", "load_plan"]

#: Fields a protocol file may not set: where the run lives, the corpus spec a snapshot supplies, and the
#: permission to replace an artifact, which is an operator's decision at run time and not part of what is measured.
_NOT_IN_A_PROTOCOL = frozenset({"root", "work_dir", "corpus", "overwrite_artifacts"})


class ProtocolError(TrainingError):
    """A protocol file is unreadable, names a field the plan does not have, or contradicts the command line."""


def load_plan(
    path: Path, *, root: Path, work_dir: Path, fixed: dict[str, Any] | None = None
) -> ExperimentPlan:
    """Build the plan a protocol file declares, with ``fixed`` applied from the command line.

    ``fixed`` carries values the operator supplies outside the file (a split seed, a corpus fingerprint). A
    field present in both places must agree: the same quantity declared twice with two values is a protocol
    nobody can say they chose.

    Raises:
        ProtocolError: the file cannot be read or parsed, has no ``[plan]`` table, sets a field a protocol may
            not set, names an unknown field, gives a value of the wrong type, or disagrees with ``fixed``.
    """
    try:
        document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        message = f"cannot read the protocol file {str(path)!r}: {error}"
        raise ProtocolError(message) from error
    table = document.get("plan")
    if not isinstance(table, dict):
        message = f"the protocol file {str(path)!r} has no [plan] table"
        raise ProtocolError(message)

    hints = typing.get_type_hints(ExperimentPlan)
    known = set(hints) - _NOT_IN_A_PROTOCOL
    forbidden = sorted(set(table) & _NOT_IN_A_PROTOCOL)
    if forbidden:
        message = (
            f"the protocol file sets {forbidden}, which name where a run lives or what the snapshot "
            "supplies; give them on the command line or leave them out"
        )
        raise ProtocolError(message)
    unknown = sorted(set(table) - known)
    if unknown:
        message = f"the protocol file names {unknown}, which are not experiment plan fields"
        raise ProtocolError(message)

    values: dict[str, Any] = {
        name: _coerce(name, value, hints[name]) for name, value in table.items()
    }
    for name, value in (fixed or {}).items():
        if value is None:
            continue
        if name in values and values[name] != value:
            message = (
                f"{name} is {values[name]!r} in the protocol file and {value!r} on the command line; "
                "a protocol declares each quantity once"
            )
            raise ProtocolError(message)
        values[name] = value
    return ExperimentPlan(root=Path(root), work_dir=Path(work_dir), **values)


def _coerce(name: str, value: Any, hint: Any) -> Any:
    """Convert a TOML value to the type the plan field declares, refusing what does not fit."""
    options = [
        option
        for option in (typing.get_args(hint) if _is_union(hint) else (hint,))
        if option is not type(None)
    ]
    if value is None:
        return None
    for option in options:
        origin = typing.get_origin(option)
        if option is Path and isinstance(value, str):
            return Path(value)
        if option is bool and isinstance(value, bool):
            return value
        if option is float and isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if option is int and isinstance(value, int) and not isinstance(value, bool):
            return value
        if option is str and isinstance(value, str):
            return value
        if origin is tuple and isinstance(value, list):
            return tuple(value)
    message = f"{name} has the wrong type in the protocol file: {value!r} does not fit {hint}"
    raise ProtocolError(message)


def _is_union(hint: Any) -> bool:
    return typing.get_origin(hint) in (typing.Union, types.UnionType)
