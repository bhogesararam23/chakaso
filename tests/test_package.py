"""Smoke tests for the installed distribution.

These tests exist because the project uses a src layout precisely so that tests
exercise the installed package. If they pass from a source tree without an
install, the layout has stopped doing its job.
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import chakaso

REPO_ROOT = Path(__file__).resolve().parent.parent

# A deliberately permissive but structural check: the version must have the shape
# of a PEP 440 release with an optional pre-release segment. The point is to catch
# a version that is not a version, not to reimplement packaging's parser.
_VERSION_PATTERN = re.compile(r"^\d+(\.\d+)*((a|b|rc|\.dev|\.post)\d+)*$")


def test_version_is_a_version_string() -> None:
    assert _VERSION_PATTERN.match(chakaso.__version__), chakaso.__version__


def test_installed_metadata_is_available() -> None:
    # Fails if the distribution is not installed, which is how the src layout
    # surfaces a missing editable install instead of silently testing the tree.
    metadata.version("chakaso")


def test_installed_version_matches_package_version() -> None:
    # pyproject.toml reads the version from chakaso.__version__ rather than
    # repeating it, and this asserts the two did not drift apart.
    assert metadata.version("chakaso") == chakaso.__version__


def test_console_script_entry_point_is_installed() -> None:
    entries = metadata.entry_points(group="console_scripts")
    assert "chakaso" in {entry.name for entry in entries}


def test_py_typed_marker_exists_and_ships_as_package_data() -> None:
    # py.typed must ship, or downstream type checkers silently ignore the package.
    # An editable install reads the source tree, so its file list cannot prove the
    # wheel contains the marker; the declaration in pyproject.toml is what does,
    # and it is asserted here alongside the file's existence.
    marker = Path(chakaso.__file__).parent / "py.typed"
    assert marker.is_file()

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "py.typed" in package_data["chakaso"]
