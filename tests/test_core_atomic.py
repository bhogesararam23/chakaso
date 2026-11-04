"""Tests for the staged write every artifact layer depends on.

The property is that a destination holds a complete file or the one it held before: a report, checkpoint or
corpus snapshot that is half-written is worse than a missing one, because it looks like an artifact until
someone verifies it. These tests interrupt a write on purpose and check what is left on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chakaso.core.atomic import staged_write, write_text_atomically
from chakaso.core.hashing import sha256_hex


def _staged_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.endswith(".partial"))


def test_write_text_atomically_lands_the_payload_and_nothing_else(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "report.json"

    write_text_atomically(json.dumps({"a": 1}, indent=2), destination)

    assert destination.read_text(encoding="utf-8") == json.dumps({"a": 1}, indent=2)
    assert _staged_names(destination.parent) == []


def test_the_bytes_on_disk_are_the_characters_that_were_hashed(tmp_path: Path) -> None:
    """A digest is only reproducible if writing did not translate the newlines it was computed over."""
    payload = json.dumps({"lines": ["one", "two"], "note": "indented"}, indent=2, sort_keys=True)
    destination = tmp_path / "report.json"

    write_text_atomically(payload, destination)

    raw = destination.read_bytes()
    assert b"\r\n" not in raw
    assert sha256_hex(raw.decode("utf-8")) == sha256_hex(payload)


def test_a_write_that_raises_leaves_the_previous_file_intact(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.txt"
    write_text_atomically("the whole artifact", destination)

    with (
        pytest.raises(RuntimeError, match="midway"),
        staged_write(destination) as staged,
    ):
        staged.write_text("half-written", encoding="utf-8")
        failure = "midway"
        raise RuntimeError(failure)

    assert destination.read_text(encoding="utf-8") == "the whole artifact"
    assert _staged_names(tmp_path) == []


def test_a_staged_write_replaces_the_previous_file_only_when_it_finishes(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.txt"

    with staged_write(destination) as staged:
        staged.write_text("the new artifact", encoding="utf-8")
        assert not destination.exists()

    assert destination.read_text(encoding="utf-8") == "the new artifact"


def test_an_interrupted_process_leaves_no_staged_file_behind(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.txt"

    class _Interrupted(BaseException):
        pass

    with pytest.raises(_Interrupted), staged_write(destination):
        interruption = "keyboard interrupt"
        raise _Interrupted(interruption)

    assert _staged_names(tmp_path) == []
    assert not destination.exists()


def test_a_staged_write_to_a_missing_directory_creates_it(tmp_path: Path) -> None:
    destination = tmp_path / "data" / "experiments" / "E2606" / "report.json"

    with staged_write(destination) as staged:
        staged.write_text("x", encoding="utf-8")

    assert destination.is_file()


def test_a_swap_the_operating_system_refuses_leaves_the_previous_artifact_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Windows case, tested without branching on the platform.

    On Windows `os.replace` onto a file another handle holds fails with `PermissionError [WinError 5]`, where
    POSIX would have moved it: a checkpoint rewrite racing an open reader, an editor holding the file, or
    Defender scanning bytes that were just written. The behaviour the artifact layer depends on is not the
    error — it is that the error costs nothing, so the same assertion holds on either OS and a reader on
    Linux is testing the branch that ships to Windows.
    """
    destination = tmp_path / "E2698-transformer-resumable.pt"
    write_text_atomically("the resume point a run wrote", destination)
    real_replace = Path.replace

    def refuse(self: Path, target: Path) -> Path:
        error = 5
        raise PermissionError(error, "Access is denied", str(self), target, None)

    monkeypatch.setattr(Path, "replace", refuse)

    with pytest.raises(PermissionError, match="Access is denied"), staged_write(destination):
        pass

    monkeypatch.setattr(Path, "replace", real_replace)
    assert destination.read_text(encoding="utf-8") == "the resume point a run wrote"
    assert _staged_names(tmp_path) == []
