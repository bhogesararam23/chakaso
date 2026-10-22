"""Tests for the process-memory reading that lets a record state its own footprint.

The point of this module is that a measured run can finally say how much memory it used, or say plainly that
the platform would not tell it. Both branches are checked: the numbers, if they come back, must be real and
must grow when the process allocates; and where they cannot come back, the type must still be well-formed
rather than a crash or a fabricated zero.
"""

from __future__ import annotations

import platform

import pytest

from chakaso.core import process_memory
from chakaso.core.process_memory import MemoryUsage, current_memory, peak_memory


def test_a_reading_reports_which_interface_answered() -> None:
    usage = peak_memory()

    assert isinstance(usage, MemoryUsage)
    assert usage.source
    # The two names are the same call, so an integration test that picks either gets the same account of
    # which interface answered.
    assert current_memory().source == usage.source
    if platform.system() in ("Windows", "Linux"):
        assert usage.measured
        assert usage.resident_bytes and usage.peak_resident_bytes
        assert usage.resident_bytes > 0
        assert usage.peak_resident_bytes >= usage.resident_bytes


def test_the_peak_moves_when_the_process_allocates() -> None:
    if not peak_memory().measured:
        pytest.skip(f"this platform does not report peak memory: {peak_memory().source}")

    before = peak_memory().peak_resident_bytes
    held = bytearray(96 * 1_048_576)
    for index in range(0, len(held), 4096):
        held[index] = 1
    after = peak_memory().peak_resident_bytes

    assert after is not None and before is not None
    assert after > before
    del held


def test_the_serialised_form_states_bytes_and_megabytes() -> None:
    usage = peak_memory()

    recorded = usage.as_dict()

    assert recorded["source"] == usage.source
    assert recorded["peak_resident_bytes"] == usage.peak_resident_bytes
    if usage.peak_resident_bytes is None:
        assert recorded["peak_resident_megabytes"] is None
    else:
        assert recorded["peak_resident_megabytes"] == round(
            usage.peak_resident_bytes / 1_048_576, 1
        )


def test_an_unsupported_platform_returns_a_well_formed_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chakaso.core.process_memory.platform.system", lambda: "Plan9")

    usage = peak_memory()

    assert usage.peak_resident_bytes is None
    assert usage.resident_bytes is None
    assert not usage.measured
    assert usage.source == "unsupported:Plan9"


def test_a_failing_windows_call_is_reported_as_a_failure_not_a_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_kernel32() -> MemoryUsage:
        message = "no kernel32"
        raise OSError(message)

    # Patched one layer out, so the wrapper's error policy is tested on every platform rather than only where
    # a kernel32 exists to fail.
    monkeypatch.setattr(process_memory, "_windows_memory_reading", _no_kernel32)

    usage = process_memory._windows_memory()

    # The module's contract is that it never raises for a platform reason: an unreadable interface becomes a
    # None reading with the failing source named.
    assert usage.peak_resident_bytes is None
    assert usage.source == "windows:call-failed:OSError"


def test_the_windows_reader_refuses_a_platform_that_has_no_win_dll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``WinDLL`` is a Windows build of CPython's name, and calling the reader elsewhere is a stated absence.

    A Linux runner cannot even type-check the attribute, which is what made this a real CI failure rather than
    a hypothetical one: the code has to answer for the wrong platform instead of raising, and the answer has to
    be checkable from any operating system.
    """
    monkeypatch.setattr(process_memory.sys, "platform", "linux")

    usage = process_memory._windows_memory()

    assert usage.source == "windows:not-a-windows-build"
    assert not usage.measured
