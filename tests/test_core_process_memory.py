"""Tests for the process-memory reading that lets a record state its own footprint.

The point of this module is that a measured run can finally say how much memory it used, or say plainly that
the platform would not tell it. Both branches are checked: the numbers, if they come back, must be real and
must grow when the process allocates; and where they cannot come back, the type must still be well-formed
rather than a crash or a fabricated zero.
"""

from __future__ import annotations

import platform
from pathlib import Path

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


_MEMINFO = """MemTotal:       16374520 kB
MemFree:          912340 kB
MemAvailable:    5242880 kB
Buffers:          262144 kB
SwapCached:            0 kB
HugePages_Total:       0
Garbage:          not a number kB
"""


def test_meminfo_parsing_reads_kibibytes_as_bytes_and_skips_what_is_not_a_number() -> None:
    values = process_memory._meminfo_values(_MEMINFO)

    assert values["MemAvailable"] == 5242880 * 1024
    assert values["MemTotal"] == 16374520 * 1024
    assert values["MemFree"] == 912340 * 1024
    # A line whose first token is not a number is skipped rather than read as zero, because "this field
    # exists and is 0" and "this field did not parse" are different claims and the second one must not
    # become a measurement. (The kB assumption is fine for the two fields this module reads; the
    # HugePages_* counters are page counts, which is why nothing else asks for them.)
    assert "Garbage" not in values
    assert values["HugePages_Total"] == 0


def test_the_available_reading_names_its_interface_and_serialises_both_units(
    tmp_path: Path,
) -> None:
    path = tmp_path / "meminfo"
    path.write_text(_MEMINFO, encoding="utf-8")

    reading = process_memory._proc_meminfo_availability(path)

    assert reading.measured
    assert reading.available_bytes == 5242880 * 1024
    assert reading.source == "linux:/proc/meminfo"
    assert reading.as_dict()["available_mebibytes"] == 5120.0
    assert reading.as_dict()["total_mebibytes"] == round(16374520 * 1024 / 1_048_576, 1)
    # MemFree is not the figure a run needs: on a machine that has been up for a week it ignores what the
    # kernel can reclaim, which is why this module asks for MemAvailable by name.
    assert reading.available_bytes != 912340 * 1024


def test_an_unreadable_meminfo_is_a_stated_absence(tmp_path: Path) -> None:
    reading = process_memory._proc_meminfo_availability(tmp_path / "does-not-exist")

    assert not reading.measured
    assert reading.available_bytes is None
    assert reading.source == "linux:meminfo-unreadable"


def test_the_availability_reader_refuses_a_platform_with_no_win_dll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_memory.sys, "platform", "linux")

    reading = process_memory._windows_availability()

    assert reading.source == "windows:not-a-windows-build"
    assert not reading.measured


def test_a_failing_availability_call_is_reported_as_a_failure_not_a_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_kernel32() -> process_memory.MemoryAvailability:
        message = "no kernel32"
        raise OSError(message)

    monkeypatch.setattr(process_memory, "_windows_availability_reading", _no_kernel32)

    reading = process_memory._windows_availability()

    assert reading.available_bytes is None
    assert reading.source == "windows:call-failed:OSError"


def test_an_unsupported_platform_answers_with_a_well_formed_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chakaso.core.process_memory.platform.system", lambda: "Plan9")

    reading = process_memory.available_memory()

    assert not reading.measured
    assert reading.source == "unsupported:Plan9"


def test_the_running_machine_reports_a_sane_available_figure() -> None:
    """On the two platforms this module asks, the figure has to exist and sit inside the machine."""
    reading = process_memory.available_memory()

    if platform.system() not in {"Linux", "Windows"}:
        assert not reading.measured
        return
    assert reading.measured, f"the machine would not answer: {reading.source}"
    assert 0 < reading.available_bytes <= reading.total_bytes
