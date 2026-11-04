"""What the operating system says about memory: this process's high-water mark, and what the machine has free.

Every experiment record in this repository has so far had to write "peak memory is not stated anywhere",
because the only interfaces reachable from the shell it ran in returned nothing. That is a missing
measurement, not a law: Windows and Linux both report a process's high-water mark, and a training run on a
16 GB machine with other things open needs that number to be honest about what it consumed rather than about
what it could have consumed. :func:`available_memory` is the other half of the same honesty — what was free
**before** the run spent it, which is the figure a record needs to say "one arm at a time on this machine"
from a measurement instead of from memory.

Returns ``None`` where the platform is not one this module asks. Callers must treat that as "not measured"
and say so, which is the same rule the rest of the package applies to an absent number: it is never replaced
with an estimate.

Standard library only: the operating system's own calls, through :mod:`ctypes`.
"""

from __future__ import annotations

import ctypes
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MemoryAvailability",
    "MemoryUsage",
    "available_memory",
    "current_memory",
    "peak_memory",
]


@dataclass(frozen=True, slots=True)
class MemoryUsage:
    """What the operating system reports about this process's memory.

    Attributes:
        peak_resident_bytes: The high-water mark of the resident set since the process started, or ``None``
            where the platform does not expose one.
        resident_bytes: What is resident right now, or ``None``.
        source: Which interface answered, recorded because "measured from /proc/self/status" and "measured
            from the Windows job object" are different claims.
    """

    peak_resident_bytes: int | None
    resident_bytes: int | None
    source: str

    @property
    def measured(self) -> bool:
        """Whether anything at all was read; false means the record must say so."""
        return self.peak_resident_bytes is not None or self.resident_bytes is not None

    def as_dict(self) -> dict[str, object]:
        """Plain data for an experiment report, with megabytes alongside bytes for a human reader."""
        return {
            "peak_resident_bytes": self.peak_resident_bytes,
            "resident_bytes": self.resident_bytes,
            "peak_resident_megabytes": (
                None
                if self.peak_resident_bytes is None
                else round(self.peak_resident_bytes / 1_048_576, 1)
            ),
            "source": self.source,
        }


def _windows_memory() -> MemoryUsage:
    try:
        return _windows_memory_reading()
    except (OSError, AttributeError, ValueError) as exc:
        # The module's contract is that a platform that will not answer is reported, not raised through:
        # an experiment record must be able to print "not measured" with the reason attached.
        return MemoryUsage(None, None, f"windows:call-failed:{type(exc).__name__}")


def _windows_memory_reading() -> MemoryUsage:
    # ``ctypes.WinDLL`` exists only in a Windows build of CPython, and typeshed declares it under a
    # ``sys.platform == "win32"`` guard, so a Linux or macOS type check cannot see the name at all. The guard
    # is what lets one source file be checked on every platform CI runs on: the branch below is analysed only
    # where the attribute exists. At runtime the dispatcher already selects by ``platform.system()``; this
    # returns the honest reading for the case where something calls it anyway.
    if sys.platform != "win32":
        return MemoryUsage(None, None, "windows:not-a-windows-build")

    class _Counters(ctypes.Structure):
        # PROCESS_MEMORY_COUNTERS_EX, field for field. The first two members are DWORDs packed together,
        # so declaring the second one as a pointer shifts every later reading by eight bytes and returns a
        # plausible number for the wrong quantity — which is worse than returning nothing.
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("page_fault_count", ctypes.c_uint32),
            ("peak_working_set", ctypes.c_size_t),
            ("working_set", ctypes.c_size_t),
            ("quota_peak_paged_pool", ctypes.c_size_t),
            ("quota_paged_pool", ctypes.c_size_t),
            ("quota_peak_nonpaged_pool", ctypes.c_size_t),
            ("quota_nonpaged_pool", ctypes.c_size_t),
            ("pagefile", ctypes.c_size_t),
            ("peak_pagefile", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    getter = kernel32.K32GetProcessMemoryInfo
    getter.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_uint32]
    handle = kernel32.GetCurrentProcess()
    if not getter(handle, ctypes.byref(counters), counters.cb):
        return MemoryUsage(None, None, "windows:k32-call-failed")
    return MemoryUsage(int(counters.peak_working_set), int(counters.working_set), "windows:k32")


def _proc_status_memory() -> MemoryUsage:
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return MemoryUsage(None, None, "linux:proc-unreadable")
    values: dict[str, int] = {}
    for line in lines:
        key, _, rest = line.partition(":")
        if key in {"VmHWM", "VmRSS"}:
            fields = rest.split()
            if fields:
                values[key] = int(fields[0]) * 1024  # the kernel reports kibibytes
    return MemoryUsage(values.get("VmHWM"), values.get("VmRSS"), "linux:/proc/self/status")


def peak_memory() -> MemoryUsage:
    """Read this process's memory high-water mark.

    Returns:
        A :class:`MemoryUsage`. On macOS, and on any platform whose call fails, both numbers are ``None``
        and ``source`` says which was attempted — a caller writing a record must copy that rather than
        dropping the field.
    """
    system = platform.system()
    if system == "Windows":
        return _windows_memory()
    if system == "Linux":
        return _proc_status_memory()
    return MemoryUsage(None, None, f"unsupported:{system}")


def current_memory() -> MemoryUsage:
    """The same reading, named for the case that wants the present value rather than the peak."""
    return peak_memory()


@dataclass(frozen=True, slots=True)
class MemoryAvailability:
    """What the machine has free **for new work**, and which interface said so.

    ``available_bytes`` is the kernel's own answer to "how much can be had without paging": on Linux
    ``MemAvailable``, which counts reclaimable page cache, and not ``MemFree``, which on a machine that has been
    up for a week sits at a few hundred megabytes and means nothing about whether a training run will fit.
    Windows answers with ``ullAvailPhys`` from the global memory status call.

    Attributes:
        available_bytes: The free-for-new-work figure, or ``None`` where the platform was not asked or would not
            answer.
        total_bytes: The machine's physical memory, or ``None``. Recorded beside the available figure because
            "5 GB free" is a different claim on a 16 GB machine than on a 64 GB one, and a cost paragraph in a
            record quotes one of the two without saying which.
        source: Which interface answered, or why it did not.
    """

    available_bytes: int | None
    total_bytes: int | None
    source: str

    @property
    def measured(self) -> bool:
        """Whether anything at all was read; false means a caller writes "not measured" and says why."""
        return self.available_bytes is not None

    def as_dict(self) -> dict[str, object]:
        """Plain data for an experiment report, with mebibytes alongside bytes for a human reader."""
        return {
            "available_bytes": self.available_bytes,
            "available_mebibytes": (
                None if self.available_bytes is None else round(self.available_bytes / 1_048_576, 1)
            ),
            "total_bytes": self.total_bytes,
            "total_mebibytes": (
                None if self.total_bytes is None else round(self.total_bytes / 1_048_576, 1)
            ),
            "source": self.source,
        }


def _meminfo_values(text: str) -> dict[str, int]:
    """`/proc/meminfo` as a mapping of field name to bytes; the kernel reports every field in kibibytes."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if fields:
            try:
                values[key.strip()] = int(fields[0]) * 1024
            except ValueError:
                continue
    return values


def _proc_meminfo_availability(path: Path = Path("/proc/meminfo")) -> MemoryAvailability:
    try:
        values = _meminfo_values(path.read_text(encoding="utf-8"))
    except OSError:
        return MemoryAvailability(None, None, "linux:meminfo-unreadable")
    return MemoryAvailability(
        values.get("MemAvailable"), values.get("MemTotal"), "linux:/proc/meminfo"
    )


def _windows_availability_reading() -> MemoryAvailability:
    # The same `sys.platform` guard `peak_memory` uses: `ctypes.WinDLL` is declared by typeshed only where it
    # exists, so this file has to be checkable on the Linux runners that CI uses while still reading the real
    # number on the Windows machine the long runs actually run on.
    if sys.platform != "win32":
        return MemoryAvailability(None, None, "windows:not-a-windows-build")

    class _MemoryStatus(ctypes.Structure):
        # MEMORYSTATUSEX, field for field. dwLength and dwMemoryLoad are two DWORDs packed into the first eight
        # bytes, so declaring only one of them shifts every 64-bit reading that follows — the same failure mode
        # `_Counters` above documents, and the same one that returns a plausible number for the wrong quantity.
        _fields_ = [
            ("length", ctypes.c_uint32),
            ("memory_load_percent", ctypes.c_uint32),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_pagefile", ctypes.c_ulonglong),
            ("avail_pagefile", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended_virtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return MemoryAvailability(None, None, "windows:globalmemorystatusex-failed")
    return MemoryAvailability(
        int(status.avail_phys), int(status.total_phys), "windows:globalmemorystatusex"
    )


def _windows_availability() -> MemoryAvailability:
    try:
        return _windows_availability_reading()
    except (OSError, AttributeError, ValueError) as exc:
        # A platform that will not answer is reported, not raised through: a run's preflight has to be able to
        # say "the guard could not read the machine" instead of failing the launch for an operating-system detail.
        return MemoryAvailability(None, None, f"windows:call-failed:{type(exc).__name__}")


def available_memory() -> MemoryAvailability:
    """How much physical memory the machine reports as free for new work.

    Used by a training launch two ways: written into the report so a cost paragraph quotes a measurement rather
    than a recollection, and compared against a declared floor when the plan names one.

    Returns:
        A :class:`MemoryAvailability`. On macOS, and where a call fails, both numbers are ``None`` and ``source``
        says which was attempted.
    """
    system = platform.system()
    if system == "Windows":
        return _windows_availability()
    if system == "Linux":
        return _proc_meminfo_availability()
    return MemoryAvailability(None, None, f"unsupported:{system}")
