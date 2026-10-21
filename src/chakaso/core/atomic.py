"""Write an artifact so a reader sees the whole file or sees nothing.

Every artifact this project cites in a record — a corpus snapshot, a checkpoint, an experiment report — is
read back later by something that assumes it is complete. A writer that dies halfway through, or a disk that
fills up, would otherwise leave a truncated file that looks exactly like a real artifact until someone tries
to verify it. Staging the write next to the destination and swapping it into place on success means the
destination is either the previous file or the new one, never a mixture.

This covers a process that dies during a write. It does not claim more than that: the staged file is
flushed, but the directory entry created by the swap is not separately fsynced, so a power cut could still
lose a rename on some filesystems.

Standard library only, and deliberately free of policy: whether an existing file may be replaced is a
question each artifact layer answers in its own words, so the guard stays with the callers.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["staged_write", "write_text_atomically"]

_SUFFIX = ".partial"


@contextmanager
def staged_write(destination: Path) -> Iterator[Path]:
    """Yield a path in ``destination``'s directory and move it onto ``destination`` when the block returns.

    Raises:
        OSError: the destination's directory cannot be created, or the swap fails. The staged file is
            removed either way, so a failed write leaves nothing behind to mistake for an artifact.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=_SUFFIX
    )
    os.close(descriptor)
    staged = Path(temporary)
    try:
        yield staged
        staged.replace(target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def write_text_atomically(payload: str, destination: Path) -> None:
    """Write ``payload`` as UTF-8 in one swap, with no newline translation.

    ``newline=""`` is the part that matters for verification: a checksum computed over this string is only
    reproducible from the file's bytes if the bytes written are the characters that were hashed. In default
    text mode on Windows they would not be — an indented JSON report would gain CRLF endings and no reader
    could recompute its digest.
    """
    with (
        staged_write(destination) as staged,
        staged.open("w", encoding="utf-8", newline="") as handle,
    ):
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
