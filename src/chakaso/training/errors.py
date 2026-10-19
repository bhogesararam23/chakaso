"""Typed errors for the training layers.

A checkpoint that is corrupt and a checkpoint that is simply from an incompatible schema are different
failures with different remedies, so they are different exception types — the convention the rest of the
package follows (a specific failure is never collapsed into a generic one).
"""

from __future__ import annotations

__all__ = [
    "CheckpointError",
    "CorruptedCheckpointError",
    "IncompatibleCheckpointError",
    "TrainingError",
]


class TrainingError(Exception):
    """Base for every error raised by :mod:`chakaso.training`."""


class CheckpointError(TrainingError):
    """Base for failures to load or validate a saved model checkpoint."""


class CorruptedCheckpointError(CheckpointError):
    """The stored bytes do not match their own checksum — the file was altered or truncated."""


class IncompatibleCheckpointError(CheckpointError):
    """The checkpoint is well-formed but speaks a schema or shape this code cannot load."""
