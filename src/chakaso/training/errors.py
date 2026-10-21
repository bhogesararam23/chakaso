"""Typed errors for the training layers.

A checkpoint that is corrupt and a checkpoint that is simply from an incompatible schema are different
failures with different remedies, so they are different exception types — the convention the rest of the
package follows (a specific failure is never collapsed into a generic one).
"""

from __future__ import annotations

__all__ = [
    "CheckpointError",
    "CorpusDriftError",
    "CorpusError",
    "CorruptedCheckpointError",
    "CorruptedCorpusArtifactError",
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


class CorpusError(TrainingError):
    """Base for failures involving a corpus manifest or the text it pins."""


class CorruptedCorpusArtifactError(CorpusError):
    """A stored manifest or vocabulary does not match its own checksum."""


class CorpusDriftError(CorpusError):
    """The files on disk no longer match what a pinned manifest recorded.

    This is the failure that makes a recorded experiment honest: the corpus a model was trained on is
    identified by the hashes of named files, so if any of those files changes, disappears or a new one
    joins the pattern, the corpus is *different* and saying so is the only acceptable response. Loading
    whatever happens to be on disk instead would silently move the dataset under a saved result.
    """
