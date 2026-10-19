"""A self-describing, integrity-checked checkpoint format.

A trained artifact is only reproducible if it can be saved and loaded exactly, and if a corrupted or
foreign checkpoint is *refused* rather than half-read. This module serializes the model's parameters
together with the configuration, vocabulary and dataset identity that produced them, into a single
document that carries a content-derived checksum.

The format is JSON with float64 arrays written by Python's exact round-tripping repr, so a save/load
cycle reproduces every parameter bit-for-bit. JSON (rather than numpy's own binary format) is chosen
because it is inspectable by a human and stable across the project's tooling; a first-experiment
checkpoint is small, so readability outweighs compactness. Large artifacts still stay out of Git per
the artifact policy — the checkpoint's *fingerprint*, not necessarily its bytes, is what a record cites.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt

from chakaso.training.config import TrainingConfig
from chakaso.training.errors import CorruptedCheckpointError, IncompatibleCheckpointError
from chakaso.training.model import Params
from chakaso.training.tokenizer import CharVocabulary

__all__ = ["Checkpoint", "checkpoint_filename"]

F64 = npt.NDArray[np.float64]

_SCHEMA_VERSION = "chakaso-checkpoint/2"
_ARRAY_NAMES = ("embedding", "hidden_weight", "hidden_bias", "output_weight", "output_bias")


def checkpoint_filename(experiment_id: str) -> str:
    """The conventional file name for one experiment's checkpoint."""
    return f"{experiment_id}-checkpoint.json"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A trained model plus everything needed to load and *use* it.

    The vocabulary characters are stored, not only their fingerprint, so a checkpoint can rebuild the
    exact identifier space and decode output — a fingerprint alone identifies a vocabulary but cannot
    turn identifiers back into text, which an inference path requires. On load the stored characters are
    checked against the stored fingerprint, so the vocabulary cannot silently disagree with the weights.
    """

    params: Params
    config: TrainingConfig
    vocab_size: int
    vocab_fingerprint: str
    dataset_digest: str
    vocab_chars: tuple[str, ...]

    def vocabulary(self) -> CharVocabulary:
        """Rebuild the exact character vocabulary the model was trained with."""
        return CharVocabulary(("".join(self.vocab_chars),))

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "config": self.config.as_dict(),
            "vocab_size": self.vocab_size,
            "vocab_fingerprint": self.vocab_fingerprint,
            "vocab_chars": list(self.vocab_chars),
            "dataset_digest": self.dataset_digest,
            "arrays": {name: _named(self.params, name).tolist() for name in _ARRAY_NAMES},
        }

    @staticmethod
    def _canonical(content: dict[str, Any]) -> str:
        # A single deterministic serialization is the checksum's input; sort_keys and fixed separators
        # make it stable regardless of dict insertion order.
        return json.dumps(content, sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        """A content hash over the whole checkpoint (parameters and identity together)."""
        return hashlib.sha256(self._canonical(self._content()).encode("utf-8")).hexdigest()

    def to_document(self) -> dict[str, Any]:
        """The full on-disk document: content plus its own checksum."""
        content = self._content()
        document = dict(content)
        document["checksum"] = self.fingerprint()
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> Checkpoint:
        """Rebuild and fully validate a checkpoint from its document.

        Raises:
            CorruptedCheckpointError: the stored checksum does not match the stored content.
            IncompatibleCheckpointError: the schema is unknown, a required field is missing, or an
                array's shape disagrees with the declared configuration.
        """
        if "checksum" not in document:
            message = "checkpoint document has no checksum field"
            raise CorruptedCheckpointError(message)

        stored_checksum = document["checksum"]
        content = {key: value for key, value in document.items() if key != "checksum"}
        recomputed = hashlib.sha256(cls._canonical(content).encode("utf-8")).hexdigest()
        if recomputed != stored_checksum:
            message = (
                "checkpoint checksum mismatch: the file was altered or truncated "
                f"(stored {stored_checksum[:12]}…, recomputed {recomputed[:12]}…)"
            )
            raise CorruptedCheckpointError(message)

        if content.get("schema_version") != _SCHEMA_VERSION:
            message = f"unsupported checkpoint schema {content.get('schema_version')!r}"
            raise IncompatibleCheckpointError(message)

        try:
            config = TrainingConfig(**content["config"])
            vocab_size = int(content["vocab_size"])
            vocab_fingerprint = str(content["vocab_fingerprint"])
            dataset_digest = str(content["dataset_digest"])
            raw_arrays = content["arrays"]
            vocab_chars = tuple(str(c) for c in content["vocab_chars"])
        except (KeyError, TypeError, ValueError) as exc:
            message = f"checkpoint is missing or mis-typed a required field: {exc}"
            raise IncompatibleCheckpointError(message) from exc

        # The stored characters must be exactly the vocabulary the weights were trained in: same size,
        # and a fingerprint that recomputes to the stored one. A mismatch is a foreign or tampered
        # vocabulary, not a corruption of the bytes (the checksum already passed).
        if len(vocab_chars) + 1 != vocab_size:
            message = (
                f"vocabulary has {len(vocab_chars)} characters but declares vocab_size "
                f"{vocab_size} (expected one fewer, the reserved unknown id)"
            )
            raise IncompatibleCheckpointError(message)
        rebuilt = CharVocabulary(("".join(vocab_chars),))
        if rebuilt.fingerprint != vocab_fingerprint:
            message = "the stored vocabulary characters do not hash to the stored vocab_fingerprint"
            raise IncompatibleCheckpointError(message)

        expected = {
            "embedding": (vocab_size, config.embed_dim),
            "hidden_weight": (config.context_size * config.embed_dim, config.hidden_dim),
            "hidden_bias": (config.hidden_dim,),
            "output_weight": (config.hidden_dim, vocab_size),
            "output_bias": (vocab_size,),
        }
        arrays: dict[str, F64] = {}
        for name in _ARRAY_NAMES:
            if name not in raw_arrays:
                message = f"checkpoint is missing the {name!r} array"
                raise IncompatibleCheckpointError(message)
            array = np.array(raw_arrays[name], dtype=np.float64)
            if array.shape != expected[name]:
                message = (
                    f"{name!r} has shape {array.shape}, expected {expected[name]} "
                    "for the declared configuration"
                )
                raise IncompatibleCheckpointError(message)
            arrays[name] = array

        params = Params(
            embedding=arrays["embedding"],
            hidden_weight=arrays["hidden_weight"],
            hidden_bias=arrays["hidden_bias"],
            output_weight=arrays["output_weight"],
            output_bias=arrays["output_bias"],
        )
        return cls(
            params=params,
            config=config,
            vocab_size=vocab_size,
            vocab_fingerprint=vocab_fingerprint,
            dataset_digest=dataset_digest,
            vocab_chars=vocab_chars,
        )

    def save(self, path: Path) -> None:
        """Write the checkpoint document to ``path`` as UTF-8 JSON."""
        text = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Read and validate a checkpoint written by :meth:`save`.

        Raises:
            CorruptedCheckpointError: the file is not valid JSON.
        """
        raw = Path(path).read_text(encoding="utf-8")
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            message = f"checkpoint at {path} is not valid JSON: {exc}"
            raise CorruptedCheckpointError(message) from exc
        if not isinstance(document, dict):
            message = f"checkpoint at {path} must be a JSON object"
            raise CorruptedCheckpointError(message)
        return cls.from_document(document)


def _named(params: Params, name: str) -> F64:
    for key, array in params.items():
        if key == name:
            return array
    message = f"parameter {name!r} is not present in the bundle"
    raise KeyError(message)
