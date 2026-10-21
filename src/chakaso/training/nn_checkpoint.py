"""A versioned, integrity-checked checkpoint for a trained Transformer.

A checkpoint is only useful if it can be loaded *as the model it was saved as*. That means the artifact has
to carry the architecture, the tokenizer the identifiers belong to, the data it was trained on, and the
seed — and it has to refuse to load when any of those disagree, because the failure mode of a mismatched
checkpoint is not an exception, it is a model that produces plausible-looking nonsense.

Format, and the two decisions in it:

* **Tensors plus metadata in one file, read with ``weights_only=True``.** ``torch``'s default loader can
  unpickle arbitrary objects; this format never needs that, so the loader is pinned to the restricted
  unpickler and a checkpoint that tries to carry anything else fails to load rather than executing it. A
  checkpoint file from an untrusted directory is therefore inert — the path is opened, the payload is
  data.
* **A content checksum over metadata *and* tensors.** The metadata block is hashed as written, each tensor
  is hashed over its bytes, and the artifact's checksum covers the pair. That distinguishes "this file was
  altered or truncated" from "this file is well-formed but from a format or configuration this code cannot
  honour" — the same distinction the JSON checkpoint of schema/2 makes, carried forward.

The tokenizer is stored **inline** (its merge table) and cross-checked against its recorded fingerprint, so
a checkpoint decodes on its own; that is the choice P25 already made for character checkpoints, and ADR-0035
records it for this format.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from chakaso.training.errors import (
    CheckpointError,
    CorruptedCheckpointError,
    IncompatibleCheckpointError,
)
from chakaso.training.subword import SubwordVocabulary
from chakaso.training.transformer import CausalTransformer, TransformerConfig

__all__ = [
    "ModelArtifact",
    "checkpoint_filename",
    "describe_environment",
    "load_artifact",
    "save_artifact",
]

_SCHEMA_VERSION = "chakaso-transformer-checkpoint/1"


def checkpoint_filename(experiment_id: str) -> str:
    """The conventional file name for one experiment's Transformer artifact."""
    return f"{experiment_id}-transformer.pt"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A trained model and everything required to reproduce its inference."""

    config: TransformerConfig
    state: Mapping[str, torch.Tensor]
    vocabulary: SubwordVocabulary
    corpus_fingerprint: str
    run_metadata: Mapping[str, Any] = field(default_factory=dict)
    step: int = 0
    optimizer_state: Mapping[str, Any] | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.corpus_fingerprint:
            message = "an artifact must record the corpus fingerprint it was trained on"
            raise IncompatibleCheckpointError(message)
        if self.step < 0:
            message = f"step must not be negative, got {self.step}"
            raise IncompatibleCheckpointError(message)

    @property
    def parameter_count(self) -> int:
        """Trainable scalars, from the declared architecture.

        Reported by configuration rather than by summing the stored tensors: a tied model writes the same
        matrix under two names, so a byte-level sum over states would double-count it and report a model
        that was never trained. The tensors are still checked against the architecture for shape and
        presence in :meth:`to_model`.
        """
        return self.config.parameter_count

    def checksum(self) -> str:
        """Digest over the metadata block and every tensor's bytes."""
        digest = hashlib.sha256()
        digest.update(self.metadata_document().encode("utf-8"))
        for name in sorted(self.state):
            tensor = self.state[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def metadata_document(self) -> str:
        """The canonical JSON the checksum covers: architecture, identity, environment, step."""
        return json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "config": self.config.as_dict(),
                "tokenizer_fingerprint": self.vocabulary.fingerprint,
                "tokenizer_size": self.vocabulary.size,
                "tokenizer_merges": [[left, right] for left, right in self.vocabulary.merges],
                "corpus_fingerprint": self.corpus_fingerprint,
                "step": self.step,
                "run_metadata": dict(self.run_metadata),
                "environment": dict(self.environment),
                "tensor_names": sorted(self.state),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_model(self) -> CausalTransformer:
        """Rebuild the model this artifact describes, loading the stored tensors into it.

        Raises:
            IncompatibleCheckpointError: a tensor is missing, extra, or has a shape the declared
                configuration does not produce.
        """
        model = CausalTransformer(self.config)
        expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        stored = dict(self.state)

        missing = sorted(set(expected) - set(stored))
        extra = sorted(set(stored) - set(expected))
        mismatched = sorted(
            name
            for name in set(expected) & set(stored)
            if tuple(stored[name].shape) != expected[name]
        )
        if missing or extra or mismatched:
            details = []
            if missing:
                details.append(f"missing tensors: {missing}")
            if extra:
                details.append(f"unexpected tensors: {extra}")
            if mismatched:
                details.append(
                    f"shape disagreement: {[(name, tuple(stored[name].shape)) for name in mismatched]}"
                )
            message = (
                "the stored tensors do not match the architecture the artifact declares "
                f"({self.config.as_dict()}) — " + "; ".join(details)
            )
            raise IncompatibleCheckpointError(message)
        # Tied embeddings make `head.weight` and `token_embedding.weight` the same object, so loading a
        # state dict that carries both under one name is expected; strict=False here is about accepting
        # that sharing, not about tolerating a genuine mismatch, which the check above already refused.
        model.load_state_dict(stored, strict=True)
        model.eval()
        return model


def save_artifact(artifact: ModelArtifact, path: Path, *, allow_overwrite: bool = False) -> None:
    """Write the artifact as one tensor-plus-metadata file.

    An existing file is not replaced unless the caller says so: a trained artifact is the expensive output
    of a run, and a second experiment pointed at the same path would otherwise destroy the first one
    silently. The interruption-safety writer passes ``allow_overwrite=True`` because rewriting its own
    resume file every evaluation is its whole purpose.

    Raises:
        IncompatibleCheckpointError: the artifact cannot be rebuilt from its own declared configuration,
            which is checked before anything is written so a bad artifact never reaches disk.
        CheckpointError: the path already exists and ``allow_overwrite`` was not given.
    """
    artifact.to_model()
    target = Path(path)
    if target.exists() and not allow_overwrite:
        message = (
            f"{str(target)!r} already holds an artifact; pass allow_overwrite=True to replace it, or "
            "choose a distinct path so the earlier run stays recoverable"
        )
        raise CheckpointError(message)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": _SCHEMA_VERSION,
            "metadata": artifact.metadata_document(),
            "checksum": artifact.checksum(),
            "tensors": {
                name: tensor.detach().cpu().contiguous() for name, tensor in artifact.state.items()
            },
            "optimizer": artifact.optimizer_state,
        },
        target,
    )


def load_artifact(path: Path) -> ModelArtifact:
    """Read and fully validate an artifact written by :func:`save_artifact`.

    The loader passes ``weights_only=True`` explicitly: an artifact must not be able to introduce a Python
    object into the process.

    Raises:
        CorruptedCheckpointError: the file is unreadable by the restricted loader, its checksum disagrees,
            or its metadata is not the JSON this format writes.
        IncompatibleCheckpointError: the schema, tokenizer fingerprint or declared vocabulary size
            disagrees with what is stored.
    """
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        # A single boundary where an unreadable, non-artifact, or hostile file becomes a typed error. The
        # exception is translated, not swallowed: the message names the path and the underlying reason.
        message = f"checkpoint at {str(path)!r} could not be loaded as an artifact: {type(exc).__name__}: {exc}"
        raise CorruptedCheckpointError(message) from exc
    if not isinstance(raw, dict):
        message = (
            f"checkpoint at {str(path)!r} is not an artifact payload (found {type(raw).__name__})"
        )
        raise IncompatibleCheckpointError(message)
    required = {"schema_version", "metadata", "checksum", "tensors"}
    missing = sorted(required - set(raw))
    unexpected = sorted(set(raw) - required - {"optimizer"})
    if missing or unexpected:
        details = f"missing blocks {missing}" if missing else f"unexpected blocks {unexpected}"
        message = f"checkpoint at {str(path)!r} is not a Chakaso artifact payload: {details}"
        raise IncompatibleCheckpointError(message)
    if raw["schema_version"] != _SCHEMA_VERSION:
        message = f"unsupported artifact schema {raw['schema_version']!r}"
        raise IncompatibleCheckpointError(message)

    metadata = raw["metadata"]
    checksum = raw["checksum"]
    tensors = raw["tensors"]
    if (
        not isinstance(metadata, str)
        or not isinstance(checksum, str)
        or not isinstance(tensors, dict)
    ):
        message = f"checkpoint at {str(path)!r} has a metadata, checksum or tensor block of the wrong type"
        raise CorruptedCheckpointError(message)

    try:
        recorded = json.loads(metadata)
    except json.JSONDecodeError as exc:
        message = (
            f"checkpoint at {str(path)!r} carries metadata that is not the JSON it claims: {exc}"
        )
        raise CorruptedCheckpointError(message) from exc

    rebuilt = hashlib.sha256()
    rebuilt.update(metadata.encode("utf-8"))
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        rebuilt.update(name.encode("utf-8"))
        rebuilt.update(str(tuple(tensor.shape)).encode("utf-8"))
        rebuilt.update(str(tensor.dtype).encode("utf-8"))
        rebuilt.update(tensor.view(torch.uint8).numpy().tobytes())
    if rebuilt.hexdigest() != checksum:
        message = (
            f"checkpoint at {str(path)!r} checksum does not match its content (stored {checksum[:12]}…, "
            f"recomputed {rebuilt.hexdigest()[:12]}…)"
        )
        raise CorruptedCheckpointError(message)

    try:
        config = TransformerConfig.from_mapping(recorded["config"])
        vocabulary = SubwordVocabulary(
            merges=tuple((int(pair[0]), int(pair[1])) for pair in recorded["tokenizer_merges"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        message = (
            f"checkpoint at {str(path)!r} declares a configuration this code cannot build: {exc}"
        )
        raise IncompatibleCheckpointError(message) from exc

    if vocabulary.fingerprint != recorded["tokenizer_fingerprint"]:
        message = (
            f"checkpoint at {str(path)!r} stores a tokenizer whose merge table does not hash to its "
            f"recorded fingerprint ({recorded['tokenizer_fingerprint']!r} stored, "
            f"{vocabulary.fingerprint!r} rebuilt)"
        )
        raise IncompatibleCheckpointError(message)
    if vocabulary.size != recorded["tokenizer_size"] or vocabulary.size != config.vocab_size:
        message = (
            f"checkpoint at {str(path)!r} pairs a {config.vocab_size}-wide architecture with a "
            f"{vocabulary.size}-identifier vocabulary"
        )
        raise IncompatibleCheckpointError(message)

    return ModelArtifact(
        config=config,
        state=dict(tensors),
        vocabulary=vocabulary,
        corpus_fingerprint=str(recorded["corpus_fingerprint"]),
        run_metadata=dict(recorded.get("run_metadata", {})),
        step=int(recorded.get("step", 0)),
        optimizer_state=raw.get("optimizer"),
        environment=dict(recorded.get("environment", {})),
    )


def describe_environment() -> dict[str, str]:
    """The environment block an artifact records: versions as observed, not claims about reproducibility."""
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
    }
