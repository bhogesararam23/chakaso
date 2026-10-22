"""A released Chakaso model: one directory, one contract document, everything verified by digest.

A checkpoint says what its weights are. It does not say what it is *for*, who may redistribute it, which
corpus it trained on under which licence, or what its measured limits are — and a developer who loads a
``.pt`` file has to take those on faith or dig them out of an experiment record by hand. This module makes
them part of the artifact, in a form a program can check.

The rules:

* **The contract is assembled, not typed.** :func:`build_release` reads the run's own ``report.json`` and
  the checkpoint's own metadata and copies provenance from them. A field nobody recorded is absent, not
  guessed — a release document that invents a corpus licence is worse than one that omits it.
* **Every byte a release names, it digests.** The checkpoint file, the report and the tokenizer fingerprint
  are recorded with SHA-256 values, and :func:`load_release` re-reads them. A swapped weights file is a
  typed refusal, not a model that happens to load.
* **Nothing here imports ``torch`` or ``numpy``.** Verification is JSON and hashes; turning the named file
  back into a model stays the job of :func:`chakaso.models.load_model`, which reports a missing extra by
  name. So a consumer can audit a release on a machine that cannot run it.
* **A release is never replaced quietly.** The same rule the checkpoint and the snapshot follow: writing
  over an existing release needs ``allow_overwrite``.
* **Limitations are part of the format.** :attr:`ModelRelease.known_limitations` is required and must be
  non-empty. An artifact that can only state what it does is how a toy model gets deployed as if it were
  not one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from chakaso.core.atomic import write_text_atomically
from chakaso.models.errors import CorruptedReleaseError, IncompatibleReleaseError

__all__ = [
    "RELEASE_DOCUMENT",
    "RELEASE_SCHEMA_VERSION",
    "ModelRelease",
    "build_release",
    "load_model_release",
    "load_release",
    "write_release",
]

RELEASE_SCHEMA_VERSION = "chakaso-model-release/1"
#: The document's own file name inside a release directory.
RELEASE_DOCUMENT = "model.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ModelRelease:
    """Everything a released Chakaso model asserts about itself.

    Attributes:
        model_id/model_version: What a consumer selects, and what a regression is measured against.
        family: The architecture family — ``transformer`` or ``decoder`` — which decides the checkpoint
            schema the file must speak.
        summary: One paragraph, written for someone deciding whether to load it.
        capabilities: Names the model can honour, from the boundary's own capability vocabulary.
        known_limitations: What the measured evidence does not support. Required and non-empty.
        checkpoint: ``file``, ``sha256``, ``schema_version`` and ``parameter_count`` of the weights file
            this release ships.
        tokenizer: ``fingerprint``, ``size`` and the merge count the checkpoint carries inline.
        training: experiment id, corpus identity, seeds, budget and the measured held-out numbers, copied
            from the run's report.
        software: interpreter, framework versions and thread count the artifact was produced with.
        licence: terms for the weights, terms for the training data, and the redistribution answer.
        created_utc: ISO timestamp of the release, not of the training.
        reproduce: the command a reader runs to rebuild the artifact from its declared inputs.
        document_sha256: checksum of this document without its own ``checksum`` field.
    """

    model_id: str
    model_version: str
    family: str
    summary: str
    capabilities: tuple[str, ...]
    known_limitations: tuple[str, ...]
    checkpoint: dict[str, Any]
    tokenizer: dict[str, Any]
    training: dict[str, Any]
    software: dict[str, Any]
    licence: dict[str, str]
    created_utc: str
    reproduce: str
    schema_version: str = RELEASE_SCHEMA_VERSION
    checksum: str = ""

    def __post_init__(self) -> None:
        for name in ("model_id", "model_version", "family", "summary", "created_utc", "reproduce"):
            if not str(getattr(self, name)).strip():
                message = f"a release must state {name}; an empty field is not an unknown"
                raise IncompatibleReleaseError(message)
        if not self.capabilities:
            message = "a release must name at least one capability it honours"
            raise IncompatibleReleaseError(message)
        if not self.known_limitations:
            message = (
                "a release must state its known limitations; an artifact that can only describe what it "
                "does is not a release, it is a claim"
            )
            raise IncompatibleReleaseError(message)
        for name, block in (
            ("checkpoint", self.checkpoint),
            ("tokenizer", self.tokenizer),
            ("training", self.training),
            ("software", self.software),
            ("licence", self.licence),
        ):
            if not isinstance(block, dict) or not block:
                message = f"a release must carry a non-empty {name} block"
                raise IncompatibleReleaseError(message)
        for key in ("file", "sha256", "schema_version", "parameter_count"):
            if key not in self.checkpoint:
                message = f"the checkpoint block is missing {key!r}"
                raise IncompatibleReleaseError(message)
        for key in ("model", "training_data", "redistribution"):
            if key not in self.licence:
                message = f"the licence block is missing {key!r}"
                raise IncompatibleReleaseError(message)

    def document(self) -> dict[str, Any]:
        """The release as stored data, without the checksum field."""
        values = asdict(self)
        values.pop("checksum")
        values["capabilities"] = list(values["capabilities"])
        values["known_limitations"] = list(values["known_limitations"])
        return values

    def compute_checksum(self) -> str:
        """Digest over everything the document states."""
        return hashlib.sha256(_canonical(self.document()).encode("utf-8")).hexdigest()

    def to_document(self) -> dict[str, Any]:
        """The stored form: the document plus its own checksum."""
        stored = self.document()
        stored["checksum"] = self.compute_checksum()
        return stored

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> ModelRelease:
        """Rebuild and verify a stored release document.

        Raises:
            CorruptedReleaseError: the checksum is absent or disagrees, or the JSON shape is wrong.
            IncompatibleReleaseError: the schema is not this one, or a required statement is missing.
        """
        if not isinstance(document, dict):
            message = "a release document must be a JSON object"
            raise CorruptedReleaseError(message)
        if document.get("schema_version") != RELEASE_SCHEMA_VERSION:
            message = (
                f"unsupported release schema {document.get('schema_version')!r}; this code reads "
                f"{RELEASE_SCHEMA_VERSION!r}"
            )
            raise IncompatibleReleaseError(message)
        stored = document.get("checksum")
        content = {key: value for key, value in document.items() if key != "checksum"}
        if not isinstance(stored, str) or not stored:
            message = "a release document must carry its own checksum"
            raise CorruptedReleaseError(message)
        if hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest() != stored:
            message = "release checksum does not match its content — the document was altered"
            raise CorruptedReleaseError(message)
        for name in (
            "model_id",
            "model_version",
            "family",
            "summary",
            "created_utc",
            "reproduce",
            "capabilities",
            "known_limitations",
        ):
            if name not in content:
                message = f"release document is missing {name!r}"
                raise IncompatibleReleaseError(message)
        try:
            return cls(
                **{
                    **content,
                    "capabilities": tuple(content["capabilities"]),
                    "known_limitations": tuple(content["known_limitations"]),
                    "checksum": stored,
                }
            )
        except (TypeError, ValueError) as exc:
            message = f"release document cannot be rebuilt: {exc}"
            raise IncompatibleReleaseError(message) from exc


def build_release(
    directory: Path,
    *,
    model_id: str,
    model_version: str,
    summary: str,
    capabilities: tuple[str, ...],
    known_limitations: tuple[str, ...],
    checkpoint_name: str,
    report_name: str = "report.json",
    licence_model: str,
    licence_training_data: str,
    redistribution: str,
    created_utc: str,
    reproduce: str,
    family: str | None = None,
) -> ModelRelease:
    """Assemble the contract from the artifacts already in ``directory``.

    Provenance is read, not typed: the corpus identity, seeds, budget and held-out numbers come from the
    run's ``report.json``, and the architecture and tokenizer identity from the checkpoint's declared
    configuration. The caller supplies only what an artifact cannot know — the model's name and version,
    its licence terms, what it is for, and what it is not.

    Raises:
        IncompatibleReleaseError: the checkpoint or the report is not in the directory, or the report is
            missing a field the contract must state.
    """
    root = Path(directory)
    checkpoint = root / checkpoint_name
    if not checkpoint.is_file():
        message = f"release directory {str(root)!r} has no checkpoint named {checkpoint_name!r}"
        raise IncompatibleReleaseError(message)
    report_path = root / report_name
    if not report_path.is_file():
        message = f"release directory {str(root)!r} has no run report named {report_name!r}"
        raise IncompatibleReleaseError(message)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        corpus = report["corpus"]
        model = report["transformer"]
        training = {
            "experiment_id": report["experiment"],
            "corpus": {
                "name": corpus["name"],
                "version": corpus["version"],
                "licence": corpus["licence"],
                "fingerprint": corpus["fingerprint"],
                "documents": corpus["documents"],
                "characters": corpus["characters"],
                "snapshot_checksum": None
                if corpus.get("snapshot") is None
                else corpus["snapshot"]["checksum"],
            },
            "seed": model["budget"]["seed"],
            "split_seed": corpus["split_seed"],
            "schedule": model["budget"]["schedule"],
            "max_steps": model["budget"]["max_steps"],
            "best_step": model["budget"]["best_step"],
            "completion": model["budget"]["completion"],
            "batch_size": model["budget"]["batch_size"],
            "effective_batch_size": model["budget"]["effective_batch_size"],
            "learning_rate": model["budget"]["learning_rate"],
            "warmup_steps": model["budget"]["warmup_steps"],
            "held_out": {
                "bits_per_character": model["test"]["bits_per_character"],
                "nats_per_token": model["test"]["loss_nats_per_token"],
                "perplexity": model["test"]["perplexity"],
                "token_accuracy": model["test"]["token_accuracy"],
                "scored_positions": model["test"]["positions_scored"],
            },
            "baselines": {
                "unigram_bits_per_character": report["baselines_subword"]["unigram"][
                    "bits_per_character"
                ],
                "uniform_bits_per_character": report["baselines_subword"]["uniform"][
                    "bits_per_character"
                ],
            },
            "report_checksum": _digest(report_path),
        }
        tokenizer = {
            "fingerprint": report["tokenizer"]["fingerprint"],
            "size": report["tokenizer"]["size"],
            "merges": report["tokenizer"]["merges"],
        }
        software = dict(report["environment"])
    except KeyError as exc:
        message = (
            f"the report in {str(root)!r} does not carry {exc}; a release cannot state provenance the run "
            "did not record"
        )
        raise IncompatibleReleaseError(message) from exc

    release = ModelRelease(
        model_id=model_id,
        model_version=model_version,
        family=report["model_family"] if family is None else family,
        summary=summary,
        capabilities=capabilities,
        known_limitations=known_limitations,
        checkpoint={
            "file": checkpoint_name,
            "sha256": _digest(checkpoint),
            # Read from the run's own artifact block rather than assembled from the family name: the file
            # states its schema, and a release that inferred it could sign a contract around a checkpoint the
            # loader would rebuild differently.
            "schema_version": model["artifact"]["schema"],
            "parameter_count": model["parameter_count"],
            "size_bytes": checkpoint.stat().st_size,
        },
        tokenizer=tokenizer,
        training=training,
        software=software,
        licence={
            "model": licence_model,
            "training_data": licence_training_data,
            "redistribution": redistribution,
        },
        created_utc=created_utc,
        reproduce=reproduce,
    )
    # The checksum is part of the object, so a release built here and a release read back from disk
    # compare equal rather than differing in one derived field.
    return replace(release, checksum=release.compute_checksum())


def write_release(
    release: ModelRelease,
    directory: Path,
    *,
    allow_overwrite: bool = False,
) -> Path:
    """Write the contract document into a release directory.

    Raises:
        IncompatibleReleaseError: the named checkpoint is not present in the directory, so the release
            would describe a file that is not there.
        ModelError: a document already exists and overwriting was not asked for.
    """
    root = Path(directory)
    checkpoint = root / release.checkpoint["file"]
    if not checkpoint.is_file():
        message = (
            f"release names {release.checkpoint['file']!r} but {str(checkpoint)!r} does not exist; a "
            "release is written after its artifact is in place"
        )
        raise IncompatibleReleaseError(message)
    target = root / RELEASE_DOCUMENT
    if target.exists() and not allow_overwrite:
        message = (
            f"{str(target)!r} already exists; a released model's contract is not replaced unless the "
            "caller passes allow_overwrite=True"
        )
        raise IncompatibleReleaseError(message)
    document = release.to_document()
    write_text_atomically(
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n", target
    )
    return target


def load_release(directory: Path) -> ModelRelease:
    """Read a release document and verify that the artifact it names is the one on disk.

    Raises:
        CorruptedReleaseError: the document is missing, unreadable, or its checksum disagrees.
        IncompatibleReleaseError: the schema is unknown, a required statement is absent, or the named
            checkpoint is missing or hashes to a different value than the contract declares.
    """
    root = Path(directory)
    document_path = root / RELEASE_DOCUMENT
    if not document_path.is_file():
        message = f"{str(root)!r} holds no release document named {RELEASE_DOCUMENT!r}"
        raise CorruptedReleaseError(message)
    try:
        document = json.loads(document_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"release document at {str(document_path)!r} is not valid JSON: {exc}"
        raise CorruptedReleaseError(message) from exc
    release = ModelRelease.from_document(document)

    checkpoint = root / release.checkpoint["file"]
    if not checkpoint.is_file():
        message = f"the release names checkpoint {release.checkpoint['file']!r} but it is not in {str(root)!r}"
        raise IncompatibleReleaseError(message)
    observed = _digest(checkpoint)
    if observed != release.checkpoint["sha256"]:
        message = (
            f"checkpoint {release.checkpoint['file']!r} hashes to {observed[:16]}…, not the "
            f"{release.checkpoint['sha256'][:16]}… the release declares — the weights are not the ones "
            "this contract describes"
        )
        raise IncompatibleReleaseError(message)
    if release.checkpoint.get("size_bytes") is not None and (
        int(release.checkpoint["size_bytes"]) != checkpoint.stat().st_size
    ):
        message = (
            f"checkpoint size on disk ({checkpoint.stat().st_size}) disagrees with the release "
            f"({release.checkpoint['size_bytes']})"
        )
        raise IncompatibleReleaseError(message)
    return release


def load_model_release(directory: Path) -> tuple[ModelRelease, Any]:
    """Verify a release, then load the model it names, in that order.

    Verification comes first and is not optional: a release whose checkpoint was swapped is refused before
    a single weight is read, so the failure a caller sees is "this artifact is not the one the contract
    describes", never a plausible generation from someone else's weights.

    Raises:
        ModelError: the release does not verify, or loading the checkpoint needs an extra that is not
            installed. The extra is named by :func:`chakaso.models.load_model`.
    """
    from chakaso.models.load import load_model

    release = load_release(directory)
    model = load_model(Path(directory) / release.checkpoint["file"])
    return release, model
