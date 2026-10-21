"""E2605: the experiment that compares two tokenizations and two architectures on one text.

This module produces a measured result rather than a demonstration. It runs the whole model path — manifest,
document split, subword tokenizer, packing, a mini-batch character model, a Transformer, validation-selected
early stopping, held-out scoring — and reports every number the record states in units both models can share.

Three things make the comparison honest, and the code enforces them:

* **One text, one split, computed once.** :func:`prepare_corpus` performs the document split a single time
  and every consumer — both tokenizations, both models, both baselines — reads the same three sides. A
  second split anywhere else would be a different experiment, and the character baseline in particular gets
  its windows from the *training documents* only.
* **One metric.** Bits per character comes from
  :func:`chakaso.training.model_metrics.metrics_from_scores` for every model and both baselines, so a
  subword model cannot look better by splitting the same sentence into more tokens. Per-token loss and
  perplexity are reported as well and labelled as tokenizer-bound.
* **Baselines on the same tokens.** A uniform and a unigram predictor are scored over the same test
  positions; a model that does not beat them has not learned anything worth recording.

Requires the optional ``[nn]`` extra, and ``[train]`` for the character baseline. Nothing in the
application layer imports this module: an experiment is a consumer of the library, not a dependency of it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chakaso.core.atomic import write_text_atomically
from chakaso.core.hashing import join_parts, sha256_hex
from chakaso.training.baselines import UnigramBaseline, uniform_next_token_loss
from chakaso.training.config import TrainingConfig
from chakaso.training.data import (
    Split,
    UnitSplit,
    character_windows,
    split_text_units,
)
from chakaso.training.decoder import DecoderConfig
from chakaso.training.errors import CorpusDriftError, CorpusError
from chakaso.training.loop import run_minibatch_training
from chakaso.training.manifest import (
    CorpusManifest,
    CorpusSnapshot,
    build_manifest,
    read_snapshot,
    save_manifest,
    verify_corpus,
    write_snapshot,
)
from chakaso.training.model import CharCausalMLP, batches_from_examples
from chakaso.training.model_metrics import ModelMetrics, metrics_from_scores
from chakaso.training.nn_checkpoint import (
    TrainedModel,
    checkpoint_filename,
    load_artifact,
    save_artifact,
)
from chakaso.training.nn_evaluation import score_model_on_examples
from chakaso.training.nn_training import (
    TransformerTrainingConfig,
    train_transformer,
    verify_resume_compatibility,
)
from chakaso.training.packing import PackedBatch, pack_units
from chakaso.training.subword import SubwordVocabulary, train_subword_vocabulary
from chakaso.training.tokenizer import UNK_ID, CharVocabulary
from chakaso.training.transformer import TransformerConfig

__all__ = [
    "CHARACTER_CONTEXT",
    "CORPUS_NAME",
    "EXPERIMENT_ID",
    "REPO_TEXT_CORPUS",
    "CorpusSpec",
    "ExperimentPlan",
    "PreparedCorpus",
    "prepare_corpus",
    "run_experiment",
    "summarise",
    "write_report",
]

EXPERIMENT_ID = "E2605"
CORPUS_NAME = "chakaso-repo-text-v1"
#: The character baseline's conditioning window in characters. The Transformer's is 128 *subword tokens*,
#: which covers far more text; the asymmetry is a property of the architectures being compared, not a
#: handicap either one was given, and the record states both.
CHARACTER_CONTEXT = 16
_VALIDATION_FRACTION = 0.15
_TEST_FRACTION = 0.15
_FAMILIES = frozenset({"transformer", "decoder"})
_SCHEDULES = frozenset({"constant", "cosine"})
_PROMPTS = (
    "# Chakaso\n\nChakaso is a",
    "## Status\n\nThe retrieval layer",
    "The tokenizer is trained by the project because",
)


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """What corpus a run may train on, and where its artifacts live.

    Attributes:
        name/version/licence/source: The provenance a record cites. Licence and source are required
            because "we trained on some markdown" is not provenance.
        include/extensions/exact_paths: The discovery rule, passed to
            :func:`chakaso.training.manifest.build_manifest` unchanged.
    """

    name: str
    version: str
    licence: str
    source: str
    include: tuple[str, ...] = ("docs/", "*.md")
    extensions: tuple[str, ...] = (".md",)
    exact_paths: tuple[str, ...] = ("LICENSE", "AGENTS.md")

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "licence", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                message = f"corpus spec field {field_name!r} must be a non-empty string"
                raise ValueError(message)
        if "/" in self.name or "\\" in self.name:
            message = f"corpus name {self.name!r} cannot be used as a file name"
            raise ValueError(message)

    def snapshot_path(self, work: Path) -> Path:
        """Where this corpus's snapshot lives in a run directory.

        The manifest and tokenizer records are named from the resolved manifest itself, because a run that
        loads a snapshot may be training on a corpus this spec was never told about.
        """
        return work / f"{self.name}-snapshot.json"


#: The corpus E2605 measured: this repository's own tracked prose, discovered from the repository root.
REPO_TEXT_CORPUS = CorpusSpec(
    name=CORPUS_NAME,
    version="1.0.0",
    licence="Apache-2.0 (the project's own licence; the text is authored for Chakaso)",
    source="Chakaso's tracked markdown and LICENSE files, discovered from the include spec on this plan.",
)


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Every knob of one run, validated on construction.

    Attributes:
        root: Repository root the corpus is prepared from.
        work_dir: Where the snapshot, tokenizer and checkpoints are written (never committed).
        experiment_id: Names this run's artifacts and the report's ``experiment`` field. A study gives each
            experiment its own so an E2606 checkpoint is never mistaken for the E2605 one.
        corpus: Which corpus the run is allowed to train on, and how it is discovered.
        vocab_size: Target subword vocabulary size, including the byte base and reserved separator.
        context_length: Transformer context, in tokens.
        batch_size: Sequences per optimizer step.
        d_model/num_layers/num_heads/feedforward_dim: Transformer geometry.
        max_steps: Absolute ceiling on optimizer steps.
        eval_every: Validation interval in steps.
        patience: Evaluations without improvement before early stopping.
        min_delta: Held-out improvement that counts against patience, in nats.
        learning_rate/weight_decay/warmup_steps/gradient_clip_norm: Optimization settings.
        seed: Master seed for initialization and batching. split_seed: the seed the document split uses.
        character_*: The character baseline's own geometry and budget.
        samples_per_prompt: Tokens generated for each qualitative sample.
        corpus_snapshot/expected_corpus_fingerprint: Which corpus this run is allowed to train on.
    """

    root: Path
    work_dir: Path
    experiment_id: str = EXPERIMENT_ID
    corpus: CorpusSpec = REPO_TEXT_CORPUS
    vocab_size: int = 1024
    #: The frequency floor the byte-pair merge search uses. It is a plan field rather than a constant
    #: because the tokenizer cache is keyed by it: a run that changed the floor while naming the same
    #: vocabulary size would otherwise be able to read a tokenizer fitted under different rules.
    tokenizer_min_frequency: int = 3
    context_length: int = 128
    batch_size: int = 32
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    feedforward_dim: int = 384
    max_steps: int = 2400
    eval_every: int = 60
    patience: int = 8
    min_delta: float = 0.001
    learning_rate: float = 6e-4
    weight_decay: float = 0.01
    warmup_steps: int = 60
    gradient_clip_norm: float = 1.0
    seed: int = 7
    #: The seed the document split is drawn with, separately from the model's. A study that varies seeds
    #: fixes the split, or the runs would differ in their data as well as their initialization — and the
    #: held-out documents would no longer be the same documents. Left unset it is ``seed``, which is what
    #: E2605 measured.
    split_seed: int | None = None
    character_context: int = CHARACTER_CONTEXT
    character_embed_dim: int = 32
    character_hidden_dim: int = 64
    character_batch_windows: int = 256
    character_steps: int = 4000
    character_eval_every: int = 200
    character_learning_rate: float = 2e-2
    samples_per_prompt: int = 96
    #: A snapshot of the corpus, read through :func:`chakaso.training.manifest.read_snapshot`, which
    #: re-checks every document digest. Given, the run trains on those bytes and never touches the tracked
    #: files; absent, the run prepares from the tree, writes a snapshot beside its manifest, and a later run
    #: in the same directory picks it up. Either way the corpus the report names is the corpus that ran.
    corpus_snapshot: Path | None = None
    #: The corpus fingerprint this run is required to use. Declared by a reproduction or a seed study, so
    #: that loading *a* snapshot is never the same as loading *the* snapshot.
    expected_corpus_fingerprint: str | None = None
    #: An experiment run rewrites its own artifact only when told to. Re-running into a directory that
    #: already holds a measured run's artifact is how a result gets silently replaced by a different one,
    #: so ADR-0035's overwrite guard applies here too.
    overwrite_artifacts: bool = False
    #: Which architecture family the run builds. ``transformer`` is what E2605, E2606 and E2607 measured and
    #: stays the default so those protocols are reproducible by name; ``decoder`` builds the release-path
    #: architecture in :mod:`chakaso.training.decoder`, whose checkpoints carry their own schema.
    model_family: str = "transformer"
    #: Rotary base for the decoder family. Ignored by the transformer family, which has a position table.
    rope_theta: float = 10_000.0
    #: Learning-rate policy after warmup: ``constant`` (what was measured) or ``cosine`` (annealed).
    schedule: str = "constant"
    decay_to: float = 0.1
    #: Micro-batches per optimizer step, for an effective batch the memory cannot hold at once.
    grad_accum_steps: int = 1
    #: How often the resumable artifact is rewritten; zero means at every evaluation.
    checkpoint_every: int = 0
    #: Whether the character-model baseline runs at all. Its windows are materialized per character, so on a
    #: corpus of tens of millions of characters it costs more memory than this project's machine has; a run
    #: that skips it says so in its report rather than pretending the number was measured.
    character_baseline: bool = True
    #: A resume point written by an earlier pass of *this* run. Verified against the plan's corpus,
    #: architecture, tokenizer and configuration before a single step is taken.
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        if not self.root.is_dir():
            message = f"experiment root {str(self.root)!r} is not a directory"
            raise ValueError(message)
        if not self.experiment_id or "/" in self.experiment_id or "\\" in self.experiment_id:
            message = f"experiment_id {self.experiment_id!r} cannot be used as a file name"
            raise ValueError(message)
        if self.model_family not in _FAMILIES:
            message = f"model_family {self.model_family!r} is not one of {sorted(_FAMILIES)}"
            raise ValueError(message)
        if self.schedule not in _SCHEDULES:
            message = f"schedule {self.schedule!r} is not one of {sorted(_SCHEDULES)}"
            raise ValueError(message)
        if isinstance(self.decay_to, bool) or not 0.0 <= self.decay_to < 1.0:
            message = f"decay_to must be in [0, 1), got {self.decay_to!r}"
            raise ValueError(message)
        if isinstance(self.tokenizer_min_frequency, bool) or self.tokenizer_min_frequency < 1:
            message = f"tokenizer_min_frequency must be >= 1, got {self.tokenizer_min_frequency!r}"
            raise ValueError(message)
        if isinstance(self.grad_accum_steps, bool) or self.grad_accum_steps < 1:
            message = f"grad_accum_steps must be >= 1, got {self.grad_accum_steps!r}"
            raise ValueError(message)
        if isinstance(self.checkpoint_every, bool) or self.checkpoint_every < 0:
            message = f"checkpoint_every must be >= 0, got {self.checkpoint_every!r}"
            raise ValueError(message)
        if self.model_family == "decoder":
            head_width = self.d_model // self.num_heads
            if self.d_model % self.num_heads or head_width % 2:
                message = (
                    "the decoder family needs a head width that is an even integer: d_model "
                    f"{self.d_model} with num_heads {self.num_heads} gives {self.d_model / self.num_heads}"
                )
                raise ValueError(message)
            if self.rope_theta <= 1.0:
                message = (
                    f"rope_theta must exceed 1.0 for the decoder family, got {self.rope_theta}"
                )
                raise ValueError(message)
        if self.vocab_size < 258:
            message = f"vocab_size must be >= 258 (byte base + separator + one merge), got {self.vocab_size}"
            raise ValueError(message)
        if self.context_length < 2 or self.batch_size < 1:
            message = "context_length must be >= 2 and batch_size >= 1"
            raise ValueError(message)
        if self.max_steps < self.eval_every:
            message = "the Transformer's step budget must reach at least one evaluation"
            raise ValueError(message)
        if self.character_batch_windows < 1 or self.character_context < 1:
            message = "the character baseline needs a positive window and batch"
            raise ValueError(message)
        if (
            self.expected_corpus_fingerprint is not None
            and len(self.expected_corpus_fingerprint) != 16
        ):
            message = (
                "expected_corpus_fingerprint must be the 16 hexadecimal characters a manifest fingerprint "
                f"has, got {self.expected_corpus_fingerprint!r}"
            )
            raise ValueError(message)

    @property
    def effective_split_seed(self) -> int:
        """The seed the document split uses: the declared one, or the run's master seed."""
        return self.seed if self.split_seed is None else self.split_seed

    def snapshot_path(self) -> Path:
        """Where this run's corpus snapshot is read from and written to."""
        return (
            self.corpus_snapshot
            if self.corpus_snapshot is not None
            else self.corpus.snapshot_path(self.work_dir)
        )

    def character_evaluation_interval(self) -> int:
        """The character baseline's validation interval, clamped to its own budget.

        A short budget must still evaluate, or early stopping has nothing to select on; the clamped value
        is what the report records, so a reader sees the interval that actually ran rather than the one a
        long run would have used.
        """
        return min(self.character_eval_every, max(1, self.character_steps // 10))


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    """The pinned corpus, its single document split, and both tokenizations of it."""

    manifest: CorpusManifest
    split: UnitSplit
    vocabulary: SubwordVocabulary
    char_vocabulary: CharVocabulary
    train: PackedBatch
    validation: PackedBatch
    test: PackedBatch
    char_windows: dict[str, tuple[np.ndarray, np.ndarray]]
    timings: dict[str, float]
    corpus_origin: str = "working tree"
    snapshot: CorpusSnapshot | None = None
    #: Whether this run fitted the tokenizer or reused the one an earlier run fitted from identical inputs.
    #: Recorded because a seed study pays the tokenizer's cost once rather than once per seed, and a reader
    #: of a single run's report should be able to tell which happened.
    tokenizer_reused: bool = False

    @property
    def tokens(self) -> dict[str, int]:
        """Packed identifiers per side — the sizes a record has to state."""
        return {
            "train": self.train.tokens_available,
            "validation": self.validation.tokens_available,
            "test": self.test.tokens_available,
        }


def architecture_for(
    plan: ExperimentPlan, vocabulary: SubwordVocabulary
) -> TransformerConfig | DecoderConfig:
    """The architecture geometry this plan names, bound to a trained vocabulary's identifier count.

    The vocabulary decides ``vocab_size`` rather than the plan, in both families: a model wider than the
    tokenizer is a shape error, and a tokenizer trained to a different target size than the model expects is
    the same error in the other direction.
    """
    if plan.model_family == "decoder":
        return DecoderConfig(
            vocab_size=vocabulary.size,
            context_length=plan.context_length,
            d_model=plan.d_model,
            num_layers=plan.num_layers,
            num_heads=plan.num_heads,
            feedforward_dim=plan.feedforward_dim,
            rope_theta=plan.rope_theta,
        )
    return TransformerConfig(
        vocab_size=vocabulary.size,
        context_length=plan.context_length,
        d_model=plan.d_model,
        num_layers=plan.num_layers,
        num_heads=plan.num_heads,
        feedforward_dim=plan.feedforward_dim,
    )


class _CharacterDecoder:
    """The character vocabulary, expressed in the shape the shared metric arithmetic consumes.

    ``document_end_id`` is the reserved unknown identifier: like a document separator it stands for no
    text, so an unknown character is scored (the model must predict *something*) but adds nothing to the
    character denominator. The count of such positions is reported by the record, never hidden.
    """

    def __init__(self, vocabulary: CharVocabulary) -> None:
        self._vocabulary = vocabulary
        self.document_end_id = UNK_ID

    def decode(self, identifiers: Any) -> str:
        return self._vocabulary.decode(list(identifiers))


def _resolve_manifest(
    plan: ExperimentPlan, work: Path
) -> tuple[CorpusManifest, str, CorpusSnapshot | None]:
    """Decide which corpus this run trains on, and keep its bytes for the next one.

    Three cases, in the order a reader would ask about them:

    * a **declared snapshot** — the run trains on exactly those bytes, and the working tree is not read;
    * a **snapshot the directory already holds** — the run that wrote it pinned this corpus, so a re-run
      trains on the measured bytes instead of whatever the documentation has since become;
    * **no snapshot** — prepare from the tracked files under the manifest the directory pins, then write a
      snapshot so this run's input outlives the next documentation edit.

    A pinned manifest and a loaded snapshot must name the same corpus; if they do not, that is drift and the
    run stops.

    Raises:
        CorpusDriftError: the working tree no longer matches the pinned manifest, or the directory's manifest
            pins a different corpus than the snapshot it was given.
    """
    snapshot_path = plan.snapshot_path()

    def pinned_record(name: str) -> CorpusManifest | None:
        # Named after the corpus that is actually pinned. A run of an external corpus must not leave a
        # record called chakaso-repo-text-v1-manifest.json behind it just because the plan's default spec
        # carries that name.
        candidate = work / f"{name}-manifest.json"
        if not candidate.is_file():
            return None
        return CorpusManifest.from_document(json.loads(candidate.read_text(encoding="utf-8")))

    def save_record(manifest: CorpusManifest) -> None:
        save_manifest(manifest, work / f"{manifest.name}-manifest.json")

    if snapshot_path.is_file():
        snapshot = read_snapshot(snapshot_path)
        origin = (
            "declared snapshot"
            if plan.corpus_snapshot is not None
            else "snapshot from this work dir"
        )
        declared = pinned_record(snapshot.manifest.name)
        if declared is None:
            save_record(snapshot.manifest)
        elif declared.fingerprint != snapshot.fingerprint:
            message = (
                f"the corpus snapshot at {str(snapshot_path)!r} pins fingerprint "
                f"{snapshot.fingerprint!r} but the directory's own record pins "
                f"{declared.fingerprint!r}; a run must not be pointed at one corpus while its directory "
                "records another"
            )
            raise CorpusDriftError(message)
        return snapshot.manifest, origin, snapshot

    corpus = plan.corpus
    manifest = build_manifest(
        plan.root,
        name=corpus.name,
        version=corpus.version,
        licence=corpus.licence,
        source=corpus.source,
        include=corpus.include,
        extensions=corpus.extensions,
        exact_paths=corpus.exact_paths,
    )
    declared = pinned_record(corpus.name)
    if declared is not None and declared.fingerprint != manifest.fingerprint:
        # The recorded manifest wins: an experiment's corpus is fixed by what it pinned, and a
        # documentation edit must not move the data under a saved run. verify_corpus still checks the
        # files against those digests.
        manifest = verify_corpus(declared, plan.root)
    elif declared is None:
        save_record(manifest)

    # Writing the snapshot is what makes this run reproducible later: the manifest pins the digests, and
    # these are the bytes that produced them.
    write_snapshot(manifest, snapshot_path, acquisition={"method": "chakaso-working-tree"})
    return manifest, "working tree", read_snapshot(snapshot_path)


def _tokenizer_identity(
    manifest: CorpusManifest, split: UnitSplit, *, target_size: int, min_frequency: int
) -> str:
    """The digest a fitted tokenizer is cached under: everything its rule sequence can depend on.

    A byte-pair vocabulary is a pure function of the training text, the target size and the frequency floor,
    and the training text is a pure function of the pinned corpus and the split. Naming all four means a
    reused file cannot be a tokenizer fitted on different data — which is the one mistake that would make two
    runs in one directory incomparable while both reports looked self-consistent.
    """
    payload = join_parts(
        manifest.fingerprint,
        split.fingerprint,
        str(target_size),
        str(min_frequency),
    )
    return sha256_hex(payload)[:16]


def _fit_or_reuse_vocabulary(
    plan: ExperimentPlan, work: Path, manifest: CorpusManifest, split: UnitSplit
) -> tuple[SubwordVocabulary, bool]:
    """Fit the subword vocabulary on the training side, or reuse the one already fitted from these inputs.

    Returns:
        The vocabulary and whether it came off disk.

    Raises:
        CorpusError: the cached file exists but is not the vocabulary its name claims.
    """
    identity = _tokenizer_identity(
        manifest,
        split,
        target_size=plan.vocab_size,
        min_frequency=plan.tokenizer_min_frequency,
    )
    path = work / f"{manifest.name}-tokenizer-{identity}.json"
    if path.is_file():
        # SubwordVocabulary.load re-checks the file's own checksum and merge-table invariants, so a cached
        # tokenizer that was edited or truncated is refused here rather than trusted because of its name.
        stored = SubwordVocabulary.load(path)
        if stored.size != plan.vocab_size:
            message = (
                f"cached tokenizer {str(path)!r} holds {stored.size} identifiers, not the "
                f"{plan.vocab_size} this plan declares"
            )
            raise CorpusError(message)
        if stored.min_frequency != plan.tokenizer_min_frequency:
            message = (
                f"cached tokenizer {str(path)!r} was fitted at min_frequency "
                f"{stored.min_frequency}, this plan declares {plan.tokenizer_min_frequency}"
            )
            raise CorpusError(message)
        return stored, True
    fitted = train_subword_vocabulary(
        (unit.text for unit in split.train),
        target_size=plan.vocab_size,
        min_frequency=plan.tokenizer_min_frequency,
    )
    fitted.save(path)
    return fitted, False


def prepare_corpus(plan: ExperimentPlan) -> PreparedCorpus:
    """Pin the corpus, split it once by document, and train both tokenizations on the training side only.

    Raises:
        CorpusDriftError: a manifest already recorded for this experiment pins different content than the
            tracked files now hold, or the corpus this run was pointed at is not the one its directory
            records.
        CorpusError: the plan declares an expected corpus fingerprint and the corpus it resolved is not that
            one — a study must not train on *a* pinned corpus once it has named the one it means.
    """
    started = time.perf_counter()
    work = plan.work_dir
    work.mkdir(parents=True, exist_ok=True)

    manifest, origin, snapshot = _resolve_manifest(plan, work)
    if (
        plan.expected_corpus_fingerprint is not None
        and manifest.fingerprint != plan.expected_corpus_fingerprint
    ):
        message = (
            f"this run requires corpus {plan.expected_corpus_fingerprint!r} and resolved "
            f"{manifest.fingerprint!r} from {origin}; the protocol names a corpus, so the run stops here "
            "rather than training on a different one"
        )
        raise CorpusError(message)

    split = split_text_units(
        manifest.text_units(),
        validation_fraction=_VALIDATION_FRACTION,
        test_fraction=_TEST_FRACTION,
        seed=plan.effective_split_seed,
    )
    vocabulary, tokenizer_reused = _fit_or_reuse_vocabulary(plan, work, manifest, split)
    char_vocabulary = CharVocabulary(tuple(unit.text for unit in split.train))

    # One window per character position of every side: on the repository's own 780 KB that is affordable,
    # on a 14.5M-character corpus it is not, and the machine running the experiment has the last word. A
    # skipped baseline is recorded as skipped rather than quietly omitted.
    char_windows: dict[str, tuple[Any, Any]] = {}
    if plan.character_baseline:
        char_windows = {
            side: _character_batches(split_side, char_vocabulary, plan.character_context)
            for side, split_side in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
        }

    return PreparedCorpus(
        manifest=manifest,
        split=split,
        vocabulary=vocabulary,
        char_vocabulary=char_vocabulary,
        train=pack_units(split.train, vocabulary, context_length=plan.context_length),
        validation=pack_units(split.validation, vocabulary, context_length=plan.context_length),
        test=pack_units(split.test, vocabulary, context_length=plan.context_length),
        char_windows=char_windows,
        timings={"prepare_seconds": round(time.perf_counter() - started, 2)},
        corpus_origin=origin,
        snapshot=snapshot,
        tokenizer_reused=tokenizer_reused,
    )


def _character_batches(
    units: tuple[Any, ...], vocabulary: CharVocabulary, context_size: int
) -> tuple[np.ndarray, np.ndarray]:
    text = "".join(unit.text for unit in units)
    examples = character_windows(text, vocabulary, context_size)
    return batches_from_examples(examples, context_size)


def _metrics(
    nats: list[float], targets: list[int], predictions: list[int], decoder: Any
) -> ModelMetrics:
    return metrics_from_scores(nats, targets, predictions, decoder)


def _character_scores(
    params: Any, model: CharCausalMLP, contexts: np.ndarray, targets: np.ndarray
) -> tuple[list[float], list[int], list[int]]:
    probabilities = model.probabilities(params, contexts)
    nats: list[float] = []
    target_list: list[int] = []
    predictions: list[int] = []
    for index in range(int(contexts.shape[0])):
        row = probabilities[index]
        target = int(targets[index])
        nats.append(float(-log(max(float(row[target]), 1e-12))))
        target_list.append(target)
        predictions.append(int(np.argmax(row)))
    return nats, target_list, predictions


def train_character_baseline(plan: ExperimentPlan, prepared: PreparedCorpus) -> dict[str, Any]:
    """Train the character model over the training documents and score it on the test documents.

    Selection reads the validation windows; the reported numbers come from the test windows at the
    validation-selected step, so the same early-stopping discipline applies to both models.
    """
    split = prepared.split
    config = TrainingConfig(
        context_size=plan.character_context,
        embed_dim=plan.character_embed_dim,
        hidden_dim=plan.character_hidden_dim,
        learning_rate=plan.character_learning_rate,
        seed=plan.seed,
        steps=plan.character_steps,
        eval_every=plan.character_evaluation_interval(),
    )
    text_split = Split(
        train_text="".join(unit.text for unit in split.train),
        validation_text="".join(unit.text for unit in split.validation),
        train_units=tuple(unit.text for unit in split.train),
        validation_units=tuple(unit.text for unit in split.validation),
    )

    started = time.perf_counter()
    result = run_minibatch_training(
        text_split,
        config,
        batch_size=plan.character_batch_windows,
        max_steps=plan.character_steps,
        eval_every=plan.character_evaluation_interval(),
        seed=plan.seed,
    )
    wall = time.perf_counter() - started

    curve = [(point.step, point.validation_loss) for point in result.curve]
    best_step, best_validation = min(curve, key=lambda point: point[1])

    # The curve's final point is not the reported model: E2601 and E2602 measured that a character model of
    # this size overfits well before its budget ends, so the selected step is re-run and scored.
    selected = run_minibatch_training(
        text_split,
        config,
        batch_size=plan.character_batch_windows,
        max_steps=max(best_step, 1),
        eval_every=plan.character_evaluation_interval(),
        seed=plan.seed,
    )
    model = CharCausalMLP(
        vocab_size=prepared.char_vocabulary.size,
        context_size=plan.character_context,
        embed_dim=plan.character_embed_dim,
        hidden_dim=plan.character_hidden_dim,
    )
    test_contexts, test_targets = prepared.char_windows["test"]
    nats, targets, predictions = _character_scores(
        selected.final_params, model, test_contexts, test_targets
    )
    metrics = _metrics(nats, targets, predictions, _CharacterDecoder(prepared.char_vocabulary))

    _, train_targets = prepared.char_windows["train"]
    uniform = uniform_next_token_loss(prepared.char_vocabulary.size)
    unigram = UnigramBaseline.fit(list(train_targets), prepared.char_vocabulary.size).cross_entropy(
        list(test_targets)
    )

    return {
        "architecture": {
            "kind": "character_causal_mlp",
            "context_characters": plan.character_context,
            "embed_dim": plan.character_embed_dim,
            "hidden_dim": plan.character_hidden_dim,
            "parameter_count": result.parameter_count,
            "vocabulary_size": prepared.char_vocabulary.size,
            "vocabulary_fingerprint": prepared.char_vocabulary.fingerprint,
            "gradients": "hand-derived, finite-difference verified (ADR-0030)",
        },
        "budget": {
            "batch_windows": plan.character_batch_windows,
            "max_steps": plan.character_steps,
            "eval_every": plan.character_evaluation_interval(),
            "selected_step": best_step,
            "selected_validation_loss_nats": round(best_validation, 6),
            "wall_clock_s": round(wall, 2),
            "train_windows": int(prepared.char_windows["train"][0].shape[0]),
            "validation_windows": int(prepared.char_windows["validation"][0].shape[0]),
            "test_windows": int(prepared.char_windows["test"][0].shape[0]),
            "unknown_character_targets_scored": metrics.separator_targets,
        },
        "test": metrics.as_dict(),
        "baselines": {"uniform_nats": round(uniform, 6), "unigram_nats": round(unigram, 6)},
        "curve": [{"step": step, "validation_loss": round(value, 6)} for step, value in curve],
    }


def baseline_metrics(prepared: PreparedCorpus) -> dict[str, Any]:
    """Score uniform and unigram predictors over the same test tokens the Transformer is scored on."""
    vocabulary = prepared.vocabulary
    test_targets = [token for example in prepared.test.examples for token in example.targets]
    train_targets = [token for example in prepared.train.examples for token in example.targets]
    if not test_targets:
        message = "the test side produced no scored tokens, so baselines cannot be computed"
        raise ValueError(message)

    uniform_value = log(vocabulary.size)
    uniform = _metrics(
        [uniform_value] * len(test_targets), test_targets, [-1] * len(test_targets), vocabulary
    )
    baseline = UnigramBaseline.fit(train_targets, vocabulary.size)
    unigram_nats = [-baseline.log_probability(token) for token in test_targets]
    unigram = _metrics(
        unigram_nats, test_targets, [baseline.most_frequent] * len(test_targets), vocabulary
    )
    return {
        "uniform": {**uniform.as_dict(), "vocabulary_size": vocabulary.size},
        "unigram": {**unigram.as_dict(), "smoothing_alpha": baseline.alpha},
    }


def score_transformer(
    model: TrainedModel, batch: PackedBatch, vocabulary: SubwordVocabulary
) -> ModelMetrics:
    """Score a trained model on one packed side of the split, through the shared arithmetic."""
    nats, targets, predictions = score_model_on_examples(model, batch.examples)
    return _metrics(nats, targets, predictions, vocabulary)


def _training_config(
    plan: ExperimentPlan, architecture: TransformerConfig | DecoderConfig
) -> TransformerTrainingConfig:
    """The optimization settings this plan names, bound to its architecture.

    One function because the resume check has to compare against exactly the configuration the run will
    use: a second, slightly different construction of the same settings is how a verified resume becomes an
    unverified one.
    """
    return TransformerTrainingConfig(
        batch_size=plan.batch_size,
        max_steps=plan.max_steps,
        learning_rate=plan.learning_rate,
        weight_decay=plan.weight_decay,
        warmup_steps=plan.warmup_steps,
        gradient_clip_norm=plan.gradient_clip_norm,
        seed=plan.seed,
        eval_every=plan.eval_every,
        patience=plan.patience,
        min_delta=plan.min_delta,
        architecture=architecture,
        schedule=plan.schedule,
        decay_to=plan.decay_to,
        grad_accum_steps=plan.grad_accum_steps,
        checkpoint_every=plan.checkpoint_every,
    )


def run_experiment(plan: ExperimentPlan, *, verbose: bool = False) -> dict[str, Any]:
    """Run E2605 end to end and return the measured report.

    Every value in the returned mapping was produced by this call: no field is a default, a target or an
    expectation. A missing dependency fails here with a named error rather than yielding a partial report.
    """
    work = plan.work_dir
    work.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and verbose:
        print("device: cpu", flush=True)  # noqa: T201 - operator-visible progress
    prepared = prepare_corpus(plan)
    architecture = architecture_for(plan, prepared.vocabulary)

    character = (
        train_character_baseline(plan, prepared)
        if plan.character_baseline
        else {
            "skipped": True,
            "reason": (
                "the character baseline materializes one window per character position; on a corpus of "
                f"{prepared.manifest.char_count:,} characters that exceeds the memory this project's "
                "machine has, and the run states that instead of silently dropping the comparison"
            ),
        }
    )

    def on_step(record: Any) -> None:
        if verbose:
            print(  # noqa: T201 - operator-visible progress
                f"step {record.step}: train {record.train_loss:.4f} "
                f"validation {record.validation_loss} grad-norm {record.gradient_norm:.3f}",
                flush=True,
            )

    started = time.perf_counter()
    resume: dict[str, Any] = {}
    budget_extended_from: int | None = None
    if plan.resume_from is not None:
        point = load_artifact(plan.resume_from)
        verified = verify_resume_compatibility(
            point,
            config=_training_config(plan, architecture),
            corpus_fingerprint=prepared.manifest.fingerprint,
            vocabulary=prepared.vocabulary,
        )
        budget_extended_from = verified.budget_extended_from
        resume = {
            "resume_state": verified.state,
            "resume_optimizer": verified.optimizer_state,
            "start_step": verified.start_step,
        }
    outcome = train_transformer(
        train_examples=prepared.train.examples,
        validation_examples=prepared.validation.examples,
        vocabulary=prepared.vocabulary,
        **resume,
        config=_training_config(plan, architecture),
        corpus_fingerprint=prepared.manifest.fingerprint,
        on_step=on_step,
        checkpoint_path=work / f"{plan.experiment_id}-{plan.model_family}-resumable.pt",
    )
    wall = time.perf_counter() - started

    artifact = outcome.artifact()
    model = artifact.to_model()
    test_metrics = score_transformer(model, prepared.test, prepared.vocabulary)
    validation_metrics = score_transformer(model, prepared.validation, prepared.vocabulary)
    artifact_path = work / checkpoint_filename(plan.experiment_id, plan.model_family)
    save_artifact(artifact, artifact_path, allow_overwrite=plan.overwrite_artifacts)

    # Generation goes through the boundary, not the adapter class: the experiment is a consumer of the
    # public interface, exactly as an application would be, so a break in load_model or in the contract
    # shows up here as well as in the adapter's own tests.
    from chakaso.models import GenerationParams, Message, Role, load_model

    inference = load_model(artifact_path)
    params = GenerationParams(max_new_tokens=plan.samples_per_prompt, temperature=0.0)
    samples = []
    for prompt in _PROMPTS:
        result = inference.generate([Message(role=Role.USER, content=prompt)], params)
        samples.append(
            {
                "prompt": prompt,
                "text": result.text,
                "finish_reason": result.finish_reason.value,
                "model_id": result.model_id,
            }
        )

    targets = [token for example in prepared.test.examples for token in example.targets]
    return {
        "experiment": plan.experiment_id,
        "corpus": {
            "name": prepared.manifest.name,
            "version": prepared.manifest.version,
            "licence": prepared.manifest.licence,
            "source": prepared.manifest.source,
            "include": list(prepared.manifest.include),
            "extensions": list(prepared.manifest.extensions),
            "exact_paths": list(prepared.manifest.exact_paths),
            "fingerprint": prepared.manifest.fingerprint,
            "prepared_from": prepared.corpus_origin,
            "snapshot": (
                None
                if prepared.snapshot is None
                else {
                    "path": plan.snapshot_path().name,
                    "checksum": prepared.snapshot.checksum,
                    "acquisition": dict(prepared.snapshot.acquisition),
                }
            ),
            "documents": len(prepared.manifest.documents),
            "characters": prepared.manifest.char_count,
            "duplicate_groups": [list(group) for group in prepared.manifest.duplicate_paths],
            "split_seed": plan.effective_split_seed,
            "model_seed": plan.seed,
            "split_fingerprint": prepared.split.fingerprint,
            "documents_per_side": {
                "train": len(prepared.split.train),
                "validation": len(prepared.split.validation),
                "test": len(prepared.split.test),
                "dropped_duplicates": len(prepared.split.dropped_duplicates),
            },
            "identifiers_per_side": {
                side: [unit.identifier for unit in getattr(prepared.split, side)]
                for side in ("train", "validation", "test")
            },
        },
        "tokenizer": {
            "kind": "byte-pair subword trained by chakaso.training.subword",
            "size": prepared.vocabulary.size,
            "merges": prepared.vocabulary.merge_count,
            "fingerprint": prepared.vocabulary.fingerprint,
            "min_frequency": prepared.vocabulary.min_frequency,
            "reused_from_cache": prepared.tokenizer_reused,
            "tokens_per_side": prepared.tokens,
            "test_characters_decoded": len(prepared.vocabulary.decode(targets)),
            "characters_per_token_test": round(
                len(prepared.vocabulary.decode(targets)) / max(1, len(targets)), 4
            ),
        },
        "character_baseline": character,
        "model_family": plan.model_family,
        "transformer": {
            "architecture": architecture.as_dict(),
            "parameter_count": outcome.parameter_count,
            "parameters_by_configuration": architecture.parameter_count,
            "budget": {
                "batch_size": plan.batch_size,
                "max_steps": plan.max_steps,
                "eval_every": plan.eval_every,
                "patience": plan.patience,
                "min_delta": plan.min_delta,
                "learning_rate": plan.learning_rate,
                "weight_decay": plan.weight_decay,
                "warmup_steps": plan.warmup_steps,
                "gradient_clip_norm": plan.gradient_clip_norm,
                "seed": plan.seed,
                "best_step": outcome.best_step,
                "stopped_early": outcome.stopped_early,
                "completion": outcome.completion,
                "resumed_from_step": resume.get("start_step"),
                "budget_extended_from": budget_extended_from,
                "schedule": plan.schedule,
                "decay_to": plan.decay_to,
                "grad_accum_steps": plan.grad_accum_steps,
                "effective_batch_size": outcome.config.effective_batch_size,
                "positions_per_second": round(outcome.positions_per_second, 1),
                "records": len(outcome.records),
                "wall_clock_s": round(wall, 2),
                "steps_per_epoch_estimate": -(-len(prepared.train.examples) // plan.batch_size),
            },
            "validation": validation_metrics.as_dict(),
            "test": test_metrics.as_dict(),
            "curve": [
                {
                    "step": record.step,
                    "train_loss": round(record.train_loss, 6),
                    "validation_loss": record.validation_loss,
                    "gradient_norm": round(record.gradient_norm, 4),
                    "learning_rate": record.learning_rate,
                }
                for record in outcome.records
            ],
            "artifact": {
                "path": artifact_path.name,
                "checksum": artifact.checksum(),
                "schema": "chakaso-transformer-checkpoint/1",
            },
        },
        "baselines_subword": baseline_metrics(prepared),
        "samples": samples,
        "timings": prepared.timings | {"transformer_wall_clock_s": round(wall, 2)},
        "environment": artifact.environment,
        "plan": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(plan).items()
            if isinstance(value, int | float | str | Path)
        },
    }


def write_report(
    report: dict[str, Any], destination: Path, *, allow_overwrite: bool = False
) -> str:
    """Write the machine-readable report deterministically and return its digest.

    The digest is what the record cites, so a reader who regenerates the report can compare hashes without
    reading every number in it. That only works if the file's bytes are the text that was hashed: the
    payload is indented, and a write in default text mode would translate its newlines on Windows and
    hand back a digest nobody else can reproduce. ``allow_overwrite`` exists because a re-run pointed at a
    recorded run's report path would otherwise replace the evidence the record names.

    Raises:
        ValueError: a file already occupies ``destination`` and ``allow_overwrite`` was not passed.
    """
    target = Path(destination)
    if target.exists() and not allow_overwrite:
        message = (
            f"{str(target)!r} already holds a report; pass allow_overwrite=True to replace it, or choose "
            "a distinct path so the earlier run stays recoverable"
        )
        raise ValueError(message)
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    write_text_atomically(payload, target)
    return sha256_hex(payload)


def summarise(report: dict[str, Any]) -> str:
    """A one-line digest of the comparison, for the record and for an operator's terminal."""
    transformer = report["transformer"]["test"]
    character = report["character_baseline"]["test"]
    uniform = report["baselines_subword"]["uniform"]
    unigram = report["baselines_subword"]["unigram"]
    budget = report["transformer"]["budget"]
    return " | ".join(
        (
            f"transformer {report['transformer']['parameter_count']:,} params, "
            f"test bits/char {transformer['bits_per_character']:.4f}, "
            f"perplexity {transformer['perplexity']:.2f} at step {budget['best_step']} "
            f"(early stopped: {budget['stopped_early']})",
            f"character test bits/char {character['bits_per_character']:.4f}",
            f"unigram {unigram['bits_per_character']:.4f}",
            f"uniform {uniform['bits_per_character']:.4f}",
        )
    )
