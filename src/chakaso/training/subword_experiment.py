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
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import e, fsum, inf, log, log2
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
    groups_per_side,
    split_text_units,
    works_on_multiple_sides,
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
from chakaso.training.mixture import ShareSelection, identifiers_by_register, select_by_share
from chakaso.training.model import CharCausalMLP, batches_from_examples
from chakaso.training.model_metrics import ModelMetrics, metrics_from_scores
from chakaso.training.nn_checkpoint import (
    TrainedModel,
    checkpoint_filename,
    load_artifact,
    save_artifact,
    schema_of,
)
from chakaso.training.nn_evaluation import document_losses, score_model_on_examples
from chakaso.training.nn_training import (
    TransformerTrainingConfig,
    optimizer_settings,
    rung_artifact,
    train_transformer,
    verify_resume_compatibility,
)
from chakaso.training.packing import PackedBatch, pack_units, select_units_by_token_cap
from chakaso.training.subword import (
    DEFAULT_PRE_TOKENIZER,
    SubwordVocabulary,
    pre_tokenize,
    train_subword_vocabulary,
)
from chakaso.training.tokenizer import UNK_ID, CharVocabulary
from chakaso.training.transformer import TransformerConfig

__all__ = [
    "CHARACTER_CONTEXT",
    "CORPUS_NAME",
    "EXPERIMENT_ID",
    "REPO_TEXT_CORPUS",
    "CorpusSpec",
    "EvaluationSides",
    "ExperimentPlan",
    "PreparedCorpus",
    "evaluation_sides",
    "pin_and_split",
    "prepare_corpus",
    "protocol_digest",
    "run_experiment",
    "score_artifacts",
    "summarise",
    "work_losses",
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
_PROMPT_SOURCES = frozenset({"repository", "held_out"})
#: Characters taken from the start of a held-out document to form a prompt.
_HELD_OUT_PROMPT_CHARACTERS = 64
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
    #: The named pre-tokenizer the vocabulary is fitted under (ADR-0048); ``None`` is the default rule every earlier
    #: record used, and is left out of the report so a default plan's protocol digest is what it always was.
    tokenizer_pre_tokenizer: str | None = None
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
    #: Where the recorded generation prompts come from. ``repository`` is the set E2605 to E2606 sampled with;
    #: ``held_out`` takes prompts from documents on this corpus's own test side, which is what a comparison
    #: on external text needs — a prompt quoting Chakaso's section titles measures memorization of a corpus
    #: the model never saw, not behaviour on the corpus it trained on.
    sample_prompts: str = "repository"
    #: A snapshot of the corpus, read through :func:`chakaso.training.manifest.read_snapshot`, which
    #: re-checks every document digest. Given, the run trains on those bytes and never touches the tracked
    #: files; absent, the run prepares from the tree, writes a snapshot beside its manifest, and a later run
    #: in the same directory picks it up. Either way the corpus the report names is the corpus that ran.
    corpus_snapshot: Path | None = None
    #: The corpus fingerprint this run is required to use. Declared by a reproduction or a seed study, so
    #: that loading *a* snapshot is never the same as loading *the* snapshot.
    expected_corpus_fingerprint: str | None = None
    #: The vocabulary fingerprint this run is required to train with. A comparison that holds the tokenizer
    #: fixed names it here, so a cache miss that silently fits a *different* vocabulary — which is how
    #: E2613's first launch went wrong — stops the run before the first training step instead of after the
    #: last. The check applies to a cached vocabulary and to a freshly fitted one alike.
    expected_tokenizer_fingerprint: str | None = None
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
    #: Whether attention is confined to the document each packed position came from. Off by default so
    #: every measured protocol remains reproducible by name; requires the decoder family.
    document_masking: bool = False
    #: Whether decoupled weight decay is withheld from the one-dimensional parameters (norm scales, biases).
    #: Off by default, which is what every committed record ran; K-030 item 3 filed the question and the
    #: published small-model recipes that use this optimizer exempt them, so the choice is an arm rather than
    #: a default change.
    weight_decay_excludes_vectors: bool = False
    #: Optimizer steps at which the run additionally writes a **usable** model artifact beside its own, named
    #: ``<experiment>-<family>-step<step>.pt``. Empty — every committed record — means one usable artifact at
    #: the end. A rung is only interpretable beside the held-out loss the loop records for it, so each step
    #: must be an evaluation point. See ADR-0049.
    curve_artifact_steps: tuple[int, ...] = ()
    #: Whether every chunk of a work lands on one side of the split. Off by default because it changes the
    #: split fingerprint, and a measured protocol's split must not move under its record; a corpus cut from
    #: books declares it on, or its held-out score is within-work rather than held-out by work.
    whole_work_splits: bool = False
    #: Group prefixes a training document's work must match to be packed for training. ``None`` trains on
    #: the whole training side. Validation, test and the tokenizer are untouched, so two arms that differ
    #: only in this field see the same vocabulary and are scored on the same held-out works; that is what
    #: makes an equal-token data-mixture comparison a comparison.
    training_group_prefixes: tuple[str, ...] | None = None
    #: Stop packing the training side after this many identifiers. ``None`` packs all of it. A corpus can
    #: be much larger than the positions a run trains on, and packing all of it costs about 46 bytes per
    #: token; the cap is what lets one corpus serve several budgeted runs on this machine.
    max_training_tokens: int | None = None
    #: Shuffle the selected training works before packing, with the split seed. Without it a capped corpus
    #: is a prefix in catalogue order, which groups registers together and makes a "mixture" arm whatever
    #: the alphabet produced; with it any prefix is a sample of the whole. Off by default so a recorded
    #: protocol's stream order does not move under its record.
    shuffle_training_units: bool = False
    #: Compose the training stream by share instead of by cap (H9). The works whose group starts with
    #: ``share_group_prefix`` supply ``share_of_tokens`` of ``max_training_tokens`` identifiers and every other
    #: training work supplies the rest, each drawn as a prefix of its own seeded shuffle so a larger share
    #: contains a smaller one. Both fields are set together or not at all, and the mode needs a token budget and
    #: shuffled units; it replaces ``training_group_prefixes``, which it would contradict. Left unset, every
    #: recorded protocol behaves as it did and its protocol digest does not move.
    share_group_prefix: str | None = None
    share_of_tokens: float | None = None
    #: Group prefixes to score separately on the test side, each packed on its own. A mixture experiment
    #: needs the technical held-out loss and the prose held-out loss, not their weighted average; a single
    #: overall number would be dominated by whichever register has more held-out text.
    test_partitions: tuple[str, ...] | None = None
    #: Fit the tokenizer on a seeded sample of at most this many characters of the training side instead of
    #: all of it. Byte-pair fitting holds every distinct word shape in memory and merges across them; on a
    #: 149M-character corpus that exceeded this machine's free memory. The sample is a seeded shuffle of the
    #: training works, so it spans registers rather than the catalogue's first alphabetical block, and the
    #: cap is part of the tokenizer's identity so a cached vocabulary cannot be reused under a different one.
    tokenizer_sample_characters: int | None = None
    #: Pack at most this many identifiers for validation and for test, sampled over a seeded shuffle of the
    #: side's documents. A 149M-character corpus makes a 15 % test side ~10M tokens: too large to score at
    #: every evaluation and too large to hold beside the training pack. The sample is drawn from documents
    #: whose works are still wholly held out, and the report states how many were packed.
    max_validation_tokens: int | None = None
    max_test_tokens: int | None = None
    #: Whether the character-model baseline runs at all. Its windows are materialized per character, so on a
    #: corpus of tens of millions of characters it costs more memory than this project's machine has; a run
    #: that skips it says so in its report rather than pretending the number was measured.
    character_baseline: bool = True
    #: Where a fitted tokenizer is cached. Defaults to the run's own work directory; a seed study points
    #: every seed at one directory, because the vocabulary depends on the corpus and the split and the
    #: study fixes both, so retraining it per seed would be paying the same cost for the same answer.
    tokenizer_cache_dir: Path | None = None
    #: A vocabulary file to train with instead of fitting or caching one. It is loaded through
    #: :meth:`SubwordVocabulary.load`, which re-checks its own checksum and merge-table invariants, and it
    #: must match ``vocab_size``, ``tokenizer_min_frequency`` and ``expected_tokenizer_fingerprint`` like a
    #: cached one. This is how a record whose vocabulary current fitting code can no longer reproduce
    #: (K-024) stays re-trainable: the file is committed beside the record, and the cache is not load-bearing.
    tokenizer_file: Path | None = None
    #: A vocabulary file used only to *choose* the held-out documents: the validation and test caps in identifiers
    #: are counted with it instead of with the vocabulary the run trains with. A more economical tokenizer reaches an
    #: identifier cap sooner, so two tokenizers capped by their own counts are scored on different text and their
    #: bits per character are not comparable (H17). ``None`` keeps the cap counted with the run's own vocabulary,
    #: which is what every earlier record did.
    evaluation_selection_tokenizer_file: Path | None = None
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
        if self.tokenizer_pre_tokenizer is not None:
            pre_tokenize("", self.tokenizer_pre_tokenizer)  # a ValueError naming the known rules
        if isinstance(self.grad_accum_steps, bool) or self.grad_accum_steps < 1:
            message = f"grad_accum_steps must be >= 1, got {self.grad_accum_steps!r}"
            raise ValueError(message)
        if self.sample_prompts not in _PROMPT_SOURCES:
            message = (
                f"sample_prompts {self.sample_prompts!r} is not one of {sorted(_PROMPT_SOURCES)}"
            )
            raise ValueError(message)
        if isinstance(self.checkpoint_every, bool) or self.checkpoint_every < 0:
            message = f"checkpoint_every must be >= 0, got {self.checkpoint_every!r}"
            raise ValueError(message)
        if not isinstance(self.document_masking, bool):
            message = f"document_masking must be a boolean, got {self.document_masking!r}"
            raise ValueError(message)
        if not isinstance(self.whole_work_splits, bool):
            message = f"whole_work_splits must be a boolean, got {self.whole_work_splits!r}"
            raise ValueError(message)
        if self.training_group_prefixes is not None:
            prefixes = self.training_group_prefixes
            if not isinstance(prefixes, tuple) or not prefixes:
                message = (
                    "training_group_prefixes must be None or a non-empty tuple of strings, got "
                    f"{prefixes!r}"
                )
                raise ValueError(message)
            if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
                message = (
                    f"training_group_prefixes holds an empty or non-string entry: {prefixes!r}"
                )
                raise ValueError(message)
        if not isinstance(self.shuffle_training_units, bool):
            message = (
                f"shuffle_training_units must be a boolean, got {self.shuffle_training_units!r}"
            )
            raise ValueError(message)
        self._validate_share()
        if not isinstance(self.weight_decay_excludes_vectors, bool):
            message = (
                "weight_decay_excludes_vectors must be a boolean, got "
                f"{self.weight_decay_excludes_vectors!r}"
            )
            raise ValueError(message)
        rungs = self.curve_artifact_steps
        if isinstance(rungs, str) or not isinstance(rungs, tuple):
            message = f"curve_artifact_steps must be a tuple of ints, got {rungs!r}"
            raise ValueError(message)
        if any(isinstance(step, bool) or not isinstance(step, int) or step < 1 for step in rungs):
            message = f"curve_artifact_steps holds a non-positive or non-integer step: {rungs!r}"
            raise ValueError(message)
        for field_name in ("max_validation_tokens", "max_test_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                message = f"{field_name} must be None or an integer >= 1, got {value!r}"
                raise ValueError(message)
        if self.tokenizer_sample_characters is not None and (
            isinstance(self.tokenizer_sample_characters, bool)
            or not isinstance(self.tokenizer_sample_characters, int)
            or self.tokenizer_sample_characters < 1
        ):
            message = (
                "tokenizer_sample_characters must be None or an integer >= 1, got "
                f"{self.tokenizer_sample_characters!r}"
            )
            raise ValueError(message)
        if self.test_partitions is not None:
            partitions = self.test_partitions
            if not isinstance(partitions, tuple) or not partitions:
                message = f"test_partitions must be None or a non-empty tuple of strings, got {partitions!r}"
                raise ValueError(message)
            if any(not isinstance(prefix, str) or not prefix for prefix in partitions):
                message = f"test_partitions holds an empty or non-string entry: {partitions!r}"
                raise ValueError(message)
        if self.max_training_tokens is not None and (
            isinstance(self.max_training_tokens, bool)
            or not isinstance(self.max_training_tokens, int)
            or self.max_training_tokens < 1
        ):
            message = (
                "max_training_tokens must be None or an integer >= 1, got "
                f"{self.max_training_tokens!r}"
            )
            raise ValueError(message)
        if self.document_masking and self.model_family != "decoder":
            message = (
                "document_masking is implemented for the decoder family; the transformer family's "
                "encoder path does not expose a per-batch attention mask here"
            )
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
        if (
            self.expected_tokenizer_fingerprint is not None
            and len(self.expected_tokenizer_fingerprint) != 16
        ):
            message = (
                "expected_tokenizer_fingerprint must be the 16 hexadecimal characters a vocabulary "
                f"fingerprint has, got {self.expected_tokenizer_fingerprint!r}"
            )
            raise ValueError(message)

    def _validate_share(self) -> None:
        """Refuse a share declaration that is half-stated or contradicts the rest of the plan."""
        prefix, share = self.share_group_prefix, self.share_of_tokens
        if prefix is None and share is None:
            return
        if prefix is None or share is None:
            message = "share_group_prefix and share_of_tokens are declared together or not at all"
            raise ValueError(message)
        if not isinstance(prefix, str) or not prefix:
            message = f"share_group_prefix must be a non-empty string, got {prefix!r}"
            raise ValueError(message)
        if isinstance(share, bool) or not isinstance(share, int | float) or not 0.0 <= share <= 1.0:
            message = f"share_of_tokens must be a number from 0 to 1, got {share!r}"
            raise ValueError(message)
        if self.max_training_tokens is None:
            message = "a share is a fraction of a token budget; set max_training_tokens"
            raise ValueError(message)
        if not self.shuffle_training_units:
            message = "a share takes each class from the front of its seeded shuffle; set shuffle_training_units"
            raise ValueError(message)
        if self.training_group_prefixes is not None:
            message = (
                "share_group_prefix replaces training_group_prefixes; a plan cannot filter the training "
                "side and compose it by share at once"
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
    #: What a share-composed run packed, so the report states the realised mixture and not the intended one.
    #: ``None`` for every other plan.
    training_composition: ShareSelection | None = None
    #: The vocabulary that chose the held-out documents, when the plan named one; ``None`` otherwise.
    selection_vocabulary: SubwordVocabulary | None = None

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
    manifest: CorpusManifest,
    split: UnitSplit,
    *,
    target_size: int,
    min_frequency: int,
    sample_characters: int | None = None,
    pre_tokenizer: str | None = None,
) -> str:
    """The digest a fitted tokenizer is cached under: everything its rule sequence can depend on.

    A byte-pair vocabulary is a pure function of the training text, the target size and the frequency floor,
    and the training text is a pure function of the pinned corpus and the split. Naming all four means a
    reused file cannot be a tokenizer fitted on different data — which is the one mistake that would make two
    runs in one directory incomparable while both reports looked self-consistent.
    """
    parts = [
        manifest.fingerprint,
        split.fingerprint,
        str(target_size),
        str(min_frequency),
        "all-training-text" if sample_characters is None else f"sample-{sample_characters}",
    ]
    if pre_tokenizer is not None:
        # A tokenizer fitted under another rule is another file. Left out for the default so every existing cache
        # key, and the file a record's reproduction command would look for, stays what it was.
        parts.append(f"pre-tokenizer-{pre_tokenizer}")
    return sha256_hex(join_parts(*parts))[:16]


def _tokenizer_sample(plan: ExperimentPlan, split: UnitSplit) -> tuple[str, ...]:
    """The texts the vocabulary is fitted on: the whole training side, or a seeded sample of it.

    The shuffle matters: training documents arrive in catalogue order, which groups registers together, so
    the first N characters of that order would fit a vocabulary on one register and encode the rest with it.
    """
    cap = plan.tokenizer_sample_characters
    if cap is None:
        return tuple(unit.text for unit in split.train)
    units = list(split.train)
    random.Random(plan.effective_split_seed).shuffle(units)
    chosen: list[str] = []
    total = 0
    for unit in units:
        chosen.append(unit.text)
        total += len(unit.text)
        if total >= cap:
            break
    return tuple(chosen)


def _fit_or_reuse_vocabulary(
    plan: ExperimentPlan, work: Path, manifest: CorpusManifest, split: UnitSplit
) -> tuple[SubwordVocabulary, bool]:
    """Fit the subword vocabulary on the training side, or reuse the one already fitted from these inputs.

    Returns:
        The vocabulary and whether it came off disk.

    Raises:
        CorpusError: the cached file exists but is not the vocabulary its name claims.
    """
    if plan.tokenizer_file is not None:
        supplied = SubwordVocabulary.load(plan.tokenizer_file)
        _check_vocabulary_matches_plan(plan, supplied, str(plan.tokenizer_file))
        _require_expected_tokenizer(
            plan, supplied, source=f"the supplied file {str(plan.tokenizer_file)!r}"
        )
        return supplied, True
    identity = _tokenizer_identity(
        manifest,
        split,
        target_size=plan.vocab_size,
        min_frequency=plan.tokenizer_min_frequency,
        sample_characters=plan.tokenizer_sample_characters,
        pre_tokenizer=plan.tokenizer_pre_tokenizer,
    )
    cache = plan.tokenizer_cache_dir if plan.tokenizer_cache_dir is not None else work
    path = cache / f"{manifest.name}-tokenizer-{identity}.json"
    if path.is_file():
        # SubwordVocabulary.load re-checks the file's own checksum and merge-table invariants, so a cached
        # tokenizer that was edited or truncated is refused here rather than trusted because of its name.
        stored = SubwordVocabulary.load(path)
        _check_vocabulary_matches_plan(plan, stored, str(path))
        _require_expected_tokenizer(plan, stored, source=f"the cached file {str(path)!r}")
        return stored, True
    fitted = train_subword_vocabulary(
        _tokenizer_sample(plan, split),
        target_size=plan.vocab_size,
        min_frequency=plan.tokenizer_min_frequency,
        pre_tokenizer=plan.tokenizer_pre_tokenizer or DEFAULT_PRE_TOKENIZER,
    )
    if fitted.size != plan.vocab_size:
        # The merge search ends early when no pair reaches the frequency floor. A model is then built at the
        # smaller size, so the run would train a different parameter count than its plan and its record
        # declare, and nothing downstream would say so until someone compared totals.
        message = (
            f"fitting stopped at {fitted.size} identifiers, short of the {plan.vocab_size} this plan declares: "
            f"no adjacent pair occurs {plan.tokenizer_min_frequency} times in the training text after "
            f"{len(fitted.merges)} merges. Lower vocab_size or tokenizer_min_frequency, or supply more text; "
            "a run does not train a smaller vocabulary than it declared"
        )
        raise CorpusError(message)
    # Checked before the save, so a vocabulary that is not the one the protocol names never enters the cache
    # where a later run would take it for the real one.
    _require_expected_tokenizer(
        plan, fitted, source="a vocabulary fitted because no cached file matched this split"
    )
    fitted.save(path)
    return fitted, False


def _check_vocabulary_matches_plan(
    plan: ExperimentPlan, vocabulary: SubwordVocabulary, where: str
) -> None:
    """Refuse a stored vocabulary whose size or frequency floor is not the one the plan declares."""
    if vocabulary.size != plan.vocab_size:
        message = (
            f"tokenizer {where!r} holds {vocabulary.size} identifiers, not the "
            f"{plan.vocab_size} this plan declares"
        )
        raise CorpusError(message)
    if vocabulary.min_frequency != plan.tokenizer_min_frequency:
        message = (
            f"tokenizer {where!r} was fitted at min_frequency {vocabulary.min_frequency}, "
            f"this plan declares {plan.tokenizer_min_frequency}"
        )
        raise CorpusError(message)
    declared_rule = plan.tokenizer_pre_tokenizer or DEFAULT_PRE_TOKENIZER
    if vocabulary.pre_tokenizer != declared_rule:
        message = (
            f"tokenizer {where!r} was fitted under the pre-tokenizer {vocabulary.pre_tokenizer!r}, "
            f"this plan declares {declared_rule!r}"
        )
        raise CorpusError(message)


def _require_expected_tokenizer(
    plan: ExperimentPlan, vocabulary: SubwordVocabulary, *, source: str
) -> None:
    """Refuse a vocabulary other than the one the plan declares it must train with.

    Raises:
        CorpusError: the plan declares ``expected_tokenizer_fingerprint`` and ``vocabulary`` differs.
    """
    expected = plan.expected_tokenizer_fingerprint
    if expected is not None and vocabulary.fingerprint != expected:
        message = (
            f"this run requires tokenizer {expected!r} and got {vocabulary.fingerprint!r} from {source}; "
            "the protocol holds the vocabulary fixed, so the run stops here rather than training on another"
        )
        raise CorpusError(message)


def _training_units(
    units: tuple[Any, ...],
    prefixes: tuple[str, ...] | None,
    *,
    shuffle_seed: int | None = None,
) -> tuple[Any, ...]:
    """The training documents this run packs: filtered by group prefix, and shuffled if asked.

    The shuffle is seeded from the split seed so every arm of one study draws the same order, and so a
    token-capped run takes a sample of the whole training side rather than the alphabetically first works.
    """
    selected = (
        units
        if prefixes is None
        else tuple(unit for unit in units if unit.group.startswith(prefixes))
    )
    if not selected:
        message = (
            f"no training document's work starts with any of {prefixes!r}; the run would train on "
            "nothing, so it stops before packing"
        )
        raise ValueError(message)
    if shuffle_seed is None:
        return selected
    shuffled = list(selected)
    random.Random(shuffle_seed).shuffle(shuffled)
    return tuple(shuffled)


def _require_share_classes(units: tuple[Any, ...], prefix: str, share: float) -> None:
    """Refuse, before the tokenizer is fitted, a share whose class has no works to draw from at all."""
    matched = sum(1 for unit in units if unit.group.startswith(prefix))
    if share > 0.0 and matched == 0:
        message = (
            f"no training document's work starts with {prefix!r}; a share of {share} of the budget "
            "cannot be drawn from nothing, so the run stops before fitting a tokenizer"
        )
        raise ValueError(message)
    if share < 1.0 and matched == len(units):
        message = (
            f"every training document's work starts with {prefix!r}; the remaining {1 - share:g} of the "
            "budget has nothing to be drawn from, so the run stops before fitting a tokenizer"
        )
        raise ValueError(message)


def pin_and_split(
    plan: ExperimentPlan,
) -> tuple[CorpusManifest, str, CorpusSnapshot | None, UnitSplit]:
    """Pin the corpus the plan names and split it once by document — nothing else.

    Split out of :func:`prepare_corpus` so a measurement that re-scores existing artifacts against the run's own
    held-out text (:func:`work_losses`) cannot drift from the path that produced that text. The guards are the run's
    guards: a corpus fingerprint the plan names and did not resolve stops the call, and a repository-prompted run on
    a foreign corpus stops too (K-026).

    Raises:
        CorpusDriftError: a recorded manifest pins different content than the tracked files now hold.
        CorpusError: the resolved corpus is not the one the plan names, or the prompts do not belong to the corpus.
    """
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
    if plan.sample_prompts == "repository" and manifest.name != CORPUS_NAME:
        # Refused here, not at generation time: the generation block runs after the last training step, so a
        # late refusal would have spent the whole run first. A prompt quoting this repository's own section
        # titles, put to a model trained on a different corpus, measures memorization of text the model never
        # saw (K-026) — and E2612's qualitative block was exactly that, by omission of a flag.
        message = (
            f"this run trains on corpus {manifest.name!r} but samples with the repository's own prompts, "
            f"which are drawn from {CORPUS_NAME!r}; pass --sample-prompts held_out so the generation "
            "block continues text from this corpus's own test side"
        )
        raise CorpusError(message)

    split = split_text_units(
        manifest.text_units(),
        validation_fraction=_VALIDATION_FRACTION,
        test_fraction=_TEST_FRACTION,
        seed=plan.effective_split_seed,
        whole_works=plan.whole_work_splits,
    )
    # Deliberately stops here. The vocabulary is fitted on the whole training side whichever prefixes the run
    # trains on, and a plan whose share prefix no work carries must be refused before that happens — a refused
    # plan that left a fitted tokenizer behind would cache an artifact no record cites.
    return manifest, origin, snapshot, split


@dataclass(frozen=True, slots=True)
class EvaluationSides:
    """The two held-out sides of a plan, packed the way the plan packs them, with the units each pack holds.

    The units travel beside the batch because a segment identifier is only meaningful against the list that was
    packed, and a caller that wants to know which work a scored document came from must ask the pack, not guess.
    """

    validation: PackedBatch
    validation_units: tuple[Any, ...]
    test: PackedBatch
    test_units: tuple[Any, ...]
    selection_vocabulary: SubwordVocabulary | None


def evaluation_sides(
    plan: ExperimentPlan, split: UnitSplit, vocabulary: SubwordVocabulary
) -> EvaluationSides:
    """The held-out packs, sampled and capped in one place.

    A run's published loss is computed over exactly these blocks, so any second implementation that sampled the
    test side differently would be scoring different text and calling it the same measurement. E2633 found that by
    being refused: taking the test side in split order gives 256,256 positions and 483,895 characters against the
    254,976 and 535,282 every report prints, because ``max_test_tokens`` caps a **seeded sample** of the side, not
    its prefix.
    """
    selection_vocabulary = (
        None
        if plan.evaluation_selection_tokenizer_file is None
        else SubwordVocabulary.load(plan.evaluation_selection_tokenizer_file)
    )
    validation_units = _sampled_units(
        split.validation, plan.max_validation_tokens, plan.effective_split_seed
    )
    test_units = _sampled_units(split.test, plan.max_test_tokens, plan.effective_split_seed)
    packed: dict[str, tuple[tuple[Any, ...], PackedBatch]] = {}
    for side, units, cap in (
        ("validation", validation_units, plan.max_validation_tokens),
        ("test", test_units, plan.max_test_tokens),
    ):
        packed[side] = (units, _pack_evaluation(units, vocabulary, plan, cap, selection_vocabulary))
    return EvaluationSides(
        validation=packed["validation"][1],
        validation_units=packed["validation"][0],
        test=packed["test"][1],
        test_units=packed["test"][0],
        selection_vocabulary=selection_vocabulary,
    )


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
    manifest, origin, snapshot, split = pin_and_split(plan)

    share_prefix, share = plan.share_group_prefix, plan.share_of_tokens
    training_units: tuple[Any, ...] = ()
    if share_prefix is None:
        training_units = _training_units(
            split.train,
            plan.training_group_prefixes,
            shuffle_seed=plan.effective_split_seed if plan.shuffle_training_units else None,
        )
    elif share is not None:
        _require_share_classes(split.train, share_prefix, share)
    # Two arms of a mixture comparison must share one tokenizer, or the loss difference is partly a different
    # alphabet. Fitted after the share checks above, never before them.
    vocabulary, tokenizer_reused = _fit_or_reuse_vocabulary(plan, work, manifest, split)
    char_vocabulary = CharVocabulary(tuple(unit.text for unit in split.train))

    composition: ShareSelection | None = None
    if share_prefix is not None and share is not None and plan.max_training_tokens is not None:
        # Needs the vocabulary: a quota is counted in identifiers, not characters.
        composition = select_by_share(
            split.train,
            vocabulary,
            prefix=share_prefix,
            share=share,
            max_tokens=plan.max_training_tokens,
            seed=plan.effective_split_seed,
        )
        training_units = composition.units

    sides = evaluation_sides(plan, split, vocabulary)
    validation_units, test_units = sides.validation_units, sides.test_units
    selection_vocabulary = sides.selection_vocabulary

    # One window per character position of every side: on the repository's own 780 KB that is affordable,
    # on a 14.5M-character corpus it is not, and the machine running the experiment has the last word. A
    # skipped baseline is recorded as skipped rather than quietly omitted.
    char_windows: dict[str, tuple[Any, Any]] = {}
    if plan.character_baseline:
        char_windows = {
            side: _character_batches(split_side, char_vocabulary, plan.character_context)
            for side, split_side in (
                ("train", training_units),
                ("validation", validation_units),
                ("test", test_units),
            )
        }

    return PreparedCorpus(
        manifest=manifest,
        split=split,
        vocabulary=vocabulary,
        char_vocabulary=char_vocabulary,
        train=pack_units(
            training_units,
            vocabulary,
            context_length=plan.context_length,
            track_documents=plan.document_masking,
            # A share-composed selection already holds its quotas to within a document; a cap applied to the
            # interleaved order would drop the last documents and move the share it just established.
            max_tokens=None if composition is not None else plan.max_training_tokens,
        ),
        validation=sides.validation,
        test=sides.test,
        char_windows=char_windows,
        timings={"prepare_seconds": round(time.perf_counter() - started, 2)},
        corpus_origin=origin,
        snapshot=snapshot,
        tokenizer_reused=tokenizer_reused,
        training_composition=composition,
        selection_vocabulary=selection_vocabulary,
    )


def _pack_evaluation(
    units: tuple[Any, ...],
    vocabulary: SubwordVocabulary,
    plan: ExperimentPlan,
    cap: int | None,
    selection_vocabulary: SubwordVocabulary | None,
) -> PackedBatch:
    """Pack a held-out side, choosing its documents with the selection vocabulary when the plan named one.

    With none, the cap is counted with ``vocabulary`` inside :func:`pack_units`, exactly as before. With one, the
    documents are the ones a pack capped under *that* vocabulary would hold, and they are packed uncapped, so every
    tokenizer given the same selection vocabulary is scored on the same text.

    Document boundaries are tracked whatever the masking setting, because a held-out pack that does not know which
    document each position came from can only be measured as a whole. Nothing reads the identifiers unless
    ``document_masking`` is on, so the scored numbers are unchanged; what they buy is :func:`work_losses`, which
    splits a published loss into the works a paired test resamples.
    """
    if selection_vocabulary is None or cap is None:
        return pack_units(
            units,
            vocabulary,
            context_length=plan.context_length,
            track_documents=True,
            max_tokens=cap,
        )
    return pack_units(
        select_units_by_token_cap(units, selection_vocabulary, cap),
        vocabulary,
        context_length=plan.context_length,
        track_documents=True,
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


def _prompt_set(plan: ExperimentPlan, prepared: PreparedCorpus) -> tuple[tuple[str, str], ...]:
    """The prompts a run samples generations from, chosen before any generation happens.

    ``held_out`` takes the opening of the first few test-side documents, in document order: the choice is a
    function of the split, not of what the model produced, and the identifiers are recorded so a reader can
    see exactly which text was used.
    """
    if plan.sample_prompts != "held_out":
        return tuple(("chakaso-repo-text-v1", prompt) for prompt in _PROMPTS)
    taken: list[tuple[str, str]] = []
    for unit in prepared.split.test:
        cleaned = " ".join(unit.text.split())
        if len(cleaned) >= _HELD_OUT_PROMPT_CHARACTERS:
            taken.append((unit.identifier, cleaned[:_HELD_OUT_PROMPT_CHARACTERS]))
        if len(taken) == len(_PROMPTS):
            break
    if not taken:
        message = (
            "the run asked for held-out prompts but no test-side document is long enough to supply one; "
            "the comparison would be against nothing"
        )
        raise ValueError(message)
    return tuple(taken)


def score_transformer(
    model: TrainedModel, batch: PackedBatch, vocabulary: SubwordVocabulary, *, plan: ExperimentPlan
) -> ModelMetrics:
    """Score a trained model on one packed side of the split, through the shared arithmetic."""
    nats, targets, predictions = score_model_on_examples(
        model, batch.examples, batch_size=plan.batch_size, document_masking=plan.document_masking
    )
    return _metrics(nats, targets, predictions, vocabulary)


def _sampled_units(units: tuple[Any, ...], cap: int | None, seed: int) -> tuple[Any, ...]:
    """Shuffle a side's documents with the split seed, so a token cap samples rather than slices.

    Slicing would take the first works in catalogue order, which is one register; a seeded shuffle makes the
    capped side a sample of every work the split held out.
    """
    if cap is None:
        return units
    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled)


def _test_partition_metrics(
    model: TrainedModel, prepared: PreparedCorpus, plan: ExperimentPlan
) -> dict[str, Any]:
    """Score each declared partition of the test side on its own packed stream.

    A partition is a group prefix (a register, in the corpora this experiment uses). Each is packed
    independently, so its tails differ by a few tokens from a slice of the whole test stream; the report
    states the tokens and characters each block was scored over, which is what makes the number checkable.
    """
    if plan.test_partitions is None:
        return {}
    packs, uniform_value, baseline = partition_packs(prepared, plan)
    return _partition_metrics(model, prepared, plan, packs, uniform_value, baseline)


def partition_packs(
    prepared: PreparedCorpus, plan: ExperimentPlan
) -> tuple[dict[str, Any], float, UnigramBaseline]:
    """Pack every declared test-side partition, and fit the unigram baseline, once.

    This is the half of the partition measurement that does not depend on which model is being scored. A run
    pays it once at its end; an analysis that scores several models of one protocol over the same pack would
    otherwise pay it per model, and the part that hurts is the baseline: it is fitted over every training
    target of the run, which is 19,660,800 identifiers at the reference configuration.
    """
    train_targets = [token for example in prepared.train.examples for token in example.targets]
    baseline = UnigramBaseline.fit(train_targets, prepared.vocabulary.size)
    uniform_value = log(prepared.vocabulary.size)
    packs: dict[str, Any] = {}
    if plan.test_partitions is None:
        return packs, uniform_value, baseline
    for prefix in plan.test_partitions:
        units = _sampled_units(
            tuple(unit for unit in prepared.split.test if unit.group.startswith(prefix)),
            plan.max_test_tokens,
            plan.effective_split_seed,
        )
        if not units:
            packs[prefix] = None
            continue
        packs[prefix] = (
            units,
            _pack_evaluation(
                units,
                prepared.vocabulary,
                plan,
                plan.max_test_tokens,
                prepared.selection_vocabulary,
            ),
        )
    return packs, uniform_value, baseline


def _partition_metrics(
    model: TrainedModel,
    prepared: PreparedCorpus,
    plan: ExperimentPlan,
    packs: Mapping[str, Any],
    uniform_value: float,
    baseline: UnigramBaseline,
) -> dict[str, Any]:
    """Score each already-packed partition with ``model``. Numbers are unchanged by the split; see
    :func:`partition_packs` for why the packing and the baseline are not recomputed here."""
    if plan.test_partitions is None:
        return {}
    produced: dict[str, Any] = {}
    for prefix in plan.test_partitions:
        entry = packs[prefix]
        if entry is None:
            produced[prefix] = {"works": 0, "reason": "no test work matched this prefix"}
            continue
        units, batch = entry
        metrics = score_transformer(model, batch, prepared.vocabulary, plan=plan)
        targets = [token for example in batch.examples for token in example.targets]
        produced[prefix] = {
            "works": len({unit.group for unit in units}),
            # Named for what they count. ``population_characters`` is every character of the held-out works
            # in this partition; ``metrics.characters_scored`` is what the sampled pack was scored over, and
            # an aggregate weighted by the wrong one moves by 4.5x the materiality band (K-027).
            "documents_packed": batch.document_count,
            "population_characters": sum(len(unit.text) for unit in units),
            "tokens_scored": len(targets),
            "metrics": metrics.as_dict(),
            "uniform_nats": round(uniform_value, 6),
            "unigram_nats": round(
                sum(-baseline.log_probability(token) for token in targets) / max(1, len(targets)), 6
            ),
        }
    return produced


def score_artifacts(
    plan: ExperimentPlan, artifacts: Mapping[str, Path]
) -> dict[str, dict[str, Any]]:
    """Measure several models of one protocol against the same held-out pack, through the run's own code path.

    A training curve written by ADR-0050's rungs, a seed study's outputs, or any two arms a record wants
    compared have to be scored over identical documents and identical arithmetic; re-implementing the pack
    selection or the metric in a script would make the comparison a claim about two code paths instead of one.
    The corpus is prepared once and every named artifact is scored against that same prepared pack, with the
    whole test side and each declared partition, in the shapes ``run_experiment`` writes into its report.

    Raises:
        CorpusError: an artifact was trained on a different corpus or a different vocabulary than the plan
            prepares. A rung from another pack would be scored on text its own run never split.
    """
    prepared = prepare_corpus(plan)
    packs, uniform_value, baseline = partition_packs(prepared, plan)
    scored: dict[str, dict[str, Any]] = {}
    for name, path in artifacts.items():
        artifact = load_artifact(path)
        if artifact.corpus_fingerprint != prepared.manifest.fingerprint:
            message = (
                f"{name} was trained on corpus {artifact.corpus_fingerprint}, but this plan prepares "
                f"{prepared.manifest.fingerprint}; its held-out numbers would not be comparable"
            )
            raise CorpusError(message)
        if artifact.vocabulary.fingerprint != prepared.vocabulary.fingerprint:
            message = (
                f"{name} carries tokenizer {artifact.vocabulary.fingerprint}, but this plan prepares "
                f"{prepared.vocabulary.fingerprint}; encoding the same text differently changes the metric"
            )
            raise CorpusError(message)
        model = artifact.to_model()
        scored[name] = {
            "path": path.name,
            "checksum": artifact.checksum(),
            "step": artifact.step,
            "test": score_transformer(
                model, prepared.test, prepared.vocabulary, plan=plan
            ).as_dict(),
            "test_partitions": _partition_metrics(
                model, prepared, plan, packs, uniform_value, baseline
            ),
        }
    return scored


def work_losses(plan: ExperimentPlan, artifacts: Mapping[str, Path]) -> dict[str, Any]:
    """Score each named artifact on the plan's own held-out pack, split into the works its documents came from.

    :func:`score_artifacts` publishes the whole-pack and per-register numbers a run reports; this publishes the same
    measurement at the unit a paired test needs. Two arms scored on identical documents differ document by document,
    and the works — the units the whole-work split holds out — are what a cluster bootstrap may resample: the dozen
    chapters of one held-out book are one observation of one prose style, not a dozen.

    The pinning, the split, the vocabulary and the pack are the run's own code path, through
    :func:`pin_and_split` and :func:`evaluation_sides`. That is not cosmetic. A reconstruction that took the test
    side in split order rather than in the seeded sample the cap is applied to scores different documents: E2633
    asked for 250,000 identifiers that way and got 256,256 positions and 483,895 characters where every committed
    report records 254,976 and 535,282, and its pooled loss came out 1.2 bits per character worse on models that had
    already published their number.

    Raises:
        CorpusError: an artifact was trained on a different corpus or carries a different vocabulary than the plan
            pins — the same guards ``score_artifacts`` applies, for the same reason.
    """
    manifest, _origin, _snapshot, split = pin_and_split(plan)
    # The cache lookup, not a refit: a measurement over existing artifacts must use the vocabulary the arms
    # trained with, and refuse if it cannot find it.
    vocabulary, _reused = _fit_or_reuse_vocabulary(plan, plan.work_dir, manifest, split)
    sides = evaluation_sides(plan, split, vocabulary)
    arms: dict[str, dict[str, Any]] = {}
    for name, path in artifacts.items():
        artifact = load_artifact(path)
        if artifact.corpus_fingerprint != manifest.fingerprint:
            message = (
                f"{name} was trained on corpus {artifact.corpus_fingerprint}, but this plan pins "
                f"{manifest.fingerprint}; its held-out numbers would not be comparable"
            )
            raise CorpusError(message)
        if artifact.vocabulary.fingerprint != vocabulary.fingerprint:
            message = (
                f"{name} carries tokenizer {artifact.vocabulary.fingerprint}, but this plan pins "
                f"{vocabulary.fingerprint}; encoding the same text differently changes the metric"
            )
            raise CorpusError(message)
        losses = document_losses(
            artifact.to_model(),
            sides.test.examples,
            artifact.vocabulary,
            sides.test_units,
            batch_size=plan.batch_size,
            document_masking=plan.document_masking,
        )
        works: dict[str, dict[str, float]] = {}
        for loss in losses:
            key = loss.group or f"document-{loss.segment}"
            row = works.setdefault(key, {"nats": 0.0, "characters": 0.0, "documents": 0.0})
            row["nats"] = fsum([row["nats"], loss.nats])
            row["characters"] += loss.characters
            row["documents"] += 1
        for row in works.values():
            row["bits_per_character"] = (
                log2(e) * row["nats"] / row["characters"] if row["characters"] else inf
            )
        nats_total = fsum(loss.nats for loss in losses)
        characters_total = sum(loss.characters for loss in losses)
        arms[name] = {
            "path": Path(path).name,
            "checksum": artifact.checksum(),
            "step": artifact.step,
            "corpus_fingerprint": artifact.corpus_fingerprint,
            "tokenizer_fingerprint": artifact.vocabulary.fingerprint,
            "positions_scored": sum(loss.positions for loss in losses),
            "characters_scored": characters_total,
            "documents": len(losses),
            "pooled_bits_per_character": (
                log2(e) * nats_total / characters_total if characters_total else inf
            ),
            "works": works,
        }
    return {
        "pack": {
            "corpus_fingerprint": manifest.fingerprint,
            "corpus_name": manifest.name,
            "split_fingerprint": split.fingerprint,
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "selection_vocabulary_fingerprint": (
                None
                if sides.selection_vocabulary is None
                else sides.selection_vocabulary.fingerprint
            ),
            "blocks": len(sides.test.examples),
            "positions": sides.test.positions,
            "documents": sides.test.document_count,
            "works": len({unit.group for unit in sides.test_units}),
            "document_masking": plan.document_masking,
        },
        "arms": arms,
    }


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
        document_masking=plan.document_masking,
        weight_decay_excludes_vectors=plan.weight_decay_excludes_vectors,
        curve_artifact_steps=plan.curve_artifact_steps,
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
    config = _training_config(plan, architecture)
    rungs: list[dict[str, Any]] = []

    def write_rung(step: int, state: Mapping[str, Any], validation_loss: float | None) -> None:
        """One point on the curve, written as a model a reader can load.

        Named by the same helper the final artifact uses, so a rung and the run's own file differ only in the
        step in their name; ``save_artifact`` re-enters the refuse-to-clobber rule, so a second ladder run in
        the same work directory is a collision the operator asked for rather than an overwrite. The checksum
        is recorded beside it, because a rung a later measurement cites has to be identifiable after the file
        has moved.
        """
        path = work / checkpoint_filename(f"{plan.experiment_id}-step{step}", plan.model_family)
        rung = rung_artifact(
            state=state,
            config=config,
            vocabulary=prepared.vocabulary,
            corpus_fingerprint=prepared.manifest.fingerprint,
            step=step,
            validation_loss_nats=validation_loss,
        )
        save_artifact(rung, path, allow_overwrite=plan.overwrite_artifacts)
        rungs.append(
            {
                "step": step,
                "path": path.name,
                "validation_loss_nats": (
                    None if validation_loss is None else round(float(validation_loss), 6)
                ),
                "checksum": rung.checksum(),
            }
        )

    resume: dict[str, Any] = {}
    budget_extended_from: int | None = None
    if plan.resume_from is not None:
        point = load_artifact(plan.resume_from)
        verified = verify_resume_compatibility(
            point,
            config=config,
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
        config=config,
        corpus_fingerprint=prepared.manifest.fingerprint,
        on_step=on_step,
        curve_artifact_writer=write_rung if plan.curve_artifact_steps else None,
        checkpoint_path=work / f"{plan.experiment_id}-{plan.model_family}-resumable.pt",
    )
    wall = time.perf_counter() - started

    artifact = outcome.artifact()
    model = artifact.to_model()
    test_metrics = score_transformer(model, prepared.test, prepared.vocabulary, plan=plan)
    test_partition_metrics = _test_partition_metrics(model, prepared, plan)
    validation_metrics = score_transformer(
        model, prepared.validation, prepared.vocabulary, plan=plan
    )
    artifact_path = work / checkpoint_filename(plan.experiment_id, plan.model_family)
    save_artifact(artifact, artifact_path, allow_overwrite=plan.overwrite_artifacts)

    # Generation goes through the boundary, not the adapter class: the experiment is a consumer of the
    # public interface, exactly as an application would be, so a break in load_model or in the contract
    # shows up here as well as in the adapter's own tests.
    from chakaso.models import GenerationParams, Message, Role, load_model

    inference = load_model(artifact_path)
    params = GenerationParams(max_new_tokens=plan.samples_per_prompt, temperature=0.0)
    samples = []
    sampling_started = time.perf_counter()
    for source, prompt in _prompt_set(plan, prepared):
        result = inference.generate([Message(role=Role.USER, content=prompt)], params)
        samples.append(
            {
                "prompt": prompt,
                "prompt_source": source,
                "text": result.text,
                "finish_reason": result.finish_reason.value,
                "model_id": result.model_id,
            }
        )
    sampling_seconds = time.perf_counter() - sampling_started

    # What the model emits is measured with the same instrument a consumer would reach for — the suite runs
    # over the boundary, not over the adapter's internals, so the numbers describe the thing a release ships.
    from chakaso.core.process_memory import peak_memory
    from chakaso.training.generation_eval import evaluate_generation

    generation = evaluate_generation(inference)
    memory = peak_memory()
    produced = sum(len(item["text"]) for item in samples)
    latency = {
        "samples": len(samples),
        "seconds": round(sampling_seconds, 3),
        "characters": produced,
        "characters_per_second": round(produced / sampling_seconds, 2)
        if sampling_seconds > 0
        else None,
        "measurement": "greedy generation through chakaso.models.load_model, warm process, no batching",
    }

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
            "whole_work_splits": plan.whole_work_splits,
            "training_group_prefixes": (
                None if plan.training_group_prefixes is None else list(plan.training_group_prefixes)
            ),
            "training_groups": sorted(prepared.training_composition.identifiers_by_group)
            if prepared.training_composition is not None
            else sorted({unit.group for unit in prepared.split.train if unit.group})
            if plan.training_group_prefixes is None
            else sorted(
                {
                    unit.group
                    for unit in prepared.split.train
                    if unit.group.startswith(plan.training_group_prefixes)
                }
            ),
            "max_training_tokens": plan.max_training_tokens,
            "max_validation_tokens": plan.max_validation_tokens,
            "max_test_tokens": plan.max_test_tokens,
            "validation_documents_packed": prepared.validation.document_count,
            "test_documents_packed": prepared.test.document_count,
            "shuffle_training_units": plan.shuffle_training_units,
            "training_documents_packed": prepared.train.document_count,
            "training_tokens_packed": prepared.train.tokens_available,
            **(
                {}
                if prepared.selection_vocabulary is None
                else {
                    "evaluation_selection": {
                        "tokenizer_fingerprint": prepared.selection_vocabulary.fingerprint,
                        "validation_documents": prepared.validation.document_count,
                        "test_documents": prepared.test.document_count,
                    }
                }
            ),
            **_composition_block(plan, prepared),
            "groups_per_side": groups_per_side(prepared.split),
            "works_on_multiple_sides": works_on_multiple_sides(prepared.split),
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
            **(
                {}
                if prepared.vocabulary.pre_tokenizer == DEFAULT_PRE_TOKENIZER
                else {"pre_tokenizer": prepared.vocabulary.pre_tokenizer}
            ),
            "sample_characters": plan.tokenizer_sample_characters,
            "sample_documents": len(_tokenizer_sample(plan, prepared.split))
            if plan.tokenizer_sample_characters is not None
            else len(prepared.split.train),
            "sample_actual_characters": sum(
                len(text) for text in _tokenizer_sample(plan, prepared.split)
            )
            if plan.tokenizer_sample_characters is not None
            else sum(len(unit.text) for unit in prepared.split.train),
            "reused_from_cache": prepared.tokenizer_reused,
            "supplied_file": plan.tokenizer_file is not None,
            "tokens_per_side": prepared.tokens,
            "test_characters_decoded": len(prepared.vocabulary.decode(targets)),
            "characters_per_token_test": round(
                len(prepared.vocabulary.decode(targets)) / max(1, len(targets)), 4
            ),
        },
        "character_baseline": character,
        "model_family": plan.model_family,
        # The key is the name three measured records' reports already use and the shape :func:`summarise`
        # and the release builder read; the family a run trained is stated beside it, not by renaming it.
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
                "optimizer": optimizer_settings(outcome.config),
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
                "document_masking": plan.document_masking,
                "whole_work_splits": plan.whole_work_splits,
                "training_group_prefixes": (
                    None
                    if plan.training_group_prefixes is None
                    else list(plan.training_group_prefixes)
                ),
                "max_training_tokens": plan.max_training_tokens,
                "shuffle_training_units": plan.shuffle_training_units,
                # Present only for a share-composed run, so a recorded protocol's digest does not move.
                **(
                    {}
                    if plan.share_group_prefix is None
                    else {
                        "share_group_prefix": plan.share_group_prefix,
                        "share_of_tokens": plan.share_of_tokens,
                    }
                ),
                "effective_batch_size": outcome.config.effective_batch_size,
                "positions_per_second": round(outcome.positions_per_second, 1),
                "records": len(outcome.records),
                "wall_clock_s": round(wall, 2),
                "steps_per_epoch_estimate": -(-len(prepared.train.examples) // plan.batch_size),
            },
            "validation": validation_metrics.as_dict(),
            "test": test_metrics.as_dict(),
            "test_partitions": test_partition_metrics,
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
                "schema": schema_of(artifact.config),
            },
            # One entry per declared rung, in step order, each naming the file and the held-out loss recorded
            # for that step. An empty list says the run wrote only its final artifact, which is every
            # committed protocol; the key is present either way so a reader can tell "no rungs" from
            # "report written before rungs existed" by the plan field beside it.
            "curve_artifacts": sorted(rungs, key=lambda rung: int(rung["step"])),
        },
        "baselines_subword": baseline_metrics(prepared),
        "samples": samples,
        "generation": generation,
        "latency": latency,
        "resources": memory.as_dict(),
        "timings": prepared.timings | {"transformer_wall_clock_s": round(wall, 2)},
        "environment": artifact.environment,
        "plan": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(plan).items()
            if isinstance(value, int | float | str | Path)
        },
    }


def _composition_block(plan: ExperimentPlan, prepared: PreparedCorpus) -> dict[str, Any]:
    """The mixture a share-composed run packed, as counted; empty for every other plan."""
    composition = prepared.training_composition
    if composition is None:
        return {}
    registers = (
        plan.test_partitions
        if plan.test_partitions is not None
        else (() if plan.share_group_prefix is None else (plan.share_group_prefix,))
    )
    return {
        "training_composition": {
            "share_group_prefix": plan.share_group_prefix,
            "declared_share": plan.share_of_tokens,
            "realised_share": round(composition.realised_share, 6),
            "identifiers_matched": composition.matched_identifiers,
            "identifiers_other": composition.other_identifiers,
            "identifiers_by_register": identifiers_by_register(
                composition.identifiers_by_group, registers
            ),
            "works_packed": len(composition.identifiers_by_group),
        }
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


#: Plan fields that name where a run lives or what it was allowed to overwrite. They change with the machine
#: and the working directory, not with what was measured.
_MACHINE_LOCAL_PLAN_FIELDS = frozenset(
    {
        "root",
        "work_dir",
        "corpus_snapshot",
        "tokenizer_cache_dir",
        "tokenizer_file",
        "evaluation_selection_tokenizer_file",
        "overwrite_artifacts",
    }
)
#: Budget fields that are results or observed conditions of one execution, not conditions it was run under: the
#: rates and wall clock vary with the machine, and where the best validation step fell, whether the run stopped
#: early, how it completed and how many evaluations it recorded are *outcomes* that can differ between two faithful
#: runs of one protocol. Leaving them in would make a re-run that picked another best step look like another protocol.
_OBSERVED_BUDGET_FIELDS = frozenset(
    {
        "positions_per_second",
        "wall_clock_s",
        "best_step",
        "stopped_early",
        "completion",
        "records",
        "resumed_from_step",
    }
)


def protocol_digest(report: dict[str, Any], *, include_seed: bool = True) -> str:
    """A digest of the conditions a report's numbers were measured under, and of none of the numbers.

    ``write_report`` returns the SHA-256 of the file, and the file embeds wall-clock time, throughput,
    memory and thread count, so a faithful re-run reproduces every metric and cannot reproduce that digest
    (K-028). Hashing the metrics instead would fail for a different reason: floating-point results on a
    different CPU, BLAS or thread count agree to the digits that matter and not to the last bit, so a hash
    over them would call a good reproduction a failure.

    This digest is the part a re-run *can* reproduce exactly: the corpus and split fingerprints, the plan
    with its machine-local paths removed, the tokenizer's identity and per-side token counts, the
    architecture and parameter count, and the budget with its observed rates removed. Two reports with equal
    protocol digests were measured under the same conditions, so their metrics are comparable and any
    difference between them is a result, not a protocol change. The metrics themselves are compared by
    tolerance, never by hash.

    ``include_seed=False`` removes the training seed (the plan's and the budget's, and the corpus block's
    ``model_seed``): a seed study compares the digests of runs that differ in exactly that and must find them equal.
    """
    transformer = report["transformer"]
    seed_keys = frozenset() if include_seed else frozenset({"seed", "model_seed"})
    protocol = {
        "experiment": report["experiment"],
        "model_family": report["model_family"],
        # How the corpus was found ("declared snapshot", "working tree", "snapshot from this work dir") is
        # provenance of the lookup, not a condition: the fingerprints beside it say which corpus it was.
        "corpus": {
            name: value
            for name, value in report["corpus"].items()
            if name not in seed_keys and name != "prepared_from"
        },
        "plan": {
            name: value
            for name, value in report["plan"].items()
            if name not in _MACHINE_LOCAL_PLAN_FIELDS and name not in seed_keys
        },
        "tokenizer": {
            name: value
            for name, value in report["tokenizer"].items()
            if name not in {"reused_from_cache", "supplied_file"}
        },
        "architecture": transformer["architecture"],
        "parameter_count": transformer["parameter_count"],
        "budget": {
            name: value
            for name, value in transformer["budget"].items()
            if name not in _OBSERVED_BUDGET_FIELDS and name not in seed_keys
        },
    }
    return sha256_hex(json.dumps(protocol, sort_keys=True, default=str, ensure_ascii=True))


def summarise(report: dict[str, Any]) -> str:
    """A one-line digest of the comparison, for the record and for an operator's terminal."""
    transformer = report["transformer"]["test"]
    character = report["character_baseline"].get("test")
    uniform = report["baselines_subword"]["uniform"]
    unigram = report["baselines_subword"]["unigram"]
    budget = report["transformer"]["budget"]
    parts = [
        f"transformer {report['transformer']['parameter_count']:,} params, "
        f"test bits/char {transformer['bits_per_character']:.4f}, "
        f"perplexity {transformer['perplexity']:.2f} at step {budget['best_step']} "
        f"(early stopped: {budget['stopped_early']})",
        (
            f"character test bits/char {character['bits_per_character']:.4f}"
            if character is not None
            else "character baseline skipped (see the report's reason)"
        ),
        f"unigram {unigram['bits_per_character']:.4f}",
        f"uniform {uniform['bits_per_character']:.4f}",
    ]

    # Both of these are read with ``get`` because reports written before the run measured them exist on disk
    # and are cited by records: a summary that only works on the newest shape would break a re-read of a
    # measured run, which is the one thing the reproducibility rule refuses to do.
    generation = report.get("generation", {}).get("automated_summary", {})
    if generation:
        parts.append(
            f"generations: {generation['samples_generated']} sampled, "
            f"{len(generation.get('looping_samples', []))} tail-looping, "
            f"{len(generation.get('periodic_run_samples', []))} cycling, "
            f"{generation['samples_refused']} refused"
        )
    resources = report.get("resources", {})
    peak = resources.get("peak_resident_megabytes")
    if peak is not None:
        parts.append(f"peak RSS {peak} MB via {resources['source']}")
    return " | ".join(parts)
