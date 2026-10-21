"""Training: the code the from-scratch experiment runs on.

This is the module [ADR-0005](../../../docs/decisions/ADR-0005-src-layout.md) reserved for model-training
code, and [ADR-0030](../../../docs/decisions/ADR-0030-minimal-optional-training-dependency.md) admits
`numpy` — as an *optional* `[train]` extra — for the numerical layers here. Nothing above the model
boundary imports this package, and importing `chakaso` never requires `numpy`.

The layers split by whether they need a numerical library. The text-facing ones — the corpus and the
character tokenizer, the deterministic train/validation split, and the corpus manifest that pins a real
document set by content digest — are pure standard library, so they install, import and type-check with no
optional dependency at all. Only the model, the training step, the checkpoint format and the loss
computations import `numpy`; the Transformer path imports `torch`, and only in the modules ADR-0034 names.

No trained Chakaso model exists yet. The whole point of this package is to build the smallest honest
experiment that produces the project's first genuinely measured learning result; until that experiment
has actually run, nothing here is a claim that the approach works.
"""

from __future__ import annotations

from chakaso.training.baselines import UnigramBaseline, uniform_next_token_loss
from chakaso.training.config import TrainingConfig
from chakaso.training.corpus import CORPUS, Corpus, corpus_text
from chakaso.training.data import (
    Example,
    Split,
    TextUnit,
    ThreeWaySplit,
    UnitSplit,
    character_windows,
    dataset_fingerprint,
    fit_vocabulary,
    split_corpus,
    split_text_units,
    split_three_way,
)
from chakaso.training.manifest import (
    CorpusDocument,
    CorpusManifest,
    build_manifest,
    normalize_text,
    read_manifest,
    read_snapshot,
    save_manifest,
    verify_corpus,
    write_snapshot,
)
from chakaso.training.tokenizer import UNK_ID, CharVocabulary

__all__ = [
    "CORPUS",
    "UNK_ID",
    "CharVocabulary",
    "Corpus",
    "CorpusDocument",
    "CorpusManifest",
    "Example",
    "Split",
    "TextUnit",
    "ThreeWaySplit",
    "TrainingConfig",
    "UnigramBaseline",
    "UnitSplit",
    "build_manifest",
    "character_windows",
    "corpus_text",
    "dataset_fingerprint",
    "fit_vocabulary",
    "normalize_text",
    "read_manifest",
    "read_snapshot",
    "save_manifest",
    "split_corpus",
    "split_text_units",
    "split_three_way",
    "uniform_next_token_loss",
    "verify_corpus",
    "write_snapshot",
]
