"""Training: the product's own module — the code that makes a Chakaso model.

[ADR-0033](../../../docs/decisions/ADR-0033-model-centred-product-contract.md) makes this package the centre
of the repository rather than an experiment living beside it: a model this project trained, on data this
project can name, behind an interface a developer can use from Python.
[ADR-0005](../../../docs/decisions/ADR-0005-src-layout.md) reserved the module for training code,
[ADR-0030](../../../docs/decisions/ADR-0030-minimal-optional-training-dependency.md) admits `numpy` for its
character-level layers and [ADR-0034](../../../docs/decisions/ADR-0034-optional-pytorch-model-extra.md)
admits `torch` for the Transformer path. Nothing above the model boundary imports this package, and
importing `chakaso` requires neither.

The layers split by what they need:

* **Dependency-free** — the corpus manifest that pins a real document set by content digest, the snapshot
  that carries a run's exact bytes (and the history read that recovers them once the tracked text has
  moved), the document-level split, the character vocabulary, the learned byte-pair subword vocabulary, and
  sequence packing. These install, import, type-check and run their tests in the dependency-free CI matrix,
  because how text becomes identifiers is a decision that must be reviewable without a numerical framework
  in the way.
* **Numerical (`[train]`)** — the character model, its hand-derived gradients, the optimizer, the training
  loop, the checkpoint format and held-out evaluation.
* **`torch` (`[nn]`)** — the Transformer, its training and its checkpoints, in the modules ADR-0034 names.

What exists here is a measured *learning loop*: E2601 and E2602 trained and reproduced a 984-parameter
character model that beats uniform and unigram baselines on held-out loss. That is a real result on a toy,
not a useful language model, and nothing in this package is allowed to read as the latter.
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
    CorpusSnapshot,
    build_manifest,
    normalize_text,
    read_manifest,
    read_snapshot,
    save_manifest,
    verify_corpus,
    write_snapshot,
)
from chakaso.training.packing import (
    PackedBatch,
    PackedExample,
    count_tokens,
    pack_units,
    token_stream,
)
from chakaso.training.snapshot import (
    AcquisitionRecord,
    GitRevisionReader,
    RevisionReader,
    materialize_revision,
    restore_corpus_snapshot,
)
from chakaso.training.subword import (
    DOCUMENT_END,
    SubwordVocabulary,
    pre_tokenize,
    train_subword_vocabulary,
)
from chakaso.training.tokenizer import UNK_ID, CharVocabulary

__all__ = [
    "CORPUS",
    "DOCUMENT_END",
    "UNK_ID",
    "AcquisitionRecord",
    "CharVocabulary",
    "Corpus",
    "CorpusDocument",
    "CorpusManifest",
    "CorpusSnapshot",
    "Example",
    "GitRevisionReader",
    "PackedBatch",
    "PackedExample",
    "RevisionReader",
    "Split",
    "SubwordVocabulary",
    "TextUnit",
    "ThreeWaySplit",
    "TrainingConfig",
    "UnigramBaseline",
    "UnitSplit",
    "build_manifest",
    "character_windows",
    "corpus_text",
    "count_tokens",
    "dataset_fingerprint",
    "fit_vocabulary",
    "materialize_revision",
    "normalize_text",
    "pack_units",
    "pre_tokenize",
    "read_manifest",
    "read_snapshot",
    "restore_corpus_snapshot",
    "save_manifest",
    "split_corpus",
    "split_text_units",
    "split_three_way",
    "token_stream",
    "train_subword_vocabulary",
    "uniform_next_token_loss",
    "verify_corpus",
    "write_snapshot",
]
