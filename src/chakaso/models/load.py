"""`load_model`: a trained Chakaso checkpoint, used from Python.

[ADR-0033](../../../docs/decisions/ADR-0033-model-centred-product-contract.md) makes the Python library
Chakaso's primary interface and states the test a developer should be able to pass: take a trained
checkpoint and use it — load it, generate from it — with no terminal command, no configuration file and no
import of the conversation runtime. This module is that surface.

```python
from chakaso.models import load_model

model = load_model("data/checkpoints/E2605-transformer.pt")
result = model.sample("The retrieval layer owns", max_new_tokens=32)
print(result.text)
```

What it deliberately does not do:

* **It does not build a second model abstraction.** `load_model` returns a
  :class:`~chakaso.models.base.LanguageModel` — the same boundary the application depends on — and selects
  the implementation through :func:`chakaso.models.registry.create_adapter`, which is where ADR-0002 puts
  name-to-class mapping. There is no parallel "GenerationModel" type to learn.
* **It does not guess which checkpoint you meant.** The format is read from the artifact: the JSON
  character checkpoint and the Transformer artifact select different adapters, and anything else is refused
  with the supported formats named.
* **It does not fall back.** A missing optional dependency, an absent path or an unreadable checkpoint is a
  typed `ModelError` naming what was missing. Silently returning the development double would turn a
  configuration mistake into a result that looks like a model answered.

Importing this module requires neither `numpy` nor `torch`; selecting an adapter may require one, and that
is reported when it does.
"""

from __future__ import annotations

import json
from pathlib import Path

from chakaso.models.base import LanguageModel
from chakaso.models.errors import ModelError
from chakaso.models.registry import (
    SUBWORD_TRANSFORMER_MODEL_ID,
    TRAINED_CHAR_MODEL_ID,
    create_adapter,
)

__all__ = ["CHAR_CHECKPOINT_SUFFIXES", "TRANSPORT_SUFFIXES", "load_model", "select_adapter_for"]

#: Checkpoint extensions this loader recognises, and the adapter each names.
CHAR_CHECKPOINT_SUFFIXES = frozenset({".json"})
TRANSPORT_SUFFIXES = frozenset({".pt"})


def select_adapter_for(path: Path) -> str:
    """Which adapter can read ``path``, decided from the artifact rather than from a filename alone.

    Raises:
        ModelError: the path is not a file, its extension is not a checkpoint format this loader knows, or
            a JSON artifact declares a schema this code does not implement.
    """
    if not path.is_file():
        message = f"no checkpoint file at {str(path)!r}"
        raise ModelError(message)
    suffix = path.suffix.lower()
    if suffix in TRANSPORT_SUFFIXES:
        return SUBWORD_TRANSFORMER_MODEL_ID
    if suffix in CHAR_CHECKPOINT_SUFFIXES:
        # Read the declared schema instead of trusting the extension: a JSON file that happens to end in
        # .json but carries another artifact is not a character checkpoint, and loading it as one would
        # produce a confusing failure deeper in the stack.
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            message = f"checkpoint at {str(path)!r} is not readable JSON: {exc}"
            raise ModelError(message) from exc
        schema = document.get("schema_version") if isinstance(document, dict) else None
        if schema == "chakaso-checkpoint/2":
            return TRAINED_CHAR_MODEL_ID
        message = (
            f"checkpoint at {str(path)!r} declares schema {schema!r}; this loader recognises "
            "'chakaso-checkpoint/2' (JSON) and the Chakaso Transformer artifact (.pt)"
        )
        raise ModelError(message)
    message = (
        f"cannot load {str(path)!r}: unrecognized checkpoint extension {suffix!r}. Supported: "
        f"{sorted(CHAR_CHECKPOINT_SUFFIXES | TRANSPORT_SUFFIXES)}"
    )
    raise ModelError(message)


def load_model(path: str | Path, *, adapter: str | None = None) -> LanguageModel:
    """Load a trained Chakaso checkpoint and return it as a :class:`LanguageModel`.

    Pass ``adapter`` to override the format detection (``"subword_transformer"`` or
    ``"trained_char"``); omit it to have the artifact's own declared schema choose. The returned object
    carries the metadata that states what it is — ``development_double`` is always false here, because a
    checkpoint is by definition not the double — and :meth:`sample` for a bare prompt or :meth:`generate`
    for the message-shaped boundary call.

    Raises:
        ModelError: the path is missing or unreadable, the format is not recognized, the artifact is
            corrupt or incompatible, or the adapter needs an optional extra that is not installed. Nothing
            is substituted, and no development double is returned.
    """
    target = Path(path)
    name = adapter or select_adapter_for(target)
    return create_adapter(name, checkpoint_path=str(target))
