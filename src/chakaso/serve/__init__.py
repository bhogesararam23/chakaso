"""The optional local HTTP interface to a trained Chakaso model.

``chakaso.serve`` is the serving adapter, not the product: it exposes one already-loaded model over HTTP so a
separate application — Python, JavaScript, anything that speaks JSON — can use it. The Python API
(:func:`chakaso.models.load_model`) stays the primary interface; this package is a thin transport in front of
it, and it is refused at startup if what it was pointed at is not a trained model.

Nothing in the base package imports this module, and nothing here imports ``torch``: a service's contract can
be exercised without a framework, and loading a model reports a missing extra by name instead of serving a
substitute.
"""

from __future__ import annotations

from chakaso.serve.app import (
    DEFAULT_MAX_INPUT_CHARACTERS,
    DEFAULT_MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_LIMIT,
    ServiceUnavailableError,
    ServingApplication,
    ServingError,
    load_serving_model,
)
from chakaso.serve.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_BODY_BYTES,
    ServingService,
    serve,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_MAX_INPUT_CHARACTERS",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PORT",
    "MAX_BODY_BYTES",
    "MAX_NEW_TOKENS_LIMIT",
    "ServiceUnavailableError",
    "ServingApplication",
    "ServingError",
    "ServingService",
    "load_serving_model",
    "serve",
]
