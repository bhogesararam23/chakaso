"""A decoder-only causal Transformer, in the framework the model path is allowed to use.

This is the architecture the project's roadmap was written around and had never built: token embeddings
plus learned positions, pre-normalization multi-head causal self-attention, a GELU feed-forward block,
residual connections, and a linear language-model head trained on next-token cross-entropy. It is
deliberately the standard, well-understood shape rather than a novel one — at this scale the question is
whether the project's data pipeline, training loop, checkpoint format and evaluation arithmetic work
together, and an unusual architecture would confound that with an untested variable.

Why `torch` and not the hand-derived gradients of :mod:`chakaso.training.model`: ADR-0030 predicted this
turn and set its own revisit conditions, and ADR-0034 records them as met — a real corpus needs
mini-batching, and verifying every scalar of a Transformer backward pass by finite differences
extrapolates to roughly 45 hours at the measured forward cost. Autograd is then the reference, and the
verification duty shifts: instead of checking *every* parameter analytically, this layer is checked
structurally (masking, shapes, loss arithmetic, gradient finiteness, seeded initialization) and against
small hand-computed values, with a sampled finite-difference cross-check. That is a weaker guarantee than
the character model's, and it is written down as one.

Importing this module requires the optional ``[nn]`` extra. Nothing else in the package imports it:
``chakaso.training`` stays free of ``torch`` so the dependency-free tier keeps working
([ADR-0034](../../../docs/decisions/ADR-0034-optional-pytorch-model-extra.md)).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = ["CausalTransformer", "TransformerConfig", "count_parameters", "mask_positions"]


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """The architecture of one causal Transformer, validated on construction.

    Attributes:
        vocab_size: Identifier count of the bound tokenizer — the width of both embedding and head.
        context_length: The longest sequence the model can condition on; positions beyond it have no
            positional embedding.
        d_model: Residual stream width. Must divide evenly by ``num_heads``.
        num_layers: Repeated attention + feed-forward blocks.
        num_heads: Attention heads per block.
        feedforward_dim: Inner width of the feed-forward block.
        dropout: Dropout probability. Zero makes evaluation deterministic at a training cost.
        tie_embeddings: Share the token-embedding matrix with the output head, the standard choice when
            the vocabulary is small relative to the data, and the reason parameter count is reported
            rather than assumed.
        initialization_std: Standard deviation of the zero-mean normal that initializes every embedding
            and linear weight. Biases and normalization weights are fixed (zero and one).
    """

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    feedforward_dim: int
    dropout: float = 0.0
    tie_embeddings: bool = True
    initialization_std: float = 0.02

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                message = f"{name} must be an integer >= 1, got {value!r}"
                raise ValueError(message)
        if isinstance(self.feedforward_dim, bool) or not isinstance(self.feedforward_dim, int):
            message = f"feedforward_dim must be an integer, got {self.feedforward_dim!r}"
            raise ValueError(message)
        if self.feedforward_dim < self.d_model:
            message = (
                f"feedforward_dim ({self.feedforward_dim}) must be at least d_model "
                f"({self.d_model}); a narrower inner block is not a feed-forward expansion"
            )
            raise ValueError(message)
        if self.d_model % self.num_heads:
            message = (
                f"d_model ({self.d_model}) must divide evenly by num_heads ({self.num_heads}), "
                "or the head width is not an integer"
            )
            raise ValueError(message)
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            message = f"dropout must be a number, got {self.dropout!r}"
            raise ValueError(message)
        if not 0.0 <= self.dropout < 1.0:
            message = f"dropout must be in [0, 1), got {self.dropout}"
            raise ValueError(message)
        if isinstance(self.initialization_std, bool) or not isinstance(
            self.initialization_std, (int, float)
        ):
            message = f"initialization_std must be a number, got {self.initialization_std!r}"
            raise ValueError(message)
        if self.initialization_std <= 0:
            message = f"initialization_std must be positive, got {self.initialization_std}"
            raise ValueError(message)

    def as_dict(self) -> dict[str, Any]:
        """The configuration as plain data, for a checkpoint or an experiment record."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
            "tie_embeddings": self.tie_embeddings,
            "initialization_std": self.initialization_std,
        }

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> TransformerConfig:
        """Rebuild a configuration from a stored mapping, refusing unknown or missing keys.

        Raises:
            ValueError: a key this code does not understand is present, or a required one is missing. The
                architecture a checkpoint declares must be exactly the architecture the code can build; a
                silently ignored field would produce a model that is not the one that was saved.
        """
        known = set(TransformerConfig.__dataclass_fields__)
        unexpected = sorted(set(values) - known)
        if unexpected:
            message = f"transformer configuration has unknown fields: {unexpected}"
            raise ValueError(message)
        missing = sorted(known - set(values) - {"dropout", "tie_embeddings", "initialization_std"})
        if missing:
            message = f"transformer configuration is missing fields: {missing}"
            raise ValueError(message)
        return cls(**{key: values[key] for key in known if key in values})

    @property
    def head_width(self) -> int:
        """The width of one attention head."""
        return self.d_model // self.num_heads

    @property
    def parameter_count(self) -> int:
        """Trainable scalars this architecture has, computed from the configuration alone.

        Derived arithmetically rather than by building a model, so a record can state model size before a
        run and a test can check that the built model agrees with it. The breakdown: token and position
        embeddings, then per layer a fused input projection (``3d x d`` plus bias), an output projection, a
        two-matrix feed-forward, and two LayerNorms, then the final LayerNorm and — when the head is not
        tied to the embedding matrix — a second ``d x vocab`` block.
        """
        d = self.d_model
        per_layer = (
            3 * d * d
            + 3 * d
            + d * d
            + d
            + d * self.feedforward_dim
            + self.feedforward_dim
            + self.feedforward_dim * d
            + d
            + 2 * (d + d)
        )
        head = 0 if self.tie_embeddings else self.vocab_size * d
        return (
            self.vocab_size * d
            + self.context_length * d
            + self.num_layers * per_layer
            + 2 * d
            + head
        )


def mask_positions(sequence_length: int, device: torch.device | None = None) -> torch.Tensor:
    """The strictly-upper-triangular additive mask that makes attention causal.

    Position ``i`` may attend to positions ``<= i`` and to nothing else, so the mask is zero on and below
    the diagonal and ``-inf`` above it. Returned on the requested device because an additive mask on the
    wrong device is a runtime error that only appears when the model is actually used.
    """
    if sequence_length < 1:
        message = f"sequence_length must be >= 1, got {sequence_length}"
        raise ValueError(message)
    return torch.full((sequence_length, sequence_length), float("-inf")).triu(1).to(device)


def count_parameters(module: nn.Module) -> int:
    """Trainable scalar count — recorded so a result can state model size without guessing it."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class CausalTransformer(nn.Module):
    """Next-token prediction over a fixed context, with causal self-attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # `enable_nested_tensor=False` is stated rather than left to the default: torch's nested-tensor fast
        # path is disabled by `norm_first` anyway and warns about it, and this project trains dense, unpadded
        # blocks, so the optimization is not something the model path is missing. The suite treats warnings
        # as errors, so a framework's own notice has to be answered by a decision, not by loosening the gate.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Tying is explicit rather than a constructor flag: the head reads the embedding matrix, so a
            # checkpoint that stores one must rebuild the same relationship on load.
            self.head.weight = self.token_embedding.weight

        self._initialize(config.initialization_std)

    def _absolute_positions(self, batch: int, length: int, device: torch.device) -> torch.Tensor:
        return torch.arange(length, device=device).unsqueeze(0).expand(batch, length)

    def _initialize(self, std: float) -> None:
        """Zero-mean normal for every weight, zeros for biases, ones for normalization scales.

        Deterministic given the seed the caller set before constructing the module; nothing inherits a
        framework default silently, because "explicit initialization" is only a claim if the values are
        named.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                # nn.Embedding has no bias parameter at all, so asking torch's module for one raises; the
                # lookup goes through getattr rather than the attribute, which is the difference between a
                # biasless embedding and a confusing AttributeError at construction.
                bias = getattr(module, "bias", None)
                if bias is not None:
                    nn.init.zeros_(bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        """Trainable scalars in this model."""
        return count_parameters(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(logits, loss)`` for ``input_ids`` of shape ``(batch, length)``.

        ``targets`` holds the next-token identifier for each position; when it is present the loss is the
        mean cross-entropy over every position, in nats. ``positions`` overrides the absolute position
        identifiers, which a caller needs when a long context is processed in windows; by default the
        sequence is positioned from zero.

        Raises:
            ValueError: the batch is empty, a sequence exceeds the configured context, or an identifier is
                outside the vocabulary. These are shape errors, not data errors, and are refused before any
                arithmetic so a mis-sliced batch cannot train on a silently truncated window.
        """
        if input_ids.ndim != 2:
            message = f"input_ids must be 2-dimensional (batch, length), got shape {tuple(input_ids.shape)}"
            raise ValueError(message)
        batch, length = input_ids.shape
        if batch == 0 or length == 0:
            message = f"a batch must contain at least one sequence of at least one token, got {batch}x{length}"
            raise ValueError(message)
        if length > self.config.context_length:
            message = f"sequence length {length} exceeds the configured context {self.config.context_length}"
            raise ValueError(message)
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            message = (
                f"an input identifier is outside the vocabulary (size {self.config.vocab_size})"
            )
            raise ValueError(message)

        if positions is None:
            expanded = self._absolute_positions(batch, length, input_ids.device)
        else:
            if positions.shape != (batch, length):
                message = (
                    f"positions must have shape {(batch, length)}, got {tuple(positions.shape)}"
                )
                raise ValueError(message)
            expanded = positions
        if int(expanded.max()) >= self.config.context_length:
            message = (
                f"a position index ({int(expanded.max())}) reaches past the context length "
                f"({self.config.context_length})"
            )
            raise ValueError(message)

        hidden = self.token_embedding(input_ids) + self.position_embedding(expanded)
        causal_mask = mask_positions(length, input_ids.device).to(hidden.dtype)
        # The additive mask is passed explicitly and `is_causal` is never set: torch requires one or the
        # other, and an explicit mask is the form a test can inspect.
        encoded = self.encoder(hidden, mask=causal_mask)
        logits = self.head(self.final_norm(encoded))

        loss: torch.Tensor | None = None
        if targets is not None:
            if targets.shape != (batch, length):
                message = f"targets must have shape {(batch, length)}, got {tuple(targets.shape)}"
                raise ValueError(message)
            loss = nn.functional.cross_entropy(
                logits.reshape(batch * length, self.config.vocab_size),
                targets.reshape(batch * length),
            )
        return logits, loss

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The distribution over the token following ``input_ids`` (one sequence, shape ``(1, length)``).

        The inference entry point: it takes the last position only, which is the whole difference between
        training arithmetic and generation.
        """
        logits, _ = self.forward(input_ids)
        return logits[:, -1, :]

    @staticmethod
    def stack_ids(batches: Sequence[Sequence[int]]) -> torch.Tensor:
        """Turn equal-length identifier rows into an int64 tensor, refusing ragged input."""
        widths = {len(row) for row in batches}
        if len(widths) != 1:
            message = f"every row in a batch must have the same length, got widths {sorted(widths)}"
            raise ValueError(message)
        return torch.tensor([list(row) for row in batches], dtype=torch.int64)
