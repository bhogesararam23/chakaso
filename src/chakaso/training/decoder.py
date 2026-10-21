"""The decoder architecture the release path trains on: causal attention with a key/value cache.

This is a **new** architecture, not an edit of :mod:`chakaso.training.transformer`. E2605, E2606 and E2607
were measured with a learned position embedding, ``nn.TransformerEncoderLayer`` blocks and a loss that
covered every position of a dense packed block; those checkpoints keep their own format and their own
meaning, and a number already in a record is never recomputed against this file. What is different here, and
why:

* **Rotary positions instead of a learned position table.** The measured model's context was a fixed
  embedding table, so its longest sequence was an architectural constant and a long prompt had to be
  truncated to fit. Rotary angles are a function of position, they add no parameters, and they make "the
  same architecture at a longer context" a configuration change rather than a redesign.
* **An explicit attention block that can be stepped.** The measured architecture re-ran the whole prefix for
  every generated token, so generation cost grew quadratically with output length and the token budget, not
  the model, decided how long a sample could be. Here every projection is named and :class:`KeyValueCache`
  lets a caller append one token and re-run only that token. A cache is correct or it is wrong, so it is
  tested against the full forward pass rather than trusted.
* **Padding is representable.** :data:`IGNORE_INDEX` targets and a key-validity mask mean a batch can hold
  rows of different lengths. The measured model could only train on dense equal-length blocks, which is a
  constraint of its loss, not of language.
* **Pre-norm with RMSNorm and a depth-scaled residual initialization.** Normalization before the block keeps
  the residual stream bounded; dividing the two residual branches by ``sqrt(2 * layers)`` keeps the identity
  path dominant early in training. Both are the standard small-Transformer stability choices and both are
  checked by a test that would fail if they were wired the wrong way round.

Deliberately absent: dropout beyond the constructor field (every measured run used none), attention biases
(parameter count is reported, so the arithmetic has to stay exact), grouped-query attention, mixture of
experts, quantization, and any kernel that would make the numerics depend on a device this project cannot
verify. The model is float32 on CPU, and that is what is tested.

Importing this module requires the optional ``[nn]`` extra, like the architecture it succeeds
([ADR-0034](../../../docs/decisions/ADR-0034-optional-pytorch-model-extra.md)).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from chakaso.training.errors import TrainingError

__all__ = [
    "IGNORE_INDEX",
    "ChakasoDecoder",
    "DecoderConfig",
    "KeyValueCache",
    "apply_rotary",
    "count_parameters",
    "mask_positions",
    "rope_angles",
]

#: The target value meaning "this position is padding"; torch's cross-entropy skips it.
IGNORE_INDEX = -100

#: Floor inside the RMS normalization, so an all-zero vector cannot divide by zero.
_NORMALIZATION_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """The architecture of one decoder, validated on construction and rebuilt from a stored mapping.

    Attributes:
        vocab_size: Identifier count of the bound tokenizer; the width of both embedding and head.
        context_length: The longest sequence the model may be given, enforced at every forward pass. With
            rotary positions this is a budget rather than a table size.
        d_model: Residual stream width; must divide evenly by ``num_heads``.
        num_layers: Repeated attention + feed-forward blocks.
        num_heads: Attention heads per block.
        feedforward_dim: Inner width of the feed-forward block.
        dropout: Dropout probability. Zero keeps a measured run reproducible token for token.
        tie_embeddings: Share the token embedding matrix with the output head.
        initialization_std: Standard deviation of the normal that initializes every weight; the residual
            branches are additionally scaled by ``1 / sqrt(2 * num_layers)``.
        rope_theta: Base wavelength of the rotary angles. 10,000 is the value the formulation was published
            with; it is a field rather than a constant so a longer context can be measured against a
            different one instead of being assumed.
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
    rope_theta: float = 10_000.0

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
                f"feedforward_dim ({self.feedforward_dim}) must be at least d_model ({self.d_model}); "
                "a narrower inner block is not a feed-forward expansion"
            )
            raise ValueError(message)
        if self.d_model % self.num_heads:
            message = (
                f"d_model ({self.d_model}) must divide evenly by num_heads ({self.num_heads}), or the "
                "head width is not an integer"
            )
            raise ValueError(message)
        if self.head_width % 2:
            message = f"head width ({self.head_width}) must be even, or rotary pairs cannot cover every dimension"
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
        if isinstance(self.rope_theta, bool) or not isinstance(self.rope_theta, (int, float)):
            message = f"rope_theta must be a number, got {self.rope_theta!r}"
            raise ValueError(message)
        if self.rope_theta <= 1.0:
            message = f"rope_theta must exceed 1.0, got {self.rope_theta}"
            raise ValueError(message)

    @property
    def head_width(self) -> int:
        """The width of one attention head."""
        return self.d_model // self.num_heads

    @property
    def rope_pairs(self) -> int:
        """Rotary frequency pairs per head: each pair turns two adjacent dimensions together."""
        return self.head_width // 2

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
            "rope_theta": self.rope_theta,
        }

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> DecoderConfig:
        """Rebuild a configuration from a stored mapping, refusing unknown or missing keys.

        Raises:
            ValueError: a key this code does not understand is present, or a required one is missing. A
                checkpoint must describe exactly the architecture that can be built from it.
        """
        known = set(DecoderConfig.__dataclass_fields__)
        unexpected = sorted(set(values) - known)
        if unexpected:
            message = f"decoder configuration has unknown fields: {unexpected}"
            raise ValueError(message)
        optional = {"dropout", "tie_embeddings", "initialization_std", "rope_theta"}
        missing = sorted(known - set(values) - optional)
        if missing:
            message = f"decoder configuration is missing fields: {missing}"
            raise ValueError(message)
        return cls(**{key: values[key] for key in known if key in values})

    @property
    def parameter_count(self) -> int:
        """Trainable scalars, computed from the configuration alone.

        Derived arithmetically so a record can state model size *before* a run and a test can insist the
        built model agrees. Per layer: q, k, v and output projections (each ``d x d``, no biases), an up and
        a down projection across the feed-forward (each ``d x ff``, with biases), and two RMS norms. Plus the
        embedding, the final norm, and a second ``d x vocab`` when the head is untied.
        """
        d = self.d_model
        per_layer = 4 * d * d + 2 * d * self.feedforward_dim + self.feedforward_dim + d + 2 * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        return self.vocab_size * d + self.num_layers * per_layer + d + head


def rope_angles(positions: torch.Tensor, pairs: int, *, theta: float = 10_000.0) -> torch.Tensor:
    """Rotary angles for ``positions`` at ``pairs`` frequency bands.

    ``positions`` is an integer tensor of any shape — usually ``(batch, length)`` — and the result has one
    extra axis of ``pairs``, so it broadcasts against a ``(batch, heads, length, head_width)`` projection.
    Frequency ``i`` turns at ``position / theta**(2i / head_width)``: low-index dimensions rotate fast,
    high-index ones slowly.
    """
    if pairs < 1:
        message = f"rotary pairs must be >= 1, got {pairs}"
        raise ValueError(message)
    if positions.dtype not in (torch.int64, torch.int32):
        message = f"rotary positions must be integer identifiers, got dtype {positions.dtype}"
        raise ValueError(message)
    if positions.numel() and int(positions.min()) < 0:
        message = f"a rotary position is negative ({int(positions.min())})"
        raise ValueError(message)
    band = torch.arange(pairs, dtype=torch.float32, device=positions.device)
    scale = theta ** (band / pairs)
    return positions.to(torch.float32).unsqueeze(-1) / scale


def apply_rotary(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate ``values`` (batch, heads, length, head_width) by ``angles`` (batch, length, pairs).

    Dimensions ``2i`` and ``2i + 1`` are one pair and turn by the same angle. The interleaved form is stated
    once here and its consequence — that a dot product between two rotated vectors depends only on the
    *difference* of their positions — is what the test suite checks.

    Raises:
        ValueError: the head width is odd, or the angles cover a different number of positions or bands.
    """
    batch, _, length, head_width = values.shape
    if head_width % 2:
        message = f"head width {head_width} is odd, so rotary pairs cannot cover it"
        raise ValueError(message)
    pairs = head_width // 2
    if angles.shape[-2] != length:
        message = f"rotary angles cover {angles.shape[-2]} position(s) but the values hold {length}"
        raise ValueError(message)
    if angles.shape[-1] != pairs:
        message = f"rotary angles cover {angles.shape[-1]} band(s), expected {pairs}"
        raise ValueError(message)
    if angles.shape[0] != batch:
        message = f"rotary angles cover {angles.shape[0]} row(s), expected {batch}"
        raise ValueError(message)
    cosine = angles.cos()[:, None, :, :]
    sine = angles.sin()[:, None, :, :]
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1)
    return rotated.reshape(values.shape)


def mask_positions(
    query_length: int,
    key_length: int,
    *,
    offset: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """The additive causal mask for ``query_length`` queries against ``key_length`` keys.

    Query ``i`` sits at absolute position ``offset + i`` and may attend to keys at or before it. The offset is
    what makes a cached step correct: one new query against a longer cached past is not a square causal
    block, and deriving the mask from the shape alone would let the new token behave as if it were the first
    token of the sequence.
    """
    if query_length < 1 or key_length < 1:
        message = f"a mask needs at least one query and one key, got {query_length}x{key_length}"
        raise ValueError(message)
    if offset < 0:
        message = f"offset must not be negative, got {offset}"
        raise ValueError(message)
    queries = torch.arange(query_length).unsqueeze(1) + offset
    keys = torch.arange(key_length).unsqueeze(0)
    mask = torch.zeros(query_length, key_length, dtype=torch.float32)
    mask[keys > queries] = float("-inf")
    return mask.to(device)


def count_parameters(module: nn.Module) -> int:
    """Trainable scalar count — recorded so a result can state model size without guessing it."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class RMSNorm(nn.Module):
    """Root-mean-square normalization: no mean subtraction, no bias.

    The same stabilizer as LayerNorm with fewer operations and one fewer parameter vector, which at this
    scale is measurably cheaper for the same effect on the residual stream.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = values.pow(2).mean(dim=-1, keepdim=True).add(_NORMALIZATION_EPSILON).rsqrt()
        return values * scale * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention over explicit q/k/v projections that can be stepped."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.heads = config.num_heads
        self.head_width = config.head_width
        self.d_model = config.d_model
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        hidden: torch.Tensor,
        angles: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend ``hidden`` (batch, length, d_model); return the output and the keys worth caching."""
        batch, length, _ = hidden.shape
        shape = (batch, length, self.heads, self.head_width)
        queries = self.query(hidden).view(shape).transpose(1, 2)
        keys = self.key(hidden).view(shape).transpose(1, 2)
        values = self.value(hidden).view(shape).transpose(1, 2)

        # Rotated before caching: a stored key already carries its own position, so a later step concatenates
        # without needing to know which positions the past was rotated at.
        queries = apply_rotary(queries, angles)
        keys = apply_rotary(keys, angles)
        if past is not None:
            cached_keys, cached_values = past
            keys = torch.cat((cached_keys, keys), dim=2)
            values = torch.cat((cached_values, values), dim=2)

        attended = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.output(merged), (keys, values)


class FeedForward(nn.Module):
    """Expand, GELU, project back — the block's only non-linearity besides attention."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, config.feedforward_dim)
        self.down = nn.Linear(config.feedforward_dim, config.d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        activated: torch.Tensor = nn.functional.gelu(self.up(hidden))
        projected: torch.Tensor = self.down(activated)
        return projected


class DecoderBlock(nn.Module):
    """Pre-norm residual attention, then pre-norm residual feed-forward."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feedforward_norm = RMSNorm(config.d_model)
        self.feedforward = FeedForward(config)

    def forward(
        self,
        hidden: torch.Tensor,
        angles: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, keys = self.attention(self.attention_norm(hidden), angles, mask=mask, past=past)
        hidden = hidden + attended
        return hidden + self.feedforward(self.feedforward_norm(hidden)), keys


class KeyValueCache:
    """The keys and values a stepped decode has already computed, one tensor pair per layer.

    A cache is only meaningful alongside the position it stopped at, so the length is part of the object
    rather than something a caller must remember — :meth:`ChakasoDecoder.forward` refuses a cache that
    disagrees with the identifiers it is given.
    """

    def __init__(self, layers: int) -> None:
        if layers < 1:
            message = f"a cache needs at least one layer, got {layers}"
            raise TrainingError(message)
        self.slots: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * layers

    @property
    def length(self) -> int:
        """How many positions are stored, or zero before the first step."""
        for slot in self.slots:
            if slot is not None:
                return int(slot[0].shape[2])
        return 0

    def record(self, index: int, pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Store one layer's keys and values after a step."""
        self.slots[index] = pair


class ChakasoDecoder(nn.Module):
    """Next-token prediction with rotary positions and an optional key/value cache."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.token_embedding.weight
        self._initialize()

    def _initialize(self) -> None:
        """Every weight from one named normal; the two residual branches scaled by depth.

        ``1 / sqrt(2 * layers)`` on the attention output and the feed-forward down projection is the standard
        way to keep a deep residual stream near identity at step zero. Applied to exactly those two tensors,
        so a test can point at what it changed instead of trusting a loop.
        """
        std = self.config.initialization_std
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                bias = getattr(module, "bias", None)
                if bias is not None:
                    nn.init.zeros_(bias)

        scale = 1.0 / math.sqrt(2.0 * self.config.num_layers)
        for module in self.blocks:
            block = cast(DecoderBlock, module)
            block.attention.output.weight.data.mul_(scale)
            block.feedforward.down.weight.data.mul_(scale)

    @property
    def parameter_count(self) -> int:
        """Trainable scalars in this model."""
        return count_parameters(self)

    @property
    def device(self) -> torch.device:
        """The device the parameters live on — CPU is the only one this project verifies."""
        return self.token_embedding.weight.device

    def _validate_input(
        self, input_ids: torch.Tensor, cache: KeyValueCache | None
    ) -> tuple[int, int]:
        """Check the identifiers and the cache's depth, and return the batch shape.

        This runs before any position arithmetic, because a tensor of the wrong rank has no shape to index —
        a caller who passes a flat sequence should get "must be 2-dimensional", not an IndexError.

        Raises:
            ValueError: the rank is wrong, the batch is empty, an identifier is outside the vocabulary, or
                the cache was built for a different number of layers.
        """
        if input_ids.ndim != 2:
            message = f"input_ids must be 2-dimensional (batch, length), got shape {tuple(input_ids.shape)}"
            raise ValueError(message)
        batch, length = input_ids.shape
        if batch == 0 or length == 0:
            message = f"a batch must contain at least one sequence of at least one token, got {batch}x{length}"
            raise ValueError(message)
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            message = (
                f"an input identifier is outside the vocabulary (size {self.config.vocab_size})"
            )
            raise ValueError(message)
        if cache is not None and len(cache.slots) != self.config.num_layers:
            message = f"the cache holds {len(cache.slots)} layer(s) but this model has {self.config.num_layers}"
            raise ValueError(message)
        return batch, length

    def _validate_positions(
        self, positions: torch.Tensor, batch: int, length: int, cache: KeyValueCache | None
    ) -> None:
        """Check that every position is real, in range, and continues the cache it was given.

        Raises:
            ValueError: the shape disagrees with the identifiers, a position is negative or beyond the
                configured context, or a non-empty cache is being stepped out of order.
        """
        if positions.shape != (batch, length):
            message = f"positions must have shape {(batch, length)}, got {tuple(positions.shape)}"
            raise ValueError(message)
        if positions.numel() and int(positions.min()) < 0:
            message = f"a position is negative ({int(positions.min())})"
            raise ValueError(message)
        reached = int(positions.max()) + 1 if positions.numel() else 0
        if reached > self.config.context_length:
            message = (
                f"a position reaches {reached} token(s), past the configured context "
                f"{self.config.context_length}"
            )
            raise ValueError(message)
        if cache is not None and cache.length and not bool((positions[:, 0] == cache.length).all()):
            message = (
                f"the cache holds {cache.length} position(s) but the new identifiers start at "
                f"{sorted(set(positions[:, 0].tolist()))}; a stepped decode must continue where its "
                "cache stopped"
            )
            raise ValueError(message)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        cache: KeyValueCache | None = None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(logits, loss)`` for ``input_ids`` of shape ``(batch, length)``.

        Args:
            input_ids: The identifiers to condition on.
            targets: The next-token identifier per position, or :data:`IGNORE_INDEX` where the position is
                padding. Present means the loss is cross-entropy per unignored position, in nats.
            positions: Absolute position of every input identifier. Without a cache this defaults to
                zero-based; with one it must continue at the cached length.
            cache: A :class:`KeyValueCache` to read from and append to. Supplying it turns generation from a
                quadratic re-run into a linear step.
            valid: Boolean tensor marking which *key* positions are real, shape ``(batch, past + length)``.
                Masked-out keys cannot influence any query, which is what makes a padded batch harmless
                rather than merely inconvenient.

        Raises:
            ValueError: shapes disagree, an identifier is outside the vocabulary, a position exceeds the
                context, a cache disagrees with the identifiers, or every target is padding — a mean over
                zero positions is not a loss.
        """
        batch, length = self._validate_input(input_ids, cache)
        already = 0 if cache is None else cache.length
        if positions is None:
            start = torch.arange(length, device=input_ids.device) + already
            positions = start.unsqueeze(0).expand(batch, length)
        self._validate_positions(positions, batch, length, cache)
        keys = already + length

        if valid is not None:
            expected = (batch, keys)
            if valid.shape != expected:
                message = (
                    f"a key-validity mask must have shape {expected}, got {tuple(valid.shape)}"
                )
                raise ValueError(message)
            if valid.dtype != torch.bool:
                message = f"a key-validity mask must be boolean, got dtype {valid.dtype}"
                raise ValueError(message)
            mask = self._validity_mask(positions, valid, already=already)
        elif keys > 1:
            mask = mask_positions(length, keys, offset=already, device=input_ids.device)
        else:
            # One query against one key attends only to itself, so there is nothing to mask.
            mask = None

        angles = rope_angles(positions, self.config.rope_pairs, theta=self.config.rope_theta)
        hidden = self.token_embedding(input_ids)
        for index, block in enumerate(self.blocks):
            hidden, stored = block(
                hidden,
                angles,
                mask=mask,
                past=None if cache is None else cache.slots[index],
            )
            if cache is not None:
                cache.record(index, stored)

        logits = self.head(self.final_norm(hidden))
        loss: torch.Tensor | None = None
        if targets is not None:
            if targets.shape != (batch, length):
                message = f"targets must have shape {(batch, length)}, got {tuple(targets.shape)}"
                raise ValueError(message)
            if int((targets != IGNORE_INDEX).sum()) == 0:
                message = (
                    "every target in this batch is IGNORE_INDEX, so the loss would be a mean over zero "
                    "positions; a batch with nothing to predict should not be trained on"
                )
                raise ValueError(message)
            loss = nn.functional.cross_entropy(
                logits.reshape(batch * length, self.config.vocab_size),
                targets.reshape(batch * length),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    @staticmethod
    def _validity_mask(
        positions: torch.Tensor, valid: torch.Tensor, *, already: int
    ) -> torch.Tensor:
        """Causality and key validity combined into one additive mask of shape ``(batch, 1, queries, keys)``.

        ``valid`` marks columns, causality marks a triangle under each row's own diagonal, and the mask is
        their intersection: a query may attend to a key only when the key is real and not in its future.
        """
        queries = (positions - already).unsqueeze(-1)
        keys = torch.arange(already + positions.shape[1], device=positions.device)
        allowed = (keys[None, None, :] <= queries) & valid[:, None, :]
        mask = torch.zeros(allowed.shape, dtype=torch.float32, device=positions.device)
        return mask.masked_fill(~allowed, float("-inf"))[:, None, :, :]

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The distribution over the token following ``input_ids`` (one sequence, shape ``(1, length)``)."""
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
