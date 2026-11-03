"""A compact, digest-checked encoding for the per-item outcome vectors a capability probe reports.

K-032 exists because of a collision nobody had named: a probe result's paired tests are computed from `outcomes`,
one row per item, and a multi-model result carries 800 rows per phenomenon per model — 586 KB for the ten models
E2626 scored. The repository's hygiene bound refuses a tracked file above 1 MiB, so the extractor stubbed the array
and kept the tables. A reader could check the arithmetic that was *reported* and could not recompute the test, let
alone run a new pair on models an old run already scored. E2629 found that by being refused.

The rows are bulky because they say the same thing twice: the item identity is a string a reader needs once and the
verdict is one bit. This module keeps the identity once per item set and stores one bit per item per model, so the
whole array survives the trip into a file two orders of magnitude smaller. The encoding is not trusted: a block
carries the SHA-256 of the canonical JSON of the array it replaced, taken before the array left the file, and
:func:`decode_outcomes` recomputes it. A block that does not rebuild its own digest is a failure, not a note — which
is what lets a record cite a compact extract and still be checkable by someone who disagrees with the reported number.

Two shapes are recognised, because that is what the probe runners write:

* ``rows`` — a flat list of ``[identifier, phenomenon, 1.0-or-0.0]``, which is what
  :func:`chakaso.training.minimal_pairs.score_pairs` reports.
* ``groups`` — a mapping of group name to a list of ``[identifier, 1.0-or-0.0]``, which is what
  :func:`chakaso.training.continuation_items.score_items` and
  :func:`chakaso.training.context_continuations.score_context_items` report.

Anything else is left alone. Dependency-free, so the format is checked in the core CI tier.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from chakaso.training.errors import TrainingError

__all__ = [
    "OutcomeEncodingError",
    "compact_probe_result",
    "decode_outcomes",
    "encode_outcomes",
    "is_compacted",
    "outcome_layout",
    "restore_probe_result",
]

#: The document-level block that holds the item identities shared by every model in one probe result.
ITEMS_KEY = "outcome_items"

#: One bit per item, so a 800-item array of rows becomes a 800-character string.
_TRUE = "1"
_FALSE = "0"


class OutcomeEncodingError(TrainingError):
    """A compact outcome block does not rebuild the array it claims to have replaced."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _verdict(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool) and value in (0.0, 1.0)


def outcome_layout(value: Any) -> str | None:
    """``"rows"``, ``"groups"`` or ``None`` for a value that is per-item outcome data.

    Strict about the row shape and about the verdict being a float 0.0 or 1.0, because the encoding drops everything
    except those fields: a row that also carried a score would be silently shortened, and a mapping of aggregates
    would be mistaken for data.
    """
    if isinstance(value, list) and value:
        if all(
            isinstance(row, list)
            and len(row) == 3
            and isinstance(row[0], str)
            and isinstance(row[1], str)
            and _verdict(row[2])
            for row in value
        ):
            return "rows"
        return None
    if isinstance(value, Mapping) and value:
        if not all(
            isinstance(group, str)
            and isinstance(rows, list)
            and rows
            and all(
                isinstance(row, list)
                and len(row) == 2
                and isinstance(row[0], str)
                and _verdict(row[1])
                for row in rows
            )
            for group, rows in value.items()
        ):
            return None
        return "groups"
    return None


def _keys_and_bits(value: Any, layout: str) -> tuple[Any, dict[str, str]]:
    if layout == "rows":
        row_keys: list[list[Any]] = [[row[0], row[1]] for row in value]
        row_bits = "".join(_TRUE if row[2] == 1.0 else _FALSE for row in value)
        return row_keys, {"": row_bits}
    group_keys: dict[str, list[Any]] = {
        group: [row[0] for row in rows] for group, rows in value.items()
    }
    group_bits: dict[str, str] = {
        group: "".join(_TRUE if row[1] == 1.0 else _FALSE for row in rows)
        for group, rows in value.items()
    }
    return group_keys, group_bits


def _rebuild(block: Mapping[str, Any], keys: Any) -> Any:
    layout = block["layout"]
    vectors = block["vectors"]
    if layout == "rows":
        bits = _bit_string(vectors, "", keys)
        return [
            [key[0], key[1], 1.0 if bit == _TRUE else 0.0]
            for key, bit in zip(keys, bits, strict=True)
        ]
    return {
        group: [
            [identifier, 1.0 if bit == _TRUE else 0.0]
            for identifier, bit in zip(
                keys[group], _bit_string(vectors, group, keys[group]), strict=True
            )
        ]
        for group in keys
    }


def _bit_string(vectors: Mapping[str, str], group: str, keys: Sequence[Any]) -> str:
    bits = vectors.get(group)
    if bits is None:
        message = f"the compact block carries no vector for {group!r}"
        raise OutcomeEncodingError(message)
    if len(bits) != len(keys):
        message = (
            f"the compact vector for {group!r} has {len(bits)} bits against {len(keys)} items; a paired test "
            "needs one verdict per item"
        )
        raise OutcomeEncodingError(message)
    if any(bit not in (_TRUE, _FALSE) for bit in bits):
        message = f"the compact vector for {group!r} holds a character that is not a verdict"
        raise OutcomeEncodingError(message)
    return bits


def encode_outcomes(value: Any, *, name: str = "") -> dict[str, Any]:
    """The ``compact_v1`` block for one outcome array, with its own rebuild proof.

    Encoded, then decoded, then digested: the digest recorded is the one taken on the value that came *in*, so a
    round-trip failure is caught here rather than by a reader months later. ``name`` labels the model in the error,
    because the only useful thing a refusal can say is which model's file is not what it claimed.
    """
    layout = outcome_layout(value)
    if layout is None:
        message = f"{name or 'this block'} is not per-item outcome data, so it cannot be compacted"
        raise OutcomeEncodingError(message)
    keys, bits = _keys_and_bits(value, layout)
    block: dict[str, Any] = {
        "compact_v1": {
            "layout": layout,
            "keys": keys,
            "vectors": bits,
            "canonical_json_sha256": sha256(_canonical(value).encode("utf-8")).hexdigest(),
        }
    }
    rebuilt = decode_outcomes(block)
    if _canonical(rebuilt) != _canonical(value):
        message = f"{name or 'a block'} did not survive the compact round trip; it is left as its own bytes"
        raise OutcomeEncodingError(message)
    return block


def decode_outcomes(block: Mapping[str, Any], *, shared: Any = None) -> Any:
    """Rebuild the outcome array a ``compact_v1`` block stands for, refusing anything the digest disagrees with.

    ``shared`` is the item listing hoisted to the document by :func:`compact_probe_result`; without it the block
    carries its own. The recorded digest is the authority: a block edited in place, or paired with a key listing for
    a different item set, fails here instead of quietly producing a McNemar count out of mismatched rows.
    """
    if not isinstance(block, Mapping) or "compact_v1" not in block:
        message = "not a compact_v1 block"
        raise OutcomeEncodingError(message)
    inner = block["compact_v1"]
    keys = shared if shared is not None else inner.get("keys")
    if keys is None:
        message = "the compact block names shared keys it does not have"
        raise OutcomeEncodingError(message)
    rebuilt = _rebuild(inner, keys)
    recorded = str(inner["canonical_json_sha256"])
    computed = sha256(_canonical(rebuilt).encode("utf-8")).hexdigest()
    if recorded != computed:
        message = (
            f"the rebuilt block's canonical JSON digests to {computed}, the extract recorded {recorded}; the "
            "compact block is not the array it replaced"
        )
        raise OutcomeEncodingError(message)
    return rebuilt


def _vectors_in(node: Any, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], Any]]:
    """Every outcome array under ``node``, with its path, walking no deeper than a model's result block goes."""
    found: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = (*path, str(key))
            layout = outcome_layout(value)
            if layout is not None:
                found.append((child, value))
            elif isinstance(value, Mapping):
                found.extend(_vectors_in(value, child))
    return found


def compact_probe_result(document: Mapping[str, Any]) -> dict[str, Any]:
    """The document with each model's per-item outcome arrays replaced by compact blocks, and their item identities
    hoisted to one shared listing where every model was asked the same items in the same order.

    The hoist is what makes the size win real: a probe result scores many models on one item set, and the identity
    half of each row is identical for all of them. Where the listings differ, each block keeps its own keys and the
    array is still compacted. A document with no ``models`` block — a run report, an analysis result — comes back
    untouched, because this function's subject is a probe result and pretending otherwise would silently rewrite
    something else's evidence.
    """
    body: dict[str, Any] = json.loads(json.dumps(document))
    models = body.get("models")
    if not isinstance(models, dict) or not models:
        return body

    listings: dict[str, list[Any]] = {}
    per_model: dict[str, dict[str, Any]] = {}
    for name, entry in models.items():
        if not isinstance(entry, dict):
            continue
        blocks: dict[str, Any] = {}
        for path, value in _vectors_in(entry, ()):
            dotted = ".".join(path)
            block = encode_outcomes(value, name=str(name))
            blocks[dotted] = block
            listings.setdefault(dotted, []).append(block["compact_v1"]["keys"])
            _place(entry, path, block)
        per_model[str(name)] = blocks

    shared: dict[str, Any] = {}
    for dotted, sets in listings.items():
        if len(sets) == len(per_model) and all(
            _canonical(keys) == _canonical(sets[0]) for keys in sets
        ):
            shared[dotted] = sets[0]
    if not shared:
        return body
    body[ITEMS_KEY] = shared
    for blocks in per_model.values():
        for dotted, block in blocks.items():
            if dotted in shared:
                inner: dict[str, Any] = block["compact_v1"]
                del inner["keys"]
                inner["keys_shared"] = f"{ITEMS_KEY}.{dotted}"
    return body


def _place(entry: dict[str, Any], path: Sequence[str], block: Mapping[str, Any]) -> None:
    node: Any = entry
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = block


def _blocks_in(node: Any, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], Mapping[str, Any]]]:
    """Every ``compact_v1`` block under ``node``, with its path — what a reader of a committed extract looks for."""
    found: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = (*path, str(key))
            if isinstance(value, Mapping) and "compact_v1" in value:
                found.append((child, value))
            elif isinstance(value, Mapping):
                found.extend(_blocks_in(value, child))
    return found


def is_compacted(document: Mapping[str, Any]) -> bool:
    """Whether a probe result stands in for its outcome arrays with compact blocks.

    A reader that accepts either form needs to know which it is holding before it asks for per-item rows, which is
    the point K-032 turned from an argument into a file format: an extract written before the fix has the arrays
    stubbed away and cannot be re-paired, one written after it can, and the difference is visible in the bytes.
    """
    models = document.get("models") if isinstance(document, Mapping) else None
    if not isinstance(models, Mapping):
        return False
    return any(_blocks_in(entry, ()) for entry in models.values() if isinstance(entry, Mapping))


def restore_probe_result(document: Mapping[str, Any]) -> dict[str, Any]:
    """The document with every compact block rebuilt into the array it stands for, so a paired test can run on it.

    The exact inverse of :func:`compact_probe_result` on the files it writes, and it refuses by way of
    :func:`decode_outcomes` if a block and the shared listing do not describe the same items.
    """
    body: dict[str, Any] = json.loads(json.dumps(document))
    models = body.get("models")
    if not isinstance(models, dict):
        return body
    listings: dict[str, Any] = body.get(ITEMS_KEY) or {}
    for entry in models.values():
        if not isinstance(entry, dict):
            continue
        for path, block in _blocks_in(entry, ()):
            inner: Mapping[str, Any] = block["compact_v1"]
            dotted = ".".join(path)
            shared = listings.get(dotted) if "keys_shared" in inner else None
            _place(entry, path, decode_outcomes(block, shared=shared))
    # The shared listing has been consumed into the arrays it describes. Leaving it in would give a reader a block
    # that is neither the compacted file's nor the original run's, and "the restored document is the document the
    # scorer wrote" is the property a reader of a record wants to be able to assert.
    body.pop(ITEMS_KEY, None)
    return body
