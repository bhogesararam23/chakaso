"""The unit a probe item's verdict should be counted as one observation of.

E2633 showed that a held-out loss difference resampled by *work* rather than by document moves enough to cross zero,
and E2634 measured the same dependence from the other side: its separations spread over 24 and 26 works, a number it
had to obtain by rebuilding the item set from a 149 MB snapshot because the per-item rows name only a register and a
hash (K-035). A clustering claim should fall out of the artifact, not cost a second command over the corpus.

This module carries that identity: one listing per probe result, pairing each item identifier with the cluster it
belongs to, with a digest of its own so a reader is not asked to trust the pairing. Two cluster units are named,
because the two probe families differ in kind:

* :data:`WORK_CLUSTER_UNIT` — the same-region and v1 continuation items are cut from held-out documents, so the unit
  a reader should treat as one observation is the **work the split held out** (``TextUnit.group``).
* :data:`PHENOMENON_CLUSTER_UNIT` — the grammatical minimal pairs are authored, not cut. No corpus work stands behind
  them; the largest shared cause a pair has is the **phenomenon** whose template it follows, and that is what the
  listing names. A clustering claim over authored pairs is a claim about templates, and this module says so in the
  artifact rather than in prose.

The listing deliberately holds one entry per *item*, in item order, and not per reading: every reading of one probe
result asks the same items in the same order, so a per-reading copy would be the same bytes several times over.

Dependency-free, like the rest of the instrument layer, so the format is checked in the core CI tier.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chakaso.core.hashing import sha256_hex
from chakaso.training.errors import TrainingError

__all__ = [
    "CLUSTER_BLOCK_SCHEMA",
    "PHENOMENON_CLUSTER_UNIT",
    "WORK_CLUSTER_UNIT",
    "ClusterListing",
    "ClusterListingError",
    "cluster_block",
    "keys_for_items",
    "load_cluster_block",
]

#: The key a probe result uses for its cluster listing, beside the item count and the item digest.
CLUSTER_BLOCK_SCHEMA = "chakaso-probe-clusters/1"

#: The items of a corpus-cut probe cluster by the work the whole-work split held out.
WORK_CLUSTER_UNIT = "held-out work"

#: The items of the authored pair probe cluster by the phenomenon whose template they follow.
PHENOMENON_CLUSTER_UNIT = "authored phenomenon"

_CLUSTER_UNITS = (WORK_CLUSTER_UNIT, PHENOMENON_CLUSTER_UNIT)


class ClusterListingError(TrainingError):
    """A probe's cluster listing does not describe the item set it is paired with."""


@dataclass(frozen=True, slots=True)
class ClusterListing:
    """``unit`` names what a cluster means here, and ``rows`` is one ``(item identifier, cluster)`` per item.

    Order is part of the identity: the rows are in the order the probe scored its items, so the digest binds both the
    pairing and the sequence, and a listing shuffled against the outcome vectors does not rebuild it.
    """

    unit: str
    rows: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.unit not in _CLUSTER_UNITS:
            message = (
                f"cluster unit {self.unit!r} is not one of {list(_CLUSTER_UNITS)}; a reader of the artifact "
                "needs to know what a cluster means before trusting a clustering claim"
            )
            raise ClusterListingError(message)
        if not self.rows:
            message = "a cluster listing over zero items is not a measurement"
            raise ClusterListingError(message)
        identifiers = [identifier for identifier, _ in self.rows]
        if any(not identifier.strip() or not cluster.strip() for identifier, cluster in self.rows):
            message = "a cluster listing entry needs both an item identifier and a cluster name"
            raise ClusterListingError(message)
        repeated = sorted({name for name in identifiers if identifiers.count(name) > 1})
        if repeated:
            message = f"the cluster listing names an item twice: {repeated}"
            raise ClusterListingError(message)

    @property
    def digest(self) -> str:
        """SHA-256 over the rows in order — the identity a record cites next to the item digest."""
        return sha256_hex(
            json.dumps([list(row) for row in self.rows], separators=(",", ":"), ensure_ascii=True)
        )

    @property
    def clusters(self) -> tuple[str, ...]:
        """The distinct cluster names, in order of first appearance."""
        seen: list[str] = []
        for _, cluster in self.rows:
            if cluster not in seen:
                seen.append(cluster)
        return tuple(seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CLUSTER_BLOCK_SCHEMA,
            "unit": self.unit,
            "items": len(self.rows),
            "clusters": len(self.clusters),
            "digest": self.digest,
            "rows": [[identifier, cluster] for identifier, cluster in self.rows],
        }


def cluster_block(items: Sequence[Any], *, unit: str) -> dict[str, Any]:
    """The serialized listing for built probe items, read off each item's ``identifier`` and ``work``.

    Takes the items rather than the pairs because the builders already know both halves; a runner that re-derived the
    work from a document identifier would be a second, drifting copy of the rule that made it.

    Raises:
        ClusterListingError: an item carries no work, or the unit is not one this module names.
    """
    rows: list[tuple[str, str]] = []
    for item in items:
        identifier = str(getattr(item, "identifier", ""))
        work = str(getattr(item, "work", ""))
        if not work.strip():
            message = (
                f"item {identifier!r} carries no work, so the probe cannot state which cluster its verdict "
                "belongs to"
            )
            raise ClusterListingError(message)
        rows.append((identifier, work))
    return ClusterListing(unit=unit, rows=tuple(rows)).as_dict()


def load_cluster_block(block: Any) -> ClusterListing:
    """The listing a probe result stored, refusing a block whose own digest does not recompute.

    Raises:
        ClusterListingError: the block is not a cluster listing, names a schema this code does not know, or its
            stored digest disagrees with its rows.
    """
    if not isinstance(block, dict):
        message = "a probe result's cluster listing must be a block, not a bare list"
        raise ClusterListingError(message)
    if block.get("schema_version") != CLUSTER_BLOCK_SCHEMA:
        message = (
            f"the cluster block declares schema {block.get('schema_version')!r}, not "
            f"{CLUSTER_BLOCK_SCHEMA!r}"
        )
        raise ClusterListingError(message)
    rows = block.get("rows")
    if not isinstance(rows, list) or not rows:
        message = "the cluster block carries no rows, so it names no pairing"
        raise ClusterListingError(message)
    if any(not isinstance(row, list) or len(row) != 2 for row in rows):
        message = "every cluster row must be an item identifier and its cluster, in that order"
        raise ClusterListingError(message)
    listing = ClusterListing(
        unit=str(block.get("unit")),
        rows=tuple((str(row[0]), str(row[1])) for row in rows),
    )
    if str(block.get("digest")) != listing.digest:
        message = (
            f"the cluster block stores digest {block.get('digest')!r} but its rows compute {listing.digest}; "
            "the pairing of items to clusters is not the one that was written"
        )
        raise ClusterListingError(message)
    if block.get("items") != len(listing.rows):
        message = (
            f"the cluster block states its item count as {block.get('items')!r} and carries "
            f"{len(listing.rows)} rows"
        )
        raise ClusterListingError(message)
    if block.get("clusters") != len(listing.clusters):
        message = (
            f"the cluster block states its cluster count as {block.get('clusters')!r} and carries "
            f"{len(listing.clusters)}"
        )
        raise ClusterListingError(message)
    return listing


def keys_for_items(listing: ClusterListing, identifiers: Sequence[str]) -> list[str]:
    """The cluster of each item in ``identifiers``, in that order, refusing an item the listing does not name.

    This is the alignment a paired statistic needs: two models' per-item verdict vectors plus their cluster keys, one
    key per verdict. An item scored but not listed, or listed but not scored, is a refusal rather than a dropped row —
    dropping it would quietly change the item set the record declared.

    Raises:
        ClusterListingError: an identifier appears twice, or one is absent from the listing.
    """
    by_identifier = dict(listing.rows)
    seen: set[str] = set()
    keys: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            message = (
                f"the outcome vectors name item {identifier!r} twice; a paired row is one item"
            )
            raise ClusterListingError(message)
        seen.add(identifier)
        cluster = by_identifier.get(identifier)
        if cluster is None:
            message = (
                f"item {identifier!r} was scored but the probe result's cluster listing does not name it, so "
                "its verdict cannot be attributed to a work"
            )
            raise ClusterListingError(message)
        keys.append(cluster)
    if len(seen) != len(listing.rows):
        message = (
            f"the listing names {len(listing.rows)} items and the outcomes carry {len(seen)}; a paired test "
            "needs the same item set on both sides"
        )
        raise ClusterListingError(message)
    return keys
