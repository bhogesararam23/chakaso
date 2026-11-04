"""Compare experiment arms from the reports on disk, and prove they were matched before reading a difference.

A two- or three-arm record (E2612, E2613) says its arms differ in exactly one declared way. That sentence is
only worth its paper if something checks it, and until now the check lived in an ad hoc script whose output was
committed and whose code was not. This module is that check as a function of reports: it lists every condition
that must be equal and whether it is, lists the fields that are *allowed* to differ and how they do, and only then
tabulates bits per character per partition against a named reference arm.

It reads reports and nothing else, so a reader can rebuild a record's comparison from the artifacts the record
cites. It has no opinion about which difference is large: the materiality band belongs to the record, and this
module reports numbers and signs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chakaso.training.errors import TrainingError

__all__ = ["ArmComparisonError", "compare_arms"]


class ArmComparisonError(TrainingError):
    """The arms cannot be compared: a reference is missing, a report lacks a field, or an aggregate is ill-posed."""


#: Conditions that must be identical across arms for a loss difference to be a result and not a protocol change.
_MATCHED: dict[str, tuple[str, ...]] = {
    "corpus_fingerprint": ("corpus", "fingerprint"),
    "split_fingerprint": ("corpus", "split_fingerprint"),
    "split_seed": ("corpus", "split_seed"),
    "tokenizer_fingerprint": ("tokenizer", "fingerprint"),
    "tokenizer_size": ("tokenizer", "size"),
    "parameter_count": ("transformer", "parameter_count"),
    "architecture": ("transformer", "architecture"),
    "seed": ("transformer", "budget", "seed"),
    "max_steps": ("transformer", "budget", "max_steps"),
    "schedule": ("transformer", "budget", "schedule"),
    "effective_batch_size": ("transformer", "budget", "effective_batch_size"),
    "learning_rate": ("transformer", "budget", "learning_rate"),
    "test_positions_scored": ("transformer", "test", "positions_scored"),
    "test_characters_scored": ("transformer", "test", "characters_scored"),
}

#: Fields an arm is allowed to differ in. A difference here is the experiment; a difference anywhere in
#: ``_MATCHED`` is a defect, and any other difference is simply not looked at.
_DECLARABLE: dict[str, tuple[str, ...]] = {
    "training_group_prefixes": ("transformer", "budget", "training_group_prefixes"),
    "max_training_tokens": ("transformer", "budget", "max_training_tokens"),
    "shuffle_training_units": ("transformer", "budget", "shuffle_training_units"),
    "share_group_prefix": ("transformer", "budget", "share_group_prefix"),
    "share_of_tokens": ("transformer", "budget", "share_of_tokens"),
    "training_composition": ("corpus", "training_composition"),
    "document_masking": ("transformer", "budget", "document_masking"),
    "grad_accum_steps": ("transformer", "budget", "grad_accum_steps"),
    "training_tokens_packed": ("corpus", "training_tokens_packed"),
    # ADR-0049's option, and the experiment E2631 asks about. Until it was listed here a comparison of two arms
    # that differed in what decay reached would have reported them as matched, because the field sat in neither
    # table and "any other difference is simply not looked at" is what the comment above this one promises.
    "decay_applied_to": ("transformer", "budget", "optimizer", "decay_applied_to"),
}

#: Declarable fields a report only carries when the run used the feature. A report written before the feature
#: existed (E2612 to E2614) has no such key, and "absent" is a real value there: the arm was not composed by
#: share, or the optimizer recipe was not declared. Every other field is required, because a missing required
#: field is a report the comparison cannot trust.
_OPTIONAL_DECLARABLE = frozenset(
    {"share_group_prefix", "share_of_tokens", "training_composition", "decay_applied_to"}
)


def compare_arms(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    reference: str,
    aggregate: Mapping[str, Sequence[str]] | None = None,
    equal_weight_aggregates: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Tabulate ``arms`` against ``reference`` after checking they were matched.

    Args:
        arms: Arm name to its report.
        reference: The arm every difference is measured against (difference = arm minus reference, so a negative
            number means the arm scored fewer bits per character).
        aggregate: Optional named groups of partition prefixes. Each is weighted by **scored characters**
            (``metrics.characters_scored``), the denominator the metric is defined over — not by the partition's
            population, which is 23 times larger in E2612 and moves the effect size by up to 4.5 times the materiality band
            (K-027).
        equal_weight_aggregates: Named groups averaged with **equal weight over the partitions**, which is the
            objective the records call O1. It is a separate argument rather than a flag because the two quantities
            differ by more than the margins they are read against, and a caller should choose which one it is asking
            for (E2631's deviation 1).

    Raises:
        ArmComparisonError: fewer than two arms, an unknown reference, a report missing a field the comparison
            reads, or an aggregate over a partition that is absent or that an arm did not score.
    """
    if len(arms) < 2:
        message = "an arm comparison needs at least two arms"
        raise ArmComparisonError(message)
    if reference not in arms:
        message = f"reference {reference!r} is not one of the arms {sorted(arms)}"
        raise ArmComparisonError(message)

    matched = {name: _across(arms, path, name=name) for name, path in _MATCHED.items()}
    matched["per_partition_tokens_and_characters"] = _partition_support(arms)
    differences = {
        name: _across(arms, path, name=name, optional=name in _OPTIONAL_DECLARABLE)
        for name, path in _DECLARABLE.items()
    }
    declared = {name: block for name, block in differences.items() if not block["equal"]}

    partitions = _partition_table(arms, reference)
    result: dict[str, Any] = {
        "reference": reference,
        "arms": sorted(arms),
        "matched_conditions": matched,
        "all_conditions_matched": all(block["equal"] for block in matched.values()),
        "declared_differences": declared,
        "whole_pack_bits_per_character": _metric_row(
            arms, reference, ("transformer", "test", "bits_per_character")
        ),
        "partitions": partitions,
    }
    if aggregate:
        result["aggregates"] = {
            group: _aggregate(arms, reference, group, prefixes, weighting="characters")
            for group, prefixes in aggregate.items()
        }
    if equal_weight_aggregates:
        result["equal_weight_aggregates"] = {
            group: _aggregate(arms, reference, group, prefixes, weighting="equal")
            for group, prefixes in equal_weight_aggregates.items()
        }
    return result


def _dig(
    report: Mapping[str, Any], path: Sequence[str], arm: str, *, optional: bool = False
) -> Any:
    current: Any = report
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            if optional:
                return None
            message = f"arm {arm!r} has no {'.'.join(path)} in its report"
            raise ArmComparisonError(message)
        current = current[key]
    return current


def _across(
    arms: Mapping[str, Mapping[str, Any]],
    path: Sequence[str],
    *,
    name: str,
    optional: bool = False,
) -> dict[str, Any]:
    values = {arm: _dig(report, path, arm, optional=optional) for arm, report in arms.items()}
    first = next(iter(values.values()))
    return {
        "values": values,
        "equal": all(value == first for value in values.values()),
        "field": name,
    }


def _partition_blocks(
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Partition prefix to arm to its report block, for every prefix any arm declares."""
    table: dict[str, dict[str, Mapping[str, Any]]] = {}
    for arm, report in arms.items():
        declared = _dig(report, ("transformer", "test_partitions"), arm)
        for prefix, block in declared.items():
            table.setdefault(prefix, {})[arm] = block
    return table


def _partition_support(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Whether every arm scored the same tokens and characters on each partition.

    Equal support is what lets two bits-per-character numbers be subtracted: they then come from the same
    held-out text, not from two different samples of it.
    """
    support: dict[str, dict[str, tuple[Any, Any]]] = {}
    for prefix, blocks in _partition_blocks(arms).items():
        support[prefix] = {
            arm: (block.get("tokens_scored"), block.get("metrics", {}).get("characters_scored"))
            for arm, block in blocks.items()
        }
    equal = all(
        len(set(per_arm.values())) == 1 and len(per_arm) == len(arms)
        for per_arm in support.values()
    )
    return {"values": support, "equal": equal, "field": "per_partition_tokens_and_characters"}


def _metric_row(
    arms: Mapping[str, Mapping[str, Any]], reference: str, path: Sequence[str]
) -> dict[str, Any]:
    values = {arm: float(_dig(report, path, arm)) for arm, report in arms.items()}
    return {
        "values": values,
        "difference_from_reference": {
            arm: round(value - values[reference], 6)
            for arm, value in values.items()
            if arm != reference
        },
    }


def _partition_table(arms: Mapping[str, Mapping[str, Any]], reference: str) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for prefix, blocks in _partition_blocks(arms).items():
        scored = {
            arm: float(block["metrics"]["bits_per_character"])
            for arm, block in blocks.items()
            if "metrics" in block
        }
        row: dict[str, Any] = {"values": scored, "scored_by": sorted(scored)}
        if reference in scored:
            row["difference_from_reference"] = {
                arm: round(value - scored[reference], 6)
                for arm, value in scored.items()
                if arm != reference
            }
        table[prefix] = row
    return table


def _aggregate(
    arms: Mapping[str, Mapping[str, Any]],
    reference: str,
    group: str,
    prefixes: Sequence[str],
    *,
    weighting: str,
) -> dict[str, Any]:
    """A mean of the partition bits-per-character over ``prefixes``, per arm, weighted as ``weighting`` says.

    Two weightings exist because the campaign has been misled by the difference. K-027 found that weighting a
    five-register aggregate by each partition's *population* moved an effect size by up to 4.5 times the materiality
    band, so the character-weighted reading was added as the honest pooled number. And **O1** — the objective six
    records declare as primary — is the *equal-weight* mean, which the character-weighted reading is not: E2631
    preregistered `--aggregate o1=` and got 2.049605 against the objective's 2.021357, because prose and science carry
    the characters and medicine carries a fifth of O1's weight. Both are reported here, each named for what it is.
    """
    if not prefixes:
        message = f"aggregate {group!r} names no partitions"
        raise ArmComparisonError(message)
    values: dict[str, float] = {}
    for arm, report in arms.items():
        blocks = _dig(report, ("transformer", "test_partitions"), arm)
        weight = total = 0.0
        for prefix in prefixes:
            block = blocks.get(prefix)
            if block is None or "metrics" not in block:
                message = f"aggregate {group!r}: arm {arm!r} did not score partition {prefix!r}"
                raise ArmComparisonError(message)
            if weighting == "equal":
                weight += 1.0
                total += float(block["metrics"]["bits_per_character"])
            else:
                characters = float(block["metrics"]["characters_scored"])
                weight += characters
                total += characters * float(block["metrics"]["bits_per_character"])
        values[arm] = round(total / weight, 6)
    return {
        "partitions": list(prefixes),
        "weighting": (
            "equal weight over the named partitions (the O1 objective)"
            if weighting == "equal"
            else "metrics.characters_scored"
        ),
        "values": values,
        "difference_from_reference": {
            arm: round(value - values[reference], 6)
            for arm, value in values.items()
            if arm != reference
        },
    }
