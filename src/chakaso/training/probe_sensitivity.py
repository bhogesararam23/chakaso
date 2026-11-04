"""What an instrument can see: scoring a capability probe against an ordering the harness already measured.

The repository has two capability probes and one honest complaint about both of them — they have only ever been
compared at one endpoint, a trained arm against a random initialisation (E2621, E2622, E2623). Whether either
one moves when the model demonstrably learns more is a separate question, and it has a separate answer: E2612
recorded held-out loss and surface repetition moving in opposite directions, and the campaign's summary says
plainly that no instrument here is *known* to detect a real improvement.

This module answers it for any set of models with a numeric criterion and per-item probe outcomes. It is
deliberately not tied to a training curve: rungs of one curve, five seeds of one arm, or two mixtures are all
the same shape — models, one measured quality criterion per model, and one score per item per model.

Everything is arithmetic over dicts, dependency-free, so it runs in the core matrix and a record can cite the
function that produced its verdict.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chakaso.training.errors import TrainingError
from chakaso.training.outcome_rules import OutcomeVocabulary, decide
from chakaso.training.paired_stats import holm_rejections, mcnemar_exact, spearman_rho

__all__ = ["VERDICT_FACT_NAMES", "compare_instruments", "ordered_models"]

#: The facts an instrument contributes to an outcome vocabulary, and the only names a record's instrument-level
#: declaration may reference (K-034). The verdict letters themselves are not here: E2626 declared `S-a`…`S-d` in its
#: own record, with its own 0.9 coverage and 0.9 tracking bounds, and a shared module that prints those letters for
#: every record is a module that reports another record's outcome space. A vocabulary file names these and decides.
VERDICT_FACT_NAMES = (
    "rho",
    "material_pairs",
    "material_pairs_separated",
    "material_coverage",
    "has_material_pairs",
    "contradicted_pairs",
)


class SensitivityError(TrainingError):
    """The models, the criterion or the probe outcomes handed to the comparison do not describe one analysis."""


def ordered_models(
    criterion: Mapping[str, float], instruments: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> tuple[str, ...]:
    """The model names ordered by the criterion, best (lowest) first, refusing a partial table.

    Every instrument must have scored every model and every model must carry a criterion value; a comparison
    that silently drops the models one instrument missed would report a family size that is not the one tested.
    Ties in the criterion are broken by name, so the ordering of a table with two equal numbers is the same in
    every run and every pair in it gets a direction of "none" rather than an arbitrary one.
    """
    names = sorted(criterion, key=lambda name: (float(criterion[name]), name))
    for instrument, scores in instruments.items():
        missing = sorted(set(names) - set(scores))
        extra = sorted(set(scores) - set(names))
        if missing or extra:
            message = (
                f"instrument {instrument!r} scored {sorted(scores)}, but the criterion names {names}; "
                "a rung missing from one of them is not the same analysis"
            )
            raise SensitivityError(message)
        for name, per_item in scores.items():
            if not per_item:
                message = f"instrument {instrument!r} has no per-item outcomes for {name!r}"
                raise SensitivityError(message)
    if len(names) < 2:
        message = f"a sensitivity comparison needs at least two models, got {names}"
        raise SensitivityError(message)
    return tuple(names)


def _item_values(per_item: Mapping[str, float], order: Sequence[str]) -> list[float]:
    """One model's per-item values in a shared item order, so two vectors are the same questions."""
    return [per_item[name] for name in order]


def _pair_row(
    first: str,
    second: str,
    criterion: Mapping[str, float],
    instruments: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    materiality: float,
) -> dict[str, Any]:
    delta = criterion[first] - criterion[second]
    row: dict[str, Any] = {
        "first": first,
        "second": second,
        # Negative when `first` is the better model, because the criterion is a loss and lower is better. The
        # direction below is read off this sign, so the convention is what a reader of a pair row needs.
        "criterion_gap": delta,
        "material": abs(delta) > materiality,
    }
    for instrument, scores in instruments.items():
        items = next(iter(scores.values()))
        order = sorted(items)
        a = {name: _item_values(model, order) for name, model in scores.items()}
        b, c, n, p_value = mcnemar_exact(a[first], a[second])
        better = first if delta < 0 else second
        # Direction: the test found a difference; which model won more items decides whether it agrees with
        # the criterion. A gap of zero is not a direction and cannot be contradicted.
        if b > c:
            winner: str | None = first
        elif c > b:
            winner = second
        else:
            winner = None
        row[instrument] = {
            "discordant": [b, c],
            "n": n,
            "p_value": p_value,
            "separated": None,
            "direction": (
                "none" if winner is None else ("agrees" if winner == better else "contradicts")
            ),
        }
    return row


def compare_instruments(
    criterion: Mapping[str, float],
    instruments: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    materiality: float,
    alpha: float = 0.05,
    verdicts: OutcomeVocabulary | None = None,
) -> dict[str, Any]:
    """Judge each instrument against the criterion over every model pair.

    ``criterion`` maps a model name to a quality number where **lower is better** (bits per character, say).
    ``instruments`` maps an instrument name to ``{model: {item: value}}``, where a value is that model's outcome
    on that item — 1.0 a pass, 0.0 a fail, 0.5 a tie.

    A pair is **material** when the criterion separates its models by more than ``materiality``, the unit the
    caller's project uses to call two models different. Each pair is tested with McNemar's exact test, and the
    family of pairs within one instrument is corrected with Holm's step-down at ``alpha`` — a sweep of dozens of
    rung pairs is one family, not dozens of experiments.

    ``verdicts`` is the calling record's outcome vocabulary, and it is the only way a letter gets into the result.
    Without it every instrument block carries its statistics and `"verdict": None`, because a verdict is declared
    before a run and this module was not part of that declaration: see
    [`ADR-0052`](../../../docs/decisions/ADR-0052-an-analysis-prints-the-verdict-its-record-declared-or-none-at-all.md)
    and K-034. The numbers are the same either way; only the letter moves.

    The facts a vocabulary may read are :data:`VERDICT_FACT_NAMES`: ``rho``, ``material_pairs``,
    ``material_pairs_separated``, ``material_coverage`` (0.0 when there are no material pairs, which is what
    ``has_material_pairs`` distinguishes), ``has_material_pairs`` and ``contradicted_pairs``.

    Raises:
        SensitivityError: the models, the criterion or an instrument's outcomes do not describe one analysis, or
            ``materiality`` is not positive.
    """
    if materiality <= 0.0:
        message = f"materiality must be positive, got {materiality!r}"
        raise SensitivityError(message)
    names = ordered_models(criterion, instruments)
    order = names  # best to worst by the criterion
    rows: list[dict[str, Any]] = []
    for index, first in enumerate(order):
        for second in order[index + 1 :]:
            rows.append(_pair_row(first, second, criterion, instruments, materiality=materiality))

    outcome: dict[str, Any] = {}
    for instrument in instruments:
        p_values = [float(row[instrument]["p_value"]) for row in rows]
        rejections = holm_rejections(p_values, alpha=alpha)
        for row, rejected in zip(rows, rejections, strict=True):
            row[instrument]["separated"] = bool(rejected)
        material_rows = [row for row in rows if row["material"]]
        separated = [row for row in material_rows if row[instrument]["separated"]]
        contradicted = [
            row
            for row in rows
            if row[instrument]["separated"] and row[instrument]["direction"] == "contradicts"
        ]
        items = next(iter(instruments[instrument].values()))
        shared_order = sorted(items)
        # The criterion is "lower is better" and an accuracy is "higher is better", so a tracking instrument
        # anti-correlates with the raw number. Negating the criterion makes +1 mean "the instrument and the
        # quality measure put these models in the same order", which is what the threshold is about.
        rho = spearman_rho(
            [-criterion[name] for name in order],
            [
                _accuracy(_item_values(instruments[instrument][name], shared_order))
                for name in order
            ],
        )
        gaps_resolved = [
            abs(float(row["criterion_gap"]))
            for row in rows
            if row[instrument]["separated"] and abs(float(row["criterion_gap"])) > 0.0
        ]
        gaps_missed = [
            abs(float(row["criterion_gap"]))
            for row in material_rows
            if not row[instrument]["separated"]
        ]
        coverage = len(separated) / len(material_rows) if material_rows else None
        facts = {
            "rho": rho,
            "material_pairs": len(material_rows),
            "material_pairs_separated": len(separated),
            "material_coverage": coverage if coverage is not None else 0.0,
            "has_material_pairs": bool(material_rows),
            "contradicted_pairs": len(contradicted),
        }
        block: dict[str, Any] = {
            "verdict": None,
            "verdict_facts": facts,
            "rho": rho,
            "pairs": len(rows),
            "material_pairs": len(material_rows),
            "material_pairs_separated": len(separated),
            "material_coverage": coverage,
            "contradicted_pairs": [[row["first"], row["second"]] for row in contradicted],
            # The headline a reader wants: the smallest quality gap the instrument can see at all, and the
            # largest it cannot. When the second is bigger than the first, sensitivity is not a threshold.
            "smallest_separated_gap": min(gaps_resolved) if gaps_resolved else None,
            "largest_unseparated_material_gap": max(gaps_missed) if gaps_missed else None,
            "alpha": alpha,
        }
        if verdicts is not None:
            decision = decide(facts, verdicts)
            block["verdict"] = decision["letter"]
            block["verdict_reading"] = decision["reading"]
            block["verdict_vocabulary"] = decision["vocabulary"]
            block["verdict_precedence"] = decision["precedence"]
            block["verdict_also_matched"] = decision["also_matched"]
        outcome[instrument] = block
    return {
        "models": list(order),
        "materiality": materiality,
        "pairs": rows,
        "instruments": outcome,
    }


def _accuracy(values: Sequence[float]) -> float:
    total = sum(values)
    return total / len(values) if values else 0.0
