"""A paired test on held-out loss itself, resampled by work rather than by row.

The project's comparisons between two trained arms have been statements about a *band*: E2614 measured the
seed-to-seed spread of the reference configuration and the campaign then judged every arm difference against 0.02 bits
per character. That is a threshold, not a test. It says a difference is larger than the noise of one machine's
initialisations, and it says nothing about how well the difference on *this held-out text* is known — which is the
question E2629 left open, because its probes could not separate the arms at all and the loss could only be compared
to a band.

Here the unit of comparison is a held-out document, and both arms are scored on the same document, so the difference
is paired and its spread across documents is the thing a claim has to clear. Documents are not independent — a book
contributes many, and a prose style repeats itself — so the bootstrap resamples **works** (the split's groups), which
is the unit the whole-work split holds out and therefore the unit a reader should treat as one observation.

Two numbers come out and they mean different things. ``mean_difference`` is the pooled bits-per-character difference:
total nats over total characters across the documents, which is the same arithmetic the run reports for its whole
pack. ``interval`` is the cluster bootstrap on that quantity. ``sign_test_p_value`` counts documents one way or the
other and assumes they are independent, so it is the *optimistic* of the two readings and is reported beside the
interval rather than instead of it.

Standard library only, so the test itself runs in the dependency-free CI matrix; the scoring that produces its inputs
needs ``[nn]`` and lives in :mod:`chakaso.training.nn_evaluation`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from chakaso.training.errors import TrainingError
from chakaso.training.paired_stats import sign_test_p_value

__all__ = ["PairedLossError", "paired_document_test"]


class PairedLossError(TrainingError):
    """A paired loss comparison was asked about documents that do not line up."""


def _validate(differences: Sequence[float], weights: Sequence[float]) -> None:
    if not differences:
        message = (
            "a paired loss test needs at least one document; over nothing there is no difference"
        )
        raise PairedLossError(message)
    if len(differences) != len(weights):
        message = (
            f"the two arms scored {len(differences)} documents but the caller supplied {len(weights)} "
            "weights; a paired test needs one weight per document"
        )
        raise PairedLossError(message)
    if any(not math.isfinite(value) for value in differences) or any(
        not math.isfinite(value) for value in weights
    ):
        message = "a paired loss test needs finite differences and weights"
        raise PairedLossError(message)
    if any(weight <= 0.0 for weight in weights):
        message = "every document's weight must be positive: a zero-weight document is not measured"
        raise PairedLossError(message)


def _clusters(
    groups: Sequence[Mapping[str, object]] | Sequence[object], count: int
) -> dict[str, list[int]]:
    """Group document indices by their cluster key, keeping first-appearance order for reproducible draws."""
    if len(groups) != count:
        message = f"{count} documents and {len(groups)} cluster keys do not describe the same set"
        raise PairedLossError(message)
    out: dict[str, list[int]] = {}
    for index, key in enumerate(groups):
        out.setdefault(str(key), []).append(index)
    return out


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """A linear-interpolated quantile of an already-sorted sequence."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def paired_document_test(
    differences: Sequence[float],
    weights: Sequence[float],
    groups: Sequence[object],
    *,
    seed: int,
    resamples: int = 1000,
    probability: float = 0.95,
) -> dict[str, object]:
    """Compare two arms document by document, and resample their works to say how well the difference is known.

    ``differences`` is one entry per held-out document — first arm's bits per character minus the second's, so a
    positive value means the first arm is worse on that document. ``weights`` is what each document contributed to
    the pooled metric (the campaign's declared weighting is scored characters), and ``groups`` names each document's
    work, the unit the whole-work split holds out. ``seed`` and ``resamples`` are reported in the result because a
    bootstrap is only reproducible if its parameters travel with it.

    Raises:
        PairedLossError: the three sequences disagree in length or are empty, a difference or weight is not finite,
            a weight is not positive, or the resample count and confidence are not usable.
    """
    numeric_differences = [float(value) for value in differences]
    numeric_weights = [float(value) for value in weights]
    _validate(numeric_differences, numeric_weights)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        message = f"resamples must be an integer >= 1, got {resamples!r}"
        raise PairedLossError(message)
    if not 0.0 < probability < 1.0:
        message = f"probability must lie strictly between 0 and 1, got {probability!r}"
        raise PairedLossError(message)

    weighted = [
        value * weight for value, weight in zip(numeric_differences, numeric_weights, strict=True)
    ]
    total_weight = math.fsum(numeric_weights)
    mean = math.fsum(weighted) / total_weight
    by_cluster = _clusters(groups, len(numeric_differences))
    keys = list(by_cluster)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        picked = rng.choices(keys, k=len(keys))
        numerator = math.fsum(
            value for key in picked for index in by_cluster[key] for value in [weighted[index]]
        )
        denominator = math.fsum(
            numeric_weights[index] for key in picked for index in by_cluster[key]
        )
        draws.append(numerator / denominator)
    draws.sort()
    tail = (1.0 - probability) / 2.0
    interval = [_percentile(draws, tail), _percentile(draws, 1.0 - tail)]

    positive = sum(1 for value in numeric_differences if value > 0.0)
    return {
        "unit": "bits per character, first minus second",
        "mean_difference": mean,
        "interval": interval,
        "probability": probability,
        "excludes_zero": interval[0] > 0.0 or interval[1] < 0.0,
        "documents": len(numeric_differences),
        "clusters": len(by_cluster),
        "cluster_means": {
            key: math.fsum(weighted[index] for index in members)
            / math.fsum(numeric_weights[index] for index in members)
            for key, members in by_cluster.items()
        },
        "total_weight": total_weight,
        "positive_documents": positive,
        "negative_documents": sum(1 for value in numeric_differences if value < 0.0),
        "sign_test_p_value": sign_test_p_value(positive, total=len(numeric_differences)),
        "sign_test_assumption": "documents independent; the interval above is the reading that resamples works",
        "bootstrap": {"seed": seed, "resamples": resamples},
    }
