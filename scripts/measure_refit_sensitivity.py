"""How much does a byte-pair vocabulary move when only its *fitting sample* moves — and does the space-prefixed rule's advantage survive the move?

H17 is measured: the space-prefixed pre-tokenizer lowers held-out bits per character on five paired seeds on one host
(E2630, Y2-a), at equal updates and equal documents (E2625, X-a), and at a second seed on a second host (E2628, Y-a).
Every one of those readings is **one vocabulary**. The question an adoption decision cannot skip is whether the gain
belongs to the rule or to the lucky 12.7-million-character sample that 4,096 merges were learned from — and that is
answerable without training anything, because what a pre-tokenizer changes is how a text is cut and how many identifiers
it costs.

So this script fits each rule on several independent samples of the same corpus side and measures, on one fixed held-out
text that none of the fits saw:

* **characters per identifier** — the compression a position buys, which is the mechanism H17 predicts;
* **the ratio between the two rules on the same sample** — the thing that has to be stable for the loss gain not to be
  an artifact of one fit;
* **merge-set overlap and rank agreement between fits of one rule** — how much a BPE fit moves at all when its sample
  moves, stated separately for each rule because a rule that fragments text into more distinct pieces has more merge
  choices to disagree about;
* **the same readings against the committed vocabularies** when they are named, so the small-sample experiment is tied
  to the artifacts the records cite rather than to a private notion of "the default".

Nothing is trained and no loss number is claimed here. A compression advantage is the *mechanism*, and the records that
measured the loss are named in the output.

Usage:
    python scripts/measure_refit_sensitivity.py --snapshot data/corpora/gutenberg-multiregister-v1/…-snapshot.json \\
        --fit-seeds 7,11,13,17 --sample-characters 4000000 --identifiers 4096 --min-frequency 3 \\
        --eval-characters 1000000 --reference-vocabulary default=experiments/E2612/vocabulary-434debe82f949347.json \\
        --reference-vocabulary space=experiments/E2620/vocabulary-4a47f0281c81a9e5.json \\
        --out experiments/E2632/refit-sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.training.errors import CorpusError, TrainingError
from chakaso.training.manifest import read_snapshot
from chakaso.training.subword import (
    LEGACY_DEFAULT_PRE_TOKENIZER,
    SPACE_PREFIXED_PRE_TOKENIZER,
    SubwordVocabulary,
    train_subword_vocabulary,
)

#: The two rules, named as the record names them. "default" is the rule that was in force when the committed
#: vocabularies were fitted — `LEGACY_DEFAULT_PRE_TOKENIZER`, a format constant — and deliberately not the library's
#: ambient default, which ADR-0051 is free to move: a measurement whose arm labels followed the ambient default would
#: relabel its own arms without changing a byte of text.
RULES = {
    "default": LEGACY_DEFAULT_PRE_TOKENIZER,
    "space-prefixed": SPACE_PREFIXED_PRE_TOKENIZER,
}


class AnalysisError(RuntimeError):
    """The inputs to the refit measurement do not describe one experiment."""


def _seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in text.split(",") if item.strip())
    except ValueError as error:
        message = f"--fit-seeds wants comma-separated integers, got {text!r}"
        raise AnalysisError(message) from error
    if len(set(seeds)) != len(seeds) or not seeds:
        message = f"--fit-seeds must name distinct seeds, got {text!r}"
        raise AnalysisError(message)
    return seeds


def compression(vocabulary: SubwordVocabulary, texts: Sequence[str]) -> dict[str, float]:
    """How the vocabulary cuts the evaluation text: identifiers, characters, and the ratio between them.

    The identifier count includes each document's separators, exactly as a packed evaluation does, so a comparison
    between two vocabularies is a comparison of what two runs would actually have spent their budget on.
    """
    identifiers = sum(len(vocabulary.encode(text)) for text in texts)
    characters = sum(len(text) for text in texts)
    return {
        "documents": len(texts),
        "characters": characters,
        "identifiers": identifiers,
        "characters_per_identifier": characters / identifiers if identifiers else 0.0,
    }


def merge_overlap(
    first: Sequence[tuple[str, str]], second: Sequence[tuple[str, str]]
) -> dict[str, float]:
    """How much of one merge list another one shares, and how well the shared part is ordered.

    The set measure (Jaccard) answers "did the two fits learn the same pieces"; the rank measure answers "in the
    same order", which is what a diverging early merge turns into. Spearman is computed over the merges both lists
    contain, on their rank within the list rather than on frequency, and an empty intersection returns ``None``
    rather than a made-up 1.0.
    """
    a, b = list(first), list(second)  # merge rank == position in the learned order
    shared = set(a) & set(b)
    union = set(a) | set(b)
    ranks_a = {pair: index for index, pair in enumerate(a)}
    ranks_b = {pair: index for index, pair in enumerate(b)}
    common = sorted(shared, key=lambda pair: ranks_a[pair])
    rho: float | None = None
    if len(common) > 2:
        from chakaso.training.paired_stats import spearman_rho

        rho = spearman_rho(
            [float(ranks_a[pair]) for pair in common],
            [float(ranks_b[pair]) for pair in common],
        )
    return {
        "jaccard": len(shared) / len(union) if union else 0.0,
        "shared_merges": len(shared),
        "rank_agreement": rho,
        "compared_pairs": len(common) if rho is not None else 0,
    }


def summarise(result: Mapping[str, Any]) -> str:
    """The reading, in words, so the command that ran is also an answer."""
    samples = [int(value) for value in result["sample_characters"]]
    lines = [
        f"fit seeds {result['fit_seeds']}, {min(samples):,}-{max(samples):,} characters of fitting text per "
        f"sample, {result['identifiers']} identifiers, min_frequency {result['min_frequency']}"
    ]
    for rule in RULES:
        per_seed = [fit["compression"][rule]["characters_per_identifier"] for fit in result["fits"]]
        lines.append(
            f"  {rule:15} characters per identifier: mean {statistics.fmean(per_seed):.4f}, "
            f"range {min(per_seed):.4f}-{max(per_seed):.4f}"
        )
    ratios = [fit["ratio_space_over_default"] for fit in result["fits"]]
    lines.append(
        f"  space-prefixed buys that many more characters per identifier, every fit: "
        f"mean {statistics.fmean(ratios):.4f}, spread {max(ratios) - min(ratios):.4f}"
    )
    for rule, stability in result["merge_stability"].items():
        lines.append(
            f"  {rule:15} merge overlap against its own first fit: jaccard "
            f"{stability['jaccard_mean']:.3f}, rank agreement {stability['rank_mean_text']}"
        )
    if result.get("references"):
        for name, block in result["references"].items():
            lines.append(
                f"  committed {name:14} ({block['fingerprint']}): "
                f"{block['compression']['characters_per_identifier']:.4f} characters per identifier"
            )
    return "\n".join(lines)


def measure(
    *,
    labels: Sequence[int],
    fitting_samples: Sequence[Sequence[str]],
    evaluation_texts: Sequence[str],
    identifiers: int,
    min_frequency: int,
    references: Mapping[str, SubwordVocabulary] | None = None,
) -> dict[str, Any]:
    """Fit both rules on every sample and compare them on the fixed held-out text.

    The two rules on one sample are fitted from the *same* tuple of texts, which is what makes their difference a
    property of the rule and not of a redraw.
    """
    if not fitting_samples or not evaluation_texts:
        message = (
            "a refit measurement needs at least one fitting sample and one evaluation document"
        )
        raise AnalysisError(message)
    if len(labels) != len(fitting_samples):
        message = f"{len(fitting_samples)} samples and {len(labels)} labels do not describe one experiment"
        raise AnalysisError(message)
    fits: list[dict[str, Any]] = []
    merges: dict[str, list[list[tuple[str, str]]]] = {rule: [] for rule in RULES}
    for index, sample in enumerate(fitting_samples):
        vocabularies: dict[str, SubwordVocabulary] = {}
        for rule, pattern in RULES.items():
            vocabularies[rule] = train_subword_vocabulary(
                sample,
                target_size=identifiers,
                min_frequency=min_frequency,
                pre_tokenizer=pattern,
            )
            merges[rule].append(list(vocabularies[rule].merges))
        per_rule = {rule: compression(vocabularies[rule], evaluation_texts) for rule in RULES}
        fits.append(
            {
                "index": index,
                "sample_characters": sum(len(text) for text in sample),
                "sample_documents": len(sample),
                "compression": per_rule,
                "ratio_space_over_default": per_rule["space-prefixed"]["characters_per_identifier"]
                / per_rule["default"]["characters_per_identifier"],
                "fingerprints": {rule: vocabularies[rule].fingerprint for rule in RULES},
                "sizes": {rule: vocabularies[rule].size for rule in RULES},
            }
        )
    stability: dict[str, Any] = {}
    for rule, lists in merges.items():
        pairs = [merge_overlap(lists[0], other) for other in lists[1:]]
        ranks = [pair["rank_agreement"] for pair in pairs if pair["rank_agreement"] is not None]
        stability[rule] = {
            "jaccard_mean": statistics.fmean(pair["jaccard"] for pair in pairs) if pairs else 1.0,
            "jaccard_values": [pair["jaccard"] for pair in pairs],
            "rank_mean": statistics.fmean(ranks) if ranks else None,
            "rank_values": ranks,
            "rank_mean_text": "n/a" if not ranks else f"{statistics.fmean(ranks):.3f}",
        }
    references_block: dict[str, Any] = {}
    for name, vocabulary in (references or {}).items():
        references_block[name] = {
            "fingerprint": vocabulary.fingerprint,
            "size": vocabulary.size,
            "pre_tokenizer": vocabulary.pre_tokenizer,
            "compression": compression(vocabulary, evaluation_texts),
        }
    return {
        "fit_seeds": list(labels),
        "sample_characters": [fit["sample_characters"] for fit in fits],
        "identifiers": identifiers,
        "min_frequency": min_frequency,
        "evaluation": {
            "documents": len(evaluation_texts),
            "characters": sum(len(text) for text in evaluation_texts),
        },
        "fits": fits,
        "merge_stability": stability,
        "references": references_block,
        "rules": dict(RULES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fit-seeds", default="7,11,13,17")
    parser.add_argument("--sample-characters", type=int, default=4_000_000)
    parser.add_argument("--identifiers", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=20261104)
    parser.add_argument("--eval-characters", type=int, default=1_000_000)
    parser.add_argument(
        "--reference-vocabulary",
        action="append",
        default=[],
        metavar="NAME=path",
        help="a committed vocabulary to measure the same way, so the small fits are tied to published artifacts",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        from chakaso.training.subword_experiment import split_of_corpus, tokenizer_sample

        manifest = read_snapshot(args.snapshot).manifest
        split = split_of_corpus(manifest, seed=args.split_seed, whole_works=True)
        seeds = _seeds(args.fit_seeds)
        samples = [
            tokenizer_sample(split.train, cap=args.sample_characters, seed=seed) for seed in seeds
        ]
        evaluation = tokenizer_sample(split.test, cap=args.eval_characters, seed=args.eval_seed)
        references: dict[str, SubwordVocabulary] = {}
        for entry in args.reference_vocabulary:
            name, _, path = entry.partition("=")
            if not path:
                message = f"--reference-vocabulary wants NAME=path, got {entry!r}"
                raise AnalysisError(message)
            references[name] = SubwordVocabulary.load(Path(path))
        result = measure(
            labels=seeds,
            fitting_samples=samples,
            evaluation_texts=evaluation,
            identifiers=args.identifiers,
            min_frequency=args.min_frequency,
            references=references,
        )
        result["snapshot"] = str(args.snapshot)
        result["corpus"] = {"name": manifest.name, "fingerprint": manifest.fingerprint}
        result["split"] = {"seed": args.split_seed, "fingerprint": split.fingerprint}
    except (AnalysisError, CorpusError, OSError, TrainingError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(summarise(result), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
