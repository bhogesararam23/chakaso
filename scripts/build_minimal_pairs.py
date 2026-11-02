"""Generate the grammatical minimal-pair probe from a fixed lexicon and templates, deterministically.

Every sentence is written by this project: a small lexicon of nouns, verbs and adjectives common in nineteenth-century
prose, and seven templates, each of which makes a good sentence and a bad one that differ in exactly one respect. For each
phenomenon the full cross-product of lexicon items and sentence frames is built, ordered by the SHA-256 of the pair's
identity (no random generator, so the selection cannot differ between Python versions), and the first ``PAIRS_PER_PHENOMENON``
are kept. Good sentences are unique across the whole set. The output is a ``chakaso-minimal-pairs/1`` file whose digest binds
its pairs.

The probe measures one narrow thing - whether a model prefers a grammatical sentence to a minimally different ungrammatical one
on constructions it has certainly met in its training prose. It is not a measure of language understanding.

Usage:
    python scripts/build_minimal_pairs.py --out configs/probes/grammatical-minimal-pairs-v1.json
    python scripts/build_minimal_pairs.py --balance-with experiments/E2612/vocabulary-434debe82f949347.json \\
        --out configs/probes/grammatical-minimal-pairs-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.core.atomic import write_text_atomically
from chakaso.training.minimal_pairs import MinimalPair, PairSet
from chakaso.training.subword import SubwordVocabulary

PAIRS_PER_PHENOMENON = 100
VERSION = "1.0.0"
BALANCED_VERSION = "2.0.0"

NOUNS = (  # (singular, plural)
    ("man", "men"),
    ("woman", "women"),
    ("child", "children"),
    ("king", "kings"),
    ("dog", "dogs"),
    ("horse", "horses"),
    ("river", "rivers"),
    ("house", "houses"),
    ("tree", "trees"),
    ("bird", "birds"),
    ("boy", "boys"),
    ("girl", "girls"),
    ("soldier", "soldiers"),
    ("ship", "ships"),
    ("letter", "letters"),
    ("door", "doors"),
)
VERBS = (  # (third person singular, base)
    ("runs", "run"),
    ("walks", "walk"),
    ("speaks", "speak"),
    ("sleeps", "sleep"),
    ("waits", "wait"),
    ("sings", "sing"),
    ("stands", "stand"),
    ("falls", "fall"),
    ("works", "work"),
    ("rests", "rest"),
)
ADJECTIVES = ("old", "young", "tall", "quiet", "poor", "brave", "kind", "strong", "small", "great")
PARTICIPLES = (  # (past participle, simple past)
    ("gone", "went"),
    ("seen", "saw"),
    ("taken", "took"),
    ("written", "wrote"),
    ("broken", "broke"),
    ("spoken", "spoke"),
    ("begun", "began"),
    ("fallen", "fell"),
    ("chosen", "chose"),
    ("given", "gave"),
)
REFLEXIVES = (
    ("She", "herself"),
    ("He", "himself"),
    ("They", "themselves"),
    ("We", "ourselves"),
    ("I", "myself"),
)
PAST_VERBS = ("hurt", "blamed", "saw", "washed", "taught")
MODALS = ("can", "will", "must", "may")
PLACES = ("by the gate", "in the field", "near the river", "at the door", "on the hill")


def _frames_agreement() -> Iterator[tuple[str, str, str]]:
    for (singular, plural), (third, base), adjective in product(NOUNS, VERBS, (None, *ADJECTIVES)):
        modifier = f"{adjective} " if adjective else ""
        yield (
            f"sva|{adjective}|{singular}|{third}",
            f"The {modifier}{singular} {third}.",
            f"The {modifier}{singular} {base}.",
        )
        yield (
            f"sva|{adjective}|{plural}|{base}",
            f"The {modifier}{plural} {base}.",
            f"The {modifier}{plural} {third}.",
        )


def _frames_attractor() -> Iterator[tuple[str, str, str]]:
    """A subject separated from its verb by a noun of the other number."""
    for (head_s, head_p), (other_s, other_p), (third, base), place in product(
        NOUNS, NOUNS, VERBS, PLACES
    ):
        if head_s == other_s:
            continue
        yield (
            f"att|s|{head_s}|{other_p}|{third}|{place}",
            f"The {head_s} near the {other_p} {third}.",
            f"The {head_s} near the {other_p} {base}.",
        )
        yield (
            f"att|p|{head_p}|{other_s}|{base}|{place}",
            f"The {head_p} near the {other_s} {base}.",
            f"The {head_p} near the {other_s} {third}.",
        )


def _frames_determiner() -> Iterator[tuple[str, str, str]]:
    for (singular, plural), place, verb in product(
        NOUNS, PLACES, ("saw", "found", "heard", "left")
    ):
        yield (
            f"det|this|{singular}|{place}|{verb}",
            f"I {verb} this {singular} {place}.",
            f"I {verb} this {plural} {place}.",
        )
        yield (
            f"det|these|{plural}|{place}|{verb}",
            f"I {verb} these {plural} {place}.",
            f"I {verb} these {singular} {place}.",
        )


def _frames_modal() -> Iterator[tuple[str, str, str]]:
    for (singular, plural), (third, base), modal in product(NOUNS, VERBS, MODALS):
        yield (
            f"mod|s|{singular}|{base}|{modal}",
            f"The {singular} {modal} {base}.",
            f"The {singular} {modal} {third}.",
        )
        yield (
            f"mod|p|{plural}|{base}|{modal}",
            f"The {plural} {modal} {base}.",
            f"The {plural} {modal} {third}.",
        )


def _frames_order() -> Iterator[tuple[str, str, str]]:
    for (singular, plural), (third, base), adjective in product(NOUNS, VERBS, ADJECTIVES):
        yield (
            f"ord|s|{adjective}|{singular}|{third}",
            f"The {adjective} {singular} {third}.",
            f"{adjective.capitalize()} the {singular} {third}.",
        )
        yield (
            f"ord|p|{adjective}|{plural}|{base}",
            f"The {adjective} {plural} {base}.",
            f"{adjective.capitalize()} the {plural} {base}.",
        )


def _frames_reflexive() -> Iterator[tuple[str, str, str]]:
    for (subject, own), verb, (_, other), place in product(
        REFLEXIVES, PAST_VERBS, REFLEXIVES, PLACES
    ):
        if own == other:
            continue
        yield (
            f"ref|{subject}|{verb}|{other}|{place}",
            f"{subject} {verb} {own} {place}.",
            f"{subject} {verb} {other} {place}.",
        )


def _frames_participle() -> Iterator[tuple[str, str, str]]:
    for (singular, _plural), (participle, past), place in product(NOUNS, PARTICIPLES, PLACES):
        yield (
            f"ppl|{singular}|{participle}|{place}",
            f"The {singular} has {participle} {place}.",
            f"The {singular} has {past} {place}.",
        )


def _frames_auxiliary() -> Iterator[tuple[str, str, str]]:
    for (singular, plural), (_third, base), adjective in product(NOUNS, VERBS, (None, *ADJECTIVES)):
        modifier = f"{adjective} " if adjective else ""
        yield (
            f"aux|s|{adjective}|{singular}|{base}",
            f"The {modifier}{singular} does not {base}.",
            f"The {modifier}{singular} do not {base}.",
        )
        yield (
            f"aux|p|{adjective}|{plural}|{base}",
            f"The {modifier}{plural} do not {base}.",
            f"The {modifier}{plural} does not {base}.",
        )


PHENOMENA = (
    ("subject_verb_agreement", _frames_agreement),
    ("subject_verb_agreement_with_attractor", _frames_attractor),
    ("determiner_noun_agreement", _frames_determiner),
    ("modal_takes_base_form", _frames_modal),
    ("determiner_adjective_order", _frames_order),
    ("reflexive_pronoun_agreement", _frames_reflexive),
    ("past_participle_form", _frames_participle),
    ("auxiliary_agreement", _frames_auxiliary),
)


def build_pairs(balance_with: SubwordVocabulary | None = None) -> PairSet:
    """The full probe: the first ``PAIRS_PER_PHENOMENON`` frames of each phenomenon in hash order, good sentences unique.

    With ``balance_with``, only frames whose two sentences have the same number of identifiers under that vocabulary are
    eligible (version 2): a summed log-probability falls with every identifier, so an untrained model prefers the
    sentence with fewer, and E2621 found that bias carried 0.95 of one phenomenon in version 1.
    """
    taken: set[str] = set()
    pairs: list[MinimalPair] = []
    for phenomenon, frames in PHENOMENA:
        ordered = sorted(
            frames(),
            key=lambda frame: hashlib.sha256(f"{phenomenon}|{frame[0]}".encode()).hexdigest(),
        )
        chosen = 0
        for key, good, bad in ordered:
            if good in taken or good == bad:
                continue
            if balance_with is not None and len(balance_with.encode(good)) != len(
                balance_with.encode(bad)
            ):
                continue
            taken.add(good)
            chosen += 1
            digest = hashlib.sha256(f"{phenomenon}|{key}".encode()).hexdigest()[:8]
            pairs.append(MinimalPair(f"{phenomenon}-{digest}", phenomenon, good, bad))
            if chosen == PAIRS_PER_PHENOMENON:
                break
        if chosen < PAIRS_PER_PHENOMENON:
            message = f"{phenomenon} yielded only {chosen} distinct pairs"
            raise ValueError(message)
    return PairSet(
        name="chakaso-grammatical-minimal-pairs",
        version=VERSION if balance_with is None else BALANCED_VERSION,
        licence="Apache-2.0; every sentence was written by this project, none is taken from any corpus",
        provenance=(
            "generated by scripts/build_minimal_pairs.py from a fixed lexicon and eight templates; selection is by SHA-256 "
            "of each frame's identity, so it does not depend on a random generator"
            + (
                ""
                if balance_with is None
                else f"; only frames whose sentences have equal identifier counts under vocabulary {balance_with.fingerprint}"
            )
        ),
        pairs=tuple(pairs),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--balance-with",
        type=Path,
        help="a vocabulary file: keep only pairs whose sentences have equal identifier counts under it (version 2)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.overwrite:
        print(f"error: {args.out} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pair_set = build_pairs(
        None if args.balance_with is None else SubwordVocabulary.load(args.balance_with)
    )
    write_text_atomically(
        json.dumps(pair_set.as_dict(), indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        args.out,
    )
    print(f"wrote {len(pair_set.pairs)} pairs, digest {pair_set.digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
