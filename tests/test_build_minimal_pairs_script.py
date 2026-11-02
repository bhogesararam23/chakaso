"""The committed probe is what its generator produces, and the generator makes valid, balanced, unique pairs."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

from chakaso.training.minimal_pairs import load_pair_set
from chakaso.training.subword import train_subword_vocabulary

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "build_minimal_pairs.py"
_COMMITTED = _ROOT / "configs" / "probes" / "grammatical-minimal-pairs-v1.json"
#: Pinned so a change to the lexicon, a template or the selection rule is a visible, reviewed change to the probe.
_DIGEST = "d2ca34018df75aff2b5fe0c7327e73ece3b2a82b9d1e166d7581c75da4752632"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_minimal_pairs_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_probe_has_a_hundred_pairs_of_each_phenomenon() -> None:
    pair_set = _script().build_pairs()  # type: ignore[attr-defined]

    counts = Counter(pair.phenomenon for pair in pair_set.pairs)
    assert len(counts) == 8
    assert set(counts.values()) == {100}
    assert len(pair_set.pairs) == 800


def test_every_good_sentence_is_unique_and_differs_from_its_bad_one() -> None:
    pair_set = _script().build_pairs()  # type: ignore[attr-defined]

    goods = [pair.good for pair in pair_set.pairs]
    assert len(set(goods)) == len(goods)
    assert all(pair.good != pair.bad for pair in pair_set.pairs)
    # No sentence is both a good one and a bad one: that would put the same string on both sides of the scoring.
    assert not set(goods) & {pair.bad for pair in pair_set.pairs}


def test_building_twice_gives_the_same_pairs() -> None:
    script = _script()

    assert script.build_pairs().digest == script.build_pairs().digest  # type: ignore[attr-defined]


def test_the_committed_file_is_exactly_what_the_generator_produces() -> None:
    built = _script().build_pairs()  # type: ignore[attr-defined]
    committed = load_pair_set(_COMMITTED)

    assert committed.pairs == built.pairs
    assert committed.digest == built.digest == _DIGEST
    assert json.loads(_COMMITTED.read_text(encoding="utf-8"))["licence"].startswith("Apache-2.0")


def test_the_generator_refuses_to_overwrite_a_file_unasked(tmp_path: Path) -> None:
    target = tmp_path / "pairs.json"
    target.write_text("{}", encoding="utf-8")

    assert _script().main([f"--out={target}"]) == 1  # type: ignore[attr-defined]
    assert target.read_text(encoding="utf-8") == "{}"


def test_every_sentence_encodes_and_decodes_exactly() -> None:
    pair_set = load_pair_set(_COMMITTED)
    vocabulary = train_subword_vocabulary(
        [sentence for pair in pair_set.pairs[:50] for sentence in (pair.good, pair.bad)],
        target_size=300,
        min_frequency=2,
    )

    for pair in pair_set.pairs:
        assert vocabulary.decode(vocabulary.encode(pair.good)) == pair.good
        assert vocabulary.decode(vocabulary.encode(pair.bad)) == pair.bad
