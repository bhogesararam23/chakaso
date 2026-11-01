"""Guards on the committed protocol files of related experiments: a sweep is a sweep only if nothing else moved."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_the_share_sweep_protocols_differ_from_e2614_and_each_other_only_in_the_share() -> None:
    directory = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    reference = tomllib.loads((directory / "E2614-decoder-seed-spread.toml").read_text("utf-8"))
    arms = {
        path.stem: tomllib.loads(path.read_text("utf-8"))["plan"]
        for path in sorted(directory.glob("E2615-s*.toml"))
    }

    assert sorted(arms) == ["E2615-s00", "E2615-s25", "E2615-s50", "E2615-s75"]
    shares = {name: plan.pop("share_of_tokens") for name, plan in arms.items()}
    assert shares == {"E2615-s00": 0.0, "E2615-s25": 0.25, "E2615-s50": 0.5, "E2615-s75": 0.75}
    for name, plan in arms.items():
        assert plan.pop("share_group_prefix") == "prose-", name
        # The sweep is a sweep only if nothing else moved: not the budget, the vocabulary, the split or the model.
        assert plan == reference["plan"], name
