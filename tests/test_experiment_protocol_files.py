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


def test_the_one_pass_control_differs_from_e2614_in_exactly_the_declared_fields() -> None:
    directory = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    reference = tomllib.loads((directory / "E2614-decoder-seed-spread.toml").read_text("utf-8"))[
        "plan"
    ]
    arm = tomllib.loads((directory / "E2616-one-pass-prose.toml").read_text("utf-8"))["plan"]

    changed = {key for key in {*reference, *arm} if reference.get(key) != arm.get(key)}

    assert changed == {
        "max_steps",
        "warmup_steps",
        "eval_every",
        "training_group_prefixes",
        "max_training_tokens",
    }
    assert arm["max_steps"] * 2 == reference["max_steps"]
    assert arm["warmup_steps"] * 2 == reference["warmup_steps"]
    assert arm["eval_every"] * 2 == reference["eval_every"]
    assert arm["training_group_prefixes"] == ["prose-"]
    # One pass: the steps read no more positions than the capped pack holds.
    positions = arm["max_steps"] * arm["batch_size"] * arm["context_length"]
    assert positions <= arm["max_training_tokens"]


def test_the_passes_arms_differ_from_e2614_only_in_the_prose_filter_and_the_cap() -> None:
    directory = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    reference = tomllib.loads((directory / "E2614-decoder-seed-spread.toml").read_text("utf-8"))[
        "plan"
    ]
    arms = {
        path.stem: tomllib.loads(path.read_text("utf-8"))["plan"]
        for path in sorted(directory.glob("E2619-prose-*.toml"))
    }

    assert sorted(arms) == ["E2619-prose-10m", "E2619-prose-20m", "E2619-prose-5m"]
    caps = {name: plan["max_training_tokens"] for name, plan in arms.items()}
    assert caps == {
        "E2619-prose-5m": 5_000_000,
        "E2619-prose-10m": 10_000_000,
        "E2619-prose-20m": 20_000_000,
    }
    for name, plan in arms.items():
        changed = {key for key in {*reference, *plan} if reference.get(key) != plan.get(key)}
        # The 20M arm's cap equals E2614's, so it differs from E2614 only in the prose filter.
        expected = {"training_group_prefixes"} | (
            {"max_training_tokens"} if plan["max_training_tokens"] != 20_000_000 else set()
        )
        assert changed == expected, name
        assert plan["training_group_prefixes"] == ["prose-"], name
        # The schedule is the point of the experiment: every arm makes the same 2,400 updates.
        assert plan["max_steps"] == reference["max_steps"] == 2400, name
        assert plan["warmup_steps"] == reference["warmup_steps"], name
    # Fewer distinct identifiers under the same updates means more passes: 5M is read about four times.
    positions = 2400 * reference["batch_size"] * reference["context_length"]
    assert round(positions / caps["E2619-prose-5m"]) == 4
    assert round(positions / caps["E2619-prose-10m"]) == 2
    assert round(positions / caps["E2619-prose-20m"]) == 1


def test_the_space_prefixed_arm_differs_from_e2614_only_in_how_the_tokenizer_is_obtained_and_the_eval_documents_chosen() -> (
    None
):
    directory = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    reference = tomllib.loads((directory / "E2614-decoder-seed-spread.toml").read_text("utf-8"))[
        "plan"
    ]
    arm = tomllib.loads((directory / "E2620-space-prefixed.toml").read_text("utf-8"))["plan"]

    changed = {key for key in {*reference, *arm} if reference.get(key) != arm.get(key)}

    assert changed == {
        "tokenizer_file",
        "expected_tokenizer_fingerprint",
        "tokenizer_pre_tokenizer",
        "evaluation_selection_tokenizer_file",
    }
    # Fitted by the run, so neither a file nor a fingerprint is declared; same sample, floor and size as the control's.
    assert "tokenizer_file" not in arm and "expected_tokenizer_fingerprint" not in arm
    for key in ("tokenizer_min_frequency", "tokenizer_sample_characters", "vocab_size"):
        assert arm[key] == reference[key], key
    # The documents scored are chosen with the control's own vocabulary.
    assert arm["evaluation_selection_tokenizer_file"] == reference["tokenizer_file"]
    assert arm["tokenizer_pre_tokenizer"].startswith("space-prefixed")
