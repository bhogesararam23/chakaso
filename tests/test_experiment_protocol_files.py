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


def test_the_equal_text_arm_differs_from_the_equal_positions_arm_only_in_the_declared_fields() -> (
    None
):
    directory = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    equal_positions = tomllib.loads((directory / "E2620-space-prefixed.toml").read_text("utf-8"))[
        "plan"
    ]
    equal_text = tomllib.loads(
        (directory / "E2624-space-prefixed-equal-text.toml").read_text("utf-8")
    )["plan"]

    changed = {
        key
        for key in {*equal_positions, *equal_text}
        if equal_positions.get(key) != equal_text.get(key)
    }

    assert changed == {
        "max_training_tokens",
        "max_steps",
        "warmup_steps",
        "eval_every",
        "tokenizer_file",
        "expected_tokenizer_fingerprint",
    }
    # The same fraction of the pool is read: updates x positions over the capped pool, to three decimals, as the control's 0.983.
    positions = equal_text["max_steps"] * equal_text["batch_size"] * equal_text["context_length"]
    assert round(positions / equal_text["max_training_tokens"], 3) == 0.983
    # The schedule keeps its shape: warmup and evaluation interval are the same fractions of the updates.
    assert round(equal_text["warmup_steps"] / equal_text["max_steps"], 3) == round(
        equal_positions["warmup_steps"] / equal_positions["max_steps"], 3
    )
    # The vocabulary is the one E2620 fitted, as a committed file, under the same rule.
    assert equal_text["tokenizer_file"].endswith(
        equal_text["expected_tokenizer_fingerprint"] + ".json"
    )
    assert equal_text["tokenizer_pre_tokenizer"] == equal_positions["tokenizer_pre_tokenizer"]


def test_the_equal_updates_and_documents_arm_is_e2620_with_only_the_vocabulary_input_and_the_pool_cap_changed() -> (
    None
):
    directory = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    e2620 = tomllib.loads((directory / "E2620-space-prefixed.toml").read_text("utf-8"))["plan"]
    arm = tomllib.loads(
        (directory / "E2625-space-prefixed-equal-updates-and-documents.toml").read_text("utf-8")
    )["plan"]
    equal_text = tomllib.loads(
        (directory / "E2624-space-prefixed-equal-text.toml").read_text("utf-8")
    )["plan"]

    changed = {key for key in {*e2620, *arm} if e2620.get(key) != arm.get(key)}

    assert changed == {
        "max_training_tokens",
        "tokenizer_file",
        "expected_tokenizer_fingerprint",
    }
    # The pool is the control's documents (E2624's cap) and the schedule is the control's own.
    assert arm["max_training_tokens"] == equal_text["max_training_tokens"] == 14_033_424
    assert (arm["max_steps"], arm["warmup_steps"], arm["eval_every"]) == (2400, 200, 400)
    positions = arm["max_steps"] * arm["batch_size"] * arm["context_length"]
    assert (
        round(positions / arm["max_training_tokens"], 2) == 1.40
    )  # about 1.4 passes over the pool


def test_the_ladder_differs_from_e2614_only_in_how_often_it_evaluates_and_what_it_writes() -> None:
    """E2626 is E2614's model measured more often, not a different model.

    The claim the ladder rests on is that an evaluation and an artifact write do not perturb the weights: the
    loop's batch plan and its learning-rate schedule are functions of the *absolute* step, so doubling the
    evaluation cadence cannot move a weight. If any other field had moved, that claim would be about a
    different run, and the rung ordering would not be E2614's curve.
    """
    directory = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    reference = tomllib.loads((directory / "E2614-decoder-seed-spread.toml").read_text("utf-8"))[
        "plan"
    ]
    arm = tomllib.loads((directory / "E2626-curve-ladder.toml").read_text("utf-8"))["plan"]

    changed = {key for key in {*reference, *arm} if reference.get(key) != arm.get(key)}

    assert changed == {"eval_every", "curve_artifact_steps"}
    assert arm["eval_every"] == reference["max_steps"] // 12
    # Every rung is an evaluation point of its own run, and the last one is the ceiling.
    rungs = list(arm["curve_artifact_steps"])
    assert rungs == sorted(rungs)
    assert all(step % arm["eval_every"] == 0 for step in rungs)
    assert rungs == list(range(arm["eval_every"], reference["max_steps"] + 1, arm["eval_every"]))


def test_the_decay_exemption_protocol_differs_from_the_ladder_in_one_declared_field() -> None:
    """E2631 asks one question, so its plan moves one field: ADR-0049's `weight_decay_excludes_vectors`.

    The control is the ladder run at E2626 on the same host, which is bit-identical to E2614 seed 7; a second
    difference anywhere in this file would make the arm pair a comparison of two protocols and not of one option.
    """
    directory = Path(__file__).resolve().parent.parent / "configs" / "experiments"
    ladder = tomllib.loads((directory / "E2626-curve-ladder.toml").read_text("utf-8"))["plan"]
    arm = tomllib.loads((directory / "E2631-vector-decay-exemption.toml").read_text("utf-8"))[
        "plan"
    ]

    changed = {key for key in {*ladder, *arm} if ladder.get(key) != arm.get(key)}

    assert changed == {"weight_decay_excludes_vectors"}
    assert arm["weight_decay_excludes_vectors"] is True
    assert ladder.get("weight_decay_excludes_vectors", False) is False
    # Everything the control's report fixes as matched, the file repeats verbatim.
    for key in (
        "split_seed",
        "expected_corpus_fingerprint",
        "expected_tokenizer_fingerprint",
        "tokenizer_file",
        "max_steps",
        "eval_every",
        "curve_artifact_steps",
        "learning_rate",
        "weight_decay",
        "test_partitions",
    ):
        assert arm[key] == ladder[key], key
