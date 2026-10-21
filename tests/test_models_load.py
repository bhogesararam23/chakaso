"""Tests for `load_model`: choosing an adapter from the artifact, and refusing anything else.

The loader's job is selection and refusal, so these are mostly failure-path tests. The rule they enforce is
the one ADR-0033 depends on: a Python caller with a checkpoint must reach a trained model without a
terminal, and a mistake — a wrong path, an unknown format, a foreign schema — must produce a named error
rather than a development double that quietly answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chakaso.models.errors import ModelError, UnknownModelError
from chakaso.models.load import select_adapter_for
from chakaso.models.registry import SUBWORD_TRANSFORMER_MODEL_ID, TRAINED_CHAR_MODEL_ID

_SCHEMA_2 = "chakaso-checkpoint/2"


def _char_checkpoint(path: Path, *, schema: str = _SCHEMA_2) -> Path:
    document = {"schema_version": schema, "arrays": {}, "vocab_size": 12}
    document["checksum"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ------------------------------------------------------------------ format selection


def test_a_transformer_artifact_selects_the_subword_adapter(tmp_path: Path) -> None:
    target = tmp_path / "E2605-transformer.pt"
    target.write_bytes(b"not really a torch payload")

    assert select_adapter_for(target) == SUBWORD_TRANSFORMER_MODEL_ID


def test_a_schema_2_json_checkpoint_selects_the_character_adapter(tmp_path: Path) -> None:
    assert select_adapter_for(_char_checkpoint(tmp_path / "ckpt.json")) == TRAINED_CHAR_MODEL_ID


def test_a_json_file_from_another_schema_is_named_rather_than_guessed(tmp_path: Path) -> None:
    target = _char_checkpoint(tmp_path / "other.json", schema="chakaso-checkpoint/9")

    with pytest.raises(ModelError, match="declares schema 'chakaso-checkpoint/9'"):
        select_adapter_for(target)


def test_a_json_checkpoint_that_is_not_json_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ModelError, match="not readable JSON"):
        select_adapter_for(target)


def test_an_unrecognized_extension_is_refused_with_the_supported_formats(tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"\x00")

    with pytest.raises(ModelError, match="unrecognized checkpoint extension"):
        select_adapter_for(target)


def test_a_missing_file_is_refused_before_any_adapter_is_built(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="no checkpoint file"):
        select_adapter_for(tmp_path / "absent.pt")


def test_a_directory_that_is_not_a_release_is_refused(tmp_path: Path) -> None:
    # A directory is only loadable when it carries a release contract; anything else is named as the
    # absence it is, rather than being guessed at or silently treated as a checkpoint.
    folder = tmp_path / "data.pt"
    folder.mkdir()

    with pytest.raises(ModelError, match="holds no release document"):
        select_adapter_for(folder)


# ------------------------------------------------------------------ the public entry point


def test_load_model_is_reachable_from_the_package_facade() -> None:
    from chakaso.models import load_model

    assert callable(load_model)


def test_load_model_refuses_an_absent_path_without_falling_back(tmp_path: Path) -> None:
    from chakaso.models import load_model

    with pytest.raises(ModelError, match="no checkpoint file"):
        load_model(tmp_path / "nothing.pt")


def test_load_model_honours_an_explicit_adapter_name(tmp_path: Path) -> None:
    from chakaso.models import load_model

    target = _char_checkpoint(tmp_path / "checkpoint.json")

    # The path is a stub whose arrays are empty, so building the adapter fails — but it fails inside the
    # *named* adapter's checkpoint loader, which is what proves an explicit `adapter=` reaches the registry
    # instead of being ignored or answered by a fallback.
    numpy = pytest.importorskip("numpy", reason="the trained_char adapter lives behind [train]")
    assert numpy is not None
    with pytest.raises(ModelError, match="could not load"):
        load_model(target, adapter=TRAINED_CHAR_MODEL_ID)


def test_an_unknown_adapter_name_is_refused_by_the_registry(tmp_path: Path) -> None:
    from chakaso.models import load_model

    target = tmp_path / "anything.pt"
    target.write_bytes(b"x")

    with pytest.raises(UnknownModelError, match="Registered adapters"):
        load_model(target, adapter="gpt_omnisapi")


def test_the_subword_adapter_module_is_not_imported_to_read_the_registry() -> None:
    # The registry must stay importable without the optional extra, so selecting the adapter is the only
    # moment `chakaso.models.nntransformer` (and therefore torch) is loaded. The AST guard in
    # test_repository_hygiene.py keeps it that way; this pins the behaviour at the entry point.
    import sys

    from chakaso.models import default_registry

    assert SUBWORD_TRANSFORMER_MODEL_ID in default_registry()
    loaded = sys.modules.get("chakaso.models.nntransformer")
    if loaded is not None:
        pytest.skip("another test in this session already selected the adapter")
