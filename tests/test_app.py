"""Tests for the application composition root.

The composition is where typed configuration becomes a running system, so these tests check
that a *configuration change actually changes runtime behaviour* — not that a config object
serialises. They assert the assembled application wires the real components: the model behind
the boundary, the store the configuration selects (never silently swapped for the in-memory
default), the strategy the planner runs, the top-k a retrieval turn applies, and the honest
absence of grounding when retrieval is disabled. The model is the deterministic double and the
embedding is the non-semantic fixture, so nothing here asserts answer quality — only wiring.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from chakaso.answers import InMemoryAnswerStore, SQLiteAnswerStore
from chakaso.app import ApplicationError, build_application, build_corpus
from chakaso.config import LoadedConfig, load_config
from chakaso.planning import QueryMode
from chakaso.retrieval import Corpus, IngestionError, ingest_text

_SPEC = ingest_text("Chakaso grounds answers in retrieved evidence.", label="spec")


def _corpus() -> Corpus:
    corpus = Corpus()
    corpus.add_document(_SPEC)
    return corpus


def _write(tmp_path: Path, text: str) -> LoadedConfig:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


# -- what the built-in configuration wires ------------------------------------


def test_default_config_wires_the_runtime_and_the_in_memory_store() -> None:
    app = build_application(load_config(), corpus=_corpus())

    assert app.retrieval_enabled is True
    assert app.strategy is QueryMode.LEXICAL
    assert isinstance(app.store, InMemoryAnswerStore)
    assert app.storage_backend == "memory"


# -- the store is the configured one, never a silent substitute ----------------


def test_injected_store_is_used_rather_than_replaced() -> None:
    store = InMemoryAnswerStore()

    app = build_application(load_config(), corpus=_corpus(), store=store)
    app.turn("What is Chakaso?")

    assert store.list()  # the turn recorded into the injected store, not a fresh one


def test_sqlite_backend_persists_answers_that_survive_closing(tmp_path: Path) -> None:
    db = tmp_path / "answers.db"
    loaded = _write(tmp_path, f'[storage]\nbackend = "sqlite"\npath = "{db.as_posix()}"\n')

    app = build_application(loaded, corpus=_corpus())
    app.turn("What is Chakaso?")
    app.close()

    # A reopen of the same file — as a new process would — sees the answer the application wrote.
    with SQLiteAnswerStore(db, initialize=False) as reopened:
        records = reopened.list()
    assert records
    assert records[0].provenance["retrieval_strategy"] in {"lexical", "none"}


def test_sqlite_backend_without_a_path_is_refused(tmp_path: Path) -> None:
    loaded = _write(tmp_path, '[storage]\nbackend = "sqlite"\n')

    with pytest.raises(ApplicationError, match=r"storage\.path"):
        build_application(loaded, corpus=_corpus())


# -- retrieval configuration drives the runtime, not just the object -----------


def test_a_strategy_change_selects_a_different_retriever(tmp_path: Path) -> None:
    loaded = _write(tmp_path, '[retrieval]\nstrategy = "dense"\n')

    app = build_application(loaded, corpus=_corpus())
    turn = app.turn("What is Chakaso?")

    assert turn.plan is not None
    assert turn.plan.mode is QueryMode.DENSE
    assert app.store.list()[0].provenance["retrieval_strategy"] == "dense"


@pytest.mark.parametrize("top_k", [1, 3])
def test_top_k_is_applied_to_a_retrieving_turn(tmp_path: Path, top_k: int) -> None:
    loaded = _write(tmp_path, f"[retrieval]\ntop_k = {top_k}\n")

    app = build_application(loaded, corpus=_corpus())
    app.turn("What is Chakaso?")

    record = app.store.list()[0]
    assert record.retrieval_metadata["top_k"] == str(top_k)


def test_disabled_retrieval_runs_the_manager_without_a_planner(tmp_path: Path) -> None:
    loaded = _write(tmp_path, "[retrieval]\nenabled = false\n")

    app = build_application(loaded, corpus=_corpus())
    turn = app.turn("What is Chakaso?")

    assert app.retrieval_enabled is False
    assert turn.plan is None
    assert turn.retrieved is False
    # Disabling grounding does not disable persistence: the answer is still recorded.
    assert app.store.list()


def test_a_greeting_retrieves_nothing_but_still_records() -> None:
    app = build_application(load_config(), corpus=_corpus())

    turn = app.turn("hello there")

    assert turn.retrieved is False
    assert turn.plan is not None
    assert turn.plan.mode is QueryMode.NONE
    assert app.store.list()[0].provenance["retrieval_strategy"] == "none"


# -- corpus construction ------------------------------------------------------


def test_build_corpus_reads_only_the_named_files(tmp_path: Path) -> None:
    document = tmp_path / "note.md"
    document.write_text("# T\n\nThe window closes in April.\n", encoding="utf-8")

    corpus = build_corpus([document], max_bytes=1_000_000)

    assert corpus.source_count == 1
    assert corpus.chunk_count >= 1


def test_build_corpus_bounds_document_size(tmp_path: Path) -> None:
    document = tmp_path / "big.md"
    document.write_text("x" * 200, encoding="utf-8")

    with pytest.raises(IngestionError):
        build_corpus([document], max_bytes=100)


def test_build_corpus_over_an_empty_path_list_is_a_valid_empty_corpus() -> None:
    paths: Sequence[str] = []

    assert build_corpus(paths, max_bytes=1000).is_empty


# -- lifecycle ----------------------------------------------------------------


def test_close_is_idempotent_and_releases_a_durable_store(tmp_path: Path) -> None:
    db = tmp_path / "answers.db"
    loaded = _write(tmp_path, f'[storage]\nbackend = "sqlite"\npath = "{db.as_posix()}"\n')

    app = build_application(loaded, corpus=_corpus())
    app.close()
    app.close()  # a second close must not raise
