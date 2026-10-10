"""Durable-storage behaviour that only a real database can exhibit.

The conformance suite proves the SQLite store enforces the same append-only rules as the
in-memory one. This file proves the things a memory store cannot: that an answer written here
outlives the process that wrote it, that a failure partway through a batch leaves no partial
chain, and that a schema or a stored document this code cannot safely interpret is refused rather
than repaired. These are the guarantees the word *durable* is being used for, so each is tested
against the actual file on disk, in a temporary directory — never the repository tree.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chakaso.answers import (
    AnswerId,
    AnswerRecord,
    AnswerSerializationError,
    AnswerStoreError,
    SQLiteAnswerStore,
    StorageSchemaError,
    new_answer_id,
)
from chakaso.answers.serialize import encode_json
from chakaso.answers.sqlite import SCHEMA_VERSION

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def make(*, conversation_id: str = "conv_1", correction_of: AnswerId | None = None) -> AnswerRecord:
    return AnswerRecord(
        answer_id=new_answer_id(),
        conversation_id=conversation_id,
        text="A durable answer.",
        created_at=NOW,
        model_id="deterministic-double",
        model_is_development_double=True,
        correction_of=correction_of,
    )


# -- restart ---------------------------------------------------------------------


def test_an_answer_survives_closing_and_reopening_the_store(tmp_path: Path) -> None:
    path = tmp_path / "answers.db"
    record = make()

    first = SQLiteAnswerStore(path)
    first.save(record)
    first.close()

    second = SQLiteAnswerStore(path)
    try:
        assert second.get(record.answer_id) == record
        assert second.list() == (record,)
    finally:
        second.close()


def test_an_answer_written_by_another_process_can_be_read_back(tmp_path: Path) -> None:
    # The strongest form of the restart claim: a genuinely separate interpreter writes the
    # answer and exits, then this process opens the same file. A connection close/reopen within
    # one process could still lean on process-local state; a fresh process cannot, so this is
    # the test that makes "durable" mean durable rather than "cached a little longer".
    path = tmp_path / "answers.db"
    record = make()
    script = (
        "import sys\n"
        "from chakaso.answers import SQLiteAnswerStore\n"
        "from chakaso.answers.serialize import decode_json\n"
        "with SQLiteAnswerStore(sys.argv[1]) as store:\n"
        "    store.save(decode_json(sys.argv[2]))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", script, str(path), encode_json(record)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    with SQLiteAnswerStore(path) as reopened:
        assert reopened.get(record.answer_id) == record
        assert reopened.list() == (record,)


# -- transactions ----------------------------------------------------------------


def test_a_failed_batch_is_rolled_back_completely(tmp_path: Path) -> None:
    store = SQLiteAnswerStore(tmp_path / "answers.db")
    existing = make()
    store.save(existing)
    fresh = make()

    with pytest.raises(AnswerStoreError, match="append-only"):
        store.save_many((fresh, existing))

    assert store.get(fresh.answer_id) is None
    assert len(store) == 1
    store.close()


def test_a_batch_written_in_the_wrong_order_persists_nothing(tmp_path: Path) -> None:
    # A child correcting a parent that appears later in the batch is not a valid history; the
    # transaction must abort and leave neither record, rather than half-writing a chain.
    store = SQLiteAnswerStore(tmp_path / "answers.db")
    parent = make()
    child = make(correction_of=parent.answer_id)

    with pytest.raises(AnswerStoreError, match="not in the store"):
        store.save_many((child, parent))

    assert len(store) == 0
    store.close()


# -- integrity and corruption ----------------------------------------------------


def test_a_corrupted_stored_document_is_reported_not_repaired(tmp_path: Path) -> None:
    path = tmp_path / "answers.db"
    record = make()
    store = SQLiteAnswerStore(path)
    store.save(record)
    store.close()

    # Corrupt the serialized document behind the store's back, then reopen: reading the answer
    # must refuse rather than guess at a plausible-but-wrong record.
    raw = sqlite3.connect(str(path))
    raw.execute(
        "UPDATE answers SET document = ? WHERE answer_id = ?",
        ("{ not json", str(record.answer_id)),
    )
    raw.commit()
    raw.close()

    reopened = SQLiteAnswerStore(path)
    with pytest.raises(AnswerSerializationError, match="not valid JSON"):
        reopened.get(record.answer_id)
    reopened.close()


def test_indexed_columns_that_disagree_with_the_document_are_corruption(tmp_path: Path) -> None:
    path = tmp_path / "answers.db"
    record = make()
    store = SQLiteAnswerStore(path)
    store.save(record)
    store.close()

    # The scalar columns are a second witness to the answer's identity. Mutating one alone, so
    # it contradicts the still-valid document, is corruption; reconstruction silently trusting
    # either half would rewrite history.
    raw = sqlite3.connect(str(path))
    raw.execute(
        "UPDATE answers SET conversation_id = ? WHERE answer_id = ?",
        ("conv_something_else", str(record.answer_id)),
    )
    raw.commit()
    raw.close()

    reopened = SQLiteAnswerStore(path)
    with pytest.raises(AnswerSerializationError, match="inconsistent"):
        reopened.get(record.answer_id)
    reopened.close()


# -- schema and versioning -------------------------------------------------------


def test_a_database_from_a_newer_version_is_refused_not_opened_for_writing(tmp_path: Path) -> None:
    path = tmp_path / "answers.db"
    SQLiteAnswerStore(path).close()

    raw = sqlite3.connect(str(path))
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.commit()
    raw.close()

    with pytest.raises(StorageSchemaError, match="newer"):
        SQLiteAnswerStore(path)


def test_an_existing_database_is_not_silently_created_when_asked_only_to_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(StorageSchemaError, match="initialize=False"):
        SQLiteAnswerStore(tmp_path / "absent.db", initialize=False)


def test_a_version_record_without_a_schema_table_is_refused(tmp_path: Path) -> None:
    # A database that claims a schema version but has lost its table is corrupt; creating a
    # table over the top of it would hide the loss, so the store reports instead.
    path = tmp_path / "answers.db"
    raw = sqlite3.connect(str(path))
    raw.execute("CREATE TABLE placeholder (x INTEGER)")
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    raw.commit()
    raw.close()

    with pytest.raises(StorageSchemaError, match="no answers table"):
        SQLiteAnswerStore(path)


# -- configuration surface and diagnostics --------------------------------------


def test_journal_mode_and_path_are_reported_as_set(tmp_path: Path) -> None:
    path = tmp_path / "answers.db"
    store = SQLiteAnswerStore(path, journal_mode="wal")
    try:
        assert store.journal_mode == "wal"
        assert store.path == path
    finally:
        store.close()


def test_a_read_only_check_leaves_the_existing_journal_untouched(tmp_path: Path) -> None:
    # journal_mode=None must not rewrite the file header, which is what a check command
    # relies on: opening to inspect is not opening to change.
    path = tmp_path / "answers.db"
    writer = SQLiteAnswerStore(path, journal_mode="wal")
    writer.close()

    reader = SQLiteAnswerStore(path, journal_mode=None, initialize=False)
    try:
        assert reader.journal_mode == "wal"
    finally:
        reader.close()


def test_an_unsupported_journal_mode_is_rejected_before_connecting(tmp_path: Path) -> None:
    path = tmp_path / "answers.db"

    with pytest.raises(AnswerStoreError, match="journal_mode"):
        SQLiteAnswerStore(path, journal_mode="not-a-real-mode")

    assert not path.exists()


def test_check_reports_the_stores_actual_health(tmp_path: Path) -> None:
    store = SQLiteAnswerStore(tmp_path / "answers.db")
    store.save(make())

    report = store.check()

    assert report["ok"] is True
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["answer_count"] == 1
    assert report["table_present"] is True
    store.migrate()  # a no-op confirmation at the current version, not a mutation
    store.close()


def test_an_in_memory_database_is_refused_because_it_is_not_durable() -> None:
    with pytest.raises(AnswerStoreError, match=":memory:"):
        SQLiteAnswerStore(":memory:")


def test_a_missing_parent_directory_is_reported_not_created(tmp_path: Path) -> None:
    with pytest.raises(AnswerStoreError, match="does not exist"):
        SQLiteAnswerStore(tmp_path / "nope" / "answers.db")
