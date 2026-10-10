"""A durable ``AnswerStore`` backed by SQLite.

ADR-0018 kept persistence behind an ``AnswerStore`` protocol precisely so that the day a
real workload needed history to survive a process, a backend could be added without touching
the conversation manager, retrieval, model or correction layers. This is that backend: the
same append-only history, written to a local database file with the standard library's
``sqlite3`` and no third-party dependency. Being local, it is honest to call *durable* — a
saved answer is readable after the process that wrote it has exited, which a test proves
rather than asserts (the restart test lives beside this module).

The design keeps every guarantee ADR-0018 established, and adds the ones a database has to
provide for research history to be trusted:

* **one append-only validator.** The overwrite, dangling-correction and cross-conversation
  checks are the in-memory store's, imported here rather than restated, so the two backends
  cannot drift apart on what a history is allowed to contain;
* **an explicit schema version** recorded in ``PRAGMA user_version``, with a migration
  boundary that refuses a newer or unsupported schema instead of opening and mutating it;
* **atomic writes.** Every append runs in one transaction that is committed only if every
  statement succeeded, so a failure partway through a batch leaves no partial correction
  chain behind;
* **strict reads.** A row's indexed scalar columns are cross-checked against its serialized
  document on every read; a divergence is reported as corruption, never reconciled silently.

Concurrency is deliberately scoped: the store keeps one connection, used from a single thread
(the intended model), and relies on SQLite's locking for cross-process safety without claiming
multi-process guarantees that are not tested here. The document column is the record's canonical
serialization (ADR-0021); the scalar columns exist to index, to enforce the correction foreign
key, and to give integrity something to check the document against — they are not a second copy
to keep in sync by hand.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from chakaso.answers.errors import AnswerSerializationError, AnswerStoreError, StorageSchemaError
from chakaso.answers.identity import AnswerId
from chakaso.answers.record import AnswerRecord
from chakaso.answers.serialize import decode_json, encode_json
from chakaso.answers.store import _ensure_appendable

__all__ = ["ANSWER_TABLE", "SCHEMA_VERSION", "SQLiteAnswerStore"]

#: The version this code writes and expects. Bump it only together with a step registered
#: in :data:`_MIGRATIONS`; a database carries its own version so a mismatch can be detected.
SCHEMA_VERSION = 1

ANSWER_TABLE = "answers"

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS answers_conversation ON answers(conversation_id);",
    "CREATE INDEX IF NOT EXISTS answers_correction ON answers(correction_of);",
)

#: SQLite journal modes this store will set. The set is a whitelist rather than free text
#: because the mode is applied with an interpolated ``PRAGMA`` (journal modes cannot be bound
#: as parameters), so an unrecognized value must be refused rather than reaching the database.
_SUPPORTED_JOURNAL_MODES = frozenset({"delete", "truncate", "persist", "memory", "wal", "off"})

_SCHEMA_SQL = f"""\
CREATE TABLE IF NOT EXISTS {ANSWER_TABLE} (
    answer_id       TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_index      INTEGER,
    created_at      TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    is_double       INTEGER NOT NULL CHECK (is_double IN (0, 1)),
    correction_of   TEXT REFERENCES {ANSWER_TABLE}(answer_id),
    document        TEXT NOT NULL
);
"""

# The migration boundary declared as an empty mapping, deliberately. There is exactly one
# schema version today, so there is nothing to migrate *from*; recording no steps is the
# honest state, not an unimplemented one. The contract the constructor and :func:`migrate`
# enforce is what makes this a boundary rather than an accident: to change the schema later,
# a contributor adds `_MIGRATIONS[old] = (sql, new_version)` and raises ``SCHEMA_VERSION``,
# and an older database is then brought forward step by step instead of silently recreated.
# A version with no registered path stays an explicit, reported refusal.
_MIGRATIONS: dict[int, tuple[str, int]] = {}


class SQLiteAnswerStore:
    """An ``AnswerStore`` persisting the append-only history in a local SQLite file."""

    def __init__(
        self,
        path: str | Path,
        *,
        journal_mode: str = "delete",
        initialize: bool = True,
        timeout: float = 5.0,
    ) -> None:
        """Open (and, unless told not to, create) the answer database at ``path``.

        Args:
            path: A file path for the database. It must name a real location; an empty path
                or the in-memory ``:memory:`` sentinel is refused because a store that does
                not survive the call that opened it is not durable, and the restart guarantee
                this class exists to provide would be a lie.
            journal_mode: A SQLite journal mode drawn from the supported set; it is applied
                with ``PRAGMA`` and read back so the value in use is the one actually set.
            initialize: Create the schema when the database is empty. ``False`` opens an
                existing database without touching its structure, so an absent schema is a
                reported error rather than something silently created.
            timeout: Seconds to wait for another writer's lock before giving up.
        """
        resolved = self._resolve_path(path)
        self._path: Path = resolved
        self._journal_mode = self._validate_journal_mode(journal_mode)
        self._closed = False

        try:
            self._conn = sqlite3.connect(str(resolved), isolation_level=None, timeout=timeout)
        except sqlite3.Error as exc:
            message = f"could not open the answer database at {resolved}: {exc}"
            raise AnswerStoreError(message) from exc

        # Foreign keys are off by default in SQLite; without them the correction foreign key
        # would not be enforced and a dangling link could be written despite the checks, so a
        # durable history would depend only on application code getting it right.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(f"PRAGMA journal_mode = {self._journal_mode}")
        self._conn.row_factory = None

        # A schema refusal must not leave the connection it opened dangling: an unclosed
        # database handle is a resource warning, and this suite turns warnings into errors, so
        # a rejected version would surface as a leak in an unrelated test rather than the
        # deliberate StorageSchemaError it should be.
        try:
            self._ensure_schema(initialize=initialize)
        except BaseException:
            self._conn.close()
            self._closed = True
            raise

    # -- lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection. Idempotent."""
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> SQLiteAnswerStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def path(self) -> Path:
        """The file this store persists to."""
        return self._path

    @property
    def journal_mode(self) -> str:
        """The journal mode actually in effect, as SQLite reports it."""
        row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]) if row else self._journal_mode

    # -- AnswerStore protocol ----------------------------------------------------

    def save(self, record: AnswerRecord) -> None:
        """Append one answer to the durable history in its own transaction."""
        self.save_many((record,))

    def save_many(self, records: Iterable[AnswerRecord]) -> None:
        """Append ``records`` atomically: the whole batch commits, or none of it does.

        Each append is validated against the live history plus the records already inserted
        in this transaction (a later record may correct an earlier one in the same batch),
        reusing :func:`~chakaso.answers.store._ensure_appendable` so the invariants are the
        in-memory store's invariants verbatim. Any rejection rolls the transaction back.
        """
        with self._transaction():
            for record in records:
                self._validate_append(record)
                self._conn.execute(
                    f"INSERT INTO {ANSWER_TABLE} "
                    "(answer_id, conversation_id, turn_index, created_at, model_id, is_double, "
                    "correction_of, document) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(record.answer_id),
                        record.conversation_id,
                        record.turn_index,
                        record.created_at.isoformat(),
                        record.model_id,
                        int(record.model_is_development_double),
                        None if record.correction_of is None else str(record.correction_of),
                        encode_json(record),
                    ),
                )

    def get(self, answer_id: AnswerId) -> AnswerRecord | None:
        """The recorded answer with ``answer_id``, or ``None`` if there is none."""
        cursor = self._conn.execute(
            f"SELECT document FROM {ANSWER_TABLE} WHERE answer_id = ?",
            (str(answer_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._decode_with_integrity(str(row[0]))

    def list(self, *, conversation_id: str | None = None) -> tuple[AnswerRecord, ...]:
        """Every recorded answer, in the order it was written (``rowid``).

        Ordering by ``rowid`` rather than by any column reproduces insertion order exactly as
        the in-memory store does, so the two backends enumerate the same history identically.
        """
        if conversation_id is None:
            cursor = self._conn.execute(f"SELECT document FROM {ANSWER_TABLE} ORDER BY rowid ASC")
        else:
            cursor = self._conn.execute(
                f"SELECT document FROM {ANSWER_TABLE} WHERE conversation_id = ? ORDER BY rowid ASC",
                (conversation_id,),
            )
        return tuple(self._decode_with_integrity(str(row[0])) for row in cursor.fetchall())

    def history(self, answer_id: AnswerId) -> tuple[AnswerRecord, ...]:
        """The correction chain ending at ``answer_id``, oldest first."""
        current = self.get(answer_id)
        if current is None:
            message = f"no answer {answer_id} is recorded; there is no history to walk"
            raise AnswerStoreError(message)

        chain = [current]
        seen = {current.answer_id}
        while current.correction_of is not None:
            parent = self.get(current.correction_of)
            if parent is None:
                message = (
                    f"answer {current.answer_id} references a missing predecessor "
                    f"{current.correction_of}; the stored history is inconsistent"
                )
                raise AnswerStoreError(message)
            if parent.answer_id in seen:
                break  # defensive: save() cannot create a cycle, but a walk must never loop
            chain.append(parent)
            seen.add(parent.answer_id)
            current = parent
        return tuple(reversed(chain))

    def __len__(self) -> int:
        """How many answers are recorded."""
        return self._count()

    # -- durable-specific surface ------------------------------------------------

    def check(self) -> Mapping[str, object]:
        """Report the store's structural health for ``chakaso storage check``.

        The values are facts read from the database, not assertions: the schema version it
        carries, whether the answer table is present, the row count, and SQLite's own
        ``integrity_check`` result. It is a diagnostic, not a repair — nothing here mutates
        the file.
        """
        integrity_cursor = self._conn.execute("PRAGMA integrity_check")
        integrity = [str(row[0]) for row in integrity_cursor.fetchall()]
        version = self._user_version()
        return {
            "path": str(self._path),
            "schema_version": version,
            "expected_version": SCHEMA_VERSION,
            "table_present": self._table_exists(),
            "answer_count": self._count(),
            "journal_mode": self.journal_mode,
            "integrity": integrity,
            "ok": version == SCHEMA_VERSION and integrity == ["ok"],
        }

    def migrate(self) -> Mapping[str, object]:
        """Bring the database to :data:`SCHEMA_VERSION`, or report why it cannot.

        With one schema version and no registered steps this is the boundary itself, not a
        migration: a fresh or current database is confirmed, and a newer or unsupported one is
        refused rather than mutated. Registered steps would run here in explicit transactions.
        """
        version = self._user_version()
        if version == 0:
            self._create_schema()
            version = SCHEMA_VERSION
        elif version > SCHEMA_VERSION:
            message = self._newer_message(version)
            raise StorageSchemaError(message)
        elif version < SCHEMA_VERSION:
            if not self._table_exists():
                message = self._missing_schema_message(version)
                raise StorageSchemaError(message)
            message = self._unsupported_message(version)
            raise StorageSchemaError(message)

        if not self._table_exists():
            message = self._missing_schema_message(SCHEMA_VERSION)
            raise StorageSchemaError(message)
        return {"schema_version": version, "expected_version": SCHEMA_VERSION, "migrated": True}

    # -- internals ---------------------------------------------------------------

    def _validate_append(self, record: AnswerRecord) -> None:
        """Enforce the shared append-only invariants against the live history.

        The decision and its wording are the in-memory store's, imported whole: this method
        only materialises the two rows the validator needs (the record's own identifier, and
        its correction target) from indexed primary-key lookups, so nothing about the
        invariants is restated here and the backends cannot drift. Within an open batch
        transaction these lookups also see rows inserted earlier in the same batch, so a
        record may correct another that appears before it.
        """
        existing: dict[AnswerId, AnswerRecord] = {}
        own = self.get(record.answer_id)
        if own is not None:
            existing[record.answer_id] = own
        if record.correction_of is not None:
            prior = self.get(record.correction_of)
            if prior is not None:
                existing[record.correction_of] = prior
        _ensure_appendable(record, existing)

    def _decode_with_integrity(self, document: str) -> AnswerRecord:
        """Decode a stored document and cross-check it against its indexed columns.

        The scalar columns are a second witness to the answer's identity, so a row whose
        ``answer_id`` or ``conversation_id`` disagrees with its own document is corruption, and
        reconstruction from either half alone would silently rewrite history. It is reported.
        """
        record = decode_json(document)
        cursor = self._conn.execute(
            f"SELECT answer_id, conversation_id FROM {ANSWER_TABLE} WHERE answer_id = ?",
            (str(record.answer_id),),
        )
        row = cursor.fetchone()
        if row is None:  # pragma: no cover - the caller read this exact row moments ago
            message = f"stored answer {record.answer_id} vanished between read and check"
            raise AnswerSerializationError(message)
        if str(row[0]) != str(record.answer_id) or str(row[1]) != record.conversation_id:
            message = (
                f"stored answer {record.answer_id} is inconsistent: its indexed columns "
                f"({row[0]!r}, {row[1]!r}) disagree with its document ({record.answer_id}, "
                f"{record.conversation_id!r})"
            )
            raise AnswerSerializationError(message)
        return record

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Run one ``BEGIN``/``COMMIT`` with ``ROLLBACK`` on any failure, nested or not."""
        self._conn.execute("BEGIN")
        try:
            yield
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _ensure_schema(self, *, initialize: bool) -> None:
        version = self._user_version()
        if version == 0:
            if not initialize:
                message = (
                    "this database is not a Chakaso answer store (no schema version is set) "
                    f"and initialize=False refused to create one; expected version {SCHEMA_VERSION}"
                )
                raise StorageSchemaError(message)
            self._create_schema()
            return
        if version > SCHEMA_VERSION:
            raise StorageSchemaError(self._newer_message(version))
        if version < SCHEMA_VERSION:
            if not self._table_exists():
                raise StorageSchemaError(self._missing_schema_message(version))
            raise StorageSchemaError(self._unsupported_message(version))
        if not self._table_exists():
            raise StorageSchemaError(self._missing_schema_message(version))

    def _create_schema(self) -> None:
        with self._transaction():
            self._conn.execute(_SCHEMA_SQL)
            for index in _INDEXES:
                self._conn.execute(index)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _user_version(self) -> int:
        row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def _count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {ANSWER_TABLE}").fetchone()
        return int(row[0]) if row else 0

    def _table_exists(self) -> bool:
        cursor = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (ANSWER_TABLE,),
        )
        return cursor.fetchone() is not None

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        """Validate and materialise a durable database path, refusing non-file targets."""
        if isinstance(path, str) and path == ":memory:":
            message = (
                "SQLiteAnswerStore requires a file path; ':memory:' would not survive the "
                "process that created it, so the durable guarantee it advertises would be false"
            )
            raise AnswerStoreError(message)
        candidate = Path(path)
        if str(candidate) in {"", "."}:
            message = "SQLiteAnswerStore requires a non-empty database path"
            raise AnswerStoreError(message)
        if not candidate.parent.is_dir():
            message = f"the database path's parent directory does not exist: {candidate.parent}"
            raise AnswerStoreError(message)
        return candidate

    @staticmethod
    def _validate_journal_mode(journal_mode: str) -> str:
        """Accept a journal mode from the supported set, rejecting anything else."""
        if journal_mode not in _SUPPORTED_JOURNAL_MODES:
            message = (
                f"unsupported journal_mode {journal_mode!r}; expected one of "
                f"{sorted(_SUPPORTED_JOURNAL_MODES)}"
            )
            raise AnswerStoreError(message)
        return journal_mode

    @staticmethod
    def _newer_message(version: int) -> str:
        return (
            f"the answer database at schema version {version} is newer than this build "
            f"understands (version {SCHEMA_VERSION}); refusing to open it for writing rather "
            "than risk writing a schema it cannot interpret back into research history"
        )

    @staticmethod
    def _unsupported_message(version: int) -> str:
        return (
            f"the answer database is at schema version {version}, older than this build's "
            f"version {SCHEMA_VERSION}, and no migration path is registered to bring it "
            "forward; the store refuses to mutate it silently"
        )

    @staticmethod
    def _missing_schema_message(version: int) -> str:
        return (
            f"the answer database reports schema version {version} but has no {ANSWER_TABLE} "
            "table; the schema is corrupt or was partially created, and the store refuses to "
            "write into it rather than compound the damage"
        )
