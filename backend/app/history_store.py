"""SQLite persistence for CastingIQ analysis history."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "castingiQ_history.sqlite3"
)

DB_PATH = Path(os.environ.get("CASTINGIQ_HISTORY_DB", str(DEFAULT_DB_PATH)))


def get_connection() -> sqlite3.Connection:
    """Open a database connection and return rows by column name."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_db() -> None:
    """Create the history table if it does not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                ),
                input_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )


def save_analysis(
    casting_input: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Persist an analysis input and its complete result."""
    initialize_db()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_history (input_json, result_json)
            VALUES (?, ?)
            """,
            (
                json.dumps(casting_input),
                json.dumps(result),
            ),
        )
        record_id = cursor.lastrowid
        row = connection.execute(
            """
            SELECT id, created_at, input_json, result_json
            FROM analysis_history
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

    return _format_record(row)


def list_analyses(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """Return recent history records, newest first."""
    initialize_db()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    with get_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM analysis_history"
        ).fetchone()[0]

        rows = connection.execute(
            """
            SELECT id, created_at, input_json, result_json
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return {
        "total": total,
        "items": [_format_record(row) for row in rows],
    }


def get_analysis(record_id: int) -> dict[str, Any] | None:
    """Return one saved analysis, or None if it does not exist."""
    initialize_db()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, created_at, input_json, result_json
            FROM analysis_history
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

    return _format_record(row) if row else None


def _format_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "input": json.loads(row["input_json"]),
        "result": json.loads(row["result_json"]),
    }
