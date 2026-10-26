"""Service layer for persistent CastingIQ analysis history."""

from typing import Any

from backend.app.history_store import (
    get_analysis,
    list_analyses,
    save_analysis,
)


def save_analysis_record(
    casting_input: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return save_analysis(casting_input, result)


def list_analysis_records(
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    return list_analyses(limit=limit, offset=offset)


def get_analysis_record(record_id: int) -> dict[str, Any] | None:
    return get_analysis(record_id)