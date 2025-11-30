import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "saved_analyses.json")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)


def _load_all() -> List[Dict[str, Any]]:
    if not os.path.exists(SAVE_PATH):
        return []
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_all(items: List[Dict[str, Any]]) -> None:
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def add_analysis(
    file_name: str,
    analysis_name: str,
    summary_row: Dict[str, Any],
    full_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Save a new analysis and return the saved record."""
    items = _load_all()
    analysis_id = str(uuid.uuid4())
    saved_at = datetime.now(timezone.utc).isoformat()

    record = {
        "id": analysis_id,
        "saved_at": saved_at,
        "file_name": file_name,
        "analysis_name": analysis_name,
        "summary": summary_row,     # used in the table
        "full_report": full_report, # used for re-rendering the full view
    }
    items.insert(0, record)
    _save_all(items)
    return record


def list_analyses() -> List[Dict[str, Any]]:
    return _load_all()


def get_analysis(analysis_id: str) -> Dict[str, Any] | None:
    for item in _load_all():
        if item.get("id") == analysis_id:
            return item
    return None

