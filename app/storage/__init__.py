import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

STORAGE_FILE = os.path.join(os.path.dirname(__file__), "reports.json")

# Ensure file exists
if not os.path.exists(STORAGE_FILE):
    os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
    with open(STORAGE_FILE, "w") as f:
        json.dump([], f)


def _load() -> List[Dict[str, Any]]:
    """Load all reports from storage file."""
    if not os.path.exists(STORAGE_FILE):
        return []
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # Handle old dict format and convert to list
            if isinstance(data, dict):
                return [
                    {"id": rid, **r}
                    for rid, r in data.items()
                ]
            return []
    except Exception:
        return []


def _save(data: List[Dict[str, Any]]) -> None:
    """Save reports to storage file."""
    os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def add_saved_report(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new report. `summary` should already be a compact dict with:
      - file_name, grade, tir, above_range, below_range, sfr_index, created_at, etc.
      - raw_json: the full JSON result we already generate for /report/json
    
    Returns the stored item with an added 'id'.
    """
    items = _load()
    new_item = {
        "id": summary.get("id") or str(uuid4()),
        "created_at": summary.get("created_at") or datetime.utcnow().isoformat() + "Z",
        **summary,
    }
    items.append(new_item)
    _save(items)
    return new_item


def get_saved_reports() -> List[Dict[str, Any]]:
    """Return list of all saved reports (newest first)."""
    items = _load()
    # Sort by created_at descending if present
    def _key(item: Dict[str, Any]) -> str:
        return item.get("created_at", "")
    return sorted(items, key=_key, reverse=True)


def get_saved_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Return full detailed report by ID."""
    items = _load()
    for item in items:
        if item.get("id") == report_id:
            return item
    return None
