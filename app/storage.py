from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_reports.json")


def _load_all_raw() -> List[Dict[str, Any]]:
    """Load all saved reports from disk. Return [] if file missing/broken."""
    if not os.path.exists(SAVE_PATH):
        return []

    try:
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_all_raw(items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def get_saved_reports() -> List[Dict[str, Any]]:
    """Return list of saved report metadata, newest first."""
    items = _load_all_raw()
    # sort by created_at descending if present
    def _key(item: Dict[str, Any]) -> str:
        return item.get("created_at", "")
    return sorted(items, key=_key, reverse=True)


def get_saved_report(report_id: str) -> Optional[Dict[str, Any]]:
    items = _load_all_raw()
    for item in items:
        if item.get("id") == report_id:
            return item
    return None


def add_saved_report(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new report. `summary` should already be a compact dict with:
      - file_name, grade, tir, above_range, below_range, sfr_index, created_at, etc.
      - raw_json: the full JSON result we already generate for /report/json

    Returns the stored item with an added 'id'.
    """
    items = _load_all_raw()
    new_item = {
        "id": summary.get("id") or str(uuid4()),
        "created_at": summary.get("created_at") or datetime.utcnow().isoformat() + "Z",
        **summary,
    }
    items.append(new_item)
    _save_all_raw(items)
    return new_item

