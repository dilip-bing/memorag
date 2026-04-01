"""Ingestion state tracker — prevents re-embedding unchanged files."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

logger = logging.getLogger("rag.ingest.tracker")


class IngestionTracker:
    def __init__(self, state_path: Path):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, dict] = self._load_state()

    def is_changed(self, path: Path) -> bool:
        """Return True if file is new or content has changed."""
        key = str(path.resolve())
        if key not in self._state:
            return True
        current_hash = self._hash_file(path)
        return current_hash != self._state[key].get("sha256")

    def mark_ingested(self, path: Path, collection: str) -> None:
        """Record file as ingested with its current hash."""
        key = str(path.resolve())
        self._state[key] = {
            "sha256": self._hash_file(path),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "collection": collection,
        }
        self._save_state()

    def remove_entry(self, path: Path) -> None:
        """Remove a file from tracker (e.g., when deleted)."""
        key = str(path.resolve())
        if key in self._state:
            del self._state[key]
            self._save_state()

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_state(self) -> Dict[str, dict]:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.warning("Tracker state corrupted, starting fresh.")
        return {}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)
        tmp.replace(self.state_path)
