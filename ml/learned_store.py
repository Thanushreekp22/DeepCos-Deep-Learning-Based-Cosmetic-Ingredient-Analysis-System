"""
Persistence for the AI-learned ingredient overlay.

Two views of the same data:

* the overlay JSON file (``data/knowledge_base/learned_ingredients.json``), read
  by :class:`ml.knowledge_base.IngredientKnowledgeBase`. A restarted process
  therefore resolves promoted ingredients locally, with no AI call;
* the MongoDB collection ``ingredient_knowledge_learned``, the queryable copy
  used by the API. It degrades to the JSON file whenever MongoDB is unreachable
  (exactly like the analyses and enrichment stores).

Only records produced by :func:`ml.ingredient_promotion.build_learned_entry` are
stored here: every one of them is labelled ``origin="ai-learned"`` and
``review_status="unreviewed"``.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from ml import config

COLLECTION = "ingredient_knowledge_learned"
DEFAULT_MONGO_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
DEFAULT_MONGO_DB = os.getenv("MONGODB_DB", "deepcos")
DEFAULT_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", "1500"))


def load_overlay_records(path: Path | str | None = None) -> list[dict]:
    """
    Read the overlay file (never raises - a broken file means "no overlay").

    Accepts either a plain list of records or ``{"entries": [...]}``.
    """
    overlay_path = Path(path or config.LEARNED_INGREDIENTS_PATH)
    if not overlay_path.exists():
        return []
    try:
        payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("[deepcos] learned ingredient overlay could not be read (ignored)")
        return []
    if isinstance(payload, dict):
        payload = payload.get("entries", [])
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


class LearnedIngredientStore:
    """JSON + MongoDB store for promoted, AI-learned knowledge-base entries."""

    def __init__(
        self,
        *,
        path: Path | str | None = None,
        mongo_uri: str | None = None,
        db_name: str | None = None,
        timeout_ms: int | None = None,
        enable_mongo: bool = True,
    ) -> None:
        self.path = Path(path or config.LEARNED_INGREDIENTS_PATH)
        self.mongo_uri = mongo_uri or DEFAULT_MONGO_URI
        self.db_name = db_name or DEFAULT_MONGO_DB
        self.timeout_ms = int(timeout_ms or DEFAULT_TIMEOUT_MS)
        self._lock = threading.Lock()
        self._collection = None
        self._backend = "json"
        self.connect_error: str | None = None
        if enable_mongo:
            self._connect()

    # -- connection ---------------------------------------------------------
    def _connect(self) -> None:
        try:
            from pymongo import MongoClient

            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=self.timeout_ms)
            client.admin.command("ping")
            collection = client[self.db_name][COLLECTION]
            collection.create_index("normalized_key", unique=True)
            self._collection = collection
            self._backend = "mongodb"
        except Exception as exc:  # noqa: BLE001 - degrade to the overlay file
            self._collection = None
            self._backend = "json"
            self.connect_error = f"{type(exc).__name__}: {exc}"
            print("[deepcos] learned overlay: MongoDB unavailable, using the JSON file")

    @property
    def backend(self) -> str:
        return self._backend

    # -- file helpers -------------------------------------------------------
    def _write_records(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, indent=1), encoding="utf-8")

    # -- CRUD ---------------------------------------------------------------
    def get(self, normalized_key: str) -> Optional[dict]:
        if not normalized_key:
            return None
        if self._backend == "mongodb" and self._collection is not None:
            try:
                doc = self._collection.find_one({"normalized_key": normalized_key})
                if doc:
                    doc.pop("_id", None)
                    return doc
            except Exception:
                pass
        for record in load_overlay_records(self.path):
            if record.get("normalized_key") == normalized_key:
                return record
        return None

    def all(self) -> list[dict]:
        """Every learned entry (the overlay file - the offline source of truth)."""
        return load_overlay_records(self.path)

    def save(self, record: dict) -> dict:
        """
        Insert or replace one entry.

        The overlay file is always written (the knowledge base reads it on
        startup); MongoDB is updated on a best-effort basis.
        """
        key = record.get("normalized_key", "")
        with self._lock:
            records = [
                r for r in load_overlay_records(self.path) if r.get("normalized_key") != key
            ]
            records.append(record)
            self._write_records(records)
            if self._backend == "mongodb" and self._collection is not None:
                try:
                    self._collection.update_one(
                        {"normalized_key": key}, {"$set": record}, upsert=True
                    )
                except Exception:
                    pass  # the file copy already holds the entry
        return record

    def list(self, limit: int = 25, offset: int = 0) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        if self._backend == "mongodb" and self._collection is not None:
            try:
                docs = []
                for doc in (
                    self._collection.find({}).sort("promoted_at", -1).skip(offset).limit(limit)
                ):
                    doc.pop("_id", None)
                    docs.append(doc)
                return docs
            except Exception:
                pass
        records = sorted(
            load_overlay_records(self.path), key=lambda r: r.get("promoted_at", ""), reverse=True
        )
        return records[offset : offset + limit]

    def delete(self, normalized_key: str) -> bool:
        removed = False
        with self._lock:
            records = load_overlay_records(self.path)
            kept = [r for r in records if r.get("normalized_key") != normalized_key]
            if len(kept) != len(records):
                self._write_records(kept)
                removed = True
            if self._backend == "mongodb" and self._collection is not None:
                try:
                    result = self._collection.delete_one({"normalized_key": normalized_key})
                    removed = removed or result.deleted_count > 0
                except Exception:
                    pass
        return removed

    def count(self) -> int:
        return len(load_overlay_records(self.path))

    def status(self) -> dict:
        return {"backend": self._backend, "count": self.count(), "path": str(self.path)}
