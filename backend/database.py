"""
Persistence layer for DeepCos.

Design
------
* MongoDB (pymongo) is the primary store: collection ``analyses``.
* If MongoDB is unreachable the store transparently falls back to a JSON file so
  the application keeps working in a demo / viva environment.
* Large base64 image previews are never stored in the database. They are
  decoded and written to ``artifacts/reports/analyses/<analysis_id>/`` and the
  report keeps an API URL for each asset. That keeps documents small and makes
  the images cacheable by the browser.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from backend import config

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except Exception:  # pragma: no cover - pymongo optional at runtime
    MongoClient = None  # type: ignore[assignment]
    PyMongoError = Exception  # type: ignore[assignment]

IMAGE_KEYS: tuple[str, ...] = (
    "original_png_base64",
    "detected_region_png_base64",
    "heatmap_png_base64",
    "cropped_region_png_base64",
)

# Fields too heavy for the history listing (kept only in the full document)
HEAVY_FIELDS: tuple[str, ...] = (
    "image_analysis",
    "explainability",
    "sequence",
    "ingredients",
    "preprocessing",
)


class AnalysisStore:
    """Stores DeepCos analysis reports in MongoDB with a JSON-file fallback."""

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        fallback_path: Path | None = None,
        asset_dir: Path | None = None,
    ) -> None:
        self.uri = uri or config.MONGODB_URI
        self.db_name = db_name or config.MONGODB_DB
        self.fallback_path = Path(fallback_path or config.JSON_STORE_PATH)
        self.asset_dir = Path(asset_dir or config.ANALYSIS_ASSET_DIR)
        self._lock = threading.Lock()
        self._client = None
        self._collection = None
        self._backend = "json"
        self.connect()

    # -- connection ---------------------------------------------------------
    def connect(self) -> None:
        """Try MongoDB; silently degrade to the JSON store when unavailable."""
        self.connect_error: str | None = None
        if MongoClient is None:
            self._backend = "json"
            self.connect_error = "pymongo is not installed"
            return
        try:
            client = MongoClient(self.uri, serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS)
            client.admin.command("ping")
            self._client = client
            self._collection = client[self.db_name]["analyses"]
            self._ensure_indexes(self._collection)
            self._backend = "mongodb"
            self._migrate_fallback()
        except Exception as exc:
            self._client = None
            self._collection = None
            self._backend = "json"
            self.connect_error = f"{type(exc).__name__}: {exc}"
            print(f"[deepcos] MongoDB unavailable ({self.connect_error}); using JSON store")

    @staticmethod
    def _ensure_indexes(collection) -> None:
        """
        Backfill legacy ``analysis_id`` values and create the unique
        ``analysis_id`` index.

        Documents saved by older builds may miss ``analysis_id``; duplicates
        would block the unique index, and even a single null would show up as
        an id-less row in the history listing, so ids are backfilled from
        ``_id`` on every start (two cheap indexed point queries). If the unique
        index still cannot be built a plain index is used instead.
        """
        legacy_ids = [doc["_id"] for doc in collection.find({"analysis_id": None}, {"_id": 1})]
        legacy_ids += [
            doc["_id"] for doc in collection.find({"analysis_id": ""}, {"_id": 1})
        ]
        for _id in legacy_ids:
            collection.update_one({"_id": _id}, {"$set": {"analysis_id": f"legacy-{_id}"}})
        try:
            collection.create_index("analysis_id", unique=True)
        except PyMongoError:
            collection.create_index("analysis_id")
        collection.create_index("created_at")

    def _migrate_fallback(self) -> None:
        """
        Reports saved while the JSON store was active would otherwise be lost
        once MongoDB takes over: upsert them (insert-only) so both eras of
        analyses stay visible in the history listing.
        """
        records = self._read_fallback()
        if not records or self._collection is None:
            return
        migrated = 0
        for record in records:
            analysis_id = record.get("analysis_id")
            if not analysis_id:
                continue
            record["created_at"] = _normalize_created_at(record.get("created_at"))
            try:
                self._collection.update_one(
                    {"analysis_id": analysis_id}, {"$setOnInsert": record}, upsert=True
                )
                migrated += 1
            except PyMongoError as exc:
                print(f"[deepcos] fallback migration stopped: {exc}")
                return
        if migrated:
            print(f"[deepcos] migrated {migrated} report(s) from the JSON store to MongoDB")

    @property
    def backend(self) -> str:
        return self._backend

    # -- asset handling ------------------------------------------------------
    def _extract_assets(self, report: dict) -> dict:
        """
        Move ``*_png_base64`` previews out of the report into PNG files and
        replace them with API URLs. Operates on a JSON deep copy so the caller
        can still use the in-memory previews (base64) after saving.
        """
        report = json.loads(json.dumps(report))  # deep copy (JSON-safe)
        image_analysis = report.get("image_analysis") or {}
        analysis_id = report.get("analysis_id") or "analysis"
        folder = self.asset_dir / analysis_id
        urls: dict[str, str] = {}
        for key in IMAGE_KEYS:
            encoded = image_analysis.pop(key, None)
            if not encoded:
                continue
            payload = encoded.split(",", 1)[-1] if "," in encoded else encoded
            try:
                folder.mkdir(parents=True, exist_ok=True)
                import base64 as _b64

                (folder / f"{key}.png").write_bytes(_b64.b64decode(payload))
            except Exception:
                continue
            urls[key] = f"/api/analyses/{analysis_id}/image/{key}"
        if urls:
            image_analysis["assets"] = urls
            report["image_analysis"] = image_analysis
        return report

    def asset_path(self, analysis_id: str, name: str) -> Optional[Path]:
        if name not in IMAGE_KEYS:
            return None
        path = self.asset_dir / analysis_id / f"{name}.png"
        return path if path.exists() else None

    # -- CRUD ---------------------------------------------------------------
    def save(self, report: dict) -> dict:
        """Persist a full report; returns the stored (asset-refactored) copy."""
        document = self._extract_assets(report)
        document["created_at"] = _normalize_created_at(document.get("created_at"))
        analysis_id = document.get("analysis_id")
        with self._lock:
            if self._backend == "mongodb" and self._collection is not None:
                try:
                    self._collection.replace_one(
                        {"analysis_id": analysis_id}, document, upsert=True
                    )
                except PyMongoError:
                    self._backend = "json"  # degrade but never lose the report
                    records = self._read_fallback()
                    records = [r for r in records if r.get("analysis_id") != analysis_id]
                    records.append(document)
                    self._write_fallback(records)
            else:
                records = self._read_fallback()
                records = [r for r in records if r.get("analysis_id") != analysis_id]
                records.append(document)
                self._write_fallback(records)
        return document

    def get(self, analysis_id: str) -> Optional[dict]:
        if self._backend == "mongodb" and self._collection is not None:
            try:
                doc = self._collection.find_one({"analysis_id": analysis_id})
                return None if doc is None else _strip_id(doc)
            except PyMongoError:
                pass
        for record in self._read_fallback():
            if record.get("analysis_id") == analysis_id:
                return record
        return None

    def status(self) -> dict:
        return {
            "backend": self._backend,
            "uri": self.uri if self._backend == "mongodb" else None,
            "database": self.db_name if self._backend == "mongodb" else None,
            "fallback_path": None if self._backend == "mongodb" else str(self.fallback_path),
            "error": getattr(self, "connect_error", None),
        }

    # -- JSON fallback helpers ---------------------------------------------
    def _read_fallback(self) -> list[dict]:
        if not self.fallback_path.exists():
            return []
        try:
            return json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _write_fallback(self, records: list[dict]) -> None:
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path.write_text(json.dumps(records, indent=1), encoding="utf-8")

    # -- queries -------------------------------------------------------------
    def list(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Most recent analyses first, without the heavy sections."""
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        if self._backend == "mongodb" and self._collection is not None:
            try:
                cursor = (
                    self._collection.find({}, _projection())
                    .sort("created_at", -1)
                    .skip(offset)
                    .limit(limit)
                )
                return [_strip_id(doc) for doc in cursor]
            except PyMongoError:
                pass
        records = sorted(
            self._read_fallback(), key=lambda r: r.get("created_at", ""), reverse=True
        )
        return [_summarise(record) for record in records[offset : offset + limit]]

    def delete(self, analysis_id: str) -> bool:
        removed = False
        with self._lock:
            if self._backend == "mongodb" and self._collection is not None:
                try:
                    removed = (
                        self._collection.delete_one({"analysis_id": analysis_id}).deleted_count
                        > 0
                    )
                except PyMongoError:
                    removed = False
            records = self._read_fallback()
            kept = [r for r in records if r.get("analysis_id") != analysis_id]
            if len(kept) != len(records):
                self._write_fallback(kept)
                removed = True
        folder = self.asset_dir / analysis_id
        if folder.exists():
            for png in folder.glob("*.png"):
                try:
                    png.unlink()
                except OSError:
                    pass
            try:
                folder.rmdir()
            except OSError:
                pass
        return removed

    def count(self) -> int:
        if self._backend == "mongodb" and self._collection is not None:
            try:
                return int(self._collection.estimated_document_count())
            except PyMongoError:
                pass
        return len(self._read_fallback())


# -- module helpers ---------------------------------------------------------
def _normalize_created_at(value: Any) -> Any:
    """
    Canonicalise timestamps so listings sort correctly as strings.

    Reports built by different code paths use ``"2026-09-29 10:48:32"`` (space)
    while older documents use ``"2026-09-29T05:17:33.351231+00:00"`` (T).
    Lexicographic descending order mixes those up, so the space form is
    converted to the ISO-8601 ``T`` separator.
    """
    if isinstance(value, str) and len(value) >= 19 and value[10] == " " and value[:4].isdigit():
        return value.replace(" ", "T", 1)
    return value


def _strip_id(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


def _projection() -> dict[str, Any]:
    """MongoDB projection that keeps only the light fields for listings."""
    return {field: 0 for field in HEAVY_FIELDS}


def _summarise(record: dict) -> dict:
    """Drop heavy sections from a record read from the JSON fallback."""
    return {k: v for k, v in record.items() if k not in HEAVY_FIELDS}


_store: Optional["AnalysisStore"] = None
_store_lock = threading.Lock()


def get_store() -> AnalysisStore:
    """Process-wide singleton used by the routers."""
    global _store
    with _store_lock:
        if _store is None:
            _store = AnalysisStore()
        return _store

