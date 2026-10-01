"""
Optional AI ingredient enrichment via the Groq API  (feature is OFF by default).

Flow for every ingredient the local knowledge base could not resolve:

    normalised key -> persistent cache (MongoDB `ingredient_enrichments`,
                      JSON file fallback) -> Groq chat completion -> strict
                      Pydantic validation -> cache write -> report

Design rules
------------
* The local knowledge base stays the only trusted source. AI data is cached
  **separately**, labelled with source / model / review status, and never
  rewrites the verified reference files (``ingredients.json`` /
  ``concern_rules.json``) or the deterministic concern engine.
* When ``AI_PROMOTE_TO_KB`` is enabled, a *validated* lookup result for an
  ingredient the local KB does not document is additionally persisted to the
  separate, clearly-labelled **AI-learned overlay**
  (``data/knowledge_base/learned_ingredients.json``), so later runs resolve that
  ingredient locally instead of calling the provider again. Promoted entries are
  always stamped ``origin="ai-learned"`` / ``review_status="unreviewed"`` and can
  be listed and deleted through ``/api/knowledge/learned``.
* The feature is disabled unless ``AI_ENRICHMENT_ENABLED`` is true **and** a
  ``GROQ_API_KEY`` is configured; without both, :func:`get_enricher` returns
  ``None`` and the analysis behaves exactly as before.
* The Groq client is injectable (:class:`GroqChatClient` conforms to
  :class:`ChatClient`), so tests can mock it and never hit the network.
* Failures (rate limits, timeouts, malformed JSON, outages) degrade to a
  "failed" enrichment item - they never fail the surrounding analysis.
* The API key is never logged, never stored and never returned by the API.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import requests

from backend import config
from backend.schemas import EnrichmentData
from ml.knowledge_base import (
    ResolvedIngredient,
    get_knowledge_base,
    normalize_ingredient_name,
)


PROVIDER = "groq"
SOURCE = "groq"
SCHEMA_VERSION = 1
REVIEW_STATUS = "unreviewed"
COLLECTION = "ingredient_enrichments"

# Conservative system prompt: JSON only, informational only, no medical or
# regulatory verdicts, no invented citations, "unknown" when unsure.
SYSTEM_PROMPT = """You are a reference assistant for cosmetic ingredients. \
Reply with ONE JSON object only - no markdown fences, no commentary.

Identify the ingredient as exactly one of: "inci" (a standard INCI name), \
"trade_name" (a branded/trade name), "botanical" (a botanical or plant name), \
or "unknown".

Return exactly these keys:
{
  "inci_name": string,
  "identification": "inci" | "trade_name" | "botanical" | "unknown",
  "aliases": [string],
  "description": string,
  "functions": [string],
  "possible_concerns": [string],
  "regulatory_note": string,
  "confidence": "low" | "medium" | "high",
  "evidence_note": string
}

Rules:
- This is informational content only. Never give medical advice and never state \
or imply that an ingredient is safe or unsafe.
- Never make definitive regulatory claims; keep regulatory_note general and \
non-binding, or "" when unknown.
- Use "" for strings and [] for lists whenever you cannot verify something; \
use "unknown" when you are unsure of the identification.
- possible_concerns are informational possibilities, not confirmed hazards.
- Never invent citations, studies, statistics or sources.
- functions are short cosmetic function words such as "emollient" or "preservative".
- description must be at most 40 words; keep every field concise.
- confidence describes how well established the information is, not how \
harmful anything is."""


class GroqClientError(RuntimeError):
    """Raised when the Groq API cannot be used (network, rate limit, HTTP error)."""


class ChatClient(Protocol):
    """Minimal client contract so tests can inject a fake."""

    def complete(self, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# Groq HTTP client
# ---------------------------------------------------------------------------
class GroqChatClient:
    """Thin ``requests``-based chat-completion client for the Groq API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else config.GROQ_API_KEY).strip()
        self.model = model or config.GROQ_MODEL
        self.url = url or config.GROQ_API_URL
        self.timeout = float(timeout if timeout is not None else config.GROQ_TIMEOUT_SECONDS)

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise GroqClientError("no API key configured")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise GroqClientError("request timed out") from exc
        except requests.RequestException as exc:
            # Never include headers (which carry the key) in the message.
            raise GroqClientError(f"network error ({type(exc).__name__})") from exc

        if response.status_code == 429:
            raise GroqClientError("rate limited by the provider (HTTP 429)")
        if response.status_code in (401, 403):
            raise GroqClientError(
                f"provider rejected the configured credentials (HTTP {response.status_code})"
            )
        if response.status_code >= 400:
            raise GroqClientError(f"provider error (HTTP {response.status_code})")

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GroqClientError("unexpected response shape from the provider") from exc
        if not isinstance(content, str) or not content.strip():
            raise GroqClientError("empty response from the provider")
        return content


def parse_enrichment_content(content: str) -> EnrichmentData:
    """
    Parse and validate one model response.

    Tolerates markdown fences and stray chatter around the JSON object, then
    applies the strict :class:`EnrichmentData` schema. Raises on anything that
    cannot be validated - callers treat that as a failed (not fatal) lookup.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("enrichment response must be a JSON object")
    return EnrichmentData.model_validate(payload)


def _safe_error(exc: BaseException) -> str:
    """Short error text with the API key defensively scrubbed out."""
    message = f"{type(exc).__name__}: {exc}"
    key = config.GROQ_API_KEY
    if key and key in message:
        message = message.replace(key, "***")
    return message[:200]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_expired(record: dict, now: datetime | None = None) -> bool:
    raw = str(record.get("expires_at", "") or "")
    if not raw:
        return True
    try:
        expires = datetime.fromisoformat(raw)
    except ValueError:
        return True
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires <= (now or _utcnow())


# ---------------------------------------------------------------------------
# Persistent cache (MongoDB with JSON fallback, mirroring AnalysisStore)
# ---------------------------------------------------------------------------
class EnrichmentStore:
    """
    Stores enrichment records in MongoDB collection ``ingredient_enrichments``
    and falls back to a JSON file when MongoDB is unavailable.
    """

    def __init__(
        self,
        *,
        fallback_path: Path | str | None = None,
        mongo_uri: str | None = None,
        db_name: str | None = None,
        enable_mongo: bool = True,
    ) -> None:
        self.fallback_path = Path(fallback_path or config.ENRICHMENT_JSON_STORE_PATH)
        self._lock = threading.Lock()
        self._collection = None
        self._backend = "json"
        self.connect_error: str | None = None
        if enable_mongo:
            self._connect(mongo_uri or config.MONGODB_URI, db_name or config.MONGODB_DB)

    # -- connection ---------------------------------------------------------
    def _connect(self, uri: str, db_name: str) -> None:
        try:
            from pymongo import MongoClient

            client = MongoClient(uri, serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS)
            client.admin.command("ping")
            collection = client[db_name][COLLECTION]
            collection.create_index("normalized_key", unique=True)
            self._collection = collection
            self._backend = "mongodb"
        except Exception as exc:  # noqa: BLE001 - degrade exactly like AnalysisStore
            self._collection = None
            self._backend = "json"
            self.connect_error = f"{type(exc).__name__}: {exc}"
            print("[deepcos] enrichment cache: MongoDB unavailable, using JSON store")

    @property
    def backend(self) -> str:
        return self._backend

    # -- JSON fallback helpers ---------------------------------------------
    def _read_json(self) -> list[dict]:
        if not self.fallback_path.exists():
            return []
        try:
            records = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return records if isinstance(records, list) else []

    def _write_json(self, records: list[dict]) -> None:
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path.write_text(json.dumps(records, indent=1), encoding="utf-8")

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
        for record in self._read_json():
            if record.get("normalized_key") == normalized_key:
                return record
        return None

    def save(self, record: dict) -> dict:
        """Insert or replace the record for its ``normalized_key``."""
        key = record.get("normalized_key", "")
        with self._lock:
            if self._backend == "mongodb" and self._collection is not None:
                try:
                    self._collection.update_one(
                        {"normalized_key": key}, {"$set": record}, upsert=True
                    )
                    return record
                except Exception:
                    pass  # fall through to the JSON store
            records = [r for r in self._read_json() if r.get("normalized_key") != key]
            records.append(record)
            self._write_json(records)
        return record

    def list(self, limit: int = 25, offset: int = 0) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        if self._backend == "mongodb" and self._collection is not None:
            try:
                cursor = (
                    self._collection.find({}).sort("updated_at", -1).skip(offset).limit(limit)
                )
                docs = []
                for doc in cursor:
                    doc.pop("_id", None)
                    docs.append(doc)
                return docs
            except Exception:
                pass
        records = sorted(
            self._read_json(), key=lambda r: r.get("updated_at", ""), reverse=True
        )
        return records[offset : offset + limit]

    def delete(self, normalized_key: str) -> bool:
        removed = False
        with self._lock:
            if self._backend == "mongodb" and self._collection is not None:
                try:
                    removed = (
                        self._collection.delete_one({"normalized_key": normalized_key}).deleted_count
                        > 0
                    )
                except Exception:
                    removed = False
            records = self._read_json()
            kept = [r for r in records if r.get("normalized_key") != normalized_key]
            if len(kept) != len(records):
                self._write_json(kept)
                removed = True
        return removed

    def count(self) -> int:
        if self._backend == "mongodb" and self._collection is not None:
            try:
                return int(self._collection.estimated_document_count())
            except Exception:
                pass
        return len(self._read_json())

    def status(self) -> dict:
        return {"backend": self._backend, "count": self.count()}


# ---------------------------------------------------------------------------
# Enrichment orchestration
# ---------------------------------------------------------------------------
class IngredientEnricher:
    """Cache-first enrichment of single unresolved ingredient names."""

    def __init__(
        self,
        *,
        client: ChatClient | None = None,
        store: EnrichmentStore | None = None,
        max_items: int | None = None,
        ttl_days: int | None = None,
        promotion: Any = None,
    ) -> None:
        self.client = client or GroqChatClient()
        self.store = store if store is not None else get_enrichment_store()
        self.max_items = (
            config.AI_ENRICHMENT_MAX_ITEMS if max_items is None else max(0, int(max_items))
        )
        self.ttl_days = config.AI_ENRICHMENT_CACHE_TTL_DAYS if ttl_days is None else ttl_days
        # Optional backend.services.kb_promotion.KnowledgePromoter. ``None`` (the
        # default) means results are only cached and shown, never persisted into
        # the knowledge base - which is also what keeps tests side-effect free.
        self.promotion = promotion

    # -- single ingredient ---------------------------------------------------
    def _record(
        self, key: str, requested_name: str, data: EnrichmentData, previous: dict | None
    ) -> dict:
        now = _utcnow()
        created_at = (previous or {}).get("created_at") or now.isoformat()
        return {
            "normalized_key": key,
            "requested_name": requested_name,
            "data": data.model_dump(),
            "source": SOURCE,
            "provider": PROVIDER,
            "model": config.GROQ_MODEL,
            "created_at": created_at,
            "updated_at": now.isoformat(),
            "expires_at": (now + timedelta(days=self.ttl_days)).isoformat(),
            "review_status": REVIEW_STATUS,
            "schema_version": SCHEMA_VERSION,
        }

    def _outcome(
        self,
        status: str,
        *,
        key: str,
        raw: str,
        record: dict | None,
        error: str | None = None,
    ) -> dict:
        data = (record or {}).get("data")
        outcome = {
            "status": status,
            "key": key,
            "raw": raw,
            "data": data if isinstance(data, dict) else None,
            "source": (record or {}).get("source"),
            "provider": (record or {}).get("provider"),
            "model": (record or {}).get("model"),
            "review_status": (record or {}).get("review_status"),
        }
        if error:
            outcome["error"] = error
        return outcome

    def _fetch(self, key: str, raw: str, previous: dict | None) -> dict:
        """Call Groq, validate, persist. Never raises."""
        try:
            content = self.client.complete(SYSTEM_PROMPT, f"Ingredient: {raw}")
            data = parse_enrichment_content(content)
        except Exception as exc:  # noqa: BLE001 - enrichment must never break analysis
            return self._outcome(
                "failed", key=key, raw=raw, record=previous, error=_safe_error(exc)
            )
        record = self._record(key, raw, data, previous)
        try:
            self.store.save(record)
        except Exception as exc:  # noqa: BLE001 - cache write failure is not fatal
            print(f"[deepcos] enrichment cache write failed: {type(exc).__name__}")
        return self._outcome("enriched", key=key, raw=raw, record=record)

    def enrich(self, raw_name: str, key: str | None = None) -> dict:
        """
        Resolve one ingredient name: cache first, then Groq.

        Returns a structured outcome with ``status`` one of
        ``enriched`` / ``cached`` / ``stale_cache`` / ``failed``.
        """
        key = key if key is not None else normalize_ingredient_name(raw_name)
        if not key:
            return self._outcome(
                "failed", key="", raw=raw_name, record=None, error="empty ingredient name"
            )

        cached = self.store.get(key)
        if cached and cached.get("data") and not _is_expired(cached):
            return self._outcome("cached", key=key, raw=raw_name, record=cached)

        outcome = self._fetch(key, raw_name, cached)
        if outcome["status"] == "enriched":
            return outcome
        if cached and cached.get("data"):
            # Expired (or refresh failed) - serve the old cache with an indicator.
            return self._outcome(
                "stale_cache", key=key, raw=raw_name, record=cached, error=outcome.get("error")
            )
        return outcome

    # -- knowledge-base promotion -------------------------------------------
    def _promotion_block(self, key: str, raw: str, outcome: dict) -> dict:
        """
        Offer one validated result to the promoter (opt-in) and describe the
        outcome for the report. Never raises.
        """
        if self.promotion is None:
            return {
                "promoted": False,
                "reason": "knowledge-base promotion is not active for this enricher",
                "origin": None,
                "review_status": None,
                "functions": [],
                "concerns": [],
            }
        try:
            return self.promotion.promote(raw, key, outcome.get("data"), outcome)
        except Exception as exc:  # noqa: BLE001 - promotion is best-effort only
            print(f"[deepcos] knowledge-base promotion skipped ({type(exc).__name__})")
            return {
                "promoted": False,
                "reason": f"promotion failed ({type(exc).__name__})",
                "origin": None,
                "review_status": None,
                "functions": [],
                "concerns": [],
            }

    # -- report block --------------------------------------------------------
    def enrich_unresolved(self, resolution: Sequence[ResolvedIngredient]) -> dict:
        """
        Build the ``ai_enrichment`` report block for unresolved ingredients.

        Only unresolved names are considered, deduplicated by normalised key
        and capped at ``AI_ENRICHMENT_MAX_ITEMS`` API lookups per analysis.
        Successful results are also handed to the optional knowledge-base
        promoter, which persists validated ones in the AI-learned overlay.
        """
        items: list[dict] = []
        seen: set[str] = set()
        requested = resolved = cached_count = failed = promoted_count = 0

        for item in resolution:
            if item.resolved or not item.key or item.key in seen:
                continue
            if requested >= self.max_items:
                break
            seen.add(item.key)
            requested += 1

            outcome = self.enrich(item.raw, key=item.key)
            entry: dict[str, Any] = {
                "ingredient": item.raw,
                "normalized_name": item.key,
                "status": outcome["status"],
                "source": outcome["source"] if outcome["data"] else None,
                "provider": outcome["provider"] if outcome["data"] else None,
                "model": outcome["model"] if outcome["data"] else None,
                "review_status": outcome["review_status"] if outcome["data"] else None,
                "data": outcome["data"],
            }
            if outcome["data"]:
                resolved += 1
                if outcome["status"] in ("cached", "stale_cache"):
                    cached_count += 1
                # Persist validated results (opt-in) and report what happened.
                entry["knowledge_base"] = self._promotion_block(item.key, item.raw, outcome)
                if entry["knowledge_base"].get("promoted"):
                    promoted_count += 1
            else:
                failed += 1
                entry["error"] = outcome.get("error", "enrichment unavailable")
            items.append(entry)

        return {
            "enabled": True,
            "requested_count": requested,
            "resolved_count": resolved,
            "cached_count": cached_count,
            "failed_count": failed,
            "promotion_enabled": self.promotion is not None,
            "promoted_count": promoted_count,
            "items": items,
        }


# ---------------------------------------------------------------------------
# Module-level accessors
# ---------------------------------------------------------------------------
def enrichment_enabled() -> bool:
    """True only when the feature is switched on AND a key is configured."""
    return bool(config.AI_ENRICHMENT_ENABLED and config.GROQ_API_KEY)


_store: Optional[EnrichmentStore] = None
_store_lock = threading.Lock()


def get_enrichment_store() -> EnrichmentStore:
    """Process-wide singleton cache store."""
    global _store
    with _store_lock:
        if _store is None:
            _store = EnrichmentStore()
        return _store


_enricher: Optional[IngredientEnricher] = None
_enricher_lock = threading.Lock()


def get_enricher() -> Optional[IngredientEnricher]:
    """
    Singleton enricher for the analysis pipeline, or ``None`` when the feature
    is disabled - in that case no Groq call can ever happen.

    When ``AI_PROMOTE_TO_KB`` is enabled the enricher also receives the
    knowledge-base promoter, which persists validated results in the
    AI-learned overlay (see :mod:`backend.services.kb_promotion`).
    """
    if not enrichment_enabled():
        return None
    global _enricher
    with _enricher_lock:
        if _enricher is None:
            promotion = None
            if config.AI_PROMOTE_TO_KB:
                try:
                    from backend.services.kb_promotion import get_promoter

                    promotion = get_promoter()
                except Exception as exc:  # noqa: BLE001 - enrichment still works
                    print(f"[deepcos] knowledge-base promotion unavailable ({type(exc).__name__})")
                    promotion = None
            _enricher = IngredientEnricher(promotion=promotion)
        return _enricher


def enrichment_status() -> dict:
    """Health/status payload - booleans only, never the API key itself."""
    try:
        store_status = get_enrichment_store().status()
    except Exception:  # noqa: BLE001
        store_status = {"backend": "unknown", "count": 0}
    try:
        learned_count = get_knowledge_base().learned_count()
    except Exception:  # noqa: BLE001
        learned_count = 0
    return {
        "enabled": enrichment_enabled(),
        "api_key_configured": bool(config.GROQ_API_KEY),
        "provider": PROVIDER,
        "model": config.GROQ_MODEL,
        "max_items": config.AI_ENRICHMENT_MAX_ITEMS,
        "cache_ttl_days": config.AI_ENRICHMENT_CACHE_TTL_DAYS,
        "cache_backend": store_status.get("backend", "unknown"),
        "cache_count": store_status.get("count", 0),
        # Knowledge-base promotion: validated AI results can be persisted in the
        # separate AI-learned overlay (never in the verified reference files).
        "promote_to_kb": bool(config.AI_PROMOTE_TO_KB),
        "promote_min_confidence": config.AI_PROMOTE_MIN_CONFIDENCE,
        "learned_count": learned_count,
    }


def _public_record(record: dict) -> dict:
    """Cache record as returned by the API (it contains no secrets)."""
    out = dict(record)
    out["expired"] = _is_expired(record)
    return out


def get_cached_enrichment(name: str) -> Optional[dict]:
    """Look up one cached enrichment by raw or normalised ingredient name."""
    key = normalize_ingredient_name(name)
    if not key:
        return None
    record = get_enrichment_store().get(key)
    return _public_record(record) if record else None


def list_cached_enrichments(limit: int = 25, offset: int = 0) -> list[dict]:
    return [_public_record(record) for record in get_enrichment_store().list(limit, offset)]


def delete_cached_enrichment(name: str) -> bool:
    key = normalize_ingredient_name(name)
    return bool(key) and get_enrichment_store().delete(key)
