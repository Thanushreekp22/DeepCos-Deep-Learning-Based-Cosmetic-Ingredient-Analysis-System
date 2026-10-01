"""
Focused tests for the optional AI ingredient enrichment feature.

Every Groq interaction is mocked - these tests never make real API requests.
Covered cases (see the task list):

* disabled feature / missing API key never calls Groq and never fails analysis
* local knowledge-base matches are never sent to Groq
* duplicate unknown names are enriched once; cache hits skip the API
* expired cache refresh, with a stale-cache fallback when refresh fails
* malformed provider JSON and timeouts degrade safely
* AI results never touch deterministic concerns or model predictions
* JSON fallback works without MongoDB; reports carry ``ai_enrichment``
* the API key is never returned by an endpoint or written to a log/record
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from backend import config
import backend.services.ingredient_enrichment as ie
from ml.knowledge_base import get_knowledge_base

# A name the local knowledge base does NOT document (verified in the tests).
UNKNOWN_A = "Xylium Extract"
UNKNOWN_A_KEY = "xylium extract"

GOOD_PAYLOAD = {
    "inci_name": UNKNOWN_A,
    "identification": "inci",
    "aliases": ["xylium"],
    "description": "A fictional example ingredient used for enrichment tests.",
    "functions": ["emollient"],
    "possible_concerns": ["fragrance-allergen"],
    "regulatory_note": "",
    "confidence": "low",
    "evidence_note": "Test fixture, not a real source.",
}
GOOD_JSON = json.dumps(GOOD_PAYLOAD)


class FakeClient:
    """Injected stand-in for GroqChatClient: records calls, never hits the net."""

    def __init__(self, content: str | None = None, error: BaseException | None = None) -> None:
        self.content = GOOD_JSON if content is None else content
        self.error = error
        self.calls: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self.error is not None:
            raise self.error
        return self.content


@pytest.fixture()
def json_store(tmp_path):
    """Enrichment cache on the JSON fallback path (MongoDB disabled)."""
    return ie.EnrichmentStore(fallback_path=tmp_path / "ingredient_enrichments.json", enable_mongo=False)


def seed_record(store, key: str = UNKNOWN_A_KEY, *, expired: bool = False, data: dict | None = None) -> dict:
    now = datetime.now(timezone.utc)
    record = {
        "normalized_key": key,
        "requested_name": UNKNOWN_A,
        "data": dict(data or GOOD_PAYLOAD),
        "source": "groq",
        "provider": "groq",
        "model": "fake-model",
        "created_at": (now - timedelta(days=60)).isoformat(),
        "updated_at": (now - timedelta(days=60)).isoformat(),
        "expires_at": (now - timedelta(days=1) if expired else now + timedelta(days=30)).isoformat(),
        "review_status": "unreviewed",
        "schema_version": ie.SCHEMA_VERSION,
    }
    store.save(record)
    return record


def make_enricher(client=None, store=None, **kwargs) -> ie.IngredientEnricher:
    return ie.IngredientEnricher(client=client or FakeClient(), store=store, **kwargs)


def resolve(names: list[str]):
    return get_knowledge_base().resolve_all(names)


# ---------------------------------------------------------------------------
# 1-2. Feature flag: disabled / missing key
# ---------------------------------------------------------------------------
def test_disabled_feature_never_calls_groq(monkeypatch):
    monkeypatch.setattr(config, "AI_ENRICHMENT_ENABLED", False)
    monkeypatch.setattr(config, "GROQ_API_KEY", "sk-test-key")
    assert ie.enrichment_enabled() is False
    assert ie.get_enricher() is None  # no enricher exists -> Groq cannot be reached

    # Enabled flag alone is not enough: without a key the feature stays off.
    monkeypatch.setattr(config, "AI_ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    assert ie.enrichment_enabled() is False
    assert ie.get_enricher() is None


def test_missing_api_key_does_not_fail_analysis():
    from ml.inference import _run_enrichment

    # enricher=None (the disabled path) must still yield a well-formed block.
    block = _run_enrichment(None, resolve([UNKNOWN_A]))
    assert block["enabled"] is False
    assert block["requested_count"] == 0
    assert block["items"] == []
    assert block["disclaimer"]


# ---------------------------------------------------------------------------
# 3-4. Only unresolved, only once
# ---------------------------------------------------------------------------
def test_local_kb_matches_do_not_call_groq(json_store):
    client = FakeClient()
    enricher = make_enricher(client, json_store)
    block = enricher.enrich_unresolved(resolve(["Aqua", "Glycerin", "Niacinamide"]))
    assert client.calls == []
    assert block["requested_count"] == 0
    assert block["items"] == []


def test_duplicate_unknown_names_enriched_once(json_store):
    client = FakeClient()
    enricher = make_enricher(client, json_store)
    resolution = resolve([UNKNOWN_A, f"{UNKNOWN_A} (5%)", UNKNOWN_A.lower(), "Aqua"])
    assert all(not item.resolved for item in resolution[:3])  # none hit the KB
    block = enricher.enrich_unresolved(resolution)
    assert block["requested_count"] == 1
    assert len(block["items"]) == 1
    assert len(client.calls) == 1
    assert json_store.count() == 1


def test_max_items_caps_requests(json_store):
    client = FakeClient()
    enricher = make_enricher(client, json_store, max_items=1)
    block = enricher.enrich_unresolved(resolve([UNKNOWN_A, "Zorbium"]))
    assert block["requested_count"] == 1
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# 5-6. Cache behaviour
# ---------------------------------------------------------------------------
def test_cached_record_avoids_api_call(json_store):
    seed_record(json_store)
    client = FakeClient()
    enricher = make_enricher(client, json_store)
    block = enricher.enrich_unresolved(resolve([UNKNOWN_A]))
    assert client.calls == []
    assert block["cached_count"] == 1
    assert block["resolved_count"] == 1
    assert block["items"][0]["status"] == "cached"
    assert block["items"][0]["data"]["description"] == GOOD_PAYLOAD["description"]


def test_expired_cache_is_refreshed(json_store):
    seed_record(json_store, expired=True)
    client = FakeClient()
    enricher = make_enricher(client, json_store)
    block = enricher.enrich_unresolved(resolve([UNKNOWN_A]))
    assert len(client.calls) == 1  # refresh attempt happened
    assert block["items"][0]["status"] == "enriched"
    stored = json_store.get(UNKNOWN_A_KEY)
    assert datetime.fromisoformat(stored["expires_at"]) > datetime.now(timezone.utc)


def test_expired_cache_served_when_refresh_fails(json_store):
    seed_record(json_store, expired=True)
    client = FakeClient(error=RuntimeError("provider down"))
    enricher = make_enricher(client, json_store)
    block = enricher.enrich_unresolved(resolve([UNKNOWN_A]))
    assert len(client.calls) == 1
    item = block["items"][0]
    assert item["status"] == "stale_cache"  # "expired cache" indicator
    assert item["data"] is not None  # old cached result still shown
    assert block["resolved_count"] == 1
    assert block["cached_count"] == 1


# ---------------------------------------------------------------------------
# 7-8. Malformed responses and timeouts degrade safely
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("content", ["this is not json", "```json\n[1,2]\n```", ""])
def test_malformed_groq_json_is_rejected_safely(json_store, content):
    client = FakeClient(content=content)
    enricher = make_enricher(client, json_store)
    block = enricher.enrich_unresolved(resolve([UNKNOWN_A]))
    assert block["failed_count"] == 1
    assert block["resolved_count"] == 0
    assert block["items"][0]["data"] is None
    assert json_store.get(UNKNOWN_A_KEY) is None  # nothing invalid is cached


def test_groq_timeout_does_not_fail_analysis(json_store):
    import requests

    client = FakeClient(error=requests.Timeout("timed out"))
    enricher = make_enricher(client, json_store)
    block = enricher.enrich_unresolved(resolve([UNKNOWN_A]))  # must not raise
    assert block["failed_count"] == 1
    assert block["items"][0]["status"] == "failed"

    # ...and a completely broken enricher must not break inference either.
    from ml.inference import _run_enrichment

    class Exploding:
        def enrich_unresolved(self, resolution):
            raise RuntimeError("boom")

    fallback = _run_enrichment(Exploding(), resolve([UNKNOWN_A]))
    assert fallback["enabled"] is False
    assert fallback["items"] == []


# ---------------------------------------------------------------------------
# 9-10-11. Full analysis: isolation, report field, JSON fallback
# ---------------------------------------------------------------------------
ANALYSIS_TEXT = f"Aqua, Glycerin, Niacinamide, {UNKNOWN_A}"
_COMPARED_FIELDS = (
    "concerns",
    "profile_scores",
    "product",
    "sequence",
    "ingredients",
    "function_histogram",
    "key_ingredients",
    "explainability",
    "knowledge_base",
    "model",
    "warnings",
)


def test_ai_results_do_not_modify_concerns_or_model_predictions(json_store):
    """The heart of the safety rule: AI data only adds a separate block."""
    from ml.inference import analyze_ingredients

    baseline = analyze_ingredients(ANALYSIS_TEXT, explain=False, enricher=None)
    enriched = analyze_ingredients(
        ANALYSIS_TEXT, explain=False, enricher=make_enricher(FakeClient(), json_store)
    )

    for field in _COMPARED_FIELDS:
        assert baseline[field] == enriched[field], f"{field} changed when enrichment ran"

    # The AI-suggested concern tag must not leak into the deterministic screening.
    assert enriched["ai_enrichment"]["items"][0]["data"]["possible_concerns"] == [
        "fragrance-allergen"
    ]
    assert all(
        f["tag"] != "fragrance-allergen" for f in enriched["concerns"]["findings"]
    )
    # Local resolution status stays "unresolved" for the AI-enriched ingredient.
    row = next(r for r in enriched["ingredients"] if r["raw"] == UNKNOWN_A)
    assert row["resolved"] is False

    # Unresolved warning is still present (existing behaviour preserved).
    assert any(UNKNOWN_A.lower() in w.lower() for w in enriched["warnings"])


def test_report_includes_ai_enrichment_field(json_store):
    from ml.inference import analyze_ingredients

    # Disabled: the field still exists, switched off.
    disabled = analyze_ingredients(ANALYSIS_TEXT, explain=False, enricher=None)
    ai = disabled["ai_enrichment"]
    assert set(ai) >= {
        "enabled",
        "requested_count",
        "resolved_count",
        "cached_count",
        "failed_count",
        "items",
        "disclaimer",
    }
    assert ai["enabled"] is False
    assert ai["items"] == []

    # Enabled: populated with the enriched unknown ingredient only.
    report = analyze_ingredients(
        ANALYSIS_TEXT, explain=False, enricher=make_enricher(FakeClient(), json_store)
    )
    ai = report["ai_enrichment"]
    assert ai["enabled"] is True
    assert ai["requested_count"] == 1
    assert ai["resolved_count"] == 1
    assert ai["failed_count"] == 0
    item = ai["items"][0]
    assert item["ingredient"] == UNKNOWN_A
    assert item["normalized_name"] == UNKNOWN_A_KEY
    assert item["status"] == "enriched"
    assert item["source"] == "groq"
    assert item["review_status"] == "unreviewed"
    assert item["data"]["confidence"] == "low"
    assert "safety" in ai["disclaimer"].lower() or "regulatory" in ai["disclaimer"].lower()

    # The exports clearly label the AI block as separate, unreviewed content.
    from ml.rules.report_builder import to_markdown, to_text

    markdown = to_markdown(report)
    assert "AI-enriched information" in markdown
    assert "Source: Groq" in markdown
    assert "Unreviewed" in markdown
    assert "Not a safety or regulatory determination" in markdown
    text_export = to_text(report)
    assert "AI-ENRICHED INFORMATION" in text_export
    assert "Source: Groq" in text_export


def test_json_fallback_works_without_mongodb(tmp_path):
    path = tmp_path / "enrichments.json"
    store = ie.EnrichmentStore(fallback_path=path, enable_mongo=False)
    assert store.backend == "json"
    seed_record(store)
    assert path.exists()
    assert store.count() == 1
    assert len(store.list()) == 1
    assert store.status() == {"backend": "json", "count": 1}

    # A brand-new store instance reads the same records back from disk.
    reopened = ie.EnrichmentStore(fallback_path=path, enable_mongo=False)
    record = reopened.get(UNKNOWN_A_KEY)
    assert record is not None
    assert record["data"]["inci_name"] == UNKNOWN_A
    assert record["review_status"] == "unreviewed"
    assert "api_key" not in json.dumps(record)

    assert reopened.delete(UNKNOWN_A_KEY) is True
    assert reopened.get(UNKNOWN_A_KEY) is None
    assert reopened.count() == 0


# ---------------------------------------------------------------------------
# 12. The API key never leaks (endpoints, logs, records) + endpoint surface
# ---------------------------------------------------------------------------
SECRET = "sk-super-secret-key-should-never-leak"


def test_api_key_is_never_returned_or_logged(monkeypatch, json_store, capsys):
    import requests as requests_lib

    monkeypatch.setattr(config, "GROQ_API_KEY", SECRET)
    monkeypatch.setattr(config, "AI_ENRICHMENT_ENABLED", True)

    # Status payload: boolean only, never the key itself.
    status = ie.enrichment_status()
    assert status["api_key_configured"] is True
    assert SECRET not in json.dumps(status)

    # Client failure messages (and anything printed) never contain the key.
    def broken_post(*args, **kwargs):
        raise requests_lib.ConnectionError(f"boom while using {SECRET}")

    monkeypatch.setattr(ie.requests, "post", broken_post)
    client = ie.GroqChatClient(api_key=SECRET)
    with pytest.raises(ie.GroqClientError) as excinfo:
        client.complete("system", "user")
    assert SECRET not in str(excinfo.value)
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err

    # Cached records never store the key.
    seed_record(json_store)
    assert SECRET not in json.dumps(json_store.get(UNKNOWN_A_KEY))

    # API responses (health + enrichment endpoints) never contain the key.
    monkeypatch.setattr(ie, "_store", json_store)
    from fastapi.testclient import TestClient

    from backend.main import app

    http = TestClient(app)
    health = http.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert SECRET not in health.text
    assert body["ai_enrichment"]["api_key_configured"] is True
    assert body["ai_enrichment"]["enabled"] is True
    assert body["ai_enrichment"]["provider"] == "groq"

    one = http.get("/api/knowledge/enrichment", params={"name": UNKNOWN_A})
    assert one.status_code == 200
    assert one.json()["review_status"] == "unreviewed"
    assert one.json()["source"] == "groq"
    assert SECRET not in one.text

    listing = http.get("/api/knowledge/enrichments", params={"limit": 25})
    assert listing.status_code == 200
    assert listing.json()["count"] >= 1
    assert SECRET not in listing.text

    missing = http.get("/api/knowledge/enrichment", params={"name": "Definitely Missing Ingredient"})
    assert missing.status_code == 404

    deleted = http.delete("/api/knowledge/enrichment", params={"name": UNKNOWN_A})
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert ie.get_cached_enrichment(UNKNOWN_A) is None

    again = http.delete("/api/knowledge/enrichment", params={"name": UNKNOWN_A})
    assert again.status_code == 404


def test_enrichment_is_off_by_default():
    """With no AI/Groq environment variables the feature stays fully disabled."""
    import os

    if any(key in os.environ for key in ("AI_ENRICHMENT_ENABLED", "GROQ_API_KEY")):
        pytest.skip("AI enrichment configured in the local environment")
    assert config.AI_ENRICHMENT_ENABLED is False
    assert config.GROQ_API_KEY == ""
    assert ie.enrichment_enabled() is False
    assert ie.get_enricher() is None