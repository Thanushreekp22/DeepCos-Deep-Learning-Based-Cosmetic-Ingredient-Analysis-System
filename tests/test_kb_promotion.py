"""
Tests for persisting validated AI enrichment into the AI-learned KB overlay.

Covers the promotion feature end to end without any network or MongoDB access:

* mapping AI wording onto the existing function/concern taxonomy
* promotion gates (identification, confidence, empty results, junk names)
* the overlay file, the learned knowledge base and the "no second AI call" path
* verified reference entries always win over AI output
* transparency: report rows/findings/export labels and the API surface
* deletion removes both the file entry and the in-process resolution

Every store and knowledge base here points at ``tmp_path``; the repository's own
``data/knowledge_base`` files are never written by these tests.
"""

from __future__ import annotations

import json

import pytest

from backend import config
import backend.services.ingredient_enrichment as ie
import backend.services.kb_promotion as kb_promotion
from ml import ingredient_promotion as ip
from ml.knowledge_base import IngredientKnowledgeBase
from ml.learned_store import LearnedIngredientStore
from ml.rules import concern_engine
from ml.rules.report_builder import to_markdown, to_text

ZINC_KEY = "zinc pca"

# Shape of a real Groq response for the reported symptom ("Zinc PCA" stays
# unknown and is enriched on every run).
ZINC_DATA = {
    "inci_name": "Zinc PCA",
    "identification": "inci",
    "aliases": ["Zinc Pyroglutamate"],
    "description": "A zinc salt of pyrrolidone carboxylic acid used as a sebum-regulating agent.",
    "functions": ["sebum regulating", "skin conditioning", "humectant"],
    "possible_concerns": ["Comedogenic potential"],
    "regulatory_note": "",
    "confidence": "medium",
    "evidence_note": "Fixture data for the promotion tests.",
}

GOOD_RECORD = {
    "normalized_key": ZINC_KEY,
    "requested_name": "Zinc PCA",
    "source": "groq",
    "provider": "groq",
    "model": "fake-model",
    "updated_at": "2026-01-01T00:00:00+00:00",
}


class FakeClient:
    """Injected stand-in for ``GroqChatClient`` (never touches the network)."""

    def __init__(self, payload: dict | None = None, error: BaseException | None = None) -> None:
        self.payload = payload or ZINC_DATA
        self.error = error
        self.calls: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload)


@pytest.fixture()
def overlay_path(tmp_path):
    return tmp_path / "learned_ingredients.json"


@pytest.fixture()
def store(overlay_path):
    return LearnedIngredientStore(path=overlay_path, enable_mongo=False)


@pytest.fixture()
def kb(overlay_path):
    return IngredientKnowledgeBase(learned_path=overlay_path)


def make_promoter(kb, store, *, enabled=True, min_confidence="medium"):
    return kb_promotion.KnowledgePromoter(
        kb=kb, store=store, enabled=enabled, min_confidence=min_confidence
    )


def promote_zinc(kb, store, **kwargs):
    return make_promoter(kb, store, **kwargs).promote("Zinc PCA", ZINC_KEY, ZINC_DATA, GOOD_RECORD)


# ---------------------------------------------------------------------------
# 1. Taxonomy mapping: AI wording -> real tags
# ---------------------------------------------------------------------------
def test_ai_wording_is_mapped_onto_the_existing_taxonomy(kb):
    functions, unmapped_functions = ip.map_functions(ZINC_DATA["functions"], kb)
    assert "anti-acne" in functions  # "sebum regulating"
    assert "skin-conditioning" in functions
    assert "humectant" in functions
    assert unmapped_functions == []
    assert all(tag in kb.function_taxonomy for tag in functions)

    concerns, unmapped_concerns = ip.map_concerns(ZINC_DATA["possible_concerns"], kb)
    assert concerns == ["comedogenic-potential"]
    assert unmapped_concerns == []
    assert all(tag in kb.concern_rules for tag in concerns)


def test_unmappable_phrases_become_plain_slug_tags(kb):
    # A known word inside a longer phrase still maps (word-boundary match)...
    functions, unmapped = ip.map_functions(["zwitterionic buffer"], kb)
    assert functions == ["ph-adjuster"]
    assert unmapped == []

    # ...while a phrase with no known word becomes a plain slug instead of
    # inventing a taxonomy entry.
    functions, unmapped = ip.map_functions(
        ["zwitterionic stabiliser", "NOT A REAL FUNCTION"], kb
    )
    assert functions == []
    assert unmapped == ["zwitterionic-stabiliser", "not-a-real-function"]

    # Without a KB the synonym tables themselves are trusted.
    concerns, unmapped_concerns = ip.map_concerns(["fragrance allergen"], kb=None)
    assert concerns == ["fragrance-allergen", "fragrance"]
    assert unmapped_concerns == []


# ---------------------------------------------------------------------------
# 2. Promotion gates
# ---------------------------------------------------------------------------
def test_disabled_promotion_writes_nothing(kb, store, overlay_path):
    result = promote_zinc(kb, store, enabled=False)
    assert result["promoted"] is False
    assert "AI_PROMOTE_TO_KB" in result["reason"]
    assert store.count() == 0
    assert not overlay_path.exists()
    assert kb.learned_count() == 0


@pytest.mark.parametrize(
    "data, expected",
    [
        ({**ZINC_DATA, "confidence": "low"}, "below the promotion threshold"),
        ({**ZINC_DATA, "identification": "unknown"}, "could not identify"),
        ({**ZINC_DATA, "functions": [], "possible_concerns": []}, "neither functions nor concerns"),
        ({**ZINC_DATA, "functions": "sebum regulating"}, None),  # a string still counts
    ],
)
def test_confidence_and_identification_gates(kb, store, data, expected):
    result = make_promoter(kb, store).promote("Zinc PCA", ZINC_KEY, data, GOOD_RECORD)
    if expected is None:
        assert result["promoted"] is True
    else:
        assert result["promoted"] is False
        assert expected in result["reason"]


def test_short_or_meaningless_names_are_not_promoted(kb, store):
    promoter = make_promoter(kb, store)
    for name in ("x", "and", "a b", "Other"):
        result = promoter.promote(name, None, ZINC_DATA, GOOD_RECORD)
        assert result["promoted"] is False, name
    assert store.count() == 0


def test_no_data_is_not_promoted(kb, store):
    result = make_promoter(kb, store).promote("Zinc PCA", ZINC_KEY, None, GOOD_RECORD)
    assert result["promoted"] is False
    assert "no enrichment data" in result["reason"]


# ---------------------------------------------------------------------------
# 3. Writing the overlay + resolving it later without an AI call
# ---------------------------------------------------------------------------
def test_promotion_writes_overlay_and_resolves_locally(kb, store, overlay_path):
    assert kb.resolve("Zinc PCA").resolved is False  # the reported symptom

    result = promote_zinc(kb, store)
    assert result["promoted"] is True
    assert result["origin"] == "ai-learned"
    assert result["review_status"] == "unreviewed"
    assert "anti-acne" in result["functions"]
    assert result["concerns"] == ["comedogenic-potential"]

    # 1. persisted in the overlay file
    records = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["normalized_key"] == ZINC_KEY
    assert records[0]["origin"] == "ai-learned"
    assert records[0]["review_status"] == "unreviewed"
    assert records[0]["ai_functions"] == ZINC_DATA["functions"]
    assert records[0]["model"] == "fake-model"

    # 2. served by this process right away...
    resolved = kb.resolve("Zinc PCA")
    assert resolved.resolved is True
    assert resolved.match_type == "learned"
    assert resolved.inci == "Zinc PCA"
    assert "anti-acne" in resolved.info.functions
    assert resolved.info.origin == "ai-learned"
    assert kb.learned_count() == 1

    # ...and by a fresh knowledge base on the next start (no AI call needed)
    reopened = IngredientKnowledgeBase(learned_path=overlay_path)
    assert reopened.resolve("Zinc PCA").resolved is True
    assert reopened.learned_count() == 1


def test_alias_and_fuzzy_matches_are_marked_as_learned(kb, store):
    promote_zinc(kb, store)
    alias = kb.resolve("Zinc Pyroglutamate")
    assert alias.resolved is True
    assert alias.match_type == "learned-alias"

    fuzzy = kb.resolve("Zinc PCA (5%)".replace("PCA", "PCAs"))
    assert fuzzy.info is not None
    assert fuzzy.match_type in ("learned-fuzzy", "learned-alias", "learned")


def test_promoted_entry_never_claims_a_regulatory_status(kb, store):
    promote_zinc(kb, store)
    info = kb.resolve("Zinc PCA").info
    assert info.regulatory_status == "unknown"
    assert info.regulatory_note == ""
    record = store.get(ZINC_KEY)
    assert record["ai_regulatory_note"] == ""
    assert "unreviewed" in record["note"]


def test_verified_reference_entries_are_never_overwritten(kb, store, overlay_path):
    """Curated data wins: a documented ingredient is never promoted over."""
    result = make_promoter(kb, store).promote(
        "Niacinamide", None, {**ZINC_DATA, "inci_name": "Niacinamide"}, GOOD_RECORD
    )
    assert result["promoted"] is False
    assert "already documents" in result["reason"]
    assert kb.resolve("Niacinamide").match_type == "exact"
    assert kb.learned_count() == 0
    assert not overlay_path.exists()


def test_repromoting_the_same_result_is_idempotent(kb, store, overlay_path):
    promoter = make_promoter(kb, store)
    first = promoter.promote("Zinc PCA", ZINC_KEY, ZINC_DATA, GOOD_RECORD)
    second = promoter.promote("Zinc PCA", ZINC_KEY, ZINC_DATA, GOOD_RECORD)
    assert first["promoted"] and second["promoted"]
    assert len(json.loads(overlay_path.read_text(encoding="utf-8"))) == 1
    assert kb.learned_count() == 1


def test_broken_overlay_file_is_ignored(overlay_path):
    overlay_path.write_text("{not json", encoding="utf-8")
    kb = IngredientKnowledgeBase(learned_path=overlay_path)
    assert kb.learned_count() == 0
    assert kb.resolve("Zinc PCA").resolved is False


def test_overlay_accepts_wrapped_entries(overlay_path):
    entry = ip.build_learned_entry(
        key=ZINC_KEY, requested_name="Zinc PCA", data=ZINC_DATA, record=GOOD_RECORD
    )
    overlay_path.write_text(json.dumps({"version": 1, "entries": [entry]}), encoding="utf-8")
    kb = IngredientKnowledgeBase(learned_path=overlay_path)
    assert kb.learned_count() == 1
    assert kb.resolve("Zinc PCA").resolved is True


def test_delete_removes_entry_and_local_resolution(kb, store):
    promote_zinc(kb, store)
    promoter = make_promoter(kb, store)
    assert promoter.get("Zinc PCA") is not None
    assert promoter.delete("Zinc PCA") is True
    assert kb.learned_count() == 0
    assert kb.resolve("Zinc PCA").resolved is False
    assert store.count() == 0
    assert promoter.delete("Zinc PCA") is False


# ---------------------------------------------------------------------------
# 4. Enricher integration: promoted results replace later AI calls
# ---------------------------------------------------------------------------
def test_enricher_promotes_and_next_run_needs_no_ai_call(kb, store, tmp_path):
    cache_path = tmp_path / "enrichments.json"
    client = FakeClient()
    enricher = ie.IngredientEnricher(
        client=client,
        store=ie.EnrichmentStore(fallback_path=cache_path, enable_mongo=False),
        promotion=make_promoter(kb, store),
        max_items=5,
    )
    first = enricher.enrich_unresolved(kb.resolve_all(["Zinc PCA", "Aqua"]))
    assert len(client.calls) == 1  # only the undocumented ingredient
    assert first["promotion_enabled"] is True
    assert first["promoted_count"] == 1
    assert first["items"][0]["knowledge_base"]["promoted"] is True
    assert first["items"][0]["knowledge_base"]["review_status"] == "unreviewed"

    # A later analysis finds the ingredient in the AI-learned overlay, so the
    # enrichment step has nothing left to ask the provider about.
    second_client = FakeClient()
    second = ie.IngredientEnricher(
        client=second_client,
        store=ie.EnrichmentStore(fallback_path=cache_path, enable_mongo=False),
        promotion=make_promoter(kb, store),
        max_items=5,
    )
    second_block = second.enrich_unresolved(kb.resolve_all(["Zinc PCA", "Aqua"]))
    assert second_client.calls == []
    assert second_block["requested_count"] == 0
    assert kb.resolve("Zinc PCA").match_type == "learned"


def test_enricher_without_promoter_never_writes_the_overlay(kb, store, overlay_path, tmp_path):
    client = FakeClient()
    block = ie.IngredientEnricher(
        client=client,
        store=ie.EnrichmentStore(fallback_path=tmp_path / "e.json", enable_mongo=False),
    ).enrich_unresolved(kb.resolve_all(["Zinc PCA"]))
    assert len(client.calls) == 1
    assert block["promotion_enabled"] is False
    assert block["promoted_count"] == 0
    assert block["items"][0]["knowledge_base"]["promoted"] is False
    assert not overlay_path.exists()
    assert kb.resolve("Zinc PCA").resolved is False


def test_enricher_survives_a_broken_promoter(kb, tmp_path):
    class BrokenPromoter:
        def promote(self, *args, **kwargs):
            raise RuntimeError("boom")

    block = ie.IngredientEnricher(
        client=FakeClient(),
        store=ie.EnrichmentStore(fallback_path=tmp_path / "e.json", enable_mongo=False),
        promotion=BrokenPromoter(),
    ).enrich_unresolved(kb.resolve_all(["Zinc PCA"]))
    # The AI result is still delivered; only the persistence step degraded.
    assert block["resolved_count"] == 1
    assert block["promoted_count"] == 0
    assert block["items"][0]["knowledge_base"]["promoted"] is False


# ---------------------------------------------------------------------------
# 5. Transparency: screening rows, concern findings and exports
# ---------------------------------------------------------------------------
def test_learned_entry_screens_as_unreviewed_ai_content(kb, store):
    promote_zinc(kb, store)
    resolved = kb.resolve_all(["Zinc PCA", "Aqua"])

    findings = {f["tag"]: f for f in concern_engine.screen(resolved, kb)["findings"]}
    finding = findings["comedogenic-potential"]
    assert finding["ai_derived"] is True
    assert finding["documented_rule"] is True  # a local rule exists for the tag
    assert finding["ingredients"] == ["Zinc PCA"]

    rows = concern_engine.ingredient_breakdown(resolved, kb)
    zinc = next(row for row in rows if row["ingredient"] == "Zinc PCA")
    assert zinc["resolved"] is True
    assert zinc["ai_learned"] is True
    assert zinc["origin"] == "ai-learned"
    assert zinc["review_status"] == "unreviewed"
    assert zinc["regulatory_status"] == "unknown"
    assert zinc["primary_function"] == kb.function_label("anti-acne")


def test_undocumented_ai_concern_keeps_the_lowest_severity(kb, store):
    data = {**ZINC_DATA, "possible_concerns": ["cytotoxic potential"]}
    make_promoter(kb, store).promote("Zinc PCA", ZINC_KEY, data, GOOD_RECORD)
    record = store.get(ZINC_KEY)
    assert record["concerns"] == []
    assert record["unmapped_concerns"] == ["cytotoxic-potential"]

    # Without a local rule there is nothing to screen: the tag stays an
    # informational note on the learned entry instead of a severity level.
    block = concern_engine.screen(kb.resolve_all(["Zinc PCA"]), kb)
    assert block["findings"] == []
    assert block["highest_severity"] == "info"


def test_report_exports_label_learned_and_promoted_content(kb, store):
    promote_zinc(kb, store)
    resolved = kb.resolve_all(["Zinc PCA", "Aqua"])
    report = {
        "input": {"ingredient_count": 2},
        "product": {"category": "Serum", "category_confidence": 0.42},
        "profile": [],
        "key_ingredients": [],
        "ingredients": concern_engine.ingredient_breakdown(resolved, kb),
        "concerns": concern_engine.screen(resolved, kb),
        "knowledge_base": {"version": "1.0", "learned_count": 1, "learned_used": 1},
        "ai_enrichment": {
            "enabled": True,
            "promotion_enabled": True,
            "promoted_count": 1,
            "items": [
                {
                    "ingredient": "Zinc PCA",
                    "normalized_name": ZINC_KEY,
                    "status": "enriched",
                    "source": "groq",
                    "model": "fake-model",
                    "review_status": "unreviewed",
                    "data": dict(ZINC_DATA),
                    "knowledge_base": {
                        "promoted": True,
                        "reason": "saved to the AI-learned knowledge-base overlay (unreviewed)",
                        "origin": "ai-learned",
                        "review_status": "unreviewed",
                    },
                }
            ],
        },
        "warnings": [],
        "disclaimer": "fixture",
    }

    markdown = to_markdown(report)
    assert "*(AI-learned, unreviewed)*" in markdown  # ingredient row + finding
    assert "saved to the AI-learned overlay" in markdown
    assert "AI-learned knowledge-base overlay" in markdown
    assert "AI-enriched information" in markdown

    text_export = to_text(report)
    assert "saved to the AI-learned KB (unreviewed)" in text_export
    assert "(AI-learned)" in text_export


# ---------------------------------------------------------------------------
# 6. API surface: list / inspect / delete + status
# ---------------------------------------------------------------------------
def test_learned_overlay_endpoints(kb, store, monkeypatch):
    promote_zinc(kb, store)
    monkeypatch.setattr(kb_promotion, "_promoter", make_promoter(kb, store))

    from fastapi.testclient import TestClient

    from backend.main import app

    http = TestClient(app)

    listing = http.get("/api/knowledge/learned", params={"limit": 10})
    assert listing.status_code == 200
    body = listing.json()
    assert body["count"] == 1
    assert body["origin"] == "ai-learned"
    assert body["min_confidence"]
    assert body["entries"][0]["normalized_key"] == ZINC_KEY

    one = http.get("/api/knowledge/learned/entry", params={"name": "Zinc PCA"})
    assert one.status_code == 200
    assert one.json()["inci"] == "Zinc PCA"
    assert one.json()["review_status"] == "unreviewed"

    missing = http.get("/api/knowledge/learned/entry", params={"name": "Xylium Extract"})
    assert missing.status_code == 404

    # Status endpoints expose counts only (never any credential).
    health = http.get("/api/health")
    assert health.status_code == 200
    learned_status = health.json()["ai_learned_kb"]
    assert learned_status["count"] == 1
    assert learned_status["min_confidence"]
    assert "learned_count" in health.json()["ai_enrichment"]
    assert "learned_count" in http.get("/api/knowledge/stats").json()

    deleted = http.delete("/api/knowledge/learned", params={"name": "Zinc PCA"})
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert kb.learned_count() == 0
    assert kb.resolve("Zinc PCA").resolved is False
    assert store.count() == 0

    again = http.delete("/api/knowledge/learned", params={"name": "Zinc PCA"})
    assert again.status_code == 404


def test_promoter_status_reports_configuration(monkeypatch, kb, store):
    monkeypatch.setattr(config, "AI_PROMOTE_TO_KB", True)
    monkeypatch.setattr(config, "AI_PROMOTE_MIN_CONFIDENCE", "high")
    promoter = kb_promotion.KnowledgePromoter(kb=kb, store=store)
    status = promoter.status()
    assert status["enabled"] is True
    assert status["min_confidence"] == "high"
    assert status["origin"] == "ai-learned"
    assert status["store_backend"] == "json"
    assert "path" not in status  # never expose filesystem paths through the API
