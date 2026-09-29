import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { Chip, Empty, ErrorBanner, Spinner } from "../components/ui.jsx";

export default function Dashboard({ health }) {
  const [recent, setRecent] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    api
      .listAnalyses(6)
      .then((data) => setRecent(data.items))
      .catch((err) => setError(err.message));
    api
      .knowledgeStats()
      .then(setStats)
      .catch(() => {});
  }, []);

  const kb = health?.knowledge_base || stats || {};
  const ocr = health?.image_pipeline?.ocr;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Dashboard</h1>
          <p>
            DeepCos reads cosmetic labels with a CNN + OCR, profiles the formulation with an
            Embedding→LSTM→MLP network and screens documented ingredient concerns - every
            prediction comes with its evidence.
          </p>
        </div>
        <Link className="btn primary" to="/analyze">
          ✚ New analysis
        </Link>
      </div>

      <ErrorBanner error={error} />

      <div className="grid cols-4">
        <div className="card">
          <h3>Ingredient model</h3>
          <div className="big">{health?.ingredient_model?.trained ? "Ready" : "Missing"}</div>
          <div className="sub">
            {health?.ingredient_model?.trained
              ? "Embedding → LSTM → MLP (multi-task)"
              : "Train with python -m ml.train_ingredient_model"}
          </div>
        </div>
        <div className="card">
          <h3>Label CNN</h3>
          <div className="big">
            {health?.image_pipeline?.text_region_cnn ? "Trained" : "Pending"}
          </div>
          <div className="sub">
            Text detection {health?.image_pipeline?.text_detector_loaded ? "loaded" : "not loaded"} ·{" "}
            OCR {ocr?.available ? `v${ocr.version ?? ""}`.trim() : "unavailable"}
          </div>
        </div>
        <div className="card">
          <h3>Knowledge base</h3>
          <div className="big">{kb.ingredient_count ?? "—"}</div>
          <div className="sub">
            ingredients · {kb.function_taxonomy_count ?? "—"} functions ·{" "}
            {kb.concern_rule_count ?? "—"} concern rules
          </div>
        </div>
        <div className="card">
          <h3>Storage</h3>
          <div className="big" style={{ fontSize: 22 }}>
            {health?.database?.backend === "mongodb"
              ? "MongoDB"
              : health?.database?.backend === "json"
                ? "JSON fallback"
                : "…"}
          </div>
          <div className="sub">
            {health?.database?.backend === "mongodb"
              ? health.database.database
              : health?.database?.fallback_path?.split(/[\\/]/).slice(-1)[0] ?? ""}
          </div>
        </div>
      </div>

      <div className="report-section">
        <h2>Recent analyses</h2>
        {recent === null ? (
          <Spinner />
        ) : recent.length === 0 ? (
          <Empty
            glyph="◌"
            title="No analyses yet"
            hint="Run your first analysis from the New analysis page."
          />
        ) : (
          <div className="grid cols-3">
            {recent.map((item) => {
              const id = item.analysis_id;
              const legacyCategory = item.product_category || item.product?.category;
              const category = legacyCategory || "Unknown";
              const summary = item.profile_summary || item.summary;
              const tone = item.mode === "image" ? "lilac" : "rose";
              return (
                <div
                  key={id}
                  className="card clickable"
                  onClick={() => navigate(`/report/${id}`)}
                >
                  <div className="spread">
                    <strong>{item.product?.display_name || category}</strong>
                    <Chip tone={tone}>{item.mode || "legacy"}</Chip>
                  </div>
                  <div className="sub" style={{ marginTop: 8, minHeight: 36 }}>
                    {summary ? `${summary.slice(0, 110)}${summary.length > 110 ? "…" : ""}` : "—"}
                  </div>
                  <div className="spread mt small muted">
                    <span>{(item.created_at || "").replace("T", " ")}</span>
                    <span>{item.input?.ingredient_count ?? "—"} ingredients</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="report-section">
        <h2>How it works</h2>
        <div className="grid cols-3">
          <div className="card">
            <h3>1 · Vision</h3>
            <div className="sub">
              An AlexNet-style CNN locates the ingredient block on the label photo; Tesseract OCR
              turns it into text with confidence scores and variant fallback.
            </div>
          </div>
          <div className="card">
            <h3>2 · Sequence model</h3>
            <div className="sub">
              Ingredients are embedded and passed through an LSTM with multi-task heads predicting
              the product category and four formulation-profile scores.
            </div>
          </div>
          <div className="card">
            <h3>3 · Explainability</h3>
            <div className="sub">
              Occlusion analysis re-runs the model without each ingredient to show exactly which
              inputs pushed a score up or down - alongside knowledge-base concern screening.
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
