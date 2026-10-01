import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { Empty, ErrorBanner, Spinner, formatWhen } from "../components/ui.jsx";

export default function Dashboard() {
  const [recent, setRecent] = useState(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    api
      .listAnalyses(6)
      .then((data) => setRecent(data.items))
      .catch((err) => setError(err.message));
  }, []);

  return (
    <>
      <div className="hero">
        <div>
          <h1>DeepCos</h1>
          <p className="tagline">Understand what's inside your cosmetic products.</p>
        </div>
        <Link className="btn primary" to="/analyze">
          ✚ New Analysis
        </Link>
      </div>

      <ErrorBanner error={error} />

      <div className="report-section">
        <h2>Recent Analyses</h2>
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
              const type =
                item.product?.display_name ||
                item.product?.category ||
                item.product_category ||
                "Unknown";
              const summary = item.profile_summary || item.summary || "";
              const mode =
                item.mode === "image" ? "Image" : item.mode === "text" ? "Text" : item.mode || "Text";
              return (
                <div
                  key={id}
                  className="card clickable compact"
                  onClick={() => navigate(`/report/${id}`)}
                >
                  <div className="recent-type">{type}</div>
                  <div className="recent-summary">
                    {summary ? `${summary.slice(0, 130)}${summary.length > 130 ? "…" : ""}` : "—"}
                  </div>
                  <div className="recent-meta">
                    <span>{item.input?.ingredient_count ?? "—"} ingredients</span>
                    <span>
                      {mode} • {formatWhen(item.created_at)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="report-section">
        <h2>How DeepCos works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-num">1</div>
            <div>
              <h3>Read</h3>
              <p>Product labels are converted into ingredient information.</p>
            </div>
          </div>
          <div className="step">
            <div className="step-num">2</div>
            <div>
              <h3>Analyze</h3>
              <p>DeepCos identifies patterns in the ingredient combination.</p>
            </div>
          </div>
          <div className="step">
            <div className="step-num">3</div>
            <div>
              <h3>Explain</h3>
              <p>
                The result shows the formulation characteristics and the ingredients that influenced
                the prediction.
              </p>
            </div>
          </div>
        </div>
        <div className="small muted mt">Powered by computer vision and deep learning.</div>
      </div>
    </>
  );
}
