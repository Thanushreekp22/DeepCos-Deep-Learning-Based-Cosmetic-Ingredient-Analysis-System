import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { Chip, Empty, ErrorBanner, Spinner } from "../components/ui.jsx";

export default function History() {
  const [items, setItems] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  const load = useCallback(() => {
    setError("");
    api
      .listAnalyses(200)
      .then((data) => setItems(data.items))
      .catch((err) => setError(err));
  }, []);

  useEffect(load, [load]);

  const remove = async (id) => {
    if (!window.confirm(`Delete analysis ${id}?`)) return;
    setBusy(true);
    try {
      await api.deleteAnalysis(id);
      load();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>History</h1>
          <p>Every saved analysis — newest first.</p>
        </div>
        <button className="btn small" onClick={load} disabled={busy}>
          ⟳ Refresh
        </button>
      </div>

      <ErrorBanner error={error} onRetry={load} />

      {items === null ? (
        <Spinner />
      ) : items.length === 0 ? (
        <Empty glyph="◌" title="No saved analyses yet" hint="Start one from New analysis." />
      ) : (
        <div className="card" style={{ overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Date</th>
                <th>Mode</th>
                <th>Product</th>
                <th>Top concern</th>
                <th>Ingredients</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {items.map((it) => {
                const concerns = it.concerns || {};
                const product = it.product || {};
                const category =
                  product.category || it.product_category || it.product?.category || "—";
                const findings = concerns.findings || [];
                return (
                  <tr key={it.analysis_id}>
                    <td className="num">
                      {String(it.created_at || "").replace("T", " ").slice(0, 19)}
                    </td>
                    <td>
                      <Chip tone={it.mode === "image" ? "lilac" : "rose"}>
                        {it.mode === "image" ? "Image" : it.mode === "text" ? "Text" : it.mode || "legacy"}
                      </Chip>
                    </td>
                    <td>
                      <a href={`/report/${it.analysis_id}`}>
                        {product.display_name || category}
                      </a>
                      <div className="small muted">
                        {String(it.profile_summary || "").slice(0, 70)}
                        {(it.profile_summary || "").length > 70 ? "…" : ""}
                      </div>
                    </td>
                    <td>
                      {findings.length > 0 ? (
                        <span className={`sev ${(findings[0].severity || "low").toLowerCase()}`}>
                          {findings[0].label || findings[0].tag}
                        </span>
                      ) : (
                        <span className="small muted">
                          {concerns.highest_severity_label || "none flagged"}
                        </span>
                      )}
                    </td>
                    <td className="num">{it.input?.ingredient_count ?? "—"}</td>
                    <td>
                      <div className="row" style={{ flexWrap: "nowrap" }}>
                        <a className="btn small" href={`/report/${it.analysis_id}`}>
                          View
                        </a>
                        <a
                          className="btn small"
                          href={api.exportUrl(it.analysis_id, "markdown")}
                          target="_blank"
                          rel="noreferrer"
                        >
                          ↓ MD
                        </a>
                        <button
                          className="btn small danger"
                          disabled={busy}
                          onClick={() => remove(it.analysis_id)}
                        >
                          ✕
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="row mt">
        <button
          className="btn small"
          onClick={() => items?.[0] && navigate(`/report/${items[0].analysis_id}`)}
          disabled={!items?.length}
        >
          Open newest report
        </button>
      </div>
    </>
  );
}
