import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, apiUrl } from "../api.js";
import { Chip, Empty, ErrorBanner, Meter, SeverityBadge, Spinner } from "../components/ui.jsx";

// profile band -> chip tone
const TONE_BY_BAND = { high: "rose", moderate: "gold", low: "info" };

// stored preview PNGs, in a sensible display order
const ASSET_LABELS = {
  original_png_base64: "Original photo",
  detected_region_png_base64: "Detected text region",
  heatmap_png_base64: "CNN text heatmap",
  cropped_region_png_base64: "Cropped region",
};
const ASSET_ORDER = Object.keys(ASSET_LABELS);

const var_safe = (name) => `var(${name})`;

function Section({ title, children }) {
  return (
    <div className="report-section">
      <h2>{title}</h2>
      {children}
    </div>
  );
}

export default function Report() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [previewing, setPreviewing] = useState(false);
  const [inlineImages, setInlineImages] = useState(null);

  const load = useCallback(() => {
    setError("");
    setReport(null);
    api
      .getAnalysis(id)
      .then(setReport)
      .catch((err) => setError(err));
  }, [id]);

  useEffect(load, [load]);

  // The stored preview PNGs are also readable inline as data URLs.
  const runPreview = async () => {
    setPreviewing(true);
    try {
      const res = await api.previewImages(id);
      setInlineImages(res.images || {});
    } catch (err) {
      setError(err);
    } finally {
      setPreviewing(false);
    }
  };

  const remove = async () => {
    if (!window.confirm("Delete this report?")) return;
    await api.deleteAnalysis(id);
    navigate("/history");
  };

  if (error && !report) return <ErrorBanner error={error} onRetry={load} />;
  if (!report) return <Spinner />;

  // The API returns the report document at the top level (no wrapper object).
  const r = report;
  const model = r.model || {};
  const product = r.product || {};
  const image = r.image_analysis || {};
  const preprocessing = r.preprocessing || {};
  const ocrText = preprocessing.ocr_text || "";
  const profile = r.profile || [];
  const profileConfidence = r.profile_confidence || {};
  const explain = r.explainability || {};
  const contributors = explain.top_contributors || {};
  const explainOn = explain.method === "occlusion" && Object.keys(contributors).length > 0;
  const keyIngredients = r.key_ingredients || [];
  const concerns = r.concerns || {};
  const findings = concerns.findings || [];
  const activeFlags = Object.entries(concerns.flags || {})
    .filter(([, on]) => on)
    .map(([name]) => name);
  const ingredients = r.ingredients || [];
  const seq = r.sequence || {};
  const input = r.input || {};
  const warnings = r.warnings || [];

  const categoryProbs = Object.entries(product.category_probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([label, probability]) => ({ label, probability }));

  const assetSources = { ...(image.assets || {}), ...(inlineImages || {}) };
  const assets = ASSET_ORDER.filter((k) => assetSources[k]).map((k) => ({
    key: k,
    label: ASSET_LABELS[k],
    src: String(assetSources[k]).startsWith("data:")
      ? assetSources[k]
      : apiUrl(assetSources[k]),
  }));

  const detector = image.cnn_text_detector || {};
  const quality = image.quality || {};
  const reliability = Object.entries(model.profile_reliability || {});
  const sevTone =
    { none: "ok", low: "", medium: "warn", high: "warn" }[concerns.highest_severity] || "";

  return (
    <>
      <div className="page-head">
        <div>
          <h1>{r.title || "Analysis report"}</h1>
          <p>
            ID {r.analysis_id} · {input.ingredient_count ?? ingredients.length} ingredients in
            list · {((r.processing_ms ?? 0) / 1000).toFixed(1)}s
          </p>
          <div className="row">
            <Chip tone={String(r.mode || "").startsWith("image") ? "lilac" : "rose"}>
              {r.mode}
            </Chip>
            <Chip tone="info">{product.category || "unknown product"}</Chip>
            {product.display_name && product.display_name !== product.category ? (
              <Chip tone="gold">{product.display_name}</Chip>
            ) : null}
            <Chip tone="lilac">{model.name}</Chip>
          </div>
        </div>
        <div className="row">
          <button className="btn" onClick={() => window.print()}>⎙ Print</button>
          <button className="btn danger" onClick={remove}>Delete</button>
          <Link className="btn primary" to="/analyze">Run another analysis</Link>
        </div>
      </div>

      <ErrorBanner error={error} onRetry={load} />

      {warnings.length > 0 && <div className="banner warn">{warnings.join(" · ")}</div>}

      <div className="grid cols-2">
        <div className="card">
          <h3>Product identification</h3>
          <div className="confidence-ring">
            <div
              className="ring"
              style={{ "--pct": Math.round((product.category_confidence || 0) * 100) }}
            >
              <span>{Math.round((product.category_confidence || 0) * 100)}%</span>
            </div>
            <div>
              <div style={{ fontSize: 18, fontWeight: 800 }}>
                {product.display_name || product.category || "unknown"}
              </div>
              <div className="small muted">top-1 category probability (model argmax)</div>
            </div>
          </div>
          {categoryProbs.length > 0 && (
            <div className="mt">
              {categoryProbs.map((p) => (
                <Meter key={p.label} label={p.label} value={p.probability} />
              ))}
            </div>
          )}
          {product.identification_note ? (
            <div className="small muted mt">{product.identification_note}</div>
          ) : null}
        </div>

        <div className="card">
          <h3>Model run</h3>
          <dl className="kv">
            <dt>Architecture</dt>
            <dd>{model.name || "embedding → lstm → mlp (multi-task)"}</dd>
            <dt>Category head</dt>
            <dd>
              {model.categories?.length ? `${model.categories.length}-class softmax` : "—"}
              {model.category_accuracy != null
                ? ` · ${(model.category_accuracy * 100).toFixed(1)}% held-out accuracy`
                : ""}
            </dd>
            <dt>Profile heads</dt>
            <dd>
              {reliability.length
                ? reliability
                    .map(([k, v]) => `${k} ${v.level} (R² ${Number(v.r2).toFixed(2)})`)
                    .join(" · ")
                : "—"}
            </dd>
            <dt>Sequence</dt>
            <dd>
              {seq.tokens?.length ?? 0} of {seq.seq_len ?? "—"} positions read
              {input.extra_ingredients_beyond_model_window
                ? ` · ${input.extra_ingredients_beyond_model_window} past the model window`
                : ""}
            </dd>
            <dt>Disclaimer</dt>
            <dd className="small">{r.disclaimer || "—"}</dd>
          </dl>
        </div>
      </div>

      <Section title="Formulation profile">
        {profile.length === 0 ? (
          <Empty glyph="◈" title="No profile returned" hint="The model profile head returned no scores." />
        ) : (
          <div className="grid cols-2">
            <div className="card">
              {profile.map((p) => (
                <Meter
                  key={p.key}
                  label={p.icon ? `${p.icon} ${p.label}` : p.label}
                  value={p.score}
                  note={`band ${p.band}${profileConfidence[p.key] ? ` · ${profileConfidence[p.key].level}` : ""}`}
                />
              ))}
            </div>
            <div className="card">
              <h3>Summary</h3>
              <p style={{ margin: 0, lineHeight: 1.6 }}>{r.profile_summary || "—"}</p>
              <div className="row mt">
                {profile.map((p) => (
                  <Chip key={p.key} tone={TONE_BY_BAND[String(p.band || "").toLowerCase()] || ""}>
                    {p.label}: {p.band}
                  </Chip>
                ))}
              </div>
              {reliability.length ? (
                <div className="small muted mt">
                  Reliability from held-out data:{" "}
                  {reliability.map(([k, v]) => `${k} ${v.level}`).join(", ")}.
                </div>
              ) : null}
            </div>
          </div>
        )}
      </Section>

      <Section title="Key ingredients">
        {keyIngredients.length === 0 ? (
          <Empty glyph="✦" title="Not available for this record" />
        ) : (
          <div className="grid cols-3">
            {keyIngredients.slice(0, 9).map((ing, i) => (
              <div className="card" key={`${ing.ingredient}-${i}`}>
                <div className="spread">
                  <strong>{ing.ingredient}</strong>
                  <span className="chip gold">#{i + 1}</span>
                </div>
                <div className="mt">
                  <Meter label="Importance" value={ing.importance ?? 0} gold />
                </div>
                <div className="row">
                  {(ing.function_labels || []).slice(0, 3).map((f) => (
                    <Chip key={f} tone="lilac">{f}</Chip>
                  ))}
                </div>
                {ing.note ? (
                  <div className="small muted" style={{ marginTop: 8 }}>{ing.note}</div>
                ) : null}
                {explainOn && ing.model_attribution != null ? (
                  <div className="small muted" style={{ marginTop: 6 }}>
                    occlusion Δ {Number(ing.model_attribution).toFixed(3)}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        )}
      </Section>

      <Section title="Concern screening">
        <div className="grid cols-2">
          <div className="card">
            <div className="spread">
              <h3 style={{ margin: 0 }}>Findings</h3>
              <Chip tone={sevTone}>{concerns.highest_severity_label || "—"}</Chip>
            </div>
            <p className="small muted" style={{ marginTop: 10 }}>
              {concerns.summary || "No documented concerns matched this formulation."}
            </p>
            {activeFlags.length > 0 && (
              <div className="row mt">
                {activeFlags.map((f) => (
                  <Chip key={f} tone="rose">{f}</Chip>
                ))}
              </div>
            )}
            {concerns.regulatory_notes?.length ? (
              <div className="small mt">
                {concerns.regulatory_notes.map((n, i) => (
                  <div key={i}>⚑ {n}</div>
                ))}
              </div>
            ) : null}
            {concerns.ingredient_assessment ? (
              <div className="small muted mt">{concerns.ingredient_assessment}</div>
            ) : null}
          </div>
          <div className="card">
            {findings.length === 0 ? (
              <Empty glyph="✓" title="No rule-based findings" hint="Nothing in this list matched a documented concern rule." />
            ) : (
              findings.map((f, i) => (
                <div key={f.tag || i} style={{ marginTop: i ? 14 : 0 }}>
                  <div className="spread">
                    <strong>{f.label}</strong>
                    <SeverityBadge severity={f.severity} />
                  </div>
                  <div className="small muted" style={{ marginTop: 4 }}>{f.message}</div>
                  {f.advice ? <div className="small" style={{ marginTop: 4 }}>→ {f.advice}</div> : null}
                  {f.ingredients?.length ? (
                    <div className="small muted" style={{ marginTop: 4 }}>
                      {f.ingredients.join(", ")}
                      {f.ingredient_count > 1 ? ` (${f.ingredient_count} ingredients)` : ""}
                    </div>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </div>
        <div className="banner info mt">
          Screened against a local knowledge base of documented concern rules. Educational
          information only — not medical advice or a regulatory assessment.
        </div>
      </Section>

      <Section title="Why? (model explainability)">
        {!explainOn ? (
          <Empty
            glyph="◐"
            title="Explainability not available for this record"
            hint={explain.description || "Occlusion attributions are only present on newer records."}
          />
        ) : (
          <>
            <div className="small muted" style={{ marginBottom: 10 }}>
              {explain.description}
            </div>
            <div className="grid cols-2">
              {Object.entries(contributors).map(([metric, entry]) => {
                const p = profile.find((x) => x.key === metric);
                return (
                  <div className="card" key={metric}>
                    <div className="spread">
                      <h3 style={{ margin: 0 }}>{p?.label || metric}</h3>
                      <span className="small muted">Δ vs. baseline</span>
                    </div>
                    <div className="grid cols-2 mt" style={{ gap: 12 }}>
                      <div>
                        <div className="small" style={{ color: var_safe("--ok"), fontWeight: 700 }}>
                          ↑ pushed up
                        </div>
                        {(entry.positive || []).slice(0, 5).map((e) => (
                          <div key={e.ingredient} className="spread small" style={{ marginTop: 6 }}>
                            <span>{e.ingredient}</span>
                            <span className="delta pos">+{Number(e.delta ?? 0).toFixed(3)}</span>
                          </div>
                        ))}
                        {(entry.positive || []).length === 0 && (
                          <div className="small muted" style={{ marginTop: 6 }}>— none —</div>
                        )}
                      </div>
                      <div>
                        <div className="small" style={{ color: var_safe("--danger"), fontWeight: 700 }}>
                          ↓ pushed down
                        </div>
                        {(entry.negative || []).slice(0, 5).map((e) => (
                          <div key={e.ingredient} className="spread small" style={{ marginTop: 6 }}>
                            <span>{e.ingredient}</span>
                            <span className="delta neg">{Number(e.delta ?? 0).toFixed(3)}</span>
                          </div>
                        ))}
                        {(entry.negative || []).length === 0 && (
                          <div className="small muted" style={{ marginTop: 6 }}>— none —</div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </Section>

      {(r.mode === "image" || r.image_analysis) && (
        <Section title="Label image & OCR">
          <div className="grid cols-2">
            <div className="card">
              <div className="spread">
                <h3 style={{ margin: 0 }}>Pipeline outputs</h3>
                <button className="btn small" onClick={runPreview} disabled={previewing}>
                  {previewing ? "Loading…" : "⟳ Reload previews"}
                </button>
              </div>
              {assets.length === 0 ? (
                <Empty glyph="▣" title="No image assets stored" hint="Previews are written during image analysis." />
              ) : (
                <div className="gallery mt">
                  {assets.map((a) => (
                    <figure key={a.key}>
                      <img src={a.src} alt={a.label} />
                      <figcaption>{a.label}</figcaption>
                    </figure>
                  ))}
                </div>
              )}
              <dl className="kv mt">
                <dt>CNN text region</dt>
                <dd>
                  {detector.box
                    ? `[${detector.box.join(", ")}] · mean p ${Number(
                        detector.mean_text_probability ?? 0
                      ).toFixed(2)} · coverage ${Math.round((detector.coverage || 0) * 100)}%`
                    : detector.note || "not available"}
                </dd>
                <dt>Image quality</dt>
                <dd>
                  {quality.width
                    ? `${quality.width}×${quality.height} · blur ${Math.round(quality.blur_score ?? 0)}`
                    : "—"}
                </dd>
              </dl>
              {quality.warnings?.length ? (
                <div className="small" style={{ marginTop: 8 }}>
                  {quality.warnings.map((w, i) => (
                    <div key={i}>⚠ {w}</div>
                  ))}
                </div>
              ) : null}
            </div>
            <div className="card">
              <h3>OCR result</h3>
              <dl className="kv">
                <dt>Confidence</dt>
                <dd>{image.ocr_confidence != null ? `${Math.round(image.ocr_confidence)}%` : "—"}</dd>
                <dt>Engine</dt>
                <dd>
                  Tesseract{image.ocr_variant_used ? ` · variant ${image.ocr_variant_used}` : ""}
                </dd>
                <dt>Text source</dt>
                <dd>{image.ingredient_text_source || "ocr"}</dd>
                <dt>OCR time</dt>
                <dd>{image.ocr_ms_estimate != null ? `${Math.round(image.ocr_ms_estimate)} ms` : "—"}</dd>
              </dl>
              <label className="field">Extracted text</label>
              <div className="mono">{ocrText || "—"}</div>
            </div>
          </div>
        </Section>
      )}

      <Section title={`Ingredients (${ingredients.length})`}>
        <div className="card" style={{ overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Ingredient (INCI)</th>
                <th>Resolved</th>
                <th>Functions</th>
                <th>Concerns</th>
              </tr>
            </thead>
            <tbody>
              {ingredients.map((ing, i) => (
                <tr key={`${ing.ingredient}-${i}`}>
                  <td>
                    <strong>{ing.ingredient}</strong>
                    {ing.raw && ing.raw !== ing.ingredient ? (
                      <div className="small muted">raw: {ing.raw}</div>
                    ) : null}
                  </td>
                  <td>
                    <span className="small">
                      <Chip tone={ing.matched ? "info" : ""}>{ing.match_type || (ing.matched ? "matched" : "unknown")}</Chip>
                    </span>
                  </td>
                  <td className="muted">{(ing.function_labels || []).join(", ") || "—"}</td>
                  <td>
                    {(ing.concerns || []).map((c) => (
                      <span key={c.tag || c} className="chip rose" style={{ marginRight: 6 }}>
                        {c.label || c.tag || c}
                      </span>
                    ))}
                    {ing.note ? <div className="small muted">{ing.note}</div> : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <div className="banner info">
        DeepCos reports are educational: predictions come from a trained model with confidence
        scores, concerns come from documented sources, and neither replaces a dermatologist or a
        regulatory review. Model: {model.name || "deepcos"} · generated{" "}
        {String(r.created_at || "").replace("T", " ")} UTC.
      </div>
      {r.disclaimer ? <div className="small muted mt">{r.disclaimer}</div> : null}
    </>
  );
}

