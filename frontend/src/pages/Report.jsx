import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, apiUrl } from "../api.js";
import {
  Chip,
  Disclosure,
  Empty,
  ErrorBanner,
  ProfileBar,
  Spinner,
  categoryIcon,
  confidenceWord,
} from "../components/ui.jsx";

// profile band -> chip tone
const TONE_BY_BAND = { high: "rose", moderate: "gold", low: "info" };

// stored preview PNGs, in a sensible display order
const ASSET_LABELS = {
  original_png_base64: "Original photo",
  detected_region_png_base64: "Detected text region",
  heatmap_png_base64: "Text heatmap",
  cropped_region_png_base64: "Cropped region",
};
const ASSET_ORDER = Object.keys(ASSET_LABELS);

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
  const product = r.product || {};
  const image = r.image_analysis || {};
  const ocrText = (r.preprocessing || {}).ocr_text || "";
  const profile = r.profile || [];
  const explain = r.explainability || {};
  const contributors = explain.top_contributors || {};
  const explainOn = explain.method === "occlusion" && Object.keys(contributors).length > 0;
  const keyIngredients = r.key_ingredients || [];
  const concerns = r.concerns || {};
  const findings = concerns.findings || [];
  const ingredients = r.ingredients || [];
  const input = r.input || {};
  const warnings = r.warnings || [];

  // Optional AI enrichment: shown separately (and collapsed) from the reference.
  const ai = r.ai_enrichment || {};
  const aiItems = (ai.items || []).filter((it) => it.data);

  const ingredientCount = input.ingredient_count ?? ingredients.length;
  const category = product.display_name || product.category || "Unknown";
  const confidence = confidenceWord(product.category_confidence);
  const confidenceLevel = String(confidence).toLowerCase();
  // Alternative product types (everything except the top prediction).
  const otherTypes = Object.entries(product.category_probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .map(([label]) => label)
    .filter((label) => label !== product.category)
    .slice(0, 2);

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

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Analysis Report</h1>
          <p>
            Likely Product Type: <strong>{category}</strong> · {ingredientCount} ingredients
            analyzed
          </p>
          <div className="small muted mt">
            Analysis based on ingredient composition
            {r.mode === "image"
              ? " · analyzed from a label image"
              : r.mode
                ? " · analyzed from ingredient text"
                : ""}
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

      <Section title="Likely Product Type">
        <div className="card product-id">
          <div className="product-id-main">
            <span className="product-id-icon">{categoryIcon(product.category)}</span>
            <div>
              <div className="product-id-name">{category}</div>
              <div className="small muted">Most likely product type for this formulation</div>
            </div>
          </div>
          <div className="product-id-conf">
            Confidence: <span className={`band-word ${confidenceLevel}`}>{confidence}</span>
          </div>
          {otherTypes.length > 0 && (
            <div className="other-types">
              <div className="small muted">Other possible types</div>
              <div className="row">
                {otherTypes.map((label) => (
                  <Chip key={label}>{label}</Chip>
                ))}
              </div>
            </div>
          )}
        </div>
      </Section>

      <Section title="Formulation Profile">
        {profile.length === 0 ? (
          <Empty
            glyph="◈"
            title="No profile available"
            hint="No formulation characteristics were returned."
          />
        ) : (
          <div className="card">
            {profile.map((p) => (
              <ProfileBar key={p.key} icon={p.icon} label={p.label} band={p.band} value={p.score} />
            ))}
          </div>
        )}
      </Section>

      <Section title="Summary">
        <div className="card">
          <p className="summary-text">{r.profile_summary || "No summary available."}</p>
          <div className="row mt">
            {profile.map((p) => (
              <Chip key={p.key} tone={TONE_BY_BAND[String(p.band || "").toLowerCase()] || ""}>
                {p.label}: {p.band}
              </Chip>
            ))}
          </div>
        </div>
      </Section>

      <Section title="Key Ingredients">
        {keyIngredients.length === 0 ? (
          <Empty glyph="✦" title="Not available for this analysis" />
        ) : (
          <>
            <div className="grid cols-3">
              {keyIngredients.slice(0, 9).map((ing, i) => (
                <div className="card" key={`${ing.ingredient}-${i}`}>
                  <strong>{ing.ingredient}</strong>
                  {(ing.function_labels || []).length > 0 ? (
                    <div className="row" style={{ marginTop: 8 }}>
                      {(ing.function_labels || []).slice(0, 3).map((f) => (
                        <Chip key={f} tone="lilac">{f}</Chip>
                      ))}
                    </div>
                  ) : null}
                  {ing.note ? (
                    <div className="small muted" style={{ marginTop: 8 }}>{ing.note}</div>
                  ) : null}
                </div>
              ))}
            </div>
            <Disclosure title="Why is this important?">
              <p className="small muted" style={{ margin: 0 }}>
                Key ingredients are the ones that most influenced this prediction, combined with
                their documented function in the ingredient reference. The section below shows how
                each one affected the result.
              </p>
            </Disclosure>
          </>
        )}
      </Section>

      <Section title="Potential Concerns">
        {findings.length === 0 ? (
          <div className="card">
            <div style={{ color: "var(--ok)", fontWeight: 700 }}>
              ✓ No documented concern detected
            </div>
            <div className="small muted" style={{ marginTop: 6 }}>
              No ingredient matched the available concern rules.
            </div>
          </div>
        ) : (
          <div className="card">
            {findings.map((f, i) => (
              <div key={f.tag || i} style={{ marginTop: i ? 14 : 0 }}>
                <div style={{ fontWeight: 700 }}>ⓘ {f.label}</div>
                <div className="small muted" style={{ marginTop: 4 }}>{f.message}</div>
                {f.advice ? (
                  <div className="small" style={{ marginTop: 4 }}>→ {f.advice}</div>
                ) : null}
                {f.ingredients?.length ? (
                  <div className="small muted" style={{ marginTop: 4 }}>
                    {f.ingredients.join(", ")}
                    {f.ingredient_count > 1 ? ` (${f.ingredient_count} ingredients)` : ""}
                  </div>
                ) : null}
              </div>
            ))}
            {concerns.regulatory_notes?.length ? (
              <div className="small muted mt">
                {concerns.regulatory_notes.map((n, i) => (
                  <div key={i}>{n}</div>
                ))}
              </div>
            ) : null}
            <div className="small muted mt">
              This is informational and does not determine personal suitability.
            </div>
          </div>
        )}
      </Section>

      <Section title="Why these ingredients matter">
        {!explainOn ? (
          <Empty
            glyph="◐"
            title="Explanation not available for this analysis"
            hint="Ingredient-level explanations are only present on newer analyses."
          />
        ) : (
          <>
            <div className="small muted" style={{ marginBottom: 12 }}>
              DeepCos checks how individual ingredients influence each formulation prediction.
            </div>
            <div className="grid cols-2">
              {Object.entries(contributors).map(([metric, entry]) => {
                const p = profile.find((x) => x.key === metric);
                const positive = entry.positive || [];
                const negative = entry.negative || [];
                return (
                  <div className="card" key={metric}>
                    <h3 style={{ marginTop: 0 }}>
                      {p?.icon ? `${p.icon} ` : ""}
                      {p?.label || metric}
                    </h3>
                    <div className="impact">
                      <div className="impact-head supports">✓ Supports prediction</div>
                      {positive.length ? (
                        positive.map((e) => (
                          <div key={e.ingredient} className="impact-item">{e.ingredient}</div>
                        ))
                      ) : (
                        <div className="impact-none">No single ingredient stood out.</div>
                      )}
                    </div>
                    <div className="impact" style={{ marginTop: 12 }}>
                      <div className="impact-head reduces">− Reduces prediction</div>
                      {negative.length ? (
                        negative.map((e) => (
                          <div key={e.ingredient} className="impact-item">{e.ingredient}</div>
                        ))
                      ) : (
                        <div className="impact-none">No ingredient reduced this prediction.</div>
                      )}
                    </div>
                    <Disclosure title="Show technical explanation">
                      <div className="small muted" style={{ marginBottom: 8 }}>
                        These values represent the change in model output when an individual
                        ingredient is temporarily removed (occlusion-based attribution).
                      </div>
                      <table className="data">
                        <thead>
                          <tr>
                            <th>Ingredient</th>
                            <th style={{ textAlign: "right" }}>Attribution</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...positive, ...negative].map((e, idx) => (
                            <tr key={`${e.ingredient}-${idx}`}>
                              <td>{e.ingredient}</td>
                              <td className="num" style={{ textAlign: "right" }}>
                                {Number(e.delta ?? 0) >= 0 ? "+" : ""}
                                {Number(e.delta ?? 0).toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </Disclosure>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </Section>

      <Section title={`Ingredients (${ingredients.length})`}>
        <div className="card" style={{ overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Ingredient</th>
                <th>Function</th>
                <th>Information</th>
              </tr>
            </thead>
            <tbody>
              {ingredients.map((ing, i) => (
                <tr key={`${ing.ingredient}-${i}`}>
                  <td>
                    <strong>{ing.ingredient}</strong>
                    {ing.raw && ing.raw !== ing.ingredient ? (
                      <div className="small muted">label: {ing.raw}</div>
                    ) : null}
                  </td>
                  <td className="muted">{(ing.function_labels || []).join(", ") || "—"}</td>
                  <td className="muted">
                    {ing.note || "—"}
                    {(ing.concerns || []).length > 0 ? (
                      <div className="row" style={{ marginTop: 6 }}>
                        {(ing.concerns || []).map((c) => (
                          <span key={c.tag || c} className="chip rose">
                            {c.label || c.tag || c}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {(r.mode === "image" || r.image_analysis) && (
        <Disclosure title="Label image & OCR (technical)">
          <div className="grid cols-2">
            <div className="card">
              <div className="spread">
                <h3 style={{ margin: 0 }}>Image pipeline outputs</h3>
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
                <dt>Text region</dt>
                <dd>
                  {detector.box
                    ? `[${detector.box.join(", ")}] · coverage ${Math.round((detector.coverage || 0) * 100)}%`
                    : detector.note || "not available"}
                </dd>
                <dt>Image quality</dt>
                <dd>{quality.width ? `${quality.width}×${quality.height}` : "—"}</dd>
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
                <dt>OCR time</dt>
                <dd>{image.ocr_ms_estimate != null ? `${Math.round(image.ocr_ms_estimate)} ms` : "—"}</dd>
              </dl>
              <label className="field">Extracted text</label>
              <div className="mono">{ocrText || "—"}</div>
            </div>
          </div>
        </Disclosure>
      )}

      {aiItems.length > 0 && (
        <Disclosure title={`AI-assisted ingredient information (${aiItems.length})`}>
          <div className="small muted" style={{ marginBottom: 12 }}>
            AI-generated information for ingredients outside the reference data. This is
            informational only — it is not part of the reference data and not a safety
            determination.
          </div>
          <div className="grid cols-2">
            {aiItems.map((item) => {
              const data = item.data || {};
              return (
                <div
                  className="card"
                  key={item.normalized_name || item.ingredient}
                  style={{ borderLeft: "4px solid var(--gold)" }}
                >
                  <div className="spread">
                    <strong>{item.ingredient}</strong>
                    <Chip tone="gold">AI-assisted</Chip>
                  </div>
                  <p className="small" style={{ margin: "10px 0 0", lineHeight: 1.55 }}>
                    {data.description || "No description available."}
                  </p>
                  {(data.functions || []).length > 0 && (
                    <div className="mt">
                      <div className="small muted">Functions (AI-suggested)</div>
                      <div className="row" style={{ marginTop: 4 }}>
                        {data.functions.map((f) => (
                          <span className="chip" key={f}>{f}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {(data.possible_concerns || []).length > 0 && (
                    <div className="mt">
                      <div className="small muted">Possible concerns (informational)</div>
                      <div className="row" style={{ marginTop: 4 }}>
                        {data.possible_concerns.map((c) => (
                          <span className="chip rose" key={c}>{c}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Disclosure>
      )}

      <div className="banner info mt">
        <strong>ⓘ About this analysis</strong>
        <div className="small" style={{ marginTop: 6 }}>
          DeepCos provides educational information based on ingredient composition and trained
          model predictions. It does not determine personal safety or replace medical or
          dermatological advice.
        </div>
      </div>
    </>
  );
}

