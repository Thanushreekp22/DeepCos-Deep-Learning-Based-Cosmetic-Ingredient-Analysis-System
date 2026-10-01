import { Fragment, useEffect, useState } from "react";
import { api } from "../api.js";
import { Disclosure, Empty, ErrorBanner, Meter, Spinner } from "../components/ui.jsx";

// Vertical flow diagram: Node → Node → ...
function Flow({ steps }) {
  return (
    <div className="flow">
      {steps.map((step, i) => (
        <Fragment key={step}>
          <div className={`flow-node${i === steps.length - 1 ? " accent" : ""}`}>{step}</div>
          {i < steps.length - 1 && <div className="flow-arrow">↓</div>}
        </Fragment>
      ))}
    </div>
  );
}

function AvailabilityRow({ label, ok, detail }) {
  return (
    <div
      className="spread small"
      style={{ padding: "6px 0", borderBottom: "1px solid rgba(46,46,58,.6)" }}
    >
      <span>
        <span
          className={`status-dot ${ok ? "ok" : "warn"}`}
          style={{ display: "inline-block", marginRight: 8 }}
        />
        {label}
      </span>
      <span className="muted">{detail}</span>
    </div>
  );
}

// CNN vision metrics - the test-set context is always attached, especially
// important when a metric reads 100%.
function CnnMetrics({ title, data }) {
  if (!data) return null;
  const m = data.metrics || {};
  return (
    <div className="card">
      <h3>{title}</h3>
      <div className="small muted" style={{ marginBottom: 10 }}>
        Evaluated on the available image test set
        {data.dataset?.splits?.test ? ` (${data.dataset.splits.test} samples)` : ""}
      </div>
      <Meter label="Accuracy" value={m.accuracy} />
      <Meter label="Macro F1" value={m.macro_f1} />
      <Meter label="Precision" value={m.precision} />
      <Meter label="Recall" value={m.recall} />
      {data.notes ? (
        <div className="small muted mt">
          {Array.isArray(data.notes) ? data.notes.join(" ") : data.notes}
        </div>
      ) : null}
    </div>
  );
}

export default function ModelPage() {
  const [info, setInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.modelInfo().then(setInfo).catch((err) => setError(err.message));
    api.modelMetrics().then(setMetrics).catch(() => {});
  }, []);

  const ing = metrics?.ingredient_model;
  const cat = ing?.category_metrics || {};
  const prof = ing?.profile_metrics || {};
  const base = (p) => (p ? String(p).split(/[\\/]/).pop() : null);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>How DeepCos analyzes ingredients</h1>
          <p>
            A simplified technical overview of the models behind every report. The normal interface
            stays plain-language; this page is for inspection during a project demo.
          </p>
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="report-section">
        <h2>Ingredient Sequence Model</h2>
        <div className="card">
          <Flow
            steps={[
              "Ingredients",
              "Embedding",
              "LSTM",
              "Prediction Heads",
              "Product Type + Formulation Profiles",
            ]}
          />
          <dl className="kv mt">
            <dt>Embedding</dt>
            <dd>Converts ingredient names into numerical representations.</dd>
            <dt>LSTM</dt>
            <dd>Learns patterns from the order and combination of ingredients.</dd>
            <dt>Prediction heads</dt>
            <dd>Generate product category and formulation-profile predictions.</dd>
          </dl>
        </div>
      </div>

      <div className="report-section">
        <h2>Model Performance</h2>
        {!ing ? (
          <Empty
            glyph="◫"
            title="No metrics available yet"
            hint="Train the ingredient model to populate performance metrics."
          />
        ) : (
          <div className="grid cols-2">
            <div className="card">
              <h3>Product category classification</h3>
              <Meter label="Accuracy" value={cat.accuracy} />
              <Meter label="Macro F1" value={cat.macro_f1} />
            </div>
            <div className="card">
              <h3>Formulation prediction</h3>
              {Object.keys(prof).length === 0 ? (
                <span className="muted small">No profile metrics available.</span>
              ) : (
                Object.entries(prof).map(([name, v]) => (
                  <div key={name} className="meter">
                    <div className="meter-head">
                      <span style={{ textTransform: "capitalize" }}>{name.replace(/_/g, " ")}</span>
                      <span className="val">R² {Number(v.r2).toFixed(2)}</span>
                    </div>
                    <div className="meter-track">
                      <div
                        className="meter-fill gold"
                        style={{
                          width: `${Math.round(Math.max(0, Math.min(1, Number(v.r2))) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                ))
              )}
              <div className="small muted mt">
                R² shows how well the model explains each formulation characteristic on held-out
                data (higher is better).
              </div>
            </div>
          </div>
        )}
        {cat.per_category && (
          <Disclosure title="Per-category detail">
            <div className="card" style={{ overflowX: "auto" }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(cat.per_category).map(([name, v]) => (
                    <tr key={name}>
                      <td>
                        <strong>{name}</strong>
                      </td>
                      <td className="num">{Number(v.precision).toFixed(3)}</td>
                      <td className="num">{Number(v.recall).toFixed(3)}</td>
                      <td className="num">{Number(v.f1).toFixed(3)}</td>
                      <td className="num">{v.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Disclosure>
        )}
      </div>

      <div className="report-section">
        <h2>Image Analysis</h2>
        <div className="card">
          <Flow
            steps={[
              "Product label image",
              "Image processing",
              "CNN-based text-region detection",
              "OCR",
              "Ingredient text",
            ]}
          />
          <p className="small muted" style={{ margin: "14px 0 0" }}>
            The vision pipeline identifies the relevant text region and OCR converts the ingredient
            label into text for analysis.
          </p>
        </div>
      </div>

      <div className="report-section">
        <h2>Vision Pipeline</h2>
        <div className="grid cols-2">
          <CnnMetrics title="Text-region detection" data={metrics?.text_region_cnn} />
          <CnnMetrics title="Label category recognition" data={metrics?.label_cnn} />
        </div>
      </div>

      <div className="report-section">
        <h2>Explainability</h2>
        <div className="card">
          <p style={{ marginTop: 0 }}>DeepCos uses ingredient-level occlusion analysis.</p>
          <p>
            Each ingredient is temporarily removed from the input. The prediction is recalculated
            and compared with the original prediction.
          </p>
          <p style={{ marginBottom: 0 }}>
            This identifies ingredients that support or reduce each predicted formulation profile.
          </p>
        </div>
      </div>

      <div className="report-section">
        <h2>Training Behaviour</h2>
        <div className="card">
          <img
            src={api.curvesUrl()}
            alt="Training and validation loss curves"
            style={{ width: "100%", borderRadius: 10, display: "block" }}
            onError={(e) => {
              e.currentTarget.style.display = "none";
            }}
          />
          <div className="small muted mt">
            Training and validation loss recorded by the training scripts.
          </div>
        </div>
      </div>

      <Disclosure title="Model availability">
        {info === null ? (
          <Spinner />
        ) : (
          <>
            <AvailabilityRow
              label="Ingredient sequence model"
              ok={info.ingredient_model}
              detail={base(info.ingredient_model_path) || "missing"}
            />
            <AvailabilityRow
              label="Ingredient vocabulary"
              ok={!!info.vocabulary_path}
              detail={base(info.vocabulary_path) || "missing"}
            />
            <AvailabilityRow
              label="Text-region CNN"
              ok={info.text_region_cnn}
              detail={info.text_region_cnn ? "loaded at startup" : "not trained"}
            />
            <AvailabilityRow
              label="Label-category CNN"
              ok={info.label_category_cnn}
              detail={info.label_category_cnn ? "loaded at startup" : "not trained"}
            />
            <AvailabilityRow
              label="Text detector integration"
              ok={info.text_detector_loaded}
              detail={info.text_detector_loaded ? "wired into /api/analyze/image" : "fallback OCR"}
            />
            <AvailabilityRow
              label="OCR (Tesseract)"
              ok={info.ocr?.available}
              detail={info.ocr?.available ? `v${info.ocr.version}` : info.ocr?.hint || "unavailable"}
            />
            <AvailabilityRow
              label="Knowledge base"
              ok={(info.knowledge_base?.ingredient_count ?? 0) > 0}
              detail={
                info.knowledge_base
                  ? `v${info.knowledge_base.version} · ${info.knowledge_base.ingredient_count} ingredients`
                  : "—"
              }
            />
            {info.hint ? <div className="small muted mt">{info.hint}</div> : null}
          </>
        )}
      </Disclosure>

      <Disclosure title="Reproduce training">
        <div className="mono">
{`python -m ml.generate_dataset            # build dataset + splits
python -m ml.train_ingredient_model       # Embedding->LSTM->MLP (multi-task)
python -m ml.train_label_cnn              # AlexNet-style label category CNN
python -m ml.train_text_region_detector   # ingredient text region detector`}
        </div>
      </Disclosure>
    </>
  );
}
