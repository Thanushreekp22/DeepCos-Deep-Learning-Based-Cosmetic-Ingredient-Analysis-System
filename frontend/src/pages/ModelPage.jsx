import { useEffect, useState } from "react";
import { api } from "../api.js";
import { Empty, ErrorBanner, Meter, Spinner } from "../components/ui.jsx";

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

function CnnMetrics({ title, data }) {
  if (!data) return null;
  const m = data.metrics || {};
  return (
    <div className="card">
      <h3>{title}</h3>
      <div className="small muted" style={{ marginBottom: 10 }}>
        {data.task} · trained {data.trained_at || "—"}
        {data.training_seconds ? ` · ${Math.round(data.training_seconds)}s` : ""}
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
          <h1>Models</h1>
          <p>
            The trained artefacts behind every report: an Embedding→LSTM→MLP multi-task network
            for formulation profiling, plus two AlexNet-style CNNs (ingredient text-region
            detection and label category recognition).
          </p>
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="grid cols-2">
        <div className="card">
          <h3>Availability</h3>
          {info === null ? (
            <Spinner />
          ) : (
            <>
              <AvailabilityRow
                label="Ingredient model (LSTM + MLP heads)"
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
                detail={info.text_region_cnn ? "loaded at startup" : "train it"}
              />
              <AvailabilityRow
                label="Label-category CNN"
                ok={info.label_category_cnn}
                detail={info.label_category_cnn ? "loaded at startup" : "train it"}
              />
              <AvailabilityRow
                label="Text detector integration"
                ok={info.text_detector_loaded}
                detail={info.text_detector_loaded ? "wired into /api/analyze/image" : "fallback OCR"}
              />
              <AvailabilityRow
                label="OCR (Tesseract)"
                ok={info.ocr?.available}
                detail={
                  info.ocr?.available ? `v${info.ocr.version}` : info.ocr?.hint || "unavailable"
                }
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
        </div>

        <div className="card">
          <h3>Ingredient model · held-out test</h3>
          {!ing ? (
            <Empty
              glyph="◫"
              title="No metrics file yet"
              hint="Run python -m ml.train_ingredient_model"
            />
          ) : (
            <>
              <div className="small muted" style={{ marginBottom: 10 }}>
                trained {ing.trained_at} · {Math.round(ing.training_seconds || 0)}s ·{" "}
                {(ing.vocabulary_size ?? 0).toLocaleString()} tokens
              </div>
              <Meter label="Category accuracy" value={cat.accuracy} note="8-class product head" />
              <Meter label="Category macro F1" value={cat.macro_f1} />
              <div className="spread small" style={{ marginTop: 4 }}>
                <span className="muted">Test loss</span>
                <span>{ing.test_loss != null ? Number(ing.test_loss).toFixed(4) : "—"}</span>
              </div>
              <div className="mt small" style={{ fontWeight: 700, marginBottom: 6 }}>
                Profile regression (per metric)
              </div>
              {Object.entries(prof).map(([name, v]) => (
                <div key={name} className="meter">
                  <div className="meter-head">
                    <span style={{ textTransform: "capitalize" }}>
                      {name.replace(/_/g, " ")}
                    </span>
                    <span className="val">
                      MAE {Number(v.mae).toFixed(3)} · R² {Number(v.r2).toFixed(3)}
                    </span>
                  </div>
                  <div className="meter-track">
                    <div
                      className="meter-fill gold"
                      style={{ width: `${Math.round((v.band_accuracy || 0) * 100)}%` }}
                    />
                  </div>
                  <div className="small muted" style={{ marginTop: 3 }}>
                    band accuracy {Math.round((v.band_accuracy || 0) * 100)}%
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      <div className="report-section">
        <h2>Per-category breakdown</h2>
        {!cat.per_category ? (
          <Empty glyph="▦" title="No per-category metrics on this record" />
        ) : (
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
        )}
      </div>

      <div className="report-section">
        <h2>CNN metrics</h2>
        <div className="grid cols-2">
          <CnnMetrics title="Ingredient text-region detector" data={metrics?.text_region_cnn} />
          <CnnMetrics title="Label category CNN" data={metrics?.label_cnn} />
        </div>
      </div>

      <div className="report-section">
        <h2>Training curves</h2>
        <div className="card">
          <img
            src={api.curvesUrl()}
            alt="Training curves for all models"
            style={{ width: "100%", borderRadius: 10, display: "block" }}
            onError={(e) => {
              e.currentTarget.style.display = "none";
            }}
          />
          <div className="small muted mt">
            Loss and metric history exported by the training scripts (keep
            artifacts/models/curves.png).
          </div>
        </div>
      </div>

      <div className="report-section">
        <h2>Reproduce training</h2>
        <div className="mono">
{`python -m ml.generate_dataset            # build dataset + splits
python -m ml.train_ingredient_model       # Embedding->LSTM->MLP (multi-task)
python -m ml.train_label_cnn              # AlexNet-style label category CNN
python -m ml.train_text_region_detector   # ingredient text region detector`}
        </div>
      </div>
    </>
  );
}
