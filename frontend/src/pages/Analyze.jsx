import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { ErrorBanner, Spinner } from "../components/ui.jsx";

export default function Analyze() {
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [samples, setSamples] = useState([]);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    api
      .samples()
      .then((data) => setSamples(data.samples || []))
      .catch(() => {});
  }, []);

  const pickFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError("");
  };

  const run = async () => {
    setError("");
    if (mode === "text" && !text.trim()) {
      setError("Paste an ingredient list first (or pick an example below).");
      return;
    }
    if (mode === "image" && !file) {
      setError("Choose a label photo first.");
      return;
    }
    setBusy(true);
    try {
      const report =
        mode === "text"
          ? await api.analyzeText(text.trim())
          : await api.analyzeImage(file);
      navigate(`/report/${report.analysis_id}`);
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
          <h1>New analysis</h1>
          <p>
            Paste the ingredient list from a product page, or upload a photo of the label. The
            report predicts the product category, formulation profile and potential concerns with
            per-ingredient explanations.
          </p>
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="tabs">
        <button
          className={`tab${mode === "text" ? " active" : ""}`}
          onClick={() => setMode("text")}
          disabled={busy}
        >
          ✎ Ingredient text
        </button>
        <button
          className={`tab${mode === "image" ? " active" : ""}`}
          onClick={() => setMode("image")}
          disabled={busy}
        >
          ▣ Label photo
        </button>
      </div>

      <div className="card">
        {mode === "text" ? (
          <>
            <label className="field" htmlFor="ingredients">
              Ingredient list (INCI order, comma or newline separated)
            </label>
            <textarea
              id="ingredients"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Aqua, Glycerin, Niacinamide, Sodium Hyaluronate, Panthenol, ..."
            />
            {samples.length > 0 && (
              <>
                <label className="field">Try an example</label>
                <div className="row">
                  {samples.map((s) => (
                    <button
                      key={s.id}
                      className="btn small"
                      onClick={() => setText(s.text)}
                      type="button"
                    >
                      {s.title}
                    </button>
                  ))}
                </div>
              </>
            )}
          </>
        ) : (
          <>
            <label className="field">Photo of the ingredient label</label>
            <div
              className={`file-drop${file ? " filled" : ""}`}
              onClick={() => fileInput.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                pickFile(e.dataTransfer.files?.[0]);
              }}
            >
              {file ? (
                <>
                  <div>
                    <strong>{file.name}</strong> · {(file.size / 1024).toFixed(0)} KB
                  </div>
                  <img src={preview} alt="selected label preview" />
                </>
              ) : (
                <>
                  <div>▣ Drop a photo here or click to browse</div>
                  <div className="small muted mt">PNG / JPEG up to 12 MB</div>
                </>
              )}
              <input
                ref={fileInput}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                style={{ display: "none" }}
                onChange={(e) => pickFile(e.target.files?.[0])}
              />
            </div>
            <div className="small muted mt">
              The CNN first locates the ingredient text block, then OCR reads it - you can correct
              the text from the report if needed.
            </div>
          </>
        )}

        <div className="row mt">
          <button className="btn primary" onClick={run} disabled={busy}>
            {busy ? "Analysing…" : "Analyse"}
          </button>
          <span className="small muted">
            {busy
              ? mode === "image"
                ? "CNN text detection → OCR → LSTM profiling (a few seconds)…"
                : "Running occlusion explainability…"
              : "The report is saved to history automatically."}
          </span>
        </div>
        {busy && <Spinner />}
      </div>
    </>
  );
}
