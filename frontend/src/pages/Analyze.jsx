import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { ErrorBanner, Spinner } from "../components/ui.jsx";

export default function Analyze() {
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

  const clearFile = () => {
    setFile(null);
    setPreview("");
    if (fileInput.current) fileInput.current.value = "";
  };

  // Picking an example switches the analysis to the pasted ingredient list.
  const useSample = (value) => {
    clearFile();
    setText(value);
  };

  const run = async () => {
    setError("");
    const useImage = !!file;
    if (!useImage && !text.trim()) {
      setError("Upload a product label image or paste the ingredient list to analyze.");
      return;
    }
    setBusy(true);
    try {
      const report = useImage ? await api.analyzeImage(file) : await api.analyzeText(text.trim());
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
          <h1>Analyze a Product</h1>
          <p>Upload a product label or enter the ingredient list manually.</p>
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="card">
        <h3>Upload Image</h3>
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
              <div className="mt">
                <button
                  type="button"
                  className="btn small"
                  onClick={(e) => {
                    e.stopPropagation();
                    clearFile();
                  }}
                >
                  Remove image
                </button>
              </div>
            </>
          ) : (
            <>
              <div>Product label</div>
              <div className="small muted mt">
                Drop an image here or click to browse · PNG or JPEG
              </div>
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
      </div>

      <div className="or-divider">OR</div>

      <div className="card">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste ingredients here..."
        />
        {samples.length > 0 && (
          <>
            <div className="field-label">Try an example</div>
            <div className="row">
              {samples.map((s) => (
                <button
                  key={s.id}
                  className="btn small"
                  onClick={() => useSample(s.text)}
                  type="button"
                >
                  {s.title}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="row mt">
        <button className="btn primary" onClick={run} disabled={busy}>
          {busy ? "Analyzing…" : "Analyze Product"}
        </button>
        {busy ? <span className="small muted">Analyzing the ingredients…</span> : null}
      </div>
      {busy && <Spinner />}
    </>
  );
}
