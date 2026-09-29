// Small shared UI primitives used across pages.

export function Meter({ label, value, suffix = "", note, gold }) {
  const pct = Math.round(Math.max(0, Math.min(1, value ?? 0)) * 100);
  return (
    <div className="meter">
      <div className="meter-head">
        <span>
          {label}
          {note ? <span className="muted small"> · {note}</span> : null}
        </span>
        <span className="val">
          {value == null ? "—" : `${(value * 100).toFixed(1)}${suffix}`}
        </span>
      </div>
      <div className="meter-track">
        <div
          className={`meter-fill${gold ? " gold" : ""}`}
          style={{ width: `${value == null ? 0 : pct}%` }}
        />
      </div>
    </div>
  );
}

export function Chip({ tone = "", children }) {
  return <span className={`chip ${tone}`}>{children}</span>;
}

export function SeverityBadge({ severity }) {
  const s = (severity || "info").toLowerCase();
  const label = s === "high-watch" ? "high watch" : s;
  return <span className={`sev ${s}`}>{label}</span>;
}

export function Spinner() {
  return <div className="spinner" aria-label="loading" />;
}

export function Empty({ glyph = "◇", title, hint }) {
  return (
    <div className="empty">
      <span className="glyph">{glyph}</span>
      <div>{title}</div>
      {hint ? <div className="small">{hint}</div> : null}
    </div>
  );
}

export function ErrorBanner({ error, onRetry }) {
  if (!error) return null;
  const message = typeof error === "string" ? error : error.message;
  const hint = typeof error === "object" ? error.hint : "";
  return (
    <div className="banner error">
      <strong>Something went wrong:</strong> {message}
      {hint ? <div className="small mt">{hint}</div> : null}
      {onRetry ? (
        <div className="mt">
          <button className="btn small" onClick={onRetry}>
            Retry
          </button>
        </div>
      ) : null}
    </div>
  );
}

export function confidenceTone(confidence) {
  if (confidence >= 0.7) return "ok";
  if (confidence >= 0.45) return "warn";
  return "danger";
}
