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

// ---------------------------------------------------------------------------
// Shared presentation helpers (plain-language UI: no raw ML jargon)
// ---------------------------------------------------------------------------

const CATEGORY_ICONS = {
  serum: "🧪",
  moisturizer: "🧴",
  cleanser: "🫧",
  sunscreen: "☀️",
  toner: "💧",
  mask: "🎭",
  exfoliant: "🧪",
  "eye cream": "👁️",
};

// A friendly glyph per predicted product type (falls back to a neutral bottle).
export function categoryIcon(category) {
  return CATEGORY_ICONS[String(category || "").trim().toLowerCase()] || "🧴";
}

// Map a model probability to a plain confidence word for the normal UI.
export function confidenceWord(probability) {
  if (probability == null || Number.isNaN(Number(probability))) return "—";
  const p = Number(probability);
  if (p >= 0.7) return "High";
  if (p >= 0.45) return "Moderate";
  return "Low";
}

// Band name ("High"/"Moderate"/"Low"/"Minimal") -> tone class for .band-word.
export function bandTone(bandWord) {
  const b = String(bandWord || "").toLowerCase();
  if (b === "high") return "high";
  if (b === "low" || b === "minimal") return "low";
  return "moderate";
}

// Human date label for lists ("Today", "Yesterday", "Sep 29").
export function formatWhen(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso).replace("T", " ").slice(0, 16);
  const now = new Date();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === now.toDateString()) return "Today";
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// Simple visual bar for a formulation characteristic: emoji + name + band word.
// The numeric score drives the bar width only - it is never shown as a raw value.
export function ProfileBar({ icon, label, band, value }) {
  const pct = Math.round(Math.max(0, Math.min(1, value ?? 0)) * 100);
  return (
    <div className="profile-bar">
      <div className="profile-bar-head">
        <span className="profile-bar-label">
          {icon ? `${icon} ` : ""}
          {label}
        </span>
        {band ? <span className={`band-word ${bandTone(band)}`}>{band}</span> : null}
      </div>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${value == null ? 0 : pct}%` }} />
      </div>
    </div>
  );
}

// Progressive disclosure: technical / optional detail that stays collapsed
// until the user (or the demo) explicitly opens it.
export function Disclosure({ title, children, className = "" }) {
  return (
    <details className={`disclosure ${className}`.trim()}>
      <summary>{title}</summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}
