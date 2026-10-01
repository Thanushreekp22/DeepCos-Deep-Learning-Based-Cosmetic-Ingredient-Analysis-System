import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { Chip, Disclosure, Empty, ErrorBanner } from "../components/ui.jsx";

// Technical view of the ingredient reference behind DeepCos. This is the
// "how is it stored / mapped / enriched" level, reached via the /knowledgebase
// command - it is intentionally not part of the normal navigation.
export default function KnowledgeBase() {
  const [stats, setStats] = useState(null);
  const [functions, setFunctions] = useState([]);
  const [concerns, setConcerns] = useState([]);
  const [learned, setLearned] = useState(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    api.knowledgeStats().then(setStats).catch((err) => setError(err.message));
    api.knowledgeFunctions().then((d) => setFunctions(d.taxonomy || [])).catch(() => {});
    api
      .knowledgeConcerns()
      .then((d) => {
        // rules is a map of tag -> {label, severity, message, advice}
        setConcerns(Object.entries(d.rules || {}).map(([tag, rule]) => ({ tag, ...rule })));
      })
      .catch(() => {});
    api.knowledgeLearned().then(setLearned).catch(() => {});
  }, []);

  // Remove one entry from the AI-assisted overlay: the ingredient stops
  // resolving locally, and the next analysis would enrich it again.
  const removeLearned = async (name) => {
    setError("");
    try {
      await api.deleteLearned(name);
      setLearned(await api.knowledgeLearned());
      setStats(await api.knowledgeStats());
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Knowledge Base</h1>
          <p>
            DeepCos uses a structured ingredient reference alongside the machine-learning models.
          </p>
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="card">
        <h3>Reference information includes</h3>
        <ul className="guide-list">
          <li>Ingredient names</li>
          <li>Ingredient functions</li>
          <li>Ingredient aliases</li>
          <li>Potential concern rules</li>
          <li>Ingredient descriptions</li>
        </ul>
      </div>

      <div className="report-section">
        <h2>Reference coverage</h2>
        <div className="grid cols-3">
          <div className="card">
            <h3>Ingredients</h3>
            <div className="big">{stats?.ingredient_count ?? "—"}</div>
          </div>
          <div className="card">
            <h3>Functions</h3>
            <div className="big">{stats?.function_taxonomy_count ?? functions.length ?? "—"}</div>
          </div>
          <div className="card">
            <h3>Concern rules</h3>
            <div className="big">{stats?.concern_rule_count ?? concerns.length ?? "—"}</div>
          </div>
        </div>
      </div>

      <Disclosure title="Function taxonomy">
        <div className="row">
          {functions.map((f) => (
            <Chip key={f.tag} tone="gold">
              {f.label || f.tag}
            </Chip>
          ))}
          {functions.length === 0 && <span className="muted small">Loading…</span>}
        </div>
      </Disclosure>

      <Disclosure title="Concern rules">
        {concerns.length === 0 ? (
          <Empty glyph="⚠" title="Loading rules…" />
        ) : (
          <div className="card" style={{ overflowX: "auto" }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Rule</th>
                  <th>Tag</th>
                  <th>Severity</th>
                  <th>Message</th>
                  <th>Advice</th>
                </tr>
              </thead>
              <tbody>
                {concerns.map((c, i) => (
                  <tr key={`${c.tag}-${i}`}>
                    <td>
                      <strong>{c.label || c.tag}</strong>
                    </td>
                    <td className="muted">{c.tag}</td>
                    <td>
                      <span className={`sev ${(c.severity || "low").toLowerCase()}`}>
                        {c.severity || "low"}
                      </span>
                    </td>
                    <td className="muted">{c.message || "—"}</td>
                    <td className="muted">{c.advice || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Disclosure>

      <Disclosure title={`AI-assisted entries${learned?.count ? ` (${learned.count})` : ""}`}>
        <p className="small muted" style={{ marginTop: 0 }}>
          Entries created when validated AI enrichment supplements the reference data. They stay
          labelled as unreviewed AI content and can be removed here.
        </p>
        {!learned?.entries?.length ? (
          <Empty
            glyph="✦"
            title="No AI-assisted entries"
            hint={
              learned?.promotion_enabled
                ? "Analyze a product with an undocumented ingredient to create one."
                : "Knowledge-base promotion is currently switched off."
            }
          />
        ) : (
          <div className="grid cols-2">
            {learned.entries.map((entry) => (
              <div
                key={entry.normalized_key}
                className="card"
                style={{ borderLeft: "4px solid var(--gold)" }}
              >
                <div className="spread">
                  <strong>{entry.inci}</strong>
                  <Chip tone="gold">AI-assisted · unreviewed</Chip>
                </div>
                <div className="row" style={{ marginTop: 8 }}>
                  {(entry.functions || []).map((f) => (
                    <Chip key={f} tone="lilac">
                      {f}
                    </Chip>
                  ))}
                  {(entry.concerns || []).map((c) => (
                    <Chip key={c} tone="rose">
                      {c}
                    </Chip>
                  ))}
                </div>
                {(entry.unmapped_functions?.length || entry.unmapped_concerns?.length) ? (
                  <div className="small muted" style={{ marginTop: 8 }}>
                    Unmapped tags:{" "}
                    {[...(entry.unmapped_functions || []), ...(entry.unmapped_concerns || [])].join(
                      ", "
                    )}
                  </div>
                ) : null}
                <div className="small muted" style={{ marginTop: 8 }}>
                  promoted {String(entry.promoted_at || "").slice(0, 19).replace("T", " ")}
                  {entry.confidence ? ` · confidence ${entry.confidence}` : ""}
                </div>
                <div className="row mt">
                  <button
                    className="btn small"
                    onClick={() =>
                      navigate(
                        `/knowledge?q=${encodeURIComponent(entry.inci || entry.normalized_key)}`
                      )
                    }
                  >
                    Resolve
                  </button>
                  <button
                    className="btn danger small"
                    onClick={() => removeLearned(entry.normalized_key)}
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Disclosure>
    </>
  );
}