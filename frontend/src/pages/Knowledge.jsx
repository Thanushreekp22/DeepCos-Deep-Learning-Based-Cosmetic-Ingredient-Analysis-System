import { useEffect, useState } from "react";
import { api } from "../api.js";
import { Chip, Empty, ErrorBanner, Spinner } from "../components/ui.jsx";

export default function Knowledge() {
  const [stats, setStats] = useState(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [detail, setDetail] = useState(null);
  const [functions, setFunctions] = useState([]);
  const [concerns, setConcerns] = useState([]);
  const [error, setError] = useState("");
  const [searching, setSearching] = useState(false);

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
  }, []);

  const search = async (e) => {
    e?.preventDefault();
    setSearching(true);
    setError("");
    setDetail(null);
    try {
      const data = await api.knowledgeSearch(query.trim());
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  };

  const openDetail = async (name) => {
    try {
      setDetail(await api.knowledgeIngredient(name));
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Knowledge base</h1>
          <p>
            Local, source-documented ingredient data used by the rule engine: INCI facts,
            functional taxonomy and concern rules with confidence levels.
          </p>
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="grid cols-4">
        <div className="card">
          <h3>Ingredients</h3>
          <div className="big">{stats?.ingredient_count ?? "—"}</div>
        </div>
        <div className="card">
          <h3>Functions</h3>
          <div className="big">{stats?.function_taxonomy_count ?? functions.length}</div>
        </div>
        <div className="card">
          <h3>Concern rules</h3>
          <div className="big">{stats?.concern_rule_count ?? concerns.length}</div>
        </div>
        <div className="card">
          <h3>Aliases</h3>
          <div className="big">{stats?.alias_count ?? "—"}</div>
        </div>
      </div>

      <div className="report-section">
        <h2>Search</h2>
        <form onSubmit={search} className="row" style={{ alignItems: "stretch" }}>
          <input
            type="search"
            placeholder="e.g. niacinamide, retinol, fragrance, preservative…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ flex: 1, minWidth: 240 }}
          />
          <button className="btn primary" type="submit" disabled={searching}>
            {searching ? "Searching…" : "Search"}
          </button>
        </form>

        {searching ? (
          <Spinner />
        ) : results === null ? (
          <Empty glyph="⌕" title="Search the ingredient knowledge base" />
        ) : results.length === 0 ? (
          <Empty glyph="⌀" title="No matches" hint="Try a partial name or a function word." />
        ) : (
          <div className="grid cols-2 mt">
            {results.map((item) => (
              <div
                key={item.inci}
                className="card clickable"
                onClick={() => openDetail(item.inci)}
              >
                <div className="spread">
                  <strong>{item.inci}</strong>
                  <span className="small muted">{item.common ? "common" : ""}</span>
                </div>
                <div className="row" style={{ marginTop: 8 }}>
                  {(item.functions || []).slice(0, 4).map((f) => (
                    <Chip key={f} tone="lilac">
                      {f}
                    </Chip>
                  ))}
                  {(item.concerns || []).map((c) => (
                    <Chip key={c} tone="rose">
                      {c}
                    </Chip>
                  ))}
                </div>
                {item.note ? (
                  <div className="small muted" style={{ marginTop: 8 }}>
                    {item.note}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        )}

        {detail && (
          <div className="card mt">
            <div className="spread">
              <h3 style={{ margin: 0 }}>{detail.inci || detail.raw || detail.name || "Ingredient"}</h3>
              <button className="btn small" onClick={() => setDetail(null)}>
                ✕ Close
              </button>
            </div>
            <dl className="kv mt">
              <dt>INCI name</dt>
              <dd>{detail.inci || detail.raw || "—"}</dd>
              <dt>Match</dt>
              <dd>
                {detail.match_type || "—"}
                {detail.match_score != null ? ` · score ${Number(detail.match_score).toFixed(2)}` : ""}
              </dd>
              <dt>Aliases</dt>
              <dd>{(detail.info?.aliases || []).join(", ") || "—"}</dd>
              <dt>Functions</dt>
              <dd>{(detail.functions || []).join(", ") || "—"}</dd>
              <dt>Concerns</dt>
              <dd>{(detail.concerns || []).join(", ") || "none documented"}</dd>
              <dt>Description</dt>
              <dd>{detail.info?.note || "—"}</dd>
              <dt>Regulatory</dt>
              <dd>
                {detail.info?.regulatory_status || "—"}
                {detail.info?.regulatory_note ? ` · ${detail.info.regulatory_note}` : ""}
              </dd>
            </dl>
          </div>
        )}
      </div>

      <div className="report-section">
        <h2>Function taxonomy</h2>
        <div className="row">
          {functions.map((f) => (
            <Chip key={f.tag} tone="gold" >
              {f.label || f.tag}
            </Chip>
          ))}
          {functions.length === 0 && <span className="muted small">Loading…</span>}
        </div>
      </div>

      <div className="report-section">
        <h2>Concern rules</h2>
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
      </div>
    </>
  );
}

