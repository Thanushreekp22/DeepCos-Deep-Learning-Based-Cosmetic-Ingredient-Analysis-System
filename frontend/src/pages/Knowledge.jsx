import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { Chip, Empty, ErrorBanner, Spinner } from "../components/ui.jsx";

export default function Knowledge() {
  const [params] = useSearchParams();
  const [query, setQuery] = useState(params.get("q") || "");
  const [results, setResults] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [searching, setSearching] = useState(false);

  const runSearch = async (term) => {
    const value = (term ?? query).trim();
    if (!value) return;
    setSearching(true);
    setError("");
    setDetail(null);
    try {
      const data = await api.knowledgeSearch(value);
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  };

  // The sidebar search box can deep-link here with ?q=<ingredient>.
  useEffect(() => {
    const q = params.get("q");
    if (q) {
      setQuery(q);
      runSearch(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const search = (e) => {
    e.preventDefault();
    runSearch();
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
          <h1>Ingredient Guide</h1>
          <p>Search an ingredient to see what it is and what it does.</p>
        </div>
      </div>

      <ErrorBanner error={error} />

      <div className="card">
        <form onSubmit={search} className="row" style={{ alignItems: "stretch" }}>
          <input
            type="search"
            placeholder="Search an ingredient... e.g. niacinamide, retinol"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ flex: 1, minWidth: 240 }}
          />
          <button className="btn primary" type="submit" disabled={searching}>
            {searching ? "Searching…" : "Search"}
          </button>
        </form>
      </div>

      {searching ? (
        <Spinner />
      ) : results === null ? (
        <Empty
          glyph="⌕"
          title="Search the ingredient guide"
          hint="Try an ingredient name, e.g. niacinamide."
        />
      ) : results.length === 0 ? (
        <Empty glyph="⌀" title="No matches" hint="Try a partial name or a function word." />
      ) : (
        <div className="grid cols-2 mt">
          {results.map((item) => (
            <div key={item.inci} className="card clickable" onClick={() => openDetail(item.inci)}>
              <strong>{item.inci}</strong>
              {(item.functions || []).length > 0 ? (
                <div className="row" style={{ marginTop: 8 }}>
                  {(item.functions || []).slice(0, 4).map((f) => (
                    <Chip key={f} tone="lilac">{f}</Chip>
                  ))}
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
            <button className="btn small" onClick={() => setDetail(null)}>✕ Close</button>
          </div>
          <div className="guide-block">
            <div className="guide-label">What it is</div>
            <div>{detail.info?.note || "No description available."}</div>
          </div>
          <div className="guide-block">
            <div className="guide-label">Functions</div>
            {(detail.functions || []).length > 0 ? (
              <ul className="guide-list">
                {(detail.functions || []).map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            ) : (
              <span className="muted">No documented functions.</span>
            )}
          </div>
          {(detail.concerns || []).length > 0 ? (
            <div className="guide-block">
              <div className="guide-label">Documented considerations</div>
              <div className="row">
                {(detail.concerns || []).map((c) => (
                  <Chip key={c.tag || c} tone="rose">{c.label || c.tag || c}</Chip>
                ))}
              </div>
            </div>
          ) : null}
          <div className="guide-block">
            <div className="guide-label">DeepCos information</div>
            <div className="small muted">
              Used as an ingredient reference when analyzing cosmetic formulations.
            </div>
          </div>
        </div>
      )}

    </>
  );
}

