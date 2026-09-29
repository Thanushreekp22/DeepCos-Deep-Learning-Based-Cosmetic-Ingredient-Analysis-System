// DeepCos API client. All requests go through the Vite dev proxy in
// development; set VITE_API_URL for a deployed backend.

const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export const apiUrl = (path) => `${BASE}${path}`;

async function parseJson(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    /* empty body */
  }
  if (!response.ok) {
    const message = payload?.detail || `${response.status} ${response.statusText}`;
    const error = new Error(message);
    error.hint = payload?.hint || "";
    error.status = response.status;
    throw error;
  }
  return payload;
}

export const api = {
  health: () => fetch(apiUrl("/api/health")).then(parseJson),
  modelInfo: () => fetch(apiUrl("/api/model/info")).then(parseJson),
  modelMetrics: () => fetch(apiUrl("/api/model/metrics")).then(parseJson),
  curvesUrl: () => apiUrl("/api/model/curves"),

  analyzeText: (text, { explain = true } = {}) =>
    fetch(apiUrl("/api/analyze/text"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, explain, persist: true }),
    }).then(parseJson),

  analyzeImage: (file, { explain = true } = {}) => {
    const form = new FormData();
    form.append("file", file);
    form.append("explain", String(explain));
    form.append("persist", "true");
    return fetch(apiUrl("/api/analyze/image"), { method: "POST", body: form }).then(parseJson);
  },

  listAnalyses: (limit = 50, offset = 0) =>
    fetch(apiUrl(`/api/analyses?limit=${limit}&offset=${offset}`)).then(parseJson),
  getAnalysis: (id) => fetch(apiUrl(`/api/analyses/${id}`)).then(parseJson),
  deleteAnalysis: (id) =>
    fetch(apiUrl(`/api/analyses/${id}`), { method: "DELETE" }).then(parseJson),
  previewImages: (id) =>
    fetch(apiUrl(`/api/analyses/${id}/preview`), { method: "POST" }).then(parseJson),
  exportUrl: (id, format) =>
    apiUrl(`/api/analyses/${id}/export?format=${encodeURIComponent(format)}`),

  knowledgeStats: () => fetch(apiUrl("/api/knowledge/stats")).then(parseJson),
  knowledgeSearch: (q, limit = 25) =>
    fetch(apiUrl(`/api/knowledge/search?q=${encodeURIComponent(q)}&limit=${limit}`)).then(parseJson),
  knowledgeIngredient: (name) =>
    fetch(apiUrl(`/api/knowledge/ingredient?name=${encodeURIComponent(name)}`)).then(parseJson),
  knowledgeFunctions: () => fetch(apiUrl("/api/knowledge/functions")).then(parseJson),
  knowledgeConcerns: () => fetch(apiUrl("/api/knowledge/concerns")).then(parseJson),
  samples: () => fetch(apiUrl("/api/knowledge/samples")).then(parseJson),
};
