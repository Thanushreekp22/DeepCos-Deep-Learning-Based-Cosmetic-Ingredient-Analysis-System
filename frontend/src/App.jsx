import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api } from "./api.js";
import Dashboard from "./pages/Dashboard.jsx";
import Analyze from "./pages/Analyze.jsx";
import Report from "./pages/Report.jsx";
import History from "./pages/History.jsx";
import Knowledge from "./pages/Knowledge.jsx";
import ModelPage from "./pages/ModelPage.jsx";

const NAV = [
  { to: "/", label: "Dashboard", icon: "◈", end: true },
  { to: "/analyze", label: "New analysis", icon: "✚" },
  { to: "/history", label: "History", icon: "≣" },
  { to: "/knowledge", label: "Knowledge base", icon: "⬡" },
  { to: "/model", label: "Models", icon: "◉" },
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((err) => setHealthError(err.message));
  }, []);

  const ready = health?.ingredient_model?.trained;
  const db = health?.database?.backend;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">DeepCos</span>
          <span className="brand-sub">Ingredient Intelligence</span>
        </div>
        <nav>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className={`status-dot ${ready ? "ok" : "warn"}`} />
          <div className="sidebar-status">
            <div>{healthError ? "API unreachable" : ready ? "Models ready" : "Models missing"}</div>
            <small>
              {db === "mongodb" ? "MongoDB" : db === "json" ? "JSON fallback" : "connecting…"}
              {health ? ` · KB ${health.knowledge_base?.ingredient_count ?? "—"}` : ""}
            </small>
          </div>
        </div>
      </aside>

      <main className="main">
        {!healthError && health && health.status !== "ok" && (
          <div className="banner warn">
            DeepCos is running in <strong>degraded</strong> mode:{" "}
            {health.ingredient_model?.hint || "check /api/health for details"}
          </div>
        )}
        {healthError && <div className="banner error">API unreachable: {healthError}</div>}
        <Routes>
          <Route path="/" element={<Dashboard health={health} />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/report/:id" element={<Report />} />
          <Route path="/history" element={<History />} />
          <Route path="/knowledge" element={<Knowledge />} />
          <Route path="/model" element={<ModelPage />} />
        </Routes>
      </main>
    </div>
  );
}
