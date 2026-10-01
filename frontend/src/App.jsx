import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { api } from "./api.js";
import Dashboard from "./pages/Dashboard.jsx";
import Analyze from "./pages/Analyze.jsx";
import Report from "./pages/Report.jsx";
import History from "./pages/History.jsx";
import Knowledge from "./pages/Knowledge.jsx";
import KnowledgeBase from "./pages/KnowledgeBase.jsx";
import ModelPage from "./pages/ModelPage.jsx";
import Help from "./pages/Help.jsx";

// The sidebar is deliberately user-facing only. Technical views (/model,
// /knowledgebase, /help) are reached through the Search & commands box below.
const NAV = [
  { to: "/", label: "Dashboard", icon: "◈", end: true },
  { to: "/analyze", label: "New Analysis", icon: "✚" },
  { to: "/history", label: "History", icon: "≣" },
  { to: "/knowledge", label: "Ingredient Guide", icon: "⬡" },
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [command, setCommand] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((err) => setHealthError(err.message));
  }, []);

  // Search box doubles as a command router: /model, /knowledgebase, /help.
  // Anything else is treated as an ingredient-guide search.
  const runCommand = (event) => {
    event.preventDefault();
    const raw = command.trim();
    if (!raw) return;
    const verb = raw.split(/\s+/)[0].toLowerCase();
    if (verb === "/model") navigate("/model");
    else if (verb === "/knowledgebase") navigate("/knowledgebase");
    else if (verb === "/help") navigate("/help");
    else navigate(`/knowledge?q=${encodeURIComponent(raw)}`);
    setCommand("");
  };

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
        <form className="sidebar-search" onSubmit={runCommand}>
          <input
            type="search"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="Search or type a command"
            aria-label="Search or type a command"
          />
          <div className="hint">
            Commands: <code>/model</code> <code>/knowledgebase</code> <code>/help</code>
          </div>
        </form>
      </aside>

      <main className="main">
        {!healthError && health && health.status !== "ok" && (
          <div className="banner warn">DeepCos is running with limited functionality.</div>
        )}
        {healthError && <div className="banner error">DeepCos cannot reach the analysis service.</div>}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/report/:id" element={<Report />} />
          <Route path="/history" element={<History />} />
          <Route path="/knowledge" element={<Knowledge />} />
          <Route path="/knowledgebase" element={<KnowledgeBase />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/help" element={<Help />} />
        </Routes>
      </main>
    </div>
  );
}
