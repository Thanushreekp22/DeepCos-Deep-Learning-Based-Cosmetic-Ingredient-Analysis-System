// Command reference for the sidebar search box (invoked with /help).
export default function Help() {
  return (
    <>
      <div className="page-head">
        <div>
          <h1>DeepCos Commands</h1>
          <p>Type a command in the sidebar search box.</p>
        </div>
      </div>

      <div className="card">
        <div className="command-row">
          <code>/model</code>
          <span>View a simplified technical overview.</span>
        </div>
        <div className="command-row">
          <code>/knowledgebase</code>
          <span>View the ingredient knowledge system.</span>
        </div>
        <div className="command-row">
          <code>/help</code>
          <span>Show available commands.</span>
        </div>
      </div>
    </>
  );
}