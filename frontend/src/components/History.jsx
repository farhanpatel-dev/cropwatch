export default function History({ analyses, currentId, onOpen }) {
  if (!analyses.length) return null;
  return (
    <section className="panel">
      <h2>Recent analyses</h2>
      <ul className="history">
        {analyses.map((a) => (
          <li key={a.id}>
            <button
              type="button"
              className={a.id === currentId ? "active" : ""}
              onClick={() => onOpen(a.id)}
            >
              <span>{a.name || a.source_filename || a.id.slice(0, 8)}</span>
              <time dateTime={a.created_at}>{new Date(a.created_at).toLocaleDateString()}</time>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
