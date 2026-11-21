import { formatArea } from "../geo.js";

export default function Results({ analysis, classes }) {
  const byId = Object.fromEntries(analysis.stats.map((s) => [s.class_id, s]));
  const stress = classes.filter((c) => c.id !== 0);
  const total = analysis.stats.reduce((sum, s) => sum + s.area_ha, 0);

  return (
    <section className="panel">
      <h2>{analysis.name || analysis.source_filename || "Result"}</h2>
      <p className="meta">
        {new Date(analysis.created_at).toLocaleString()} · model {analysis.model.name} v
        {analysis.model.version}
      </p>
      {analysis.stats.length === 0 ? (
        <p className="ok">No crop stress found in this image.</p>
      ) : (
        <p className="summary">
          <strong>{formatArea(total)}</strong> flagged as stressed
        </p>
      )}
      <table className="stats">
        <thead>
          <tr>
            <th>Class</th>
            <th>Patches</th>
            <th>Area</th>
          </tr>
        </thead>
        <tbody>
          {stress.map((c) => {
            const s = byId[c.id];
            return (
              <tr key={c.id} className={s ? "" : "muted"}>
                <td>
                  <i className="swatch" style={{ background: c.colour }} />
                  {c.name.replace(/_/g, " ")}
                </td>
                <td>{s ? s.polygon_count : 0}</td>
                <td>{s ? formatArea(s.area_ha) : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="hint">
        The model finds about two thirds of nutrient-deficiency areas and half of weed areas, and
        sometimes marks healthy ground as stressed. Treat results as a guide for scouting.
      </p>
    </section>
  );
}
