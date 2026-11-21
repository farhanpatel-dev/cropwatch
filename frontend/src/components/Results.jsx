import { CLASS_INFO, label } from "../classInfo.js";
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
        <p className="ok">No crop stress found: the whole image looks healthy to the model.</p>
      ) : (
        <p className="summary">
          The model marked <strong>{formatArea(total)}</strong> of this field as possibly stressed.
          On the map, each coloured outline is one problem area; click it to see its size and how
          sure the model is.
        </p>
      )}
      <table className="stats">
        <thead>
          <tr>
            <th>Problem</th>
            <th title="Number of separate outlined areas">Areas found</th>
            <th>Total size</th>
          </tr>
        </thead>
        <tbody>
          {stress.map((c) => {
            const s = byId[c.id];
            return (
              <tr key={c.id} className={s ? "" : "muted"}>
                <td title={CLASS_INFO[c.name]?.what}>
                  <i className="swatch" style={{ background: c.colour }} />
                  {label(c.name)}
                  <small className="what">{CLASS_INFO[c.name]?.what}</small>
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
