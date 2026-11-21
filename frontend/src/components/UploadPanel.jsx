import { useState } from "react";

import {
  ALLOWED,
  boundsFromMap,
  boxAround,
  extension,
  MAX_PIXELS_SIDE,
  MAX_UPLOAD_MB,
  NEEDS_BOUNDS,
  parseBounds,
} from "../geo.js";

// Agriculture-Vision images are 512 x 512 px at about 10 cm per pixel, so about 51 m wide.
// Their real locations are not published: these map positions are made up (Iowa farmland).
const SAMPLE_SIZE_M = 51.2;
const SAMPLES = [
  { file: "drydown.jpg", label: "Drydown", lat: 41.9905, lon: -93.6195 },
  { file: "nutrient_deficiency.jpg", label: "Nutrient deficiency", lat: 41.9905, lon: -93.6185 },
  { file: "weed_cluster.jpg", label: "Weed cluster", lat: 41.9898, lon: -93.6195 },
  { file: "healthy.jpg", label: "Healthy field", lat: 41.9898, lon: -93.6185 },
];

async function imageSize(file) {
  try {
    const bitmap = await createImageBitmap(file);
    const size = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return size;
  } catch {
    return null; // the server checks again; TIFFs cannot be decoded by browsers
  }
}

export default function UploadPanel({ busy, onAnalyse, mapRef }) {
  const [file, setFile] = useState(null);
  const [bounds, setBounds] = useState("");
  const [name, setName] = useState("");
  const [problem, setProblem] = useState("");
  const [dragging, setDragging] = useState(false);

  const ext = file ? extension(file.name) : "";
  const needsBounds = NEEDS_BOUNDS.includes(ext);

  function pick(f) {
    setProblem("");
    if (!f) return;
    if (!ALLOWED.includes(extension(f.name))) {
      setProblem("Use a GeoTIFF (.tif), PNG or JPEG image.");
      return;
    }
    setFile(f);
    if (!name) setName(f.name.replace(/\.[^.]+$/, ""));
  }

  function useMapView() {
    const map = mapRef.current;
    if (map) setBounds(boundsFromMap(map.getBounds()));
  }

  async function check(f, b) {
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) return `The file is larger than ${MAX_UPLOAD_MB} MB.`;
    if (NEEDS_BOUNDS.includes(extension(f.name))) {
      try {
        parseBounds(b);
      } catch (e) {
        return `${e.message}. PNG and JPEG images carry no location, so tell us where the image is.`;
      }
    }
    const size = await imageSize(f);
    if (size && (size.width > MAX_PIXELS_SIDE || size.height > MAX_PIXELS_SIDE)) {
      return `The image is ${size.width} x ${size.height} px; the limit is ${MAX_PIXELS_SIDE} x ${MAX_PIXELS_SIDE}. Crop or split it first.`;
    }
    return "";
  }

  async function submit(f, b, n) {
    const issue = await check(f, b);
    setProblem(issue);
    if (issue) return;
    const imageUrl = NEEDS_BOUNDS.includes(extension(f.name)) ? URL.createObjectURL(f) : null;
    onAnalyse({ file: f, bounds: NEEDS_BOUNDS.includes(extension(f.name)) ? b : null, name: n, imageUrl });
  }

  async function runSample(s) {
    setProblem("");
    const res = await fetch(`/samples/${s.file}`);
    const blob = await res.blob();
    const f = new File([blob], s.file, { type: "image/jpeg" });
    const b = boxAround(s.lat, s.lon, SAMPLE_SIZE_M);
    setFile(f);
    setBounds(b);
    setName(`Sample: ${s.label}`);
    submit(f, b, `Sample: ${s.label}`);
  }

  return (
    <section className="panel">
      <h2>Analyse an image</h2>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (file) submit(file, bounds, name);
        }}
      >
        <label
          className={`drop ${dragging ? "drop--over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            pick(e.dataTransfer.files[0]);
          }}
        >
          <input
            type="file"
            accept={ALLOWED.join(",")}
            onChange={(e) => pick(e.target.files[0])}
            disabled={busy}
          />
          <strong>{file ? file.name : "Choose or drop an aerial image"}</strong>
          <span>
            {file
              ? `${(file.size / 1024 / 1024).toFixed(1)} MB`
              : `GeoTIFF, PNG or JPEG · 8-bit RGB · up to ${MAX_PIXELS_SIDE} px and ${MAX_UPLOAD_MB} MB`}
          </span>
        </label>

        {(needsBounds || !file) && (
          <div className="field">
            <label htmlFor="bounds">
              Image location <small>minLon, minLat, maxLon, maxLat (needed for PNG/JPEG)</small>
            </label>
            <div className="row">
              <input
                id="bounds"
                value={bounds}
                onChange={(e) => setBounds(e.target.value)}
                placeholder="-93.62,41.99,-93.61,42.00"
                spellCheck="false"
                disabled={busy}
              />
              <button type="button" className="secondary" onClick={useMapView} disabled={busy}>
                Use map view
              </button>
            </div>
          </div>
        )}
        {file && !needsBounds && (
          <p className="hint">GeoTIFFs are placed on the map by their own coordinates.</p>
        )}

        <div className="field">
          <label htmlFor="name">Name (optional)</label>
          <input id="name" value={name} onChange={(e) => setName(e.target.value)} disabled={busy} />
        </div>

        {problem && <p className="problem">{problem}</p>}
        <button type="submit" disabled={!file || busy}>
          {busy ? "Analysing…" : "Analyse"}
        </button>
      </form>

      <div className="samples">
        <span>Or try a sample field:</span>
        <div className="sample-grid">
          {SAMPLES.map((s) => (
            <button
              key={s.file}
              type="button"
              className="sample"
              onClick={() => runSample(s)}
              disabled={busy}
              title={`Analyse ${s.file}`}
            >
              <img src={`/samples/${s.file}`} alt="" loading="lazy" />
              <span>{s.label}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
