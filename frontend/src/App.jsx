import { useCallback, useEffect, useRef, useState } from "react";

import { createAnalysis, getAnalysis, getClasses, getPolygons, listAnalyses } from "./api.js";
import History from "./components/History.jsx";
import MapView from "./components/MapView.jsx";
import Results from "./components/Results.jsx";
import UploadPanel from "./components/UploadPanel.jsx";

// Used until GET /api/v1/classes answers (same frozen list as docs/api.md).
const DEFAULT_CLASSES = [
  { id: 0, name: "background", colour: "transparent" },
  { id: 1, name: "nutrient_deficiency", colour: "#e6194b" },
  { id: 2, name: "drydown", colour: "#f58231" },
  { id: 3, name: "weed_cluster", colour: "#911eb4" },
];

export default function App() {
  const [classes, setClasses] = useState(DEFAULT_CLASSES);
  const [analyses, setAnalyses] = useState([]);
  const [current, setCurrent] = useState(null); // { analysis, polygons, imageUrl }
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [apiDown, setApiDown] = useState(false);
  const mapRef = useRef(null);

  const refreshHistory = useCallback(() => {
    listAnalyses()
      .then((list) => {
        setAnalyses(list);
        setApiDown(false);
      })
      .catch(() => setApiDown(true));
  }, []);

  useEffect(() => {
    getClasses()
      .then(setClasses)
      .catch(() => {});
    refreshHistory();
  }, [refreshHistory]);

  // Show a new result and free the previous image's object URL (it is no longer drawn).
  function show(next) {
    setCurrent((prev) => {
      if (prev?.imageUrl && prev.imageUrl !== next.imageUrl) URL.revokeObjectURL(prev.imageUrl);
      return next;
    });
  }

  async function analyse({ file, bounds, name, imageUrl }) {
    setBusy(true);
    setError("");
    try {
      const analysis = await createAnalysis({ file, bounds, name });
      const polygons = await getPolygons(analysis.id);
      show({ analysis, polygons, imageUrl });
      refreshHistory();
    } catch (e) {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function openAnalysis(id) {
    setError("");
    try {
      const [analysis, polygons] = await Promise.all([getAnalysis(id), getPolygons(id)]);
      show({ analysis, polygons, imageUrl: null });
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="brand">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="7" fill="currentColor" />
            <path d="M16 25c0-7 2-12 8-15-1 7-4 11-8 15zm0 0c0-5-2-9-7-11 0 5 3 9 7 11z" fill="#fff" />
          </svg>
          <div>
            <h1>CropWatch</h1>
            <p>Find crop stress in aerial farm images</p>
          </div>
        </header>

        <details className="intro" open={!current}>
          <summary>How it works</summary>
          <ol>
            <li>
              <strong>Give it a photo</strong> of a field taken from above (drone or plane), or
              click one of the sample fields below.
            </li>
            <li>
              <strong>An AI model checks every pixel</strong> and decides whether it looks
              healthy or shows one of three problems: nutrient deficiency, drydown or weeds.
            </li>
            <li>
              <strong>The result appears on the map:</strong> your photo is placed where the field
              is, and coloured outlines mark the problem areas. Click an outline for details.
            </li>
          </ol>
        </details>

        {apiDown && (
          <div className="notice">
            The analysis backend is offline right now. You can still look around; uploads will
            work when it is running.
          </div>
        )}

        <UploadPanel busy={busy} onAnalyse={analyse} mapRef={mapRef} />
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}
        {current && <Results analysis={current.analysis} classes={classes} />}
        <History analyses={analyses} currentId={current?.analysis.id} onOpen={openAnalysis} />

        <footer className="credits">
          Model v3 (U-Net, ResNet34) trained on{" "}
          <a href="https://github.com/SHI-Labs/Agriculture-Vision" target="_blank" rel="noreferrer">
            Agriculture-Vision 2021
          </a>{" "}
          (non-commercial use). Results are estimates: see the model card for accuracy.
        </footer>
      </aside>
      <main className="map-wrap">
        <MapView current={current} classes={classes} mapRef={mapRef} />
        {!current && !busy && (
          <div className="map-hint">
            Nothing analysed yet. Click a <strong>sample field</strong> on the left to see an
            example result here.
          </div>
        )}
        {busy && (
          <div className="busy" role="status">
            <span className="spinner" aria-hidden="true" />
            Analysing image… large images can take up to a minute.
          </div>
        )}
      </main>
    </div>
  );
}
