import { useEffect, useMemo, useState } from "react";
import {
  GeoJSON,
  ImageOverlay,
  LayersControl,
  MapContainer,
  Rectangle,
  TileLayer,
  useMap,
} from "react-leaflet";

import { formatArea, polygonToLatLngBounds } from "../geo.js";

const START = { center: [41.99, -93.619], zoom: 17 };

function FitTo({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [40, 40], maxZoom: 21 });
  }, [map, bounds]);
  return null;
}

function ExposeMap({ mapRef }) {
  const map = useMap();
  useEffect(() => {
    mapRef.current = map;
  }, [map, mapRef]);
  return null;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

export default function MapView({ current, classes, mapRef }) {
  const [imageOpacity, setImageOpacity] = useState(0.9);
  const colours = useMemo(() => Object.fromEntries(classes.map((c) => [c.id, c.colour])), [classes]);
  const imageBounds = useMemo(
    () => polygonToLatLngBounds(current?.analysis.image_bounds),
    [current?.analysis.image_bounds],
  );

  const style = (feature) => {
    const colour = colours[feature.properties.class_id] || "#333";
    return { color: colour, weight: 2, fillColor: colour, fillOpacity: 0.35 };
  };

  const onEachFeature = (feature, layer) => {
    const p = feature.properties;
    layer.bindPopup(
      `<strong>${escapeHtml(p.class_name.replace(/_/g, " "))}</strong><br/>` +
        `Area: ${formatArea((p.area_m2 || 0) / 10000)}<br/>` +
        `Confidence: ${Math.round((p.confidence || 0) * 100)}%`,
    );
  };

  return (
    <>
      <MapContainer {...START} maxZoom={22} className="map" zoomControl>
        <ExposeMap mapRef={mapRef} />
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Satellite">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution="Tiles &copy; Esri, Maxar, Earthstar Geographics"
              maxNativeZoom={19}
              maxZoom={22}
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Streets">
            <TileLayer
              url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              maxNativeZoom={19}
              maxZoom={22}
            />
          </LayersControl.BaseLayer>
        </LayersControl>

        {current?.imageUrl && imageBounds && (
          <ImageOverlay url={current.imageUrl} bounds={imageBounds} opacity={imageOpacity} />
        )}
        {imageBounds && (
          <Rectangle
            bounds={imageBounds}
            pathOptions={{ color: "#fff", weight: 1.5, dashArray: "6 6", fill: false }}
          />
        )}
        {current && (
          <GeoJSON
            key={current.analysis.id}
            data={current.polygons}
            style={style}
            onEachFeature={onEachFeature}
          />
        )}
        <FitTo bounds={imageBounds} />
      </MapContainer>

      <div className="legend" aria-label="Legend">
        {classes
          .filter((c) => c.id !== 0)
          .map((c) => (
            <span key={c.id}>
              <i style={{ background: c.colour }} />
              {c.name.replace(/_/g, " ")}
            </span>
          ))}
        {current?.imageUrl && (
          <label className="opacity">
            Image
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={imageOpacity}
              onChange={(e) => setImageOpacity(Number(e.target.value))}
            />
          </label>
        )}
      </div>
    </>
  );
}
