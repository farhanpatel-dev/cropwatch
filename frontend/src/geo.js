// Small helpers with no React or network code (tested in geo.test.js).

// docs/api.md / ADR 0003: inference rejects images wider or taller than this.
export const MAX_PIXELS_SIDE = 4000;
export const MAX_UPLOAD_MB = 50;
export const NEEDS_BOUNDS = [".png", ".jpg", ".jpeg"];
export const ALLOWED = [".tif", ".tiff", ...NEEDS_BOUNDS];

export function extension(filename) {
  const i = filename.lastIndexOf(".");
  return i < 0 ? "" : filename.slice(i).toLowerCase();
}

/** "minLon,minLat,maxLon,maxLat" -> [4 numbers]; throws Error with a readable message. */
export function parseBounds(text) {
  const parts = String(text).split(",").map((v) => v.trim());
  const values = parts.map(Number);
  if (parts.length !== 4 || parts.some((p) => p === "") || values.some((v) => !Number.isFinite(v))) {
    throw new Error("Bounds must be four numbers: minLon,minLat,maxLon,maxLat");
  }
  const [minLon, minLat, maxLon, maxLat] = values;
  if (!(minLon < maxLon && minLat < maxLat)) throw new Error("Bounds need min < max");
  if (minLon < -180 || maxLon > 180 || minLat < -90 || maxLat > 90) {
    throw new Error("Longitude must be -180..180 and latitude -90..90");
  }
  return values;
}

/** Leaflet LatLngBounds-like object -> "minLon,minLat,maxLon,maxLat" (6 decimals). */
export function boundsFromMap(b) {
  return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(6)).join(",");
}

/** GeoJSON Polygon (closed ring of [lon, lat]) -> [[south, west], [north, east]] for Leaflet. */
export function polygonToLatLngBounds(polygon) {
  const ring = polygon?.coordinates?.[0];
  if (!ring?.length) return null;
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  return [
    [Math.min(...lats), Math.min(...lons)],
    [Math.max(...lats), Math.max(...lons)],
  ];
}

/** A square box of `sizeM` metres centred on lat/lon, as "minLon,minLat,maxLon,maxLat". */
export function boxAround(lat, lon, sizeM) {
  const dLat = sizeM / 2 / 111320;
  const dLon = sizeM / 2 / (111320 * Math.cos((lat * Math.PI) / 180));
  return [lon - dLon, lat - dLat, lon + dLon, lat + dLat].map((v) => v.toFixed(6)).join(",");
}

export function formatArea(ha) {
  if (ha >= 1) return `${ha.toFixed(2)} ha`;
  return `${Math.round(ha * 10000).toLocaleString()} m²`;
}
