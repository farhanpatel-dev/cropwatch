// Calls to the gateway (docs/api.md, "Gateway (public)"). Base URL from VITE_API_URL.
const BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error(`Cannot reach the CropWatch API at ${BASE}. Is the backend running?`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body && body.detail;
    const text = typeof detail === "string" ? detail : JSON.stringify(detail ?? res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return body;
}

export const apiBase = BASE;
export const getClasses = () => request("/api/v1/classes");
export const listAnalyses = () => request("/api/v1/analyses?limit=50");
export const getAnalysis = (id) => request(`/api/v1/analyses/${id}`);
export const getPolygons = (id) => request(`/api/v1/analyses/${id}/polygons`);

export function createAnalysis({ file, bounds, name }) {
  const form = new FormData();
  form.append("file", file);
  if (bounds) form.append("bounds", bounds);
  if (name) form.append("name", name);
  return request("/api/v1/analyses", { method: "POST", body: form });
}
