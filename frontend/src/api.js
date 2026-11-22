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
export const getModelInfo = () => request("/api/v1/model-info");
export const listAnalyses = () => request("/api/v1/analyses?limit=50");
export const getAnalysis = (id) => request(`/api/v1/analyses/${id}`);
export const getPolygons = (id) => request(`/api/v1/analyses/${id}/polygons`);

// On AWS the file goes straight to S3 (a Lambda request may be at most 6 MB): ask the gateway
// for an upload slot, post the file there, then start the analysis with the upload key.
// Locally the gateway answers 501 and the file is sent to the gateway directly.
async function uploadToS3(file) {
  let res;
  try {
    res = await fetch(`${BASE}/api/v1/uploads`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, size: file.size }),
    });
  } catch {
    throw new Error(`Cannot reach the CropWatch API at ${BASE}. Is the backend running?`);
  }
  if (res.status === 501 || res.status === 404) return null;
  const slot = await res.json().catch(() => null);
  if (!res.ok) throw new Error(`${res.status}: ${slot?.detail ?? res.statusText}`);

  const form = new FormData();
  Object.entries(slot.fields).forEach(([k, v]) => form.append(k, v));
  form.append("file", file); // must be the last field
  const put = await fetch(slot.url, { method: "POST", body: form });
  if (!put.ok) throw new Error(`Upload failed (${put.status}). Try again.`);
  return slot.upload_key;
}

export async function createAnalysis({ file, bounds, name }) {
  const uploadKey = await uploadToS3(file);
  const form = new FormData();
  if (uploadKey) {
    form.append("upload_key", uploadKey);
    form.append("filename", file.name);
  } else {
    form.append("file", file);
  }
  if (bounds) form.append("bounds", bounds);
  if (name) form.append("name", name);
  return request("/api/v1/analyses", { method: "POST", body: form });
}
