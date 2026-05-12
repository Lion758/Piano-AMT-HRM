const API_PORT = "8000";

function trimTrailingSlash(value) {
  return value.replace(/\/+$/, "");
}

export function getApiBase() {
  const envBase = import.meta.env.VITE_API_BASE?.trim();
  if (envBase) {
    return trimTrailingSlash(envBase);
  }

  if (typeof window !== "undefined" && window.location?.hostname) {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    return `${protocol}//${window.location.hostname}:${API_PORT}`;
  }

  return `http://localhost:${API_PORT}`;
}

export const API_BASE = getApiBase();

export function resolveApiUrl(pathOrUrl) {
  if (!pathOrUrl) {
    return null;
  }

  return new URL(String(pathOrUrl), `${API_BASE}/`).toString();
}
