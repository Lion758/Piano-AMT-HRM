const FALLBACK_API_BASE = "http://134.208.3.192:8000";
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

function trimTrailingSlash(value) {
  return value.replace(/\/+$/, "");
}

export function getApiBase() {
  const envBase = import.meta.env.VITE_API_BASE?.trim();
  if (envBase) {
    return trimTrailingSlash(envBase);
  }

  if (
    typeof window !== "undefined" &&
    window.location?.hostname &&
    LOCAL_HOSTS.has(window.location.hostname)
  ) {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    return `${protocol}//${window.location.hostname}:8000`;
  }

  return FALLBACK_API_BASE;
}

export const API_BASE = getApiBase();

export function resolveApiUrl(pathOrUrl) {
  if (!pathOrUrl) {
    return null;
  }

  return new URL(String(pathOrUrl), `${API_BASE}/`).toString();
}
