export async function api(path, options = {}) {
  const headers = options.headers || {};
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

export function clone(value) {
  return JSON.parse(JSON.stringify(value == null ? null : value));
}
