export const API_BASE_URL = "http://localhost:8000";

async function post(path, body) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(`Can't reach the backend at ${API_BASE_URL}. Is it running?`);
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data.detail) detail = data.detail;
    } catch {
      /* response wasn't JSON */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const ingest = (video_id) => post("/ingest", { video_id });
export const chat = (video_id, question) => post("/chat", { video_id, question });