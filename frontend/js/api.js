/**
 * frontend/js/api.js
 * Single Responsibility: all HTTP + WebSocket communication with the backend.
 * All other JS modules call functions here — never fetch() directly.
 */

const API_BASE = "http://localhost:8000/api/v1";
const WS_URL   = "ws://localhost:8000/ws/play";

// ─── HTTP helpers ─────────────────────────────────────────

export async function uploadPGN(file, username = "anonymous") {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/games/upload?username=${encodeURIComponent(username)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export async function getProfile(username) {
  const res = await fetch(`${API_BASE}/games/profile/${encodeURIComponent(username)}`);
  if (!res.ok) throw new Error(`No profile found for "${username}"`);
  return res.json();
}

export async function getStyleSimilarity(usernameA, usernameB) {
  const url = `${API_BASE}/games/similarity?username_a=${encodeURIComponent(usernameA)}&username_b=${encodeURIComponent(usernameB)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Could not compute similarity.");
  return res.json();
}

// ─── WebSocket client ─────────────────────────────────────

export class EngineSocket {
  constructor({ onEvaluation, onInfo, onError, onConnect, onDisconnect }) {
    this._handlers = { onEvaluation, onInfo, onError, onConnect, onDisconnect };
    this._ws = null;
  }

  connect() {
    this._ws = new WebSocket(WS_URL);
    this._ws.onopen  = () => this._handlers.onConnect?.();
    this._ws.onclose = () => this._handlers.onDisconnect?.();
    this._ws.onerror = () => this._handlers.onError?.("WebSocket error");
    this._ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "evaluation") this._handlers.onEvaluation?.(msg);
      if (msg.type === "info")       this._handlers.onInfo?.(msg.message);
      if (msg.type === "error")      this._handlers.onError?.(msg.message);
    };
  }

  disconnect() { this._ws?.close(); }

  setStyle(history) {
    this._send({ type: "set_style", history });
  }

  evaluate(fen, topK = 5) {
    this._send({ type: "evaluate", fen, top_k: topK });
  }

  _send(obj) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(obj));
    }
  }
}
