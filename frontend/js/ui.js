/**
 * frontend/js/ui.js
 * Single Responsibility: UI utilities — toasts, loading states, DOM helpers.
 * No API or chess logic here.
 */

// ─── Toast notifications ──────────────────────────────────

function _ensureToastContainer() {
  let el = document.getElementById("toast-container");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast-container";
    document.body.appendChild(el);
  }
  return el;
}

export function toast(message, type = "info", durationMs = 3500) {
  const container = _ensureToastContainer();
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), durationMs);
}

// ─── Loading button state ─────────────────────────────────

export function setLoading(btn, loading, loadingText = "Loading…") {
  if (loading) {
    btn.dataset.originalText = btn.textContent;
    btn.textContent = loadingText;
    btn.disabled = true;
  } else {
    btn.textContent = btn.dataset.originalText || btn.textContent;
    btn.disabled = false;
  }
}

// ─── Show / hide ──────────────────────────────────────────

export function show(selector) {
  const el = typeof selector === "string" ? document.querySelector(selector) : selector;
  if (el) el.style.display = "";
}

export function hide(selector) {
  const el = typeof selector === "string" ? document.querySelector(selector) : selector;
  if (el) el.style.display = "none";
}

// ─── Blunder risk badge ───────────────────────────────────

export function blunderBadge(risk) {
  if (risk === null || risk === undefined) return "";
  const pct = (risk * 100).toFixed(1);
  if (risk > 0.5) return `<span class="badge badge-red">⚠ Blunder Risk ${pct}%</span>`;
  if (risk > 0.25) return `<span class="badge badge-amber">⚡ Risk ${pct}%</span>`;
  return `<span class="badge badge-green">✓ Safe ${pct}%</span>`;
}
