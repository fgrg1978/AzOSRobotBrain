/*
 * Robot Fleet Dashboard — vanilla JS (no deps).
 *
 * Polls /fleet/robots every POLL_INTERVAL_MS. Degrades gracefully if
 * the endpoint is missing (shows single-robot fallback via /status).
 */

"use strict";

// ── Constants (NO MAGIC NUMBERS) ────────────────────────────────────────────
const POLL_INTERVAL_MS       = 2000;
const REQUEST_TIMEOUT_MS     = 4000;
const MAX_RECENT_EVENTS      = 20;
const MAP_ROBOT_RADIUS_PX    = 8;
const MAP_PADDING_PX         = 24;
const MAP_DEFAULT_RANGE_MM   = 10000;   // +/- 5m default bounds if no data
const MAP_GRID_STEP_MM       = 1000;
const STATUS_ONLINE          = "online";
const STATUS_OFFLINE         = "offline";
const STATUS_BUSY            = "busy";
const BATTERY_LOW_MV         = 6500;
const BATTERY_CRIT_MV        = 5800;
const SECONDS_PER_MIN        = 60;
const SECONDS_PER_HOUR       = 3600;
const API_BASE               = "";      // same-origin
const HTTP_UNAUTHORIZED      = 401;
const TOKEN_STORAGE_KEY      = "robotBrainApiKey";

// ── State ───────────────────────────────────────────────────────────────────
const state = {
  robots: [],              // latest list from /fleet/robots
  events: {},              // robot_id -> array of events
  selectedId: null,
  apiOk: false,
  fleetEndpointAvailable: true,
  lastUpdateTime: 0,
};

// ── DOM refs ────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const dom = {
  tbody:            $("fleet-tbody"),
  connStatus:       $("connection-status"),
  lastUpdate:       $("last-update"),
  robotCount:       $("robot-count"),
  canvas:           $("map-canvas"),
  detailEmpty:      $("detail-empty"),
  detailContent:    $("detail-content"),
  detailId:         $("detail-id"),
  detailFields:     $("detail-fields"),
  detailEvents:     $("detail-events"),
  detailClose:      $("detail-close"),
  modeSelect:       $("mode-select"),
  cmdMode:          $("cmd-mode"),
  cmdEstop:         $("cmd-estop"),
  cmdFeedback:      $("cmd-feedback"),
};

// ── API token ───────────────────────────────────────────────────────────────
// The API tier requires "Authorization: Bearer <AZOS_BRAIN_API_KEY>" on every
// non-public route (api.py APIServer._is_authorised). /dashboard/* itself is
// public so this page loads, but every call it makes — /fleet/robots, /status,
// /mode, /stop, /fleet/command — is not. This page used to send no header at
// all, so switching the API to authenticated mode made the whole dashboard go
// dark, which is a strong incentive to switch it back off. Ask for the token
// once, keep it in localStorage, re-ask on 401.
//
// The token is deliberately NOT taken from (or put in) the URL: query strings
// end up in browser history, referrers and access logs.
function getApiToken() {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY) || "";
  } catch (e) {
    return ""; // storage blocked (private mode) — degrade to insecure-mode use
  }
}

function setApiToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_STORAGE_KEY, token);
    else localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch (e) { /* storage blocked — token lives for this page load only */ }
}

// Prompt at most once per page load so a 401 storm from the 2s poll loop
// cannot produce a prompt storm.
let tokenPromptShown = false;
function promptForApiToken() {
  if (tokenPromptShown) return "";
  tokenPromptShown = true;
  const entered = window.prompt(
    "This brain requires an API token (AZOS_BRAIN_API_KEY). Paste it to continue:"
  );
  setApiToken(entered || "");
  return entered || "";
}

// ── HTTP helper with timeout ────────────────────────────────────────────────
async function fetchJson(url, opts = {}, allowRetry = true) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
  try {
    const token = getApiToken();
    // MERGE the caller's headers — the command POSTs pass Content-Type, and
    // replacing their headers object would break exactly the two endpoints
    // that move the robot.
    const headers = { ...(opts.headers || {}) };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const resp = await fetch(API_BASE + url, { ...opts, headers, signal: ctrl.signal });
    if (!resp.ok) {
      if (resp.status === HTTP_UNAUTHORIZED && allowRetry) {
        // Either no token yet, or a stale one. Ask, then replay once.
        const fresh = promptForApiToken();
        if (fresh) {
          clearTimeout(timer);
          return fetchJson(url, opts, false);
        }
      }
      return { ok: false, status: resp.status };
    }
    const data = await resp.json();
    return { ok: true, status: resp.status, data };
  } catch (e) {
    return { ok: false, status: 0, error: String(e) };
  } finally {
    clearTimeout(timer);
  }
}

// ── Formatters ──────────────────────────────────────────────────────────────
function fmtBattery(mv) {
  if (!mv) return "—";
  const v = (mv / 1000).toFixed(2);
  let cls = "";
  if (mv < BATTERY_CRIT_MV) cls = "status-offline";
  else if (mv < BATTERY_LOW_MV) cls = "status-busy";
  else cls = "status-online";
  return `<span class="${cls}">${v} V</span>`;
}

function fmtAge(seconds) {
  if (seconds == null || seconds < 0) return "never";
  if (seconds < SECONDS_PER_MIN) return `${Math.round(seconds)}s ago`;
  if (seconds < SECONDS_PER_HOUR) return `${Math.round(seconds / SECONDS_PER_MIN)}m ago`;
  return `${(seconds / SECONDS_PER_HOUR).toFixed(1)}h ago`;
}

function fmtStatus(robot) {
  if (!robot.online) return `<span class="status-offline">offline</span>`;
  if (robot.busy)    return `<span class="status-busy">busy</span>`;
  if (robot.docked)  return `<span class="status-online">docked</span>`;
  return `<span class="status-online">online</span>`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// ── Table rendering ─────────────────────────────────────────────────────────
function renderTable() {
  dom.robotCount.textContent = `${state.robots.length} robots`;

  if (state.robots.length === 0) {
    dom.tbody.innerHTML = `<tr><td colspan="8" class="empty-row">No robots registered</td></tr>`;
    return;
  }

  const rows = state.robots.map((r) => {
    const sel = (r.id === state.selectedId) ? " selected" : "";
    const posTxt = (r.x_mm != null && r.y_mm != null)
      ? `${Math.round(r.x_mm)}, ${Math.round(r.y_mm)}` : "—";
    return `
      <tr class="row${sel}" data-id="${escapeHtml(r.id)}">
        <td>${escapeHtml(r.id)}</td>
        <td>${escapeHtml(r.type || "wheeled")}</td>
        <td>${escapeHtml(r.name || r.id)}</td>
        <td>${fmtStatus(r)}</td>
        <td>${fmtBattery(r.battery_mv)}</td>
        <td>${fmtAge(r.last_seen_age_s)}</td>
        <td>${posTxt}</td>
        <td>${escapeHtml(r.mode || "—")}</td>
      </tr>`;
  }).join("");

  dom.tbody.innerHTML = rows;
  for (const tr of dom.tbody.querySelectorAll("tr.row")) {
    tr.addEventListener("click", () => selectRobot(tr.dataset.id));
  }
}

// ── Map rendering ───────────────────────────────────────────────────────────
function renderMap() {
  const canvas = dom.canvas;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  // Background
  ctx.fillStyle = "#15151a";
  ctx.fillRect(0, 0, W, H);

  // Compute bounds from robots
  const robots = state.robots.filter(r => r.x_mm != null && r.y_mm != null);
  let minX, maxX, minY, maxY;
  if (robots.length === 0) {
    minX = -MAP_DEFAULT_RANGE_MM / 2; maxX = MAP_DEFAULT_RANGE_MM / 2;
    minY = -MAP_DEFAULT_RANGE_MM / 2; maxY = MAP_DEFAULT_RANGE_MM / 2;
  } else {
    minX = Math.min(...robots.map(r => r.x_mm)) - MAP_GRID_STEP_MM;
    maxX = Math.max(...robots.map(r => r.x_mm)) + MAP_GRID_STEP_MM;
    minY = Math.min(...robots.map(r => r.y_mm)) - MAP_GRID_STEP_MM;
    maxY = Math.max(...robots.map(r => r.y_mm)) + MAP_GRID_STEP_MM;
    if (maxX - minX < MAP_DEFAULT_RANGE_MM) {
      const midX = (maxX + minX) / 2;
      minX = midX - MAP_DEFAULT_RANGE_MM / 2;
      maxX = midX + MAP_DEFAULT_RANGE_MM / 2;
    }
    if (maxY - minY < MAP_DEFAULT_RANGE_MM) {
      const midY = (maxY + minY) / 2;
      minY = midY - MAP_DEFAULT_RANGE_MM / 2;
      maxY = midY + MAP_DEFAULT_RANGE_MM / 2;
    }
  }

  const spanX = maxX - minX, spanY = maxY - minY;
  const drawW = W - 2 * MAP_PADDING_PX, drawH = H - 2 * MAP_PADDING_PX;

  function worldToCanvas(xMm, yMm) {
    const px = MAP_PADDING_PX + (xMm - minX) / spanX * drawW;
    // Y flipped: +y up in world, down in canvas
    const py = MAP_PADDING_PX + (1 - (yMm - minY) / spanY) * drawH;
    return [px, py];
  }

  // Grid lines
  ctx.strokeStyle = "#2a2a30";
  ctx.lineWidth = 1;
  for (let x = Math.ceil(minX / MAP_GRID_STEP_MM) * MAP_GRID_STEP_MM; x <= maxX; x += MAP_GRID_STEP_MM) {
    const [px] = worldToCanvas(x, 0);
    ctx.beginPath(); ctx.moveTo(px, MAP_PADDING_PX); ctx.lineTo(px, H - MAP_PADDING_PX); ctx.stroke();
  }
  for (let y = Math.ceil(minY / MAP_GRID_STEP_MM) * MAP_GRID_STEP_MM; y <= maxY; y += MAP_GRID_STEP_MM) {
    const [, py] = worldToCanvas(0, y);
    ctx.beginPath(); ctx.moveTo(MAP_PADDING_PX, py); ctx.lineTo(W - MAP_PADDING_PX, py); ctx.stroke();
  }

  // Origin marker
  const [ox, oy] = worldToCanvas(0, 0);
  ctx.strokeStyle = "#3a3a44";
  ctx.beginPath(); ctx.moveTo(ox - 6, oy); ctx.lineTo(ox + 6, oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox, oy - 6); ctx.lineTo(ox, oy + 6); ctx.stroke();

  // Waypoints (if provided per-robot)
  for (const r of state.robots) {
    if (!Array.isArray(r.waypoints) || r.waypoints.length === 0) continue;
    ctx.strokeStyle = "#4da3ff88";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let first = true;
    for (const wp of r.waypoints) {
      if (wp.x_mm == null || wp.y_mm == null) continue;
      const [px, py] = worldToCanvas(wp.x_mm, wp.y_mm);
      if (first) { ctx.moveTo(px, py); first = false; } else { ctx.lineTo(px, py); }
    }
    ctx.stroke();
    for (const wp of r.waypoints) {
      if (wp.x_mm == null || wp.y_mm == null) continue;
      const [px, py] = worldToCanvas(wp.x_mm, wp.y_mm);
      ctx.fillStyle = "#4da3ff";
      ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  // Robots
  for (const r of robots) {
    const [px, py] = worldToCanvas(r.x_mm, r.y_mm);
    let color = "#ef4444";  // offline
    if (r.online) color = "#4ade80";
    if (r.id === state.selectedId) color = "#a78bfa";

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, MAP_ROBOT_RADIUS_PX, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#0008";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = "#e8e8ee";
    ctx.font = "11px -apple-system, sans-serif";
    ctx.fillText(r.id, px + MAP_ROBOT_RADIUS_PX + 4, py + 4);
  }
}

// ── Detail panel ────────────────────────────────────────────────────────────
function selectRobot(id) {
  state.selectedId = id;
  renderTable();
  renderMap();
  renderDetail();
}

function closeDetail() {
  state.selectedId = null;
  renderTable();
  renderMap();
  renderDetail();
}

function renderDetail() {
  if (!state.selectedId) {
    dom.detailContent.classList.add("hidden");
    dom.detailEmpty.classList.remove("hidden");
    return;
  }
  const r = state.robots.find(x => x.id === state.selectedId);
  if (!r) {
    dom.detailContent.classList.add("hidden");
    dom.detailEmpty.classList.remove("hidden");
    dom.detailEmpty.textContent = `Robot '${state.selectedId}' not found.`;
    return;
  }

  dom.detailContent.classList.remove("hidden");
  dom.detailEmpty.classList.add("hidden");
  dom.detailId.textContent = r.id;

  const fields = [
    ["Type",        r.type || "wheeled"],
    ["Name",        r.name || r.id],
    ["Status",      r.online ? "online" : "offline"],
    ["Battery",     r.battery_mv ? `${(r.battery_mv / 1000).toFixed(2)} V` : "—"],
    ["Position",    (r.x_mm != null) ? `${Math.round(r.x_mm)}, ${Math.round(r.y_mm)} mm` : "—"],
    ["Mode",        r.mode || "—"],
    ["Zones",       (r.zones || []).join(", ") || "—"],
    ["Docked",      r.docked ? "yes" : "no"],
    ["Busy",        r.busy ? "yes" : "no"],
    ["Last seen",   fmtAge(r.last_seen_age_s)],
    ["Port",        r.port || "—"],
  ];
  dom.detailFields.innerHTML = fields
    .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`)
    .join("");

  // Events
  const events = state.events[r.id] || [];
  if (events.length === 0) {
    dom.detailEvents.innerHTML = `<li class="empty-row">No events</li>`;
  } else {
    dom.detailEvents.innerHTML = events.slice(-MAX_RECENT_EVENTS).reverse()
      .map(e => `<li><span class="status-busy">${escapeHtml(e.t)}</span> ${escapeHtml(e.msg)}</li>`)
      .join("");
  }
}

function pushEvent(robotId, msg) {
  if (!state.events[robotId]) state.events[robotId] = [];
  state.events[robotId].push({
    t: new Date().toLocaleTimeString(),
    msg,
  });
  while (state.events[robotId].length > MAX_RECENT_EVENTS) {
    state.events[robotId].shift();
  }
}

// ── Commands ────────────────────────────────────────────────────────────────
function setFeedback(msg, ok) {
  dom.cmdFeedback.textContent = msg;
  dom.cmdFeedback.className = "feedback " + (ok ? "ok" : "err");
}

// Mapping: mode name -> MODE packet payload byte. The backend's
// /mode endpoint (single-robot) accepts a name, so we always use that path.
// For fleet-scoped modes we POST /fleet/command with PKT_MODE=0x81 and a
// 1-byte payload mapping the mode to its id.
const PKT_MODE  = 0x81;
const PKT_ESTOP = 0x88;
const MODE_IDS = { idle: 0, patrulla: 1, guard: 2, dock: 3, offline: 4 };

function toHex(bytes) {
  return bytes.map(b => b.toString(16).padStart(2, "0")).join("");
}

async function sendModeCmd() {
  const id = state.selectedId;
  if (!id) return;
  const mode = dom.modeSelect.value;

  // Fleet path: POST /fleet/command with id/pkt_type/payload_hex.
  const modeId = MODE_IDS[mode];
  let result = { ok: false, status: 0 };
  if (modeId != null && state.fleetEndpointAvailable) {
    result = await fetchJson(`/fleet/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id, pkt_type: PKT_MODE, payload_hex: toHex([modeId]),
      }),
    });
  }
  // Fallback: single-robot /mode endpoint (takes a name).
  if (!result.ok) {
    result = await fetchJson(`/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
  }
  if (result.ok) {
    setFeedback(`Mode set to '${mode}'`, true);
    pushEvent(id, `mode -> ${mode}`);
  } else {
    setFeedback(`Mode command failed (${result.status || "network"})`, false);
  }
  renderDetail();
}

async function sendEstop() {
  const id = state.selectedId;
  if (!id) return;
  if (!confirm(`Send EMERGENCY STOP to '${id}'?`)) return;

  let result = { ok: false, status: 0 };
  if (state.fleetEndpointAvailable) {
    result = await fetchJson(`/fleet/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id, pkt_type: PKT_ESTOP, payload_hex: "",
      }),
    });
  }
  if (!result.ok) {
    result = await fetchJson(`/stop`, { method: "POST" });
  }
  if (result.ok) {
    setFeedback("ESTOP sent", true);
    pushEvent(id, "ESTOP sent");
  } else {
    setFeedback(`ESTOP failed (${result.status || "network"})`, false);
  }
  renderDetail();
}

// ── Normalize fleet payload ────────────────────────────────────────────────
// The real /fleet/robots endpoint returns:
//   { total, online, timeout_s, robots: { "id1": {online, type, name,
//                                                 last_seen, battery_mv,
//                                                 location: [x, y]}, ... } }
// We normalize it to a flat array with stable keys used by the UI.
function normalizeFleet(payload) {
  const ROBOT_TYPE_NAMES = ["wheeled", "drone", "humanoid", "ackermann"];
  if (Array.isArray(payload)) return payload;
  if (!payload || payload.error) return [];
  const bag = payload.robots || {};
  const now_s = Date.now() / 1000;
  // `robots` can be a dict {id: data} or an array of objects.
  const entries = Array.isArray(bag)
    ? bag.map(r => [r.id || r.robot_id, r])
    : Object.entries(bag);
  return entries.map(([rid, r]) => {
    const loc = Array.isArray(r.location) ? r.location : [0, 0];
    const lastSeen = r.last_seen || 0;
    const ageS = lastSeen > 0 ? Math.max(0, now_s - lastSeen) : null;
    const typeIdx = typeof r.type === "number" ? r.type : -1;
    const typeName = (typeIdx >= 0 && typeIdx < ROBOT_TYPE_NAMES.length)
      ? ROBOT_TYPE_NAMES[typeIdx]
      : (r.type || "unknown");
    return {
      id:               rid,
      name:             r.name || rid,
      type:             typeName,
      online:           !!r.online,
      busy:             !!r.busy,
      docked:           !!r.docked,
      battery_mv:       r.battery_mv || 0,
      x_mm:             loc[0],
      y_mm:             loc[1],
      last_seen_age_s:  ageS,
      zones:            r.zones || [],
      mode:             r.mode || r.meta?.mode || "",
      port:             r.port || null,
      waypoints:        r.waypoints || [],
    };
  });
}

// ── Poll loop ───────────────────────────────────────────────────────────────
async function pollFleet() {
  // Try /fleet/robots first
  if (state.fleetEndpointAvailable) {
    const res = await fetchJson("/fleet/robots");
    if (res.ok) {
      // Endpoint returns {error} if fleet manager disabled.
      if (res.data && res.data.error) {
        state.fleetEndpointAvailable = false;
      } else {
        state.robots = normalizeFleet(res.data);
        state.apiOk = true;
        state.lastUpdateTime = Date.now();
        updateConnectionBadge();
        renderAll();
        return;
      }
    } else if (res.status === 404) {
      state.fleetEndpointAvailable = false; // fall through to degraded mode
    }
  }

  // Degraded mode: query /status for single robot
  const res = await fetchJson("/status");
  if (res.ok) {
    const s = res.data;
    state.robots = [{
      id: "local",
      name: "local",
      type: s.robot_type != null ? `type-${s.robot_type}` : "wheeled",
      online: !!s.connected,
      busy: false,
      docked: false,
      battery_mv: (s.sensors && s.sensors.battery_mv) || 0,
      x_mm: 0, y_mm: 0,
      mode: s.mode,
      last_seen_age_s: s.last_sensor_age_s,
      zones: [],
      port: null,
    }];
    state.apiOk = true;
    state.lastUpdateTime = Date.now();
    updateConnectionBadge("degraded");
    renderAll();
  } else {
    state.apiOk = false;
    updateConnectionBadge();
  }
}

function updateConnectionBadge(forced) {
  const badge = dom.connStatus;
  if (forced === "degraded") {
    badge.textContent = "Degraded (single robot)";
    badge.className = "status-pill degraded";
  } else if (state.apiOk) {
    badge.textContent = "Connected";
    badge.className = "status-pill online";
  } else {
    badge.textContent = "Disconnected";
    badge.className = "status-pill offline";
  }
  dom.lastUpdate.textContent = state.lastUpdateTime
    ? `Last update: ${new Date(state.lastUpdateTime).toLocaleTimeString()}`
    : "Last update: never";
}

function renderAll() {
  renderTable();
  renderMap();
  renderDetail();
}

// ── Init ────────────────────────────────────────────────────────────────────
function init() {
  dom.detailClose.addEventListener("click", closeDetail);
  dom.cmdMode.addEventListener("click", sendModeCmd);
  dom.cmdEstop.addEventListener("click", sendEstop);
  pollFleet();
  setInterval(pollFleet, POLL_INTERVAL_MS);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
