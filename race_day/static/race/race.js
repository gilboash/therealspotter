// Race view — a heat played back as a race, in floating windows over one shared playhead: the heat's video (all
// pilots, or one pilot's quadrant), a timeline lane per pilot, the standings, the race map and the race in 3D.
// From gilboash/therealspotter's web app (web/static/app.js, Race mode). Opened in the dashboard's Race view tab
// (an iframe: ?e=<event>&heat=<n>&pilot=<k>); read-only, from /api/race/… (race_day/raceview.py).
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const store = {
  get(k, def) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : def; } catch { return def; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const QS = new URLSearchParams(location.search);
const EV = QS.get("e") || "";

async function api(p) {
  const r = await fetch(p + (p.includes("?") ? "&" : "?") + "e=" + encodeURIComponent(EV), { cache: "no-store" });
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).error || m; } catch { /* not json */ }
    throw new Error(m);
  }
  return r.json();
}

function status(msg, isErr = false) {
  const el = $("#status");
  el.textContent = msg;
  el.style.color = isErr ? "var(--bad)" : "";
}

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────

const S = {
  heats: [], H: null,                  // the heats to pick from; the heat open (/api/race/heat/<n>)
  pilot: null,                         // the pilot open (its index), or null: every pilot (a combo heat)
  run: null,                           // the open pilot's run (passes, laps, stats)
  combo: null,                         // every pilot of the heat, raced against each other at the playhead
  solo: null,                          // the open pilot as a one-pilot race (the race map, call-outs)
  crop: null,                          // the part of the video shown: the open pilot's quadrant ([x, y, w, h], fractions)
  raceSet: { sound: true, voice: true, ...store.get("raceview.sound", {}) },   // sounds while playing
  t: 0, duration: 0,
  ov: store.get("raceview.overlay", { hud: true }),
  videoEl: null,
};

// ─────────────────────────────────────────────────────────────
// Window manager
// ─────────────────────────────────────────────────────────────

const PANELS = {
  video:     { title: "Video", init: initVideo },
  timeline:  { title: "Timeline", init: initTimeline },
  racestand: { title: "Race — standings", render: renderRaceStand, live: true },
  stats:     { title: "Race stats — laps", render: renderStats, live: true },
  splits:    { title: "Gate splits — time between gates", render: renderSplits, live: true },
  livemap:   { title: "Race map", init: initLiveMap, live: true },
  race3d:    { title: "Race 3D", init: initRace3D },
  gates:     { title: "Gates (learned)", render: renderGates },
};

// default windows, as fractions of the desk: [x, y, w, h]
const DEFAULTS = {
  pilot: {         // one pilot: its laps, splits, and the 3D view chasing it
    video: [0, 0, 0.56, 0.7], timeline: [0, 0.7, 0.56, 0.3],
    stats: [0.56, 0, 0.44, 0.36], splits: [0.56, 0.36, 0.44, 0.3], race3d: [0.56, 0.66, 0.44, 0.34],
  },
  combo: {         // every pilot: the standings and the 3D overview
    video: [0, 0, 0.56, 0.7], timeline: [0, 0.7, 0.56, 0.3], racestand: [0.56, 0, 0.44, 0.48], race3d: [0.56, 0.48, 0.44, 0.52],
  },
};

const wins = {};          // id → {el, body, panel}
let zTop = 10;

// two layouts: every pilot's and one pilot's (switched as you open one or the other)
const layoutKind = () => (S.combo ? "combo" : "pilot");
const layoutKey = () => `raceview.layout.${layoutKind()}`;

function deskRect() { return $("#desk").getBoundingClientRect(); }

function saveLayout() {
  const L = {};
  for (const [id, w] of Object.entries(wins)) {
    const r = w.el;
    L[id] = { x: r.offsetLeft, y: r.offsetTop, w: r.offsetWidth, h: r.offsetHeight, z: +r.style.zIndex || 1 };
  }
  store.set(layoutKey(), L);
}

function applyLayout() {
  S.layoutKind = layoutKind();
  for (const id of Object.keys(wins)) closeWin(id, false);
  const saved = store.get(layoutKey(), null);
  const d = deskRect();
  if (saved) {
    for (const [id, g] of Object.entries(saved).sort((a, b) => a[1].z - b[1].z)) if (PANELS[id]) openWin(id, g);
  } else {
    for (const [id, f] of Object.entries(DEFAULTS[S.layoutKind])) {
      openWin(id, { x: f[0] * d.width + 4, y: f[1] * d.height + 4, w: f[2] * d.width - 8, h: f[3] * d.height - 8 });
    }
  }
  buildWinMenu();
}

function openWin(id, geo) {
  if (wins[id]) { focusWin(id); return; }
  const P = PANELS[id];
  const d = deskRect();
  geo = geo || { x: 40 + Math.random() * 80, y: 40 + Math.random() * 60, w: Math.min(520, d.width * 0.4), h: Math.min(360, d.height * 0.45) };
  const el = document.createElement("div");
  el.className = "win";
  el.dataset.win = id;                 // the phone layout (race.css) orders and sizes windows by it
  el.innerHTML = `<div class="title"><span>${esc(P.title)}</span><span class="x" title="Close">✕</span></div>
                  <div class="body"></div><div class="grip"></div>`;
  Object.assign(el.style, { left: geo.x + "px", top: geo.y + "px", width: geo.w + "px", height: geo.h + "px" });
  $("#desk").appendChild(el);
  const w = { el, body: $(".body", el), panel: P };
  wins[id] = w;
  focusWin(id);
  $(".x", el).onclick = () => closeWin(id);
  el.addEventListener("mousedown", () => focusWin(id));
  drag($(".title", el), (dx, dy, g) => { el.style.left = g.x + dx + "px"; el.style.top = Math.max(0, g.y + dy) + "px"; }, el);
  drag($(".grip", el), (dx, dy, g) => {
    el.style.width = Math.max(160, g.w + dx) + "px";
    el.style.height = Math.max(80, g.h + dy) + "px";
  }, el);
  if (P.init) P.init(w);
  else P.render(w);
  if (geo.z) el.style.zIndex = geo.z;
  buildWinMenu();
}

function closeWin(id, save = true) {
  const w = wins[id];
  if (!w) return;
  if (w.cleanup) w.cleanup();
  w.el.remove();
  delete wins[id];
  if (save) { saveLayout(); buildWinMenu(); }
}

function focusWin(id) {
  for (const w of Object.values(wins)) w.el.classList.remove("focus");
  wins[id].el.classList.add("focus");
  wins[id].el.style.zIndex = ++zTop;
}

function drag(handle, move, el) {
  handle.addEventListener("mousedown", (e) => {
    if (e.target.classList.contains("x")) return;
    e.preventDefault();
    const g = { x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight };
    const sx = e.clientX, sy = e.clientY;
    const mm = (ev) => move(ev.clientX - sx, ev.clientY - sy, g);
    const mu = () => { removeEventListener("mousemove", mm); removeEventListener("mouseup", mu); saveLayout(); };
    addEventListener("mousemove", mm);
    addEventListener("mouseup", mu);
  });
}

function buildWinMenu() {
  const m = $("#winMenu");
  m.innerHTML = Object.entries(PANELS).map(([id, p]) =>
    `<label><input type="checkbox" data-w="${id}" ${wins[id] ? "checked" : ""}> ${esc(p.title)}</label>`).join("")
    + `<hr><button id="resetLayout">Reset this layout</button>`;
  m.querySelectorAll("input[data-w]").forEach((cb) => cb.onchange = () => {
    if (cb.checked) { openWin(cb.dataset.w); saveLayout(); } else closeWin(cb.dataset.w);
  });
  $("#resetLayout").onclick = () => { localStorage.removeItem(layoutKey()); applyLayout(); };
}

function render(ids) {
  for (const [id, w] of Object.entries(wins)) {
    if ((!ids || ids.includes(id)) && w.panel.render) {
      try { w.panel.render(w); } catch (e) { console.error(id, e); }
    }
  }
  if (wins.timeline && (!ids || ids.includes("timeline"))) wins.timeline.draw();
  if (wins.livemap && (!ids || ids.includes("livemap"))) wins.livemap.draw();
}

// ─────────────────────────────────────────────────────────────
// Heats and pilots
// ─────────────────────────────────────────────────────────────

async function loadHeats() {
  S.heats = (await api("/api/race/heats")).heats;
  fillHeatSel();
}

function fillHeatSel() {
  const classes = new Set(S.heats.map((h) => h.layout)).size > 1;
  $("#heatSel").title = "* a pilot has a missed gate to check (Review in the app)";
  $("#heatSel").innerHTML = `<option value="">— pick a heat —</option>` + S.heats.map((h) =>
    `<option value="${h.heat}" ${S.H?.heat === h.heat ? "selected" : ""}>${esc(h.label)}${classes ? ` · ${esc(h.layout)}` : ""} · ${esc(h.pilots.join(", "))}${h.check.length ? " *" : ""}</option>`).join("");
}

function fillPilotSel() {
  const H = S.H, sel = $("#pilotSel");
  sel.parentElement.hidden = !H;
  if (!H) return;
  sel.innerHTML = (H.combo ? `<option value="">⊞ All pilots</option>` : "") + H.pilots.map((p) =>
    `<option value="${p.i}" ${S.pilot === p.i ? "selected" : ""}>${esc(flagName(p))}${p.channel ? ` · ${esc(p.channel)}` : ""}</option>`).join("");
  sel.title = "Every pilot of the heat racing each other, or one pilot's run (🚩: a missed gate to check)";
}

async function selectHeat(n, pilot = null) {
  S.H = null; S.combo = null; S.solo = null; S.run = null; S.pilot = null;
  if (!n) { fillPilotSel(); render(); return; }
  try {
    status("Loading the heat…");
    const H = await api(`/api/race/heat/${n}`);
    H.track = H.label;
    for (const p of H.pilots) {
      p.passes = p.passes.map((x) => ({ ...x, source: "RACE" }));
      Object.assign(p, racePilot(p, p.i, p.name));
    }
    S.H = H;
    fillHeatSel();
    openView(H.combo && pilot === null ? null : pilot ?? 0, false);
    status(`${H.label} — ${H.pilots.length} pilot${H.pilots.length === 1 ? "" : "s"}, ${H.n_gates} gates`
      + (H.combo ? " · a click on a pilot's picture opens its run" : ""));
  } catch (e) { status(e.message, true); fillPilotSel(); render(); }
}

// every pilot (k = null) or one pilot's run: same video file in a combo heat, so the playhead stays where it is
function openView(k, keepT = true) {
  const H = S.H, t = S.t;
  if (!H) return;
  S.pilot = k;
  S.combo = k === null ? H : null;
  const p = k === null ? null : H.pilots[k];
  S.run = p && { n_gates: H.n_gates, passes: p.passes, laps: p.laps, stats: p.stats, name: p.name, stem: p.stem, fps: p.fps };
  S.solo = p && { n_gates: H.n_gates, legs: H.legs, layout: H.layout, track: H.track, pilots: [racePilot(p, 0, p.name)] };
  S.crop = p && H.combo ? p.crop : null;
  comboMemo = null;
  fillPilotSel();
  if (layoutKind() !== S.layoutKind) applyLayout();
  const url = (p || H.pilots[0]).video_url;
  setVideoSource(url);
  if (keepT) seekReady(t);
  else if (H.span) seekReady(H.span[0]);
  fitVideo();
  tellParent();
  render();
}

// the dashboard keeps the heat and pilot in its address (a link to share)
function tellParent() {
  try { parent.postMessage({ raceview: { heat: S.H?.heat ?? null, pilot: S.pilot } }, location.origin); } catch { /* opened alone */ }
}

// ─────────────────────────────────────────────────────────────
// Laps of the open pilot
// ─────────────────────────────────────────────────────────────

// The laps as both the timeline and the HUD show them: the completed laps, then the unfinished
// one — from the last G1 to the pilot's last gate pass (after that the run is over, no lap is running).
function lapSegments() {
  if (!S.run) return [];
  const laps = S.run.laps || [];
  const segs = laps.map((L, i) => ({ i, n: L.lap, t0: L.t0, t1: L.t1, dt: L.dt, done: true, missed: lapMissed(L), rec: lapRecovered(L) }));
  const g1 = S.run.passes.filter((p) => p.gate_id === 1);
  const start = laps.length ? laps[laps.length - 1].t1 : g1[0]?.t;
  if (start !== undefined) {
    const after = S.run.passes.filter((p) => p.t > start + 1e-3);
    if (after.length) segs.push({ i: -1, n: laps.length + 1, t0: start, t1: after[after.length - 1].t, dt: null, done: false });
  }
  return segs;
}

// The gates a completed lap skipped (known once the lap closes at G1): [{gate, t}], t placed between the gates
// passed before and after it. The pilot cut it or the detector missed it: a lap to check (Review in the app).
function lapGaps(L, n = S.run?.n_gates || 0) {
  if (n < 2 || !L.splits) return [];
  const seen = new Map(L.splits.map((s) => [s.gate_id, s.t]));
  const out = [];
  for (let g = 2; g <= n; g++) {
    if (seen.has(g)) continue;
    let a = g - 1; while (a > 1 && !seen.has(a)) a--;
    let b = g + 1; while (b <= n && !seen.has(b)) b++;
    const ta = seen.get(a) ?? L.t0, tb = b <= n ? seen.get(b) : L.t1;
    out.push({ gate: g, t: ta + (tb - ta) * (g - a) / (b - a) });
  }
  return out;
}
const lapMissed = (L, n) => lapGaps(L, n);
const missTxt = (m) => m.map((x) => `G${x.gate}`).join(", ");
// gates the lap rescue filled in (the scorer's near-misses at the right time): checked too
const lapRecovered = (L) => (L.splits || []).filter((s) => s.recovered).map((s) => ({ gate: s.gate_id, t: s.t }));

function lapAt(t) {
  // the lap segment at time t, or null (before the first G1, or after the run ended)
  return lapSegments().find((L) => t >= L.t0 && (L.done ? t < L.t1 : t <= L.t1 + 0.5)) || null;
}

// race time (from the race start, as FPVTrackside's replay shows it) of a video time
const raceT0 = () => S.H?.pilots[S.pilot ?? 0]?.live_start_s || 0;

// ─────────────────────────────────────────────────────────────
// Video window (with overlay)
// ─────────────────────────────────────────────────────────────

function initVideo(w) {
  w.body.style.padding = "0";
  w.body.innerHTML = `<div class="vidwrap"><div class="vidbox"><video preload="auto" playsinline muted></video><canvas></canvas>
    <div class="vspin" hidden><i></i><span>Loading video…</span></div></div>
    <div class="vidctl">
      <button data-a="play" title="Space">▶</button>
      <button data-a="-1s" title="←">−1s</button><button data-a="-f" title="↓">−1f</button>
      <button data-a="+f" title="↑">+1f</button><button data-a="+1s" title="→">+1s</button>
      <span class="t">0.00 s</span>
      <select data-a="rate"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
      <label><input type="checkbox" data-ov="hud"> lap HUD</label>
      <label title="While it plays: a beep per gate, a fanfare on the start gate"><input type="checkbox" data-rs="sound"> 🔊 sound</label>
      <label title="While it plays: lap times called out with the pilot's name, and with every pilot the places as they change: who takes the lead, who's now second, who's down, who finishes"><input type="checkbox" data-rs="voice"> 🗣 voice</label>
    </div></div>`;
  const v = $("video", w.body), c = $("canvas", w.body);
  S.videoEl = v;
  setVideoSource(v.dataset.src || (S.H && (S.run ? S.H.pilots[S.pilot] : S.H.pilots[0]).video_url) || "");
  w.body.querySelectorAll("[data-ov]").forEach((cb) => {
    cb.checked = !!S.ov[cb.dataset.ov];
    cb.onchange = () => { S.ov[cb.dataset.ov] = cb.checked; store.set("raceview.overlay", S.ov); drawOverlay(); };
  });
  w.body.querySelectorAll("[data-rs]").forEach((cb) => {
    cb.checked = !!S.raceSet[cb.dataset.rs];
    cb.onchange = () => { S.raceSet[cb.dataset.rs] = cb.checked; store.set("raceview.sound", S.raceSet); if (cb.checked) unlockSound(); };
  });
  w.body.querySelector("[data-a=rate]").onchange = (e) => { v.playbackRate = +e.target.value; };
  w.body.querySelectorAll("button[data-a]").forEach((b) => b.onclick = () => control(b.dataset.a));
  v.addEventListener("play", () => { $("[data-a=play]", w.body).textContent = "⏸"; });
  v.addEventListener("pause", () => { $("[data-a=play]", w.body).textContent = "▶"; });
  v.addEventListener("seeked", tick);
  v.addEventListener("play", unlockSound);
  v.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(v.duration) && v.duration > 0) S.duration = v.duration;
    fitVideo(); tick();
  });
  v.addEventListener("loadeddata", tick);
  // spinner while the video loads or buffers
  const spin = $(".vspin", w.body);
  const busy = (on, msg = "Loading video…") => { spin.hidden = !on; $("span", spin).textContent = msg; };
  v.addEventListener("loadstart", () => busy(!!v.getAttribute("src")));
  v.addEventListener("waiting", () => { if (!v.paused) busy(true, "Buffering…"); });
  // loadeddata too: iPhone Safari doesn't fire canplay before the first tap on play
  for (const e of ["loadeddata", "canplay", "playing", "seeked", "emptied"]) v.addEventListener(e, () => busy(false));
  v.addEventListener("error", () => { if (v.getAttribute("src")) busy(true, "Couldn't load the video"); spin.classList.add("err"); });
  v.addEventListener("loadstart", () => spin.classList.remove("err"));
  const spinCheck = setInterval(() => {            // browsers don't always send the events above: hide it once a frame is there
    if (!spin.hidden && !spin.classList.contains("err") && v.readyState >= 2 && !v.seeking) busy(false);
  }, 400);
  c.onclick = (e) => {                            // every pilot: a click on a pilot's picture opens its run
    const C = S.combo, g = viewGeom(c);
    if (!C || !g) return;
    const r = c.getBoundingClientRect();
    const fx = (e.clientX - r.left - g.ox) / (g.s * g.vw), fy = (e.clientY - r.top - g.oy) / (g.s * g.vh);
    const p = C.pilots.find((q) => q.crop && fx >= q.crop[0] && fx < q.crop[0] + q.crop[2] && fy >= q.crop[1] && fy < q.crop[1] + q.crop[3]);
    if (p) openView(p.i);
  };
  const ro = new ResizeObserver(() => { fitVideo(); drawOverlay(); });
  ro.observe(c);
  w.overlay = c;
  w.cleanup = () => { ro.disconnect(); clearInterval(spinCheck); v.pause(); S.videoEl = null; };
}

function setVideoSource(url) {
  const v = S.videoEl;
  if (!v) return;
  if (v.dataset.src !== url) {
    v.dataset.src = url;
    if (url) v.src = url; else v.removeAttribute("src");
    S.t = 0;
  }
}

// one pilot of a 2×2 heat video: the video is placed so that only its quadrant shows, as big as the window allows
function fitVideo() {
  const v = S.videoEl;
  if (!v) return;
  const W = v.videoWidth, Hh = v.videoHeight, cr = S.crop;
  if (!cr || !W) { v.removeAttribute("style"); return; }
  const bw = v.parentElement.clientWidth, bh = v.parentElement.clientHeight;
  const s = Math.min(bw / (cr[2] * W), bh / (cr[3] * Hh));
  Object.assign(v.style, { objectFit: "fill", inset: "auto", width: W * s + "px", height: Hh * s + "px",
    left: (bw - cr[2] * W * s) / 2 - cr[0] * W * s + "px", top: (bh - cr[3] * Hh * s) / 2 - cr[1] * Hh * s + "px" });
}

// the picture shown in the canvas c: its size in video pixels (a quadrant: its own), scale and offset
function viewGeom(c) {
  const v = S.videoEl;
  if (!v || !v.videoWidth) return null;
  const cr = S.crop || [0, 0, 1, 1], vw = v.videoWidth * cr[2], vh = v.videoHeight * cr[3];
  const cw = c.clientWidth, ch = c.clientHeight, s = Math.min(cw / vw, ch / vh);
  return { vw, vh, s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2, cw, ch };
}

function control(a) {
  const v = S.videoEl;
  if (!v || !S.H) return;
  const f = 1 / (S.run?.fps || 30);
  if (a === "play") { if (v.paused) { unlockSound(); v.play(); } else v.pause(); return; }
  const d = { "-1s": -1, "+1s": 1, "-f": -f, "+f": f, "-10s": -10, "+10s": 10 }[a];
  if (d !== undefined) { v.pause(); seek(v.currentTime + d); }
}

function seek(t) {
  const v = S.videoEl;
  S.t = Math.max(0, Math.min(t, S.duration || t));
  if (v && S.H) v.currentTime = S.t;
  tick();
}

// seek once the video's length is known (else the browser drops it)
function seekReady(t) {
  const v = S.videoEl;
  if (v && v.readyState < 1) v.addEventListener("loadedmetadata", () => seek(t), { once: true });
  seek(t);
}

function drawOverlay() {
  const w = wins.video;
  if (!w || !w.overlay || !S.videoEl) return;
  const c = w.overlay;
  const dpr = devicePixelRatio || 1;
  const cw = c.clientWidth, ch = c.clientHeight;
  if (c.width !== Math.round(cw * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cw, ch);
  const G = viewGeom(c);
  if (!G) return;
  const { vw, vh, s, ox, oy } = G;
  g.font = "12px -apple-system, sans-serif";
  g.textBaseline = "top";
  if (S.ov.hud && S.run) {
    // recent pass flash + lap timer
    const recent = S.run.passes.filter((p) => p.t <= S.t && S.t - p.t < 0.6);
    g.font = "bold 22px -apple-system, sans-serif";
    recent.forEach((p, i) => {
      g.fillStyle = p.recovered ? "#ff9f43" : "#4cd97b";
      g.fillText(`G${p.gate_id} ${p.recovered ? "?" : "✓"}`, ox + 12, oy + 10 + i * 26);
    });
    const la = lapAt(S.t);
    if (la) {
      const txt = `Lap ${la.n}${la.done ? "" : " (unfinished)"}  ${fmt(S.t - la.t0)}s`;
      const tw = g.measureText(txt).width;
      g.fillStyle = "rgba(0,0,0,.6)"; g.fillRect(ox + vw * s - tw - 20, oy + 6, tw + 14, 30);
      g.fillStyle = "#ffb020"; g.fillText(txt, ox + vw * s - tw - 13, oy + 10);
    }
    // a lap that just closed with a skipped gate: flagged to check
    const ended = lapSegments().find((L) => L.done && (L.missed.length || L.rec.length) && S.t >= L.t1 && S.t - L.t1 < 4);
    if (ended) {
      const bad = ended.missed.length > 0;
      const parts = [bad ? `missed ${missTxt(ended.missed)}` : "", ended.rec.length ? `recovered ${missTxt(ended.rec)}` : ""].filter(Boolean);
      const txt = `${bad ? "🚩" : "🟠"} Lap ${ended.n}: ${parts.join(" · ")} · to check`;
      const tw = g.measureText(txt).width;
      g.fillStyle = bad ? "rgba(120,0,0,.75)" : "rgba(110,60,0,.75)"; g.fillRect(ox + vw * s - tw - 20, oy + 40, tw + 14, 30);
      g.fillStyle = bad ? "#ff5c5c" : "#ff9f43"; g.fillText(txt, ox + vw * s - tw - 13, oy + 44);
    }
  }
  c.style.cursor = S.combo ? "pointer" : "";
  c.title = S.combo ? "Click a pilot: its own run, laps and missed gates" : "";
  if (S.combo && S.ov.hud) drawComboOverlay(g, G);
  const co = S.raceCallout;                     // the last call-out, a few seconds, at the bottom
  if (co && performance.now() - co.wall < 4000) {
    g.font = "bold 18px -apple-system, sans-serif";
    const txt = `🗣 ${co.text}`, tw = g.measureText(txt).width;
    g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(ox + (vw * s - tw) / 2 - 10, oy + vh * s - 44, tw + 20, 30);
    g.fillStyle = "#ffd36b"; g.fillText(txt, ox + (vw * s - tw) / 2, oy + vh * s - 40);
  }
}

// one tick per displayed frame while playing; also after seeks
let lastLive = 0;
function tick() {
  const was = S.t;
  if (S.videoEl) S.t = S.videoEl.currentTime;
  raceSounds(was, S.t, S.videoEl && !S.videoEl.paused);
  const w = wins.video;
  if (w) $(".t", w.body).textContent = `${fmt(S.t - raceT0())} s${raceT0() ? " race time" : ""}`;
  drawOverlay();
  if (wins.timeline) wins.timeline.draw();
  const now = performance.now();
  const playing = S.videoEl && !S.videoEl.paused;
  if (!playing || now - lastLive > 120) {
    lastLive = now;
    render(Object.keys(PANELS).filter((id) => PANELS[id].live));
  }
}

function loop() {
  if (S.videoEl && !S.videoEl.paused) tick();
  requestAnimationFrame(loop);
}

// ─────────────────────────────────────────────────────────────
// Timeline: the race (its pilots' passes) and a little around it, not the whole video (staging, the next race)
// ─────────────────────────────────────────────────────────────

const span = () => S.H?.span || [0, S.duration || 0];

function initTimeline(w) {
  w.body.style.padding = "0";
  w.body.innerHTML = `<canvas class="timeline"></canvas>`;
  const c = $("canvas", w.body);
  let hover = null;
  const tOf = (e) => {
    const r = c.getBoundingClientRect(), [a, b] = span();
    return a + ((e.clientX - r.left) / r.width) * (b - a);
  };
  c.onmousedown = (e) => { if (S.H) { S.videoEl?.pause(); seek(tOf(e)); } };
  c.ondblclick = (e) => {                         // every pilot: a double click on a pilot's lane opens its run
    const np = S.combo?.pilots.length;
    if (!np) return;
    const r = c.getBoundingClientRect(), k = Math.floor((e.clientY - r.top - 2) / ((r.height - 20) / np));
    if (k >= 0 && k < np) openView(k);
  };
  c.onmousemove = (e) => { hover = tOf(e); w.draw(); };
  c.onmouseleave = () => { hover = null; w.draw(); };
  w.draw = () => {
    const dpr = devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#181b20"; g.fillRect(0, 0, cw, ch);
    const [A, B] = span(), D = B - A;
    if (!S.H || !(D > 0)) { g.fillStyle = "#9aa3ad"; g.fillText("Pick a heat", 10, 20); return; }
    const X = (t) => ((t - A) / D) * cw, t0 = raceT0();
    const mid = ch * 0.55;
    g.fillStyle = "#2a2f37"; g.fillRect(0, mid - 1, cw, 2);
    g.font = "10px -apple-system, sans-serif";
    g.textBaseline = "top";
    // seconds grid, in race time
    const step = D > 300 ? 60 : D > 120 ? 20 : 10;
    g.fillStyle = "#59616b";
    for (let s = Math.ceil((A - t0) / step) * step; s + t0 <= B; s += step) { g.fillRect(X(s + t0), ch - 6, 1, 6); g.fillText(`${s}s`, X(s + t0) + 2, ch - 14); }
    if (S.combo) drawComboLanes(g, cw, ch, X);
    if (S.run) {
      for (const L of lapSegments()) {
        g.fillStyle = L.done ? "rgba(255,176,32,.08)" : "rgba(255,176,32,.03)";
        g.fillRect(X(L.t0), 0, X(L.t1) - X(L.t0), ch);
        g.fillStyle = "#ffb020"; g.fillRect(X(L.t0), 2, 2, ch - 4);
        if (L.done) g.fillRect(X(L.t1) - 1, 2, 2, ch - 4);
        if (L.done && L.missed.length) g.fillStyle = "#ff5c5c";
        else if (L.done && L.rec.length) g.fillStyle = "#ff9f43";
        g.fillText(L.done ? `${L.missed.length ? "🚩 " : L.rec.length ? "🟠 " : ""}L${L.n} ${fmt(L.dt)}s` : `L${L.n} unfinished`, X(L.t0) + 4, 4);
        for (const m of L.done ? L.missed : []) {      // skipped gate: red mark where it should have been
          g.fillStyle = "#ff5c5c";
          g.fillRect(X(m.t) - 1, mid - 10, 2, 20);
          g.fillText(`G${m.gate}✕`, X(m.t) - 8, mid - 24);
        }
      }
      for (const p of S.run.passes) {
        g.fillStyle = p.recovered ? "#ff9f43" : "#4cd97b";
        g.fillRect(X(p.t) - 1, mid - (p.gate_id === 1 ? 16 : 10), 2, p.gate_id === 1 ? 32 : 20);
        if (cw / D > 8) g.fillText(String(p.gate_id), X(p.t) - 3, mid + 12);
      }
    }
    g.fillStyle = "#ff4d4d"; g.fillRect(X(S.t) - 1, 0, 2, ch);
    if (hover !== null) { g.fillStyle = "#e6e6e6"; g.fillText(`${fmt(hover - t0)}s`, Math.min(X(hover) + 4, cw - 40), ch / 2 - 30); }
  };
  const ro = new ResizeObserver(() => w.draw());
  ro.observe(c);
  w.cleanup = () => ro.disconnect();
  w.draw();
}

// ─────────────────────────────────────────────────────────────
// Race stats + gate splits (one pilot)
// ─────────────────────────────────────────────────────────────

function needRun(w) {
  if (!S.H) { w.body.innerHTML = `<div class="empty">Pick a heat.</div>`; return true; }
  if (!S.run) {
    w.body.innerHTML = `<div class="empty">Every pilot of the heat: their places are in <b>Race — standings</b>. One pilot's laps and
      splits: pick the pilot in the top bar, or click their picture.</div>`;
    return true;
  }
  return false;
}

function renderStats(w) {
  if (needRun(w)) return;
  const st = S.run.stats, laps = S.run.laps;
  const la = lapAt(S.t);
  const cur = la ? S.t - la.t0 : null;
  const k = (l, v, cls = "") => `<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  let h = `<div class="kpis">
    ${k("Current lap", cur !== null ? fmt(cur) + "s" : "—", "live")}
    ${k("Laps", laps.length)}
    ${k("Best lap", st?.best_lap != null ? fmt(st.best_lap) + "s" : "—")}
    ${k("Last lap", laps.length ? fmt(laps[laps.length - 1].dt) + "s" : "—")}
    ${k("Average", st?.mean_lap != null ? fmt(st.mean_lap) + "s" : "—")}
    ${k("Consistency (CV)", st?.lap_cv != null ? fmt(st.lap_cv * 100, 1) + "%" : "—")}
    ${k("Theoretical best", st?.theoretical_best ? fmt(st.theoretical_best) + "s" : "—")}
    ${k("Best 3 in a row", st?.best3 ? fmt(st.best3) + "s" : "—")}
    ${k("Holeshot", st?.holeshot != null ? fmt(st.holeshot) + "s" : "—")}
  </div>`;
  if (!laps.length) {
    h += `<div class="empty">No complete lap: the start gate (G1) was not crossed twice.</div>`;
  } else {
    const best = Math.min(...laps.map((l) => l.dt)), fm = S.H.field?.median_lap;
    h += `<table><tr><th>Lap</th><th>Time</th><th>Δ best</th>${fm ? "<th>vs field</th>" : ""}<th>Gates</th><th>Missed</th><th>Start</th></tr>`;
    let flagged = 0, recov = 0;
    laps.forEach((L, i) => {
      const n = new Set(L.splits.map((s) => s.gate_id)).size;
      const miss = lapMissed(L), rec = lapRecovered(L);
      flagged += miss.length > 0;
      recov += rec.length > 0 && !miss.length;
      const chk = [miss.length ? `<span class="bad">missed ${missTxt(miss)}</span>` : "",
        rec.length ? `<span class="warn">recovered ${missTxt(rec)}</span>` : ""].filter(Boolean).join(" · ");
      h += `<tr class="click ${la && la.i === i ? "cur" : ""} ${miss.length ? "flag" : rec.length ? "flag-rec" : ""}" data-t="${L.t0}">
        <td>${miss.length ? "🚩 " : rec.length ? "🟠 " : ""}Lap ${L.lap}</td><td class="${L.dt === best ? "best" : ""}">${fmt(L.dt)}s</td>
        <td>${L.dt === best ? "★" : "+" + fmt(L.dt - best)}</td>${fm ? `<td class="${L.dt < fm ? "good" : "bad"}">${L.dt < fm ? "" : "+"}${fmt(L.dt - fm)}</td>` : ""}
        <td>${n}/${S.run.n_gates}</td><td>${chk || "—"}</td><td>${fmt(L.t0 - raceT0(), 1)}s</td></tr>`;
    });
    h += `</table>`;
    if (flagged) h += `<div class="hint"><span class="bad">🚩 ${flagged} lap${flagged > 1 ? "s" : ""} skipped a gate</span>: the pilot cut it,
      or the video lost it. Check the lap (Review in the app) before it counts.</div>`;
    if (recov) h += `<div class="hint"><span class="warn">🟠 ${recov} lap${recov > 1 ? "s" : ""} complete with recovered gates</span>: a weak
      detection at the right time and place was taken as the pass.</div>`;
    if (fm) h += `<div class="hint">vs field: against the field's typical lap (${fmt(fm)}s). Start: race time.</div>`;
  }
  w.body.innerHTML = h;
  w.body.querySelectorAll("tr[data-t]").forEach((tr) => tr.onclick = () => { S.videoEl?.pause(); seek(+tr.dataset.t); });
}

function renderSplits(w) {
  if (needRun(w)) return;
  const st = S.run.stats, n = S.run.n_gates, field = S.H.field;
  let h = "";
  // live: the lap at the playhead, gate by gate
  const la = lapAt(S.t);
  // the lap that closed last (before the playhead): skipped gates are known once it closes
  const closed = lapSegments().filter((L) => L.done && L.t1 <= S.t).pop();
  if (closed?.missed.length) {
    h += `<div class="spotter">🚩 Lap ${closed.n} (${fmt(closed.dt)}s) missed ${missTxt(closed.missed)}${closed.rec.length ? ` · recovered ${missTxt(closed.rec)}` : ""} · to check</div>`;
  } else if (closed?.rec.length) {
    h += `<div class="spotter rec">🟠 Lap ${closed.n} (${fmt(closed.dt)}s) recovered ${missTxt(closed.rec)} · to check</div>`;
  }
  if (la) {
    const t0 = la.t0;
    const ps = S.run.passes.filter((p) => p.t >= t0 - 1e-3 && p.t <= S.t && (!la.done || p.t < la.t1));
    h += `<h4>Lap ${la.n}${la.done ? "" : " (unfinished)"} at the playhead — ${fmt(S.t - t0)}s</h4>
      <table><tr><th>Gate</th><th>From start</th><th>Since prev</th><th>Best section</th><th>Δ</th>${field ? "<th>Field</th>" : ""}</tr>`;
    // skipped gates show up as soon as a later gate is passed
    const skipped = la.done ? la.missed : [];
    let prev = null;
    for (const p of ps) {
      for (const m of skipped.filter((m) => m.gate < p.gate_id && m.gate > (prev ? prev.gate_id : 1)).sort((a, b) => a.gate - b.gate)) {
        h += `<tr class="flag"><td class="bad">G${m.gate} ✕</td><td colspan="${field ? 5 : 4}" class="bad">missed · to check</td></tr>`;
      }
      const leg = prev ? p.t - prev.t : null;
      const step = prev && (p.gate_id - prev.gate_id + n) % n === 1;
      const sec = step ? st?.section_best?.[prev.gate_id - 1] : null, med = step ? field?.section_median?.[prev.gate_id - 1] : null;
      h += `<tr class="click ${p.recovered ? "flag-rec" : ""}" data-t="${p.t}" ${p.recovered ? 'title="recovered: a weak detection at the expected time"' : ""}>
        <td class="${p.recovered ? "warn" : ""}">G${p.gate_id}${p.recovered ? " ?" : ""}</td><td>${fmt(p.t - t0)}s</td>
        <td>${leg !== null ? fmt(leg) + "s" : "—"}</td><td>${sec ? fmt(sec) + "s" : "—"}</td>
        <td>${sec && leg !== null ? (leg - sec < 0.005 ? "★" : "+" + fmt(leg - sec)) : ""}</td>
        ${field ? `<td class="${med && leg !== null ? (leg < med ? "good" : "bad") : ""}">${med && leg !== null ? (leg < med ? "" : "+") + fmt(leg - med) : ""}</td>` : ""}</tr>`;
      prev = p;
    }
    h += `</table>`;
  }
  if (st && st.section_grid?.length) {
    const secs = st.sections;
    h += `<h4>All laps — seconds between gates</h4><div style="overflow:auto"><table><tr><th>Lap</th>${secs.map((s) => `<th>${s.replace("→", "→<br>")}</th>`).join("")}<th>Lap</th></tr>`;
    st.section_grid.forEach((row, i) => {
      const L = S.run.laps[i];
      if (!L) return;
      const miss = lapMissed(L), rec = lapRecovered(L);
      h += `<tr class="click ${la && la.i === i ? "cur" : ""} ${miss.length ? "flag" : ""}" data-t="${L.t0}"
        title="${[miss.length ? "missed " + missTxt(miss) : "", rec.length ? "recovered " + missTxt(rec) : ""].filter(Boolean).join(" · ")}"><td>${miss.length ? "🚩 " : rec.length ? "🟠 " : ""}L${i + 1}</td>` + row.map((x, g) =>
        x === null ? `<td class="miss ${miss.length ? "bad" : ""}">·</td>` : `<td class="${st.section_best[g] !== null && Math.abs(x - st.section_best[g]) < 1e-6 ? "best" : ""}">${fmt(x)}</td>`).join("")
        + `<td>${fmt(st.lap_times[i])}</td></tr>`;
    });
    h += `<tr><td><b>Best</b></td>${st.section_best.map((x) => `<td class="best">${fmt(x)}</td>`).join("")}<td class="best">${fmt(st.theoretical_best)}</td></tr>`;
    h += `<tr><td>Median</td>${st.section_median.map((x) => `<td>${fmt(x)}</td>`).join("")}<td>${fmt(st.median_lap)}</td></tr>`;
    if (field?.section_median?.length === secs.length)
      h += `<tr><td class="hint">Field</td>${field.section_median.map((x) => `<td class="hint">${fmt(x)}</td>`).join("")}<td class="hint">${fmt(field.median_lap)}</td></tr>`;
    h += `</table></div><div class="hint">Best row sum = theoretical best lap. · = that leg wasn't flown cleanly (a gate missed in between). Field = the typical pilot's time (median).</div>`;
  } else if (!la) {
    h += `<div class="empty">No laps yet.</div>`;
  }
  w.body.innerHTML = h;
  w.body.querySelectorAll("tr[data-t]").forEach((tr) => tr.onclick = () => { S.videoEl?.pause(); seek(+tr.dataset.t); });
  const last = la && S.videoEl && !S.videoEl.paused ? w.body.querySelector("table")?.querySelector("tr:last-child") : null;
  if (last) {                                     // playing: the gate just passed stays in sight
    const b = w.body.getBoundingClientRect(), r = last.getBoundingClientRect();
    if (r.bottom > b.bottom) w.body.scrollTop += r.bottom - b.bottom + 6;
    else if (r.top < b.top) w.body.scrollTop -= b.top - r.top + 6;
  }
}

function renderGates(w) {
  const G = S.H?.gates;
  if (!S.H) { w.body.innerHTML = `<div class="empty">Pick a heat.</div>`; return; }
  if (!G || !G.length) { w.body.innerHTML = `<div class="empty">This track has no learned gates yet.</div>`; return; }
  w.body.innerHTML = `<div class="gates">${G.map((g) => `<div class="gate"><div class="h">G${g.id} <span>${esc(g.type)}${g.offset_s != null ? " · +" + fmt(g.offset_s, 1) + "s" : ""}</span></div>
    <div class="imgs">${g.images.slice(0, 6).map((u) => `<img loading="lazy" src="${esc(u)}">`).join("")}</div></div>`).join("")}</div>`;
}

// ─────────────────────────────────────────────────────────────
// Sounds and voice (a page plays sound only after a click on it: ▶ does that)
// ─────────────────────────────────────────────────────────────

let audioCtx = null;
function audio() {
  if (!audioCtx) { try { audioCtx = new AudioContext(); } catch { return null; } }
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

// notes: [frequency Hz, start s, length s, wave]
const SFX = {
  gate: [[988, 0, 0.06], [1319, 0.06, 0.2]],                                   // a gate: coin
  start: [[523, 0, 0.07], [659, 0.07, 0.07], [784, 0.14, 0.07], [1047, 0.21, 0.28]],   // power-up: start / finish
};

function sfx(kind) {
  if (!S.raceSet.sound) return;
  const ac = audio();
  if (!ac || ac.state !== "running") return;
  const t0 = ac.currentTime + 0.005, out = ac.createGain();
  out.gain.value = 0.22;
  out.connect(ac.destination);
  for (const [f, at, len, wave] of SFX[kind] || SFX.gate) {
    const o = ac.createOscillator(), g = ac.createGain(), t = t0 + at;
    o.type = wave || "square";
    o.frequency.setValueAtTime(f, t);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(1, t + 0.004);
    g.gain.setValueAtTime(1, t + len * 0.5);
    g.gain.exponentialRampToValueAtTime(0.0001, t + len);
    o.connect(g).connect(out);
    o.start(t); o.stop(t + len + 0.01);
  }
}

let utterance = null;                          // kept referenced: Chrome drops an utterance that gets garbage-collected

function say(text) {
  S.raceCallout = { text, wall: performance.now() };   // shown over the video
  if (!S.raceSet.voice) return;
  const ss = window.speechSynthesis;
  if (!ss) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.1;
  u.lang = "en-US";
  const v = ss.getVoices().find((x) => x.lang === "en-US" && x.localService) || ss.getVoices().find((x) => x.lang?.startsWith("en"));
  if (v) u.voice = v;
  utterance = u;
  ss.resume();                                  // Chrome can leave the queue paused
  ss.speak(u);                                  // several pilots: every call-out, one after another
}

function unlockSound() {
  audio();
  const ss = window.speechSynthesis;
  if (ss && !unlockSound.done) {                // a silent utterance inside the click lets later ones play
    const u = new SpeechSynthesisUtterance(" ");
    u.volume = 0;
    ss.speak(u);
    unlockSound.done = true;
  }
}

// ─────────────────────────────────────────────────────────────
// The race: every pilot's place at the playhead — standings, call-outs, the race map, the race in 3D
// ─────────────────────────────────────────────────────────────

const PILOT_COLORS = ["#ff5050", "#3caaff", "#5adc5a", "#ffc83c"];
const pilotName = (i, R) => R.pilots?.[i]?.name || `Pilot ${i + 1}`;

// Each pilot's progress: its laps as they started (G1), and within the current lap the furthest gate reached —
// forward only, never more than half the track at once (a misnamed gate doesn't move the pilot) — so pilots compare
// across laps. The leader is the furthest; at the same gate, the first.
function raceState(L) {
  const n = L.gates || 0;
  const ps = (L.pilots || []).map((p, i) => ({ ...p, i, cum: -1, g: 0, t: null, cross: {}, laps: [], lap: 0 }));
  if (!n) return ps;
  const starts = [...(L.starts || [])].sort((a, b) => a.t - b.t);
  const passes = L.passes.filter((x) => x.pilot != null && x.gate_id > 0).sort((a, b) => a.t - b.t);
  for (const q of ps) {
    const st = starts.filter((x) => x.pilot === q.i);
    if (!st.length) continue;
    const mine = passes.filter((x) => x.pilot === q.i);
    st.forEach((s0, k) => {                           // lap k+1: from its start to the next one
      const end = k + 1 < st.length ? st[k + 1].t : Infinity;
      let g = 1, t = s0.t;
      q.cross[k * n] = s0.t;
      for (const x of mine) {
        if (x.t < s0.t || x.t >= end) continue;
        if (x.gate_id > g && x.gate_id - g <= n / 2) { g = x.gate_id; t = x.t; q.cross[k * n + g - 1] = t; }
      }
      q.lap = k + 1; q.g = g; q.t = t; q.cum = k * n + g - 1;
    });
  }
  for (const l of L.laps || []) if (ps[l.pilot]) ps[l.pilot].laps.push(l);
  const order = [...ps].sort((a, b) => b.cum - a.cum || (a.cross[a.cum] ?? 1e9) - (b.cross[b.cum] ?? 1e9));
  order.forEach((q, k) => { q.pos = q.cum >= 0 ? k + 1 : null; });
  const lead = order[0];
  for (const q of ps) {                              // down: a pilot whose run ended short of the race (crashed, out)
    q.done = L.now != null && q.end != null && L.now > q.end + 3;
    q.finished = !!q.fin && L.now >= q.end;          // its last lap closed: right away
    q.down = q.cum >= 0 && !q.finished && q.done;
  }
  for (const q of ps) {
    q.best = q.laps.length ? Math.min(...q.laps.map((l) => l.dt)) : null;
    q.gap = lead && q !== lead && q.cum >= 0 && lead.cross[q.cum] != null ? q.t - lead.cross[q.cum] : null;
    q.behind = lead && q.cum >= 0 ? Math.floor((lead.cum - q.cum) / n) : 0;     // whole laps down
  }
  ps.order = order;
  return ps;
}

// Call-outs as the order changes: "X takes the lead", "X is now second" — said of who moved up (whoever they passed
// isn't) —, "X is down", "X finishes, second". A new place counts once held POS_HOLD s of race time and clearly (a
// gate ahead of the next pilot, not a tie at the same gate); one place call every POS_GAP s at most (a lead change:
// LEAD_GAP), the lead first. quiet: only take the race as it is now (a seek).
const POS_HOLD = 1.5, POS_GAP = 3, LEAD_GAP = 5;
const ORDINAL = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];
const newPlaces = () => ({ pos: {}, down: new Set(), fin: new Set(), said: -1e9, lead: -1e9 });
function placeCalls(ps, now, st, R, quiet = false) {
  const out = [], name = (q) => pilotName(q.i, R);
  if (!ps.order) return out;
  let rebase = quiet;
  for (const q of ps) {
    if (q.finished && !st.fin.has(q.i)) {
      st.fin.add(q.i);
      if (!quiet) out.push(`${name(q)} finishes${ps.length > 1 && q.pos ? ", " + (ORDINAL[q.pos] || "place " + q.pos) : ""}`);
    }
    if (q.down !== st.down.has(q.i)) {
      q.down ? st.down.add(q.i) : st.down.delete(q.i);
      rebase = true;
      if (!quiet) out.push(`${name(q)} ${q.down ? "is down" : "is back"}`);
    }
  }
  const run = ps.order.filter((q) => q.cum >= 0 && !q.down), ups = [];
  run.forEach((q, k) => {
    const p = k + 1, s = st.pos[q.i];
    if (!s || rebase || q.cum < 1) { st.pos[q.i] = { p, cand: null }; return; }     // new here, or still at the start
    if (p === s.p) { s.cand = null; return; }
    if (!s.cand || s.cand.p !== p) { s.cand = { p, t: now }; return; }
    if (now - s.cand.t < POS_HOLD) return;
    if (p > s.p) { s.p = p; s.cand = null; return; }            // moved down: said of whoever passed them
    if (run[k + 1] && run[k + 1].cum === q.cum) return;         // at the same gate as the next one: not clear yet
    ups.push({ q, p, s });
  });
  const u = ups.sort((a, b) => a.p - b.p)[0];
  if (u && ps.length > 1 && now - st.said >= POS_GAP && (u.p > 1 || now - st.lead >= LEAD_GAP)) {
    u.s.p = u.p; u.s.cand = null; st.said = now;
    if (u.p === 1) st.lead = now;
    out.push(u.p === 1 ? `${name(u.q)} takes the lead` : `${name(u.q)} is now ${ORDINAL[u.p] || "place " + u.p}`);
  }
  return out;
}

// a pilot with a missed gate to check (not once the run is reviewed in the app)
const FLAG = ` <span class="bad" title="a missed gate to check (pick the pilot: its laps)">🚩</span>`;
const pilotFlag = (p) => p.reviewed !== "complete" && pilotMissed(p).length > 0;
const flagName = (p) => (pilotFlag(p) ? "🚩 " : "") + p.name;

function renderStandings(R) {
  const ps = raceState(R), n = R.gates;
  let h = `<table class="standings"><tr><th>#</th><th>Pilot</th><th>Lap</th><th>Gate</th><th>Gap</th><th>Last lap</th><th>Best</th></tr>`;
  for (const q of [...ps].sort((a, b) => (a.pos ?? 99) - (b.pos ?? 99))) {
    const last = q.laps[q.laps.length - 1];
    h += `<tr class="click ${q.pos === 1 ? "cur" : ""}" data-pilot="${q.i}" title="Open ${esc(q.name)}'s run"><td>${q.pos ?? "—"}</td>
      <td><span class="dot" style="background:${PILOT_COLORS[q.i % 4]}"></span><b>${esc(q.name)}</b>${R.pilots?.[q.i]?.flag ? FLAG : ""}</td>
      <td>${q.lap || "—"}</td><td>${q.cum >= 0 ? "G" + q.g : "—"}</td>
      <td>${q.down ? "down" : q.done ? "finished" : q.pos === 1 ? "leader" : q.behind > 0 ? `+${q.behind} lap${q.behind > 1 ? "s" : ""}` : q.gap != null ? "+" + fmt(q.gap, 1) + "s" : "—"}</td>
      <td>${last ? fmt(last.dt) + "s" : "—"}</td><td>${q.best != null ? fmt(q.best) + "s" : "—"}</td></tr>`;
  }
  h += "</table>";
  return h + `<div class="hint">${n} gates · gap: behind the leader at the same gate · a click on a pilot: their run</div>`;
}

// a layout's gates ([{x, y}], seconds of flight, y down) fitted into a w×h canvas: their points and the middle
function layoutFit(gs, w, h, pad) {
  const xs = gs.map((q) => q.x), ys = gs.map((q) => q.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const k = Math.min((w - 2 * pad) / Math.max(1e-6, x1 - x0), (h - 2 * pad) / Math.max(1e-6, y1 - y0));
  const ox = (w - (x1 - x0) * k) / 2 - x0 * k, oy = (h - (y1 - y0) * k) / 2 - y0 * k;
  const P = gs.map((q) => [ox + q.x * k, oy + q.y * k]);
  return { P, k, ox, oy, cx: P.reduce((a, p) => a + p[0], 0) / P.length, cy: P.reduce((a, p) => a + p[1], 0) / P.length };
}

// the closed path through points P at lap fraction f — gate i at fraction at[i] — a centripetal Catmull-Rom curve
// (no loops or spikes at tight turns between uneven gates)
function splineAt(P, at, f) {
  const n = P.length;
  f = ((f % 1) + 1) % 1;
  let i = n - 1;
  while (i > 0 && at[i] > f) i--;
  const a1 = i + 1 < n ? at[i + 1] : 1, u = (f - at[i]) / Math.max(1e-9, a1 - at[i]);
  const p0 = P[(i - 1 + n) % n], p1 = P[i], p2 = P[(i + 1) % n], p3 = P[(i + 2) % n];
  const d = (a, b) => Math.max(1e-4, Math.sqrt(Math.hypot(b[0] - a[0], b[1] - a[1])));
  const t1 = d(p0, p1), t2 = t1 + d(p1, p2), t3 = t2 + d(p2, p3), t = t1 + (t2 - t1) * u;
  const mix = (a, b, ta, tb) => [0, 1].map((k) => ((tb - t) * a[k] + (t - ta) * b[k]) / (tb - ta));
  const A1 = mix(p0, p1, 0, t1), A2 = mix(p1, p2, t1, t2), A3 = mix(p2, p3, t2, t3);
  return mix(mix(A1, A2, 0, t2), mix(A2, A3, t1, t3), t1, t2);
}

const SLOW_AT = 0.8, SLOW_PACE = 0.35;     // the race map: past 80% of a leg without the next gate, 35% of the pace

// the race's track for the race map and the 3D view: each leg's learned time, the lap's, where gate i+1 is (at[i],
// a fraction of the lap)
function raceTrack(L) {
  const n = L.gates, legs = L.legs && L.legs.length === n ? L.legs : Array(n).fill(1), lap = legs.reduce((a, b) => a + b, 0);
  const at = [0]; for (let i = 0; i < n - 1; i++) at.push(at[i] + legs[i] / lap);
  return { n, legs, lap, at };
}

// where each pilot is shown ({pilot: learned seconds since its first G1}; kept in w between frames): the gate it
// passed last, then on at the learned pace; close to the next gate it slows down (that gate may be missed) but keeps
// going, slower, until a pass places it again. A pass further on than shown: it catches up quickly (no jump); a
// finished run stops.
function raceProgress(w, ps, T, now) {
  const { legs, lap, at } = T, dNow = now - (w.lastNow ?? now);
  w.lastNow = now;
  w.disp = w.disp || {};
  for (const q of ps) {
    if (q.cum < 0) { delete w.disp[q.i]; continue; }
    const gi = q.g - 1, leg = legs[gi];
    const since = Math.max(0, (q.end != null ? Math.min(now, q.end) : now) - q.t);
    const on = since <= SLOW_AT * leg ? since : SLOW_AT * leg + (since - SLOW_AT * leg) * SLOW_PACE;
    const target = (q.lap - 1) * lap + at[gi] * lap + Math.min(on, leg + lap * 0.25);
    let v = w.disp[q.i];
    if (v === undefined || dNow < 0 || dNow > 2 || Math.abs(target - v) > lap * 0.6) v = target;   // a seek, a new run
    else if (target > v) v += Math.min(target - v, Math.max(dNow, (target - v) * Math.min(1, dNow * 5)));
    w.disp[q.i] = v;
  }
  return w.disp;
}

// the race map: the gates on the track's layout (or a stadium loop), spaced by the learned time of each leg; each
// pilot a dot moving from its last gate towards the next at the learned pace
function initLiveMap(w) {
  w.body.style.padding = "0";
  w.body.innerHTML = `<canvas class="timeline" style="cursor:default"></canvas><button class="to3d" title="The race in 3D">🧊 3D</button>`;
  const c = $("canvas", w.body);
  $(".to3d", w.body).onclick = () => { openWin("race3d"); saveLayout(); };
  w.draw = () => {
    const dpr = devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#181b20"; g.fillRect(0, 0, cw, ch);
    g.font = "11px -apple-system, sans-serif"; g.textBaseline = "middle"; g.textAlign = "center";
    const L = raceSrc() ? comboRace() : null, n = L?.gates;
    if (!L || !n) {
      g.fillStyle = "#9aa3ad";
      g.fillText(L ? "The track isn't learned yet: no race map." : "Pick a heat: its pilots' progress shows here as it plays.", cw / 2, ch / 2);
      return;
    }
    const { lap, at } = raceTrack(L);
    // the track's layout when there is one: a smooth path through the gates; else a stadium-shaped loop — two
    // straights and two half circles, the start gate mid bottom straight
    const pad = 34, W = cw - 2 * pad, H = ch - 2 * pad;
    const lay = L.layout?.gates?.length === n ? layoutFit(L.layout.gates, cw, ch, pad) : null;
    const r = lay ? Math.min(W, H) / 4 : Math.max(10, Math.min(H / 2, W / 4)), sl = Math.max(0, W - 2 * r);
    const per = 2 * sl + 2 * Math.PI * r, cx = lay ? lay.cx : cw / 2, cy = lay ? lay.cy : ch / 2;
    const pt = lay ? (f) => splineAt(lay.P, at, f) : (f) => {
      let d = ((f % 1) + 1) % 1 * per;
      const segs = [[sl / 2, (d) => [cx + d, cy + r]], [Math.PI * r, (d) => { const a = Math.PI / 2 - d / r; return [cx + sl / 2 + r * Math.cos(a), cy + r * Math.sin(a)]; }],
        [sl, (d) => [cx + sl / 2 - d, cy - r]], [Math.PI * r, (d) => { const a = -Math.PI / 2 - d / r; return [cx - sl / 2 + r * Math.cos(a), cy + r * Math.sin(a)]; }],
        [sl / 2, (d) => [cx - sl / 2 + d, cy + r]]];
      for (const [len, f2] of segs) { if (d <= len) return f2(d); d -= len; }
      return [cx, cy + r];
    };
    g.strokeStyle = "#2f353e"; g.lineWidth = 14; g.beginPath();
    for (let i = 0; i <= 200; i++) { const [x, y] = pt(i / 200); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.closePath(); g.stroke();
    for (let i = 0; i < n; i++) {                                     // the gates: a bar across the path, its label outside
      const [x, y] = pt(at[i]), [x2, y2] = pt(at[i] + 0.002), tl = Math.hypot(x2 - x, y2 - y) || 1;
      const nx = -(y2 - y) / tl, ny = (x2 - x) / tl, dl = Math.hypot(x - cx, y - cy) || 1;
      g.strokeStyle = i === 0 ? "#ffb020" : "#59616b"; g.lineWidth = 4;
      g.beginPath(); g.moveTo(x - nx * 10, y - ny * 10); g.lineTo(x + nx * 10, y + ny * 10); g.stroke();
      g.fillStyle = i === 0 ? "#ffb020" : "#9aa3ad"; g.fillText(`G${i + 1}`, x + (x - cx) / dl * 22, y + (y - cy) / dl * 22);
    }
    const now = S.t, ps = raceState(L), np = ps.length;
    const lead = ps.order?.[0];
    if (lead && lead.cum >= 0) {             // the leader's lap, big: inside the stadium loop; a real layout: top right
      const big = lay ? Math.round(Math.min(ch * 0.12, 40)) : Math.round(Math.min(r * 0.9, 64));
      const lx = lay ? cw - 12 : cx, ly = lay ? 10 + big / 2 : cy - 6;
      g.textAlign = lay ? "right" : "center";
      g.fillStyle = lay ? "#59616b" : "#3a414b"; g.font = `bold ${big}px -apple-system, sans-serif`;
      g.fillText(`LAP ${lead.lap}`, lx, ly);
      g.font = "12px -apple-system, sans-serif"; g.fillStyle = "#59616b";
      g.fillText(np > 1 ? `leader ${lead.name}` : lead.name || "", lx, lay ? ly + big / 2 + 10 : cy + Math.min(r * 0.45, 34));
      g.textAlign = "center";
    }
    g.font = "bold 11px -apple-system, sans-serif";
    const disp = raceProgress(w, ps, raceTrack(L), now);
    const lane = (i) => (np > 1 ? (i - (np - 1) / 2) * 6 : 0);
    const at3 = (f, i) => {                                           // a point of pilot i's lane (beside the loop's centre line)
      const [x, y] = pt(f), [x2, y2] = pt(f + 0.002), dl = Math.hypot(x2 - x, y2 - y) || 1;
      return [x - (y2 - y) / dl * lane(i), y + (x2 - x) / dl * lane(i)];
    };
    for (const q of ps) {
      if (disp[q.i] === undefined) continue;
      const f = disp[q.i] / lap, col = PILOT_COLORS[q.i % 4];
      g.strokeStyle = col; g.globalAlpha = 0.45; g.lineWidth = 4; g.beginPath();   // a trail: the last ~3 s
      for (let k = 0; k <= 12; k++) { const [x, y] = at3(f - (3 / lap) * (1 - k / 12), q.i); k ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke(); g.globalAlpha = 1;
      const [x, y] = at3(f, q.i);
      g.beginPath(); g.arc(x, y, 9, 0, 2 * Math.PI);
      g.fillStyle = col; g.fill();
      if (q.pos === 1 && np > 1) { g.strokeStyle = "#fff"; g.lineWidth = 2; g.stroke(); }
      g.fillStyle = "#101216"; g.fillText(String(q.lap), x, y + 0.5);              // its lap, in the dot
      const lx = x + (x - cx) / (Math.hypot(x - cx, y - cy) || 1) * (26 + (q.i % 2) * 14), ly = y + (y - cy) / (Math.hypot(x - cx, y - cy) || 1) * (22 + (q.i % 2) * 12);
      g.fillStyle = q.done || q.down ? "#59616b" : "#e6e9ee";
      g.fillText(`${np > 1 && q.pos ? q.pos + " " : ""}${q.name}${q.down ? " ✕" : q.done ? " ✓" : ""}`, lx, ly);
    }
    g.textAlign = "left"; g.fillStyle = "#9aa3ad"; g.font = "11px -apple-system, sans-serif";
    g.fillText(lead?.cum >= 0 ? `Lap ${lead.lap}${np > 1 ? ` · leader ${lead.name}` : ""}` : "Waiting for the start (G1)", 10, 14);
  };
  const ro = new ResizeObserver(() => w.draw());
  ro.observe(c);
  w.cleanup = () => ro.disconnect();
  w.draw();
}

// ── the race in 3D: the track's layout with each gate modelled by its type and the pilots flying it, where the
// race map has them; an overview to turn and zoom, or a camera chasing a pilot or on board ──
// three.js (vendor/three.min.js, MIT) is loaded when the window opens: no internet needed at the field
let threeP = null;
const loadThree = () => threeP || (threeP = new Promise((ok, no) => {
  const el = document.createElement("script");
  el.src = "vendor/three.min.js?v=r128";
  el.onload = () => ok(window.THREE);
  el.onerror = () => { threeP = null; no(new Error("three.js didn't load (vendor/three.min.js)")); };
  document.head.append(el);
}));
const M_PER_S = 8, FLY_Y = 1.6, LANE_M = 0.6;    // a second of flight on the layout ≈ 8 m; the gates' opening at 1.6 m

// a label that faces the camera, the same size on screen however far (h: a share of the view's height)
function textSprite(T, h = 0.045) {
  const c = document.createElement("canvas"), tex = new T.CanvasTexture(c);
  const sp = new T.Sprite(new T.SpriteMaterial({ map: tex, sizeAttenuation: false, depthTest: false, transparent: true }));
  sp.renderOrder = 10;
  sp.set = (text, col = "#e6e9ee") => {
    if (sp.text === text + col) return;
    sp.text = text + col;
    const g = c.getContext("2d"), f = "bold 40px -apple-system, sans-serif";
    g.font = f;
    c.width = Math.ceil(g.measureText(text).width) + 24; c.height = 56;
    g.font = f; g.textBaseline = "middle";
    g.fillStyle = "rgba(10,12,16,.6)"; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = col; g.fillText(text, 12, 30);
    tex.needsUpdate = true;
    sp.scale.set(h * c.width / c.height, h, 1);
  };
  return sp;
}

// a gate by its learned type: square (a frame on two posts), arch (half a ring), else a square
function gateMesh(T, type, col) {
  const g = new T.Group(), mat = new T.MeshLambertMaterial({ color: col });
  const box = (w, h, d, x, y) => { const m = new T.Mesh(new T.BoxGeometry(w, h, d), mat); m.position.set(x, y, 0); g.add(m); };
  if (type === "arch") {
    const m = new T.Mesh(new T.TorusGeometry(1.6, 0.16, 8, 28, Math.PI), mat);
    g.add(m);
  } else {
    box(0.2, 2.8, 0.2, -1.3, 1.4); box(0.2, 2.8, 0.2, 1.3, 1.4);
    box(2.8, 0.2, 0.2, 0, 2.7); box(2.8, 0.2, 0.2, 0, 0.6);
  }
  g.mat = mat;
  return g;
}

function initRace3D(w) {
  w.body.style.padding = "0";
  w.body.innerHTML = `<div class="vidwrap"><div class="vidctl">
      <label>Camera <select data-x="cam"></select></label>
      <span class="hint" data-x="tip">drag: turn · wheel: zoom</span></div>
    <div class="vidbox" style="background:#151a20"><div class="hud3d"></div></div></div>`;
  const box = $(".vidbox", w.body), hud = $(".hud3d", w.body), sel = $("[data-x=cam]", w.body);
  let alive = true, raf = 0, T = null, R3 = null, G = null, camKey = "";
  // the camera picked, per kind: every pilot's (the overview) and one pilot's (chasing it)
  const camKind = () => (S.combo ? ".combo" : ".pilot");
  const camStore = (k) => store.get("raceview.cam3d" + k, k === ".pilot" ? "chase:0" : "orbit");
  const st = { disp: {} }, view = { az: 0, el: 1.0, dist: null, kind: camKind(), cam: camStore(camKind()) };
  const look = { p: null, t: null };
  sel.onchange = () => { view.cam = sel.value; store.set("raceview.cam3d" + view.kind, view.cam); look.p = null; };
  w.cleanup = () => { alive = false; cancelAnimationFrame(raf); if (R3) { R3.dispose(); R3.forceContextLoss?.(); } };

  // the scene of one track: ground, the path, the gates; rebuilt when the track or its layout changes
  const build = (L) => {
    const { n, lap, at } = raceTrack(L);
    let P = L.layout?.gates?.length === n ? L.layout.gates.map((q) => [q.x * M_PER_S, q.y * M_PER_S])
      : at.map((a) => [Math.cos(2 * Math.PI * a - Math.PI / 2), Math.sin(2 * Math.PI * a - Math.PI / 2)].map((v) => -v * lap * M_PER_S / (2 * Math.PI)));
    const mx = P.reduce((a, p) => a + p[0], 0) / n, mz = P.reduce((a, p) => a + p[1], 0) / n;
    P = P.map(([x, z]) => [x - mx, z - mz]);
    const ext = Math.max(20, ...P.map(([x, z]) => Math.hypot(x, z)));
    const scene = new T.Scene();
    scene.background = new T.Color(0x151a20);
    scene.fog = new T.Fog(0x151a20, ext * 2.5, ext * 6);
    scene.add(new T.HemisphereLight(0xdfe8ff, 0x1c2a1c, 0.95));
    const sun = new T.DirectionalLight(0xffffff, 0.55); sun.position.set(ext, ext * 2, ext * 0.6); scene.add(sun);
    const ground = new T.Mesh(new T.PlaneGeometry(ext * 8, ext * 8), new T.MeshLambertMaterial({ color: 0x1d2a1f }));
    ground.rotation.x = -Math.PI / 2; scene.add(ground);
    const grid = new T.GridHelper(ext * 8, Math.round(ext * 8 / 10), 0x2a3a2c, 0x243226); grid.position.y = 0.01; scene.add(grid);
    const path = (f) => splineAt(P, at, f);
    const dir = (f) => { const a = path(f), b = path(f + 0.001), l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1; return [(b[0] - a[0]) / l, (b[1] - a[1]) / l]; };
    const K = 480, pos = [], idx = [];                // the path: a ribbon on the ground
    for (let k = 0; k <= K; k++) {
      const [x, z] = path(k / K), [tx, tz] = dir(k / K);
      pos.push(x - tz * 2.2, 0.03, z + tx * 2.2, x + tz * 2.2, 0.03, z - tx * 2.2);
      if (k < K) idx.push(2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2);
    }
    const rg = new T.BufferGeometry();
    rg.setAttribute("position", new T.Float32BufferAttribute(pos, 3)); rg.setIndex(idx);
    scene.add(new T.Mesh(rg, new T.MeshBasicMaterial({ color: 0x343b45, side: T.DoubleSide })));
    const gates = [];
    for (let i = 0; i < n; i++) {
      const type = L.layout?.gates?.[i]?.type || "square", [x, z] = path(at[i]), [tx, tz] = dir(at[i]);
      const gm = gateMesh(T, type, i === 0 ? 0xffb020 : 0xcfd6de);
      gm.position.set(x, 0, z); gm.rotation.y = Math.atan2(tx, tz);
      scene.add(gm);
      const lb = textSprite(T, 0.03); lb.set(`G${i + 1}`, i === 0 ? "#ffb020" : "#9aa3ad"); lb.position.set(x, 3.4, z); scene.add(lb);
      gates.push(gm);
    }
    return { key: G?.key, scene, path, dir, ext, gates, lap, at, n, pilots: {} };
  };

  // a pilot: a glowing drone, a trail of its last ~3 s and its name
  const pilotObj = (i) => {
    const col = new T.Color(PILOT_COLORS[i % 4]), g = new T.Group();
    const body = new T.Mesh(new T.ConeGeometry(0.6, 1.5, 14), new T.MeshLambertMaterial({ color: col, emissive: col, emissiveIntensity: 0.45 }));
    body.rotation.x = Math.PI / 2; g.add(body);
    const gc = document.createElement("canvas"); gc.width = gc.height = 64;
    const gg = gc.getContext("2d"), grd = gg.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, "rgba(255,255,255,1)"); grd.addColorStop(1, "rgba(255,255,255,0)"); gg.fillStyle = grd; gg.fillRect(0, 0, 64, 64);
    const glow = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(gc), color: col, blending: T.AdditiveBlending, transparent: true, depthWrite: false }));
    glow.scale.set(4.5, 4.5, 1); g.add(glow);
    const trail = new T.Line(new T.BufferGeometry().setAttribute("position", new T.Float32BufferAttribute(new Array(25 * 3).fill(0), 3)),
      new T.LineBasicMaterial({ color: col, transparent: true, opacity: 0.7 }));
    const label = textSprite(T);
    G.scene.add(g, trail, label);
    return { g, trail, label, body };
  };

  const at3 = (f, i, np) => {                         // pilot i's lane, beside the path, at flying height
    const [x, z] = G.path(f), [tx, tz] = G.dir(f), o = np > 1 ? (i - (np - 1) / 2) * LANE_M : 0;
    return new T.Vector3(x - tz * o, FLY_Y, z + tx * o);
  };

  const camOptions = (ps) => {
    const kind = camKind(), key = kind + ps.map((q) => q.name).join("|");
    if (key === camKey) return;
    camKey = key;
    if (kind !== view.kind) { view.kind = kind; view.cam = camStore(kind); look.p = null; }
    const o = [["orbit", "Overview"]];
    if (ps.length > 1) o.push(["lead", "Chase the leader"]);
    ps.forEach((q) => o.push([`chase:${q.i}`, `Chase ${q.name}`], [`fpv:${q.i}`, `On board ${q.name}`]));
    if (!o.some(([v]) => v === view.cam)) view.cam = "orbit";
    sel.innerHTML = o.map(([v, t]) => `<option value="${v}" ${v === view.cam ? "selected" : ""}>${esc(t)}</option>`).join("");
  };

  const orbit = () => {
    look.p = null;
    view.dist = view.dist || G.ext * 1.9;
    const e = view.el, a = view.az;
    cam.fov = 50;
    cam.position.set(view.dist * Math.cos(e) * Math.sin(a), view.dist * Math.sin(e), view.dist * Math.cos(e) * Math.cos(a));
    cam.lookAt(0, 0, 0);
  };

  let cam = null, lastMs = 0;
  const frame = (ms) => {
    if (!alive) return;
    raf = requestAnimationFrame(frame);
    if (document.hidden) return;
    const dt = Math.min(0.1, (ms - (lastMs || ms)) / 1000); lastMs = ms;
    const cw = box.clientWidth, ch = box.clientHeight;
    if (!cw || !ch) return;
    const L = raceSrc() ? comboRace() : null;
    if (!L || !L.gates) {
      hud.innerHTML = `<div class="hint">${S.H ? "The track isn't learned yet: no 3D view." : "Pick a heat."}</div>`;
      R3.domElement.style.visibility = "hidden";
      return;
    }
    R3.domElement.style.visibility = "";
    const key = JSON.stringify([L.track, L.gates, L.legs, (L.layout?.gates || []).map((q) => [q.x, q.y, q.type])]);
    if (!G || G.key !== key) { G = build(L); G.key = key; st.disp = {}; look.p = null; view.dist = null; camKey = null; }
    if (R3.domElement.width !== Math.round(cw * devicePixelRatio)) R3.setSize(cw, ch, false);
    cam.aspect = cw / ch;
    const now = S.t, ps = raceState(L), np = ps.length, lead = ps.order?.[0];
    camOptions(ps);
    const disp = raceProgress(st, ps, raceTrack(L), now);
    for (const gm of G.gates) gm.mat.emissive.setHex(0);              // a pass: its gate lights up in the pilot's colour
    for (const x of L.passes) {
      if (x.gate_id > 0 && x.pilot != null && now - x.t >= 0 && now - x.t < 0.6 && G.gates[x.gate_id - 1])
        G.gates[x.gate_id - 1].mat.emissive.set(PILOT_COLORS[x.pilot % 4]);
    }
    const where = {};
    for (const q of ps) {
      const o = G.pilots[q.i] || (G.pilots[q.i] = pilotObj(q.i)), on = disp[q.i] !== undefined;
      o.g.visible = o.trail.visible = o.label.visible = on;
      if (!on) continue;
      const f = disp[q.i] / G.lap, p = at3(f, q.i, np), ahead = at3(f + 0.004, q.i, np);
      o.g.position.copy(p); o.g.lookAt(ahead);
      o.g.position.y = q.down ? 0.4 : p.y + Math.sin(ms / 300 + q.i) * 0.08;      // down: on the ground
      const tp = o.trail.geometry.attributes.position;
      for (let k = 0; k < 25; k++) { const v = at3(f - (3 / G.lap) * (1 - k / 24), q.i, np); tp.setXYZ(k, v.x, v.y, v.z); }
      tp.needsUpdate = true;
      o.label.position.set(p.x, o.g.position.y + 1.5 + (q.i % 2) * 1.6, p.z);
      o.label.set(`${np > 1 && q.pos ? q.pos + " " : ""}${q.name} · L${q.lap}${q.down ? " ✕" : q.done ? " ✓" : ""}`,
        q.down || q.done ? "#8a929c" : PILOT_COLORS[q.i % 4]);
      o.body.material.emissiveIntensity = q.down ? 0 : q.pos === 1 && np > 1 ? 0.9 : 0.45;
      o.body.material.color.set(q.down ? "#59616b" : PILOT_COLORS[q.i % 4]);
      o.g.children[1].visible = o.trail.visible = !q.down;
      where[q.i] = { p, f };
    }
    // the camera: an overview to turn and zoom, or behind / on board a pilot (the leader: whoever leads now)
    const m = view.cam.match(/^(chase|fpv):(\d+)$/), who = view.cam === "lead" ? lead?.i : m ? +m[2] : null;
    const tgt = who != null ? where[who] : null;
    $("[data-x=tip]", w.body).textContent = tgt ? "" : "drag: turn · wheel: zoom";
    const fpv = !!tgt && m && m[1] === "fpv";
    if (fpv) G.pilots[who].g.visible = G.pilots[who].label.visible = G.pilots[who].trail.visible = false;     // on board: not its own drone
    if (tgt) {
      const fa = (fpv ? 1.2 : 0.6) / G.lap;
      const [tx, tz] = G.dir(tgt.f), ahead = at3(tgt.f + fa * 2, who, np);
      const goal = fpv ? tgt.p.clone().add(new T.Vector3(0, 0.25, 0)) : tgt.p.clone().add(new T.Vector3(-tx * 7, 3.2, -tz * 7));
      const aim = fpv ? ahead : tgt.p.clone().add(new T.Vector3(tx * 4, 0, tz * 4));
      const k = 1 - Math.exp(-dt * (fpv ? 12 : 4));
      if (!look.p) { look.p = goal.clone(); look.t = aim.clone(); }
      look.p.lerp(goal, k); look.t.lerp(aim, k);
      cam.fov = fpv ? 80 : 60;
      cam.position.copy(look.p); cam.lookAt(look.t);
    } else orbit();
    for (const o of Object.values(G.pilots))                          // the glow: smaller up close
      o.g.children[1].scale.setScalar(Math.min(4.5, Math.max(1.2, cam.position.distanceTo(o.g.position) / 10)));
    cam.near = 0.1; cam.far = G.ext * 12; cam.updateProjectionMatrix();
    R3.render(G.scene, cam);
    const lines = (ps.order || []).filter((q) => q.cum >= 0).map((q) => `<div><span class="dot" style="background:${PILOT_COLORS[q.i % 4]}"></span>`
      + `${np > 1 ? q.pos + " " : ""}<b>${esc(q.name)}</b> L${q.lap} G${q.g}${q.down ? " · down" : q.finished ? " · finished"
        : q.pos > 1 ? q.behind > 0 ? ` · +${q.behind}L` : q.gap != null ? ` · +${fmt(q.gap, 1)}s` : "" : ""}</div>`).join("");
    const head = lead?.cum >= 0 ? `<div class="lap">LAP ${lead.lap}</div>` : `<div class="hint">Waiting for the start (G1)</div>`;
    if (hud.dataset.h !== head + lines) { hud.dataset.h = head + lines; hud.innerHTML = head + lines; }
  };

  loadThree().then((three) => {
    if (!alive) return;
    T = three;
    try { R3 = new T.WebGLRenderer({ antialias: true }); }
    catch (e) { hud.innerHTML = `<div class="hint">No 3D here: WebGL isn't available in this browser.</div>`; return; }
    R3.setPixelRatio(devicePixelRatio || 1);
    Object.assign(R3.domElement.style, { position: "absolute", inset: "0", width: "100%", height: "100%", touchAction: "none" });
    box.prepend(R3.domElement);
    cam = new T.PerspectiveCamera(50, 1, 0.1, 2000);
    let drag = null;                                  // the overview: drag to turn, wheel / pinch to zoom
    const c = R3.domElement;
    c.onpointerdown = (e) => { drag = { x: e.clientX, y: e.clientY }; c.setPointerCapture(e.pointerId); };
    c.onpointermove = (e) => {
      if (!drag) return;
      if (view.cam !== "orbit") { view.cam = sel.value = "orbit"; store.set("raceview.cam3d" + view.kind, "orbit"); }
      view.az -= (e.clientX - drag.x) * 0.006;
      view.el = Math.min(1.5, Math.max(0.08, view.el + (e.clientY - drag.y) * 0.006));
      drag = { x: e.clientX, y: e.clientY };
    };
    c.onpointerup = c.onpointercancel = () => { drag = null; };
    c.onwheel = (e) => {
      e.preventDefault();
      if (G) view.dist = Math.min(G.ext * 8, Math.max(5, (view.dist || G.ext * 1.9) * Math.exp(e.deltaY * 0.0012)));
    };
    raf = requestAnimationFrame(frame);
  }).catch((e) => { hud.innerHTML = `<div class="hint err">${esc(e.message)}</div>`; });
}

// ── every pilot of a heat at once: their runs, on the same clock (the heat's video), raced against each other at
// the playhead — the standings, race map and lap times ──

// one pilot's race from its run: its gate passes, laps, and where each lap starts (lap k at its t0; the unfinished
// last lap at the last t1; no lap closed: at the first G1)
function racePilot(r, i, name) {
  const passes = (r.passes || []).filter((x) => x.gate_id >= 1);
  const laps = r.laps || [], last = laps[laps.length - 1], starts = [];
  const end = r.end !== undefined ? r.end : passes.length ? passes[passes.length - 1].t : null;
  for (const L of laps) starts.push({ lap: L.lap, t: L.t0 });
  if (last && end > last.t1 + 1e-3) starts.push({ lap: last.lap + 1, t: last.t1 });
  if (!laps.length) { const g1 = passes.find((x) => x.gate_id === 1); if (g1) starts.push({ lap: 1, t: g1.t }); }
  return { i, name, passes, laps, end, starts };
}

// the race at time t, in raceState's shape: what had happened by t — of every pilot, or of the one pilot open
let comboMemo = null;
const raceSrc = () => S.combo || S.solo;
function comboRace(t = S.t) {
  const C = raceSrc();
  if (comboMemo && comboMemo.C === C && comboMemo.t === t) return comboMemo.R;
  const R = {
    now: t, gates: C.n_gates, legs: C.legs, layout: C.layout, track: C.track,
    pilots: C.pilots.map((p) => ({ name: p.name, end: p.end, fin: raceFinished(C, p), flag: C === S.combo && pilotFlag(p) })),
    passes: C.pilots.flatMap((p) => (p.passes || []).filter((x) => x.t <= t).map((x) => ({ ...x, pilot: p.i }))),
    starts: C.pilots.flatMap((p) => p.starts.filter((x) => x.t <= t).map((x) => ({ ...x, pilot: p.i }))),
    laps: C.pilots.flatMap((p) => (p.laps || []).filter((L) => L.t1 <= t).map((L) => ({ ...L, pilot: p.i, t: L.t1 }))),
  };
  comboMemo = { C, t, R };
  return R;
}

// a pilot whose run ended with a lap (its last pass closed it) of as many laps as anyone's: finished, not down
function raceFinished(C, p) {
  const most = Math.max(...C.pilots.map((q) => q.laps.length)), last = p.laps[p.laps.length - 1];
  return !!last && p.laps.length === most && p.end != null && p.end - last.t1 < 5;
}

// while the video plays: sounds and call-outs for what happens between two frames (not on a seek)
let racePlaces = newPlaces();
function raceSounds(t0, t1, playing) {
  if (!raceSrc()) return;
  const C = raceSrc(), multi = !!S.combo;
  if (!playing || !(t1 > t0) || t1 - t0 > 1.5) {          // paused or jumped: the places as of now, nothing said
    racePlaces = newPlaces();
    if (multi) placeCalls(raceState(comboRace(t1)), t1, racePlaces, comboRace(t1), true);
    return;
  }
  const inside = (t) => t > t0 && t <= t1;
  for (const p of C.pilots) {
    if (!multi) for (const x of p.passes) if (inside(x.t)) sfx(x.gate_id === 1 ? "start" : "gate");
    if (p.starts[0] && p.starts[0].lap === 1 && inside(p.starts[0].t)
        && !C.pilots.some((q) => q !== p && q.starts[0] && q.starts[0].t <= p.starts[0].t)) say(multi ? "Go!" : p.name ? `${p.name}, go!` : "Go!");
    for (const L of p.laps) {
      if (!inside(L.t1)) continue;
      if (multi) sfx("start");
      const best = Math.min(...p.laps.filter((x) => x.t1 <= L.t1).map((x) => x.dt));
      say(`${p.name ? p.name + ", " : ""}lap ${L.lap}, ${L.dt.toFixed(1)}${L.dt <= best && L.lap > 1 ? ", best lap" : ""}`);
    }
  }
  if (!multi) return;
  const R = comboRace(t1);
  for (const text of placeCalls(raceState(R), t1, racePlaces, R)) say(text);
}

function renderRaceStand(w) {
  const C = S.combo;
  if (!C) {
    w.body.innerHTML = `<div class="empty">${S.H ? `${esc(S.run?.name || "")}'s run is open: pick ⊞ All pilots in the top bar to race them against each other.`
      : "Pick a heat: its pilots race each other here as it plays."}</div>`;
    return;
  }
  const R = comboRace();
  let h = renderStandings(R);
  // every lap of every pilot so far, side by side
  const laps = R.laps, maxLap = Math.max(0, ...laps.map((l) => l.lap));
  if (maxLap) {
    const best = (i) => Math.min(...laps.filter((l) => l.pilot === i).map((l) => l.dt));
    const fastest = Math.min(...laps.map((l) => l.dt));
    h += `<h4>Laps</h4><table><tr><th>Lap</th>${C.pilots.map((p) => `<th><span class="dot" style="background:${PILOT_COLORS[p.i % 4]}"></span>${esc(p.name)}</th>`).join("")}</tr>`;
    for (let n = 1; n <= maxLap; n++) {
      h += `<tr><td>${n}</td>${C.pilots.map((p) => {
        const l = laps.find((x) => x.pilot === p.i && x.lap === n);
        return `<td class="${l && l.dt === fastest ? "cur" : ""}">${l ? fmt(l.dt) + "s" + (l.dt === best(p.i) ? " ★" : "") : ""}</td>`;
      }).join("")}</tr>`;
    }
    h += `</table><div class="hint">★ the pilot's best · highlighted: the fastest lap of the race · as of ${fmt(S.t - raceT0(), 1)} s</div>`;
  }
  w.body.innerHTML = h;
  w.body.querySelectorAll("tr[data-pilot]").forEach((tr) => tr.onclick = () => openView(+tr.dataset.pilot));
}

// over the heat's video: each pilot's picture framed in its colour, with its place, lap, gate and lap clock; a pass flashes
function drawComboOverlay(g, G) {
  const C = S.combo, R = comboRace(), ps = raceState(R);
  for (const p of C.pilots) {
    if (!p.crop) continue;
    const q = ps[p.i], col = PILOT_COLORS[p.i % 4], cr = p.crop;
    const x = G.ox + cr[0] * G.vw * G.s, y = G.oy + cr[1] * G.vh * G.s, ww = cr[2] * G.vw * G.s, hh = cr[3] * G.vh * G.s;
    g.strokeStyle = col; g.lineWidth = q.pos === 1 ? 4 : 2;
    g.strokeRect(x + 1, y + 1, ww - 2, hh - 2);
    const lapT = q.cum >= 0 ? S.t - R.starts.filter((s) => s.pilot === p.i).pop().t : null;
    const line = q.cum < 0 ? "waiting for G1"
      : q.down || q.done ? `${q.down ? "down" : "finished"} · ${q.laps.length} lap${q.laps.length === 1 ? "" : "s"}${q.best != null ? ` · best ${fmt(q.best)}s` : ""}`
      : `L${q.lap} · G${q.g} · ${fmt(lapT, 1)}s${q.pos > 1 ? q.behind > 0 ? ` · +${q.behind}L` : q.gap != null ? ` · +${fmt(q.gap, 1)}s` : "" : ""}`;
    g.font = "bold 16px -apple-system, sans-serif";
    const head = `${q.pos ? q.pos + " · " : ""}${flagName(p)}`;
    const tw = Math.max(g.measureText(head).width, (g.font = "13px -apple-system, sans-serif", g.measureText(line).width));
    g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(x + 6, y + 6, tw + 14, 40);
    g.fillStyle = col; g.font = "bold 16px -apple-system, sans-serif"; g.fillText(head, x + 13, y + 9);
    g.fillStyle = "#e6e9ee"; g.font = "13px -apple-system, sans-serif"; g.fillText(line, x + 13, y + 29);
    const lp = q.laps[q.laps.length - 1];                  // a lap just closed: its time, big
    g.font = "bold 22px -apple-system, sans-serif";
    if (lp && S.t - lp.t < 3) {
      const txt = `Lap ${lp.lap} ${fmt(lp.dt)}s${lp.dt === q.best && q.laps.length > 1 ? " ★" : ""}`;
      g.fillStyle = "rgba(0,0,0,.6)"; g.fillRect(x + ww / 2 - g.measureText(txt).width / 2 - 8, y + hh - 44, g.measureText(txt).width + 16, 32);
      g.fillStyle = "#ffb020"; g.textAlign = "center"; g.fillText(txt, x + ww / 2, y + hh - 40); g.textAlign = "left";
    }
    const rec = (p.passes || []).filter((e) => e.t <= S.t && S.t - e.t < 0.6);
    rec.forEach((e, k) => { g.fillStyle = e.recovered ? "#ff9f43" : "#4cd97b"; g.fillText(`G${e.gate_id} ✓`, x + 13, y + 54 + k * 26); });
  }
}

// a pilot's missed gates, of every lap: [{lap, gate, t}]
function pilotMissed(p) {
  return (p.laps || []).flatMap((L) => lapMissed(L, S.H?.n_gates).map((m) => ({ lap: L.lap, gate: m.gate, t: m.t })));
}

// the timeline with every pilot: a lane per pilot — its laps and gate passes, each pilot's progress at the playhead:
// what's flown so far bright, what's to come dim, the running lap filled to the playhead (lap, its clock, last gate)
function drawComboLanes(g, cw, ch, X) {
  const C = S.combo, np = C.pilots.length;
  if (!np) return;
  const R = comboRace(), ps = raceState(R), now = S.t, xNow = X(now);
  const top = 2, lh = (ch - 18 - top) / np;
  C.pilots.forEach((p, k) => {
    const y = top + k * lh, col = PILOT_COLORS[p.i % 4], q = ps[p.i];
    const dim = (t) => { g.globalAlpha = t > now ? 0.3 : 1; };
    g.fillStyle = k % 2 ? "rgba(255,255,255,.02)" : "rgba(255,255,255,.045)"; g.fillRect(0, y, cw, lh);
    for (const L of p.laps || []) {
      const past = L.t1 <= now;
      g.fillStyle = col; g.globalAlpha = (L.lap % 2 ? 0.16 : 0.08) * (past ? 1.4 : 0.5); g.fillRect(X(L.t0), y, X(L.t1) - X(L.t0), lh);
      dim(L.t0); g.fillRect(X(L.t0), y, 2, lh); g.globalAlpha = 1;
      if (past && lh >= 14 && X(L.t1) - X(L.t0) > 46) { g.fillStyle = "#e6e9ee"; g.fillText(`L${L.lap} ${fmt(L.dt)}s`, X(L.t0) + 4, y + 2); }
    }
    g.fillStyle = col;
    for (const x of p.passes || []) {
      dim(x.t);
      g.fillRect(X(x.t) - 0.5, y + lh * (x.gate_id === 1 ? 0.25 : 0.5), 1.5, lh * (x.gate_id === 1 ? 0.7 : 0.45));
    }
    if (p.reviewed !== "complete") for (const m of pilotMissed(p)) {     // a missed gate to check
      dim(m.t);
      g.fillStyle = "#ff5c5c";
      g.beginPath(); g.moveTo(X(m.t), y + lh * 0.45); g.lineTo(X(m.t) - 4, y + lh * 0.45 - 6); g.lineTo(X(m.t) + 4, y + lh * 0.45 - 6); g.fill();
      g.fillRect(X(m.t) - 0.5, y + lh * 0.45, 1.5, lh * 0.5);
      if (lh >= 22) g.fillText(`G${m.gate}`, X(m.t) + 3, y + lh * 0.45 - 4);
    }
    g.globalAlpha = 1;
    const st = R.starts.filter((s) => s.pilot === p.i).pop();      // the lap running at the playhead
    const running = st && q.cum >= 0 && !q.done && !q.finished;
    if (running) {
      g.fillStyle = col; g.globalAlpha = 0.28; g.fillRect(X(st.t), y, xNow - X(st.t), lh); g.globalAlpha = 1;
      g.fillRect(X(st.t), y + lh - 3, xNow - X(st.t), 3);             // the lap so far, as a bar
    }
    const note = q.cum < 0 ? "waiting for G1" : q.finished ? `finished · ${q.laps.length} laps`
      : q.down ? "down" : running ? `L${q.lap} ${fmt(now - st.t, 1)}s · G${q.g}` : "";
    if (lh >= 12) {
      g.beginPath(); g.moveTo(xNow + 1, y + lh / 2); g.lineTo(xNow + 7, y + lh / 2 - 4); g.lineTo(xNow + 7, y + lh / 2 + 4);
      g.fillStyle = col; g.fill();                                       // the pilot at the playhead
      if (note) {
        const txt = `${q.pos ? q.pos + " · " : ""}${flagName(p)} · ${note}`, tw = g.measureText(txt).width;
        const tx = xNow + 10 + tw > cw - 4 ? xNow - 10 - tw : xNow + 10;
        g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(tx - 3, y + lh / 2 - 7, tw + 6, 14);
        g.fillStyle = "#e6e9ee"; g.textBaseline = "middle"; g.fillText(txt, tx, y + lh / 2); g.textBaseline = "top";
      }
    }
    if ((!(p.laps || []).length || X(p.laps[0].t0) > 50) && Math.abs(xNow - 4) > 120) {
      g.fillStyle = col; g.fillText(flagName(p), 4, y + 2);
    }
  });
}

// ─────────────────────────────────────────────────────────────
// Keyboard, start-up
// ─────────────────────────────────────────────────────────────

addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const k = e.key;
  if (k === " ") { e.preventDefault(); control("play"); }
  else if (k === "ArrowLeft") { e.preventDefault(); control(e.shiftKey ? "-10s" : "-1s"); }
  else if (k === "ArrowRight") { e.preventDefault(); control(e.shiftKey ? "+10s" : "+1s"); }
  else if (k === "ArrowDown") { e.preventDefault(); control("-f"); }
  else if (k === "ArrowUp") { e.preventDefault(); control("+f"); }
  else if (k === "Escape" && S.H?.combo && S.pilot !== null) openView(null);
});

$("#heatSel").onchange = (e) => selectHeat(+e.target.value || null);
$("#pilotSel").onchange = (e) => openView(e.target.value === "" ? null : +e.target.value);
$("#winBtn").onclick = (e) => { e.stopPropagation(); $("#winMenu").classList.toggle("open"); };
document.addEventListener("click", (e) => { if (!$("#winMenu").contains(e.target)) $("#winMenu").classList.remove("open"); });

(async function start() {
  try {
    if (!EV) throw new Error("open the race view from the dashboard (an event's Race view tab)");
    $("#pilotSel").parentElement.hidden = true;
    applyLayout();
    await loadHeats();
    const n = +QS.get("heat") || null, k = QS.get("pilot");
    if (n && S.heats.some((h) => h.heat === n)) await selectHeat(n, k === null || k === "" ? null : +k);
    else status(S.heats.length ? "Pick a heat" : "No analysed heat yet: heats show here once their runs are analysed and labelled.");
    requestAnimationFrame(loop);
    setInterval(() => { if (!document.hidden) loadHeats().catch(() => {}); }, 30000);    // heats analysed meanwhile
  } catch (e) { status(e.message, true); }
})();
