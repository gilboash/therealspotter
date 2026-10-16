// FPV Spotter web UI — floating windows over one shared state (track, video, analysis, playhead).
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const store = {
  get(k, def) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : def; } catch { return def; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

async function api(url, opts = {}) {
  if (opts.json !== undefined) {
    opts = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(opts.json) };
  }
  const r = await fetch(url, opts);
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch { /* not json */ }
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
  mode: store.get("spotter.mode", "race"),
  tracks: [], track: null, video: null,
  run: null, frames: null, gates: null,
  marks: [], nextGate: 1,
  jobs: [], seenDone: new Set(),
  t: 0,
  ov: store.get("spotter.overlay", { tracks: true, dets: false, labels: true, hud: true }),
  videoEl: null,
};

const curTrack = () => S.tracks.find((t) => t.name === S.track);

// ─────────────────────────────────────────────────────────────
// Window manager
// ─────────────────────────────────────────────────────────────

const PANELS = {
  video:    { title: "Video", init: initVideo },
  timeline: { title: "Timeline", init: initTimeline },
  stats:    { title: "Race stats — laps", render: renderStats, live: true },
  splits:   { title: "Gate splits — time between gates", render: renderSplits, live: true },
  passdet:  { title: "Pass detector · tracks", render: renderPassdet, live: true },
  dets:     { title: "Detections (YOLO)", render: renderDets, live: true },
  passes:   { title: "Passes", render: renderPasses, live: true },
  gates:    { title: "Gates (learned)", render: renderGates },
  library:  { title: "Videos", render: renderLibrary },
  learn:    { title: "Learn track — auto", render: renderLearn },
  marks:    { title: "Manual marks", render: renderMarks, live: true },
  jobs:     { title: "Jobs", render: renderJobs },
};

// default windows per mode, as fractions of the desk: [x, y, w, h]
const DEFAULTS = {
  race: {
    video: [0, 0, 0.56, 0.66], timeline: [0, 0.66, 0.56, 0.16], passdet: [0, 0.82, 0.56, 0.18],
    stats: [0.56, 0, 0.44, 0.42], splits: [0.56, 0.42, 0.44, 0.40], jobs: [0.56, 0.82, 0.44, 0.18],
  },
  learn: {
    library: [0, 0, 0.42, 0.45], learn: [0, 0.45, 0.42, 0.55], gates: [0.42, 0, 0.58, 0.62], jobs: [0.42, 0.62, 0.58, 0.38],
  },
  manual: {
    video: [0, 0, 0.6, 0.68], timeline: [0, 0.68, 0.6, 0.16], passes: [0, 0.84, 0.6, 0.16],
    marks: [0.6, 0, 0.4, 0.5], gates: [0.6, 0.5, 0.4, 0.5],
  },
};

const wins = {};          // id → {el, body, panel}
let zTop = 10;

function layoutKey() { return `spotter.layout.${S.mode}`; }

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
  for (const id of Object.keys(wins)) closeWin(id, false);
  const saved = store.get(layoutKey(), null);
  const d = deskRect();
  if (saved) {
    for (const [id, g] of Object.entries(saved).sort((a, b) => a[1].z - b[1].z)) if (PANELS[id]) openWin(id, g);
  } else {
    for (const [id, f] of Object.entries(DEFAULTS[S.mode])) {
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
    + `<hr><button id="resetLayout">Reset this mode's layout</button>`;
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
}

// ─────────────────────────────────────────────────────────────
// Data loading
// ─────────────────────────────────────────────────────────────

async function loadTracks(keep = true) {
  const d = await api("/api/tracks");
  S.tracks = d.tracks;
  if (!keep || !curTrack()) S.track = store.get("spotter.track", null);
  if (!curTrack()) S.track = S.tracks[0]?.name || null;
  fillTrackSel();
  fillVideoSel();
  if (!S.tracks.length) status(`No tracks yet in ${d.data_root} — click ＋ to create one.`);
}

function fillTrackSel() {
  $("#trackSel").innerHTML = S.tracks.map((t) =>
    `<option ${t.name === S.track ? "selected" : ""}>${esc(t.name)}</option>`).join("");
}

function videosForMode() {
  const t = curTrack();
  if (!t) return [];
  return t.videos;
}

function fillVideoSel() {
  const vids = videosForMode();
  const cur = S.video?.name;
  $("#videoSel").innerHTML = `<option value="">— pick a video —</option>` + vids.map((v) =>
    `<option value="${esc(v.name)}" ${v.name === cur ? "selected" : ""}>${esc(v.name)} · ${v.role}${v.analysed ? " ✓" : v.learn_analysed ? " (learn run)" : ""}</option>`).join("");
  if (cur) S.video = vids.find((v) => v.name === cur) || null;
  renderActions();
}

async function selectTrack(name) {
  S.track = name;
  store.set("spotter.track", name);
  writeHash();
  S.video = null; S.run = null; S.frames = null; S.marks = [];
  fillVideoSel();
  await loadGates();
  setVideoSource();
  render();
}

async function loadGates() {
  S.gates = S.track ? await api(`/api/tracks/${encodeURIComponent(S.track)}/gates`).catch(() => null) : null;
  render(["gates", "learn", "marks"]);
}

async function selectVideo(name) {
  const t = curTrack();
  S.video = t?.videos.find((v) => v.name === name) || null;
  S.run = null; S.frames = null; S.marks = []; S.nextGate = 1;
  writeHash();
  setVideoSource();
  renderActions();
  if (S.video) {
    status(`${S.video.name}: ${S.video.w}×${S.video.h}, ${S.video.fps} fps, ${S.video.duration}s`);
    await loadRun();
    if (S.mode === "manual") await loadMarks();
  }
  render();
}

async function loadRun() {
  if (!S.video || !(S.video.analysed || S.video.learn_analysed)) { S.run = null; S.frames = null; return; }
  const base = `/api/tracks/${encodeURIComponent(S.track)}/runs/${encodeURIComponent(S.video.stem)}`;
  try {
    S.run = await api(base);
    status(`Loaded results for ${S.video.name} — ${S.run.passes.length} passes, ${S.run.laps.length} laps`
      + (S.run.info?.timing ? ` · ${S.run.info.timing}` : ""));
    S.frames = await api(base + "/frames");
    render();
  } catch (e) { status(e.message, true); }
}

async function loadMarks() {
  if (!S.video) return;
  const d = await api(`/api/tracks/${encodeURIComponent(S.track)}/marks/${encodeURIComponent(S.video.stem)}`);
  S.marks = d.marks || [];
  const last = S.marks[S.marks.length - 1];
  S.nextGate = last ? last.gate + 1 : 1;
  render(["marks", "timeline", "gates"]);
}

let marksTimer = null;
function saveMarksSoon() {
  clearTimeout(marksTimer);
  marksTimer = setTimeout(async () => {
    if (!S.video) return;
    await api(`/api/tracks/${encodeURIComponent(S.track)}/marks/${encodeURIComponent(S.video.stem)}`, { json: { marks: S.marks } });
    const v = curTrack()?.videos.find((x) => x.name === S.video.name);
    if (v) v.has_marks = S.marks.length > 0;
  }, 400);
}

// ─────────────────────────────────────────────────────────────
// Top bar actions
// ─────────────────────────────────────────────────────────────

function renderActions() {
  const a = $("#actions");
  a.innerHTML = "";
  const v = S.video;
  if (!v || S.mode === "learn") return;
  const running = S.jobs.some((j) => j.kind === "analyse" && j.video === v.name && ["running", "queued"].includes(j.status));
  const b = document.createElement("button");
  b.className = v.analysed ? "" : "primary";
  b.textContent = running ? "Analysing…" : v.analysed ? "▶ Re-run analysis" : "▶ Run analysis";
  b.disabled = running;
  b.title = "YOLO + pass detection + gate labelling (saved, loads instantly next time)";
  b.onclick = () => startJob({ kind: "analyse", track: S.track, video: v.name });
  a.appendChild(b);
}

async function startJob(body) {
  try {
    const j = await api("/api/jobs", { json: body });
    S.jobs.unshift(j);
    status(`Started: ${j.title}`);
    if (!wins.jobs) openWin("jobs");
    renderActions();
    render(["jobs", "library", "learn"]);
    pollJobs();
  } catch (e) { status(e.message, true); }
}

let pollTimer = null;
async function pollJobs() {
  clearTimeout(pollTimer);
  try {
    const d = await api("/api/jobs");
    S.jobs = d.jobs;
    for (const j of S.jobs) {
      if (["done", "failed"].includes(j.status) && !S.seenDone.has(j.id)) {
        S.seenDone.add(j.id);
        if (j.status === "failed") { status(`Failed: ${j.title} — see Jobs`, true); continue; }
        status(`Finished: ${j.title}`);
        await loadTracks();
        await loadGates();
        if (S.video && (j.kind === "learn" || j.video === S.video.name)) {
          S.video = curTrack()?.videos.find((v) => v.name === S.video.name) || S.video;
          await loadRun();
        }
      }
    }
    renderActions();
    render(["jobs", "library", "learn"]);
  } catch { /* server restarting */ }
  const busy = S.jobs.some((j) => ["running", "queued"].includes(j.status));
  pollTimer = setTimeout(pollJobs, busy ? 1000 : 5000);
}

// ─────────────────────────────────────────────────────────────
// Frame lookup at the playhead
// ─────────────────────────────────────────────────────────────

function frameAt(t) {
  const F = S.frames?.frames;
  if (!F || !F.length) return null;
  let lo = 0, hi = F.length - 1;
  if (F[0][0] > t) return null;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (F[mid][0] <= t + 1e-4) lo = mid; else hi = mid - 1;
  }
  const fps = S.video?.fps || 30;
  return t - F[lo][0] <= 1.5 / fps ? F[lo] : null;
}

// The laps as both the timeline and the HUD show them: the completed laps, then the unfinished
// one — from the last G1 to the pilot's last gate pass (after that the run is over, no lap is running).
function lapSegments() {
  if (!S.run) return [];
  const laps = S.run.laps || [];
  const segs = laps.map((L, i) => ({ i, n: L.lap, t0: L.t0, t1: L.t1, dt: L.dt, done: true, missed: lapMissed(L), rec: lapRecovered(L) }));
  const g1 = S.run.passes.filter((p) => p.gate_id === 1);
  const start = laps.length ? laps[laps.length - 1].t1 : g1[0]?.t;
  if (start !== undefined) {
    const after = S.run.passes.filter((p) => p.source === "RACE" && p.t > start + 1e-3);
    if (after.length) segs.push({ i: -1, n: laps.length + 1, t0: start, t1: after[after.length - 1].t, dt: null, done: false });
  }
  return segs;
}

// The gates a completed lap skipped (known once the lap closes at G1): [{gate, t, after}], t placed
// between the gates passed before and after it. The pilot cut it or the detector missed it, so a
// spotter has to check the lap.
function lapMissed(L) {
  const n = S.run?.n_gates || 0;
  if (n < 2 || !L.splits) return [];
  const seen = new Map(L.splits.map((s) => [s.gate_id, s.t]));
  const out = [];
  for (let g = 2; g <= n; g++) {
    if (seen.has(g)) continue;
    let a = g - 1; while (a > 1 && !seen.has(a)) a--;
    let b = g + 1; while (b <= n && !seen.has(b)) b++;
    const ta = seen.get(a) ?? L.t0, tb = b <= n ? seen.get(b) : L.t1;
    out.push({ gate: g, t: ta + (tb - ta) * (g - a) / (b - a), after: ta });
  }
  return out;
}

const missTxt = (m) => m.map((x) => `G${x.gate}`).join(", ");
// gates the lap rescue filled in (the scorer's near-misses at the right time): a spotter checks them too
const lapRecovered = (L) => (L.splits || []).filter((s) => s.recovered).map((s) => ({ gate: s.gate_id, t: s.t }));
const srcClass = (p) => p.recovered ? "src-REC" : `src-${p.source}`;

function lapAt(t) {
  // the lap segment at time t, or null (before the first G1, or after the run ended)
  return lapSegments().find((L) => t >= L.t0 && (L.done ? t < L.t1 : t <= L.t1 + 0.5)) || null;
}

const passLabel = (p) => p.gate_id >= 1 ? `G${p.gate_id}` : p.source === "EXTRA" ? `E${p.extra_id}` : "—";

// ─────────────────────────────────────────────────────────────
// Video window (with overlay)
// ─────────────────────────────────────────────────────────────

const STAGE_COL = { idle: "#9aa3ad", aligned: "#ffc83d", passed: "#4cd97b" };

function initVideo(w) {
  w.body.style.padding = "0";
  w.body.innerHTML = `<div class="vidwrap"><div class="vidbox"><video preload="auto" playsinline muted></video><canvas></canvas></div>
    <div class="vidctl">
      <button data-a="play" title="Space">▶</button>
      <button data-a="-1s" title="←">−1s</button><button data-a="-f" title="↓">−1f</button>
      <button data-a="+f" title="↑">+1f</button><button data-a="+1s" title="→">+1s</button>
      <span class="t">0.00 s</span>
      <select data-a="rate"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
      <label><input type="checkbox" data-ov="tracks"> tracks</label>
      <label><input type="checkbox" data-ov="dets"> detections</label>
      <label><input type="checkbox" data-ov="labels"> labels</label>
      <label><input type="checkbox" data-ov="hud"> lap HUD</label>
    </div></div>`;
  const v = $("video", w.body), c = $("canvas", w.body);
  S.videoEl = v;
  setVideoSource();
  w.body.querySelectorAll("[data-ov]").forEach((cb) => {
    cb.checked = !!S.ov[cb.dataset.ov];
    cb.onchange = () => { S.ov[cb.dataset.ov] = cb.checked; store.set("spotter.overlay", S.ov); drawOverlay(); };
  });
  w.body.querySelector("[data-a=rate]").onchange = (e) => { v.playbackRate = +e.target.value; };
  w.body.querySelectorAll("button[data-a]").forEach((b) => b.onclick = () => control(b.dataset.a));
  v.addEventListener("play", () => { $("[data-a=play]", w.body).textContent = "⏸"; });
  v.addEventListener("pause", () => { $("[data-a=play]", w.body).textContent = "▶"; });
  v.addEventListener("seeked", tick);
  v.addEventListener("loadedmetadata", () => {
    // the browser knows the real length (a file's frame count can be wrong, e.g. variable frame rate)
    if (S.video && Number.isFinite(v.duration) && v.duration > 0) { S.video.duration = v.duration; tick(); }
  });
  v.addEventListener("loadeddata", tick);
  const ro = new ResizeObserver(() => drawOverlay());
  ro.observe(c);
  w.overlay = c;
  w.cleanup = () => { ro.disconnect(); v.pause(); S.videoEl = null; };
}

function setVideoSource() {
  const v = S.videoEl;
  if (!v) return;
  const url = S.video?.url || "";
  if (v.dataset.src !== url) {
    v.dataset.src = url;
    if (url) v.src = url; else v.removeAttribute("src");
    S.t = 0;
  }
}

function control(a) {
  const v = S.videoEl;
  if (!v || !S.video) return;
  const f = 1 / (S.video.fps || 30);
  if (a === "play") { v.paused ? v.play() : v.pause(); return; }
  const d = { "-1s": -1, "+1s": 1, "-f": -f, "+f": f, "-10s": -10, "+10s": 10 }[a];
  if (d !== undefined) { v.pause(); seek(v.currentTime + d); }
}

function seek(t) {
  const v = S.videoEl;
  S.t = Math.max(0, Math.min(t, S.video?.duration || t));
  if (v && S.video) v.currentTime = S.t;
  tick();
}

function drawOverlay() {
  const w = wins.video;
  if (!w || !w.overlay || !S.videoEl) return;
  const c = w.overlay, v = S.videoEl;
  const dpr = devicePixelRatio || 1;
  const cw = c.clientWidth, ch = c.clientHeight;
  if (c.width !== Math.round(cw * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cw, ch);
  const vw = v.videoWidth || S.video?.w, vh = v.videoHeight || S.video?.h;
  if (!vw || !vh) return;
  const s = Math.min(cw / vw, ch / vh), ox = (cw - vw * s) / 2, oy = (ch - vh * s) / 2;
  const box = (b) => [ox + b[0] * s, oy + b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s];
  const fr = frameAt(S.t);
  g.font = "12px -apple-system, sans-serif";
  g.textBaseline = "top";
  if (fr && S.ov.dets) {
    g.setLineDash([4, 3]);
    g.strokeStyle = "#5aa0ff"; g.lineWidth = 1;
    for (const d of fr[2]) {
      const [x, y, ww, hh] = box(d);
      g.strokeRect(x, y, ww, hh);
      if (S.ov.labels) { g.fillStyle = "#5aa0ff"; g.fillText(`${S.frames.types[d[4]]} ${fmt(d[5])}`, x + 2, y + hh + 2); }
    }
    g.setLineDash([]);
  }
  if (fr && S.ov.tracks) {
    for (const tr of fr[1]) {
      const stage = S.frames.stages[tr[7]];
      const col = STAGE_COL[stage] || "#fff";
      const [x, y, ww, hh] = box(tr.slice(1, 5));
      g.strokeStyle = col; g.lineWidth = stage === "idle" ? 1.5 : 3;
      g.strokeRect(x, y, ww, hh);
      if (S.ov.labels) {
        const txt = `#${tr[0]} ${S.frames.types[tr[5]]} ${stage} ${fmt(tr[6])}`;
        const tw = g.measureText(txt).width;
        g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(x, y - 16, tw + 6, 16);
        g.fillStyle = col; g.fillText(txt, x + 3, y - 15);
      }
    }
  }
  if (S.ov.hud && S.run) {
    // recent pass flash + lap timer
    const recent = S.run.passes.filter((p) => p.t <= S.t && S.t - p.t < 0.6 && p.source !== "NOPASS");
    g.font = "bold 22px -apple-system, sans-serif";
    recent.forEach((p, i) => {
      g.fillStyle = p.recovered ? "#ff9f43" : p.source === "RACE" ? "#4cd97b" : "#5aa0ff";
      g.fillText(`${passLabel(p)} ${p.recovered ? "?" : "✓"}`, ox + 12, oy + 10 + i * 26);
    });
    const la = lapAt(S.t);
    if (la) {
      const txt = `Lap ${la.n}${la.done ? "" : " (unfinished)"}  ${fmt(S.t - la.t0)}s`;
      const tw = g.measureText(txt).width;
      g.fillStyle = "rgba(0,0,0,.6)"; g.fillRect(ox + vw * s - tw - 20, oy + 6, tw + 14, 30);
      g.fillStyle = "#ffb020"; g.fillText(txt, ox + vw * s - tw - 13, oy + 10);
    }
    // a lap that just closed with a skipped gate: flag it for the spotter
    const ended = lapSegments().find((L) => L.done && (L.missed.length || L.rec.length) && S.t >= L.t1 && S.t - L.t1 < 4);
    if (ended) {
      const bad = ended.missed.length > 0;
      const parts = [bad ? `missed ${missTxt(ended.missed)}` : "", ended.rec.length ? `recovered ${missTxt(ended.rec)}` : ""].filter(Boolean);
      const txt = `${bad ? "🚩" : "🟠"} Lap ${ended.n}: ${parts.join(" · ")} · spotter check`;
      const tw = g.measureText(txt).width;
      g.fillStyle = bad ? "rgba(120,0,0,.75)" : "rgba(110,60,0,.75)"; g.fillRect(ox + vw * s - tw - 20, oy + 40, tw + 14, 30);
      g.fillStyle = bad ? "#ff5c5c" : "#ff9f43"; g.fillText(txt, ox + vw * s - tw - 13, oy + 44);
    }
  }
  if (S.mode === "manual") {
    const near = S.marks.filter((m) => Math.abs(m.t - S.t) < 0.4);
    g.font = "bold 22px -apple-system, sans-serif";
    near.forEach((m, i) => { g.fillStyle = "#ffb020"; g.fillText(`◆ G${m.gate}`, ox + 12, oy + vh * s - 34 - i * 26); });
  }
}

// one tick per displayed frame while playing; also after seeks
let lastLive = 0;
function tick() {
  if (S.videoEl) S.t = S.videoEl.currentTime;
  const w = wins.video;
  if (w) $(".t", w.body).textContent = `${fmt(S.t)} s / ${fmt(S.video?.duration || 0, 1)} s`;
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
// Timeline
// ─────────────────────────────────────────────────────────────

function initTimeline(w) {
  w.body.style.padding = "0";
  w.body.innerHTML = `<canvas class="timeline"></canvas>`;
  const c = $("canvas", w.body);
  let hover = null;
  const tOf = (e) => {
    const r = c.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * (S.video?.duration || 0);
  };
  c.onmousedown = (e) => { if (S.video) { S.videoEl?.pause(); seek(tOf(e)); } };
  c.onmousemove = (e) => { hover = tOf(e); w.draw(); };
  c.onmouseleave = () => { hover = null; w.draw(); };
  w.draw = () => {
    const dpr = devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#181b20"; g.fillRect(0, 0, cw, ch);
    const D = S.video?.duration || 0;
    if (!D) { g.fillStyle = "#9aa3ad"; g.fillText("Pick a video", 10, 20); return; }
    const X = (t) => (t / D) * cw;
    const mid = ch * 0.55;
    g.fillStyle = "#2a2f37"; g.fillRect(0, mid - 1, cw, 2);
    g.font = "10px -apple-system, sans-serif";
    g.textBaseline = "top";
    // seconds grid
    const step = D > 300 ? 60 : D > 120 ? 20 : 10;
    g.fillStyle = "#59616b";
    for (let s = 0; s <= D; s += step) { g.fillRect(X(s), ch - 6, 1, 6); g.fillText(`${s}s`, X(s) + 2, ch - 14); }
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
        const col = p.recovered ? "#ff9f43" : p.source === "RACE" ? "#4cd97b" : p.source === "EXTRA" ? "#5aa0ff" : "#666";
        g.fillStyle = col;
        g.fillRect(X(p.t) - 1, mid - (p.gate_id === 1 ? 16 : 10), 2, p.gate_id === 1 ? 32 : 20);
        if (p.source === "RACE" && cw / D > 8) { g.fillText(String(p.gate_id), X(p.t) - 3, mid + 12); }
      }
    }
    if (S.mode === "manual") {
      g.fillStyle = "#ffb020";
      for (const m of S.marks) {
        const x = X(m.t);
        g.beginPath(); g.moveTo(x, mid - 22); g.lineTo(x + 5, mid - 16); g.lineTo(x, mid - 10); g.lineTo(x - 5, mid - 16); g.fill();
      }
    }
    g.fillStyle = "#ff4d4d"; g.fillRect(X(S.t) - 1, 0, 2, ch);
    if (hover !== null) { g.fillStyle = "#e6e6e6"; g.fillText(`${fmt(hover)}s`, Math.min(X(hover) + 4, cw - 40), ch / 2 - 30); }
  };
  const ro = new ResizeObserver(() => w.draw());
  ro.observe(c);
  w.cleanup = () => ro.disconnect();
  w.draw();
}

// ─────────────────────────────────────────────────────────────
// Race stats + gate splits
// ─────────────────────────────────────────────────────────────

function needRun(w) {
  if (!S.video) { w.body.innerHTML = `<div class="empty">Pick a video.</div>`; return true; }
  if (!S.run) {
    w.body.innerHTML = `<div class="empty">${esc(S.video.name)} has not been analysed yet — press <b>▶ Run analysis</b>.</div>`;
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
    ${k("Best lap", st ? fmt(st.best_lap) + "s" : "—")}
    ${k("Last lap", laps.length ? fmt(laps[laps.length - 1].dt) + "s" : "—")}
    ${k("Average", st ? fmt(st.mean_lap) + "s" : "—")}
    ${k("Consistency (CV)", st?.lap_cv != null ? fmt(st.lap_cv * 100, 1) + "%" : "—")}
    ${k("Theoretical best", st?.theoretical_best ? fmt(st.theoretical_best) + "s" : "—")}
    ${k("Best 3 in a row", st?.best3 ? fmt(st.best3) + "s" : "—")}
  </div>`;
  if (!laps.length) {
    h += `<div class="empty">No complete lap: the start gate (G1) was not crossed twice.</div>`;
  } else {
    const best = Math.min(...laps.map((l) => l.dt));
    h += `<table><tr><th>Lap</th><th>Time</th><th>Δ best</th><th>Gates</th><th>Missed</th><th>Start</th></tr>`;
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
        <td>${L.dt === best ? "★" : "+" + fmt(L.dt - best)}</td><td>${n}/${S.run.n_gates}</td>
        <td>${chk ? chk + " · spotter check" : "—"}</td><td>${fmt(L.t0, 1)}s</td></tr>`;
    });
    h += `</table>`;
    if (flagged) h += `<div class="hint"><span class="bad">🚩 ${flagged} lap${flagged > 1 ? "s" : ""} skipped a gate</span>: the pilot cut it,
      or the detector missed it. A spotter should check before the lap counts.</div>`;
    if (recov) h += `<div class="hint"><span class="warn">🟠 ${recov} lap${recov > 1 ? "s" : ""} complete with recovered gates</span>: a weak
      detection at the right time and place was taken as the pass. Quick spotter check.</div>`;
  }
  h += `<h4>Analysis</h4><div class="hint">${esc(S.run.info.timing || "")} · passes: ${esc(S.run.info.pass_logic || "")} · detector on ${esc(S.run.info.det_device || "?")}</div>`;
  w.body.innerHTML = h;
  w.body.querySelectorAll("tr[data-t]").forEach((tr) => tr.onclick = () => { S.videoEl?.pause(); seek(+tr.dataset.t); });
}

function renderSplits(w) {
  if (needRun(w)) return;
  const st = S.run.stats, n = S.run.n_gates;
  let h = "";
  // live: the lap at the playhead, gate by gate
  const la = lapAt(S.t);
  // the lap that closed last (before the playhead): skipped gates are known once it closes
  const closed = lapSegments().filter((L) => L.done && L.t1 <= S.t).pop();
  if (closed?.missed.length) {
    h += `<div class="spotter">🚩 Lap ${closed.n} (${fmt(closed.dt)}s) missed ${missTxt(closed.missed)}${closed.rec.length ? ` · recovered ${missTxt(closed.rec)}` : ""} · spotter check</div>`;
  } else if (closed?.rec.length) {
    h += `<div class="spotter rec">🟠 Lap ${closed.n} (${fmt(closed.dt)}s) recovered ${missTxt(closed.rec)} · spotter check</div>`;
  }
  if (la) {
    const t0 = la.t0;
    const ps = S.run.passes.filter((p) => p.source === "RACE" && p.t >= t0 - 1e-3 && p.t <= S.t
      && (!la.done || p.t < la.t1));
    h += `<h4>Lap ${la.n}${la.done ? "" : " (unfinished)"} at the playhead — ${fmt(S.t - t0)}s</h4>
      <table><tr><th>Gate</th><th>From start</th><th>Since prev</th><th>Best section</th><th>Δ</th></tr>`;
    // skipped gates show up as soon as a later gate is passed
    const skipped = la.done ? la.missed : [];
    let prev = null;
    for (const p of ps) {
      for (const m of skipped.filter((m) => m.gate < p.gate_id && m.gate > (prev ? prev.gate_id : 1))) {
        h += `<tr class="flag"><td class="bad">G${m.gate} ✕</td><td colspan="4" class="bad">missed · spotter check</td></tr>`;
      }
      const leg = prev ? p.t - prev.t : null;
      const sec = prev && (p.gate_id - prev.gate_id + n) % n === 1 ? st?.section_best?.[prev.gate_id - 1] : null;
      h += `<tr class="click ${p.recovered ? "flag-rec" : ""}" data-t="${p.t}" ${p.recovered ? 'title="recovered: a weak detection at the expected time · spotter check"' : ""}>
        <td class="${p.recovered ? "warn" : ""}">G${p.gate_id}${p.recovered ? " ?" : ""}</td><td>${fmt(p.t - t0)}s</td>
        <td>${leg !== null ? fmt(leg) + "s" : "—"}</td><td>${sec ? fmt(sec) + "s" : "—"}</td>
        <td>${sec && leg !== null ? (leg - sec < 0.005 ? "★" : "+" + fmt(leg - sec)) : ""}</td></tr>`;
      prev = p;
    }
    h += `</table>`;
  }
  if (st && st.section_grid?.length) {
    const secs = st.sections;
    h += `<h4>All laps — seconds between gates</h4><div style="overflow:auto"><table><tr><th>Lap</th>${secs.map((s) => `<th>${s.replace("→", "→<br>")}</th>`).join("")}<th>Lap</th></tr>`;
    st.section_grid.forEach((row, i) => {
      const miss = lapMissed(S.run.laps[i]), rec = lapRecovered(S.run.laps[i]);
      h += `<tr class="click ${la && la.i === i ? "cur" : ""} ${miss.length ? "flag" : ""}" data-t="${S.run.laps[i].t0}"
        title="${[miss.length ? "missed " + missTxt(miss) : "", rec.length ? "recovered " + missTxt(rec) : ""].filter(Boolean).join(" · ")}"><td>${miss.length ? "🚩 " : rec.length ? "🟠 " : ""}L${i + 1}</td>` + row.map((x, g) =>
        x === null ? `<td class="miss ${miss.length ? "bad" : ""}">·</td>` :`<td class="${st.section_best[g] !== null && Math.abs(x - st.section_best[g]) < 1e-6 ? "best" : ""}">${fmt(x)}</td>`).join("")
        + `<td>${fmt(st.lap_times[i])}</td></tr>`;
    });
    h += `<tr><td><b>Best</b></td>${st.section_best.map((x) => `<td class="best">${fmt(x)}</td>`).join("")}<td class="best">${fmt(st.theoretical_best)}</td></tr>`;
    h += `<tr><td>Median</td>${st.section_median.map((x) => `<td>${fmt(x)}</td>`).join("")}<td>${fmt(st.median_lap)}</td></tr></table></div>
      <div class="hint">Best row sum = theoretical best lap. · = that leg wasn't flown cleanly (a gate missed in between).</div>`;
  } else if (!la) {
    h += `<div class="empty">No laps yet.</div>`;
  }
  w.body.innerHTML = h;
  w.body.querySelectorAll("tr[data-t]").forEach((tr) => tr.onclick = () => { S.videoEl?.pause(); seek(+tr.dataset.t); });
}

// ─────────────────────────────────────────────────────────────
// Debug windows: pass detector / tracks, detections, passes
// ─────────────────────────────────────────────────────────────

function renderPassdet(w) {
  if (needRun(w)) return;
  const fr = frameAt(S.t);
  let h = `<div class="hint">t = ${fmt(S.t, 3)}s · ${fr ? fr[1].length : 0} tracks · stage: <span class="stage-idle">idle</span> → <span class="stage-aligned">aligned</span> → <span class="stage-passed">passed</span></div>`;
  if (!fr || !fr[1].length) { w.body.innerHTML = h + `<div class="empty">No tracks in this frame.</div>`; return; }
  h += `<table><tr><th>tid</th><th>type</th><th>stage</th><th>score</th><th>area %</th><th>c-dist</th><th>bbox</th></tr>`;
  for (const tr of [...fr[1]].sort((a, b) => a[0] - b[0])) {
    const stage = S.frames.stages[tr[7]];
    h += `<tr><td>#${tr[0]}</td><td>${esc(S.frames.types[tr[5]])}</td><td class="stage-${esc(stage)}">${esc(stage)}</td>
      <td>${fmt(tr[6])}</td><td>${fmt(tr[8] * 100)}</td><td>${fmt(tr[9], 3)}</td><td class="hint">${tr.slice(1, 5).join(",")}</td></tr>`;
  }
  w.body.innerHTML = h + `</table>`;
}

function renderDets(w) {
  if (needRun(w)) return;
  if (!S.frames?.has_dets) {
    w.body.innerHTML = `<div class="empty">This analysis has no raw detections saved — re-run the analysis to get them (tracks are in the Pass detector window).</div>`;
    return;
  }
  const fr = frameAt(S.t);
  let h = `<div class="hint">t = ${fmt(S.t, 3)}s · ${fr ? fr[2].length : 0} detections (top 10 by confidence)</div>`;
  if (fr && fr[2].length) {
    h += `<table><tr><th>#</th><th>type</th><th>conf</th><th>w×h</th><th>bbox</th></tr>`;
    fr[2].forEach((d, i) => {
      h += `<tr><td>${i + 1}</td><td>${esc(S.frames.types[d[4]])}</td><td>${fmt(d[5])}</td><td>${d[2] - d[0]}×${d[3] - d[1]}</td><td class="hint">${d.slice(0, 4).join(",")}</td></tr>`;
    });
    h += `</table>`;
  }
  w.body.innerHTML = h;
}

function renderPasses(w) {
  if (needRun(w)) return;
  const ps = S.run.passes;
  let near = -1, bd = 1e9;
  ps.forEach((p, i) => { const d = Math.abs(p.t - S.t); if (d < bd) { bd = d; near = i; } });
  const counts = { RACE: 0, EXTRA: 0, NOPASS: 0 };
  ps.forEach((p) => { counts[p.source] = (counts[p.source] || 0) + 1; });
  let h = `<div class="hint"><span class="src-RACE">${counts.RACE} gate passes</span> · <span class="src-EXTRA">${counts.EXTRA || 0} extras</span> · <span class="src-NOPASS">${counts.NOPASS || 0} not used</span></div>
    <table><tr><th>t</th><th>label</th><th>source</th><th>p(pass)</th><th>type</th><th>reason</th></tr>`;
  ps.forEach((p, i) => {
    h += `<tr class="click ${i === near && bd < 1 ? "cur" : ""}" data-t="${p.t}"><td>${fmt(p.t)}s</td><td class="${srcClass(p)}">${passLabel(p)}</td>
      <td class="${srcClass(p)}">${esc(p.recovered ? "RECOVERED" : p.source)}</td><td>${fmt(p.p)}</td><td>${esc(p.type)}</td><td class="hint">${esc(p.reason)}</td></tr>`;
  });
  w.body.innerHTML = h + `</table>`;
  w.body.querySelectorAll("tr[data-t]").forEach((tr) => tr.onclick = () => { S.videoEl?.pause(); seek(+tr.dataset.t - 0.5); });
  const cur = $("tr.cur", w.body);
  if (cur && S.videoEl && !S.videoEl.paused) cur.scrollIntoView({ block: "nearest" });
}

// ─────────────────────────────────────────────────────────────
// Gates, videos, learning, manual marks, jobs
// ─────────────────────────────────────────────────────────────

function renderGates(w) {
  const G = S.gates;
  if (!S.track) { w.body.innerHTML = `<div class="empty">Pick a track.</div>`; return; }
  if (!G || !G.gates.length) { w.body.innerHTML = `<div class="empty">This track has no learned gates yet — use Learn · auto or Learn · manual.</div>`; return; }
  const L = G.learned;
  let h = L ? `<div class="hint">${L.n_gates} gates · ${L.method} · lap ≈ ${fmt(L.lap_seconds, 1)}s · ${L.extras || 0} tracked extras · from ${esc((L.learned_from || []).join(", "))}</div>` : "";
  h += `<div class="gates">`;
  for (const g of G.gates) {
    h += `<div class="gate ${S.mode === "manual" && g.id === S.nextGate ? "next" : ""}"><div class="h">G${g.id} <span>${esc(g.type)}${g.offset_s != null ? " · +" + fmt(g.offset_s, 1) + "s" : ""}${g.seen_laps != null ? " · " + g.seen_laps + " laps" : ""}</span></div>
      <div class="imgs">${g.images.slice(0, 6).map((u) => `<img loading="lazy" src="${esc(u)}">`).join("")}</div></div>`;
  }
  h += `</div>`;
  if (G.extras.length) {
    h += `<h4>Tracked extras (not counted as gates)</h4><div class="hint">${G.extras.map((e) => `E${e.id} +${fmt(e.offset_s, 1)}s (${e.seen_laps} laps)`).join(" · ")}</div>`;
  }
  w.body.innerHTML = h;
}

function uploadForm(role) {
  return `<div class="row"><input type="file" accept="video/*" data-up="${role}"><button data-upbtn="${role}">Upload</button></div>
    <div class="row"><input type="text" placeholder="…or a path on this computer, e.g. ~/Downloads/run.mp4" data-path="${role}" style="flex:1;min-width:200px"><button data-impbtn="${role}">Add</button></div>`;
}

function wireUpload(w) {
  w.body.querySelectorAll("[data-upbtn]").forEach((b) => b.onclick = async () => {
    const role = b.dataset.upbtn, f = $(`[data-up="${role}"]`, w.body).files[0];
    if (!f) return status("Choose a video file first.", true);
    const fd = new FormData();
    fd.append("file", f);
    status(`Uploading ${f.name} (${(f.size / 1e6).toFixed(0)} MB)…`);
    try {
      await api(`/api/tracks/${encodeURIComponent(S.track)}/upload?role=${role}`, { method: "POST", body: fd });
      status(`Added ${f.name}`);
      await loadTracks(); render();
    } catch (e) { status(e.message, true); }
  });
  w.body.querySelectorAll("[data-impbtn]").forEach((b) => b.onclick = async () => {
    const role = b.dataset.impbtn, p = $(`[data-path="${role}"]`, w.body).value.trim();
    if (!p) return;
    status(`Copying ${p}…`);
    try {
      const r = await api(`/api/tracks/${encodeURIComponent(S.track)}/import`, { json: { path: p, role } });
      status(`Added ${r.name}`);
      await loadTracks(); render();
    } catch (e) { status(e.message, true); }
  });
}

function videoRows(vids, withLearn) {
  if (!vids.length) return `<div class="empty">No videos.</div>`;
  let h = `<table><tr><th>Video</th><th>Role</th><th>Length</th><th>fps</th><th>Status</th><th></th></tr>`;
  for (const v of vids) {
    const job = S.jobs.find((j) => j.video === v.name && ["running", "queued"].includes(j.status));
    const st = job ? `<span class="badge warn">${fmt(job.progress, 0)}%</span>`
      : v.analysed ? `<span class="badge ok">analysed</span>` : v.learn_analysed ? `<span class="badge">learn run</span>` : `<span class="badge">new</span>`;
    h += `<tr><td>${esc(v.name)}</td><td>${v.role}</td><td>${fmt(v.duration, 0)}s</td><td>${v.fps}</td><td>${st}</td>
      <td><button data-open="${esc(v.name)}">Open</button>${withLearn ? "" : ` <button data-an="${esc(v.name)}" ${job ? "disabled" : ""}>Analyse</button>`}
      <button data-rm="${esc(v.name)}" data-folder="${v.folder}" ${job ? "disabled" : ""} title="Move out of the track (to removed_videos/)">✕ Remove</button></td></tr>`;
  }
  return h + `</table>`;
}

function wireVideoRows(w) {
  w.body.querySelectorAll("[data-open]").forEach((b) => b.onclick = () => { $("#videoSel").value = b.dataset.open; selectVideo(b.dataset.open); });
  w.body.querySelectorAll("[data-an]").forEach((b) => b.onclick = () => startJob({ kind: "analyse", track: S.track, video: b.dataset.an }));
  w.body.querySelectorAll("[data-rm]").forEach((b) => b.onclick = async () => {
    const { rm: name, folder } = b.dataset;
    const what = folder === "learn_videos" ? "learn videos" : "race videos";
    if (!confirm(`Remove ${name} from the ${what} of ${S.track}?\nThe file is moved to removed_videos/, not deleted.`)) return;
    try {
      await api(`/api/tracks/${encodeURIComponent(S.track)}/remove`, { json: { folder, name } });
      status(`Removed ${name} from the ${what}.`);
      await loadTracks(); render(["learn", "library"]);
    } catch (e) { status(e.message, true); }
  });
}

function renderLibrary(w) {
  const t = curTrack();
  if (!t) { w.body.innerHTML = `<div class="empty">Create a track with ＋ first.</div>`; return; }
  if (w.body.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;  // don't wipe typing
  let h = videoRows(t.videos, false);
  h += `<h4>Add a race video</h4>${uploadForm("race")}<h4>Add a learn video</h4>${uploadForm("learn")}`;
  w.body.innerHTML = h;
  wireVideoRows(w);
  wireUpload(w);
}

function renderLearn(w) {
  const t = curTrack();
  if (!t) { w.body.innerHTML = `<div class="empty">Create a track with ＋ first.</div>`; return; }
  if (w.body.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
  const learnVids = t.videos.filter((v) => v.folder === "learn_videos");
  const job = S.jobs.find((j) => j.kind === "learn" && j.track === t.name && ["running", "queued"].includes(j.status));
  const L = t.learned;
  let h = `<div class="hint">Learns the gate order, timing and looks from the videos below. The first gate passed is
    the start/finish gate (G1). Every video is labelled in gate order with one shared timing and look per gate, then
    the gates are re-learned from those labels until they settle. New learn videos are analysed first (YOLO), which takes a while.</div>
    <div class="hint">Wrong gates or lap time? Usually the start gate was missed on some laps. In <b>Manual</b> mode open
    one learn video and press <kbd>L</kbd> at every start/finish crossing (no other marks needed), then learn here again:
    those times anchor the laps of every video.</div>
    <h4>Learn videos</h4>${videoRows(learnVids, true)}
    <h4>Add a learn video</h4>${uploadForm("learn")}
    <h4>Learn</h4><div class="row">Gates on the track <input type="number" min="2" max="40" id="nGates" value="${L?.n_gates || store.get("spotter.ngates." + t.name, 14)}" style="width:70px">
      <button class="primary" id="learnBtn" ${job || !learnVids.length ? "disabled" : ""}>${job ? `Learning… ${fmt(job.progress, 0)}%` : "🎓 Learn track"}</button></div>`;
  if (L) {
    h += `<h4>Current track model</h4><div class="hint">${L.n_gates} gates · method <b>${esc(L.method)}</b> · lap ≈ ${fmt(L.lap_seconds, 1)}s ·
      ${L.extras || 0} extras · learned from ${esc((L.learned_from || []).join(", "))}</div>
      <div class="row"><button id="resetLearn" ${job ? "disabled" : ""}>🗑 Reset learning</button>
      <span class="hint">moves the learned gates to learn_backups/; analyses are kept</span></div>`;
  }
  w.body.innerHTML = h;
  wireVideoRows(w);
  wireUpload(w);
  const btn = $("#learnBtn", w.body);
  if (btn) btn.onclick = () => {
    const n = +$("#nGates", w.body).value;
    store.set("spotter.ngates." + t.name, n);
    startJob({ kind: "learn", track: t.name, gates: n });
  };
  const rst = $("#resetLearn", w.body);
  if (rst) rst.onclick = async () => {
    if (!confirm(`Forget the learned gates of ${t.name}? They are moved to a backup folder, not deleted.`)) return;
    try {
      const r = await api(`/api/tracks/${encodeURIComponent(t.name)}/reset`, { json: {} });
      status(r.backup ? `Learning reset; old files in ${r.backup}` : "Nothing was learned yet.");
      await loadTracks(); await loadGates();
    } catch (e) { status(e.message, true); }
  };
}

function renderMarks(w) {
  if (!S.video) { w.body.innerHTML = `<div class="empty">Pick a video of this track to mark.</div>`; return; }
  if (w.body.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
  const analysed = S.video.analysed || S.video.learn_analysed;
  const t = curTrack();
  const marked = t.videos.filter((v) => v.has_marks).map((v) => v.stem);
  if (S.marks.length && !marked.includes(S.video.stem)) marked.push(S.video.stem);
  let h = `<div class="hint">Play the video and press <kbd>L</kbd> when the drone crosses the <b>start/finish</b> gate (G1),
    <kbd>G</kbd> at every next gate, <kbd>L</kbd> again at the end of each lap. <kbd>Backspace</kbd> removes the last mark.
    Marks snap to the nearest detected pass, which gives each gate its images.</div>`;
  if (!analysed) h += `<div class="row"><span class="badge warn">not analysed</span> Run the analysis first, so marks can snap to detected passes.</div>`;
  h += `<div class="row">Next mark: <b style="color:var(--accent)">G${S.nextGate}</b>
    <label>set <input type="number" min="1" max="40" id="nextG" value="${S.nextGate}" style="width:60px"></label>
    <button id="clrMarks">Clear all</button></div>`;
  const near = S.marks.reduce((b, m, i) => (Math.abs(m.t - S.t) < Math.abs((S.marks[b]?.t ?? 1e9) - S.t) ? i : b), -1);
  h += `<table><tr><th>#</th><th>Gate</th><th>t</th><th>Lap</th><th></th></tr>`;
  let lap = 0;
  S.marks.forEach((m, i) => {
    if (m.gate === 1) lap++;
    h += `<tr class="click ${i === near && Math.abs(m.t - S.t) < 0.5 ? "cur" : ""}" data-t="${m.t}"><td>${i + 1}</td><td>G${m.gate}</td><td>${fmt(m.t)}s</td><td>${lap || "—"}</td><td><span class="x" data-del="${i}" style="cursor:pointer">✕</span></td></tr>`;
  });
  h += `</table>`;
  const g1 = S.marks.filter((m) => m.gate === 1).length;
  h += `<h4>Build the track</h4><div class="hint">Marked videos: ${esc(marked.join(", ") || "none")} · ${g1} start-gate marks on this video
    (${Math.max(0, g1 - 1)} full laps)</div>
    <div class="row"><button class="primary" id="learnMarks" ${g1 < 2 ? "disabled" : ""}>🎓 Learn track from marks</button></div>`;
  w.body.innerHTML = h;
  w.body.querySelectorAll("tr[data-t]").forEach((tr) => tr.onclick = (e) => {
    if (e.target.dataset.del !== undefined) {
      S.marks.splice(+e.target.dataset.del, 1); saveMarksSoon(); render(["marks", "timeline"]); return;
    }
    S.videoEl?.pause(); seek(+tr.dataset.t);
  });
  $("#nextG", w.body).onchange = (e) => { S.nextGate = Math.max(1, +e.target.value || 1); render(["marks", "gates"]); };
  $("#clrMarks", w.body).onclick = () => { if (confirm("Remove all marks on this video?")) { S.marks = []; S.nextGate = 1; saveMarksSoon(); render(["marks", "timeline"]); } };
  $("#learnMarks", w.body).onclick = async () => {
    clearTimeout(marksTimer);
    await api(`/api/tracks/${encodeURIComponent(S.track)}/marks/${encodeURIComponent(S.video.stem)}`, { json: { marks: S.marks } });
    status("Learning from marks…");
    try {
      const r = await api(`/api/tracks/${encodeURIComponent(S.track)}/manual`, { json: { stems: marked } });
      status(`Learned ${r.n_gates} gates (lap ≈ ${r.lap_seconds}s) from ${r.videos.join(", ")}; re-labelled ${r.relabelled.length} analysed videos.`);
      await loadTracks(); await loadGates(); await loadRun();
    } catch (e) { status(e.message, true); }
  };
}

function addMark(gate) {
  if (!S.video) return;
  S.marks.push({ t: +S.t.toFixed(3), gate });
  S.marks.sort((a, b) => a.t - b.t);
  S.nextGate = gate + 1;
  saveMarksSoon();
  status(`Marked G${gate} at ${fmt(S.t)}s`);
  render(["marks", "timeline", "gates"]);
  drawOverlay();
}

function renderJobs(w) {
  if (!S.jobs.length) { w.body.innerHTML = `<div class="empty">No jobs yet. Analysis and learning run here, with progress.</div>`; return; }
  const open = new Set([...w.body.querySelectorAll("details[open]")].map((d) => d.dataset.id));
  let h = "";
  for (const j of S.jobs.slice(0, 12)) {
    const el = j.ended ? fmt(j.ended - j.started, 0) + "s" : j.started ? fmt(Date.now() / 1000 - j.started, 0) + "s" : "";
    h += `<div class="job ${j.status}"><div class="h"><b>${esc(j.title)}</b><span>${esc(j.status)} · ${el}
      ${["running", "queued"].includes(j.status) ? `<button data-cancel="${j.id}">Cancel</button>` : ""}</span></div>
      <div class="bar"><i style="width:${j.progress || 0}%"></i></div><div class="hint">${esc(j.detail || "")}</div>
      <details data-id="${j.id}" ${open.has(j.id) || j.status === "failed" ? "open" : ""}><summary class="hint">log</summary><pre class="log">${esc(j.log.slice(-60).join("\n"))}</pre></details></div>`;
  }
  w.body.innerHTML = h;
  w.body.querySelectorAll("[data-cancel]").forEach((b) => b.onclick = () => api(`/api/jobs/${b.dataset.cancel}/cancel`, { method: "POST" }).then(pollJobs));
}

// ─────────────────────────────────────────────────────────────
// Modes, keyboard, start-up
// ─────────────────────────────────────────────────────────────

function setMode(m) {
  S.mode = m;
  store.set("spotter.mode", m);
  writeHash();
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  applyLayout();
  renderActions();
  if (m === "manual") loadMarks();
  render();
}

addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const k = e.key;
  if (k === " ") { e.preventDefault(); control("play"); }
  else if (k === "ArrowLeft") { e.preventDefault(); control(e.shiftKey ? "-10s" : "-1s"); }
  else if (k === "ArrowRight") { e.preventDefault(); control(e.shiftKey ? "+10s" : "+1s"); }
  else if (k === "ArrowDown") { e.preventDefault(); control("-f"); }
  else if (k === "ArrowUp") { e.preventDefault(); control("+f"); }
  else if (S.mode === "manual" && (k === "l" || k === "L")) addMark(1);
  else if (S.mode === "manual" && (k === "g" || k === "G")) addMark(Math.max(2, S.nextGate));
  else if (S.mode === "manual" && k === "Backspace") {
    e.preventDefault();
    const m = S.marks.pop();
    if (m) { S.nextGate = m.gate; saveMarksSoon(); render(["marks", "timeline", "gates"]); drawOverlay(); }
  }
});

$("#modes").onclick = (e) => { if (e.target.dataset.mode) setMode(e.target.dataset.mode); };
$("#trackSel").onchange = (e) => selectTrack(e.target.value);
$("#videoSel").onchange = (e) => selectVideo(e.target.value);
$("#winBtn").onclick = (e) => { e.stopPropagation(); $("#winMenu").classList.toggle("open"); };
document.addEventListener("click", (e) => { if (!$("#winMenu").contains(e.target)) $("#winMenu").classList.remove("open"); });
$("#newTrack").onclick = async () => {
  const name = prompt("New track name (a folder in the data folder):");
  if (!name) return;
  try {
    await api("/api/tracks", { json: { name } });
    await loadTracks();
    $("#trackSel").value = name;
    await selectTrack(name);
  } catch (e) { status(e.message, true); }
};

function hashState() {
  const h = new URLSearchParams(location.hash.slice(1));
  return { mode: h.get("mode"), track: h.get("track"), video: h.get("video"), t: h.get("t") };
}

function writeHash() {
  const h = new URLSearchParams({ mode: S.mode });
  if (S.track) h.set("track", S.track);
  if (S.video) h.set("video", S.video.name);
  history.replaceState(null, "", "#" + h.toString());
}

// ─────────────────────────────────────────────────────────────
// Add video: browse this computer (served by the local server)
// ─────────────────────────────────────────────────────────────

const BR = { dir: store.get("spotter.browseDir", "~/Downloads"), sel: null };

async function browseTo(dir) {
  try {
    const d = await api(`/api/browse?dir=${encodeURIComponent(dir)}`);
    BR.dir = d.dir; BR.sel = null;
    store.set("spotter.browseDir", d.dir);
    $("#brDir").value = d.dir;
    $("#brUp").disabled = !d.parent;
    $("#brUp").onclick = () => d.parent && browseTo(d.parent);
    $("#brPlaces").innerHTML = d.places.map((p) => `<button data-p="${esc(p.path)}">${esc(p.name)}</button>`).join("");
    $("#brPlaces").querySelectorAll("[data-p]").forEach((b) => b.onclick = () => browseTo(b.dataset.p));
    const have = new Set((curTrack()?.videos || []).map((v) => v.name));
    $("#brList").innerHTML = d.dirs.map((n) => `<div class="dir" data-dir="${esc(n)}">📁 ${esc(n)}</div>`).join("")
      + d.videos.map((v) => `<div data-v="${esc(v.path)}" data-n="${esc(v.name)}">🎞 ${esc(v.name)}
          <span class="meta">${have.has(v.name) ? "already in track · " : ""}${v.mb} MB · ${new Date(v.mtime * 1000).toLocaleDateString()}</span></div>`).join("")
      || `<div class="hint">No folders or videos here.</div>`;
    $("#brList").querySelectorAll("[data-dir]").forEach((el) => el.onclick = () => browseTo(d.dir + "/" + el.dataset.dir));
    $("#brList").querySelectorAll("[data-v]").forEach((el) => {
      el.onclick = () => {
        $("#brList").querySelectorAll(".sel").forEach((x) => x.classList.remove("sel"));
        el.classList.add("sel");
        BR.sel = el.dataset.v;
        $("#brSel").textContent = el.dataset.n;
        $("#brAdd").disabled = false;
      };
      el.ondblclick = () => addBrowsed();
    });
    $("#brSel").textContent = "";
    $("#brAdd").disabled = true;
  } catch (e) { status(e.message, true); }
}

async function addBrowsed() {
  if (!BR.sel || !S.track) return;
  const role = document.querySelector("input[name=brRole]:checked").value;
  $("#brAdd").disabled = true;
  status(`Copying ${BR.sel} into ${S.track}…`);
  try {
    const r = await api(`/api/tracks/${encodeURIComponent(S.track)}/import`, { json: { path: BR.sel, role } });
    $("#browser").classList.remove("open");
    await loadTracks();
    $("#videoSel").value = r.name;
    await selectVideo(r.name);
    status(`Added ${r.name} as a ${role} video${role === "race" ? " — press ▶ Run analysis" : " — learn the track in Learn · auto"}`);
  } catch (e) { status(e.message, true); $("#brAdd").disabled = false; }
}

$("#addVideo").onclick = () => {
  if (!S.track) return status("Create or pick a track first.", true);
  $("#brTrack").textContent = S.track;
  const role = S.mode === "race" ? "race" : "learn";
  document.querySelector(`input[name=brRole][value=${role}]`).checked = true;
  $("#browser").classList.add("open");
  browseTo(BR.dir);
};
$("#brClose").onclick = () => $("#browser").classList.remove("open");
$("#browser").onclick = (e) => { if (e.target.id === "browser") $("#browser").classList.remove("open"); };
$("#brGo").onclick = () => browseTo($("#brDir").value);
$("#brDir").onkeydown = (e) => { if (e.key === "Enter") browseTo($("#brDir").value); };
$("#brAdd").onclick = addBrowsed;

(async function start() {
  try {
    const H = hashState();
    if (H.mode && DEFAULTS[H.mode]) S.mode = H.mode;
    if (H.track) store.set("spotter.track", H.track);
    await loadTracks(false);
    document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === S.mode));
    applyLayout();
    if (S.track) await selectTrack(S.track);
    status(S.track ? `Track ${S.track} — pick a video` : "Create a track with ＋");
    if (H.video && curTrack()?.videos.some((v) => v.name === H.video)) {
      $("#videoSel").value = H.video;
      await selectVideo(H.video);
      if (H.t) { const t = +H.t; const v = S.videoEl; if (v) v.addEventListener("loadedmetadata", () => seek(t), { once: true }); seek(t); }
    }
    pollJobs();
    requestAnimationFrame(loop);
  } catch (e) { status("Server not reachable: " + e.message, true); }
})();
