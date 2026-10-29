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
  league: false,                       // the shared page for pilots (spotter_web.py --league)
  mode: store.get("spotter.mode", "race"),
  tracks: [], track: null, video: null,
  run: null, frames: null, gates: null,
  combo: null,                         // a combo view picked in Race mode: its pilots' analyses (loadCombo)
  solo: null,                          // the picked analysed video as a one-pilot race (the race map, call-outs)
  raceSet: { sound: true, voice: true, ...store.get("spotter.raceSound", {}) },   // Race mode: sounds while playing
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
  livefeed: { title: "Live feed", init: initLiveFeed },
  livestats: { title: "Live — laps", render: renderLiveStats },
  livelog:  { title: "Live — gate log", render: renderLiveLog },
  livemap:  { title: "Race map", init: initLiveMap, live: true },
  racestand: { title: "Race — standings (combo)", render: renderRaceStand, live: true },
  layout:   { title: "Track layout", init: initLayout, render: (w) => w.load() },
  race3d:   { title: "Race 3D", init: initRace3D },
};
const LIVE_PANELS = ["livefeed", "livestats", "livelog", "livemap"];

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
  live: {
    livefeed: [0, 0, 0.64, 0.82], timeline: [0, 0.82, 0.64, 0.18], livestats: [0.64, 0, 0.36, 0.42], livelog: [0.64, 0.42, 0.36, 0.58],
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
  el.dataset.win = id;                 // the phone layout (style.css) orders and sizes windows by it
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
  m.innerHTML = Object.entries(PANELS).filter(([id]) => !S.league || !["learn", "marks", "layout", ...LIVE_PANELS].includes(id)).map(([id, p]) =>
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
  if (wins.livemap && (!ids || ids.includes("livemap") || ids.includes("livestats"))) wins.livemap.draw();
}

// ─────────────────────────────────────────────────────────────
// Data loading
// ─────────────────────────────────────────────────────────────

async function loadTracks(keep = true) {
  const d = await api("/api/tracks");
  S.tracks = d.tracks;
  S.limits = d.upload_limits;
  if (d.version) { $("#ver").textContent = "v" + d.version.split(" ")[0]; $("#ver").title = "lazyspotter " + d.version; }
  if (d.league && !S.league) applyLeague();
  if (!keep || !curTrack()) S.track = store.get("spotter.track", null);
  if (S.track !== "" && !curTrack()) S.track = S.tracks[0]?.name || null;      // "" = none picked on purpose
  if (S.track === "") S.track = null;
  fillTrackSel();
  fillVideoSel();
  if (!S.tracks.length) status(S.league ? "No tracks yet." : `No tracks yet in ${d.data_root} — click ＋ to create one.`);
}

// the model (model_profiles.py): v2 (current) or v3 (cozYnut's new detector + two-stage scorer); each keeps its own
// analyses, so switching shows a video's analysis by the other model (or "new" if it has none yet)
async function loadModel() {
  const d = await api("/api/model").catch(() => null);
  if (!d) return;
  S.model = d.active;
  $("#modelSel").innerHTML = d.models.map((m) => `<option value="${m.name}" ${m.name === d.active ? "selected" : ""} ${m.ready ? "" : "disabled"}
    title="${esc(m.about)} (${esc(m.detector)})">${esc(m.label)}${m.ready ? "" : " (files missing)"}</option>`).join("");
  $("#modelSel").title = d.models.find((m) => m.name === d.active)?.about || "";
}

$("#modelSel").onchange = async (e) => {
  try {
    const d = await api("/api/model", { json: { model: e.target.value } });
    S.model = d.active;
    status(`Model ${d.active}: analyses, Live and new jobs use it — a video it hasn't analysed shows as new`);
    $("#modelSel").title = d.models.find((m) => m.name === d.active)?.about || "";
    await loadTracks();
    await loadGates();
    if (S.video) await selectVideo(S.video.name);
    render();
  } catch (err) { status(err.message, true); $("#modelSel").value = S.model; }
};

// the league page: watch the analysed runs, upload and analyse your own; nothing else
function applyLeague() {
  S.league = true;
  S.mode = "race";
  document.querySelectorAll("#modes button").forEach((b) => { if (b.dataset.mode !== "race") b.remove(); });
  $("#newTrack").remove();
  $("#addVideo").textContent = "⬆ Upload your video";
  buildWinMenu();
}

function fillTrackSel() {
  $("#trackSel").innerHTML = (S.league ? "" : `<option value="" ${!S.track ? "selected" : ""}>— none —</option>`) +
    S.tracks.map((t) => `<option value="${esc(t.name)}" ${t.name === S.track ? "selected" : ""}>${esc(t.name)}</option>`).join("");
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
  S.track = name || null;
  store.set("spotter.track", name || "");
  writeHash();
  S.video = null; S.run = null; S.frames = null; S.marks = []; S.combo = null; S.solo = null;
  fillVideoSel();
  await loadGates();
  setVideoSource();
  syncLapMode();
  render();
}

async function loadGates() {
  S.gates = S.track ? await api(`/api/tracks/${encodeURIComponent(S.track)}/gates`).catch(() => null) : null;
  render(["gates", "learn", "marks", "layout"]);
}

async function selectVideo(name) {
  const t = curTrack();
  S.video = t?.videos.find((v) => v.name === name) || null;
  S.run = null; S.frames = null; S.marks = []; S.nextGate = 1; S.combo = null; S.solo = null;
  writeHash();
  syncPilotBox();
  setVideoSource();
  renderActions();
  if (S.video) {
    status(`${S.video.name}: ${S.video.w}×${S.video.h}, ${S.video.fps} fps, ${S.video.duration}s`);
    await loadRun();
    await loadCombo();
    renderActions();
    if (S.mode === "manual") await loadMarks();
  }
  render();
}

async function loadRun() {
  S.solo = null;
  if (!S.video || !(S.video.analysed || S.video.learn_analysed)) { S.run = null; S.frames = null; return; }
  const base = `/api/tracks/${encodeURIComponent(S.track)}/runs/${encodeURIComponent(S.video.stem)}`;
  try {
    S.run = await api(base);
    S.solo = { n_gates: S.run.n_gates, legs: S.run.legs, layout: S.run.layout, track: S.track, pilots: [racePilot(S.run, 0, soloName())] };
    comboMemo = null;
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

// a race-day combo (quad) view of several pilots: split into one video per pilot, each analysed (combo_split.py)
const COMBO_W = 1600;
const maybeCombo = (v) => v && v.role !== "combo" && v.w >= COMBO_W && !v.analysed;
// an uploaded race video: a combo is split per pilot only if you say so; any other video is just analysed
async function analyseOrSplit(track, v) {
  if (maybeCombo(v)) {
    status(`Is ${v.name} a combo view? Looking…`);
    const L = await api(`/api/tracks/${encodeURIComponent(track)}/combo?video=${encodeURIComponent(v.name)}`).catch(() => null);
    if (L?.combo) {
      const on = L.tiles.filter((t) => t.active).map((t) => `CH${t.ch}`);
      if (confirm(`${v.name} is a combo view: ${L.grid[0]}×${L.grid[1]}, pilots on ${on.join(", ")}.\n\n`
        + `OK: split it into one video per pilot (…_CH1 …) and analyse each\nCancel: analyse it as one video`))
        return startJob({ kind: "split", analyse: true, track, video: v.name });
    }
  }
  return startJob({ kind: "analyse", track, video: v.name });
}

function renderActions() {
  const a = $("#actions");
  a.innerHTML = "";
  const v = S.video;
  if (!v || S.mode === "learn" || S.mode === "live") return;
  if (v.role === "combo") {                     // its pilots' videos (…_CH1 …): split, analysed one by one, then raced
    const ps = S.combo?.video === v.name ? S.combo.pilots : null, todo = (ps || []).filter((p) => !p.analysed);
    const busy = S.jobs.some((j) => ["running", "queued"].includes(j.status) && j.track === S.track
      && (j.video === v.name || (ps || []).some((p) => p.video === j.video)));
    if (!ps) return;
    if (!S.league && (!ps.length || todo.length)) {
      const b = document.createElement("button");
      b.className = "primary";
      b.disabled = busy;
      b.textContent = busy ? "Analysing pilots…" : !ps.length ? "⊞ Split & analyse pilots" : `▶ Analyse ${todo.length} pilot${todo.length > 1 ? "s" : ""} (${ps.length - todo.length}/${ps.length} done)`;
      b.title = !ps.length ? "One video per pilot (…_CH1 …, in test_videos/), each analysed in full; then the combo races them as it plays"
        : `Analyse ${todo.map((p) => p.video).join(", ")} with model ${S.model} (one at a time)`;
      b.onclick = async () => {
        if (!ps.length) return startJob({ kind: "split", analyse: true, track: S.track, video: v.name });
        for (const p of todo) await startJob({ kind: "analyse", track: S.track, video: p.video });
      };
      a.appendChild(b);
    }
    if (!a.children.length) a.innerHTML = `<span class="hint">combo · ${ps.length} pilots</span>`;
    return;
  }
  if (maybeCombo(v) && !S.league) {             // big enough to be a combo: split it (asked), or analyse as is
    const busy = S.jobs.some((j) => j.video === v.name && ["running", "queued"].includes(j.status));
    const b = document.createElement("button");
    b.textContent = busy ? "Working…" : "⊞ Split & analyse pilots";
    b.disabled = busy;
    b.title = "A combo (quad) view of several pilots: one video per pilot (…_CH1 …), each analysed; the combo moves to combo_videos/";
    b.onclick = async () => {
      if (!confirm(`Split ${v.name} into one video per pilot and analyse each?\nThe combo moves to combo_videos/.`)) return;
      startJob({ kind: "split", analyse: true, track: S.track, video: v.name });
    };
    a.appendChild(b);
  }
  const running = S.jobs.some((j) => j.kind === "analyse" && j.video === v.name && ["running", "queued"].includes(j.status));
  if (S.league && v.analysed && !running) return;
  const b = document.createElement("button");
  b.className = v.analysed ? "" : "primary";
  b.textContent = running ? "Analysing…" : v.analysed ? "▶ Re-run analysis" : "▶ Run analysis";
  b.disabled = running;
  b.title = "YOLO + pass detection + gate labelling (saved, loads instantly next time)";
  b.onclick = () => {
    if (v.analysed && !confirm(`${v.name} is already analysed.\nRun the analysis again over it? Its current results are replaced.`)) return;
    startJob({ kind: "analyse", track: S.track, video: v.name });
  };
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
        if (S.video?.role === "combo" && (j.kind === "learn" || j.video === S.video.name || S.combo?.pilots.some((p) => p.video === j.video))) {
          S.video = curTrack()?.videos.find((v) => v.name === S.video.name) || S.video;
          await loadCombo();
          render();
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
  w.body.innerHTML = `<div class="vidwrap"><div class="vidbox"><video preload="auto" playsinline muted></video><canvas></canvas>
    <div class="vspin" hidden><i></i><span>Loading video…</span></div></div>
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
      <label title="While it plays: a beep per gate pass, a coin when the gate is named, a fanfare on the start gate"><input type="checkbox" data-rs="sound"> 🔊 sound</label>
      <label title="While it plays: lap times called out (with the pilot's name), and on a combo the places as they change: who takes the lead, who's now second, who's down, who finishes"><input type="checkbox" data-rs="voice"> 🗣 voice</label>
      <input data-x="pilot" size="12" maxlength="40" placeholder="Pilot name" title="Called out with every lap time (a combo: name its pilots in Race — standings)">
    </div></div>`;
  const v = $("video", w.body), c = $("canvas", w.body);
  S.videoEl = v;
  setVideoSource();
  w.body.querySelectorAll("[data-ov]").forEach((cb) => {
    cb.checked = !!S.ov[cb.dataset.ov];
    cb.onchange = () => { S.ov[cb.dataset.ov] = cb.checked; store.set("spotter.overlay", S.ov); drawOverlay(); };
  });
  w.body.querySelectorAll("[data-rs]").forEach((cb) => {
    cb.checked = !!S.raceSet[cb.dataset.rs];
    cb.onchange = () => { S.raceSet[cb.dataset.rs] = cb.checked; store.set("spotter.raceSound", S.raceSet); if (cb.checked) unlockSound(); };
  });
  const pin = $("[data-x=pilot]", w.body);
  pin.oninput = () => {
    if (!S.video) return;
    store.set(pilotKey(S.video), pin.value.trim());
    if (S.solo) S.solo.pilots[0].name = soloName();
    comboMemo = null;
  };
  syncPilotBox();
  w.body.querySelector("[data-a=rate]").onchange = (e) => { v.playbackRate = +e.target.value; };
  w.body.querySelectorAll("button[data-a]").forEach((b) => b.onclick = () => control(b.dataset.a));
  v.addEventListener("play", () => { $("[data-a=play]", w.body).textContent = "⏸"; });
  v.addEventListener("pause", () => { $("[data-a=play]", w.body).textContent = "▶"; });
  v.addEventListener("seeked", tick);
  v.addEventListener("play", unlockSound);
  v.addEventListener("loadedmetadata", () => {
    // the browser knows the real length (a file's frame count can be wrong, e.g. variable frame rate)
    if (S.video && Number.isFinite(v.duration) && v.duration > 0) { S.video.duration = v.duration; tick(); }
  });
  v.addEventListener("loadeddata", tick);
  // spinner while the video loads or buffers (slow over a tunnel)
  const spin = $(".vspin", w.body);
  // the hidden attribute (not only a class) keeps it out of sight even with a stale or missing stylesheet
  const busy = (on, msg = "Loading video…") => { spin.hidden = !on; $("span", spin).textContent = msg; };
  v.addEventListener("loadstart", () => busy(!!v.getAttribute("src")));
  v.addEventListener("waiting", () => { if (!v.paused) busy(true, "Buffering…"); });   // paused on a frame: nothing to wait for
  // loadeddata too: iPhone Safari doesn't fire canplay before the first tap on play
  for (const e of ["loadeddata", "canplay", "playing", "seeked", "emptied"]) v.addEventListener(e, () => busy(false));
  v.addEventListener("error", () => { if (v.getAttribute("src")) busy(true, "Couldn't load the video"); spin.classList.add("err"); });
  v.addEventListener("loadstart", () => spin.classList.remove("err"));
  // browsers don't always send the events above (seeks, iPhone Safari): hide it once a frame is there
  const spinCheck = setInterval(() => {
    if (!spin.hidden && !spin.classList.contains("err") && v.readyState >= 2 && !v.seeking) busy(false);
  }, 400);
  const ro = new ResizeObserver(() => drawOverlay());
  ro.observe(c);
  w.overlay = c;
  w.cleanup = () => { ro.disconnect(); clearInterval(spinCheck); v.pause(); S.videoEl = null; };
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
  if (a === "play") { if (v.paused) { unlockSound(); v.play(); } else v.pause(); return; }
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
  if (S.combo && S.ov.hud) drawComboOverlay(g, box);
  const co = S.raceCallout;                     // the last call-out, a few seconds, at the bottom
  if (S.mode === "race" && co && performance.now() - co.wall < 4000) {
    g.font = "bold 18px -apple-system, sans-serif";
    const txt = `🗣 ${co.text}`, tw = g.measureText(txt).width;
    g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(ox + (vw * s - tw) / 2 - 10, oy + vh * s - 44, tw + 20, 30);
    g.fillStyle = "#ffd36b"; g.fillText(txt, ox + (vw * s - tw) / 2, oy + vh * s - 40);
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
  const was = S.t;
  if (S.videoEl) S.t = S.videoEl.currentTime;
  raceSounds(was, S.t, S.videoEl && !S.videoEl.paused);
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
    return ((e.clientX - r.left) / r.width) * (S.mode === "live" ? liveSpan() : S.video?.duration || 0);
  };
  c.onmousedown = (e) => { if (S.mode !== "live" && S.video) { S.videoEl?.pause(); seek(tOf(e)); } };
  c.onmousemove = (e) => { hover = tOf(e); w.draw(); };
  c.onmouseleave = () => { hover = null; w.draw(); };
  w.draw = () => {
    const dpr = devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#181b20"; g.fillRect(0, 0, cw, ch);
    if (S.mode === "live") return drawLiveTimeline(g, cw, ch, hover);
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

// Live: the timeline is built as the run goes — passes as they're detected and named, laps as they close
const liveNow = () => {
  const st = S.live.stats, P = S.live.passes, last = P[P.length - 1];
  let t = st ? st.t : 0;
  if (LIVE_ON() && st?.wall) t += (performance.now() - st.wall) / 1000 * (S.live.speed || 1);   // between stats
  return Math.max(t, last ? last.t : 0);
};
const liveSpan = () => {                          // a file: the whole video; a live feed: grows with the run
  const d = S.live.stats?.duration;
  return d > 0 ? d : Math.max(60, Math.ceil(liveNow() / 30 + 1) * 30);
};

// gates a closed lap has no pass for: [{gate, t}], t between the passes of the gates around it
function liveLapMissed(L) {
  const n = S.live.gates || 0;
  const inLap = S.live.passes.filter((p) => p.gate_id >= 1 && p.t >= L.t0 - 1e-3 && p.t < L.t1 - 1e-3);
  const seen = new Map(inLap.map((p) => [p.gate_id, p.t]));
  seen.set(1, L.t0);
  const out = [];
  for (let gid = 2; gid <= n; gid++) {
    if (seen.has(gid)) continue;
    let a = gid - 1; while (a > 1 && !seen.has(a)) a--;
    let b = gid + 1; while (b <= n && !seen.has(b)) b++;
    const ta = seen.get(a), tb = b <= n ? seen.get(b) : L.t1;
    out.push({ gate: gid, t: ta + (tb - ta) * (gid - a) / (b - a) });
  }
  return out;
}

function drawLiveTimeline(g, cw, ch, hover) {
  const L = S.live, D = liveSpan(), now = liveNow(), X = (t) => (t / D) * cw, mid = ch * 0.55;
  g.font = "10px -apple-system, sans-serif";
  g.textBaseline = "top";
  if (!L.passes.length && !LIVE_ON()) {
    g.fillStyle = "#9aa3ad"; g.fillText("The live run's passes and laps show up here as they happen.", 10, 20); return;
  }
  if (liveMulti()) { g.fillStyle = "#9aa3ad"; g.fillText("All pilots of a combo: see the Race map and the standings.", 10, 20); return; }
  g.fillStyle = "#2a2f37"; g.fillRect(0, mid - 1, cw, 2);
  const step = D > 300 ? 60 : D > 120 ? 20 : 10;
  g.fillStyle = "#59616b";
  for (let s = 0; s <= D; s += step) { g.fillRect(X(s), ch - 6, 1, 6); g.fillText(`${s}s`, X(s) + 2, ch - 14); }
  for (const lap of L.laps || []) {                         // closed laps
    const miss = liveLapMissed(lap);
    g.fillStyle = "rgba(255,176,32,.08)"; g.fillRect(X(lap.t0), 0, X(lap.t1) - X(lap.t0), ch);
    g.fillStyle = "#ffb020"; g.fillRect(X(lap.t0), 2, 2, ch - 4); g.fillRect(X(lap.t1) - 1, 2, 2, ch - 4);
    g.fillStyle = miss.length ? "#ff5c5c" : "#ffb020";
    g.fillText(`${miss.length ? "🚩 " : ""}L${lap.lap} ${lap.estimated ? "~" : ""}${fmt(lap.dt)}s`, X(lap.t0) + 4, 4);
    g.fillStyle = "#ff5c5c";
    for (const m of miss) { g.fillRect(X(m.t) - 1, mid - 10, 2, 20); g.fillText(`G${m.gate}✕`, X(m.t) - 8, mid - 24); }
  }
  if (L.lap_start && (!L.laps.length || L.lap_start.t >= L.laps[L.laps.length - 1].t1 - 1e-3)) {   // the lap running
    const t0 = L.lap_start.t, t1 = Math.max(t0, now);
    g.fillStyle = "rgba(255,176,32,.04)"; g.fillRect(X(t0), 0, X(t1) - X(t0), ch);
    g.fillStyle = "#ffb020"; g.fillRect(X(t0), 2, 2, ch - 4);
    g.fillText(`L${L.lap_start.lap}${L.lap_start.estimated ? " ~" : ""} ${LIVE_ON() ? fmt(t1 - t0, 1) + "s…" : "unfinished"}`, X(t0) + 4, 4);
  }
  for (const p of L.passes) {                               // passes: green named, grey not a pass, blue not named
    const named = p.gate_id >= 1, g1 = p.gate_id === 1;
    g.fillStyle = named ? "#4cd97b" : p.gate_id === 0 ? "#666" : "#5aa0ff";
    g.fillRect(X(p.t) - 1, mid - (g1 ? 16 : 10), 2, g1 ? 32 : 20);
    if (named && cw / D > 6) g.fillText(String(p.gate_id), X(p.t) - 3, mid + 12);
  }
  g.fillStyle = "#ff4d4d"; g.fillRect(X(now) - 1, 0, 2, ch);           // now
  if (hover !== null) { g.fillStyle = "#e6e6e6"; g.fillText(`${fmt(hover)}s`, Math.min(X(hover) + 4, cw - 40), ch / 2 - 30); }
}

// ─────────────────────────────────────────────────────────────
// Race stats + gate splits
// ─────────────────────────────────────────────────────────────

function needRun(w) {
  if (!S.video) { w.body.innerHTML = `<div class="empty">Pick a video.</div>`; return true; }
  if (S.video.role === "combo") {
    w.body.innerHTML = `<div class="empty">${esc(S.video.name)} is a combo view: its pilots race each other in <b>Race — standings</b>
      and the <b>Race map</b> (Windows ▾). A pilot's own laps and splits: pick its …_CH video.</div>`;
    return true;
  }
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

const UPLOAD_CHUNK = 32 << 20;         // bytes per request: a Cloudflare tunnel refuses bodies over 100 MB

// upload a video from this device in chunks; onProgress(0…1). Returns {name, folder}.
async function uploadVideo(track, file, role, onProgress = () => {}) {
  const id = [...crypto.getRandomValues(new Uint8Array(12))].map((b) => b.toString(16).padStart(2, "0")).join("");
  const n = Math.max(1, Math.ceil(file.size / UPLOAD_CHUNK));
  let r;
  for (let i = 0; i < n; i++) {
    const q = new URLSearchParams({ role, name: file.name, upload_id: id, index: i, last: i === n - 1 ? 1 : 0, size: file.size });
    r = await api(`/api/tracks/${encodeURIComponent(track)}/upload?${q}`,
      { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file.slice(i * UPLOAD_CHUNK, (i + 1) * UPLOAD_CHUNK) });
    onProgress((i + 1) / n);
  }
  return r;
}

function uploadForm(role) {
  return `<div class="row"><button data-upopen="${role}">⬆ Upload a ${role} video from this device</button></div>`;
}

function wireUpload(w) {
  w.body.querySelectorAll("[data-upopen]").forEach((b) => b.onclick = () => openUploader(b.dataset.upopen));
}

function videoRows(vids, withLearn) {
  if (!vids.length) return `<div class="empty">No videos.</div>`;
  let h = `<table><tr><th>Video</th><th>Role</th><th>Length</th><th>fps</th><th>Status</th><th></th></tr>`;
  for (const v of vids) {
    const job = S.jobs.find((j) => j.video === v.name && ["running", "queued"].includes(j.status));
    const st = job ? `<span class="badge warn">${fmt(job.progress, 0)}%</span>`
      : v.analysed ? `<span class="badge ok">analysed</span>` : v.learn_analysed ? `<span class="badge">learn run</span>` : `<span class="badge">new</span>`;
    h += `<tr><td>${esc(v.name)}</td><td>${v.role}</td><td>${fmt(v.duration, 0)}s</td><td>${v.fps}</td><td>${st}</td>
      <td><button data-open="${esc(v.name)}">Open</button>${withLearn || (S.league && v.analysed) || v.role === "combo" ? "" : ` <button data-an="${esc(v.name)}" ${job ? "disabled" : ""}>Analyse</button>`}
      ${!S.league && v.role !== "combo" && v.w >= 1600 ? `<button data-split="${esc(v.name)}" ${job ? "disabled" : ""} title="A combo (quad) view of several pilots? One video per pilot (_CH1 … next to it, 30 fps); the combo moves to combo_videos/">⊞ Split</button>` : ""}
      ${S.league ? "" : `<button data-rm="${esc(v.name)}" data-folder="${v.folder}" ${job ? "disabled" : ""} title="Move out of the track (to removed_videos/)">✕ Remove</button>`}</td></tr>`;
  }
  return h + `</table>`;
}

function wireVideoRows(w) {
  w.body.querySelectorAll("[data-open]").forEach((b) => b.onclick = () => { $("#videoSel").value = b.dataset.open; selectVideo(b.dataset.open); });
  w.body.querySelectorAll("[data-an]").forEach((b) => b.onclick = () => {
    const v = curTrack()?.videos.find((x) => x.name === b.dataset.an);
    if (v?.analysed && !confirm(`${v.name} is already analysed.\nRun the analysis again over it? Its current results are replaced.`)) return;
    startJob({ kind: "analyse", track: S.track, video: b.dataset.an });
  });
  w.body.querySelectorAll("[data-split]").forEach((b) => b.onclick = () => {
    if (!confirm(`Split ${b.dataset.split} into one video per pilot (a channel with no pilot is left out)?\nThe combo itself moves to combo_videos/.`)) return;
    startJob({ kind: "split", track: S.track, video: b.dataset.split });
  });
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
  h += `<h4>Add a race video</h4>${uploadForm("race")}` + (S.league ? "" : `<h4>Add a learn video</h4>${uploadForm("learn")}`);
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
  const M = S.model || "v2";
  if (L) {
    h += `<h4>Learned track · model ${esc(M)}</h4>`
      + (L.own ? "" : `<div class="hint warn">Model ${esc(M)} hasn't learned ${esc(t.name)} itself: it uses v2's learned track below. 🎓 Learn track here makes ${esc(M)}'s own (v2's stays).</div>`)
      + `<div class="hint">${L.n_gates} gates · method <b>${esc(L.method)}</b> · lap ≈ ${fmt(L.lap_seconds, 1)}s ·
      ${L.extras || 0} extras · learned from ${esc((L.learned_from || []).join(", "))}</div>`;
    const Y = L.layout;                                 // the map: made right after learning, better with each analysis
    h += `<h4>Map · made after learning</h4><div class="hint">${Y ? `Estimated ${esc(Y.made || "")} from <b>${Y.laps_used}</b> clean
      lap${Y.laps_used === 1 ? "" : "s"}${Y.laps_used >= 2 ? ` (closed ${fmt(100 * (Y.closure || 0), 0)}% of a lap)` : " — a circle until laps are analysed"}
      · ${Y.moved} gate${Y.moved === 1 ? "" : "s"} moved by hand.` : "Not made yet: it's made on the next 🎓 Learn track, or open it below."}
      Made again after each learn and each analysed video (more clean laps, a better shape); the gates you moved stay.</div>
      <div class="row"><button id="openLayout" title="Where the gates roughly are: drag them where they really are — the race map and 3D view draw on it">🗺 Track layout</button>
      <button id="open3d" title="The race in 3D on this layout">🧊 Race 3D</button></div>`;
  }
  const nRuns = t.videos.filter((v) => v.analysed || v.learn_analysed).length;
  if ((L && L.own) || nRuns) {
    h += `<div class="row"><button id="resetLearn" ${job ? "disabled" : ""}>🗑 Delete ${esc(M)}'s learned track</button>
      <label title="Every analysis of this track by model ${esc(M)} (${M === "v2" ? "runs/" : "runs_" + esc(M) + "/"}): race and learn videos"><input type="checkbox" id="resetRuns"> and its analyses (${nRuns})</label></div>
      <div class="hint">Only model ${esc(M)}'s: the other model's learned track and analyses stay. Moved to learn_backups/, not deleted for good.</div>`;
  }
  w.body.innerHTML = h;
  const ol = $("#openLayout", w.body);
  if (ol) ol.onclick = () => { openWin("layout"); saveLayout(); };
  const o3 = $("#open3d", w.body);
  if (o3) o3.onclick = () => { openWin("race3d"); saveLayout(); };
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
    const runs = $("#resetRuns", w.body).checked, M = S.model || "v2";
    if (!confirm(`Delete model ${M}'s learned track of ${t.name}${runs ? " and all its analyses by " + M : ""}?\n`
      + `The other model's stay. The files are moved to learn_backups/, not deleted for good.`)) return;
    try {
      const r = await api(`/api/tracks/${encodeURIComponent(t.name)}/reset`, { json: { analyses: runs } });
      status(r.backup ? `Deleted ${r.model}'s ${r.moved.join(", ")} — kept in ${r.backup}` : `Model ${r.model} had nothing of its own to delete.`);
      await loadTracks(); await loadGates();
      if (S.video) await selectVideo(S.video.name);
    } catch (e) { status(e.message, true); }
  };
}

// Track layout: where the gates roughly are — estimated after learning from how the camera turned between them
// (learn_layout.py); drag a gate where it really is (kept, also when it's estimated again). The race map draws on it.
const layoutImgs = {};
function initLayout(w) {
  w.body.style.padding = "0";
  w.body.innerHTML = `<div class="vidwrap"><div class="vidctl">
      <button data-a="mirror" title="Mirror it, left ↔ right">⇋ Mirror</button>
      <button data-a="rotate" title="Turn it a quarter, clockwise">⟳ Rotate</button>
      <button data-a="reset" title="Every gate back where it was estimated">↺ Undo my moves</button>
      <button data-a="estimate" title="Estimate it again from the analysed videos (the gates you moved stay)">↻ Estimate again</button>
      <label><input type="checkbox" data-x="imgs" checked> gate pictures</label>
      <span class="hint" data-x="msg"></span></div>
    <div class="vidbox" style="background:#181b20"><canvas class="timeline" style="cursor:default"></canvas></div></div>`;
  const c = $("canvas", w.body), msg = (t, err) => { const m = $("[data-x=msg]", w.body); m.textContent = t; m.classList.toggle("err", !!err); };
  let D = null, fit = null, drag = null, hover = null;
  const url = () => `/api/tracks/${encodeURIComponent(S.track)}/layout`;
  w.load = async () => {
    if (drag) return;
    if (!S.track || !curTrack()?.learned) { D = null; w.draw(); return; }
    const d = await api(url()).catch((e) => { msg(e.message, true); return null; });
    if (drag) return;
    D = d;
    if (D) msg(`${D.method} · moved by hand: ${D.gates.filter((q) => q.manual).length}`);
    w.draw();
  };
  const post = async (body, label) => {
    msg(label);
    try { D = await api(url(), { json: body }); msg(`${D.method} · moved by hand: ${D.gates.filter((q) => q.manual).length}`); }
    catch (e) { msg(e.message, true); }
    w.draw(); render(["livemap"]); refreshRaceLayout();
  };
  w.body.querySelectorAll("button[data-a]").forEach((b) => b.onclick = () => {
    const a = b.dataset.a;
    if (a === "reset" && !confirm("Put every gate back where it was estimated?")) return;
    post(a === "mirror" || a === "rotate" ? { transform: a } : { [a]: true }, a === "estimate" ? "estimating…" : "…");
  });
  $("[data-x=imgs]", w.body).onchange = () => w.draw();
  const near = (e) => {
    if (!D || !fit) return null;
    const r = c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 14;
    fit.P.forEach((p, i) => { const d = Math.hypot(p[0] - x, p[1] - y); if (d < bd) { bd = d; best = i; } });
    return best;
  };
  c.onpointerdown = (e) => {
    const i = near(e);
    if (i === null || S.league) return;
    drag = { i, fit };
    c.setPointerCapture(e.pointerId);
  };
  c.onpointermove = (e) => {
    if (!drag) { const h = near(e); if (h !== hover) { hover = h; c.style.cursor = h !== null ? "grab" : "default"; w.draw(); } return; }
    const r = c.getBoundingClientRect(), f = drag.fit, q = D.gates[drag.i];
    q.x = (e.clientX - r.left - f.ox) / f.k; q.y = (e.clientY - r.top - f.oy) / f.k; q.manual = true;
    c.style.cursor = "grabbing";
    w.draw();
  };
  c.onpointerup = () => {
    if (!drag) return;
    const q = D.gates[drag.i];
    drag = null;
    post({ move: { [q.id]: [q.x, q.y] } }, `G${q.id} moved`);
  };
  w.draw = () => {
    const dpr = devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#181b20"; g.fillRect(0, 0, cw, ch);
    g.font = "12px -apple-system, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
    if (!D) {
      g.fillStyle = "#9aa3ad";
      g.fillText(!S.track ? "Pick a track in the top bar." : "Learn the track first: its layout is estimated then.", cw / 2, ch / 2);
      return;
    }
    const imgs = $("[data-x=imgs]", w.body).checked, pad = imgs ? 70 : 34, n = D.gates.length;
    fit = drag ? drag.fit : layoutFit(D.gates, cw, ch, pad);
    if (drag) fit = { ...fit, P: D.gates.map((q) => [fit.ox + q.x * fit.k, fit.oy + q.y * fit.k]) };
    const legs = D.legs?.length === n ? D.legs : Array(n).fill(1), lap = legs.reduce((a, b) => a + b, 0);
    const at = [0]; for (let i = 0; i < n - 1; i++) at.push(at[i] + legs[i] / lap);
    g.strokeStyle = "#2f353e"; g.lineWidth = 12; g.beginPath();
    for (let i = 0; i <= 300; i++) { const [x, y] = splineAt(fit.P, at, i / 300); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.closePath(); g.stroke();
    g.fillStyle = "#59616b";                               // which way it's flown: an arrow mid-leg
    for (let i = 0; i < n; i++) {
      const f = at[i] + legs[i] / lap / 2, [x, y] = splineAt(fit.P, at, f), [x2, y2] = splineAt(fit.P, at, f + 0.003);
      const a = Math.atan2(y2 - y, x2 - x);
      g.save(); g.translate(x, y); g.rotate(a); g.beginPath(); g.moveTo(6, 0); g.lineTo(-4, -4); g.lineTo(-4, 4); g.fill(); g.restore();
    }
    D.gates.forEach((q, i) => {
      const [x, y] = fit.P[i], dl = Math.hypot(x - fit.cx, y - fit.cy) || 1, ux = (x - fit.cx) / dl, uy = (y - fit.cy) / dl;
      if (imgs && q.img) {                                 // its picture, outside the loop
        const im = layoutImgs[q.img] || (layoutImgs[q.img] = Object.assign(new Image(), { src: q.img }));
        if (!im.complete) im.onload = () => w.draw();
        else if (im.naturalWidth) {
          const h = 44, ww = Math.min(70, h * im.naturalWidth / im.naturalHeight), bx = x + ux * 52 - ww / 2, by = y + uy * 46 - h / 2;
          g.drawImage(im, bx, by, ww, h);
          g.strokeStyle = i === 0 ? "#ffb020" : "#59616b"; g.lineWidth = 1; g.strokeRect(bx, by, ww, h);
        }
      }
      g.beginPath(); g.arc(x, y, i === hover || (drag && drag.i === i) ? 10 : 8, 0, 2 * Math.PI);
      g.fillStyle = i === 0 ? "#ffb020" : q.manual ? "#e6e9ee" : "#3caaff"; g.fill();
      g.fillStyle = "#101216"; g.font = "bold 10px -apple-system, sans-serif"; g.fillText(String(q.id), x, y + 0.5);
      if (!imgs) { g.fillStyle = "#9aa3ad"; g.font = "11px -apple-system, sans-serif"; g.fillText(`G${q.id} ${q.type || ""}`, x + ux * 22, y + uy * 18); }
    });
    g.textAlign = "left"; g.fillStyle = "#9aa3ad"; g.font = "11px -apple-system, sans-serif";
    g.fillText(`${S.track} · ${n} gates · drag a gate where it really is · blue: estimated, white: moved by you, orange: G1 (start)`, 8, 12);
  };
  const ro = new ResizeObserver(() => w.draw());
  ro.observe(c);
  w.cleanup = () => ro.disconnect();
  w.load();
}

// the layout changed: the race map of the video picked, and Live's, draw on the new one
async function refreshRaceLayout() {
  const d = S.track ? await api(`/api/tracks/${encodeURIComponent(S.track)}/layout`).catch(() => null) : null;
  if (S.solo) S.solo.layout = d;
  if (S.combo) S.combo.layout = d;
  if (S.live?.track === S.track && S.live.layout) S.live.layout = d;
  comboMemo = null;
  render(["livemap"]);
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
      ${["running", "queued"].includes(j.status) && !S.league ? `<button data-cancel="${j.id}">Cancel</button>` : ""}</span></div>
      <div class="bar"><i style="width:${j.progress || 0}%"></i></div><div class="hint">${esc(j.detail || "")}</div>
      <details data-id="${j.id}" ${open.has(j.id) || j.status === "failed" ? "open" : ""}><summary class="hint">log</summary><pre class="log">${esc(j.log.slice(-60).join("\n"))}</pre></details></div>`;
  }
  w.body.innerHTML = h;
  w.body.querySelectorAll("[data-cancel]").forEach((b) => b.onclick = () => api(`/api/jobs/${b.dataset.cancel}/cancel`, { method: "POST" }).then(pollJobs));
}

// ─────────────────────────────────────────────────────────────
// Live: the real-time spotter on one feed (a video file played in real time, a capture device or a stream).
// The server runs it and sends every pass the moment it's detected. With a learned track (top bar) each pass is
// then named (G1…GN) and laps run G1 to G1; this page plays a sound per gate and speaks the lap times.
// ─────────────────────────────────────────────────────────────

S.live = { status: "idle", passes: [], stats: null, laps: [], lap_start: null };
S.liveSet = { voice: true, lapmode: "track", where: "mac", boxStride: 2, boxPace: "box", ...store.get("spotter.live", { kind: "file", device: 0, imgsz: 320, speed: 1, sound: true }) };
if (!["file", "device", "listen"].includes(S.liveSet.kind)) S.liveSet.kind = "listen";   // "Stream URL" was dropped (2.3.3)
delete S.liveSet.url;
let liveES = null, audioCtx = null, liveTimer = null, lapWall = null;
const LIVE_ON = () => ["deploying", "loading", "running"].includes(S.live.status);
S.preview = { status: "off" };                // the picked live feed, shown before ▶ Start (server: PREVIEW)
const PV_ON = () => ["opening", "on"].includes(S.preview.status);
const liveFeedPicked = () => S.liveSet.where === "mac" && ["device", "listen"].includes(S.liveSet.kind);

// the preview shows the combo channel picked (once it's open; again when it's reopened)
let comboBusy = false;
async function previewCombo() {
  const want = Math.max(0, S.liveSet.combo || 0);                 // all pilots: the whole picture
  if (comboBusy || LIVE_ON() || S.preview.status !== "on" || (S.preview.combo?.ch || 0) === want) return;
  comboBusy = true;
  try { S.preview = await api("/api/live/preview/combo", { json: { combo: want } }); }
  catch (e) { status(e.message, true); S.liveSet.combo = 0; saveLiveSet(); const sel = wins.livefeed && $("[data-l=combo]", wins.livefeed.body); if (sel) sel.value = "0"; }
  finally { comboBusy = false; syncLiveFeed(); }
}

// show the picked live feed now (▶ Start then spots / saves it); a video file or the box: no preview
async function requestPreview() {
  if (LIVE_ON()) return;
  const L = S.liveSet;
  if (!liveFeedPicked()) {
    if (PV_ON()) api("/api/live/preview/stop", { method: "POST" }).catch(() => {});
    return;
  }
  const body = L.kind === "device" ? { kind: "device", device: L.device, device_id: L.deviceId || undefined }
    : { kind: "listen", proto: L.proto || "rtmp" };
  try { S.preview = await api("/api/live/preview", { json: body }); syncLiveFeed(); }
  catch (e) { status(e.message, true); }
}

function connectLive() {
  if (liveES) return;
  liveES = new EventSource("/api/live/events");
  liveES.onmessage = (m) => onLiveEvent(JSON.parse(m.data));
  liveES.onerror = () => { if (S.mode === "live") status("Live: reconnecting to the server…", true); };
  clearInterval(liveTimer);
  let tick = 0;
  liveTimer = setInterval(() => {
    if (!LIVE_ON()) return;
    renderLiveClock();
    if (++tick % 3 === 0 && wins.timeline) wins.timeline.draw();          // the playhead moves between events
    if (wins.livemap && liveMulti()) wins.livemap.draw();                 // the pilots move between gates
  }, 100);
}

function disconnectLive() {
  if (liveES) { liveES.close(); liveES = null; }
  clearInterval(liveTimer);
}

const liveNamed = () => !!S.live.gates;          // a learned track: passes get gate names, laps are timed

function onLiveEvent(e) {
  const L = S.live;
  if (e.type === "preview") { S.preview = e; syncLiveFeed(); previewCombo(); return; }
  if (e.type === "snapshot") {
    S.live = { ...e, passes: e.passes || [], laps: e.laps || [] };
    S.preview = e.preview || { status: "off" };
    livePlaces = newPlaces();
    if (e.pilots) raceCallouts(true);                     // opened mid-race: the places as they are, nothing said
    const st = S.live.stats, ls = S.live.lap_start;      // page opened mid-run: the lap clock from the feed time
    lapWall = ls && st ? performance.now() - 1000 * (st.t - ls.t) / (S.live.speed || 1) : null;
  } else if (e.type === "state") {
    const was = L.status;
    Object.assign(L, e);
    if (e.status === "running" && was !== "running") status(`Live: ${e.source}`);
    if (e.status === "error") status(`Live: ${e.error}`, true);
    if (e.status === "loading" && was !== "loading") { L.passes = []; L.laps = []; L.starts = []; L.lap_start = null; lapWall = null; livePlaces = newPlaces(); }
  } else if (e.type === "pass") {
    e.wall = performance.now();
    L.passes.push(e);
    if (e.pilot == null) sfx("pass");           // every pass: a beep; a named gate adds its arcade sound (4 pilots: quiet)
    flashLive();
  } else if (e.type === "gate") {
    const p = L.passes.find((q) => q.n === e.n);
    if (p) Object.assign(p, { gate_id: e.gate_id, conf: e.conf, wall: performance.now() });
    if (e.pilot != null) raceCallouts();
    else if (e.gate_id) sfx(e.gate_id === 1 ? "start" : "gate");
  } else if (e.type === "start" && e.pilot != null) {
    (L.starts = L.starts || []).push(e);
    if (e.lap === 1 && L.starts.filter((x) => x.lap === 1).length === 1) say("Go!", true);
    raceCallouts();
  } else if (e.type === "lap" && e.pilot != null) {
    L.laps.push(e);
    const mine = L.laps.filter((l) => l.pilot === e.pilot), best = Math.min(...mine.map((l) => l.dt));
    sfx("start");
    say(`${pilotName(e.pilot)}, lap ${e.lap}, ${e.estimated ? "about " : ""}${e.dt.toFixed(1)}${e.dt <= best && mine.length > 1 ? ", best lap" : ""}`, true);
  } else if (e.type === "start") {
    L.lap_start = e;
    lapWall = performance.now() - 1000 * (e.at - e.t) / (L.speed || 1);
    if (e.lap === 1) say(L.pilot ? `${L.pilot}, go!` : "Go!");
  } else if (e.type === "lap") {
    L.laps.push(e);
    const best = Math.min(...L.laps.map((l) => l.dt));
    say(`${L.pilot ? L.pilot + ", " : ""}Lap ${e.lap}, ${e.estimated ? "about " : ""}${e.dt.toFixed(1)}${e.dt <= best && L.laps.length > 1 ? ", best lap" : ""}`);
  } else if (e.type === "stats") { e.wall = performance.now(); L.stats = e; if (e.recording) L.recording = e.recording; }
  else if (e.type === "end") {
    L.end = e;
    status(`Live ${e.stopped ? "stopped" : "ended"}: ${e.passes} passes`
      + (liveNamed() ? ` · ${e.laps} laps${e.best_lap ? `, best ${fmt(e.best_lap)}s` : ""}` : "")
      + ` · ${e.frames_done}/${e.frames_read} frames analysed · delay ${e.delay_ms_median} ms (95%: ${e.delay_ms_p95} ms)`);
  }
  syncLiveFeed();
  render(e.type === "stats" ? ["livestats", "timeline"] : ["livestats", "livelog", "timeline"]);
}

// sounds and voice (the browser lets a page play sound only after a click on it: Start does that)
function audio() {
  if (!audioCtx) { try { audioCtx = new AudioContext(); } catch { return null; } }
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

// notes: [frequency Hz, start s, length s, wave]
const SFX = {
  pass: [[1320, 0, 0.12, "triangle"]],                                          // a pass was detected: beep
  gate: [[988, 0, 0.06], [1319, 0.06, 0.2]],                                   // the gate was named: coin
  start: [[523, 0, 0.07], [659, 0.07, 0.07], [784, 0.14, 0.07], [1047, 0.21, 0.28]],   // power-up: start / finish
};

function sfx(kind, race = false) {
  if (!(race ? S.raceSet.sound : S.liveSet.sound)) return;
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

function say(text, queue = false, race = false) {
  if (race) S.raceCallout = { text, wall: performance.now() };   // Race mode: shown over the video
  else { S.live.callout = { text, wall: performance.now() }; render(["livestats"]); }   // shown in the laps window too
  if (!(race ? S.raceSet.voice : S.liveSet.voice)) return;
  const ss = window.speechSynthesis;
  if (!ss) return status("This browser can't speak: lap times are shown in the laps window only.", true);
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.1;
  u.lang = "en-US";
  const v = ss.getVoices().find((x) => x.lang === "en-US" && x.localService) || ss.getVoices().find((x) => x.lang?.startsWith("en"));
  if (v) u.voice = v;
  u.onerror = (e) => { if (e.error !== "interrupted" && e.error !== "canceled") status(`Voice call-out failed: ${e.error}`, true); };
  utterance = u;
  ss.resume();                                  // Chrome can leave the queue paused
  if (queue) return ss.speak(u);                // several pilots: every call-out, one after another
  if (ss.speaking || ss.pending) {              // the newest call-out wins; Chrome drops a speak() right after cancel()
    ss.cancel();
    setTimeout(() => ss.speak(u), 80);
  } else ss.speak(u);
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

function flashLive() {
  for (const id of ["livefeed", "livestats"]) {
    const el = wins[id]?.el;
    if (!el) continue;
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  }
}

function saveLiveSet() { store.set("spotter.live", S.liveSet); }

// laps need a learned track in the top bar: without one (or with "none") the choice is "off"
const trackLearned = () => !!curTrack()?.learned;
const lapMode = () => (S.liveSet.lapmode === "track" && trackLearned() ? "track" : "off");

function syncLapMode() {
  const sel = wins.livefeed && $("[data-l=lapmode]", wins.livefeed.body);
  if (!sel) return;
  const opt = $("option[value=track]", sel);
  opt.disabled = !trackLearned();
  opt.textContent = !S.track ? "Laps: learned track (no track picked)" : trackLearned() ? `Laps: ${S.track}` : `Laps: ${S.track} isn't learned`;
  sel.value = lapMode();
}

function initLiveFeed(w) {
  w.body.style.padding = "0";
  const L = S.liveSet;
  w.body.innerHTML = `<div class="vidwrap"><div class="vidctl">
      <select data-l="where" title="Where the spotter runs: on this Mac, or on the naco media box (the Mac only shows and controls it)">
        <option value="mac">Run on: this Mac</option><option value="box">Run on: media box (alpha)</option></select>
      <span data-where="box" class="hint">video file (top bar) →</span>
      <select data-l="kind" data-where="mac" title="Where the feed comes from">
        <option value="file">Video file (top bar), real time</option>
        <option value="device">Capture device (USB / HDMI)</option>
        <option value="listen">Receive a stream (sent to this Mac)</option></select>
      <span data-for="file"><select data-l="speed" title="Playback speed of the file">
        <option value="0.5">×0.5</option><option value="1">×1</option><option value="2">×2 (drops frames)</option></select></span>
      <span data-for="device"><select data-l="device"></select></span>
      <span data-for="listen" title="This Mac is the server: set the sender (the media box's stream out) to this address, then ▶ Start here — it waits for the stream, and again if the sender drops off">
        <select data-l="proto"><option value="rtmp">RTMP</option><option value="rtsp">RTSP</option></select>
        send to <select data-x="host"></select><input data-x="addr" size="34" readonly><button data-a="copyaddr" title="Copy the address">📋</button></span>
      <span data-for="device listen"><input data-l="pilot" size="12" maxlength="40" placeholder="Pilot name"
        title="Called out with every lap time (e.g. “Niv, lap 3, 24.1”); also names a saved feed"></span>
      <span data-for="device listen" data-where="mac"><label title="Save this feed (every frame, full size) under a new track's name, to learn the track from later: <datasets>/live_recordings/<track>/"><input type="checkbox" data-l="record"> ⏺ save feed</label>
        <input data-l="recTrack" size="14" maxlength="60" placeholder="new track name" title="The new track's name: the feed is saved in live_recordings/<this name>/"></span>
      <select data-l="combo" data-where="mac" title="A combo (quad) view of several pilots: show, spot and save one pilot's channel">
        <option value="0">Combo: off</option><option value="1">Combo: CH1 ◰</option><option value="2">Combo: CH2 ◳</option>
        <option value="3">Combo: CH3 ◱</option><option value="4">Combo: CH4 ◲</option>
        <option value="-1">Combo: all pilots — race</option></select>
      <span data-x="names" title="The pilots' names, by channel (called out with their laps and the lead)">${[1, 2, 3, 4].map((c) =>
        `<input data-name="${c}" size="8" maxlength="30" placeholder="CH${c} pilot" style="border-left:4px solid ${PILOT_COLORS[c - 1]}">`).join("")}</span>
      <select data-l="lapmode" title="Gate names and laps need a learned track in the top bar; without one: beeps only">
        <option value="track">Laps: learned track (top bar)</option>
        <option value="off">Laps: off — beeps only</option></select>
      <span data-where="box"><select data-l="boxStride" title="How many frames the box analyses (it can't do all 30 a second)">
        <option value="2">every 2nd frame (15 fps)</option><option value="3">every 3rd frame (10 fps)</option>
        <option value="1">every frame (30 fps)</option></select>
      <select data-l="boxPace" title="At the box's pace: no frame is dropped, the video plays slower than real time (to judge how well it spots). Real time: frames it can't take in time are dropped, as live">
        <option value="box">at the box's pace</option><option value="realtime">real time (drops)</option></select>
</span>
      <select data-l="imgsz" title="Detector input size: smaller is faster (the offline analysis uses 640)">
        <option value="320">det 320 (best)</option><option value="416">det 416</option><option value="640">det 640 (too slow live)</option></select>
      <button class="primary" data-a="start">▶ Start</button><button data-a="stop">■ Stop</button>
      <label title="A beep on every gate pass; with a learned track also a coin when the gate is named, a fanfare on the start gate"><input type="checkbox" data-l="sound"> 🔊 sound</label>
      <label title="Lap times spoken out loud (needs a learned track)"><input type="checkbox" data-l="voice"> 🗣 voice</label>
      <button data-a="test" title="Play the sounds and a lap call-out">test</button>
    </div>
    <div class="vidctl boxcfg" hidden>
      <span>Media box</span><input data-b="user" size="6" placeholder="user" title="SSH user on the box">@<input data-b="host" size="16" placeholder="host or IP" title="The box's address (host name or IP): 🔍 Find looks for it">
      <button data-a="boxsave">Save</button><button data-a="boxfind" title="Look for boxes on this network">🔍 Find</button>
      <button data-a="boxcheck" title="Is it reachable? Memory, temperature, what this app keeps there">Check</button>
      <button data-a="boxclean" title="Stop and delete everything this app put on the box (its temporary folder)">🧹 Clean box</button>
      <span class="hint" data-b="msg"></span><span data-b="found"></span>
    </div>
    <div class="vidbox"><img class="livefeed" alt=""><div class="vspin" hidden><i></i><span></span></div>
      <div class="vbadge" hidden></div></div></div>`;
  const q = (s) => $(s, w.body);
  q("[data-l=kind]").value = L.kind;
  q("[data-l=where]").value = L.where;
  q("[data-l=boxStride]").value = String(L.boxStride);
  q("[data-l=boxPace]").value = L.boxPace;
  q("[data-l=speed]").value = String(L.speed);
  q("[data-l=imgsz]").value = String(L.imgsz);
  q("[data-l=proto]").value = L.proto || "rtmp";
  q("[data-l=combo]").value = String(L.combo || 0);
  w.body.querySelectorAll("[data-name]").forEach((el) => {
    el.value = (L.names || [])[+el.dataset.name - 1] || "";
    el.oninput = () => { L.names = [1, 2, 3, 4].map((c) => q(`[data-name="${c}"]`).value.trim()); saveLiveSet(); };
  });
  q("[data-l=pilot]").value = L.pilot || "";
  q("[data-l=record]").checked = !!L.record;
  q("[data-l=recTrack]").value = L.recTrack || "";
  syncLapMode();
  q("[data-l=sound]").checked = !!L.sound;
  q("[data-l=voice]").checked = !!L.voice;
  const showKind = () => {
    const kind = L.where === "box" ? "file" : L.kind;          // the box runs a video file of the track
    w.body.querySelectorAll("[data-for]").forEach((el) => { el.hidden = !el.dataset.for.split(" ").includes(kind); });
    w.body.querySelectorAll("[data-where]").forEach((el) => {           // [data-for] already hid it by source
      if (el.dataset.where !== L.where) el.hidden = true; else if (!el.dataset.for) el.hidden = false;
    });
    q("[data-l=recTrack]").hidden = !L.record;
    const all = L.where === "mac" && +L.combo === -1;                  // every pilot: a name per channel
    q("[data-x=names]").hidden = !all;
    if (all) q("[data-l=pilot]").closest("span").hidden = true;
    const row = q(".boxcfg"), show = L.where === "box";
    if (show && row.hidden) loadBoxCfg();
    row.hidden = !show;                                        // the box's address: only when running on it
  };
  showKind();
  const dev = q("[data-l=device]");
  const fillDevices = async () => {
    const d = await api("/api/live/devices").catch(() => ({ devices: [] }));
    const opts = d.devices.length ? d.devices : [0, 1, 2, 3].map((i) => ({ index: i, name: `device ${i}` }));
    dev.innerHTML = opts.map((x) => `<option value="${esc(x.id ?? String(x.index))}" data-index="${x.index}">${esc(x.name)}</option>`).join("");
    const pick = opts.find((x) => x.id && x.id === L.deviceId) || opts.find((x) => !L.deviceId && x.index === L.device) || opts[0];
    if (pick) { dev.value = pick.id ?? String(pick.index); L.device = pick.index; L.deviceId = pick.id; }
    requestPreview();
  };
  if (L.kind === "device" && L.where === "mac") fillDevices(); else requestPreview();
  // a stream sent to this Mac: the addresses the sender can reach it at (one per network it's on)
  const hostSel = q("[data-x=host]"), addr = q("[data-x=addr]");
  let listenAddrs = null;
  const showAddr = () => {
    const urls = (listenAddrs || {})[L.proto || "rtmp"] || [];
    hostSel.innerHTML = urls.map((u, i) => `<option value="${i}">${esc(u.split("/")[2].split(":")[0])}</option>`).join("");
    hostSel.hidden = urls.length < 2;
    hostSel.value = String(Math.min(+(L.listenHost || 0), Math.max(0, urls.length - 1)));
    addr.value = urls[+hostSel.value] || "";
  };
  const fillListen = async () => { listenAddrs = await api("/api/live/listen").catch(() => ({})); showAddr(); };
  hostSel.onchange = () => { L.listenHost = +hostSel.value; saveLiveSet(); showAddr(); };
  addr.onclick = () => addr.select();
  q("[data-a=copyaddr]").onclick = async () => {
    addr.select();
    try { await navigator.clipboard.writeText(addr.value); } catch { document.execCommand("copy"); }
    status(`Copied ${addr.value}: set the box to stream there, then ▶ Start`);
  };
  if (L.kind === "listen") fillListen();
  w.body.querySelectorAll("[data-l]").forEach((el) => el.onchange = () => {
    const k = el.dataset.l;
    if (k === "device") { L.deviceId = el.value; L.device = +el.selectedOptions[0].dataset.index; saveLiveSet(); requestPreview(); return; }
    L[k] = el.type === "checkbox" ? el.checked : ["imgsz", "speed", "boxStride", "combo"].includes(k) ? +el.value : el.value;
    if (k === "combo") { saveLiveSet(); showKind(); previewCombo(); return; }
    saveLiveSet();
    if (k === "kind" || k === "where" || k === "record") { showKind(); if (L.kind === "device" && L.where === "mac") fillDevices(); }
    if ((k === "kind" || k === "where") && !(L.kind === "device" && L.where === "mac")) requestPreview();
    if (k === "proto") requestPreview();
    if (k === "kind" && L.kind === "listen") fillListen();
    if (k === "proto") showAddr();
    if (["kind", "device", "proto", "where"].includes(k) && S.live.status === "error") {   // a new source: the old run's error is gone
      S.live.status = "idle"; S.live.error = null; syncLiveFeed(); render(["livestats"]);
    }
    if ((k === "sound" && L.sound) || (k === "voice" && L.voice)) unlockSound();
  });
  q("[data-a=test]").onclick = () => {
    unlockSound();
    setTimeout(() => sfx("pass"), 30);
    setTimeout(() => sfx("gate"), 400);
    setTimeout(() => sfx("start"), 850);
    setTimeout(() => say("Lap 2, 27.4"), 1500);
  };
  q("[data-a=start]").onclick = liveStart;
  q("[data-a=stop]").onclick = liveStop;
  // the media box: where it is, find it, check it, clean it
  const b = (s) => q(`[data-b=${s}]`), msg = (t, err) => { b("msg").textContent = t; b("msg").classList.toggle("err", !!err); };
  const boxCall = async (label, fn) => { msg(label); try { await fn(); } catch (e) { msg(e.message, true); } };
  async function loadBoxCfg() {
    const c = await api("/api/live/box").catch(() => null);
    if (!c) return;
    b("host").value = c.host || ""; b("user").value = c.user || "";
    if (!c.host || !c.user) msg("set the box's address and user (🔍 Find looks for it)", true);
  }
  q("[data-a=boxsave]").onclick = () => boxCall("saving…", async () => {
    const c = await api("/api/live/box", { json: { host: b("host").value, user: b("user").value } });
    msg(`saved: ${c.user}@${c.host}`);
  });
  q("[data-a=boxfind]").onclick = () => boxCall("looking for boxes…", async () => {
    const d = await api("/api/live/box/find");
    msg(d.boxes.length ? `${d.boxes.length} found — pick one:` : "no box answered (is it on this network?)", !d.boxes.length);
    b("found").innerHTML = d.boxes.map((x) => `<button data-addr="${esc(x.addr)}" title="${esc(x.how)}${x.ver ? " · app " + esc(x.ver) : ""}">${esc(x.name)} (${esc(x.addr)})</button>`).join("");
    b("found").querySelectorAll("button").forEach((btn) => btn.onclick = () => { b("host").value = btn.dataset.addr; b("found").innerHTML = ""; q("[data-a=boxsave]").click(); });
  });
  q("[data-a=boxcheck]").onclick = () => boxCall("checking…", async () => {
    const s = await api("/api/live/box/status");
    msg(s.reachable ? `${s.model || "reachable"} · ${s.mem_free_mb} of ${s.mem_total_mb} MB free · ${s.cpu_temp ?? "?"} °C · this app's temporary folder: ${s.temp_folder_mb} MB${s.spotter_running ? " · a spotter is running" : ""}`
      : `can't reach ${s.host}: ${s.error || ""}`, !s.reachable);
  });
  q("[data-a=boxclean]").onclick = () => boxCall("cleaning…", async () => {
    if (!confirm("Delete everything this app put on the media box (code, models, track, videos, its Python environment)?")) return msg("");
    const s = await api("/api/live/box/clean", { method: "POST" });
    msg(`cleaned · temporary folder: ${s.temp_folder_mb ?? 0} MB`);
  });
  w.feed = q("img");
  w.spin = q(".vspin");
  w.spin.onclick = () => { if (!LIVE_ON() && !PV_ON() && liveFeedPicked()) requestPreview(); };   // open it (again)
  w.cleanup = () => { w.feed.removeAttribute("src"); delete w.feed.dataset.started; };
  syncLiveFeed();
}

async function liveStart() {
  unlockSound();                               // this click unlocks the sound and the voice
  const L = S.liveSet;
  const box = L.where === "box";
  const body = { kind: box ? "file" : L.kind, imgsz: L.imgsz, track: S.track || undefined, lapmode: lapMode(),   // the track names the gates
                 where: L.where, stride: box ? L.boxStride : 1, pace: box ? L.boxPace : "realtime" };
  if (body.kind === "file") {
    if (!S.track) return status("Pick a track in the top bar first: the video file comes from it.", true);
    if (!S.video) return status("Pick a video in the top bar first (it plays in real time as if it were live).", true);
    Object.assign(body, { video: S.video.name, speed: L.speed });
  } else if (body.kind === "device") Object.assign(body, { device: L.device, device_id: L.deviceId || undefined });
  else body.proto = L.proto || "rtmp";
  if (!box && L.combo) body.combo = +L.combo === -1 ? "all" : L.combo;   // one pilot of a combo view, or all of them
  if (!box && +L.combo === -1) {                                 // every pilot: their names, and the race map
    body.pilots = L.names || [];
    if (!wins.livemap) { openWin("livemap"); saveLayout(); }
  }
  if (body.kind !== "file") body.pilot = (L.pilot || "").trim();  // a live feed: whose laps are called out
  if (body.kind !== "file" && !box && L.record) {                 // save the feed under a new track's name
    if (!(L.recTrack || "").trim()) return status("Name the new track to save the feed under (next to ⏺ save feed).", true);
    Object.assign(body, { record: true, record_track: L.recTrack.trim() });
  }
  if (S.jobs.some((j) => ["running", "queued"].includes(j.status)))
    status("An analysis job is running: it shares the GPU, so the live spotter may drop frames.", true);
  try { const r = await api("/api/live/start", { json: body }); if (S.live.started !== r.started) S.live = { ...r }; syncLiveFeed(); render(["livestats", "livelog"]); }
  catch (e) { status(e.message, true); }
}

// the MJPEG stream: one connection per run, kept until the run ends (the last frame stays on screen)
function syncLiveFeed() {
  const w = wins.livefeed;
  if (!w || !w.feed) return;
  const L = S.live, img = w.feed, P = S.preview, pv = !LIVE_ON() && liveFeedPicked();
  // one MJPEG connection while the preview or a run shows frames (the server ends it ~5 s after both stop;
  // the last frame stays on screen)
  if ((LIVE_ON() || PV_ON()) && !img.dataset.started) {
    img.dataset.started = String(Date.now());
    img.src = `/api/live/mjpeg?t=${img.dataset.started}`;
  } else if (!LIVE_ON() && !PV_ON()) delete img.dataset.started;
  const msg = L.status === "deploying" ? (L.detail || "Deploying to the media box…")
    : L.status === "loading" ? (L.detail || "Loading the detector…") : L.status === "error" ? `Error: ${L.error}`
    : pv && P.status === "opening" ? (P.detail || "Opening the feed…")
    : pv && P.status === "error" ? `Feed: ${P.error} — click to try again`
    : pv && !PV_ON() ? "No preview — click here to show the feed, or ▶ Start"
    : !img.getAttribute("src") ? "Not running — pick a source and ▶ Start" : "";
  w.spin.hidden = !msg;
  w.spin.style.cursor = w.spin.style.pointerEvents = "";
  if (pv && !PV_ON()) Object.assign(w.spin.style, { cursor: "pointer", pointerEvents: "auto" });   // click: open it
  w.spin.classList.toggle("err", L.status === "error" || (pv && P.status === "error") || (!LIVE_ON() && !PV_ON() && !img.getAttribute("src")));
  $("span", w.spin).textContent = msg;
  const badge = $(".vbadge", w.body);
  badge.hidden = !(pv && P.status === "on");
  badge.textContent = `PREVIEW${P.combo ? ` · CH${P.combo.ch}` : ""}${P.size ? ` · ${P.size}` : ""}${P.fps ? ` · ${P.fps} fps` : ""} — ▶ Start to spot gates${S.liveSet.record ? " and save" : ""}`;
  $("[data-a=start]", w.body).disabled = LIVE_ON();
  $("[data-a=stop]", w.body).disabled = !LIVE_ON() && !PV_ON();
}

function sinceLast() {
  const p = S.live.passes[S.live.passes.length - 1];
  if (!p) return null;
  if (p.wall) return (performance.now() - p.wall) / 1000;
  return S.live.stats ? S.live.stats.t - p.t : null;
}

const curLap = () => (lapWall !== null && S.live.lap_start ? (performance.now() - lapWall) / 1000 * (S.live.speed || 1) : null);
const gname = (p) => p.gate_id === undefined ? "…" : p.gate_id === null ? "·" : p.gate_id ? `G${p.gate_id}` : "✕";

function renderLiveClock() {
  const b = wins.livestats?.body;
  if (!b) return;
  const el = $("[data-k=since]", b), lap = $("[data-k=lap]", b);
  if (el) el.textContent = fmt(sinceLast(), 1) + "s";
  if (lap) lap.textContent = curLap() !== null ? fmt(curLap(), 1) + "s" : "—";
}

// ── several pilots of a combo at once: race state, standings, call-outs, the race map ──
const PILOT_COLORS = ["#ff5050", "#3caaff", "#5adc5a", "#ffc83c"];     // as online_spotter.PILOT_BGR
const liveMulti = () => !!(S.live.pilots && S.live.pilots.length);
const pilotName = (i, R = S.live) => R.pilots?.[i]?.name || `CH${(R.pilots?.[i]?.ch) ?? i + 1}`;

// Each pilot's progress: its laps as the decoder started them (G1, or estimated when G1 was missed), and within the
// current lap the furthest gate reached — forward only, never more than half the track at once (a misnamed gate
// doesn't move the pilot) — so pilots compare across laps. The leader is the furthest; at the same gate, the first.
const QUIET_LAP = 0.75, QUIET_MIN = 15;
function raceState(L = S.live) {
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
  // down: a combo's pilot whose run ended short of the race (crashed, out); in Live, a pilot not seen at any gate for
  // QUIET_LAP of a lap (at least QUIET_MIN s; a few missed gates in a row happen) while the others race on
  const now = L.now ?? Math.max(0, ...passes.map((x) => x.t));
  const quiet = Math.max(QUIET_MIN, QUIET_LAP * (L.legs || []).reduce((a, b) => a + b, 0));
  for (const q of ps) {
    q.last = Math.max(-1, ...passes.filter((x) => x.pilot === q.i).map((x) => x.t));
    q.done = L.now != null && q.end != null && L.now > q.end + 3;                  // a combo's pilot whose run is over
    q.finished = !!q.fin && L.now >= q.end;                                        // its last lap closed: right away
    q.down = q.cum >= 0 && !q.finished && (q.done || (q.end === undefined && now - Math.max(q.last, q.t ?? -1) > quiet));
  }
  for (const q of ps) {
    q.best = q.laps.length ? Math.min(...q.laps.map((l) => l.dt)) : null;
    q.gap = lead && q !== lead && q.cum >= 0 && lead.cross[q.cum] != null ? q.t - lead.cross[q.cum] : null;
    q.behind = lead && q.cum >= 0 ? Math.floor((lead.cum - q.cum) / n) : 0;     // whole laps down
  }
  ps.order = order;
  return ps;
}

// Call-outs as the order changes (Live and Race mode): "X takes the lead", "X is now second" — said of who moved up
// (whoever they passed isn't) —, "X is down" (see raceState; "X is back" if they turn up again), "X finishes, second".
// A new place counts once held POS_HOLD s of race time and clearly (a gate ahead of the next pilot, not a tie at the
// same gate); one place call every POS_GAP s at most (a lead change: LEAD_GAP), the lead first. A pilot down or back
// re-places the others silently. quiet: only take the race as it is now (opened mid-race, a seek).
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

let livePlaces = newPlaces();
function raceCallouts(quiet = false) {
  const now = Math.max(0, ...S.live.passes.map((p) => p.t));
  for (const text of placeCalls(raceState(), now, livePlaces, S.live, quiet)) say(text, true);
}

function renderStandings(R = S.live) {
  const ps = raceState(R), n = R.gates;
  const k = (l, v) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  let h = "";
  if (!n) return `<div class="hint">No learned track: passes per pilot only (pick a learned track in the top bar for laps and the race map).</div>`
    + `<div class="kpis">${ps.map((q) => k(`<span class="dot" style="background:${PILOT_COLORS[q.i % 4]}"></span>${esc(q.name)}`,
      R.passes.filter((p) => p.pilot === q.i).length)).join("")}</div>`;
  h += `<table class="standings"><tr><th>#</th><th>Pilot</th><th>Lap</th><th>Gate</th><th>Gap</th><th>Last lap</th><th>Best</th></tr>`;
  for (const q of [...ps].sort((a, b) => (a.pos ?? 99) - (b.pos ?? 99))) {
    const last = q.laps[q.laps.length - 1];
    h += `<tr class="${q.pos === 1 ? "cur" : ""}"><td>${q.pos ?? "—"}</td>
      <td><span class="dot" style="background:${PILOT_COLORS[q.i % 4]}"></span><b>${esc(q.name)}</b></td>
      <td>${q.lap || "—"}</td><td>${q.cum >= 0 ? "G" + q.g : "—"}</td>
      <td>${q.down ? "down" : q.done ? "finished" : q.pos === 1 ? "leader" : q.behind > 0 ? `+${q.behind} lap${q.behind > 1 ? "s" : ""}` : q.gap != null ? "+" + fmt(q.gap, 1) + "s" : "—"}</td>
      <td>${last ? (last.estimated ? "~" : "") + fmt(last.dt) + "s" : "—"}</td><td>${q.best != null ? fmt(q.best) + "s" : "—"}</td></tr>`;
  }
  h += "</table>";
  if (R.callout) h += `<div class="row callout">🗣 ${esc(R.callout.text)}</div>`;
  return h + `<div class="hint">${esc(R.track)} · ${n} gates · gap: behind the leader at the same gate</div>`;
}

// the race map: the gates on a loop, spaced by the learned time of each leg; each pilot a dot moving from its last
// gate towards the next at the learned pace (it stops short of the next gate until it's passed)
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
    const live = S.mode === "live", L = live ? S.live : raceSrc() ? comboRace() : null, n = L?.gates;
    const multi = live ? liveMulti() : !!L?.pilots.length;
    if (!multi || !n) {
      g.fillStyle = "#9aa3ad";
      g.fillText(multi ? "Pick a learned track in the top bar for the race map."
        : live ? "Live with Combo: all pilots — the pilots' progress shows here."
        : "Pick an analysed video or a combo view in the top bar: the pilots' progress shows here as it plays.", cw / 2, ch / 2);
      return;
    }
    const { legs, lap, at } = raceTrack(L);
    // the track's layout (learn_layout.py, moved by hand in Track layout) when there is one: a smooth path through the
    // gates; else a stadium-shaped loop — two straights and two half circles, the start gate mid bottom straight
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
    const now = live ? liveNow() : S.t, ps = raceState(L), np = ps.length;
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

// ── the race in 3D: the track's layout (Track layout) with each gate modelled by its type and the pilots flying it,
// where the race map has them; an overview to turn and zoom, or a camera chasing a pilot or on board ──
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
  const st = { disp: {} }, view = { az: 0, el: 1.0, dist: null, cam: store.get("spotter.cam3d", "orbit") };
  const look = { p: null, t: null };
  sel.onchange = () => { view.cam = sel.value; store.set("spotter.cam3d", view.cam); look.p = null; };
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
    const key = ps.map((q) => q.name).join("|");
    if (key === camKey) return;
    camKey = key;
    const o = [["orbit", "Overview"]];
    if (ps.length > 1) o.push(["lead", "Chase the leader"]);
    ps.forEach((q) => o.push([`chase:${q.i}`, `Chase ${q.name}`], [`fpv:${q.i}`, `On board ${q.name}`]));
    if (!o.some(([v]) => v === view.cam)) view.cam = "orbit";
    sel.innerHTML = o.map(([v, t]) => `<option value="${v}" ${v === view.cam ? "selected" : ""}>${esc(t)}</option>`).join("");
  };

  let cam = null, lastMs = 0;
  const frame = (ms) => {
    if (!alive) return;
    raf = requestAnimationFrame(frame);
    if (document.hidden) return;
    const dt = Math.min(0.1, (ms - (lastMs || ms)) / 1000); lastMs = ms;
    const cw = box.clientWidth, ch = box.clientHeight;
    if (!cw || !ch) return;
    const live = S.mode === "live", L = live ? S.live : raceSrc() ? comboRace() : null;
    if (!L || !L.gates) {
      hud.innerHTML = `<div class="hint">${live ? "Live on a learned track: the pilots fly here." : "Pick an analysed video or a combo view of a learned track in the top bar: its pilots fly here as it plays."}</div>`;
      R3.domElement.style.visibility = "hidden";
      return;
    }
    R3.domElement.style.visibility = "";
    const key = JSON.stringify([L.track, L.gates, L.legs, L.layout?.gates]);
    if (!G || G.key !== key) { G = build(L); G.key = key; st.disp = {}; look.p = null; view.dist = null; }
    if (R3.domElement.width !== Math.round(cw * devicePixelRatio)) R3.setSize(cw, ch, false);
    cam.aspect = cw / ch;
    const now = live ? liveNow() : S.t, ps = raceState(L), np = ps.length, lead = ps.order?.[0];
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
    } else {
      look.p = null;
      view.dist = view.dist || G.ext * 1.9;
      const e = view.el, a = view.az;
      cam.fov = 50;
      cam.position.set(view.dist * Math.cos(e) * Math.sin(a), view.dist * Math.sin(e), view.dist * Math.cos(e) * Math.cos(a));
      cam.lookAt(0, 0, 0);
    }
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
      if (view.cam !== "orbit") { view.cam = sel.value = "orbit"; store.set("spotter.cam3d", "orbit"); }
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

// ── a combo view in Race mode: its pilots' own videos (…_CH1 …, split from it, on the same clock) are analysed one by
// one, in full; playing the combo then races them against each other at the playhead — the same standings, race
// map and lap times as Live's all pilots, from those analyses ──
const comboNamesKey = (stem) => `spotter.combo.${S.track}.${stem}`;

async function loadCombo() {
  S.combo = null;
  const v = S.video;
  if (!v || v.role !== "combo" || !S.track) return;
  try {
    const d = await api(`/api/tracks/${encodeURIComponent(S.track)}/combo_race?video=${encodeURIComponent(v.name)}`);
    if (S.video?.name !== v.name) return;
    const names = store.get(comboNamesKey(v.stem), []);
    d.pilots.forEach((p, i) => Object.assign(p, racePilot(p, i, names[p.ch - 1] || `CH${p.ch}`)));
    d.track = S.track;
    S.combo = d;
    comboMemo = null;
    if (S.mode === "race" && d.pilots.some((p) => p.analysed)) {    // its windows: the standings and the race map
      const r = deskRect(), f = DEFAULTS.race, geo = (k) => ({ x: f[k][0] * r.width + 4, y: f[k][1] * r.height + 4, w: f[k][2] * r.width - 8, h: f[k][3] * r.height - 8 });
      if (!wins.racestand) openWin("racestand", geo("stats"));
      if (!wins.livemap) openWin("livemap", geo("splits"));
      saveLayout();
    }
  } catch (e) { status(e.message, true); }
}

// one pilot's race from its analysis: its named gate passes, laps, and where each lap starts (lap k at its t0; the
// unfinished last lap at the last t1; no lap closed: at the first G1)
function racePilot(r, i, name) {
  const passes = (r.passes || []).filter((x) => (x.source ?? "RACE") === "RACE" && x.gate_id >= 1);
  const laps = r.laps || [], last = laps[laps.length - 1], starts = [];
  const end = r.end !== undefined ? r.end : passes.length ? passes[passes.length - 1].t : null;
  for (const L of laps) starts.push({ lap: L.lap, t: L.t0 });
  if (last && end > last.t1 + 1e-3) starts.push({ lap: last.lap + 1, t: last.t1 });
  if (!laps.length) { const g1 = passes.find((x) => x.gate_id === 1); if (g1) starts.push({ lap: 1, t: g1.t }); }
  return { i, name, passes, laps, end, starts };
}

// the pilot's name of a video: typed in the video window, else guessed from the file name (h001_R6_Nivazu → Nivazu)
const pilotKey = (v) => `spotter.pilot.${S.track}.${v.stem}`;
const guessPilot = (v) => (v?.stem || "").split(/[_ ]+/).reverse().find((x) => /^[A-Za-z][A-Za-z-]{2,}$/.test(x) && !/^(learn|combo)$/i.test(x)) || "";
const soloName = () => (S.video && store.get(pilotKey(S.video), "")) || guessPilot(S.video);
function syncPilotBox() {
  const el = wins.video && $("[data-x=pilot]", wins.video.body);
  if (!el) return;
  el.hidden = !S.video || S.video.role === "combo";
  el.value = S.video ? store.get(pilotKey(S.video), "") : "";
  el.placeholder = guessPilot(S.video) || "Pilot name";
}

// the race at time t, in Live's shape (raceState, renderStandings and the race map take it): what had happened by t —
// of the combo's pilots, or of the one pilot of the video picked
let comboMemo = null;
const raceSrc = () => S.combo || S.solo;
function comboRace(t = S.t) {
  const C = raceSrc();
  if (comboMemo && comboMemo.C === C && comboMemo.t === t) return comboMemo.R;
  const R = {
    now: t, gates: C.n_gates, legs: C.legs, layout: C.layout, track: C.track, pilots: C.pilots.map((p) => ({ name: p.name, ch: p.ch, end: p.end, fin: raceFinished(C, p) })),
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

// Race mode, while the video plays: Live's sounds and call-outs for what happens between two frames (not on a seek)
let racePlaces = newPlaces();
function raceSounds(t0, t1, playing) {
  if (S.mode !== "race" || !raceSrc()) return;
  const C = raceSrc(), multi = !!S.combo;
  if (!playing || !(t1 > t0) || t1 - t0 > 1.5) {          // paused or jumped: the places as of now, nothing said
    racePlaces = newPlaces();
    if (multi) placeCalls(raceState(comboRace(t1)), t1, racePlaces, comboRace(t1), true);
    return;
  }
  const inside = (t) => t > t0 && t <= t1;
  for (const p of C.pilots) {
    if (!multi) for (const x of p.passes) if (inside(x.t)) sfx(x.gate_id === 1 ? "start" : "gate", true);
    if (p.starts[0] && p.starts[0].lap === 1 && inside(p.starts[0].t)
        && !C.pilots.some((q) => q !== p && q.starts[0] && q.starts[0].t <= p.starts[0].t)) say(multi ? "Go!" : p.name ? `${p.name}, go!` : "Go!", true, true);
    for (const L of p.laps) {
      if (!inside(L.t1)) continue;
      if (multi) sfx("start", true);
      const best = Math.min(...p.laps.filter((x) => x.t1 <= L.t1).map((x) => x.dt));
      say(`${p.name ? p.name + ", " : ""}lap ${L.lap}, ${L.dt.toFixed(1)}${L.dt <= best && L.lap > 1 ? ", best lap" : ""}`, true, true);
    }
  }
  if (!multi) return;
  const R = comboRace(t1);
  for (const text of placeCalls(raceState(R), t1, racePlaces, R)) say(text, true, true);
}

function renderRaceStand(w) {
  const C = S.combo;
  if (!C) {
    w.body.innerHTML = `<div class="empty">${S.video?.role === "combo" ? "Loading the pilots…"
      : "Pick a combo view (combo_videos) in the top bar: its pilots race each other here as it plays."}</div>`;
    w.key = null;
    return;
  }
  if (w.key !== C) {                               // the pilots' names: kept, not redrawn while you type
    w.key = C;
    w.body.innerHTML = `<div class="row">${C.pilots.map((p) =>
      `<input data-cname="${p.i}" size="9" maxlength="30" value="${esc(/^CH\d+$/.test(p.name) ? "" : p.name)}" placeholder="CH${p.ch} pilot"
        title="CH${p.ch}: ${esc(p.video)}${p.analysed ? "" : " — not analysed yet"}" style="border-left:4px solid ${PILOT_COLORS[p.i % 4]}">`).join("")}</div><div data-x="std"></div>`;
    w.body.querySelectorAll("[data-cname]").forEach((el) => el.oninput = () => {
      const p = C.pilots[+el.dataset.cname];
      p.name = el.value.trim() || `CH${p.ch}`;
      const names = []; for (const q of C.pilots) names[q.ch - 1] = q.name.match(/^CH\d+$/) ? "" : q.name;
      store.set(comboNamesKey(S.video.stem), names);
      comboMemo = null;
      render(["racestand", "livemap"]); drawOverlay(); wins.timeline?.draw();
    });
  }
  const todo = C.pilots.filter((p) => !p.analysed);
  let h = !C.pilots.length ? `<div class="hint warn">No pilot videos of it yet: ⊞ Split & analyse pilots (top bar).</div>`
    : todo.length ? `<div class="hint warn">Not analysed yet: ${todo.map((p) => `CH${p.ch}`).join(", ")} — ▶ Analyse pilots (top bar).</div>` : "";
  const R = comboRace();
  h += renderStandings(R);
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
    h += `</table><div class="hint">★ the pilot's best · highlighted: the fastest lap of the race · as of ${fmt(S.t, 1)} s</div>`;
  }
  $("[data-x=std]", w.body).innerHTML = h;
}

// over the combo view: each pilot's tile framed in its colour, with its place, lap, gate and lap clock; a pass flashes
function drawComboOverlay(g, box) {
  const C = S.combo, R = comboRace(), ps = raceState(R);
  for (const p of C.pilots) {
    const T = p.tile, q = ps[p.i], col = PILOT_COLORS[p.i % 4];
    const [x, y, ww, hh] = box([T.x, T.y, T.x + T.w, T.y + T.h]);
    g.strokeStyle = col; g.lineWidth = q.pos === 1 ? 4 : 2;
    g.strokeRect(x + 1, y + 1, ww - 2, hh - 2);
    const lapT = q.cum >= 0 ? S.t - R.starts.filter((s) => s.pilot === p.i).pop().t : null;
    const line = !p.analysed ? "not analysed" : q.cum < 0 ? "waiting for G1"
      : q.down || q.done ? `${q.down ? "down" : "finished"} · ${q.laps.length} lap${q.laps.length === 1 ? "" : "s"}${q.best != null ? ` · best ${fmt(q.best)}s` : ""}`
      : `L${q.lap} · G${q.g} · ${fmt(lapT, 1)}s${q.pos > 1 ? q.behind > 0 ? ` · +${q.behind}L` : q.gap != null ? ` · +${fmt(q.gap, 1)}s` : "" : ""}`;
    g.font = "bold 16px -apple-system, sans-serif";
    const head = `${q.pos ? q.pos + " · " : ""}${p.name}`;
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
    const rec = (p.passes || []).filter((x) => x.t <= S.t && S.t - x.t < 0.6);
    rec.forEach((e, k) => { g.fillStyle = e.recovered ? "#ff9f43" : "#4cd97b"; g.fillText(`G${e.gate_id} ✓`, x + 13, y + 54 + k * 26); });
  }
}

// the timeline of a combo view: a lane per pilot — its laps and gate passes
function drawComboLanes(g, cw, ch, X) {
  const C = S.combo, np = C.pilots.length;
  if (!np) return;
  const top = 2, lh = (ch - 18 - top) / np;
  C.pilots.forEach((p, k) => {
    const y = top + k * lh, col = PILOT_COLORS[p.i % 4];
    g.fillStyle = k % 2 ? "rgba(255,255,255,.02)" : "rgba(255,255,255,.045)"; g.fillRect(0, y, cw, lh);
    for (const L of p.laps || []) {
      g.fillStyle = col; g.globalAlpha = L.lap % 2 ? 0.16 : 0.08; g.fillRect(X(L.t0), y, X(L.t1) - X(L.t0), lh); g.globalAlpha = 1;
      g.fillRect(X(L.t0), y, 2, lh);
      if (lh >= 14 && X(L.t1) - X(L.t0) > 46) { g.fillStyle = "#e6e9ee"; g.fillText(`L${L.lap} ${fmt(L.dt)}s`, X(L.t0) + 4, y + 2); }
    }
    g.fillStyle = col;
    for (const x of p.passes || []) g.fillRect(X(x.t) - 0.5, y + lh * (x.gate_id === 1 ? 0.25 : 0.5), 1.5, lh * (x.gate_id === 1 ? 0.7 : 0.45));
    if (!(p.laps || []).length || X(p.laps[0].t0) > 50) { g.fillStyle = col; g.fillText(p.analysed ? p.name : `${p.name} (not analysed)`, 4, y + 2); }
  });
}

function renderLiveStats(w) {
  const L = S.live, st = L.stats, P = L.passes, laps = L.laps || [];
  const last = P[P.length - 1], prev = P[P.length - 2];
  const badge = { running: "ok", loading: "warn", error: "", ended: "", stopped: "", idle: "" }[L.status] ?? "";
  const k = (l, v, cls = "", attr = "") => `<div class="kpi ${cls}"><div class="v" ${attr}>${v}</div><div class="l">${l}</div></div>`;
  let h = `<div class="row"><span class="badge ${badge}">${esc(L.status)}</span>${L.pilot ? `<b>${esc(L.pilot)}</b>` : ""}<span class="hint">${esc(L.source || "no feed yet")}${L.where === "box" && L.box ? ` · ${esc(L.box.host)}` : ""}</span>
    ${LIVE_ON() ? `<span class="grow"></span><button data-a="livestop">■ Stop</button>` : ""}</div>`;
  const R = L.recording;
  if (R && R.track) {
    const len = R.seconds != null ? `${Math.floor(R.seconds / 60)}:${String(Math.floor(R.seconds % 60)).padStart(2, "0")}` : "0:00";
    h += R.error ? `<div class="row rec err">⏺ saving the feed failed: ${esc(R.error)}</div>`
      : R.done ? `<div class="row rec done">✓ saved ${len} · ${esc(R.path || "")}</div>`
      : `<div class="row rec">⏺ REC ${len}${R.size ? " · " + esc(R.size) : ""} → live_recordings/${esc(R.track)}/${esc(R.file || "")}</div>`;
  }
  if (liveMulti()) {
    h += renderStandings();
  } else if (liveNamed()) {
    const best = laps.length ? Math.min(...laps.map((l) => l.dt)) : null, lastLap = laps[laps.length - 1];
    const named = [...P].reverse().find((p) => p.gate_id);
    h += `<div class="kpis">
      ${k(L.lap_start ? `Lap ${L.lap_start.lap}` : "Waiting for G1", fmt(curLap(), 1) + "s", "live big", 'data-k="lap"')}
      ${k("Last lap", lastLap ? (lastLap.estimated ? "~" : "") + fmt(lastLap.dt) + "s" : "—", "big")}
      ${k("Best lap", best !== null ? fmt(best) + "s" : "—")}
      ${k("Last gate", named ? `G${named.gate_id}<small> / ${L.gates}</small>` : "—")}
      ${k("Since last pass", fmt(sinceLast(), 1) + "s", "", 'data-k="since"')}
    </div>`;
    if (laps.length) {
      h += `<table><tr><th>Lap</th><th>Time</th><th>Δ best</th><th>Gates</th></tr>`;
      for (let i = laps.length - 1; i >= 0; i--) {
        const l = laps[i];
        h += `<tr class="${l.dt === best ? "cur" : ""}"><td>${l.lap}</td><td>${fmt(l.dt)}s${l.estimated ? ' <span class="hint" title="G1 was missed: the lap end is estimated from the learned timing">~</span>' : ""}</td>
          <td>${l.dt === best ? "best" : "+" + fmt(l.dt - best)}</td><td>${l.gates}/${L.gates}</td></tr>`;
      }
      h += "</table>";
    }
    if (L.callout) h += `<div class="row callout">🗣 ${esc(L.callout.text)}</div>`;
    h += `<div class="hint">${esc(L.track)} · ${L.gates} gates · timing ${esc(L.timing || "none: order + appearance only")}</div>`;
  } else {
    h += `<div class="kpis">
      ${k("Gate passes", P.length, "live big")}
      ${k("Since last pass", fmt(sinceLast(), 1) + "s", "", 'data-k="since"')}
      ${k("Last gap", last && prev ? fmt(last.t - prev.t) + "s" : "—")}
    </div>`;
    if (L.status !== "idle") h += `<div class="hint">${L.lapmode === "off" ? "Laps off" : L.track ? esc(L.track) + " isn't learned yet" : "No track"}: passes only, no gate names or laps.</div>`;
  }
  if (liveMulti()) h += `<div class="hint sounds">🔊 <b>fanfare</b> a pilot finished a lap · 🗣 lap times, “… takes the lead”, “… is now second”, “… is down”`
    + ` (no beep per pass: four pilots)</div>`;
  else h += `<div class="hint sounds">🔊 <b>beep</b> a gate pass was detected` + (!liveNamed() ? ""
    : ` · <b>coin</b> ~0.1 s later: the gate was identified (G2…) · <b>fanfare</b> the start gate (G1) · 🗣 the lap time`) + `</div>`;
  if (st) {
    const slow = st.fps_spotter < 0.95 * st.fps_feed || st.dropped_pct > 3;      // a dropped frame breaks tracks
    h += `<h4>Spotter</h4><div class="kpis">
      <div class="kpi"><div class="v ${slow ? "warn" : ""}">${fmt(st.fps_spotter, 0)}<small> / ${fmt(st.fps_feed, 0)}</small></div><div class="l">fps analysed / feed</div></div>
      <div class="kpi"><div class="v ${st.dropped_pct > 3 ? "warn" : ""}">${fmt(st.dropped_pct, 0)}%</div><div class="l">Frames dropped</div></div>
      <div class="kpi"><div class="v">${st.delay_ms}<small> ms</small></div><div class="l">Processing delay</div></div>
      ${st.cpu_temp != null ? `<div class="kpi"><div class="v ${st.cpu_temp >= 80 ? "warn" : ""}">${fmt(st.cpu_temp, 0)}<small> °C</small></div><div class="l">Box CPU${st.cpu_temp >= 80 ? " (throttles at 85)" : ""}</div></div>` : ""}
      <div class="kpi"><div class="v">${fmt(st.t, 0)}${st.duration ? `<small> / ${fmt(st.duration, 0)}</small>` : ""}s</div><div class="l">Feed time</div></div>
    </div><div class="hint">${esc(st.size)} feed · detector input ${st.imgsz}${st.stride > 1 ? ` · every ${st.stride === 2 ? "2nd" : "3rd"} frame` : ""}${L.where === "box" ? ` · on the media box${L.box?.realtime === false ? `, at its own pace (${fmt(st.fps_spotter * (st.stride || 1) / 30, 2)}× real time, nothing dropped)` : ""}` : ""}${slow ? " · the spotter can't keep up and passes get lost: pick a smaller detector size" + (L.where === "box" ? " or the box's pace" : "") : ""}</div>`;
  }
  w.body.innerHTML = h;
  // redrawn twice a second, so react on press (a click could span a redraw)
  w.body.onpointerdown = (ev) => { if (ev.target.closest("[data-a=livestop]")) liveStop(); };
}

function liveStop() {
  if (!LIVE_ON() && PV_ON()) return api("/api/live/preview/stop", { method: "POST" }).catch((e) => status(e.message, true));
  api("/api/live/stop", { method: "POST" }).catch((e) => status(e.message, true));
}

function renderLiveLog(w) {
  const P = S.live.passes, named = liveNamed();
  if (!P.length) { w.body.innerHTML = `<div class="empty">Every gate pass shows up here the moment it's detected.</div>`; return; }
  const multi = liveMulti();
  let h = `<table><tr><th>#</th>${multi ? "<th>Pilot</th>" : ""}${named ? "<th>Gate</th>" : ""}<th>Feed time</th><th>Gap</th><th>Gate type</th><th>How</th><th>Delay</th></tr>`;
  for (let i = P.length - 1; i >= Math.max(0, P.length - 200); i--) {
    const p = P[i], g = i > 0 ? p.t - P[i - 1].t : null;
    const gt = named ? `<td title="${p.gate_id === 0 ? "judged not a real pass" : p.conf != null ? `sure ${Math.round(p.conf * 100)}%` : ""}"><b>${gname(p)}</b></td>` : "";
    const who = multi ? `<td><span class="dot" style="background:${PILOT_COLORS[p.pilot % 4]}"></span>${esc(pilotName(p.pilot))}</td>` : "";
    h += `<tr class="${i === P.length - 1 ? "cur" : ""}${p.gate_id === 0 ? " dim" : ""}"><td>${p.n}</td>${who}${gt}<td>${fmt(p.t)}s</td><td>${g === null ? "—" : fmt(g) + "s"}</td>
      <td>${esc(p.gate_type)}</td><td class="hint">${esc(p.reason)}</td><td>${p.delay_ms} ms</td></tr>`;
  }
  w.body.innerHTML = h + "</table>";
}

// ─────────────────────────────────────────────────────────────
// Modes, keyboard, start-up
// ─────────────────────────────────────────────────────────────

function setMode(m) {
  if (S.league) m = "race";
  S.mode = m;
  store.set("spotter.mode", m);
  writeHash();
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  applyLayout();
  renderActions();
  if (m === "manual") loadMarks();
  if (m === "live") connectLive(); else disconnectLive();
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
// Upload a video from this device (the browser's own file picker, never this computer's folders); a race video is
// then analysed against the chosen track. A file the track already has (same name and size) is not uploaded again.
// ─────────────────────────────────────────────────────────────

function openUploader(role) {
  if (!S.tracks.length) return status("Create a track with ＋ first.", true);
  role = S.league ? "race" : role || (S.mode === "race" ? "race" : "learn");
  document.querySelector(`input[name=upRole][value=${role}]`).checked = true;
  $("#upRoleRow").style.display = S.league ? "none" : "";
  $("#upTrack").innerHTML = S.tracks.map((t) => `<option ${t.name === S.track ? "selected" : ""}>${esc(t.name)}</option>`).join("");
  $("#upFile").value = "";
  $("#upProg").value = 0;
  const L = S.limits;
  $("#upLimits").textContent = L ? `Up to ${fmt(L.video_gb, 0)} GB per video, ${L.per_hour} uploads per hour.` : "";
  $("#upMsg").textContent = L?.paused ? "Uploads are paused: the league folder is full. Ask the organiser." : "";
  $("#upGo").disabled = !!L?.paused;
  $("#uploader").classList.add("open");
}

async function openVideo(track, name) {
  await loadTracks();
  if (track !== S.track) { $("#trackSel").value = track; await selectTrack(track); }
  $("#videoSel").value = name;
  await selectVideo(name);
}

$("#upGo").onclick = async () => {
  const f = $("#upFile").files[0], track = $("#upTrack").value;
  const role = S.league ? "race" : document.querySelector("input[name=upRole]:checked").value;
  if (!f) { $("#upMsg").textContent = "Choose a video first."; return; }
  const same = S.tracks.find((t) => t.name === track)?.videos.find((v) => v.name === f.name && v.bytes === f.size);
  if (same) {                                  // already in the track: open it, analyse only if asked
    $("#uploader").classList.remove("open");
    await openVideo(track, same.name);
    const busy = S.jobs.some((j) => j.video === same.name && j.track === track && ["running", "queued"].includes(j.status));
    if (busy) return status(`${same.name} is already in ${track} and being analysed — opened it.`);
    if (same.role !== "race") return status(`${same.name} is already a ${same.role} video of ${track} — opened it.`);
    if (!same.analysed) return analyseOrSplit(track, same);
    if (S.league || !confirm(`${same.name} is already in ${track} and analysed.\n\nOK: run the analysis again over it (its results are replaced)\nCancel: just open it`))
      return status(`${same.name} is already analysed — opened it.`);
    return startJob({ kind: "analyse", track, video: same.name });
  }
  if (S.limits && f.size > S.limits.video_gb * 2 ** 30) {
    $("#upMsg").textContent = `That video is ${(f.size / 2 ** 30).toFixed(1)} GB; up to ${fmt(S.limits.video_gb, 0)} GB, please.`;
    return;
  }
  $("#upGo").disabled = true;
  try {
    const r = await uploadVideo(track, f, role, (x) => {
      $("#upProg").value = 100 * x;
      $("#upMsg").textContent = `Uploading ${(f.size / 1e6).toFixed(0)} MB… ${fmt(100 * x, 0)}%`;
    });
    $("#uploader").classList.remove("open");
    await openVideo(track, r.name);
    if (role !== "race") return status(`Added ${r.name} as a learn video of ${track} — learn the track in Learn · auto.`);
    const v = S.tracks.find((t) => t.name === track)?.videos.find((x) => x.name === r.name);
    await analyseOrSplit(track, v || { name: r.name });
  } catch (e) { $("#upMsg").textContent = e.message; $("#upGo").disabled = false; }
};
$("#upClose").onclick = () => $("#uploader").classList.remove("open");
$("#addVideo").onclick = () => openUploader();

(async function start() {
  try {
    const H = hashState();
    if (H.mode && DEFAULTS[H.mode]) S.mode = H.mode;
    if (H.track) store.set("spotter.track", H.track);
    await loadModel();
    await loadTracks(false);
    if (S.league) { S.mode = "race"; $("#modelBox").remove(); }
    document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === S.mode));
    applyLayout();
    if (S.mode === "live") connectLive();
    if (S.track) await selectTrack(S.track);
    status(S.track ? `Track ${S.track} — pick a video` + (S.league ? " or ⬆ upload your own" : "") : S.league ? "No tracks yet."
      : S.tracks.length ? "No track picked (Live works without one: beeps only)" : "Create a track with ＋");
    if (H.video && curTrack()?.videos.some((v) => v.name === H.video)) {
      $("#videoSel").value = H.video;
      await selectVideo(H.video);
      if (H.t) { const t = +H.t; const v = S.videoEl; if (v) v.addEventListener("loadedmetadata", () => seek(t), { once: true }); seek(t); }
    }
    pollJobs();
    if (S.league) setInterval(() => { if (!document.hidden) loadTracks().catch(() => {}); }, 60000);  // runs analysed meanwhile
    requestAnimationFrame(loop);
  } catch (e) { status("Server not reachable: " + e.message, true); }
})();
