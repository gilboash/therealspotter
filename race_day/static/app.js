// Race-day dashboard — read-only views of every event of the data folder.
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sec = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const delta = (x, d = 2) => (x == null ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(d));
const cls = (x, tol = 0.005) => (x == null ? "" : x < -tol ? "good" : x > tol ? "bad" : "ok");
const pct = (x) => (x == null ? "–" : (100 * x).toFixed(1) + "%");
// The event shown (its folder relative to the data folder); every API call is about it
let EV = null;
const api = (p) => {
  if (EV && p.startsWith("/api/") && p !== "/api/events") p += (p.includes("?") ? "&" : "?") + "e=" + encodeURIComponent(EV);
  return fetch(p, { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(r.status)));
};
// a run checked by hand in the app (its laps and gates are the reviewed ones)
const reviewPill = (r) => r.reviewed === "complete" ? '<span class="pill good" title="Checked by hand in the app: laps and gates are the reviewed ones">reviewed</span>'
  : r.reviewed ? '<span class="pill ok" title="Partly checked by hand in the app">review in progress</span>' : "";
// a link inside the event shown: B() + "run/…"
const B = () => `#/e/${encodeURIComponent(EV)}/`;
const MODES = { live: "FPVTRACKSIDE · OPEN", capture: "LIVE CAPTURE · OPEN", replays: "REPLAYS · OPEN" };
const PILOT_COLORS = ["#4da3ff", "#ff9f43", "#3ecf8e", "#c56cf0", "#ff6b6b", "#f5c451", "#00d2d3", "#ff9ff3"];
const colorFor = (() => { const m = {}; return (n) => (m[n] ??= PILOT_COLORS[Object.keys(m).length % PILOT_COLORS.length]); })();

// A short "how to read it" line under a title or number
const hint = (t) => `<div class="hint">${t}</div>`;
// A number tile with its meaning underneath
const fact = (k, v, h, extra = "") => `<div class="fact"><div class="k">${k}</div>${v}${extra}${hint(h)}</div>`;

function steadiness(cv) {
  if (cv == null) return "";
  if (cv < 0.03) return '<span class="pill good">steady</span>';
  if (cv < 0.07) return '<span class="pill ok">fair</span>';
  return '<span class="pill bad">variable</span>';
}

// Colour a section time against the field median (green faster, red slower)
function heat(v, ref) {
  if (v == null || !ref) return "";
  const r = Math.max(-1, Math.min(1, (v - ref) / (ref * 0.15)));
  const c = r < 0 ? [62, 207, 142] : [255, 107, 107];
  return `background:rgba(${c[0]},${c[1]},${c[2]},${(0.15 + 0.55 * Math.abs(r)).toFixed(2)})`;
}

// A run's video: a whole file, or one pilot's quadrant of a 2×2 heat video
// (FPVTrackside event) — the quadrant is cropped here, the file is streamed as is.
function videoTag(r, attrs) {
  if (!r.crop) return `<video ${attrs} src="${r.video_url}" controls playsinline preload="metadata"></video>`;
  const [x, y, w, h] = r.crop;
  return `<div class="cropbox" onclick="const v=this.querySelector('video');v.paused?v.play():v.pause()">
    <video ${attrs} src="${r.video_url}" playsinline preload="metadata"
      style="width:${100 / w}%;left:${-100 * x / w}%;top:${-100 * y / h}%"></video><span class="tap">tap to play / pause</span></div>`;
}

// ── header: status every 3 s ───────────────────────────────────
let STATUS = null;
async function refreshStatus() {
  if (!EV) {
    SCHED = null;
    for (const id of ["#nav-up", "#nav-tv"]) $(id).style.display = "none";
    $("#track").textContent = "Race day"; $("#mode").textContent = ""; $("#phase").textContent = "";
    $("#phase").className = "badge"; $("#live").innerHTML = "";
    return;
  }
  try {
    STATUS = await api("/api/status");
    SCHED = await api("/api/schedule").catch(() => SCHED);
    for (const id of ["#nav-up", "#nav-tv"]) $(id).style.display = SCHED && SCHED.has ? "" : "none";
    $("#track").textContent = STATUS.track;
    $("#mode").textContent = MODES[STATUS.mode] || (STATUS.mode === "track" ? "track" : "FPVTrackside event");
    const ph = $("#phase");
    if (STATUS.phase === "learn") { ph.className = "badge learn"; ph.textContent = "track not learned yet"; }
    else { ph.className = "badge result"; ph.textContent = "results"; }
    const ch = (STATUS.live && STATUS.live.channels) || {};
    const names = (STATUS.pilot_info && STATUS.pilot_info.pilots) || {};
    $("#live").innerHTML = (SCHED && SCHED.current ? `<a class="chip on" href="${B()}tv">${esc(SCHED.current.label)}</a>` : "") +
      (STATUS.live && STATUS.live.heat ? `<span class="chip">Heat ${STATUS.live.heat}</span>` : "") +
      (STATUS.pilot_info && STATUS.pilot_info.round != null && ["scheduled", "prestart", "running"].includes(STATUS.pilot_info.phase)
        ? `<span class="chip">${STATUS.pilot_info.stage ? esc(STATUS.pilot_info.stage) + " · " : ""}Round ${esc(STATUS.pilot_info.round)} · Race ${esc(STATUS.pilot_info.race ?? "–")} · ${esc(STATUS.pilot_info.phase)}</span>` : "") +
      Object.entries(ch).map(([c, s]) => {
        const on = s.state === "live", drop = s.state === "dropout";
        const who = names[String(c).replace(/\s/g, "").toUpperCase()];
        return `<span class="chip ${on ? "on" : drop ? "drop" : ""}">${esc(c)}${who ? " " + esc(who) : ""} ${on || drop ? sec(s.seconds, 0) + "s" : "waiting"}</span>`;
      }).join("") +
      (STATUS.queue.analysing + STATUS.queue.waiting ? `<span class="chip">analysing ${STATUS.queue.analysing + STATUS.queue.waiting}</span>` : "");
  } catch (e) { /* server restarting */ }
}

// ── router ─────────────────────────────────────────────────────
let current = { key: null, timer: null };
function route() {
  // #/ = every event; #/e/<event>/[page/arg] = one event (parts decoded one by one: ids contain "/")
  const parts = (location.hash.slice(1) || "/").split("/").map(decodeURIComponent);
  const key = location.hash;
  if (current.timer) clearInterval(current.timer);
  current = { key, timer: null };
  let view, arg;
  if (parts[1] === "e" && parts[2]) {
    EV = parts[2];
    const views = { "": home, run: runPage, heat: heatPage, pilot: pilotPage, track: trackPage,
                    upcoming: upcomingPage, tv: tvPage, ghost: ghostPage };
    view = views[parts[3] || ""] || home;
    arg = parts[4];
  } else { EV = null; view = eventsPage; }
  $("#nav-res").style.display = $("#nav-track").style.display = EV ? "" : "none";
  if (EV) { $("#nav-res").href = B(); $("#nav-track").href = B() + "track"; $("#nav-up").href = B() + "upcoming"; $("#nav-tv").href = B() + "tv"; }
  refreshStatus();
  const arg2 = parts[5];
  view(arg, true, arg2).catch(showErr);
  current.timer = setInterval(() => view(arg, false, arg2).catch(() => {}), view.every || 5000);
  window.scrollTo(0, 0);
}
function showErr(e) { $("#app").innerHTML = `<p class="muted">Couldn't load (${esc(e)}).</p>`; }
window.addEventListener("hashchange", route);

// ── every event of the data folder ─────────────────────────────
async function eventsPage(_, first) {
  const d = await api("/api/events");
  const date = (t) => new Date(t * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  $("#app").innerHTML = `<h1>Events</h1>
    ${hint(`Every track and FPVTrackside event in the data folder (${esc(d.root)}), newest first. The one open in the race-day app is marked “open”.`)}
    <div class="grid cards">${d.events.map((e) => `<a class="card" href="#/e/${encodeURIComponent(e.id)}/">
      <div class="who">${esc(e.name)}</div>
      <div class="meta">${e.kind === "track" ? "track" : "FPVTrackside event"} · ${date(e.time)}${e.open ? ' · <span class="good">open</span>' : ""}</div>
      <div style="margin-top:8px">${e.runs ? `<b>${e.runs}</b> runs` : '<span class="muted">no results yet</span>'}
        ${e.learned ? '<span class="pill good">learned</span>' : '<span class="pill">not learned</span>'}</div></a>`).join("")
      || '<p class="muted">No events in the data folder.</p>'}</div>`;
}

// ── lap bar: the track drawn as sections ───────────────────────
function lapBar(sections, times) {
  if (!sections || !times || !times.some((x) => x)) return "";
  const tot = times.reduce((a, b) => a + (b || 0), 0);
  return `<div class="lapbar">${sections.map((s, i) => {
    const w = (100 * (times[i] || 0)) / tot;
    return `<div style="width:${w}%;background:hsl(${(i * 360) / sections.length},45%,${i % 2 ? 26 : 32}%)" title="${esc(s)} ${sec(times[i])}s">${w > 6 ? esc(s.replace("→", "›")) : ""}</div>`;
  }).join("")}</div>`;
}

// ── tables sorted by a click on a column's title ───────────────
// cols: [{k, label, tip, val(row) → number/text to sort by, html(row), cls(row), dir: 1 smallest first / -1 biggest first}]
const SORT = {};
function sortTable(id, cols, rows, def, attrs, rank = true) {
  const s = (SORT[id] ??= { k: def, dir: (cols.find((c) => c.k === def) || cols[0]).dir || 1 });
  const col = cols.find((c) => c.k === s.k) || cols[0];
  const sorted = [...rows].sort((a, b) => {
    const x = col.val(a), y = col.val(b);
    if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1;       // empty ones last
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * s.dir;
  });
  return `<table id="${id}" class="sort${rank ? " rank" : ""}"><thead><tr>${rank ? "<th>#</th>" : ""}${cols.map((c) =>
    `<th data-k="${c.k}" data-dir="${c.dir || 1}" title="${esc(c.tip || "")}" class="${c.k === s.k ? "sorted" : ""}">${c.label}${c.k === s.k ? (s.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr></thead>
    <tbody>${sorted.map((r, i) => `<tr ${attrs ? attrs(r) : ""}>${rank ? `<td>${i + 1}</td>` : ""}${cols.map((c) => `<td class="${c.cls ? c.cls(r) : ""}">${c.html(r)}</td>`).join("")}</tr>`).join("")
      || `<tr><td colspan="${cols.length + 1}" class="muted">Nothing yet.</td></tr>`}</tbody></table>`;
}
document.addEventListener("click", (e) => {
  const th = e.target.closest("table.sort th[data-k]");
  if (!th) return;
  const id = th.closest("table").id, s = SORT[id];
  SORT[id] = s && s.k === th.dataset.k ? { k: s.k, dir: -s.dir } : { k: th.dataset.k, dir: +th.dataset.dir };
  if (current.draw) current.draw();
});

// ── compare with: the field (median) or another pilot of the class ──
const CMP = {};
const cmpKey = (lay) => `cmp:${EV}:${lay}`;
function cmpGet(lay) {
  const k = cmpKey(lay);
  if (!(k in CMP)) { try { CMP[k] = localStorage.getItem(k) || ""; } catch (e) { CMP[k] = ""; } }
  return CMP[k];
}
function cmpSet(lay, v) {
  CMP[cmpKey(lay)] = v;
  try { localStorage.setItem(cmpKey(lay), v); } catch (e) { /* private window */ }
  if (current.draw) current.draw();
}
// What a pilot is compared with. The field: its median lap and median section times. Another pilot: their
// best lap and their best time in every section (typical = their median lap).
function refOf(cls, me) {
  const f = (cls && cls.field) || {};
  const name = cls ? cmpGet(cls.layout) : "";
  const p = name && name !== me ? cls.pilots.find((x) => x.pilot === name) : null;
  if (!p) return { field: true, who: "the field", label: "field median", lap: f.median_lap, typical: f.median_lap,
                   sections: f.section_median || [], named: ((cls && cls.named) || []).map((x) => x.median) };
  return { field: false, who: name, label: `${name}'s best`, lap: p.best_lap, typical: p.median_lap,
           sections: p.section_best, named: p.named_best, p };
}
function cmpSelect(cls, me) {
  if (!cls || cls.pilots.length < 2) return "";
  const cur = cmpGet(cls.layout);
  const others = cls.pilots.filter((p) => p.pilot !== me).sort((a, b) => (a.best_lap ?? 1e9) - (b.best_lap ?? 1e9));
  return `<div class="cmp panel"><label>Compare with
    <select data-lay="${esc(cls.layout)}" onchange="cmpSet(this.dataset.lay,this.value)">
      <option value="">The field (median)</option>
      ${others.map((p) => `<option value="${esc(p.pilot)}"${p.pilot === cur ? " selected" : ""}>${esc(p.pilot)}${p.best_lap != null ? " · best " + sec(p.best_lap) : ""}</option>`).join("")}
    </select></label>
    ${hint("The field = the typical pilot here (median). A pilot = their best lap and their best time in every section. The laps, charts and bars below follow your choice.")}</div>`;
}

// ── home: the event's results ──────────────────────────────────
const rowN = (pb) => (pb >= 2 ? `Best ${pb} laps` : null);
const clock = (t) => (t ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "–");
const runLink = (stem, text) => (stem ? `<a href="${B()}run/${encodeURIComponent(stem)}" onclick="event.stopPropagation()">${text}</a>` : text);

async function home(_, first) {
  const [st, res] = await Promise.all([api("/api/status"), api("/api/results")]);
  if (!res.classes.length && st.phase === "learn") {
    const [data, sc] = await Promise.all([api("/api/runs"), api("/api/schedule").catch(() => ({ has: false }))]);
    return learnView(st, data.runs, sc);
  }
  const sig = JSON.stringify(res);
  if (!first && sig === current.sig) return;            // nothing new: the page stays as it is (scroll, sorting)
  current.sig = sig;
  current.draw = () => drawHome(res);
  drawHome(res);
}

function board(c, i, pb) {
  const tm = (k, d = 2) => ({ val: (p) => p[k], html: (p) => sec(p[k], d) });
  const cols = [
    { k: "pilot", label: "Pilot", val: (p) => p.pilot.toLowerCase(), html: (p) => `<span style="color:${colorFor(p.pilot)}">●</span> ${esc(p.pilot)}` },
    { k: "best_lap", label: "Best lap", tip: "Fastest single lap", val: (p) => p.best_lap, html: (p) => `<b>${runLink(p.best_lap_run, sec(p.best_lap))}</b>` },
    ...(rowN(pb) ? [{ k: "best_row", label: rowN(pb), tip: `Fastest ${pb} laps in a row, in one heat`, val: (p) => p.best_row, html: (p) => runLink(p.best_row_run, sec(p.best_row)) }] : []),
    ...(pb !== 3 ? [{ k: "best3", label: "Best 3 laps", tip: "Fastest 3 laps in a row, in one heat", ...tm("best3") }] : []),
    { k: "median_lap", label: "Typical lap", tip: "The median of all their laps", ...tm("median_lap") },
    { k: "theoretical_best", label: "Theoretical best", tip: "Their best time in every section of one heat, added up", ...tm("theoretical_best") },
    { k: "to_gain", label: "To gain", tip: "Best lap minus theoretical best: the time still in it", val: (p) => p.to_gain, html: (p) => (p.to_gain == null ? "–" : sec(p.to_gain)) },
    { k: "lap_cv", label: "Consistency", tip: "How much their laps vary (smaller = steadier)", val: (p) => p.lap_cv, html: (p) => `${pct(p.lap_cv)} ${steadiness(p.lap_cv)}` },
    { k: "speed_index", label: "Speed vs field", dir: -1, tip: "Field's typical lap ÷ their typical lap", val: (p) => p.speed_index,
      html: (p) => (p.speed_index ? "×" + sec(p.speed_index) : "–"), cls: (p) => (p.speed_index > 1.005 ? "good" : p.speed_index < 0.995 ? "bad" : "") },
    { k: "holeshot", label: "Holeshot", tip: "Fastest from the start to the first crossing", val: (p) => p.holeshot, html: (p) => runLink(p.holeshot_run, sec(p.holeshot)) },
    { k: "laps", label: "Laps", dir: -1, tip: "Full laps flown in the event", val: (p) => p.laps, html: (p) => p.laps },
    { k: "runs", label: "Heats", dir: -1, tip: "Heats flown (with an analysed video)", val: (p) => p.runs, html: (p) => p.runs },
  ];
  return sortTable("board" + i, cols, c.pilots, "best_lap", (p) => `class="click" onclick="location.hash='${B()}pilot/${encodeURIComponent(p.pilot)}'"`);
}

function records(c, pb) {
  const top = (k, dir = 1) => c.pilots.filter((p) => p[k] != null).sort((a, b) => (a[k] - b[k]) * dir)[0];
  const tile = (title, p, k, run, h, fmt = sec) => (p ? fact(title, `<div class="v">${runLink(p[run], fmt(p[k]))}</div>`, `${esc(p.pilot)}${h ? " · " + h : ""}`) : "");
  const steady = c.pilots.filter((p) => p.lap_cv != null && p.laps >= 6).sort((a, b) => a.lap_cv - b.lap_cv)[0];
  return `<div class="facts">
    ${tile("Fastest lap", top("best_lap"), "best_lap", "best_lap_run")}
    ${rowN(pb) ? tile(rowN(pb), top("best_row"), "best_row", "best_row_run", "in a row") : ""}
    ${tile("Best holeshot", top("holeshot"), "holeshot", "holeshot_run")}
    ${tile("Steadiest", steady, "lap_cv", "", "laps vary least", pct)}
    ${tile("Most improved", top("improved_pct", -1), "improved", "", "off their first heat's best lap", (x) => "−" + sec(x) + " s")}
    ${tile("Most laps", top("laps", -1), "laps", "", "", (x) => x)}
    ${fact("Typical lap", `<div class="v">${sec(c.field.median_lap)}</div>`, "The field's median lap.")}
  </div>`;
}

function fastestSections(c) {
  const f = c.field;
  const named = c.named.filter((x) => x.best).map((x) => `<span class="pill sec"><b>${esc(x.name)}</b> <span class="muted">G${x.from}→G${x.to}</span> <b>${sec(x.best.time)}</b> ${runLink(x.best.run, esc(x.best.pilot))}</span>`).join(" ");
  const legs = (f.section_best || []).map((b, i) => (b ? `<span class="pill sec">${esc(f.sections[i])} <b>${sec(b.time)}</b> ${runLink(b.run, esc(b.pilot))}</span>` : "")).join(" ");
  if (!named && !legs) return "";
  return `<h3>Fastest sections</h3>
    ${hint("Who holds the best time in each part of the track. Example: “G9→G10 0.48 Talper” — nobody flew from gate 9 to gate 10 faster than 0.48 s. Tap a name to open that run.")}
    <div class="panel">${named ? `<div class="k">Named sections</div><div>${named}</div><div class="k" style="margin-top:10px">Gate to gate</div>` : ""}<div>${legs}</div></div>`;
}

// Racing against others: the event's races (not its time trials)
function craftTable(c, i) {
  const rows = c.pilots.filter((p) => p.craft && p.craft.races);
  if (!rows.length) return "";
  const num = (k, d = 1) => ({ val: (p) => p.craft[k], html: (p) => sec(p.craft[k], d) });
  const cols = [
    { k: "pilot", label: "Pilot", val: (p) => p.pilot.toLowerCase(), html: (p) => `<span style="color:${colorFor(p.pilot)}">●</span> ${esc(p.pilot)}` },
    { k: "races", label: "Races", dir: -1, val: (p) => p.craft.races, html: (p) => p.craft.races },
    { k: "wins", label: "Wins", dir: -1, val: (p) => p.craft.wins, html: (p) => `<b>${p.craft.wins}</b>` },
    { k: "avg_place", label: "Average place", tip: "Finishing place, averaged over their races", ...num("avg_place") },
    { k: "avg_holeshot_place", label: "Place at the holeshot", tip: "Their place at the first crossing, averaged", ...num("avg_holeshot_place") },
    { k: "gained", label: "Places gained", dir: -1, tip: "From the holeshot to the finish, a race on average: +1 = one place gained",
      val: (p) => p.craft.gained, html: (p) => (p.craft.gained == null ? "–" : (p.craft.gained > 0 ? "+" : "") + sec(p.craft.gained, 1)),
      cls: (p) => (p.craft.gained > 0.05 ? "good" : p.craft.gained < -0.05 ? "bad" : "") },
  ];
  return `<h3>Racing</h3>${hint("The races flown against others (time trials aside). Place at the holeshot = where they were at the first crossing; Places gained +1.0 = they finish one place better than they started, on average — passing in the race; a minus = places lost after the start.")}
    <div class="panel scroll">${sortTable("craft" + i, cols, rows, "avg_place", (p) => `class="click" onclick="location.hash='${B()}pilot/${encodeURIComponent(p.pilot)}'"`)}</div>`;
}

// Best lap heat by heat, one line per pilot; tap a name to bring its line out
const HI = {};
function hiSet(lay, name) { HI[lay] = HI[lay] === name ? null : name; if (current.draw) current.draw(); }
function progressChart(c) {
  const ps = c.pilots.filter((p) => p.progress.length >= 2);
  if (ps.length < 1) return "";
  const pts = ps.flatMap((p) => p.progress), heats = pts.map((x) => x[0]), laps = pts.map((x) => x[1]).sort((a, b) => a - b);
  const h0 = Math.min(...heats), h1 = Math.max(...heats), lo = laps[0] * 0.97, hi = Math.min(laps[laps.length - 1], laps[Math.floor(laps.length / 2)] * 1.5) * 1.02;
  const W = 900, H = 280, pl = 40, pr = 12, pt = 12, pb = 26;
  const X = (h) => pl + ((h - h0) * (W - pl - pr)) / Math.max(1, h1 - h0), Y = (t) => H - pb - ((Math.min(t, hi) - lo) / (hi - lo)) * (H - pt - pb);
  const cur = HI[c.layout];
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo));
  const lines = [...ps].sort((a, b) => (a.pilot === cur) - (b.pilot === cur)).map((p) => {
    const on = !cur || p.pilot === cur, col = colorFor(p.pilot);
    return `<g opacity="${on ? 1 : 0.12}"><polyline points="${p.progress.map((x) => `${X(x[0])},${Y(x[1])}`).join(" ")}" fill="none" stroke="${col}" stroke-width="${p.pilot === cur ? 3 : 1.6}"/>
      ${p.progress.map((x) => `<circle cx="${X(x[0])}" cy="${Y(x[1])}" r="${p.pilot === cur ? 4 : 2.6}" fill="${col}"><title>${esc(p.pilot)} · heat ${x[0]} · ${sec(x[1])}</title></circle>`).join("")}</g>`;
  }).join("");
  return `<h3>Progress through the event</h3>${hint("Each pilot's best lap in every heat they flew, in the order flown: a line going down = getting faster. Tap a name to bring its line out (again to show all).")}
    <div class="panel"><svg viewBox="0 0 ${W} ${H}">
      ${ticks.map((t) => `<line x1="${pl}" x2="${W - pr}" y1="${Y(t)}" y2="${Y(t)}" stroke="#2a3347"/><text x="${pl - 6}" y="${Y(t) + 4}" fill="#8a96ad" font-size="11" text-anchor="end">${sec(t, 1)}</text>`).join("")}
      <text x="${pl}" y="${H - 6}" fill="#8a96ad" font-size="11">heat ${h0}</text><text x="${W - pr}" y="${H - 6}" fill="#8a96ad" font-size="11" text-anchor="end">heat ${h1}</text>${lines}</svg>
      <div style="margin-top:8px">${[...ps].sort((a, b) => a.best_lap - b.best_lap).map((p) => `<span class="pill pick${p.pilot === cur ? " on" : ""}" data-lay="${esc(c.layout)}" data-p="${esc(p.pilot)}" onclick="hiSet(this.dataset.lay,this.dataset.p)"><span style="color:${colorFor(p.pilot)}">●</span> ${esc(p.pilot)}${p.improved ? ` <span class="good">−${sec(p.improved)}</span>` : ""}</span>`).join("")}</div></div>`;
}

// Pilots × sections: every pilot's best time in every section, coloured by their rank in it
function sectionMap(c) {
  const f = c.field, ps = [...c.pilots].sort((a, b) => (a.best_lap ?? 1e9) - (b.best_lap ?? 1e9));
  if (ps.length < 2 || !f.sections) return "";
  const colsN = c.named.map((x, j) => ({ label: esc(x.name), get: (p) => p.named_best[j] }));
  const colsL = f.sections.map((s, j) => ({ label: esc(s.replace("→", "›")), get: (p) => p.section_best[j] }));
  const cols = [...colsN, ...colsL];
  const rank = cols.map((col) => { const v = ps.map(col.get).filter((x) => x != null).sort((a, b) => a - b); return (x) => (v.length > 1 ? v.indexOf(x) / (v.length - 1) : 0); });
  const paint = (r) => { const g = [62, 207, 142], b = [255, 107, 107], k = r < 0.5 ? g : b, a = Math.abs(r - 0.5) * 2; return `background:rgba(${k[0]},${k[1]},${k[2]},${(0.08 + 0.6 * a).toFixed(2)})`; };
  return `<h3>Section map</h3>${hint("Every pilot's best time in every section, fastest lap first. Green = among the fastest of the class in that section, red = among the slowest; bold = the fastest. A fast pilot's red cell is where they give time away; a green cell low in the table is that pilot's strong point.")}
    <div class="panel scroll"><table class="map"><thead><tr><th>Pilot</th>${cols.map((x, j) => `<th class="${j < colsN.length ? "named" : ""}">${x.label}</th>`).join("")}</tr></thead><tbody>
      ${ps.map((p) => `<tr class="click" onclick="location.hash='${B()}pilot/${encodeURIComponent(p.pilot)}'"><td><span style="color:${colorFor(p.pilot)}">●</span> ${esc(p.pilot)}</td>${cols.map((col, j) => { const v = col.get(p); return v == null ? '<td class="muted">–</td>' : `<td><div class="cell" style="${paint(rank[j](v))}${rank[j](v) === 0 ? ";font-weight:800" : ""}">${sec(v)}</div></td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function heatsTable(res) {
  const many = res.classes.length > 1;
  const cname = Object.fromEntries(res.classes.map((c) => [c.layout, c.name || c.layout]));
  const win = (h) => h.pilots.find((p) => p.position === 1);
  const fast = (h) => h.pilots.filter((p) => p.best_lap != null).sort((a, b) => a.best_lap - b.best_lap)[0];
  const cols = [
    { k: "start", label: "Heat", val: (h) => h.start, html: (h) => (h.heat != null ? `<b>${h.heat}</b>` : '<span class="muted">–</span>') },
    { k: "label", label: "Race", val: (h) => h.label, html: (h) => `${esc(h.label)} ${h.type === "TimeTrial" ? '<span class="pill">time trial</span>' : ""}${h.heat == null ? ' <span class="pill">no video analysed</span>' : ""}` },
    ...(many ? [{ k: "cls", label: "Class", val: (h) => cname[h.layout] || "", html: (h) => esc(cname[h.layout] || "–") }] : []),
    { k: "time", label: "Time", val: (h) => h.start, html: (h) => clock(h.start) },
    { k: "winner", label: "Winner", val: (h) => (win(h) || {}).pilot || null, html: (h) => (win(h) ? `<span style="color:${colorFor(win(h).pilot)}">●</span> ${esc(win(h).pilot)}` : "–") },
    { k: "fast", label: "Fastest lap", val: (h) => (fast(h) || {}).best_lap ?? null, html: (h) => (fast(h) ? `<b>${sec(fast(h).best_lap)}</b> ${esc(fast(h).pilot)}` : "–") },
    { k: "laps", label: "Laps", dir: -1, tip: "Laps flown by all the pilots together", val: (h) => h.pilots.reduce((a, p) => a + p.laps, 0), html: (h) => h.pilots.reduce((a, p) => a + p.laps, 0) },
    { k: "pilots", label: "Finishing order", val: (h) => h.pilots.length, html: (h) => h.pilots.map((p) => `<span class="upchip">${p.position ?? "–"}. ${esc(p.pilot)} <span class="muted">&nbsp;${p.laps} lap${p.laps === 1 ? "" : "s"}</span></span>`).join("") },
  ];
  SORT.heats ??= { k: "start", dir: 1 };
  return sortTable("heats", cols, res.heats, "start", (h) => (h.heat != null ? `class="click" onclick="location.hash='${B()}heat/${h.heat}'"` : ""), false);
}

function drawHome(res) {
  const pb = res.pb_laps;
  $("#app").innerHTML = `<h1>Results</h1>
    ${hint(`Tap a column's title to sort by it (again to turn it round). Best lap = fastest single lap. ${rowN(pb) ? `${rowN(pb)} = fastest ${pb} laps in a row. ` : ""}Typical lap = the median of all the pilot's laps. Theoretical best = the best time in every section of one heat added up — the lap they could fly; To gain = how far the best lap is from it. Consistency 2.5% = laps vary about 2.5% (smaller = steadier). Speed vs field ×1.05 = 5% faster than the typical pilot here. Tap a pilot for their page.`)}
    ${res.classes.map((c, i) => `${c.name ? `<h2>${esc(c.name)} <span class="muted">· ${c.n_gates} gates · ${c.pilots.length} pilots</span></h2>` : "<h2>Leaderboard</h2>"}
      ${lapBar(c.field.sections, c.field.section_median)}
      ${records(c, pb)}
      <div class="panel scroll" style="margin-top:12px">${board(c, i, pb)}</div>
      ${craftTable(c, i)}
      ${progressChart(c)}
      ${fastestSections(c)}
      ${sectionMap(c)}`).join("") || '<p class="muted">No runs with laps yet.</p>'}
    <h2>Heats</h2>
    ${hint("Every heat of the event in the order flown. Tap a heat to see it: the pilots' videos side by side, the result with every lap, and the gap to the leader at every gate.")}
    <div class="panel scroll">${heatsTable(res)}</div>`;
}

function roundRace(r) {
  if (!r || r.round == null) return "";
  return `${r.stage ? esc(r.stage) + " · " : ""}Round ${esc(r.round)} · Race ${esc(r.race ?? "–")} · `;
}

function learnView(st, runs, sc) {
  $("#app").innerHTML = `${raceDay(sc)}
    <h1>The track isn't learned yet</h1>
    ${hint("Gates and laps appear once the track is learned: in the race-day app, tick ☑ a few heats with full laps and press Learn track.")}
    <h2>Runs analysed</h2>
    <div class="panel scroll"><table><thead><tr><th>Run</th><th>Pilot</th><th>Heat</th><th>Channel</th><th>Length</th><th>Status</th></tr></thead>
      <tbody>${runs.map((r) => `<tr><td>${esc(r.stem)}</td><td>${esc(r.pilot)}</td><td>${r.heat}</td><td>${esc(r.channel)}</td><td>${sec(r.live_s, 0)}s</td><td>${esc(r.status)}</td></tr>`).join("")}</tbody></table></div>`;
}

// ── run page ───────────────────────────────────────────────────
async function runPage(stem, first, at) {
  const [r, res] = await Promise.all([api("/api/run/" + encodeURIComponent(stem)), api("/api/results").catch(() => null)]);
  const st = r.stats;
  if (!first) { if (st && !$("#lapTable")) return runPage(stem, true); return; }   // keep the video playing
  const head = `<h1><span style="color:${colorFor(r.pilot)}">●</span> ${esc(r.pilot)} ${reviewPill(r)}</h1>
    <p class="muted">${roundRace(r)}<a href="${B()}heat/${r.heat}">Heat ${r.heat}</a> · ${esc(r.channel)} · <a href="${B()}pilot/${encodeURIComponent(r.pilot)}">all runs of ${esc(r.pilot)}</a></p>`;
  if (!st) { $("#app").innerHTML = head + `<div class="panel">This run is ${esc(r.status)} — results appear when it's analysed${STATUS && STATUS.phase === "learn" ? " and the track is learned" : ""}.</div>
    ${videoTag(r, "")}`; return; }
  const cls = res && res.classes.find((c) => c.layout === r.layout && c.n_gates === st.n_gates);
  $("#app").innerHTML = head + `
    ${videoTag(r, 'id="vid"')}
    <div id="timeline" class="panel" style="padding:6px 10px"></div>
    ${hint("Timeline: thin blue lines = gates passed, yellow lines = start/finish (G1, a new lap), L1 L2 … = laps, red = where the video is now. Tap the bar to jump there.")}
    <div id="parts"></div>`;
  current.draw = () => { $("#parts").innerHTML = runParts(r, cls); };       // the video keeps playing
  current.draw();
  drawTimeline(r);
  if (at != null && isFinite(+at)) {                    // opened at a moment (a missed gate, a slow lap)
    const v = $("#vid"), go = () => { v.currentTime = Math.max(0, +at); };
    if (v.readyState >= 1) go(); else v.addEventListener("loadedmetadata", go, { once: true });
  }
}

// Everything under a run's video: it follows "Compare with"
function runParts(r, klass) {
  const st = r.stats, field = r.field || {}, best = st.best_lap_index, live0 = r.live_start_s || 0;
  const ref = klass ? refOf(klass, r.pilot) : { field: true, who: "the field", label: "field median", lap: field.median_lap,
                                             typical: field.median_lap, sections: field.section_median || [], named: [] };
  const si = ref.typical && st.median_lap ? ref.typical / st.median_lap : null;
  const diffs = st.section_best.map((v, i) => (v != null && ref.sections[i] != null ? v - ref.sections[i] : null));
  const named = (r.named || []).map((x, i) => (x.best != null && ref.named[i] != null ? x.best - ref.named[i] : null));
  const against = ref.field ? "the typical pilot" : `${esc(ref.who)}'s best`;
  return `${cmpSelect(klass, r.pilot)}${ghostLink(r.pilot, ref)}
    <div class="facts">
      ${fact("Best lap", `<div class="v good">${sec(st.best_lap)}</div>`, "Your fastest full lap (G1 to G1), in seconds.")}
      ${fact("Average lap", `<div class="v">${sec(st.mean_lap)}</div>`, "The mean of all your full laps in this run.")}
      ${fact("Laps", `<div class="v">${st.laps}</div>`, "Full laps flown (start/finish to start/finish).")}
      ${fact("Consistency", `<div class="v">±${sec(st.lap_sd)}</div>`, "±0.40 = your laps vary about 0.4 s around the average. Smaller = steadier.", steadiness(st.lap_cv))}
      ${fact("Theoretical best", `<div class="v">${sec(st.theoretical_best)}</div>`, "Your best time in each section added up — the lap you could fly if you put it all together.")}
      ${fact(`Speed vs ${esc(ref.field ? "field" : ref.who)}`, `<div class="v ${si > 1.005 ? "good" : si < 0.995 ? "bad" : ""}">${si ? "×" + sec(si) : "–"}</div>`,
        ref.field ? "×1.05 = 5% faster than the typical pilot on this track, ×0.97 = 3% slower." : `Your typical lap in this run against ${esc(ref.who)}'s typical lap: ×1.05 = 5% faster, ×0.97 = 3% slower.`)}
      ${fact(`Holeshot (to G${st.first_gate || 1})`, `<div class="v">${sec(st.holeshot)}</div>`, "Seconds from take-off to the first gate.")}
      ${fact("Missed gates", `<div class="v ${st.missed_gates.length ? "bad" : ""}">${st.missed_gates.length}</div>`, "Gates skipped (or not seen) between two passes. Each one is listed below with a link to the moment in the video.")}
    </div>
    ${st.missed_gates.length ? `<p class="muted">Missed: ${st.missed_gates.map((m) => `<a href="#" onclick="seek(${m.t - 1});return false">G${m.gate} at ${sec(m.t - live0, 1)}s</a>`).join(", ")}${st.ended_mid_lap ? " · run ended mid-lap" : ""}</p>` : st.ended_mid_lap ? '<p class="muted">Run ended mid-lap.</p>' : ""}
    <h2>Laps</h2>
    ${hint(`vs own best +0.35 = this lap was 0.35 s slower than your best lap. vs ${esc(ref.label)}: green = faster than ${ref.field ? "the typical lap on this track" : esc(ref.who) + "'s best lap"}, red = slower. In the chart, the dashed line is ${ref.field ? "the typical lap" : esc(ref.who) + "'s best lap"}. Tap a lap to watch it.`)}
    <div class="panel scroll"><table id="lapTable"><thead><tr><th>Lap</th><th>Time</th><th>vs own best</th><th>vs ${esc(ref.label)}</th></tr></thead><tbody>
      ${st.lap_times.map((t, i) => `<tr class="click ${i === best ? "best" : ""}" onclick="seek(${r.lap_marks[i].t0 - 0.5})"><td>${i + 1}</td><td>${sec(t)}</td>
        <td>${delta(t - st.best_lap)}</td><td class="${cls(ref.lap ? t - ref.lap : null)}">${ref.lap ? delta(t - ref.lap) : "–"}</td></tr>`).join("")}
    </tbody></table>${lapChart(st.lap_times, ref.lap, ref.label)}</div>
    <h2>Sections</h2>
    ${hint(`One row per lap, one column per section of the track. Green cell = faster than ${against} in that section, red = slower. Example: a red column means that section costs you time on every lap.`)}
    <div class="panel scroll"><table><thead><tr><th>Lap</th>${st.sections.map((s) => `<th>${esc(s)}</th>`).join("")}</tr></thead><tbody>
      ${st.section_grid.map((row, i) => `<tr><td>${i + 1}</td>${row.map((v, j) => `<td><div class="cell" style="${heat(v, ref.sections[j])}">${sec(v)}</div></td>`).join("")}</tr>`).join("")}
      <tr class="best"><td>Best</td>${st.section_best.map((v) => `<td>${sec(v)}</td>`).join("")}</tr>
      ${ref.field ? "" : `<tr><td>${esc(ref.label)}</td>${ref.sections.map((v) => `<td>${sec(v)}</td>`).join("")}</tr>`}
      <tr><td class="muted">Field median</td>${(field.section_median || []).map((v) => `<td class="muted">${sec(v)}</td>`).join("")}</tr>
      <tr><td class="muted">Field best</td>${(field.section_best || []).map((b) => `<td class="muted">${b ? sec(b.time) : "–"}</td>`).join("")}</tr>
    </tbody></table></div>
    <h2>Where time is gained or lost</h2>
    ${hint(`Your best time in each section against ${against}. Green bar to the left = you gain time there (−0.20 = 0.2 s faster). Red bar to the right = you lose time there.`)}
    ${named.some((d) => d != null) ? `<div class="panel"><div class="k">Named sections</div>${sectionBars(r.named.map((x) => `${x.name} (G${x.from}→G${x.to})`), named, 170)}</div>` : ""}
    <div class="panel">${sectionBars(st.sections, diffs)}</div>`;
}

function seek(t) { const v = $("#vid"); if (v) { v.currentTime = Math.max(0, t); v.play().catch(() => {}); } }

function drawTimeline(r) {
  const v = $("#vid"), box = $("#timeline");
  if (!v || !box) return;
  const draw = () => {
    const dur = v.duration || Math.max(1, ...r.passes.map((p) => p.t)) + 2;
    const X = (t) => (100 * t) / dur;
    // drawn in screen pixels (no stretched viewBox), so gate lines stay 1.5 px thin on any screen
    const laps = r.lap_marks.map((l, i) => `<rect x="${X(l.t0)}%" y="0" width="${Math.max(0, X(l.t1) - X(l.t0))}%" height="34" fill="${i % 2 ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0.05)"}"/>
      <text x="${X((l.t0 + l.t1) / 2)}%" y="11" fill="#8a96ad" font-size="10" text-anchor="middle">L${l.lap}</text>`).join("");
    const passes = r.passes.filter((p) => p.gate >= 1).map((p) => p.gate === 1
      ? `<line x1="${X(p.t)}%" x2="${X(p.t)}%" y1="2" y2="34" stroke="#f5c451" stroke-width="1.5"/>`
      : `<line x1="${X(p.t)}%" x2="${X(p.t)}%" y1="15" y2="34" stroke="#4da3ff" stroke-width="1.5"/>`).join("");
    box.innerHTML = `<svg width="100%" height="34" style="cursor:pointer;display:block">${laps}${passes}
      <line id="ph" x1="${X(v.currentTime)}%" x2="${X(v.currentTime)}%" y1="0" y2="34" stroke="#ff3b5c" stroke-width="2"/></svg>`;
    box.querySelector("svg").onclick = (e) => { const b = e.currentTarget.getBoundingClientRect(); seek(((e.clientX - b.left) / b.width) * dur); };
  };
  v.addEventListener("loadedmetadata", draw);
  v.addEventListener("timeupdate", () => { const ph = $("#ph"); if (ph && v.duration) { const x = (100 * v.currentTime) / v.duration + "%"; ph.setAttribute("x1", x); ph.setAttribute("x2", x); } });
  draw();
}

function lapChart(times, ref, label = "field median") {
  if (!times || times.length < 2) return "";
  const W = 600, H = 120, pad = 24, lo = Math.min(...times, ref ?? 1e9) * 0.97, hi = Math.max(...times, ref ?? 0) * 1.03;
  const x = (i) => pad + (i * (W - 2 * pad)) / (times.length - 1), y = (t) => H - pad - ((t - lo) / (hi - lo)) * (H - 2 * pad);
  const pts = times.map((t, i) => `${x(i)},${y(t)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" style="margin-top:10px">
    ${ref ? `<line x1="${pad}" x2="${W - pad}" y1="${y(ref)}" y2="${y(ref)}" stroke="#8a96ad" stroke-dasharray="4 4"/><text x="${W - pad}" y="${y(ref) - 4}" fill="#8a96ad" font-size="11" text-anchor="end">${esc(label)}</text>` : ""}
    <polyline points="${pts}" fill="none" stroke="#4da3ff" stroke-width="2.5"/>
    ${times.map((t, i) => `<circle cx="${x(i)}" cy="${y(t)}" r="4" fill="#4da3ff"/><text x="${x(i)}" y="${H - 6}" fill="#8a96ad" font-size="11" text-anchor="middle">L${i + 1}</text>`).join("")}</svg>`;
}

function sectionBars(sections, diffs, w0 = 76) {
  if (!diffs || !diffs.some((d) => d != null)) return '<p class="muted">Needs more runs on the track.</p>';
  const m = Math.max(0.05, ...diffs.filter((d) => d != null).map(Math.abs));
  return sections.map((s, i) => {
    const d = diffs[i]; if (d == null) return "";
    const w = (50 * Math.abs(d)) / m;
    return `<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><div style="width:${w0}px;max-width:42%;font-size:12px;overflow-wrap:anywhere">${esc(s)}</div>
      <div style="flex:1;position:relative;height:16px;background:var(--panel2);border-radius:4px">
        <div style="position:absolute;top:0;bottom:0;${d < 0 ? `right:50%` : `left:50%`};width:${w}%;background:${d < 0 ? "var(--good)" : "var(--bad)"};border-radius:3px"></div>
        <div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted)"></div></div>
      <div style="width:56px;text-align:right" class="${cls(d)}">${delta(d)}</div></div>`;
  }).join("");
}

// ── heat page: 4 videos in sync + gap chart ────────────────────
async function heatPage(n, first) {
  const h = await api("/api/heat/" + n);
  if (!first) return;
  const runs = h.runs;
  const L = Math.min(...runs.filter((r) => r.stats && r.stats.laps).map((r) => r.stats.laps));
  const order = runs.filter((r) => r.stats && r.stats.laps >= L && isFinite(L)).map((r) => {
    const lt = r.stats.lap_times.slice(0, L).reduce((a, b) => a + b, 0);
    return { r, t: lt };
  }).sort((a, b) => a.t - b.t);
  const rr = runs.find((r) => r.round != null), res = h.result;
  $("#app").innerHTML = `<h1>${rr ? `${rr.stage ? esc(rr.stage) + " · " : ""}Round ${esc(rr.round)} · Race ${esc(rr.race ?? "–")}` : `Heat ${n}`}</h1>
    <p><span class="btn" onclick="heatPlay(true)">▶ Play all</span> <span class="btn" onclick="heatPlay(false)">❚❚ Pause</span></p>
    ${hint("▶ Play all starts the pilots' videos together, as they flew. Tap a name to open that pilot's run.")}
    <div class="quad">${runs.map((r) => `<div><div class="meta"><span style="color:${colorFor(r.pilot)}">●</span> <a href="${B()}run/${encodeURIComponent(r.stem)}">${esc(r.pilot)}</a> · ${esc(r.channel)}</div>
      ${r.crop ? videoTag(r, `class="hv" muted data-off="${r.live_start_s}"`) : `<video class="hv" data-off="${r.live_start_s}" src="${r.video_url}" playsinline muted preload="metadata"></video>`}</div>`).join("")}</div>
    ${res ? `<h2>Result</h2>${hint(`The finishing order${res.type === "TimeTrial" ? " (a time trial: by the best laps in a row)" : ""}, with every lap. Green = the pilot's best lap. Tap a name to open that pilot's run.`)}<div class="panel scroll"><table class="race"><thead><tr><th>Pos</th><th>Pilot</th><th>Laps</th><th>Total</th><th>Lap times</th><th>Best lap</th>${rowN(h.pb_laps) ? `<th>${rowN(h.pb_laps)}</th>` : ""}<th>Holeshot</th></tr></thead><tbody>
      ${res.pilots.map((p) => `<tr><td><b>${p.position ?? "–"}</b>${p.dnf ? ' <span class="muted">DNF</span>' : ""}</td><td><span style="color:${colorFor(p.pilot)}">●</span> ${runLink(p.stem, esc(p.pilot))}</td><td>${p.laps}</td><td>${sec(p.total)}</td>
        <td>${lapChips(p.lap_times) || '<span class="muted">–</span>'}</td><td><b>${sec(p.best_lap)}</b></td>${rowN(h.pb_laps) ? `<td>${sec(p.best_row)}</td>` : ""}<td>${sec(p.holeshot)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    ${!res && isFinite(L) && order.length ? `<h2>Order over the first ${L} lap${L > 1 ? "s" : ""}</h2>${hint("Total time for the laps everyone completed. Gap +1.20 = 1.2 s behind the leader.")}<div class="panel scroll"><table><thead><tr><th>Pilot</th><th>Time</th><th>Gap</th><th>Best lap</th></tr></thead><tbody>
      ${order.map((o, i) => `<tr><td><b>${i + 1}.</b> ${esc(o.r.pilot)}</td><td>${sec(o.t)}</td><td>${i ? "+" + sec(o.t - order[0].t) : "–"}</td><td>${sec(o.r.stats.best_lap)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    <h2>Gap to the leader at every gate</h2>${hint("One line per pilot. The leader runs along the top. A line dropping lower = falling further behind; a line climbing = catching up. L1 L2 … mark the laps.")}<div class="panel">${gapChart(runs, h.runs[0]?.stats?.n_gates)}</div>`;
}

function heatPlay(on) {
  const vs = [...document.querySelectorAll(".hv")];
  if (!on) return vs.forEach((v) => v.pause());
  const t = Math.max(0, (vs[0]?.currentTime || 0) - (+vs[0]?.dataset.off || 0));
  vs.forEach((v) => { v.currentTime = t + (+v.dataset.off || 0); v.play().catch(() => {}); });
}

function gapChart(runs, n) {
  if (!n) return '<p class="muted">Results appear when the runs are analysed.</p>';
  // k = gates passed since the first G1: lap*n + gate
  const seqs = runs.map((r) => {
    let lap = -1, out = {};
    for (const p of r.passes) { if (p.gate === 1) lap++; if (lap >= 0) out[lap * n + p.gate - 1] = p.t; }
    return { r, out };
  });
  const ks = [...new Set(seqs.flatMap((s) => Object.keys(s.out).map(Number)))].sort((a, b) => a - b);
  const lead = {}; for (const k of ks) lead[k] = Math.min(...seqs.map((s) => s.out[k]).filter((x) => x != null));
  const maxGap = Math.max(1, ...seqs.flatMap((s) => ks.filter((k) => s.out[k] != null).map((k) => s.out[k] - lead[k])));
  const W = 700, H = 200, pad = 28, X = (k) => pad + ((k - ks[0]) * (W - 2 * pad)) / Math.max(1, ks[ks.length - 1] - ks[0]), Y = (g) => pad + (g / maxGap) * (H - 2 * pad);
  const lines = seqs.map((s) => {
    const pts = ks.filter((k) => s.out[k] != null).map((k) => `${X(k)},${Y(s.out[k] - lead[k])}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${colorFor(s.r.pilot)}" stroke-width="2.5"/>`;
  }).join("");
  const lapTicks = ks.filter((k) => k % n === 0).map((k) => `<line x1="${X(k)}" x2="${X(k)}" y1="${pad}" y2="${H - pad}" stroke="#2a3347"/><text x="${X(k)}" y="${H - 8}" fill="#8a96ad" font-size="11" text-anchor="middle">L${k / n + 1}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}">${lapTicks}<text x="${pad}" y="16" fill="#8a96ad" font-size="11">seconds behind the leader (down = further behind)</text>${lines}</svg>
    <div>${seqs.map((s) => `<span class="pill"><span style="color:${colorFor(s.r.pilot)}">●</span> ${esc(s.r.pilot)}</span>`).join("")}</div>`;
}

// ── pilot page ─────────────────────────────────────────────────
async function pilotPage(name, first) {
  const [p, sc, res] = await Promise.all([api("/api/pilot/" + encodeURIComponent(name)).catch(() => null),
                                          api("/api/schedule").catch(() => ({ has: false })), api("/api/results").catch(() => null)]);
  SCHED = sc;
  // the schedule part changes during the day; the rest only with new runs
  const sig = JSON.stringify([p && p.runs.map((r) => [r.stem, r.laps, r.best_lap]), sc.has && sc.next[name], sc.has && sc.upcoming.length, sc.current && sc.current.id]);
  if (!first && sig === current.sig) return;
  current.sig = sig;
  current.draw = () => drawPilot(name, p, sc, res);
  drawPilot(name, p, sc, res);
}

// Two pilots side by side: every number of the leaderboard
function headToHead(me, him, pb) {
  const rows = [["Best lap", "best_lap"], ...(rowN(pb) ? [[rowN(pb), "best_row"]] : []), ["Typical lap", "median_lap"],
                ["Theoretical best", "theoretical_best"], ["Best holeshot", "holeshot"]];
  const line = (k, a, b, fmt, d) => `<tr><td>${k}</td><td><b>${fmt(a)}</b></td><td>${fmt(b)}</td><td class="${cls(d)}">${d == null ? "–" : delta(d)}</td></tr>`;
  return `<div class="panel scroll"><table><thead><tr><th></th><th>${esc(me.pilot)}</th><th>${esc(him.pilot)}</th><th>Difference</th></tr></thead><tbody>
    ${rows.map(([k, f]) => line(k, me[f], him[f], sec, me[f] != null && him[f] != null ? me[f] - him[f] : null)).join("")}
    ${line("Consistency", me.lap_cv, him.lap_cv, pct, null)}
    ${line("Laps flown", me.laps, him.laps, (x) => x ?? "–", null)}</tbody></table></div>`;
}

function drawPilot(name, p, sc, res) {
  const runs = p ? p.runs : [];
  const klass = p && res ? res.classes.find((c) => c.layout === p.layout) : null;
  const me = klass ? klass.pilots.find((x) => x.pilot === name) : null;
  const ref = klass ? refOf(klass, name) : { field: true, who: "the field", label: "field median", lap: p && p.field.median_lap, sections: (p && p.field.section_median) || [], named: [] };
  const pb = res ? res.pb_laps : 3;
  // the run that set a number: its card opens that run
  const bestOf = (key) => runs.filter((r) => r[key] != null && (key !== "holeshot" || r[key] > 0)).reduce((a, r) => (a && a[key] <= r[key] ? a : r), null);
  const card = (k, r, key, h) => fact(k, r ? `<a class="v good" href="${B()}run/${encodeURIComponent(r.stem)}">${sec(r[key])}</a>` : '<div class="v">–</div>',
    h + (r ? ` Heat ${r.heat} ↗` : ""));
  const tile = (k, v, run, h) => fact(k, v != null ? `<div class="v">${runLink(run, sec(v))}</div>` : '<div class="v">–</div>', h);
  const mine = sc.has ? [...(sc.current ? [sc.current] : []), ...sc.upcoming].filter((r) => r.pilots.some((x) => x.name === name)) : [];
  const best = p ? p.best_sections : [];
  const diffs = best.map((v, i) => (v != null && ref.sections[i] != null ? v - ref.sections[i] : null));
  const named = me && klass ? klass.named.map((x, i) => (me.named_best[i] != null && ref.named[i] != null ? me.named_best[i] - ref.named[i] : null)) : [];
  const against = ref.field ? "the typical pilot" : `${esc(ref.who)}'s best`;
  $("#app").innerHTML = `<h1><span style="color:${colorFor(name)}">●</span> ${esc(name)}${klass && klass.name ? ` <span class="muted" style="font-size:15px">· ${esc(klass.name)}</span>` : ""}</h1>
    <div class="facts">
      ${card("Best lap", bestOf("best_lap"), "best_lap", "Fastest lap over all your runs.")}
      ${me && rowN(pb) ? tile(rowN(pb), me.best_row, me.best_row_run, `Fastest ${pb} laps in a row, in one heat.`) : card("Best 3 laps", bestOf("best3"), "best3", "Fastest 3 laps in a row.")}
      ${me ? tile("Typical lap", me.median_lap, "", "The median of all your laps.") : ""}
      ${me ? tile("Theoretical best", me.theoretical_best, "", "Your best time in every section of one heat, added up.") : ""}
      ${card("Best holeshot", bestOf("holeshot"), "holeshot", "Fastest from take-off to the first gate.")}
      ${fact("Laps", `<div class="v">${runs.reduce((a, r) => a + r.laps, 0)}</div>`, `Full laps flown, ${runs.length} run${runs.length === 1 ? "" : "s"} together.`)}
      ${me && me.improved ? fact("Improved", `<div class="v good">−${sec(me.improved)}</div>`, "Your best lap against your first heat's best lap.") : ""}
      ${me && me.craft && me.craft.races ? fact("Races won", `<div class="v">${me.craft.wins} <span class="muted" style="font-size:13px">of ${me.craft.races}</span></div>`, `Average place ${sec(me.craft.avg_place, 1)}.`)
        + (me.craft.gained != null ? fact("Places gained", `<div class="v ${me.craft.gained > 0.05 ? "good" : me.craft.gained < -0.05 ? "bad" : ""}">${me.craft.gained > 0 ? "+" : ""}${sec(me.craft.gained, 1)}</div>`, "From the holeshot to the finish, a race on average.") : "") : ""}
      ${sc.has && sc.next[name] != null ? fact("Next race", `<div class="v">${nextText(sc.next[name])}</div>`, "From FPVTrackside's running order.") : ""}
    </div>
    ${mine.length ? `<h2>Upcoming races</h2>${hint("Your races to come, with who you fly against.")}
      ${mine.map((r) => `<div class="panel">${r === sc.current ? `<div class="up-head"><b>${esc(r.label)}</b> <span class="pill live">racing now</span></div>
        <div>${r.pilots.map((x) => `<span class="upchip">${chan(x)} ${pilotLink(x.name)}</span>`).join("")}</div>` : upRow(r, sc.upcoming.indexOf(r))}</div>`).join("")}` : ""}
    ${runs.length ? `${cmpSelect(klass, name)}${ghostLink(name, ref)}
    ${!ref.field && me ? `<h2>${esc(name)} and ${esc(ref.who)}</h2>${hint("Difference: green = you are faster (−0.30 = 0.3 s faster), red = slower.")}${headToHead(me, ref.p, pb)}` : ""}
    <h2>Where you gain and lose time</h2>${hint(`Your best time in each section, from all your runs, against ${against}. Green bar to the left = a strength (you're faster there), red bar to the right = where you lose time.`)}
      ${named.some((d) => d != null) ? `<div class="panel"><div class="k">Named sections</div>${sectionBars(klass.named.map((x) => `${x.name} (G${x.from}→G${x.to})`), named, 170)}</div>` : ""}
      <div class="panel">${sectionBars(p.sections || [], diffs)}</div>
    ${lostLaps(runs, me, p.sections || [])}
    <h2>Every lap</h2>${hint("All your laps of the event in the order you flew them, one bar per lap (shorter bar = faster lap). Green = your best. Tap a bar to open that run.")}
      <div class="panel scroll">${lapsChart(runs)}</div>
    <h2>Runs</h2>${hint(`Each of your runs — tap one to open it. The chart shows your best lap in each run through the event (R1, R2 …): going down = you got faster. Dashed line = ${ref.field ? "the typical lap on this track" : esc(ref.who) + "'s best lap"}.`)}<div class="panel scroll"><table><thead><tr><th>Run</th><th>Laps</th><th>Best</th><th>vs ${esc(ref.label)}</th><th>Average</th><th>Consistency</th></tr></thead><tbody>
      ${runs.map((r) => `<tr class="click" onclick="location.hash='${B()}run/${encodeURIComponent(r.stem)}'"><td>Heat ${r.heat} · ${esc(r.channel)}</td><td>${r.laps}</td><td>${sec(r.best_lap)}</td>
        <td class="${cls(r.best_lap != null && ref.lap ? r.best_lap - ref.lap : null)}">${r.best_lap != null && ref.lap ? delta(r.best_lap - ref.lap) : "–"}</td><td>${sec(r.mean_lap)}</td><td>${pct(r.lap_cv)} ${steadiness(r.lap_cv)}</td></tr>`).join("")}</tbody></table>
      ${lapChart(runs.map((r) => r.best_lap).filter((x) => x), ref.lap, ref.label).replaceAll(">L", ">R")}</div>`
    : '<p class="muted">No analysed runs for this pilot yet.</p>'}`;
}

// Where a pilot's laps are lost: their slowest laps, the gates they miss most, the sections they fly least evenly
function lostLaps(runs, me, sections) {
  const at = (r, t) => `${B()}run/${encodeURIComponent(r.stem)}/${Math.max(0, t - 1).toFixed(1)}`;
  const med = me ? me.median_lap : null;
  const slow = med ? runs.flatMap((r) => (r.lap_times || []).map((t, i) => ({ r, i, t, m: (r.lap_marks || [])[i] }))).filter((x) => x.m && x.t > med * 1.15)
    .sort((a, b) => b.t - a.t).slice(0, 5) : [];
  const miss = {};
  for (const r of runs) for (const m of r.missed_gates || []) (miss[m.gate] ??= []).push({ r, t: m.t });
  const missed = Object.entries(miss).sort((a, b) => b[1].length - a[1].length).slice(0, 5);
  const uneven = me ? sections.map((s, i) => ({ s, b: me.section_best[i], m: me.section_median[i] })).filter((x) => x.b != null && x.m != null)
    .map((x) => ({ ...x, d: x.m - x.b })).sort((a, b) => b.d - a.d).slice(0, 4).filter((x) => x.d > 0.02) : [];
  if (!slow.length && !missed.length && !uneven.length) return "";
  return `<h2>Where laps are lost</h2>${hint("Your slowest laps, the gates you miss most often and the sections you fly least evenly. Tap one to watch the moment. A “missed” gate is one the video didn't see you pass: a real miss, a crash, or a gate the video lost — watch the moment to tell which.")}
    <div class="panel">
      ${slow.length ? `<div class="k">Slowest laps (your typical lap is ${sec(med)})</div><div style="margin-bottom:10px">${slow.map((x) => `<a class="pill sec" href="${at(x.r, x.m.t0)}">Heat ${x.r.heat} · lap ${x.i + 1} · <b>${sec(x.t)}</b> <span class="bad">${delta(x.t - med)}</span> ▶</a>`).join(" ")}</div>` : ""}
      ${missed.length ? `<div class="k">Gates missed most</div><div style="margin-bottom:10px">${missed.map(([g, v]) => `<span class="pill sec"><b>G${g}</b> ×${v.length} ${v.slice(0, 6).map((x) => `<a href="${at(x.r, x.t)}">H${x.r.heat}</a>`).join(" ")}</span>`).join(" ")}</div>` : ""}
      ${uneven.length ? `<div class="k">Least even sections (typical against your own best)</div><div>${uneven.map((x) => `<span class="pill sec">${esc(x.s)} · typical <b>${sec(x.m)}</b> · best ${sec(x.b)} <span class="bad">${delta(x.d)}</span></span>`).join(" ")}</div>` : ""}
    </div>`;
}

// ── ghost lap: two pilots' best laps side by side, started together ──
const ghostLink = (me, ref) => (ref.field ? "" : `<p><a class="btn" href="${B()}ghost/${encodeURIComponent(me)}/${encodeURIComponent(ref.who)}">▶ Watch ${esc(me)}'s best lap next to ${esc(ref.who)}'s</a></p>`);
async function ghostPage(a, first, b) {
  if (!first) return;
  const g = await api(`/api/ghost?p=${encodeURIComponent(a)}&p=${encodeURIComponent(b)}`);
  const [A, Bb] = g.laps;
  // the time at every gate both were seen at: who is ahead there, and by how much
  const tB = Object.fromEntries(Bb.gates.map((x) => [x.gate, x.t]));
  const rows = A.gates.filter((x) => tB[x.gate] != null).map((x) => ({ gate: x.gate, d: x.t - tB[x.gate] }));
  rows.push({ gate: "Lap", d: A.time - Bb.time });
  $("#app").innerHTML = `<h1>Best laps: <span style="color:${colorFor(A.pilot)}">●</span> ${esc(A.pilot)} and <span style="color:${colorFor(Bb.pilot)}">●</span> ${esc(Bb.pilot)}</h1>
    <p><span class="btn" onclick="ghostPlay(true)">▶ Play from the lap's start</span> <span class="btn" onclick="ghostPlay(false)">❚❚ Pause</span>
      <label class="muted" style="margin-left:8px">Speed <select id="grate" onchange="ghostRate()"><option value="1">1×</option><option value="0.5">½×</option><option value="0.25">¼×</option></select></label>
      <span class="clock big" id="gclock">0.00</span></p>
    ${hint("Both laps start at the start/finish at the same moment, so what you see is the gap growing or shrinking round the lap. Slow it down to compare lines through a gate.")}
    <div class="quad">${g.laps.map((l) => `<div><div class="meta"><span style="color:${colorFor(l.pilot)}">●</span> <a href="${B()}run/${encodeURIComponent(l.stem)}">${esc(l.pilot)}</a> · heat ${l.heat} · lap ${l.lap} · <b>${sec(l.time)}</b></div>
      ${l.crop ? videoTag(l, `class="gv" muted data-t0="${l.t0}" data-t1="${l.t1}"`) : `<video class="gv" muted data-t0="${l.t0}" data-t1="${l.t1}" src="${l.video_url}" playsinline preload="metadata"></video>`}</div>`).join("")}</div>
    <h2>The gap at every gate</h2>${hint(`Time since the lap's start when each pilot passed the gate. Green bar to the left = ${esc(A.pilot)} is ahead there (−0.20 = 0.2 s ahead), red to the right = behind. Watch where the bar grows: that is where the time goes.`)}
    <div class="panel">${rows.length > 3 ? sectionBars(rows.map((x) => (x.gate === "Lap" ? "Lap" : "G" + x.gate)), rows.map((x) => x.d)) : '<p class="muted">The gates of these laps weren\'t seen well enough to compare.</p>'}</div>`;
  for (const v of document.querySelectorAll(".gv")) {
    const start = () => { v.currentTime = +v.dataset.t0; };
    if (v.readyState >= 1) start(); else v.addEventListener("loadedmetadata", start, { once: true });
    v.addEventListener("timeupdate", () => {
      if (v.currentTime >= +v.dataset.t1 + 0.4) v.pause();
      const first = document.querySelector(".gv"), c = $("#gclock");
      if (c && v === first) c.textContent = sec(Math.max(0, v.currentTime - +v.dataset.t0));
    });
  }
}
function ghostRate() { const r = +$("#grate").value; document.querySelectorAll(".gv").forEach((v) => { v.playbackRate = r; }); }
function ghostPlay(on) {
  const vs = [...document.querySelectorAll(".gv")];
  if (!on) return vs.forEach((v) => v.pause());
  vs.forEach((v) => { v.currentTime = +v.dataset.t0; v.playbackRate = +$("#grate").value; v.play().catch(() => {}); });
}

// every lap of a pilot's day as bars, heat by heat
function lapsChart(runs) {
  const order = [...runs].sort((a, b) => a.heat - b.heat);
  const all = order.flatMap((r) => (r.lap_times || []).map((t, i) => ({ t, r, i })));
  if (!all.length) return '<p class="muted">No laps yet.</p>';
  const best = Math.min(...all.map((x) => x.t)), hi = Math.max(...all.map((x) => x.t)) * 1.05;
  const bw = 16, gap = 4, sepw = 12, H = 150, top = 14, base = H - 30;
  let x = 30, out = "", prev = null;
  for (const a of all) {
    if (prev && a.r !== prev) { x += sepw; out += `<line x1="${x - sepw / 2}" x2="${x - sepw / 2}" y1="${top}" y2="${base}" stroke="#2a3347"/>`; }
    if (a.r !== prev) out += `<text x="${x}" y="${H - 8}" fill="#8a96ad" font-size="10">H${a.r.heat}</text>`;
    const h = ((base - top) * a.t) / hi;
    out += `<a href="${B()}run/${encodeURIComponent(a.r.stem)}"><rect x="${x}" y="${base - h}" width="${bw}" height="${h}" rx="2" fill="${a.t === best ? "var(--good)" : "#4da3ff"}"><title>Heat ${a.r.heat} lap ${a.i + 1}: ${sec(a.t)} s</title></rect></a>`;
    x += bw + gap; prev = a.r;
  }
  return `<svg width="${x + 10}" height="${H}" style="width:${x + 10}px;max-width:none">
    <text x="0" y="${top - 2}" fill="#8a96ad" font-size="10">s</text><text x="0" y="${base - ((base - top) * best) / hi + 4}" fill="var(--good)" font-size="10">${sec(best, 1)}</text>${out}</svg>`;
}

// ── race day (FPVTrackside events): the race on now, the races to come ──
let SCHED = null;
const pilotLink = (n) => `<a href="${B()}pilot/${encodeURIComponent(n)}">${esc(n)}</a>`;
const chan = (p) => `<span class="dot" style="background:${esc(p.color || "#8a96ad")}"></span>${esc(p.channel)}`;
// a pilot's next race: in the running race, in the next one, or how many races fly before theirs
const nextText = (s) => s == null ? '<span class="muted">–</span>' : s === "Racing" ? '<span class="pill live">racing</span>'
  : s === "Staging" ? '<span class="pill ok">next race</span>' : `after ${s} race${s === "1" ? "" : "s"}`;
const typePill = (r) => r.practice ? '<span class="pill">practice</span>' : r.type === "TimeTrial" ? '<span class="pill">time trial</span>' : "";
const whenText = (i) => i === 0 ? '<span class="pill ok">next</span>' : `<span class="muted">after ${i} race${i > 1 ? "s" : ""}</span>`;

function lapChips(laps) {
  const best = Math.min(...laps);
  return laps.map((l) => `<span class="lap${l === best && laps.length > 1 ? " best" : ""}">${sec(l)}</span>`).join("");
}

function raceTable(r) {
  return `<div class="scroll"><table class="race"><thead><tr><th>Pos</th><th>Pilot</th><th>Chan</th><th>Holeshot</th><th>Laps</th><th>Total</th></tr></thead><tbody>
    ${r.pilots.map((p) => `<tr><td>${p.position ?? "–"}${p.dnf ? ' <span class="muted">DNF</span>' : ""}</td><td>${pilotLink(p.name)}</td><td>${chan(p)}</td>
      <td>${sec(p.holeshot)}</td><td>${lapChips(p.laps) || '<span class="muted">–</span>'}</td><td>${sec(p.total)}</td></tr>`).join("")}</tbody></table></div>`;
}

function upRow(r, i) {
  return `<div class="up"><div class="up-head"><b>${esc(r.label)}</b> ${typePill(r)} ${whenText(i)}</div>
    <div>${r.pilots.map((p) => `<span class="upchip">${chan(p)} ${pilotLink(p.name)}</span>`).join("")}</div></div>`;
}

function nowPanel(sc) {
  const r = sc.current || sc.last;
  if (!r) return "";
  return `<div class="panel"><div class="up-head"><b>${sc.current ? "Racing now" : "Last race"}</b> · ${esc(r.label)} ${typePill(r)}
    ${sc.current ? `<span class="pill live">LIVE</span> <span class="clock" data-start="${r.start}"></span>` : ""}</div>${raceTable(r)}</div>`;
}

// Home's race-day block: the race on now (or the last one) and the next three
function raceDay(sc) {
  if (!sc || !sc.has) return "";
  const up = sc.upcoming;
  return `<h1>Race day</h1>
    ${hint("From FPVTrackside: the race on now (or the last one flown), then the races to come with their pilots and channels.")}
    ${nowPanel(sc)}
    <div class="panel"><div class="up-head"><b>Next races</b> <span class="muted">${up.length} to fly</span></div>
      ${up.slice(0, 3).map(upRow).join("") || '<span class="muted">No races to come in FPVTrackside.</span>'}
      ${up.length > 3 ? `<a href="${B()}upcoming">See all ${up.length} →</a>` : ""}</div>`;
}

async function upcomingPage(_, first) {
  const sc = SCHED = await api("/api/schedule");
  if (!sc.has) { $("#app").innerHTML = '<p class="muted">Upcoming races are shown for FPVTrackside events.</p>'; return; }
  $("#app").innerHTML = `<h1>Upcoming races</h1>
    ${hint("In running order, from FPVTrackside. Find your name to see when you fly: “after 2 races” = two races fly before yours.")}
    ${sc.current ? nowPanel(sc) : ""}
    ${sc.upcoming.map((r, i) => `<div class="panel">${upRow(r, i)}</div>`).join("") || '<p class="muted">No races to come. They show here as soon as they are set up in FPVTrackside.</p>'}`;
}

// TV: the race on now, big, for a screen at the pits
async function tvPage(_, first) {
  const sc = SCHED = await api("/api/schedule");
  const r = sc.has && (sc.current || sc.last);
  if (!r) { $("#app").innerHTML = '<p class="muted">No race yet.</p>'; return; }
  $("#app").innerHTML = `<div class="tv-head"><h1>${esc(r.label)}</h1>${typePill(r)}
      ${sc.current ? `<span class="pill live">LIVE</span><span class="clock big" data-start="${r.start}"></span>` : '<span class="muted">last race</span>'}</div>
    ${sc.current && !sc.web ? hint("Laps show when the race ends. To see them live, switch on FPVTrackside's web server (Menu → Application Settings → Web → Auto start HTTP Server).") : ""}
    <div class="tiles">${r.pilots.map((p) => `<div class="tile" style="border-left-color:${esc(p.color || "#8a96ad")}">
      <div class="tile-head"><span class="who">${pilotLink(p.name)}</span><span class="pos">${p.position ? "P" + p.position : ""}</span></div>
      <div class="kv"><div><span class="k">Chan</span>${esc(p.channel)}</div><div><span class="k">Laps</span>${p.laps.length}</div><div><span class="k">Total</span>${sec(p.total)}</div></div>
      <div class="laps">${lapChips(p.laps)}</div></div>`).join("")}</div>
    ${sc.upcoming.length ? `<h2>Next</h2>${upRow(sc.upcoming[0], 0)}` : ""}`;
}
tvPage.every = 1000;

// race clocks: seconds since the race start, ticking
setInterval(() => document.querySelectorAll(".clock[data-start]").forEach((c) => {
  const t = Date.now() / 1000 - Number(c.dataset.start);
  c.textContent = t >= 0 ? `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}` : "";
}), 100);

// ── track page ─────────────────────────────────────────────────
async function trackPage(_, first) {
  const [t, res] = await Promise.all([api("/api/track"), api("/api/results").catch(() => null)]);
  if (!first) return;
  const named = ((res && res.classes) || []).filter((c) => c.named.length).map((c) => `${c.name ? `<div class="k">${esc(c.name)}</div>` : ""}<div class="scroll"><table><thead><tr><th>Section</th><th>Gates</th><th>Typical</th><th>Fastest</th><th>By</th></tr></thead><tbody>
    ${c.named.map((x) => `<tr><td><b>${esc(x.name)}</b></td><td>G${x.from}→G${x.to}</td><td>${sec(x.median)}</td><td><b>${x.best ? sec(x.best.time) : "–"}</b></td><td>${x.best ? runLink(x.best.run, esc(x.best.pilot)) : "–"}</td></tr>`).join("")}</tbody></table></div>`).join("");
  const f = t.field || {};
  const hard = (f.sections || []).map((s, i) => ({ s, sp: f.section_spread?.[i], med: f.section_median?.[i] }))
    .filter((x) => x.sp != null).sort((a, b) => b.sp - a.sp).slice(0, 4);
  $("#app").innerHTML = `<h1>Track — ${t.gates.length} gates</h1>
    ${hint("The track drawn as sections, each block as wide as the time it usually takes.")}
    ${lapBar(f.sections, f.section_median || t.legs_s)}
    ${hard.length ? `<h2>Hardest sections</h2>${hint("Where pilots differ most. Spread 0.90 s = the gap between fast and slow pilots in that section — where races are won or lost.")}<div class="panel">${hard.map((h) => `<span class="pill">${esc(h.s)}: median ${sec(h.med)}s, spread ${sec(h.sp)}s</span>`).join(" ")}</div>` : ""}
    ${named ? `<h2>Named sections</h2>${hint("The parts of the track named in Event setup → Sections: the typical time through each, and who flew it fastest.")}<div class="panel">${named}</div>` : ""}
    <h2>Gates</h2>${hint("Pictures the system learned for each gate, in the order they're flown, with the typical time to the next gate.")}<div class="gates">${t.gates.map((g) => `<div class="gate panel"><b>G${g.gate}</b> <span class="muted">${esc(g.type)}</span>
      ${g.images.slice(0, 2).map((src) => `<img loading="lazy" src="${src}">`).join("")}
      <div class="muted" style="font-size:12px">G${g.gate}→G${g.gate % t.gates.length + 1}: ${sec(f.section_median?.[g.gate - 1] ?? t.legs_s?.[g.gate - 1])}s</div></div>`).join("") || '<p class="muted">The track isn\'t learned yet.</p>'}</div>`;
}

refreshStatus(); setInterval(refreshStatus, 3000); route();
