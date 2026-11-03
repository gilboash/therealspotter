# Changelog

The version shows next to the logo in the web app (hover it for the git commit; `+` means uncommitted
changes). It is set in `version.py`.


### 2.11.9 — Live: gate class names on Core ML, the fast centred gate; post-race: a feed gray after a crash (Nov 2026)

By [@HarelZrihen](https://github.com/HarelZrihen), #5.

- Live on Core ML (its default) called every gate "class0": an exported model keeps its class names in `det.names`,
  not `det.model.names`. So flagpoles weren't ignored, as they are after the race. Now they are: on the 3 test
  combos 4 fewer extra passes, and 3 gate passes Live had only got from a flagpole beside the gate (post-race's
  own detector misses them too; its scorer finds them). Laps unchanged.
- Live's pass detector takes a gate flown through fast and centred, whose box vanishes before reaching two sides of
  the picture, with its top up to 20% down (was 10%; `LIVE_PASSDET`, post-race keeps its setting). On the 3 combos:
  ft35 CH4's last lap now closes, 1 more real pass, 3 more extra; the 8 learn videos of both tracks: no change.
- Post-race (v2): a pass's camera-motion "chaos" is measured against the noise of the frames that move, as v3
  already did, not of all frames. A feed gray for half the video (a crash) made every pass look chaotic and lost them
  all (ft35 CH2: 0 passes → 26, its 21.10 s lap). On the 38 stored v2 runs, only feeds gray for long change (2).
- Live from the command line loads CLIP before the feed starts, as the web app does.
- Live, M2, 3 combos at 640 (command line, every frame): 19 → 20 of 23 post-race laps.

### 2.11.8 — Live combo's timeline: a lane per pilot (Nov 2026)

- The Timeline in Live with a combo showed only a hint (one feed had its live timeline). It now has a lane per pilot,
  as a combo's timeline after the race, built as the run goes: its closed laps (🚩 and a red mark at a gate it has no
  pass for; the gate's name when the lane is tall enough), the lap it's on growing, its passes in its colour (blue:
  being named; grey: not a pass), and the time now.
- `requirements.txt`: `coremltools` on macOS (Live's Core ML detector; without it Live falls back to PyTorch).

### 2.11.7 — a pilot's tile goes to its detector process at the detector's size (Nov 2026)

- Each tile was sent to its detector process whole (a 2048×1152 combo: 1024×576, 1.8 MB) and shrunk there: the pipe
  took ~18% of each pilot's time. It's now shrunk first, as the detector would (Ultralytics' letterbox scale,
  INTER_LINEAR), and the boxes scaled back: the same detections (240 tiles; corners within 1 px). M2, browser, 3
  combos at 640: 15 → 18 of 23 post-race laps, Python CPU ~320% → ~250%; each pilot ~35% idle (was ~25%).
- Measured on an M2 (no app code): the Neural Engine does ~245 predictions a second at 640×384 in 2+ processes (4
  pilots at 30 fps: 120), the GPU ~45; the detector as the app calls it ~200 in 4 processes. The limit was the app,
  not the Neural Engine. Input auto stays 416 for a combo on an M1–M3.

### 2.11.6 — a detector process per pilot; input auto (Nov 2026)

- Live combo on Core ML: each pilot's detector in its own process (pilots: a process each, the new auto; by
  [@HarelZrihen](https://github.com/HarelZrihen), #4). In one process the detectors took turns on Python's lock: an
  M4 Pro's Core ML at 640 does ~145 frames a second there however many threads call it, ~290 in two or more; a combo
  at 640 needs 120. M4 Pro, browser, 640: 30 fps and 0–0.1% dropped (before 27.3, up to 8.9%), also at the feed's
  60 fps and with the map and Race 3D open. An M2 at 640: 29 fps and ~1% dropped (before 21, ~30%), 15 of 23
  post-race laps — 416 still gets 17. Each process: ~40% CPU and ~230 MB, kept loaded between runs; another input
  size's are freed, and one that dies stops the run with an error (before: its pilot just stopped). The four start
  together before the feed plays (one after another they took ~18 s of a run's start). An M2 at 416: 29.7–29.9 fps,
  0.1–0.2% dropped, the same 17 of 23 laps as a thread each, ~230% CPU (vs ~165%).
- ⚙ input: auto, the new default (saved settings at 320, the old default, move to it): 640 on Core ML for one feed
  or for a combo on an M4 or later, 416 for a combo on an M1–M3, 320 on PyTorch. The media box keeps its own size.
- Command line: `--procs` with `--combo-all`.

### 2.11.5 — the race map and Race 3D for one feed in Live (Nov 2026)

- Live on a learned track with one feed (not a combo): the race map and Race 3D show the pilot flying the track's
  layout, as a race of one — its lap starts from its laps, its passes as that pilot's. The server sends the layout and
  leg times for one feed too (before: only for a combo). Open them from Windows ▾.

### 2.11.4 — Race 3D without a race (Nov 2026)

- Race 3D showed only a hint when there was no race to fly: opened from the Learn window's map (🧊 Race 3D), in Live
  before a run, or with a video not analysed. It now shows the picked track's learned layout to turn and zoom, and
  the pilots on it once there's a race.

### 2.11.3 — Live combo at 30 fps on an M2, in the browser (Nov 2026)

- *Live (fast)*: Core ML whenever it's there (one feed too, before PyTorch), and a combo's pilots each on their own
  worker (pilots: auto). Saved settings with "one batch" (the old default) move to auto.
- The page's picture is drawn, scaled and encoded in its own thread (newest frame only): in the thread that hands
  frames to the pilots it took ~30% of its time, and the pilots dropped frames.
- PyTorch (the detector's NMS, CLIP on the CPU) and numpy's OpenBLAS on one thread while live: their spinning pools
  took ~1.5 cores. `threadpoolctl` added to requirements.
- Measured in Chrome on an M2, 3 combos (12 pilots) against post-race: the server's CPU ~300% → ~150%; dropped
  8–11% → 0–2%; laps the same as post-race 9 / 23 (one batch) → 17 / 23 at 320 and at 416, as many as with no frame
  dropped (640 with nothing dropped: 19 / 23, but the M2 runs it at ~21 fps). One feed: 30 fps at 640, 11 / 13 laps.

### 2.11.2 — fixes from the M4 Pro test (Nov 2026)

- `online_spotter.py`: a symlinked video inside a track folder finds its track (it followed the link out, and ran
  with no gate names or laps, silently); a video with no learned track now says so.

### 2.11.1 — CLIP downloaded as a setup step (2.10.4 on master) (Nov 2026)

- `setup_models.py` downloads CLIP (open_clip ViT-B-32 `openai`, 605 MB from Hugging Face) once after
  `pip install`, with a progress bar. It was downloaded silently in the first analysis or Live run, which looked
  stuck, and failed with no hint without internet.
- `spotter_web.py` downloads it at start if it isn't there yet, and says so in the terminal. If it can't, it prints why
  and how to fix it (certificates, offline copy, rate limit) and starts anyway: watching analysed races still works.
- The README's setup has the step and the fixes; `requirements.txt` lists `huggingface_hub`.

### 2.11.0 — Live: performance settings, speed per feed (branch live-perf) (Oct 2026)

- **⚙ Performance** in the Live feed window: presets *Live (fast)* (as before) and *Post-race quality* (PyTorch, 640,
  every frame at 30 fps, a worker per pilot), or each setting: detector Core ML / PyTorch (Core ML exported once per
  input size), input 320 / 416 / 512 / 640, frames analysed a second 15 / 20 / 30 / every frame of the feed (the
  tracker and pass detector rescaled to it, also above 30 fps), a combo's pilots in one batch or a detector and
  thread per pilot (a slow pilot drops only its own frames), the picture's rate to the page, CLIP on CPU or GPU.
- Speed per feed: each pilot's tile shows its fps and detector time; Live — laps a table per pilot (fps analysed,
  detector ms, dropped, delay, passes). One feed: the detector's time per frame.
- `online_spotter.py --fps 15|20|30|feed --per-pilot`, and per-pilot speeds in its stats lines.
- Checked on ft35 (M2, every frame at 30 fps, PyTorch 320): one batch and a worker per pilot give the same laps,
  the post-race ones; a worker per pilot on PyTorch is slower on an M2 (one GPU shared, ~8 vs ~14 fps). Real time,
  Core ML 320 with a worker per pilot: 28 of 30 fps on every pilot, 0% dropped per pilot, ~37 ms delay.

### 2.10.3 — v2 works from a fresh clone; a README to set up and use it (Oct 2026)

- v2's pass scorer ships in the repo (`models/pass_scorer_v2.json`); one re-trained on this machine
  (`<data folder>/training/pass_scorer.json`) is still used instead. A clone with an empty data folder analyses with v2
  as before (same passes and times on a test clip).
- README.md replaces README.yaml (GitHub didn't render it): what it does, setup step by step, a first track, the
  modes and their options, launch options, sharing, the data folder, command-line tools; screenshots and badges
  (with ideas from Harel's PR #3). The detail moved to `docs/REFERENCE.md`, the version history here.
- `requirements.txt` installs on Python 3.14 (from PR #3): onnx / onnxruntime stay pinned up to 3.13.

### 2.10.2 — a combo's pilots analysed again (Oct 2026)

- A combo whose pilots are all analysed: ↻ Re-analyse pilots in the top bar (all of them, asked first), and in
  Race — standings each pilot's analysis (passes, laps; few passes in orange) with ↻ to analyse that one again.

### 2.10.1 — the top bar fits a narrow window (Oct 2026)

- Narrower than 1500 px the words before Track / Video / Model and "Add video" go (the hover title stays); narrower
  than 1150 px the name next to the logo too; the track, video and action buttons are cut short (…) before the
  Windows menu is pushed off the bar. The phone layout is as before.

### 2.10.0 — Race 3D; the map as a learn stage; place call-outs (Oct 2026)

- **Race 3D** window (Windows ▾, 🧊 3D on the race map, or the Learn window): the race on the track's layout in 3D
  — each gate modelled by its learned type (square, arch), the path on the ground, each pilot a glowing drone with
  a trail and its place, name and lap; a gate lights up in the pilot's colour on a pass. Camera: an overview (drag
  to turn, wheel to zoom), chase the leader, chase a pilot, or on board a pilot. Live and Race mode (a combo or one
  pilot's video), where the race map has them. three.js is served from `web/static/vendor/` (no internet needed).
- The map is a stage of learning in the Learn window ("Map · made after learning": when, from how many clean laps,
  gates moved by hand). It's made after each 🎓 Learn track (as in 2.9.0) and now also made again after each analysed
  video of the track (more clean laps, a better shape; the gates moved by hand, mirror and rotation stay).
- Call-outs as the places change (Live and Race mode combos): "X takes the lead", "X is now second" (of who moved up,
  held 1.5 s and a gate clear), "X is down" (a combo pilot whose run ended short; in Live, not seen at any gate for
  ¾ of a lap, at least 15 s) / "X is back", "X finishes, second". One place call every 3 s at most (lead changes
  every 5 s). The standings and race map show a pilot down (✕) or finished.

### 2.9.0 — the track's layout: estimated after learning, gates dragged into place (Oct 2026)

- `learn_layout.py`: after learning (auto or manual), where the gates roughly are — from how the camera turned on
  each leg of every clean lap (the roll gives the side: the same on 14–16 of 16 fril-27 laps for most legs; the
  pitch how much — banked, a turn shows as pitch) and the learned leg times (as lengths). Rough: it ends ~45% of a
  lap from its start before it's closed. `<track>/gate_layout[.v3].json`, per model like every learned file.
- **Track layout** window (Learn window: 🗺 Track layout): the gates with their pictures on the path, drag a gate
  where it really is (kept when it's estimated again); ⇋ Mirror, ⟳ Rotate (kept too), ↺ Undo my moves, ↻ Estimate
  again. `GET/POST /api/tracks/<t>/layout`.
- The race map (Live and Race mode) draws the race on the layout — a smooth path through the gates — instead of
  the stadium loop (still used for a track without one).

### 2.8.0 — Race mode: sounds and call-outs; a race map that keeps moving (Oct 2026)

- Race mode plays Live's sounds while the video plays (🔊 sound, 🗣 voice in the video window): a coin per named
  gate, a fanfare on G1, lap times called out with the pilot's name (typed in the video window; guessed from the
  file name, h001_R6_Nivazu → Nivazu); a combo: per pilot (names in Race — standings), "… takes the lead". The last
  call-out shows over the video. Nothing is said on a seek.
- Race map (Live and Race mode; in Race mode also for one pilot's video): the leader's lap big inside the loop,
  each pilot's lap in its dot, a lane and a short trail per pilot, its name beside it. A pilot no longer stops
  short of the next gate: past 80% of the leg it slows to 35% of the pace and keeps going (the gate may be
  missed) until a pass places it again; a pass further on is caught up smoothly; a finished run stops.

### 2.7.0 — Race mode: a combo view's pilots raced against each other (Oct 2026)

- Pick a combo view in Race mode: its pilots' own analyses (`<combo>_CH1` …, analysed in full, offline) race each
  other as the combo plays — tile overlays, **Race — standings (combo)** with a lap table, the race map, a timeline
  lane per pilot. `GET /api/tracks/<t>/combo_race?video=`.
- Top bar on a combo: ▶ Analyse N pilots; ⊞ Split & analyse pilots again for one whose pilot videos are gone.
- The race map's leg times follow the model's learned track (were always v2's).

### 2.6.1 — learned tracks per model, delete per model (Oct 2026)

- Each model learns its own track (v3: `*.v3.json`; until it has, it uses v2's); learning works under v3.
- Learn window: delete the picked model's learned track, optionally with all its analyses (to learn_backups/).
- Top bar: short model names (the long ones pushed ▶ Run analysis off the bar).

### 2.6.0 — a Model setting: v2 (current) or v3 (cozYnut's new detector + two-stage scorer) (Oct 2026)

- Top bar **Model**: v2 / v3 (`model_profiles.py`); each keeps its own analyses (`runs/`, `runs_v3/`); learning is
  v2's; Live follows the model's detector. `GET/POST /api/model`.
- From cozYnut's rest-api branch: detector `cyn_current-20261021_best.pt`, `models/pass_scorer_v3.json`, two-stage
  pass deciding (pass_scorer, gate_decoder — vectorised, 16× faster, same labels — extract_race, relabel_run),
  switched on by the v3 scorer only. v2's results unchanged (checked on 51 videos).

### 2.5.1 — Live: combo recordings timed by their frames (Oct 2026)

- A feed faster than 30 fps (a combo recording: ~60 fps, uneven) is handed on at an even 30 fps by its frames'
  own times, not every 2nd frame: lap starts as the split videos' (within 0.03 s), gate names far closer.
- The combo's picture to the page at most 15 a second (drawing every frame took 8 ms of the 33).
- Race standings follow the decoder's lap starts (a misnamed gate no longer jumps a pilot laps ahead);
  "… takes the lead" only when held 1.5 s, at most every 5 s.

### 2.5.0 — Live: every pilot of a combo at once — standings, race map, lead call-outs (Oct 2026)

- Live **Combo: all pilots — race**: each tile its own pilot (`OnlineSpotter.run_multi`), names per channel,
  standings, **Race map** (automatic loop by the learned leg times), lap call-outs per pilot, "… takes the lead".
- The detector exported to Core ML for it (once, `exports/`): four tiles in 14 ms — all pilots at 30 fps.
- `online_spotter.py --combo-all`; events carry `pilot` / `ch`; `GET /api/live` has `pilots`, `starts`, `legs`.

### 2.4.1 — combo views uploaded for analysis (Oct 2026)

- An uploaded race video that's a combo view asks whether to split it per pilot and analyse each (or analyse it
  as one video); Race mode: ⊞ Split & analyse pilots. The split job can queue its pilots' analyses
  (`analyse: true`; a video that isn't a combo is then analysed as it is); `GET /api/tracks/<t>/combo?video=`.
- A combo in `combo_videos/` can't be analysed as one video.

### 2.4.0 — combo (quad) views: split per pilot, one pilot live (Oct 2026)

- `combo_split.py`: finds a combo view's tiles (where the picture jumps between feeds), splits it into one 30 fps
  video per pilot; Learn / Videos **⊞ Split** (a job; the combo moves to `combo_videos/`).
- Live: **Combo: CH1…CH4** on a file, capture device or stream — preview, spotting and saving of that pilot only;
  60 fps feeds spotted at every 2nd frame. `online_spotter.py --combo N`.
- Fix: stopping a stream preview before the sender connected crashed its thread.

### 2.3.5 — Live: the preview doesn't get stuck opening (Oct 2026)

- A capture device is released before it's opened again, preview requests are taken one at a time (never two
  captures of one device), and a device giving no picture for 6 s is closed and opened again (3 tries, then an
  error). No preview: click the feed area to open it.

### 2.3.4 — Live: preview, the right capture device, 30 fps (Oct 2026)

- Live: a capture device or a stream sent to this Mac **shows as soon as it's picked** (PREVIEW badge: size, fps)
  — nothing spotted or saved; **▶ Start** takes the open feed over (a sender isn't dropped) to spot gates / time
  laps / save, **■ Stop** hands it back to the preview; ■ Stop again closes it. A preview nobody watches for a
  minute is closed (camera light off).
- Capture devices are listed in the order OpenCV opens them (by the cameras' unique IDs, not ffmpeg's order: picking
  "USB3 Video" opened the FaceTime camera) and remembered by unique ID, so a replugged card is still the one.
- A capture device is asked for 30 fps: left alone, AVFoundation opened a USB3 HDMI card at 20 fps (1080p).

### 2.3.3 — Live: no "Stream URL" feed (Oct 2026)

- Live: the *Stream URL* feed (the Mac fetching from a stream server) is gone from the page: the box sends its
  stream, so *Receive a stream* is the one. A saved *Stream URL* choice turns into it. The CLI still takes a URL
  (`online_spotter.py rtsp://…`).

### 2.3.2 — receive a stream sent to this Mac (Oct 2026)

- Live: **Receive a stream (sent to this Mac)**: the Mac listens (RTMP :1935 or RTSP :8554, `/live/feed`) and shows
  the address to set the sender (the media box) to; it waits for the sender and again after it drops off. Works
  with ⏺ save feed and the pilot name. `online_spotter.py <rtmp://0.0.0.0:1935/live/feed> --listen` from the CLI.

### 2.3.1 — save live feeds for a new track (Oct 2026)

- Live: **⏺ save feed** with a new track's name (capture device / stream): the feed is saved, full frames on the
  wall clock, to `live_recordings/<track>/<pilot>_<time>.mp4`, to learn the track from later.

### 2.3.0 — Live on the media box (alpha), live timeline (Oct 2026)

- Live: the Timeline window is built as the run goes — passes and gate names as they're detected, laps as they
  close (gates a lap skipped in red), the running lap, the current feed time.

- Live: **Run on: media box** (alpha) — the Mac's page controls and shows a spotter running on a naco media box (Pi 5):
  deploy (temporary folder), run a track video, passes / gate names / laps / preview / temperature back. Find,
  check and clean the box from the page.
- The spotter at 15 / 10 fps (every 2nd / 3rd frame, tracker rescaled); detector size and ARM runtime benchmarks.
- `track_core.py`: the tracker without heavy imports. Requirements: `onnx` and `onnxruntime` (exports for the box).

### 2.2.0 — Live tab: real-time spotter (Oct 2026)

- **Live** mode: one feed (a video file played in real time, a capture device, an RTSP / MJPEG stream) analysed as
  it plays, the tracked gates drawn on the feed.
- Stage 1: an event per gate pass. Stage 2 (learned track): gate names and laps from `online_decoder.py`, arcade
  sounds per gate, lap times spoken.
- The Mac compute benchmark: where the time goes, clean runs only.

### 2.1.1 — phones (Oct 2026)

- Phone layout (portrait and landscape): the top bar wraps with full-width Track / Video pickers, windows stack in
  one scrolling column with the video first, larger touch targets, no zoom on focus.
- The page, scripts and styles are re-checked on every load (`Cache-Control: no-cache`), so an update reaches every
  browser at once; a stale stylesheet had left "Loading video…" showing under the video.
- The loading spinner hides as soon as a frame is shown, and no longer shows for a paused video.

### 2.1.0 — back to sequence decoding + near-misses, league page (Oct 2026)


**v2 (commit 1353a85)** added the web app and a *lap-first* learner/labeller (`lap_learner.py`: laps from the start
gate, the N most reliable pass points as gates, the rest as tracked extras), plus a lap rescue for skipped gates.
Race results often showed a skipped gate on laps where the pilot clearly flew every gate, so both pipelines were
benchmarked on the same detections.

**Benchmark:** `datastuff/fril-26-race-6` (13 gates, 4 learn + 20 race videos) and `fril-27-race-1` (14 gates,
3 learn + 25 race videos). Every race video was analysed once (YOLO + pass scorer) and each version labelled
the same passes. There is no reviewed ground truth for these tracks, so the measure is *laps with every gate
present* (pilots almost always fly them all) and *complete laps* (every gate once, in order):

| Version | fril-26 laps | every gate | complete | missing gates | fril-27 laps | every gate | complete | missing gates |
|---|---|---|---|---|---|---|---|---|
| v2 lap-first | 26 | 8% | — | 76 | 63 | 0% | — | 269 |
| v2 + fixes + lap rescue | 52 | 62% | 62% | 29 | 65 | 40% | 40% | 78 |
| original (template + sequence decoder) | 62 | 66% | 48% | 22 | 67 | 27% | 16% | 73 |
| **original + near-misses (current)** | **62** | **77%** | **56%** | **16** | **69** | **62%** | **35%** | **30** |

Findings:
- The lap-first learner learned both tracks wrong without help (fril-26 as double laps, fril-27 at 33.7 s instead
  of ~26 s) and found no laps at all on 7 slow or messy race videos; the template learner gets both right.
- With the original labeller, ~3 in 4 missing gates were *near-misses*: the tracker saw the crossing, but the pass
  scorer gave it p < 0.35 (typically a gate taken in a hard bank that leaves the frame while still small).
  Adding those crossings and letting the decoder place them by timing fills most gaps; 11 of 12 randomly checked
  fills were real passes in the video.
- About 10 missing gates per track were never detected at all; no labelling change can recover those.
- Remaining duplicates (25–38 per track) come mostly from the decoder giving two real passes the same gate.

**Changes:**
- `learn_track.py`: the template method is the only automatic method again (`--method laps` removed); start-gate
  marks from the web app are used as G1 times.
- `gate_decoder.label_race`: always the sequence decoder (the lap-model labeller is removed).
- `near_miss.py` (new) replaces the lap rescue in `extract_race.py` and `relabel_run.py`.
- Web app: Breaking Carbon branding, reset learning, remove video, missed (red) / recovered (amber) spotter flags.
- Tracks learned with v2 should be learned again (**🎓 Learn track**); older results are re-labelled automatically.
- League page (`--league`, `--tracks`) for sharing with pilots, with upload limits; chunked uploads that fit
  through a Cloudflare tunnel; a loading spinner over the video; the version number in the header.

### 2.0.0 — lazyspotter v2 (commit 1353a85)

Web app (Race, Learn · auto, Learn · manual), lap-first learning, scene-checked start gate.
