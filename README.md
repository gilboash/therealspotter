# therealspotter — Lazy Spotter v2 🏁

**The first version of the realspotter app.** One desktop app, `lazy_spotter_v2.py`,
for FPV drone races: it watches the pilots' video feeds, learns the track's gates,
finds every gate pass and lap, checks FPVTrackside's lap times against the video,
and shows each run's results in a browser that anyone on the Wi-Fi can open.

Everything runs locally on a laptop (no cloud), in Python. Demo of an earlier
version: https://www.youtube.com/watch?v=6A2gmpzc6iU

---

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision          # first: the right build for your machine (Apple MPS is included on macOS)
pip install -r requirements.txt
pip install ffmpeg                    # recommended: makes recordings stream in phone browsers
```

The first run downloads the CLIP model (once, ~350 MB) into the Hugging Face
cache; after that the app works without internet.

## Run

```bash
source venv/bin/activate
python lazy_spotter_v2.py
```

Or in one line, from the project folder (no need to activate first):

```bash
venv/bin/python lazy_spotter_v2.py
```

**The first time**, the app asks for the **data folder**: the folder that holds
your tracks and FPVTrackside events (for example `FPVdatasets`). It's remembered
(`~/.lazy_spotter_v2.json`) and can be changed on the start screen.

---

## The start screen

- **Data folder: …  [Change…]** — the folder every event is picked from.
- **Browser address** — the dashboard (see below), already running.
- Three tabs, one per way of working:

| Tab | For | You pick |
|---|---|---|
| ⏱ **FPVTrackside event** | checking a race that FPVTrackside timed | one of the FPVTrackside events in the data folder |
| ● **Live capture** | recording heats from the capture card yourself | a track of the data folder, or a new one |
| ▶ **Replays** | looking at races afterwards | any track or FPVTrackside event of the data folder |

Events are always chosen from a list of what's in the data folder (newest first,
with a filter), never browsed for.

### ⏱ FPVTrackside event

For races timed by FPVTrackside (LapRF). The app reads the event folder in place
— it never copies or moves the videos — and checks every pilot's laps against
the video.

- **Heat list (left):** every finished heat, newest first, with its pilots and
  their LapRF lap counts, and a status: `•` new · `⏳` analysing · `◦` analysed ·
  `✓` checked · `↻` laps updated · `⚠ n` lap findings to look at.
  Tick ☑ a heat (or single pilots in the heat view) to learn the track from it.
- **Learn track / Relearn track (top):** learns the gates from the ticked heats
  (asks the number of gates). LapRF's start/finish times help, so one full lap
  per run is enough. Gate 1 is the gate each pilot passes first after the start
  (the holeshot) — the pilots line up behind the start/finish. If later heats show
  the first gate after the start labelled as another gate, a red bar says the
  numbering is shifted: relearn the track.
- **Auto (top):** new heats are analysed and checked by themselves, and confident
  lap corrections are written to FPVTrackside; unsure ones wait for you.
  With the track not learned yet, Auto learns it by itself once a layout's first 4
  heats are analysed: from their pilots' runs, with LapRF's crossings as the
  start/finish (G1) moments — as if you had ticked those heats and pressed Learn. Each
  layout (Event setup) on its own; it needs the layout's number of gates set in Event
  setup (a red bar says so if it isn't). Those heats and every later one are then
  checked as usual.
  LapRF is the reference: a lap the video doesn't confirm, or a crossing the timing
  doesn't have, is only shown when the lap time itself is odd (under ¾ or over 1.6 ×
  the usual lap).
  A lap is added only where nothing in the timing can be that crossing (LapRF
  may fire anywhere from the start/finish gate to the gate after it, up to
  2.5 s later), and never where FPVTrackside wouldn't count one: shorter than
  the event's minimum lap time, on a crossing FPVTrackside discounted, after
  the pilot's target laps in a race, or in a time trial after the race time
  ran out. Those are shown for you to check instead.
- **The heat (right):** the 2×2 video with each pilot's channel, name and lap;
  a timeline per pilot (white = LapRF laps, green = laps seen in the video,
  orange = things to check, blue = your marks); and per pilot a table of laps
  with what the video says about each.
- **Buttons:** **Analyse** · **Review** (see below) · **✓ Accept** / **✗ Reject** a
  finding · **+ Lap / HS here** (add a lap or the holeshot at the video's time) ·
  **− Remove lap** · **Update laps in FPVTrackside** (writes the changes like a
  marshal's correction) · **Restore original laps**.
- **Racing now · Next** (above the heat list): the race FPVTrackside is on and the
  next one, with their pilots.

#### With FPVTrackside's REST API (FPVTrackside 2.79 and the alpha)

**FPVTrackside…** (top bar) sets where FPVTrackside's web server is (default
`http://localhost:8080`; switch it on in FPVTrackside: *Menu → Application Settings → Web →
Auto start HTTP Server*) and, only when FPVTrackside runs on another computer, its API
token. The top bar shows what the app found:

- **REST API ✓** — the alpha, with this event open: **Update laps in FPVTrackside** makes
  the changes in FPVTrackside itself, through its API, one lap correction per pilot
  (an added lap is placed after the pilot's previous crossing, so the two computers'
  clocks don't matter). They show there at once; nothing has to be reopened. **Restore
  original laps** undoes exactly those changes. Auto mode works the same way. A race
  running is seen from FPVTrackside's race control. A heat that's already flown can be
  corrected through the API while the next race runs (when *Pause during races* is off).
- **fril live…:** after the app corrects a heat, and whenever FPVTrackside saves a heat
  again (a marshal's edit), the heat's laps as they are now go to fril.co.il/live
  (FPVTrackside never resends a past heat's laps). So does the event's consecutive-laps
  setting, which the qualifying board ranks by.
- **web server only** / **not reachable** — stock FPVTrackside: the changes go to its
  files, as below.

#### Making the changes in FPVTrackside (stock FPVTrackside)

FPVTrackside (without its REST API, which only the alpha has) reads an event's files
when the event opens and keeps them in memory: laps written by the app show in
FPVTrackside **only after the event is reopened**, and if FPVTrackside saves that heat
first (any edit there), the written laps are lost. So the app also lists every
accepted change as a step to make in FPVTrackside itself — it shows there at once:

- **To do in FPVTrackside** (under the table): one step per change, by **race time**
  (the time from the race start, as FPVTrackside's replay shows it) — e.g.
  *R1 cozYnut — add lap 3 at race time 1:29.41 (21.86 s lap): Replay → seek to 1:29.41 →
  right-click the pilot's laps → Add Lap Now*, or *disqualify lap 2 at race time 0:41.90:
  Lap Editor → right-click that lap → Disqualify Lap*. Click a step to see it in the video;
  **Copy race time** copies its time. The table has a **Race time** column too.
- When FPVTrackside saves the heat, the app checks it again: steps FPVTrackside now has
  go away (the heat shows *n done in FPVTrackside*).
- If FPVTrackside saved a heat over laps the app wrote, the heat says so and those
  changes are back on the list.
- **Update laps in FPVTrackside** still writes them all at once — then reopen the event in
  FPVTrackside, and don't edit those heats there before you do. Auto mode writes the same way.
- While FPVTrackside runs a race, the app pauses completely so FPVTrackside has
  the computer; it continues when the race ends.

#### Event setup: classes, track layouts, qualifying, brackets, points

**Event setup…** (top bar) is done at the start of an event. It is kept with the event's
data (`event_setup.json`) and needs FPVTrackside's REST API, with the event open there, for
the three steps that make things in FPVTrackside.

- **Track layouts.** `main` is the event's own track, the one the app always used.
  **+ Layout** adds one for a class that flies other gates, or the same gates in another
  order, for example `beginners`, 10 gates. Each layout is a track of its own: learned
  from its own runs (**Learn track** learns each layout from the ticked runs on it), and a
  run is analysed, labelled and checked on its pilot's class's layout. The top bar shows
  each layout. Runs analysed before their class got a layout are copied over and labelled
  again, not analysed again.
- **Sections…** names the parts of each layout's lap the way the club talks about them ("the dive", from
  gate 5 to gate 8; a section can be one leg or several), with each one's usual time from the analysed
  runs. The announcer walks the track by them, tells a pilot which section to work on ("if X takes the
  dive like Pennz, that's half a second a lap"), and names them in the live calls ("down at the dive").
  The venue's track file (Announcer window → Track) still names single gates.
- **Classes** (**+ Class**: Open, Beginners…): the class's pilots (☑), its track layout,
  and a pilot's **Main**: *auto* (by qualifying) or a letter, to put a pilot in that main
  whatever their time (their level).
- **Create qualifying in FPVTrackside…** (top of the window) makes the time-trial rounds
  "Qualifying 1"…, **every class in each round**, in the order of the class tabs (Open's
  races, then Beginners'). Each class's pilots fly in races of up to its *Races* pilots (as
  even as possible, the same races every round). Each pilot sits on their own channel
  in the event when it's free in that race (a DJI pilot keeps DR8), else on the next free
  **Seats** channel.
  FPVTrackside first says what it would create (a dry run); nothing changes until you
  agree. Doing it again only adds or changes what's missing; a race already flown is
  never touched.
- **Make the brackets…** (top of the window, once for the whole event, when qualifying is done) takes every
  class's qualifying order from FPVTrackside, one tab per class:
  the best *Seeded by* consecutive laps over the class's qualifying rounds (also any time trial round the
  race director added by hand in FPVTrackside), with pilots
  without a time last. It cuts that order into mains, A being the fastest:
  - mains of *Mains* pilots;
  - a remainder of at least *Own bracket from* pilots gets a main of its own;
  - a smaller remainder follows *Leftover*: a **series main** that races *Series main*
    times for points (3 pilots or more; with fewer, the slowest of the last main make up the 3 when that
    main keeps *Own bracket from* pilots — 17 pilots by 8 make 8, 6 and a series of 3 — else they join it),
    **join the last main**, or **spread** the pilots so the mains are about equal;
  - a whole class smaller than *Own bracket from* (2 pilots, say) races a **series main**, whatever the
    *Leftover* rule: a bracket needs more pilots than that;
  - pinned pilots keep their main, and the rest fill up around them in qualifying order.

  The window shows every main and every seed. *Move to* moves a pilot to another main,
  and that's kept as their level. **Create in FPVTrackside** then makes every class's mains stages at once
  in FPVTrackside's own bracket builder, its **Seeder** (`http://localhost:8080/seeder/`):
  - a bracket main is a double elimination, *Races* a race and *Through* going on;
  - the *Chase the ace in* main ends in a chase the ace to *Chase the ace to* wins;
  - the other mains have the same bracket and end in *Other mains end in*: **one final
    race** (the club's way), a chase the ace too, or the Seeder's classic grand final (a
    2-pilot winners final and more rounds).

  With the announcer on, it then explains the brackets just made, class by class: each main,
  who's in it, and whether it's a double elimination or a points series and how that works.
  Then it says each main and its pilots once more, so every pilot hears where they are twice.

  The Seeder would put each stage's races in a round of its own, so the app places them
  itself (`race_day/finals.py`). Once every class has its brackets, the races whose
  line-up is known go into shared rounds **"Finals 1", "Finals 2"…**, one race per main
  in turn: A, B, C, D, then the next class's mains, then back to A. A race goes in the
  Finals round of its own round in its bracket (a chase the ace's 2nd, 3rd… race in the
  rounds after), and a main with no race left in a round is skipped there.

  The next round appears as soon as one pilot is known for it (only the round after the
  one being flown): all of its races at once, in turn, each with the pilots known so far.
  For example, when W2-1 is flown its 3rd and 4th are put in L2-1 at once, and its other
  two pilots join after L1-1. All at once because FPVTrackside keeps races in the order
  they're made. A race that isn't full yet is named in the top bar ("not full yet, don't
  start"). A chase the ace race the bracket no longer needs is deleted from FPVTrackside,
  unless it has been flown. After each race, by itself, the app gives the race's finishing order
  to its stage, so the Seeder moves the pilots on, and the races that are now known go
  into the next rounds. A marshal's later change in FPVTrackside (a new finishing order) is
  sent again while the Seeder still accepts it; once the bracket's next race is flown it
  can't be, and a red bar names the race with its old and new order, to fix by hand in the
  Seeder. The top bar shows how far the finals are.
- **Places & points…**: where each pilot finished, A's pilots first, then B's.
  - **In a bracket main:** the champion first, then by the last race each pilot flew
    (a later race is better), and within it by place.
  - **In a chase the ace:** by wins, then best place.
  - **FRIL points:** place N of TOTAL pilots in the class earns TOTAL+1−N, plus
    ceil(TOTAL/5), ceil(TOTAL/8) and ceil(TOTAL/13) for 1st, 2nd and 3rd.
  - **Copy** puts the table on the clipboard for a spreadsheet.
- **The announcer** follows the setup:
  - each class's *Qualifying calls* and *Finals calls* are lap times, places or both;
  - in a bracket it says who goes through, who drops to the losers bracket, who is out,
    and a chase the ace win.

### ● Live capture

For recording heats from the race director's HDMI output yourself.

1. **Setup:** **Choose track…** (a track of the data folder, or **New track…**),
   or one of the **Recent** three; **Feed**: 2×2 (four pilots) or single pilot;
   the **channel** of each quadrant; **Number of gates**; the **Capture device**
   (**Refresh**, **Preview device** to check it — the card should give 1920×1080);
   **Longest recording** (default 3:00). **Start**.
   *Test without a card:* choose "Test: play a video file as if live".
2. **The screen:** heat list on the left, the **live feed** top right, the chosen
   heat below it.
3. **● Record** saves the whole feed as `<track>/videos/heat_NNN.mp4` until
   **■ Stop** or the time limit. While it records, analysis and learning wait,
   so the recording is never choppy.
4. Select heats (⌘/Shift-click for several) → **Analyse selected heats**, or
   **Analyse this heat**. Nothing is analysed by itself.
5. Tick ☑ a few heats with full laps → **Learn track**. Every track is learned in
   the app at least once; after that analysed heats get gate numbers and laps.
6. Videos already in `videos/` that the app didn't record are asked about once
   (all 2×2 / all single / one by one, with optional pilot names).

Pilots are known by their channel unless a name was given.

### ▶ Replays

The Live capture screen without the setup, the recording and the live feed:
pick a track, and its videos (`<track>/videos/`) are the heat list. **Analyse**,
**Learn track** and **Review** work the same way. Picking an FPVTrackside event
opens it in the FPVTrackside screen, without Auto.

### Review — checking one pilot's run

In any tab: select a pilot in the heat view and press **Review**. The video shows
only that pilot's quadrant, enlarged; the timeline shows the passes the system
found; a list shows each with its time, lap and gate, next to the system's crop
and the learned picture of that gate. Go through them:

| Key | Action |
|---|---|
| `N` / `Shift+N` | next / previous pass to check |
| `0`–`9` | gate number (`1` then `2` quickly → G12) |
| `Enter` | accept, at the video's time |
| `←` / `→` | one frame (fix the exact pass frame first); `Shift` = one second; `Space` = play |
| `X` | false pass (no real pass here) |
| `A` | add a pass the system missed |
| `M` | a gate the pilot skipped |
| `T` | tag: crash/clipped → bad video → reverse → unsure → none |
| `Backspace` | undo |

Every change is saved (`lazySpotter/gt/<run>.gt.json`); **Done reviewing** goes
back. A reviewed run then shows **the reviewed passes and laps everywhere** — the
heat view (✔ in the heat list), the FPVTrackside lap check (so corrections can be
written back) and the browser (a "reviewed" badge) — updated 2 s after the last
change. Passes not checked yet stay as the system had them; the system's own
result is kept (`system_passes`) and is what `score.py` measures. Reviews also
make gate numbers better on that track and are what the system is retrained on. A clipped gate counts as a pass if the drone went
through it; a crash in the gate only if the drone flies on.

### The browser dashboard

Starts with the app; the address is on the start screen and in every tab's top
bar. Open it on any phone or laptop on the same Wi-Fi. It shows **every event of
the data folder** (the one open in the app live): results and leaderboard, each
run (its video with gate and lap marks, laps, sections against the field), heats
(the pilots' videos side by side, gap chart), pilots, and the track's gates.
Read-only: nothing can be changed from the browser.

For an **FPVTrackside event** it also follows the race day, from FPVTrackside's files
(running order: round order, round, race):

- **Race day** (top of Results): the race on now — LIVE, its clock, each pilot's
  position, holeshot, laps and total — or the last one flown; then the next races with
  their pilots and channels. The leaderboard gets a **Next race** column: *racing*,
  *next race*, or *after 2 races* (two races fly before yours).
- **Upcoming:** every race to come in running order — pilots find their name to see
  when they fly.
- **TV:** the race on now, big, for a screen at the pits.
- **Pilot page:** best lap, best 3 laps and best holeshot (each opens the run that set
  it), the next race and the races to come with who they fly against, and every lap
  of the day as a chart.

**On fril.co.il/live too:** **fril live…** (top bar of the FPVTrackside screen) takes fril's
live ingest address with its token — the same one FPVTrackside's notifier sends to,
`https://fril.co.il/api/live/ingest?token=…`. The app then sends the race on now and
every race to come there (one `UpcomingRaces` event, `race_day/fril.py`) whenever that
changes, and once a minute; fril.co.il/live shows them as **המרוצים הבאים**. Once the
brackets are made, it also sends the finals' mains (one `Brackets` event: each main as
FPVTrackside's Seeder has it, read from <http://localhost:8080> every 10 s and as each race
ends, with the race on now and the next one marked), and fril.co.il/live switches to its
finals order: next up, then every main's bracket like the Seeder's page. The top bar
says when it was last sent, or why it couldn't be.

The race being run is never read from disk; its laps show when it ends — or live, if
FPVTrackside's web server is on (*Menu → Application Settings → Web → Auto start HTTP
Server*, at <http://localhost:8080>).

### 🗣 The announcer

The app can speak the race in place of FPVTrackside, in English and in your club's own
words: **Announcer…** in the top bar of the FPVTrackside screen. No call is written by a
model: every one is a line from the club's files, so it's instant and never says
anything the club didn't write.

- **What it says.**
  - From FPVTrackside's live events: "Arm your quads…" the moment FPVTrackside
    announces the start, who takes the holeshot, every lap with its time and place,
    who takes the lead, the fastest lap of the race, the last lap, who finishes where
    and by how much, a kind word for the last one home, a pilot out, "Race over".
  - From the live video (below): who takes the lead mid-lap and at which gate, the gap
    between the first two through a gate ("Can Slow catch Pennz? 4 tenths in it!"),
    who is down (and so who moves up), who is back in it.
  - Everything else FPVTrackside decides to say (next up, results, hurry up…) is read
    from its log and said as it is. A sound switched off in FPVTrackside isn't said.
  - **Safety first:** FPVTrackside's emergency stop or stand-down is said at once and
    alone, and the race calls stop until the next race.
- **FPVTrackside → the app.** In FPVTrackside, *Menu → Application Settings → Gate / LED
  POST notifications*: **Notifier enabled / mode = ExtensionMode**, **NotificationURL =
  `http://localhost:8766/`**, then restart FPVTrackside. The app passes every event on to
  fril (the **fril live…** address), unchanged, so fril's live board works as before —
  but only while the app is open. The top bar counts the events.
- **FPVTrackside's own voice off**, so you don't hear both: *Turn it off* in the
  Announcer window, with FPVTrackside closed (it sets FPVTrackside's speech volume to 0;
  it writes its settings back when it exits). The start tone and the horn still play.
- **Voice and speed:** any English voice of the Mac (`say`); *Test* says a lap call.

#### Mid-lap positions from the video

Tick **Mid-lap positions from the video** and pick FPVTrackside's capture card (macOS
lets two apps read it). With the FPVTrackside event open and its track learned, a process
of its own reads the card, cuts the 2×2 picture into the pilots' pictures as the event's
last heat recorded them, and names each pilot's gates as they're flown: YOLO on the
Neural Engine → tracker → pass detector → CLIP → the live gate decoder
(`online_decoder.py`, from gilboash/therealspotter) on the learned track.

- It analyses the newest frame each time (15 a second per pilot by default): a slower
  Mac names a gate a little later, but never falls further behind. The window shows its
  frames a second, the share dropped, and how far behind it is.
- Where a pilot is = LapRF's lap + the video's gate in it. A gate counts only a few
  gates on from the last one and only if there was time to fly there (the learned
  gate-to-gate times); a new leader must be a gate (or half a second) ahead and pass
  one more gate first. Nothing from the video is called in the first 6 seconds or
  once the winner is home.
- Tested on two heats of Race 1 (replayed, graded against the full analysis): the
  video's lead and gap calls were right every time; "moves up to second" only about
  half the time, so it's off unless a club turns it on (`video_places: true`).
- About 1.5 GB of memory (its own YOLO and CLIP).
- **▶ Replay the selected heat** plays the heat picked on the FPVTrackside screen to
  the announcer as if it were being raced — its video for the capture card, its
  crossings for FPVTrackside's events — to try it all at home. Nothing goes to fril.
  The same from the command line, printing the calls:
  `venv/bin/python -m race_day.heatreplay <event folder> <heat> --club <club> --track <track>`
  (add `--speak` to hear them). `race_day.ftreplay` plays only the crossings, through the
  app's port.

#### Your club's announcer (clubs/)

Each club has a folder in `clubs/`, picked in the Announcer window. `clubs/example/` (a
made-up club) shows every file; copy it to `clubs/<your club>/` and make it yours. Your
club's folder stays on your computer: git ignores everything in `clubs/` but the example,
so pilots' names and details aren't shared.

| File | What it holds |
|---|---|
| `club.yaml` | the club's name, the announcer's name, how lap times are read (`natural`: "twenty-one point nine six", or `digits`: "twenty-one ninety-six"), the safety line, `video_places`; for the talk: `crew` (league managers, founders, volunteers), `sponsors`, `spectators` (the safety reminders), `next_event`; `calendar_ics` (the club's iCalendar feed — FRIL's is the Google Calendar behind fril.co.il/calendar): the next race in the closing and filler, and today's venue; a copy is kept in `calendar.ics` for a venue without internet |
| `lines.yaml` | the calls: for each kind (`lap`, `lap_lead`, `holeshot`, `finish_win`, `gap`, `down`, `kind`, …), several ways to say it. Heats called by places only use `lap_place`, `lap_lead_place` and `lap_leader_place` (no `{time}`), and by lap times only `lap_time`; brackets add `through`, `to_losers`, `eliminated` and `ace_win` (Event setup). One is picked at random, never the one said last. Words in braces are filled in: `{pilot} {lap} {time} {place} {gate} {margin} {ahead} {behind} {next}`; a line whose words aren't known for that call is skipped for another (so "{pilot} is down at {gate}" is only said when the gate is). Missing kinds fall back to plain defaults. |
| `pilots.yaml` | pilot cards: `callsign` exactly as in FPVTrackside, `say_it_like` (how the voice should say it — try it with *Test*), `new: true` for a new or young pilot (kinder lines, `kind_new`) |
| `tracks/<track>.yaml` | a track: what to call its gates (`4: {say: "the dive gate", crash_spot: true}`, numbered as the event's learned track numbers them, 1 = start/finish) and the expected lap time. Pick it in the window before the race. |
| `talk.yaml` | the talk between races (*Talk between races* in the Announcer window): for each moment, several ways to say it, with the facts filled in. A line that uses more of the known facts wins over a shorter one. When it talks: **▶ Opening** (about 10 minutes: welcome, the day's plan, safety, the track from the analysed section times, the rules of qualifying and the brackets with examples, the points, the season, meet the field); each race's preview (standings, a section tip such as "if X takes gate 5 to 6 like Pennz, that's half a second a lap", what's at stake, the pick, the final walk-out) and wrap (in qualifying, who moved up the seeding; in a bracket, who stays on the winners side, who drops to the losers bracket, who's in the final, and how many races until they fly again; the pick's verdict; then **next up**: the next race's pilots, with the time to the start when the Auto Runner is on, and the race after it); the qualifying results; the champion; filler in a quiet gap; and the closing when **Places & points** shows final results. Never during a race: it stops at the start call, and every race call and FPVTrackside line goes first. |
| `seasons.yaml` | the season standings the talk uses: a finished season's table, and this season's events' points (**Places & points → Save to the season** adds an event's; never a test event's) |
| the countdown | with FPVTrackside's **Auto Runner** on and its **Announce next race in** on, the app follows its countdown ("{time} until the race start", read from FPVTrackside's log). Its clock starts when a race ends: the Auto Runner's results time plus its time to the next race (`AutoRunnerConfig.xml`), so the next race's pilots are reminded while the results still show. It calls it in the club's words (`start_in`, `start_in_first`, `start_in_last` in `lines.yaml`): every step in the first qualifying round, with the pilots told to the start line and to turn their video on, fewer after it. **20 seconds** before the start, a pilot whose picture the live video doesn't have is hurried up by name (`hurry_video`; else `video_ok`) — the live video has to be on. The talk between races stops 35 seconds before the start. "Here we go" and "let's go" are only said in the last 20 seconds (or in the opening and closing): any other line with one is skipped, or said without that sentence. |

Keep race calls short (3–8 words, present tense): they're said while the race is flown.
A `persona.md` next to them is a good place for the style notes the lines follow.

---

## Where the data is

```
<data folder>/
  <track>/                       a track (Live capture, Replays)
    videos/                      its heat videos (video recordings go here)
    lazySpotter/                 everything the app keeps:
      race_day.json                setup and heat states
      videos.json                  what each video is (single / 2×2, channels, pilots)
      gate_memory.json             the learned gates (+ gate_timing.learned.json)
      runs/                        each pilot run's analysis
      gt/                          reviews
  events/<FPVTrackside event>/   an FPVTrackside event folder (heats read in place)
    lazySpotter/                 the same, plus backups/ of every Race.json the app changed
  training/                      models and datasets (FPV_YOLO_tool)
```

Learning data is only ever taken from `lazySpotter/`.

---

## How it works

1. **Gate detector** — YOLO (`cyn_current-20261021_best.pt`) finds the gates in every frame.
2. **Tracking** (`tracking.py`) — each gate is followed from frame to frame; CLIP
   gives each gate crop a picture fingerprint.
3. **Pass detection in two stages** (`pass_scorer.py`, `pass_scorer.json`) — every
   moment a big gate disappears is a candidate; a trained scorer keeps almost all
   of them (low threshold), and the gate-sequence decoder drops the ones that don't
   fit the track.
4. **Gate numbers and laps** (`gate_decoder.py`) — the most likely sequence of
   gates from their order, the time between gates, and how each gate looks;
   each pass's time is corrected by its gate's usual offset.
5. **Learning a track** (`learn_track.py`) — gates and gate-to-gate timing are
   learned from a few runs, given only the number of gates.

On 52 reviewed runs of three tracks: 96.2% of passes found on time, 55 false
passes, gate number right 98.7%.

**A heat's pilots are analysed together** (`race_day/heat_analysis.py`), in one
worker process, in three stages that run at the same time:

- **decoding** — the heat video is read once, by the Mac's hardware decoder
  (VideoToolbox, through ffmpeg; OpenCV without it), and cut into the pilots'
  pictures; at most 30 frames a second (a 60 fps heat video: every 2nd frame),
  and for an FPVTrackside heat only the race (from 5 s before the start to 30 s
  after the end — the staging isn't analysed);
- **the detector** — YOLO as a Core ML model on the **Neural Engine**
  (`race_day/models.py`: exported once, at exactly the picture size the
  analysis uses, fp16), running a few moments ahead of the pilots; a pilot whose
  picture has been gray (no signal) or still (crashed, landed, on the pad) for
  2 s gets no YOLO until it moves again;
- **the pilots** — each its own pipeline (steps 2–3); CLIP is loaded once.

Without Core ML the detector is the .pt on the GPU, the pilots' pictures of a
moment in one call. On 2027Race1 (4 heats, 16 pilots) the laps and the lap check
against LapRF match the GPU's (all 38 timing laps; one crossing it left unsure is
now confirmed), and 98% of gate passes are within 0.5 s of it; the analysis uses
about 1.3 GB. Labelling and learning run in a second, light process.

## Project structure

```
therealspotter/
├── lazy_spotter_v2.py       # the app
├── race_day/                # its screens, controller, capture, review, analysis workers, dashboard (static/)
├── extract_race.py          # the analysis of one video (run by the app's workers)
├── tracking.py              # gate tracker and CLIP embedder
├── pass_detector.py         # pass candidates
├── pass_scorer.py           # two-stage pass detection   (+ pass_scorer.json, the trained scorer)
├── gate_decoder.py          # gate numbers and laps
├── online_decoder.py        # gate numbers while the race is flown (the announcer's live video)
├── learn_track.py           # learning a track
├── extract_motion.py        # camera motion (for the pass scorer)
├── gate_db.py               # gate memory
├── pipeline_cfg.py          # pipeline settings (debug_ui_defaults.json)
├── score.py                 # measures the system against the reviews
├── dataset_paths.py         # where the data is
├── clubs/                   # each club's announcer words (only clubs/example/ is in git)
└── cyn_current-20261021_best.pt   # the gate detector
```
