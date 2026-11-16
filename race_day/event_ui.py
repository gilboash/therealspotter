"""
The FPVTrackside event screen of lazy_spotter_v2 (gate validation for FPVTrackside).

    ┌ top bar: event · Learn / Relearn track · Auto · status ──────────────┐
    │ heats (newest first, ☑ = learn) │ heat: 2×2 video with pilot names  │
    │                                 │ timeline (timing / video / checks) │
    │                                 │ per pilot: laps, disagreements     │
    │                                 │ Accept · Reject · Add lap · Remove │
    │                                 │ To do in FPVTrackside (the steps)  │
    │                                 │ Update laps in FPVTrackside        │
    └─────────────────────────────────┴────────────────────────────────────┘
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from race_day.flowlayout import FlowLayout

JUMP_FRAMES = 6         # clicking a lap jumps this many frames before its pass
STATUS_ICON = {"new": "•", "analysing": "⏳", "analysed": "◦", "checked": "✓", "updated": "↻"}
LANE_COLORS = {"timing": QColor(235, 235, 235), "video": QColor(80, 200, 120), "flag": QColor(255, 160, 40),
               "mark": QColor(90, 160, 255), "gate": QColor(60, 130, 90),
               "remove": QColor(230, 60, 60), "add": QColor(80, 200, 120)}
UPPER = ("timing", "remove", "add")      # the upper half of a pilot's row: FPVTrackside's laps
TIMELINE_LEGEND = ("Upper half — FPVTrackside's laps:  <b>white</b> now  ·  <span style='color:#e63c3c'>red</span> removed "
                   "at the next update  ·  <span style='color:#50c878'>green</span> added at the next update.     "
                   "Lower half — the video:  <span style='color:#50c878'>green</span> start/finish passes  ·  "
                   "<span style='color:#ffa028'>orange</span> to check (the video has no start/finish pass there)")


def _mmss(t: Optional[float]) -> str:
    if t is None:
        return "–"
    return f"{int(t // 60)}:{t % 60:05.2f}"


def _who(channel: str, pilot: str) -> str:
    """A pilot's label: the channel, and the name when there is one."""
    return channel if pilot in (None, "", "?", channel) else f"{channel} {pilot}"


GATE_TOL_S = 0.5      # a lap's own start/finish pass may land just after its lap line (the antenna offset)


def lap_gates(passes: List[dict], n_gates: int, t0: Optional[float], t1: Optional[float]) -> Optional[tuple]:
    """The different gates a pilot passed in one lap — between the lap lines
    at video times t0 and t1 — and the ones not seen: (seen, n_gates, missing).
    None when it can't be told (no lap start, or the run's gates aren't
    labelled yet). The lap's own start/finish pass is counted once: its end
    (the pass at t1), not its start."""
    if t0 is None or t1 is None or n_gates < 1 or not any(x.get("gate", -1) >= 1 for x in passes):
        return None
    seen = {x["gate"] for x in passes if x.get("gate", -1) >= 1 and t0 + GATE_TOL_S < x["t"] <= t1 + GATE_TOL_S}
    return len(seen), n_gates, [g for g in range(1, n_gates + 1) if g not in seen]


def step_text(st: dict) -> str:
    """A change as a marshal's step in FPVTrackside (two lines). Stock
    FPVTrackside's Lap Editor adds a lap only after the last one, so a lap in
    between is added from the Replay, at its race time."""
    lap = "the holeshot" if st["lap"] == 0 else f"lap {st['lap']}"
    took = f" ({st['lap_time']:.2f} s lap)" if st.get("lap_time") else ""
    who = _who(st["channel"], st["pilot"])
    at = _mmss(st["race_t"])
    if st["op"] == "remove":
        return (f"{who} — disqualify {lap} at race time {at}{took}\n"
                f"      Lap Editor → right-click that lap → Disqualify Lap")
    how = f"Replay → seek to {at} → right-click {st['pilot'] if st['pilot'] not in ('', '?') else st['channel']}'s laps → Add Lap Now"
    if st.get("last") and st.get("lap_time"):
        how += f"   (or Lap Editor → Add Lap → {st['lap_time']:.2f})"
    return f"{who} — add {lap} at race time {at}{took}\n      {how}"


def _clock(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 10)}"


# one colour per pilot row (the dashboard uses the same palette)
PILOT_COLORS = [QColor(77, 163, 255), QColor(255, 159, 67), QColor(62, 207, 142), QColor(197, 108, 240),
                QColor(255, 107, 107), QColor(245, 196, 81), QColor(0, 210, 211), QColor(255, 159, 243)]

REVIEW_COLORS = {"pending": QColor(150, 150, 150), "accepted": QColor(60, 220, 120), "fixed_id": QColor(255, 160, 40),
                 "added": QColor(80, 160, 255), "skipped": QColor(180, 100, 255), "rejected": QColor(230, 60, 60)}


class Timeline(QWidget):
    """One lane per pilot; click to seek. FPVTrackside events: the upper half
    is FPVTrackside's laps (white now, red to be removed, green to be added
    at the next update), the lower half the video (green start/finish
    passes, orange things to check). Replays / Live capture: green start/finish
    passes and the other gates (dark green).
    While reviewing, a row on top with the review marks (click one to select it)."""
    seek = pyqtSignal(float)
    mark_clicked = pyqtSignal(object)

    LABEL_W = 44                         # the pilot label column
    MIN_LANE = 14                        # a pilot row is at least this tall; it grows with the timeline

    def __init__(self):
        super().__init__()
        self.duration, self.pos = 1.0, 0.0
        self.lanes: List[dict] = []
        self.review: list = []                   # race_day.review.Mark
        self.review_sel = None
        self.highlight = None                    # the selected pilot's row is drawn brighter
        self._fit()

    def _fit(self):
        self.setMinimumHeight(self._top() + self.MIN_LANE * max(1, len(self.lanes)) + 4)

    def set_review(self, marks: list, sel=None):
        self.review, self.review_sel = marks, sel
        self._fit()
        self.update()

    def _top(self) -> int:
        return 20 if self.review else 0

    def _lane_h(self) -> float:
        return max(self.MIN_LANE, (self.height() - self._top() - 2) / max(1, len(self.lanes)))

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(600, self._top() + 26 * max(1, len(self.lanes)) + 4)

    def set_data(self, duration: float, lanes: List[dict]):
        self.duration, self.lanes = max(duration, 1.0), lanes
        self._fit()
        self.update()

    def set_pos(self, t: float):
        self.pos = t
        self.update()

    def _x(self, t):
        return self.LABEL_W + 4 + (self.width() - self.LABEL_W - 14) * t / self.duration

    def mousePressEvent(self, e):
        x = e.position().x()
        if self.review and e.position().y() < self._top():
            near = [m for m in self.review if abs(self._x(m.t) - x) <= 5]
            if near:
                return self.mark_clicked.emit(min(near, key=lambda m: abs(self._x(m.t) - x)))
        t = (x - self.LABEL_W - 4) / max(1, self.width() - self.LABEL_W - 14) * self.duration
        self.seek.emit(float(min(max(t, 0.0), self.duration)))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        p.setFont(QFont("Helvetica", 9))
        if self.review:
            p.setPen(QColor(150, 150, 150))
            p.drawText(QRectF(2, 3, 36, 14), Qt.AlignmentFlag.AlignVCenter, "review")
            for m in self.review:
                x = int(self._x(m.t))
                pen = QPen(REVIEW_COLORS.get(m.state, REVIEW_COLORS["pending"]))
                pen.setWidthF(3.0 if m is self.review_sel else 1.5)
                p.setPen(pen)
                p.drawLine(x, 3, x, 17)
        lh = self._lane_h()
        W = self.width()
        for i, lane in enumerate(self.lanes):
            y0 = self._top() + 1 + lh * i
            # alternate shades, a separator and the pilot's colour: rows easy to tell apart
            p.fillRect(QRectF(0, y0, W, lh), QColor(52, 56, 68) if lane["name"] == self.highlight
                       else QColor(34, 36, 42) if i % 2 == 0 else QColor(24, 25, 30))
            p.fillRect(QRectF(0, y0 + 1, 4, lh - 2), PILOT_COLORS[i % len(PILOT_COLORS)])
            p.setPen(QColor(55, 58, 66))
            p.drawLine(0, int(y0 + lh), W, int(y0 + lh))
            p.setPen(QPen(QColor(48, 50, 58), 1, Qt.PenStyle.DotLine))      # upper half | lower half
            p.drawLine(self.LABEL_W, int(y0 + lh / 2), W, int(y0 + lh / 2))
            f = QFont("Helvetica", int(min(12, max(9, lh * 0.38))))
            f.setBold(True)
            p.setFont(f)
            p.setPen(PILOT_COLORS[i % len(PILOT_COLORS)])
            p.drawText(QRectF(8, y0, self.LABEL_W - 8, lh), Qt.AlignmentFlag.AlignVCenter, lane["name"])
            pad = max(1.5, lh * 0.08)
            half = lh / 2
            for kind, ts in lane["marks"].items():
                pen = QPen(LANE_COLORS[kind])
                pen.setWidthF(2.0 if kind in ("flag", "mark", "remove", "add") else 1.5)
                p.setPen(pen)
                top = y0 + (0 if kind in UPPER else half)
                y1, y2 = top + pad + (half * 0.35 if kind == "gate" else 0), top + half - pad
                for t in ts:
                    if t is None:
                        continue
                    x = int(self._x(t))
                    p.drawLine(x, int(y1), x, int(y2))
        pen = QPen(QColor(255, 60, 60))
        pen.setWidth(2)
        p.setPen(pen)
        x = self._x(self.pos)
        p.drawLine(int(x), 0, int(x), self.height())


class HeatPlayer(QWidget):
    """Plays a heat video in place with each quadrant's channel and pilot."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.view = QLabel("Choose a heat")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(480, 270)
        self.view.setStyleSheet("background:#111;color:#666")
        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek)
        self.split = QSplitter(Qt.Orientation.Vertical)      # drag the boundary: a taller timeline
        self.split.addWidget(self.view)
        self.split.addWidget(self.timeline)
        self.split.setChildrenCollapsible(False)
        self.split.setStretchFactor(0, 1)
        self.split.setSizes([400, 110])
        v.addWidget(self.split, 1)
        self.which = QComboBox()
        self.which.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.which.setMinimumContentsLength(18)
        self.which.currentIndexChanged.connect(self._switch)
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.toggle)
        back, fwd = QPushButton("◀ 1s"), QPushButton("1s ▶")
        back.clicked.connect(lambda: self.seek(self.t - 1.0))
        fwd.clicked.connect(lambda: self.seek(self.t + 1.0))
        fb, ff = QPushButton("◀ frame"), QPushButton("frame ▶")
        fb.clicked.connect(self.frame_back)
        ff.clicked.connect(self.frame_forward)
        self.time_lbl = QLabel("")
        self.live_btn = QPushButton("⟲ Back to live")
        self.live_btn.setToolTip("Back in step with the heat being replayed to the announcer: the frame the live video\n"
                                 "is reading right now, as on race day")
        self.live_btn.clicked.connect(self._sync)
        self.live_btn.hide()
        bar = QWidget()                          # the controls wrap onto more lines in a narrow window
        row = FlowLayout(bar)
        for w in (self.which, self.play_btn, self.live_btn, back, fb, ff, fwd, self.time_lbl):
            row.addWidget(w)
        v.addWidget(bar)
        self.cap = None
        self.videos: List[str] = []
        self.quads: Dict[str, List[dict]] = {}    # video → [{"rect", "channel", "pilot"}]
        self.laps_at = {}                        # channel → sorted video times of timing laps
        self.t, self.fps, self.duration = 0.0, 30.0, 1.0
        self.playing, self._t0, self._p0 = False, 0.0, 0.0
        self.frame = None
        self.focus = None                        # (rect, channel): show only this quadrant, enlarged
        self.highlight = None                    # channel whose feed stands out (the selected pilot)
        self.colors: Dict[str, QColor] = {}      # channel → its colour (the timeline row's)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.clock = None                        # a heat replayed to the announcer: {"video": s, "wall": clock time}

    def follow(self, path: str, clock: dict) -> bool:
        """Play `path` in step with a heat replayed to the announcer: at each moment the frame that the live
        video is reading then (clock: video second `video` plays at clock time `wall`). False if this heat's
        videos aren't the ones loaded."""
        if self.path != path:
            idx = self.which.findData(path)
            if idx < 0:
                return False
            self.which.setCurrentIndex(idx)
        self.clock = clock
        self.live_btn.show()
        self._sync()
        return True

    def unfollow(self):
        self.clock = None
        self.live_btn.hide()

    def _sync(self):
        """Back in step with the replay's clock."""
        if not self.clock or not self.cap:
            return
        c = self.clock
        self.stop()
        self.seek(max(0.0, c["video"] + (time.time() - c["wall"])))
        self.playing = True
        self._t0, self._p0 = c["wall"], c["video"]      # before the video's first frame: it waits on it
        self.play_btn.setText("❚❚")
        self.timer.start(15)

    def load(self, videos: List[str], quads: Dict[str, List[dict]], laps_at: Dict[str, List[float]]):
        self.stop()
        self.unfollow()
        self.videos, self.quads, self.laps_at = videos, quads, laps_at
        self.which.blockSignals(True)
        self.which.clear()
        for vpath in videos:
            dji = any(q.get("kind") == "dji" for q in quads.get(vpath, []))
            self.which.addItem("DJI feeds" if dji else "Analog 2×2", vpath)
        self.which.setVisible(len(videos) > 1)
        self.which.blockSignals(False)
        self._open(videos[0] if videos else None)

    def _switch(self, i):
        t = self.t
        self._open(self.which.itemData(i))
        self.seek(t)

    def _open(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path) if path else None
        self.path = path
        if not self.cap or not self.cap.isOpened():
            self.view.setText("Can't open the video")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.seek(0.0)

    def seek(self, t: float):
        if not self.cap:
            return
        t = min(max(t, 0.0), self.duration)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        self.step()
        if self.playing:
            self._t0, self._p0 = time.time(), self.t

    def seek_frames_before(self, t: float, frames: int):
        """Show the frame `frames` frames before time t (counted in real frames:
        heat videos don't have evenly spaced frames)."""
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_MSEC, min(max(t, 0.0), self.duration) * 1000.0)
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - frames))
        self.step()

    def step(self):
        if not self.cap:
            return
        ok, f = self.cap.read()
        if ok:
            self.frame, self.t = f, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self._show()

    def frame_forward(self):
        """Pause and show the next frame."""
        self.stop()
        self.step()

    def frame_back(self):
        """Pause and show the previous frame — by frame number: heat videos
        don't have evenly spaced frames, so going back by time can land on the
        same frame again."""
        if not self.cap:
            return
        self.stop()
        nxt = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))       # after a read: the next frame's index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, nxt - 2))
        self.step()

    def toggle(self):
        if self.playing:
            self.stop()
        elif self.cap:
            self.playing = True
            self._t0, self._p0 = time.time(), self.t
            self.play_btn.setText("❚❚")
            self.timer.start(15)

    def stop(self):
        self.playing = False
        self.play_btn.setText("▶")
        self.timer.stop()

    def _tick(self):
        target = self._p0 + (time.time() - self._t0)
        if target >= self.duration:
            return self.stop()
        n = 0
        while self.t < target - 0.5 / self.fps and n < 8:      # catch up: skip frames
            if not self.cap.grab():
                return self.stop()
            self.t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            n += 1
        if self.t < target:
            self.step()

    def set_highlight(self, channel: Optional[str]):
        """Make this pilot's feed stand out: a border in their colour, the other feeds a bit darker."""
        if channel != self.highlight:
            self.highlight = channel
            self.timeline.highlight = channel
            self.timeline.update()
            if self.frame is not None:
                self._show()

    def set_focus(self, rect=None, channel: str = ""):
        """Show only this quadrant of the video, enlarged (reviewing one pilot)."""
        self.focus = (tuple(rect), channel) if rect else None
        if self.frame is not None:
            self._show()

    def _show(self):
        img = self.frame
        if self.focus:
            (x, y, w, h), ch = self.focus
            H0, W0 = img.shape[:2]
            img = img[int(round(y * H0)):int(round((y + h) * H0)), int(round(x * W0)):int(round((x + w) * W0))]
        W, H = self.view.width(), self.view.height()
        h0, w0 = img.shape[:2]
        s = min(W / w0, H / h0)
        img = cv2.resize(img, (max(1, int(w0 * s)), max(1, int(h0 * s))), interpolation=cv2.INTER_AREA)
        h1, w1 = img.shape[:2]
        quads = self.quads.get(self.path, [])
        if self.highlight and not self.focus and len(quads) > 1 and any(q["channel"] == self.highlight for q in quads):
            for q in quads:
                x0, y0 = int(q["rect"][0] * w1), int(q["rect"][1] * h1)
                x1, y1 = int((q["rect"][0] + q["rect"][2]) * w1), int((q["rect"][1] + q["rect"][3]) * h1)
                if q["channel"] != self.highlight:
                    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.55).astype(img.dtype)     # the others a bit darker
            q = next(q for q in quads if q["channel"] == self.highlight)
            c = self.colors.get(self.highlight, QColor(77, 163, 255))
            x0, y0 = int(q["rect"][0] * w1), int(q["rect"][1] * h1)
            x1, y1 = int((q["rect"][0] + q["rect"][2]) * w1) - 1, int((q["rect"][1] + q["rect"][3]) * h1) - 1
            cv2.rectangle(img, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), (c.blue(), c.green(), c.red()), 4)
        for q in quads:
            if self.focus:
                if q["channel"] != self.focus[1]:
                    continue
                x, y = 0, 0
            else:
                x, y = int(q["rect"][0] * w1), int(q["rect"][1] * h1)
            laps = sum(1 for t in self.laps_at.get(q["channel"], []) if t <= self.t)
            text = _who(q["channel"], q["pilot"]).replace(" ", "  ", 1) + (f"   lap {laps - 1}" if laps > 1 else (" HS" if laps == 1 else ""))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x + 4, y + 4), (x + 12 + tw, y + 12 + th), (0, 0, 0), -1)
            cv2.putText(img, text, (x + 8, y + 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(qi.copy()))
        self.time_lbl.setText(f"{_mmss(self.t)} / {_mmss(self.duration)}")
        self.timeline.set_pos(self.t)

    def close_video(self):
        self.stop()
        if self.cap:
            self.cap.release()
        self.cap = None


class EventPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win, self.day = win, None
        self.dash_url = ""
        self.heat_id: Optional[str] = None
        self._detail_sig = None
        v = QVBoxLayout(self)
        self.title = QLabel("")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        self.title.setFont(f)
        self.learn_btn = QPushButton("Learn track")
        self.learn_btn.clicked.connect(self._learn)
        self.setup_btn = QPushButton("Event setup…")
        self.setup_btn.setToolTip("The event's classes (Open, Beginners…): their pilots, the track layout each flies,\n"
                                  "qualifying, the brackets and points — made in FPVTrackside through its API")
        self.setup_btn.clicked.connect(self._setup)
        self.auto = QCheckBox("Auto")
        self.auto.setToolTip("New heats are analysed and checked by themselves, and confident lap corrections\n"
                             "are written to FPVTrackside's files. Uncertain ones wait for you.\n"
                             "FPVTrackside shows written laps after the event is reopened; don't edit those\n"
                             "heats in FPVTrackside before then, or the change is lost.")
        self.auto.toggled.connect(lambda on: self.day and self.day.set_auto(on))
        self.pause_race = QCheckBox("Pause during races")
        self.pause_race.setToolTip("Stop analysing while FPVTrackside runs a race, so it has the computer to\n"
                                   "itself (on), or keep analysing (off). Lap checks and writing laps to\n"
                                   "FPVTrackside always wait for the race to end.")
        self.pause_race.toggled.connect(lambda on: self.day and self.day.set_pause_in_race(on))
        self.status = QLabel("")
        self.status.setStyleSheet("color:#aaa")
        self.ft_btn = QPushButton("FPVTrackside…")
        self.ft_btn.setToolTip("Where FPVTrackside's web server is. With the FPVTrackside alpha (its REST API) and\n"
                               "this event open there, lap changes are made in FPVTrackside directly: they show at once.")
        self.ft_btn.clicked.connect(lambda: win.set_fpvtrackside())
        self.fril_btn = QPushButton("fril live…")
        self.fril_btn.setToolTip("Send the race on now and the races to come to fril.co.il/live\n"
                                 "(its live ingest address, the same one FPVTrackside sends to)")
        self.fril_btn.clicked.connect(lambda: win.set_fril())
        self.ann_btn = QPushButton("Announcer…")
        self.ann_btn.setToolTip("The app speaks the race calls in place of FPVTrackside, from its live events\n"
                                "(which it passes on to fril)")
        self.ann_btn.clicked.connect(lambda: win.set_announcer())
        stop = QPushButton("Close")
        stop.clicked.connect(lambda: win.stop_event())
        # the title and buttons wrap onto more lines when the window is narrow; the status has a line of its own
        bar = QWidget()
        flow = FlowLayout(bar)
        for w in (self.title, self.setup_btn, self.learn_btn, self.auto, self.pause_race, self.ft_btn,
                  self.fril_btn, self.ann_btn, stop):
            flow.addWidget(w)
        v.addWidget(bar)
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(100)
        v.addWidget(self.status)
        self.banner = QLabel("")
        self.banner.setStyleSheet("background:#7a4b00;color:white;padding:4px")
        self.banner.setVisible(False)
        v.addWidget(self.banner)
        self.alert = QLabel("")                               # what needs a person: a bracket, the gate numbering
        self.alert.setStyleSheet("background:#8b1a1a;color:white;padding:4px")
        self.alert.setWordWrap(True)
        self.alert.setVisible(False)
        v.addWidget(self.alert)

        split = self.split = QSplitter(Qt.Orientation.Horizontal)
        left = self.left = QWidget()          # the heat list's side (LivePage adds its buttons here)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.upnext = QLabel("")               # FPVTrackside events: the race on now and the next one
        self.upnext.setWordWrap(True)
        self.upnext.setStyleSheet("background:#1b2232;color:#ddd;padding:5px;border-radius:4px")
        self.upnext.hide()
        lv.addWidget(self.upnext)
        self.heats = QListWidget()
        self.heats.setMinimumWidth(180)
        self.heats.currentItemChanged.connect(self._select)
        self.heats.itemChanged.connect(self._checked)
        lv.addWidget(self.heats, 1)
        split.addWidget(left)

        right = self.right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.heat_title = QLabel("")
        self.heat_title.setStyleSheet("font-weight:bold")
        rv.addWidget(self.heat_title)
        vsplit = QSplitter(Qt.Orientation.Vertical)
        self.player = HeatPlayer()
        self.legend = QLabel(TIMELINE_LEGEND)
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet("color:#999;font-size:11px")
        self.player.layout().insertWidget(1, self.legend)        # under the timeline
        vsplit.addWidget(self.player)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Video", "Clock", "Race time", "What", "Lap time", "Gates", "Check"])
        self.tree.headerItem().setToolTip(5, "Gates the video saw the pilot pass in this lap (hover a lap for the "
                                             "ones not seen)")
        self.tree.headerItem().setToolTip(2, "Time from the race start: what FPVTrackside's replay shows")
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemDoubleClicked.connect(self._jump)
        self.tree.itemClicked.connect(self._jump)
        self.tree.currentItemChanged.connect(self._tree_selected)
        self.tree.itemChanged.connect(self._pilot_checked)
        vsplit.addWidget(self.tree)
        # the accepted changes as steps for a marshal in FPVTrackside: stock FPVTrackside shows laps
        # written to Race.json only after the event is reopened, a change made there shows at once
        self.todo_box = QWidget()
        tv = QVBoxLayout(self.todo_box)
        tv.setContentsMargins(0, 0, 0, 0)
        th = QHBoxLayout()
        self.todo_title = QLabel("")
        self.todo_title.setStyleSheet("font-weight:bold")
        th.addWidget(self.todo_title, 1)
        b_copy = QPushButton("Copy race time")
        b_copy.setToolTip("The selected step's race time, to type into FPVTrackside")
        b_copy.clicked.connect(self._copy_step)
        th.addWidget(b_copy)
        tv.addLayout(th)
        self.todo = QListWidget()
        self.todo.setStyleSheet("font-family:Menlo,monospace;font-size:12px")
        self.todo.itemClicked.connect(self._step_clicked)
        tv.addWidget(self.todo)
        hint = QLabel("Make each change in FPVTrackside: it shows there at once, and the step goes away here when "
                      "FPVTrackside saves the heat. Or Update laps in FPVTrackside writes them all to the file — "
                      "FPVTrackside shows those only after the event is reopened.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#999;font-size:11px")
        tv.addWidget(hint)
        self.todo_box.hide()
        vsplit.addWidget(self.todo_box)
        from race_day.review_ui import ReviewPanel
        self.review_panel = ReviewPanel(self, self.player)
        self.review_panel.finished.connect(self._review_end)
        # a review updates the run's results (tabs, lap check, dashboard) 2 s after the last change
        self._review_heat: Optional[str] = None
        self._review_timer = QTimer(self)
        self._review_timer.setSingleShot(True)
        self._review_timer.setInterval(2000)
        self._review_timer.timeout.connect(self._review_apply)
        self.review_panel.changed.connect(self._review_timer.start)
        self.review_panel.hide()
        vsplit.addWidget(self.review_panel)
        vsplit.setSizes([520, 300, 160, 300])
        rv.addWidget(vsplit, 1)
        bbar = QWidget()                         # the buttons wrap onto more lines in a narrow window
        btns = FlowLayout(bbar)
        self.b_analyse = QPushButton("Analyse")
        self.b_accept = QPushButton("✓ Accept")
        self.b_reject = QPushButton("✗ Reject")
        self.b_add = QPushButton("+ Lap / HS here")
        self.b_remove = QPushButton("− Remove lap")
        self.b_write = QPushButton("Update laps in FPVTrackside")
        self.b_write.setStyleSheet("QPushButton{background:#2b6cb0;color:white;font-weight:bold;padding:6px 12px}"
                                   "QPushButton:disabled{background:#444;color:#888}")
        self.b_restore = QPushButton("Restore original laps")
        self.b_review = QPushButton("Review")
        self.b_review.setToolTip("Check every gate pass of the selected pilot: accept, fix or add passes "
                                 "(saved as the run's review, used to measure and retrain the system)")
        self.b_review.clicked.connect(self._review)
        self.b_analyse.clicked.connect(lambda: self.heat_id and self.day.analyse([self.heat_id]))
        self.b_accept.clicked.connect(lambda: self._decide("accepted"))
        self.b_reject.clicked.connect(lambda: self._decide("rejected"))
        self.b_add.clicked.connect(self._add)
        self.b_remove.clicked.connect(self._remove)
        self.b_write.clicked.connect(self._write)
        self.b_restore.clicked.connect(self._restore)
        for b in (self.b_analyse, self.b_review, self.b_accept, self.b_reject, self.b_add, self.b_remove,
                  self.b_restore, self.b_write):
            btns.addWidget(b)
        rv.addWidget(bbar)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#888")
        rv.addWidget(self.note)
        split.addWidget(right)
        split.setSizes([300, 980])
        v.addWidget(split, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        # the video keys, the same in every screen (and while reviewing)
        p = self.player
        for key, fn in (("Space", p.toggle), ("Left", p.frame_back), ("Right", p.frame_forward),
                        ("Shift+Left", lambda: p.seek(p.t - 1.0)), ("Shift+Right", lambda: p.seek(p.t + 1.0))):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ── setup ──────────────────────────────────────────────────

    def open(self, day):
        self.day = day
        self.heat_id, self._detail_sig = None, None
        self.heats.clear()
        self.tree.clear()
        self.auto.blockSignals(True)
        self.auto.setChecked(bool(day.state.get("auto")))
        self.auto.blockSignals(False)
        self.auto.setVisible(day.mode == "live")
        self.pause_race.blockSignals(True)
        self.pause_race.setChecked(bool(day.state.get("pause_in_race", True)))
        self.pause_race.blockSignals(False)
        self.pause_race.setVisible(day.event.has_timing and day.mode != "capture")
        timing = day.event.has_timing
        for b in (self.b_accept, self.b_reject, self.b_add, self.b_remove, self.b_write, self.b_restore):
            b.setVisible(timing)
        self.ft_btn.setVisible(timing and day.mode != "capture")
        self.fril_btn.setVisible(timing and day.mode == "live")
        self.ann_btn.setVisible(timing and day.mode == "live")
        self.b_review.setVisible(True)
        self.legend.setVisible(timing)
        self._asked = set()
        self.timer.start(1000)
        self.tick()

    def close_day(self):
        self._review_end()
        self.timer.stop()
        self.player.close_video()

    # ── periodic refresh (state only — cheap) ──────────────────

    def tick(self):
        if not self.day:
            return
        st = self.day.status()
        self.title.setText(f"{ {'live': 'LIVE', 'capture': 'TRACK'}.get(st['mode'], 'REPLAYS')} — {st['event']}")
        self.learn_btn.setText("Relearn track" if st["learned"] else "Learn track")
        lays = st.get("layouts") or {}
        if len(lays) > 1:                                     # one track layout per class (Event setup)
            bits = [" · ".join(f"{n}: {'learned' if l['learned'] else 'not learned'} ({l['n_gates']} gates)"
                               for n, l in lays.items())]
        else:
            bits = [f"learned · {st['n_gates']} gates" if st["learned"] else "not learned yet"]
        if st["learning"] or st["learn_waiting"]:
            bits.append("learning" + (f" {st['learning_layout']}" if len(lays) > 1 else "") + "…")
        if st["analysing"]:
            bits.append(f"analysing {st['analysing']} pilots")
        if st["labelling"]:
            bits.append(f"labelling {st['labelling']} heats")
        bits.append(f"{st['heats']} heats")
        ft = st.get("fpvtrackside") or {}
        if ft.get("url") and self.day.event.has_timing:
            bits.append("FPVTrackside: " + ("REST API ✓ — changes show there at once" if ft["serves"]
                                            else "REST API, another event open" if ft["level"] == "rest"
                                            else "web server only — changes go to its files" if ft["level"] == "web"
                                            else "not reachable — changes go to its files"))
        if st.get("finals"):
            bits.append(st["finals"])
        if self.dash_url:
            bits.append(f"dashboard {self.dash_url}")
        fr = st.get("fril")
        if fr:
            bits.append("fril live: ⚠ " + fr["error"] if fr["error"] else
                        f"fril live: sent {time.strftime('%H:%M:%S', time.localtime(fr['last_ok']))}"
                        if fr["last_ok"] else "fril live: sending…")
        relay, ann = getattr(self.win, "relay", None), getattr(self.win, "announcer", None)
        if relay and st["mode"] == "live" and self.day.event.has_timing:
            rs = relay.status()
            if rs["received"]:
                bits.append(f"FPVTrackside events: {rs['received']}" + (f" (⚠ fril: {rs['forward_error']})"
                                                                         if rs["forward_error"] else ""))
            if ann and ann.enabled:
                bits.append("🗣 announcer on")
        self.status.setText("   ·   ".join(bits))
        racing_on = st["race_running"] and not st["paused"] and st["mode"] != "capture"
        self.banner.setVisible(st["paused"] or racing_on)
        if st["paused"]:
            self.banner.setStyleSheet("background:#7a4b00;color:white;padding:4px")
            self.banner.setText("● Recording — analysis and learning wait until it stops" if st["mode"] == "capture"
                                else "⏸  Race running in FPVTrackside — race_day is paused until it ends")
            if st["mode"] != "capture":
                self.player.stop()
        elif racing_on:
            self.banner.setStyleSheet("background:#1e4e79;color:white;padding:4px")
            self.banner.setText("▶  Race running in FPVTrackside — analysis goes on (pause during races is off); "
                                "lap checks and writing laps wait for the race to end")
        alerts = ["⚠ " + w for w in (st.get("finals_warnings") or []) + (st.get("warnings") or [])]
        self.alert.setVisible(bool(alerts))
        self.alert.setText("\n".join(alerts[-4:]))
        self._fill_upnext()
        self._fill_heats()
        if self.heat_id:
            self._show_heat()
        self._ask_videos(st.get("unconfigured", []))

    def _fill_upnext(self):
        sc = self.day.race_day()
        if not sc or not sc.get("has") or not (sc["current"] or sc["upcoming"]):
            self.upnext.hide()
            return
        line = lambda r: f"<b>{r['label']}</b> — " + ", ".join(_who(p["channel"], p["name"]) for p in r["pilots"])
        bits = []
        if sc["current"]:
            bits.append("<span style='color:#ff3b5c'>● Racing now</span>  " + line(sc["current"]))
        if sc["upcoming"]:
            bits.append("Next  " + line(sc["upcoming"][0]) + (f"  <span style='color:#999'>(+{len(sc['upcoming']) - 1} more)</span>"
                                                              if len(sc["upcoming"]) > 1 else ""))
        self.upnext.setText("<br>".join(bits))
        self.upnext.show()

    def _ask_videos(self, names: List[str]):
        """Videos nobody has described yet: ask about one (the next tick asks the next)."""
        for name in names:
            if name not in self._asked:
                self._asked.add(name)
                self._configure_video(name)
                break

    def _fill_heats(self):
        rows = self.day.heat_list()
        self.heats.blockSignals(True)
        cur = self.heat_id
        if self.heats.count() != len(rows):
            self.heats.clear()
            for _ in rows:
                it = QListWidgetItem()
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                self.heats.addItem(it)
        for i, r in enumerate(rows):
            it = self.heats.item(i)
            laps = ", ".join(_who(c, r["pilots"][c]) + (" ✔" if c in r.get("reviewed", []) else "")
                             + (f" ({r['timing_laps'][c]})" if r["timing_laps"].get(c) is not None
                                                          else "") + (" 📵" if c in r.get("no_video", []) else "")
                             for c in sorted(r["pilots"]))
            flag = f"  ⚠ {r['open']}" if r["open"] else ""
            it.setText(f"{STATUS_ICON.get(r['status'], '•')} {r['number']}  {r['label']}{flag}\n    {laps}")
            it.setData(Qt.ItemDataRole.UserRole, r["id"])
            it.setCheckState({"all": Qt.CheckState.Checked, "some": Qt.CheckState.PartiallyChecked}
                             .get(r["learn"], Qt.CheckState.Unchecked))
            it.setToolTip("☑ = learn the track from this heat's pilots (or tick single pilots in the heat view)")
            if r["id"] == cur and self.heats.currentRow() != i:
                self.heats.setCurrentRow(i)
        self.heats.blockSignals(False)

    def _checked(self, item):
        hid = item.data(Qt.ItemDataRole.UserRole)
        if hid and self.day:
            self.day.set_checked(hid, item.checkState() == Qt.CheckState.Checked)

    def _select(self, item, _prev=None):
        if not item:
            return
        hid = item.data(Qt.ItemDataRole.UserRole)
        if hid == self.heat_id:
            return
        self._review_end()                       # another heat: the review is saved as it is
        self.heat_id, self._detail_sig = hid, None
        self.player.set_highlight(None)
        self.note.setText("")
        self._show_heat(load_video=True)

    # ── heat view ──────────────────────────────────────────────

    def _show_heat(self, load_video: bool = False):
        d = self.day.heat_detail(self.heat_id)
        if not d:
            return
        self.detail = d
        sig = (d["status"], len(d["marks"]), len(d["written"]), str(d["decisions"]), str(d["learn_pilots"]),
               len(d["todo"]), d["done_in_fpvtrackside"], bool(d["overwritten"]), d["api"], d["api_error"],
               d["api_writes"],
               sum(len(p["findings"]) for p in d["pilots"].values()), sum(len(p["timing"]) for p in d["pilots"].values()),
               tuple((len(p["passes"]), len(p["laps"]), str(p.get("review"))) for p in d["pilots"].values()))
        if load_video:
            quads = {}
            for ch, p in d["pilots"].items():
                quads.setdefault(p["quad"]["video"], []).append({"rect": p["quad"]["rect"], "channel": ch,
                                                                 "pilot": p["pilot"], "kind": p["quad"]["kind"]})
            self.player.load(d["videos"], quads, {ch: [x["video_t"] for x in p["timing"]] for ch, p in d["pilots"].items()})
        if sig == self._detail_sig and not load_video:
            return
        self._detail_sig = sig
        self.heat_title.setText(f"Heat {d['number']} — {d['label']}   ·   {_clock(d['start'])}–{_clock(d['end'])}"
                                f"   ·   {d['status']}" + (f"   ·   updated {len(d['written'])}×" if d["written"] else "")
                                + (f"   ·   {d['done_in_fpvtrackside']} done in FPVTrackside"
                                   if d["done_in_fpvtrackside"] else ""))
        if d["overwritten"]:
            self.note.setText(f"⚠ FPVTrackside no longer has {d['overwritten']['changes']} of the laps written to it — "
                              "it saved this heat before the event was reopened. They are back on the list: make them in "
                              "FPVTrackside, or Update laps again and reopen the event in FPVTrackside.")
        if d["api_error"]:
            self.note.setText(f"⚠ FPVTrackside refused: {d['api_error']}")
        self._fill_todo(d)
        self.player.laps_at = {ch: [x["video_t"] for x in p["timing"]] for ch, p in d["pilots"].items()}
        lanes = []
        for ch, p in sorted(d["pilots"].items()):
            if self.day.event.has_timing:
                lanes.append({"name": ch, "marks": self._timing_lane(d, ch, p)})
            else:
                lanes.append({"name": ch, "marks": {
                    "video": p["video_g1"], "gate": [x["t"] for x in p["passes"] if x["gate"] != 1]}})
        self.player.colors = {ch: PILOT_COLORS[i % len(PILOT_COLORS)] for i, ch in enumerate(sorted(d["pilots"]))}
        self.player.timeline.set_data(self.player.duration, lanes)
        self._fill_tree(d)
        has_changes = bool(d["changes"])
        self.b_write.setEnabled(has_changes)
        self.b_write.setText(f"Update laps in FPVTrackside ({len(d['changes'])})" if has_changes
                             else "Update laps in FPVTrackside")
        self.b_restore.setEnabled(bool(d["written"]))
        self.b_write.setToolTip("Made in FPVTrackside through its REST API: shows there at once" if d["api"] else
                                "Written to the heat's Race.json: FPVTrackside shows it after the event is reopened")

    @staticmethod
    def _timing_lane(d: dict, ch: str, p: dict) -> dict:
        """A pilot's row in an FPVTrackside event: FPVTrackside's laps now and
        what the next update changes (upper), the video (lower)."""
        dec = lambda f: d["decisions"].get(f"{ch}|{f['key']}")
        removed = {f["detection_id"] for f in p["findings"] if f.get("detection_id") and dec(f) == "accepted"}
        removed |= {m["detection_id"] for m in d["marks"] if m["channel"] == ch and m["op"] == "remove"}
        added = [f.get("video_t") for f in p["findings"] if not f.get("detection_id") and dec(f) == "accepted"]
        added += [m.get("video_t") for m in d["marks"] if m["channel"] == ch and m["op"] == "add"]
        # to check: no start/finish pass in the video there (a timing lap it doesn't confirm, or a
        # crossing worked out from the gates around it); stays until accepted
        flags = [f.get("video_t") for f in p["findings"]
                 if dec(f) != "accepted" and (f.get("detection_id") or not f.get("exact"))]
        return {"timing": [x["video_t"] for x in p["timing"] if x["id"] not in removed],
                "remove": [x["video_t"] for x in p["timing"] if x["id"] in removed],
                "add": added, "video": p["video_g1"], "flag": flags}

    def _fill_tree(self, d):
        sel = self._selected()
        sel_key = (sel or {}).get("key")
        self.tree.blockSignals(True)
        self.tree.clear()
        for ch, p in sorted(d["pilots"].items()):
            run = p.get("run", {})
            n_t = max(0, len(p["timing"]) - 1)
            n_v = max(0, len(p["video_g1"]) - 1)
            open_n = sum(1 for f in p["findings"] if f"{ch}|{f['key']}" not in d["decisions"])
            if not self.day.event.has_timing:
                n_t = len(p.get("laps", []))
            head = QTreeWidgetItem([ch, "", "", f"{_who(ch, p['pilot'])}  —  {'timing' if self.day.event.has_timing else 'video'} "
                                    f"{n_t} laps" + (f" · video {n_v} laps" if self.day.event.has_timing else "")
                                    + (f" · ⚠ {open_n} to check" if open_n else "")
                                    + ((" · reviewed ✔" if p["review"].get("status") == "complete"
                                        else f" · review {p['review'].get('checked', 0)}/{p['review'].get('marks', 0)}")
                                       if p.get("review") else "")
                                    + (" · 📵 no video — no picture recorded for this pilot" if p.get("no_video") else "")
                                    + (f" · {p['no_video_findings']} not checked (no picture then)"
                                       if p.get("no_video_findings") else "")
                                    + ("" if run.get("status") in (None, "analysed") else f" · {run.get('status')}"),
                                    "", "", ""])
            head.setData(0, Qt.ItemDataRole.UserRole, {"channel": ch, "pilot_row": True})
            head.setFlags(head.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            head.setCheckState(0, Qt.CheckState.Checked if ch in d["learn_pilots"] else Qt.CheckState.Unchecked)
            head.setToolTip(0, "☑ = learn the track from this pilot's run")
            f0 = head.font(3)
            f0.setBold(True)
            head.setFont(3, f0)
            self.tree.addTopLevelItem(head)
            flagged = {f.get("detection_id"): f for f in p["findings"] if f.get("detection_id")}
            removed = {m.get("detection_id") for m in d["marks"] if m["op"] == "remove"}
            n_gates = d.get("n_gates") or 0
            no_pic = p.get("no_video")
            gray = p.get("gray") or []
            in_gray = (lambda vt: vt is not None and any(a - 1.0 <= vt <= b + 1.0 for a, b in gray))

            def gates(a, b):
                """A lap's gates; none when the lap ended with no picture, marked when part of it had none."""
                if no_pic or in_gray(b):
                    return None
                g = lap_gates(p["passes"], n_gates, a, b)
                part = a is not None and b is not None and any(x < b and y > a for x, y in gray)
                return (*g, True) if g and part else g
            confirmed = p.get("confirmed")
            prev = prev_vt = None
            rows = []
            for i, x in enumerate(p["timing"]):
                f = flagged.get(x["id"])
                lap = x["t"] - prev if prev is not None else None
                lap_g = None if x["holeshot"] else gates(prev_vt, x["video_t"])
                prev, prev_vt = x["t"], x["video_t"]
                what = "Holeshot" if x["holeshot"] else f"Lap {i}"
                what += "  (manual)" if x["type"] == "Manual" else ""
                if x["id"] in removed:
                    chk = "removed by hand"
                elif f:
                    dec = d["decisions"].get(f"{ch}|{f['key']}")
                    chk = (f"⚠ {f['reason']}" if not dec else f"{dec}: {f['reason']}")
                elif p.get("no_video"):
                    chk = "no video"
                elif in_gray(x["video_t"]):
                    chk = "📵 no picture then — not checked"
                elif p.get("checked") and (confirmed is None or x["id"] in confirmed):
                    chk = "✓ seen in the video"
                elif p.get("checked"):
                    chk = "not checked by the video"
                else:
                    chk = ""
                rows.append((x["video_t"], x["t"], what, lap, chk,
                             {"channel": ch, "detection_id": x["id"], "video_t": x["video_t"],
                              "key": f["key"] if f else None, "race_t": x["race_t"], "gates": lap_g}))
            # a lap that isn't a LapRF lap starts at the lap line before it, whichever kind: LapRF's,
            # one only the video saw, or one added by hand
            line_vts = sorted([x["video_t"] for x in p["timing"] if x["id"] not in removed]
                              + [f["video_t"] for f in p["findings"] if f.get("video_t") is not None
                                 and (f["kind"] == "missed" or (f["kind"] == "unsure" and not f.get("detection_id")))]
                              + [m["video_t"] for m in d["marks"] if m["channel"] == ch and m["op"] == "add"
                                 and m.get("video_t") is not None])

            def since_line(vt):
                before = [t for t in line_vts if t < vt - GATE_TOL_S]
                return gates(before[-1], vt) if before and vt is not None else None
            for f in p["findings"]:
                if f["kind"] == "missed" or (f["kind"] == "unsure" and not f.get("detection_id")):
                    dec = d["decisions"].get(f"{ch}|{f['key']}")
                    rows.append((f.get("video_t"), f["t"], "Lap in the video, not in timing", f.get("lap_time"),
                                 (f"⚠ {f['reason']}" if not dec else f"{dec}: {f['reason']}"),
                                 {"channel": ch, "key": f["key"], "video_t": f.get("video_t"),
                                  "race_t": f.get("race_t"),
                                  "gates": None if f.get("holeshot") else since_line(f.get("video_t"))}))
            if not self.day.event.has_timing:                 # Replays without timing: the video's laps
                for l in p.get("laps", []):
                    rows.append((l["t1"], l["t1"], f"Lap {l['lap']}", l["t1"] - l["t0"],
                                 "reviewed ✓" if p.get("reviewed") else "",
                                 {"channel": ch, "video_t": l["t1"], "gates": gates(l["t0"], l["t1"])}))
            for mi, m in enumerate(d["marks"]):
                if m["channel"] == ch and m["op"] == "add":
                    rows.append((m.get("video_t"), m["t"], "Lap added by hand", None, "will be written",
                                 {"channel": ch, "mark_index": mi, "video_t": m.get("video_t"),
                                  "race_t": m["t"] - d["start"], "gates": since_line(m.get("video_t"))}))
            for vt, ct, what, lap, chk, data in sorted(rows, key=lambda r: r[1]):
                timed = self.day.event.has_timing
                g = data.get("gates")
                it = QTreeWidgetItem([_mmss(vt), _clock(ct) if timed else "",
                                      _mmss(data.get("race_t")) if timed and data.get("race_t") is not None else "",
                                      what, f"{lap:.2f}s" if lap else "", f"{g[0]}/{g[1]}" if g else "", chk])
                it.setData(0, Qt.ItemDataRole.UserRole, data)
                if g:
                    part = len(g) > 3
                    it.setToolTip(5, ("All gates seen in this lap" if not g[2] else
                                      "Not seen in this lap: " + ", ".join(f"G{x}" for x in g[2]))
                                  + (" — no picture for part of this lap" if part else ""))
                    if g[2] and not part:
                        it.setForeground(5, QColor(255, 170, 60))
                    elif part:
                        it.setForeground(5, QColor(150, 150, 150))
                if chk.startswith("📵") or chk.startswith("not checked"):
                    for c in range(7):
                        it.setForeground(c, QColor(150, 150, 150))
                if chk.startswith("⚠"):
                    for c in range(7):
                        it.setForeground(c, QColor(255, 170, 60))
                head.addChild(it)
                if sel_key and data.get("key") == sel_key:
                    self.tree.setCurrentItem(it)
            head.setExpanded(True)
        for c in range(6):
            self.tree.resizeColumnToContents(c)
        self.tree.blockSignals(False)

    def _fill_todo(self, d: dict):
        steps = d["todo"] if self.day.event.has_timing and not d["api"] else []
        self.todo_box.setVisible(bool(steps))
        row = self.todo.currentRow()
        self.todo.clear()
        for st in steps:
            it = QListWidgetItem(step_text(st))
            it.setData(Qt.ItemDataRole.UserRole, st)
            it.setForeground(LANE_COLORS["remove"] if st["op"] == "remove" else LANE_COLORS["add"])
            self.todo.addItem(it)
        self.todo_title.setText(f"To do in FPVTrackside — {len(steps)} change{'s' if len(steps) != 1 else ''}"
                                " (by race time)")
        if 0 <= row < self.todo.count():
            self.todo.setCurrentRow(row)

    def _step_clicked(self, item):
        st = item.data(Qt.ItemDataRole.UserRole) or {}
        self.player.set_highlight(st.get("channel"))
        if st.get("video_t") is not None:
            self.player.stop()
            self.player.seek_frames_before(st["video_t"], JUMP_FRAMES)

    def _copy_step(self):
        it = self.todo.currentItem() or (self.todo.item(0) if self.todo.count() else None)
        if it:
            st = it.data(Qt.ItemDataRole.UserRole)
            QGuiApplication.clipboard().setText(_mmss(st["race_t"]))
            self.note.setText(f"Copied {_mmss(st['race_t'])} ({_who(st['channel'], st['pilot'])}).")

    def _pilot_checked(self, item, col):
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        if col == 0 and data.get("pilot_row") and self.heat_id:
            self.day.set_pilot_checked(self.heat_id, data["channel"], item.checkState(0) == Qt.CheckState.Checked)
            self._detail_sig = None

    def _selected(self) -> Optional[dict]:
        it = self.tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None

    def _jump(self, item, _col=0):
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        if data.get("video_t") is not None:
            self.player.stop()                       # 6 frames before the pass, to watch it happen
            self.player.seek_frames_before(data["video_t"], JUMP_FRAMES)

    # ── actions ────────────────────────────────────────────────

    def _tree_selected(self, item, _prev=None):
        """The selected line's pilot: their feed stands out in the video and the timeline."""
        data = (item.data(0, Qt.ItemDataRole.UserRole) or {}) if item else {}
        self.player.set_highlight(data.get("channel"))

    def _decide(self, decision):
        s = self._selected()
        if not s or not s.get("key"):
            return QMessageBox.information(self, "Check", "Select a ⚠ line first.")
        self.day.decide(self.heat_id, s["channel"], s["key"], decision)
        self._detail_sig = None
        self._next_finding(s["channel"], s["key"])

    def _next_finding(self, ch: str, key: str):
        """After Accept / Reject: the next undecided finding — the rest of this
        pilot's, then the next pilot's (as the table lists them), then any
        skipped earlier — selected, and the video 6 frames before it."""
        d = self.day.heat_detail(self.heat_id)
        if not d:
            return
        order = [(c, f) for c in sorted(d["pilots"]) for f in sorted(d["pilots"][c]["findings"], key=lambda f: f["t"])]
        here = next((i for i, (c, f) in enumerate(order) if c == ch and f["key"] == key), -1)
        todo = [(c, f) for c, f in order[here + 1:] + order[:max(here, 0)] if f"{c}|{f['key']}" not in d["decisions"]]
        self._show_heat()                            # the table with the decision in it
        if not todo:
            self.note.setText("All findings in this heat are decided.")
            return
        c, f = todo[0]
        for i in range(self.tree.topLevelItemCount()):
            head = self.tree.topLevelItem(i)
            for k in range(head.childCount()):
                it = head.child(k)
                data = it.data(0, Qt.ItemDataRole.UserRole) or {}
                if data.get("channel") == c and data.get("key") == f["key"]:
                    self.tree.setCurrentItem(it)
                    self.tree.scrollToItem(it)
                    self._jump(it)
                    left = len(todo)
                    self.note.setText(f"Next: {c} at {_mmss(f.get('video_t'))} — {left} finding{'s' if left > 1 else ''} "
                                      "left to decide in this heat.")
                    return

    def _add(self):
        s = self._selected()
        if not s:
            return QMessageBox.information(self, "Add lap", "Select the pilot (or one of their laps) first.")
        self.day.add_lap(self.heat_id, s["channel"], self.player.t)
        self._detail_sig = None

    def _remove(self):
        s = self._selected()
        if not s or not (s.get("detection_id") or "mark_index" in s):
            return QMessageBox.information(self, "Remove lap", "Select a timing lap (or a lap you added) first.")
        self.day.remove_lap(self.heat_id, s["channel"], detection_id=s.get("detection_id"),
                            mark_index=s.get("mark_index", -1))
        self._detail_sig = None

    def _write(self):
        d = self.day.heat_detail(self.heat_id)
        if not d or not d["changes"]:
            return
        lines = []
        for c in d["changes"]:
            name = d["pilots"].get(c["channel"], {}).get("pilot", "")
            if c["op"] == "add":
                lines.append(f"+ add a lap for {c['channel']} {name} at {_clock(c['t'])}")
            else:
                t = next((x["t"] for x in d["pilots"].get(c["channel"], {}).get("timing", [])
                          if x["id"] == c["detection_id"]), None)
                lines.append(f"− remove {c['channel']} {name}'s lap at {_clock(t) if t else '?'}")
        if QMessageBox.question(self, "Update laps in FPVTrackside",
                                f"Heat {d['number']} — {d['label']}\n\n" + "\n".join(lines) +
                                ("\n\nMade in FPVTrackside through its REST API, as a marshal's correction: "
                                 "it shows there at once. Restore original laps undoes it." if d["api"] else
                                 "\n\nWritten like a marshal's correction. The original Race.json is kept "
                                 "(Restore original laps)."
                                 "\n\nFPVTrackside shows these laps after the event is reopened. Don't edit this "
                                 "heat in FPVTrackside before then, or the change is lost.")) \
                == QMessageBox.StandardButton.Yes:
            self.day.write(self.heat_id)
            self._detail_sig = None

    def _restore(self):
        d = self.detail if self.heat_id else None
        how = (" (through FPVTrackside's REST API — it undoes the changes made through it)" if d and d.get("api_writes")
               else "")
        if QMessageBox.question(self, "Restore", f"Put this heat's original laps back in FPVTrackside?{how}") \
                == QMessageBox.StandardButton.Yes:
            self.day.restore(self.heat_id)
            self._detail_sig = None

    def _review(self):
        """Review the selected pilot's run here: their quadrant enlarged, the
        marks on the timeline, the review panel in place of the table."""
        from dataset_paths import TrackPaths
        from race_day.review import Review
        s = self._selected()
        d = self.day.heat_detail(self.heat_id) if self.heat_id else None
        if not s or not d or s.get("channel") not in d["pilots"]:
            return QMessageBox.information(self, "Review", "Select a pilot (or one of their laps) first.")
        ch = s["channel"]
        p = d["pilots"][ch]
        if not Path(p["race_data"]).exists():
            return QMessageBox.information(self, "Review", "Analyse this heat first.")
        tp = TrackPaths(Path(p["track_dir"])) if p.get("track_dir") else self.day.tp   # the pilot's layout
        if not tp.gate_memory.exists():
            return QMessageBox.information(self, "Review", "Learn the track first"
                                           + (f" ({p.get('layout')} layout)." if p.get("layout") not in (None, "main") else "."))
        q = p["quad"]
        rect = tuple(q["rect"])
        full = rect == (0.0, 0.0, 1.0, 1.0)
        try:
            r = Review(tp, p["stem"], q["video"], None if full else rect, track_name=self.day.event.name)
        except Exception as e:
            return QMessageBox.critical(self, "Review", f"Can't open the review: {e}")
        if self.player.path != q["video"]:
            idx = self.player.which.findData(q["video"])
            if idx >= 0:
                self.player.which.setCurrentIndex(idx)
        self.tree.hide()
        self.review_panel.show()
        self._review_heat = self.heat_id
        self.review_panel.start(r, ch, p["pilot"], None if full else rect)
        self.review_panel.setFocus()

    def _review_apply(self):
        self._review_timer.stop()
        if self.day and self._review_heat:
            self.day.review_changed(self._review_heat)
            self._detail_sig = None

    def _review_end(self):
        if getattr(self, "review_panel", None) is None or self.review_panel.review is None:
            return
        if self._review_timer.isActive():            # changes not applied yet
            self._review_apply()
        self.review_panel.stop()
        self.review_panel.hide()
        self.tree.show()
        self._detail_sig = None

    def _configure_video(self, name: str):
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle(f"What is {name}?")
        f = QFormLayout(dlg)
        kind = QComboBox()
        kind.addItems(["Single pilot", "2×2 (four pilots)"])
        heat = QSpinBox()
        heat.setRange(1, 999)
        heat.setValue(1 + max([h["number"] for h in self.day.heat_list()] + [0]))
        f.addRow("Video", QLabel(name))
        f.addRow("Kind", kind)
        f.addRow("Heat number", heat)
        single = QLineEdit()
        f.addRow("Pilot (optional)", single)
        quads = []
        for pos, ch in zip(["top left", "top right", "bottom left", "bottom right"], ["R1", "R3", "R6", "R8"]):
            c, n = QLineEdit(ch), QLineEdit()
            c.setMaximumWidth(60)
            row = QHBoxLayout()
            row.addWidget(c)
            row.addWidget(n)
            f.addRow(f"{pos}: channel, pilot (optional)", row)
            quads.append((c, n, row))

        def show(i):
            for c, n, _ in quads:
                c.setEnabled(i == 1)
                n.setEnabled(i == 1)
            single.setEnabled(i == 0)
        kind.currentIndexChanged.connect(show)
        show(0)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if kind.currentIndex() == 0:
            info = {"kind": "single", "heat": heat.value(), "pilots": {"P1": single.text().strip() or "P1"}}
        else:
            chans = [c.text().strip() or f"Q{i + 1}" for i, (c, _, _) in enumerate(quads)]
            info = {"kind": "2x2", "heat": heat.value(), "channels": chans,       # no name: the channel is the name
                    "pilots": {ch: n.text().strip() or ch for ch, (_, n, _) in zip(chans, quads)}}
        self.day.configure_video(name, info)

    def _learn(self):
        runs = self.day.learn_runs()
        if not runs:
            return QMessageBox.information(self, "Learn", "Tick ☑ the pilot runs to learn the track from first — "
                                           "in the heat list (all pilots of a heat) or per pilot in the heat view.\n\n"
                                           "About 5 runs with at least one full lap each are enough.")
        heats = {r["id"]: r for r in self.day.heat_list()}
        timing = self.day.event.has_timing
        lines = [f"heat {heats[h]['number']} · {_who(ch, heats[h]['pilots'].get(ch, '?'))}"
                 + (f" — {heats[h]['timing_laps'].get(ch, '?')} laps" if timing else "") for h, ch in runs if h in heats]
        few = [l for l, (h, ch) in zip(lines, runs)
               if timing and (heats.get(h, {}).get("timing_laps", {}).get(ch) or 0) < 1]
        by_layout: Dict[str, int] = {}
        for h, ch in runs:
            hh = self.day.event.heats.get(h)
            if hh:
                lay = self.day.layout_of(hh, ch)
                by_layout[lay] = by_layout.get(lay, 0) + 1
        others = {lay: k for lay, k in by_layout.items() if lay != "main"}
        if others:                                           # layouts of their own: each learned from its runs
            lines.append("")
            lines += [f"{k} of them are on the {lay} layout ({self.day._n_gates(lay)} gates, Event setup): "
                      f"learned on their own" for lay, k in others.items()]
        if "main" not in by_layout:
            if QMessageBox.question(self, "Learn the track", f"Learning from {len(runs)} pilot runs:\n  "
                                    + "\n  ".join(lines) + "\n\nLearn?") == QMessageBox.StandardButton.Yes:
                self.day.learn(int(self.day.state.get("n_gates") or 10))
            return
        n, ok = QInputDialog.getInt(self, "Learn the track",
                                    f"Learning from {len(runs)} pilot runs:\n  " + "\n  ".join(lines) +
                                    (f"\n\n⚠ {len(few)} of them have no full lap — they add little." if few else "") +
                                    "\n\nNumber of gates on the track" + (" (main layout)" if others else "") + ":",
                                    int(self.day._n_gates("main") or 10), 3, 40)
        if ok:
            self.day.learn(n)

    def _setup(self):
        from race_day.setup_ui import SetupDialog
        SetupDialog(self, self.day).exec()
        self.day._lay_at = 0
        if self.heat_id:
            self._show_heat()
