#!/usr/bin/env python3
"""
Race Review UI — race_ui.py

No command-line arguments needed.  Open everything from inside the UI.

Workflow:
  1. python race_ui.py
  2. 📁 Open Track   → pick <data_root>/<track>; its gate_memory.json loads
  3. 📂 Open Video   → a video from the track (saved results load if present)
  4. ▶ Run Race Analysis → extraction runs in background and saves to
     <track>/runs/<video>.race_data.json (see dataset_paths.py)
  5. Scrub through the video to review detections, matches, and lap splits

🧠 Open Gate Memory can still load a different memory for experiments.

Timeline legend:
  Green tick  = gate pass matched (RACE)
  Red tick    = gate detected but no match (NOMATCH)
  Orange tick = duplicate gate within a lap (DUP)
  Yellow line = lap boundary

Keyboard shortcuts:
  Space            Play / Pause
  Up / Down        ±1 frame
  Left / Right     ±1 second
  Shift+Left/Right ±10 seconds
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dataset_paths import VIDEO_FILTER, TrackPaths, data_root, track_of

import cv2
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
        QSizePolicy, QMessageBox, QToolBar, QStatusBar, QFrame,
        QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
        QScrollArea, QSplitter, QDoubleSpinBox, QCheckBox,
    )
    from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, pyqtSignal, QProcess
    from PyQt6.QtGui import (
        QImage, QPixmap, QPainter, QColor, QPen, QFont, QBrush,
        QAction, QKeySequence, QShortcut, QPalette,
    )
except ImportError:
    sys.exit("PyQt6 is required.  Install with:  pip install PyQt6")


# ──────────────────────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────────────────────

_SOURCE_COLORS = {
    "RACE":    QColor( 60, 220, 100),
    "NOMATCH": QColor(220,  60,  60),
    "DUP":     QColor(220, 140,  40),
    "EXTRA":   QColor( 90, 160, 255),    # tracked pass point that isn't one of the gates (lap model)
    "NOPASS":  QColor(110, 110, 110),
}
_LAP_COLOR      = QColor(255, 200,   0)
_PLAYHEAD_COLOR = QColor(255,  60,  60)
_BG_COLOR       = QColor( 24,  24,  24)
_TRACK_COLOR    = QColor( 55,  55,  55)
_FILL_COLOR     = QColor( 80,  80,  80)

_STAGE_CV2 = {
    "idle":    (128, 128, 128),
    "aligned": (  0, 220, 255),
    "passed":  (  0, 255,   0),
}
_TYPE_CV2 = {
    "square":   (255,  80,  80),
    "arch":     (  0, 220, 220),
    "circle":   ( 80, 220,  80),
    "flagpole": (220,  80, 220),
    "NONE":     (110, 110, 110),
}


def _src_color(source: str) -> QColor:
    return _SOURCE_COLORS.get(source.upper(), QColor(160, 160, 160))


def _auto_clip_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ──────────────────────────────────────────────────────────────
# Timeline widget
# ──────────────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    seeked = pyqtSignal(float)

    _PAD = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

        self._duration: float = 0.0
        self._current_t: float = 0.0
        self._passes: List[dict] = []
        self._laps: List[dict] = []
        self._hover_t: Optional[float] = None

    def set_duration(self, d: float):
        self._duration = float(d)
        self.update()

    def set_current_t(self, t: float):
        self._current_t = float(t)
        self.update()

    def set_passes(self, passes: List[dict]):
        self._passes = passes
        self.update()

    def set_laps(self, laps: List[dict]):
        self._laps = laps
        self.update()

    def _t_to_x(self, t: float) -> int:
        if self._duration <= 0:
            return self._PAD
        return int(self._PAD + (t / self._duration) * (self.width() - 2 * self._PAD))

    def _x_to_t(self, x: int) -> float:
        if self._duration <= 0:
            return 0.0
        w = self.width() - 2 * self._PAD
        return max(0.0, min(self._duration, (x - self._PAD) / max(1, w) * self._duration))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        mid = H // 2

        p.fillRect(0, 0, W, H, _BG_COLOR)

        if self._duration <= 0:
            p.setPen(QColor(100, 100, 100))
            p.drawText(QRect(0, 0, W, H), Qt.AlignmentFlag.AlignCenter, "No data loaded")
            p.end()
            return

        # Track bar + progress fill
        bar_h = 6
        p.fillRect(self._PAD, mid - bar_h // 2, W - 2 * self._PAD, bar_h, _TRACK_COLOR)
        px = self._t_to_x(self._current_t)
        if px > self._PAD:
            p.fillRect(self._PAD, mid - bar_h // 2, px - self._PAD, bar_h, _FILL_COLOR)

        # Lap markers
        for lap in self._laps:
            lx = self._t_to_x(float(lap.get("t0", lap.get("t", 0.0))))
            p.setPen(QPen(_LAP_COLOR, 2))
            p.drawLine(lx, mid - 24, lx, mid + 24)
            p.setFont(QFont("Arial", 7))
            p.setPen(_LAP_COLOR)
            p.drawText(lx - 6, mid - 26, f"L{int(lap.get('lap', lap.get('lap_no', 0)))}")

        # Pass ticks
        for ps in self._passes:
            cx = self._t_to_x(float(ps.get("t", 0.0)))
            col = QColor(_src_color(str(ps.get("source", "NOMATCH"))))
            col.setAlpha(210)
            p.setPen(QPen(col, 2))
            p.drawLine(cx, mid - 13, cx, mid + 13)

        # Playhead
        ph = self._t_to_x(self._current_t)
        p.setPen(QPen(_PLAYHEAD_COLOR, 2))
        p.drawLine(ph, 2, ph, H - 2)

        # Hover time tooltip
        if self._hover_t is not None:
            hx = self._t_to_x(self._hover_t)
            p.setFont(QFont("Arial", 8))
            p.setPen(QColor(200, 200, 200))
            p.drawText(max(4, min(W - 44, hx - 14)), H - 5, f"{self._hover_t:.2f}s")

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.seeked.emit(self._x_to_t(event.pos().x()))

    def mouseMoveEvent(self, event):
        self._hover_t = self._x_to_t(event.pos().x())
        self.update()

    def leaveEvent(self, _event):
        self._hover_t = None
        self.update()


# ──────────────────────────────────────────────────────────────
# Video player (cv2-based, with overlay callback)
# ──────────────────────────────────────────────────────────────

class VideoPlayer(QWidget):
    position_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._duration: float = 0.0
        self._current_t: float = 0.0
        self._playing: bool = False
        self._last_frame: Optional[np.ndarray] = None
        self._frame_processor = None   # fn(frame_bgr, t) -> frame_bgr

        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: black;")
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lv.addWidget(self._label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_frame_processor(self, fn):
        self._frame_processor = fn

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def current_t(self) -> float:
        return self._current_t

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_playing(self) -> bool:
        return self._playing

    def load(self, path: str) -> bool:
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = total / max(self._fps, 1.0)
        self._current_t = 0.0
        self._render_next()
        return True

    def play(self):
        if not self._cap:
            return
        self._playing = True
        self._timer.start(max(1, int(1000.0 / self._fps)))

    def pause(self):
        self._playing = False
        self._timer.stop()

    def toggle(self):
        self.pause() if self._playing else self.play()

    def seek(self, t: float):
        if not self._cap:
            return
        t = max(0.0, min(self._duration, t))
        self._cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        self._render_next()
        self.position_changed.emit(self._current_t)

    def seek_frames(self, delta: int):
        if not self._cap:
            return
        self.seek(self._current_t + delta / max(self._fps, 1.0))

    def _tick(self):
        if not self._cap:
            return
        self._render_next()
        self.position_changed.emit(self._current_t)
        if self._current_t >= self._duration - 0.05:
            self.pause()

    def _render_next(self):
        if not self._cap:
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        self._current_t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._last_frame = frame.copy()
        display = frame.copy()
        if self._frame_processor:
            display = self._frame_processor(display, self._current_t)
        self._push(display)

    def _push(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self._label.width() or 640, self._label.height() or 360,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_frame is not None:
            display = self._last_frame.copy()
            if self._frame_processor:
                display = self._frame_processor(display, self._current_t)
            self._push(display)


# ──────────────────────────────────────────────────────────────
# Match Candidates detail window
# ──────────────────────────────────────────────────────────────

class MatchCandidatesWindow(QWidget):
    """Standalone resizable window — shows all candidate images with similarity scores."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Match Candidates — Detail View")
        self.resize(1100, 700)
        self.setStyleSheet("background:#0e0e0e;")

        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setStyleSheet("background:#0e0e0e; color:#555; font-size:13px;")
        lv.addWidget(self._canvas)

        self._passes: List[dict] = []
        self._gate_mem: List[dict] = []
        self._race_lookahead: int = 3

    def update_data(self, passes: List[dict], gate_mem: List[dict], race_lookahead: int):
        self._passes = passes
        self._gate_mem = gate_mem
        self._race_lookahead = race_lookahead
        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def _render(self):
        if not self._passes or not self._gate_mem:
            self._canvas.setPixmap(QPixmap())
            self._canvas.setText("No pass at current frame")
            return

        self._canvas.setText("")

        ps          = self._passes[0]
        matched_gid = int(ps.get("gate_id", -1))
        source      = str(ps.get("source", "NOMATCH"))
        top_sim     = float(ps.get("sim", 0.0))
        exp_before  = int(ps.get("exp_before", -1))
        query_img   = str(ps.get("query_img", ""))

        # Query embedding for per-image cosine similarity (absent in old JSONs)
        q_vec: Optional[np.ndarray] = None
        raw_qe = ps.get("query_embedding")
        if raw_qe:
            v = np.array(raw_qe, dtype=np.float32)
            n = np.linalg.norm(v)
            q_vec = v / n if n > 1e-9 else None

        def _clip_sim(gate: dict, clip_idx: int) -> Optional[float]:
            if q_vec is None:
                return None
            embeds = gate.get("embeds", [])
            if clip_idx >= len(embeds):
                return None
            e = np.array(embeds[clip_idx], dtype=np.float32)
            n = np.linalg.norm(e)
            if n < 1e-9:
                return None
            return float(np.dot(q_vec, e / n))

        n = len(self._gate_mem)
        if n > 0:
            if exp_before >= 0:
                exp = exp_before % n
            else:
                gid_to_pos  = {int(g["gate_id"]): i for i, g in enumerate(self._gate_mem)}
                exp = gid_to_pos.get(matched_gid, 0)
            idxs   = list(dict.fromkeys([exp, (exp + 1) % n, 0]))
            window = [self._gate_mem[i] for i in idxs]
        else:
            window = []

        if not window:
            self._canvas.setPixmap(QPixmap())
            self._canvas.setText("No gates in comparison window")
            return

        gate_imgs: List[List[str]] = []
        for g in window:
            imgs = [p for p in g.get("embed_imgs", []) if p]
            if not imgs:
                li = g.get("last_img", "")
                imgs = [li] if li else []
            gate_imgs.append(imgs)

        max_rows = max((len(x) for x in gate_imgs), default=1)
        n_cols   = 1 + len(window)

        W = max(self.width()  - 2, 300)
        H = max(self.height() - 2, 200)

        header_h = 36
        sim_h    = 20
        pad      = 4

        col_w = max(60, (W - pad * (n_cols + 1)) // n_cols)
        row_h = max(40, (H - header_h - sim_h - pad * (max_rows + 1)) // max_rows)

        canvas_w = n_cols * (col_w + pad) + pad
        canvas_h = header_h + sim_h + max_rows * (row_h + pad) + pad
        canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (14, 14, 14)

        def _load_fit(path: str, w: int, h: int) -> np.ndarray:
            """Load image preserving aspect ratio, letterboxed into w×h cell."""
            cell = np.full((h, w, 3), 25, dtype=np.uint8)
            if path and Path(path).exists():
                raw = cv2.imread(path)
                if raw is not None and raw.shape[0] > 0 and raw.shape[1] > 0:
                    ih, iw = raw.shape[:2]
                    scale = min(w / iw, h / ih)
                    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
                    resized = cv2.resize(raw, (nw, nh))
                    ox = (w - nw) // 2
                    oy = (h - nh) // 2
                    cell[oy:oy + nh, ox:ox + nw] = resized
            return cell

        fs = max(0.38, col_w / 350.0)
        ft = max(1, round(fs * 1.5))

        # ── Query column (full height, aspect-ratio preserved) ─
        x0     = pad
        full_h = max_rows * (row_h + pad) - pad
        cv2.putText(canvas, "Query frame", (x0 + 4, header_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), ft, cv2.LINE_AA)
        canvas[header_h + sim_h: header_h + sim_h + full_h, x0: x0 + col_w] = \
            _load_fit(query_img, col_w, full_h)
        cv2.rectangle(canvas,
                      (x0, header_h + sim_h),
                      (x0 + col_w - 1, header_h + sim_h + full_h - 1),
                      (90, 90, 90), 1)

        # ── Gate columns ──────────────────────────────────────
        for ci, (g, imgs) in enumerate(zip(window, gate_imgs)):
            gid      = int(g["gate_id"])
            is_match = (gid == matched_gid)
            x0       = pad + (ci + 1) * (col_w + pad)

            # Best per-clip sim for the gate header
            clip_sims = [_clip_sim(g, ei) for ei in range(len(imgs))]
            valid_sims = [s for s in clip_sims if s is not None]
            best_sim   = max(valid_sims) if valid_sims else None

            if is_match and source == "RACE":
                border_col = (60, 220, 100)
                if best_sim is not None:
                    hdr_sim = f"best {best_sim * 100:.1f}%  MATCH"
                else:
                    hdr_sim = f"{top_sim * 100:.1f}%  MATCH"
                hdr_col = (60, 220, 100)
            elif is_match:
                border_col = (60, 60, 220)
                if best_sim is not None:
                    hdr_sim = f"best {best_sim * 100:.1f}%  NO-MATCH"
                else:
                    hdr_sim = f"{top_sim * 100:.1f}%  NO-MATCH"
                hdr_col = (100, 100, 220)
            else:
                border_col = (70, 70, 70)
                if best_sim is not None:
                    hdr_sim = f"best {best_sim * 100:.1f}%"
                else:
                    hdr_sim = ""
                hdr_col = (140, 140, 140)

            lbl = f"G{gid}  ({len(imgs)} clip{'s' if len(imgs) != 1 else ''})"
            cv2.putText(canvas, lbl, (x0 + 4, header_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, border_col, ft, cv2.LINE_AA)
            if hdr_sim:
                cv2.putText(canvas, hdr_sim, (x0 + 4, header_h + sim_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.3, fs * 0.8),
                            hdr_col, 1, cv2.LINE_AA)

            for ei, img_path in enumerate(imgs):
                y0  = header_h + sim_h + ei * (row_h + pad)
                img = _load_fit(img_path, col_w, row_h)

                # Per-image similarity overlay at bottom of cell
                s = clip_sims[ei] if ei < len(clip_sims) else None
                if s is not None:
                    bar_h   = max(18, int(row_h * 0.12))
                    overlay = img[row_h - bar_h:, :]
                    overlay[:] = (overlay * 0.35).astype(np.uint8)
                    pct_txt = f"{s * 100:.1f}%"
                    fs_bar  = max(0.28, col_w / 500.0)
                    cv2.putText(img, pct_txt,
                                (4, row_h - bar_h // 2 + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, fs_bar,
                                (255, 255, 255), 1, cv2.LINE_AA)

                canvas[y0: y0 + row_h, x0: x0 + col_w] = img

            col_h = len(imgs) * (row_h + pad) - pad if imgs else full_h
            cv2.rectangle(canvas,
                          (x0, header_h + sim_h),
                          (x0 + col_w - 1, header_h + sim_h + col_h - 1),
                          border_col, 3 if is_match else 1)

        rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi   = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._canvas.setPixmap(QPixmap.fromImage(qi))


# ──────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPV Race Review UI")
        self.resize(1540, 900)

        self._video_path: Optional[str] = None
        self._track: Optional[TrackPaths] = None
        self._gate_memory_path: Optional[str] = None
        self._det_model_path: Optional[str] = self._find_model_auto()
        self._clip_device: str = _auto_clip_device()
        self._sim_thresh, self._min_match_margin, self._g1_sim_thresh, self._g1_margin = self._load_gatedb_defaults()
        self._require_same_type: bool = False

        # Race data
        self._race_data: Optional[dict] = None
        self._frames_by_t: List[Tuple[float, dict]] = []
        self._all_passes: List[dict] = []
        self._all_laps: List[dict] = []
        self._gate_mem: List[dict] = []   # ordered gates from gate_memory.json
        self._race_lookahead: int = 3

        # Detail popup
        self._match_cands_win: Optional[MatchCandidatesWindow] = None

        # Background process
        self._proc: Optional[QProcess] = None
        self._proc_json_path: str = ""

        self._build_ui()
        self._build_shortcuts()

    def _find_model_auto(self) -> Optional[str]:
        # the detector the pass scorer was trained with, else the original model
        for name in ("cyn_current-20260715_best.pt", "current_best_non_vocab.pt"):
            default = Path(__file__).parent / name
            if default.exists():
                return str(default)
        return None

    @staticmethod
    def _load_gatedb_defaults() -> tuple:
        defaults_path = Path(__file__).parent / "debug_ui_defaults.json"
        try:
            with open(defaults_path, encoding="utf-8") as f:
                d = json.load(f)
            gdb = d.get("gatedb", {})
            sim   = float(gdb.get("sim_thresh", 0.88))
            marg  = float(gdb.get("min_match_margin", 0.03))
            g1sim = float(gdb.get("g1_sim_thresh", sim))
            g1marg= float(gdb.get("g1_margin", marg))
            return sim, marg, g1sim, g1marg
        except Exception:
            return 0.88, 0.03, 0.88, 0.03

    def _ensure_model(self) -> bool:
        if self._det_model_path and Path(self._det_model_path).exists():
            return True
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "PyTorch Models (*.pt)")
        if path:
            self._det_model_path = path
            return True
        return False

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        open_track = QAction("📁 Open Track", self)
        open_track.setToolTip("Pick a track folder — its gate memory loads and results are saved there")
        open_track.triggered.connect(self._on_open_track)
        tb.addAction(open_track)
        self._track_label = QLabel("  no track  ")
        self._track_label.setStyleSheet("color:#ffa028; font-weight:bold; font-size:12px;")
        tb.addWidget(self._track_label)

        open_vid = QAction("📂 Open Video", self)
        open_vid.triggered.connect(self._on_open_video)
        tb.addAction(open_vid)

        # right after Open Video, so it never falls off the end of a narrow toolbar
        run_act = QAction("▶  Run Race Analysis", self)
        run_act.triggered.connect(self._on_run)
        tb.addAction(run_act)
        tb.addSeparator()

        open_mem = QAction("🧠 Open Gate Memory", self)
        open_mem.triggered.connect(self._on_open_memory)
        tb.addAction(open_mem)

        set_model = QAction("🔧 Set Model", self)
        set_model.triggered.connect(self._on_set_model)
        tb.addAction(set_model)

        self._model_label = QLabel()
        self._model_label.setStyleSheet("color:#ffa028; font-size:11px; padding: 0 4px;")
        self._update_model_label()
        tb.addWidget(self._model_label)

        tb.addSeparator()

        tb.addWidget(QLabel("  Sim ≥ "))
        self._sim_spin = QDoubleSpinBox()
        self._sim_spin.setRange(50.0, 99.9)
        self._sim_spin.setSingleStep(0.1)
        self._sim_spin.setDecimals(1)
        self._sim_spin.setSuffix("%")
        self._sim_spin.setValue(self._sim_thresh * 100)
        self._sim_spin.setFixedWidth(80)
        self._sim_spin.setToolTip("Minimum cosine similarity for a gate to count as MATCH")
        self._sim_spin.valueChanged.connect(lambda v: setattr(self, "_sim_thresh", v / 100.0))
        tb.addWidget(self._sim_spin)

        tb.addWidget(QLabel("  Margin ≥ "))
        self._margin_spin = QDoubleSpinBox()
        self._margin_spin.setRange(0.0, 20.0)
        self._margin_spin.setSingleStep(0.1)
        self._margin_spin.setDecimals(1)
        self._margin_spin.setSuffix("%")
        self._margin_spin.setValue(self._min_match_margin * 100)
        self._margin_spin.setFixedWidth(80)
        self._margin_spin.setToolTip("Minimum gap between best and second-best gate similarity")
        self._margin_spin.valueChanged.connect(lambda v: setattr(self, "_min_match_margin", v / 100.0))
        tb.addWidget(self._margin_spin)

        tb.addWidget(QLabel("  G1 Sim ≥ "))
        self._g1_sim_spin = QDoubleSpinBox()
        self._g1_sim_spin.setRange(50.0, 99.9)
        self._g1_sim_spin.setSingleStep(0.1)
        self._g1_sim_spin.setDecimals(1)
        self._g1_sim_spin.setSuffix("%")
        self._g1_sim_spin.setValue(self._g1_sim_thresh * 100)
        self._g1_sim_spin.setFixedWidth(80)
        self._g1_sim_spin.setToolTip("Minimum cosine similarity for G1 (start/finish gate) to count as MATCH")
        self._g1_sim_spin.valueChanged.connect(lambda v: setattr(self, "_g1_sim_thresh", v / 100.0))
        tb.addWidget(self._g1_sim_spin)

        tb.addWidget(QLabel("  G1 Margin ≥ "))
        self._g1_margin_spin = QDoubleSpinBox()
        self._g1_margin_spin.setRange(0.0, 20.0)
        self._g1_margin_spin.setSingleStep(0.1)
        self._g1_margin_spin.setDecimals(1)
        self._g1_margin_spin.setSuffix("%")
        self._g1_margin_spin.setValue(self._g1_margin * 100)
        self._g1_margin_spin.setFixedWidth(80)
        self._g1_margin_spin.setToolTip("Minimum margin for G1 (start/finish gate) — raise to prevent false lap resets")
        self._g1_margin_spin.valueChanged.connect(lambda v: setattr(self, "_g1_margin", v / 100.0))
        tb.addWidget(self._g1_margin_spin)

        self._same_type_chk = QCheckBox("Match type")
        self._same_type_chk.setChecked(self._require_same_type)
        self._same_type_chk.setToolTip(
            "Only match a detected gate against memory slots of the same type.\n"
            "Enable if you have mixed gate types (squares, flagpoles, etc.) in the same course."
        )
        self._same_type_chk.stateChanged.connect(
            lambda s: setattr(self, "_require_same_type", bool(s))
        )
        tb.addWidget(self._same_type_chk)

        tb.addSeparator()
        self._status_badge = QLabel("  Open a video and gate memory to begin")
        self._status_badge.setStyleSheet("color:#aaa; font-size:12px;")
        tb.addWidget(self._status_badge)

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # Left column
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)

        self._video = VideoPlayer()
        self._video.set_frame_processor(self._draw_overlay)
        self._video.position_changed.connect(self._on_position)
        lv.addWidget(self._video, stretch=1)

        self._time_label = QLabel("0.00 s  /  0.00 s")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._time_label.setStyleSheet("color:#999; font-size:11px;")
        lv.addWidget(self._time_label)

        self._timeline = TimelineWidget()
        self._timeline.seeked.connect(self._on_seek)
        lv.addWidget(self._timeline)

        lv.addWidget(self._build_legend())

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar{background:#222;border:1px solid #444;border-radius:3px;}"
            "QProgressBar::chunk{background:#1a4a7a;}"
        )
        lv.addWidget(self._progress_bar)

        lv.addLayout(self._build_playback_row())
        lv.addLayout(self._build_seek_row())

        root.addWidget(left, stretch=3)

        # Right column
        right = QWidget()
        right.setFixedWidth(360)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.setSpacing(4)

        rv.addWidget(self._lbl("Laps", bold=True, size=13))

        self._lap_list = QListWidget()
        self._lap_list.setMaximumHeight(160)
        self._lap_list.setStyleSheet(
            "QListWidget{background:#1e1e1e;border:1px solid #444;}"
            "QListWidget::item:selected{background:#1a4a2a;}"
        )
        self._lap_list.currentRowChanged.connect(self._on_lap_selected)
        rv.addWidget(self._lap_list)

        rv.addWidget(self._lbl("Lap Splits", bold=True, size=11))

        self._splits_table = QTableWidget(0, 4)
        self._splits_table.setHorizontalHeaderLabels(["Gate", "Type", "From Start", "Δ Prev"])
        self._splits_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._splits_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._splits_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._splits_table.verticalHeader().setVisible(False)
        self._splits_table.setStyleSheet(
            "QTableWidget{background:#1a1a1a;color:#ddd;gridline-color:#333;}"
            "QHeaderView::section{background:#2a2a2a;color:#aaa;border:none;}"
        )
        rv.addWidget(self._splits_table, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        rv.addWidget(sep)

        info_splitter = QSplitter(Qt.Orientation.Vertical)
        info_splitter.setStyleSheet("QSplitter::handle{background:#555; height:5px;}")

        # top half: frame info
        fi_w = QWidget()
        fi_v = QVBoxLayout(fi_w)
        fi_v.setContentsMargins(0, 0, 0, 0)
        fi_v.setSpacing(2)
        fi_v.addWidget(self._lbl("Frame Info", bold=True, size=11))
        self._frame_info = QLabel("Scrub the video to see per-frame detection data.")
        self._frame_info.setStyleSheet(
            "color:#aaa; font-size:10px; background:#1a1a1a; padding:5px; border:1px solid #333;"
        )
        self._frame_info.setWordWrap(True)
        self._frame_info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        fi_v.addWidget(self._frame_info, stretch=1)
        info_splitter.addWidget(fi_w)

        # bottom half: match candidates (scrollable)
        mc_w = QWidget()
        mc_v = QVBoxLayout(mc_w)
        mc_v.setContentsMargins(0, 0, 0, 0)
        mc_v.setSpacing(2)
        mc_hdr = QHBoxLayout()
        mc_hdr.setContentsMargins(0, 0, 0, 0)
        mc_hdr.addWidget(self._lbl("Match Candidates", bold=True, size=11))
        mc_hdr.addStretch()
        self._match_detail_btn = QPushButton("⊞")
        self._match_detail_btn.setFixedSize(22, 22)
        self._match_detail_btn.setToolTip("Open full-size Match Candidates window")
        self._match_detail_btn.setStyleSheet(
            "QPushButton{background:#2a2a2a; color:#aaa; border:1px solid #555; border-radius:3px; font-size:13px;}"
            "QPushButton:hover{background:#3a3a3a; color:#fff;}"
        )
        self._match_detail_btn.clicked.connect(self._on_open_match_window)
        mc_hdr.addWidget(self._match_detail_btn)
        mc_v.addLayout(mc_hdr)
        self._match_scroll = QScrollArea()
        self._match_scroll.setWidgetResizable(False)
        self._match_scroll.setStyleSheet(
            "QScrollArea{background:#111; border:1px solid #333;}"
        )
        self._match_strip = QLabel("No pass at current frame")
        self._match_strip.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._match_strip.setStyleSheet("background:#111; color:#555; font-size:10px; padding:4px;")
        self._match_scroll.setWidget(self._match_strip)
        mc_v.addWidget(self._match_scroll, stretch=1)
        info_splitter.addWidget(mc_w)

        info_splitter.setSizes([160, 240])
        rv.addWidget(info_splitter, stretch=1)
        self._info_splitter = info_splitter
        info_splitter.splitterMoved.connect(self._on_splitter_moved)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color:#555; font-size:9px;")
        self._summary_label.setWordWrap(True)
        rv.addWidget(self._summary_label)

        root.addWidget(right)

        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        model_name = Path(self._det_model_path).name if self._det_model_path else "not found"
        self._sb.showMessage(f"Ready — device: {self._clip_device}  |  model: {model_name}")

    def _build_legend(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(12)
        h.addWidget(self._lbl("Passes:"))
        h.addWidget(self._lbl("● RACE",    color=_SOURCE_COLORS["RACE"]))
        h.addWidget(self._lbl("● NOMATCH", color=_SOURCE_COLORS["NOMATCH"]))
        h.addWidget(self._lbl("● DUP",     color=_SOURCE_COLORS["DUP"]))
        h.addWidget(self._lbl("● EXTRA",   color=_SOURCE_COLORS["EXTRA"]))
        h.addWidget(self._lbl("● NOPASS",  color=_SOURCE_COLORS["NOPASS"]))
        h.addWidget(self._lbl("│ Lap",     color=_LAP_COLOR))
        h.addStretch()
        return w

    @staticmethod
    def _lbl(text: str, color: Optional[QColor] = None, bold: bool = False, size: int = 10) -> QLabel:
        l = QLabel(text)
        c = f"rgb({color.red()},{color.green()},{color.blue()})" if color else "#999"
        w = "bold" if bold else "normal"
        l.setStyleSheet(f"font-size:{size}px; color:{c}; font-weight:{w};")
        return l

    def _build_playback_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedHeight(34)
        self._play_btn.clicked.connect(self._on_play_pause)
        row.addWidget(self._play_btn)

        self._cancel_btn = QPushButton("⏹ Cancel")
        self._cancel_btn.setFixedHeight(34)
        self._cancel_btn.setStyleSheet("background:#5a3000; color:white;")
        self._cancel_btn.clicked.connect(self._cancel_extraction)
        self._cancel_btn.setVisible(False)
        row.addWidget(self._cancel_btn)

        return row

    def _build_seek_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self._lbl("Seek:", size=11))
        for label, delta in [("◀◀ -10s", -10), ("◀ -1s", -1), ("+1s ▶", 1), ("+10s ▶▶", 10)]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet("font-size:11px;")
            btn.clicked.connect(lambda _, d=delta: self._on_seek(self._video.current_t + d))
            row.addWidget(btn)
        for label, df in [("◀ -1f", -1), ("+1f ▶", 1)]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet("font-size:11px; background:#2a2a2a;")
            btn.clicked.connect(lambda _, d=df: self._video.seek_frames(d))
            row.addWidget(btn)
        return row

    def _build_shortcuts(self):
        pairs = [
            ("Space",        self._on_play_pause),
            ("Up",           lambda: self._video.seek_frames(+1)),
            ("Down",         lambda: self._video.seek_frames(-1)),
            ("Right",        lambda: self._on_seek(self._video.current_t + 1.0)),
            ("Left",         lambda: self._on_seek(self._video.current_t - 1.0)),
            ("Shift+Right",  lambda: self._on_seek(self._video.current_t + 10.0)),
            ("Shift+Left",   lambda: self._on_seek(self._video.current_t - 10.0)),
        ]
        for key, fn in pairs:
            QShortcut(QKeySequence(key), self).activated.connect(fn)

    # ── File loading ──────────────────────────────────────────

    def _on_open_track(self):
        root = data_root()
        root.mkdir(parents=True, exist_ok=True)
        d = QFileDialog.getExistingDirectory(
            self, "Open a track folder", str(self._track.dir if self._track else root)
        )
        if not d:
            return
        tp = track_of(d)
        if tp is None:
            QMessageBox.warning(self, "Not a track folder", f"Pick a track folder inside:\n{root}")
            return
        self._set_track(tp)

    def _set_track(self, tp: TrackPaths):
        self._track = tp.ensure()
        self._track_label.setText(f"  track: {tp.name}  ")
        self._track_label.setStyleSheet("color:#60dc78; font-weight:bold; font-size:12px;")
        if tp.gate_memory.exists():
            self._load_memory(str(tp.gate_memory))
        else:
            self._sb.showMessage(f"Track '{tp.name}' has no gate_memory.json yet — learn it in learn_ui first.")

    def _place_video(self, path: str) -> Optional[str]:
        """The video must be in a track folder; one from outside is copied into
        <track>/test_videos/ of the open track."""
        tp = track_of(path)
        if tp is not None:
            if not self._track or tp.dir != self._track.dir:
                self._set_track(tp)
            return path
        if self._track is None:
            QMessageBox.information(self, "Pick a track", "This video is not in the dataset yet — pick its track folder.")
            self._on_open_track()
            if self._track is None:
                return None
        dest = self._track.test_videos / Path(path).name
        if not dest.exists():
            ans = QMessageBox.question(
                self, "Copy video into the track?",
                f"Copy\n{Path(path).name}\ninto\n{dest.parent}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                return None
            self._sb.showMessage(f"Copying {Path(path).name}…")
            QApplication.processEvents()
            shutil.copy2(path, dest)
        return str(dest)

    def _on_open_video(self):
        start = self._track.test_videos if self._track else data_root()
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", str(start), VIDEO_FILTER)
        if not path:
            return
        path = self._place_video(path)
        if not path:
            return
        if not self._video.load(path):
            QMessageBox.critical(self, "Error", f"Cannot open video:\n{path}")
            return
        self._video_path = path
        self._timeline.set_duration(self._video.duration)
        self.setWindowTitle(f"FPV Race Review — {Path(path).name}")
        self._sb.showMessage(f"Video loaded: {Path(path).name}  ({self._video.duration:.1f}s)")
        self._check_ready()
        saved = track_of(path).race_data(Path(path).stem)
        if saved.exists():
            self._load_race_data(str(saved))
            self._sb.showMessage(f"Loaded saved results {saved.name} — ▶ Run Race Analysis to redo them")

    def _on_open_memory(self):
        start = self._track.dir if self._track else data_root()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gate Memory", str(start), "JSON Files (*.json)"
        )
        if path:
            self._load_memory(path)

    def _load_memory(self, path: str):
        if path:
            self._gate_memory_path = path
            try:
                with open(path, "r", encoding="utf-8") as f:
                    mem_data = json.load(f)
                gates = mem_data.get("memory", []) or []
                gates.sort(key=lambda g: int(g.get("order_idx", 0)))
                self._gate_mem = gates
                self._race_lookahead = int(mem_data.get("race_lookahead", 3))
            except Exception:
                self._gate_mem = []
                self._race_lookahead = 3
            self._sb.showMessage(f"Gate memory: {Path(path).name}  ({len(self._gate_mem)} gates)")
            self._check_ready()

    def _update_model_label(self):
        name = Path(self._det_model_path).name if self._det_model_path else "no model"
        self._model_label.setText(name)

    def _on_set_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "PyTorch Models (*.pt)")
        if path:
            self._det_model_path = path
            self._update_model_label()
            self._sb.showMessage(f"Model set: {Path(path).name}")

    def _check_ready(self):
        if self._video_path and self._gate_memory_path:
            self._status_badge.setText("  Ready — click ▶ Run Race Analysis")
            self._status_badge.setStyleSheet("color:#60dc78; font-size:12px; font-weight:bold;")

    # ── Background extraction ─────────────────────────────────

    def _on_run(self):
        if not self._video_path:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if not self._gate_memory_path:
            QMessageBox.warning(self, "No gate memory", "Open a gate_memory.json first.")
            return
        if not self._ensure_model():
            return

        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()

        stem = Path(self._video_path).stem
        here = Path(__file__).parent
        tp = track_of(self._video_path)
        tp.runs_dir.mkdir(parents=True, exist_ok=True)
        out_json = str(tp.race_data(stem))
        self._proc_json_path = out_json
        script = str(here / "extract_race.py")

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_proc_output)
        self._proc.finished.connect(self._on_proc_finished)

        args = [
            script,
            "--video",              self._video_path,
            "--det-model",          self._det_model_path,
            "--gate-memory",        self._gate_memory_path,
            "--output",             out_json,
            "--clip-device",        self._clip_device,
            "--sim-thresh",         str(round(self._sim_thresh, 3)),
            "--min-match-margin",   str(round(self._min_match_margin, 3)),
            "--g1-sim-thresh",      str(round(self._g1_sim_thresh, 3)),
            "--g1-margin",          str(round(self._g1_margin, 3)),
        ]
        if self._require_same_type:
            args.append("--require-same-type")
        self._proc.start(sys.executable, ["-u"] + args)        # unbuffered: progress lines arrive live

        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._status_badge.setText("  Running race analysis…")
        self._status_badge.setStyleSheet("color:#ffa028; font-size:12px; font-weight:bold;")
        self._sb.showMessage("Race analysis running in background…")

    def _on_proc_output(self):
        if not self._proc:
            return
        raw = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Progress:"):
                try:
                    pct = int(float(line.split("%")[0].split()[-1]))
                    self._progress_bar.setValue(pct)
                    parts = [p for p in line.split() if p.startswith("passes=")]
                    n = int(parts[0].split("=")[1]) if parts else 0
                    self._sb.showMessage(f"Analysing…  {pct}%   ({n} passes so far)")
                except Exception:
                    pass
            elif line.startswith("Done."):
                self._sb.showMessage(line)

    def _on_proc_finished(self, exit_code: int, _):
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)
        if exit_code != 0:
            self._status_badge.setText("  Extraction failed")
            self._status_badge.setStyleSheet("color:#ff4444; font-size:12px;")
            self._sb.showMessage("Race analysis failed or was cancelled.")
            return
        if not Path(self._proc_json_path).exists():
            self._sb.showMessage("No output JSON found.")
            return
        self._load_race_data(self._proc_json_path)

    def _cancel_extraction(self):
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._sb.showMessage("Extraction cancelled.")

    # ── Load + index race data ────────────────────────────────

    def _load_race_data(self, path: str):
        try:
            with open(path, encoding="utf-8") as f:
                self._race_data = json.load(f)
        except Exception as e:
            self._sb.showMessage(f"Could not load race data: {e}")
            return

        # Build timestamp-sorted index for fast lookup
        self._frames_by_t = []
        for entry in self._race_data.get("frames", []):
            self._frames_by_t.append((float(entry.get("t", 0.0)), entry))
        self._frames_by_t.sort(key=lambda x: x[0])

        self._all_passes = self._race_data.get("passes", [])
        self._all_laps   = self._race_data.get("laps", [])

        self._timeline.set_passes(self._all_passes)
        self._timeline.set_laps(self._all_laps)
        self._refresh_lap_list()

        n_passes = len(self._all_passes)
        n_race   = sum(1 for p in self._all_passes if p.get("source") == "RACE")
        n_nomatch= sum(1 for p in self._all_passes if p.get("source") == "NOMATCH")
        n_extra  = sum(1 for p in self._all_passes if p.get("source") == "EXTRA")
        n_laps   = len(self._all_laps)

        self._summary_label.setText(
            f"Total passes: {n_passes}  |  Matched: {n_race}  |  "
            f"No match: {n_nomatch}  |  Extras: {n_extra}  |  Laps: {n_laps}"
        )
        self._status_badge.setText(
            f"  {n_laps} laps  ·  {n_race} matched  ·  {n_nomatch} no-match"
        )
        self._status_badge.setStyleSheet("color:#60dc78; font-size:12px; font-weight:bold;")
        self._sb.showMessage(
            f"Loaded {Path(path).name}  —  {n_passes} passes  {n_laps} laps"
        )

    def _nearest_frame(self, t: float) -> Optional[dict]:
        if not self._frames_by_t:
            return None
        lo, hi = 0, len(self._frames_by_t) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._frames_by_t[mid][0] < t:
                lo = mid + 1
            else:
                hi = mid
        best, best_dt = None, 1e9
        for i in (lo - 1, lo, lo + 1):
            if 0 <= i < len(self._frames_by_t):
                dt = abs(self._frames_by_t[i][0] - t)
                if dt < best_dt:
                    best_dt = dt
                    best = self._frames_by_t[i][1]
        return best

    # ── Frame overlay (drawn on video frames) ─────────────────

    def _draw_overlay(self, frame: np.ndarray, t: float) -> np.ndarray:
        data = self._nearest_frame(t)
        if data is None:
            return frame

        H, W = frame.shape[:2]

        for tr in data.get("tracks", []):
            stage = str(tr.get("stage", "idle"))
            if stage != "aligned":
                continue

            x1, y1, x2, y2 = tr["bbox"]
            ttype = str(tr.get("type", "NONE"))
            score = float(tr.get("score", 0.0))
            area  = float(tr.get("area_ratio", 0.0))
            cdist = float(tr.get("cdist", 0.0))

            box_col   = _TYPE_CV2.get(ttype, (110, 110, 110))
            stage_col = _STAGE_CV2.get("aligned", (0, 220, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 4)
            cv2.line(frame, (x1, max(0, y1 - 4)), (x2, max(0, y1 - 4)), stage_col, 3)

            label = f"#{tr['track_id']} {ttype} {score:.2f}"
            cv2.putText(frame, label, (x1, max(16, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 2, cv2.LINE_AA)
            detail = f"area={area*100:.1f}%  cd={cdist:.2f}"
            cv2.putText(frame, detail, (x1, min(H - 4, y2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, stage_col, 1, cv2.LINE_AA)

        for ps in data.get("passes", []):
            src  = str(ps.get("source", "NOMATCH"))
            gid  = int(ps.get("gate_id", -1))
            sim  = float(ps.get("sim", 0.0))
            src_cv2 = {"RACE": (60, 220, 100), "NOMATCH": (60, 60, 220), "DUP": (40, 140, 220),
                       "EXTRA": (255, 160, 90)}.get(src, (180, 180, 180))
            msg = (f"G{gid}  sim={sim:.3f}  [{src}]" if gid >= 0 else
                   f"E{ps.get('extra_id')}  [EXTRA]" if src == "EXTRA" else src)
            cv2.putText(frame, msg, (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, src_cv2, 2, cv2.LINE_AA)

        for lp in data.get("laps", []):
            cv2.putText(frame, f"LAP {lp.get('lap', '?')}!",
                        (W // 2 - 70, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3, cv2.LINE_AA)

        return frame

    # ── Playback ──────────────────────────────────────────────

    def _on_play_pause(self):
        self._video.toggle()
        self._play_btn.setText("⏸  Pause" if self._video.is_playing else "▶  Play")

    def _on_seek(self, t: float):
        self._video.pause()
        self._play_btn.setText("▶  Play")
        self._video.seek(t)

    def _on_position(self, t: float):
        self._timeline.set_current_t(t)
        self._time_label.setText(f"{t:.2f} s  /  {self._video.duration:.2f} s")
        self._refresh_frame_info(t)

    # ── Frame info panel ──────────────────────────────────────

    def _refresh_frame_info(self, t: float):
        data = self._nearest_frame(t)
        if data is None:
            self._frame_info.setText("No frame data at this position.\nRun analysis first.")
            self._update_match_strip([])
            return

        lines = [f"t = {data.get('t', t):.3f}s   frame {data.get('idx', '?')}"]

        tracks = data.get("tracks", [])
        if tracks:
            lines.append(f"\nTracks ({len(tracks)}):")
            for tr in tracks:
                lines.append(
                    f"  #{tr['track_id']} {tr['type']}  "
                    f"stage={tr['stage']}  score={tr['score']:.2f}  "
                    f"area={tr['area_ratio']*100:.1f}%  cdist={tr['cdist']:.3f}"
                )
        else:
            lines.append("\nTracks: none")

        passes = data.get("passes", [])
        if passes:
            lines.append(f"\nPasses ({len(passes)}):")
            for ps in passes:
                gid = ps.get("gate_id", -1)
                sim = ps.get("sim", 0.0)
                src = ps.get("source", "?")
                reason = ps.get("reason", "")
                name = f"E{ps.get('extra_id')}" if src == "EXTRA" else f"G{gid}"
                lines.append(f"  {name}  sim={sim:.3f}  [{src}]  {reason}")

        for lp in data.get("laps", []):
            lines.append(f"\n  LAP {lp.get('lap', '?')} completed")

        self._frame_info.setText("\n".join(lines))
        self._update_match_strip(data.get("passes", []))

    def _on_splitter_moved(self):
        data = self._nearest_frame(self._video.current_t)
        if data:
            self._update_match_strip(data.get("passes", []))

    def _on_open_match_window(self):
        if self._match_cands_win is None:
            self._match_cands_win = MatchCandidatesWindow()
        self._match_cands_win.show()
        self._match_cands_win.raise_()
        data = self._nearest_frame(self._video.current_t)
        passes = data.get("passes", []) if data else []
        self._match_cands_win.update_data(passes, self._gate_mem, self._race_lookahead)

    def _update_match_strip(self, passes: List[dict]):
        if not passes or not self._gate_mem:
            self._match_strip.setPixmap(QPixmap())
            self._match_strip.setText("No pass at current frame")
            self._match_strip.adjustSize()
            return

        ps           = passes[0]
        matched_gid  = int(ps.get("gate_id", -1))
        source       = str(ps.get("source", "NOMATCH"))
        sim          = float(ps.get("sim", 0.0))
        exp_before   = int(ps.get("exp_before", -1))
        query_img    = str(ps.get("query_img", ""))

        n = len(self._gate_mem)
        if n > 0:
            if exp_before >= 0:
                exp = exp_before % n
            else:
                gid_to_pos = {int(g["gate_id"]): i for i, g in enumerate(self._gate_mem)}
                exp = gid_to_pos.get(matched_gid, 0)
            idxs = list(dict.fromkeys([exp, (exp + 1) % n, 0]))
            window = [self._gate_mem[i] for i in idxs]
        else:
            window = []

        if not window:
            self._match_strip.setPixmap(QPixmap())
            self._match_strip.setText("No gates in comparison window")
            self._match_strip.adjustSize()
            return

        # collect per-gate image lists (embed_imgs if present, else last_img)
        gate_imgs: List[List[str]] = []
        for g in window:
            imgs = [p for p in g.get("embed_imgs", []) if p]
            if not imgs:
                li = g.get("last_img", "")
                imgs = [li] if li else []
            gate_imgs.append(imgs)

        max_rows = max((len(x) for x in gate_imgs), default=1)
        n_cols   = 1 + len(window)   # query col + one per gate
        header_h = 14
        pad      = 3

        # Scale thumbnails to fill the available viewport
        vp       = self._match_scroll.viewport()
        avail_w  = max(vp.width(), n_cols * 40 + pad)
        avail_h  = max(vp.height(), header_h + max_rows * 30)
        thumb_w  = max(30, (avail_w - pad * (n_cols + 1)) // n_cols)
        thumb_h  = max(20, (avail_h - header_h - pad * (max_rows + 1)) // max_rows)

        canvas_w = n_cols * (thumb_w + pad) + pad
        canvas_h = header_h + max_rows * (thumb_h + pad) + pad
        canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)

        def _load(path: str) -> np.ndarray:
            if path and Path(path).exists():
                raw = cv2.imread(path)
                if raw is not None:
                    return cv2.resize(raw, (thumb_w, thumb_h))
            return np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)

        font_scale = max(0.28, thumb_w / 220.0)

        # ── Query column ──────────────────────────────────
        x0     = pad
        full_h = max_rows * (thumb_h + pad) - pad
        if query_img and Path(query_img).exists():
            raw = cv2.imread(query_img)
            if raw is not None:
                canvas[header_h: header_h + full_h, x0: x0 + thumb_w] = \
                    cv2.resize(raw, (thumb_w, full_h))
        cv2.putText(canvas, "Query", (x0 + 2, header_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, header_h), (x0 + thumb_w - 1, header_h + full_h),
                      (100, 100, 100), 1)

        # ── Gate columns ──────────────────────────────────
        for ci, (g, imgs) in enumerate(zip(window, gate_imgs)):
            gid      = int(g["gate_id"])
            is_match = (gid == matched_gid)
            x0       = pad + (ci + 1) * (thumb_w + pad)

            if is_match and source == "RACE":
                col = (60, 220, 100)
            elif is_match:
                col = (60, 60, 220)
            else:
                col = (80, 80, 80)

            lbl = f"G{gid} {sim:.2f}" if is_match else f"G{gid} ({len(imgs)})"
            cv2.putText(canvas, lbl, (x0 + 2, header_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1, cv2.LINE_AA)

            for ei, img_path in enumerate(imgs):
                y0 = header_h + ei * (thumb_h + pad)
                canvas[y0: y0 + thumb_h, x0: x0 + thumb_w] = _load(img_path)

            col_h = len(imgs) * (thumb_h + pad) - pad
            cv2.rectangle(canvas, (x0, header_h), (x0 + thumb_w - 1, header_h + col_h),
                          col, 2 if is_match else 1)

        rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi   = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qi)
        self._match_strip.setText("")
        self._match_strip.setPixmap(pix)
        self._match_strip.resize(pix.size())

        if self._match_cands_win and self._match_cands_win.isVisible():
            self._match_cands_win.update_data(passes, self._gate_mem, self._race_lookahead)

    # ── Lap list + splits ─────────────────────────────────────

    def _refresh_lap_list(self):
        self._lap_list.clear()
        if not self._all_laps:
            self._lap_list.addItem("No laps completed")
            return

        best_dt = min((float(L.get("dt", 1e9)) for L in self._all_laps), default=0.0)

        for L in self._all_laps:
            lap_no = int(L.get("lap", L.get("lap_no", 0)))
            dt = float(L.get("dt", 0.0))
            t0 = float(L.get("t0", 0.0))
            star = " ★" if abs(dt - best_dt) < 0.01 else ""
            item = QListWidgetItem(f"Lap {lap_no:>2d}   {dt:.2f}s{star}")
            item.setForeground(QColor(60, 220, 100) if star else QColor(200, 200, 200))
            item.setData(Qt.ItemDataRole.UserRole, t0)
            self._lap_list.addItem(item)

    def _on_lap_selected(self, row: int):
        if row < 0 or row >= len(self._all_laps):
            return
        lap = self._all_laps[row]
        t0 = float(lap.get("t0", 0.0))

        # Jump to lap start
        self._on_seek(t0)

        # Fill splits table
        splits = lap.get("splits", [])
        self._splits_table.setRowCount(len(splits))
        for r, s in enumerate(splits):
            gid   = int(s.get("gate_id", 0))
            gt    = str(s.get("type", ""))
            dt0   = float(s.get("dt0", 0.0))
            dprev = float(s.get("dprev", 0.0))
            self._splits_table.setItem(r, 0, QTableWidgetItem(f"G{gid}"))
            self._splits_table.setItem(r, 1, QTableWidgetItem(gt))
            self._splits_table.setItem(r, 2, QTableWidgetItem(f"{dt0:.2f}s"))
            self._splits_table.setItem(r, 3, QTableWidgetItem("—" if r == 0 else f"{dprev:.2f}s"))


# ──────────────────────────────────────────────────────────────
# Dark palette + entry point
# ──────────────────────────────────────────────────────────────

def _dark_palette() -> QPalette:
    pal = QPalette()
    for role, rgb in [
        (QPalette.ColorRole.Window,          (26,  26,  26)),
        (QPalette.ColorRole.WindowText,      (220, 220, 220)),
        (QPalette.ColorRole.Base,            (34,  34,  34)),
        (QPalette.ColorRole.AlternateBase,   (42,  42,  42)),
        (QPalette.ColorRole.Text,            (220, 220, 220)),
        (QPalette.ColorRole.Button,          (50,  50,  50)),
        (QPalette.ColorRole.ButtonText,      (220, 220, 220)),
        (QPalette.ColorRole.Highlight,       (42,  90, 138)),
        (QPalette.ColorRole.HighlightedText, (255, 255, 255)),
    ]:
        pal.setColor(role, QColor(*rgb))
    return pal


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
