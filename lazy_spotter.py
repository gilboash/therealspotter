# vocab_spotter.py
import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from collections import deque
from pathlib import Path

# CLIP
import torch
import open_clip
from PIL import Image

# The tracker and box helpers (light: the live spotter on the media box uses them too)
from track_core import iou, clamp_bbox, Track, TimeTracker, _cls_to_name  # noqa: F401

# Your separate pass detector module
from pass_detector import PassDetector, detect_camera_edges

# GateDB (NEW learn=append-to-index, race=match-only)
from gate_db import GateDB, build_gate_db_panel, compute_track_gate_hints, build_race_stats_panel
import pipeline_cfg


# ============================================================
# DEBUG VISUALIZATION SWITCH
# ============================================================
DEBUG_PASS_VIZ = False         # per-box pass debug text (inside/near bbox)
DEBUG_PASS_PANEL_VIZ = True    # pass detector panel window

PASS_DEBUG_PANEL_WIDTH = 1040            # pixels
PASS_DEBUG_PANEL_BG = (18, 18, 18)      # dark background
PASS_DEBUG_PANEL_TEXT = (235, 235, 235)
PASS_DEBUG_PANEL_DIM = (170, 170, 170)
PASS_DEBUG_PANEL_WARN = (0, 255, 255)
PASS_DEBUG_PANEL_GOOD = (0, 255, 0)

# ============================================================
# GATE DATABASE SWITCHES
# ============================================================
ENABLE_GATE_DB = True

# Visualization throttling (per-frame gate-id overlay is expensive)
GATE_ID_VIZ = True
GATE_ID_VIZ_EVERY_N_FRAMES = 6   # compute CLIP for tracks only every N frames
GATE_ID_VIZ_MAX_TRACKS = 3       # only annotate top-N tracks by area

# Optional: show a GateDB window (laps + gates table)
GATE_DB_PANEL_VIZ = True
GATE_DB_PANEL_WIDTH = 1300

GATE_DB_RACE_STATS = True
# ============================================================
# Embedding capture timing (FOCUS: aligned snapshot)
# ============================================================
ALIGNED_EMBED_PAD_FRAC = 0.18
ALIGNED_SNAPSHOT_TTL_SEC = 3.0
ALIGNED_SNAPSHOT_LOCK_ONCE = True

# If False: we still compute embeddings, but do not save JPGs to disk
SAVE_PASS_CROPS_TODISK = False

# ============================================================
# Learning memory controls (keyboard)
# ============================================================
GATE_MEMORY_PATH = "gate_memory.json"
RACE_LOOKAHEAD = 3


# ============================================================
# Utilities
# ============================================================

def crop_with_padding(frame: np.ndarray, bbox: Tuple[int, int, int, int], pad_frac: float = 0.10) -> np.ndarray:
    """
    Safe crop with padding around bbox.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = int(bw * pad_frac)
    py = int(bh * pad_frac)

    xx1 = max(0, x1 - px)
    yy1 = max(0, y1 - py)
    xx2 = min(W, x2 + px)
    yy2 = min(H, y2 + py)

    if xx2 <= xx1 or yy2 <= yy1:
        return frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)].copy()

    return frame[yy1:yy2, xx1:xx2].copy()


def _alpha_rect(img, x1, y1, x2, y2, alpha=0.55, color=(0, 0, 0)):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


# ============================================================
# Mode-aware status text builder (top-left)
# ============================================================

def build_status_text(
    gatedb: Optional[GateDB],
    last_pass_candidate: Optional[dict],
    lap_flash_active: bool,
) -> str:
    if gatedb is None:
        return ""

    mode = str(getattr(gatedb, "mode", "learn")).lower()
    memsz = int(gatedb.memory_size())
    expi = int(gatedb.memory_expected_index())
    look = int(getattr(gatedb, "race_lookahead", 0))

    # lightweight preview if GateDB provides it (optional)
    win_prev = ""
    if hasattr(gatedb, "memory_window_preview"):
        try:
            win_prev = str(gatedb.memory_window_preview(max_items=3))
        except Exception:
            win_prev = ""

    lines: List[str] = []
    if lap_flash_active:
        lines.append("*** LAP END / WRAP ***")

    if mode == "learn":
        lines.append(f"MODE=LEARN  mem={memsz}  exp={expi}")
        if win_prev:
            lines.append(f"next slots: {win_prev}")

        if last_pass_candidate is not None:
            lines.append(
                f"last pass: type={last_pass_candidate['gate_type']}  (press 'c' to learn into idx={expi})"
            )
        else:
            lines.append("last pass: (none yet)")
        lines.append("LEARN: c=append to current index, x=skip(create), k=lap end(reset idx->0)")
    elif mode == "auto":
        lines.append(f"MODE=AUTO  mem={memsz}  exp={expi}  look={look}")
        if win_prev:
            lines.append(f"expected: {win_prev}")
        lines.append("AUTO: match-or-create. ")
    else:
        lines.append(f"MODE=RACE  mem={memsz}  exp={expi}  look={look}")
        if win_prev:
            lines.append(f"expected: {win_prev}")
        lines.append("RACE: match-only (no creation). x=skip idx, k=lap end(reset idx->0)")


    # NEW: show recent lap durations (start gate -> start gate)
    if mode == "race" and hasattr(gatedb, "lap_history"):
        try:
            laps = list(getattr(gatedb, "lap_history", []))[-3:]  # last 3 laps
        except Exception:
            laps = []
        if laps:
            parts = []
            for L in laps:
                lap_idx = max(0, int(L.get("lap", 0)) - 1)
                dt = float(L.get("dt", 0.0))
                parts.append(f"#{lap_idx} {dt:.2f}s")
            lines.append("laps: " + " | ".join(parts))

    return "\n".join(lines)


# ============================================================
# Always-on HOTKEYS overlay (main screen)
# ============================================================

def draw_hotkeys_overlay(
    frame: np.ndarray,
    *,
    paused: bool,
    gate_memory_path: str,
    pass_enabled: bool,
    gatedb: Optional[GateDB],
    has_last_candidate: bool = False,
):
    """
    Draw a persistent hotkeys menu on the main output frame.
    Always visible (even before first pass).
    """
    H, W = frame.shape[:2]
    x = 10
    y = 10
    pad = 8
    line_h = 18

    db_mode = (str(getattr(gatedb, "mode", "learn")).lower() if gatedb is not None else "off")
    if db_mode == "race" or db_mode == "auto":
        return

    pflag = " [PAUSED]" if paused else ""

    lines: List[str] = []
    lines.append(f"HOTKEYS  mode={db_mode.upper()}{pflag}")
    lines.append("q: quit   SPACE: pause/resume   n: step (paused)")
    lines.append("")
    lines.append(f"PassDetector: {'ON' if pass_enabled else 'OFF'}   Pass debug panel: {'ON' if DEBUG_PASS_PANEL_VIZ else 'OFF'}")
    lines.append("")

    gatedb_on = (gatedb is not None)
    memsz = int(gatedb.memory_size()) if gatedb_on else 0
    expi = int(gatedb.memory_expected_index()) if gatedb_on else 0
    look = int(getattr(gatedb, "race_lookahead", 0)) if gatedb_on else 0

    lines.append(f"Gate memory: {'ON' if gatedb_on else 'OFF'}")
    if gatedb_on:
        if db_mode == "learn":
            lines.append(f"  c: learn last pass into expected idx  [{'READY' if has_last_candidate else 'no candidate'}]")
            lines.append("  x: skip expected idx (still creates slot)")
            lines.append("  k: lap end / wrap (expected idx -> 0)")
            lines.append("  r: reset memory (RAM)")
            lines.append("  s: save memory file")
            lines.append("  l: load memory file")
            lines.append(f"  mem={memsz} exp={expi}")
        else:
            lines.append("  x: skip expected idx (recovery)")
            lines.append("  k: lap end / wrap (expected idx -> 0)")
            lines.append("  l: load memory file")
            lines.append("  s: save memory file (optional)")
            lines.append("  r: reset memory (optional)")
            lines.append(f"  mem={memsz} exp={expi} look={look}")
    else:
        lines.append("  (GateDB disabled because CLIP is off)")

    if gate_memory_path:
        lines.append(f"  file: {gate_memory_path}")

    # box size
    box_w = 0
    for t in lines:
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        box_w = max(box_w, tw)
    box_h = line_h * len(lines) + pad * 2

    x2 = min(W - 2, x + box_w + pad * 2)
    y2 = min(H - 2, y + box_h)

    _alpha_rect(frame, x, y, x2, y2, alpha=0.55, color=(0, 0, 0))

    ty = y + pad + 14
    for t in lines:
        color = (235, 235, 235)
        if "READY" in t:
            color = (0, 255, 0)
        if "no candidate" in t:
            color = (0, 255, 255)
        if "(GateDB disabled" in t:
            color = (170, 170, 170)
        cv2.putText(frame, t, (x + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        ty += line_h


# ============================================================
# Crop cache (for delayed pass event)
# ============================================================

class GateCropCache:
    """
    Keeps a short rolling buffer of crops per track_id so when pass is emitted
    (often after disappearance) we can still save the most relevant crop.
    """
    def __init__(self, max_age_sec: float = 1.0, max_items_per_track: int = 12, pad_frac: float = 0.10):
        self.max_age_sec = float(max_age_sec)
        self.max_items_per_track = int(max_items_per_track)
        self.pad_frac = float(pad_frac)
        self.buffers: Dict[int, deque] = {}

    def update_from_tracks(self, frame: np.ndarray, tracks: List["Track"], now: float, frame_w: int, frame_h: int):
        for tr in tracks:
            tid = int(tr.track_id)
            x1, y1, x2, y2 = tr.bbox

            crop = crop_with_padding(frame, (x1, y1, x2, y2), pad_frac=self.pad_frac)

            area = float(max(0, x2 - x1) * max(0, y2 - y1))
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            nx = cx / max(frame_w, 1)
            ny = cy / max(frame_h, 1)
            cdist = float(((nx - 0.5) ** 2 + (ny - 0.5) ** 2) ** 0.5)

            buf = self.buffers.get(tid)
            if buf is None:
                buf = deque(maxlen=self.max_items_per_track)
                self.buffers[tid] = buf

            buf.append({
                "t": float(now),
                "crop": crop,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "cdist": cdist,
            })

        self._prune(now)

    def _prune(self, now: float):
        cutoff = now - self.max_age_sec
        for tid in list(self.buffers.keys()):
            buf = self.buffers[tid]
            while buf and buf[0]["t"] < cutoff:
                buf.popleft()
            if not buf:
                self.buffers.pop(tid, None)

    def get_best_crop(self, track_id: int) -> Optional[dict]:
        """
        Best cached item: largest area, tie-breaker smallest center distance.
        """
        buf = self.buffers.get(int(track_id))
        if not buf:
            return None
        best = None
        for item in buf:
            if best is None:
                best = item
                continue
            if item["area"] > best["area"]:
                best = item
            elif item["area"] == best["area"] and item["cdist"] < best["cdist"]:
                best = item
        return best


# ============================================================
# CLIP embedder
# ============================================================

class ClipEmbedder:
    def __init__(self, device: str = "cpu", model: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_bgr(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


def save_pass_crop(out_dir: str, pass_idx: int, gate_type: str, track_id: int, now: float, item: dict,
                   prefix: str = ""):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}pass{pass_idx:04d}_tid{track_id}_{gate_type}_{now:.3f}s.jpg"
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, item["crop"])
    return path


# ============================================================
# Tracking + locking
# ============================================================

# ============================================================
# Visualization
# ============================================================

DEFAULT_COLORS = {
    "square": (255, 0, 0),
    "circle": (0, 255, 0),
    "arch": (0, 255, 255),
    "flagpole": (255, 0, 255),
    "gate": (0, 165, 255),
    "NONE": (128, 128, 128),
}

def type_color(type_name: str, palette: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    return palette.get(type_name, (0, 165, 255))

def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _norm_type(s: str) -> str:
    return (s or "").strip().lower()

def _cooldown_remaining(passdet: PassDetector, track_id: int, ttype: str, now: float) -> Tuple[float, float, float]:
    g_last = getattr(passdet, "_last_pass_time_global", -1e9)
    by_type = getattr(passdet, "_last_pass_time_by_type", {})
    by_track = getattr(passdet, "_last_pass_time_by_track", {})

    pass_cd = float(getattr(passdet, "pass_cooldown_sec", 0.0))
    type_cd = float(getattr(passdet, "type_cooldown_sec", 0.0))
    track_cd = float(getattr(passdet, "track_cooldown_sec", 0.0))

    rem_g = max(0.0, pass_cd - (now - float(g_last)))
    tkey = _norm_type(ttype)
    rem_t = max(0.0, type_cd - (now - float(by_type.get(tkey, -1e9))))
    rem_tr = max(0.0, track_cd - (now - float(by_track.get(int(track_id), -1e9))))
    return rem_g, rem_t, rem_tr


# ============================================================
# PASS DETECTOR DEBUG PANEL
# ============================================================

def build_pass_debug_panel(
    frame_h: int,
    frame_w: int,
    now: float,
    tracks: List[Track],
    passdet: PassDetector,
    title: str = "PassDetector Debug",
) -> np.ndarray:
    panel_w = PASS_DEBUG_PANEL_WIDTH
    panel = np.zeros((frame_h, panel_w, 3), dtype=np.uint8)
    panel[:, :] = PASS_DEBUG_PANEL_BG

    tr_by_id: Dict[int, Track] = {int(t.track_id): t for t in tracks}
    st_by_id = getattr(passdet, "states", {}) or {}

    y = 26
    cv2.putText(panel, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, PASS_DEBUG_PANEL_TEXT, 2, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, f"t={now:.3f}s  states={len(st_by_id)}  tracks={len(tracks)}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PASS_DEBUG_PANEL_DIM, 2, cv2.LINE_AA)
    y += 20

    col = "tid  type       stg     score   area%   cdist   seenAgo  misc"
    cv2.putText(panel, col, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, PASS_DEBUG_PANEL_TEXT, 2, cv2.LINE_AA)
    y += 10
    cv2.line(panel, (10, y), (panel_w - 10, y), (70, 70, 70), 1)
    y += 18

    frame_area = float(max(1, frame_w * frame_h))
    items = sorted(st_by_id.items(), key=lambda kv: int(kv[0]))

    line_h = 18
    max_lines = (frame_h - y - 10) // line_h

    if len(items) > max_lines:
        items = items[:max_lines]
        cv2.putText(panel, f"(showing first {max_lines} of {len(st_by_id)})",
                    (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PASS_DEBUG_PANEL_WARN, 2, cv2.LINE_AA)

    for tid, st in items:
        tid = int(tid)
        tr = tr_by_id.get(tid)

        ttype = str(getattr(st, "ttype", ""))
        stage = str(getattr(st, "stage", ""))
        last_area = float(getattr(st, "last_area", 0.0))
        last_cx = float(getattr(st, "last_cx", 0.0))
        last_cy = float(getattr(st, "last_cy", 0.0))
        last_seen_time = float(getattr(st, "last_seen_time", 0.0))

        area_ratio = last_area / frame_area
        nx = last_cx / max(frame_w, 1)
        ny = last_cy / max(frame_h, 1)
        cdist = ((abs(nx - 0.5) ** 2 + abs(ny - 0.5) ** 2) ** 0.5)
        seen_ago = max(0.0, now - last_seen_time)

        score = float(tr.score_ema) if tr is not None else -1.0

        flag_centered = bool(getattr(st, "flag_was_centered", False))

        color = PASS_DEBUG_PANEL_TEXT
        if stage == "aligned":
            color = PASS_DEBUG_PANEL_WARN
        if stage == "passed":
            color = PASS_DEBUG_PANEL_GOOD

        short_type = (ttype[:9] + "…") if len(ttype) > 10 else ttype.ljust(10)
        short_stage = stage.ljust(7)[:7]

        misc = f"ctr={'Y' if flag_centered else 'N'}" if ("flag" in ttype.lower() or "pole" in ttype.lower()) else ""
        line = f"{tid:>3d}  {short_type} {short_stage} {score:>5.2f}  {area_ratio*100:>5.2f}  {cdist:>6.3f}  {seen_ago:>6.2f}  {misc}"

        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        y += line_h

    return panel


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    palette: Dict[str, Tuple[int, int, int]],
    show_none: bool,
    hide_after: float,
    now: float,
    passhud: Optional["PassHUD"] = None,
    passdet: Optional[PassDetector] = None,
    gate_hint: Optional[Dict[int, Tuple[int, float]]] = None,
    status_text: str = "",
):
    H, W = frame.shape[:2]
    frame_area = float(W * H)
    st_map = getattr(passdet, "states", {}) if passdet is not None else {}

    for tr in tracks:
        t = tr.locked_type if tr.locked_type else "NONE"
        if (not show_none) and t == "NONE":
            continue
        if hide_after > 0 and (now - tr.last_seen) > hide_after:
            continue

        x1, y1, x2, y2 = tr.bbox
        c = type_color(t, palette)

        highlighted = (passhud is not None) and passhud.is_highlighted(tr.track_id, now)
        thick = 6 if highlighted else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thick)

        label = f"#{tr.track_id} {t} {tr.score_ema:.2f}"
        cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)

        if gate_hint is not None and int(tr.track_id) in gate_hint:
            gid, sim = gate_hint[int(tr.track_id)]
            hint = f"G{gid} sim={sim:.2f}"
            cv2.putText(frame, hint, (x1, min(H - 6, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

        if DEBUG_PASS_VIZ and passdet is not None and int(tr.track_id) in st_map:
            st = st_map[int(tr.track_id)]
            stage = str(getattr(st, "stage", ""))
            la = float(getattr(st, "last_area", 0.0))
            ar = la / max(frame_area, 1.0)

            cx, cy = _center(tr.bbox)
            nx = cx / max(W, 1)
            ny = cy / max(H, 1)
            dx = abs(nx - 0.5)
            dy = abs(ny - 0.5)
            center_dist = (dx * dx + dy * dy) ** 0.5

            rem_g, rem_t, rem_tr = _cooldown_remaining(passdet, int(tr.track_id), str(getattr(st, "ttype", "")), now)
            dbg1 = f"stage={stage}  area={ar:.3f}  cDist={center_dist:.3f}"
            dbg2 = f"cd(g/t/tr)={rem_g:.2f}/{rem_t:.2f}/{rem_tr:.2f}"

            box_w = max(120, min(340, x2 - x1))
            bx1 = x1
            by2 = max(0, y1 - 2)
            by1 = max(0, by2 - 34)
            _alpha_rect(frame, bx1, by1, bx1 + box_w, by2, alpha=0.55, color=(0, 0, 0))
            cv2.putText(frame, dbg1, (bx1 + 3, by1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, dbg2, (bx1 + 3, by1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        if highlighted:
            cv2.putText(frame, "PASSED", (x1, min(H - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    if status_text:
        _alpha_rect(frame, 6, 6, min(frame.shape[1] - 6, 980), 110, alpha=0.55, color=(0, 0, 0))
        y0 = 26
        for line in status_text.splitlines()[:5]:
            cv2.putText(frame, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
            y0 += 20


# ============================================================
# PASSED HUD
# ============================================================

@dataclass
class PassEvent:
    idx: int
    track_id: int
    type: str
    reason: str
    t: float


class PassHUD:
    def __init__(self, keep_seconds: float = 3.0, max_events: int = 8, highlight_seconds: float = 1.2):
        self.keep_seconds = float(keep_seconds)
        self.max_events = int(max_events)
        self.highlight_seconds = float(highlight_seconds)

        self._events: List[PassEvent] = []
        self._next_idx = 1
        self._highlight_until: Dict[int, float] = {}

    def add(self, now: float, track_id: int, gate_type: str, reason: str):
        self._events.append(PassEvent(
            idx=self._next_idx,
            track_id=int(track_id),
            type=str(gate_type),
            reason=str(reason),
            t=float(now),
        ))
        self._next_idx += 1
        self._highlight_until[int(track_id)] = now + self.highlight_seconds
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]

    def _prune(self, now: float):
        if self.keep_seconds > 0:
            cutoff = now - self.keep_seconds
            self._events = [e for e in self._events if e.t >= cutoff]
        for tid in list(self._highlight_until.keys()):
            if now >= self._highlight_until[tid]:
                self._highlight_until.pop(tid, None)

    def is_highlighted(self, track_id: int, now: float) -> bool:
        t = self._highlight_until.get(int(track_id), 0.0)
        return now < t

    def draw(self, frame: np.ndarray, now: float, x: int = 10, y: int = 60):
        self._prune(now)
        if not self._events:
            return
        cv2.putText(frame, "PASSED:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        y += 26
        for e in reversed(self._events):
            line = f"{e.idx}) #{e.track_id} {e.type} ({e.reason})"
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
            y += 22


# ============================================================
# YOLO multiclass helpers
# ============================================================

def _get_yolo_names(det: YOLO) -> Dict[int, str]:
    names = getattr(det.model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {}



# ============================================================
# Main
# ============================================================

def main():
    _pipe_cfg = pipeline_cfg.load()

    parser = argparse.ArgumentParser(
        description="FPV gate spotter: multiclass YOLO detector + type-lock tracking + pass detector + GateDB (learned order memory)"
    )

    parser.add_argument("--mode", choices=["calib", "learn", "race", "auto"], default="learn")
    parser.add_argument("--video", type=str, default=None, help="Path to MP4 file (optional). If omitted, uses camera 0.")

    parser.add_argument("--show-none", action="store_true", help="Visualize NONE-class tracks (default: hidden)")
    parser.add_argument("--max-candidates", type=int, default=10, help="Max detections per frame after filtering/sort")
    parser.add_argument("--min-type-score", type=float, default=0.20, help="Min detector confidence to accept class; else NONE")

    parser.add_argument("--det-model", type=str, required=True, help="Path to your trained YOLO .pt model")
    parser.add_argument("--det-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--det-maxdet", type=int, default=50, help="YOLO max detections per frame")

    parser.add_argument("--iou-match", type=float, default=_pipe_cfg["tracker"]["iou_match_thresh"], help="IOU association threshold")
    parser.add_argument("--ttl-seconds", type=float, default=_pipe_cfg["tracker"]["ttl_seconds"], help="Track TTL in seconds")
    parser.add_argument("--min-det-conf", type=float, default=_pipe_cfg["tracker"]["min_det_conf"], help="Min detection confidence for tracker (0 disables)")
    parser.add_argument("--hide-after", type=float, default=0.1, help="Hide tracks if not seen for this many seconds (0 disables)")
    parser.add_argument("--lock-min-score", type=float, default=_pipe_cfg["tracker"]["lock_min_score"])
    parser.add_argument("--lock-hysteresis", type=float, default=_pipe_cfg["tracker"]["lock_hysteresis"])
    parser.add_argument("--lock-streak", type=int, default=_pipe_cfg["tracker"]["lock_streak"])

    # Pass detector
    parser.add_argument("--pass-enable", action="store_true", help="Enable gate pass detection")
    parser.add_argument("--pass-min-score", type=float, default=_pipe_cfg["gates"]["min_track_score"], help="Min track score_ema for pass logic")
    parser.add_argument("--pass-min-area", type=float, default=_pipe_cfg["gates"]["min_area_ratio"], help="Min bbox area ratio (gate close enough)")
    parser.add_argument("--pass-center-tol", type=float, default=_pipe_cfg["gates"]["center_tol"], help="Center tolerance for alignment (0..1)")
    parser.add_argument("--pass-disappear", type=float, default=_pipe_cfg["gates"]["disappear_timeout"], help="Seconds after alignment where disappearance counts as pass")
    parser.add_argument("--pass-hud-seconds", type=float, default=3.0, help="How long to keep PASSED lines on screen")
    parser.add_argument("--pass-hud-max", type=int, default=8, help="Max PASSED lines shown")

    # embeddings
    parser.add_argument("--save-pass-crops", action="store_true", help="Enable CLIP embeddings (and optionally save crops)")
    parser.add_argument("--pass-crops-dir", type=str, default=None, help="Folder to save pass crops into (default: <track>/pass_crops)")
    parser.add_argument("--clip-device", type=str, default="cpu", help="cpu / mps / cuda for CLIP embedding")

    # memory persistence
    parser.add_argument("--gate-memory", type=str, default=None, help="Path to gate memory json file (default: the video's track)")
    parser.add_argument("--race-lookahead", type=int, default=RACE_LOOKAHEAD, help="How many gates ahead to consider in race mode")

    args = parser.parse_args()

    # Default file locations come from the video's track folder (dataset_paths.py)
    from dataset_paths import data_root, track_of
    _tp = track_of(args.video) if args.video else None
    if args.gate_memory is None:
        args.gate_memory = str(_tp.gate_memory) if _tp else str(data_root() / GATE_MEMORY_PATH)
    if args.pass_crops_dir is None:
        stem = Path(args.video).stem if args.video else "camera"
        args.pass_crops_dir = str((_tp.runs_dir if _tp else data_root()) / f"{stem}.pass_crops")

    det = YOLO(args.det_model)
    names = _get_yolo_names(det)

    tracker = TimeTracker(
        iou_match_thresh=args.iou_match,
        ttl_seconds=args.ttl_seconds,
        lock_min_score=args.lock_min_score,
        lock_hysteresis=args.lock_hysteresis,
        lock_streak=args.lock_streak,
        min_det_conf=args.min_det_conf,
    )

    passdet: Optional[PassDetector] = None
    passhud: Optional[PassHUD] = None
    if args.pass_enable:
        passdet = PassDetector(
            min_track_score=args.pass_min_score,
            min_area_ratio=args.pass_min_area,
            center_tol=args.pass_center_tol,
            disappear_timeout=args.pass_disappear,
            ignore_flagpoles=False,
        )
        passhud = PassHUD(keep_seconds=args.pass_hud_seconds, max_events=args.pass_hud_max)

    crop_cache = GateCropCache(max_age_sec=1.2, max_items_per_track=14, pad_frac=0.12)

    clip: Optional[ClipEmbedder] = None
    if args.save_pass_crops:
        clip = ClipEmbedder(device=args.clip_device)

    gatedb: Optional[GateDB] = None
    if ENABLE_GATE_DB and clip is not None:
        # NOTE: new GateDB signature (no update_sim_thresh/proto_ema/gate_revisit_cooldown_sec)
        gatedb = GateDB(
            sim_thresh=0.88,
            require_same_type=False,
            min_lap_gap_sec=6.0,
            min_gates_between_laps=2,
            min_match_margin=0.03,
            race_lookahead=int(args.race_lookahead),
            max_embeds_per_gate=6,
        )
        gatedb.set_mode(str(args.mode))
        gatedb.set_race_lookahead(int(args.race_lookahead))

        # load memory automatically if exists (your existing behavior)
        if args.gate_memory and os.path.exists(args.gate_memory):
            if str(args.mode).lower() == "race":
                try:
                    gatedb.load_memory(args.gate_memory)
                    print(f"[GateDB] Loaded memory from {args.gate_memory}")
                except Exception as e:
                    print(f"[GateDB] Failed loading memory: {e}")

    palette = dict(DEFAULT_COLORS)
    for _, n in names.items():
        palette.setdefault(str(n), (0, 165, 255))

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")

    if passdet is not None:
        ok, first_frame = cap.read()
        if ok:
            left_norm, right_norm = detect_camera_edges(first_frame)
            passdet.set_camera_edges(left_norm, right_norm)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"Detector model: {args.det_model}")
    print(f"Classes: {list(names.values()) if names else '(unknown names)'}")
    if args.pass_enable:
        print("Pass detector: ENABLED")
    print("Press 'q' to quit.")
    print("Controls: SPACE pause/resume | n step (when paused)")
    print("LEARN: c=learn(pass->idx) | x=skip(create) | k=lap end(wrap) | s/l/r")
    print("RACE : x=skip idx | k=lap end(wrap) | l (and optional s/r)")

    last_t = 0.0
    paused = False
    step_once = False
    frame = None

    track_gate_hint: Dict[int, Tuple[int, float]] = {}
    frame_idx = 0

    aligned_snapshots: Dict[int, dict] = {}

    def _prune_aligned_snapshots(now_ts: float):
        cutoff = now_ts - float(ALIGNED_SNAPSHOT_TTL_SEC)
        for tid in list(aligned_snapshots.keys()):
            if float(aligned_snapshots[tid].get("t", -1e9)) < cutoff:
                aligned_snapshots.pop(tid, None)

    # IMPORTANT: in new LEARN, last_pass_candidate has NO gate_id/sim
    last_pass_candidate: Optional[dict] = None

    # lap flash (main screen)
    lap_flash_until = 0.0

    while True:
        if not paused or step_once:
            ok, frame = cap.read()
            step_once = False
            if not ok:
                break
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_idx += 1

        now = pos_msec / 1000.0

        dt = now - last_t
        last_t = now

        H, W = frame.shape[:2]

        res = det(frame, conf=args.det_conf, verbose=False, max_det=int(args.det_maxdet))[0]

        typed: List[dict] = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bb = clamp_bbox((x1, y1, x2, y2), W, H)

            conf = float(b.conf[0]) if b.conf is not None else 0.0
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            cls_name = _cls_to_name(cls_id, names)

            gate_type = cls_name if conf >= args.min_type_score else "NONE"

            typed.append({
                "bbox": bb,
                "det_conf": conf,
                "type": gate_type,
                "type_score": conf,
            })

        typed_sorted = sorted(typed, key=lambda d: d["det_conf"], reverse=True)
        typed_topk = typed_sorted[:max(1, args.max_candidates)]

        tracks = tracker.update(typed_topk, now)
        crop_cache.update_from_tracks(frame, tracks, now, frame_w=W, frame_h=H)

        # GateID hints for current tracks (VIZ ONLY)
        if ENABLE_GATE_DB and GATE_ID_VIZ and gatedb is not None and clip is not None:
            new_hints = compute_track_gate_hints(
                frame=frame,
                tracks=tracks,
                gatedb=gatedb,
                clip_embedder=clip,
                crop_cache=crop_cache,
                frame_idx=frame_idx,
                enabled=True,
                viz_every_n_frames=GATE_ID_VIZ_EVERY_N_FRAMES,
                viz_max_tracks=GATE_ID_VIZ_MAX_TRACKS,
                pad_frac=0.12,
            )
            if new_hints:
                track_gate_hint = new_hints

        # pass detector update + pass events
        if passdet is not None:
            passdet.update(tracks, now, frame_w=W, frame_h=H)

            # capture embedding when a track FIRST becomes "aligned"
            if clip is not None and args.save_pass_crops and gatedb is not None:
                _prune_aligned_snapshots(now)
                st_map = getattr(passdet, "states", {}) or {}
                tr_by_id = {int(t.track_id): t for t in tracks}

                for tid, st in st_map.items():
                    tid = int(tid)
                    stage = str(getattr(st, "stage", ""))

                    if stage == "aligned":
                        if ALIGNED_SNAPSHOT_LOCK_ONCE and tid in aligned_snapshots:
                            continue
                        tr = tr_by_id.get(tid)
                        if tr is not None and tr.locked_type != "NONE":
                            crop = crop_with_padding(frame, tr.bbox, pad_frac=ALIGNED_EMBED_PAD_FRAC)
                            emb = clip.embed_bgr(crop)
                            aligned_snapshots[tid] = {
                                "t": float(now),
                                "gate_type": str(tr.locked_type),
                                "crop": crop,
                                "emb": emb,
                                "bbox": tuple(tr.bbox),
                            }

            while True:
                evt = passdet.pop_any_passed()
                if evt is None:
                    break

                if passhud is not None:
                    passhud.add(
                        now,
                        track_id=evt.get("track_id", -1),
                        gate_type=evt.get("type", "UNKNOWN"),
                        reason=evt.get("reason", ""),
                    )

                # embed + GateDB handling
                if args.save_pass_crops and passhud is not None and clip is not None and gatedb is not None:
                    tid = int(evt.get("track_id", -1))
                    evt_type = str(evt.get("type", "UNKNOWN"))

                    snap = aligned_snapshots.get(tid)
                    if snap is not None and str(snap.get("gate_type", "")) == evt_type:
                        crop = snap["crop"]
                        emb = snap["emb"]
                        item_for_save = {"crop": crop}
                    else:
                        best = crop_cache.get_best_crop(tid)
                        if best is None:
                            continue
                        crop = best["crop"]
                        emb = clip.embed_bgr(crop)
                        item_for_save = best

                    saved_path = ""
                    if SAVE_PASS_CROPS_TODISK:
                        saved_path = save_pass_crop(
                            out_dir=args.pass_crops_dir,
                            prefix=f"{Path(args.video).stem}_" if args.video else "",
                            pass_idx=passhud._next_idx,
                            gate_type=evt_type,
                            track_id=tid,
                            now=now,
                            item=item_for_save,
                        )

                    db_mode = str(getattr(gatedb, "mode", "learn")).lower()

                    if db_mode == "race" or db_mode == "auto":
                        # RACE: match-only (no creation)
                        gid, sim, _ = gatedb.match_or_create(
                            now=now,
                            gate_type=evt_type,
                            emb=emb,
                            img_path=saved_path,
                        )
                        if int(gid) >= 0:

                            prev_laps = int(getattr(gatedb, "lap_count", 0))
                            gatedb.on_pass(
                                now=now,
                                gate_id=gid,
                                gate_type=evt_type,
                                sim=sim,
                                reason=str(evt.get("reason", "")),
                                track_id=tid,
                                img_path=saved_path,
                            )
                            if int(getattr(gatedb, "lap_count", 0)) != prev_laps:
                                lap_flash_until = now + 1.0

                            # in race, last_pass_candidate is informational only
                            last_pass_candidate = {
                                "t": float(now),
                                "gate_type": str(evt_type),
                                "emb": emb,
                                "img_path": str(saved_path),
                                "gate_id": int(gid),
                                "sim": float(sim),
                            }
                        else:
                            # NO MATCH / DUP: don't count a pass, don't update last_pass_candidate
                            pass

                    else:
                        # LEARN: DO NOT match, DO NOT create here.
                        # Only store candidate; learning happens on key 'c'
                        last_pass_candidate = {
                            "t": float(now),
                            "gate_type": str(evt_type),
                            "emb": emb,
                            "img_path": str(saved_path),
                        }

                    aligned_snapshots.pop(tid, None)

        # status text
        status_text = build_status_text(
            gatedb=gatedb,
            last_pass_candidate=last_pass_candidate,
            lap_flash_active=(now < lap_flash_until),
        )

        # Visualization
        if args.mode in ("calib", "learn", "race", "auto"):
            draw_tracks(
                frame,
                tracks,
                palette=palette,
                show_none=args.show_none,
                hide_after=args.hide_after,
                now=now,
                passhud=passhud,
                passdet=passdet,
                gate_hint=track_gate_hint,
                status_text=status_text,
            )

            cv2.putText(frame, f"MODE: {args.mode.upper()}  dt={dt*1000:.1f}ms",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Detections={len(typed)} TopK={len(typed_topk)} Tracks={len(tracks)}",
                        (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

            if ENABLE_GATE_DB and gatedb is not None:
                cv2.putText(frame, f"LAPS: {gatedb.lap_count}",
                            (10, H - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

            if passhud is not None:
                passhud.draw(frame, now, x=300, y=60)

            draw_hotkeys_overlay(
                frame,
                paused=paused,
                gate_memory_path=str(args.gate_memory),
                pass_enabled=(passdet is not None),
                gatedb=gatedb,
                has_last_candidate=(last_pass_candidate is not None),
            )

            if DEBUG_PASS_PANEL_VIZ and passdet is not None:
                panel = build_pass_debug_panel(
                    frame_h=H,
                    frame_w=W,
                    now=now,
                    tracks=tracks,
                    passdet=passdet,
                    title="PassDetector Debug",
                )
                cv2.imshow("PassDetector Debug", panel)


            if GATE_DB_RACE_STATS and args.mode == "race" and gatedb is not None:
                rsp = build_race_stats_panel(
                    frame_h=H,
                    now=now,
                    gatedb=gatedb,
                    title="Race Stats (laps + splits)",
                    panel_w=980,
                    bg=PASS_DEBUG_PANEL_BG,
                    text=PASS_DEBUG_PANEL_TEXT,
                    dim=PASS_DEBUG_PANEL_DIM,
                    accent=(0, 200, 255),
                    good=PASS_DEBUG_PANEL_GOOD,
                )
                cv2.imshow("Race Stats", rsp)

            if GATE_DB_PANEL_VIZ and ENABLE_GATE_DB and gatedb is not None:
                gpanel = build_gate_db_panel(
                    frame_h=H,
                    now=now,
                    gatedb=gatedb,
                    title="GateDB (Re-ID + Laps)",
                    panel_w=GATE_DB_PANEL_WIDTH,
                    bg=PASS_DEBUG_PANEL_BG,
                    text=PASS_DEBUG_PANEL_TEXT,
                    dim=PASS_DEBUG_PANEL_DIM,
                    good=PASS_DEBUG_PANEL_GOOD,
                )
                cv2.imshow("GateDB", gpanel)

            cv2.imshow("Gate Spotter (YOLO multiclass)", frame)

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print(f"[DEBUG] paused = {paused}")
        elif key == ord("n") and paused:
            step_once = True

        # keyboard controls (mode-aware)
        if gatedb is not None:
            db_mode = str(getattr(gatedb, "mode", "learn")).lower()

            if key == ord("c"):
                if db_mode != "learn":
                    print("[GateDB] 'c' ignored in RACE mode.")
                else:
                    if last_pass_candidate is None:
                        print("[GateDB] learn ignored: no last pass candidate")
                    else:
                        # LEARN: append embedding into expected index (create slot if missing)
                        gid, _ = gatedb.learn_confirm_pass(
                            now=float(last_pass_candidate.get("t", now)),
                            gate_type=str(last_pass_candidate.get("gate_type", "UNKNOWN")),
                            emb=last_pass_candidate.get("emb", None),
                            img_path=str(last_pass_candidate.get("img_path", "")),
                        )
                        print(f"[GateDB] LEARN confirm -> G{gid}  mem={gatedb.memory_size()} exp={gatedb.memory_expected_index()}")
            elif key == ord("x"):
                if db_mode == "learn":
                    gid = gatedb.learn_skip_expected_gate(now=float(now))
                    print(f"[GateDB] LEARN skip -> G{gid}  exp={gatedb.memory_expected_index()}")
                else:
                    if hasattr(gatedb, "skip_expected_in_race"):
                        gatedb.skip_expected_in_race(step=1, now=float(now))
                        print(f"[GateDB] RACE skip expected -> exp={gatedb.memory_expected_index()}")
                    else:
                        # if you haven't updated gate_db.py yet
                        print("[GateDB] RACE skip expected ignored: GateDB.skip_expected_in_race() not found")

            elif key == ord("k"):
                gatedb.force_lap_end(float(now))
                lap_flash_until = now + 1.0
                print(f"[GateDB] lap end -> laps={gatedb.lap_count} exp={gatedb.memory_expected_index()}")

            elif key == ord("r"):
                gatedb.reset_memory()
                print("[GateDB] memory reset (RAM).")

            elif key == ord("s"):
                try:
                    gatedb.save_memory(args.gate_memory)
                    print(f"[GateDB] memory saved to {args.gate_memory}")
                except Exception as e:
                    print(f"[GateDB] save failed: {e}")

            elif key == ord("l"):
                try:
                    gatedb.load_memory(args.gate_memory)
                    print(f"[GateDB] memory loaded from {args.gate_memory}")
                except Exception as e:
                    print(f"[GateDB] load failed: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
