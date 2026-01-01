# gate_db.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class GateInfo:
    gate_id: int
    gate_type: str
    proto: np.ndarray              # normalized CLIP embedding
    num_updates: int = 1
    first_seen_t: float = 0.0
    last_seen_t: float = 0.0
    last_pass_t: float = 0.0
    pass_count: int = 0
    last_sim: float = 0.0
    last_img: str = ""


class GateDB:
    """
    Stable GateIDs using CLIP embedding similarity.
    Matching: cosine similarity (dot product of normalized embeddings).
    """
    def __init__(
        self,
        sim_thresh: float,
        require_same_type: bool = True,
        min_lap_gap_sec: float = 6.0,
        min_gates_between_laps: int = 2,
        start_gate_id: int = 1,
        # --- NEW (uniqueness + stability) ---
        min_match_margin: float = 0.0,          # require best - second_best >= margin
        update_sim_thresh: float = 0.95,        # only update proto if sim >= this
        proto_ema: float = 0.15,                # EMA factor for proto update
        gate_revisit_cooldown_sec: float = 0.0, # skip matching to same gate for this time after it was passed
    ):
        self.sim_thresh = float(sim_thresh)
        self.require_same_type = bool(require_same_type)

        self.min_lap_gap_sec = float(min_lap_gap_sec)
        self.min_gates_between_laps = int(min_gates_between_laps)

        # NEW params
        self.min_match_margin = float(min_match_margin)
        self.update_sim_thresh = float(update_sim_thresh)
        self.proto_ema = float(proto_ema)
        self.gate_revisit_cooldown_sec = float(gate_revisit_cooldown_sec)

        self.gates: Dict[int, GateInfo] = {}
        self.next_gate_id = 1

        # Lap tracking
        self.start_gate_id = int(start_gate_id)
        self.lap_count = 0
        self._last_lap_t = -1e9
        self._passes_since_lap = 0

        # UI
        self.last_events: List[dict] = []
        self.max_events = 12

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # embeddings expected normalized

    def match_or_create(
        self,
        now: float,
        gate_type: str,
        emb: np.ndarray,
        img_path: str = "",
    ) -> Tuple[int, float, bool]:
        """
        Returns: (gate_id, best_sim, is_new_gate)

        Matching rules:
          - candidate gates can be filtered by type (optional)
          - candidate gates can be skipped if they were passed very recently (gate_revisit_cooldown_sec)
          - accept match if best_sim >= sim_thresh AND (best_sim - second_best) >= min_match_margin
          - update proto only if best_sim >= update_sim_thresh (EMA update with proto_ema)
        """
        gate_type = str(gate_type)

        best_id: Optional[int] = None
        best_sim = -1.0
        second_best = -1.0

        for gid, g in self.gates.items():
            if self.require_same_type and (str(g.gate_type) != gate_type):
                continue

 #           print("test against gid " , gid)

            # NEW: don't let the same gate "win again" immediately after you passed it
            if self.gate_revisit_cooldown_sec > 0 and g.last_pass_t > 0:
                if (now - float(g.last_pass_t)) < self.gate_revisit_cooldown_sec:
                    continue
            s = self._cos(emb, g.proto)

#            print("not aged, sim ", s)

            if s > best_sim:
#                print("new best ")
                second_best = best_sim
                best_sim = s
                best_id = gid
            elif s > second_best:
#                print("second best ")
                second_best = s

        margin = best_sim - second_best if second_best > -0.5 else 1e9  # if only one candidate, treat margin as huge
#        print("margin  ", margin)

        if best_id is not None and best_sim >= self.sim_thresh and margin >= self.min_match_margin:
            g = self.gates[best_id]
            g.last_seen_t = now
            g.last_sim = best_sim
            if img_path:
                g.last_img = img_path

            # NEW: only update proto when match is VERY confident to prevent drift
            if best_sim >= self.update_sim_thresh and self.proto_ema > 0:
                ema = float(np.clip(self.proto_ema, 0.0, 1.0))
                new_proto = (1.0 - ema) * g.proto + ema * emb
                new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-9)
                g.proto = new_proto.astype(np.float32)
                g.num_updates += 1

            return best_id, best_sim, False

        # create new gate
        gid = self.next_gate_id
        self.next_gate_id += 1

        self.gates[gid] = GateInfo(
            gate_id=gid,
            gate_type=gate_type,
            proto=emb.astype(np.float32),
            num_updates=1,
            first_seen_t=now,
            last_seen_t=now,
            last_sim=1.0,
            last_img=img_path or "",
        )
        return gid, 1.0, True

    def on_pass(
        self,
        now: float,
        gate_id: int,
        gate_type: str,
        sim: float,
        reason: str,
        track_id: int,
        img_path: str = "",
    ):
        g = self.gates.get(int(gate_id))
        if g is not None:
            g.last_pass_t = now
            g.pass_count += 1
            g.last_sim = float(sim)
            if img_path:
                g.last_img = img_path

        # lap bookkeeping
        if int(gate_id) == int(self.start_gate_id):
            if (now - self._last_lap_t) >= self.min_lap_gap_sec and self._passes_since_lap >= self.min_gates_between_laps:
                self.lap_count += 1
                self._last_lap_t = now
                self._passes_since_lap = 0
        else:
            self._passes_since_lap += 1

        evt = {
            "t": float(now),
            "gate_id": int(gate_id),
            "type": str(gate_type),
            "sim": float(sim),
            "reason": str(reason),
            "track_id": int(track_id),
        }
        self.last_events.append(evt)
        if len(self.last_events) > self.max_events:
            self.last_events = self.last_events[-self.max_events:]

    def summary_rows(self) -> List[GateInfo]:
        return [self.gates[k] for k in sorted(self.gates.keys())]


def build_gate_db_panel(
    frame_h: int,
    now: float,
    gatedb: GateDB,
    title: str,
    panel_w: int,
    bg=(18, 18, 18),
    text=(235, 235, 235),
    dim=(170, 170, 170),
    good=(0, 255, 0),
) -> np.ndarray:
    panel = np.zeros((frame_h, panel_w, 3), dtype=np.uint8)
    panel[:, :] = bg

    y = 26
    cv2.putText(panel, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text, 2, cv2.LINE_AA)
    y += 22
    cv2.putText(
        panel,
        f"t={now:.3f}s | gates={len(gatedb.gates)} | laps={gatedb.lap_count} | sinceLap={gatedb._passes_since_lap}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        dim,
        2,
        cv2.LINE_AA,
    )
    y += 14
    cv2.line(panel, (10, y), (panel_w - 10, y), (70, 70, 70), 1)
    y += 18

    cv2.putText(panel, "GATES:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text, 2, cv2.LINE_AA)
    y += 20
    # ONLY CHANGE: add 2 columns (best/2nd best sim)
    cv2.putText(
        panel,
        "gid  type       passes  lastPassAgo  updates  lastSim  bestSim  2ndSim",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text,
        2,
        cv2.LINE_AA,
    )
    y += 16

    rows = gatedb.summary_rows()
    line_h = 18
    max_lines = max(1, (frame_h - y - 170) // line_h)
    rows = rows[:max_lines]

    # ONLY CHANGE: compute per-gate best/2nd-best similarity among gates of same type
    for g in rows:
        best_sim = -1.0
        second_sim = -1.0
        for gid2, g2 in gatedb.gates.items():
            if gid2 == g.gate_id:
                continue
            if gatedb.require_same_type and str(g2.gate_type) != str(g.gate_type):
                continue
            s = float(np.dot(g.proto, g2.proto))  # normalized
            if s > best_sim:
                second_sim = best_sim
                best_sim = s
            elif s > second_sim:
                second_sim = s

        last_pass_ago = (now - g.last_pass_t) if g.last_pass_t > 0 else 1e9

        # Format: if there is no other candidate, show "---"
        best_str = f"{best_sim:>7.3f}" if best_sim > -0.5 else "   --- "
        second_str = f"{second_sim:>7.3f}" if second_sim > -0.5 else "   --- "

        s = (
            f"{g.gate_id:>3d}  {g.gate_type[:10].ljust(10)}  {g.pass_count:>6d}  "
            f"{last_pass_ago:>10.2f}  {g.num_updates:>7d}  {g.last_sim:>7.3f}  "
            f"{best_str}  {second_str}"
        )
        color = good if g.gate_id == gatedb.start_gate_id else text
        cv2.putText(panel, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        y += line_h

    y += 10
    cv2.line(panel, (10, y), (panel_w - 10, y), (70, 70, 70), 1)
    y += 20
    cv2.putText(panel, "LAST PASSES (newest first):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text, 2, cv2.LINE_AA)
    y += 20

    evs = list(reversed(gatedb.last_events))[:10]
    for e in evs:
        s = f"G{e['gate_id']}  {e['type']:<9} sim={e['sim']:.3f}  tid={e['track_id']}  {e['reason']}"
        cv2.putText(panel, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text, 2, cv2.LINE_AA)
        y += 18

    return panel


def best_gate_match_viz_only(gatedb: GateDB, gate_type: str, emb: np.ndarray) -> Tuple[Optional[int], float]:
    """
    Returns (best_gate_id or None, best_sim). Does NOT create a new gate.
    """
    gate_type = str(gate_type)
    best_id = None
    best_sim = -1.0
    for gid, g in gatedb.gates.items():
        if gatedb.require_same_type and str(g.gate_type) != gate_type:
            continue
        s = float(np.dot(emb, g.proto))  # normalized
        if s > best_sim:
            best_sim = s
            best_id = gid
    return best_id, best_sim


def compute_track_gate_hints(
    frame: np.ndarray,
    tracks,
    gatedb: GateDB,
    clip_embedder,
    crop_cache,
    frame_idx: int,
    *,
    enabled: bool,
    viz_every_n_frames: int,
    viz_max_tracks: int,
    pad_frac: float = 0.12,
) -> Dict[int, Tuple[int, float]]:
    """
    Computes GateID hints for a small number of tracks (top-by-area),
    and only every N frames. Returns mapping: track_id -> (gate_id, sim).
    """
    if not enabled:
        return {}
    if len(gatedb.gates) == 0:
        return {}
    if (frame_idx % max(1, viz_every_n_frames)) != 0:
        return {}

    def area_of(t) -> float:
        x1, y1, x2, y2 = t.bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    sorted_tracks = sorted(tracks, key=area_of, reverse=True)[:max(1, viz_max_tracks)]

    hints: Dict[int, Tuple[int, float]] = {}
    for tr in sorted_tracks:
        if tr.locked_type == "NONE":
            continue

        best = crop_cache.get_best_crop(int(tr.track_id))
        if best is not None:
            crop = best["crop"]
        else:
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = tr.bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            px = int(bw * pad_frac)
            py = int(bh * pad_frac)
            xx1 = max(0, x1 - px)
            yy1 = max(0, y1 - py)
            xx2 = min(W, x2 + px)
            yy2 = min(H, y2 + py)
            crop = frame[yy1:yy2, xx1:xx2].copy()

        emb = clip_embedder.embed_bgr(crop)
        gid, sim = best_gate_match_viz_only(gatedb, gate_type=str(tr.locked_type), emb=emb)
        if gid is not None:
            hints[int(tr.track_id)] = (int(gid), float(sim))

    return hints
