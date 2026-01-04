# gate_db.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

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


@dataclass
class MemoryGate:
    """
    Track memory entry: order matters.
    We MUST store an embedding to be usable across sessions.
    """
    order_idx: int
    gate_id: int
    gate_type: str
    proto: np.ndarray             # normalized CLIP embedding (float32)
    created_t: float = 0.0
    last_img: str = ""


class GateDB:
    """
    Stable GateIDs using CLIP embedding similarity.
    Matching: cosine similarity (dot product of normalized embeddings).

    NEW:
      - mode: learn / race
      - ordered "track memory": list of MemoryGate
      - persistence: save/load memory JSON
      - race: bias matching to expected gates (lookahead K)
      - keyboard integration expects:
          set_mode(), set_race_lookahead()
          confirm_last_pass_into_memory(), mark_skipped_expected_gate()
          save_memory(), load_memory(), reset_memory()
          memory_size(), memory_expected_index()
    """

    def __init__(
        self,
        sim_thresh: float,
        require_same_type: bool = True,
        min_lap_gap_sec: float = 6.0,
        min_gates_between_laps: int = 2,
        start_gate_id: int = 1,
        # --- uniqueness + stability ---
        min_match_margin: float = 0.0,          # require best - second_best >= margin
        update_sim_thresh: float = 0.95,        # only update proto if sim >= this
        proto_ema: float = 0.15,                # EMA factor for proto update
        gate_revisit_cooldown_sec: float = 0.0, # skip matching to same gate for this time after it was passed
    ):
        self.sim_thresh = float(sim_thresh)
        self.require_same_type = bool(require_same_type)

        self.min_lap_gap_sec = float(min_lap_gap_sec)
        self.min_gates_between_laps = int(min_gates_between_laps)

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

        # ============================
        # NEW: Memory / Mode
        # ============================
        self.mode: str = "learn"         # learn | race
        self.race_lookahead: int = 3     # how many expected gates to consider
        self._memory: List[MemoryGate] = []
        self._expected_idx: int = 0      # next expected memory index (0-based)

    # ----------------------------
    # Basic helpers
    # ----------------------------
    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # embeddings expected normalized

    @staticmethod
    def _l2norm(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        n = float(np.linalg.norm(x) + 1e-9)
        return (x / n).astype(np.float32)

    # ----------------------------
    # Mode / Memory public API
    # ----------------------------
    def set_mode(self, mode: str):
        m = (mode or "").strip().lower()
        if m not in ("learn", "race", "calib"):
            m = "learn"
        # treat "calib" like learn for DB logic
        self.mode = "race" if m == "race" else "learn"

    def set_race_lookahead(self, k: int):
        self.race_lookahead = int(max(1, k))

    def memory_size(self) -> int:
        return int(len(self._memory))

    def memory_expected_index(self) -> int:
        return int(self._expected_idx)

    def reset_memory(self):
        self._memory = []
        self._expected_idx = 0

    def mark_skipped_expected_gate(self):
        """
        In learn mode (or race), allow user to indicate that the next expected gate was missed.
        This advances the expected pointer.
        """
        if self._memory:
            self._expected_idx = min(self._expected_idx + 1, len(self._memory))
        else:
            self._expected_idx = 0

    def confirm_last_pass_into_memory(
        self,
        gate_type: str,
        emb: np.ndarray,
        gate_id: int,
        img_path: str = "",
        t: float = 0.0,
    ) -> bool:
        """
        Confirm the last pass as "the next gate in order" and store embedding.
        This is what makes it usable in a *separate future session* (race mode).

        Returns True if added, False if rejected (e.g., NONE type or missing emb).
        """
        gate_type = str(gate_type)
        if gate_type == "NONE":
            return False
        if emb is None:
            return False
        embn = self._l2norm(np.asarray(emb))

        mg = MemoryGate(
            order_idx=len(self._memory),
            gate_id=int(gate_id),
            gate_type=gate_type,
            proto=embn,
            created_t=float(t),
            last_img=str(img_path or ""),
        )
        self._memory.append(mg)

        # after confirming, next expected becomes the next entry
        self._expected_idx = min(len(self._memory), self._expected_idx + 1)
        return True

    def save_memory(self, path: str):
        """
        Save ordered memory (including embeddings) to JSON.
        Embeddings stored as float lists.
        """
        data = {
            "version": 1,
            "mode": self.mode,
            "race_lookahead": int(self.race_lookahead),
            "expected_idx": int(self._expected_idx),
            "memory": [
                {
                    "order_idx": int(m.order_idx),
                    "gate_id": int(m.gate_id),
                    "gate_type": str(m.gate_type),
                    "proto": m.proto.astype(np.float32).tolist(),
                    "created_t": float(m.created_t),
                    "last_img": str(m.last_img or ""),
                }
                for m in self._memory
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_memory(self, path: str):
        """
        Load ordered memory from JSON (including embeddings).
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mem_in = data.get("memory", []) or []
        loaded: List[MemoryGate] = []
        for i, d in enumerate(mem_in):
            proto = np.asarray(d.get("proto", []), dtype=np.float32)
            if proto.size == 0:
                continue
            proto = self._l2norm(proto)

            loaded.append(
                MemoryGate(
                    order_idx=int(d.get("order_idx", i)),
                    gate_id=int(d.get("gate_id", i + 1)),
                    gate_type=str(d.get("gate_type", "gate")),
                    proto=proto,
                    created_t=float(d.get("created_t", 0.0)),
                    last_img=str(d.get("last_img", "")),
                )
            )

        # sort by order_idx to be safe
        loaded.sort(key=lambda m: int(m.order_idx))
        # re-index order_idx cleanly
        for j, m in enumerate(loaded):
            m.order_idx = j

        self._memory = loaded
        self.race_lookahead = int(max(1, data.get("race_lookahead", self.race_lookahead)))
        self._expected_idx = int(min(max(0, data.get("expected_idx", 0)), len(self._memory)))
        # mode in file is informational; caller sets runtime mode via set_mode()

    # ----------------------------
    # Memory-aware candidate selection (race)
    # ----------------------------
    def _memory_candidates(self, gate_type: str) -> List[MemoryGate]:
        """
        Return memory gates for matching.
        In race mode: only consider expected_idx .. expected_idx+lookahead-1
        In learn mode: consider all memory (useful for viz or warm-start)
        """
        gate_type = str(gate_type)
        if not self._memory:
            return []

        if self.mode == "race":
            s = int(self._expected_idx)
            e = int(min(len(self._memory), s + max(1, int(self.race_lookahead))))
            cands = self._memory[s:e]
        else:
            cands = list(self._memory)

        if self.require_same_type:
            cands = [m for m in cands if str(m.gate_type) == gate_type]
        return cands

    # ----------------------------
    # Core matching logic
    # ----------------------------
    def match_or_create(
        self,
        now: float,
        gate_type: str,
        emb: np.ndarray,
        img_path: str = "",
    ) -> Tuple[int, float, bool]:
        """
        Returns: (gate_id, best_sim, is_new_gate)

        Matching order:
          1) If memory exists:
              - try memory candidates first (race: expected window; learn: all memory)
              - if best memory sim >= sim_thresh and margin >= min_match_margin:
                  - return that memory gate_id (NO create)
                  - and in race mode, advance expected_idx by 1 (best effort)
          2) Otherwise fallback to classic gate DB (self.gates) match/create.

        Classic GateDB rules:
          - candidate gates filtered by type (optional)
          - candidate gates can be skipped if they were passed very recently (gate_revisit_cooldown_sec)
          - accept match if best_sim >= sim_thresh AND (best_sim - second_best) >= min_match_margin
          - update proto only if best_sim >= update_sim_thresh (EMA update with proto_ema)
        """
        gate_type = str(gate_type)
        emb = self._l2norm(np.asarray(emb))

        # 1) memory-first
        mem_cands = self._memory_candidates(gate_type)
        if mem_cands:
            best_m: Optional[MemoryGate] = None
            best_sim = -1.0
            second_best = -1.0
            for m in mem_cands:
                s = self._cos(emb, m.proto)
                if s > best_sim:
                    second_best = best_sim
                    best_sim = s
                    best_m = m
                elif s > second_best:
                    second_best = s

            margin = best_sim - second_best if second_best > -0.5 else 1e9
            if best_m is not None and best_sim >= self.sim_thresh and margin >= self.min_match_margin:
                # race: move expected forward (best effort)
                if self.mode == "race":
                    # advance expected idx if this was the first expected element;
                    # since we filtered to window starting at expected_idx, the
                    # best match corresponds to some position in that window.
                    # simplest: advance by 1.
                    self._expected_idx = min(self._expected_idx + 1, len(self._memory))

                # also ensure GateInfo exists (so UI table shows it)
                gid = int(best_m.gate_id)
                if gid not in self.gates:
                    self.gates[gid] = GateInfo(
                        gate_id=gid,
                        gate_type=gate_type,
                        proto=best_m.proto.copy(),
                        num_updates=1,
                        first_seen_t=float(now),
                        last_seen_t=float(now),
                        last_sim=float(best_sim),
                        last_img=str(img_path or best_m.last_img or ""),
                    )
                    self.next_gate_id = max(self.next_gate_id, gid + 1)
                else:
                    g = self.gates[gid]
                    g.last_seen_t = float(now)
                    g.last_sim = float(best_sim)
                    if img_path:
                        g.last_img = str(img_path)

                return gid, float(best_sim), False

        # 2) classic gate DB match/create (your previous logic)
        best_id: Optional[int] = None
        best_sim = -1.0
        second_best = -1.0

        for gid, g in self.gates.items():
            if self.require_same_type and (str(g.gate_type) != gate_type):
                continue

            # skip same gate immediately after passing it
            if self.gate_revisit_cooldown_sec > 0 and g.last_pass_t > 0:
                if (now - float(g.last_pass_t)) < self.gate_revisit_cooldown_sec:
                    continue

            s = self._cos(emb, g.proto)

            if s > best_sim:
                second_best = best_sim
                best_sim = s
                best_id = gid
            elif s > second_best:
                second_best = s

        margin = best_sim - second_best if second_best > -0.5 else 1e9

        if best_id is not None and best_sim >= self.sim_thresh and margin >= self.min_match_margin:
            g = self.gates[best_id]
            g.last_seen_t = now
            g.last_sim = best_sim
            if img_path:
                g.last_img = img_path

            # update proto only if very confident to prevent drift
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
                # In race mode, a lap boundary often means "start over expected order"
                if self.mode == "race" and self._memory:
                    self._expected_idx = 0
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


# ============================================================
# UI panel
# ============================================================

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

    mode = getattr(gatedb, "mode", "learn")
    memsz = getattr(gatedb, "memory_size", lambda: 0)()
    expi = getattr(gatedb, "memory_expected_index", lambda: 0)()

    cv2.putText(
        panel,
        f"t={now:.3f}s | gates={len(gatedb.gates)} | laps={gatedb.lap_count} | sinceLap={gatedb._passes_since_lap} | mode={mode} | mem={memsz} | exp={expi}",
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

    # compute per-gate best/2nd-best similarity among gates of same type
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


# ============================================================
# Viz-only helpers
# ============================================================

def best_gate_match_viz_only(gatedb: GateDB, gate_type: str, emb: np.ndarray) -> Tuple[Optional[int], float]:
    """
    Returns (best_gate_id or None, best_sim). Does NOT create a new gate.
    Uses memory bias in race mode (same as match_or_create memory-first),
    but without creating.
    """
    gate_type = str(gate_type)
    emb = gatedb._l2norm(np.asarray(emb))

    # try memory candidates first
    mem_cands = gatedb._memory_candidates(gate_type)
    if mem_cands:
        best_id = None
        best_sim = -1.0
        for m in mem_cands:
            s = float(np.dot(emb, m.proto))
            if s > best_sim:
                best_sim = s
                best_id = int(m.gate_id)
        if best_id is not None:
            return best_id, float(best_sim)

    # fallback to gates dict
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
    if len(gatedb.gates) == 0 and gatedb.memory_size() == 0:
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
