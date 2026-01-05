# gate_db.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


# ============================================================
# Data structures
# ============================================================

@dataclass
class GateInfo:
    gate_id: int
    gate_type: str
    proto: np.ndarray              # normalized (centroid-ish; for UI only)
    num_updates: int = 1
    first_seen_t: float = 0.0
    last_seen_t: float = 0.0
    last_pass_t: float = 0.0
    pass_count: int = 0
    last_sim: float = 0.0
    last_img: str = ""

    # decision/debug metadata
    last_match_source: str = ""    # "LEARN" | "RACE" | "NOMATCH"
    last_second_sim: float = 0.0
    last_margin: float = 0.0
    last_expected_idx: int = -1
    last_window_size: int = 0

    # memory linkage
    memory_index: int = -1         # index in ordered memory if present, else -1
    memory_embed_count: int = 0    # embeddings stored for that memory slot


@dataclass
class MemoryGate:
    """
    Ordered memory slot:
      - order_idx defines the lap order
      - gate_id is stable (we create once per slot)
      - embeds holds multiple embeddings collected over time (2-3+ per gate)
    """
    order_idx: int
    gate_id: int
    gate_type: str
    embeds: List[np.ndarray]       # list of normalized float32 vectors
    created_t: float = 0.0
    last_img: str = ""


# ============================================================
# GateDB
# ============================================================

class GateDB:
    """
    LEARN mode:
      - NO matching
      - user builds ordered memory slots (index 0..N-1)
      - confirm appends embedding into current expected slot
      - skip creates a placeholder slot (still advances)

    RACE mode:
      - NO creation
      - match current embedding to expected window of slots using embed banks
      - IMPORTANT: candidate set MUST be (expected_idx + [0..lookahead]) with wrap-around
      - IMPORTANT: sim per gate is max dot over ALL embeddings in that slot
    """

    def __init__(
        self,
        sim_thresh: float,
        require_same_type: bool = True,
        min_lap_gap_sec: float = 6.0,
        min_gates_between_laps: int = 2,
        start_gate_id: int = 1,
        # stability knobs (race)
        min_match_margin: float = 0.0,
        race_lookahead: int = 3,     # how many *ahead* to consider, inclusive window is lookahead+1
        max_embeds_per_gate: int = 6,  # bank size per memory slot
    ):
        self.sim_thresh = float(sim_thresh)
        self.require_same_type = bool(require_same_type)
        self.min_match_margin = float(min_match_margin)

        self.min_lap_gap_sec = float(min_lap_gap_sec)
        self.min_gates_between_laps = int(min_gates_between_laps)
        self.start_gate_id = int(start_gate_id)

        self.mode: str = "learn"  # "learn" or "race"
        self.race_lookahead: int = int(max(0, race_lookahead))  # allow 0 (only expected)

        self.max_embeds_per_gate = int(max(1, max_embeds_per_gate))

        self.gates: Dict[int, GateInfo] = {}
        self.next_gate_id = 1

        self._seen_gate_ids_this_lap: set[int] = set()

        # ordered memory
        self._memory: List[MemoryGate] = []
        self._expected_idx: int = 0

        # laps
        self.lap_count = 0
        self._last_lap_t = -1e9
        self._passes_since_lap = 0

        # UI events
        self.last_events: List[dict] = []
        self.max_events = 12

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _l2norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = float(np.linalg.norm(x) + 1e-9)
        return (x / n).astype(np.float32)

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # expects normalized

    def _track_len(self) -> int:
        return int(len(self._memory))

    def set_mode(self, mode: str):
        m = (mode or "").strip().lower()
        self.mode = "race" if m == "race" else "learn"

    def _start_new_lap_after_start_gate(self, now: float):
        """
        Called when start gate is passed.
        Lap increments, seen-set resets (start gate marked as seen),
        expected advances to start_idx+1 (NOT 0).
        """
        self.lap_count += 1
        self._last_lap_t = float(now)
        self._passes_since_lap = 0

        # reset seen gates for new lap
        self._seen_gate_ids_this_lap = {int(self.start_gate_id)}

        # expected should be right after the start gate slot (usually idx 0 -> 1)
        start_idx = 0
        for m in self._memory:
            if int(m.gate_id) == int(self.start_gate_id):
                start_idx = int(m.order_idx)
                break
        self._expected_idx = min(start_idx + 1, len(self._memory))

        # log
        self.last_events.append({"t": float(now), "evt": "LAP", "start_gate": int(self.start_gate_id)})
        if len(self.last_events) > self.max_events:
            self.last_events = self.last_events[-self.max_events:]

    def set_race_lookahead(self, k: int):
        self.race_lookahead = int(max(0, k))

    def memory_size(self) -> int:
        return int(len(self._memory))

    def memory_expected_index(self) -> int:
        return int(self._expected_idx)

    def _expected_window_indices(self) -> List[int]:
        """
        RACE candidate indices:
          expected_idx, expected_idx+1, ... expected_idx+lookahead  (wrap-around)
        If memory empty -> [].
        """
        n = self._track_len()
        if n <= 0:
            return []

        exp = int(self._expected_idx) % n
        look = int(max(0, self.race_lookahead))

        idxs: List[int] = []
        for k in range(look + 1):
            idxs.append((exp + k) % n)
        return idxs

    def memory_window_preview(self, max_items: int = 3) -> str:
        """
        Small UI helper used by vocab_spotter overlays.
        Shows gate_ids in expected window (race) or from expected forward (learn).
        """
        n = self._track_len()
        if n <= 0:
            return ""

        if str(self.mode).lower() == "race":
            idxs = self._expected_window_indices()
            gids = [self._memory[i].gate_id for i in idxs[:max_items]]
            more = "" if len(idxs) <= max_items else "…"
            return "[" + ",".join(f"G{g}" for g in gids) + more + "]"
        else:
            exp = int(self._expected_idx)
            gids = [self._memory[i].gate_id for i in range(exp, min(n, exp + max_items))]
            more = "" if (exp + max_items) >= n else "…"
            return "[" + ",".join(f"G{g}" for g in gids) + more + "]"

    def skip_expected_in_race(self, step: int = 1, now: float = 0.0):
        """
        RACE recovery: advance expected index by `step` (wrap-around), without creating anything.
        """
        n = int(len(self._memory))
        if n <= 0:
            return
        step = int(max(1, step))
        self._expected_idx = (int(self._expected_idx) + step) % n

        self.last_events.append({"t": float(now), "evt": "RACE_SKIP", "step": int(step), "exp": int(self._expected_idx)})
        if len(self.last_events) > self.max_events:
            self.last_events = self.last_events[-self.max_events:]

    # ----------------------------
    # GateInfo creation / refresh
    # ----------------------------
    def _ensure_gateinfo(self, gid: int, gate_type: str, proto: np.ndarray, now: float, img_path: str = "") -> GateInfo:
        gid = int(gid)
        gate_type = str(gate_type)
        proto = self._l2norm(proto) if (proto is not None and np.asarray(proto).size > 0) else np.zeros((1,), dtype=np.float32)

        if gid not in self.gates:
            self.gates[gid] = GateInfo(
                gate_id=gid,
                gate_type=gate_type,
                proto=proto.copy(),
                num_updates=1,
                first_seen_t=float(now),
                last_seen_t=float(now),
                last_sim=0.0,
                last_img=str(img_path or ""),
            )
            self.next_gate_id = max(self.next_gate_id, gid + 1)
        return self.gates[gid]

    def _refresh_memory_flags(self):
        mem_map = {int(m.gate_id): int(m.order_idx) for m in self._memory}
        mem_cnt = {int(m.gate_id): int(len(m.embeds)) for m in self._memory}

        for gid, g in self.gates.items():
            g.memory_index = int(mem_map.get(int(gid), -1))
            g.memory_embed_count = int(mem_cnt.get(int(gid), 0))

    def _update_gate_proto_from_bank(self, mg: MemoryGate) -> np.ndarray:
        """
        Compute a simple centroid proto from embeds bank.
        Used only for UI / uniqueness comparisons.
        """
        if not mg.embeds:
            return np.zeros((1,), dtype=np.float32)
        M = np.stack(mg.embeds, axis=0).astype(np.float32)
        v = np.mean(M, axis=0)
        return self._l2norm(v)

    # ----------------------------
    # Memory persistence
    # ----------------------------
    def save_memory(self, path: str):
        data = {
            "version": 2,
            "mode": self.mode,
            "race_lookahead": int(self.race_lookahead),
            "expected_idx": int(self._expected_idx),
            "max_embeds_per_gate": int(self.max_embeds_per_gate),
            "memory": [
                {
                    "order_idx": int(m.order_idx),
                    "gate_id": int(m.gate_id),
                    "gate_type": str(m.gate_type),
                    "embeds": [e.astype(np.float32).tolist() for e in m.embeds],
                    "created_t": float(m.created_t),
                    "last_img": str(m.last_img or ""),
                }
                for m in self._memory
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_memory(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded: List[MemoryGate] = []
        for i, d in enumerate(data.get("memory", []) or []):
            embeds_in = d.get("embeds", []) or []
            embeds: List[np.ndarray] = []
            for e in embeds_in:
                arr = np.asarray(e, dtype=np.float32)
                if arr.size == 0:
                    continue
                embeds.append(self._l2norm(arr))

            loaded.append(
                MemoryGate(
                    order_idx=int(d.get("order_idx", i)),
                    gate_id=int(d.get("gate_id", i + 1)),
                    gate_type=str(d.get("gate_type", "gate")),
                    embeds=embeds,
                    created_t=float(d.get("created_t", 0.0)),
                    last_img=str(d.get("last_img", "")),
                )
            )

        loaded.sort(key=lambda m: int(m.order_idx))
        for j, m in enumerate(loaded):
            m.order_idx = j

        self._memory = loaded
        self.race_lookahead = int(max(0, data.get("race_lookahead", self.race_lookahead)))
        # clamp expected idx into [0..len-1] if len>0 else 0
        if len(self._memory) > 0:
            self._expected_idx = int(min(max(0, data.get("expected_idx", 0)), len(self._memory) - 1))
            # NEW: in race mode we typically always start from 0
            if self.mode == "race" and len(self._memory) > 0:
                self._expected_idx = 0
        else:
            self._expected_idx = 0
        self.max_embeds_per_gate = int(max(1, data.get("max_embeds_per_gate", self.max_embeds_per_gate)))

        # ensure GateInfo exists for UI
        for m in self._memory:
            proto = self._update_gate_proto_from_bank(m) if m.embeds else np.zeros((1,), dtype=np.float32)
            g = self._ensure_gateinfo(m.gate_id, m.gate_type, proto, now=float(m.created_t), img_path=str(m.last_img))
            g.memory_index = int(m.order_idx)
            g.memory_embed_count = int(len(m.embeds))

        self._refresh_memory_flags()

    # ----------------------------
    # LEARN API (NO matching)
    # ----------------------------
    def reset_memory(self):
        self._memory = []
        self._expected_idx = 0
        self._refresh_memory_flags()

    def force_lap_end(self, now: float = 0.0):
        """
        Manual lap boundary.
        In LEARN: reset expected index to 0 and increment lap counter.
        In RACE: also reset expected index to 0.
        """
        self.lap_count += 1
        self._last_lap_t = float(now)
        self._passes_since_lap = 0
        self._expected_idx = 0
        self._seen_gate_ids_this_lap = set()


        self.last_events.append({"t": float(now), "evt": "LAP_END"})
        if len(self.last_events) > self.max_events:
            self.last_events = self.last_events[-self.max_events:]

    def learn_skip_expected_gate(self, now: float = 0.0) -> int:
        """
        Skip still creates an entry (placeholder slot) and advances expected index.
        Returns the gate_id of the created/selected slot.
        """
        idx = int(self._expected_idx)

        # ensure memory slot exists
        if idx >= len(self._memory):
            gid = int(self.next_gate_id)
            self.next_gate_id += 1
            mg = MemoryGate(
                order_idx=idx,
                gate_id=gid,
                gate_type="UNKNOWN",
                embeds=[],
                created_t=float(now),
                last_img="",
            )
            self._memory.append(mg)
            # ensure GateInfo for UI
            g = self._ensure_gateinfo(gid, "UNKNOWN", np.zeros((1,), dtype=np.float32), now=float(now))
            g.last_match_source = "LEARN"
            g.memory_index = idx
            g.memory_embed_count = 0

        gid_out = int(self._memory[idx].gate_id)
        # advance (learn is building the lap; clamp to len)
        self._expected_idx = min(self._expected_idx + 1, len(self._memory))
        self._refresh_memory_flags()

        self.last_events.append({"t": float(now), "evt": "SKIP", "idx": int(idx), "gate_id": int(gid_out)})
        if len(self.last_events) > self.max_events:
            self.last_events = self.last_events[-self.max_events:]
        return gid_out

    def learn_confirm_pass(
        self,
        now: float,
        gate_type: str,
        emb: np.ndarray,
        img_path: str = "",
    ) -> Tuple[int, float]:
        """
        Confirm the current pass into the CURRENT expected index:
          - if slot exists: append embedding to that slot
          - if slot doesn't exist: create new slot
        Returns (gate_id, sim=1.0) (sim is not meaningful in LEARN).
        """
        gate_type = str(gate_type)
        if gate_type == "NONE":
            return -1, 0.0
        if emb is None:
            return -1, 0.0

        embn = self._l2norm(emb)
        idx = int(self._expected_idx)

        # create slot if missing
        if idx >= len(self._memory):
            gid = int(self.next_gate_id)
            self.next_gate_id += 1
            mg = MemoryGate(
                order_idx=idx,
                gate_id=gid,
                gate_type=gate_type,
                embeds=[],
                created_t=float(now),
                last_img=str(img_path or ""),
            )
            self._memory.append(mg)
        else:
            mg = self._memory[idx]
            gid = int(mg.gate_id)
            # fill type if placeholder
            if str(mg.gate_type) in ("UNKNOWN", "NONE", ""):
                mg.gate_type = gate_type

        # append embedding into bank (ring buffer)
        mg.embeds.append(embn)
        if len(mg.embeds) > self.max_embeds_per_gate:
            mg.embeds = mg.embeds[-self.max_embeds_per_gate:]

        if img_path:
            mg.last_img = str(img_path)

        # update GateInfo proto from bank (for UI only)
        proto = self._update_gate_proto_from_bank(mg)
        g = self._ensure_gateinfo(gid, mg.gate_type, proto, now=float(now), img_path=str(mg.last_img))
        g.proto = proto.copy()
        g.last_seen_t = float(now)
        g.last_sim = 1.0
        g.last_match_source = "LEARN"
        g.last_expected_idx = idx
        g.last_window_size = 0
        g.memory_index = idx
        g.memory_embed_count = int(len(mg.embeds))
        if img_path:
            g.last_img = str(img_path)

        # advance expected
        self._expected_idx = min(self._expected_idx + 1, len(self._memory))
        self._refresh_memory_flags()

        # event log
        self.last_events.append({
            "t": float(now),
            "evt": "CONFIRM",
            "idx": int(idx),
            "gate_id": int(gid),
            "type": str(mg.gate_type),
            "embeds": int(len(mg.embeds)),
        })
        if len(self.last_events) > self.max_events:
            self.last_events = self.last_events[-self.max_events:]

        return int(gid), 1.0

    # ----------------------------
    # RACE matching (NO creation)
    # ----------------------------
    def _race_window_indices(self) -> List[int]:
        """
        Internal helper for panel/debug/viz: returns the exact candidate indices (wrap-around).
        """
        return self._expected_window_indices()

    def _race_window_span_for_panel(self) -> Tuple[int, int]:
        """
        Backward-compatible-ish helper for your panel:
        return (s, e) for the NON-wrapped contiguous span starting at expected,
        clamped to end. (This is *only* for display; matching uses wrap indices.)
        """
        n = self._track_len()
        if n <= 0:
            return 0, 0
        s = int(self._expected_idx) % n
        e = int(min(n, s + (max(0, int(self.race_lookahead)) + 1)))
        return s, e

    def _max_sim_to_bank(self, embn: np.ndarray, bank: List[np.ndarray]) -> float:
        """
        Max cosine similarity vs all embeddings stored for a slot.
        """
        if bank is None or len(bank) == 0:
            return -1.0
        best = -1.0
        for v in bank:
            s = float(np.dot(embn, v))
            if s > best:
                best = s
        return best

    def race_match(self, now: float, gate_type: str, emb: np.ndarray) -> Tuple[int, float, str, float, float, int, int]:
        """
        Returns:
          (gate_id, best_sim, source, second_sim, margin, exp_before, window_size)

        Candidate set is STRICTLY the expected window indices with wrap-around.
        Similarity per slot is max over its embeddings bank.

        source:
          - "RACE" if above threshold
          - "NOMATCH" otherwise
        """
        gate_type = str(gate_type)
        embn = self._l2norm(emb)

        exp_before = int(self._expected_idx)
        if not self._memory:
            return -1, 0.0, "NOMATCH", 0.0, 0.0, exp_before, 0

        idxs = self._race_window_indices()
        # build candidate list (preserve indices; do NOT slice)
        candidates: List[MemoryGate] = [self._memory[i] for i in idxs]

        # type filter (optional)
        if self.require_same_type:
            print("exit match for same type")
            candidates = [m for m in candidates if str(m.gate_type) == gate_type]

        window_size = int(len(candidates))
        if not candidates:
            print("exit match no candidates ")
            return -1, 0.0, "NOMATCH", 0.0, 0.0, exp_before, window_size

        best_sim = -1.0
        second_sim = -1.0
        best_m: Optional[MemoryGate] = None

        for m in candidates:
            print(" match candidate  ",int(m.gate_id))

            if not m.embeds:
                continue
            smax = self._max_sim_to_bank(embn, m.embeds)
            if smax > best_sim:
                second_sim = best_sim
                best_sim = smax
                best_m = m
            elif smax > second_sim:
                second_sim = smax


        if best_m is None:
            return -1, 0.0, "NOMATCH", 0.0, 0.0, exp_before, window_size

        margin = (best_sim - second_sim) if second_sim > -0.5 else 1e9
        ok = (best_sim >= self.sim_thresh) and (margin >= self.min_match_margin)

        gid = int(best_m.gate_id)

        if ok and gid == int(self.start_gate_id):
            # optional debounce (keep your knob)
            if (now - self._last_lap_t) >= self.min_lap_gap_sec:
                self._start_new_lap_after_start_gate(now=float(now))
            src = "RACE"
        # Enforce: each gate can only be counted once per lap
        elif ok and gid in self._seen_gate_ids_this_lap:
            ok = False
            src = "DUP"   # duplicate within lap
        else:
            src = "RACE" if ok else "NOMATCH"

        if ok:
            # mark seen
            self._seen_gate_ids_this_lap.add(gid)

            # advance expected to matched slot + 1
            self._expected_idx = min(int(best_m.order_idx) + 1, len(self._memory))

          




        print(" match best candidate is  ", gid)

        # update GateInfo decision fields for UI
        proto = self._update_gate_proto_from_bank(best_m) if best_m.embeds else np.zeros((1,), dtype=np.float32)
        g = self._ensure_gateinfo(gid, best_m.gate_type, proto, now=float(now), img_path=str(best_m.last_img))
        g.last_seen_t = float(now)
        g.last_sim = float(best_sim)
        g.last_match_source = src
        g.last_second_sim = float(second_sim if second_sim > -0.5 else 0.0)
        g.last_margin = float(margin if margin < 1e8 else 0.0)
        g.last_expected_idx = exp_before
        g.last_window_size = window_size
        g.memory_index = int(best_m.order_idx)
        g.memory_embed_count = int(len(best_m.embeds))

        self._refresh_memory_flags()
        return (
            gid,
            float(best_sim),
            src,
            float(second_sim if second_sim > -0.5 else 0.0),
            float(margin if margin < 1e8 else 0.0),
            exp_before,
            window_size,
        )

    # ----------------------------
    # Backward-compatible API name
    # ----------------------------
    def match_or_create(self, now: float, gate_type: str, emb: np.ndarray, img_path: str = "") -> Tuple[int, float, bool]:
        """
        Compatibility layer:
          - in LEARN: should NOT be used (use learn_confirm_pass)
          - in RACE: match only, never create new gates
        Returns (gate_id, sim, is_new=False)
        """
        if self.mode != "race":
            # safe fallback if caller forgot:
            gid, _sim = self.learn_confirm_pass(now=now, gate_type=gate_type, emb=emb, img_path=img_path)
            return int(gid), 1.0, False

        gid, sim, src, *_ = self.race_match(now=now, gate_type=gate_type, emb=emb)

        # IMPORTANT: only a real RACE match yields a valid gate_id.
        # NOMATCH or DUP should return -1 so caller will not count a pass.
        if src != "RACE":
            return -1, float(sim), False

        return int(gid), float(sim), False


    # ----------------------------
    # pass bookkeeping
    # ----------------------------
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
        if int(gate_id) < 0:
            return

        g = self.gates.get(int(gate_id))
        if g is not None:
            g.last_pass_t = float(now)
            g.pass_count += 1
            g.last_sim = float(sim)
            if img_path:
                g.last_img = str(img_path)

        # (optional) lap auto-bookkeeping based on start gate_id
        if int(gate_id) == int(self.start_gate_id):
            if (now - self._last_lap_t) >= self.min_lap_gap_sec and self._passes_since_lap >= self.min_gates_between_laps:
                self.lap_count += 1
                self._last_lap_t = float(now)
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
            "src": str(getattr(g, "last_match_source", "")) if g is not None else "",
            "second": float(getattr(g, "last_second_sim", 0.0)) if g is not None else 0.0,
            "margin": float(getattr(g, "last_margin", 0.0)) if g is not None else 0.0,
            "exp": int(getattr(g, "last_expected_idx", -1)) if g is not None else -1,
            "win": int(getattr(g, "last_window_size", 0)) if g is not None else 0,
        }
        self.last_events.append(evt)
        if len(self.last_events) > self.max_events:
            self.last_events = self.last_events[-self.max_events:]

    def summary_rows(self) -> List[GateInfo]:
        return [self.gates[k] for k in sorted(self.gates.keys())]


# ============================================================
# UI panel
# ============================================================

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
    memsz = gatedb.memory_size()
    expi = gatedb.memory_expected_index()
    look = int(getattr(gatedb, "race_lookahead", 0))

    cv2.putText(
        panel,
        f"t={now:.3f}s | gates={len(gatedb.gates)} | laps={gatedb.lap_count} | sinceLap={gatedb._passes_since_lap} | mode={mode} | mem={memsz} | exp={expi} | look={look}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        dim,
        2,
        cv2.LINE_AA,
    )
    y += 14

    # show race candidate indices/gates (wrap-aware)
    if mode == "race" and memsz > 0:
        idxs = gatedb._race_window_indices()
        gids = []
        for i in idxs[:6]:
            try:
                gids.append(int(gatedb._memory[int(i)].gate_id))
            except Exception:
                pass
        more = "…" if len(idxs) > 6 else ""
        cv2.putText(
            panel,
            f"race window idx={idxs[:6]}{more}  gids={[f'G{g}' for g in gids]}{more}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
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
        "gid  type       memIdx embeds passes lastPassAgo lastSim  src    2nd   marg",
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
    max_lines = max(1, (frame_h - y - 190) // line_h)
    rows = rows[:max_lines]

    for g in rows:
        last_pass_ago = (now - g.last_pass_t) if g.last_pass_t > 0 else 1e9
        src = (g.last_match_source or "---")[:7].ljust(7)
        memi = int(getattr(g, "memory_index", -1))
        mcnt = int(getattr(g, "memory_embed_count", 0))

        sline = (
            f"{g.gate_id:>3d}  {g.gate_type[:10].ljust(10)}  "
            f"{memi:>5d}  {mcnt:>6d}  {g.pass_count:>5d}  "
            f"{last_pass_ago:>10.2f}  {g.last_sim:>6.3f}  "
            f"{src}  {float(getattr(g,'last_second_sim',0.0)):>5.3f}  {float(getattr(g,'last_margin',0.0)):>5.3f}"
        )
        color = good if g.gate_id == gatedb.start_gate_id else text
        cv2.putText(panel, sline, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        y += line_h

    y += 8
    cv2.line(panel, (10, y), (panel_w - 10, y), (70, 70, 70), 1)
    y += 18
    cv2.putText(panel, "EVENTS (newest first):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text, 2, cv2.LINE_AA)
    y += 20

    evs = list(reversed(gatedb.last_events))[:10]
    for e in evs:
        if "evt" in e:
            s = f"{e.get('evt')}  t={e.get('t',0.0):.2f}  idx={e.get('idx','')} gate={e.get('gate_id','')}"
        else:
            s = (
                f"G{e['gate_id']} {e['type']:<8} sim={e['sim']:.3f} "
                f"2nd={e.get('second',0.0):.3f} m={e.get('margin',0.0):.3f} "
                f"src={str(e.get('src','---')):<7} exp={int(e.get('exp',-1)):>2d} win={int(e.get('win',0)):>2d} "
                f"tid={e['track_id']} {e['reason']}"
            )
        cv2.putText(panel, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text, 2, cv2.LINE_AA)
        y += 18

    return panel


# ============================================================
# Viz-only helpers (GateID hints)
# ============================================================

def best_gate_match_viz_only(gatedb: GateDB, gate_type: str, emb: np.ndarray) -> Tuple[Optional[int], float]:
    """
    For visualization:
      - if race: evaluate only current expected window (wrap-aware indices)
      - if learn: evaluate all memory slots that have embeddings
    """
    gate_type = str(gate_type)
    embn = gatedb._l2norm(emb)

    if not gatedb._memory:
        return None, -1.0

    if str(gatedb.mode).lower() == "race":
        idxs = gatedb._race_window_indices()
        window = [gatedb._memory[i] for i in idxs]
    else:
        window = list(gatedb._memory)

    if gatedb.require_same_type:
        window = [m for m in window if str(m.gate_type) == gate_type]

    best_id = None
    best_sim = -1.0
    for m in window:
        if not m.embeds:
            continue
        smax = gatedb._max_sim_to_bank(embn, m.embeds)
        if smax > best_sim:
            best_sim = smax
            best_id = int(m.gate_id)

    return best_id, float(best_sim)


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
    if gatedb.memory_size() == 0:
        return {}
    if (frame_idx % max(1, viz_every_n_frames)) != 0:
        return {}

    def area_of(t) -> float:
        x1, y1, x2, y2 = t.bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    sorted_tracks = sorted(tracks, key=area_of, reverse=True)[:max(1, viz_max_tracks)]

    hints: Dict[int, Tuple[int, float]] = {}
    for tr in sorted_tracks:
        if getattr(tr, "locked_type", "NONE") == "NONE":
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
