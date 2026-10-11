"""
Scene embeddings: CLIP of the whole frame just before each pass.

The pass crop (query_embedding) shows the gate; the scene shows where on the
track the drone is — the view of the start gate with the tent beside it, the
trees behind. Within one video every crossing of the same gate has nearly the
same scene, so the start gate's crossings can be checked against each other
(lap_learner.label), even when another camera makes the learned images a poor
match.

    add_scene_embeddings(video, passes)         # in place, for passes missing one
    ensure_file(video, race_data_json)          # the same for a saved analysis (saves it back)
"""

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

SCENE_OFFSET = 0.3          # seconds before the pass: the gate ahead, not yet filling the frame

_clip = None


def _embedder(clip=None):
    global _clip
    if clip is not None:
        return clip
    if _clip is None:
        import torch
        from lazy_spotter import ClipEmbedder
        _clip = ClipEmbedder(device="mps" if torch.backends.mps.is_available() else "cpu")
    return _clip


def add_scene_embeddings(video: str, passes: List[dict], clip=None) -> int:
    """Adds "scene_embedding" to every pass that has none. Returns how many were added."""
    todo = [p for p in passes if not p.get("scene_embedding")]
    if not todo:
        return 0
    emb = _embedder(clip)
    cap = cv2.VideoCapture(str(video))
    n = 0
    for p in sorted(todo, key=lambda p: p["t"]):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(p["t"]) - SCENE_OFFSET) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        p["scene_embedding"] = [round(float(x), 5) for x in emb.embed_bgr(frame)]
        n += 1
    cap.release()
    return n


def ensure_file(video: str, data_file, clip=None) -> int:
    """Adds missing scene embeddings to a saved analysis (passes) and saves it."""
    f = Path(data_file)
    data = json.loads(f.read_text(encoding="utf-8"))
    n = add_scene_embeddings(video, data.get("passes", []), clip)
    if n:
        f.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return n


def norm(e) -> Optional[np.ndarray]:
    if e is None or len(e) == 0:
        return None
    v = np.asarray(e, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)
