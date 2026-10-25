#!/usr/bin/env python3
"""
model_profiles.py — which model the pipeline runs with: a detector, a pass scorer and how passes are decided.

  v2   the detector of 2026-07-15 and the pass scorer trained on it (<data_root>/training/pass_scorer.json):
       one threshold decides the passes, the gate decoder names them. What this app has run since 2.0.
  v3   cozYnut's of 2026-10-21 (rest-api branch): a new detector, the pass scorer re-trained on its analyses of
       52 reviewed videos of 3 tracks (models/pass_scorer_v3.json), two stages — the scorer keeps almost every
       candidate (threshold 0.03) and the gate decoder drops the ones that don't fit the track, using each pass's
       p_pass, then shifts each pass by its gate's usual offset. Their numbers on the 52 videos: errors 331 → 149,
       false passes 162 → 55, gate ID 97.1% → 98.7%.

Each model keeps its own analyses (<track>/runs/ for v2, <track>/runs_v3/ for v3), so a video analysed with both
can be compared by switching. The learned tracks (gate memory, timing) are shared — learned with v2; learning is
v2's only (v3 would re-learn them from its own analyses).

The model picked: FPV_MODEL (the web app sets it for its jobs), else "model" in local_config.json, else v2.

    python model_profiles.py            # the models, and the one picked
    python model_profiles.py v3         # pick v3 (local_config.json)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent
DEFAULT = "v2"
PROFILES: Dict[str, dict] = {
    "v2": {"label": "v2 · detector 07-15 · one threshold", "detector": "cyn_current-20260715_best.pt",
           "scorer": None, "runs": "runs", "learn": True},
    "v3": {"label": "v3 · cozYnut 10-21 · two stages", "detector": "cyn_current-20261021_best.pt",
           "scorer": "models/pass_scorer_v3.json", "runs": "runs_v3", "learn": False},
}
_CFG = ROOT / "local_config.json"


_cached = (None, None)                  # (local_config.json's mtime, its "model")


def active() -> str:
    global _cached
    name = os.environ.get("FPV_MODEL")
    if not name and _CFG.exists():
        try:
            mt = _CFG.stat().st_mtime
            if _cached[0] != mt:
                _cached = (mt, json.loads(_CFG.read_text(encoding="utf-8")).get("model"))
            name = _cached[1]
        except (OSError, ValueError):
            name = None
    return name if name in PROFILES else DEFAULT


def profile(name: str = None) -> dict:
    return PROFILES[name or active()]


def set_active(name: str):
    if name not in PROFILES:
        raise ValueError(f"model must be one of {', '.join(PROFILES)}")
    cfg = {}
    if _CFG.exists():
        try:
            cfg = json.loads(_CFG.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
    cfg["model"] = name
    _CFG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.environ["FPV_MODEL"] = name


def detector_path(name: str = None) -> Path:
    return ROOT / profile(name)["detector"]


def scorer_path(name: str = None) -> Path:
    s = profile(name)["scorer"]
    if s:
        return ROOT / s
    from dataset_paths import training_dir
    return training_dir() / "pass_scorer.json"


def runs_name(name: str = None) -> str:
    return profile(name)["runs"]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        set_active(sys.argv[1])
    for k, p in PROFILES.items():
        print(f"{'→' if k == active() else ' '} {k}: {p['label']}  ({p['detector']}, scorer {scorer_path(k)})")
