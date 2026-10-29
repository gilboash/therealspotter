"""
Where dataset files live — shared by every UI and script.

Nothing generated from videos is written inside the code repo.  Everything
lives under one data root, one folder per track.  Every file or folder made
from a video starts with the video's name (<video>.<what>), so one video's
data sorts together:

    <data_root>/
      <track>/                       e.g. track1
        gate_memory.json             the track's learned gates
        memory_videos/               videos used to learn the gates
        test_videos/                 videos used only for scoring
        learn_videos/                unreviewed videos the track is learned from (learn_track.py)
        videos/                      race_day: runs recorded live or cut from 2×2 replays
        race_day.json                race_day: setup + state of the race day
        candidates/<video>.candidates.json   learn_ui candidate extraction
        candidates/<video>.crops/            candidate / Force Clip crops (memory images)
        runs/<video>.race_data.json          extract_race.py output
        runs/<video>.race_query/             its query crops
        runs/<video>.motion.json             per-frame camera motion (extract_motion.py)
        runs/<video>.learn_passes.json       passes of a learn video (learn_track.py, never shown in UIs)
        gt/<video>.gt.json                   reviewed marks (learn_ui REVIEW mode)
        velocidrone/                 same layout, for the sim version of the track
      training/                      shared YOLO data (gate_annotator)
        gate_annotations/  gate_models/  runs/

Data root: $FPV_DATA_ROOT, else "data_root" in local_config.json next to this
file, else /Users/eyalcozac/Codes and apps/FPVdatasets.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_DATA_ROOT = "/Users/eyalcozac/Codes and apps/FPVdatasets"
VIDEO_FILTER = "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.MOV)"
SIM_DIR = "velocidrone"
TRAINING_DIR = "training"
RECORDINGS_DIR = "live_recordings"     # live feeds saved by the Live tab, per new track (not a track itself)


def data_root() -> Path:
    env = os.environ.get("FPV_DATA_ROOT")
    if env:
        return Path(env)
    cfg = Path(__file__).parent / "local_config.json"
    if cfg.exists():
        try:
            val = json.loads(cfg.read_text(encoding="utf-8")).get("data_root")
            if val:
                return Path(val)
        except Exception:
            pass
    return Path(DEFAULT_DATA_ROOT)


def recordings_dir() -> Path:
    """<data root>/live_recordings/<new track>/<pilot>_<time>.mp4: live feeds saved for learning a track later."""
    return data_root() / RECORDINGS_DIR


def training_dir() -> Path:
    return data_root() / TRAINING_DIR


def under_root(path) -> Optional[Path]:
    """Path relative to the data root, or None if it lies outside.
    Tries the path as given first, so symlinked videos inside the dataset count."""
    root = data_root()
    for p, r in ((Path(os.path.abspath(path)), Path(os.path.abspath(root))),
                 (Path(path).resolve(), root.resolve())):
        try:
            return p.relative_to(r)
        except ValueError:
            pass
    return None


def rel(path) -> str:
    """Path relative to the data root (unchanged if it lies outside)."""
    if not path:
        return ""
    r = under_root(path)
    return str(r) if r is not None else str(path)


def abs_path(path) -> str:
    """Inverse of rel(): resolve a data-root-relative path."""
    if not path:
        return ""
    p = Path(path)
    return str(p if p.is_absolute() else data_root() / p)


@dataclass(frozen=True)
class TrackPaths:
    """All file locations for one track folder (or its velocidrone/ subfolder)."""
    dir: Path

    @property
    def name(self) -> str:
        return f"{self.dir.parent.name} (sim)" if self.dir.name == SIM_DIR else self.dir.name

    # The learned track is per model (model_profiles.py): v2's files as always, another model's with its tag
    # (gate_memory.v3.json …). A model that hasn't learned the track yet reads v2's (*_own: where it writes).
    def _own(self, name: str) -> Path:
        import model_profiles
        tag = model_profiles.files_tag()
        return self.dir / (name[:-5] + tag + ".json" if tag else name)

    def _learned_file(self, name: str) -> Path:
        own = self._own(name)
        return own if own.exists() else self.dir / name

    @property
    def learned_own(self) -> bool:
        """The picked model has learned this track itself (v2: always its own files)."""
        import model_profiles
        return not model_profiles.files_tag() or self.gate_memory_own.exists()

    @property
    def gate_memory(self) -> Path:
        return self._learned_file("gate_memory.json")

    @property
    def gate_memory_own(self) -> Path:
        return self._own("gate_memory.json")

    @property
    def memory_videos(self) -> Path:
        return self.dir / "memory_videos"

    @property
    def test_videos(self) -> Path:
        return self.dir / "test_videos"

    @property
    def videos(self) -> Path:
        # race_day: every run recorded (Live) or cut from a 2×2 file (Replays)
        return self.dir / "videos"

    @property
    def race_day_state(self) -> Path:
        return self.dir / "race_day.json"

    @property
    def learn_videos(self) -> Path:
        return self.dir / "learn_videos"

    @property
    def learned_memory(self) -> Path:
        # learn_track.py: gates (order, type, images) learned from learn_videos/
        return self._learned_file("gate_memory.learned.json")

    @property
    def learned_timing(self) -> Path:
        # learn_track.py: gate-to-gate durations learned from learn_videos/
        return self._learned_file("gate_timing.learned.json")

    @property
    def learned_memory_own(self) -> Path:
        return self._own("gate_memory.learned.json")

    @property
    def learned_timing_own(self) -> Path:
        return self._own("gate_timing.learned.json")

    @property
    def gate_layout(self) -> Path:
        # learn_layout.py: where the gates roughly are (a top view), moved by hand in the web app
        return self._learned_file("gate_layout.json")

    @property
    def gate_layout_own(self) -> Path:
        return self._own("gate_layout.json")

    @property
    def gt_dir(self) -> Path:
        return self.dir / "gt"

    @property
    def runs_dir(self) -> Path:
        """The analyses of the model picked (model_profiles.py): runs/ (v2), runs_v3/ (v3)."""
        import model_profiles
        return self.dir / model_profiles.runs_name()

    @property
    def candidates_dir(self) -> Path:
        return self.dir / "candidates"

    def gt(self, stem: str) -> Path:
        return self.gt_dir / f"{stem}.gt.json"

    def race_data(self, stem: str) -> Path:
        return self.runs_dir / f"{stem}.race_data.json"

    def learn_passes(self, stem: str) -> Path:
        # learn_track.py: passes of a learn video, found with no track knowledge.
        # Kept apart from race_data.json so Review / race_ui never load them.
        return self.runs_dir / f"{stem}.learn_passes.json"

    def motion(self, stem: str) -> Path:
        # extract_motion.py: per-frame camera motion from optical flow
        return self.runs_dir / f"{stem}.motion.json"

    def race_query_dir(self, stem: str) -> Path:
        # extract_race.py writes this next to its --output
        return self.runs_dir / f"{stem}.race_query"

    def candidates_json(self, stem: str) -> Path:
        return self.candidates_dir / f"{stem}.candidates.json"

    def candidate_crops(self, stem: str) -> Path:
        return self.candidates_dir / f"{stem}.crops"

    def ensure(self) -> "TrackPaths":
        for d in (self.memory_videos, self.test_videos, self.gt_dir, self.runs_dir, self.candidates_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def track_of(path) -> Optional[TrackPaths]:
    """The track a file or folder inside the data root belongs to, else None."""
    r = under_root(path)
    root = data_root()
    if r is None or not r.parts or r.parts[0] in (TRAINING_DIR, RECORDINGS_DIR):
        return None
    if len(r.parts) == 1 and not (root / r.parts[0]).is_dir():
        return None                     # a loose file in the root, not a track
    if len(r.parts) >= 2 and r.parts[1] == SIM_DIR:
        return TrackPaths(root / r.parts[0] / SIM_DIR)
    return TrackPaths(root / r.parts[0])


def list_tracks() -> List[TrackPaths]:
    root = data_root()
    if not root.exists():
        return []
    return [TrackPaths(d) for d in sorted(root.iterdir())
            if d.is_dir() and d.name not in (TRAINING_DIR, RECORDINGS_DIR) and not d.name.startswith(".")]
