#!/usr/bin/env python3
"""
The files the watch-only page (spotter_web.py --view) needs from the data folder, one path per line, relative to
the data root: what deploy-optiplex.sh copies to the other computer.

    python deploy/view_files.py fril-27-second-track [more tracks]       # --sizes: "<bytes>\t<mtime>\t<path>"

Per track: the learned track (gate_memory.json, gate_timing.learned.json, gate_layout.json), the combos' pilots'
names and the spotter's approvals; the analysed videos the page lists (races, combos with an analysed pilot,
learn videos raced in a combo) with their analyses (runs/<stem>.race_data.json or .learn_passes.json, .combo.json)
and the pictures the page shows (the passes', the learned gates'). Not the motion tracks, the learning's crops,
candidates, backups or other models' runs: nothing there is analysed again.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_paths import TrackPaths, data_root  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
TRACK_FILES = ("gate_memory.json", "gate_timing.learned.json", "gate_layout.json", "combo_pilots.json",
               "spotter_approved.json")


def _image(tp: TrackPaths, p: str):
    """An image path saved in an analysis (maybe of another computer): the file in this track, if there."""
    parts = Path(p).parts
    if tp.dir.name in parts:
        f = tp.dir.joinpath(*parts[parts.index(tp.dir.name) + 1:])
        if f.is_file():
            return f
    return None


def files(tp: TrackPaths):
    for name in TRACK_FILES:
        if (tp.dir / name).is_file():
            yield tp.dir / name
    images = set()
    if (tp.dir / "gate_memory.json").is_file():          # the learned gates' pictures
        for g in json.loads((tp.dir / "gate_memory.json").read_text(encoding="utf-8")).get("memory", []):
            images.update(g.get("embed_imgs", []))
    runs = tp.dir / "runs"
    for folder in ("test_videos", "videos", "combo_videos", "learn_videos"):
        d = tp.dir / folder
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            got = [p for p in (runs / f"{f.stem}.race_data.json", runs / f"{f.stem}.learn_passes.json",
                               runs / f"{f.stem}.combo.json") if p.is_file()]
            race = runs / f"{f.stem}.race_data.json"
            pilots = any(next(runs.glob(f"{glob.escape(f.stem)}_CH*.{k}.json"), None)
                         for k in ("race_data", "learn_passes"))
            if not (race in got or (folder == "learn_videos" and got) or (folder == "combo_videos" and pilots)):
                continue                          # not analysed: the page doesn't list it
            yield f
            yield from got
            for a in got:                                 # the passes' pictures (not the learning's crops)
                if not a.name.endswith(".combo.json"):
                    images.update(p["query_img"] for p in json.loads(a.read_text(encoding="utf-8")).get("passes", [])
                                  if p.get("query_img"))
    for p in sorted(images):
        f = _image(tp, p)
        if f is not None:
            yield f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--sizes", action="store_true")
    args = ap.parse_args()
    root = data_root()
    for t in args.tracks:
        tp = TrackPaths(root / t)
        if not tp.dir.is_dir():
            sys.exit(f"no track {t} in {root}")
        for f in files(tp):
            r = f.relative_to(root).as_posix()
            if args.sizes:
                st = f.stat()
                print(f"{st.st_size}\t{int(st.st_mtime)}\t{r}")
            else:
                print(r)


if __name__ == "__main__":
    main()
