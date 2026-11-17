"""
Analysis workers, in two pools:

  heavy  analysing videos — by default ONE process, which analyses a heat's
         pilots together (race_day/heat_analysis.py: the heat video decoded
         once, by the hardware decoder; YOLO on the Neural Engine; only the
         race; no YOLO for pilots with no picture). ~1.3 GB, where a process
         per pilot took ~3 GB each (11.6 GB for four).
  light  labelling and learning — one process with no models (fast, ~0.1 GB),
         so they never wait behind a heat being analysed.

Each run goes through extract_race.run_race_extraction with a blank gate
memory: this finds the passes (pass scorer) and their gate images; gate IDs
and laps are added afterwards by labeling.py, once the track is learned.

Everything heavy runs here, never in the app: analysing (a heat's pilots,
read in place from the heat video), labelling and learning. pause() stops the
worker processes outright (SIGSTOP) — used while FPVTrackside runs a race, so
it has the computer to itself — and resume() lets them continue where they
were.
"""

import contextlib
import json
import multiprocessing as mp
import os
import signal
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

_W = {}          # per worker process: model path, device, clip device


def _init(slots, clip_device: str):
    path, device = slots.get()
    _W.update(model=path, device=device, clip=clip_device)


def _blank_memory(n_gates: int) -> str:
    mem = {"version": 2, "mode": "blank", "race_lookahead": 3, "expected_idx": 0, "max_embeds_per_gate": 6,
           "memory": [{"order_idx": i, "gate_id": i + 1, "gate_type": "unknown", "embeds": [],
                       "created_t": 0.0, "last_img": "", "embed_imgs": []} for i in range(n_gates)]}
    fd, p = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(mem, f)
    return p


def _analyse(video: str, out_json: str, n_gates: int, log_path: str) -> dict:
    from extract_race import run_race_extraction
    t0 = time.time()
    mem = _blank_memory(n_gates)
    try:
        with open(log_path, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            run_race_extraction(video, _W["model"], mem, out_json, clip_device=_W["clip"],
                                gate_id_logic="greedy", det_device=_W["device"])
    finally:
        os.unlink(mem)
    return {"seconds": round(time.time() - t0, 1),
            "device": "coreml" if _W["model"].endswith(".mlpackage") else _W["device"]}


def _analyse_quad(video: str, rect, out_json: str, n_gates: int, log_path: str, stem: str) -> dict:
    """One pilot's quadrant of a heat video, read in place (QuadReader)."""
    from extract_race import run_race_extraction
    from race_day.capture import QuadReader
    t0 = time.time()
    mem = _blank_memory(n_gates)
    try:
        with open(log_path, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            reader = QuadReader(video, rect)
            run_race_extraction(video, _W["model"], mem, out_json, clip_device=_W["clip"],
                                gate_id_logic="greedy", det_device=_W["device"], reader=reader, stem=stem)
        from race_day.capture import gray_intervals
        _write_picture(out_json, stem, {"duration": round(reader.frame_count / reader.fps, 2),
                                        "gray": gray_intervals(reader.gray_t, 1.0 / reader.fps)})
    finally:
        os.unlink(mem)
    return {"seconds": round(time.time() - t0, 1), "device": _W["device"]}


def _analyse_heat(video: str, runs: List[dict], n_gates: int, window=None) -> dict:
    """The pilots `runs` of one heat video, together (race_day/heat_analysis.py).
    runs: [{"stem", "rect", "out_json", "log"}]; window: (start, end) in video
    seconds, or None for the whole video → {stem: {"seconds", "device"} or {"error"}}.
    With the detector on the Neural Engine (Core ML), CLIP runs on the CPU: the
    process then never starts PyTorch on the GPU (~1.5 GB)."""
    from race_day.heat_analysis import analyse_heat
    clip = "cpu" if _W["model"].endswith(".mlpackage") else _W["clip"]
    res = analyse_heat(video, runs, _W["model"], n_gates, device=_W["device"], clip_device=clip,
                       blank_memory=_blank_memory, window=window)
    for r in runs:
        got = res.get(r["stem"]) or {"error": "not analysed"}
        if "picture" in got:
            _write_picture(r["out_json"], r["stem"], got.pop("picture"))
    return res


def _write_picture(race_json, stem: str, data: dict):
    """<stem>.picture.json: when the pilot's feed had no picture (gray)."""
    Path(race_json).with_name(f"{stem}.picture.json").write_text(json.dumps(data), encoding="utf-8")


def _label(track_dir: str, stems: List[str], quads=None, g1_hint=None) -> dict:
    from dataset_paths import TrackPaths
    from race_day import labeling
    from race_day.capture import sample_no_picture
    tp = TrackPaths(Path(track_dir))
    for s, (video, rect) in (quads or {}).items():      # runs analysed before pictures were recorded
        if not (tp.runs_dir / f"{s}.picture.json").exists():
            _write_picture(tp.race_data(s), s, sample_no_picture(video, rect))
    return {s: labeling.label_run(tp, s, (g1_hint or {}).get(s)) for s in stems}


def _learn(track_dir: str, stems: List[str], n_gates: int, log_path: str, g1_times=None) -> dict:
    from dataset_paths import TrackPaths
    from race_day import labeling
    t0 = time.time()
    with open(log_path, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        res = labeling.learn(TrackPaths(Path(track_dir)), stems, n_gates, verbose=True, g1_times=g1_times)
    res["seconds"] = round(time.time() - t0, 1)
    return res


class AnalysisPool:
    def __init__(self, devices: List[Tuple[str, Optional[str]]], clip_device: str = "mps"):
        """devices: (model path, device) per heavy worker (models.worker_devices)."""
        ctx = mp.get_context("spawn")
        self._mgr = ctx.Manager()
        slots = self._mgr.Queue()
        for d in devices:
            slots.put(d)
        self.pool = ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx,
                                        initializer=_init, initargs=(slots, clip_device))
        self.light = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
        self.n = len(devices)

    def submit_heat(self, video: Path, runs: List[dict], n_gates: int, window=None) -> Future:
        """A heat video's pilots, analysed together. runs: [{"stem", "rect",
        "out_json"}]; each pilot's log is <stem>.analysis.log next to its result.
        window: (start, end) in video seconds — only that part (the race)."""
        runs = [dict(r, rect=tuple(r["rect"]), out_json=str(r["out_json"]),
                     log=str(Path(r["out_json"]).with_name(f"{r['stem']}.analysis.log"))) for r in runs]
        return self.pool.submit(_analyse_heat, str(video), runs, n_gates, window)

    def submit(self, video: Path, out_json: Path, n_gates: int) -> Future:
        log = Path(out_json).with_name(Path(out_json).name.replace(".race_data.json", ".analysis.log"))
        return self.pool.submit(_analyse, str(video), str(out_json), n_gates, str(log))

    def submit_quad(self, video: Path, rect, out_json: Path, n_gates: int, stem: str) -> Future:
        log = Path(out_json).with_name(f"{stem}.analysis.log")
        return self.pool.submit(_analyse_quad, str(video), tuple(rect), str(out_json), n_gates, str(log), stem)

    def submit_label(self, track_dir: Path, stems: List[str], quads=None, g1_hint=None) -> Future:
        """quads: stem → (video, rect), to record missing no-picture intervals.
        g1_hint: stem → the timing system's start/finish crossings (video seconds)."""
        return self.light.submit(_label, str(track_dir), list(stems), quads, g1_hint)

    def submit_learn(self, track_dir: Path, stems: List[str], n_gates: int, g1_times=None) -> Future:
        log = Path(track_dir) / "learn.log"
        return self.light.submit(_learn, str(track_dir), list(stems), n_gates, str(log), g1_times)

    def _pids(self) -> List[int]:
        out = []
        for ex in (self.pool, self.light):
            procs = getattr(ex, "_processes", None) or {}
            out += [p.pid for p in procs.values() if p.pid]
        return out

    def pause(self):
        """Stop the workers where they are (no CPU, memory kept)."""
        for pid in self._pids():
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                pass

    def resume(self):
        for pid in self._pids():
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass

    def shutdown(self, wait: bool = True):
        self.resume()
        self.pool.shutdown(wait=wait, cancel_futures=not wait)
        self.light.shutdown(wait=wait, cancel_futures=not wait)
        self._mgr.shutdown()
