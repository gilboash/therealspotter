#!/usr/bin/env python3
"""
box_remote.py — run the live spotter on a naco media box (Raspberry Pi 5) and control it from the Mac.

The Mac's web app stays the control and the display; the box is only a compute machine. Everything on the box lives
in one temporary folder (~/lazyspotter_tmp: its own Python environment, the code, the models, the track bundle,
the video) and clean() deletes it. Nothing is installed elsewhere on the box and its own app is not touched.

  - config: the box's address and SSH user, set from the page and kept only in local_config.json "box" (not in
    the repo); discover() finds boxes on the network with the box's own discovery (UDP broadcast on port 8557).
  - deploy: the code, the detector and CLIP exported to ONNX (made once on the Mac, exports/), the track bundle
    (gate_memory.json + gate_timing.learned.json) and the video; a file already there unchanged is not sent again.
  - run: online_spotter.py on the box over SSH, every event as a JSON line back (passes, gate names, laps, stats,
    preview frames); stopping the SSH session stops it.

Uses the Mac's SSH key (the box's user must accept it); the box's host key is kept in .box_known_hosts here.
"""

import hashlib
import json
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
KNOWN_HOSTS = ROOT / ".box_known_hosts"
CONFIG = ROOT / "local_config.json"
REMOTE = "lazyspotter_tmp"                    # in the box user's home
SPOTTER_PAT = "[l]azyspotter_tmp/code/online_spotter.py"   # [l]: pgrep/pkill must not match their own command line
PRESENCE_PORT = 8557
DEFAULT = {"host": "", "user": ""}             # no default: found (discover) or typed in, saved locally
CODE = ["online_spotter.py", "online_decoder.py", "gate_decoder.py", "pass_detector.py", "track_core.py",
        "pipeline_cfg.py", "dataset_paths.py", "box_runtime.py", "debug_ui_defaults.json"]
PIP = ["numpy", "opencv-python-headless", "pillow", "ncnn", "onnxruntime", "onnx"]
CLIP_ONNX = "clip_vitb32_openai.onnx"
CLIP_INT8 = "clip_vitb32_openai_int8.onnx"         # quantised here: on the 1 GB box it doesn't finish

Progress = Callable[[str], None]


# ── configuration and discovery ──────────────────────────────────

def get_config() -> dict:
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    return {**DEFAULT, **(cfg.get("box") or {})}


def set_config(host: str, user: str) -> dict:
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    cfg["box"] = {"host": host.strip(), "user": user.strip() or DEFAULT["user"]}
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return get_config()


def discover(timeout: float = 1.5) -> List[dict]:
    """Boxes answering the box's own presence probe (UDP broadcast)."""
    found: Dict[str, dict] = {}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(0.2)
        s.sendto(json.dumps({"naco": "who", "v": 1}).encode(), ("255.255.255.255", PRESENCE_PORT))
        end = time.time() + timeout
        while time.time() < end:
            try:
                data, addr = s.recvfrom(4096)
                msg = json.loads(data.decode("utf-8", "replace"))
            except (socket.timeout, ValueError):
                continue
            if isinstance(msg, dict) and msg.get("naco") == "here":
                found.setdefault(msg.get("name") or addr[0], {"name": msg.get("name") or addr[0], "addr": addr[0],
                                                               "ver": msg.get("ver"), "how": "discovery"})
        s.close()
    except OSError:
        pass
    return list(found.values())


# ── local files ──────────────────────────────────────────────────

_sha_cache: Dict[tuple, str] = {}


def sha1(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if key not in _sha_cache:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def export_models(det_model: Path, imgsz: int, log: Progress) -> Dict[str, Path]:
    """The detector at imgsz wide (16:9 input) and CLIP's image encoder as ONNX, made once into exports/."""
    import math
    EXPORTS.mkdir(exist_ok=True)
    h = int(math.ceil(imgsz * 9 / 16 / 32) * 32)
    det = EXPORTS / f"{Path(det_model).stem}_{imgsz}x{h}.onnx"
    if not det.exists() or det.stat().st_mtime < Path(det_model).stat().st_mtime:
        log(f"exporting the detector to ONNX ({imgsz}×{h})…")
        import shutil
        from ultralytics import YOLO
        shutil.move(YOLO(str(det_model)).export(format="onnx", imgsz=[h, imgsz], simplify=False, device="cpu",
                                                verbose=False), det)
    clip = EXPORTS / CLIP_ONNX
    if not clip.exists():
        log("exporting CLIP's image encoder to ONNX…")
        import open_clip
        import torch
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        m.eval()

        class Visual(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                f = self.m.encode_image(x)
                return f / f.norm(dim=-1, keepdim=True)

        torch.onnx.export(Visual(m), torch.zeros(1, 3, 224, 224), str(clip), input_names=["image"],
                          output_names=["emb"], opset_version=17, dynamo=False)
    q = EXPORTS / CLIP_INT8
    if not q.exists() or q.stat().st_mtime < clip.stat().st_mtime:
        log("quantising CLIP to int8…")
        from onnxruntime.quantization import QuantType, quantize_dynamic
        part = q.with_name(q.stem + ".part.onnx")
        quantize_dynamic(str(clip), str(part), weight_type=QuantType.QInt8)
        part.replace(q)
    return {"detector": det, "clip": q}


# ── the box ──────────────────────────────────────────────────────

class Box:
    def __init__(self, host: Optional[str] = None, user: Optional[str] = None):
        cfg = get_config()
        self.host, self.user = host or cfg["host"], user or cfg["user"]
        if not self.host or not self.user:
            raise RuntimeError("Set the media box's address and user first (Run on: media box → Find, or type them).")
        self.proc: Optional[subprocess.Popen] = None

    def _ssh(self) -> List[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
                "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=5", f"{self.user}@{self.host}"]

    def sh(self, cmd: str, timeout: float = 60, check: bool = True) -> str:
        r = subprocess.run(self._ssh() + [cmd], capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"box: {(r.stderr or r.stdout).strip()[-300:] or f'exit {r.returncode}'}")
        return r.stdout

    def status(self) -> dict:
        """Reachable? Memory, CPU temperature, what's in the temporary folder, whether a spotter runs."""
        try:
            out = self.sh(f"cat /proc/device-tree/model 2>/dev/null; echo; free -m | awk '/Mem:/{{print $2, $7}}'; "
                          f"cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0; "
                          f"(du -sm ~/{REMOTE} 2>/dev/null || echo 0) | cut -f1; "
                          f"pgrep -f '{SPOTTER_PAT}' >/dev/null && echo running || echo idle",
                          timeout=12)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            return {"reachable": False, "host": self.host, "error": str(e)}
        lines = [x.strip() for x in out.splitlines()]
        model = lines[0].strip("\x00") if lines else ""
        mem = lines[1].split() if len(lines) > 1 else []
        temp = int(lines[2]) / 1000.0 if len(lines) > 2 and lines[2].isdigit() else None
        tmp_mb = int(lines[3]) if len(lines) > 3 and lines[3].isdigit() else 0
        return {"reachable": True, "host": self.host, "model": model,
                "mem_total_mb": int(mem[0]) if mem else None, "mem_free_mb": int(mem[1]) if len(mem) > 1 else None,
                "cpu_temp": round(temp, 1) if temp else None, "temp_folder_mb": tmp_mb,
                "spotter_running": lines[-1] == "running" if lines else False}

    # files
    def _manifest(self) -> dict:
        try:
            return json.loads(self.sh(f"cat ~/{REMOTE}/manifest.json 2>/dev/null || echo '{{}}'", timeout=20) or "{}")
        except ValueError:
            return {}

    def upload(self, local: Path, rel: str, manifest: dict, log: Progress, label: Optional[str] = None) -> bool:
        """local → ~/lazyspotter_tmp/<rel>, unless the same file is already there. True if it was sent."""
        sig = f"{sha1(local)}:{local.stat().st_size}"
        if manifest.get(rel) == sig:
            return False
        dst = f"~/{REMOTE}/{rel}"
        size, sent, last = local.stat().st_size, 0, -1
        p = subprocess.Popen(self._ssh() + [f"mkdir -p $(dirname {dst}) && cat > {dst}.part && mv {dst}.part {dst}"],
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(local, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                p.stdin.write(chunk)
                sent += len(chunk)
                pct = 100 * sent // size
                if size > 4 << 20 and pct // 10 != last:
                    log(f"sending {label or local.name}: {pct}% of {size >> 20} MB")
                    last = pct // 10
        p.stdin.close()
        if p.wait(timeout=600) != 0:
            raise RuntimeError(f"box: sending {local.name} failed: {p.stderr.read().decode()[-300:]}")
        manifest[rel] = sig
        return True

    def setup(self, log: Progress):
        """The temporary folder and its Python environment (once; --no-cache-dir so pip leaves nothing behind)."""
        if "ok" in self.sh(f"test -f ~/{REMOTE}/venv/.ready && echo ok || true", timeout=20):
            return
        log("setting up a temporary Python environment on the box (first time, ~1–2 min)…")
        self.sh(f"mkdir -p ~/{REMOTE} && cd ~/{REMOTE} && rm -rf venv && python3 -m venv venv && "
                f"venv/bin/pip install -q --no-cache-dir {' '.join(PIP)} && "
                f"venv/bin/pip install -q --no-cache-dir --no-deps pnnx && touch venv/.ready", timeout=900)

    def deploy(self, track_dir: Optional[Path], track_name: str, video: Path, models: Dict[str, Path],
               log: Progress) -> dict:
        """track_dir None: no track bundle (passes only)."""
        self.setup(log)
        man = self._manifest()
        for f in CODE:
            self.upload(ROOT / f, f"code/{f}", man, log)
        for k in ("detector", "clip"):
            self.upload(models[k], f"models/{models[k].name}", man, log, label=f"the {k} model")
        tdir = f"tracks/{track_name}"
        for f in ("gate_memory.json", "gate_timing.learned.json") if track_dir else ():
            src = Path(track_dir) / f
            if src.exists():
                self.upload(src, f"{tdir}/{f}", man, log, label=f"the track ({f})")
        self.upload(video, f"videos/{video.name}", man, log, label=video.name)
        self.sh(f"cat > ~/{REMOTE}/manifest.json <<'EOF'\n{json.dumps(man)}\nEOF", timeout=20)
        return {"video": f"../videos/{video.name}", "detector": f"../models/{models['detector'].name}",
                "clip": f"../models/{models['clip'].name}", "track": f"../{tdir}"}

    # running
    def start(self, paths: dict, imgsz: int, stride: int, speed: float, preview: int = 3,
              realtime: bool = True) -> subprocess.Popen:
        """realtime: the video plays at its own pace and frames the box can't take in time are dropped; otherwise
        every (stride-th) frame is analysed at the box's own pace, slower than real time."""
        args = ["../venv/bin/python", "online_spotter.py", paths["video"], "--box-detector", paths["detector"],
                "--box-clip", paths["clip"], "--track-dir", paths["track"], "--imgsz", str(imgsz),
                "--stride", str(stride), "--speed", str(speed), "--jsonl", "--preview", str(preview), "--exit-on-eof"]
        if not realtime:
            args.append("--fast")
        cmd = f"cd ~/{REMOTE}/code && exec {' '.join(shlex.quote(a) for a in args)}"
        self.proc = subprocess.Popen(self._ssh() + [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        return self.proc

    def stop(self):
        p, self.proc = self.proc, None
        if p is not None:
            try:
                p.stdin.close()               # the box's spotter stops when its stdin closes
            except OSError:
                pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.terminate()
        try:
            self.sh(f"pkill -f '{SPOTTER_PAT}' || true", timeout=10, check=False)
        except subprocess.TimeoutExpired:
            pass

    def clean(self) -> dict:
        """Stop any spotter and delete the whole temporary folder."""
        self.stop()
        self.sh(f"rm -rf ~/{REMOTE}", timeout=60)
        return self.status()


def read_events(proc: subprocess.Popen, on_event: Callable[[dict], None], on_log: Callable[[str], None]):
    """The box's JSON lines → on_event (blocking until the box's spotter ends); stderr → on_log."""
    def errs():
        for line in proc.stderr:
            on_log(line.rstrip())
    threading.Thread(target=errs, daemon=True).start()
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            on_event(json.loads(line))
        except ValueError:
            continue
    proc.wait()
