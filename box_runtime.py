#!/usr/bin/env python3
"""
box_runtime.py — the live spotter's detector and CLIP without PyTorch, for the media box (Raspberry Pi 5).

Only numpy, OpenCV, Pillow, NCNN and ONNX Runtime: the box has 1 GB of RAM and no GPU the detector can use.
  - NcnnDetector: the YOLO detector exported to ONNX on the Mac (box_remote.py), converted to NCNN on the box
    (pnnx); fp16 on ARM. Letterbox, confidence filter, per-class NMS and box scaling as Ultralytics does them.
  - OrtClip: CLIP ViT-B/32's image encoder exported to ONNX, quantised to int8 on the box (ONNX Runtime);
    preprocessing as open_clip's (Pillow bicubic resize, centre crop, normalise), so its embeddings match the
    learned track's.
  - prepare(models_dir): the conversions, once per model (cached in the box's temporary folder).
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> List[int]:
    """Greedy NMS on xyxy boxes, highest score first (torchvision's: areas without +1)."""
    order = np.argsort(-scores)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0]); yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2]); yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ov = inter / (areas[i] + areas[r] - inter + 1e-9)
        order = r[ov <= iou]
    return keep


class NcnnDetector:
    """detect(frame) → [(x1, y1, x2, y2, conf, cls)] in frame pixels, like Ultralytics' predict."""

    def __init__(self, param: str, names: Dict[int, str], imgsz_hw: Tuple[int, int] = (192, 320),
                 threads: int = 4, iou: float = 0.7, max_det: int = 50):
        import ncnn
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.load_param(str(param))
        self.net.load_model(str(param).replace(".param", ".bin"))
        self.ncnn = ncnn
        self.names, self.hw, self.iou, self.max_det = names, tuple(imgsz_hw), iou, max_det

    def _letterbox(self, frame):
        H, W = frame.shape[:2]
        h, w = self.hw
        r = min(h / H, w / W)
        nw, nh = round(W * r), round(H * r)
        dw, dh = (w - nw) / 2, (h - nh) / 2
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (W, H) else frame
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        return x, r, left, top

    def __call__(self, frame: np.ndarray, conf: float = 0.25):
        x, r, left, top = self._letterbox(frame)
        with self.net.create_extractor() as ex:
            ex.input("in0", self.ncnn.Mat(x).clone())
            out = np.array(ex.extract("out0")[1])            # (4 + classes, anchors): cx cy w h, class scores
        scores = out[4:]
        cls = scores.argmax(0)
        sc = scores[cls, np.arange(scores.shape[1])]
        m = sc > conf
        if not m.any():
            return []
        b, sc, cls = out[:4, m].T, sc[m], cls[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        keep = _nms(xyxy + cls[:, None] * 7680.0, sc, self.iou)[: self.max_det]      # per class, as Ultralytics
        xyxy = (xyxy[keep] - [left, top, left, top]) / r
        H, W = frame.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
        return [(*map(float, bb), float(s), int(c)) for bb, s, c in zip(xyxy, sc[keep], cls[keep])]


class OrtClip:
    """embed_bgr(crop) → unit-length 512-d CLIP embedding, as lazy_spotter.ClipEmbedder."""

    def __init__(self, onnx_path: str, threads: int = 2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    @staticmethod
    def preprocess(bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
        w, h = im.size
        s = 224
        nw, nh = (s, int(s * h / w)) if w <= h else (int(s * w / h), s)      # shorter side → 224 (torchvision)
        im = im.resize((nw, nh), Image.BICUBIC)
        left, top = int(round((nw - s) / 2.0)), int(round((nh - s) / 2.0))
        im = im.crop((left, top, left + s, top + s))
        x = (np.asarray(im, np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def embed_bgr(self, bgr: np.ndarray) -> np.ndarray:
        e = self.s.run(None, {"image": self.preprocess(bgr)})[0][0]
        return (e / (np.linalg.norm(e) + 1e-9)).astype(np.float32)


def prepare(models: Path, detector_onnx: str, clip_onnx: str, hw=(192, 320), log=print) -> Tuple[Path, Path]:
    """The box-side model files, made once: NCNN from the detector's ONNX (pnnx), int8 CLIP (ONNX Runtime)."""
    models = Path(models)
    det = models / Path(detector_onnx).name
    stem = det.stem.replace("-", "_")                  # pnnx names its output so
    param = models / f"{stem}.ncnn.param"
    done = models / f"{stem}.ncnn.ok"                  # written last: a conversion that was cut off is redone
    if not done.exists() or done.stat().st_mtime < det.stat().st_mtime:
        log(f"converting the detector to NCNN ({det.name})…")
        import importlib.util
        exe = Path(importlib.util.find_spec("pnnx").submodule_search_locations[0]) / "pnnx"   # the binary only:
        # importing the package needs torch, which the box doesn't have
        subprocess.run([str(exe), str(det), f"inputshape=[1,3,{hw[0]},{hw[1]}]"], cwd=models, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in models.glob(stem + ".pnnx*"):
            f.unlink()
        for f in (models / f"{stem}_pnnx.py", models / f"{stem}_ncnn.py"):
            if f.exists():
                f.unlink()
        done.touch()
    clip = Path(models) / Path(clip_onnx).name
    if clip.stem.endswith("_int8"):                    # quantised on the Mac already (box_remote.py)
        return param, clip
    q = clip.with_name(clip.stem + "_int8.onnx")
    if not q.exists() or q.stat().st_mtime < clip.stat().st_mtime:
        log("quantising CLIP to int8…")
        from onnxruntime.quantization import QuantType, quantize_dynamic
        part = q.with_name(q.stem + ".part.onnx")
        quantize_dynamic(str(clip), str(part), weight_type=QuantType.QInt8)
        os.replace(part, q)
    return param, q


def detector_names(detector_onnx: str) -> Dict[int, str]:
    """Class names from the exported model's metadata (written by Ultralytics)."""
    import ast
    import onnx
    m = onnx.load(str(detector_onnx), load_external_data=False)
    meta = {p.key: p.value for p in m.metadata_props}
    return {int(k): str(v) for k, v in ast.literal_eval(meta.get("names", "{}")).items()}




if __name__ == "__main__":       # python box_runtime.py <models dir> <detector.onnx> <clip.onnx>: prepare only
    import sys
    p, q = prepare(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
    print(json.dumps({"detector": str(p), "clip": str(q), "names": detector_names(sys.argv[2])}))
