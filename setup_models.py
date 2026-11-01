#!/usr/bin/env python3
"""
Download the models that don't ship in the repo, once, after `pip install -r requirements.txt`:

    ./venv/bin/python setup_models.py

Today that is one model: CLIP (open_clip ViT-B-32 'openai', about 605 MB from Hugging Face), which tells the
learned gates apart. The gate detectors and pass scorers ship in the repo. Without this step the app downloads
CLIP itself the first time it learns, analyses or goes Live, and says so in the terminal.
"""

import os
import sys
from pathlib import Path

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_REPO = "timm/vit_base_patch32_clip_224.openai"     # where open_clip takes these weights from
CLIP_FILE = "open_clip_model.safetensors"
CLIP_MB = 605

FIXES = f"""\
  How to fix it:
  - SSL / certificate error (python.org Python on macOS): run "/Applications/Python 3.x/Install Certificates.command"
    once, then try again.
  - No internet, a proxy or huggingface.co blocked: try another network, or copy the folder
      ~/.cache/huggingface/hub/models--{CLIP_REPO.replace('/', '--')}
    from a computer that has it to the same place here, and set HF_HUB_OFFLINE=1.
  - 429 Too Many Requests: wait a few minutes and try again (or log in: huggingface-cli login).
  Then run: ./venv/bin/python setup_models.py"""


def hub_cache() -> str:
    """The Hugging Face cache folder the download goes to (HF_HUB_CACHE / HF_HOME, or ~/.cache/huggingface/hub)."""
    return os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")


def clip_cached() -> bool:
    """Is CLIP already downloaded, so loading it needs no internet?"""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(CLIP_REPO, CLIP_FILE), str)
    except Exception:
        return (Path(hub_cache()) / f"models--{CLIP_REPO.replace('/', '--')}").exists()


def ensure_clip() -> bool:
    """Download CLIP if it isn't here yet, with a progress bar in the terminal. True when it's ready."""
    if clip_cached():
        return True
    print(f"\nDownloading the CLIP image model ({CLIP_MB} MB, only this once) from Hugging Face\n"
          f"  {CLIP_REPO} → {hub_cache()}", flush=True)
    try:
        from open_clip import pretrained
        pretrained.download_pretrained(pretrained.get_pretrained_cfg(CLIP_MODEL, CLIP_PRETRAINED))
    except Exception as e:
        print(f"\nCould not download the CLIP model: {type(e).__name__}: {e}\n{FIXES}\n", file=sys.stderr, flush=True)
        return False
    print("CLIP downloaded.\n", flush=True)
    return True


def main() -> int:
    ok = ensure_clip()
    if ok:
        print(f"CLIP is ready ({CLIP_REPO}).")
    detectors = sorted(p.name for p in Path(__file__).parent.glob("cyn_current-*.pt"))
    print(f"Gate detectors in the repo: {', '.join(detectors) or 'none found'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
