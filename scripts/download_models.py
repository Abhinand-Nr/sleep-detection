"""Download every pretrained model the pipeline needs into models/.

Usage:  python scripts/download_models.py
Files that already exist are skipped, so it is safe to run again.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# name -> (url, minimum expected size in bytes; catches HTML error pages saved as models)
MODELS: dict[str, tuple[str, int]] = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
        200_000,
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx",
        30_000_000,
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
        "float16/1/face_landmarker.task",
        3_000_000,
    ),
    "yolov8n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
        5_000_000,
    ),
}


def download(name: str, url: str, min_size: int) -> bool:
    """Download one file atomically (to .part, then rename). Returns True on success."""
    target = MODELS_DIR / name
    if target.exists() and target.stat().st_size >= min_size:
        print(f"[skip] {name} already present ({target.stat().st_size / 1e6:.1f} MB)")
        return True

    tmp = target.with_suffix(target.suffix + ".part")
    print(f"[get ] {name}\n       from {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:  # network errors, 404s, ...
        print(f"[FAIL] {name}: {exc}")
        tmp.unlink(missing_ok=True)
        return False

    size = tmp.stat().st_size
    if size < min_size:
        print(f"[FAIL] {name}: only {size} bytes, expected >= {min_size} (bad download?)")
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(target)
    print(f"[ ok ] {name} ({size / 1e6:.1f} MB)")
    return True


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    results = [download(name, url, min_size) for name, (url, min_size) in MODELS.items()]
    if all(results):
        print("\nAll models ready in", MODELS_DIR)
        return 0
    print("\nSome downloads failed. Check your internet connection and run again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
