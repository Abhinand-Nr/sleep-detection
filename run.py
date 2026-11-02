"""Sleep detector entry point.

    python run.py --video data/videos/clip1.mp4 --refs refs/ --out outputs/clip1

Without --refs the video is treated as single-person ("Person 1").
Writes results.csv, summary.json, annotated.mp4, timeline.png and run.log to --out.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.outputs import (build_summary, log_warnings, plot_timeline, write_annotated_video, write_csv,
                         write_summary)
from src.pipeline import analyze, extract
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.video_io import VideoOpenError, VideoReader

ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ("yunet", "sface", "face_landmarker", "yolo_pose")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect who is asleep in a video.")
    p.add_argument("--video", type=Path, required=True, help="input video file")
    p.add_argument("--refs", type=Path, default=None,
                   help="folder of reference photos <Name>.jpg (optional for single-person videos)")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: outputs/<video name>)")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="config file")
    p.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4 (faster)")
    return p.parse_args(argv)


def check_models(cfg: dict) -> list[str]:
    """Return the model files listed in config that are missing on disk."""
    return [cfg["models"][k] for k in MODEL_KEYS if not (ROOT / cfg["models"][k]).exists()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    out_dir = args.out or ROOT / "outputs" / args.video.stem
    log = setup_logging(out_dir, cfg["logging"]["level"])

    missing = check_models(cfg)
    if missing:
        log.error("Missing model files: %s. Run `python scripts/download_models.py` first.", ", ".join(missing))
        return 2

    refs = args.refs
    if refs is not None and not refs.is_dir():
        log.warning("--refs folder %s does not exist; continuing in single-person mode", refs)
        refs = None
    if refs is None:
        log.info("Single-person mode: the person will be labelled 'Person 1'")

    try:
        with VideoReader(args.video, cfg["video"]["process_fps"], cfg["video"]["max_side"]) as reader:
            info = reader.info
            log.info("Video %s: %dx%d, %.2f fps, %d frames (~%.1f s); analysing at %.1f fps",
                     info.path.name, info.width, info.height, info.fps, info.frame_count,
                     info.duration_sec, reader.effective_fps)
    except VideoOpenError as exc:
        log.error("%s", exc)
        return 2

    start = time.perf_counter()
    ext = extract(args.video, refs, cfg, ROOT, cache_dir=out_dir)
    if len(ext.processed_times) == 0:
        log.error("No frames could be decoded from %s", args.video)
        return 2
    result = analyze(ext, cfg)
    log.info("Analysis took %.1f s in total", time.perf_counter() - start)

    write_csv(result.per_second, out_dir / "results.csv")
    summary = build_summary(result.per_second, result.calibration, args.video, cfg, result.identities)
    write_summary(summary, out_dir / "summary.json")
    plot_timeline(result.per_second, out_dir / "timeline.png", f"Sleep states - {args.video.name}")
    log_warnings(result.per_second, cfg)
    if cfg["outputs"]["annotated_video"] and not args.no_video:
        log.info("Writing annotated video ...")
        write_annotated_video(args.video, result.frames, result.per_second, cfg, out_dir / "annotated.mp4")

    log.info("---- Summary ----")
    for line in summary["summary"]:
        log.info("  %s", line)
    log.info("Outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
