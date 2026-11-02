"""Video reading.

VideoReader opens a file, reports its metadata, and yields frames at a chosen
processing rate (e.g. 10 fps from a 30 fps video) with a timestamp in seconds.
Timestamps come from frame_index / fps, which is more reliable than the
container's millisecond position for many phone videos.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


class VideoOpenError(RuntimeError):
    """Raised when a video file is missing or cannot be decoded."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int      # from the container header; may be approximate
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Frame:
    index: int            # index in the original video
    time_sec: float
    image: np.ndarray     # BGR, possibly resized


class VideoReader:
    """Iterate over a video's frames, optionally subsampled and resized.

    Use as a context manager:
        with VideoReader(path, process_fps=10) as reader:
            for frame in reader:
                ...
    """

    def __init__(self, path: Path, process_fps: float | None = None, max_side: int | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise VideoOpenError(f"Video not found: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise VideoOpenError(f"OpenCV could not open video: {self.path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            raise VideoOpenError(f"Video reports an invalid FPS ({fps}): {self.path}")

        self.info = VideoInfo(
            path=self.path,
            fps=float(fps),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.max_side = max_side

        # Keep every `step`-th frame. If the requested rate is >= the video rate, keep all.
        if process_fps and process_fps < self.info.fps:
            self.step = max(1, round(self.info.fps / process_fps))
        else:
            self.step = 1

    @property
    def effective_fps(self) -> float:
        """Frames per second actually delivered to the pipeline."""
        return self.info.fps / self.step

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if not self.max_side:
            return image
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self.max_side:
            return image
        scale = self.max_side / longest
        return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            # grab() is cheap (no decode); only retrieve() the frames we keep.
            if not self._cap.grab():
                break
            if index % self.step == 0:
                ok, image = self._cap.retrieve()
                if not ok or image is None:
                    log.warning("Could not decode frame %d, skipping", index)
                else:
                    yield Frame(index=index, time_sec=index / self.info.fps, image=self._resize(image))
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
