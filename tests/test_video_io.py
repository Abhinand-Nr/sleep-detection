"""Tests for video reading: subsampling, timestamps, and error handling."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video_io import VideoOpenError, VideoReader


def _make_video(path: Path, n_frames: int = 30, fps: float = 30.0, size: tuple[int, int] = (64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i, dtype=np.uint8))
    writer.release()
    return path


def test_reads_metadata_and_all_frames(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "v.avi")
    with VideoReader(video) as reader:
        assert reader.info.fps == pytest.approx(30.0)
        frames = list(reader)
    assert len(frames) == 30
    assert frames[-1].time_sec == pytest.approx(29 / 30)


def test_subsamples_to_process_fps(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "v.avi")
    with VideoReader(video, process_fps=10) as reader:
        assert reader.step == 3
        frames = list(reader)
    assert [f.index for f in frames] == list(range(0, 30, 3))


def test_resizes_to_max_side(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "v.avi", size=(200, 100))
    with VideoReader(video, max_side=100) as reader:
        frame = next(iter(reader))
    assert frame.image.shape[:2] == (50, 100)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoOpenError):
        VideoReader(tmp_path / "nope.mp4")


def test_garbage_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoOpenError):
        VideoReader(bad)
