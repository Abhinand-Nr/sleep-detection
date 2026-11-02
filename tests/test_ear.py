"""EAR and head-pose maths."""
import math

import numpy as np
import pytest

from src.face_signals import eye_aspect_ratio, rotation_to_pitch_yaw


def eye(width: float, height: float) -> np.ndarray:
    """Six eye points p1..p6 for an eye of the given width and opening height."""
    h = height / 2
    return np.array([[0, 0], [width / 3, -h], [2 * width / 3, -h], [width, 0], [2 * width / 3, h], [width / 3, h]])


def test_ear_is_height_over_width() -> None:
    assert eye_aspect_ratio(eye(30, 9)) == pytest.approx(0.3)


def test_ear_is_scale_invariant() -> None:
    assert eye_aspect_ratio(eye(30, 9)) == pytest.approx(eye_aspect_ratio(eye(300, 90)))


def test_closed_eye_has_low_ear() -> None:
    assert eye_aspect_ratio(eye(30, 1)) < 0.05 < eye_aspect_ratio(eye(30, 9))


def test_degenerate_eye_gives_nan() -> None:
    assert math.isnan(eye_aspect_ratio(np.zeros((6, 2))))


def test_pitch_positive_when_rotating_face_down() -> None:
    a = math.radians(20)
    rx = np.eye(4)
    rx[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    pitch, yaw = rotation_to_pitch_yaw(rx)
    assert pitch == pytest.approx(20, abs=1e-6)
    assert yaw == pytest.approx(0, abs=1e-6)


def test_yaw_from_rotation_about_vertical_axis() -> None:
    a = math.radians(30)
    ry = np.eye(4)
    ry[0, 0], ry[0, 2], ry[2, 0], ry[2, 2] = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
    _, yaw = rotation_to_pitch_yaw(ry)
    assert abs(yaw) == pytest.approx(30, abs=1e-6)
