"""Face signals: find faces (YuNet), then measure eye openness (EAR) and head pose (MediaPipe).

EAR (Eye Aspect Ratio, Soukupova & Cech 2016) = eye height / eye width, from 6 points per eye:

        p2   p3
    p1          p4        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        p6   p5

Open eyes give a roughly constant EAR (~0.25-0.35); it drops towards ~0.1 when the eye closes.
Because it is a ratio, it does not depend on how far the person is from the camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# MediaPipe Face Mesh indices, ordered p1..p6 as in the diagram above.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)   # subject's right eye (left side of the image)
LEFT_EYE = (362, 385, 387, 263, 373, 380)


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR from a (6, 2) array of eye points ordered p1..p6. Returns nan if the eye has no width."""
    p1, p2, p3, p4, p5, p6 = pts
    width = np.linalg.norm(p1 - p4)
    if width < 1e-6:
        return float("nan")
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width))


def rotation_to_pitch_yaw(matrix: np.ndarray) -> tuple[float, float]:
    """Pitch and yaw in degrees from MediaPipe's 4x4 facial transformation matrix.

    MediaPipe uses a y-up camera space, so a positive rotation about x turns the face
    downwards: pitch > 0 means "head tilted down". Yaw is left/right turning.
    """
    r = np.asarray(matrix, dtype=float)[:3, :3]
    # Remove any uniform scale so the angles are well defined.
    r = r / np.cbrt(max(np.linalg.det(r), 1e-9))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    return pitch, yaw


@dataclass
class FaceBox:
    """One YuNet detection. `raw` is the 15-value row SFace needs for alignment."""
    x: float
    y: float
    w: float
    h: float
    score: float
    raw: np.ndarray

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class FaceMeasurement:
    ear: float            # mean of both eyes
    ear_left: float
    ear_right: float
    pitch: float          # degrees, + = head down (raw, before per-person baseline)
    yaw: float            # degrees


class FaceAnalyzer:
    """Wraps YuNet (detection) and MediaPipe Face Landmarker (landmarks + pose)."""

    def __init__(self, models_cfg: dict, face_cfg: dict, root: Path):
        self.cfg = face_cfg
        self._detector = cv2.FaceDetectorYN.create(
            str(root / models_cfg["yunet"]), "", (320, 320),
            face_cfg["yunet_score"], face_cfg["yunet_nms"], 5000,
        )
        self._input_size: tuple[int, int] | None = None
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(root / models_cfg["face_landmarker"])),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=face_cfg["landmark_conf"],
            min_face_presence_confidence=face_cfg["landmark_conf"],
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """All faces in a BGR image (any orientation YuNet can see, frontal to ~profile)."""
        h, w = image.shape[:2]
        if self._input_size != (w, h):
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self._detector.detect(image)
        if faces is None:
            return []
        return [FaceBox(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14]), f.copy())
                for f in faces]

    def measure(self, image: np.ndarray, face: FaceBox) -> FaceMeasurement | None:
        """EAR and head pose for one face, or None if it is too small or landmarks fail."""
        if face.w < self.cfg["min_face_px"]:
            return None
        crop = _crop_around(image, face, self.cfg["crop_margin"])
        if crop.size == 0:
            return None
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None

        ch, cw = crop.shape[:2]
        pts = np.array([(lm.x * cw, lm.y * ch) for lm in result.face_landmarks[0]])
        ear_r = eye_aspect_ratio(pts[list(RIGHT_EYE)])
        ear_l = eye_aspect_ratio(pts[list(LEFT_EYE)])
        pitch, yaw = rotation_to_pitch_yaw(result.facial_transformation_matrixes[0])
        return FaceMeasurement(float(np.nanmean([ear_l, ear_r])), ear_l, ear_r, pitch, yaw)

    def close(self) -> None:
        self._landmarker.close()


def _crop_around(image: np.ndarray, face: FaceBox, margin: float) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = int(max(0, face.x - margin * face.w))
    y1 = int(max(0, face.y - margin * face.h))
    x2 = int(min(w, face.x + (1 + margin) * face.w))
    y2 = int(min(h, face.y + (1 + margin) * face.h))
    return image[y1:y2, x1:x2]
