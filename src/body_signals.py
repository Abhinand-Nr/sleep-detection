"""Body signals: motion energy and posture, which work even when the face is hidden.

Motion energy = fraction of pixels inside the person's box that changed since the previous
processed frame, minus the fraction that changed in the background (lighting flicker,
camera noise). Pixel differencing picks up small movements (typing, fidgeting) that are
smaller than YOLO's keypoint jitter, so it is the main stillness signal. Keypoint motion
is also recorded for comparison.

Posture uses COCO body keypoints (image y grows downwards):
  head height  = how far the head is above the shoulder line, in shoulder widths
  arm contact  = distance from head to nearest wrist/elbow, in shoulder widths
  torso angle  = angle of the shoulder->hip line from vertical (0 = upright, 90 = lying)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP = 11, 12
HEAD = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
ARMS = (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)

# Shoulder width shrinks to ~0 when someone is side-on; never let the scale drop below this
# fraction of the box height, so ratios stay sane.
_MIN_SCALE_OF_BOX_HEIGHT = 0.2

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def to_gray(image: np.ndarray, blur_kernel: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)


def changed_mask(gray: np.ndarray, prev_gray: np.ndarray, threshold: int) -> np.ndarray:
    """Boolean mask of pixels whose grey level changed by more than `threshold`."""
    return cv2.absdiff(gray, prev_gray) > threshold


def _clip_box(box: Box, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def region_motion(mask: np.ndarray, box: Box) -> float:
    """Fraction of changed pixels inside a box."""
    x1, y1, x2, y2 = _clip_box(box, mask.shape)
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    return float(mask[y1:y2, x1:x2].mean())


def background_motion(mask: np.ndarray, boxes: list[Box]) -> float:
    """Fraction of changed pixels outside every person box (global noise / lighting change)."""
    outside = np.ones(mask.shape, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = _clip_box(box, mask.shape)
        outside[y1:y2, x1:x2] = False
    return float(mask[outside].mean()) if outside.any() else 0.0


def keypoint_motion(kps: np.ndarray, ok: np.ndarray, prev_kps: np.ndarray, prev_ok: np.ndarray,
                    box_height: float) -> float:
    """Mean displacement of keypoints visible in both frames, as a fraction of box height."""
    both = ok & prev_ok
    if not both.any() or box_height <= 0:
        return float("nan")
    return float(np.linalg.norm(kps[both] - prev_kps[both], axis=1).mean() / box_height)


def posture_metrics(kps: np.ndarray, ok: np.ndarray, box: Box) -> dict[str, float]:
    """Raw posture numbers (nan when the needed keypoints are not visible)."""
    nan = float("nan")
    out = {"head_height": nan, "arm_contact": nan, "torso_angle": nan}
    if not (ok[L_SHOULDER] and ok[R_SHOULDER]):
        return out

    shoulder_mid = (kps[L_SHOULDER] + kps[R_SHOULDER]) / 2
    box_h = box[3] - box[1]
    scale = max(float(np.linalg.norm(kps[L_SHOULDER] - kps[R_SHOULDER])), _MIN_SCALE_OF_BOX_HEIGHT * box_h)
    if scale <= 0:
        return out

    head_idx = [i for i in HEAD if ok[i]]
    if head_idx:
        head = kps[head_idx].mean(axis=0)
        out["head_height"] = float((shoulder_mid[1] - head[1]) / scale)
        arm_idx = [i for i in ARMS if ok[i]]
        if arm_idx:
            out["arm_contact"] = float(np.linalg.norm(kps[arm_idx] - head, axis=1).min() / scale)

    if ok[L_HIP] and ok[R_HIP]:
        hip_mid = (kps[L_HIP] + kps[R_HIP]) / 2
        dx, dy = hip_mid - shoulder_mid
        out["torso_angle"] = float(math.degrees(math.atan2(abs(dx), abs(dy))))
    return out


def posture_flag(head_height: float, arm_contact: float, torso_angle: float, cfg: dict) -> str:
    """Name the posture: lying / head_on_arms / slumped / upright / unknown.

    Checked from strongest to weakest sleep evidence. nan comparisons are False, so a
    missing measurement never triggers a flag.
    """
    if torso_angle > cfg["lying_torso_angle_deg"]:
        return "lying"
    if arm_contact < cfg["arm_contact_ratio"] and head_height < cfg["arm_contact_max_head_height"]:
        return "head_on_arms"
    if head_height < cfg["slump_head_height"]:
        return "slumped"
    if not math.isnan(head_height):
        return "upright"
    return "unknown"


SLEEPY_POSTURES = frozenset({"lying", "head_on_arms", "slumped"})
