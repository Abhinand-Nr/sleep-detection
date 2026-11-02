"""Person detection, body keypoints and tracking: YOLOv8n-pose with ByteTrack.

YOLO finds every person and 17 body keypoints (nose, eyes, ears, shoulders, elbows,
wrists, hips, knees, ankles). It works from behind too, which is why the body path
does not need a face. ByteTrack links boxes across frames and gives each person a
track ID that stays the same while they are visible, so a name learned from the face
once can be kept when the face turns away.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class PersonDetection:
    track_id: int                 # -1 = not from the tracker (face-only fallback)
    box: tuple[float, float, float, float]   # x1, y1, x2, y2
    conf: float
    keypoints: np.ndarray         # (17, 2) pixel coords
    kp_conf: np.ndarray           # (17,)

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


class PersonTracker:
    """Runs YOLO-pose + ByteTrack frame by frame. Create one per video (tracker state persists)."""

    def __init__(self, model_path: Path, det_cfg: dict, root: Path):
        self.cfg = det_cfg
        self._model = YOLO(str(model_path))
        self._tracker_cfg = str(root / det_cfg["tracker"])

    def update(self, image: np.ndarray) -> list[PersonDetection]:
        result = self._model.track(
            image, persist=True, tracker=self._tracker_cfg, classes=[0],
            conf=self.cfg["person_conf"], imgsz=self.cfg["imgsz"], device=self.cfg["device"],
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []   # ByteTrack only reports confirmed tracks

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        if result.keypoints is not None and result.keypoints.conf is not None:
            kps = result.keypoints.xy.cpu().numpy()
            kp_conf = result.keypoints.conf.cpu().numpy()
        else:
            kps = np.zeros((len(ids), 17, 2))
            kp_conf = np.zeros((len(ids), 17))

        return [
            PersonDetection(int(ids[i]), tuple(float(v) for v in xyxy[i]), float(confs[i]), kps[i], kp_conf[i])
            for i in range(len(ids))
        ]


def assign_faces(people: list[PersonDetection], face_centers: list[tuple[float, float]],
                 kp_conf_min: float) -> dict[int, int]:
    """Match faces to people. Returns {person index: face index}, one face per person at most.

    A face can belong to a person only if its centre lies in the top 60% of their box.
    Among candidates, the closest face to the person's nose keypoint (or to the top-centre
    of the box when the nose is not visible) wins, greedily.
    """
    head_region = 0.6   # fraction of box height from the top where a face can be
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(people):
        x1, y1, x2, y2 = p.box
        h = y2 - y1
        if p.kp_conf[0] >= kp_conf_min:
            anchor = p.keypoints[0]
        else:
            anchor = np.array([(x1 + x2) / 2, y1 + 0.15 * h])
        for fi, (cx, cy) in enumerate(face_centers):
            if x1 <= cx <= x2 and y1 <= cy <= y1 + head_region * h:
                dist = float(np.hypot(cx - anchor[0], cy - anchor[1]))
                pairs.append((dist, pi, fi))

    used_p: set[int] = set()
    used_f: set[int] = set()
    out: dict[int, int] = {}
    for _, pi, fi in sorted(pairs):
        if pi not in used_p and fi not in used_f:
            out[pi] = fi
            used_p.add(pi)
            used_f.add(fi)
    return out
