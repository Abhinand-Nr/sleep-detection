"""Who is who: face recognition (SFace) plus a clothing-colour backup, decided per track.

How it works (offline, over the whole video):
1. During extraction, every few frames each track's face (if visible) is compared with
   every reference photo; the cosine similarities are stored as votes for that track.
   A sample of the track's torso colour histogram is stored too.
2. After the video is read, tracks are named greedily, strongest face evidence first.
   A name cannot be given to two tracks that are on screen at the same time.
3. A track that never showed its face (e.g. the tracker lost someone and gave them a new
   ID while they faced away) can inherit a name from clothing colour, if it is similar
   enough to an already-named person who is not on screen at the same time.
Because a name belongs to the whole track, a person stays named while their face is hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.body_signals import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
from src.utils.logging import get_logger

log = get_logger(__name__)

REF_EXTENSIONS = (".jpg", ".jpeg", ".png")


class FaceEmbedder:
    """SFace face recognizer: aligned face crop -> 128-d feature vector."""

    def __init__(self, model_path: Path):
        self._rec = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(self, image: np.ndarray, yunet_row: np.ndarray) -> np.ndarray:
        aligned = self._rec.alignCrop(image, yunet_row)
        return self._rec.feature(aligned).copy()

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self._rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


def load_references(refs_dir: Path, face_analyzer, embedder: FaceEmbedder) -> dict[str, np.ndarray]:
    """Embed refs/<Name>.jpg files. Uses the largest face in each photo; skips photos without one."""
    refs: dict[str, np.ndarray] = {}
    for path in sorted(refs_dir.iterdir()):
        if path.suffix.lower() not in REF_EXTENSIONS:
            continue
        image = cv2.imread(str(path))
        if image is None:
            log.warning("Could not read reference photo %s", path.name)
            continue
        faces = face_analyzer.detect(image)
        if not faces:
            log.warning("No face found in reference photo %s; skipping it", path.name)
            continue
        best = max(faces, key=lambda f: f.w * f.h)
        refs[path.stem] = embedder.embed(image, best.raw)
        log.info("Reference loaded: %s", path.stem)
    return refs


def torso_histogram(image: np.ndarray, box: tuple[float, float, float, float], kps: np.ndarray,
                    ok: np.ndarray, bins: list[int]) -> np.ndarray | None:
    """Normalised hue-saturation histogram of the torso (clothing colour).

    Uses the shoulder-to-hip region when those keypoints are visible, otherwise the middle
    of the person box. Hue/saturation ignore brightness, so mild lighting changes matter less.
    """
    x1, y1, x2, y2 = box
    if ok[L_SHOULDER] and ok[R_SHOULDER] and ok[L_HIP] and ok[R_HIP]:
        pts = kps[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
        tx1, ty1 = pts.min(axis=0)
        tx2, ty2 = pts.max(axis=0)
    else:
        w, h = x2 - x1, y2 - y1
        tx1, tx2 = x1 + 0.2 * w, x2 - 0.2 * w
        ty1, ty2 = y1 + 0.25 * h, y1 + 0.6 * h
    ih, iw = image.shape[:2]
    tx1, ty1 = max(0, int(tx1)), max(0, int(ty1))
    tx2, ty2 = min(iw, int(tx2)), min(ih, int(ty2))
    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return None
    hsv = cv2.cvtColor(image[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


@dataclass
class TrackEvidence:
    """Everything collected about one tracker ID during extraction."""
    track_id: int
    first_time: float = float("inf")
    last_time: float = float("-inf")
    n_frames: int = 0
    sims: dict[str, list[float]] = field(default_factory=dict)   # ref name -> similarities
    hist_sum: np.ndarray | None = None
    hist_n: int = 0

    def add_frame(self, t: float) -> None:
        self.first_time = min(self.first_time, t)
        self.last_time = max(self.last_time, t)
        self.n_frames += 1

    def add_similarities(self, sims: dict[str, float]) -> None:
        for name, s in sims.items():
            self.sims.setdefault(name, []).append(s)

    def add_histogram(self, hist: np.ndarray) -> None:
        self.hist_sum = hist.copy() if self.hist_sum is None else self.hist_sum + hist
        self.hist_n += 1

    @property
    def duration(self) -> float:
        return max(0.0, self.last_time - self.first_time)

    @property
    def mean_hist(self) -> np.ndarray | None:
        return None if self.hist_sum is None else (self.hist_sum / self.hist_n).astype(np.float32)


@dataclass
class Identity:
    name: str
    source: str        # "face", "clothing", "single", or "none"
    score: float       # face: summed similarity of matching votes; clothing: correlation


def _overlaps(a: TrackEvidence, spans: list[tuple[float, float]], tolerance: float) -> bool:
    """True if the track shares more than `tolerance` seconds with any span. At an ID switch the
    old and new track can coexist for a few frames; that is a handover, not two people."""
    return any(min(a.last_time, end) - max(a.first_time, start) > tolerance for start, end in spans)


def resolve_identities(tracks: dict[int, TrackEvidence], ref_names: list[str],
                       cfg: dict) -> dict[int, Identity]:
    """Name tracks from face votes first, then clothing; see the module docstring."""
    candidates = [t for t in tracks.values() if t.duration >= cfg["min_track_seconds"] and t.track_id >= 0]
    thr = cfg["sface_threshold"]

    # (score, track, name) for every track/name with enough matching face votes
    options: list[tuple[float, TrackEvidence, str]] = []
    for t in candidates:
        for name in ref_names:
            good = [s for s in t.sims.get(name, []) if s >= thr]
            if len(good) >= cfg["min_face_votes"]:
                options.append((float(sum(good)), t, name))

    result: dict[int, Identity] = {}
    spans: dict[str, list[tuple[float, float]]] = {n: [] for n in ref_names}
    hists: dict[str, list[np.ndarray]] = {n: [] for n in ref_names}

    def give(t: TrackEvidence, name: str, source: str, score: float) -> None:
        result[t.track_id] = Identity(name, source, score)
        spans[name].append((t.first_time, t.last_time))
        if t.mean_hist is not None:
            hists[name].append(t.mean_hist)

    for score, t, name in sorted(options, key=lambda o: -o[0]):
        if t.track_id in result or _overlaps(t, spans[name], cfg["max_handover_overlap_seconds"]):
            continue
        give(t, name, "face", score)

    # Clothing backup, longest unnamed tracks first.
    for t in sorted(candidates, key=lambda t: -t.n_frames):
        if t.track_id in result or t.mean_hist is None:
            continue
        best_name, best_sim = None, cfg["clothing_threshold"]
        for name in ref_names:
            if not hists[name] or _overlaps(t, spans[name], cfg["max_handover_overlap_seconds"]):
                continue
            ref_hist = np.mean(hists[name], axis=0).astype(np.float32)
            sim = histogram_similarity(t.mean_hist, ref_hist)
            if sim >= best_sim:
                best_name, best_sim = name, sim
        if best_name is not None:
            give(t, best_name, "clothing", best_sim)

    for t in candidates:
        if t.track_id not in result and t.duration >= cfg["min_unknown_seconds"]:
            result[t.track_id] = Identity(f"Unknown {t.track_id}", "none", 0.0)
    return result
