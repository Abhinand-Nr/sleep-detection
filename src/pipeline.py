"""Orchestration: video -> per-frame signals (extraction) -> people -> per-second states (analysis).

Extraction runs the neural networks once and is cached to <out>/signals.pkl, because it is
the slow part. Analysis is pure rules on those signals and takes well under a second, so
evaluate.py can try many threshold settings without re-running the models.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregation import aggregate
from src.body_signals import (background_motion, changed_mask, keypoint_motion, posture_metrics,
                              region_motion, to_gray)
from src.detection_tracking import PersonDetection, PersonTracker, assign_faces
from src.face_signals import FaceAnalyzer
from src.identity import (FaceEmbedder, Identity, TrackEvidence, load_references, resolve_identities,
                          torso_histogram)
from src.sleep_state import classify
from src.utils.logging import get_logger
from src.video_io import VideoReader

log = get_logger(__name__)

EXTRACTION_SECTIONS = ("models", "video", "detection", "face", "body", "identity")
CACHE_VERSION = 1       # bump when the extraction code changes what it stores
FALLBACK_TRACK_ID = -1  # face found but YOLO saw no person (single-person mode only)
SINGLE_PERSON_NAME = "Person 1"


@dataclass
class Extraction:
    frames: pd.DataFrame                 # one row per (processed frame, detected person)
    tracks: dict[int, TrackEvidence]
    processed_times: np.ndarray          # timestamp of every processed frame, even empty ones
    fps: float                           # processing rate
    duration: float
    ref_names: list[str] = field(default_factory=list)

    @property
    def single_person(self) -> bool:
        return not self.ref_names


@dataclass
class Analysis:
    frames: pd.DataFrame                 # frames with a `person` column
    per_second: pd.DataFrame             # one row per (second, person) with state + confidence
    identities: dict[int, Identity]
    calibration: dict[str, dict]


# ---------------------------------------------------------------- extraction

def _cache_key(video: Path, refs_dir: Path | None, cfg: dict) -> str:
    stat = video.stat()
    refs = sorted((p.name, p.stat().st_mtime) for p in refs_dir.iterdir()) if refs_dir else []
    payload = {
        "v": CACHE_VERSION,
        "video": [str(video.resolve()), stat.st_size, stat.st_mtime],
        "refs": refs,
        "cfg": {k: cfg[k] for k in EXTRACTION_SECTIONS},
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def extract(video: Path, refs_dir: Path | None, cfg: dict, root: Path,
            cache_dir: Path | None = None) -> Extraction:
    """Run all models over the video (or load the cached result)."""
    key = _cache_key(video, refs_dir, cfg)
    cache = cache_dir / "signals.pkl" if cache_dir else None
    if cache and cache.exists():
        with cache.open("rb") as f:
            stored = pickle.load(f)
        if stored.get("key") == key:
            log.info("Loaded cached signals from %s (delete it to force re-extraction)", cache)
            return stored["extraction"]
        log.info("Config or video changed since the cache was made; re-extracting")

    ext = _extract(video, refs_dir, cfg, root)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump({"key": key, "extraction": ext}, f)
    return ext


def _extract(video: Path, refs_dir: Path | None, cfg: dict, root: Path) -> Extraction:
    models = cfg["models"]
    face_an = FaceAnalyzer(models, cfg["face"], root)
    tracker = PersonTracker(root / models["yolo_pose"], cfg["detection"], root)

    embedder, refs = None, {}
    if refs_dir is not None:
        embedder = FaceEmbedder(root / models["sface"])
        refs = load_references(refs_dir, face_an, embedder)
        if not refs:
            log.warning("No usable reference photos in %s; falling back to single-person mode", refs_dir)
    single = not refs

    kp_min = cfg["detection"]["keypoint_conf"]
    body, ident = cfg["body"], cfg["identity"]
    rows: list[dict] = []
    tracks: dict[int, TrackEvidence] = {}
    prev_gray = None
    prev_kps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    times: list[float] = []
    progress_every = cfg["logging"]["progress_every_seconds"]
    next_progress = progress_every

    with VideoReader(video, cfg["video"]["process_fps"], cfg["video"]["max_side"]) as reader:
        fps, duration = reader.effective_fps, reader.info.duration_sec
        for n, frame in enumerate(reader):
            img, t = frame.image, frame.time_sec
            times.append(t)
            if t >= next_progress:
                log.info("  ... %.0f / %.0f s processed", t, duration)
                next_progress += progress_every

            people = tracker.update(img)
            faces = face_an.detect(img)
            face_of = assign_faces(people, [f.center for f in faces], kp_min)

            if single and not people and faces:
                # Close-up webcam shot: YOLO may see no body, but the face is enough for EAR.
                f = max(faces, key=lambda f: f.w * f.h)
                m = 0.5   # enlarge the face box so motion covers the head, not just the face
                people = [PersonDetection(FALLBACK_TRACK_ID,
                                          (f.x - m * f.w, f.y - m * f.h, f.x + (1 + m) * f.w, f.y + (1 + m) * f.h),
                                          f.score, np.zeros((17, 2)), np.zeros(17))]
                face_of = {0: faces.index(f)}

            gray = to_gray(img, body["blur_kernel"])
            mask = changed_mask(gray, prev_gray, body["pixel_diff_threshold"]) if prev_gray is not None else None
            bg = background_motion(mask, [p.box for p in people]) if mask is not None else float("nan")

            for pi, p in enumerate(people):
                ok = p.kp_conf >= kp_min
                ev = tracks.setdefault(p.track_id, TrackEvidence(p.track_id))
                ev.add_frame(t)
                box_h = p.box[3] - p.box[1]
                row = {
                    "frame_index": frame.index, "time_sec": t, "track_id": p.track_id,
                    "x1": p.box[0], "y1": p.box[1], "x2": p.box[2], "y2": p.box[3],
                    "det_conf": p.conf, "n_kp": int(ok.sum()),
                    "motion": max(0.0, region_motion(mask, p.box) - bg) if mask is not None else float("nan"),
                    "bg_motion": bg,
                    "kp_motion": keypoint_motion(p.keypoints, ok, *prev_kps[p.track_id], box_h)
                    if p.track_id in prev_kps else float("nan"),
                    **posture_metrics(p.keypoints, ok, p.box),
                    "face": False, "face_w": float("nan"), "ear": float("nan"), "ear_left": float("nan"),
                    "ear_right": float("nan"), "pitch": float("nan"), "yaw": float("nan"),
                    "kps": np.where(ok[:, None], p.keypoints, np.nan).astype(np.float32),
                }
                prev_kps[p.track_id] = (p.keypoints, ok)

                if pi in face_of:
                    face = faces[face_of[pi]]
                    row["face_w"] = face.w
                    meas = face_an.measure(img, face)
                    if meas is not None:
                        row.update(face=True, ear=meas.ear, ear_left=meas.ear_left, ear_right=meas.ear_right,
                                   pitch=meas.pitch, yaw=meas.yaw)
                    if embedder is not None and refs and n % ident["embed_every_n_frames"] == 0 \
                            and face.w >= cfg["face"]["min_face_px"]:
                        feat = embedder.embed(img, face.raw)
                        ev.add_similarities({name: embedder.similarity(feat, r) for name, r in refs.items()})

                if n % ident["hist_every_n_frames"] == 0 and p.track_id != FALLBACK_TRACK_ID:
                    hist = torso_histogram(img, p.box, p.keypoints, ok, ident["hist_bins"])
                    if hist is not None:
                        ev.add_histogram(hist)
                rows.append(row)
            prev_gray = gray

    face_an.close()
    frames = pd.DataFrame(rows)
    if frames.empty:
        log.warning("No person or face was detected anywhere in the video")
    log.info("Extraction done: %d processed frames, %d person-detections, %d track(s)",
             len(times), len(frames), len([t for t in tracks if t >= 0]))
    return Extraction(frames, tracks, np.asarray(times), fps, duration, sorted(refs))


# ---------------------------------------------------------------- analysis

def assign_people(ext: Extraction, cfg: dict) -> tuple[pd.DataFrame, dict[int, Identity]]:
    """Add a `person` column (name) to the frame rows; drop rows that belong to nobody."""
    frames = ext.frames.copy()
    if frames.empty:
        return frames.assign(person=pd.Series(dtype=str), identity_source=pd.Series(dtype=str)), {}

    if ext.single_person:
        # One person: in every frame keep the biggest detection and call it Person 1.
        frames["area"] = (frames.x2 - frames.x1) * (frames.y2 - frames.y1)
        frames = frames.loc[frames.groupby("frame_index")["area"].idxmax()].drop(columns="area")
        frames["person"] = SINGLE_PERSON_NAME
        frames["identity_source"] = "single"
        return frames.reset_index(drop=True), {}

    identities = resolve_identities(ext.tracks, ext.ref_names, cfg["matching"])
    for tid, ev in ext.tracks.items():
        if tid >= 0 and tid not in identities and ev.duration >= cfg["matching"]["min_track_seconds"]:
            log.warning("Track %d (%.0f-%.0f s) could not be identified", tid, ev.first_time, ev.last_time)
    for tid, idn in identities.items():
        log.info("Track %d -> %s (via %s, score %.2f)", tid, idn.name, idn.source, idn.score)
    for name in ext.ref_names:
        if not any(i.name == name for i in identities.values()):
            log.warning("%s (from refs) was never recognised in the video", name)

    frames["person"] = frames.track_id.map(lambda t: identities[t].name if t in identities else None)
    frames["identity_source"] = frames.track_id.map(lambda t: identities[t].source if t in identities else None)
    frames = frames.dropna(subset=["person"])
    # Safety net: never two rows for the same person in one frame.
    frames = frames.sort_values("det_conf", ascending=False).drop_duplicates(["frame_index", "person"])
    return frames.sort_values(["frame_index", "person"]).reset_index(drop=True), identities


def people_to_report(ext: Extraction, frames: pd.DataFrame) -> list[str]:
    names = list(ext.ref_names) if not ext.single_person else [SINGLE_PERSON_NAME]
    extra = sorted(set(frames.person) - set(names)) if not frames.empty else []
    return names + extra


def analyze(ext: Extraction, cfg: dict) -> Analysis:
    """Cheap, rule-based half of the pipeline (identity, aggregation, state machine)."""
    frames, identities = assign_people(ext, cfg)
    people = people_to_report(ext, frames)
    per_second, frames, calibration = aggregate(frames, people, ext.processed_times, ext.fps, cfg)
    # Unnamed tracks that were never clearly visible for a single second are detector noise.
    unnamed = {i.name for i in identities.values() if i.source == "none"}
    noise = [p for p in unnamed if not per_second[per_second.person == p].visible.any()]
    if noise:
        per_second = per_second[~per_second.person.isin(noise)]
        frames = frames[~frames.person.isin(noise)]
    per_second = classify(per_second, cfg["sleep"])
    return Analysis(frames, per_second, identities, calibration)
