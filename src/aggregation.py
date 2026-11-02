"""Per-person calibration and per-second aggregation.

At 10 fps each second has ~10 frames. Single frames are noisy (a blink, a landmark glitch),
so every decision is made per second from the fractions of frames showing something,
e.g. "eyes closed in >= 50% of this second's measurable frames".

Calibration: EAR differs between people (eye shape, glasses, camera angle), so the
"eyes closed" threshold is relative to each person's own open-eye EAR, measured in the
first seconds their face is seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.body_signals import SLEEPY_POSTURES, posture_flag
from src.utils.logging import get_logger

log = get_logger(__name__)

# How many frame intervals may separate two frames of the same eye closure (tolerates one
# dropped frame); a larger gap ends the closure.
_MAX_FRAME_GAP = 1.5


def calibrate(df_p: pd.DataFrame, eyes: dict) -> dict:
    """Baseline open-eye EAR, closed-eye threshold and baseline head pitch for one person."""
    # Calibrate only on frames facing the camera: someone who starts the video looking down at a
    # phone must not have "head down, eyes half-shut" learned as their normal.
    frontal = df_p[df_p.face & (df_p.yaw.abs() <= eyes["max_yaw_deg"])
                   & (df_p.pitch.abs() <= eyes["max_calibration_pitch_deg"])].dropna(subset=["ear"])
    cal = {"method": "fallback", "baseline_ear": float("nan"),
           "threshold": eyes["fallback_threshold"], "baseline_pitch": 0.0}
    if frontal.empty:
        return cal

    start = frontal.time_sec.min()
    first = frontal[frontal.time_sec < start + eyes["calibration_seconds"]]
    candidates = []
    if len(first) >= eyes["min_calibration_frames"]:
        candidates.append(("first_seconds", first))
    candidates.append(("whole_video", frontal))   # if the person started with eyes closed
    for method, data in candidates:
        baseline = float(np.percentile(data.ear, eyes["baseline_percentile"]))
        if baseline >= eyes["min_plausible_baseline"]:
            cal.update(method=method, baseline_ear=baseline, threshold=eyes["closed_ratio"] * baseline,
                       baseline_pitch=float(data.pitch.median()))
            return cal
    cal["baseline_pitch"] = float(frontal.pitch.median())
    return cal


def frame_flags(df_p: pd.DataFrame, cal: dict, cfg: dict) -> pd.DataFrame:
    """Add per-frame booleans: is the EAR trustworthy, are the eyes closed, posture name."""
    eyes, posture = cfg["eyes"], cfg["posture"]
    df = df_p.copy()
    df["rel_pitch"] = eyes["pitch_sign"] * (df.pitch - cal["baseline_pitch"])
    df["ear_reliable"] = (df.face & df.ear.notna() & (df.yaw.abs() <= eyes["max_yaw_deg"])
                          & (df.rel_pitch <= eyes["max_pitch_down_deg"])
                          & (eyes["pitch_sign"] * df.pitch <= eyes["max_abs_pitch_down_deg"]))
    df["closed"] = df.ear_reliable & (df.ear < cal["threshold"])
    df["closed_raw"] = df.ear.notna() & (df.ear < cal["threshold"])   # naive baseline: no pose gating
    df["posture"] = [posture_flag(h, a, t, posture)
                     for h, a, t in zip(df.head_height, df.arm_contact, df.torso_angle)]
    df.loc[df.n_kp < posture["min_visible_keypoints"], "posture"] = "unknown"
    return df


def long_closure_seconds(df: pd.DataFrame, fps: float, blink_max: float) -> set[int]:
    """Seconds touched by an eye closure longer than a blink."""
    out: set[int] = set()
    dt = 1.0 / fps
    run_start = prev_t = None
    for t, closed in zip(df.time_sec, df.closed):
        if closed and run_start is not None and t - prev_t <= _MAX_FRAME_GAP * dt:
            prev_t = t
            continue
        if run_start is not None and (prev_t - run_start + dt) > blink_max:
            out.update(range(int(run_start), int(prev_t) + 1))
        run_start, prev_t = (t, t) if closed else (None, None)
    if run_start is not None and (prev_t - run_start + dt) > blink_max:
        out.update(range(int(run_start), int(prev_t) + 1))
    return out


def _mode(values: pd.Series) -> str:
    counts = values.value_counts()
    return str(counts.index[0]) if len(counts) else "unknown"


def aggregate_person(df: pd.DataFrame, n_frames: np.ndarray, fps: float, cfg: dict) -> pd.DataFrame:
    """One row per second for one person. `n_frames[s]` = processed frames in second s."""
    agg, motion = cfg["aggregation"], cfg["motion"]
    n_sec = len(n_frames)
    idx = pd.RangeIndex(n_sec, name="time_sec")
    df = df.assign(sec=df.time_sec.astype(int))
    g = df.groupby("sec")

    out = pd.DataFrame(index=idx)
    out["n_frames"] = n_frames
    out["n_det"] = g.size().reindex(idx, fill_value=0)
    frames_safe = np.maximum(out.n_frames, 1)
    out["present_frac"] = out.n_det / frames_safe
    out["visible"] = out.present_frac >= agg["visible_frac"]
    out["n_kp"] = g.n_kp.median().reindex(idx)
    out["body_visible"] = out.n_kp >= cfg["posture"]["min_visible_keypoints"]
    out["face_frac"] = g.face.sum().reindex(idx, fill_value=0) / frames_safe
    out["face_visible"] = out.face_frac >= agg["face_frac"]

    n_ear = g.ear_reliable.sum().reindex(idx, fill_value=0)
    n_closed = g.closed.sum().reindex(idx, fill_value=0)
    out["ear"] = df[df.ear_reliable].groupby("sec").ear.median().reindex(idx)
    out["eyes_measurable"] = n_ear / frames_safe >= agg["eye_frac"]
    out["closed_frac"] = (n_closed / n_ear.replace(0, np.nan))
    out["eyes_closed"] = out.eyes_measurable & (out.closed_frac >= agg["closed_frac"])
    out["eyes_open"] = out.eyes_measurable & (out.closed_frac < agg["closed_frac"])
    n_raw = g.ear.count().reindex(idx, fill_value=0)
    out["closed_frac_raw"] = g.closed_raw.sum().reindex(idx, fill_value=0) / n_raw.replace(0, np.nan)
    out["long_closure"] = out.index.isin(long_closure_seconds(df, fps, cfg["eyes"]["blink_max_seconds"]))

    # Only over a full window: early in the video a short window would turn one closure into
    # a huge fraction (6 s closed out of the first 14 s = 0.43).
    win = agg["perclos_window_seconds"]
    ear_sum = n_ear.rolling(win, min_periods=win).sum()
    frame_sum = out.n_frames.rolling(win, min_periods=win).sum()
    perclos = n_closed.rolling(win, min_periods=win).sum() / ear_sum.replace(0, np.nan)
    out["perclos"] = perclos.where(ear_sum >= agg["perclos_min_coverage"] * frame_sum)

    out["head_pitch"] = g.rel_pitch.median().reindex(idx)
    nod = agg["nod_window_seconds"]
    pmax = g.rel_pitch.max().reindex(idx).rolling(nod, min_periods=1).max()
    pmin = g.rel_pitch.min().reindex(idx).rolling(nod, min_periods=1).min()
    # A nod is the head dropping DOWN and swinging; looking up or shifting in the chair is not.
    out["nodding"] = ((pmax - pmin) >= agg["nod_range_deg"]) & (pmax >= agg["nod_down_deg"])

    out["motion"] = g.motion.median().reindex(idx)
    out["kp_motion"] = g.kp_motion.median().reindex(idx)
    out["still"] = out.motion < motion["still_threshold"]          # nan -> False
    out["moving"] = out.motion >= motion["wake_threshold"]

    out["posture_flag"] = g.posture.agg(_mode).reindex(idx).fillna("unknown")
    sleepy_frac = g.posture.agg(lambda s: s.isin(SLEEPY_POSTURES).mean()).reindex(idx, fill_value=0)
    out["posture_sleepy"] = sleepy_frac >= agg["posture_frac"]
    out["identity_source"] = g.identity_source.agg(_mode).reindex(idx)
    return out.reset_index()


def aggregate(frames: pd.DataFrame, people: list[str], processed_times: np.ndarray, fps: float,
              cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Calibrate and aggregate every person. Returns (per_second, frames_with_flags, calibration)."""
    n_sec = int(np.floor(processed_times.max())) + 1 if len(processed_times) else 0
    n_frames = np.bincount(np.floor(processed_times).astype(int), minlength=n_sec)[:n_sec]

    per_second, flagged, calibration = [], [], {}
    for person in people:
        df_p = frames[frames.person == person].sort_values("time_sec") if not frames.empty else frames
        if df_p.empty:
            log.warning("%s was never detected", person)
            df_p = _empty_frames()
        cal = calibrate(df_p, cfg["eyes"])
        calibration[person] = cal
        if not df_p.empty:
            log.info("%s: EAR baseline %.3f -> closed below %.3f (%s)", person,
                     cal["baseline_ear"], cal["threshold"], cal["method"])
        df_p = frame_flags(df_p, cal, cfg)
        flagged.append(df_p)
        per_second.append(aggregate_person(df_p, n_frames, fps, cfg).assign(person=person))

    per_sec = pd.concat(per_second, ignore_index=True) if per_second else pd.DataFrame()
    flagged_df = pd.concat(flagged, ignore_index=True) if flagged else frames
    return per_sec, flagged_df, calibration


def _empty_frames() -> pd.DataFrame:
    cols = {"time_sec": float, "n_kp": float, "face": bool, "ear": float, "pitch": float, "yaw": float,
            "motion": float, "kp_motion": float, "head_height": float, "arm_contact": float,
            "torso_angle": float, "identity_source": object}
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})
