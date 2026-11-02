"""Per-second aggregation, calibration and blink detection on synthetic frame data."""
import numpy as np
import pandas as pd
import pytest

from src.aggregation import aggregate, calibrate, frame_flags, long_closure_seconds

FPS = 10


def frames(ears: list[float], motion: float = 0.0) -> pd.DataFrame:
    """One person, one row per processed frame at 10 fps, frontal face."""
    n = len(ears)
    nan = float("nan")
    return pd.DataFrame({
        "frame_index": range(n), "time_sec": np.arange(n) / FPS, "track_id": 1, "person": "Person 1",
        "x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 200.0, "det_conf": 0.9, "n_kp": 12,
        "motion": motion, "kp_motion": 0.0, "head_height": 0.8, "arm_contact": 2.0, "torso_angle": 5.0,
        "face": True, "ear": ears, "pitch": 0.0, "yaw": 0.0, "identity_source": "single", "kps": [None] * n,
    }).assign(face_w=60.0, ear_left=nan, ear_right=nan)


def test_calibration_uses_open_eye_level(cfg) -> None:
    ears = [0.3] * 90 + [0.1] * 10       # mostly open, a few blinks
    cal = calibrate(frames(ears), cfg["eyes"])
    assert cal["method"] == "first_seconds"
    assert cal["baseline_ear"] == pytest.approx(0.3)
    assert cal["threshold"] == pytest.approx(0.3 * cfg["eyes"]["closed_ratio"])


def test_calibration_falls_back_when_eyes_start_closed(cfg) -> None:
    ears = [0.1] * 150 + [0.3] * 400     # asleep for the first 15 s
    cal = calibrate(frames(ears), cfg["eyes"])
    assert cal["method"] == "whole_video"
    assert cal["baseline_ear"] == pytest.approx(0.3)


def test_blink_is_not_a_long_closure(cfg) -> None:
    ears = [0.3] * 20 + [0.1] * 3 + [0.3] * 20 + [0.1] * 12 + [0.3] * 5   # 0.3 s blink, 1.2 s closure
    df = frame_flags(frames(ears), calibrate(frames(ears), cfg["eyes"]), cfg)
    long = long_closure_seconds(df, FPS, cfg["eyes"]["blink_max_seconds"])
    assert 2 not in long                 # blink at 2.0-2.3 s
    assert {4, 5} <= long                # closure at 4.3-5.5 s


def test_per_second_rows_and_fractions(cfg) -> None:
    ears = [0.3] * 10 + [0.3] * 7 + [0.1] * 3 + [0.1] * 10    # second 1 has a blink, second 2 closed
    df = frames(ears)
    per_sec, _, _ = aggregate(df, ["Person 1"], df.time_sec.to_numpy(), FPS, cfg)
    assert list(per_sec.time_sec) == [0, 1, 2]
    assert list(per_sec.eyes_closed) == [False, False, True]
    assert list(per_sec.eyes_open) == [True, True, False]
    assert per_sec.closed_frac.iloc[1] == pytest.approx(0.3)
    assert per_sec.visible.all() and per_sec.face_visible.all()


def test_perclos_needs_a_full_window(cfg) -> None:
    win = cfg["aggregation"]["perclos_window_seconds"]
    ears = ([0.3] * 80 + [0.1] * 60) + [0.3] * (win * FPS)   # 6 s closure early in the video
    df = frames(ears)
    per_sec, _, _ = aggregate(df, ["Person 1"], df.time_sec.to_numpy(), FPS, cfg)
    assert per_sec.perclos.iloc[: win - 1].isna().all()
    assert per_sec.perclos.iloc[win - 1] == pytest.approx(60 / (win * FPS))


def test_nodding_needs_the_head_to_drop(cfg) -> None:
    pitch = [0.0] * 30 + [-20.0] * 10 + [0.0] * 30 + [25.0] * 10 + [0.0] * 20   # look up, then drop down
    df = frames([0.3] * len(pitch)).assign(pitch=pitch)
    per_sec, _, _ = aggregate(df, ["Person 1"], df.time_sec.to_numpy(), FPS, cfg)
    assert not per_sec.nodding.iloc[3:5].any()     # looking up is not a nod
    assert per_sec.nodding.iloc[7]                 # head dropped 25 degrees


def test_absent_seconds_are_filled(cfg) -> None:
    df = frames([0.3] * 10)
    times = np.arange(30) / FPS                       # video is 3 s, person only in the first
    per_sec, _, _ = aggregate(df, ["Person 1"], times, FPS, cfg)
    assert list(per_sec.visible) == [True, False, False]


def test_turned_head_makes_ear_unreliable(cfg) -> None:
    df = frames([0.1] * 20).assign(yaw=cfg["eyes"]["max_yaw_deg"] + 10)
    per_sec, _, _ = aggregate(df, ["Person 1"], df.time_sec.to_numpy(), FPS, cfg)
    assert not per_sec.eyes_measurable.any()
    assert not per_sec.eyes_closed.any()


def test_motion_threshold(cfg) -> None:
    still = cfg["motion"]["still_threshold"]
    df = frames([0.3] * 10, motion=still / 2)
    per_sec, _, _ = aggregate(df, ["Person 1"], df.time_sec.to_numpy(), FPS, cfg)
    assert per_sec.still.iloc[0]


def test_looking_down_from_the_start_is_not_learned_as_normal(cfg) -> None:
    df = frames([0.08] * 200).assign(pitch=25.0)     # phone from second 0: head down, eyes half-shut
    per_sec, _, _ = aggregate(df, ["Person 1"], df.time_sec.to_numpy(), FPS, cfg)
    assert not per_sec.eyes_closed.any()
