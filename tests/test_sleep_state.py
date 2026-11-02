"""State machine: blinks vs sleep, hysteresis, visibility handling, body path."""
import math

import pandas as pd

from src.sleep_state import classify_person, intervals


def seconds(*segments: tuple[int, dict]) -> pd.DataFrame:
    """Build per-second rows from (n_seconds, overrides) segments. Default = awake, eyes open, still."""
    base = {"visible": True, "body_visible": True, "face_visible": True, "eyes_closed": False,
            "eyes_open": True, "still": True, "moving": False, "posture_sleepy": False,
            "posture_flag": "upright", "long_closure": False, "perclos": math.nan, "nodding": False}
    rows = []
    for n, overrides in segments:
        rows += [{**base, **overrides} for _ in range(n)]
    return pd.DataFrame(rows).assign(time_sec=range(len(rows)))


CLOSED = {"eyes_closed": True, "eyes_open": False}
FACE_HIDDEN = {"face_visible": False, "eyes_closed": False, "eyes_open": False}


def states(df: pd.DataFrame, cfg: dict) -> list[str]:
    return list(classify_person(df, cfg["sleep"]).state)


def test_blink_or_short_closure_is_not_sleep(cfg) -> None:
    # Eyes closed for 5 s (shorter than face_sleep_seconds = 10) -> drowsy at most, never asleep.
    out = states(seconds((10, {}), (5, CLOSED), (10, {})), cfg)
    assert "asleep" not in out
    assert out[10:15] == ["drowsy"] * 5
    assert out[-1] == "awake"


def test_sustained_closure_is_sleep_with_backfilled_onset(cfg) -> None:
    out = states(seconds((5, {}), (30, CLOSED), (5, {})), cfg)
    assert out[5:35] == ["asleep"] * 30          # onset relabelled back to second 5
    assert out[:5] == ["awake"] * 5
    assert out[-1] == "awake"


def test_no_backfill_gives_delayed_onset(cfg) -> None:
    cfg["sleep"]["backfill"] = False
    out = states(seconds((5, {}), (30, CLOSED)), cfg)
    first = out.index("asleep")
    assert first == 5 + cfg["sleep"]["face_sleep_seconds"] - 1


def test_hysteresis_one_second_of_open_eyes_does_not_wake(cfg) -> None:
    out = states(seconds((20, CLOSED), (1, {}), (20, CLOSED)), cfg)
    assert out == ["asleep"] * 41


def test_hysteresis_wakes_after_wake_seconds(cfg) -> None:
    wake = cfg["sleep"]["wake_seconds"]
    out = states(seconds((20, CLOSED), (wake, {}), (5, {})), cfg)
    assert out[:20] == ["asleep"] * 20
    assert out[20:] == ["awake"] * (wake + 5)    # wake-up seconds are backfilled too


def test_closed_eyes_but_moving_is_not_sleep(cfg) -> None:
    moving = {**CLOSED, "still": False, "moving": True}
    assert "asleep" not in states(seconds((30, moving)), cfg)


def test_face_only_mode_ignores_motion(cfg) -> None:
    cfg["sleep"]["use_body"] = False
    moving = {**CLOSED, "still": False, "moving": True}
    assert states(seconds((30, moving)), cfg) == ["asleep"] * 30


def test_body_path_needs_long_stillness_and_posture(cfg) -> None:
    n = cfg["sleep"]["body_sleep_seconds"]
    slumped = {**FACE_HIDDEN, "posture_sleepy": True, "posture_flag": "head_on_arms"}
    out = states(seconds((n + 10, slumped)), cfg)
    assert out == ["asleep"] * (n + 10)
    assert classify_person(seconds((n + 10, slumped)), cfg["sleep"]).confidence.iloc[0] == \
        cfg["sleep"]["confidence"]["asleep_body"]


def test_still_facing_away_without_sleep_posture_is_not_asleep(cfg) -> None:
    n = cfg["sleep"]["body_sleep_seconds"]
    out = states(seconds((n + 30, FACE_HIDDEN)), cfg)   # e.g. reading, facing away, upright
    assert "asleep" not in out


def test_body_path_shorter_than_threshold_is_not_sleep(cfg) -> None:
    n = cfg["sleep"]["body_sleep_seconds"]
    slumped = {**FACE_HIDDEN, "posture_sleepy": True}
    assert "asleep" not in states(seconds((n - 5, slumped), (10, {"still": False, "moving": True})), cfg)


def test_not_visible_is_unknown_and_pauses_counters(cfg) -> None:
    hidden = {"body_visible": False, "face_visible": False}
    out = states(seconds((6, CLOSED), (3, hidden), (6, CLOSED)), cfg)
    assert out[6:9] == ["unknown"] * 3                 # never guessed
    assert out[:6] == ["asleep"] * 6 and out[9:] == ["asleep"] * 6   # 6 + 6 >= 10 closed seconds


def test_absent_person(cfg) -> None:
    out = states(seconds((5, {}), (5, {"visible": False}), (5, {})), cfg)
    assert out[5:10] == ["absent"] * 5


def test_intervals_helper() -> None:
    s = pd.Series(range(8))
    st = pd.Series(["awake", "asleep", "asleep", "awake", "asleep", "asleep", "asleep", "awake"])
    assert intervals(s, st) == [(1, 3), (4, 7)]
