"""Visibility-aware fusion and the per-person sleep state machine.

Each second the person is in one of: awake, drowsy, asleep, unknown, absent.

Entering asleep needs SUSTAINED evidence, from whichever signal is available:
  face path : eyes closed AND body still, for face_sleep_seconds in a row   -> high confidence
  body path : body still AND sleep posture (eyes not seen open), for
              body_sleep_seconds in a row                                   -> medium confidence
Leaving asleep needs clear wake evidence (eyes open, or clear movement) for wake_seconds
in a row. This asymmetry (hysteresis) stops the state flickering on a single noisy second.

Seconds where the person is present but barely visible are `unknown` (never guessed), and
they pause the counters instead of resetting them, so a 1 s occlusion does not undo 40 s of
evidence. `absent` (not in frame) resets everything.

Offline backfill: when sleep is confirmed after N seconds, those N seconds are relabelled
asleep, because that is when sleep actually started. A real-time version would skip this
and accept an N-second delay.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

AWAKE, DROWSY, ASLEEP, UNKNOWN, ABSENT = "awake", "drowsy", "asleep", "unknown", "absent"
STATES = (AWAKE, DROWSY, ASLEEP, UNKNOWN, ABSENT)


@dataclass
class _Tracker:
    """Counters for one person."""
    closed_run: int = 0      # consecutive seconds of face-path sleep evidence
    still_run: int = 0       # consecutive seconds of body-path sleep evidence
    wake_run: int = 0        # consecutive seconds of wake evidence while asleep
    asleep: bool = False
    source: str = ""         # "face" or "body": how the current sleep was detected
    last_wake_sec: int = -10**9


def _num(v) -> bool:
    """True for a real True; pandas may hand us numpy bools or NaN."""
    return bool(v) if v == v else False   # NaN != NaN


def classify_person(rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add state / confidence / reason columns to one person's per-second rows (sorted by time)."""
    conf = cfg["confidence"]
    use_body = cfg["use_body"]
    st = _Tracker()
    states: list[str] = []
    confs: list[float] = []
    reasons: list[str] = []

    def backfill(n: int, state: str, c: float, reason: str) -> None:
        """Relabel the previous n seconds (offline only), skipping unknown/absent ones."""
        if not cfg["backfill"]:
            return
        i = len(states) - 1
        while n > 0 and i >= 0 and states[i] != ABSENT:
            if states[i] != UNKNOWN:      # unknown seconds were paused, not counted; leave them
                states[i], confs[i], reasons[i] = state, c, reason
                n -= 1
            i -= 1

    for r in rows.itertuples(index=False):
        sec = int(r.time_sec)
        eyes_closed, eyes_open = _num(r.eyes_closed), _num(r.eyes_open)
        still, moving = _num(r.still), _num(r.moving)

        # 1) Is there anything to judge?
        if not _num(r.visible):
            st = _Tracker(last_wake_sec=st.last_wake_sec)
            states.append(ABSENT); confs.append(conf["absent"]); reasons.append("not in frame")
            continue
        if not (_num(r.body_visible) or _num(r.face_visible)):
            # Present but barely visible: do not guess; keep counters as they are.
            states.append(UNKNOWN); confs.append(conf["unknown"]); reasons.append("mostly not visible")
            continue

        # 2) Evidence this second.
        face_sleep = eyes_closed and (still or not use_body)
        body_sleep = use_body and still and _num(r.posture_sleepy) and not eyes_open
        wake = eyes_open or (use_body and moving)

        st.closed_run = st.closed_run + 1 if face_sleep else 0
        st.still_run = st.still_run + 1 if body_sleep else 0

        # 3) Transitions.
        if st.asleep:
            st.wake_run = st.wake_run + 1 if wake else 0
            if st.wake_run >= cfg["wake_seconds"]:
                why = "eyes open" if eyes_open else "moving"
                n_back = st.wake_run - 1
                st.asleep, st.closed_run, st.still_run, st.wake_run = False, 0, 0, 0
                st.last_wake_sec = sec
                states.append(AWAKE); confs.append(conf["awake_eyes_open"] if eyes_open else conf["awake_moving"])
                reasons.append(f"woke up ({why} {cfg['wake_seconds']}s)")
                backfill(n_back + 1, AWAKE, confs[-1], reasons[-1])
                continue
            c = conf["asleep_face"] if st.source == "face" else conf["asleep_body"]
            states.append(ASLEEP); confs.append(c)
            reasons.append("still asleep" + (" (wake signs, waiting)" if st.wake_run else ""))
            continue

        if st.closed_run >= cfg["face_sleep_seconds"]:
            st.asleep, st.source, st.wake_run = True, "face", 0
            reason = f"eyes closed + still {st.closed_run}s"
            states.append(ASLEEP); confs.append(conf["asleep_face"]); reasons.append(reason)
            backfill(st.closed_run, ASLEEP, conf["asleep_face"], reason)
            continue
        if st.still_run >= cfg["body_sleep_seconds"]:
            st.asleep, st.source, st.wake_run = True, "body", 0
            reason = f"still + {r.posture_flag} posture {st.still_run}s (face not usable)"
            states.append(ASLEEP); confs.append(conf["asleep_body"]); reasons.append(reason)
            backfill(st.still_run, ASLEEP, conf["asleep_body"], reason)
            continue

        # 4) Not asleep: drowsy or awake.
        # PERCLOS looks back over a window, so right after waking it still "remembers" the
        # sleep; ignore it during the cooldown so a woken person is not called drowsy.
        recently_woke = sec - st.last_wake_sec < cfg["perclos_cooldown_seconds"]
        perclos_high = (r.perclos == r.perclos) and r.perclos >= cfg["perclos_drowsy"] and not recently_woke
        if eyes_closed or _num(r.long_closure):
            states.append(DROWSY); confs.append(conf["drowsy"]); reasons.append("eyes closed longer than a blink")
        elif perclos_high:
            states.append(DROWSY); confs.append(conf["drowsy"]); reasons.append(f"PERCLOS {r.perclos:.2f}")
        elif _num(r.nodding):
            states.append(DROWSY); confs.append(conf["drowsy"]); reasons.append("head nodding")
        elif eyes_open:
            states.append(AWAKE); confs.append(conf["awake_eyes_open"]); reasons.append("eyes open")
        elif use_body and not still:
            states.append(AWAKE); confs.append(conf["awake_moving"]); reasons.append("moving")
        else:
            states.append(AWAKE); confs.append(conf["awake_default"]); reasons.append("no sleep evidence yet")

    return rows.assign(state=states, confidence=confs, reason=reasons)


def classify(per_second: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run the state machine independently for every person."""
    if per_second.empty:
        return per_second.assign(state=pd.Series(dtype=str), confidence=pd.Series(dtype=float),
                                 reason=pd.Series(dtype=str))
    parts = [classify_person(g.sort_values("time_sec"), cfg) for _, g in per_second.groupby("person", sort=False)]
    return pd.concat(parts, ignore_index=True)


def naive_states(per_second: pd.DataFrame, closed_frac: float) -> pd.Series:
    """Baseline 1: a second is asleep if EAR was below threshold in >= closed_frac of its frames.

    No temporal logic, no pose gating, no body signals.
    """
    closed = per_second.closed_frac_raw.fillna(0) >= closed_frac
    state = pd.Series(AWAKE, index=per_second.index)
    state[closed] = ASLEEP
    state[~per_second.visible.astype(bool)] = ABSENT
    return state


def intervals(seconds: pd.Series, state: pd.Series, target: str = ASLEEP) -> list[tuple[int, int]]:
    """Contiguous [start, end) second ranges where state == target."""
    out: list[tuple[int, int]] = []
    start = prev = None
    for s, v in zip(seconds.astype(int), state):
        if v == target:
            if start is None or s != prev + 1:
                if start is not None:
                    out.append((start, prev + 1))
                start = s
            prev = s
        elif start is not None:
            out.append((start, prev + 1))
            start = prev = None
    if start is not None:
        out.append((start, prev + 1))
    return out
