"""Write results: results.csv, summary.json, annotated.mp4, timeline.png, and log warnings."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")   # no window; just files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.sleep_state import ABSENT, ASLEEP, AWAKE, DROWSY, STATES, UNKNOWN, intervals  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.video_io import VideoReader  # noqa: E402

log = get_logger(__name__)

CSV_COLUMNS = ["time_sec", "person", "visible", "face_visible", "ear", "head_pitch", "motion",
               "posture_flag", "state", "confidence", "reason",
               # extra diagnostic columns
               "closed_frac", "perclos", "kp_motion", "n_kp", "identity_source"]

# Validated categorical slots 1-3 (blue / orange / aqua) + neutrals for "no judgement".
STATE_COLORS = {ASLEEP: "#2a78d6", DROWSY: "#eb6834", AWAKE: "#1baf7a", UNKNOWN: "#a3a29c", ABSENT: "#e4e3df"}
_BGR = {k: tuple(int(v[i:i + 2], 16) for i in (5, 3, 1)) for k, v in STATE_COLORS.items()}


def hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}" if sec < 3600 else f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def write_csv(per_second: pd.DataFrame, path: Path) -> None:
    df = per_second.reindex(columns=CSV_COLUMNS).copy()
    for col in ("ear", "head_pitch", "motion", "closed_frac", "perclos", "kp_motion", "confidence"):
        df[col] = df[col].astype(float).round(4)
    df.to_csv(path, index=False)


def build_summary(per_second: pd.DataFrame, calibration: dict, video: Path, cfg: dict,
                  identities: dict | None = None) -> dict:
    people = {}
    lines = []
    for person, g in per_second.groupby("person", sort=False):
        g = g.sort_values("time_sec")
        ivs = intervals(g.time_sec, g.state, ASLEEP)
        total = sum(e - s for s, e in ivs)
        mean_conf = [float(g[(g.time_sec >= s) & (g.time_sec < e)].confidence.mean()) for s, e in ivs]
        counts = {s: int((g.state == s).sum()) for s in STATES}
        people[person] = {
            "asleep_intervals": [
                {"start_sec": s, "end_sec": e, "start": hms(s), "end": hms(e), "duration_sec": e - s,
                 "mean_confidence": round(c, 3)} for (s, e), c in zip(ivs, mean_conf)],
            "total_asleep_sec": total,
            "seconds_per_state": counts,
            "identity_source": g.identity_source.dropna().mode().iloc[0] if g.identity_source.notna().any() else None,
            "calibration": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in calibration.get(person, {}).items()},
        }
        if ivs:
            lines.append(f"{person} asleep " + ", ".join(f"{hms(s)}-{hms(e)}" for s, e in ivs)
                         + f" (total {hms(total)})")
        elif counts[AWAKE] + counts[DROWSY] > 0:
            lines.append(f"{person} awake throughout")
        else:
            lines.append(f"{person} not judged (never clearly visible)")

    return {"video": str(video), "summary": lines, "people": people,
            "definition": {"face_sleep_seconds": cfg["sleep"]["face_sleep_seconds"],
                           "body_sleep_seconds": cfg["sleep"]["body_sleep_seconds"],
                           "wake_seconds": cfg["sleep"]["wake_seconds"]}}


def write_summary(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def log_warnings(per_second: pd.DataFrame, cfg: dict) -> None:
    """Warnings that help a human decide how much to trust the output."""
    out = cfg["outputs"]
    for person, g in per_second.groupby("person", sort=False):
        g = g.sort_values("time_sec")
        # Long stretches where the person is visible but their face is not.
        hidden = g.visible.astype(bool) & ~g.face_visible.astype(bool)
        for s, e in intervals(g.time_sec, hidden.map({True: "x", False: ""}), "x"):
            if e - s >= out["face_unseen_warn_seconds"]:
                log.warning("%s face not seen for %d s (%s-%s); relying on body signals", person, e - s, hms(s), hms(e))
        for s, e in intervals(g.time_sec, g.state, ASLEEP):
            c = g[(g.time_sec >= s) & (g.time_sec < e)].confidence.mean()
            if c < out["low_confidence_warn"]:
                log.warning("%s asleep %s-%s has low confidence (%.2f): body-only evidence", person, hms(s), hms(e), c)
        if (g.state == UNKNOWN).mean() > 0.2:
            log.warning("%s is 'unknown' for %.0f%% of the video (mostly not visible)", person,
                        100 * (g.state == UNKNOWN).mean())


def plot_timeline(per_second: pd.DataFrame, path: Path, title: str,
                  truth: pd.DataFrame | None = None) -> None:
    """Per-person state bands, plus EAR and motion traces. `truth` adds hand-label rows."""
    people = list(dict.fromkeys(per_second.person))
    rows: list[tuple[str, pd.DataFrame]] = []
    for p in people:
        if truth is not None and p in set(truth.person):
            rows.append((f"{p} (labels)", truth[truth.person == p]))
        rows.append((p, per_second[per_second.person == p]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 2.0 + 0.5 * len(rows) + 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [max(1.2, 0.5 * len(rows)), 1.7, 1.7]})
    ax_state, ax_ear, ax_mot = axes
    for i, (label, g) in enumerate(rows):
        y = len(rows) - 1 - i
        for state in STATES:
            for s, e in intervals(g.time_sec, g.state, state):
                ax_state.broken_barh([(s, e - s)], (y - 0.35, 0.7), facecolors=STATE_COLORS[state],
                                     edgecolor="white", linewidth=1, hatch="//" if state == UNKNOWN else None)
    ax_state.set_yticks(range(len(rows)))
    ax_state.set_yticklabels([r[0] for r in reversed(rows)])
    ax_state.set_title(title, loc="left", fontsize=11)
    ax_state.legend(handles=[Patch(facecolor=STATE_COLORS[s], label=s, hatch="//" if s == UNKNOWN else None)
                             for s in STATES], ncol=5, loc="upper left", bbox_to_anchor=(0, -0.08),
                    frameon=False, fontsize=8)

    line_styles = ["-", "--", ":", "-."]
    for i, p in enumerate(people):
        g = per_second[per_second.person == p]
        ax_ear.plot(g.time_sec, g.ear, color="#52514e", linestyle=line_styles[i % 4], linewidth=1.5, label=p)
        ax_mot.plot(g.time_sec, g.motion, color="#52514e", linestyle=line_styles[i % 4], linewidth=1.5, label=p)
    ax_ear.set_ylabel("EAR")
    ax_mot.set_ylabel("motion")
    ax_mot.set_xlabel("time (s)")
    if len(people) > 1:
        ax_ear.legend(frameon=False, fontsize=8)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_annotated_video(video: Path, frames: pd.DataFrame, per_second: pd.DataFrame, cfg: dict,
                          path: Path) -> None:
    """Re-read the video and draw each person's box, name, state and confidence."""
    state_at = {(r.person, int(r.time_sec)): (r.state, r.confidence)
                for r in per_second[["person", "time_sec", "state", "confidence"]].itertuples(index=False)}
    by_frame = {k: g for k, g in frames.groupby("frame_index")} if not frames.empty else {}

    with VideoReader(video, cfg["video"]["process_fps"], cfg["video"]["max_side"]) as reader:
        writer = None
        for frame in reader:
            img = frame.image
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), reader.effective_fps, (w, h))
            for r in by_frame.get(frame.index, pd.DataFrame()).itertuples(index=False):
                state, conf = state_at.get((r.person, int(r.time_sec)), (UNKNOWN, 0.0))
                color = _BGR[state]
                p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
                cv2.rectangle(img, p1, p2, color, 3)
                kps = r.kps
                for x, y in kps[~np.isnan(kps).any(axis=1)]:
                    cv2.circle(img, (int(x), int(y)), 3, color, -1)
                text = f"{r.person}: {state} ({conf:.2f})"
                ear = f" EAR {r.ear:.2f}" if r.ear == r.ear else ""
                _label(img, text + ear, (p1[0], max(20, p1[1] - 8)), color)
            _label(img, hms(frame.time_sec), (10, 30), (40, 40, 40))
            writer.write(img)
        if writer is not None:
            writer.release()


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = org
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), color, -1)
    cv2.putText(img, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
