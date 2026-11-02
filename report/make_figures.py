"""Build the figures used in the 2-page write-up.

Run from the project root with the project venv:
    .venv/Scripts/python.exe report/make_figures.py

Reads only existing outputs (results.csv, labels, annotated.mp4); writes to report/figures/.
Numbers in the baseline chart are copied from outputs/evaluation/ and REPORT_FACTS.md.
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1B2330"
MUTED = "#6B7482"
GRID = "#E3E6EB"
STATE = {"awake": "#3C9D6E", "drowsy": "#E39A3B", "asleep": "#2F5FB3",
         "unknown": "#A9AFB8", "absent": "#E6E8EC"}
SYSTEM = {"naive": "#C3C8D0", "face-only": "#8DAAD9", "full": "#1E3F7A"}

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 7.5,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "svg.fonttype": "none",
})


def still_frame(t_sec=45):
    cap = cv2.VideoCapture(str(ROOT / "outputs/clip08_two_people/annotated.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ok, frame = cap.read()
    assert ok, "could not read frame"
    h, w = frame.shape[:2]
    frame = frame[int(h * 0.075):, :]  # drop the timestamp strip at the top
    cv2.imwrite(str(OUT / "frame_clip08.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])


def baselines():
    systems = ["naive", "face-only", "full"]
    panels = [
        ("Asleep F1, eval split", [0.72, 0.88, 0.95], 1.0, "{:.2f}"),
        ("False ‘asleep’ seconds, phone clip (of 126)", [117, 48, 0], 126, "{:d}"),
        ("Missed sleep seconds, head on desk (of 79)", [None, 79, 1], 79, "{:d}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.12), sharey=True)
    for ax, (title, vals, vmax, fmt) in zip(axes, panels):
        for i, (s, v) in enumerate(zip(systems, vals)):
            y = len(systems) - 1 - i
            if v is None:
                ax.text(0, y, "not measured", va="center", ha="left", color=MUTED,
                        fontsize=6.8, style="italic")
                continue
            ax.barh(y, v, height=0.62, color=SYSTEM[s])
            ax.text(v + vmax * 0.02, y, fmt.format(v), va="center", ha="left",
                    fontsize=7.2, color=INK, fontweight="bold" if s == "full" else "normal")
        ax.set_xlim(0, vmax * 1.16)
        ax.set_title(title, fontsize=7.6, loc="left", color=INK, pad=4, fontweight="semibold")
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(systems)))
    axes[0].set_yticklabels(list(reversed(systems)))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.78, bottom=0.03, wspace=0.12)
    fig.savefig(OUT / "baselines.svg")
    plt.close(fig)


def runs(states):
    """Turn a per-second state list into (start, length, state) runs."""
    out, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            out.append((start, i - start, states[start]))
            start = i
    return out


def timeline():
    pred = pd.read_csv(ROOT / "outputs/clip08_two_people/results.csv")
    labels = pd.read_csv(ROOT / "data/labels/clip08_two_people.csv")
    rows = []
    for person in ["Abhinand", "Rajni"]:
        lab = labels[labels.person == person]
        rows.append((f"{person} · label", [(r.start_sec, r.end_sec - r.start_sec, r.state)
                                           for r in lab.itertuples()]))
        p = pred[pred.person == person].sort_values("time_sec")
        rows.append((f"{person} · system", runs(list(p.state))))

    fig, ax = plt.subplots(figsize=(7.1, 1.3))
    ypos = [3.25, 2.45, 1.0, 0.2]
    for (name, segs), y in zip(rows, ypos):
        ax.broken_barh([(s, l) for s, l, _ in segs], (y - 0.32, 0.64),
                       facecolors=[STATE[st] for _, _, st in segs], linewidth=0)
        ax.text(-3, y, name, ha="right", va="center", fontsize=7.2,
                color=INK if "label" in name else MUTED)
    ax.vlines(127, -0.15, 1.38, colors=MUTED, linestyles=(0, (2, 2)), lw=0.6)
    ax.text(128.5, 1.62, "Rajni's track ID switches at 2:07; the name is kept", ha="left",
            va="center", fontsize=6.4, color=MUTED)
    ax.set_xlim(0, 212)
    ax.set_ylim(-0.25, 3.7)
    ax.set_yticks([])
    ax.set_xticks(range(0, 211, 30))
    ax.set_xticklabels([f"{t // 60}:{t % 60:02d}" for t in range(0, 211, 30)], fontsize=6.8)
    ax.tick_params(axis="x", length=2, color=GRID)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE[s]) for s in ("awake", "drowsy", "asleep")]
    ax.legend(handles, ["awake", "drowsy", "asleep"], ncol=3, frameon=False, fontsize=7,
              loc="lower right", bbox_to_anchor=(1.0, 0.98), handlelength=1.0,
              handleheight=0.8, columnspacing=1.2, borderaxespad=0)
    fig.subplots_adjust(left=0.135, right=0.995, top=0.87, bottom=0.17)
    fig.savefig(OUT / "timeline_clip08.svg")
    plt.close(fig)


if __name__ == "__main__":
    still_frame()
    baselines()
    timeline()
    print("figures written to", OUT)
