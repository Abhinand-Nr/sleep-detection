"""Cut a short highlight reel from the annotated videos in outputs/<clip>/annotated.mp4.

Usage: python scripts/make_demo.py --ffmpeg <path to ffmpeg> [--out outputs/demo/demo_highlights.mp4]
Each segment gets a caption bar; title cards separate the parts. Output is H.264 at 10 fps.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FPS = 10
W, H = 1280, 720
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"

# (title card, [(clip, start_s, end_s, caption)])
PARTS = [
    ("1. Eyes closed and still for 10 s = asleep", [
        ("clip03a_fake_sleep_face", 36, 58, "Eyes close at 0:40. Short closure = drowsy; closed and still for 10 s = ASLEEP (face path, confidence 0.9)"),
        ("clip03a_fake_sleep_face", 84, 95, "Eyes open for 2 s = AWAKE again (hysteresis: one noisy frame can't flip the state)"),
    ]),
    ("2. Not every eye closure is sleep", [
        ("clip02_eyes_closed_5s", 28, 40, "A 5 s eye closure is only DROWSY, never asleep. Blinks under 0.5 s are ignored."),
        ("clip05_phone", 4, 22, "Looking down at a phone makes eyes look closed. Head angle > 20 deg down = eyes not trusted, so it stays AWAKE."),
    ]),
    ("3. Face hidden: the body still tells the story", [
        ("clip04a_head_on_desk", 36, 60, "Head down on the arms, face not visible: still + head-on-arms posture for 45 s = ASLEEP (body path, 0.65)"),
        ("clip04a_head_on_desk", 124, 134, "Sits up and moves = AWAKE"),
        ("clip06_dim_light", 40, 56, "Dim room, webcam at 16 fps: still detected"),
    ]),
    ("4. Two people: who is asleep, by name", [
        ("clip08_two_people", 0, 8, "Each person is named by matching their face to one reference photo (SFace)"),
        ("clip08_two_people", 18, 40, "Abhinand closes his eyes and is tagged ASLEEP; Rajni stays AWAKE"),
        ("clip08_two_people", 78, 100, "Rajni falls ASLEEP; the name stays on her track the whole time"),
    ]),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def title_card(text: str, sub: str = "") -> np.ndarray:
    img = Image.new("RGB", (W, H), (20, 32, 52))
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - (20 if sub else 0)), text, font=font(46, True), fill="white", anchor="mm")
    if sub:
        d.text((W // 2, H // 2 + 45), sub, font=font(26), fill=(170, 195, 230), anchor="mm")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def captioned(frame: np.ndarray, caption: str) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, H - 64, W, H), fill=(20, 32, 52, 215))
    d.text((W // 2, H - 32), caption, font=font(24), fill="white", anchor="mm")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def segment(clip: str, start: float, end: float):
    cap = cv2.VideoCapture(str(ROOT / "outputs" / clip / "annotated.mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * src_fps))
    idx, frame = int(start * src_fps) - 1, None
    for t in np.arange(start, end, 1 / FPS):       # resample to FPS by time
        while idx < int(t * src_fps):
            ok, f = cap.read()
            if not ok:
                cap.release()
                return
            idx, frame = idx + 1, f
        yield frame
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo" / "demo_highlights.mp4"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([args.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "24", "-preset", "medium", "-movflags", "+faststart", args.out],
                            stdin=subprocess.PIPE)

    def hold(img: np.ndarray, seconds: float) -> None:
        for _ in range(int(seconds * FPS)):
            proc.stdin.write(img.tobytes())

    hold(title_card("Sleep Detection from Video", "Demo highlights  ·  pretrained models + explainable rules  ·  Abhinand"), 3.5)
    for title, segments in PARTS:
        hold(title_card(title), 2.5)
        for clip, start, end, caption in segments:
            for frame in segment(clip, start, end):
                proc.stdin.write(captioned(frame, caption).tobytes())
    hold(title_card("Eval clips: asleep precision 0.95 · recall 0.95 · F1 0.95",
                    "Naive 'eyes closed = asleep' baseline: F1 0.72  ·  full details in the write-up"), 4)
    proc.stdin.close()
    proc.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
