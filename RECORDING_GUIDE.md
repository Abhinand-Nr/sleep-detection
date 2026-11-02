# What you need to record and label (about 1–1.5 hours in total)

The code is finished, but the numbers for the report have to come from **your** videos.
This guide covers what to film, how to write the labels, and which commands to run.

---

## 1. Camera setup (same for every clip)

- **Fixed camera.** Put the laptop or phone on a stand and don't touch it while recording. If the camera moves, the motion signal breaks.
- **Landscape**, 720p or 1080p, 30 fps. A normal phone or laptop webcam is fine.
- **Frame head, shoulders, arms and the desk.** Sit about 1–1.5 m from the camera, not a close-up selfie. The body path needs to see your shoulders and arms, especially when you put your head down.
- Normal room light (except the dim-light clip). Avoid a bright window behind you.
- Save files as `.mp4` in `data/videos/`, using **exactly** the names below.

**Tip for easy labelling:** before each clip, write its "script" with times on paper, and start a stopwatch
on your phone together with the recording. Then labels = your script. Check them afterwards in VLC, which
shows the current time at the bottom.

---

## 2. The clips

"asleep" below means: **eyes closed, relaxed, not moving**, like really trying to nap.
Fake it convincingly: no fidgeting, slow breathing.

| # | File name | Length | What to do (script) | Labels to write |
|---|---|---|---|---|
| # | File name | Length | What to do (script) | Labels to write |
|---|---|---|---|---|
| 1 | `clip01_normal_work.mp4` | 2 min | Work normally on the laptop: type, read, blink naturally, look around a bit. | 0–120 awake |
| 2 | `clip02_eyes_closed_5s.mp4` | 2 min | Work normally. At **0:30, 1:00, 1:30** close your eyes for **5 s** and open them again. | awake, except each 5 s closure = **drowsy** |
| 3a | `clip03a_fake_sleep_face.mp4` | 3 min | 0:00–0:40 work. **0:40–1:40 eyes closed, face towards camera, sit still** (lean back in chair is fine). 1:40–2:10 work. **2:10–2:50 eyes closed again**. 2:50–3:00 work. | 0–40 awake, 40–100 asleep, 100–130 awake, 130–170 asleep, 170–180 awake |
| 3b | `clip03b_fake_sleep_face.mp4` | 3 min | Same idea as 3a but with **different times** (e.g. sleep 0:30–1:30 and 2:00–2:45). | as you did it |
| 4a | `clip04a_head_on_desk.mp4` | 3 min | 0:00–0:40 work. **0:40–2:10 fold your arms on the desk and put your head on them**, face down/hidden, still. 2:10–3:00 sit up and work. | 0–40 awake, 40–130 asleep, 130–180 awake |
| 4b | `clip04b_facing_away.mp4` | 3 min | **Turn the chair so your back or side faces the camera** for the whole clip. 0:00–0:40 do something (phone, stretch). **0:40–2:00 slump/lean your head down or onto the desk/wall, still.** 2:00–3:00 move around again. | 0–40 awake, 40–120 asleep, 120–180 awake |
| 5 | `clip05_phone.mp4` | 2 min | **Look down at your phone and scroll** the whole time (head down, eyes looking down). This is the known false-alarm risk. | 0–120 awake |
| 6 | `clip06_dim_light.mp4` | 3 min | Like 3a (sleep e.g. 0:40–1:40), but with only a dim lamp or the screen on. | as you did it |
| 7 | `clip07_glasses.mp4` | 3 min | Like 3a, wearing glasses (borrow a pair if needed). | as you did it |
| 8 | `clip08_two_people.mp4` | 4 min | Two people side by side. **First 15 s: both look at the camera** (needed for naming). Then person B fakes sleep from **1:00 to 2:30** (eyes closed; for extra credit put the head down facing away for part of it). Person A stays awake the whole time. | A: 0–240 awake. B: 0–60 awake, 60–150 asleep, 150–240 awake |

**Total filming: about 30 minutes.** Re-record any clip where the camera moved.

---

## 3. Reference photos (for clip 8 only)

- One clear, front-facing photo per person, good light, just their face (a phone selfie is fine).
- Save as `refs/<Name>.jpg`, e.g. `refs/Abhinand.jpg` and `refs/Rahul.jpg`.
- The file name **is** the name used in the output **and** in the labels CSV.

---

## 4. Writing the labels

One CSV per clip in `data/labels/`, with the same name as the video but ending in `.csv`. Copy `data/labels/TEMPLATE.csv` as a starting point:

```
start_sec,end_sec,person,state
0,40,Person 1,awake
40,100,Person 1,asleep
100,130,Person 1,awake
```

- Times are in seconds; `end_sec` is exclusive (40–100 means seconds 40 up to 99).
- `person` is **`Person 1`** for single-person clips, and the **refs file name** (e.g. `Abhinand`) for clip 8.
- `state` is one of: `awake`, `drowsy`, `asleep`, `unknown`, `absent`.
- Label what you **did**, starting from the moment your eyes closed or your head went down. The system's
  10 s / 45 s thresholds are its own business, so don't shift your labels to match them.
- If you walk out of frame, label that stretch `absent`.

---

## 5. Commands to run afterwards

Open PowerShell in the project folder:

```powershell
.\.venv\Scripts\Activate.ps1

# (a) quick look at one clip: creates outputs/clip03a_fake_sleep_face/
python run.py --video data/videos/clip03a_fake_sleep_face.mp4

# (b) the two-person clip, with names
python run.py --video data/videos/clip08_two_people.mp4 --refs refs/

# (c) full evaluation: tunes on the "tune" clips, reports on the "eval" clips
copy data\manifest.example.csv data\manifest.csv     # then edit names if yours differ
python evaluate.py
```

- Each clip takes roughly **2–3× its length** to process on a laptop CPU the first time.
  The signals are cached in `outputs/<clip>/signals.pkl`, so re-runs and `evaluate.py` are fast.
- `evaluate.py` writes `outputs/evaluation/results_table.md` (paste into the report),
  `metrics.json`, `tuned_params.yaml`, and one timeline per clip comparing labels vs. predictions.

---

## 6. Two quick checks while you record

1. **Head-pitch direction.** Run clip 5 (phone). In `outputs/clip05_phone/results.csv`, `head_pitch`
   should be **positive** (roughly +10 to +40) while you look down. If it is negative, set `pitch_sign: -1`
   in `config.yaml`. My maths says positive, but I could not test it on a real face.
2. **Annotated video.** Open `outputs/<clip>/annotated.mp4` in VLC and check that the box follows you and the
   state label makes sense.

When you're done, tell Claude "clips recorded, run the evaluation". It will run everything,
tune, look at the failure cases and write up the facts and numbers for your report.
