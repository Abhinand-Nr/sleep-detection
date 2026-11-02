# Sleep Detection from Video

Given a video, this system outputs a **per-second judgement** of whether each person is asleep, plus a
summary of asleep intervals. With reference photos it names people, e.g.
*"Rahul asleep 01:00–02:30, Abhinand awake throughout"*.

It uses only **pretrained models plus explainable rules**. Nothing is trained. It keeps working when
people **face away from the camera**.

## Assumptions

- Fixed camera, indoor lighting, offline processing of recorded video (designed so a real-time
  version needs only small changes; see Limitations).
- People may face the camera, be side-on, or face away.
- For naming, each person shows a clear face at least once, and a reference photo is supplied as `refs/<Name>.jpg`.
- Without `--refs` the video is treated as single-person and the person is called "Person 1".

## Operational definition of "asleep"

Sleep can't be measured directly from video, so we define **observable behavioural sleep**:

| What is visible | Asleep if… | Confidence |
|---|---|---|
| Face, eyes measurable | eyes closed (EAR < 0.5 × the person's own open-eye EAR) continuously **≥ 10 s** AND body still | high (0.9) |
| Body, but no usable eyes (facing away, head down) | body still **≥ 45 s** AND sleep posture (slumped / head on arms / lying) | medium (0.65) |
| Person present but mostly not visible | `unknown` (never guessed) | — |

Other states: `awake` (eyes open or moving), `drowsy` (closure longer than a 0.5 s blink but shorter than
10 s, high PERCLOS, or head nodding), `absent` (not in frame). Blinks are ignored.
**Hysteresis:** leaving `asleep` needs 2 s in a row of open eyes or clear movement.
EAR is only trusted when the head is roughly frontal (|yaw| ≤ 35°, not tilted down more than 25° from the
person's normal, and never more than 20° down from the camera), because looking down makes the eyes *look*
closed. Calibration only uses frames facing the camera, so starting a video on the phone can't teach the
system that "head down" is normal.

All numbers above are in `config.yaml`.

## Pipeline

```
video → YOLOv8n-pose + ByteTrack (people, 17 keypoints, track IDs)
      → identity: SFace face match vs refs/ (votes per track, kept while the face is hidden)
                  + clothing-colour histogram as backup re-ID
      → per person, per frame: EAR + head pitch/yaw (YuNet → MediaPipe Face Landmarker),
                               motion energy (pixel change in box − background), posture (keypoints)
      → per-second aggregation (fractions of frames)
      → visibility-aware fusion + state machine with hysteresis
      → results.csv, summary.json, annotated.mp4, timeline.png, run.log
```

| Module | Role |
|---|---|
| `src/video_io.py` | read video, subsample to `process_fps`, timestamps |
| `src/detection_tracking.py` | YOLO-pose + ByteTrack, face-to-person matching |
| `src/face_signals.py` | YuNet faces, MediaPipe landmarks, EAR, head pose |
| `src/body_signals.py` | motion energy, keypoint motion, posture |
| `src/identity.py` | SFace votes, clothing histograms, whole-video name assignment |
| `src/aggregation.py` | per-person calibration, per-second features |
| `src/sleep_state.py` | fusion, state machine, hysteresis, naive baseline |
| `src/outputs.py` | CSV, JSON, annotated video, timeline plot, warnings |
| `src/pipeline.py` | wires it together; caches extracted signals |

## Install

Python 3.12 (Windows or Linux, CPU only).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
```

On Linux, install the CPU build of PyTorch first to avoid a 2 GB CUDA download (see `Dockerfile`).
Docker (tested: image builds, the test suite passes inside it, a 2-minute clip runs in about 2.3 minutes on CPU):

```bash
docker build -t sleep-detector .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/refs:/app/refs" -v "$PWD/outputs:/app/outputs"     sleep-detector --video data/videos/clip01_normal_work.mp4
```

On Windows PowerShell, use `${PWD}` instead of `$PWD`. The image is about 3.3 GB, mostly PyTorch.

## Run

```bash
python run.py --video data/videos/clip1.mp4 --refs refs/ --out outputs/clip1
python run.py --video data/videos/clip2.mp4                 # single person, "Person 1"
```

Options: `--no-video` skips the annotated video; `--config` uses another config file.

## Outputs (in `--out`, default `outputs/<video name>/`)

| File | Content |
|---|---|
| `results.csv` | one row per second per person: `time_sec, person, visible, face_visible, ear, head_pitch, motion, posture_flag, state, confidence, reason` (+ diagnostics) |
| `summary.json` | asleep intervals per person with times, durations, confidence, totals, calibration |
| `annotated.mp4` | boxes with name, state, confidence and EAR on every processed frame |
| `timeline.png` | per-person state bands with EAR and motion traces |
| `run.log` | the log, including warnings such as "face not seen for 60 s", "could not be identified", "low confidence" |
| `signals.pkl` | cached model outputs; delete it to force re-extraction |

## Evaluation

```bash
copy data\manifest.example.csv data\manifest.csv   # list clips, labels, refs, split
python evaluate.py
```

- Labels: `data/labels/<clip>.csv` with `start_sec,end_sec,person,state` (see `RECORDING_GUIDE.md`).
- Thresholds (`eyes.closed_ratio`, `motion.still_threshold`) are grid-searched on **tune** clips only;
  all systems are then scored on held-out **eval** clips. The split is printed and written to the report.
- Baselines: **naive** (EAR below threshold = asleep, no time logic), **face-only** (state machine without body
  signals), **full** system.
- Metrics: per-second accuracy, asleep precision / recall / F1, confusion matrix, IoU of asleep seconds,
  and mean IoU per labelled sleep interval.
- Writes `outputs/evaluation/results_table.md`, `metrics.json`, `tuned_params.yaml`, and timeline plots
  with labels shown above predictions.

### Results

9 self-recorded clips (25 min; 8 single-person, 1 with two people), labelled per second by hand.
Tune split: clips 1, 3a, 4a. Eval split: clips 2, 3b, 4b, 5, 6, 8. Tuned values: `closed_ratio` 0.5,
`still_threshold` 0.02.

**Eval split, asleep as the positive class (1161 person-seconds):**

| system | accuracy (all states) | asleep precision | asleep recall | asleep F1 | asleep IoU |
|---|---|---|---|---|---|
| naive (EAR below threshold = asleep) | 0.796 | 0.581 | 0.962 | 0.724 | 0.568 |
| face-only state machine | 0.768 | 0.820 | 0.955 | 0.882 | 0.789 |
| **full system** | 0.780 | **0.952** | 0.952 | **0.952** | **0.909** |

Per clip (full system): eyes-closed sleep F1 0.98 (3b) and 0.95 (dim light, 16 fps); two people 0.94
with both people named correctly; phone use and 5 s eye closures produce **no** false asleep seconds.

**How the numbers were reached (read before trusting them).** The first evaluation gave eval F1 0.61.
Looking at the eval failures found three bugs, which were fixed (each with a test):
the per-event IoU mixed clips together; the phone clip was calibrated while looking down (74 false asleep
seconds); and a 0.4 s track overlap at an ID switch left half of one person unnamed. Then four disputed label
spans in clips 4b and 8 were re-checked against the video and corrected by the person who acted them.
Because both steps used eval clips, the final eval numbers are optimistic; the tune-split F1 (0.96) did not
change. The first run is kept in `outputs/evaluation_v1_before_fixes/`.

**Main remaining error: too much `drowsy`.** 225 labelled-awake seconds are called drowsy (mostly the
head-nodding rule firing on normal head movement). This doesn't affect the asleep metrics but lowers
all-state accuracy to 0.78.

## Tests

```bash
python -m pytest -q
```

47 tests: EAR maths, head-pose sign, blink vs sleep, hysteresis, backfill, body path, unknown handling,
calibration, per-second aggregation, identity carry-over by track ID, clothing re-ID, metrics.

## Known limitations

1. Labels are acted ("fake") sleep; real sleep looks different (gradual slumping, deeper breathing).
2. Behavioural, not physiological, sleep: 10 s of still, closed eyes counts as asleep by definition.
3. Looking down at a phone and dozing off with the chin dropped look alike: measured head pitch was +15..+31° on
   the phone and about +18° when dozing. The 20° limit separates them on these clips, but only just.
4. Body-only sleep needs 45 s and is lower-confidence; short naps facing away are missed.
5. About 5–9 processed frames/s on a laptop CPU (1–2.2× video length). Real-time needs a GPU or a lower `process_fps`;
   the online version would also turn off `backfill` (onset reported 10 s late) and keep running identity votes.
6. Naming needs a clear face at least once per person; similar clothing can confuse the backup re-ID.
7. Tuned on 3 clips; evaluated on 6 from two people in one room, with acted sleep.
8. No eval clip has sleep while facing away: the body-only path (45 s, posture) is only tested on clip 4a (tune).
9. The head-nodding rule over-triggers `drowsy` (see Results).
