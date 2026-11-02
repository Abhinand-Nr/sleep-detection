"""Compare predictions with hand labels, tune thresholds, and compare against baselines.

    python evaluate.py --manifest data/manifest.csv

manifest.csv columns: video,labels,refs,split   (split = tune | eval; refs may be empty)
labels CSV columns : start_sec,end_sec,person,state

Systems compared (same signals, same calibration, different decision logic):
  1. naive     : a second is asleep if EAR is below threshold in most of its frames
  2. face_only : full state machine, but no body signals (no motion, no posture)
  3. full      : the complete system
Thresholds are tuned on the "tune" clips only, then all systems are scored on the "eval"
clips, so the reported numbers are not fitted to the data they are measured on.
Signals are cached per clip, so after the first run this takes seconds.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.outputs import plot_timeline
from src.pipeline import Extraction, analyze, extract
from src.sleep_state import ASLEEP, STATES, intervals, naive_states
from src.utils.config import load_config
from src.utils.logging import LOGGER_NAME, setup_logging

ROOT = Path(__file__).resolve().parent
SYSTEMS = ("naive", "face_only", "full")


# ---------------------------------------------------------------- labels & metrics

def load_labels(path: Path) -> pd.DataFrame:
    """Expand interval labels into one row per (person, second). Later rows win on overlap."""
    raw = pd.read_csv(path, skipinitialspace=True)
    raw.columns = [c.strip() for c in raw.columns]
    seconds: dict[tuple[str, int], str] = {}
    for r in raw.itertuples(index=False):
        state = str(r.state).strip().lower()
        if state not in STATES:
            raise ValueError(f"{path.name}: unknown state '{r.state}' (use one of {', '.join(STATES)})")
        for s in range(int(math.floor(r.start_sec)), int(math.ceil(r.end_sec))):
            seconds[(str(r.person).strip(), s)] = state
    return pd.DataFrame([(p, s, v) for (p, s), v in seconds.items()], columns=["person", "time_sec", "truth"])


def join(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Labelled seconds with predictions; a missing prediction counts as 'absent'."""
    merged = truth.merge(pred[["person", "time_sec", "state"]], on=["person", "time_sec"], how="left")
    return merged.fillna({"state": "absent"})


def asleep_metrics(df: pd.DataFrame, positive: str = ASLEEP) -> dict:
    """Per-second accuracy (all states) and precision/recall/F1/IoU for the asleep class."""
    if df.empty:
        return {"seconds": 0}
    y_true = df.truth == positive
    y_pred = df.state == positive
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    union = int((y_true | y_pred).sum())
    return {
        "seconds": int(len(df)),
        "accuracy": float(accuracy_score(df.truth, df.state)),
        "asleep_accuracy": float((y_true == y_pred).mean()),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "iou": float((y_true & y_pred).sum() / union) if union else float("nan"),
        "event_iou": event_iou(df, positive),
    }


def event_iou(df: pd.DataFrame, positive: str = ASLEEP) -> float:
    """Mean IoU over labelled asleep intervals (each vs. the predicted intervals it overlaps)."""
    scores = []
    keys = ["clip", "person"] if "clip" in df else ["person"]   # "Person 1" in two clips is two people
    for _, g in df.groupby(keys):
        g = g.sort_values("time_sec")
        preds = intervals(g.time_sec, g.state, positive)
        for s, e in intervals(g.time_sec, g.truth, positive):
            hits = [(ps, pe) for ps, pe in preds if ps < e and s < pe]
            inter = sum(max(0, min(e, pe) - max(s, ps)) for ps, pe in hits)
            union = (e - s) + sum(pe - ps for ps, pe in hits) - inter
            scores.append(inter / union if union else 0.0)
    return float(sum(scores) / len(scores)) if scores else float("nan")


# ---------------------------------------------------------------- systems

def with_params(cfg: dict, params: dict) -> dict:
    """Deep copy of cfg with dotted keys overridden, e.g. {'eyes.closed_ratio': 0.6}."""
    out = copy.deepcopy(cfg)
    for key, value in params.items():
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return out


def get_param(cfg: dict, key: str):
    node = cfg
    for part in key.split("."):
        node = node[part]
    return node


def predict(ext: Extraction, cfg: dict, system: str) -> pd.DataFrame:
    if system == "face_only":
        cfg = with_params(cfg, {"sleep.use_body": False})
    per_second = analyze(ext, cfg).per_second
    if system == "naive":
        per_second = per_second.assign(state=naive_states(per_second, cfg["aggregation"]["closed_frac"]).values)
    return per_second


def score(clips: list[dict], cfg: dict, system: str) -> tuple[dict, pd.DataFrame]:
    joined = pd.concat([join(predict(c["ext"], cfg, system), c["truth"]).assign(clip=c["name"]) for c in clips],
                       ignore_index=True)
    return asleep_metrics(joined, cfg["evaluation"]["positive_state"]), joined


def tune(clips: list[dict], cfg: dict, log) -> dict:
    grid = cfg["evaluation"]["tuning_grid"]
    keys = list(grid)

    def run(params: dict) -> tuple[float, float]:
        m, _ = score(clips, with_params(cfg, params), "full")
        key = (m.get("f1", 0.0), m.get("accuracy", 0.0))
        log.info("  tune %s -> F1 %.3f, acc %.3f", params, *key)
        return key

    # Start from the config defaults; another setting must be strictly better to replace them.
    best = {k: get_param(cfg, k) for k in keys}
    best_key = run(best)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        key = run(params)
        if key > best_key:
            best, best_key = params, key
    log.info("Best on tune split: %s (F1 %.3f)", best, best_key[0])
    return best


# ---------------------------------------------------------------- report

def fmt(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def markdown_report(split_rows: list[tuple[str, str]], tuned: dict, results: dict, per_clip: pd.DataFrame,
                    confusion: pd.DataFrame, eval_name: str) -> str:
    lines = ["# Evaluation results", "", "## Data split", "", "| clip | split |", "|---|---|"]
    lines += [f"| {c} | {s} |" for c, s in split_rows]
    lines += ["", "## Tuned parameters (chosen on the tune split only)", ""]
    lines += [f"- `{k}` = {v}" for k, v in tuned.items()] or ["- (no tuning: defaults from config.yaml)"]
    for split, res in results.items():
        lines += ["", f"## Systems on the **{split}** split", "",
                  "| system | seconds | accuracy (all states) | asleep precision | asleep recall | asleep F1 | asleep IoU | event IoU |",
                  "|---|---|---|---|---|---|---|---|"]
        for system, m in res.items():
            lines.append(f"| {system} | {m.get('seconds', 0)} | {fmt(m.get('accuracy'))} | {fmt(m.get('precision'))} | "
                         f"{fmt(m.get('recall'))} | {fmt(m.get('f1'))} | {fmt(m.get('iou'))} | {fmt(m.get('event_iou'))} |")
    lines += ["", "## Full system per clip (tuned parameters)", "",
              "| clip | split | seconds | accuracy | asleep precision | asleep recall | asleep F1 |", "|---|---|---|---|---|---|---|"]
    for r in per_clip.itertuples(index=False):
        lines.append(f"| {r.clip} | {r.split} | {r.seconds} | {fmt(r.accuracy)} | {fmt(r.precision)} | "
                     f"{fmt(r.recall)} | {fmt(r.f1)} |")
    lines += ["", f"## Confusion matrix, full system, {eval_name} split (rows = labels, columns = predicted)", "",
              "| label \\ predicted | " + " | ".join(confusion.columns) + " |",
              "|---|" + "---|" * len(confusion.columns)]
    for state, row in confusion.iterrows():
        lines.append(f"| {state} | " + " | ".join(str(int(v)) for v in row) + " |")
    lines += ["", "Metrics are per second. 'asleep' is the positive class; IoU = overlap of predicted and "
              "labelled asleep seconds; event IoU = mean IoU per labelled asleep interval."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the sleep detector against hand labels.")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "evaluation")
    ap.add_argument("--no-tune", action="store_true", help="use config.yaml values as they are")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(args.out, cfg["logging"]["level"])
    if not args.manifest.exists():
        log.error("Manifest %s not found. Copy data/manifest.example.csv to data/manifest.csv and edit it.",
                  args.manifest)
        return 2
    manifest = pd.read_csv(args.manifest, skipinitialspace=True, dtype=str).fillna("")

    clips = []
    for r in manifest.itertuples(index=False):
        video, labels = ROOT / r.video, ROOT / r.labels
        if not video.exists() or not labels.exists():
            log.error("Skipping %s: video or labels file missing", r.video)
            continue
        refs = ROOT / r.refs if r.refs else None
        log.info("Clip %s (%s split)", video.name, r.split)
        ext = extract(video, refs, cfg, ROOT, cache_dir=ROOT / "outputs" / video.stem)
        clips.append({"name": video.stem, "split": r.split.strip().lower(), "ext": ext, "truth": load_labels(labels)})
    if not clips:
        log.error("No usable clips in the manifest")
        return 2

    # Identity/calibration messages were shown once during extraction; the many re-analyses
    # below would repeat them hundreds of times.
    for name in ("pipeline", "aggregation"):
        logging.getLogger(f"{LOGGER_NAME}.{name}").setLevel(logging.WARNING)

    tune_clips = [c for c in clips if c["split"] == "tune"]
    eval_clips = [c for c in clips if c["split"] == "eval"]
    log.info("Tune clips: %s", ", ".join(c["name"] for c in tune_clips) or "none")
    log.info("Eval clips: %s", ", ".join(c["name"] for c in eval_clips) or "none")
    if not eval_clips:
        log.warning("No eval clips: results below are on the tune clips and therefore optimistic")
    eval_set, eval_name = (eval_clips, "eval") if eval_clips else (clips, "all (no held-out clips)")

    tuned = {}
    if tune_clips and not args.no_tune:
        log.info("Tuning on %d clip(s) ...", len(tune_clips))
        tuned = tune(tune_clips, cfg, log)
    cfg_t = with_params(cfg, tuned)

    results: dict[str, dict] = {}
    for split, subset in (("tune", tune_clips), (eval_name, eval_set)):
        if subset:
            results[split] = {s: score(subset, cfg_t, s)[0] for s in SYSTEMS}

    per_clip_rows, timelines = [], args.out / "timelines"
    timelines.mkdir(parents=True, exist_ok=True)
    for c in clips:
        m, joined = score([c], cfg_t, "full")
        per_clip_rows.append({"clip": c["name"], "split": c["split"], **m})
        pred = predict(c["ext"], cfg_t, "full")
        plot_timeline(pred, timelines / f"{c['name']}.png", f"{c['name']}: labels vs full system",
                      truth=c["truth"].rename(columns={"truth": "state"}))
    per_clip = pd.DataFrame(per_clip_rows).reindex(
        columns=["clip", "split", "seconds", "accuracy", "precision", "recall", "f1"])

    _, joined = score(eval_set, cfg_t, "full")
    present = [s for s in STATES if s in set(joined.truth) | set(joined.state)]
    confusion = pd.DataFrame(confusion_matrix(joined.truth, joined.state, labels=present), index=present, columns=present)

    report = markdown_report([(c["name"], c["split"]) for c in clips], tuned, results, per_clip, confusion, eval_name)
    (args.out / "results_table.md").write_text(report, encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps({"tuned": tuned, "results": results}, indent=2), encoding="utf-8")
    (args.out / "tuned_params.yaml").write_text(yaml.safe_dump(tuned), encoding="utf-8")
    print("\n" + report)
    log.info("Wrote %s (paste into the report), timelines in %s", args.out / "results_table.md", timelines)
    if tuned:
        log.info("To adopt the tuned values, copy them from tuned_params.yaml into config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
