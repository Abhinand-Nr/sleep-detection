"""Label parsing and metrics."""
from pathlib import Path

import pandas as pd
import pytest

from evaluate import asleep_metrics, event_iou, join, load_labels, with_params


def test_labels_expand_to_seconds(tmp_path: Path) -> None:
    p = tmp_path / "l.csv"
    p.write_text("start_sec,end_sec,person,state\n0,3,Person 1,awake\n3,5,Person 1,asleep\n")
    df = load_labels(p)
    assert list(df.sort_values("time_sec").truth) == ["awake"] * 3 + ["asleep"] * 2


def test_bad_state_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "l.csv"
    p.write_text("start_sec,end_sec,person,state\n0,3,Person 1,sleeping\n")
    with pytest.raises(ValueError):
        load_labels(p)


def test_metrics_and_iou() -> None:
    truth = pd.DataFrame({"person": "A", "time_sec": range(10), "truth": ["awake"] * 4 + ["asleep"] * 6})
    pred = pd.DataFrame({"person": "A", "time_sec": range(10), "state": ["awake"] * 6 + ["asleep"] * 4})
    m = asleep_metrics(join(pred, truth))
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(4 / 6)
    assert m["iou"] == pytest.approx(4 / 6)
    assert event_iou(join(pred, truth)) == pytest.approx(4 / 6)


def test_event_iou_keeps_clips_apart() -> None:
    one = pd.DataFrame({"person": "Person 1", "time_sec": range(4),
                        "truth": ["awake", "asleep", "asleep", "awake"],
                        "state": ["awake", "asleep", "asleep", "awake"]})
    both = pd.concat([one.assign(clip="a"), one.assign(clip="b")])
    assert event_iou(both) == pytest.approx(1.0)


def test_missing_prediction_counts_as_absent() -> None:
    truth = pd.DataFrame({"person": "A", "time_sec": [0], "truth": ["asleep"]})
    pred = pd.DataFrame({"person": [], "time_sec": [], "state": []})
    assert join(pred, truth).state.iloc[0] == "absent"


def test_with_params_does_not_modify_original(cfg) -> None:
    new = with_params(cfg, {"eyes.closed_ratio": 0.99})
    assert new["eyes"]["closed_ratio"] == 0.99
    assert cfg["eyes"]["closed_ratio"] != 0.99
