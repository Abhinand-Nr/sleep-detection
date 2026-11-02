"""Identity: names stick to a track, no double-booking, clothing backup."""
import numpy as np

from src.identity import TrackEvidence, resolve_identities


def track(tid: int, start: float, end: float, sims: dict | None = None, hist: np.ndarray | None = None) -> TrackEvidence:
    t = TrackEvidence(tid)
    for time in np.arange(start, end + 1e-9, 0.1):
        t.add_frame(float(time))
    for name, values in (sims or {}).items():
        for v in values:
            t.add_similarities({name: v})
    if hist is not None:
        t.add_histogram(hist.astype(np.float32))
    return t


def hist(peak: int) -> np.ndarray:
    h = np.full((16, 8), 0.001, dtype=np.float32)
    h[peak, 3] = 1.0
    return h / h.sum()


def test_name_from_a_few_face_frames_covers_the_whole_track(cfg) -> None:
    # Face seen only 5 times at the start of a 60 s track; the rest faces away.
    tracks = {1: track(1, 0, 60, {"Alice": [0.6] * 5, "Bob": [0.1] * 5})}
    ids = resolve_identities(tracks, ["Alice", "Bob"], cfg["matching"])
    assert ids[1].name == "Alice" and ids[1].source == "face"


def test_two_people_on_screen_get_different_names(cfg) -> None:
    tracks = {
        1: track(1, 0, 60, {"Alice": [0.7] * 10, "Bob": [0.40] * 10}),
        2: track(2, 0, 60, {"Alice": [0.45] * 10, "Bob": [0.6] * 10}),
    }
    ids = resolve_identities(tracks, ["Alice", "Bob"], cfg["matching"])
    assert ids[1].name == "Alice" and ids[2].name == "Bob"


def test_name_not_given_twice_at_the_same_time(cfg) -> None:
    tracks = {
        1: track(1, 0, 60, {"Alice": [0.8] * 10}),
        2: track(2, 10, 50, {"Alice": [0.5] * 10}),   # overlaps track 1, weaker
    }
    ids = resolve_identities(tracks, ["Alice", "Bob"], cfg["matching"])
    assert ids[1].name == "Alice"
    assert ids[2].name != "Alice"


def test_too_few_votes_is_not_enough(cfg) -> None:
    n = cfg["matching"]["min_face_votes"]
    tracks = {1: track(1, 0, 60, {"Alice": [0.9] * (n - 1)})}
    ids = resolve_identities(tracks, ["Alice"], cfg["matching"])
    assert ids[1].source == "none"


def test_clothing_backup_after_track_id_switch(cfg) -> None:
    # Alice's track is lost at 30 s; a new track (no face) appears at 35 s wearing the same colours.
    tracks = {
        1: track(1, 0, 30, {"Alice": [0.7] * 10}, hist(4)),
        2: track(2, 0, 90, {"Bob": [0.7] * 10}, hist(12)),
        3: track(3, 35, 90, None, hist(4)),
    }
    ids = resolve_identities(tracks, ["Alice", "Bob"], cfg["matching"])
    assert ids[3].name == "Alice" and ids[3].source == "clothing"


def test_short_glitch_tracks_are_ignored(cfg) -> None:
    tracks = {9: track(9, 0, 0.5, {"Alice": [0.9] * 5})}
    assert resolve_identities(tracks, ["Alice"], cfg["matching"]) == {}


def test_track_handover_with_a_brief_overlap_keeps_the_name(cfg) -> None:
    tracks = {
        1: track(1, 0, 127.2, {"Alice": [0.8] * 10}),
        2: track(2, 126.9, 212, {"Alice": [0.75] * 10}),   # new ID, 0.3 s overlap with the old one
    }
    ids = resolve_identities(tracks, ["Alice", "Bob"], cfg["matching"])
    assert ids[1].name == "Alice" and ids[2].name == "Alice"
