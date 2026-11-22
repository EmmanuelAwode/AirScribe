"""Replay a gesture log (python -m airscribe --log FILE.csv) through the pinch
detector with the current config, or with other settings, without a camera.

    python scripts/replay_log.py gestures.csv
    python scripts/replay_log.py gestures.csv --measure 3d --start 0.30 --release 0.42

Prints how many strokes the session would produce, how many are tiny (< 10
frames, usually accidental) and what share of hand-visible frames would ink.

    python scripts/replay_log.py gestures.csv --full replay.png

--full re-runs the WHOLE app (smoothing, strokes, re-joining, backfill) on the
recorded hand landmarks - needs a log made by the current version, which stores
them - and saves the resulting drawing as a picture.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from airscribe.config import Config  # noqa: E402
from airscribe.gestures import PinchDetector, combine_ratios  # noqa: E402


def num(v):
    return float(v) if v else None


def replay(rows, measure, start, release, confirm, warmup):
    from dataclasses import replace
    g = replace(Config().gesture, pinch_start=start, pinch_release=release,
                confirm_frames=confirm, warmup_frames=warmup)
    det = PinchDetector.from_config(g)
    pen = []
    for r in rows:
        r2d = num(r.get("ratio2dt")) or num(r.get("ratio2d"))  # tilt-corrected when the log has it
        g = Config().gesture
        ratio = combine_ratios(r2d, num(r["ratio3d"]), measure, g.blend_weight_2d) if r["hand"] == "1" else None
        pen.append(det.update(ratio))
    starts = [i for i in range(1, len(pen)) if pen[i] and not pen[i - 1]]
    lengths = []
    for s in starts:
        e = s
        while e < len(pen) and pen[e]:
            e += 1
        lengths.append(e - s)
    seen = sum(r["hand"] == "1" for r in rows) or 1
    return len(starts), sum(n < 10 for n in lengths), 100 * sum(pen) / seen


def full_replay(rows, cfg, out_png):
    import cv2
    import numpy as np
    from airscribe.app import AirScribe

    def pts(cell, n):
        v = [float(a) for a in cell.split()]
        return [tuple(v[i:i + n]) for i in range(0, len(v), n)]

    if not rows or not rows[0].get("landmarks_px") and not any(r.get("landmarks_px") for r in rows[:500]):
        raise SystemExit("This log has no hand landmarks; record a new one with the current version.")
    app = AirScribe(1280, 720, cfg)
    t0 = float(rows[0]["t"])
    for r in rows:
        t = float(r["t"]) - t0
        if r["hand"] == "1" and r.get("landmarks_px"):
            world = pts(r["landmarks_world"], 3) if r.get("landmarks_world") else None
            app.process_hand(pts(r["landmarks_px"], 2), t, world)
        else:
            app.process_hand(None, t)
    app.strokes.end()
    img = np.full((720, 1280, 3), 245, np.uint8)
    colors = [(60, 60, 230), (40, 150, 250), (60, 170, 60), (235, 140, 40), (200, 80, 160), (30, 30, 30)]
    for i, s in enumerate(app.canvas.strokes):
        if len(s.points) > 1:
            cv2.polylines(img, [np.int32(s.points)], False, colors[i % len(colors)], 4, cv2.LINE_AA)
        elif s.points:
            cv2.circle(img, s.points[0], 4, colors[i % len(colors)], -1)
    cv2.imwrite(out_png, img)
    return len(app.canvas.strokes)


def main() -> int:
    g = Config().gesture
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--measure", default=g.pinch_measure, choices=["2d", "3d", "blend"])
    p.add_argument("--start", type=float, default=g.pinch_start)
    p.add_argument("--release", type=float, default=g.pinch_release)
    p.add_argument("--confirm", type=int, default=g.confirm_frames)
    p.add_argument("--warmup", type=int, default=g.warmup_frames)
    p.add_argument("--full", metavar="PNG", help="replay the whole app on the recorded landmarks and save the drawing")
    a = p.parse_args()
    with open(a.log, newline="") as f:
        rows = list(csv.DictReader(f))
    strokes, tiny, ink = replay(rows, a.measure, a.start, a.release, a.confirm, a.warmup)
    print(f"{a.log}: {len(rows)} frames | measure={a.measure} start={a.start} release={a.release}")
    print(f"  strokes {strokes}  (tiny < 10 frames: {tiny})  |  pen down in {ink:.1f}% of hand frames")
    if a.full:
        from dataclasses import replace
        cfg = Config()
        cfg.gesture = replace(cfg.gesture, pinch_measure=a.measure, pinch_start=a.start,
                              pinch_release=a.release, confirm_frames=a.confirm, warmup_frames=a.warmup)
        n = full_replay(rows, cfg, a.full)
        print(f"  full replay: {n} strokes drawn -> {a.full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
