"""Replay a gesture log (python -m airscribe --log FILE.csv) through the pinch
detector with the current config, or with other settings, without a camera.

    python scripts/replay_log.py gestures.csv
    python scripts/replay_log.py gestures.csv --measure 3d --start 0.30 --release 0.42

Prints how many strokes the session would produce, how many are tiny (< 10
frames, usually accidental) and what share of hand-visible frames would ink.
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
        r2d = num(r.get("ratio2dt")) or num(r["ratio2d"])  # tilt-corrected when the log has it
        ratio = combine_ratios(r2d, num(r["ratio3d"]), measure) if r["hand"] == "1" else None
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


def main() -> int:
    g = Config().gesture
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--measure", default=g.pinch_measure, choices=["2d", "3d", "blend"])
    p.add_argument("--start", type=float, default=g.pinch_start)
    p.add_argument("--release", type=float, default=g.pinch_release)
    p.add_argument("--confirm", type=int, default=g.confirm_frames)
    p.add_argument("--warmup", type=int, default=g.warmup_frames)
    a = p.parse_args()
    with open(a.log, newline="") as f:
        rows = list(csv.DictReader(f))
    strokes, tiny, ink = replay(rows, a.measure, a.start, a.release, a.confirm, a.warmup)
    print(f"{a.log}: {len(rows)} frames | measure={a.measure} start={a.start} release={a.release}")
    print(f"  strokes {strokes}  (tiny < 10 frames: {tiny})  |  pen down in {ink:.1f}% of hand frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
