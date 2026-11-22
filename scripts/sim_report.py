"""Camera-free report: a simulated hand writes "ROSH" (and traces around the
writer's head) through the real AirScribe pipeline, with the settings before
this round of fixes and with the current settings.

    python scripts/sim_report.py            # 10 simulated writers per scenario
    python scripts/sim_report.py --seeds 30

Writes sim_report/summary.png (old vs new, one row per scenario) and prints a
table: strokes (ideal 6), share of the letters inked, stray ink off the letters.
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from airscribe.config import Config  # noqa: E402
from tests.writing_sim import SCENARIOS, old_config, run, run_trace_around_yourself  # noqa: E402


def label(img, text):
    cv2.putText(img, text, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40), 2, cv2.LINE_AA)
    return img


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=10, help="simulated writers per scenario")
    p.add_argument("--out", default=os.path.join(ROOT, "sim_report"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    configs = {"before": old_config(), "now": Config()}
    rows = []
    print(f"{'scenario':24s} {'':7s} strokes  inked  worst-inked  stray")
    for sc in SCENARIOS + ["trace around yourself"]:
        name = sc if isinstance(sc, str) else sc.name
        pics = []
        for key, cfg in configs.items():
            if isinstance(sc, str):
                rs = [run_trace_around_yourself(cfg, s) for s in range(a.seeds)]
                for r in rs:
                    r["stray"] = 0.0
                ideal = 1
            else:
                rs = [run(sc, cfg, s) for s in range(a.seeds)]
                ideal = 6
            cov = [r["coverage"] for r in rs]
            print(f"{name:24s} {key:7s} {np.mean([r['strokes'] for r in rs]):5.1f}/{ideal} "
                  f"{np.mean(cov) * 100:5.1f}%  {min(cov) * 100:5.1f}%  {np.mean([r['stray'] for r in rs]) * 100:5.2f}%")
            worst = min(rs, key=lambda r: r["coverage"])
            pics.append(label(cv2.resize(worst["image"], (640, 360)), f"{name} - {key} (worst of {a.seeds})"))
        rows.append(np.hstack(pics))
    summary = np.vstack(rows)
    path = os.path.join(a.out, "summary.png")
    cv2.imwrite(path, summary)
    print(f"\nPictures (worst run of each, before | now): {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
