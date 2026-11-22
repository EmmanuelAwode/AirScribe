"""Camera-free end-to-end tests: a simulated hand (tests/handsim.py) writes
"ROSH" and traces around the user's own head through the real pipeline, and a
real recorded session is replayed. See scripts/sim_report.py for pictures."""
import csv
import os

import numpy as np
import pytest

from airscribe.config import Config
from tests.handsim import POSES, observe
from tests.writing_sim import (SCENARIOS, old_config, run, run_hover_only,
                               run_trace_around_yourself)

SEEDS = range(6)
CORE = [s for s in SCENARIOS if not s.dropout]


@pytest.mark.parametrize("sc", CORE, ids=[s.name for s in CORE])
def test_rosh_is_six_clean_strokes(sc):
    rs = [run(sc, seed=seed) for seed in SEEDS]
    for seed, r in zip(SEEDS, rs):
        assert r["strokes"] == 6, f"{sc.name} seed {seed}: {r['strokes']} strokes"
        assert r["stray"] < 0.005, f"{sc.name} seed {seed}: stray ink"
        assert r["tiny"] == 0
        # Worst case: a writer whose relaxed thumb rests near the index finger,
        # writing in front of their body with the thumb hidden, loses ~10%.
        assert r["coverage"] > 0.88, f"{sc.name} seed {seed}: {r['coverage']:.2f}"
    assert np.mean([r["coverage"] for r in rs]) > 0.95


def test_brief_dropouts_split_strokes_but_lose_little_ink():
    sc = next(s for s in SCENARIOS if s.dropout)
    rs = [run(sc, seed=s) for s in SEEDS]
    assert np.mean([r["coverage"] for r in rs]) > 0.93
    assert max(r["stray"] for r in rs) < 0.005   # never a line across a gap


def test_tracing_around_yourself_is_one_stroke():
    for seed in SEEDS:
        r = run_trace_around_yourself(seed=seed)
        assert r["strokes"] == 1 and r["coverage"] > 0.97


def test_old_3d_only_setting_fails_in_front_of_body():
    """Guards the simulator itself: it must reproduce the real bug report."""
    sc = next(s for s in SCENARIOS if s.pose == "in front of body")
    rs = [run(sc, old_config(), seed=s) for s in SEEDS]
    assert np.mean([r["strokes"] for r in rs]) > 7 or np.mean([r["coverage"] for r in rs]) < 0.85


@pytest.mark.parametrize("pose", list(POSES))
def test_hovering_with_half_closed_thumb_leaves_almost_no_ink(pose):
    # 6 writers x 20 s each, thumb drifting up to halfway to a pinch
    total = sum(run_hover_only(seed=s, pose=pose, max_pinch=0.5)["strokes"] for s in SEEDS)
    assert total <= 1


def test_adapting_to_a_close_resting_thumb_stops_joined_letters():
    """A writer whose relaxed thumb sits near the index finger: with fixed
    thresholds the pen never lifts between letters (tails / joined letters)."""
    from dataclasses import replace
    fixed = Config()
    fixed.gesture = replace(fixed.gesture, adapt=False)
    stray_fixed = max(run(sc, fixed, 101)["stray"] for sc in CORE)
    stray_adapt = max(run(sc, Config(), 101)["stray"] for sc in CORE)
    assert stray_fixed > 0.01 and stray_adapt < 0.005


def test_tilt_corrected_ratio_matches_3d_geometry():
    """Without noise, the tilt-corrected 2D ratio must not blow up when the
    hand tilts towards the camera, unlike the plain 2D ratio."""
    from tests.handsim import Noise
    from airscribe.gestures import pinch_ratio, pinch_ratio_tilt_corrected
    quiet = Noise(px=0, world_xy=0, world_z=0, occlusion_px=0, occlusion_world=0)
    flat = observe(1.0, noise=quiet)
    tilt = observe(1.0, pitch=60, noise=quiet)
    plain_err = pinch_ratio(tilt.px) / pinch_ratio(flat.px)
    fixed_err = pinch_ratio_tilt_corrected(tilt.px, tilt.world) / pinch_ratio_tilt_corrected(flat.px, flat.world)
    assert plain_err > 1.5                       # plain 2D balloons when tilted
    # Corrected stays much closer. Not exact: tilted fingertips sit nearer the
    # camera than the palm, so perspective still enlarges the gap a little.
    assert fixed_err < 1.4 and fixed_err < plain_err - 0.2


def test_real_session_replay_regression():
    """Replays real recorded pinch values (tests/data) through the detector."""
    from airscribe.gestures import PinchDetector, combine_ratios
    path = os.path.join(os.path.dirname(__file__), "data", "session1_ratios.csv")
    rows = list(csv.DictReader(open(path)))
    g = Config().gesture
    det = PinchDetector.from_config(g)
    pen = []
    for r in rows:
        ratio = None
        if r["hand"] == "1":
            num = lambda v: float(v) if v else None   # noqa: E731
            ratio = combine_ratios(num(r["ratio2d"]), num(r["ratio3d"]), g.pinch_measure)
        pen.append(det.update(ratio))
    starts = [i for i in range(1, len(pen)) if pen[i] and not pen[i - 1]]
    lengths = []
    for s in starts:
        e = s
        while e < len(pen) and pen[e]:
            e += 1
        lengths.append(e - s)
    assert len(starts) <= 24                       # old 3D-only setting: 31
    assert sum(n < 10 for n in lengths) <= 2       # old: 7 accidental bits
