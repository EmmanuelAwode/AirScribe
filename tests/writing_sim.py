"""End-to-end writing simulation: a simulated hand writes "ROSH" through the
real AirScribe pipeline (AirScribe.process_hand -> gestures -> strokes -> canvas)
and the resulting ink is scored against the intended letters.

Scenarios vary the hand pose, noise, writing speed and tracking dropouts.
Used by tests/test_writing_sim.py and scripts/sim_report.py.
"""
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from airscribe.app import AirScribe
from airscribe.config import Config
from tests.handsim import POSES, W, H, Noise, observe


def _arc(cx, cy, rx, ry, a0, a1, n=40):
    a = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + rx * np.cos(a), cy + ry * np.sin(a)))


def _line(p, q, n=20):
    return list(zip(np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)))


def rosh_strokes():
    """Intended strokes for R, O, S, H (pixel polylines). 6 strokes in total."""
    top, bot = 230.0, 470.0
    mid = (top + bot) / 2
    r = (_line((200, bot), (200, top)) + _arc(200, (top + mid) / 2, 75, (mid - top) / 2, -90, 90)[1:]
         + _line((200, mid), (285, bot))[1:])
    o = _arc(470, mid, 85, 120, -90, 270, 70)
    s = (_arc(700, (top + mid) / 2, 70, (mid - top) / 2, -20, -270, 40)
         + _arc(700, (mid + bot) / 2, 70, (bot - mid) / 2, -90, 160, 40)[1:])
    h = [_line((900, top), (900, bot)), _line((1050, top), (1050, bot)), _line((900, mid), (1050, mid))]
    return [r, o, s] + h


@dataclass
class Scenario:
    name: str
    pose: str = "beside body"
    noise: Noise = field(default_factory=Noise)
    speed: float = 450.0       # px/s while writing
    fps: float = 40.0
    dropout: float = 0.0       # per-frame chance tracking is lost for 1-3 frames
    pinch_wobble: float = 0.12 # how much the pinch loosens while writing (0..1 of pinch)


SCENARIOS = [
    Scenario("beside body"),
    Scenario("in front of body", pose="in front of body"),
    Scenario("tilted towards camera", pose="tilted towards camera"),
    Scenario("edge-on hand", pose="edge-on"),
    Scenario("fast writing", speed=900.0),
    Scenario("shaky + noisy", noise=Noise(px=3.0, world_xy=0.003, world_z=0.014), pinch_wobble=0.25),
    Scenario("dim light dropouts", dropout=0.02),
]


def _pose_walk(pose, rng, n):
    """Slowly drifting hand orientation inside the pose's range."""
    lo = {k: POSES[pose][k][0] for k in ("yaw", "pitch", "roll")}
    hi = {k: POSES[pose][k][1] for k in ("yaw", "pitch", "roll")}
    cur = {k: rng.uniform(lo[k], hi[k]) for k in lo}
    out = []
    for _ in range(n):
        for k in cur:
            cur[k] = float(np.clip(cur[k] + rng.normal(0, 0.8), lo[k], hi[k]))
        out.append(dict(cur))
    return out


def make_session(sc: Scenario, rng):
    """Per-frame plan: list of (tip_xy, pinch amount, visible)."""
    frames = []
    dt = 1 / sc.fps
    pos = np.array([120.0, 600.0])

    def hold(p, pinch, secs):
        for _ in range(max(1, int(secs / dt))):
            frames.append((p.copy(), pinch))

    def travel(path, pinch_fn):
        path = np.asarray(path)
        seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        total = seglen[-1]
        n = max(2, int(total / sc.speed / dt))
        for i in range(n + 1):
            s = total * i / n
            k = min(np.searchsorted(seglen, s, side="right") - 1, len(path) - 2)
            f = (s - seglen[k]) / max(seglen[k + 1] - seglen[k], 1e-9)
            frames.append((path[k] + f * (path[k + 1] - path[k]), pinch_fn(i / n)))

    hold(pos, 0.0, 0.6)
    wob = 0.0
    for stroke in rosh_strokes():
        start = np.asarray(stroke[0])
        travel([pos, start], lambda _: 0.0)                      # move with open hand
        for k in range(int(0.12 / dt)):                          # pinch closes
            frames.append((start.copy(), (k + 1) / int(0.12 / dt)))

        def wobble(_):
            nonlocal wob                                         # pinch loosens a little
            wob = float(np.clip(wob + rng.normal(0, 0.05), 0, sc.pinch_wobble))
            return 1.0 - wob
        travel(stroke, wobble)
        end = np.asarray(stroke[-1])
        for k in range(int(0.12 / dt)):                          # pinch opens
            frames.append((end.copy(), 1 - (k + 1) / int(0.12 / dt)))
        hold(end, 0.0, 0.15)
        pos = end
    hold(pos, 0.0, 0.4)

    # tracking dropouts
    visible = np.ones(len(frames), bool)
    i = 0
    while i < len(frames):
        if rng.random() < sc.dropout:
            visible[i:i + rng.integers(1, 4)] = False
            i += 4
        i += 1
    return frames, visible


def run(sc: Scenario, cfg: Config = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    cfg = cfg or Config()
    app = AirScribe(W, H, cfg)
    plan, visible = make_session(sc, rng)
    poses = _pose_walk(sc.pose, rng, len(plan))
    depth = rng.uniform(0.5, 0.75)
    spread = rng.uniform(0.25, 0.45)
    for i, ((tip, pinch), vis, pose) in enumerate(zip(plan, visible, poses)):
        t = i / sc.fps
        if not vis:
            app.process_hand(None, t)
            continue
        fr = observe(pinch, **pose, depth=depth, spread=spread, noise=sc.noise, rng=rng, tip_at=tip)
        app.process_hand(fr.px, t, fr.world)
    app.strokes.end()
    return score(app, sc)


def score(app, sc):
    intended = rosh_strokes()
    target = np.zeros((H, W), np.uint8)
    for s in intended:
        cv2.polylines(target, [np.int32(s)], False, 255, 1)
    dist_to_target = cv2.distanceTransform(255 - target, cv2.DIST_L2, 3)
    ink = app.canvas.alpha > 128
    ink_dist = cv2.distanceTransform((~ink).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    samples = np.int32([p for s in intended for p in s])
    covered = ink_dist[samples[:, 1], samples[:, 0]] <= 6
    stray = (ink & (dist_to_target > 14)).sum() / max(ink.sum(), 1)
    tiny = sum(len(s.points) < 4 for s in app.canvas.strokes)
    return dict(scenario=sc.name, strokes=len(app.canvas.strokes), intended=len(intended),
                coverage=float(covered.mean()), stray=float(stray), tiny=tiny,
                image=render(app, intended))


def render(app, intended):
    img = np.full((H, W, 3), 250, np.uint8)
    for s in intended:
        cv2.polylines(img, [np.int32(s)], False, (215, 215, 215), 14, cv2.LINE_AA)
    colors = [(60, 60, 220), (40, 150, 250), (60, 170, 60), (220, 130, 40), (180, 70, 160), (30, 30, 30)]
    for i, s in enumerate(app.canvas.strokes):
        if len(s.points) > 1:
            cv2.polylines(img, [np.int32(s.points)], False, colors[i % len(colors)], 4, cv2.LINE_AA)
        elif s.points:
            cv2.circle(img, s.points[0], 4, colors[i % len(colors)], -1)
    return img


def old_config() -> Config:
    """The settings before this round of fixes (3D-only pinch)."""
    cfg = Config()
    cfg.gesture = replace(cfg.gesture, pinch_measure="3d", pinch_start=0.30, pinch_release=0.42,
                          resume_frames=0, smooth_frames=1, backfill_frames=0, adapt=False)
    return cfg


def _lerp_pose(a, b, f):
    pa = {k: sum(POSES[a][k]) / 2 for k in ("yaw", "pitch", "roll")}
    pb = {k: sum(POSES[b][k]) / 2 for k in ("yaw", "pitch", "roll")}
    return {k: pa[k] + (pb[k] - pa[k]) * f for k in pa}


def run_trace_around_yourself(cfg: Config = None, seed: int = 0, fps: float = 40.0, laps: float = 1.0):
    """One held pinch tracing a big loop around your head and shoulders. The
    hand turns from 'beside body' (sides of the loop) to 'in front of body'
    (top, over your head) and back. Ideal result: ONE stroke covering the loop."""
    rng = np.random.default_rng(seed)
    cfg = cfg or Config()
    app = AirScribe(W, H, cfg)
    cx, cy, rx, ry = 640, 380, 330, 280
    n = int(5.0 * laps * fps)
    path = [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in np.linspace(np.pi * 0.9, np.pi * (0.9 + 2 * laps), n)]
    depth, spread = rng.uniform(0.5, 0.75), rng.uniform(0.25, 0.45)
    frames = [(path[0], 0.0)] * int(0.5 * fps)
    frames += [(path[0], (k + 1) / 5) for k in range(5)]
    wob = 0.0
    for p in path:
        wob = float(np.clip(wob + rng.normal(0, 0.05), 0, 0.15))
        frames.append((p, 1.0 - wob))
    for i, (tip, pinch) in enumerate(frames):
        # how far "over your head" the hand is: 0 at the sides, 1 at the top
        over = float(np.clip((cy - tip[1]) / ry, 0, 1)) ** 0.7
        pose = _lerp_pose("beside body", "in front of body", over)
        pose = {k: v + rng.normal(0, 4) for k, v in pose.items()}
        fr = observe(pinch, **pose, depth=depth, spread=spread, rng=rng, tip_at=tip)
        app.process_hand(fr.px, i / fps, fr.world)
    app.strokes.end()
    ink = app.canvas.alpha > 128
    ink_dist = cv2.distanceTransform((~ink).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    pts = np.int32(path)
    covered = ink_dist[np.clip(pts[:, 1], 0, H - 1), np.clip(pts[:, 0], 0, W - 1)] <= 6
    img = np.full((H, W, 3), 250, np.uint8)
    cv2.polylines(img, [pts], False, (215, 215, 215), 14, cv2.LINE_AA)
    for k, s in enumerate(app.canvas.strokes):
        if len(s.points) > 1:
            cv2.polylines(img, [np.int32(s.points)], False, [(60, 60, 220), (40, 150, 250), (60, 170, 60)][k % 3], 4, cv2.LINE_AA)
    return dict(strokes=len(app.canvas.strokes), coverage=float(covered.mean()), image=img)


def run_hover_only(cfg: Config = None, seed: int = 0, pose: str = "beside body", fps: float = 40.0,
                   secs: float = 20.0, max_pinch: float = 0.5):
    """Moving around without meaning to draw. The thumb drifts between fully
    open and `max_pinch` of the way to a pinch. Ideal result: no ink at all."""
    rng = np.random.default_rng(seed)
    cfg = cfg or Config()
    app = AirScribe(W, H, cfg)
    poses = _pose_walk(pose, rng, int(secs * fps))
    depth, spread = rng.uniform(0.5, 0.75), rng.uniform(0.25, 0.45)
    pos, vel, pinch = np.array([640.0, 360.0]), np.zeros(2), 0.0
    for i, pz in enumerate(poses):
        vel = 0.9 * vel + rng.normal(0, 6, 2)
        pos = np.clip(pos + vel, [80, 80], [W - 80, H - 80])
        pinch = float(np.clip(pinch + rng.normal(0, 0.06), 0, max_pinch))
        fr = observe(pinch, **pz, depth=depth, spread=spread, rng=rng, tip_at=pos)
        app.process_hand(fr.px, i / fps, fr.world)
    return dict(strokes=len(app.canvas.strokes), ink_px=int((app.canvas.alpha > 0).sum()))
