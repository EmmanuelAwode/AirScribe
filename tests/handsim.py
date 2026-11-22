"""A simulated hand and webcam for testing AirScribe without a camera.

The hand is 21 MediaPipe-style landmarks with adult proportions (metres). A
`pinch` value blends between a writing "hover" pose (index pointing, thumb
tucked) and a thumb-index pinch. The hand is rotated (yaw / pitch / roll),
placed in front of a pinhole camera matching a 1280x720 webcam, and projected.

It returns what MediaPipe returns: 2D pixel landmarks and 3D "world" landmarks
(hand-centred, metres). Three error sources seen in real logs are modelled:

* 2D noise: pixel jitter on every landmark.
* 3D depth noise: world-landmark depth (z) is far noisier than x/y.
* Thumb occlusion: when the thumb tip is hidden behind the index finger from
  the camera's view, its estimate is pulled off by a sizeable random error in
  both 2D and 3D (this is what inflates a real pinch in front of the body).
"""
from dataclasses import dataclass

import numpy as np

# Joint positions in the hand frame: x = towards thumb, y = towards fingertips,
# z = out of the palm. Wrist at the origin.
_FIXED = {
    0: (0.000, 0.000, 0.000),
    1: (0.022, 0.020, 0.012),   # thumb CMC
    5: (0.025, 0.085, 0.000),   # index MCP
    9: (0.005, 0.090, 0.000),   # middle MCP
    13: (-0.015, 0.085, 0.000),
    17: (-0.033, 0.075, 0.000),
    # curled middle / ring / pinky (writing grip)
    10: (0.005, 0.112, 0.033), 11: (0.005, 0.094, 0.050), 12: (0.005, 0.074, 0.046),
    14: (-0.015, 0.105, 0.031), 15: (-0.015, 0.088, 0.046), 16: (-0.015, 0.070, 0.042),
    18: (-0.033, 0.091, 0.026), 19: (-0.033, 0.078, 0.038), 20: (-0.033, 0.064, 0.035),
}
_HOVER = {  # index pointing, thumb resting beside the middle finger
    2: (0.040, 0.048, 0.026), 3: (0.042, 0.075, 0.040), 4: (0.036, 0.100, 0.048),
    6: (0.027, 0.125, 0.004), 7: (0.028, 0.150, 0.008), 8: (0.029, 0.170, 0.012),
}
_PINCH = {  # index slightly curled, thumb tip meeting the index tip
    2: (0.042, 0.060, 0.032), 3: (0.045, 0.095, 0.046), 4: (0.034, 0.143, 0.050),
    6: (0.027, 0.122, 0.012), 7: (0.026, 0.140, 0.028), 8: (0.024, 0.152, 0.040),
}

W, H, FOCAL = 1280, 720, 900.0


def hand_points(pinch: float, spread: float = 0.35) -> np.ndarray:
    """21x3 hand-frame landmarks. pinch 0 = hover, 1 = pinched.
    spread scales how far the thumb sits from the index when hovering."""
    pts = np.zeros((21, 3))
    for i, p in _FIXED.items():
        pts[i] = p
    for i in _HOVER:
        hover = np.array(_HOVER[i])
        if i in (2, 3, 4):  # wider or narrower hover thumb
            hover = np.array(_PINCH[i]) + (hover - np.array(_PINCH[i])) * spread
        pts[i] = hover + (np.array(_PINCH[i]) - hover) * pinch
    return pts


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Degrees. Base pose: palm facing the camera, fingers up the image."""
    y, p, r = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    base = np.diag([1.0, -1.0, -1.0])  # hand +y -> image up, palm normal -> camera
    return rz @ rx @ ry @ base


@dataclass
class Noise:
    px: float = 1.5            # 2D landmark jitter (pixels)
    world_xy: float = 0.002    # 3D jitter in x/y (metres)
    world_z: float = 0.009     # 3D jitter in depth (metres)
    occlusion_px: float = 14.0      # extra 2D error on a hidden thumb tip
    occlusion_world: float = 0.015  # extra 3D error on a hidden thumb tip


@dataclass
class Frame:
    px: list            # 21 (x, y) pixels, as AirScribe receives them
    world: list         # 21 (x, y, z) metres
    thumb_hidden: bool


def observe(pinch, yaw=0.0, pitch=0.0, roll=0.0, depth=0.6, spread=0.35,
            noise: Noise = None, rng=None, tip_at=None) -> Frame:
    """Render one frame. tip_at=(x, y) moves the hand so the index tip lands
    there on screen (the hand is translated, not re-posed)."""
    noise = noise or Noise()
    rng = rng if rng is not None else np.random.default_rng()
    local = hand_points(pinch, spread)
    cam = local @ rotation(yaw, pitch, roll).T
    cam = cam - cam[9] + np.array([0.0, 0.0, depth])  # hand centre at given depth
    u = FOCAL * cam[:, 0] / cam[:, 2] + W / 2
    v = FOCAL * cam[:, 1] / cam[:, 2] + H / 2
    px = np.stack([u, v], 1)

    # Thumb tip hidden if it projects close to the index finger and lies behind it.
    seg = px[[6, 7, 8]]
    d2 = np.min(np.linalg.norm(seg - px[4], axis=1))
    hand_px = np.linalg.norm(px[0] - px[9])
    hidden = d2 < 0.25 * hand_px and cam[4, 2] > cam[8, 2] + 0.004

    px = px + rng.normal(0, noise.px, px.shape)
    world = cam - cam[0]
    world = world + rng.normal(0, 1, world.shape) * [noise.world_xy, noise.world_xy, noise.world_z]
    if hidden:
        px[4] += rng.normal(0, noise.occlusion_px, 2)
        world[4] += rng.normal(0, noise.occlusion_world, 3)
    if tip_at is not None:
        px = px - px[8] + np.asarray(tip_at, float)
    return Frame([tuple(p) for p in px], [tuple(p) for p in world], bool(hidden))


# Hand orientations seen while writing (degrees).
POSES = {
    "beside body":    dict(yaw=(-20, 20), pitch=(-15, 20), roll=(-20, 20)),
    # drawing around your own head/torso: hand turned, back of hand to camera
    "in front of body": dict(yaw=(130, 170), pitch=(10, 35), roll=(-20, 20)),
    "tilted towards camera": dict(yaw=(-25, 25), pitch=(40, 65), roll=(-20, 20)),
    "edge-on": dict(yaw=(70, 85), pitch=(-10, 20), roll=(-20, 20)),
}


def sample_pose(name, rng):
    p = POSES[name]
    return {k: rng.uniform(*p[k]) for k in ("yaw", "pitch", "roll")}
