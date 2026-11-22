"""All tunable numbers in one place.

Edit these values (or pass a different Config to the app) to tune feel.
Nothing here imports OpenCV or MediaPipe, so it is safe to use in tests.
"""
from dataclasses import dataclass, field


@dataclass
class GestureConfig:
    # Pinch ratio = distance(thumb tip, index tip) / hand size,
    # where hand size = distance(wrist, middle-finger MCP knuckle).
    # Lower ratio = fingers closer together. Watch the live "p" value in the
    # status bar (or run with --log) to pick thresholds for your hand.
    pinch_start: float = 0.28    # ratio must drop BELOW this to start drawing
    pinch_release: float = 0.44  # ratio must rise ABOVE this to stop drawing
    # The gap between the two (hysteresis) stops flicker near the boundary.

    # Which landmarks the ratio is measured on:
    #   "2d"    on-screen pixels. Tilting the hand changes it.
    #   "3d"    MediaPipe world landmarks (metric). Steady to the side of the
    #           body, but in front of the body/face a real pinch reads ~0.31
    #           (vs ~0.25 to the side), so pinches there were missed.
    #   "blend" average of the two (default). Measured on a real logged session:
    #           it keeps the same pinched/open split in front of and beside the
    #           body, and its errors partly cancel out.
    # Falls back to whichever is available. Check settings against a recording
    # with:  python scripts/replay_log.py gestures.csv
    pinch_measure: str = "blend"
    # Share of the 2D (tilt-corrected) reading in the blend; the rest is 3D.
    blend_weight_2d: float = 0.5
    # A transition must hold this many consecutive frames (about 55 ms at 37 fps)
    # so one noisy frame can neither break a stroke nor start one.
    confirm_frames: int = 2
    # Frames a newly seen hand must be tracked before a pinch can start, so a
    # hand entering the frame does not leave stray dots.
    warmup_frames: int = 3
    # After a tracking dropout of at most this many frames (~150 ms) in the middle
    # of a stroke, a hand that is still pinched resumes drawing at once (as a new
    # stroke - nothing is drawn across the gap) instead of waiting out warm-up.
    resume_frames: int = 6
    # Average the pinch ratio over this many frames before thresholding.
    smooth_frames: int = 3
    # When the pen goes down late (a pinch that reads close to the threshold),
    # the new stroke reaches back over up to this many recent frames in which
    # the fingers were already clearly closed, so the start of
    # the letter is not lost. An open hand moving to the letter never qualifies.
    backfill_frames: int = 12

    # Adapt to the user's hand. pinch_start / pinch_release above become upper
    # limits; the live levels sit these margins below the learned open-hand
    # value (see PinchDetector). Floors keep a pinch always possible.
    adapt: bool = True
    release_margin: float = 0.05
    start_margin: float = 0.17
    min_start: float = 0.22
    min_release: float = 0.30
    adapt_rate: float = 0.03   # per frame (~1 s to settle at 40 fps)


@dataclass
class SmoothingConfig:
    # One Euro filter (Casiez et al. 2012). Coordinates are in pixels.
    # min_cutoff: lower = smoother when the finger is slow/still (more lag).
    # beta:       higher = less lag when the finger moves fast (keeps corners).
    min_cutoff: float = 1.5
    beta: float = 0.015
    d_cutoff: float = 1.0


@dataclass
class StrokeConfig:
    # A single-frame jump larger than this fraction of the frame diagonal is
    # treated as a tracking glitch: the stroke is ended, not connected.
    max_jump_fraction: float = 0.25
    # Ignore new points closer than this many pixels to the last point.
    min_point_distance: float = 1.5
    # Pixels inset from the frame edges; the fingertip outside this inner
    # rectangle counts as "left the drawing area" and ends the stroke.
    edge_margin: int = 12
    # A pinch that opens for only a moment and closes again where it left off
    # is a flicker, not a real release: if the pen goes back down within this
    # many frames (~120 ms at 57 fps) and this many pixels of where the stroke
    # ended, the SAME stroke continues. Only pinch releases are re-joined -
    # tracking dropouts, leaving the area and glitch jumps always start fresh.
    rejoin_frames: int = 7
    rejoin_px: float = 40.0


@dataclass
class TrackerConfig:
    num_hands: int = 1
    # Slightly permissive so a dim room or fast motion blur drops tracking less often.
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class Config:
    gesture: GestureConfig = field(default_factory=GestureConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    brush_sizes: tuple = (2, 4, 6, 9, 13, 18, 26)
    default_brush_index: int = 2
    eraser_scale: float = 3.0  # eraser width = brush size * this
    # BGR colours with display names.
    palette: tuple = (
        ("White", (245, 245, 245)),
        ("Black", (25, 25, 25)),
        ("Red", (60, 60, 230)),
        ("Orange", (40, 150, 250)),
        ("Yellow", (60, 220, 245)),
        ("Green", (90, 200, 70)),
        ("Blue", (235, 140, 40)),
        ("Purple", (200, 80, 160)),
    )
    paper_color: tuple = (238, 241, 244)  # plain-canvas background (BGR)
