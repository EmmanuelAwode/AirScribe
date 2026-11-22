"""Gesture detection: turns hand landmarks into a pen-down / pen-up signal.

Pure Python + math. No camera, OpenCV or MediaPipe needed, so it is unit tested.
"""
import math
from typing import Optional, Sequence, Tuple

Point = Sequence[float]  # (x, y) or (x, y, z)

# MediaPipe hand landmark indices (21 points per hand).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9


def _dist(a: Point, b: Point) -> float:
    return math.dist(a, b)  # works for 2D pixels or 3D world points


def pinch_ratio(landmarks: Sequence[Point]) -> Optional[float]:
    """Thumb-index gap divided by hand size, so it does not depend on how far
    the hand is from the camera. Returns None if hand size is degenerate."""
    hand_size = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if hand_size < 1e-6:
        return None
    return _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_size


# Palm segments used to measure how foreshortened the hand is on screen.
# The palm is close to rigid, so their true (3D) lengths do not change.
PALM_SEGMENTS = ((0, 5), (0, 9), (0, 17), (5, 9), (9, 17), (5, 17))


def pinch_ratio_tilt_corrected(px: Sequence[Point], world: Sequence[Point]) -> Optional[float]:
    """2D pinch ratio with the hand's tilt taken out.

    Plain 2D ratio = pixel gap / pixel hand size. When the hand tilts, the
    on-screen hand size shrinks (foreshortening) and the ratio balloons, so a
    real pinch can read 0.4. Here the pixel gap is converted to metres using
    the palm segment that is LEAST foreshortened (pixels per metre is highest
    for a segment facing the camera), then divided by the true 3D hand size.
    The thumb-index gap still comes from the 2D image, which MediaPipe locates
    more reliably than depth.
    """
    ppm = 0.0
    for a, b in PALM_SEGMENTS:
        metres = _dist(world[a], world[b])
        if metres > 1e-4:
            ppm = max(ppm, _dist(px[a], px[b]) / metres)
    hand_size = _dist(world[WRIST], world[MIDDLE_MCP])
    if ppm < 1e-6 or hand_size < 1e-6:
        return None
    return _dist(px[THUMB_TIP], px[INDEX_TIP]) / ppm / hand_size


def combine_ratios(r2d: Optional[float], r3d: Optional[float], measure: str = "blend",
                   weight_2d: float = 0.5) -> Optional[float]:
    """Pick or blend the 2D and 3D pinch ratios (see GestureConfig.pinch_measure).
    r2d should be the tilt-corrected 2D ratio whenever world landmarks exist."""
    if r2d is None or r3d is None:
        return r3d if r2d is None else r2d
    if measure == "2d":
        return r2d
    if measure == "3d":
        return r3d
    if measure == "blend":
        return weight_2d * r2d + (1 - weight_2d) * r3d
    raise ValueError(f"unknown pinch_measure {measure!r}")


class PinchDetector:
    """Hysteresis switch with frame confirmation.

    Released -> Pinched when ratio < start for `confirm_frames` frames in a row.
    Pinched  -> Released when ratio > release for `confirm_frames` frames in a row.
    Between the two thresholds the previous state is kept, so a ratio wobbling
    around a single boundary cannot flicker the pen on and off, and a one-frame
    spike (landmark noise) is ignored.

    `warmup_frames`: a pinch cannot start until the hand has been tracked this
    many frames, so a hand entering the frame (often mis-tracked for a moment)
    does not leave stray dots.

    `resume_frames`: if tracking drops out for at most this many frames while
    pinched, and the first frame back still reads as pinched (below release),
    the pen goes straight back down with no warm-up or confirmation. The app
    still starts a NEW stroke, so nothing is drawn across the gap.

    `smooth_frames`: the decision uses the average of the last N ratios.

    Adapting to the user's hand (`adapt=True`): people differ in where a
    relaxed thumb rests. With the thumb resting near the index finger an open
    hand can read ~0.43, under a fixed release level, so the pen never lifts
    between letters (letters get joined) and hovering dips under the start
    level (stray marks). The detector keeps `open_level`, a slow running
    average of the ratio while the pen is up and the fingers are not closing,
    (learning only from frames that are clearly open by the configured
    levels) and lowers the thresholds to sit below it:
        release_now = open_level - release_margin   (never above `release`)
        start_now   = open_level - start_margin     (never above `start`)
    with floors so a pinch is always possible.

    `pending_release` is True while a release is seen but not yet confirmed;
    the app holds the pen still during those frames so that, if the release is
    real, the fingers opening never adds a tail to the stroke.
    """

    def __init__(self, start: float, release: float, confirm_frames: int = 1, warmup_frames: int = 0,
                 resume_frames: int = 0, smooth_frames: int = 1, adapt: bool = False,
                 release_margin: float = 0.07, start_margin: float = 0.17,
                 min_start: float = 0.20, min_release: float = 0.30, adapt_rate: float = 0.03):
        if not start < release:
            raise ValueError("pinch start threshold must be below release threshold")
        self.start = start            # upper limits; the live values may be lower
        self.release = release
        self.adapt = adapt
        self.release_margin, self.start_margin = release_margin, start_margin
        self.min_start, self.min_release, self.adapt_rate = min_start, min_release, adapt_rate
        self.open_level = release + release_margin  # until learned: no change
        self.confirm_frames = max(1, confirm_frames)
        self.warmup_frames = max(0, warmup_frames)
        self.resume_frames = max(0, resume_frames)
        self.smooth_frames = max(1, smooth_frames)
        self.reset()

    def update(self, ratio: Optional[float]) -> bool:
        if ratio is None:          # hand lost -> pen up, start over
            if self.pinched:
                self._resume_left = self.resume_frames
            else:
                self._resume_left = max(0, self._resume_left - 1)
            self.pinched = False
            self._count = 0
            self.tracked_frames = 0
            self._recent = []
            self.value = self.raw = None
            return False
        # Average the last few ratios: landmark noise is mostly frame-to-frame,
        # so this steadies the value at the cost of a frame or two of delay.
        self.raw = ratio
        self._recent.append(ratio)
        del self._recent[:-self.smooth_frames]
        ratio = sum(self._recent) / len(self._recent)
        self.value = ratio
        start, release = self.start_now, self.release_now
        if self.tracked_frames == 0 and self._resume_left > 0 and ratio <= release:
            self.pinched = True    # brief dropout mid-stroke: resume at once
        self._resume_left = 0
        self.tracked_frames += 1
        if self.pinched:
            crossing = ratio > release
        else:
            crossing = ratio < start and self.tracked_frames > self.warmup_frames
            if (self.adapt and self.tracked_frames > self.warmup_frames
                    and ratio > (self.start + self.release) / 2):
                # Pen up and the hand clearly open by the DEFAULT levels: learn
                # this hand's open level. Gating on the fixed defaults (not the
                # live, adapted levels) matters: a pinch that failed to register
                # must never teach the detector that "open" is lower, or the
                # levels spiral down and pinches fail more and more.
                self.open_level += self.adapt_rate * (ratio - self.open_level)
        self._count = self._count + 1 if crossing else 0
        if self._count >= self.confirm_frames:
            self.pinched = not self.pinched
            self._count = 0
        return self.pinched

    @property
    def pending_release(self) -> bool:
        # Also true as soon as the raw (unsmoothed) ratio says the fingers are
        # opening, so the smoothing delay never lets an opening add a tail.
        return self.pinched and (self._count > 0 or (self.raw is not None and self.raw > self.release_now))

    @property
    def release_now(self) -> float:
        if not self.adapt:
            return self.release
        return max(self.min_release, min(self.release, self.open_level - self.release_margin))

    @property
    def start_now(self) -> float:
        if not self.adapt:
            return self.start
        s = max(self.min_start, min(self.start, self.open_level - self.start_margin))
        return min(s, self.release_now - 0.06)   # always keep some hysteresis

    @classmethod
    def from_config(cls, g) -> "PinchDetector":
        return cls(g.pinch_start, g.pinch_release, g.confirm_frames, g.warmup_frames,
                   g.resume_frames, g.smooth_frames, g.adapt, g.release_margin,
                   g.start_margin, g.min_start, g.min_release, g.adapt_rate)

    def reset(self) -> None:
        self.pinched = False
        self._count = 0
        self.tracked_frames = 0
        self._resume_left = 0
        self._recent = []
        self.value = None  # the (smoothed) ratio used for the last decision
        self.raw = None    # the latest unsmoothed ratio
