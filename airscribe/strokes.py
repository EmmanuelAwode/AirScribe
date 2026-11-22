"""Stroke controller: decides, frame by frame, when a stroke begins, grows or ends.

Inputs per frame: the (smoothed) fingertip position or None, whether the pen is
down (pinch or keyboard toggle), and whether the point is inside the drawing
area. Output: calls on the canvas. It never connects two points across a gap:
any frame where drawing is not allowed ends the stroke, and the next allowed
frame begins a brand-new one. The one exception is a momentary pinch release
(see StrokeConfig.rejoin_frames): that continues the same stroke.
"""
import math
from typing import Optional, Tuple

from .config import StrokeConfig

Point = Tuple[float, float]


class StrokeController:
    def __init__(self, canvas, cfg: StrokeConfig):
        self.canvas = canvas
        self.cfg = cfg
        self.active = False
        self.last: Optional[Point] = None
        self.end_reason = ""
        self.rejoined = False           # True on the frame a stroke was re-joined
        self._since_end = 10**9         # frames since the last pinch release
        self._end_point: Optional[Point] = None
        self._end_style = None
        self.diag = math.hypot(canvas.width, canvas.height)

    def in_area(self, p: Point) -> bool:
        m = self.cfg.edge_margin
        return m <= p[0] < self.canvas.width - m and m <= p[1] < self.canvas.height - m

    def update(self, point: Optional[Point], pen_down: bool, color, size: int, erase: bool,
               hold: bool = False) -> bool:
        """Advance one frame. Returns True while a stroke is being drawn.

        hold: keep the current stroke open but add no point this frame (used
        while a pinch release is being confirmed, so opening fingers leave no tail).
        """
        self.rejoined = False
        self._since_end += 1
        if point is None or not pen_down or not self.in_area(point):
            reason = "no_hand" if point is None else ("pen_up" if not pen_down else "out_of_area")
            self.end(reason)
            return False

        if hold:                # add nothing; never start a stroke on a held frame
            return self.active
        if self.active and self.last is not None:
            jump = math.hypot(point[0] - self.last[0], point[1] - self.last[1])
            if jump > self.cfg.max_jump_fraction * self.diag:
                # Tracking glitch (or hand swap): do not draw a line across the canvas.
                self.end("jump")
            elif jump < self.cfg.min_point_distance:
                return True

        if not self.active:
            style = (tuple(color), size, erase)
            close = (self._end_point is not None
                     and math.hypot(point[0] - self._end_point[0], point[1] - self._end_point[1]) <= self.cfg.rejoin_px)
            if (self._since_end <= self.cfg.rejoin_frames and close and style == self._end_style
                    and self.canvas.reopen_last()):
                self.rejoined = True        # momentary release: carry on the same stroke
            else:
                self.canvas.begin_stroke(color, size, erase)
            self.active = True
            self._end_point = None
        self.canvas.add_point(point)
        self.last = point
        return True

    def end(self, reason: str = "action") -> None:
        if self.active:
            self.canvas.end_stroke()
            self.end_reason = reason
            if reason == "pen_up" and self.canvas.strokes:
                s = self.canvas.strokes[-1]
                self._since_end = 0
                self._end_point = self.last
                self._end_style = (tuple(s.color), s.size, s.erase)
            else:
                self._end_point = None   # dropouts, edges, jumps, tool changes: never re-join
        self.active = False
        self.last = None
