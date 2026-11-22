"""Ink layer, kept separate from the webcam.

The drawing is a list of strokes (the source of truth, which makes undo-by-stroke
trivial) plus two cached images rendered from them:
  * color: HxWx3 BGR ink colour
  * alpha: HxW uint8 coverage (0 = no ink, 255 = solid ink)
New points are drawn incrementally; undo/clear re-render from the stroke list.
An eraser stroke is just a stroke that writes alpha = 0, so undoing an erase
brings the ink back.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class Stroke:
    color: Tuple[int, int, int]
    size: int
    erase: bool
    points: List[Tuple[int, int]] = field(default_factory=list)


class InkCanvas:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.strokes: List[Stroke] = []
        self.current: Stroke | None = None
        self.color = np.zeros((height, width, 3), np.uint8)
        self.alpha = np.zeros((height, width), np.uint8)

    # ---- stroke building -------------------------------------------------
    def begin_stroke(self, color, size: int, erase: bool) -> None:
        self.current = Stroke(tuple(int(c) for c in color), int(size), erase)
        self.strokes.append(self.current)

    def add_point(self, p: Point) -> None:
        if self.current is None:
            return
        q = (int(round(p[0])), int(round(p[1])))
        prev = self.current.points[-1] if self.current.points else None
        self.current.points.append(q)
        self._draw_segment(self.current, prev, q)

    def prepend_points(self, pts) -> None:
        """Add points to the START of the current stroke (used to backfill a
        late pen-down). Draws them joined to the stroke's first point."""
        if self.current is None or not pts:
            return
        pts = [(int(round(x)), int(round(y))) for x, y in pts]
        first = self.current.points[0] if self.current.points else None
        prev = None
        for q in pts + ([first] if first else []):
            self._draw_segment(self.current, prev, q)
            prev = q
        self.current.points[:0] = pts

    def end_stroke(self) -> None:
        self.current = None

    # ---- editing ---------------------------------------------------------
    def undo(self) -> bool:
        if self.current is not None:
            self.end_stroke()
        if not self.strokes:
            return False
        self.strokes.pop()
        self._rerender()
        return True

    def clear(self) -> None:
        self.current = None
        self.strokes.clear()
        self._rerender()

    @property
    def is_empty(self) -> bool:
        return not self.alpha.any()

    # ---- rendering -------------------------------------------------------
    def _draw_segment(self, s: Stroke, a, b) -> None:
        a = a or b
        if s.erase:
            cv2.line(self.alpha, a, b, 0, s.size, cv2.LINE_AA)
            cv2.circle(self.alpha, b, s.size // 2, 0, -1, cv2.LINE_AA)
        else:
            # Colour is painted slightly wider without anti-aliasing so the
            # anti-aliased alpha edge never reveals black/neighbour pixels.
            cv2.line(self.color, a, b, s.color, s.size + 2, cv2.LINE_8)
            cv2.circle(self.color, b, s.size // 2 + 1, s.color, -1, cv2.LINE_8)
            cv2.line(self.alpha, a, b, 255, s.size, cv2.LINE_AA)
            cv2.circle(self.alpha, b, s.size // 2, 255, -1, cv2.LINE_AA)

    def _rerender(self) -> None:
        self.color[:] = 0
        self.alpha[:] = 0
        for s in self.strokes:
            prev = None
            for q in s.points:
                self._draw_segment(s, prev, q)
                prev = q

    def composite(self, background: np.ndarray) -> np.ndarray:
        """Blend ink over a BGR background of the same size.

        Only the bounding box of the ink is blended, using uint8 OpenCV maths,
        which keeps this a few milliseconds per frame at 720p.
        """
        out = background.copy()
        x, y, w, h = cv2.boundingRect(self.alpha)
        if w == 0 or h == 0:
            return out
        roi = (slice(y, y + h), slice(x, x + w))
        a = cv2.merge([self.alpha[roi]] * 3)
        ink = cv2.multiply(self.color[roi], a, scale=1 / 255)
        bg = cv2.multiply(background[roi], cv2.bitwise_not(a), scale=1 / 255)
        out[roi] = cv2.add(ink, bg)
        return out

    def to_bgra(self) -> np.ndarray:
        """Ink only, transparent background. No webcam pixels ever reach here."""
        bgra = np.dstack([self.color, self.alpha])
        bgra[self.alpha == 0, :3] = 0
        return bgra

    def save_png(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not cv2.imwrite(path, self.to_bgra()):
            raise IOError(f"Could not write {path}")
        return path
