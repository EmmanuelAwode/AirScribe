"""AirScribe main loop: camera -> tracker -> gesture/smoothing -> strokes -> render.

The per-frame logic lives in `AirScribe.process_hand`, which takes landmarks
(not a camera frame), so it can be driven by tests without any hardware.
"""
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from .canvas import InkCanvas
from .config import Config
from .gestures import INDEX_TIP, PinchDetector, combine_ratios, pinch_ratio, pinch_ratio_tilt_corrected
from .smoothing import PointSmoother
from .strokes import StrokeController
from . import ui

WINDOW = "AirScribe"


class CameraError(RuntimeError):
    pass


@dataclass
class ToolState:
    tool: str = "pen"           # "pen" or "eraser"
    color_idx: int = 0
    brush_idx: int = 2
    show_camera: bool = True
    mirror: bool = True         # flip the camera like a mirror (natural for writing)
    show_landmarks: bool = False
    manual_ink: bool = False    # keyboard fallback: ink while hovering, no pinch
    brush_size: int = 6


class AirScribe:
    def __init__(self, width: int, height: int, cfg: Optional[Config] = None, export_dir: str = "exports"):
        self.cfg = cfg or Config()
        self.width, self.height = width, height
        self.export_dir = export_dir
        self.state = ToolState(brush_idx=self.cfg.default_brush_index)
        self._sync_brush()
        self.canvas = InkCanvas(width, height)
        self.strokes = StrokeController(self.canvas, self.cfg.stroke)
        g, s = self.cfg.gesture, self.cfg.smoothing
        self.pinch = PinchDetector.from_config(g)
        self.smoother = PointSmoother(s.min_cutoff, s.beta, s.d_cutoff)
        self.toolbar = ui.Toolbar(width, self.cfg.palette)
        self.cursor = None
        self.landmarks = None
        self.ratio = self.ratio2d = self.ratio2dt = self.ratio3d = None
        self.drawing = False
        self.confirm: Optional[ui.ConfirmDialog] = None
        self.toast = ("", 0.0)
        self.mouse = None
        self.quit = False
        self._recent = []  # (cursor, pinch value) for the last few frames

    # ---------------------------------------------------------------- input
    @property
    def color(self):
        return self.cfg.palette[self.state.color_idx][1]

    def _sync_brush(self):
        self.state.brush_size = self.cfg.brush_sizes[self.state.brush_idx]

    def stroke_width(self) -> int:
        s = self.state.brush_size
        return int(round(s * self.cfg.eraser_scale)) if self.state.tool == "eraser" else s

    def process_hand(self, landmarks, t: float, world=None) -> None:
        """One frame of hand input.

        landmarks: 21 (x, y) pixel points, or None when no hand is seen.
        world: optional 21 (x, y, z) metric points from MediaPipe; when present
        and enabled in config, the pinch is measured in 3D (tilt-proof).
        """
        self.landmarks = landmarks
        self.world = world
        self.ratio2d = self.ratio2dt = self.ratio3d = None
        if self.confirm is not None:
            landmarks = None  # never ink behind a modal dialog
        if landmarks is None:
            self.ratio = None
            self._recent.clear()
            self.pinch.update(None)
            self.smoother.reset()       # next sighting starts fresh, no glide-in
            self.cursor = None
            self.drawing = self.strokes.update(None, False, self.color, 1, False)
            return
        self.ratio2d = pinch_ratio(landmarks)
        if world is not None:
            self.ratio3d = pinch_ratio(world)
            self.ratio2dt = pinch_ratio_tilt_corrected(landmarks, world)
        r2d = self.ratio2dt if self.ratio2dt is not None else self.ratio2d
        self.ratio = combine_ratios(r2d, self.ratio3d, self.cfg.gesture.pinch_measure)
        pinched = self.pinch.update(self.ratio)
        self.cursor = self.smoother(landmarks[INDEX_TIP], t)
        manual = self.state.manual_ink
        was_active = self.strokes.active
        self.drawing = self.strokes.update(
            self.cursor, pinched or manual, self.color, self.stroke_width(),
            self.state.tool == "eraser", hold=self.pinch.pending_release and not manual,
        )
        if self.drawing and not was_active and not manual:
            self.canvas.prepend_points(self._backfill())
        keep = self.cfg.gesture.backfill_frames
        self._recent.append((self.cursor, self.pinch.value))
        del self._recent[:-keep or len(self._recent)]  # keep=0 keeps nothing

    def _backfill(self):
        """Recent positions (oldest first) where the fingers were already
        clearly closed (below the midpoint of the start and release levels),
        stopping at the first frame that was not, or was outside the area.
        Some people's relaxed thumb sits close to the index finger, so an open
        hand can read just under the release level; the midpoint keeps the
        hand's approach to a letter out of the stroke."""
        closed = (self.pinch.start_now + self.pinch.release_now) / 2
        pts = []
        for p, value in reversed(self._recent):
            if value is None or value > closed or not self.strokes.in_area(p):
                break
            pts.append(p)
        return pts[::-1]

    def status(self) -> str:
        if self.landmarks is None:
            return "No hand detected"
        if self.drawing:
            return "Erasing" if self.state.tool == "eraser" else "Drawing"
        return "Ready"

    # ------------------------------------------------------------- actions
    def action(self, name: str) -> None:
        st = self.state
        if name in ("pen", "eraser"):
            self.strokes.end()
            st.tool = name
        elif name.startswith("color"):
            self.strokes.end()
            st.color_idx = int(name[5:])
            st.tool = "pen"
        elif name == "size-":
            st.brush_idx = max(0, st.brush_idx - 1)
            self._sync_brush()
        elif name == "size+":
            st.brush_idx = min(len(self.cfg.brush_sizes) - 1, st.brush_idx + 1)
            self._sync_brush()
        elif name == "undo":
            self.strokes.end()
            self._notify("Undid last stroke" if self.canvas.undo() else "Nothing to undo")
        elif name == "clear":
            self.strokes.end()
            if self.canvas.strokes:
                self.confirm = ui.ConfirmDialog("Clear all ink?")
        elif name == "confirm_clear":
            self.canvas.clear()
            self.confirm = None
            self._notify("Canvas cleared")
        elif name == "cancel_clear":
            self.confirm = None
        elif name == "save":
            self.save()
        elif name == "view":
            st.show_camera = not st.show_camera
        elif name == "mirror":
            # Ink stays where it is on screen; only the live video is flipped.
            self.strokes.end()
            self.smoother.reset()
            self.pinch.reset()
            self._recent.clear()
            st.mirror = not st.mirror
            self._notify("Mirror on" if st.mirror else "Mirror off (true camera view)")
        elif name == "landmarks":
            st.show_landmarks = not st.show_landmarks
        elif name == "manual":
            st.manual_ink = not st.manual_ink
            self.strokes.end()
        elif name == "quit":
            self.quit = True

    def save(self) -> Optional[str]:
        self.strokes.end()
        if self.canvas.is_empty:
            self._notify("Nothing to save")
            return None
        name = datetime.now().strftime("airscribe_%Y%m%d_%H%M%S.png")
        path = self.canvas.save_png(os.path.join(self.export_dir, name))
        self._notify(f"Saved {path}")
        return path

    def _notify(self, text: str) -> None:
        self.toast = (text, time.monotonic() + 2.5)

    def on_key(self, key: int) -> None:
        if key < 0:
            return
        k = key & 0xFF
        ch = chr(k).lower() if 32 <= k < 127 else ""
        if self.confirm is not None:
            if ch == "y" or k == 13:
                self.action("confirm_clear")
            elif ch == "n" or k == 27:
                self.action("cancel_clear")
            return
        if k == 27 or ch == "q":
            self.action("quit")
        elif ch == " ":
            self.action("manual")
        elif ch == "p":
            self.action("pen")
        elif ch == "e":
            self.action("eraser")
        elif ch.isdigit() and 1 <= int(ch) <= len(self.cfg.palette):
            self.action(f"color{int(ch) - 1}")
        elif ch in ("[", "-"):
            self.action("size-")
        elif ch in ("]", "=", "+"):
            self.action("size+")
        elif ch == "u" or k == 26:  # 26 = Ctrl+Z
            self.action("undo")
        elif ch == "c":
            self.action("clear")
        elif ch == "s" or k == 19:  # 19 = Ctrl+S
            self.action("save")
        elif ch == "b":
            self.action("view")
        elif ch == "l":
            self.action("landmarks")
        elif ch == "m":
            self.action("mirror")

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.mouse = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.confirm is not None:
            vy = y - self.toolbar.height  # dialog is drawn in video coordinates
            if self.confirm.yes and self.confirm.yes.hit(x, vy):
                self.action("confirm_clear")
            elif self.confirm.no and self.confirm.no.hit(x, vy):
                self.action("cancel_clear")
            return
        hit = self.toolbar.hit(x, y)
        if hit:
            self.action(hit)

    # --------------------------------------------------------------- render
    def render(self, frame_bgr: np.ndarray) -> np.ndarray:
        st = self.state
        if st.show_camera:
            bg = frame_bgr
        else:
            bg = np.full((self.height, self.width, 3), self.cfg.paper_color, np.uint8)
        view = self.canvas.composite(bg)
        if st.show_landmarks and self.landmarks is not None:
            from .tracking import draw_landmarks
            draw_landmarks(view, self.landmarks)
        if self.cursor is not None:
            ui.draw_cursor(view, self.cursor, self.color, self.stroke_width() // 2,
                           self.drawing, st.tool == "eraser")
        if st.manual_ink:
            ui.draw_badge(view, "KEY INK ON (Space)")
        if self.confirm is not None:
            self.confirm.render(view)
        text, until = self.toast
        if text and time.monotonic() < until:
            ui.draw_toast(view, text)
        hover = self.mouse if self.mouse and self.mouse[1] < self.toolbar.height else None
        detail = f"p {self.ratio:.2f}" if self.ratio is not None else ""
        bar = self.toolbar.render(st, self.status(), hover, detail)
        return np.vstack([bar, view])


def _flat(points, digits) -> str:
    """21 landmarks as one space-separated cell, so a log can be replayed offline."""
    if points is None:
        return ""
    return " ".join(f"{v:.{digits}f}" for p in points for v in p)


class GestureLog:
    """Optional per-frame CSV for tuning: python -m airscribe --log gestures.csv"""

    def __init__(self, path: str):
        import csv
        # Line-buffered: every frame reaches disk at once, even if the app is
        # never closed with Q.
        self.f = open(path, "w", newline="", buffering=1)
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "hand", "ratio2d", "ratio3d", "ratio2dt", "pinched", "drawing",
                         "x", "y", "strokes", "end_reason", "mirror", "manual",
                         "landmarks_px", "landmarks_world"])
        self.prev_strokes = 0

    def write(self, t, app) -> None:
        n = len(app.canvas.strokes)
        c = app.cursor or (None, None)
        f = lambda v: "" if v is None else f"{v:.4f}"
        self.w.writerow([f"{t:.4f}", int(app.landmarks is not None), f(app.ratio2d), f(app.ratio3d), f(app.ratio2dt),
                         int(app.pinch.pinched), int(app.drawing), f(c[0]), f(c[1]), n,
                         app.strokes.end_reason, int(app.state.mirror), int(app.state.manual_ink),
                         _flat(app.landmarks, 1), _flat(getattr(app, "world", None), 5)])
        app.strokes.end_reason = ""

    def close(self) -> None:
        self.f.close()


def open_camera(index: int, width: int, height: int):
    # DirectShow opens quickly on Windows; fall back to the default backend.
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame
        cap.release()
    raise CameraError(
        f"Could not open camera index {index}. Check that a webcam is connected, "
        "not in use by another app (Teams, Zoom, browser), and allowed under "
        "Windows Settings > Privacy & security > Camera. "
        "Try another index with --camera 1."
    )


def run(camera_index: int, width: int, height: int, model_path: str,
        export_dir: str, cfg: Optional[Config] = None, mirror: bool = True,
        log_path: Optional[str] = None) -> None:
    from .tracking import HandTracker  # imported here so tests need no MediaPipe

    cfg = cfg or Config()
    tracker = HandTracker(model_path, cfg.tracker)  # fails fast if model missing
    cap = log = None
    try:
        cap, frame = open_camera(camera_index, width, height)
        h, w = frame.shape[:2]
        app = AirScribe(w, h, cfg, export_dir)
        app.state.mirror = mirror
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)  # 1:1 pixels, so mouse coords match
        cv2.setMouseCallback(WINDOW, app.on_mouse)
        print(f"AirScribe running on camera {camera_index} at {w}x{h}. Press Q or Esc to quit.")
        failures = 0
        log = GestureLog(log_path) if log_path else None
        if log:
            print(f"Logging every frame (with all hand landmarks) to {os.path.abspath(log_path)}")
        while not app.quit:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 30:
                    raise CameraError("Camera stopped delivering frames.")
                continue
            failures = 0
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            if app.state.mirror:
                frame = cv2.flip(frame, 1)  # flip BEFORE tracking, so landmarks are in screen space
            t = time.monotonic()
            app.process_hand(tracker.detect(frame), t, tracker.last_world)
            if log:
                log.write(t, app)
            cv2.imshow(WINDOW, app.render(frame))
            app.on_key(cv2.waitKey(1))
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
    finally:
        if log is not None:
            log.close()
            print(f"Gesture log written to {log_path}")
        if cap is not None:
            cap.release()
        tracker.close()
        cv2.destroyAllWindows()
