"""Camera-free tests: pinch hysteresis, stroke breaks, smoothing, canvas, export.

Hands are synthesised as 21 landmarks so the whole per-frame pipeline
(AirScribe.process_hand) runs without a webcam or the MediaPipe model.
"""
import cv2
import numpy as np
import pytest

from airscribe.app import AirScribe
from airscribe.canvas import InkCanvas
from airscribe.config import Config, StrokeConfig
from airscribe.gestures import PinchDetector, combine_ratios, pinch_ratio
from airscribe.smoothing import PointSmoother
from airscribe.strokes import StrokeController

W, H = 640, 480
HAND = 100.0  # synthetic hand size in px (wrist -> middle MCP)


def hand(tip, ratio, scale=1.0):
    """21 landmarks with the index tip at `tip` and the given pinch ratio."""
    x, y = tip
    size = HAND * scale
    lm = [(x, y + 2 * size)] * 21
    lm[0] = (x, y + 2 * size)          # wrist
    lm[9] = (x, y + 1 * size)          # middle MCP -> hand size = size
    lm[8] = (x, y)                     # index tip
    lm[4] = (x + ratio * size, y)      # thumb tip
    return lm


# ------------------------------------------------------------ pinch ratio
def test_pinch_ratio_is_scale_invariant():
    near = pinch_ratio(hand((200, 100), 0.3, scale=2.0))
    far = pinch_ratio(hand((200, 100), 0.3, scale=0.5))
    assert near == pytest.approx(0.3) and far == pytest.approx(0.3)


def test_pinch_ratio_degenerate_hand():
    assert pinch_ratio([(5.0, 5.0)] * 21) is None


# ------------------------------------------------------ pinch hysteresis
def test_debounced_pinch_needs_consecutive_frames():
    d = PinchDetector(0.30, 0.45, confirm_frames=2)
    assert d.update(0.1) is False          # first frame below start: not yet
    assert d.update(0.5) is False          # streak broken
    assert d.update(0.1) is False
    assert d.update(0.1) is True           # two in a row
    assert d.update(0.9) is True and d.pending_release
    assert d.update(0.2) is True and not d.pending_release
    assert d.update(0.9) is True
    assert d.update(0.9) is False


def test_warmup_blocks_pinch_for_new_hand():
    d = PinchDetector(0.30, 0.45, confirm_frames=1, warmup_frames=3)
    assert [d.update(0.1) for _ in range(4)] == [False, False, False, True]
    d.update(None)                         # hand lost -> warm-up starts again
    assert d.update(0.1) is False


def test_pinch_starts_only_below_start_threshold():
    d = PinchDetector(start=0.30, release=0.45)
    assert d.update(0.50) is False
    assert d.update(0.31) is False   # between thresholds while released: stay released
    assert d.update(0.29) is True    # crosses start


def test_pinch_releases_only_above_release_threshold():
    d = PinchDetector(0.30, 0.45)
    d.update(0.2)
    assert d.update(0.40) is True    # between thresholds while pinched: stay pinched
    assert d.update(0.45) is True    # equal is not above
    assert d.update(0.46) is False


def test_no_flicker_when_wobbling_around_one_boundary():
    d = PinchDetector(0.30, 0.45)
    d.update(0.2)
    states = [d.update(r) for r in (0.29, 0.31, 0.29, 0.33, 0.30, 0.44, 0.31)]
    assert all(states)
    d2 = PinchDetector(0.30, 0.45)
    states = [d2.update(r) for r in (0.46, 0.44, 0.46, 0.44, 0.31, 0.45)]
    assert not any(states)


def test_lost_hand_releases_pinch():
    d = PinchDetector(0.30, 0.45)
    d.update(0.1)
    assert d.update(None) is False
    assert d.update(0.40) is False   # must cross start again


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        PinchDetector(0.5, 0.4)


# ---------------------------------------------------------- stroke breaks
def make_controller():
    canvas = InkCanvas(W, H)
    return canvas, StrokeController(canvas, StrokeConfig(edge_margin=10))


def feed(ctrl, frames):
    for p, down in frames:
        ctrl.update(p, down, (255, 255, 255), 4, False)


def test_release_ends_stroke_and_next_pinch_starts_new_one():
    canvas, ctrl = make_controller()
    feed(ctrl, [((100, 100), True), ((120, 100), True), ((140, 100), False),
                ((300, 300), True), ((320, 300), True)])
    assert len(canvas.strokes) == 2
    assert canvas.strokes[0].points == [(100, 100), (120, 100)]
    assert canvas.strokes[1].points[0] == (300, 300)


def test_tracking_dropout_breaks_stroke_without_connecting_line():
    canvas, ctrl = make_controller()
    feed(ctrl, [((100, 100), True), ((110, 100), True), (None, True),
                ((500, 400), True), ((510, 400), True)])
    assert len(canvas.strokes) == 2
    # Nothing was drawn along the straight path between the two strokes.
    mid = (300, 250)
    assert canvas.alpha[mid[1], mid[0]] == 0


def test_leaving_drawing_area_ends_stroke():
    canvas, ctrl = make_controller()
    feed(ctrl, [((100, 100), True), ((50, 100), True), ((5, 100), True),   # x=5 < margin
                ((50, 100), True)])
    assert len(canvas.strokes) == 2
    assert all(p[0] >= 10 for s in canvas.strokes for p in s.points)


def test_large_jump_is_treated_as_glitch():
    canvas, ctrl = make_controller()
    feed(ctrl, [((50, 50), True), ((60, 50), True), ((600, 450), True), ((605, 450), True)])
    assert len(canvas.strokes) == 2
    assert canvas.alpha[250, 330] == 0


class Clock:
    """Feeds frames into the app at 30 fps."""

    def __init__(self, app):
        self.app, self.t = app, 0.0

    def __call__(self, tip=None, ratio=None, frames=1, world=None):
        for _ in range(frames):
            self.t += 1 / 30
            lm = None if tip is None else hand(tip, ratio)
            self.app.process_hand(lm, self.t, world)


SETTLE = 6  # frames for warm-up + confirmation with default config


def test_full_pipeline_dropout_resets_smoothing_and_stroke():
    app = AirScribe(W, H)
    feed_ = Clock(app)
    feed_((100, 150), 0.1, frames=SETTLE)
    for x in range(100, 200, 10):             # pinched stroke moving right
        feed_((x, 150), 0.1)
    assert app.status() == "Drawing"
    feed_(None)                                # tracking dropout
    assert app.status() == "No hand detected" and not app.strokes.active
    feed_((450, 350), 0.1)
    # Smoother was reset, so the cursor is exactly at the new tip (no glide-in)
    assert app.cursor == pytest.approx((450, 350))
    feed_((450, 350), 0.1, frames=SETTLE)
    assert len(app.canvas.strokes) == 2
    assert app.canvas.strokes[1].points[0] == (450, 350)


def test_pipeline_status_ready_vs_drawing_vs_erasing():
    app = AirScribe(W, H)
    feed_ = Clock(app)
    feed_((200, 200), 0.8, frames=4)
    assert app.status() == "Ready"
    feed_((200, 200), 0.1, frames=4)       # 3-frame average + 2-frame confirmation
    assert app.status() == "Drawing"
    app.action("eraser")
    feed_((210, 200), 0.1)
    assert app.status() == "Erasing"


def test_keyboard_toggle_inks_without_pinch():
    app = AirScribe(W, H)
    app.on_key(ord(" "))
    assert app.state.manual_ink
    app.process_hand(hand((200, 200), 0.9), 0.0)
    app.process_hand(hand((230, 200), 0.9), 0.03)
    assert app.status() == "Drawing" and len(app.canvas.strokes) == 1
    app.on_key(ord(" "))
    app.process_hand(hand((260, 200), 0.9), 0.06)
    assert app.status() == "Ready" and not app.strokes.active


def test_no_ink_while_clear_dialog_open():
    app = AirScribe(W, H)
    feed_ = Clock(app)
    feed_((200, 200), 0.1, frames=SETTLE)
    feed_(None)
    app.action("clear")
    assert app.confirm is not None
    feed_((300, 300), 0.1, frames=SETTLE)
    assert len(app.canvas.strokes) == 1
    app.on_key(ord("n"))
    assert app.confirm is None and len(app.canvas.strokes) == 1
    app.action("clear")
    app.on_key(ord("y"))
    assert app.canvas.strokes == [] and app.canvas.is_empty


# ------------------------------------------- consistency (debounce, 3D, tails)
def test_one_frame_ratio_spike_does_not_break_stroke():
    app = AirScribe(W, H)
    feed_ = Clock(app)
    feed_((100, 200), 0.1, frames=SETTLE)
    feed_((120, 200), 0.1)
    feed_((140, 200), 0.9)            # single noisy frame: looks released
    feed_((160, 200), 0.1)
    feed_((180, 200), 0.1)
    assert len(app.canvas.strokes) == 1 and app.strokes.active


def test_one_frame_false_pinch_does_not_start_stroke():
    app = AirScribe(W, H)
    feed_ = Clock(app)
    feed_((200, 200), 0.8, frames=5)
    feed_((200, 200), 0.1)            # one noisy frame
    feed_((200, 200), 0.8, frames=3)
    assert app.canvas.strokes == []


def test_hand_entering_frame_already_pinched_leaves_no_stray_dot():
    app = AirScribe(W, H)
    feed_ = Clock(app)
    feed_((300, 400), 0.1, frames=3)  # warm-up: tracking may be wrong here
    assert app.canvas.strokes == []
    feed_((300, 400), 0.1, frames=2)
    assert len(app.canvas.strokes) == 1


def test_release_leaves_no_tail_from_opening_fingers():
    app = AirScribe(W, H)
    feed_ = Clock(app)
    feed_((100, 200), 0.1, frames=SETTLE)
    for x in (120, 140, 160):
        feed_((x, 200), 0.1)
    feed_((200, 240), 0.9)            # fingers opening while the hand moves on
    feed_((240, 280), 0.9)
    feed_((280, 320), 0.9)            # release confirmed (averaged value)
    pts = app.canvas.strokes[0].points
    assert not app.strokes.active
    assert max(p[0] for p in pts) <= 165      # nothing from the release frames


def test_combine_ratios_modes_and_fallback():
    assert combine_ratios(0.2, 0.4, "blend") == pytest.approx(0.3)
    assert combine_ratios(0.2, 0.4, "2d") == 0.2
    assert combine_ratios(0.2, 0.4, "3d") == 0.4
    assert combine_ratios(0.2, None, "3d") == 0.2      # no world landmarks -> 2D
    assert combine_ratios(None, 0.4, "2d") == 0.4
    with pytest.raises(ValueError):
        combine_ratios(0.2, 0.4, "4d")


def test_pinch_in_front_of_body_is_detected_with_blend():
    """Logged values in front of the body: 3D reads a real pinch as ~0.31,
    above the start threshold, while 2D reads ~0.18. The blend (~0.25) draws."""
    world = [(x / 1000, y / 1000, 0.0) for x, y in hand((200, 200), 0.31)]
    app = AirScribe(W, H)
    Clock(app)((200, 200), 0.18, frames=SETTLE, world=world)
    assert app.ratio == pytest.approx(0.245) and app.status() == "Drawing"
    cfg = Config()
    cfg.gesture.pinch_measure = "3d"
    app3d = AirScribe(W, H, cfg)
    Clock(app3d)((200, 200), 0.18, frames=SETTLE, world=world)
    assert app3d.status() == "Ready"                     # the old behaviour: missed


def test_noisy_handwriting_gives_one_stroke_per_letter():
    """Simulates writing four letters with realistic pinch noise and occasional
    one-frame landmark glitches. Each letter must be exactly one stroke."""
    rng = np.random.default_rng(7)
    app = AirScribe(W, H)
    feed_ = Clock(app)
    feed_((60, 200), 0.7, frames=SETTLE)
    for letter in range(4):
        x0 = 80 + letter * 130
        for i in range(40):                       # pen down: ratio ~0.15
            r = 0.15 + rng.normal(0, 0.05)
            if rng.random() < 0.05:
                r = 0.6                           # one-frame glitch
            feed_((x0 + 60 * np.sin(i / 6), 200 + 2.5 * i), max(r, 0.0))
        for i in range(15):                       # pen up, move to next letter
            feed_((x0 + 60 + i * 5, 200 - i * 3), 0.7 + rng.normal(0, 0.05))
    assert len(app.canvas.strokes) == 4


# --------------------------------------------------------------- smoothing
def test_smoother_reduces_jitter_but_follows_fast_motion():
    rng = np.random.default_rng(0)
    s = PointSmoother(1.5, 0.015, 1.0)
    raw, out = [], []
    for i in range(60):                       # still finger with 3 px noise
        p = (300 + rng.normal(0, 3), 200 + rng.normal(0, 3))
        raw.append(p)
        out.append(s(p, i / 30))
    assert np.std([p[0] for p in out[10:]]) < 0.5 * np.std([p[0] for p in raw[10:]])
    s.reset()
    t = 0.0
    for x in range(0, 600, 40):               # fast sweep: 1200 px/s
        t += 1 / 30
        y = s((x, 100), t)
    assert abs(y[0] - 560) < 40               # lags less than one frame of motion


# ------------------------------------------------------------ canvas/export
def draw_line(canvas, a, b, color=(0, 0, 255), size=8, erase=False):
    canvas.begin_stroke(color, size, erase)
    for i in range(11):
        canvas.add_point((a[0] + (b[0] - a[0]) * i / 10, a[1] + (b[1] - a[1]) * i / 10))
    canvas.end_stroke()


def test_eraser_removes_ink_and_undo_restores_it():
    c = InkCanvas(W, H)
    draw_line(c, (100, 200), (500, 200))
    assert c.alpha[200, 300] == 255
    draw_line(c, (300, 150), (300, 250), size=30, erase=True)
    assert c.alpha[200, 300] == 0 and c.alpha[200, 150] == 255
    assert c.undo()
    assert c.alpha[200, 300] == 255
    assert c.undo() and c.is_empty
    assert not c.undo()


def test_png_export_is_transparent_and_contains_no_camera_pixels(tmp_path):
    app = AirScribe(W, H, export_dir=str(tmp_path))
    draw_line(app.canvas, (100, 100), (400, 100), color=(0, 0, 255))
    camera = np.full((H, W, 3), (10, 200, 30), np.uint8)   # pretend webcam frame
    app.render(camera)
    path = app.save()
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img.shape == (H, W, 4)
    assert img[300, 300, 3] == 0                         # background transparent
    assert tuple(img[100, 250]) == (0, 0, 255, 255)      # ink kept, opaque
    opaque = img[img[..., 3] > 0][:, :3]
    assert not np.any(np.all(opaque == (10, 200, 30), axis=1))


def test_save_with_empty_canvas_writes_nothing(tmp_path):
    app = AirScribe(W, H, export_dir=str(tmp_path))
    assert app.save() is None
    assert list(tmp_path.iterdir()) == []


def test_view_toggle_keeps_drawing():
    app = AirScribe(W, H)
    draw_line(app.canvas, (100, 300), (400, 300), color=(255, 0, 0))
    camera = np.zeros((H, W, 3), np.uint8)
    a = app.render(camera)
    app.action("view")
    b = app.render(camera)
    tb = app.toolbar.height
    assert tuple(a[tb + 300, 250]) == (255, 0, 0) == tuple(b[tb + 300, 250])
    assert tuple(b[tb + 400, 250]) == app.cfg.paper_color


def test_toolbar_buttons_map_to_actions():
    app = AirScribe(1280, 720)
    ids = {b.id for b in app.toolbar.buttons}
    assert {"pen", "eraser", "undo", "clear", "save", "view", "size-", "size+"} <= ids
    assert sum(i.startswith("color") for i in ids) >= 6
    b = next(b for b in app.toolbar.buttons if b.id == "color3")
    x, y, w, h = b.rect
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, x + w // 2, y + h // 2, 0, None)
    assert app.state.color_idx == 3 and app.state.tool == "pen"


def test_toolbar_wraps_on_narrow_frames():
    app = AirScribe(640, 480)
    for b in app.toolbar.buttons:
        x, y, w, h = b.rect
        assert x + w <= 640
    assert app.toolbar.height >= 48


def test_mirror_toggle_by_key_and_button_keeps_ink_and_breaks_stroke():
    app = AirScribe(1280, 720)
    assert app.state.mirror is True
    Clock(app)((200, 200), 0.1, frames=SETTLE)
    assert app.strokes.active
    app.on_key(ord("m"))
    assert app.state.mirror is False and not app.strokes.active
    assert len(app.canvas.strokes) == 1           # ink kept
    b = next(b for b in app.toolbar.buttons if b.id == "mirror")
    x, y, w, h = b.rect
    assert x + w <= 1280 - app.toolbar.STATUS_W   # fits on the first row
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, x + 5, y + 5, 0, None)
    assert app.state.mirror is True


# ----------------------------------------------------- adapting to the hand
def test_adaptive_thresholds_follow_a_close_resting_thumb():
    d = PinchDetector(0.28, 0.44, adapt=True, release_margin=0.05, start_margin=0.17,
                      min_start=0.22, min_release=0.30, adapt_rate=0.05)
    assert d.release_now == pytest.approx(0.44)          # unchanged until learned
    for _ in range(200):
        d.update(0.42)                                   # relaxed hand reads 0.42
    assert d.open_level == pytest.approx(0.42, abs=0.01)
    assert d.release_now == pytest.approx(0.37, abs=0.01)    # now lifts below 0.42
    assert d.start_now == pytest.approx(0.25, abs=0.01)
    assert d.release_now - d.start_now >= 0.06


def test_adaptive_thresholds_never_exceed_config_or_floors():
    d = PinchDetector(0.28, 0.44, adapt=True, min_start=0.22, min_release=0.30, adapt_rate=0.2)
    for _ in range(100):
        d.update(0.9)                                    # very open hand
    assert d.release_now == 0.44 and d.start_now == 0.28
    d2 = PinchDetector(0.28, 0.44, adapt=True, min_start=0.22, min_release=0.30, adapt_rate=0.2)
    for _ in range(100):
        d2.update(0.31)                                  # hovering almost closed
    assert d2.release_now >= 0.30 and d2.start_now >= 0.22 - 1e-9
    assert d2.start_now <= d2.release_now - 0.06 + 1e-9


def test_pinching_does_not_teach_the_open_level():
    d = PinchDetector(0.28, 0.44, adapt=True, adapt_rate=0.2)
    before = d.open_level
    for _ in range(100):
        d.update(0.10)                                   # holding a pinch
    assert d.pinched and d.open_level == before
