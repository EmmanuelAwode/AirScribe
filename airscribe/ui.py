"""Toolbar, status indicator, cursor and dialogs, drawn with OpenCV.

The toolbar is a separate strip stacked ABOVE the video, so it never covers ink
and the fingertip can never "press" a button: buttons respond to the mouse only.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BAR_BG = (34, 30, 28)
BTN_BG = (58, 52, 48)
BTN_HOVER = (78, 70, 64)
BTN_ACTIVE = (190, 130, 60)
TEXT = (235, 235, 235)
TEXT_DIM = (160, 160, 160)
ROW_H = 48
PAD = 8

STATUS_COLORS = {
    "No hand detected": (90, 90, 220),
    "Ready": (80, 190, 90),
    "Drawing": (60, 200, 250),
    "Erasing": (230, 160, 90),
}


@dataclass
class Button:
    id: str
    rect: Tuple[int, int, int, int]  # x, y, w, h
    label: str = ""
    swatch: Optional[Tuple[int, int, int]] = None

    def hit(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self.rect
        return bx <= x < bx + bw and by <= y < by + bh


class Toolbar:
    STATUS_W = 210

    def __init__(self, width: int, palette):
        self.width = width
        self.palette = palette
        self.buttons: List[Button] = []
        self.height = ROW_H
        self._layout()

    def _layout(self) -> None:
        # (id, width, label, swatch) ; None entries are group gaps.
        items = [("pen", 52, "Pen", None), ("eraser", 66, "Eraser", None), None]
        items += [(f"color{i}", 30, "", c) for i, (_, c) in enumerate(self.palette)]
        items += [None, ("size-", 28, "-", None), ("sizeval", 40, "", None), ("size+", 28, "+", None), None]
        items += [("undo", 58, "Undo", None), ("clear", 58, "Clear", None), ("save", 56, "Save", None), None]
        items += [("view", 74, "Camera", None), ("mirror", 92, "Mirror", None)]

        x, row = PAD, 0
        limit = self.width - self.STATUS_W - PAD  # first row leaves room for status
        self.buttons = []
        for it in items:
            if it is None:
                x += 10
                continue
            bid, w, label, sw = it
            if x + w > limit:
                row += 1
                x = PAD
                limit = self.width - PAD
            h = 30
            y = row * ROW_H + (ROW_H - h) // 2
            self.buttons.append(Button(bid, (x, y, w, h), label, sw))
            x += w + 4
        self.height = (row + 1) * ROW_H

    def hit(self, x: int, y: int) -> Optional[str]:
        for b in self.buttons:
            if b.id != "sizeval" and b.hit(x, y):
                return b.id
        return None

    def render(self, state, status: str, hover: Optional[Tuple[int, int]], detail: str = "") -> np.ndarray:
        bar = np.full((self.height, self.width, 3), BAR_BG, np.uint8)
        for b in self.buttons:
            x, y, w, h = b.rect
            hovered = hover is not None and b.hit(*hover) and b.id != "sizeval"
            if b.swatch is not None:
                idx = int(b.id[5:])
                selected = idx == state.color_idx and state.tool == "pen"
                cv2.rectangle(bar, (x, y), (x + w, y + h), b.swatch, -1)
                cv2.rectangle(bar, (x, y), (x + w, y + h), (90, 90, 90), 1)
                if idx == state.color_idx:
                    ring = (255, 255, 255) if selected else (130, 130, 130)
                    cv2.rectangle(bar, (x - 3, y - 3), (x + w + 3, y + h + 3), ring, 2)
                continue
            active = (b.id == "pen" and state.tool == "pen") or (b.id == "eraser" and state.tool == "eraser")
            bg = BTN_ACTIVE if active else (BTN_HOVER if hovered else BTN_BG)
            if b.id != "sizeval":
                cv2.rectangle(bar, (x, y), (x + w, y + h), bg, -1)
            label = b.label
            if b.id == "sizeval":
                label = f"{state.brush_size}px"
            elif b.id == "view":
                label = "Camera" if state.show_camera else "Canvas"
            elif b.id == "mirror":
                label = "Mirror: on" if state.mirror else "Mirror: off"
            _center_text(bar, label, (x, y, w, h), 0.45, TEXT)

        # Status pill (top-right)
        sx = self.width - self.STATUS_W
        cy = ROW_H // 2
        color = STATUS_COLORS.get(status, TEXT)
        cv2.rectangle(bar, (sx, cy - 15), (self.width - PAD, cy + 15), (48, 44, 40), -1)
        cv2.circle(bar, (sx + 14, cy), 6, color, -1, cv2.LINE_AA)
        cv2.putText(bar, status, (sx + 26, cy + 5), FONT, 0.47, TEXT, 1, cv2.LINE_AA)
        if detail:
            cv2.putText(bar, detail, (self.width - PAD - 52, cy + 5), FONT, 0.38, TEXT_DIM, 1, cv2.LINE_AA)
        return bar


def _center_text(img, text, rect, scale, color):
    x, y, w, h = rect
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    cv2.putText(img, text, (x + (w - tw) // 2, y + (h + th) // 2), FONT, scale, color, 1, cv2.LINE_AA)


def draw_cursor(img, p, color, radius: int, drawing: bool, erase: bool) -> None:
    c = (int(p[0]), int(p[1]))
    if erase:
        cv2.circle(img, c, max(radius, 4), (40, 40, 40), 3, cv2.LINE_AA)
        cv2.circle(img, c, max(radius, 4), (245, 245, 245), 1, cv2.LINE_AA)
    else:
        cv2.circle(img, c, max(radius, 3) + 3, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.circle(img, c, max(radius, 3) + 3, color, 1, cv2.LINE_AA)
    # The small dot marks exactly where the next ink lands.
    cv2.circle(img, c, 3, color if not erase else (245, 245, 245), -1, cv2.LINE_AA)
    if drawing:
        cv2.circle(img, c, 3, (255, 255, 255), 1, cv2.LINE_AA)


class ConfirmDialog:
    def __init__(self, message: str):
        self.message = message
        self.yes: Optional[Button] = None
        self.no: Optional[Button] = None

    def render(self, img) -> None:
        h, w = img.shape[:2]
        bw, bh = 340, 130
        x, y = (w - bw) // 2, (h - bh) // 2
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), BAR_BG, -1)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (90, 90, 90), 1)
        _center_text(img, self.message, (x, y + 14, bw, 30), 0.55, TEXT)
        _center_text(img, "Y / Enter = clear    N / Esc = cancel", (x, y + 42, bw, 20), 0.38, TEXT_DIM)
        self.yes = Button("yes", (x + 40, y + 78, 120, 34), "Clear all")
        self.no = Button("no", (x + bw - 160, y + 78, 120, 34), "Cancel")
        cv2.rectangle(img, self.yes.rect[:2], (self.yes.rect[0] + 120, self.yes.rect[1] + 34), (60, 60, 200), -1)
        cv2.rectangle(img, self.no.rect[:2], (self.no.rect[0] + 120, self.no.rect[1] + 34), BTN_BG, -1)
        _center_text(img, "Clear all", self.yes.rect, 0.5, TEXT)
        _center_text(img, "Cancel", self.no.rect, 0.5, TEXT)


def draw_toast(img, text: str) -> None:
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    x, y = (w - tw) // 2, h - 24
    cv2.rectangle(img, (x - 12, y - th - 10), (x + tw + 12, y + 10), BAR_BG, -1)
    cv2.putText(img, text, (x, y), FONT, 0.5, TEXT, 1, cv2.LINE_AA)


def draw_badge(img, text: str) -> None:
    """Small top-left badge on the video (e.g. manual ink mode)."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    cv2.rectangle(img, (8, 8), (20 + tw, 16 + th + 6), (60, 160, 230), -1)
    cv2.putText(img, text, (14, 14 + th), FONT, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
