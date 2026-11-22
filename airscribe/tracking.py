"""Hand tracking: wraps MediaPipe Hand Landmarker (Tasks API), runs fully locally.

Input: a mirrored BGR frame. Output: a list of 21 (x, y) pixel landmarks for the
first detected hand, or None. Nothing else in the app touches MediaPipe.
"""
import os
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .config import TrackerConfig

Landmarks = List[Tuple[float, float]]

# Bone connections between the 21 landmarks, for the optional skeleton overlay.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class HandTracker:
    def __init__(self, model_path: str, cfg: TrackerConfig):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at '{model_path}'.\n"
                "Download it with:  python scripts\\download_model.py"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        self.last_world = None  # 21 (x, y, z) metric points for the last detection

    def detect(self, frame_bgr) -> Optional[Landmarks]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            self.last_world = None
            return None
        self.last_world = ([(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]]
                           if result.hand_world_landmarks else None)
        # Landmarks are normalised [0, 1]; convert to pixel coordinates.
        return [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]

    def close(self) -> None:
        self._landmarker.close()


def draw_landmarks(img, landmarks: Landmarks) -> None:
    pts = [(int(x), int(y)) for x, y in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 2, (90, 220, 255), -1, cv2.LINE_AA)
