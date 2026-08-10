"""Face detection wrapper around MediaPipe.

Why this exists:
    The vision emotion model assumes a tight crop of a face. Feeding it a
    full photo (background + body + face) tanks accuracy. Examiners and demo
    images rarely give us perfect crops, so we detect first.

Falls back gracefully when no face is found (uses the whole image rather
than crashing) and exposes the bounding box so the UI can draw it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FaceCrop:
    """Face detection result."""

    image: Image.Image          # the cropped (or whole) RGB image
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) in original coords or None
    confidence: float           # detector confidence in [0, 1]; 0 if fell back
    detected: bool              # whether a real face was found


class FaceDetector:
    """Lazy MediaPipe face detector. Singleton-style to avoid re-init costs."""

    def __init__(self, min_confidence: float = 0.5, padding_ratio: float = 0.15):
        self._mp = None
        self._detector = None
        self._min_confidence = min_confidence
        self._padding_ratio = padding_ratio

    def _ensure_loaded(self) -> None:
        if self._detector is not None:
            return
        try:
            import mediapipe as mp

            self._mp = mp
            # model_selection=1 = full-range (better for far / small faces)
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self._min_confidence,
            )
        except Exception as e:  # pragma: no cover -- graceful degrade
            logger.warning("MediaPipe unavailable (%s); falling back to whole-frame.", e)
            self._detector = None

    def detect(self, image: Image.Image) -> FaceCrop:
        """Detect the most prominent face. Returns FaceCrop (with fallback if none)."""
        self._ensure_loaded()

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self._detector is None:
            return FaceCrop(image=image, bbox=None, confidence=0.0, detected=False)

        np_img = np.array(image)
        results = self._detector.process(np_img)

        if not results.detections:
            return FaceCrop(image=image, bbox=None, confidence=0.0, detected=False)

        # Pick highest-confidence detection.
        det = max(results.detections, key=lambda d: d.score[0])
        rel = det.location_data.relative_bounding_box
        h, w, _ = np_img.shape
        x = max(0, int(rel.xmin * w))
        y = max(0, int(rel.ymin * h))
        bw = int(rel.width * w)
        bh = int(rel.height * h)

        # Add padding so we don't crop hairline / chin.
        pad_x = int(bw * self._padding_ratio)
        pad_y = int(bh * self._padding_ratio)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)

        if x1 <= x0 or y1 <= y0:
            return FaceCrop(image=image, bbox=None, confidence=0.0, detected=False)

        crop = image.crop((x0, y0, x1, y1))
        return FaceCrop(
            image=crop,
            bbox=(x0, y0, x1 - x0, y1 - y0),
            confidence=float(det.score[0]),
            detected=True,
        )


# Module-level singleton.
_DETECTOR: Optional[FaceDetector] = None


def get_face_detector() -> FaceDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FaceDetector()
    return _DETECTOR


__all__ = ["FaceDetector", "FaceCrop", "get_face_detector"]
