"""Bubble state analyser: determine whether bubbles are filled."""
from __future__ import annotations

import cv2
import numpy as np

from bubble_mark.processing.image_utils import apply_threshold, to_grayscale


class BubbleAnalyzer:
    """Analyse bubble regions in a normalised sheet image.

    Parameters
    ----------
    fill_threshold:
        Fraction of dark pixels (in the binarised bubble region) above which a
        bubble is considered *filled*.  Default is ``0.5``.
    """

    def __init__(self, fill_threshold: float = 0.5) -> None:
        if not 0.0 < fill_threshold <= 1.0:
            raise ValueError("fill_threshold must be in (0, 1]")
        self.fill_threshold = fill_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_bubble(self, image: np.ndarray, bubble_region: tuple) -> bool:
        """Return *True* if the bubble at *bubble_region* is filled.

        Parameters
        ----------
        image:
            Full normalised sheet image (grayscale or BGR).
        bubble_region:
            ``(x, y, w, h)`` bounding box of the bubble.
        """
        x, y, w, h = bubble_region
        if w <= 0 or h <= 0:
            return False

        # Clamp to image bounds
        ih, iw = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            return False

        roi = image[y1:y2, x1:x2]
        gray = to_grayscale(roi)

        # Use a simple mean-threshold: dark pixels < 128
        dark_fraction = np.mean(gray < 128)
        return bool(dark_fraction >= self.fill_threshold)

    def analyze_answer_row(
        self, image: np.ndarray, bubbles: list[tuple]
    ) -> str:
        """Analyse one question row and return the selected answer string.

        Returns
        -------
        str
            * ``"1"``–``"5"`` for a single filled bubble (choice index + 1).
            * ``"M"`` if more than one bubble is filled.
            * ``" "`` if no bubble is filled.
        """
        filled = [
            idx for idx, region in enumerate(bubbles)
            if self.analyze_bubble(image, region)
        ]
        if len(filled) == 0:
            return " "
        if len(filled) > 1:
            return "M"
        return str(filled[0] + 1)

    def analyze_id_column(
        self, image: np.ndarray, bubbles: list[tuple]
    ) -> str:
        """Analyse one ID digit column and return the recognised digit string.

        Parameters
        ----------
        bubbles:
            List of 10 bubble regions for digits 0–9 in order.

        Returns
        -------
        str
            * ``"0"``–``"9"`` for a single filled bubble.
            * ``"?"`` if more than one bubble is filled.
            * ``" "`` if no bubble is filled.
        """
        filled = [
            idx for idx, region in enumerate(bubbles)
            if self.analyze_bubble(image, region)
        ]
        if len(filled) == 0:
            return " "
        if len(filled) > 1:
            return "?"
        return str(filled[0])
