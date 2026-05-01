"""Bubble sheet detector: find and normalise the sheet, locate bubbles."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from bubble_mark.processing.image_utils import (
    apply_threshold,
    find_page_contour,
    perspective_transform,
    resize_image,
    to_grayscale,
)

# Standard normalised sheet dimensions used internally
_SHEET_WIDTH = 850
_SHEET_HEIGHT = 1100


class BubbleSheetDetector:
    """Detect a bubble sheet in an image and locate its bubble regions.

    Parameters
    ----------
    layout_config:
        A dictionary describing the sheet layout.  Recognised keys:

        * ``num_questions``        – total answer questions (default 30)
        * ``num_choices``          – choices per question (default 5, i.e. A-E)
        * ``num_id_digits``        – digits in the student-ID field (default 9)
        * ``id_choices_per_digit`` – choices per ID digit (default 10, i.e. 0-9)
    """

    DEFAULT_LAYOUT: dict = {
        "num_questions": 30,
        "num_choices": 5,
        "num_id_digits": 9,
        "id_choices_per_digit": 10,
    }

    def __init__(self, layout_config: Optional[dict] = None) -> None:
        self.layout_config: dict = {**self.DEFAULT_LAYOUT, **(layout_config or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Pre-process *image* and return a normalised sheet image or *None*."""
        contour = find_page_contour(image)
        if contour is not None:
            normalised = perspective_transform(image, contour)
        else:
            # Fall back: assume entire image is the sheet
            normalised = image.copy()

        normalised = resize_image(normalised, width=_SHEET_WIDTH, height=_SHEET_HEIGHT)
        return normalised

    def locate_answer_bubbles(
        self, normalised_image: np.ndarray
    ) -> list[list[tuple[int, int, int, int]]]:
        """Return a grid of answer-bubble bounding boxes.

        Returns a list of rows; each row is a list of ``(x, y, w, h)`` tuples
        (one per choice column).
        """
        num_q = self.layout_config["num_questions"]
        num_c = self.layout_config["num_choices"]

        # Answer section occupies roughly the top 70% of the sheet,
        # leaving room for an ID section at the bottom.
        section = self._answer_section(normalised_image)
        return self._build_bubble_grid(section, num_q, num_c, offset_y=_answer_offset_y())

    def locate_id_bubbles(
        self, normalised_image: np.ndarray
    ) -> list[list[tuple[int, int, int, int]]]:
        """Return a grid of student-ID bubble bounding boxes.

        Returns a list of columns (one per digit); each column is a list of
        ``(x, y, w, h)`` tuples (one per digit choice row).
        """
        num_d = self.layout_config["num_id_digits"]
        num_choices = self.layout_config["id_choices_per_digit"]

        section = self._id_section(normalised_image)
        # For ID we build a grid where rows = digit choices, cols = digits
        # then transpose to get per-digit lists
        raw_grid = self._build_bubble_grid(section, num_choices, num_d, offset_y=_id_offset_y())
        # raw_grid[row][col] → we want [col][row]
        columns: list[list[tuple[int, int, int, int]]] = [
            [raw_grid[row][col] for row in range(num_choices)]
            for col in range(num_d)
        ]
        return columns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _answer_section(self, image: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        return image[: int(h * 0.72), :]

    def _id_section(self, image: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        return image[int(h * 0.74) :, :]

    def _build_bubble_grid(
        self,
        section: np.ndarray,
        num_rows: int,
        num_cols: int,
        offset_y: int = 0,
    ) -> list[list[tuple[int, int, int, int]]]:
        """Divide *section* into a uniform grid of bubble regions."""
        h, w = section.shape[:2]
        # Add margins (5% each side)
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        usable_w = w - 2 * margin_x
        usable_h = h - 2 * margin_y

        cell_w = usable_w // num_cols
        cell_h = usable_h // num_rows

        # Shrink each cell to get a tighter bubble region (80% of cell)
        bub_w = int(cell_w * 0.8)
        bub_h = int(cell_h * 0.8)
        pad_x = (cell_w - bub_w) // 2
        pad_y = (cell_h - bub_h) // 2

        grid: list[list[tuple[int, int, int, int]]] = []
        for row in range(num_rows):
            row_bubbles: list[tuple[int, int, int, int]] = []
            for col in range(num_cols):
                x = margin_x + col * cell_w + pad_x
                y = margin_y + row * cell_h + pad_y + offset_y
                row_bubbles.append((x, y, bub_w, bub_h))
            grid.append(row_bubbles)
        return grid


def _answer_offset_y() -> int:
    return 0


def _id_offset_y() -> int:
    return 0
