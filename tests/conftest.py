"""Shared pytest fixtures and synthetic image helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure the src package is importable without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------


def create_blank_sheet(width: int = 850, height: int = 1100) -> np.ndarray:
    """Return a white (255) BGR image of the given dimensions."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def create_filled_bubble_image(w: int = 30, h: int = 30) -> np.ndarray:
    """Return an all-black (0) grayscale image representing a filled bubble."""
    return np.zeros((h, w), dtype=np.uint8)


def create_empty_bubble_image(w: int = 30, h: int = 30) -> np.ndarray:
    """Return an all-white (255) grayscale image representing an empty bubble."""
    return np.full((h, w), 255, dtype=np.uint8)


def create_synthetic_bubble_sheet(
    width: int = 850,
    height: int = 1100,
    answers: str = "12345",
    student_id: str = "123456789",
    num_questions: int = 5,
    num_choices: int = 5,
    num_id_digits: int = 9,
) -> np.ndarray:
    """Create a realistic synthetic bubble sheet image.

    Bubbles for the given *answers* and *student_id* are drawn as filled
    dark circles; all others are drawn as outlined circles (empty).

    The layout mirrors the grid produced by :class:`BubbleSheetDetector`.
    """
    import cv2

    sheet = np.full((height, width, 3), 240, dtype=np.uint8)

    # ---- Answer section (top 72 % of sheet) ---------------------------
    ans_h = int(height * 0.72)
    margin_x = int(width * 0.05)
    margin_y = int(ans_h * 0.05)
    usable_w = width - 2 * margin_x
    usable_h = ans_h - 2 * margin_y

    cell_w = usable_w // num_choices
    cell_h = usable_h // num_questions
    bub_w = int(cell_w * 0.8)
    bub_h = int(cell_h * 0.8)
    pad_x = (cell_w - bub_w) // 2
    pad_y = (cell_h - bub_h) // 2

    for row in range(num_questions):
        answer_ch = answers[row] if row < len(answers) else " "
        for col in range(num_choices):
            x = margin_x + col * cell_w + pad_x
            y = margin_y + row * cell_h + pad_y
            cx, cy = x + bub_w // 2, y + bub_h // 2
            rx, ry = bub_w // 2, bub_h // 2
            chosen = str(col + 1) == answer_ch
            color = (30, 30, 30) if chosen else (200, 200, 200)
            thickness = -1 if chosen else 2
            cv2.ellipse(sheet, (cx, cy), (rx, ry), 0, 0, 360, color, thickness)

    # ---- ID section (bottom 26 % of sheet) ----------------------------
    id_top = int(height * 0.74)
    id_section_h = height - id_top
    id_margin_x = int(width * 0.05)
    id_margin_y = int(id_section_h * 0.05)
    id_usable_w = width - 2 * id_margin_x
    id_usable_h = id_section_h - 2 * id_margin_y

    id_choices_per_digit = 10
    id_cell_w = id_usable_w // num_id_digits
    id_cell_h = id_usable_h // id_choices_per_digit
    id_bub_w = int(id_cell_w * 0.8)
    id_bub_h = int(id_cell_h * 0.8)
    id_pad_x = (id_cell_w - id_bub_w) // 2
    id_pad_y = (id_cell_h - id_bub_h) // 2

    for digit_col in range(num_id_digits):
        digit_ch = student_id[digit_col] if digit_col < len(student_id) else " "
        for row in range(id_choices_per_digit):
            x = id_margin_x + digit_col * id_cell_w + id_pad_x
            y = id_top + id_margin_y + row * id_cell_h + id_pad_y
            cx, cy = x + id_bub_w // 2, y + id_bub_h // 2
            rx, ry = id_bub_w // 2, id_bub_h // 2
            chosen = str(row) == digit_ch
            color = (30, 30, 30) if chosen else (200, 200, 200)
            thickness = -1 if chosen else 2
            cv2.ellipse(sheet, (cx, cy), (rx, ry), 0, 0, 360, color, thickness)

    return sheet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_sheet():
    return create_blank_sheet()


@pytest.fixture
def filled_bubble():
    return create_filled_bubble_image()


@pytest.fixture
def empty_bubble():
    return create_empty_bubble_image()


@pytest.fixture
def synthetic_sheet():
    return create_synthetic_bubble_sheet(
        answers="12345",
        student_id="123456789",
        num_questions=5,
        num_choices=5,
    )


@pytest.fixture
def sample_answer_key():
    from bubble_mark.models.answer_key import AnswerKey

    return AnswerKey("12345", name="test_key")
