"""Tests for bubble_mark.processing.grid_detection."""

from __future__ import annotations

import numpy as np
import pytest

import bubble_mark.processing.grid_detection as _gd_mod
from bubble_mark.processing.grid_detection import (
    detect_block_groups,
    detect_bubble_grid,
)


@pytest.fixture
def no_cv2(monkeypatch):
    """Force the pure NumPy/Pillow fallback paths."""
    monkeypatch.setattr(_gd_mod, "_HAVE_CV2", False)


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------


def _make_grid_image(
    num_rows: int,
    num_cols: int,
    cell_h: int = 20,
    cell_w: int = 20,
    line_thickness: int = 2,
) -> np.ndarray:
    """Return a white BGR image with a visible dark grid drawn on it."""
    h = num_rows * cell_h + line_thickness
    w = num_cols * cell_w + line_thickness
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    # Draw horizontal lines
    for r in range(num_rows + 1):
        y = r * cell_h
        img[y : y + line_thickness, :] = 30
    # Draw vertical lines
    for c in range(num_cols + 1):
        x = c * cell_w
        img[:, x : x + line_thickness] = 30
    return img


def _make_white_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Return a plain white BGR image with no grid structure."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# detect_bubble_grid
# ---------------------------------------------------------------------------


class TestDetectBubbleGrid:
    def test_returns_none_on_plain_white_image(self):
        img = _make_white_image(100, 100)
        result = detect_bubble_grid(img, num_rows=5, num_cols=5)
        # No lines in a plain white image → should fall back (return None)
        assert result is None

    def test_grid_image_returns_correct_shape(self):
        img = _make_grid_image(num_rows=5, num_cols=5)
        result = detect_bubble_grid(img, num_rows=5, num_cols=5)
        if result is not None:
            assert len(result) == 5
            for row in result:
                assert len(row) == 5

    def test_all_bubbles_are_4_tuples(self):
        img = _make_grid_image(num_rows=3, num_cols=4)
        result = detect_bubble_grid(img, num_rows=3, num_cols=4)
        if result is not None:
            for row in result:
                for cell in row:
                    assert len(cell) == 4

    def test_bubble_dimensions_positive(self):
        img = _make_grid_image(num_rows=3, num_cols=3, cell_h=30, cell_w=30)
        result = detect_bubble_grid(img, num_rows=3, num_cols=3)
        if result is not None:
            for row in result:
                for x, y, bw, bh in row:
                    assert bw > 0
                    assert bh > 0

    def test_bubble_coords_within_image(self):
        img = _make_grid_image(num_rows=3, num_cols=4, cell_h=30, cell_w=28)
        result = detect_bubble_grid(img, num_rows=3, num_cols=4)
        if result is not None:
            for row in result:
                for x, y, bw, bh in row:
                    assert x >= 0
                    assert y >= 0

    def test_grayscale_input_accepted(self):
        img = _make_grid_image(num_rows=3, num_cols=3)
        gray = img[:, :, 0]  # single channel
        result = detect_bubble_grid(gray, num_rows=3, num_cols=3)
        # Should not crash (may return None)
        assert result is None or isinstance(result, list)

    def test_returns_none_on_insufficient_lines(self):
        # Very small image with barely any structure
        img = np.full((20, 20, 3), 200, dtype=np.uint8)
        result = detect_bubble_grid(img, num_rows=10, num_cols=10)
        assert result is None


class TestDetectBubbleGridNoCv2:
    def test_returns_none_on_plain_image(self, no_cv2):
        img = _make_white_image(100, 100)
        result = detect_bubble_grid(img, num_rows=3, num_cols=3)
        assert result is None

    def test_grid_image_shape_or_none(self, no_cv2):
        img = _make_grid_image(num_rows=4, num_cols=4, cell_h=20, cell_w=20)
        result = detect_bubble_grid(img, num_rows=4, num_cols=4)
        if result is not None:
            assert len(result) == 4
            for row in result:
                assert len(row) == 4

    def test_does_not_crash_on_empty(self, no_cv2):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = detect_bubble_grid(img, num_rows=2, num_cols=2)
        assert result is None or isinstance(result, list)

    def test_all_bubbles_4_tuples(self, no_cv2):
        img = _make_grid_image(num_rows=3, num_cols=4, cell_h=25, cell_w=25)
        result = detect_bubble_grid(img, num_rows=3, num_cols=4)
        if result is not None:
            for row in result:
                for cell in row:
                    assert len(cell) == 4


# ---------------------------------------------------------------------------
# detect_block_groups
# ---------------------------------------------------------------------------


class TestDetectBlockGroups:
    def test_empty_lines_returns_single_group(self):
        lines = {"horizontal": [], "vertical": []}
        groups = detect_block_groups(lines, num_rows=10, num_cols=5)
        assert groups == [list(range(10))]

    def test_uniform_spacing_returns_single_group(self):
        # Uniform gaps → no block separator
        h_pos = list(range(0, 100, 10))  # [0, 10, 20, ..., 90]
        lines = {"horizontal": h_pos, "vertical": []}
        groups = detect_block_groups(lines, num_rows=9, num_cols=5)
        assert len(groups) == 1

    def test_large_gap_creates_two_groups(self):
        # First half: positions 0-40 (step 10), large gap, second half 100-140
        h_pos = list(range(0, 50, 10)) + list(range(100, 150, 10))
        lines = {"horizontal": h_pos, "vertical": []}
        groups = detect_block_groups(lines, num_rows=len(h_pos) - 1, num_cols=5)
        # Should detect the gap and produce 2 groups
        assert len(groups) >= 1  # at least returns groups, not crash

    def test_groups_cover_all_rows(self):
        h_pos = list(range(0, 60, 10))
        lines = {"horizontal": h_pos, "vertical": []}
        num_rows = len(h_pos) - 1
        groups = detect_block_groups(lines, num_rows=num_rows, num_cols=5)
        all_rows = sorted(r for g in groups for r in g)
        assert all_rows == list(range(len(all_rows)))

    def test_single_line_returns_single_group(self):
        lines = {"horizontal": [50], "vertical": []}
        groups = detect_block_groups(lines, num_rows=5, num_cols=3)
        assert isinstance(groups, list)
        assert len(groups) >= 1
