"""Tests for bubble_mark.processing.distortion."""

from __future__ import annotations

import numpy as np
import pytest

import bubble_mark.processing.distortion as _dist_mod
from bubble_mark.processing.distortion import (
    apply_homography,
    correct_distortion_from_lines,
    estimate_distortion_from_reference,
    find_intersection,
)


@pytest.fixture
def no_cv2(monkeypatch):
    """Force the pure NumPy/Pillow fallback paths."""
    monkeypatch.setattr(_dist_mod, "_HAVE_CV2", False)


# ---------------------------------------------------------------------------
# find_intersection
# ---------------------------------------------------------------------------


class TestFindIntersection:
    def test_known_intersection(self):
        # Two lines crossing at (5, 5)
        line1 = [0, 0, 10, 10]
        line2 = [0, 10, 10, 0]
        pt = find_intersection(line1, line2)
        assert pt is not None
        x, y = pt
        assert abs(x - 5.0) < 1e-6
        assert abs(y - 5.0) < 1e-6

    def test_horizontal_and_vertical(self):
        h_line = [0, 50, 200, 50]
        v_line = [100, 0, 100, 200]
        pt = find_intersection(h_line, v_line)
        assert pt is not None
        x, y = pt
        assert abs(x - 100.0) < 1e-6
        assert abs(y - 50.0) < 1e-6

    def test_parallel_lines_return_none(self):
        line1 = [0, 0, 100, 0]  # horizontal y=0
        line2 = [0, 10, 100, 10]  # horizontal y=10
        assert find_intersection(line1, line2) is None

    def test_coincident_lines_return_none(self):
        line1 = [0, 0, 100, 0]
        line2 = [10, 0, 90, 0]
        assert find_intersection(line1, line2) is None

    def test_returns_floats(self):
        line1 = [0, 0, 4, 4]
        line2 = [0, 4, 4, 0]
        pt = find_intersection(line1, line2)
        assert pt is not None
        assert isinstance(pt[0], float)
        assert isinstance(pt[1], float)

    def test_non_axis_aligned(self):
        # Line from (0, 0) to (4, 8) and from (0, 8) to (4, 0)
        line1 = [0, 0, 4, 8]
        line2 = [0, 8, 4, 0]
        pt = find_intersection(line1, line2)
        assert pt is not None
        x, y = pt
        assert abs(x - 2.0) < 1e-5
        assert abs(y - 4.0) < 1e-5


# ---------------------------------------------------------------------------
# correct_distortion_from_lines
# ---------------------------------------------------------------------------


class TestCorrectDistortionFromLines:
    def _make_rect_lines(self, x1=50, y1=50, x2=350, y2=450):
        """Four lines forming a rectangle."""
        return [
            [x1, y1, x2, y1],  # top
            [x1, y2, x2, y2],  # bottom
            [x1, y1, x1, y2],  # left
            [x2, y1, x2, y2],  # right
        ]

    def test_returns_ndarray(self):
        img = np.full((500, 400, 3), 200, dtype=np.uint8)
        lines = self._make_rect_lines()
        result = correct_distortion_from_lines(img, lines)
        assert result is None or isinstance(result, np.ndarray)

    def test_rectangular_lines_return_image(self):
        img = np.full((500, 400, 3), 200, dtype=np.uint8)
        lines = self._make_rect_lines()
        result = correct_distortion_from_lines(img, lines)
        # Should succeed and return an image (the corners are well-defined)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_output_is_3d_bgr(self):
        img = np.full((500, 400, 3), 200, dtype=np.uint8)
        lines = self._make_rect_lines()
        result = correct_distortion_from_lines(img, lines)
        if result is not None:
            assert result.ndim == 3
            assert result.shape[2] == 3

    def test_wrong_number_of_lines_returns_none(self):
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        lines = [[0, 0, 100, 0], [0, 100, 100, 100]]  # only 2
        result = correct_distortion_from_lines(img, lines)
        assert result is None

    def test_parallel_lines_gracefully_handled(self):
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        # Two pairs of parallel lines – cannot form 4 corners
        lines = [
            [0, 0, 200, 0],
            [0, 200, 200, 200],
            [0, 0, 200, 0],   # duplicate top
            [0, 200, 200, 200],  # duplicate bottom
        ]
        # Should either return None or not crash
        try:
            result = correct_distortion_from_lines(img, lines)
            assert result is None or isinstance(result, np.ndarray)
        except Exception:
            pytest.fail("correct_distortion_from_lines raised unexpectedly")

    def test_no_cv2_fallback_works(self, no_cv2):
        img = np.full((500, 400, 3), 200, dtype=np.uint8)
        lines = self._make_rect_lines()
        result = correct_distortion_from_lines(img, lines)
        if result is not None:
            assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# estimate_distortion_from_reference
# ---------------------------------------------------------------------------


class TestEstimateDistortionFromReference:
    def test_identical_images_returns_identity_like(self):
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        H = estimate_distortion_from_reference(img, img.copy())
        assert H is not None
        assert H.shape == (3, 3)
        # Translation should be near zero
        assert abs(H[0, 2]) < 5
        assert abs(H[1, 2]) < 5

    def test_returns_3x3_matrix(self):
        img = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
        ref = img.copy()
        H = estimate_distortion_from_reference(img, ref)
        assert H is not None
        assert H.shape == (3, 3)
        assert H.dtype == np.float64

    def test_different_sizes_handled(self):
        sheet = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ref = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        H = estimate_distortion_from_reference(sheet, ref)
        # Should not crash; may return None or a matrix
        assert H is None or (isinstance(H, np.ndarray) and H.shape == (3, 3))

    def test_no_cv2_fallback_works(self, no_cv2):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        H = estimate_distortion_from_reference(img, img.copy())
        assert H is not None
        assert H.shape == (3, 3)

    def test_explicit_channel(self):
        img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        H = estimate_distortion_from_reference(img, img.copy(), channel=2)
        assert H is not None

    def test_grayscale_images(self):
        img = np.random.randint(0, 255, (80, 80), dtype=np.uint8)
        H = estimate_distortion_from_reference(img, img.copy())
        assert H is not None
        assert H.shape == (3, 3)


# ---------------------------------------------------------------------------
# apply_homography
# ---------------------------------------------------------------------------


class TestApplyHomography:
    def test_identity_returns_same_content(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        result = apply_homography(img, H)
        assert result.shape == img.shape
        # With identity transform the output should be very close to input
        diff = np.abs(result.astype(int) - img.astype(int))
        assert diff.mean() < 5  # allow tiny interpolation differences

    def test_output_shape_preserved(self):
        img = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        result = apply_homography(img, H)
        assert result.shape == img.shape

    def test_no_cv2_identity(self, no_cv2):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        H = np.eye(3, dtype=np.float64)
        result = apply_homography(img, H)
        assert result.shape == img.shape

    def test_no_cv2_translation(self, no_cv2):
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        img[10:20, 10:20] = 200
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = 5  # shift right by 5
        H[1, 2] = 3  # shift down by 3
        result = apply_homography(img, H)
        assert result.shape == img.shape
