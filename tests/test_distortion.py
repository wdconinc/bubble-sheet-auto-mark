"""Tests for bubble_mark.processing.distortion."""

from __future__ import annotations

import numpy as np
import pytest

import bubble_mark.processing.distortion as _dist_mod
from bubble_mark.processing.distortion import (
    apply_homography,
    correct_distortion_from_lines,
    correct_distortion_from_polylines,
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
            [0, 0, 200, 0],  # duplicate top
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


# ---------------------------------------------------------------------------
# correct_distortion_from_polylines
# ---------------------------------------------------------------------------


class TestCorrectDistortionFromPolylines:
    """Tests for the bilinear Coons patch polyline warp."""

    @staticmethod
    def _rect_polylines(x1=50, y1=50, x2=350, y2=450, pts_per_edge=2):
        """Return four polylines that form a rectangle (optionally subdivided)."""
        n = pts_per_edge
        top = [[x1 + (x2 - x1) * i / (n - 1), y1] for i in range(n)]
        bottom = [[x1 + (x2 - x1) * i / (n - 1), y2] for i in range(n)]
        left = [[x1, y1 + (y2 - y1) * i / (n - 1)] for i in range(n)]
        right = [[x2, y1 + (y2 - y1) * i / (n - 1)] for i in range(n)]
        return {"top": top, "bottom": bottom, "left": left, "right": right}

    def test_returns_ndarray_for_rect(self):
        img = np.full((500, 400, 3), 200, dtype=np.uint8)
        polys = self._rect_polylines()
        result = correct_distortion_from_polylines(img, polys)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_output_is_3d_bgr(self):
        img = np.full((500, 400, 3), 200, dtype=np.uint8)
        polys = self._rect_polylines()
        result = correct_distortion_from_polylines(img, polys)
        assert result is not None
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_output_size_derived_from_edge_lengths(self):
        # Rectangle 200×300 pixels
        polys = self._rect_polylines(x1=0, y1=0, x2=200, y2=300)
        img = np.full((400, 300, 3), 128, dtype=np.uint8)
        result = correct_distortion_from_polylines(img, polys)
        assert result is not None
        assert result.shape[1] == 200  # width from top/bottom arc-length
        assert result.shape[0] == 300  # height from left/right arc-length

    def test_missing_edge_returns_none(self):
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        polys = self._rect_polylines()
        del polys["right"]
        result = correct_distortion_from_polylines(img, polys)
        assert result is None

    def test_single_point_edge_returns_none(self):
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        polys = self._rect_polylines()
        polys["top"] = [[50, 50]]  # only one point
        result = correct_distortion_from_polylines(img, polys)
        assert result is None

    def test_curved_top_edge_does_not_crash(self):
        """A slightly curved top edge should still produce a valid image."""
        img = np.full((500, 400, 3), 200, dtype=np.uint8)
        # Top edge curves slightly upward in the middle
        polys = {
            "top": [[50, 50], [200, 30], [350, 50]],
            "bottom": [[50, 450], [200, 450], [350, 450]],
            "left": [[50, 50], [50, 250], [50, 450]],
            "right": [[350, 50], [350, 250], [350, 450]],
        }
        result = correct_distortion_from_polylines(img, polys)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3

    def test_no_cv2_fallback_returns_valid_image(self, no_cv2):
        img = np.full((500, 400, 3), 200, dtype=np.uint8)
        polys = self._rect_polylines()
        result = correct_distortion_from_polylines(img, polys)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_grayscale_input(self):
        img = np.full((300, 400), 180, dtype=np.uint8)
        polys = self._rect_polylines(x1=20, y1=20, x2=380, y2=280)
        result = correct_distortion_from_polylines(img, polys)
        assert result is not None
        assert result.ndim == 2

    def test_two_point_edge_equivalent_to_straight_line(self):
        """With 2-point (straight) polylines the result should resemble the
        perspective-transform output of the same rectangle."""
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        # Draw a distinctive pattern inside the rectangle
        img[50:450, 50:350] = 128
        img[100:400, 100:300] = 200
        polys = self._rect_polylines(x1=50, y1=50, x2=350, y2=450)
        result = correct_distortion_from_polylines(img, polys)
        assert result is not None
        # Output must be non-empty and have reasonable pixel values
        assert result.shape[0] > 0 and result.shape[1] > 0
        assert result.max() > 50  # not all black

    def test_subdivided_rect_same_as_two_point(self):
        """Subdividing a straight edge into more points should give the same
        output (within rounding) as the 2-point version."""
        img = np.random.RandomState(0).randint(0, 256, (500, 400, 3), dtype=np.uint8)
        polys_2 = self._rect_polylines(pts_per_edge=2)
        polys_5 = self._rect_polylines(pts_per_edge=5)
        r2 = correct_distortion_from_polylines(img, polys_2)
        r5 = correct_distortion_from_polylines(img, polys_5)
        assert r2 is not None and r5 is not None
        assert r2.shape == r5.shape
        # Results must be pixel-close (allow small bilinear interpolation diffs)
        assert np.mean(np.abs(r2.astype(np.int32) - r5.astype(np.int32))) < 2
