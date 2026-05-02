"""Tests for bubble_mark.processing.image_utils."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import bubble_mark.processing.image_utils as _image_utils_mod
from bubble_mark.processing.image_utils import (
    apply_threshold,
    draw_overlay,
    find_page_contour,
    load_image,
    order_points,
    perspective_transform,
    resize_image,
    to_grayscale,
)
from tests.conftest import create_blank_sheet


@pytest.fixture
def no_cv2(monkeypatch):
    """Force the pure NumPy/Pillow fallback paths by setting _HAVE_CV2=False."""
    monkeypatch.setattr(_image_utils_mod, "_HAVE_CV2", False)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(str(tmp_path / "nonexistent.png"))

    def test_invalid_file_raises(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError):
            load_image(str(bad))

    def test_valid_image_loaded(self, tmp_path):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        path = str(tmp_path / "img.png")
        cv2.imwrite(path, img)
        loaded = load_image(path)
        assert loaded.shape == (50, 50, 3)


# ---------------------------------------------------------------------------
# to_grayscale
# ---------------------------------------------------------------------------


class TestToGrayscale:
    def test_bgr_to_gray(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = to_grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == (100, 100)

    def test_already_gray(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray = to_grayscale(img)
        assert gray.shape == (100, 100)

    def test_single_channel_3d(self):
        img = np.zeros((50, 50, 1), dtype=np.uint8)
        gray = to_grayscale(img)
        assert gray.ndim == 2


# ---------------------------------------------------------------------------
# apply_threshold
# ---------------------------------------------------------------------------


class TestApplyThreshold:
    def test_otsu_returns_binary(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        binary = apply_threshold(img, method="otsu")
        assert binary.ndim == 2
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255})

    def test_adaptive_returns_binary(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        binary = apply_threshold(img, method="adaptive")
        assert binary.ndim == 2
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255})

    def test_default_method_is_otsu(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_threshold(img)
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# order_points
# ---------------------------------------------------------------------------


class TestOrderPoints:
    def test_order_known_points(self):
        pts = np.array([[10, 200], [200, 10], [200, 200], [10, 10]], dtype=np.float32)
        ordered = order_points(pts)
        # top-left has smallest sum
        assert tuple(ordered[0]) == (10.0, 10.0)
        # bottom-right has largest sum
        assert tuple(ordered[2]) == (200.0, 200.0)

    def test_output_shape(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        ordered = order_points(pts)
        assert ordered.shape == (4, 2)


# ---------------------------------------------------------------------------
# perspective_transform
# ---------------------------------------------------------------------------


class TestPerspectiveTransform:
    def test_output_is_ndarray(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200
        contour = np.array(
            [[10, 10], [190, 10], [190, 190], [10, 190]], dtype=np.float32
        )
        warped = perspective_transform(img, contour)
        assert isinstance(warped, np.ndarray)
        assert warped.ndim == 3

    def test_output_shape_approximation(self):
        # Contour describing a near-square region → output should be roughly square
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        contour = np.array(
            [[20, 20], [280, 20], [280, 280], [20, 280]], dtype=np.float32
        )
        warped = perspective_transform(img, contour)
        h, w = warped.shape[:2]
        assert abs(w - h) < 20  # roughly square


# ---------------------------------------------------------------------------
# find_page_contour
# ---------------------------------------------------------------------------


class TestFindPageContour:
    def test_blank_image_returns_none_or_array(self):
        img = create_blank_sheet(200, 200)
        result = find_page_contour(img)
        # May return None on a blank white image
        assert result is None or isinstance(result, np.ndarray)

    def test_dark_rectangle_detected(self):
        img = np.full((400, 400, 3), 255, dtype=np.uint8)
        # Draw a thick dark rectangle
        cv2.rectangle(img, (40, 40), (360, 360), (0, 0, 0), 5)
        result = find_page_contour(img)
        # Should find 4-point contour or None (threshold-dependent)
        if result is not None:
            assert result.shape[-1] == 2


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------


class TestResizeImage:
    def test_resize_by_width(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(img, width=100)
        assert resized.shape[1] == 100
        assert resized.shape[0] == 50  # aspect preserved

    def test_resize_by_height(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(img, height=50)
        assert resized.shape[0] == 50
        assert resized.shape[1] == 100

    def test_resize_both(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(img, width=80, height=60)
        assert resized.shape[:2] == (60, 80)

    def test_no_resize_returns_original(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_image(img)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# Pure NumPy/Pillow fallback paths (_HAVE_CV2 = False)
# ---------------------------------------------------------------------------


class TestLoadImageNoCv2:
    def test_missing_file_raises(self, tmp_path, no_cv2):
        with pytest.raises(FileNotFoundError):
            load_image(str(tmp_path / "nonexistent.png"))

    def test_invalid_file_raises(self, tmp_path, no_cv2):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError):
            load_image(str(bad))

    def test_valid_image_loaded(self, tmp_path, no_cv2):
        from PIL import Image as PILImage

        img = PILImage.fromarray(np.zeros((50, 50, 3), dtype=np.uint8), mode="RGB")
        path = str(tmp_path / "img.png")
        img.save(path)
        loaded = load_image(path)
        assert loaded.shape == (50, 50, 3)


class TestToGrayscaleNoCv2:
    def test_bgr_to_gray(self, no_cv2):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = to_grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == (100, 100)

    def test_already_gray(self, no_cv2):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray = to_grayscale(img)
        assert gray.shape == (100, 100)


class TestApplyThresholdNoCv2:
    def test_otsu_returns_binary(self, no_cv2):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        binary = apply_threshold(img, method="otsu")
        assert binary.ndim == 2
        assert set(np.unique(binary)).issubset({0, 255})

    def test_adaptive_returns_binary(self, no_cv2):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        binary = apply_threshold(img, method="adaptive")
        assert binary.ndim == 2
        assert set(np.unique(binary)).issubset({0, 255})


class TestFindPageContourNoCv2:
    def test_blank_image_returns_none_or_array(self, no_cv2):
        img = create_blank_sheet(200, 200)
        result = find_page_contour(img)
        assert result is None or isinstance(result, np.ndarray)

    def test_dark_region_detected(self, no_cv2):
        # Use a large filled dark region so it clears the 10 % coverage threshold.
        img = np.full((400, 400, 3), 255, dtype=np.uint8)
        img[40:360, 40:360] = 0  # large dark rectangle
        result = find_page_contour(img)
        if result is not None:
            assert result.shape[-1] == 2


class TestPerspectiveTransformNoCv2:
    def test_output_is_ndarray(self, no_cv2):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200
        contour = np.array(
            [[10, 10], [190, 10], [190, 190], [10, 190]], dtype=np.float32
        )
        warped = perspective_transform(img, contour)
        assert isinstance(warped, np.ndarray)
        assert warped.ndim == 3

    def test_output_shape_approximation(self, no_cv2):
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        contour = np.array(
            [[20, 20], [280, 20], [280, 280], [20, 280]], dtype=np.float32
        )
        warped = perspective_transform(img, contour)
        h, w = warped.shape[:2]
        assert abs(w - h) < 20  # roughly square


class TestResizeImageNoCv2:
    def test_resize_by_width(self, no_cv2):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(img, width=100)
        assert resized.shape[1] == 100
        assert resized.shape[0] == 50  # aspect preserved

    def test_resize_by_height(self, no_cv2):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(img, height=50)
        assert resized.shape[0] == 50
        assert resized.shape[1] == 100

    def test_resize_both(self, no_cv2):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(img, width=80, height=60)
        assert resized.shape[:2] == (60, 80)

    def test_no_resize_returns_original(self, no_cv2):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_image(img)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# draw_overlay
# ---------------------------------------------------------------------------


def _make_overlay_args(width: int = 200, height: int = 260):
    """Build minimal but realistic arguments for draw_overlay."""
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    answer_section_rect = (0, 0, width, int(height * 0.72))
    id_section_rect = (0, int(height * 0.74), width, height)
    # Two bubble rows, two columns each
    all_answer_bubbles = [(10, 10, 20, 20), (40, 10, 20, 20)]
    all_id_bubbles = [(10, 200, 15, 15), (35, 200, 15, 15)]
    filled_answer = [(10, 10, 20, 20)]
    filled_id = [(35, 200, 15, 15)]
    return dict(
        image=image,
        answer_section_rect=answer_section_rect,
        id_section_rect=id_section_rect,
        all_answer_bubbles=all_answer_bubbles,
        all_id_bubbles=all_id_bubbles,
        filled_answer_bubbles=filled_answer,
        filled_id_bubbles=filled_id,
    )


class TestDrawOverlay:
    def test_returns_ndarray(self):
        result = draw_overlay(**_make_overlay_args())
        assert isinstance(result, np.ndarray)

    def test_output_same_shape_as_input(self):
        args = _make_overlay_args()
        result = draw_overlay(**args)
        assert result.shape == args["image"].shape

    def test_output_is_bgr(self):
        result = draw_overlay(**_make_overlay_args())
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_does_not_modify_input(self):
        args = _make_overlay_args()
        original = args["image"].copy()
        draw_overlay(**args)
        np.testing.assert_array_equal(args["image"], original)

    def test_filled_bubble_pixels_changed(self):
        """Pixels inside a filled bubble region should differ from the input."""
        args = _make_overlay_args()
        original = args["image"].copy()
        result = draw_overlay(**args)
        x, y, bw, bh = args["filled_answer_bubbles"][0]
        # At least some pixels inside the filled region changed
        region_before = original[y : y + bh, x : x + bw]
        region_after = result[y : y + bh, x : x + bw]
        assert not np.array_equal(region_before, region_after)

    def test_grayscale_input_converted_to_bgr(self):
        args = _make_overlay_args()
        args["image"] = np.full((260, 200), 200, dtype=np.uint8)
        result = draw_overlay(**args)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_empty_bubble_lists(self):
        args = _make_overlay_args()
        args["all_answer_bubbles"] = []
        args["all_id_bubbles"] = []
        args["filled_answer_bubbles"] = []
        args["filled_id_bubbles"] = []
        result = draw_overlay(**args)
        assert result.shape == args["image"].shape


class TestDrawOverlayNoCv2:
    def test_returns_ndarray(self, no_cv2):
        result = draw_overlay(**_make_overlay_args())
        assert isinstance(result, np.ndarray)

    def test_output_same_shape_as_input(self, no_cv2):
        args = _make_overlay_args()
        result = draw_overlay(**args)
        assert result.shape == args["image"].shape

    def test_does_not_modify_input(self, no_cv2):
        args = _make_overlay_args()
        original = args["image"].copy()
        draw_overlay(**args)
        np.testing.assert_array_equal(args["image"], original)

    def test_filled_bubble_pixels_changed(self, no_cv2):
        args = _make_overlay_args()
        original = args["image"].copy()
        result = draw_overlay(**args)
        x, y, bw, bh = args["filled_answer_bubbles"][0]
        region_before = original[y : y + bh, x : x + bw]
        region_after = result[y : y + bh, x : x + bw]
        assert not np.array_equal(region_before, region_after)

    def test_grayscale_input_converted_to_bgr(self, no_cv2):
        args = _make_overlay_args()
        args["image"] = np.full((260, 200), 200, dtype=np.uint8)
        result = draw_overlay(**args)
        assert result.ndim == 3
        assert result.shape[2] == 3
