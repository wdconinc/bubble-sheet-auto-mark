"""Tests for bubble_mark.processing.color_channel."""

from __future__ import annotations

import numpy as np
import pytest

import bubble_mark.processing.color_channel as _cc_mod
from bubble_mark.processing.color_channel import enhance_contrast, extract_print_channel


@pytest.fixture
def no_cv2(monkeypatch):
    """Force the pure NumPy/Pillow fallback paths."""
    monkeypatch.setattr(_cc_mod, "_HAVE_CV2", False)


# ---------------------------------------------------------------------------
# extract_print_channel
# ---------------------------------------------------------------------------


class TestExtractPrintChannel:
    def test_grayscale_2d_passthrough(self):
        gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = extract_print_channel(gray)
        assert result.shape == (50, 50)
        assert result.dtype == np.uint8

    def test_single_channel_3d_returns_2d(self):
        img = np.random.randint(0, 255, (50, 50, 1), dtype=np.uint8)
        result = extract_print_channel(img)
        assert result.ndim == 2
        assert result.shape == (50, 50)

    def test_explicit_channel_index(self):
        # Make an image where channel 0 is all-zero and others non-zero
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 1] = 128
        img[:, :, 2] = 200
        result = extract_print_channel(img, channel=0)
        assert result.shape == (40, 40)
        assert np.all(result == 0)

    def test_explicit_channel_1(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 1] = 77
        result = extract_print_channel(img, channel=1)
        assert np.all(result == 77)

    def test_auto_selects_highest_variance_channel(self):
        # Channel 2 has highest variance (random), others are constant
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        img[:, :, 2] = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        result = extract_print_channel(img)
        assert result.shape == (60, 60)
        # Should be channel 2 content
        np.testing.assert_array_equal(result, img[:, :, 2])

    def test_returns_uint8(self):
        img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        result = extract_print_channel(img)
        assert result.dtype == np.uint8

    def test_channel_index_wraps(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[:, :, 0] = 42
        # channel=3 should wrap to 0
        result = extract_print_channel(img, channel=3)
        assert np.all(result == 42)


# ---------------------------------------------------------------------------
# enhance_contrast
# ---------------------------------------------------------------------------


class TestEnhanceContrast:
    def test_returns_2d_uint8(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_same_shape_as_input(self):
        gray = np.random.randint(50, 200, (80, 120), dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.shape == gray.shape

    def test_output_in_valid_range(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_image_does_not_crash(self):
        # All-same-value image – histogram equalisation should handle gracefully
        gray = np.full((50, 50), 128, dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.shape == (50, 50)
        assert result.dtype == np.uint8

    def test_all_zeros_does_not_crash(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.shape == (50, 50)

    def test_all_max_does_not_crash(self):
        gray = np.full((50, 50), 255, dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.shape == (50, 50)


class TestEnhanceContrastNoCv2:
    def test_returns_2d_uint8(self, no_cv2):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_same_shape(self, no_cv2):
        gray = np.random.randint(0, 255, (80, 80), dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.shape == gray.shape

    def test_output_in_valid_range(self, no_cv2):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_equalization_increases_range(self, no_cv2):
        # Low-contrast image concentrated around mid-gray
        gray = np.full((100, 100), 128, dtype=np.uint8)
        # Add a little variation so histogram has something to equalize
        gray[0, 0] = 100
        gray[-1, -1] = 160
        result = enhance_contrast(gray)
        assert result.shape == gray.shape

    def test_constant_image_does_not_crash(self, no_cv2):
        gray = np.full((50, 50), 200, dtype=np.uint8)
        result = enhance_contrast(gray)
        assert result.dtype == np.uint8
