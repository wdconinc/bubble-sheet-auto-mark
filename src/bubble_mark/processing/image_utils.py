"""Image pre-processing utilities for bubble sheet detection."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from *path* and return it as a BGR NumPy array.

    Raises FileNotFoundError if the path does not exist and ValueError if
    OpenCV cannot decode the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert *image* to grayscale.

    Accepts BGR (3-channel) or already-grayscale (1/2-channel) arrays.
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image if image.ndim == 2 else image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_threshold(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Binarize *image* using the chosen thresholding *method*.

    Supported methods: ``"otsu"`` (default) and ``"adaptive"``.
    """
    gray = to_grayscale(image)
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    # Default: Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def find_page_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest roughly-rectangular contour in *image*.

    Returns a (4, 1, 2) contour array suitable for ``perspective_transform``,
    or *None* if no suitable contour is found.
    """
    binary = apply_threshold(image, method="otsu")
    # Dilate slightly to close gaps on page border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = image.shape[0] * image.shape[1]

    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.1:
            break
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx

    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left].

    *pts* must be shaped (4, 2).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # top-left: smallest sum
    ordered[2] = pts[np.argmax(s)]   # bottom-right: largest sum

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right: smallest diff (x-y)
    ordered[3] = pts[np.argmax(diff)]  # bottom-left: largest diff

    return ordered


def perspective_transform(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Apply a 4-point perspective warp to normalise the bubble sheet.

    *contour* should be the output of ``find_page_contour`` (shape (4, 1, 2)
    or (4, 2)).
    """
    pts = contour.reshape(4, 2).astype(np.float32)
    ordered = order_points(pts)

    tl, tr, br, bl = ordered
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (width, height))


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """Resize *image* while preserving aspect ratio.

    Provide exactly one of *width* or *height*; if both are given both are
    used directly (aspect ratio may change).  Returns unchanged image if
    neither is given.
    """
    if width is None and height is None:
        return image

    h, w = image.shape[:2]
    if width is not None and height is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if width is not None:
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)

    ratio = height / h
    new_w = int(w * ratio)
    return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)
