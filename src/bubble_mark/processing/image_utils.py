"""Image pre-processing utilities for bubble sheet detection.

All public functions work with or without OpenCV.  When ``cv2`` is available
(desktop / CI) it is used as-is.  When it is absent (Android with Chaquopy,
which does not yet ship a cp312 OpenCV wheel) every function falls back to an
equivalent implementation built on NumPy and Pillow, both of which are always
available.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Pure-NumPy / Pillow helpers (used only when cv2 is unavailable)
# ---------------------------------------------------------------------------

def _otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu's optimal binarization threshold using only NumPy."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = int(gray.size)
    sum_all = float(np.dot(np.arange(256, dtype=np.float64), hist))
    sum_b = 0.0
    w_b = 0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * int(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return threshold


def _dilate_pure(binary: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Morphological binary dilation using PIL MaxFilter."""
    from PIL import Image, ImageFilter
    pil = Image.fromarray(binary)
    return np.array(pil.filter(ImageFilter.MaxFilter(kernel_size)))


def _perspective_coeffs(src_pts: np.ndarray, dst_pts: np.ndarray) -> list:
    """Compute the 8 PIL PERSPECTIVE coefficients that map dst→src.

    PIL's transform(PERSPECTIVE) evaluates:
        x = (a*X + b*Y + c) / (g*X + h*Y + 1)
        y = (d*X + e*Y + f) / (g*X + h*Y + 1)
    where (X, Y) are output-image coordinates and (x, y) are source
    coordinates.  We solve the 8×8 linear system given 4 point pairs.
    """
    A = []
    b_vec: list[float] = []
    for (xs, ys), (xd, yd) in zip(src_pts, dst_pts):
        A.append([xd, yd, 1.0, 0.0, 0.0, 0.0, -xs * xd, -xs * yd])
        A.append([0.0, 0.0, 0.0, xd, yd, 1.0, -ys * xd, -ys * yd])
        b_vec.append(float(xs))
        b_vec.append(float(ys))
    coeffs, _, _, _ = np.linalg.lstsq(
        np.array(A, dtype=np.float64), np.array(b_vec, dtype=np.float64), rcond=None
    )
    return coeffs.tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image from *path* and return it as a BGR NumPy array.

    Raises FileNotFoundError if the path does not exist and ValueError if
    the file cannot be decoded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    if _HAVE_CV2:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        return image
    from PIL import Image as PILImage, UnidentifiedImageError
    try:
        with PILImage.open(path) as pil_raw:
            pil = pil_raw.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {path}") from exc
    # Convert RGB → BGR to match cv2 convention.
    return np.array(pil)[:, :, ::-1].copy()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert *image* to grayscale.

    Accepts BGR (3-channel) or already-grayscale (1/2-channel) arrays.
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image if image.ndim == 2 else image[:, :, 0]
    if _HAVE_CV2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Pure NumPy: ITU-R 601 coefficients for BGR channel order.
    return (
        0.114 * image[:, :, 0].astype(np.float32)
        + 0.587 * image[:, :, 1].astype(np.float32)
        + 0.299 * image[:, :, 2].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)


def apply_threshold(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Binarize *image* using the chosen thresholding *method*.

    Supported methods: ``"otsu"`` (default) and ``"adaptive"``.
    """
    gray = to_grayscale(image)
    if _HAVE_CV2:
        if method == "adaptive":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary
    # Pure fallback -------------------------------------------------------
    if method == "adaptive":
        from PIL import Image, ImageFilter
        blurred = np.array(
            Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=5))
        )
        # Pixels darker than the local mean by more than 2 → foreground.
        return np.where(
            gray.astype(np.int16) - blurred.astype(np.int16) < -2,
            np.uint8(255), np.uint8(0)
        )
    t = _otsu_threshold(gray)
    return np.where(gray < t, np.uint8(255), np.uint8(0))


def find_page_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest roughly-rectangular contour in *image*.

    Returns a (4, 1, 2) contour array suitable for ``perspective_transform``,
    or *None* if no suitable contour is found.
    """
    if _HAVE_CV2:
        binary = apply_threshold(image, method="otsu")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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

    # Pure-NumPy fallback -------------------------------------------------
    # Strategy: threshold + dilate, then find the 4 extreme corner pixels of
    # the white region.  For a page that fills most of the frame this reliably
    # recovers the 4 corners even without a connected-components library.
    binary = apply_threshold(image, method="otsu")
    dilated = _dilate_pure(binary, kernel_size=5)

    ys, xs = np.where(dilated > 0)
    if len(xs) == 0:
        return None

    # Require the white region to cover at least 10 % of the image.
    if len(xs) < image.shape[0] * image.shape[1] * 0.1:
        return None

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    s = coords[:, 0] + coords[:, 1]   # x + y
    d = coords[:, 0] - coords[:, 1]   # x - y

    tl = coords[np.argmin(s)]   # top-left:     min(x+y)
    br = coords[np.argmax(s)]   # bottom-right: max(x+y)
    tr = coords[np.argmax(d)]   # top-right:    max(x-y)
    bl = coords[np.argmin(d)]   # bottom-left:  min(x-y)

    return np.array([tl, tr, br, bl], dtype=np.float32).reshape(4, 1, 2)


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

    if _HAVE_CV2:
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image, M, (width, height))

    # Pure Pillow fallback ------------------------------------------------
    from PIL import Image as PILImage
    coeffs = _perspective_coeffs(ordered, dst)
    is_color = image.ndim == 3 and image.shape[2] >= 3
    if is_color:
        pil = PILImage.fromarray(image[:, :, ::-1].astype(np.uint8))  # BGR → RGB
        result = pil.transform((width, height), PILImage.PERSPECTIVE, coeffs, PILImage.BICUBIC)
        return np.array(result)[:, :, ::-1].copy()  # RGB → BGR
    pil = PILImage.fromarray(image.astype(np.uint8))
    result = pil.transform((width, height), PILImage.PERSPECTIVE, coeffs, PILImage.BICUBIC)
    return np.array(result)


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

    if _HAVE_CV2:
        if width is not None and height is not None:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if width is not None:
            return cv2.resize(image, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        return cv2.resize(image, (int(w * height / h), height), interpolation=cv2.INTER_AREA)

    # Pure Pillow fallback ------------------------------------------------
    from PIL import Image as PILImage
    if width is not None and height is not None:
        new_size = (width, height)
    elif width is not None:
        new_size = (width, int(h * width / w))
    else:
        new_size = (int(w * height / h), height)  # type: ignore[assignment]

    is_color = image.ndim == 3 and image.shape[2] >= 3
    if is_color:
        pil = PILImage.fromarray(image[:, :, ::-1].astype(np.uint8))  # BGR → RGB
        resized = pil.resize(new_size, PILImage.LANCZOS)
        return np.array(resized)[:, :, ::-1].copy()  # RGB → BGR
    pil = PILImage.fromarray(image.astype(np.uint8))
    return np.array(pil.resize(new_size, PILImage.LANCZOS))
