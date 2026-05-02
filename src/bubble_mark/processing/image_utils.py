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
    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError

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
        (
            0.114 * image[:, :, 0].astype(np.float32)
            + 0.587 * image[:, :, 1].astype(np.float32)
            + 0.299 * image[:, :, 2].astype(np.float32)
        )
        .clip(0, 255)
        .astype(np.uint8)
    )


def apply_threshold(
    image: np.ndarray, method: str = "otsu", invert: bool = True
) -> np.ndarray:
    """Binarize *image* using the chosen thresholding *method*.

    Supported methods: ``"otsu"`` (default) and ``"adaptive"``.

    Parameters
    ----------
    image:
        Input image (BGR or grayscale).
    method:
        Thresholding method: ``"otsu"`` (default) or ``"adaptive"``.
    invert:
        When ``True`` (default) dark pixels become 255 and bright pixels
        become 0 — suitable for detecting dark objects (e.g. filled bubbles)
        on a bright background.  When ``False`` the polarity is flipped:
        bright pixels become 255 and dark pixels become 0 — suitable for
        detecting a bright object (e.g. a white page) on a dark background.
    """
    gray = to_grayscale(image)
    if _HAVE_CV2:
        if method == "adaptive":
            thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 11, 2
            )
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, thresh_type | cv2.THRESH_OTSU)
        return binary
    # Pure fallback -------------------------------------------------------
    if method == "adaptive":
        from PIL import Image, ImageFilter

        blurred = np.array(
            Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=5))
        )
        diff = gray.astype(np.int16) - blurred.astype(np.int16)
        # Match cv2 THRESH_BINARY_INV / THRESH_BINARY with C=2:
        #   THRESH_BINARY_INV (invert=True):  foreground when gray <= mean - C
        #                                     ↔ diff < -C  (i.e. diff < -2)
        #   THRESH_BINARY     (invert=False): foreground when gray >  mean - C
        #                                     ↔ diff > -C  (i.e. diff > -2)
        condition = diff < -2 if invert else diff > -2
        return np.where(condition, np.uint8(255), np.uint8(0))
    t = _otsu_threshold(gray)
    # Match cv2 convention: THRESH_BINARY_INV → foreground when gray <= thresh;
    # THRESH_BINARY → foreground when gray > thresh.
    condition = gray <= t if invert else gray > t
    return np.where(condition, np.uint8(255), np.uint8(0))


def find_page_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest roughly-rectangular contour in *image*.

    Returns a (4, 1, 2) contour array suitable for ``perspective_transform``,
    or *None* if no suitable contour is found.
    """
    if _HAVE_CV2:
        binary = apply_threshold(image, method="otsu", invert=False)
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
    binary = apply_threshold(image, method="otsu", invert=False)
    dilated = _dilate_pure(binary, kernel_size=5)

    ys, xs = np.where(dilated > 0)
    if len(xs) == 0:
        return None

    # Require the white region to cover at least 10 % of the image.
    if len(xs) < image.shape[0] * image.shape[1] * 0.1:
        return None

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    s = coords[:, 0] + coords[:, 1]  # x + y
    d = coords[:, 0] - coords[:, 1]  # x - y

    tl = coords[np.argmin(s)]  # top-left:     min(x+y)
    br = coords[np.argmax(s)]  # bottom-right: max(x+y)
    tr = coords[np.argmax(d)]  # top-right:    max(x-y)
    bl = coords[np.argmin(d)]  # bottom-left:  min(x-y)

    return np.array([tl, tr, br, bl], dtype=np.float32).reshape(4, 1, 2)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left].

    *pts* must be shaped (4, 2).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left: smallest sum
    ordered[2] = pts[np.argmax(s)]  # bottom-right: largest sum

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
        result = pil.transform(
            (width, height), PILImage.PERSPECTIVE, coeffs, PILImage.BICUBIC
        )
        return np.array(result)[:, :, ::-1].copy()  # RGB → BGR
    pil = PILImage.fromarray(image.astype(np.uint8))
    result = pil.transform(
        (width, height), PILImage.PERSPECTIVE, coeffs, PILImage.BICUBIC
    )
    return np.array(result)


# ---------------------------------------------------------------------------
# Overlay / annotation helpers
# ---------------------------------------------------------------------------


def _draw_rect_np(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple,
    thickness: int = 1,
) -> None:
    """Draw a rectangle outline on *image* in-place using NumPy indexing.

    *x2* and *y2* are **exclusive** endpoints (Python/NumPy convention).
    """
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    c = np.array(color, dtype=np.uint8)
    for t in range(thickness):
        ty1 = min(y1 + t, y2 - 1)
        ty2 = max(y2 - 1 - t, y1)
        tx1 = min(x1 + t, x2 - 1)
        tx2 = max(x2 - 1 - t, x1)
        image[ty1, x1:x2] = c
        image[ty2, x1:x2] = c
        image[y1:y2, tx1] = c
        image[y1:y2, tx2] = c


def _fill_rect_np(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple,
) -> None:
    """Fill a rectangle on *image* in-place using NumPy indexing.

    *x2* and *y2* are **exclusive** endpoints (Python/NumPy convention).
    """
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    image[y1:y2, x1:x2] = np.array(color, dtype=np.uint8)


def draw_overlay(
    image: np.ndarray,
    answer_section_rect: tuple[int, int, int, int],
    id_section_rect: tuple[int, int, int, int],
    all_answer_bubbles: list[tuple[int, int, int, int]],
    all_id_bubbles: list[tuple[int, int, int, int]],
    filled_answer_bubbles: list[tuple[int, int, int, int]],
    filled_id_bubbles: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Return a BGR copy of *image* with a grading overlay drawn on it.

    The overlay visualises:

    * **Page outline** – a green border around the entire image.
    * **Answer section** – a blue bounding rectangle for the answer region.
    * **ID section** – a purple bounding rectangle for the student-ID region.
    * **All bubble outlines** – light-gray rectangles for every bubble cell.
    * **Filled bubbles** – bright-green filled rectangles for identified
      filled bubbles.

    Parameters
    ----------
    image:
        Normalised (perspective-corrected, resized) sheet image in BGR format.
    answer_section_rect:
        ``(x1, y1, x2, y2)`` bounding box of the answer section.
    id_section_rect:
        ``(x1, y1, x2, y2)`` bounding box of the ID section.
    all_answer_bubbles:
        Flat list of ``(x, y, w, h)`` for every answer bubble region.
    all_id_bubbles:
        Flat list of ``(x, y, w, h)`` for every ID bubble region.
    filled_answer_bubbles:
        Subset of *all_answer_bubbles* identified as filled.
    filled_id_bubbles:
        Subset of *all_id_bubbles* identified as filled.

    Returns
    -------
    np.ndarray
        A new BGR ``uint8`` array with the overlay applied.
    """
    # Ensure output is a 3-channel BGR uint8 copy
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        gray = image if image.ndim == 2 else image[:, :, 0]
        if _HAVE_CV2:
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            out = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    else:
        out = image.copy().astype(np.uint8)

    h, w = out.shape[:2]

    # Colours (BGR)
    _GREEN = (0, 200, 0)
    _BLUE = (200, 80, 0)
    _PURPLE = (180, 0, 180)
    _GRAY = (160, 160, 160)
    _FILL_GREEN = (0, 200, 50)

    ax1, ay1, ax2, ay2 = answer_section_rect
    ix1, iy1, ix2, iy2 = id_section_rect

    if _HAVE_CV2:
        # cv2.rectangle uses inclusive endpoints, so we subtract 1 from the
        # exclusive upper-right corner to match the exclusive-convention callers.
        # Page outline
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), _GREEN, 4)
        # Section rectangles
        cv2.rectangle(out, (ax1, ay1), (ax2 - 1, ay2 - 1), _BLUE, 2)
        cv2.rectangle(out, (ix1, iy1), (ix2 - 1, iy2 - 1), _PURPLE, 2)
        # All bubble outlines (answer + ID)
        for x, y, bw, bh in all_answer_bubbles:
            cv2.rectangle(out, (x, y), (x + bw - 1, y + bh - 1), _GRAY, 1)
        for x, y, bw, bh in all_id_bubbles:
            cv2.rectangle(out, (x, y), (x + bw - 1, y + bh - 1), _GRAY, 1)
        # Filled bubbles
        for x, y, bw, bh in filled_answer_bubbles:
            cv2.rectangle(out, (x, y), (x + bw - 1, y + bh - 1), _FILL_GREEN, -1)
        for x, y, bw, bh in filled_id_bubbles:
            cv2.rectangle(out, (x, y), (x + bw - 1, y + bh - 1), _FILL_GREEN, -1)
    else:
        # Pure NumPy fallback — uses exclusive (x2, y2) endpoints throughout.
        _draw_rect_np(out, 0, 0, w, h, _GREEN, thickness=4)
        _draw_rect_np(out, ax1, ay1, ax2, ay2, _BLUE, thickness=2)
        _draw_rect_np(out, ix1, iy1, ix2, iy2, _PURPLE, thickness=2)
        for x, y, bw, bh in all_answer_bubbles:
            _draw_rect_np(out, x, y, x + bw, y + bh, _GRAY, thickness=1)
        for x, y, bw, bh in all_id_bubbles:
            _draw_rect_np(out, x, y, x + bw, y + bh, _GRAY, thickness=1)
        for x, y, bw, bh in filled_answer_bubbles:
            _fill_rect_np(out, x, y, x + bw, y + bh, _FILL_GREEN)
        for x, y, bw, bh in filled_id_bubbles:
            _fill_rect_np(out, x, y, x + bw, y + bh, _FILL_GREEN)

    return out


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
            return cv2.resize(
                image, (width, int(h * width / w)), interpolation=cv2.INTER_AREA
            )
        return cv2.resize(
            image, (int(w * height / h), height), interpolation=cv2.INTER_AREA
        )

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
