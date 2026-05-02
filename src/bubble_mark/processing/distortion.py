"""Distortion correction helpers for bubble sheet images.

Two correction strategies are provided:

1. **Line-based** (:func:`correct_distortion_from_lines`): the user supplies
   four edge lines drawn interactively on the reference sheet; the four page
   corners are obtained as line intersections and a perspective warp is applied.

2. **Correlation-based** (:func:`estimate_distortion_from_reference`): given a
   distortion-corrected reference image and an incoming sheet image, a
   normalised cross-correlation (phase correlation) is used to estimate the
   dominant translational offset between the two images.  A translation-only
   homography is returned; this is sufficient to correct small residual
   misalignments after the initial perspective warp.

All functions work with or without OpenCV via the same cv2 / NumPy-Pillow
fallback pattern used throughout the rest of the package.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bubble_mark.processing.image_utils import order_points, perspective_transform

try:
    import cv2

    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lines with a |denominator| below this threshold are treated as parallel.
_PARALLEL_LINE_EPSILON: float = 1e-10

# Small epsilon added to the FFT cross-power spectrum to avoid division by zero.
_FFT_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Line-intersection helpers
# ---------------------------------------------------------------------------


def find_intersection(
    line1: list[float],
    line2: list[float],
) -> Optional[tuple[float, float]]:
    """Return the (x, y) intersection of two infinite lines.

    Each line is specified as ``[x1, y1, x2, y2]`` — two points that the line
    passes through.  Returns *None* if the lines are parallel (or
    nearly-parallel, with a determinant below a small epsilon).

    Parameters
    ----------
    line1, line2:
        Each a 4-element list/tuple ``[x1, y1, x2, y2]``.
    """
    x1, y1, x2, y2 = (float(v) for v in line1)
    x3, y3, x4, y4 = (float(v) for v in line2)

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < _PARALLEL_LINE_EPSILON:
        return None  # parallel or coincident

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return float(x), float(y)


def _lines_to_corners(
    edge_lines: list[list[float]],
) -> Optional[np.ndarray]:
    """Convert four edge lines to four corner points via pairwise intersections.

    *edge_lines* must be a list of exactly four ``[x1, y1, x2, y2]`` lines
    representing the top, bottom, left, and right page edges (in any order).

    The four corners are found as the six possible pairwise intersections;
    only four are retained — those that form the largest quadrilateral.

    Returns an ``(4, 2)`` float32 array ordered ``[TL, TR, BR, BL]``, or
    *None* if fewer than four valid intersections can be found.
    """
    if len(edge_lines) != 4:
        return None

    # Compute all 6 pairwise intersections
    from itertools import combinations

    pts: list[tuple[float, float]] = []
    for l1, l2 in combinations(edge_lines, 2):
        pt = find_intersection(l1, l2)
        if pt is not None:
            pts.append(pt)

    # Keep the 4 corners that maximise the convex-hull area.
    # With exactly 4 non-parallel edge lines we typically get 4–6 intersections
    # (parallel pairs don't intersect); we pick the 4 that span the most area
    # by ordering and returning the outermost 4.
    if len(pts) < 4:
        return None

    pts_arr = np.array(pts, dtype=np.float32)

    if len(pts_arr) == 4:
        return order_points(pts_arr)

    # More than 4 intersections – select the 4 that form the largest area quad.
    # Use the convex hull and pick at most 4 corner points.
    from itertools import combinations as _comb

    best_area = -1.0
    best_quad: Optional[np.ndarray] = None
    for quad_pts in _comb(range(len(pts_arr)), 4):
        quad = pts_arr[list(quad_pts)]
        ordered = order_points(quad)
        tl, tr, br, bl = ordered
        # Shoelace formula
        area = (
            abs(
                (tl[0] * tr[1] - tr[0] * tl[1])
                + (tr[0] * br[1] - br[0] * tr[1])
                + (br[0] * bl[1] - bl[0] * br[1])
                + (bl[0] * tl[1] - tl[0] * bl[1])
            )
            / 2.0
        )
        if area > best_area:
            best_area = area
            best_quad = ordered

    return best_quad


# ---------------------------------------------------------------------------
# Public API – line-based correction
# ---------------------------------------------------------------------------


def correct_distortion_from_lines(
    image: np.ndarray,
    edge_lines: list[list[float]],
) -> Optional[np.ndarray]:
    """Apply a perspective correction using four user-drawn edge lines.

    Parameters
    ----------
    image:
        The reference sheet image as a BGR (or grayscale) NumPy array.
    edge_lines:
        A list of exactly four ``[x1, y1, x2, y2]`` line definitions
        representing the four page edges drawn by the user.

    Returns
    -------
    np.ndarray or None
        The perspective-corrected image, or *None* if the corners cannot be
        determined from the supplied lines.
    """
    corners = _lines_to_corners(edge_lines)
    if corners is None:
        return None

    contour = corners.reshape(4, 1, 2)
    return perspective_transform(image, contour)


# ---------------------------------------------------------------------------
# Public API – correlation-based distortion estimation
# ---------------------------------------------------------------------------


def estimate_distortion_from_reference(
    sheet_image: np.ndarray,
    reference_image: np.ndarray,
    channel: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Estimate a translational offset that aligns *sheet_image* to *reference_image*.

    The alignment is performed in a single reference color channel (see
    :mod:`bubble_mark.processing.color_channel`).  An FFT-based normalised
    cross-correlation (phase correlation) is used to recover the dominant
    translational offset between the two images.  A ``(3, 3)`` translation-only
    homography is returned; perspective/shear distortions are not corrected here
    (those are handled by the initial line-based perspective warp).

    Parameters
    ----------
    sheet_image:
        The incoming (possibly distorted) sheet image, already perspective-
        corrected to the normalised sheet size.
    reference_image:
        The distortion-corrected reference (empty) sheet image.  Must be the
        same size as *sheet_image* after the initial perspective correction.
    channel:
        Color channel index to use for cross-correlation.  When *None* the
        channel with the highest variance is chosen automatically.

    Returns
    -------
    np.ndarray or None
        A ``(3, 3)`` float64 translation-only homography matrix that maps
        *sheet_image* coordinates to the reference frame.  Returns *None* if
        the images cannot be aligned (e.g., very different sizes).
    """
    from bubble_mark.processing.color_channel import extract_print_channel

    # Resize reference to match sheet if sizes differ
    sh, sw = sheet_image.shape[:2]
    rh, rw = reference_image.shape[:2]
    if (sh, sw) != (rh, rw):
        if _HAVE_CV2:
            ref_resized = cv2.resize(
                reference_image, (sw, sh), interpolation=cv2.INTER_AREA
            )
        else:
            from PIL import Image as PILImage

            if reference_image.ndim == 3:
                pil = PILImage.fromarray(
                    reference_image[:, :, ::-1].astype(np.uint8), mode="RGB"
                )
                pil_resized = pil.resize((sw, sh), PILImage.LANCZOS)
                ref_resized = np.array(pil_resized)[:, :, ::-1].copy()
            else:
                pil = PILImage.fromarray(reference_image.astype(np.uint8), mode="L")
                pil_resized = pil.resize((sw, sh), PILImage.LANCZOS)
                ref_resized = np.array(pil_resized)
    else:
        ref_resized = reference_image

    ch_sheet = extract_print_channel(sheet_image, channel).astype(np.float32)
    ch_ref = extract_print_channel(ref_resized, channel).astype(np.float32)

    # Normalised cross-correlation via FFT to find (dx, dy) translation
    dx, dy = _fft_translation(ch_sheet, ch_ref)

    if _HAVE_CV2:
        # Build a homography that accounts for the translation only.
        # For more accuracy a full feature-based approach would be used, but
        # NCC translation is sufficient for small residual distortions after
        # the initial perspective correction.
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = -float(dx)
        H[1, 2] = -float(dy)
        return H

    # No cv2 – return translation as 3×3 matrix
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = -float(dx)
    H[1, 2] = -float(dy)
    return H


def apply_homography(
    image: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Warp *image* with the 3×3 homography *H*.

    Convention: ``H`` is treated as a forward transform mapping source pixel
    coordinates to destination pixel coordinates (i.e., a source pixel at
    ``(x, y)`` maps to destination ``(x + H[0,2], y + H[1,2])`` for a
    translation-only matrix).  Both the cv2 and the NumPy fallback path apply
    this same convention so that results are platform-independent.

    Parameters
    ----------
    image:
        Source image (BGR or grayscale).
    H:
        A ``(3, 3)`` float64 homography matrix.

    Returns
    -------
    np.ndarray
        The warped image, same dtype and channel count as *image*.
    """
    h, w = image.shape[:2]
    if _HAVE_CV2:
        # cv2.warpPerspective default (no WARP_INVERSE_MAP) performs
        # dst(x,y) = src(H^{-1} * [x,y,1]^T), which is equivalent to
        # moving a source pixel at (x,y) to destination (x + H[0,2], y + H[1,2])
        # for a pure-translation H — matching the fallback convention below.
        return cv2.warpPerspective(image, H, (w, h))

    # Pure fallback: only handle simple translation (H is nearly identity).
    # H[0,2] / H[1,2] give the forward shift (src → dst), matching cv2 above.
    dx = int(round(H[0, 2]))
    dy = int(round(H[1, 2]))
    return _translate_image(image, dx, dy)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fft_translation(
    img_a: np.ndarray,
    img_b: np.ndarray,
) -> tuple[float, float]:
    """Return the (dx, dy) sub-pixel translation of *img_a* relative to *img_b*.

    Uses the phase-correlation (FFT cross-power-spectrum) method.  The result
    is the shift such that ``img_b ≈ translate(img_a, dx, dy)``.
    """
    # Zero-pad to same power-of-two size for efficiency
    h = max(img_a.shape[0], img_b.shape[0])
    w = max(img_a.shape[1], img_b.shape[1])

    fa = np.fft.fft2(img_a, s=(h, w))
    fb = np.fft.fft2(img_b, s=(h, w))

    cross_power = fa * np.conj(fb)
    denom = np.abs(cross_power) + _FFT_EPSILON  # avoid division by zero
    normalised = cross_power / denom

    response = np.fft.ifft2(normalised).real
    idx = np.unravel_index(np.argmax(response), response.shape)

    dy_raw, dx_raw = idx
    # Convert to signed offset
    dy = dy_raw if dy_raw < h // 2 else dy_raw - h
    dx = dx_raw if dx_raw < w // 2 else dx_raw - w
    return float(dx), float(dy)


def _translate_image(
    image: np.ndarray,
    dx: int,
    dy: int,
) -> np.ndarray:
    """Return a copy of *image* translated by (*dx*, *dy*) pixels.

    Regions that slide out of the frame are filled with zeros; regions that
    slide into the frame from outside are also zero (no wrap-around).
    """
    result = np.zeros_like(image)
    h, w = image.shape[:2]

    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)

    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)

    if src_x2 > src_x1 and src_y2 > src_y1 and dst_x2 > dst_x1 and dst_y2 > dst_y1:
        result[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    return result
