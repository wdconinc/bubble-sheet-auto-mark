"""In-image grid detection for bubble sheet regions.

Replaces the uniform arithmetic grid produced by
:meth:`BubbleSheetDetector._build_bubble_grid` with a grid derived from
actual line/structure information in the image.  This allows the grid to
accommodate rotation, shear, and block-grouped layouts.

When OpenCV is available Hough line detection is used.  When it is absent a
projection-profile / peak-finding approach implemented in pure NumPy is used
as a fallback.

Public surface
--------------
``detect_bubble_grid(region_image, num_rows, num_cols)``
    Primary entry-point.  Returns a list-of-rows of ``(x, y, w, h)`` bounding
    boxes in the same format as :meth:`BubbleSheetDetector._build_bubble_grid`.

``detect_block_groups(lines, num_rows)``
    Helper that partitions detected lines into the expected row groups
    to handle sheets that arrange questions in labelled blocks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2

    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of the cell size used as inner margin around each detected bubble.
_BUBBLE_CELL_MARGIN_RATIO: float = 0.1

# A gap between consecutive horizontal line positions larger than this multiple
# of the median gap is treated as a block separator.
_BLOCK_GAP_THRESHOLD_MULTIPLIER: float = 1.8

# Minimum projection profile value (fraction of pixels that are "dark") for a
# position to be considered a candidate grid line.
_DEFAULT_MIN_PROMINENCE: float = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_bubble_grid(
    region_image: np.ndarray,
    num_rows: int,
    num_cols: int,
) -> Optional[list[list[tuple[int, int, int, int]]]]:
    """Detect the bubble grid layout from in-image structure information.

    1. Extracts the primary print channel and enhances contrast.
    2. Detects horizontal and vertical lines via Hough transform (cv2) or
       projection-profile peaks (pure NumPy).
    3. Infers the cell grid from those lines.
    4. Returns *None* if not enough lines were found to build the grid — the
       caller should then fall back to the arithmetic uniform grid.

    Parameters
    ----------
    region_image:
        Cropped section of the normalised sheet that contains the bubbles
        (BGR or grayscale NumPy array).
    num_rows:
        Expected number of bubble rows (questions or digit-choice rows).
    num_cols:
        Expected number of bubble columns (choices per question or ID digits).

    Returns
    -------
    list[list[tuple[int,int,int,int]]] or None
        A list of *num_rows* lists, each containing *num_cols* ``(x, y, w, h)``
        tuples giving the bounding box of each bubble cell in the coordinate
        space of *region_image*.  Returns *None* on failure.
    """
    from bubble_mark.processing.color_channel import (
        enhance_contrast,
        extract_print_channel,
    )

    ch = extract_print_channel(region_image)
    enhanced = enhance_contrast(ch)

    h, w = enhanced.shape[:2]

    h_lines, v_lines = _detect_lines(enhanced)

    # We need at least (num_rows+1) horizontal and (num_cols+1) vertical lines
    # to build an unambiguous grid.  Fall back if not enough were found.
    if len(h_lines) < num_rows + 1 or len(v_lines) < num_cols + 1:
        return None

    # Snap detected line positions to the expected grid positions.
    row_ys = _snap_to_grid(h_lines, num_rows + 1, h)
    col_xs = _snap_to_grid(v_lines, num_cols + 1, w)

    if len(row_ys) < 2 or len(col_xs) < 2:
        return None

    # Build (x, y, w, h) bubble cells from the grid lines, applying a small
    # inner margin so the cell boundaries themselves don't count as filled.
    inner_margin = _BUBBLE_CELL_MARGIN_RATIO  # fraction of cell size
    grid: list[list[tuple[int, int, int, int]]] = []
    for row in range(num_rows):
        row_cells: list[tuple[int, int, int, int]] = []
        y0 = row_ys[row]
        y1 = row_ys[row + 1]
        cell_h = y1 - y0
        margin_y = max(1, int(cell_h * inner_margin))
        for col in range(num_cols):
            x0 = col_xs[col]
            x1 = col_xs[col + 1]
            cell_w = x1 - x0
            margin_x = max(1, int(cell_w * inner_margin))
            bub_x = x0 + margin_x
            bub_y = y0 + margin_y
            bub_w = max(1, cell_w - 2 * margin_x)
            bub_h = max(1, cell_h - 2 * margin_y)
            row_cells.append((bub_x, bub_y, bub_w, bub_h))
        grid.append(row_cells)

    return grid


def detect_grid_lines(
    region_image: np.ndarray,
    num_rows: int,
    num_cols: int,
) -> Optional[tuple[list[int], list[int]]]:
    """Return the snapped grid-line pixel positions for a bubble region.

    Parameters
    ----------
    region_image:
        Cropped section of the normalised sheet that contains the bubbles
        (BGR or grayscale NumPy array).
    num_rows:
        Expected number of bubble rows.
    num_cols:
        Expected number of bubble columns.

    Returns
    -------
    tuple[list[int], list[int]] or None
        ``(row_ys, col_xs)`` — two sorted lists of pixel positions giving the
        ``num_rows + 1`` horizontal and ``num_cols + 1`` vertical grid-line
        positions in the coordinate space of *region_image*.  Returns *None*
        if not enough lines could be detected.
    """
    from bubble_mark.processing.color_channel import (
        enhance_contrast,
        extract_print_channel,
    )

    ch = extract_print_channel(region_image)
    enhanced = enhance_contrast(ch)

    h, w = enhanced.shape[:2]

    h_lines, v_lines = _detect_lines(enhanced)

    if len(h_lines) < num_rows + 1 or len(v_lines) < num_cols + 1:
        return None

    row_ys = _snap_to_grid(h_lines, num_rows + 1, h)
    col_xs = _snap_to_grid(v_lines, num_cols + 1, w)

    if len(row_ys) < 2 or len(col_xs) < 2:
        return None

    return row_ys, col_xs


def detect_block_groups(
    lines: dict[str, list[int]],
    num_rows: int,
) -> list[list[int]]:
    """Partition detected line positions into block groups.

    On some bubble sheets questions are grouped in labelled blocks (e.g., two
    columns of 15 questions each).  This function detects unusually large gaps
    between consecutive line positions and uses them to infer block boundaries.

    Parameters
    ----------
    lines:
        A dictionary with keys ``"horizontal"`` and ``"vertical"``, each
        containing a sorted list of pixel positions for the detected grid
        lines in the region image.
    num_rows:
        Expected total number of bubble rows.

    Returns
    -------
    list[list[int]]
        A list of row-index groups.  For a non-grouped sheet this is
        ``[[0, 1, …, num_rows-1]]``; for a 2-block sheet it might be
        ``[[0, …, 14], [15, …, 29]]``.
    """
    h_positions = sorted(lines.get("horizontal", []))
    if len(h_positions) < 2:
        return [list(range(num_rows))]

    gaps = [h_positions[i + 1] - h_positions[i] for i in range(len(h_positions) - 1)]
    if not gaps:
        return [list(range(num_rows))]

    median_gap = float(np.median(gaps))
    threshold = median_gap * _BLOCK_GAP_THRESHOLD_MULTIPLIER

    groups: list[list[int]] = []
    current_group: list[int] = []
    row_idx = 0
    for i, gap in enumerate(gaps):
        current_group.append(row_idx)
        row_idx += 1
        if gap > threshold:
            groups.append(current_group)
            current_group = []
    current_group.append(row_idx)
    groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_lines(
    binary_or_gray: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Detect horizontal and vertical lines in *binary_or_gray*.

    Returns two sorted lists of pixel positions: (horizontal_ys, vertical_xs).
    """
    if _HAVE_CV2:
        return _detect_lines_cv2(binary_or_gray)
    return _detect_lines_projection(binary_or_gray)


def _detect_lines_cv2(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Hough-line detection using cv2.HoughLinesP."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )

    h_ys: list[int] = []
    v_xs: list[int] = []

    if raw is None:
        return h_ys, v_xs

    for line in raw:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 15 or angle > 165:
            # Roughly horizontal
            h_ys.append((y1 + y2) // 2)
        elif 75 < angle < 105:
            # Roughly vertical
            v_xs.append((x1 + x2) // 2)

    return sorted(h_ys), sorted(v_xs)


def _detect_lines_projection(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Projection-profile peak detection (pure NumPy fallback)."""
    # Binarise: dark pixels < 128 → 1, bright → 0
    binary = (gray < 128).astype(np.float32)

    h_proj = binary.mean(axis=1)  # row sums → horizontal line evidence
    v_proj = binary.mean(axis=0)  # col sums → vertical line evidence

    h_ys = _find_projection_peaks(h_proj)
    v_xs = _find_projection_peaks(v_proj)

    return h_ys, v_xs


def _find_projection_peaks(
    proj: np.ndarray, min_prominence: float = _DEFAULT_MIN_PROMINENCE
) -> list[int]:
    """Return indices where *proj* has local maxima above *min_prominence*.

    A simple non-maximum suppression with a window equal to ~2 % of the
    signal length is applied so that nearby peaks are merged.
    """
    n = len(proj)
    if n == 0:
        return []

    window = max(3, n // 50)
    peaks: list[int] = []

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbourhood = proj[lo:hi]
        if proj[i] == neighbourhood.max() and proj[i] >= min_prominence:
            peaks.append(i)

    # Merge adjacent peaks (within *window* pixels) by keeping the maximum
    merged: list[int] = []
    for p in peaks:
        if merged and p - merged[-1] <= window:
            # Replace last if this one is higher
            if proj[p] > proj[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)

    return merged


def _snap_to_grid(detected: list[int], n: int, extent: int) -> list[int]:
    """Return *n* evenly-spaced positions inferred from *detected* lines.

    When *detected* has ``≥ n`` positions they are clustered into *n* groups
    (1-D k-means) and the cluster centres are returned as integer pixel
    positions.

    When *detected* has fewer than *n* positions, the gaps between the known
    positions are subdivided until *n* positions are available; if fewer than
    two positions are known an arithmetic fallback based on *extent* is used.

    The returned list is sorted and has exactly *n* elements (or fewer if not
    enough input is provided).  Endpoint positions near 0 or *extent* are not
    explicitly enforced — the output reflects the geometry implied by
    *detected*.
    """
    if len(detected) == 0:
        return []

    detected_arr = np.array(sorted(detected), dtype=np.float32)

    if len(detected_arr) >= n:
        # Cluster into n groups using a simple 1-D k-means with initialisation
        # spread evenly across the detected range.
        centers = np.linspace(detected_arr[0], detected_arr[-1], n)
        for _ in range(20):  # iterate to convergence
            labels = np.argmin(np.abs(detected_arr[:, None] - centers[None, :]), axis=1)
            new_centers = np.array(
                [
                    detected_arr[labels == k].mean()
                    if np.any(labels == k)
                    else centers[k]
                    for k in range(n)
                ]
            )
            if np.allclose(new_centers, centers, atol=0.5):
                break
            centers = new_centers
        return sorted(int(round(c)) for c in centers)

    # Fewer lines than expected → interpolate the missing ones
    if len(detected_arr) < 2:
        step = extent / n
        return [int(round(i * step)) for i in range(n)]

    step = float(detected_arr[-1] - detected_arr[0]) / (len(detected_arr) - 1)
    positions = list(detected_arr)
    while len(positions) < n:
        # Fill the largest gap
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        big = int(np.argmax(gaps))
        mid = (positions[big] + positions[big + 1]) / 2.0
        positions.insert(big + 1, mid)

    return sorted(int(round(p)) for p in positions[:n])
