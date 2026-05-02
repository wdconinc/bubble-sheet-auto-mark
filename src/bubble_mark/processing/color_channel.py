"""Color channel extraction and contrast enhancement for bubble sheet detection.

All public functions work with or without OpenCV.  When ``cv2`` is available it
is used as-is; when absent every function falls back to a NumPy / Pillow
equivalent that is always available.
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

# CLAHE parameters used by enhance_contrast() when cv2 is available.
_CLAHE_CLIP_LIMIT: float = 2.0
_CLAHE_TILE_SIZE: tuple[int, int] = (8, 8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_print_channel(
    image: np.ndarray,
    channel: Optional[int] = None,
) -> np.ndarray:
    """Return a single-channel (grayscale) array that best captures dark ink.

    For typical bubble sheets printed with black or dark-blue ink on white
    paper, the channel with the highest variance carries the most information
    about the printed marks.

    Parameters
    ----------
    image:
        Input image as a NumPy array.  Both BGR (3-channel) and grayscale
        (1-channel / 2-D) arrays are accepted.
    channel:
        If given, extract exactly this channel index (0=Blue, 1=Green, 2=Red
        for BGR images).  When *None* (default) the channel with the highest
        pixel variance is selected automatically.

    Returns
    -------
    np.ndarray
        A 2-D ``uint8`` array of the same height and width as *image*.
    """
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].astype(np.uint8)

    num_channels = image.shape[2]

    if channel is not None:
        idx = int(channel) % num_channels
        return image[:, :, idx].astype(np.uint8)

    # Auto-select: pick the channel with the highest variance
    best_idx = 0
    best_var = -1.0
    for i in range(num_channels):
        var = float(np.var(image[:, :, i]))
        if var > best_var:
            best_var = var
            best_idx = i

    return image[:, :, best_idx].astype(np.uint8)


def enhance_contrast(channel: np.ndarray) -> np.ndarray:
    """Enhance contrast of a single-channel image for grid/line detection.

    Uses CLAHE (Contrast-Limited Adaptive Histogram Equalisation) when OpenCV
    is available; otherwise falls back to a global histogram equalisation
    implemented with NumPy/Pillow.

    Parameters
    ----------
    channel:
        A 2-D ``uint8`` grayscale array.

    Returns
    -------
    np.ndarray
        A 2-D ``uint8`` array of the same shape as *channel* with enhanced
        local contrast.
    """
    gray = channel if channel.ndim == 2 else channel[:, :, 0]
    gray = gray.astype(np.uint8)

    if _HAVE_CV2:
        clahe = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_SIZE
        )
        return clahe.apply(gray)

    # Pure NumPy / Pillow fallback: global histogram equalisation
    return _equalize_hist_np(gray)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _equalize_hist_np(gray: np.ndarray) -> np.ndarray:
    """Global histogram equalisation using NumPy only.

    For each pixel value *v* the mapping is::

        new_v = round((cdf(v) - cdf_min) / (N - cdf_min) * 255)

    where *N* is the total number of pixels.
    """
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    # Mask zeros in the CDF so we can compute the minimum non-zero value
    cdf_masked = np.ma.array(cdf, mask=(cdf == 0))
    cdf_min = int(cdf_masked.min())
    n = int(gray.size)
    # Build the lookup table
    denom = n - cdf_min
    if denom == 0:
        return gray.copy()
    lut = np.round((cdf - cdf_min) / denom * 255).clip(0, 255).astype(np.uint8)
    return lut[gray]
