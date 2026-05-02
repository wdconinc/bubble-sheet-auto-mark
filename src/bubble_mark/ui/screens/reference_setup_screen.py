"""Reference setup screen: interactive distortion correction and region selection.

This screen guides the user through a two-step reference sheet setup:

**Step 1 – Page edge lines**
    The user draws four lines on the raw reference sheet image that correspond
    to the four page edges.  The lines are used to compute a perspective warp
    that corrects the distortion introduced by camera angle.

**Step 2 – Region selection**
    On the distortion-corrected image the user draws two rectangles:
    one for the answer bubble region and one for the student-ID bubble region.
    The rectangles are stored as normalized ``[x1, y1, x2, y2]`` bounds in
    :class:`~bubble_mark.models.settings.AppSettings`.

Because Toga does not expose a raw pointer/touch-event canvas that supports
arbitrary line drawing on all platforms (desktop, Android, iOS) without the
Briefcase build environment, this screen uses a text-coordinate approach as a
universally runnable fallback: the user types four line endpoints and two
rectangle corners.  A visual preview is rendered by drawing the entered
geometry onto a copy of the image and displaying it in a Toga ``ImageView``.

When the user confirms the setup the corrected reference image is saved
alongside the source image (as ``<name>_corrected.png``) so that the grader
can load it for cross-correlation alignment.
"""

from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

if TYPE_CHECKING:
    from bubble_mark.ui.app import BubbleMarkApp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_line(text: str) -> list[float] | None:
    """Parse ``"x1, y1, x2, y2"`` into a ``[float, …]`` list or *None*."""
    try:
        parts = [float(v.strip()) for v in text.split(",")]
        if len(parts) == 4:
            return parts
    except ValueError:
        pass
    return None


def _parse_rect(text: str) -> list[float] | None:
    """Parse ``"x1, y1, x2, y2"`` (pixel coords) into a list or *None*."""
    return _parse_line(text)


def _lines_to_normalized_rect(
    rect_px: list[float],
    img_w: int,
    img_h: int,
) -> list[float] | None:
    """Convert pixel-coordinate rect to normalized [0-1] bounds."""
    x1, y1, x2, y2 = rect_px
    nx1, ny1 = x1 / img_w, y1 / img_h
    nx2, ny2 = x2 / img_w, y2 / img_h
    # Ensure proper ordering
    if nx1 > nx2:
        nx1, nx2 = nx2, nx1
    if ny1 > ny2:
        ny1, ny2 = ny2, ny1
    from bubble_mark.models.settings import _validate_region

    return _validate_region([nx1, ny1, nx2, ny2])


def _draw_geometry_on_image(
    image: np.ndarray,
    lines: list[list[float]],
    answer_rect: list[float] | None,
    id_rect: list[float] | None,
) -> np.ndarray:
    """Return a BGR copy of *image* with the entered geometry drawn on it."""
    try:
        import cv2 as _cv2

        out = image.copy()
        for ln in lines:
            x1, y1, x2, y2 = [int(v) for v in ln]
            _cv2.line(out, (x1, y1), (x2, y2), (0, 180, 0), 2)
        if answer_rect:
            ax1, ay1, ax2, ay2 = [int(v) for v in answer_rect]
            _cv2.rectangle(out, (ax1, ay1), (ax2, ay2), (200, 80, 0), 2)
        if id_rect:
            ix1, iy1, ix2, iy2 = [int(v) for v in id_rect]
            _cv2.rectangle(out, (ix1, iy1), (ix2, iy2), (180, 0, 180), 2)
        return out
    except ImportError:
        pass

    # Pure NumPy fallback: just return a copy (geometry not drawn)
    return image.copy()


def _bgr_to_toga_image(bgr: np.ndarray) -> toga.Image:
    """Convert a BGR NumPy array to a ``toga.Image``."""
    from PIL import Image as PILImage

    rgb = bgr[:, :, ::-1].copy()
    pil = PILImage.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return toga.Image(data=buf.getvalue())


def _save_corrected_image(corrected: np.ndarray, source_path: str) -> str:
    """Save the corrected image next to *source_path* and return its path."""
    base, ext = os.path.splitext(source_path)
    out_path = base + "_corrected" + (ext or ".png")
    from PIL import Image as PILImage

    if corrected.ndim == 3:
        rgb = corrected[:, :, ::-1].copy()
        pil = PILImage.fromarray(rgb.astype(np.uint8), mode="RGB")
    else:
        pil = PILImage.fromarray(corrected.astype(np.uint8), mode="L")
    pil.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Screen builder
# ---------------------------------------------------------------------------


def build_reference_setup_screen(app: "BubbleMarkApp") -> toga.Box:
    """Return a Box containing the reference sheet setup UI.

    The screen is divided into two phases:
    - Phase A: Enter four edge lines → preview → apply perspective correction.
    - Phase B: Enter two region rectangles on the corrected image → save.
    """
    box = toga.Box(style=Pack(direction=COLUMN, padding=10))

    title = toga.Label(
        "Reference Sheet Setup",
        style=Pack(padding_bottom=6, font_size=16),
    )
    hint = toga.Label(
        "Step 1: Enter four page-edge lines (pixel coords) and press "
        "'Apply Correction'.  Step 2: Enter answer and ID regions "
        "(pixel coords on corrected image) and press 'Save Regions'.",
        style=Pack(padding_bottom=8),
    )

    status_label = toga.Label("", style=Pack(padding_bottom=6))
    preview = toga.ImageView(style=Pack(flex=1))

    # -- Internal state -------------------------------------------------------
    _state: dict = {
        "raw_image": None,  # BGR ndarray of the loaded reference image
        "corrected_image": None,  # BGR ndarray after perspective correction
    }

    # -- Load reference image -------------------------------------------------
    def _load_reference() -> None:
        path = app.app_settings.reference_image_path
        if not path or not os.path.exists(path):
            status_label.text = "No reference image path configured in Settings."
            return
        try:
            from bubble_mark.processing.image_utils import load_image

            _state["raw_image"] = load_image(path)
            _state["corrected_image"] = None
            preview.image = _bgr_to_toga_image(_state["raw_image"])
            status_label.text = f"Loaded: {os.path.basename(path)}"
        except Exception as exc:
            logger.exception("Failed to load reference image: %s", exc)
            status_label.text = f"Error loading image: {exc}"

    # -------------------------------------------------------------------------
    # Phase A: edge lines
    # -------------------------------------------------------------------------

    lines_header = toga.Label(
        "Page edge lines (one per row, format: x1, y1, x2, y2 in pixels):",
        style=Pack(padding_top=8, padding_bottom=4),
    )

    line_inputs: list[toga.TextInput] = [
        toga.TextInput(
            placeholder=f"Edge {i + 1}: x1, y1, x2, y2",
            style=Pack(padding_bottom=4),
        )
        for i in range(4)
    ]

    def _refresh_preview(_=None) -> None:
        img = _state["corrected_image"] or _state["raw_image"]
        if img is None:
            return
        lines = [_parse_line(inp.value) for inp in line_inputs]
        valid_lines = [ln for ln in lines if ln is not None]
        ar = _parse_rect(inp_answer_rect.value)
        ir = _parse_rect(inp_id_rect.value)
        annotated = _draw_geometry_on_image(img, valid_lines, ar, ir)
        preview.image = _bgr_to_toga_image(annotated)

    for inp in line_inputs:
        inp.on_change = _refresh_preview

    def apply_correction(_=None) -> None:
        img = _state["raw_image"]
        if img is None:
            status_label.text = "Load the reference image first."
            return
        lines = [_parse_line(inp.value) for inp in line_inputs]
        if any(ln is None for ln in lines):
            status_label.text = "Please enter all four edge lines."
            return
        try:
            from bubble_mark.processing.distortion import correct_distortion_from_lines

            corrected = correct_distortion_from_lines(img, lines)  # type: ignore[arg-type]
            if corrected is None:
                status_label.text = "Could not determine corners from the given lines."
                return
            _state["corrected_image"] = corrected
            preview.image = _bgr_to_toga_image(corrected)
            status_label.text = (
                "Distortion correction applied. Now enter region rectangles."
            )
        except Exception as exc:
            logger.exception("Correction failed: %s", exc)
            status_label.text = f"Error: {exc}"

    btn_apply = toga.Button(
        "Apply Correction",
        on_press=apply_correction,
        style=Pack(padding_bottom=8),
    )

    # -------------------------------------------------------------------------
    # Phase B: region rectangles
    # -------------------------------------------------------------------------

    regions_header = toga.Label(
        "Regions on corrected image (pixel coords, format: x1, y1, x2, y2):",
        style=Pack(padding_top=8, padding_bottom=4),
    )

    inp_answer_rect = toga.TextInput(
        placeholder="Answer region: x1, y1, x2, y2",
        style=Pack(padding_bottom=4),
    )
    inp_id_rect = toga.TextInput(
        placeholder="ID region: x1, y1, x2, y2",
        style=Pack(padding_bottom=4),
    )
    inp_answer_rect.on_change = _refresh_preview
    inp_id_rect.on_change = _refresh_preview

    def save_regions(_=None) -> None:
        corrected = _state["corrected_image"]
        if corrected is None:
            status_label.text = "Apply distortion correction first."
            return

        h, w = corrected.shape[:2]

        ar_px = _parse_rect(inp_answer_rect.value)
        ir_px = _parse_rect(inp_id_rect.value)

        answer_region = _lines_to_normalized_rect(ar_px, w, h) if ar_px else None
        id_region = _lines_to_normalized_rect(ir_px, w, h) if ir_px else None

        if answer_region is None and ar_px is not None:
            status_label.text = "Invalid answer region coordinates."
            return
        if id_region is None and ir_px is not None:
            status_label.text = "Invalid ID region coordinates."
            return

        # Persist the corrected reference image next to the source
        ref_path = app.app_settings.reference_image_path
        if ref_path:
            try:
                corrected_path = _save_corrected_image(corrected, ref_path)
                app.app_settings.reference_image_path = corrected_path
                logger.info("Saved corrected reference image to %s", corrected_path)
            except Exception as exc:
                logger.warning("Could not save corrected image: %s", exc)

        if answer_region is not None:
            app.app_settings.answer_region = answer_region
        if id_region is not None:
            app.app_settings.id_region = id_region

        # Store the edge lines in settings for reproducibility
        lines = [_parse_line(inp.value) for inp in line_inputs]
        valid_lines = [ln for ln in lines if ln is not None]
        if len(valid_lines) == 4:
            app.app_settings.page_edge_lines = valid_lines

        status_label.text = "Reference setup saved successfully."

    btn_save = toga.Button(
        "Save Regions",
        on_press=save_regions,
        style=Pack(padding_bottom=8),
    )

    btn_row = toga.Box(style=Pack(direction=ROW, gap=8, padding_top=6))
    btn_row.add(
        toga.Button(
            "Load Reference Image",
            on_press=lambda _: _load_reference(),
            style=Pack(flex=1),
        ),
        toga.Button("Back", on_press=lambda _: app.go_settings(), style=Pack(flex=1)),
    )

    box.add(
        title,
        hint,
        preview,
        status_label,
        lines_header,
        *line_inputs,
        btn_apply,
        regions_header,
        toga.Label("Answer region:", style=Pack(padding_bottom=2)),
        inp_answer_rect,
        toga.Label("ID region:", style=Pack(padding_bottom=2)),
        inp_id_rect,
        btn_save,
        btn_row,
    )
    return box
