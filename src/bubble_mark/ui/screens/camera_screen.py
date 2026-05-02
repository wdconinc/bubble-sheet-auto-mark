"""Camera / image import screen.

On Android the screen opens a live CameraX viewfinder with an OpenCV overlay
that highlights the detected bubble-sheet page contour in real time.  A
"Capture" button freezes the current frame and feeds it into the grading
pipeline.

On desktop the "Open Camera" button falls back to a file-import dialog so the
rest of the workflow can be exercised without camera hardware.
"""
from __future__ import annotations

import io
import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

if TYPE_CHECKING:
    from bubble_mark.ui.app import BubbleMarkApp

logger = logging.getLogger(__name__)


def _is_android() -> bool:
    return sys.platform == "android" or "ANDROID_DATA" in __import__("os").environ


# When cv2 is absent, overlay detection runs every Nth frame to reduce CPU load.
_OVERLAY_SKIP_FRAMES = 5
_overlay_frame_count: int = 0
_overlay_cached_result: "np.ndarray | None" = None


def _draw_overlay(rgb: np.ndarray) -> np.ndarray:
    """Return a copy of *rgb* with a green page-contour overlay if found.

    Works with or without OpenCV.  When cv2 is available the contour is drawn
    directly on a BGR copy; otherwise the Pillow ImageDraw API is used.

    When cv2 is absent, contour detection is CPU-intensive so results are
    cached and refreshed only every ``_OVERLAY_SKIP_FRAMES`` frames.
    """
    global _overlay_frame_count, _overlay_cached_result

    try:
        import cv2 as _cv2
        _have_cv2_local = True
    except ImportError:
        _have_cv2_local = False

    # Throttle the expensive pure-NumPy/Pillow detection path.
    if not _have_cv2_local:
        _overlay_frame_count += 1
        if _overlay_frame_count % _OVERLAY_SKIP_FRAMES != 0:
            return _overlay_cached_result if _overlay_cached_result is not None else rgb

    try:
        from bubble_mark.processing.image_utils import find_page_contour
        bgr = rgb[:, :, ::-1].copy()  # RGB → BGR (pure numpy, no cv2 needed)
        contour = find_page_contour(bgr)
        if contour is None:
            return rgb

        if _have_cv2_local:
            _cv2.drawContours(bgr, [contour], -1, (0, 255, 0), 3)
            return bgr[:, :, ::-1].copy()  # BGR → RGB
        # Pillow fallback: draw the 4-sided polygon on the RGB image.
        from PIL import Image as PILImage, ImageDraw
        pts = contour.reshape(4, 2)
        poly = [(int(x), int(y)) for x, y in pts]
        pil = PILImage.fromarray(rgb.copy())
        draw = ImageDraw.Draw(pil)
        draw.polygon(poly, outline=(0, 255, 0))
        # Draw a thicker border by drawing multiple times with slight offsets.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
            shifted = [(x + dx, y + dy) for x, y in poly]
            draw.polygon(shifted, outline=(0, 255, 0))
        result = np.array(pil)
        _overlay_cached_result = result
        return result
    except Exception:
        return rgb


def _numpy_to_toga_image(rgb: np.ndarray) -> toga.Image:
    """Convert an HxWx3 uint8 RGB array to a ``toga.Image``."""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return toga.Image(data=buf.getvalue())


def build_camera_screen(app: BubbleMarkApp) -> toga.Box:
    """Return a Box containing the camera/import screen UI."""
    box = toga.Box(style=Pack(direction=COLUMN, padding=10))

    title = toga.Label(
        "Camera / Import",
        style=Pack(padding_bottom=10, font_size=18),
    )
    status_label = toga.Label(
        "Press 'Open Camera' to start the live viewfinder.",
        style=Pack(padding_bottom=8),
    )

    # Live viewfinder widget (updated each frame on Android).
    image_view = toga.ImageView(style=Pack(flex=1))

    # Thread-safe reference to the latest raw RGB frame for capture.
    _last_frame: list[np.ndarray | None] = [None]

    # ------------------------------------------------------------------ #
    # Frame callback – called on a CameraX worker thread                  #
    # ------------------------------------------------------------------ #

    def _on_frame(rgb: np.ndarray) -> None:
        _last_frame[0] = rgb.copy()
        annotated = _draw_overlay(rgb)
        toga_img = _numpy_to_toga_image(annotated)

        def _update(_=None) -> None:
            image_view.image = toga_img
            status_label.text = "Live preview — point at a bubble sheet."

        app.loop.call_soon_threadsafe(_update)

    # ------------------------------------------------------------------ #
    # Button callbacks                                                     #
    # ------------------------------------------------------------------ #

    def open_camera(widget: toga.Widget) -> None:
        if _is_android():
            from bubble_mark.ui.camerax_bridge import start_camera
            status_label.text = "Starting camera…"
            logger.info("Opening camera (Android CameraX).")
            started = start_camera(_on_frame)
            if not started:
                status_label.text = (
                    "Camera permission requested. Please grant it and tap 'Open Camera' again."
                )
        else:
            logger.info("Desktop: opening file-import dialog.")
            _import_from_file()

    def _import_from_file() -> None:
        async def _pick() -> None:
            try:
                result = await app.main_window.open_file_dialog(
                    title="Open bubble-sheet image",
                    file_types=["jpg", "jpeg", "png", "bmp"],
                )
            except Exception as exc:
                status_label.text = "Error opening file dialog."
                logger.exception("File dialog raised an unexpected error: %s", exc)
                return
            if result is None:
                status_label.text = "File import cancelled."
                logger.info("File import cancelled (no selection).")
                return
            try:
                from PIL import Image as PILImage
                with PILImage.open(str(result)) as img:
                    pil = img.convert("RGB")
                rgb = np.array(pil, dtype=np.uint8)
                logger.info("Loaded image: %s (%dx%d)", result.name, rgb.shape[1], rgb.shape[0])
                _on_frame(rgb)
                status_label.text = f"Loaded: {result.name}"
            except Exception as exc:
                logger.exception("Error loading image: %s", exc)
                status_label.text = f"Error loading image: {exc}"

        import asyncio
        asyncio.ensure_future(_pick())

    def capture(widget: toga.Widget) -> None:
        frame = _last_frame[0]
        if frame is None:
            status_label.text = "No frame to capture yet."
            logger.warning("Capture pressed but no frame is available.")
            return
        logger.info("Capturing frame for grading.")
        _grade_frame(frame)

    def _grade_frame(rgb: np.ndarray) -> None:
        try:
            from bubble_mark.processing.analyzer import BubbleAnalyzer
            from bubble_mark.processing.detector import BubbleSheetDetector
            from bubble_mark.processing.grader import BubbleSheetGrader
        except ImportError:
            msg = "Image processing is not available on this platform."
            status_label.text = msg
            logger.error(msg)
            return

        if app.answer_key is None:
            msg = "Please configure an answer key in Settings first."
            status_label.text = msg
            logger.warning(msg)
            return

        bgr = rgb[:, :, ::-1].copy()  # RGB → BGR (pure numpy)
        detector = BubbleSheetDetector(app.app_settings.layout_config)
        analyzer = BubbleAnalyzer(app.app_settings.fill_threshold)
        grader = BubbleSheetGrader(app.answer_key, detector, analyzer)
        try:
            result = grader.grade_image(bgr)
            if result is None:
                msg = "Could not detect a bubble sheet in the image."
                status_label.text = msg
                logger.warning(msg)
                return
            app.results = [result]
            msg = f"Graded: {result.num_correct}/{result.num_questions} correct."
            status_label.text = msg
            logger.info(msg)
            # Stop the camera before navigating away so the CameraX session
            # is not left running against a discarded screen.
            from bubble_mark.ui.camerax_bridge import stop_camera as _stop
            _stop()
            app.go_results()
        except Exception as exc:
            logger.exception("Grading failed: %s", exc)
            status_label.text = f"Grading failed: {exc}"

    def stop_camera(widget: toga.Widget) -> None:
        from bubble_mark.ui.camerax_bridge import stop_camera as _stop
        _stop()
        logger.info("Camera stopped by user.")
        status_label.text = "Camera stopped."

    # ------------------------------------------------------------------ #
    # Layout                                                               #
    # ------------------------------------------------------------------ #

    btn_row = toga.Box(style=Pack(direction=ROW, padding_bottom=8, gap=8))
    btn_row.add(
        toga.Button("Open Camera", on_press=open_camera, style=Pack(flex=1)),
        toga.Button("Capture", on_press=capture, style=Pack(flex=1)),
        toga.Button("Stop", on_press=stop_camera, style=Pack(flex=1)),
    )

    btn_back = toga.Button("Back", on_press=lambda w: app.go_home())

    box.add(title, image_view, status_label, btn_row, btn_back)
    return box
