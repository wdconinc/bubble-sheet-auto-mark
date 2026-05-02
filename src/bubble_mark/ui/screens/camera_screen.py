"""Camera / image import screen."""
from __future__ import annotations

from typing import TYPE_CHECKING

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

if TYPE_CHECKING:
    from bubble_mark.ui.app import BubbleMarkApp


def build_camera_screen(app: BubbleMarkApp) -> toga.Box:
    """Return a Box containing the camera/import screen UI."""
    box = toga.Box(style=Pack(direction=COLUMN, padding=10))

    title = toga.Label(
        "Camera / Import",
        style=Pack(padding_bottom=10, font_size=18),
    )
    status_label = toga.Label(
        "No images captured",
        style=Pack(padding_bottom=10),
    )

    def import_image(widget: toga.Widget) -> None:
        status_label.text = "Import not yet implemented in this build."

    def process_images(widget: toga.Widget) -> None:
        from bubble_mark.processing.analyzer import BubbleAnalyzer
        from bubble_mark.processing.detector import BubbleSheetDetector
        from bubble_mark.processing.grader import BubbleSheetGrader

        if app.answer_key is None:
            status_label.text = "Please configure an answer key in Settings first."
            return

        detector = BubbleSheetDetector(app.app_settings.layout_config)
        analyzer = BubbleAnalyzer(app.app_settings.fill_threshold)
        grader = BubbleSheetGrader(app.answer_key, detector, analyzer)  # noqa: F841
        status_label.text = "Processing complete (no images loaded)."

    btn_row = toga.Box(style=Pack(direction=ROW, padding_bottom=10))
    btn_row.add(
        toga.Button("Capture / Import", on_press=import_image, style=Pack(flex=1, padding_right=5)),
        toga.Button("Process All", on_press=process_images, style=Pack(flex=1)),
    )

    btn_back = toga.Button("Back", on_press=lambda w: app.go_home())

    box.add(title, status_label, btn_row, btn_back)
    return box
