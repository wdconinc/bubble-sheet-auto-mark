"""Home screen: main landing page."""
from __future__ import annotations

from typing import TYPE_CHECKING

import toga
from toga.style import Pack
from toga.style.pack import COLUMN

if TYPE_CHECKING:
    from bubble_mark.ui.app import BubbleMarkApp


def build_home_screen(app: BubbleMarkApp) -> toga.Box:
    """Return a Box containing the home screen UI."""
    box = toga.Box(style=Pack(direction=COLUMN, padding=20))

    title = toga.Label(
        "Bubble Sheet Auto-Mark",
        style=Pack(padding_bottom=20, font_size=20, text_align="center"),
    )
    btn_import = toga.Button(
        "Import Image(s)",
        on_press=lambda w: app.go_camera(),
        style=Pack(padding_bottom=10),
    )
    btn_camera = toga.Button(
        "Open Camera",
        on_press=lambda w: app.go_camera(),
        style=Pack(padding_bottom=10),
    )
    btn_settings = toga.Button(
        "Settings",
        on_press=lambda w: app.go_settings(),
        style=Pack(padding_bottom=10),
    )
    btn_results = toga.Button(
        "View Results",
        on_press=lambda w: app.go_results(),
        style=Pack(padding_bottom=10),
    )

    box.add(title, btn_import, btn_camera, btn_settings, btn_results)
    return box
