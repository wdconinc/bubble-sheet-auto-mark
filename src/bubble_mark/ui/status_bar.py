"""Collapsible log-status-bar drawer for the bottom of the app.

The bar is always visible as a thin strip showing the most recent log line
and a toggle button.  Tapping "▲ Logs" pulls the panel up to occupy half
the screen (the screen-content area and the log panel each take ``flex=1``),
giving a half-screen drawer effect without requiring a native drawer widget.

Usage::

    from bubble_mark.ui.status_bar import LogStatusBar

    bar = LogStatusBar(app, handler)
    wrapper_box.add(screen_area, bar.widget)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

if TYPE_CHECKING:
    from bubble_mark.ui.app import BubbleMarkApp
    from bubble_mark.ui.log_handler import StatusBarHandler


class LogStatusBar:
    """Collapsible log-status bar that sits at the bottom of the app wrapper.

    When collapsed the bar shows only a single-line strip with the latest log
    message and an expand button.  When expanded the bar grows to share half
    the available vertical space with the screen-content area, showing a
    scrollable ``MultilineTextInput`` with all retained log lines.
    """

    def __init__(self, app: BubbleMarkApp, handler: StatusBarHandler) -> None:
        self._app = app
        self._handler = handler
        self._expanded = False

        # ── Toggle row (always visible) ──────────────────────────────────
        self._last_line_label = toga.Label(
            "No log messages yet.",
            style=Pack(flex=1, padding_left=6, padding_right=4, font_size=11),
        )
        self._toggle_btn = toga.Button(
            "▲ Logs",
            on_press=self._on_toggle,
            style=Pack(padding=4, font_size=11),
        )
        self._toggle_row = toga.Box(style=Pack(direction=ROW, padding=2))
        self._toggle_row.add(self._last_line_label, self._toggle_btn)

        # ── Log panel (hidden until first expansion) ─────────────────────
        self._log_text = toga.MultilineTextInput(
            readonly=True,
            style=Pack(flex=1),
        )
        self._log_panel = toga.Box(style=Pack(direction=COLUMN, flex=1))
        self._log_panel.add(self._log_text)

        # ── Outer container ───────────────────────────────────────────────
        # flex=0 keeps the bar at its natural (collapsed) height by default.
        self._container = toga.Box(style=Pack(direction=COLUMN, flex=0))
        self._container.add(self._toggle_row)

        # Register callback (invoked from the thread that emits the record).
        handler.add_callback(self._on_log_record)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def widget(self) -> toga.Box:
        """The root Toga widget to add to the app's wrapper box."""
        return self._container

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_log_record(self, line: str) -> None:
        """Called from any thread when a new log line is emitted."""

        def _update() -> None:
            self._last_line_label.text = line
            if self._expanded:
                self._log_text.value = "\n".join(self._handler.lines)

        self._app.loop.call_soon_threadsafe(_update)

    def _on_toggle(self, widget: toga.Widget) -> None:
        if self._expanded:
            self._container.remove(self._log_panel)
            self._container.style.flex = 0
            self._toggle_btn.text = "▲ Logs"
            self._expanded = False
        else:
            # Re-populate log text before revealing the panel.
            self._log_text.value = "\n".join(self._handler.lines)
            # Remove toggle_row so we can re-add it after the log panel.
            self._container.remove(self._toggle_row)
            self._container.add(self._log_panel)
            self._container.add(self._toggle_row)
            # flex=1 lets this container share space equally with screen_area.
            self._container.style.flex = 1
            self._toggle_btn.text = "▼ Logs"
            self._expanded = True
