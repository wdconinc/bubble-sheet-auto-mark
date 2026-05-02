"""Settings screen: configure layout, answer key, thresholds."""
from __future__ import annotations

from typing import TYPE_CHECKING

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

if TYPE_CHECKING:
    from bubble_mark.ui.app import BubbleMarkApp


def build_settings_screen(app: BubbleMarkApp) -> toga.Box:
    """Return a Box containing the settings screen UI."""
    box = toga.Box(style=Pack(direction=COLUMN, padding=15))

    title = toga.Label(
        "Settings",
        style=Pack(padding_bottom=10, font_size=18),
    )

    def _row(label_text: str, input_widget: toga.Widget) -> toga.Box:
        row = toga.Box(style=Pack(direction=ROW, padding_bottom=6))
        row.add(
            toga.Label(label_text, style=Pack(flex=1)),
            input_widget,
        )
        return row

    inp_questions = toga.TextInput(value="30", style=Pack(width=80))
    inp_choices = toga.TextInput(value="5", style=Pack(width=80))
    inp_id_digits = toga.TextInput(value="9", style=Pack(width=80))
    inp_threshold = toga.TextInput(value="0.5", style=Pack(width=80))
    inp_answer_key = toga.TextInput(
        placeholder="e.g. 134521345213452...",
        style=Pack(padding_bottom=10),
    )
    status_label = toga.Label("", style=Pack(padding_bottom=6))

    def save_settings(widget: toga.Widget) -> None:
        from bubble_mark.models.answer_key import AnswerKey
        from bubble_mark.models.settings import AppSettings

        try:
            layout = {
                "num_questions": int(inp_questions.value or 30),
                "num_choices": int(inp_choices.value or 5),
                "num_id_digits": int(inp_id_digits.value or 9),
                "id_choices_per_digit": 10,
            }
            threshold = float(inp_threshold.value or 0.5)
            app.app_settings = AppSettings(
                layout_config=layout,
                fill_threshold=threshold,
            )
            key_text = inp_answer_key.value.strip()
            if key_text:
                app.answer_key = AnswerKey(key_text)
            status_label.text = "Settings saved."
        except Exception as exc:
            status_label.text = f"Error: {exc}"

    btn_row = toga.Box(style=Pack(direction=ROW, padding_top=6))
    btn_row.add(
        toga.Button("Save Settings", on_press=save_settings, style=Pack(flex=1, padding_right=5)),
        toga.Button("Back", on_press=lambda w: app.go_home(), style=Pack(flex=1)),
    )

    box.add(
        title,
        _row("Number of Questions", inp_questions),
        _row("Number of Choices (A-E=5)", inp_choices),
        _row("ID Digits", inp_id_digits),
        _row("Fill Threshold (0-1)", inp_threshold),
        toga.Label("Answer Key (string of 1-5 or A-E)", style=Pack(padding_bottom=4)),
        inp_answer_key,
        status_label,
        btn_row,
    )
    return box
