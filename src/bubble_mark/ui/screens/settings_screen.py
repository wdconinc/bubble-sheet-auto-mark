"""Settings screen: configure layout, answer key, thresholds, and sheet regions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

if TYPE_CHECKING:
    from bubble_mark.ui.app import BubbleMarkApp

logger = logging.getLogger(__name__)


def _parse_region(text: str) -> list | None:
    """Parse a region string ``"x1, y1, x2, y2"`` into a validated list or *None*."""
    from bubble_mark.models.settings import _validate_region

    try:
        parts = [float(v.strip()) for v in text.split(",")]
        if len(parts) != 4:
            return None
        return _validate_region(parts)
    except ValueError:
        return None


def build_settings_screen(app: "BubbleMarkApp") -> toga.Box:
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

    # Pre-populate from current settings
    s = app.app_settings
    if s.layout_config:
        inp_questions.value = str(s.layout_config.get("num_questions", 30))
        inp_choices.value = str(s.layout_config.get("num_choices", 5))
        inp_id_digits.value = str(s.layout_config.get("num_id_digits", 9))
    inp_threshold.value = str(s.fill_threshold)

    # ── Reference sheet section ──────────────────────────────────────────

    ref_label = toga.Label(
        "Reference Sheet",
        style=Pack(padding_top=10, padding_bottom=4, font_size=14),
    )
    ref_hint = toga.Label(
        "Upload a blank sheet to define answer and ID regions for improved detection.",
        style=Pack(padding_bottom=6),
    )

    ref_path_label = toga.Label(
        s.reference_image_path or "No file selected.",
        style=Pack(padding_bottom=6),
    )

    def _pick_ref_image(widget: toga.Widget) -> None:
        async def _pick() -> None:
            try:
                result = await app.main_window.open_file_dialog(
                    title="Select blank bubble sheet",
                    file_types=["jpg", "jpeg", "png", "bmp"],
                )
            except Exception as exc:
                logger.exception("File dialog raised an unexpected error: %s", exc)
                status_label.text = "Error opening file dialog."
                return
            if result is not None:
                ref_path_label.text = str(result)

        asyncio.ensure_future(_pick())

    btn_ref = toga.Button(
        "Upload Blank Sheet",
        on_press=_pick_ref_image,
        style=Pack(padding_bottom=6),
    )

    region_hint = toga.Label(
        'Regions as "x1, y1, x2, y2" fractions (0–1). '
        "Example answer region: 0, 0, 1, 0.72  |  ID region: 0, 0.74, 1, 1",
        style=Pack(padding_bottom=6),
    )

    def _region_value(region: list | None) -> str:
        if region is None:
            return ""
        return ", ".join(f"{v:.3f}" for v in region)

    inp_answer_region = toga.TextInput(
        placeholder="0, 0, 1, 0.72",
        value=_region_value(s.answer_region),
        style=Pack(padding_bottom=6),
    )
    inp_id_region = toga.TextInput(
        placeholder="0, 0.74, 1, 1",
        value=_region_value(s.id_region),
        style=Pack(padding_bottom=6),
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

            ref_path = ref_path_label.text
            if ref_path == "No file selected.":
                ref_path = None

            answer_region = _parse_region(inp_answer_region.value)
            id_region = _parse_region(inp_id_region.value)

            app.app_settings = AppSettings(
                layout_config=layout,
                reference_image_path=ref_path,
                fill_threshold=threshold,
                answer_region=answer_region,
                id_region=id_region,
            )
            key_text = inp_answer_key.value.strip()
            if key_text:
                app.answer_key = AnswerKey(key_text)
            status_label.text = "Settings saved."
        except Exception as exc:
            status_label.text = f"Error: {exc}"

    btn_row = toga.Box(style=Pack(direction=ROW, padding_top=6))
    btn_row.add(
        toga.Button(
            "Save Settings", on_press=save_settings, style=Pack(flex=1, padding_right=5)
        ),
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
        ref_label,
        ref_hint,
        btn_ref,
        ref_path_label,
        region_hint,
        _row("Answer Region (x1,y1,x2,y2)", inp_answer_region),
        _row("ID Region (x1,y1,x2,y2)", inp_id_region),
        status_label,
        btn_row,
    )
    return box
