"""Results screen: display and export grading results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

if TYPE_CHECKING:
    from bubble_mark.ui.app import BubbleMarkApp


def build_results_screen(app: BubbleMarkApp) -> toga.Box:
    """Return a Box containing the results screen UI."""
    box = toga.Box(style=Pack(direction=COLUMN, padding=10))

    title = toga.Label("Results", style=Pack(padding_bottom=8, font_size=18))
    status_label = toga.Label("", style=Pack(padding_bottom=6))

    table = toga.Table(
        headings=["Student ID", "Answers", "Score"],
        data=[
            (r.student_id, r.answers, f"{r.score:.1%}" if r.score is not None else "-")
            for r in app.results
        ],
        style=Pack(flex=1, padding_bottom=8),
    )

    def export_csv(widget: toga.Widget) -> None:
        from bubble_mark.export.csv_exporter import CSVExporter

        if not app.results:
            status_label.text = "No results to export."
            return
        try:
            path = "results.csv"
            CSVExporter().export(app.results, path)
            status_label.text = f"Exported to {path}"
        except Exception as exc:
            status_label.text = f"Export error: {exc}"

    def clear_results(widget: toga.Widget) -> None:
        app.results.clear()
        table.data = []
        status_label.text = "Results cleared."

    btn_row = toga.Box(style=Pack(direction=ROW, padding_top=4))
    btn_row.add(
        toga.Button(
            "Export CSV", on_press=export_csv, style=Pack(flex=1, padding_right=5)
        ),
        toga.Button(
            "Clear Results", on_press=clear_results, style=Pack(flex=1, padding_right=5)
        ),
        toga.Button("Back", on_press=lambda w: app.go_home(), style=Pack(flex=1)),
    )

    box.add(title, table, status_label, btn_row)
    return box
