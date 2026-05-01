"""Results screen: display and export grading results."""
from __future__ import annotations

from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

Builder.load_string("""
<ResultsScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 8

        Label:
            text: 'Results'
            font_size: '20sp'
            size_hint_y: 0.08

        ScrollView:
            size_hint_y: 0.72
            GridLayout:
                id: results_grid
                cols: 3
                size_hint_y: None
                height: self.minimum_height
                row_default_height: 30

        Label:
            id: status_label
            text: ''
            size_hint_y: 0.06

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.1
            spacing: 8
            Button:
                text: 'Export CSV'
                on_press: root.export_csv()
            Button:
                text: 'Clear Results'
                on_press: root.clear_results()
            Button:
                text: 'Back'
                on_press: app.go_home()
""")


class ResultsScreen(Screen):
    """Screen showing a list of graded results with export functionality."""

    def on_enter(self):
        self._refresh()

    def _refresh(self):
        from kivy.app import App as KivyApp
        from kivy.uix.label import Label

        app = KivyApp.get_running_app()
        grid = self.ids.results_grid
        grid.clear_widgets()

        # Header row
        for header in ("Student ID", "Answers", "Score"):
            grid.add_widget(Label(text=header, bold=True))

        for r in app.results:
            grid.add_widget(Label(text=r.student_id))
            grid.add_widget(Label(text=r.answers))
            score_txt = f"{r.score:.1%}" if r.score is not None else "-"
            grid.add_widget(Label(text=score_txt))

    def export_csv(self):
        from kivy.app import App as KivyApp
        from bubble_mark.export.csv_exporter import CSVExporter

        app = KivyApp.get_running_app()
        if not app.results:
            self.ids.status_label.text = "No results to export."
            return
        try:
            path = "results.csv"
            CSVExporter().export(app.results, path)
            self.ids.status_label.text = f"Exported to {path}"
        except Exception as exc:
            self.ids.status_label.text = f"Export error: {exc}"

    def clear_results(self):
        from kivy.app import App as KivyApp
        app = KivyApp.get_running_app()
        app.results.clear()
        self._refresh()
        self.ids.status_label.text = "Results cleared."
