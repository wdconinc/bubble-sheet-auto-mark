"""Camera / image import screen."""
from __future__ import annotations

from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

Builder.load_string("""
<CameraScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 8

        Label:
            text: 'Camera / Import'
            font_size: '20sp'
            size_hint_y: 0.08

        Label:
            id: status_label
            text: 'No images captured'
            size_hint_y: 0.1

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.12
            spacing: 8
            Button:
                text: 'Capture / Import'
                on_press: root.import_image()
            Button:
                text: 'Process All'
                on_press: root.process_images()

        Button:
            text: 'Back'
            size_hint_y: 0.08
            on_press: app.go_home()
""")


class CameraScreen(Screen):
    """Screen for capturing or importing images of bubble sheets."""

    def import_image(self):
        """Open a file chooser to import an image (stub for desktop)."""
        self.ids.status_label.text = "Import not yet implemented in this build."

    def process_images(self):
        """Process captured/imported images through the grading pipeline."""
        from bubble_mark.processing.analyzer import BubbleAnalyzer
        from bubble_mark.processing.detector import BubbleSheetDetector
        from bubble_mark.processing.grader import BubbleSheetGrader
        from kivy.app import App as KivyApp

        app = KivyApp.get_running_app()

        if app is None or app.answer_key is None:
            self.ids.status_label.text = "Please configure an answer key in Settings first."
            return

        detector = BubbleSheetDetector(app.settings.layout_config)
        analyzer = BubbleAnalyzer(app.settings.fill_threshold)
        grader = BubbleSheetGrader(app.answer_key, detector, analyzer)

        self.ids.status_label.text = "Processing complete (no images loaded)."
