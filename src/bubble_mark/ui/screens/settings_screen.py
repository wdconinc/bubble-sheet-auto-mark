"""Settings screen: configure layout, answer key, thresholds."""
from __future__ import annotations

from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

Builder.load_string("""
<SettingsScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 15
        spacing: 8

        Label:
            text: 'Settings'
            font_size: '20sp'
            size_hint_y: 0.07

        GridLayout:
            cols: 2
            size_hint_y: 0.45
            spacing: 5

            Label:
                text: 'Number of Questions'
            TextInput:
                id: num_questions
                text: '30'
                input_filter: 'int'
                multiline: False

            Label:
                text: 'Number of Choices (A-E=5)'
            TextInput:
                id: num_choices
                text: '5'
                input_filter: 'int'
                multiline: False

            Label:
                text: 'ID Digits'
            TextInput:
                id: num_id_digits
                text: '9'
                input_filter: 'int'
                multiline: False

            Label:
                text: 'Fill Threshold (0-1)'
            TextInput:
                id: fill_threshold
                text: '0.5'
                multiline: False

        Label:
            text: 'Answer Key (string of 1-5 or A-E)'
            size_hint_y: 0.05

        TextInput:
            id: answer_key_input
            hint_text: 'e.g. 134521345213452...'
            size_hint_y: 0.12
            multiline: False

        Label:
            id: status_label
            text: ''
            size_hint_y: 0.05

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.1
            spacing: 8
            Button:
                text: 'Save Settings'
                on_press: root.save_settings()
            Button:
                text: 'Back'
                on_press: app.go_home()
""")


class SettingsScreen(Screen):
    """Screen for configuring application settings and answer key."""

    def save_settings(self):
        from kivy.app import App as KivyApp
        from bubble_mark.models.answer_key import AnswerKey
        from bubble_mark.models.settings import AppSettings

        app = KivyApp.get_running_app()
        try:
            layout = {
                "num_questions": int(self.ids.num_questions.text or 30),
                "num_choices": int(self.ids.num_choices.text or 5),
                "num_id_digits": int(self.ids.num_id_digits.text or 9),
                "id_choices_per_digit": 10,
            }
            threshold = float(self.ids.fill_threshold.text or 0.5)
            app.settings = AppSettings(
                layout_config=layout,
                fill_threshold=threshold,
            )
            key_text = self.ids.answer_key_input.text.strip()
            if key_text:
                app.answer_key = AnswerKey(key_text)
            self.ids.status_label.text = "Settings saved."
        except Exception as exc:
            self.ids.status_label.text = f"Error: {exc}"
