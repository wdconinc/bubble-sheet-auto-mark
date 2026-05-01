"""Home screen: main landing page."""
from __future__ import annotations

from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

Builder.load_string("""
<HomeScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 10

        Label:
            text: 'Bubble Sheet Auto-Mark'
            font_size: '24sp'
            size_hint_y: 0.2

        Button:
            text: 'Import Image(s)'
            on_press: app.go_camera()

        Button:
            text: 'Open Camera'
            on_press: app.go_camera()

        Button:
            text: 'Settings'
            on_press: app.go_settings()

        Button:
            text: 'View Results'
            on_press: app.go_results()
""")


class HomeScreen(Screen):
    """Landing screen with navigation buttons."""
    pass
