"""BubbleMarkApp – main Kivy application class."""
from __future__ import annotations

import os
import sys

# Ensure src is on sys.path when app.py is the entry point
_src = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if _src not in sys.path:
    sys.path.insert(0, _src)

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, SlideTransition

from bubble_mark.models.answer_key import AnswerKey
from bubble_mark.models.grade_result import GradeResult
from bubble_mark.models.settings import AppSettings
from bubble_mark.ui.screens.camera_screen import CameraScreen
from bubble_mark.ui.screens.home_screen import HomeScreen
from bubble_mark.ui.screens.results_screen import ResultsScreen
from bubble_mark.ui.screens.settings_screen import SettingsScreen


class BubbleMarkApp(App):
    """Main Kivy application for Bubble Sheet Auto-Mark."""

    def build(self):
        # In-memory state (privacy-first: nothing leaves device)
        self.settings: AppSettings = AppSettings.default()
        self.answer_key: AnswerKey | None = None
        self.results: list[GradeResult] = []

        self.sm = ScreenManager(transition=SlideTransition())
        self.sm.add_widget(HomeScreen(name="home"))
        self.sm.add_widget(CameraScreen(name="camera"))
        self.sm.add_widget(SettingsScreen(name="settings"))
        self.sm.add_widget(ResultsScreen(name="results"))
        return self.sm

    # ------------------------------------------------------------------
    # Navigation helpers (called from screens)
    # ------------------------------------------------------------------

    def go_home(self):
        self.sm.current = "home"

    def go_camera(self):
        self.sm.current = "camera"

    def go_settings(self):
        self.sm.current = "settings"

    def go_results(self):
        self.sm.current = "results"
