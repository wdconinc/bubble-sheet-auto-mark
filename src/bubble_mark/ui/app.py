"""BubbleMarkApp – main Toga application class."""
from __future__ import annotations

import toga
from toga.style import Pack
from toga.style.pack import COLUMN

from bubble_mark.models.answer_key import AnswerKey
from bubble_mark.models.grade_result import GradeResult
from bubble_mark.models.settings import AppSettings


class BubbleMarkApp(toga.App):
    """Main Toga application for Bubble Sheet Auto-Mark."""

    def startup(self) -> None:
        # In-memory state (privacy-first: nothing leaves device)
        self.app_settings: AppSettings = AppSettings.default()
        self.answer_key: AnswerKey | None = None
        self.results: list[GradeResult] = []

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = self._build_home()
        self.main_window.show()

        # Schedule update check 2 s after startup
        import threading
        threading.Timer(2.0, self._trigger_update_check).start()

    def _trigger_update_check(self) -> None:
        from bubble_mark.updater import check_and_prompt_update
        check_and_prompt_update(self)

    # ------------------------------------------------------------------
    # Screen builders (each returns a toga.Box)
    # ------------------------------------------------------------------

    def _build_home(self) -> toga.Box:
        from bubble_mark.ui.screens.home_screen import build_home_screen
        return build_home_screen(self)

    def _build_camera(self) -> toga.Box:
        from bubble_mark.ui.screens.camera_screen import build_camera_screen
        return build_camera_screen(self)

    def _build_settings(self) -> toga.Box:
        from bubble_mark.ui.screens.settings_screen import build_settings_screen
        return build_settings_screen(self)

    def _build_results(self) -> toga.Box:
        from bubble_mark.ui.screens.results_screen import build_results_screen
        return build_results_screen(self)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def go_home(self) -> None:
        self.main_window.content = self._build_home()

    def go_camera(self) -> None:
        self.main_window.content = self._build_camera()

    def go_settings(self) -> None:
        self.main_window.content = self._build_settings()

    def go_results(self) -> None:
        self.main_window.content = self._build_results()


def main() -> BubbleMarkApp:
    """Entry point for Briefcase."""
    return BubbleMarkApp("Bubble Sheet Auto-Mark", "com.wdconinc.bubblemark")
