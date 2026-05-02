"""BubbleMarkApp – main Toga application class."""
from __future__ import annotations

import logging

import toga
from toga.style import Pack
from toga.style.pack import COLUMN

from bubble_mark.models.answer_key import AnswerKey
from bubble_mark.models.grade_result import GradeResult
from bubble_mark.models.settings import AppSettings

logger = logging.getLogger(__name__)


class BubbleMarkApp(toga.App):
    """Main Toga application for Bubble Sheet Auto-Mark."""

    def startup(self) -> None:
        # In-memory state (privacy-first: nothing leaves device)
        self.app_settings: AppSettings = AppSettings.default()
        self.answer_key: AnswerKey | None = None
        self.results: list[GradeResult] = []

        # ── Logging handler (captures records from all modules) ───────────
        from bubble_mark.ui.log_handler import StatusBarHandler
        self._log_handler = StatusBarHandler()
        self._log_handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(self._log_handler)
        # Ensure INFO-level records reach the status bar.  We lower the root
        # level only when it still has no handlers other than our own (i.e. the
        # app has not set up its own logging configuration), or when the level
        # would suppress INFO messages entirely.
        if not any(h for h in root_logger.handlers if h is not self._log_handler) \
                or root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

        # ── Status bar (collapsible log drawer) ───────────────────────────
        from bubble_mark.ui.status_bar import LogStatusBar
        self._status_bar = LogStatusBar(self, self._log_handler)

        # ── Persistent wrapper layout ─────────────────────────────────────
        # The wrapper stays as main_window.content for the lifetime of the
        # app.  Only the _screen_area portion is swapped during navigation.
        self._screen_area = toga.Box(style=Pack(direction=COLUMN, flex=1))
        self._current_screen: toga.Box | None = None
        self._wrapper = toga.Box(style=Pack(direction=COLUMN, flex=1))
        self._wrapper.add(self._screen_area, self._status_bar.widget)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = self._wrapper

        # ── Help → Check for Updates menu command ─────────────────────────
        check_updates_cmd = toga.Command(
            lambda _: self._check_for_updates(),
            text="Check for Updates…",
            group=toga.Group.HELP,
        )
        self.commands.add(check_updates_cmd)

        self._set_screen(self._build_home())
        self.main_window.show()

        logger.info("BubbleMarkApp started.")

        # Schedule update check 2 s after startup (daemon thread so it
        # doesn't keep the process alive on shutdown)
        import threading
        t = threading.Timer(2.0, self._trigger_update_check)
        t.daemon = True
        t.start()

    def _trigger_update_check(self) -> None:
        from bubble_mark.updater import check_and_prompt_update
        check_and_prompt_update(self)

    def _check_for_updates(self) -> None:
        from bubble_mark.updater import check_for_updates
        check_for_updates(self)

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

    def _set_screen(self, screen: toga.Box) -> None:
        """Replace the current screen in the persistent wrapper layout."""
        screen.style.flex = 1
        if self._current_screen is not None:
            self._screen_area.remove(self._current_screen)
        self._current_screen = screen
        self._screen_area.add(screen)

    def go_home(self) -> None:
        self._set_screen(self._build_home())

    def go_camera(self) -> None:
        self._set_screen(self._build_camera())

    def go_settings(self) -> None:
        self._set_screen(self._build_settings())

    def go_results(self) -> None:
        self._set_screen(self._build_results())


def main() -> BubbleMarkApp:
    """Entry point for Briefcase."""
    return BubbleMarkApp("Bubble Sheet Auto-Mark", "com.wdconinc.bubble_mark")
