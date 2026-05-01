"""In-app update checker for Bubble Sheet Auto-Mark."""
from __future__ import annotations

import importlib.util
import sys
import threading
import webbrowser

import requests

try:
    from bubble_mark import __version__ as CURRENT_VERSION
except ImportError:
    CURRENT_VERSION = "0.1.0"

GITHUB_REPO = "wdconinc/bubble-sheet-auto-mark"


def get_latest_release() -> tuple[str, str | None]:
    """Query the GitHub Releases API and return (latest_version, apk_url).

    Returns ``("0.0.0", None)`` on any network or HTTP error so the app
    never crashes when offline or when the API is unavailable.
    """
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        latest_version = data["tag_name"].lstrip("v")
        apk_url: str | None = next(
            (
                asset["browser_download_url"]
                for asset in data.get("assets", [])
                if asset["name"].endswith(".apk")
            ),
            None,
        )
        return latest_version, apk_url
    except (requests.RequestException, KeyError, ValueError):
        return "0.0.0", None


def is_update_available() -> tuple[bool, str | None]:
    """Return ``(update_available, apk_url)`` by comparing against the latest release."""
    latest_version, apk_url = get_latest_release()
    return latest_version != CURRENT_VERSION, apk_url


def check_and_prompt_update(app_instance) -> None:  # noqa: ANN001
    """Check for an update and, on Android, prompt the user to download it.

    On non-Android platforms this function returns immediately without
    showing any UI so desktop development is unaffected.
    """
    # Only show the prompt when running on Android.
    on_android = sys.platform == "linux" and importlib.util.find_spec("android") is not None
    if not on_android:
        return

    def _check() -> None:
        update_available, apk_url = is_update_available()
        if not update_available or not apk_url:
            return

        from kivy.clock import Clock

        Clock.schedule_once(lambda dt: _show_popup(apk_url))

    def _show_popup(apk_url: str) -> None:
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.label import Label
        from kivy.uix.popup import Popup

        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        content.add_widget(
            Label(text="A new version is available.\nWould you like to download it?")
        )

        buttons = BoxLayout(size_hint_y=None, height=44, spacing=10)

        popup = Popup(
            title="Update Available",
            content=content,
            size_hint=(0.8, 0.4),
            auto_dismiss=False,
        )

        def _on_download(instance) -> None:  # noqa: ANN001
            popup.dismiss()
            _open_url(apk_url)

        def _on_later(instance) -> None:  # noqa: ANN001
            popup.dismiss()

        download_btn = Button(text="Download")
        download_btn.bind(on_release=_on_download)
        later_btn = Button(text="Later")
        later_btn.bind(on_release=_on_later)

        buttons.add_widget(download_btn)
        buttons.add_widget(later_btn)
        content.add_widget(buttons)

        popup.open()

    threading.Thread(target=_check, daemon=True).start()


def _open_url(url: str) -> None:
    """Open *url* using Android's Intent system, falling back to webbrowser."""
    try:
        from android.intent import Intent  # type: ignore[import]
        from android.net import Uri  # type: ignore[import]

        intent = Intent(Intent.ACTION_VIEW)
        intent.setData(Uri.parse(url))
        from android import activity  # type: ignore[import]

        activity.startActivity(intent)
    except Exception:
        webbrowser.open(url)
