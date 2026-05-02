"""In-app update checker for Bubble Sheet Auto-Mark."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
import webbrowser

try:
    from bubble_mark import __version__ as CURRENT_VERSION
except ImportError:
    CURRENT_VERSION = "0.1.0"

GITHUB_REPO = "wdconinc/bubble-sheet-auto-mark"
_GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


def get_latest_release() -> tuple[str, str | None]:
    """Query the GitHub Releases API and return (latest_version, apk_url).

    Returns ``("0.0.0", None)`` on any network or HTTP error so the app
    never crashes when offline or when the API is unavailable.
    """
    try:
        req = urllib.request.Request(
            _GITHUB_API_URL,
            headers={
                "User-Agent": f"bubble-sheet-auto-mark/{CURRENT_VERSION}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data: dict = json.loads(resp.read())
        latest_version: str = data["tag_name"].lstrip("v")
        apk_url: str | None = next(
            (
                asset["browser_download_url"]
                for asset in data.get("assets", [])
                if asset["name"].endswith(".apk")
            ),
            None,
        )
        return latest_version, apk_url
    except (urllib.error.URLError, KeyError, ValueError):
        return "0.0.0", None


def _parse_version(v: str) -> tuple[int, int, int]:
    """Parse a dotted-version string into a normalised 3-tuple of ints.

    Pads short versions (e.g. ``"1.2"`` → ``(1, 2, 0)``) and returns
    ``(0, 0, 0)`` for any malformed input so comparisons remain consistent.
    """
    parts = (v.split(".") + ["0", "0", "0"])[:3]
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return (0, 0, 0)


def is_update_available() -> tuple[bool, str | None]:
    """Return ``(update_available, apk_url)`` by comparing against the latest release."""
    latest_version, apk_url = get_latest_release()
    return _parse_version(latest_version) > _parse_version(CURRENT_VERSION), apk_url


def check_and_prompt_update(app_instance) -> None:  # noqa: ANN001
    """Check for an update and, on Android, prompt the user to download it.

    On non-Android platforms this function returns immediately without
    showing any UI so desktop development is unaffected.
    """
    on_android = sys.platform == "linux" and importlib.util.find_spec("android") is not None
    if not on_android:
        return

    def _check() -> None:
        update_available, apk_url = is_update_available()
        if not update_available or not apk_url:
            return

        async def _show_dialog() -> None:
            result = await app_instance.main_window.dialog(
                toga.QuestionDialog(
                    "Update Available",
                    "A new version is available.\nWould you like to download it?",
                )
            )
            if result:
                _open_url(apk_url)

        import toga
        app_instance.loop.call_soon_threadsafe(
            lambda: app_instance.loop.create_task(_show_dialog())
        )

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
