"""In-app update checker for Bubble Sheet Auto-Mark."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from typing import Any

try:
    from bubble_mark import __version__ as CURRENT_VERSION
except ImportError:
    CURRENT_VERSION = "0.1.0"

GITHUB_REPO = "wdconinc/bubble-sheet-auto-mark"
_GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


def get_latest_release() -> tuple[str | None, str | None]:
    """Query the GitHub Releases API and return ``(latest_version, apk_url)``.

    Returns ``(None, None)`` on any network or HTTP error so callers can
    distinguish a genuine error from a real ``v0.0.0`` release.
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
        return None, None


def _parse_version(v: str) -> tuple[int, int, int]:
    """Parse a dotted-version string into a normalised 3-tuple of ints.

    Strips PEP 440 pre-release, dev, and local segments before parsing
    (e.g. ``"1.2.3.dev4+gabcdef"`` → ``(1, 2, 3)``).  Pads short versions
    (e.g. ``"1.2"`` → ``(1, 2, 0)``) and returns ``(0, 0, 0)`` for any
    malformed input so comparisons remain consistent.
    """
    # Extract the leading numeric release segment, discarding any
    # pre/dev/post/local suffix that starts with a non-digit character.
    release = re.split(r"[^0-9.]", v)[0].rstrip(".")
    if not release:
        return (0, 0, 0)
    parts = (release.split(".") + ["0", "0", "0"])[:3]
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return (0, 0, 0)


def is_update_available() -> tuple[bool, str | None]:
    """Return ``(update_available, apk_url)`` by comparing against the latest release.

    Returns ``(False, None)`` when the release API is unreachable.
    """
    latest_version, apk_url = get_latest_release()
    if latest_version is None:
        return False, None
    return _parse_version(latest_version) > _parse_version(CURRENT_VERSION), apk_url


def _schedule_coro(app_instance: Any, coro: Any) -> None:
    """Schedule an async *coro* on the app's event loop from a worker thread."""
    app_instance.loop.call_soon_threadsafe(
        lambda: app_instance.loop.create_task(coro)
    )


def _handle_update_result(
    app_instance: Any,
    latest_version: str | None,
    apk_url: str | None,
    *,
    silent: bool,
) -> None:
    """Decide which dialog to show (if any) based on the update-check result.

    Called from a worker thread.  When *silent* is ``True`` only the
    "update available" prompt is shown; "up to date" and error outcomes are
    silently ignored (legacy Android startup-check behaviour).

    When *silent* is ``False`` every outcome surfaces a dialog so the user
    always receives clear feedback from an explicit check.
    """
    import toga  # optional dependency; imported here so tests stay headless

    on_android = (
        sys.platform == "linux" and importlib.util.find_spec("android") is not None
    )

    if latest_version is None:
        # Network or API error
        if silent:
            return

        async def _show_error() -> None:
            await app_instance.main_window.dialog(
                toga.InfoDialog(
                    "Update Check Failed",
                    "Could not retrieve update information.\n"
                    "Please check your network connection and try again.",
                )
            )

        _schedule_coro(app_instance, _show_error())
        return

    update_available = _parse_version(latest_version) > _parse_version(CURRENT_VERSION)

    if not update_available:
        if silent:
            return

        async def _show_up_to_date() -> None:
            await app_instance.main_window.dialog(
                toga.InfoDialog(
                    "No Update Available",
                    f"You are already running the latest version ({CURRENT_VERSION}).",
                )
            )

        _schedule_coro(app_instance, _show_up_to_date())
        return

    # An update is available.
    # Silent (startup) mode on Android requires an APK URL to be actionable.
    if silent and not apk_url:
        return

    async def _show_update_available() -> None:
        result = await app_instance.main_window.dialog(
            toga.QuestionDialog(
                "Update Available",
                f"A new version ({latest_version}) is available.\n"
                "Would you like to download it now?",
            )
        )
        if result:
            if on_android and apk_url:
                _open_url(apk_url)
            else:
                webbrowser.open(f"https://github.com/{GITHUB_REPO}/releases/latest")

    _schedule_coro(app_instance, _show_update_available())


def check_and_prompt_update(app_instance: Any) -> None:
    """Check for an update and, on Android, prompt the user to download it.

    On non-Android platforms this function returns immediately without
    showing any UI so desktop development is unaffected.
    """
    on_android = sys.platform == "linux" and importlib.util.find_spec("android") is not None
    if not on_android:
        return

    def _check() -> None:
        latest_version, apk_url = get_latest_release()
        _handle_update_result(app_instance, latest_version, apk_url, silent=True)

    threading.Thread(target=_check, daemon=True).start()


def check_for_updates(app_instance: Any) -> None:
    """Explicitly check for updates and show feedback to the user.

    Unlike :func:`check_and_prompt_update`, this function is intended to be
    called from an explicit user action (e.g. a menu item) and therefore
    always shows a result dialog on all platforms:

    * If an update is available the user is asked whether to download it.
      On Android the APK is opened via the Intent system; on other platforms
      the releases page is opened in the default browser.
    * If no update is available a short "up to date" information dialog is
      shown so the user gets clear feedback.
    * Network or API errors are surfaced as an information dialog rather than
      being swallowed silently.
    """
    def _check() -> None:
        latest_version, apk_url = get_latest_release()
        _handle_update_result(app_instance, latest_version, apk_url, silent=False)

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
