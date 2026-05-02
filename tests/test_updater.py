"""Tests for bubble_mark.updater."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import bubble_mark.updater as updater_module
from bubble_mark.updater import (
    _parse_version,
    check_for_updates,
    get_latest_release,
    is_update_available,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(data: dict) -> MagicMock:
    """Return a mock urllib response whose .read() returns JSON bytes."""
    body = json.dumps(data).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------


class TestParseVersion:
    def test_plain_release(self):
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_pads_short_version(self):
        assert _parse_version("1.2") == (1, 2, 0)

    def test_dev_version_stripped(self):
        # setuptools-scm produces versions like "0.2.1.dev3+gabcdef"
        assert _parse_version("0.2.1.dev3+gabcdef") == (0, 2, 1)

    def test_dev_only_fallback(self):
        # "0.0.0.dev0" is the configured fallback_version for untagged checkouts
        assert _parse_version("0.0.0.dev0") == (0, 0, 0)

    def test_local_segment_stripped(self):
        assert _parse_version("1.0.0+local") == (1, 0, 0)

    def test_malformed_returns_zero(self):
        assert _parse_version("not-a-version") == (0, 0, 0)

    def test_empty_string(self):
        assert _parse_version("") == (0, 0, 0)


# ---------------------------------------------------------------------------
# get_latest_release
# ---------------------------------------------------------------------------


class TestGetLatestRelease:
    def test_returns_version_and_apk_url(self):
        data = {
            "tag_name": "v1.2.3",
            "assets": [
                {
                    "name": "app-release.apk",
                    "browser_download_url": "https://example.com/app.apk",
                },
                {
                    "name": "checksums.txt",
                    "browser_download_url": "https://example.com/checksums.txt",
                },
            ],
        }
        with patch("urllib.request.urlopen", return_value=_make_response(data)):
            version, apk_url = get_latest_release()
        assert version == "1.2.3"
        assert apk_url == "https://example.com/app.apk"

    def test_strips_v_prefix(self):
        data = {"tag_name": "v2.0.0", "assets": []}
        with patch("urllib.request.urlopen", return_value=_make_response(data)):
            version, _ = get_latest_release()
        assert version == "2.0.0"

    def test_no_v_prefix(self):
        data = {"tag_name": "3.1.0", "assets": []}
        with patch("urllib.request.urlopen", return_value=_make_response(data)):
            version, _ = get_latest_release()
        assert version == "3.1.0"

    def test_apk_url_none_when_no_apk_asset(self):
        data = {
            "tag_name": "v1.0.0",
            "assets": [
                {
                    "name": "source.tar.gz",
                    "browser_download_url": "https://example.com/src.tar.gz",
                },
            ],
        }
        with patch("urllib.request.urlopen", return_value=_make_response(data)):
            _, apk_url = get_latest_release()
        assert apk_url is None

    def test_apk_url_none_when_assets_missing(self):
        data = {"tag_name": "v1.0.0"}
        with patch("urllib.request.urlopen", return_value=_make_response(data)):
            _, apk_url = get_latest_release()
        assert apk_url is None

    def test_returns_fallback_on_url_error(self):
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")
        ):
            version, apk_url = get_latest_release()
        assert version is None
        assert apk_url is None

    def test_returns_fallback_on_missing_tag_name(self):
        data = {"assets": []}  # no tag_name key
        with patch("urllib.request.urlopen", return_value=_make_response(data)):
            version, apk_url = get_latest_release()
        assert version is None
        assert apk_url is None

    def test_returns_fallback_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not-valid-json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            version, apk_url = get_latest_release()
        assert version is None
        assert apk_url is None

    def test_returns_first_apk_asset(self):
        data = {
            "tag_name": "v1.0.0",
            "assets": [
                {
                    "name": "app-debug.apk",
                    "browser_download_url": "https://example.com/debug.apk",
                },
                {
                    "name": "app-release.apk",
                    "browser_download_url": "https://example.com/release.apk",
                },
            ],
        }
        with patch("urllib.request.urlopen", return_value=_make_response(data)):
            _, apk_url = get_latest_release()
        assert apk_url == "https://example.com/debug.apk"


# ---------------------------------------------------------------------------
# is_update_available
# ---------------------------------------------------------------------------


class TestIsUpdateAvailable:
    def test_update_available_when_latest_is_newer(self):
        with patch(
            "bubble_mark.updater.get_latest_release",
            return_value=("9.9.9", "https://example.com/app.apk"),
        ):
            available, apk_url = is_update_available()
        assert available is True
        assert apk_url == "https://example.com/app.apk"

    def test_no_update_when_versions_match(self):
        current = updater_module.CURRENT_VERSION
        with patch(
            "bubble_mark.updater.get_latest_release",
            return_value=(current, "https://example.com/app.apk"),
        ):
            available, apk_url = is_update_available()
        assert available is False

    def test_no_update_when_latest_is_older(self):
        # A server returning a downgraded version must not prompt an update.
        with patch("bubble_mark.updater.CURRENT_VERSION", "1.0.0"):
            with patch(
                "bubble_mark.updater.get_latest_release", return_value=("0.0.1", None)
            ):
                available, _ = is_update_available()
        assert available is False

    def test_partial_version_treated_as_padded(self):
        # "1.2" should compare equal to "1.2.0" and not trigger an update.
        with patch("bubble_mark.updater.CURRENT_VERSION", "1.2.0"):
            with patch(
                "bubble_mark.updater.get_latest_release", return_value=("1.2", None)
            ):
                available, _ = is_update_available()
        assert available is False

    def test_no_update_on_network_error(self):
        # get_latest_release returns (None, None) on error; no spurious update
        # prompt is shown and the function gracefully returns (False, None).
        with patch("bubble_mark.updater.get_latest_release", return_value=(None, None)):
            available, apk_url = is_update_available()
        assert available is False
        assert apk_url is None

    def test_no_update_when_current_is_dev_build(self):
        # A dev build of 1.0.0 should not be prompted to update to 1.0.0 release.
        with patch("bubble_mark.updater.CURRENT_VERSION", "1.0.0.dev3+gabcdef"):
            with patch(
                "bubble_mark.updater.get_latest_release", return_value=("1.0.0", None)
            ):
                available, _ = is_update_available()
        assert available is False

    def test_update_available_when_dev_build_has_newer_release(self):
        # A dev build of 1.0.0 should still be prompted to update to 1.1.0.
        with patch("bubble_mark.updater.CURRENT_VERSION", "1.0.0.dev3+gabcdef"):
            with patch(
                "bubble_mark.updater.get_latest_release",
                return_value=("1.1.0", "https://example.com/app.apk"),
            ):
                available, apk_url = is_update_available()
        assert available is True
        assert apk_url == "https://example.com/app.apk"

    def test_apk_url_propagated(self):
        with patch(
            "bubble_mark.updater.get_latest_release",
            return_value=("2.0.0", "https://example.com/app.apk"),
        ):
            _, apk_url = is_update_available()
        assert apk_url == "https://example.com/app.apk"


# ---------------------------------------------------------------------------
# check_for_updates
# ---------------------------------------------------------------------------

def _make_app_instance():
    """Return a minimal fake app instance with a mock event loop and window."""
    loop = MagicMock()
    loop.call_soon_threadsafe = MagicMock(side_effect=lambda fn: fn())
    loop.create_task = MagicMock()

    window = MagicMock()
    # dialog() returns a coroutine-like object; mock create_task to capture it
    window.dialog = MagicMock(return_value=MagicMock())

    app = MagicMock()
    app.loop = loop
    app.main_window = window
    return app


def _mock_toga_context():
    """Return a context manager that injects a MagicMock for the toga module."""
    import sys
    toga_mock = MagicMock()
    toga_mock.InfoDialog = MagicMock(return_value=MagicMock())
    toga_mock.QuestionDialog = MagicMock(return_value=MagicMock())
    return patch.dict(sys.modules, {"toga": toga_mock})


def _sync_thread_patch():
    """Return a patch that makes threading.Thread execute its target synchronously.

    This eliminates the race condition in tests that call check_for_updates()
    and then immediately assert on mock state: without this patch the worker
    thread may not have finished (or even started) before the assertion runs.
    """
    import threading as _threading

    class _SyncThread(_threading.Thread):
        def start(self):
            if self._target:
                self._target(*self._args, **self._kwargs)

    return patch("threading.Thread", _SyncThread)


class TestCheckForUpdates:
    def test_shows_up_to_date_when_no_newer_version(self):
        app = _make_app_instance()
        with _mock_toga_context(), _sync_thread_patch():
            with patch("bubble_mark.updater.get_latest_release", return_value=("0.1.0", None)):
                with patch("bubble_mark.updater.CURRENT_VERSION", "0.1.0"):
                    check_for_updates(app)

        # A task should have been scheduled on the loop
        app.loop.create_task.assert_called_once()

    def test_shows_update_available_when_newer_version(self):
        app = _make_app_instance()
        with _mock_toga_context(), _sync_thread_patch():
            with patch("bubble_mark.updater.get_latest_release", return_value=("9.9.9", "https://example.com/app.apk")):
                check_for_updates(app)

        app.loop.create_task.assert_called_once()

    def test_shows_error_when_network_fails(self):
        app = _make_app_instance()
        with _mock_toga_context(), _sync_thread_patch():
            with patch("bubble_mark.updater.get_latest_release", return_value=(None, None)):
                check_for_updates(app)

        app.loop.create_task.assert_called_once()

    def test_spawns_daemon_thread(self):
        """check_for_updates must not block the UI thread."""
        import threading as _threading

        app = _make_app_instance()
        spawned: list = []

        class _CapturingThread(_threading.Thread):
            def start(self):
                spawned.append(self)
                super().start()

        with _mock_toga_context():
            with patch("bubble_mark.updater.get_latest_release", return_value=("0.1.0", None)):
                with patch("bubble_mark.updater.CURRENT_VERSION", "0.1.0"):
                    with patch("threading.Thread", _CapturingThread):
                        check_for_updates(app)

        assert len(spawned) == 1
        assert spawned[0].daemon is True

    def test_opens_browser_on_desktop_when_update_accepted(self):
        """On non-Android platforms the releases page should open in browser."""
        import asyncio

        app = _make_app_instance()

        # Capture the coroutine passed to create_task
        captured_coro = []
        app.loop.create_task.side_effect = lambda coro: captured_coro.append(coro)

        with _mock_toga_context(), _sync_thread_patch():
            with patch("bubble_mark.updater.get_latest_release", return_value=("9.9.9", "https://example.com/app.apk")):
                with patch("bubble_mark.updater.CURRENT_VERSION", "0.1.0"):
                    with patch("bubble_mark.updater.importlib.util.find_spec", return_value=None):
                        check_for_updates(app)

        assert len(captured_coro) == 1

        # dialog() is awaited in the coroutine, so it must return an awaitable.
        async def _dialog_accept(_arg):
            return True

        with patch("bubble_mark.updater.webbrowser.open") as mock_open:
            app.main_window.dialog.side_effect = _dialog_accept

            asyncio.run(captured_coro[0])

            mock_open.assert_called_once_with(
                f"https://github.com/{updater_module.GITHUB_REPO}/releases/latest"
            )