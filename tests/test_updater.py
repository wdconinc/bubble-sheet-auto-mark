"""Tests for bubble_mark.updater."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import bubble_mark.updater as updater_module
from bubble_mark.updater import get_latest_release, is_update_available


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
# get_latest_release
# ---------------------------------------------------------------------------

class TestGetLatestRelease:
    def test_returns_version_and_apk_url(self):
        data = {
            "tag_name": "v1.2.3",
            "assets": [
                {"name": "app-release.apk", "browser_download_url": "https://example.com/app.apk"},
                {"name": "checksums.txt", "browser_download_url": "https://example.com/checksums.txt"},
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
                {"name": "source.tar.gz", "browser_download_url": "https://example.com/src.tar.gz"},
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
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            version, apk_url = get_latest_release()
        assert version == "0.0.0"
        assert apk_url is None

    def test_returns_fallback_on_missing_tag_name(self):
        data = {"assets": []}  # no tag_name key
        with patch("urllib.request.urlopen", return_value=_make_response(data)):
            version, apk_url = get_latest_release()
        assert version == "0.0.0"
        assert apk_url is None

    def test_returns_fallback_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not-valid-json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            version, apk_url = get_latest_release()
        assert version == "0.0.0"
        assert apk_url is None

    def test_returns_first_apk_asset(self):
        data = {
            "tag_name": "v1.0.0",
            "assets": [
                {"name": "app-debug.apk", "browser_download_url": "https://example.com/debug.apk"},
                {"name": "app-release.apk", "browser_download_url": "https://example.com/release.apk"},
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
        with patch("bubble_mark.updater.get_latest_release", return_value=("9.9.9", "https://example.com/app.apk")):
            available, apk_url = is_update_available()
        assert available is True
        assert apk_url == "https://example.com/app.apk"

    def test_no_update_when_versions_match(self):
        current = updater_module.CURRENT_VERSION
        with patch("bubble_mark.updater.get_latest_release", return_value=(current, "https://example.com/app.apk")):
            available, apk_url = is_update_available()
        assert available is False

    def test_no_update_when_latest_is_older(self):
        # A server returning a downgraded version must not prompt an update.
        with patch("bubble_mark.updater.CURRENT_VERSION", "1.0.0"):
            with patch("bubble_mark.updater.get_latest_release", return_value=("0.0.1", None)):
                available, _ = is_update_available()
        assert available is False

    def test_partial_version_treated_as_padded(self):
        # "1.2" should compare equal to "1.2.0" and not trigger an update.
        with patch("bubble_mark.updater.CURRENT_VERSION", "1.2.0"):
            with patch("bubble_mark.updater.get_latest_release", return_value=("1.2", None)):
                available, _ = is_update_available()
        assert available is False

    def test_no_update_on_network_error(self):
        # get_latest_release returns "0.0.0" on error; "0.0.0" is older than
        # the current version so no spurious update prompt is shown.
        with patch("bubble_mark.updater.get_latest_release", return_value=("0.0.0", None)):
            available, apk_url = is_update_available()
        assert available is False
        assert apk_url is None

    def test_apk_url_propagated(self):
        with patch("bubble_mark.updater.get_latest_release", return_value=("2.0.0", "https://example.com/app.apk")):
            _, apk_url = is_update_available()
        assert apk_url == "https://example.com/app.apk"
