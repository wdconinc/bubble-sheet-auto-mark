"""Tests for bubble_mark.updater."""
from __future__ import annotations

import json
import sys
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import bubble_mark.updater as updater_module
from bubble_mark.updater import CURRENT_VERSION, GITHUB_REPO, get_latest_release, is_update_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(body: dict, status: int = 200) -> MagicMock:
    """Return a mock that behaves like the context-manager returned by urlopen."""
    raw = json.dumps(body).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


_RELEASE_WITH_APK = {
    "tag_name": "v1.2.3",
    "assets": [
        {"name": "bubble_mark-1.2.3.apk", "browser_download_url": "https://example.com/app.apk"},
        {"name": "checksums.txt", "browser_download_url": "https://example.com/checksums.txt"},
    ],
}

_RELEASE_WITHOUT_APK = {
    "tag_name": "v1.2.3",
    "assets": [
        {"name": "checksums.txt", "browser_download_url": "https://example.com/checksums.txt"},
    ],
}


# ---------------------------------------------------------------------------
# get_latest_release()
# ---------------------------------------------------------------------------


class TestGetLatestRelease:
    def test_returns_version_and_apk_url(self):
        with patch("urllib.request.urlopen", return_value=_make_response(_RELEASE_WITH_APK)):
            version, apk_url = get_latest_release()

        assert version == "1.2.3"
        assert apk_url == "https://example.com/app.apk"

    def test_strips_leading_v_from_tag(self):
        with patch("urllib.request.urlopen", return_value=_make_response(_RELEASE_WITH_APK)):
            version, _ = get_latest_release()

        assert not version.startswith("v")

    def test_apk_url_is_none_when_no_apk_asset(self):
        with patch("urllib.request.urlopen", return_value=_make_response(_RELEASE_WITHOUT_APK)):
            version, apk_url = get_latest_release()

        assert version == "1.2.3"
        assert apk_url is None

    def test_returns_fallback_on_network_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            version, apk_url = get_latest_release()

        assert version == "0.0.0"
        assert apk_url is None

    def test_returns_fallback_on_http_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url=None, code=404, msg="Not Found", hdrs=None, fp=None
            ),
        ):
            version, apk_url = get_latest_release()

        assert version == "0.0.0"
        assert apk_url is None

    def test_returns_fallback_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not-json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            version, apk_url = get_latest_release()

        assert version == "0.0.0"
        assert apk_url is None

    def test_returns_fallback_when_tag_name_missing(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_make_response({"assets": []}),
        ):
            version, apk_url = get_latest_release()

        assert version == "0.0.0"
        assert apk_url is None

    def test_request_includes_user_agent_header(self):
        captured: list = []

        def _fake_urlopen(req, timeout=None):
            captured.append(req)
            return _make_response(_RELEASE_WITH_APK)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            get_latest_release()

        assert len(captured) == 1
        ua = captured[0].get_header("User-agent")
        assert ua is not None and "bubble-sheet-auto-mark" in ua

    def test_request_url_contains_repo(self):
        captured: list = []

        def _fake_urlopen(req, timeout=None):
            captured.append(req)
            return _make_response(_RELEASE_WITH_APK)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            get_latest_release()

        assert GITHUB_REPO in captured[0].full_url


# ---------------------------------------------------------------------------
# is_update_available()
# ---------------------------------------------------------------------------


class TestIsUpdateAvailable:
    def test_update_available_when_versions_differ(self):
        with patch(
            "bubble_mark.updater.get_latest_release",
            return_value=("9.9.9", "https://example.com/app.apk"),
        ):
            available, apk_url = is_update_available()

        assert available is True
        assert apk_url == "https://example.com/app.apk"

    def test_no_update_when_versions_match(self):
        with patch(
            "bubble_mark.updater.get_latest_release",
            return_value=(CURRENT_VERSION, "https://example.com/app.apk"),
        ):
            available, apk_url = is_update_available()

        assert available is False

    def test_propagates_apk_url_none(self):
        with patch(
            "bubble_mark.updater.get_latest_release",
            return_value=("9.9.9", None),
        ):
            available, apk_url = is_update_available()

        assert available is True
        assert apk_url is None

    def test_fallback_version_not_treated_as_update(self):
        """On network failure, '0.0.0' differs from CURRENT_VERSION ('0.1.0' or higher)."""
        with patch(
            "bubble_mark.updater.get_latest_release",
            return_value=("0.0.0", None),
        ):
            available, apk_url = is_update_available()

        # '0.0.0' != '0.1.0', so available is True — but apk_url is None,
        # meaning check_and_prompt_update will not show the popup.
        assert available is True
        assert apk_url is None
