"""Tests for bubble_mark.models.settings."""
from __future__ import annotations

import json
import os

import pytest

from bubble_mark.models.settings import AppSettings


class TestAppSettingsInit:
    def test_default_layout(self):
        s = AppSettings()
        assert s.layout_config["num_questions"] == 30
        assert s.layout_config["num_choices"] == 5
        assert s.layout_config["num_id_digits"] == 9
        assert s.layout_config["id_choices_per_digit"] == 10

    def test_custom_layout_merged(self):
        s = AppSettings(layout_config={"num_questions": 20})
        assert s.layout_config["num_questions"] == 20
        # Other keys remain at defaults
        assert s.layout_config["num_choices"] == 5

    def test_default_fill_threshold(self):
        assert AppSettings().fill_threshold == 0.5

    def test_custom_fill_threshold(self):
        assert AppSettings(fill_threshold=0.3).fill_threshold == 0.3

    def test_default_reference_image_path_is_none(self):
        assert AppSettings().reference_image_path is None

    def test_custom_reference_image_path(self):
        s = AppSettings(reference_image_path="/tmp/ref.png")
        assert s.reference_image_path == "/tmp/ref.png"


class TestDefaultClassMethod:
    def test_returns_app_settings(self):
        assert isinstance(AppSettings.default(), AppSettings)

    def test_default_has_default_values(self):
        s = AppSettings.default()
        assert s.fill_threshold == 0.5
        assert s.reference_image_path is None


class TestToDict:
    def test_keys_present(self):
        d = AppSettings().to_dict()
        assert "layout_config" in d
        assert "reference_image_path" in d
        assert "fill_threshold" in d

    def test_round_trip(self):
        orig = AppSettings(fill_threshold=0.7, reference_image_path="/tmp/ref.png")
        restored = AppSettings.from_dict(orig.to_dict())
        assert restored.fill_threshold == 0.7
        assert restored.reference_image_path == "/tmp/ref.png"


class TestSaveLoad:
    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "settings.json")
        AppSettings().save(path)
        assert os.path.exists(path)

    def test_saved_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "settings.json")
        AppSettings().save(path)
        with open(path) as f:
            data = json.load(f)
        assert "layout_config" in data

    def test_load_restores_values(self, tmp_path):
        path = str(tmp_path / "settings.json")
        orig = AppSettings(fill_threshold=0.6, layout_config={"num_questions": 25})
        orig.save(path)
        loaded = AppSettings.load(path)
        assert loaded.fill_threshold == 0.6
        assert loaded.layout_config["num_questions"] == 25
