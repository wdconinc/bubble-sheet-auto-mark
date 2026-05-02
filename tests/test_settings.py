"""Tests for bubble_mark.models.settings."""

from __future__ import annotations

import json
import os

from bubble_mark.models.settings import AppSettings, _validate_region


class TestValidateRegion:
    def test_none_returns_none(self):
        assert _validate_region(None) is None

    def test_valid_region(self):
        assert _validate_region([0.0, 0.0, 1.0, 0.72]) == [0.0, 0.0, 1.0, 0.72]

    def test_invalid_ordering_returns_none(self):
        assert _validate_region([0.5, 0.0, 0.3, 1.0]) is None  # x1 >= x2

    def test_out_of_range_returns_none(self):
        assert _validate_region([0.0, 0.0, 1.5, 1.0]) is None

    def test_wrong_length_returns_none(self):
        assert _validate_region([0.0, 0.0, 1.0]) is None

    def test_non_numeric_returns_none(self):
        assert _validate_region(["a", "b", "c", "d"]) is None


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

    def test_default_answer_region_is_none(self):
        assert AppSettings().answer_region is None

    def test_custom_answer_region(self):
        s = AppSettings(answer_region=[0.0, 0.0, 1.0, 0.72])
        assert s.answer_region == [0.0, 0.0, 1.0, 0.72]

    def test_invalid_answer_region_stored_as_none(self):
        s = AppSettings(answer_region=[0.9, 0.0, 0.1, 1.0])  # x1 > x2
        assert s.answer_region is None

    def test_default_id_region_is_none(self):
        assert AppSettings().id_region is None

    def test_custom_id_region(self):
        s = AppSettings(id_region=[0.0, 0.74, 1.0, 1.0])
        assert s.id_region == [0.0, 0.74, 1.0, 1.0]


class TestDefaultClassMethod:
    def test_returns_app_settings(self):
        assert isinstance(AppSettings.default(), AppSettings)

    def test_default_has_default_values(self):
        s = AppSettings.default()
        assert s.fill_threshold == 0.5
        assert s.reference_image_path is None
        assert s.answer_region is None
        assert s.id_region is None


class TestToDict:
    def test_keys_present(self):
        d = AppSettings().to_dict()
        assert "layout_config" in d
        assert "reference_image_path" in d
        assert "fill_threshold" in d
        assert "answer_region" in d
        assert "id_region" in d

    def test_round_trip(self):
        orig = AppSettings(
            fill_threshold=0.7,
            reference_image_path="/tmp/ref.png",
            answer_region=[0.0, 0.0, 1.0, 0.72],
            id_region=[0.0, 0.74, 1.0, 1.0],
        )
        restored = AppSettings.from_dict(orig.to_dict())
        assert restored.fill_threshold == 0.7
        assert restored.reference_image_path == "/tmp/ref.png"
        assert restored.answer_region == [0.0, 0.0, 1.0, 0.72]
        assert restored.id_region == [0.0, 0.74, 1.0, 1.0]

    def test_round_trip_without_regions(self):
        orig = AppSettings(fill_threshold=0.6)
        restored = AppSettings.from_dict(orig.to_dict())
        assert restored.answer_region is None
        assert restored.id_region is None


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

    def test_load_restores_regions(self, tmp_path):
        path = str(tmp_path / "settings.json")
        orig = AppSettings(
            answer_region=[0.0, 0.0, 1.0, 0.72],
            id_region=[0.0, 0.74, 1.0, 1.0],
        )
        orig.save(path)
        loaded = AppSettings.load(path)
        assert loaded.answer_region == [0.0, 0.0, 1.0, 0.72]
        assert loaded.id_region == [0.0, 0.74, 1.0, 1.0]
