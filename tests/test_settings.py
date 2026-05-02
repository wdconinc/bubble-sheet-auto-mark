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


class TestValidateEdgeLines:
    def test_valid_four_lines(self):
        from bubble_mark.models.settings import _validate_edge_lines

        lines = [
            [0, 0, 100, 0],
            [0, 200, 100, 200],
            [0, 0, 0, 200],
            [100, 0, 100, 200],
        ]
        result = _validate_edge_lines(lines)
        assert result is not None
        assert len(result) == 4

    def test_none_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_lines

        assert _validate_edge_lines(None) is None

    def test_wrong_count_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_lines

        lines = [[0, 0, 100, 0], [0, 200, 100, 200]]  # only 2
        assert _validate_edge_lines(lines) is None

    def test_wrong_length_per_line_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_lines

        lines = [[0, 0, 100], [0, 200, 100, 200], [0, 0, 0, 200], [100, 0, 100, 200]]
        assert _validate_edge_lines(lines) is None

    def test_non_numeric_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_lines

        lines = [
            ["a", "b", "c", "d"],
            [0, 200, 100, 200],
            [0, 0, 0, 200],
            [100, 0, 100, 200],
        ]
        assert _validate_edge_lines(lines) is None


class TestAppSettingsNewFields:
    def test_default_page_edge_lines_is_none(self):
        assert AppSettings().page_edge_lines is None

    def test_custom_page_edge_lines_stored(self):
        lines = [
            [0, 0, 100, 0],
            [0, 200, 100, 200],
            [0, 0, 0, 200],
            [100, 0, 100, 200],
        ]
        s = AppSettings(page_edge_lines=lines)
        assert s.page_edge_lines is not None
        assert len(s.page_edge_lines) == 4

    def test_invalid_page_edge_lines_stored_as_none(self):
        s = AppSettings(page_edge_lines=[[0, 0, 100, 0]])  # only 1 line
        assert s.page_edge_lines is None

    def test_default_reference_color_channel(self):
        assert AppSettings().reference_color_channel == 1

    def test_custom_reference_color_channel(self):
        assert AppSettings(reference_color_channel=2).reference_color_channel == 2

    def test_round_trip_with_new_fields(self):
        lines = [
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 200.0, 100.0, 200.0],
            [0.0, 0.0, 0.0, 200.0],
            [100.0, 0.0, 100.0, 200.0],
        ]
        orig = AppSettings(
            page_edge_lines=lines,
            reference_color_channel=2,
        )
        restored = AppSettings.from_dict(orig.to_dict())
        assert restored.page_edge_lines is not None
        assert len(restored.page_edge_lines) == 4
        assert restored.reference_color_channel == 2

    def test_to_dict_contains_new_keys(self):
        d = AppSettings().to_dict()
        assert "page_edge_lines" in d
        assert "reference_color_channel" in d

    def test_load_legacy_dict_without_new_fields(self):
        """Loading a dict without the new keys should use defaults."""
        d = {
            "layout_config": {},
            "reference_image_path": None,
            "fill_threshold": 0.5,
            "answer_region": None,
            "id_region": None,
        }
        s = AppSettings.from_dict(d)
        assert s.page_edge_lines is None
        assert s.reference_color_channel == 1


class TestValidateEdgePolylines:
    def test_none_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        assert _validate_edge_polylines(None) is None

    def test_non_dict_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        assert _validate_edge_polylines([[0, 0], [1, 1]]) is None

    def test_valid_all_edges(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        polys = {
            "top": [[0, 0], [100, 0]],
            "bottom": [[0, 200], [100, 200]],
            "left": [[0, 0], [0, 200]],
            "right": [[100, 0], [100, 200]],
        }
        result = _validate_edge_polylines(polys)
        assert result is not None
        assert len(result["top"]) == 2

    def test_edge_with_none_is_allowed(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        polys = {
            "top": [[0, 0], [100, 0]],
            "bottom": None,
            "left": None,
            "right": None,
        }
        result = _validate_edge_polylines(polys)
        assert result is not None
        assert result["top"] is not None
        assert result["bottom"] is None

    def test_single_point_edge_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        polys = {
            "top": [[0, 0]],  # too few points
            "bottom": [[0, 200], [100, 200]],
            "left": [[0, 0], [0, 200]],
            "right": [[100, 0], [100, 200]],
        }
        assert _validate_edge_polylines(polys) is None

    def test_wrong_point_length_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        polys = {
            "top": [[0, 0, 99], [100, 0, 99]],  # 3-element points
            "bottom": [[0, 200], [100, 200]],
            "left": [[0, 0], [0, 200]],
            "right": [[100, 0], [100, 200]],
        }
        assert _validate_edge_polylines(polys) is None

    def test_non_numeric_returns_none(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        polys = {
            "top": [["a", "b"], ["c", "d"]],
            "bottom": [[0, 200], [100, 200]],
            "left": [[0, 0], [0, 200]],
            "right": [[100, 0], [100, 200]],
        }
        assert _validate_edge_polylines(polys) is None

    def test_multi_point_edge_accepted(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        polys = {
            "top": [[0, 0], [50, 5], [100, 0]],
            "bottom": [[0, 200], [50, 195], [100, 200]],
            "left": [[0, 0], [2, 100], [0, 200]],
            "right": [[100, 0], [98, 100], [100, 200]],
        }
        result = _validate_edge_polylines(polys)
        assert result is not None
        assert len(result["top"]) == 3

    def test_coordinates_converted_to_float(self):
        from bubble_mark.models.settings import _validate_edge_polylines

        polys = {
            "top": [[0, 0], [100, 0]],
            "bottom": [[0, 200], [100, 200]],
            "left": [[0, 0], [0, 200]],
            "right": [[100, 0], [100, 200]],
        }
        result = _validate_edge_polylines(polys)
        assert result is not None
        assert isinstance(result["top"][0][0], float)


class TestAppSettingsPolylines:
    def test_default_page_edge_polylines_is_none(self):
        assert AppSettings().page_edge_polylines is None

    def test_valid_polylines_stored(self):
        polys = {
            "top": [[0.0, 0.0], [100.0, 0.0]],
            "bottom": [[0.0, 200.0], [100.0, 200.0]],
            "left": [[0.0, 0.0], [0.0, 200.0]],
            "right": [[100.0, 0.0], [100.0, 200.0]],
        }
        s = AppSettings(page_edge_polylines=polys)
        assert s.page_edge_polylines is not None

    def test_invalid_polylines_stored_as_none(self):
        s = AppSettings(page_edge_polylines={"top": [[0, 0]]})  # only 1 point
        assert s.page_edge_polylines is None

    def test_round_trip(self):
        polys = {
            "top": [[0.0, 0.0], [100.0, 0.0]],
            "bottom": [[0.0, 200.0], [100.0, 200.0]],
            "left": [[0.0, 0.0], [0.0, 200.0]],
            "right": [[100.0, 0.0], [100.0, 200.0]],
        }
        orig = AppSettings(page_edge_polylines=polys)
        restored = AppSettings.from_dict(orig.to_dict())
        assert restored.page_edge_polylines is not None
        assert len(restored.page_edge_polylines["top"]) == 2

    def test_to_dict_contains_key(self):
        d = AppSettings().to_dict()
        assert "page_edge_polylines" in d

    def test_load_legacy_dict_without_polylines(self):
        d = {
            "layout_config": {},
            "reference_image_path": None,
            "fill_threshold": 0.5,
            "answer_region": None,
            "id_region": None,
            "page_edge_lines": None,
            "reference_color_channel": 1,
        }
        s = AppSettings.from_dict(d)
        assert s.page_edge_polylines is None


class TestGetEdgeCorrectionInputs:
    """Tests for AppSettings.get_edge_correction_inputs()."""

    @staticmethod
    def _full_polylines():
        return {
            "top": [[0.0, 0.0], [100.0, 0.0]],
            "bottom": [[0.0, 200.0], [100.0, 200.0]],
            "left": [[0.0, 0.0], [0.0, 200.0]],
            "right": [[100.0, 0.0], [100.0, 200.0]],
        }

    @staticmethod
    def _four_lines():
        return [
            [0, 0, 100, 0],
            [0, 200, 100, 200],
            [0, 0, 0, 200],
            [100, 0, 100, 200],
        ]

    def test_returns_none_when_nothing_set(self):
        mode, data = AppSettings().get_edge_correction_inputs()
        assert mode == "none"
        assert data is None

    def test_returns_lines_when_only_lines_set(self):
        s = AppSettings(page_edge_lines=self._four_lines())
        mode, data = s.get_edge_correction_inputs()
        assert mode == "lines"
        assert data is not None

    def test_returns_polylines_when_all_four_edges_set(self):
        s = AppSettings(page_edge_polylines=self._full_polylines())
        mode, data = s.get_edge_correction_inputs()
        assert mode == "polylines"
        assert data is not None

    def test_polylines_take_priority_over_lines(self):
        s = AppSettings(
            page_edge_lines=self._four_lines(),
            page_edge_polylines=self._full_polylines(),
        )
        mode, data = s.get_edge_correction_inputs()
        assert mode == "polylines"

    def test_partial_polylines_falls_back_to_lines(self):
        """If only 3 polyline edges are defined, lines should be used."""
        partial = {k: v for k, v in self._full_polylines().items() if k != "right"}
        s = AppSettings(
            page_edge_lines=self._four_lines(),
            page_edge_polylines=partial,
        )
        mode, data = s.get_edge_correction_inputs()
        assert mode == "lines"

    def test_partial_polylines_no_lines_returns_none(self):
        """Partial polylines with no fallback lines → 'none'."""
        partial = {k: v for k, v in self._full_polylines().items() if k != "right"}
        s = AppSettings(page_edge_polylines=partial)
        mode, data = s.get_edge_correction_inputs()
        assert mode == "none"
