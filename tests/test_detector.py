"""Tests for bubble_mark.processing.detector."""

from __future__ import annotations

import numpy as np

from bubble_mark.processing.detector import BubbleSheetDetector
from tests.conftest import create_blank_sheet


class TestBubbleSheetDetectorInit:
    def test_default_layout(self):
        d = BubbleSheetDetector()
        assert d.layout_config["num_questions"] == 30
        assert d.layout_config["num_choices"] == 5
        assert d.layout_config["num_id_digits"] == 9
        assert d.layout_config["id_choices_per_digit"] == 10

    def test_custom_layout(self):
        d = BubbleSheetDetector({"num_questions": 20, "num_choices": 4})
        assert d.layout_config["num_questions"] == 20
        assert d.layout_config["num_choices"] == 4
        # Unspecified keys keep defaults
        assert d.layout_config["num_id_digits"] == 9

    def test_default_regions_are_none(self):
        d = BubbleSheetDetector()
        assert d.answer_region is None
        assert d.id_region is None

    def test_custom_regions_stored(self):
        d = BubbleSheetDetector(
            answer_region=[0.0, 0.0, 1.0, 0.72],
            id_region=[0.0, 0.74, 1.0, 1.0],
        )
        assert d.answer_region == [0.0, 0.0, 1.0, 0.72]
        assert d.id_region == [0.0, 0.74, 1.0, 1.0]


class TestDetect:
    def test_returns_ndarray_for_blank_sheet(self, blank_sheet):
        d = BubbleSheetDetector()
        result = d.detect(blank_sheet)
        assert isinstance(result, np.ndarray)

    def test_output_has_correct_size(self, blank_sheet):
        d = BubbleSheetDetector()
        result = d.detect(blank_sheet)
        assert result is not None
        # Should be resized to internal standard
        assert result.shape[1] == 850
        assert result.shape[0] == 1100

    def test_detect_synthetic_sheet(self, synthetic_sheet):
        d = BubbleSheetDetector({"num_questions": 5, "num_choices": 5})
        result = d.detect(synthetic_sheet)
        assert result is not None


class TestLocateAnswerBubbles:
    def test_returns_grid_of_correct_size(self):
        d = BubbleSheetDetector({"num_questions": 5, "num_choices": 5})
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        grid = d.locate_answer_bubbles(normalised)
        assert len(grid) == 5
        for row in grid:
            assert len(row) == 5

    def test_bubble_tuples_are_4_element(self):
        d = BubbleSheetDetector({"num_questions": 3, "num_choices": 4})
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        grid = d.locate_answer_bubbles(normalised)
        for row in grid:
            for bubble in row:
                assert len(bubble) == 4

    def test_bubble_coords_within_image(self):
        d = BubbleSheetDetector({"num_questions": 5, "num_choices": 5})
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        h, w = normalised.shape[:2]
        grid = d.locate_answer_bubbles(normalised)
        for row in grid:
            for x, y, bw, bh in row:
                assert x >= 0
                assert y >= 0
                assert x + bw <= w + 5  # allow tiny rounding
                assert bw > 0 and bh > 0

    def test_custom_answer_region_grid_size(self):
        """Custom answer_region must still return the correct grid dimensions."""
        d = BubbleSheetDetector(
            {"num_questions": 5, "num_choices": 5},
            answer_region=[0.0, 0.0, 1.0, 0.6],
        )
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        grid = d.locate_answer_bubbles(normalised)
        assert len(grid) == 5
        for row in grid:
            assert len(row) == 5

    def test_custom_answer_region_y_coords_bounded(self):
        """Bubbles from a custom region should lie within the region's y-range."""
        region = [0.0, 0.0, 1.0, 0.6]
        d = BubbleSheetDetector(
            {"num_questions": 5, "num_choices": 5},
            answer_region=region,
        )
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        max_y = int(normalised.shape[0] * region[3])
        grid = d.locate_answer_bubbles(normalised)
        for row in grid:
            for x, y, bw, bh in row:
                assert y + bh <= max_y + 5  # allow tiny rounding


class TestLocateIdBubbles:
    def test_returns_columns_of_correct_size(self):
        d = BubbleSheetDetector({"num_id_digits": 9, "id_choices_per_digit": 10})
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        columns = d.locate_id_bubbles(normalised)
        assert len(columns) == 9
        for col in columns:
            assert len(col) == 10

    def test_id_bubble_tuples_are_4_element(self):
        d = BubbleSheetDetector()
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        columns = d.locate_id_bubbles(normalised)
        for col in columns:
            for bubble in col:
                assert len(bubble) == 4

    def test_id_bubble_y_coords_in_full_image_space(self):
        """ID bubble y coordinates must be offset into the full normalised image.

        The ID section starts at ~74% of the image height; y values below that
        threshold confirm the offset is applied correctly.
        """
        d = BubbleSheetDetector()
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        id_section_start = int(normalised.shape[0] * 0.74)
        columns = d.locate_id_bubbles(normalised)
        for col in columns:
            for x, y, bw, bh in col:
                assert y >= id_section_start, (
                    f"ID bubble y={y} is above the ID section "
                    f"start at {id_section_start}"
                )

    def test_custom_id_region_columns_size(self):
        """Custom id_region must still return the correct number of columns/rows."""
        d = BubbleSheetDetector(
            {"num_id_digits": 9, "id_choices_per_digit": 10},
            id_region=[0.0, 0.74, 1.0, 1.0],
        )
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        columns = d.locate_id_bubbles(normalised)
        assert len(columns) == 9
        for col in columns:
            assert len(col) == 10

    def test_custom_id_region_y_coords_within_region(self):
        """ID bubbles from a custom region should start within the region's y-range."""
        region = [0.0, 0.74, 1.0, 1.0]
        d = BubbleSheetDetector(id_region=region)
        sheet = create_blank_sheet()
        normalised = d.detect(sheet)
        id_section_start = int(normalised.shape[0] * region[1])
        columns = d.locate_id_bubbles(normalised)
        for col in columns:
            for x, y, bw, bh in col:
                assert y >= id_section_start, (
                    f"ID bubble y={y} is above custom region start {id_section_start}"
                )
