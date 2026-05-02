"""Tests for bubble_mark.processing.analyzer."""

from __future__ import annotations

import numpy as np
import pytest

from bubble_mark.processing.analyzer import BubbleAnalyzer
from tests.conftest import create_empty_bubble_image, create_filled_bubble_image


class TestBubbleAnalyzerInit:
    def test_default_threshold(self):
        a = BubbleAnalyzer()
        assert a.fill_threshold == 0.5

    def test_custom_threshold(self):
        a = BubbleAnalyzer(fill_threshold=0.3)
        assert a.fill_threshold == 0.3

    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError):
            BubbleAnalyzer(fill_threshold=0.0)

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError):
            BubbleAnalyzer(fill_threshold=1.1)


class TestAnalyzeBubble:
    def _make_sheet(self, bubble: np.ndarray) -> tuple[np.ndarray, tuple]:
        """Embed *bubble* into a white sheet; return (sheet, region)."""
        h, w = bubble.shape[:2]
        sheet = np.full((h + 20, w + 20), 255, dtype=np.uint8)
        sheet[10 : 10 + h, 10 : 10 + w] = (
            bubble if bubble.ndim == 2 else bubble[:, :, 0]
        )
        return sheet, (10, 10, w, h)

    def test_filled_bubble_returns_true(self):
        a = BubbleAnalyzer(fill_threshold=0.5)
        bubble = create_filled_bubble_image(30, 30)
        sheet, region = self._make_sheet(bubble)
        assert a.analyze_bubble(sheet, region) is True

    def test_empty_bubble_returns_false(self):
        a = BubbleAnalyzer(fill_threshold=0.5)
        bubble = create_empty_bubble_image(30, 30)
        sheet, region = self._make_sheet(bubble)
        assert a.analyze_bubble(sheet, region) is False

    def test_out_of_bounds_region_returns_false(self):
        a = BubbleAnalyzer()
        sheet = np.full((50, 50), 255, dtype=np.uint8)
        assert a.analyze_bubble(sheet, (200, 200, 30, 30)) is False

    def test_zero_size_region_returns_false(self):
        a = BubbleAnalyzer()
        sheet = np.zeros((50, 50), dtype=np.uint8)
        assert a.analyze_bubble(sheet, (10, 10, 0, 20)) is False

    def test_bgr_image_supported(self):
        a = BubbleAnalyzer(fill_threshold=0.5)
        # All-black BGR image → filled
        sheet = np.zeros((50, 50, 3), dtype=np.uint8)
        assert a.analyze_bubble(sheet, (5, 5, 20, 20)) is True


class TestAnalyzeAnswerRow:
    def _build_image_with_bubbles(self, filled_idx: list[int], num_bubbles: int = 5):
        """Build a grayscale image where selected bubbles are black."""
        bub_w, bub_h = 30, 30
        gap = 5
        total_w = num_bubbles * (bub_w + gap)
        sheet = np.full((bub_h + 20, total_w + 20), 255, dtype=np.uint8)
        bubbles = []
        for i in range(num_bubbles):
            x = 10 + i * (bub_w + gap)
            y = 10
            if i in filled_idx:
                sheet[y : y + bub_h, x : x + bub_w] = 0
            bubbles.append((x, y, bub_w, bub_h))
        return sheet, bubbles

    def test_single_answer_choice_1(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_image_with_bubbles([0])
        assert a.analyze_answer_row(sheet, bubbles) == "1"

    def test_single_answer_choice_3(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_image_with_bubbles([2])
        assert a.analyze_answer_row(sheet, bubbles) == "3"

    def test_no_answer_returns_space(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_image_with_bubbles([])
        assert a.analyze_answer_row(sheet, bubbles) == " "

    def test_multiple_answers_returns_M(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_image_with_bubbles([0, 2])
        assert a.analyze_answer_row(sheet, bubbles) == "M"


class TestAnalyzeIdColumn:
    def _build_id_column(self, filled_digit: int | None, num_choices: int = 10):
        bub_w, bub_h = 25, 25
        gap = 4
        total_h = num_choices * (bub_h + gap)
        sheet = np.full((total_h + 20, bub_w + 20), 255, dtype=np.uint8)
        bubbles = []
        for i in range(num_choices):
            x = 10
            y = 10 + i * (bub_h + gap)
            if i == filled_digit:
                sheet[y : y + bub_h, x : x + bub_w] = 0
            bubbles.append((x, y, bub_w, bub_h))
        return sheet, bubbles

    def test_digit_5_recognised(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_id_column(5)
        assert a.analyze_id_column(sheet, bubbles) == "5"

    def test_digit_0_recognised(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_id_column(0)
        assert a.analyze_id_column(sheet, bubbles) == "0"

    def test_no_digit_returns_space(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_id_column(None)
        assert a.analyze_id_column(sheet, bubbles) == " "

    def test_multiple_digits_returns_question_mark(self):
        a = BubbleAnalyzer()
        bub_w, bub_h = 25, 25
        gap = 4
        num_choices = 10
        total_h = num_choices * (bub_h + gap)
        sheet = np.full((total_h + 20, bub_w + 20), 255, dtype=np.uint8)
        bubbles = []
        for i in range(num_choices):
            x, y = 10, 10 + i * (bub_h + gap)
            if i in (2, 7):
                sheet[y : y + bub_h, x : x + bub_w] = 0
            bubbles.append((x, y, bub_w, bub_h))
        assert a.analyze_id_column(sheet, bubbles) == "?"


class TestAnalyzeAnswerRowWithFilled:
    def _build_image_with_bubbles(self, filled_idx: list[int], num_bubbles: int = 5):
        bub_w, bub_h = 30, 30
        gap = 5
        total_w = num_bubbles * (bub_w + gap)
        sheet = np.full((bub_h + 20, total_w + 20), 255, dtype=np.uint8)
        bubbles = []
        for i in range(num_bubbles):
            x = 10 + i * (bub_w + gap)
            y = 10
            if i in filled_idx:
                sheet[y : y + bub_h, x : x + bub_w] = 0
            bubbles.append((x, y, bub_w, bub_h))
        return sheet, bubbles

    def test_single_answer_returns_correct_char_and_region(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_image_with_bubbles([2])
        ch, filled = a.analyze_answer_row_with_filled(sheet, bubbles)
        assert ch == "3"
        assert filled == [bubbles[2]]

    def test_no_answer_returns_space_and_empty_list(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_image_with_bubbles([])
        ch, filled = a.analyze_answer_row_with_filled(sheet, bubbles)
        assert ch == " "
        assert filled == []

    def test_multiple_answers_returns_M_and_all_filled_regions(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_image_with_bubbles([0, 3])
        ch, filled = a.analyze_answer_row_with_filled(sheet, bubbles)
        assert ch == "M"
        assert set(map(tuple, filled)) == {bubbles[0], bubbles[3]}

    def test_consistent_with_analyze_answer_row(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_image_with_bubbles([1])
        ch, _ = a.analyze_answer_row_with_filled(sheet, bubbles)
        assert a.analyze_answer_row(sheet, bubbles) == ch


class TestAnalyzeIdColumnWithFilled:
    def _build_id_column(self, filled_indices: list[int], num_choices: int = 10):
        bub_w, bub_h = 25, 25
        gap = 4
        total_h = num_choices * (bub_h + gap)
        sheet = np.full((total_h + 20, bub_w + 20), 255, dtype=np.uint8)
        bubbles = []
        for i in range(num_choices):
            x, y = 10, 10 + i * (bub_h + gap)
            if i in filled_indices:
                sheet[y : y + bub_h, x : x + bub_w] = 0
            bubbles.append((x, y, bub_w, bub_h))
        return sheet, bubbles

    def test_single_digit_returns_correct_char_and_region(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_id_column([7])
        ch, filled = a.analyze_id_column_with_filled(sheet, bubbles)
        assert ch == "7"
        assert filled == [bubbles[7]]

    def test_no_digit_returns_space_and_empty_list(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_id_column([])
        ch, filled = a.analyze_id_column_with_filled(sheet, bubbles)
        assert ch == " "
        assert filled == []

    def test_multiple_digits_returns_question_mark_and_all_regions(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_id_column([2, 8])
        ch, filled = a.analyze_id_column_with_filled(sheet, bubbles)
        assert ch == "?"
        assert set(map(tuple, filled)) == {bubbles[2], bubbles[8]}

    def test_consistent_with_analyze_id_column(self):
        a = BubbleAnalyzer()
        sheet, bubbles = self._build_id_column([4])
        ch, _ = a.analyze_id_column_with_filled(sheet, bubbles)
        assert a.analyze_id_column(sheet, bubbles) == ch
