"""Tests for bubble_mark.processing.grader."""

from __future__ import annotations

import numpy as np
import pytest

from bubble_mark.models.grade_result import GradeResult
from bubble_mark.processing.analyzer import BubbleAnalyzer
from bubble_mark.processing.detector import BubbleSheetDetector
from bubble_mark.processing.grader import BubbleSheetGrader
from tests.conftest import create_blank_sheet


@pytest.fixture
def grader(sample_answer_key):
    detector = BubbleSheetDetector({"num_questions": 5, "num_choices": 5})
    analyzer = BubbleAnalyzer()
    return BubbleSheetGrader(sample_answer_key, detector, analyzer)


class TestGradeAnswers:
    def test_returns_grade_result(self, grader):
        result = grader.grade_answers("12345", "123456789")
        assert isinstance(result, GradeResult)

    def test_student_id_stored(self, grader):
        result = grader.grade_answers("12345", "987654321")
        assert result.student_id == "987654321"

    def test_answers_stored(self, grader):
        result = grader.grade_answers("12345", "000000001")
        assert result.answers == "12345"

    def test_answer_key_stored(self, grader):
        result = grader.grade_answers("12345", "000")
        assert result.answer_key is grader.answer_key

    def test_perfect_score(self, grader):
        # answer_key is "12345"
        result = grader.grade_answers("12345", "111")
        assert result.score == 1.0

    def test_zero_score(self, grader):
        # "55551" vs key "12345": no position matches → 0.0
        result = grader.grade_answers("55551", "111")
        assert result.score == 0.0

    def test_partial_score(self, grader):
        # "12345" vs "12555" → 3/5 correct (positions 0,1,2 are right)
        # Wait: key="12345", detected="12555"
        # q0: 1==1 ✓, q1: 2==2 ✓, q2: 5≠3, q3: 5≠4, q4: 5==5 ✓ → 3/5
        result = grader.grade_answers("12555", "111")
        assert result.score == pytest.approx(3 / 5)

    def test_missing_answers_ignored_in_score(self, grader):
        # " " means missing, excluded from denominator
        result = grader.grade_answers(" 2 4 ", "111")
        # gradeable: 2 (positions 1 and 3); key="12345", pos1→2==2 ✓, pos3→4==4 ✓
        assert result.score == 1.0

    def test_multiple_answers_ignored_in_score(self, grader):
        result = grader.grade_answers("M2M4M", "111")
        # gradeable: positions 1,3 → both correct → 2/2
        assert result.score == 1.0


class TestGradeImage:
    def test_grade_image_returns_grade_result(self, grader):
        sheet = create_blank_sheet()
        result = grader.grade_image(sheet)
        # Blank sheet: detection falls back, should still return GradeResult
        assert isinstance(result, GradeResult)

    def test_grade_image_answer_key_attached(self, grader):
        sheet = create_blank_sheet()
        result = grader.grade_image(sheet)
        assert result.answer_key is grader.answer_key

    def test_grade_image_annotated_image_attached(self, grader):
        sheet = create_blank_sheet()
        result = grader.grade_image(sheet)
        assert result.annotated_image is not None

    def test_grade_image_annotated_image_is_ndarray(self, grader):
        sheet = create_blank_sheet()
        result = grader.grade_image(sheet)
        assert isinstance(result.annotated_image, np.ndarray)

    def test_grade_image_annotated_image_is_bgr(self, grader):
        sheet = create_blank_sheet()
        result = grader.grade_image(sheet)
        assert result.annotated_image.ndim == 3
        assert result.annotated_image.shape[2] == 3

    def test_grade_image_annotated_image_same_size_as_normalised(self, grader):
        """Annotated image must match the normalised sheet dimensions (850×1100)."""
        sheet = create_blank_sheet()
        result = grader.grade_image(sheet)
        assert result.annotated_image.shape[1] == 850
        assert result.annotated_image.shape[0] == 1100

    def test_grade_image_annotated_differs_from_blank_sheet(self, grader):
        """The overlay must actually modify at least one pixel."""
        sheet = create_blank_sheet()
        result = grader.grade_image(sheet)
        # The normalised blank sheet is all-white (255); overlay draws coloured
        # borders so at least one pixel must differ from 255.
        assert not np.all(result.annotated_image == 255)
