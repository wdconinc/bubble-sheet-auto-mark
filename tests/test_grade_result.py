"""Tests for bubble_mark.models.grade_result."""
from __future__ import annotations

import pytest

from bubble_mark.models.answer_key import AnswerKey
from bubble_mark.models.grade_result import GradeResult


@pytest.fixture
def key_12345():
    return AnswerKey("12345")


class TestGradeResultInit:
    def test_stores_attributes(self, key_12345):
        r = GradeResult("SID001", "12345", key_12345)
        assert r.student_id == "SID001"
        assert r.answers == "12345"
        assert r.answer_key is key_12345

    def test_no_key(self):
        r = GradeResult("SID", "123")
        assert r.answer_key is None


class TestNumQuestions:
    def test_count(self):
        r = GradeResult("s", "12345")
        assert r.num_questions == 5

    def test_empty(self):
        r = GradeResult("s", "")
        assert r.num_questions == 0


class TestNumCorrect:
    def test_all_correct(self, key_12345):
        r = GradeResult("s", "12345", key_12345)
        assert r.num_correct == 5

    def test_none_correct(self, key_12345):
        # "55551" vs key "12345": no position matches → 0
        r = GradeResult("s", "55551", key_12345)
        assert r.num_correct == 0

    def test_partial(self, key_12345):
        r = GradeResult("s", "12555", key_12345)
        assert r.num_correct == 3

    def test_missing_ignored(self, key_12345):
        r = GradeResult("s", "1 3 5", key_12345)
        # positions 0,2,4 are gradeable: 1==1✓, 3==3✓, 5==5✓ → 3
        assert r.num_correct == 3

    def test_multiple_ignored(self, key_12345):
        r = GradeResult("s", "1M3M5", key_12345)
        assert r.num_correct == 3

    def test_no_key_returns_none(self):
        r = GradeResult("s", "12345")
        assert r.num_correct is None


class TestScore:
    def test_perfect_score(self, key_12345):
        r = GradeResult("s", "12345", key_12345)
        assert r.score == pytest.approx(1.0)

    def test_zero_score(self, key_12345):
        # "55551" vs key "12345": no matches → 0.0
        r = GradeResult("s", "55551", key_12345)
        assert r.score == pytest.approx(0.0)

    def test_partial_score(self, key_12345):
        r = GradeResult("s", "12555", key_12345)
        assert r.score == pytest.approx(3 / 5)

    def test_all_missing_score_zero(self, key_12345):
        r = GradeResult("s", "     ", key_12345)
        assert r.score == 0.0

    def test_no_key_returns_none(self):
        r = GradeResult("s", "12345")
        assert r.score is None


class TestToDict:
    def test_with_key(self, key_12345):
        r = GradeResult("s", "12345", key_12345)
        d = r.to_dict()
        assert "student_id" in d
        assert "answers" in d
        assert "score" in d
        assert "num_correct" in d
        assert "num_questions" in d

    def test_without_key(self):
        r = GradeResult("s", "12345")
        d = r.to_dict()
        assert "score" not in d


class TestToCsvRow:
    def test_required_columns_present(self, key_12345):
        r = GradeResult("STU001", "12345", key_12345)
        row = r.to_csv_row()
        assert row["student_id"] == "STU001"
        assert row["answers"] == "12345"

    def test_score_columns_when_key(self, key_12345):
        r = GradeResult("s", "12345", key_12345)
        row = r.to_csv_row()
        assert "score" in row
        assert "num_correct" in row
        assert "num_questions" in row

    def test_no_score_columns_without_key(self):
        r = GradeResult("s", "12345")
        row = r.to_csv_row()
        assert "score" not in row
