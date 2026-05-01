"""Tests for bubble_mark.export.csv_exporter."""
from __future__ import annotations

import csv
import io

import pytest

from bubble_mark.export.csv_exporter import CSVExporter
from bubble_mark.models.answer_key import AnswerKey
from bubble_mark.models.grade_result import GradeResult


@pytest.fixture
def key():
    return AnswerKey("12345")


@pytest.fixture
def results(key):
    return [
        GradeResult("123456789", "12345", key),
        GradeResult("987654321", "M35 1", key),
    ]


@pytest.fixture
def results_no_key():
    return [
        GradeResult("111111111", "12345"),
        GradeResult("222222222", "54321"),
    ]


class TestExportToString:
    def test_returns_string(self, results):
        csv_str = CSVExporter().export_to_string(results)
        assert isinstance(csv_str, str)

    def test_has_header_row(self, results):
        csv_str = CSVExporter().export_to_string(results)
        lines = csv_str.strip().splitlines()
        assert lines[0].startswith("student_id,answers")

    def test_score_columns_present_when_key(self, results):
        csv_str = CSVExporter().export_to_string(results)
        header = csv_str.splitlines()[0]
        assert "score" in header
        assert "num_correct" in header
        assert "num_questions" in header

    def test_no_score_columns_without_key(self, results_no_key):
        csv_str = CSVExporter(include_score=True).export_to_string(results_no_key)
        header = csv_str.splitlines()[0]
        assert "score" not in header

    def test_include_score_false_omits_score_columns(self, results):
        csv_str = CSVExporter(include_score=False).export_to_string(results)
        header = csv_str.splitlines()[0]
        assert "score" not in header

    def test_student_id_in_rows(self, results):
        csv_str = CSVExporter().export_to_string(results)
        assert "123456789" in csv_str
        assert "987654321" in csv_str

    def test_answers_in_rows(self, results):
        csv_str = CSVExporter().export_to_string(results)
        assert "12345" in csv_str

    def test_correct_number_of_data_rows(self, results):
        csv_str = CSVExporter().export_to_string(results)
        lines = [l for l in csv_str.strip().splitlines() if l]
        # header + 2 data rows
        assert len(lines) == 3

    def test_parseable_csv(self, results):
        csv_str = CSVExporter().export_to_string(results)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["student_id"] == "123456789"

    def test_empty_results(self):
        csv_str = CSVExporter().export_to_string([])
        lines = [l for l in csv_str.strip().splitlines() if l]
        assert len(lines) == 1  # header only

    def test_score_value_format(self, results):
        csv_str = CSVExporter().export_to_string(results)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        # First result "12345" vs key "12345" → score=1.0
        assert float(rows[0]["score"]) == pytest.approx(1.0)


class TestExportToFile:
    def test_writes_file(self, tmp_path, results):
        path = str(tmp_path / "out.csv")
        CSVExporter().export(results, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "student_id" in content
        assert "123456789" in content

    def test_file_parseable(self, tmp_path, results):
        path = str(tmp_path / "out.csv")
        CSVExporter().export(results, path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
