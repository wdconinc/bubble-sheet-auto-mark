"""Tests for bubble_mark.models.answer_key."""
from __future__ import annotations

import json
import csv
import os

import pytest

from bubble_mark.models.answer_key import AnswerKey


class TestAnswerKeyInit:
    def test_digit_answers_stored_unchanged(self):
        key = AnswerKey("12345")
        assert key.answers == "12345"

    def test_letter_answers_normalised_to_digits(self):
        key = AnswerKey("ABCDE")
        assert key.answers == "12345"

    def test_mixed_case_normalised(self):
        key = AnswerKey("AaBbCcDdEe")
        assert key.answers == "1122334455"

    def test_name_stored(self):
        key = AnswerKey("123", name="quiz1")
        assert key.name == "quiz1"

    def test_default_name_empty(self):
        key = AnswerKey("12")
        assert key.name == ""


class TestNumQuestions:
    def test_length(self):
        key = AnswerKey("12345678")
        assert key.num_questions == 8


class TestNormalize:
    def test_returns_digit_string(self):
        key = AnswerKey("ABCDE")
        assert key.normalize() == "12345"


class TestValidate:
    def test_valid_digits(self):
        assert AnswerKey("12345").validate() is True

    def test_invalid_digit_6(self):
        # 6 is not a valid choice
        assert AnswerKey.__new__(AnswerKey).__class__("6").validate() is False

    def test_invalid_after_normalisation(self):
        # letters are normalised → should be valid
        key = AnswerKey("ABCDE")
        assert key.validate() is True

    def test_empty_answers(self):
        key = AnswerKey("")
        assert key.validate() is True  # vacuously true


class TestFromString:
    def test_creates_key(self):
        key = AnswerKey.from_string("12345", name="test")
        assert key.answers == "12345"
        assert key.name == "test"


class TestFromCsvFile:
    def test_loads_csv(self, tmp_path):
        path = tmp_path / "key.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question_number", "answer"])
            writer.writeheader()
            for i, ans in enumerate("12345", start=1):
                writer.writerow({"question_number": i, "answer": ans})
        key = AnswerKey.from_csv_file(str(path))
        assert key.answers == "12345"


class TestToDict:
    def test_round_trip(self):
        key = AnswerKey("12345", name="round_trip")
        d = key.to_dict()
        key2 = AnswerKey.from_dict(d)
        assert key2.answers == key.answers
        assert key2.name == key.name


class TestSaveLoadJson:
    def test_save_and_load(self, tmp_path):
        key = AnswerKey("12345", name="saved")
        path = str(tmp_path / "key.json")
        key.save_json(path)
        loaded = AnswerKey.load_json(path)
        assert loaded.answers == "12345"
        assert loaded.name == "saved"

    def test_saved_file_is_valid_json(self, tmp_path):
        key = AnswerKey("123")
        path = str(tmp_path / "k.json")
        key.save_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["answers"] == "123"
