"""AnswerKey model: stores and validates the correct answers for a test."""
from __future__ import annotations

import csv
import json
from typing import ClassVar


_LETTER_TO_DIGIT: dict[str, str] = {
    "A": "1", "B": "2", "C": "3", "D": "4", "E": "5",
    "a": "1", "b": "2", "c": "3", "d": "4", "e": "5",
}
_VALID_DIGITS = frozenset("12345")
_VALID_LETTERS = frozenset("AaBbCcDdEe")


class AnswerKey:
    """Correct answers for a bubble-sheet test.

    Parameters
    ----------
    answers:
        String of correct answers.  Each character must be a digit ``1``–``5``
        or a letter ``A``–``E`` (case-insensitive).  Letters are normalised to
        digits automatically.
    name:
        Optional human-readable name for this key.
    """

    def __init__(self, answers: str, name: str = "") -> None:
        self.name = name
        self.answers = self._normalise_answers(answers)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_questions(self) -> int:
        return len(self.answers)

    # ------------------------------------------------------------------
    # Normalisation / validation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_answers(answers: str) -> str:
        result = []
        for ch in answers:
            if ch in _LETTER_TO_DIGIT:
                result.append(_LETTER_TO_DIGIT[ch])
            else:
                result.append(ch)
        return "".join(result)

    def normalize(self) -> str:
        """Return a copy of *answers* with all letters converted to digit form."""
        return self._normalise_answers(self.answers)

    def validate(self) -> bool:
        """Return *True* if every answer is a valid digit (``1``–``5``)."""
        return all(ch in _VALID_DIGITS for ch in self.answers)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_string(cls, s: str, name: str = "") -> "AnswerKey":
        """Create an :class:`AnswerKey` from a raw answer string."""
        return cls(s, name=name)

    @classmethod
    def from_csv_file(cls, path: str, name: str = "") -> "AnswerKey":
        """Load from a CSV file with columns ``question_number,answer``.

        Rows are sorted by question number; missing rows produce ``"1"``
        (configurable in future).
        """
        rows: dict[int, str] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = int(row["question_number"])
                rows[q] = row["answer"].strip()

        max_q = max(rows.keys()) if rows else 0
        answers = "".join(rows.get(i + 1, "1") for i in range(max_q))
        return cls(answers, name=name)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"answers": self.answers, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "AnswerKey":
        return cls(d["answers"], name=d.get("name", ""))

    def save_json(self, path: str) -> None:
        """Save this answer key to a JSON file at *path*."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "AnswerKey":
        """Load an answer key from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
