"""GradeResult model: per-student grading outcome."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bubble_mark.models.answer_key import AnswerKey

_VALID_ANSWERS = frozenset("12345")


class GradeResult:
    """Grading outcome for a single student.

    Parameters
    ----------
    student_id:
        Detected or manually-set student ID string.
    answers:
        Detected answers string.  Each character is ``"1"``–``"5"``,
        ``"M"`` (multiple marked), or ``" "`` (blank/missing).
    answer_key:
        Optional :class:`~bubble_mark.models.answer_key.AnswerKey` to score
        against.  When *None*, :attr:`score` and :attr:`num_correct` are also
        *None*.
    """

    def __init__(
        self,
        student_id: str,
        answers: str,
        answer_key: Optional["AnswerKey"] = None,
    ) -> None:
        self.student_id = student_id
        self.answers = answers
        self.answer_key = answer_key

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def num_questions(self) -> int:
        """Number of answer positions in this result."""
        return len(self.answers)

    @property
    def num_correct(self) -> Optional[int]:
        """Count of correctly answered questions, or *None* if no key."""
        if self.answer_key is None:
            return None
        correct = 0
        key = self.answer_key.answers
        for i, detected in enumerate(self.answers):
            if detected not in _VALID_ANSWERS:
                continue  # skip missing / multiple
            if i < len(key) and detected == key[i]:
                correct += 1
        return correct

    @property
    def score(self) -> Optional[float]:
        """Fraction of gradeable questions answered correctly, or *None*.

        Gradeable means the detected answer is a valid single choice (not
        ``"M"`` or ``" "``).
        """
        if self.answer_key is None:
            return None
        gradeable = sum(1 for ch in self.answers if ch in _VALID_ANSWERS)
        if gradeable == 0:
            return 0.0
        nc = self.num_correct
        return round(nc / gradeable, 4) if nc is not None else None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d: dict = {
            "student_id": self.student_id,
            "answers": self.answers,
        }
        if self.answer_key is not None:
            d["score"] = self.score
            d["num_correct"] = self.num_correct
            d["num_questions"] = self.num_questions
        return d

    def to_csv_row(self) -> dict:
        """Return a flat dict suitable for :mod:`csv.DictWriter`."""
        row: dict = {
            "student_id": self.student_id,
            "answers": self.answers,
        }
        if self.answer_key is not None:
            row["score"] = self.score
            row["num_correct"] = self.num_correct
            row["num_questions"] = self.num_questions
        return row
