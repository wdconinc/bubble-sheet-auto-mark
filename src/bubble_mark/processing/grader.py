"""Full grading pipeline: detect → locate → analyse → grade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from bubble_mark.models.answer_key import AnswerKey
    from bubble_mark.models.grade_result import GradeResult

from bubble_mark.processing.analyzer import BubbleAnalyzer
from bubble_mark.processing.detector import BubbleSheetDetector


class BubbleSheetGrader:
    """Orchestrate the full grading pipeline for a single sheet image.

    Parameters
    ----------
    answer_key:
        The :class:`~bubble_mark.models.answer_key.AnswerKey` to grade against.
    detector:
        A configured :class:`BubbleSheetDetector`.
    analyzer:
        A configured :class:`BubbleAnalyzer`.
    """

    def __init__(
        self,
        answer_key: "AnswerKey",
        detector: BubbleSheetDetector,
        analyzer: BubbleAnalyzer,
    ) -> None:
        self.answer_key = answer_key
        self.detector = detector
        self.analyzer = analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grade_image(self, image: np.ndarray) -> Optional["GradeResult"]:
        """Run the full pipeline on *image* and return a :class:`GradeResult`.

        Returns *None* if the sheet cannot be detected or processed.
        """
        # Import here to avoid circular imports at module load time

        normalised = self.detector.detect(image)
        if normalised is None:
            return None

        # Detect student ID
        id_bubbles = self.detector.locate_id_bubbles(normalised)
        student_id = "".join(
            self.analyzer.analyze_id_column(normalised, col) for col in id_bubbles
        )

        # Detect answers
        answer_rows = self.detector.locate_answer_bubbles(normalised)
        detected_answers = "".join(
            self.analyzer.analyze_answer_row(normalised, row) for row in answer_rows
        )

        return self.grade_answers(detected_answers, student_id)

    def grade_answers(self, detected_answers: str, student_id: str) -> "GradeResult":
        """Build a :class:`GradeResult` from pre-detected *detected_answers*.

        Parameters
        ----------
        detected_answers:
            String of detected answers (``"1"``–``"5"``, ``"M"``, or ``" "``).
        student_id:
            Student ID string as detected from the ID bubble grid.
        """
        from bubble_mark.models.grade_result import GradeResult

        return GradeResult(
            student_id=student_id,
            answers=detected_answers,
            answer_key=self.answer_key,
        )
