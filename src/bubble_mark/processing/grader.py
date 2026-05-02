"""Full grading pipeline: detect → locate → analyse → grade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from bubble_mark.models.answer_key import AnswerKey
    from bubble_mark.models.grade_result import GradeResult

from bubble_mark.processing.analyzer import BubbleAnalyzer
from bubble_mark.processing.detector import BubbleSheetDetector
from bubble_mark.processing.image_utils import draw_overlay


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

        The returned :class:`~bubble_mark.models.grade_result.GradeResult`
        includes an :attr:`~bubble_mark.models.grade_result.GradeResult.annotated_image`
        attribute: a BGR copy of the normalised sheet with an overlay drawn on
        it that shows the page outline, answer and ID section regions, all
        bubble cells, and the filled bubbles that were identified.
        """
        normalised = self.detector.detect(image)
        if normalised is None:
            return None

        # Detect student ID
        id_bubbles = self.detector.locate_id_bubbles(normalised)
        id_chars: list[str] = []
        filled_id_regions: list = []
        for col in id_bubbles:
            ch, filled = self.analyzer.analyze_id_column_with_filled(normalised, col)
            id_chars.append(ch)
            filled_id_regions.extend(filled)
        student_id = "".join(id_chars)

        # Detect answers
        answer_rows = self.detector.locate_answer_bubbles(normalised)
        answer_chars: list[str] = []
        filled_answer_regions: list = []
        for row in answer_rows:
            ch, filled = self.analyzer.analyze_answer_row_with_filled(normalised, row)
            answer_chars.append(ch)
            filled_answer_regions.extend(filled)
        detected_answers = "".join(answer_chars)

        result = self.grade_answers(detected_answers, student_id)

        # Build overlay image ---------------------------------------------------
        all_answer_bubbles = [b for row in answer_rows for b in row]
        all_id_bubbles = [b for col in id_bubbles for b in col]
        result.annotated_image = draw_overlay(
            normalised,
            answer_section_rect=self.detector.answer_section_rect(normalised),
            id_section_rect=self.detector.id_section_rect(normalised),
            all_answer_bubbles=all_answer_bubbles,
            all_id_bubbles=all_id_bubbles,
            filled_answer_bubbles=filled_answer_regions,
            filled_id_bubbles=filled_id_regions,
        )

        return result

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
