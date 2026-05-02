"""CSV exporter for grading results."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bubble_mark.models.grade_result import GradeResult

_BASE_FIELDS = ["student_id", "answers"]
_SCORE_FIELDS = ["score", "num_correct", "num_questions"]


class CSVExporter:
    """Export a list of :class:`~bubble_mark.models.grade_result.GradeResult`
    objects to CSV format.

    Parameters
    ----------
    include_score:
        When *True* (default), add ``score``, ``num_correct``, and
        ``num_questions`` columns if any result has an answer key attached.
    """

    def __init__(self, include_score: bool = True) -> None:
        self.include_score = include_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, results: list["GradeResult"], output_path: str) -> None:
        """Write *results* to a CSV file at *output_path*.

        The file is written with UTF-8 encoding.  Existing files are
        overwritten.
        """
        content = self.export_to_string(results)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)

    def export_to_string(self, results: list["GradeResult"]) -> str:
        """Return *results* serialised as a CSV string."""
        fieldnames = self._fieldnames(results)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fieldnames(self, results: list["GradeResult"]) -> list[str]:
        fields = list(_BASE_FIELDS)
        if self.include_score and any(r.answer_key is not None for r in results):
            fields.extend(_SCORE_FIELDS)
        return fields
