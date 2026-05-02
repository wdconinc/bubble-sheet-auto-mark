"""AppSettings model: persistent application configuration."""

from __future__ import annotations

import json
from typing import Optional

_DEFAULT_LAYOUT: dict = {
    "num_questions": 30,
    "num_choices": 5,
    "num_id_digits": 9,
    "id_choices_per_digit": 10,
}


def _validate_region(region: Optional[list]) -> Optional[list]:
    """Return *region* if valid, else *None*.

    A valid region is a list of four floats ``[x1, y1, x2, y2]`` where each
    value is in [0, 1] and ``x1 < x2``, ``y1 < y2``.
    """
    if region is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in region)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return [x1, y1, x2, y2]


class AppSettings:
    """Application-wide settings.

    Parameters
    ----------
    layout_config:
        Dictionary describing the sheet layout.
    reference_image_path:
        Optional filesystem path to a reference (empty) bubble sheet image.
    fill_threshold:
        Fraction of dark pixels needed to consider a bubble filled.
    answer_region:
        Optional bounding box ``[x1, y1, x2, y2]`` (normalized 0–1) defining
        where the answer bubbles are located on the normalised sheet.  When
        *None* the detector falls back to the built-in layout heuristic
        (top ~72 % of the sheet).
    id_region:
        Optional bounding box ``[x1, y1, x2, y2]`` (normalized 0–1) defining
        where the student-ID bubbles are located on the normalised sheet.
        When *None* the detector falls back to the built-in heuristic
        (bottom ~26 % of the sheet).
    """

    def __init__(
        self,
        layout_config: Optional[dict] = None,
        reference_image_path: Optional[str] = None,
        fill_threshold: float = 0.5,
        answer_region: Optional[list] = None,
        id_region: Optional[list] = None,
    ) -> None:
        self.layout_config: dict = {**_DEFAULT_LAYOUT, **(layout_config or {})}
        self.reference_image_path: Optional[str] = reference_image_path
        self.fill_threshold: float = fill_threshold
        self.answer_region: Optional[list] = _validate_region(answer_region)
        self.id_region: Optional[list] = _validate_region(id_region)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "layout_config": self.layout_config,
            "reference_image_path": self.reference_image_path,
            "fill_threshold": self.fill_threshold,
            "answer_region": self.answer_region,
            "id_region": self.id_region,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AppSettings":
        return cls(
            layout_config=d.get("layout_config"),
            reference_image_path=d.get("reference_image_path"),
            fill_threshold=d.get("fill_threshold", 0.5),
            answer_region=d.get("answer_region"),
            id_region=d.get("id_region"),
        )

    def save(self, path: str) -> None:
        """Save settings to a JSON file at *path*."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AppSettings":
        """Load settings from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def default(cls) -> "AppSettings":
        """Return a default :class:`AppSettings` instance."""
        return cls()
