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
    """

    def __init__(
        self,
        layout_config: Optional[dict] = None,
        reference_image_path: Optional[str] = None,
        fill_threshold: float = 0.5,
    ) -> None:
        self.layout_config: dict = {**_DEFAULT_LAYOUT, **(layout_config or {})}
        self.reference_image_path: Optional[str] = reference_image_path
        self.fill_threshold: float = fill_threshold

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "layout_config": self.layout_config,
            "reference_image_path": self.reference_image_path,
            "fill_threshold": self.fill_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AppSettings":
        return cls(
            layout_config=d.get("layout_config"),
            reference_image_path=d.get("reference_image_path"),
            fill_threshold=d.get("fill_threshold", 0.5),
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
