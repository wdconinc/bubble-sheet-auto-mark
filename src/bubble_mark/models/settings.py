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


def _validate_edge_lines(lines: Optional[list]) -> Optional[list]:
    """Return *lines* if it is a valid list of four ``[x1,y1,x2,y2]`` lines.

    Each element must be a list/tuple of exactly four numeric values.  Returns
    *None* if validation fails (wrong count, wrong element type, non-numeric
    values, etc.).
    """
    if lines is None:
        return None
    if not isinstance(lines, (list, tuple)) or len(lines) != 4:
        return None
    try:
        validated = []
        for entry in lines:
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                return None
            coords = [float(v) for v in entry]
            validated.append(coords)
        return validated
    except (TypeError, ValueError):
        return None


def _validate_edge_polylines(polylines: Optional[dict]) -> Optional[dict]:
    """Return a normalised *polylines* dict if valid, else *None*.

    A valid edge-polylines value is a dict that may contain the keys
    ``'top'``, ``'bottom'``, ``'left'``, and ``'right'``.  Keys that are
    absent from the input are treated the same as an explicit ``None`` value
    (i.e., the edge has not yet been drawn).  Each non-*None* value must be a
    list of at least two ``[x, y]`` numeric coordinate pairs.  Returns *None*
    if *polylines* is not a dict or if any non-*None* edge entry fails
    validation.
    """
    if polylines is None:
        return None
    if not isinstance(polylines, dict):
        return None
    result: dict = {}
    for key in ("top", "bottom", "left", "right"):
        pts = polylines.get(key)
        if pts is None:
            result[key] = None
            continue
        if not isinstance(pts, (list, tuple)) or len(pts) < 2:
            return None
        validated_pts = []
        try:
            for pt in pts:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    return None
                validated_pts.append([float(pt[0]), float(pt[1])])
        except (TypeError, ValueError):
            return None
        result[key] = validated_pts
    return result


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
    page_edge_lines:
        Optional list of exactly four ``[x1, y1, x2, y2]`` line definitions
        drawn by the user on the reference sheet to mark the four page edges.
        Used by :func:`~bubble_mark.processing.distortion.correct_distortion_from_lines`
        to compute the perspective warp for the reference setup flow.
    page_edge_polylines:
        Optional dict with keys ``'top'``, ``'bottom'``, ``'left'``, ``'right'``,
        each mapping to a list of at least two ``[x, y]`` coordinate pairs.
        Used by
        :func:`~bubble_mark.processing.distortion.correct_distortion_from_polylines`
        when the sheet edges are curved (e.g., sheet not on a flat surface).
        When all four edges are present, polylines take priority over
        *page_edge_lines*; see :meth:`get_edge_correction_inputs`.
    reference_color_channel:
        Index of the BGR color channel to use as the primary print-color
        reference channel (0=Blue, 1=Green, 2=Red).  Defaults to ``1``
        (green), which offers good contrast for black/dark-blue ink.
    """

    def __init__(
        self,
        layout_config: Optional[dict] = None,
        reference_image_path: Optional[str] = None,
        fill_threshold: float = 0.5,
        answer_region: Optional[list] = None,
        id_region: Optional[list] = None,
        page_edge_lines: Optional[list] = None,
        page_edge_polylines: Optional[dict] = None,
        reference_color_channel: int = 1,
    ) -> None:
        self.layout_config: dict = {**_DEFAULT_LAYOUT, **(layout_config or {})}
        self.reference_image_path: Optional[str] = reference_image_path
        self.fill_threshold: float = fill_threshold
        self.answer_region: Optional[list] = _validate_region(answer_region)
        self.id_region: Optional[list] = _validate_region(id_region)
        self.page_edge_lines: Optional[list] = _validate_edge_lines(page_edge_lines)
        self.page_edge_polylines: Optional[dict] = _validate_edge_polylines(
            page_edge_polylines
        )
        self.reference_color_channel: int = int(reference_color_channel)

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
            "page_edge_lines": self.page_edge_lines,
            "page_edge_polylines": self.page_edge_polylines,
            "reference_color_channel": self.reference_color_channel,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AppSettings":
        return cls(
            layout_config=d.get("layout_config"),
            reference_image_path=d.get("reference_image_path"),
            fill_threshold=d.get("fill_threshold", 0.5),
            answer_region=d.get("answer_region"),
            id_region=d.get("id_region"),
            page_edge_lines=d.get("page_edge_lines"),
            page_edge_polylines=d.get("page_edge_polylines"),
            reference_color_channel=d.get("reference_color_channel", 1),
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

    # ------------------------------------------------------------------
    # Edge-correction selection
    # ------------------------------------------------------------------

    def get_edge_correction_inputs(
        self,
    ) -> tuple:
        """Return the best available edge-correction inputs.

        Priority:

        1. ``('polylines', dict)`` – when *page_edge_polylines* has all four
           edges defined with at least two points each.
        2. ``('lines', list)`` – when *page_edge_lines* is set.
        3. ``('none', None)`` – when neither is available.

        Returns
        -------
        tuple
            A ``(mode, data)`` pair where *mode* is one of
            ``'polylines'``, ``'lines'``, or ``'none'``.
        """
        ep = self.page_edge_polylines
        if ep is not None and all(
            isinstance(ep.get(k), list) and len(ep[k]) >= 2
            for k in ("top", "bottom", "left", "right")
        ):
            return ("polylines", ep)
        if self.page_edge_lines is not None:
            return ("lines", self.page_edge_lines)
        return ("none", None)
