"""AnswerKey model: stores and validates the correct answers for a test."""

from __future__ import annotations

import csv
import json

_LETTER_TO_DIGIT: dict[str, str] = {
    "A": "1",
    "B": "2",
    "C": "3",
    "D": "4",
    "E": "5",
    "a": "1",
    "b": "2",
    "c": "3",
    "d": "4",
    "e": "5",
}
_VALID_DIGITS = frozenset("12345")
_VALID_LETTERS = frozenset("AaBbCcDdEe")

# ──────────────────────────────────────────────────────────────────────────────
# Shuffle helpers
#
# The shuffle uses the Fisher–Yates algorithm (https://en.wikipedia.org/wiki/
# Fisher%E2%80%93Yates_shuffle) driven by a 32-bit LCG (Numerical Recipes
# parameters: a=1664525, c=1013904223, m=2**32).  A *seed* of 0 is treated as
# a special no-op value: the answers are returned unchanged.  Any other integer
# produces a deterministic, reversible permutation that is the same across
# Python versions and other language implementations that use the same LCG.
# ──────────────────────────────────────────────────────────────────────────────

_LCG_A: int = 1664525
_LCG_C: int = 1013904223
_LCG_M: int = 2**32


def _lcg_sequence(seed: int, length: int) -> list[int]:
    """Return *length* consecutive LCG values starting from *seed*."""
    state = int(seed) & 0xFFFFFFFF
    out = []
    for _ in range(length):
        state = (_LCG_A * state + _LCG_C) % _LCG_M
        out.append(state)
    return out


def _shuffle_answers(answers: str, seed: int) -> str:
    """Return *answers* permuted by a seeded Fisher–Yates shuffle.

    When *seed* is 0 the original string is returned unchanged.  The algorithm
    is deterministic and reversible via :func:`_unshuffle_answers`.

    See https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle for a
    description of the shuffle algorithm used here.
    """
    if seed == 0:
        return answers
    result = list(answers)
    n = len(result)
    if n <= 1:
        return answers
    lcg_vals = _lcg_sequence(seed, n - 1)
    for i in range(n - 1, 0, -1):
        j = lcg_vals[n - 1 - i] % (i + 1)
        result[i], result[j] = result[j], result[i]
    return "".join(result)


def _unshuffle_answers(answers: str, seed: int) -> str:
    """Reverse a :func:`_shuffle_answers` permutation.

    When *seed* is 0 the original string is returned unchanged.

    See https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle for a
    description of the shuffle algorithm used here.
    """
    if seed == 0:
        return answers
    n = len(answers)
    if n <= 1:
        return answers
    # Recompute the same swap sequence used during shuffling.
    lcg_vals = _lcg_sequence(seed, n - 1)
    result = list(answers)
    # Replay swaps in reverse order to invert the permutation.
    for i in range(1, n):
        j = lcg_vals[n - 1 - i] % (i + 1)
        result[i], result[j] = result[j], result[i]
    return "".join(result)


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
    def from_qr_string(
        cls, qr_text: str, shuffle: int = 0, name: str = ""
    ) -> "AnswerKey":
        """Create an :class:`AnswerKey` by decoding a QR-code text payload.

        Parameters
        ----------
        qr_text:
            The plain-text string decoded from a QR code.  This should be the
            (possibly shuffled) answer-key string, containing digits ``1``–``5``
            or letters ``A``–``E``.
        shuffle:
            Seed used to *unshuffle* the QR payload before constructing the key.
            A value of ``0`` (the default) means the payload is used as-is.
            Any other integer reverses the Fisher–Yates permutation that was
            applied when the QR code was generated.

            See https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle for
            a description of the shuffle algorithm.
        name:
            Optional human-readable name for this key.
        """
        unshuffled = _unshuffle_answers(qr_text.strip(), int(shuffle))
        return cls(unshuffled, name=name)

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
